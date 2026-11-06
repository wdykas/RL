# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Exact integer categorical law shared bit for bit by the engine sampler and the learner.

Policy (``engine_sampler: exact_int``). The law of the next token is the INTEGER distribution

    l   = log_softmax(z.float())                 fp32, the batch-invariant kernel both sides run
    w_v = floor(exp(float64(l_v)) * 2^52)        int64, elementwise (``exact_nucleus.mass_weights``)
    W   = sum_v w_v                               int64, exact (associative: any order / chunking / TP split)
    pi(v) = w_v / W

and the token is drawn EXACTLY from it: a uniform integer r in [0, W) by rejection on keyed
counter-based 64-bit words (no modulo bias), then the token v with C_{v-1} <= r < C_v, C the
integer prefix sums in token-id order. P(v) = w_v / W exactly, by construction (no floating
point in the law; ``test_exact_int.py`` enumerates every r and every accepted word).

Logged and trained log-prob: lambda(v) = log(w_v) - log(W) in float64 (w_v, W < 2^53 are exact
in float64), computed from the same fp32 row on both sides (engine logprob path, learner
training / mismatch / verification), so sampling law == logged law == trained law. The fp32
loss value is fp32(lambda). Gradient: straight-through d log_softmax (``_gathered_logprob_grad``,
unchanged); the quantization is piecewise constant (zero derivative a.e.).

Scale 2^52 (= ``exact_nucleus.FRAC_BITS``): the same integers as the exact top-p nucleus, so
top-p is the same law restricted to the nucleus (``law_weights``). W < 2^53 for any
log-softmax row (sum exp(l) = 1 up to fp32 rounding), so W, every partial sum, and every
TP shard sum fit int64 with 2^10 headroom; tokens with p < 2^-52 get weight 0.
Gap to the fp64 softmax of the same fp32 row: TV <= V * 2^-52 (+ the normalization by W
instead of sum exp(l)), a definitional change, not a sampling error.

Keyed words: ``word(salt, a, b, purpose, k)`` = a splitmix64 chain (Python ints, numpy uint64
and the Triton kernels give the same 64-bit value). The engine keys by (salt(seed, lane),
engine request id, token index, ENGINE, round); the verifier by (per-call key, position,
round, purpose, counter).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np
import torch

from nemo_rl.models.megatron import exact_nucleus

FRAC_BITS = exact_nucleus.FRAC_BITS  # 52
SCALE = float(2**FRAC_BITS)
MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15
M1 = 0xBF58476D1CE4E5B9
M2 = 0x94D049BB133111EB
# Word purposes (key separation).
ENGINE, ETA, RESID, RESACC, BONUS, SLOW, TEST = 1, 2, 3, 4, 5, 6, 7
INT63_MAX = (1 << 63) - 1

_STATE: dict[str, Any] = {"active": False}
_STATS: dict[str, int] = {"zero_weight_targets": 0}


def configure(active: bool) -> None:
    _STATE["active"] = bool(active)


def active() -> bool:
    return bool(_STATE["active"])


def configure_from_policy(config: Any) -> None:
    mcfg = (config or {}).get("megatron_cfg") or {}
    configure(bool(mcfg.get("exact_int_law", False)))


def stats() -> dict[str, int]:
    return dict(_STATS)


# --------------------------------------------------------------------------
# Keyed 64-bit words (splitmix64 chain)
# --------------------------------------------------------------------------


def _mix_py(z: int) -> int:
    z = (z + GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * M1) & MASK64
    z = ((z ^ (z >> 27)) * M2) & MASK64
    return z ^ (z >> 31)


def word_py(salt: int, a: int, b: int, purpose: int, k: int) -> int:
    """One keyed 64-bit word (Python ints; the reference for the vectorized versions)."""
    h = _mix_py(salt & MASK64)
    h = _mix_py(h ^ (a & MASK64))
    h = _mix_py(h ^ (b & MASK64))
    h = _mix_py(h ^ (purpose & MASK64))
    return _mix_py(h ^ (k & MASK64))


def _mix_np(z: np.ndarray) -> np.ndarray:
    z = z + np.uint64(GOLDEN)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M1)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M2)
    return z ^ (z >> np.uint64(31))


def _u64(x) -> np.ndarray:
    """Python int / numpy integer scalar or array -> uint64 (two's complement for negatives)."""
    if isinstance(x, (int, np.integer)):
        return np.uint64(int(x) & MASK64)
    a = np.asarray(x)
    if a.dtype == np.uint64:
        return a
    return a.astype(np.int64).view(np.uint64)


def words_np(salt, a, b, purpose, k) -> np.ndarray:
    """Vectorized ``word_py`` (numpy uint64, broadcasting; wraps mod 2^64 like the reference)."""
    with np.errstate(over="ignore"):
        h = _mix_np(np.atleast_1d(_u64(salt)))
        h = _mix_np(h ^ _u64(a))
        h = _mix_np(h ^ _u64(b))
        h = _mix_np(h ^ np.uint64(purpose))
        return _mix_np(h ^ _u64(k))


def eta53(words: np.ndarray) -> np.ndarray:
    """Top 53 bits of each word as int64 (eta in [k 2^-53, (k + 1) 2^-53))."""
    return (words >> np.uint64(11)).astype(np.int64)


# --------------------------------------------------------------------------
# Integer weights and row statistics
# --------------------------------------------------------------------------


def int_weights(logprobs: torch.Tensor) -> torch.Tensor:
    """int64 w = floor(exp(float64(l)) * 2^52) (-inf -> 0); ``exact_nucleus.mass_weights``.

    CUDA rows go through the Triton kernel (the same elementwise ``_w_of`` as the
    sampler and ``row_stats``), CPU rows through torch: both are checked bitwise
    equal on GPU by test_exact_int_gpu.py.
    """
    if logprobs.is_cuda and _triton_ok():
        return _int_weights_triton(logprobs)
    return exact_nucleus.mass_weights(logprobs)


def row_stats(
    logprobs: torch.Tensor, tokens: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """(W [n] int64, w_tokens [n] int64 or None) of fp32 log-prob rows [n, V].

    Never materializes [n, V] int64 on CUDA (one Triton program per row).
    """
    lp = logprobs.reshape(-1, logprobs.shape[-1])
    tok = None if tokens is None else tokens.reshape(-1).to(lp.device).long()
    if lp.is_cuda and _triton_ok():
        return _row_stats_triton(lp, tok)
    w = exact_nucleus.mass_weights(lp)
    W = w.sum(-1)
    wx = None if tok is None else w.gather(-1, tok.view(-1, 1))[:, 0]
    return W, wx


def residual_mass(
    lp: torch.Tensor, lq: torch.Tensor, Wp: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per row k of fp32 log-prob rows lp, lq [k, V]: (Wq [k] int64, r [k] float64) with
    r = sum_v max(0, b_k wp_v / Wp_k - wq_v / Wq), w = ``int_weights`` (the block-verification
    residual mass of the integer laws, fp64; summation error <= (V + 4) 2^-51 (b + 1) in any
    order). One Triton program per row on CUDA (no [k, V] temporaries)."""
    lp = lp.reshape(-1, lp.shape[-1])
    lq = lq.reshape(-1, lq.shape[-1])
    Wp = Wp.reshape(-1).to(lp.device)
    b = b.reshape(-1).to(lp.device, torch.float64)
    if lp.is_cuda and _triton_ok():
        return _residual_triton(lp, lq, Wp, b)
    wp = exact_nucleus.mass_weights(lp)
    wq = exact_nucleus.mass_weights(lq)
    Wq = wq.sum(-1)
    r = (b[:, None] * (wp.double() / Wp.double()[:, None]) - wq.double() / Wq.double()[:, None]).clamp(min=0).sum(-1)
    return Wq, r


def lam(wx: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
    """float64 lambda = log(w_x) - log(W) (-inf where w_x = 0); w, W < 2^53 are exact in fp64."""
    return torch.log(wx.double()) - torch.log(W.double())


def int_logprob(logprobs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """float64 lambda(token) of the integer law of each fp32 log-prob row ([..., V] -> [...])."""
    shape = tokens.shape
    W, wx = row_stats(logprobs, tokens)
    return lam(wx, W).view(shape)


def int_logprob_selected(logprobs: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """lambda from each row and its selected fp32 value l_x (w_x depends on l_x only)."""
    W, _ = row_stats(logprobs)
    wx = int_weights(selected.reshape(-1).float().contiguous())
    return lam(wx, W).view(selected.shape)


def training_value(logprobs: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Learner chunk hook: fp32 lambda of each row's target (the TRAINED / mismatch value).

    Targets with w = 0 (never sampled under the integer law: prompt / padding positions)
    keep the fp32 log_softmax value (finite, masked out of the loss) and are counted.
    """
    lam64 = int_logprob(logprobs, tokens)
    raw = logprobs.gather(-1, tokens.unsqueeze(-1).to(logprobs.device)).squeeze(-1)
    zero = torch.isneginf(lam64)
    capturing = lam64.is_cuda and torch.cuda.is_current_stream_capturing()
    if not capturing:  # no host sync inside a CUDA graph capture (learner graphs)
        n0 = int(zero.sum())
        if n0:
            _STATS["zero_weight_targets"] += n0
    return torch.where(zero, raw.double(), lam64).to(torch.float32)


def law_weights(logprobs: torch.Tensor, top_p: Optional[float] = None) -> torch.Tensor:
    """Integer law weights [.., V]: w, or w restricted to the exact nucleus (top-p): the
    nucleus is a subset of the same integers, so pi_top_p(v) = w_v 1[v in N] / sum_N w."""
    w = int_weights(logprobs)
    if top_p is None or float(top_p) >= 1.0:
        return w
    keep = exact_nucleus.plain_mask(w, float(top_p))
    return torch.where(keep, w, torch.zeros_like(w))


# --------------------------------------------------------------------------
# Exact sampling from integer weight rows (reference / verifier emissions)
# --------------------------------------------------------------------------


def uniform_below(W: int, words: Callable[[int], int], bits: int = 63) -> tuple[int, int]:
    """Exact uniform integer in [0, W) from ``bits``-bit words (rejection; no modulo bias).

    ``words(k)`` returns the k-th 64-bit word; its top ``bits`` bits are u in [0, 2^bits).
    Accept u < t = floor((2^bits - 1) / W) * W (a multiple of W, < 2^bits): u | accepted is
    uniform on [0, t), so u mod W is uniform on [0, W). Returns (r, words used).
    """
    if W <= 0:
        raise ValueError("uniform_below: W must be positive")
    top = (1 << bits) - 1
    t = (top // W) * W
    if t == 0:
        raise ValueError(f"uniform_below: W={W} needs more than {bits} bits")
    k = 0
    while True:
        u = words(k) >> (64 - bits)
        k += 1
        if u < t:
            return u % W, k


def select_by_prefix(w: list[int] | np.ndarray, r: int) -> int:
    """Token v with C_{v-1} <= r < C_v (C = integer prefix sums in id order)."""
    c = 0
    for v, x in enumerate(w):
        c += int(x)
        if r < c:
            return v
    raise ValueError("select_by_prefix: r >= W")


def sample_weights_py(w, words: Callable[[int], int], bits: int = 63) -> int:
    """Reference exact sampler (Python ints): P(v) = w_v / sum w."""
    W = sum(int(x) for x in w)
    r, _ = uniform_below(W, words, bits)
    return select_by_prefix(w, r)


def sample_weights(
    w: torch.Tensor, salt: int, a, b, purpose: int, rounds_cap: int = 64
) -> torch.Tensor:
    """Exact samples from int64 weight rows w [n, V] (any device), keyed per row.

    Row i draws from words(salt, a[i], b[i], purpose, k), k = 0, 1, ... (rejection, then
    the prefix-sum selection in id order). ``a``/``b``: [n] ints (numpy / list / tensor).
    One cumsum + searchsorted on the device; the rejection rounds are host-side numpy
    (a round rejects with probability <= W / 2^63, ~2^-11 at W ~ 2^52).
    """
    n = w.shape[0]
    W = w.sum(-1)
    W_h = W.cpu().numpy().astype(np.int64)
    if (W_h <= 0).any():
        raise ValueError("sample_weights: a row has no mass")
    a = np.broadcast_to(np.asarray(a, dtype=np.int64), (n,))
    b = np.broadcast_to(np.asarray(b, dtype=np.int64), (n,))
    t = (np.int64(INT63_MAX) // W_h) * W_h
    r = np.full(n, -1, dtype=np.int64)
    k = 0
    while (r < 0).any():
        if k >= rounds_cap:
            raise RuntimeError("sample_weights: rejection did not terminate")
        u = (words_np(salt, a, b, purpose, np.full(n, k, dtype=np.int64)) >> np.uint64(1)).astype(np.int64)
        take = (r < 0) & (u < t)
        r = np.where(take, u % W_h, r)
        k += 1
    cum = torch.cumsum(w, -1)
    rt = torch.as_tensor(r, device=w.device).view(-1, 1)
    return torch.searchsorted(cum, rt, right=True)[:, 0]


# --------------------------------------------------------------------------
# Tensor-parallel (vocab-sharded) exact selection
# --------------------------------------------------------------------------


def sharded_sample(
    w_local: torch.Tensor,
    rank: int,
    world: int,
    r_of_W: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
    all_reduce_max: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact sample of rows whose weights are split over ``world`` ranks in rank (= id) order.

    ``w_local`` [R, V/world] int64 (this rank's shard of the integer weights). Collectives:
    (1) one int64 all-reduce of [R, world] (each rank fills its column with its shard sum):
    every rank then knows every shard sum, hence W and the shard prefix sums; (2) ``r_of_W(W)``
    draws r in [0, W) from keyed words (identical on every rank: no collective); the owning
    shard s* (prefix_{s*} <= r < prefix_{s*+1}) selects locally with r - prefix_{s*}; (3) one
    int64 all-reduce (max) of [R] global token ids (-1 elsewhere). The result is bitwise the
    gathered-row selection (integer prefix sums are associative). Returns (token [R], W [R]).
    Per rank per row: O(V / world) + (world + 1) int64 values over the TP group.
    """
    R, Vl = w_local.shape
    part = torch.zeros(R, world, dtype=torch.int64, device=w_local.device)
    part[:, rank] = w_local.sum(-1)
    part = all_reduce_sum(part)
    W = part.sum(-1)
    r = r_of_W(W).to(w_local.device)
    pref = torch.cumsum(part, -1) - part  # exclusive shard prefix
    lo, hi = pref[:, rank], pref[:, rank] + part[:, rank]
    mine = (r >= lo) & (r < hi)
    cum = torch.cumsum(w_local, -1)
    loc = torch.searchsorted(cum, (r - lo).clamp(min=0).view(-1, 1), right=True)[:, 0]
    tok = torch.where(mine, loc + rank * Vl, torch.full_like(loc, -1))
    return all_reduce_max(tok), W


# --------------------------------------------------------------------------
# Triton kernels (CUDA): row statistics and the engine sampler
# --------------------------------------------------------------------------

_TRITON: dict[str, Any] = {}


def _triton_ok() -> bool:
    if "ok" not in _TRITON:
        try:
            _build_triton()
            _TRITON["ok"] = True
        except Exception as exc:  # pragma: no cover - no triton
            print(f"[exact_int] triton unavailable ({exc}); torch path", flush=True)
            _TRITON["ok"] = False
    return _TRITON["ok"]


def _build_triton() -> None:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _w_of(l):
        # floor(exp(float64(l)) * 2^52) as int64; -inf -> 0 (``exact_nucleus.mass_weights``).
        return libdevice.floor(libdevice.exp(l.to(tl.float64)) * 4503599627370496.0).to(tl.int64)

    @triton.jit
    def _mix(z):
        z = z + 0x9E3779B97F4A7C15
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
        return z ^ (z >> 31)

    @triton.jit
    def _word(salt, a, b, purpose, k):
        h = _mix(salt)
        h = _mix(h ^ a)
        h = _mix(h ^ b)
        h = _mix(h ^ purpose)
        return _mix(h ^ k)

    @triton.jit
    def _weights_kernel(l_ptr, out_ptr, n, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        x = tl.load(l_ptr + i, mask=m, other=float("-inf")).to(tl.float32)
        tl.store(out_ptr + i, _w_of(x), mask=m)

    @triton.jit
    def _row_stats_kernel(l_ptr, stride, tok_ptr, W_ptr, wx_ptr, vocab, HAS_TOK: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        acc = tl.zeros([BLOCK], tl.int64)
        for off in range(0, vocab, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(l_ptr + row * stride + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            acc += _w_of(x)
        tl.store(W_ptr + row, tl.sum(acc, axis=0))
        if HAS_TOK:
            t = tl.load(tok_ptr + row)
            x = tl.load(l_ptr + row * stride + t).to(tl.float32)
            tl.store(wx_ptr + row, _w_of(x))

    @triton.jit(do_not_specialize=["vocab", "salt", "purpose"])
    def _sample_kernel(
        l_ptr, stride, a_ptr, b_ptr, tok_ptr, W_ptr, wx_ptr, vocab, salt, purpose, BLOCK: tl.constexpr
    ):
        row = tl.program_id(0).to(tl.int64)
        # Pass 1: W = sum_v floor(exp(l_v) 2^52) (int64, exact).
        acc = tl.zeros([BLOCK], tl.int64)
        for off in range(0, vocab, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(l_ptr + row * stride + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            acc += _w_of(x)
        W = tl.sum(acc, axis=0)
        Wc = tl.maximum(W, 1)
        # r uniform on [0, W): 63-bit keyed words, accept u < floor((2^63 - 1) / W) W.
        a = tl.load(a_ptr + row).to(tl.uint64, bitcast=True)
        b = tl.load(b_ptr + row).to(tl.uint64, bitcast=True)
        s = salt.to(tl.int64).to(tl.uint64, bitcast=True)
        pur = purpose.to(tl.int64).to(tl.uint64)
        t = (9223372036854775807 // Wc) * Wc
        k = tl.zeros([], tl.uint64)
        u = (_word(s, a, b, pur, k) >> 1).to(tl.int64)
        while u >= t:
            k = k + 1
            u = (_word(s, a, b, pur, k) >> 1).to(tl.int64)
        r = u % Wc
        # Pass 2: first v with C_v > r (C = prefix sums in id order).
        run = tl.zeros([], tl.int64)
        tok = tl.full([], -1, tl.int64)
        wx = tl.zeros([], tl.int64)
        for off in range(0, vocab, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(l_ptr + row * stride + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            w = _w_of(x)
            cs = tl.cumsum(w, axis=0) + run
            hit = (cs > r) & (cols < vocab)
            first = tl.min(tl.where(hit, cols, 2147483647), axis=0).to(tl.int64)
            take = (tok < 0) & (first < 2147483647)
            wsel = tl.sum(tl.where(cols == first, w, 0), axis=0)
            tok = tl.where(take, first, tok)
            wx = tl.where(take, wsel, wx)
            run = run + tl.sum(w, axis=0)
        tl.store(tok_ptr + row, tl.where(W > 0, tok, 0))
        tl.store(W_ptr + row, W)
        tl.store(wx_ptr + row, wx)

    @triton.jit(do_not_specialize=["n", "salt", "purpose"])
    def _words_kernel(a_ptr, b_ptr, k_ptr, out_ptr, n, salt, purpose, BLOCK: tl.constexpr):
        i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        a = tl.load(a_ptr + i, mask=m, other=0).to(tl.uint64, bitcast=True)
        b = tl.load(b_ptr + i, mask=m, other=0).to(tl.uint64, bitcast=True)
        k = tl.load(k_ptr + i, mask=m, other=0).to(tl.uint64, bitcast=True)
        h = _word(salt.to(tl.int64).to(tl.uint64, bitcast=True), a, b, purpose.to(tl.int64).to(tl.uint64), k)
        tl.store(out_ptr + i, h.to(tl.int64, bitcast=True), mask=m)

    @triton.jit
    def _residual_kernel(p_ptr, p_stride, q_ptr, q_stride, Wp_ptr, b_ptr, Wq_ptr, r_ptr, vocab, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        # Pass 1: Wq = sum_v floor(exp(lq_v) 2^52) (int64, exact).
        acc = tl.zeros([BLOCK], tl.int64)
        for off in range(0, vocab, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            x = tl.load(q_ptr + row * q_stride + cols, mask=cols < vocab, other=float("-inf")).to(tl.float32)
            acc += _w_of(x)
        Wq = tl.sum(acc, axis=0)
        Wqf = Wq.to(tl.float64)
        Wpf = tl.load(Wp_ptr + row).to(tl.float64)
        bb = tl.load(b_ptr + row)
        # Pass 2: r = sum_v max(0, b wp_v / Wp - wq_v / Wq) in fp64.
        racc = tl.zeros([BLOCK], tl.float64)
        for off in range(0, vocab, BLOCK):
            cols = off + tl.arange(0, BLOCK)
            m = cols < vocab
            xp = tl.load(p_ptr + row * p_stride + cols, mask=m, other=float("-inf")).to(tl.float32)
            xq = tl.load(q_ptr + row * q_stride + cols, mask=m, other=float("-inf")).to(tl.float32)
            d = bb * (_w_of(xp).to(tl.float64) / Wpf) - _w_of(xq).to(tl.float64) / Wqf
            racc += tl.maximum(d, 0.0)
        tl.store(Wq_ptr + row, Wq)
        tl.store(r_ptr + row, tl.sum(racc, axis=0))

    _TRITON.update(
        weights=_weights_kernel, row_stats=_row_stats_kernel, sample=_sample_kernel, words=_words_kernel,
        residual=_residual_kernel,
    )


def _int_weights_triton(l: torch.Tensor) -> torch.Tensor:
    _triton_ok()
    x = l.contiguous()
    out = torch.empty(x.shape, dtype=torch.int64, device=x.device)
    n = x.numel()
    if n:
        B = 1024
        _TRITON["weights"][((n + B - 1) // B,)](x, out, n, BLOCK=B)
    return out


def _row_stats_triton(lp: torch.Tensor, tok: Optional[torch.Tensor]):
    _triton_ok()
    if lp.stride(-1) != 1:
        lp = lp.contiguous()
    n, V = lp.shape
    W = torch.empty(n, dtype=torch.int64, device=lp.device)
    wx = torch.empty(n, dtype=torch.int64, device=lp.device) if tok is not None else W
    if n:
        _TRITON["row_stats"][(n,)](
            lp, lp.stride(0), tok if tok is not None else W, W, wx, V,
            HAS_TOK=tok is not None, BLOCK=2048, num_warps=8,
        )
    return W, (wx if tok is not None else None)


def _residual_triton(lp: torch.Tensor, lq: torch.Tensor, Wp: torch.Tensor, b: torch.Tensor):
    _triton_ok()
    if lp.stride(-1) != 1:
        lp = lp.contiguous()
    if lq.stride(-1) != 1:
        lq = lq.contiguous()
    n, V = lp.shape
    Wq = torch.empty(n, dtype=torch.int64, device=lp.device)
    r = torch.empty(n, dtype=torch.float64, device=lp.device)
    if n:
        _TRITON["residual"][(n,)](
            lp, lp.stride(0), lq, lq.stride(0), Wp.to(torch.int64).contiguous(), b.contiguous(), Wq, r, V,
            BLOCK=1024, num_warps=8,
        )
    return Wq, r


def words_device(salt: int, a: torch.Tensor, b: torch.Tensor, purpose: int, k: torch.Tensor) -> torch.Tensor:
    """``word_py`` on the device (int64 bit pattern of the uint64 word), elementwise."""
    a, b, k = (t.to(torch.int64).contiguous() for t in torch.broadcast_tensors(a, b, k))
    out = torch.empty(a.shape, dtype=torch.int64, device=a.device)
    n = a.numel()
    if n:
        _triton_ok()
        B = 1024
        _TRITON["words"][((n + B - 1) // B,)](
            a, b, k, out, n, _as_i64(salt), int(purpose), BLOCK=B
        )
    return out


def _as_i64(x: int) -> int:
    x &= MASK64
    return x - (1 << 64) if x >= (1 << 63) else x


def sample_logprob_rows(
    logprobs: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    salt: int,
    purpose: int = ENGINE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Engine sampler: exact draw from the integer law of each fp32 log-prob row [n, V].

    Row i is keyed by words(salt, a[i], b[i], purpose, k) (engine: a = request id, b = index
    of the sampled token). One Triton program per row, two passes over the vocabulary
    (W; then the prefix-sum selection) and a rejection loop of ~1 + 2^-11 words; no host sync,
    no [n, V] temporaries. Returns (token [n] int64, W [n] int64, w_token [n] int64).
    Rows with W = 0 (all -inf / NaN logits) return token 0 and W = 0 (the caller checks W).
    """
    if not (logprobs.is_cuda and _triton_ok()):
        n = logprobs.shape[0]
        w = int_weights(logprobs)
        tok = sample_weights(w, salt, a.cpu().numpy(), b.cpu().numpy(), purpose)
        W = w.sum(-1)
        return tok, W, w.gather(-1, tok.view(-1, 1))[:, 0]
    lp = logprobs if logprobs.stride(-1) == 1 else logprobs.contiguous()
    n, V = lp.shape
    tok = torch.empty(n, dtype=torch.int64, device=lp.device)
    W = torch.empty_like(tok)
    wx = torch.empty_like(tok)
    if n:
        _TRITON["sample"][(n,)](
            lp, lp.stride(0), a.to(torch.int64).contiguous(), b.to(torch.int64).contiguous(),
            tok, W, wx, V, _as_i64(salt), int(purpose), BLOCK=2048, num_warps=8,
        )
    return tok, W, wx


def sample_logits(
    logits: torch.Tensor, a: torch.Tensor, b: torch.Tensor, salt: int, purpose: int = ENGINE
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``sample_logprob_rows`` of ``log_softmax(logits.float())`` (the engine's raw rows)."""
    return sample_logprob_rows(torch.log_softmax(logits.float(), -1), a, b, salt, purpose)


def tv_gap_bound(vocab: int) -> float:
    """Upper bound on TV(pi_int, softmax_fp64(l)) from the floor: V 2^-52 (+ |W 2^-52 - sum exp l|)."""
    return vocab * 2.0 ** -FRAC_BITS


def describe() -> str:
    return f"exact_int law (2^{FRAC_BITS} integer weights) {'on' if active() else 'off'}"


def log_scale() -> float:
    return math.log(SCALE)
