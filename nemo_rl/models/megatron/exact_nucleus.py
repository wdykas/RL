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
"""Exact top-p (nucleus) policy shared bit for bit by the engine and the learner.

Policy. With ``top_p`` = P < 1 the policy is

    pi(x | ctx) = softmax(z restricted to N(ctx))[x],   N = nucleus of softmax(z),

computed as ``log_softmax`` of the logits with every token outside N set to
-inf (fp32, the same batch-invariant ``log_softmax`` the raw-logprob path
already runs on both sides). So the only new ingredient is the membership mask,
and it must be a deterministic function of the logits row that both sides
evaluate identically, at any batch shape and any tensor-parallel vocab split.

Canonical nucleus (integer mass, so no floating-point reduction order matters):

    l   = log_softmax(z.float())                       (fp32, batch invariant)
    w_v = floor(exp(float64(l_v)) * 2^52)               (int64, elementwise)
    W   = sum_v w_v,     T = ceil(P * float64(W))       (exact: W < 2^53)
    order: w descending, ties by token id ascending
    B_v = sum of w over the tokens strictly before v in that order
    N   = { v : B_v < T }

Integer sums are associative, so W, B and the mass of any set are the same
whatever the reduction order, chunking or vocab sharding. The first token is
always in N (B = 0 < T for W > 0). Tokens with w = 0 (p < 2^-52) are never in N
at P < 1. Equivalently (used by the sharded search): with
C(t) = sum_{w_v > t} w_v, the threshold weight a = min{t >= 0 : C(t) < T};
N = {w > a} plus the first n_tie = ceil((T - C(a)) / a) tokens of weight a in
id order.

Sharded search (``sharded_mask``; vocab-parallel logits at scale): integer
bisection on t over [0, max w] (<= 53 rounds; each round one all-reduce of one
int64 per row of C(t)), then the tie count via an all-gather of one int64 per
row per rank. Exact (integers) and bitwise the gathered mask; per rank per row:
O(53 V/TP) compares + 53 scalar all-reduces (radix-256 selection: 7 rounds of
a 256-bin all-reduce instead).

Sticky nucleus (``sticky_mask``; opt-in, analogous to sticky MoE routing): an
anchor A (the drafter's own nucleus at that position, logged by the engine,
<= ``cap`` token ids, -1 padded; an empty/all -1 anchor is "self") is kept iff
it is delta-feasible under the target:

    w(A) >= ceil((P - delta) W)   and   w(A) - min_{u in A} w_u < ceil((P + delta) W)

else the target's plain nucleus. With A = the target's own plain nucleus both
branches return it, so F(theta; plain(theta)) = plain(theta) bit for bit.

Gradient. N is piecewise constant in the logits (zero derivative a.e.), so
d log pi(x) / d z = onehot(x) - softmax_N(z) on N and 0 outside: the masked
``log_softmax``'s own backward. GRPO's on-policy gradient E[A grad log pi] is
the policy gradient of the truncated policy wherever theta is not on a mask
boundary (measure zero); the importance ratio at theta = theta_old is 1 because
learner and engine use the same mask.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
from typing import Any, Callable, Iterator, Optional

import torch

FRAC_BITS = 52
_SCALE = float(2**FRAC_BITS)
_INT_MAX = torch.iinfo(torch.int64).max
# Rows per mask block (bounds the int64/sort temporaries to ~5 x rows x V x 8 B).
MASK_BLOCK_ROWS = 256

_STATE: dict[str, Any] = {"top_p": None, "delta": 0.0, "cap": 64}
_ANCHORS: contextvars.ContextVar = contextvars.ContextVar("nucleus_anchors", default=None)
_STATS: dict[str, float] = {}


def configure(top_p: Optional[float], sticky_delta: float = 0.0, cap: int = 64) -> None:
    """Enable the exact nucleus policy in this process (``top_p`` None/1: off)."""
    if top_p is not None and not (0.0 < float(top_p) <= 1.0):
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    _STATE["top_p"] = None if top_p is None or float(top_p) >= 1.0 else float(top_p)
    _STATE["delta"] = float(sticky_delta)
    _STATE["cap"] = int(cap)


def active() -> bool:
    return _STATE["top_p"] is not None


def top_p() -> Optional[float]:
    return _STATE["top_p"]


def sticky_delta() -> float:
    return _STATE["delta"]


def anchor_cap() -> int:
    return _STATE["cap"]


# --------------------------------------------------------------------------
# Integer mass and thresholds
# --------------------------------------------------------------------------


def mass_weights(logprobs: torch.Tensor) -> torch.Tensor:
    """int64 w = floor(exp(float64(l)) * 2^52) of fp32 log-prob rows (-inf -> 0)."""
    return torch.floor(torch.exp(logprobs.double()) * _SCALE).to(torch.int64)


def mass_threshold(W: torch.Tensor, frac: float) -> torch.Tensor:
    """ceil(frac * W) as int64 (W < 2^53: exact in float64 before the product)."""
    return torch.ceil(W.double() * float(frac)).to(torch.int64)


def plain_mask(w: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus membership [..., V] of integer mass rows (gathered reference)."""
    W = w.sum(-1, keepdim=True)
    T = mass_threshold(W, p)
    sw, si = torch.sort(w, dim=-1, descending=True, stable=True)
    B = torch.cumsum(sw, -1) - sw
    keep_sorted = B < T
    return torch.zeros_like(keep_sorted).scatter_(-1, si, keep_sorted)


def anchor_set_mask(anchors: torch.Tensor, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(membership [..., V] of the anchor ids, has-anchor [...]); -1 = padding."""
    valid = (anchors >= 0) & (anchors < vocab)
    idx = anchors.clamp(0, vocab - 1).long()
    counts = torch.zeros(
        (*anchors.shape[:-1], vocab), dtype=torch.int32, device=anchors.device
    ).scatter_add_(-1, idx, valid.to(torch.int32))
    return counts > 0, valid.any(-1)


def sticky_keep(
    w: torch.Tensor, a_mask: torch.Tensor, has: torch.Tensor, p: float, delta: float
) -> torch.Tensor:
    """[...] bool: the anchor set is delta-feasible under the target's masses w."""
    W = w.sum(-1)
    wA = torch.where(a_mask, w, torch.zeros_like(w)).sum(-1)
    minA = torch.where(a_mask, w, torch.full_like(w, _INT_MAX)).amin(-1)
    lo = mass_threshold(W, p - delta)
    hi = mass_threshold(W, p + delta)
    return has & (wA >= lo) & (wA - minA < hi)


def sticky_mask(
    w: torch.Tensor, anchors: Optional[torch.Tensor], p: float, delta: float
) -> torch.Tensor:
    """Sticky-nucleus membership: the anchor set if delta-feasible, else plain."""
    plain = plain_mask(w, p)
    if anchors is None:
        return plain
    a_mask, has = anchor_set_mask(anchors, w.shape[-1])
    keep = sticky_keep(w, a_mask, has, p, delta)
    if _STATS:
        _STATS["rows"] += float(has.numel())
        _STATS["anchored"] += float(has.sum())
        _STATS["kept"] += float(keep.sum())
        _STATS["differs"] += float((keep & (a_mask != plain).any(-1)).sum())
    return torch.where(keep.unsqueeze(-1), a_mask, plain)


def nucleus_mask(
    logits_fp32: torch.Tensor,
    p: Optional[float] = None,
    anchors: Optional[torch.Tensor] = None,
    delta: Optional[float] = None,
) -> torch.Tensor:
    """Membership [..., V] of fp32 logit rows (the engine's / learner's full rows)."""
    p = _STATE["top_p"] if p is None else p
    delta = _STATE["delta"] if delta is None else delta
    w = mass_weights(torch.log_softmax(logits_fp32, -1))
    return sticky_mask(w, anchors, p, delta)


def mask_logits_(
    logits_fp32: torch.Tensor,
    p: Optional[float] = None,
    anchors: Optional[torch.Tensor] = None,
    delta: Optional[float] = None,
    block_rows: int = MASK_BLOCK_ROWS,
) -> torch.Tensor:
    """In place: -inf outside the nucleus. Rows in blocks (row-wise ops only)."""
    if p is None and not active():
        return logits_fp32
    V = logits_fp32.shape[-1]
    flat = logits_fp32.view(-1, V)
    a_flat = None if anchors is None else anchors.reshape(-1, anchors.shape[-1])
    for s in range(0, flat.shape[0], block_rows):
        e = min(flat.shape[0], s + block_rows)
        m = nucleus_mask(flat[s:e], p, None if a_flat is None else a_flat[s:e], delta)
        flat[s:e].masked_fill_(~m, float("-inf"))
    return logits_fp32


def policy_log_softmax(
    logits: torch.Tensor, anchors: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """fp32 log-prob rows of the policy: plain ``log_softmax`` when inactive (canonical-block
    ``fp32(z) - LSE`` under ``NEMO_RL_VOCAB_LSE=canonical_block``, vocab_lse.py)."""
    from nemo_rl.models.megatron import vocab_lse

    if vocab_lse.enabled():
        if active():
            raise RuntimeError("NEMO_RL_VOCAB_LSE=canonical_block does not compose with exact top-p yet")
        return vocab_lse.logprob_rows(logits)
    z = logits.float()
    if active():
        if z.data_ptr() == logits.data_ptr():
            z = z.clone()
        mask_logits_(z, anchors=anchors)
    return torch.log_softmax(z, -1)


def mask_sampled_rows_(
    logits: torch.Tensor, idx: torch.Tensor, cap: int = 0
) -> Optional[torch.Tensor]:
    """Engine sampler: mask rows ``logits[idx]`` in place (-inf outside the nucleus).

    The engine's raw log-prob path then reads ``log_softmax(logits.float())`` of
    the masked row: the truncated policy's log-prob. ``cap`` > 0: also return the
    drafter-nucleus anchor ids [len(idx), cap] int32 (nucleus in (mass desc, id
    asc) order, -1 padded; an all -1 row when the nucleus is larger than cap).
    """
    rows = logits[idx].float()
    w = mass_weights(torch.log_softmax(rows, -1))
    keep = plain_mask(w, _STATE["top_p"])
    logits[idx] = rows.masked_fill(~keep, float("-inf")).to(logits.dtype)
    if cap <= 0:
        return None
    order = torch.sort(w, dim=-1, descending=True, stable=True).indices[:, :cap]
    size = keep.sum(-1)
    ar = torch.arange(order.shape[1], device=order.device)[None]
    ids = torch.where(ar < size[:, None], order, torch.full_like(order, -1)).to(torch.int32)
    ids = torch.where((size > cap)[:, None], torch.full_like(ids, -1), ids)
    if ids.shape[1] < cap:
        ids = torch.cat([ids, ids.new_full((ids.shape[0], cap - ids.shape[1]), -1)], 1)
    return ids


# --------------------------------------------------------------------------
# Sequence-chunk hook for the learner's gathered log-prob helpers
# --------------------------------------------------------------------------


@contextlib.contextmanager
def anchors_context(anchors: Optional[torch.Tensor]) -> Iterator[None]:
    """Sticky-nucleus anchors [B, S, C] (int32, -1 = self) of the next forward."""
    tok = _ANCHORS.set(anchors)
    try:
        yield
    finally:
        _ANCHORS.reset(tok)


def current_anchors() -> Optional[torch.Tensor]:
    """The anchors of the forward in progress (captured by autograd functions,
    whose backward runs outside the post-processing context)."""
    return _ANCHORS.get() if active() else None


def mask_chunk(full_fp32: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Masked copy of a [B, end - start, V] fp32 chunk of the current forward.

    Out of place: the chunk may be a view of saved logits (the TP=1 local path
    hands out ``logits[:, a:b]``, and ``.to(float32)`` of fp32 is the same
    storage). Sequence positions are absolute (the anchors' index).
    """
    if not active():
        return full_fp32
    a = _ANCHORS.get()
    if a is not None:
        if full_fp32.is_cuda and torch.cuda.is_current_stream_capturing():
            # A graph would replay the anchors of the capture call.
            raise RuntimeError("sticky-nucleus log-probs inside a CUDA graph capture")
        a = a[:, start:end].to(full_fp32.device)
        want = int(full_fp32.shape[1])
        if a.shape[1] < want:  # sequence padding past the anchored rows: self
            a = torch.cat([a, a.new_full((a.shape[0], want - a.shape[1], a.shape[2]), -1)], 1)
        if a.shape[0] != full_fp32.shape[0]:
            raise ValueError(
                f"nucleus anchors batch {a.shape[0]} != logits batch {full_fp32.shape[0]}"
            )
    V = full_fp32.shape[-1]
    flat = full_fp32.reshape(-1, V)
    a_flat = None if a is None else a.reshape(-1, a.shape[-1])
    keep = torch.empty(flat.shape, dtype=torch.bool, device=flat.device)
    for s in range(0, flat.shape[0], MASK_BLOCK_ROWS):
        e = min(flat.shape[0], s + MASK_BLOCK_ROWS)
        keep[s:e] = nucleus_mask(flat[s:e], None, None if a_flat is None else a_flat[s:e])
    return full_fp32.masked_fill(~keep.view(full_fp32.shape), float("-inf"))


def configure_from_policy(config: Any) -> None:
    """Set the process policy from ``policy.megatron_cfg`` (ThunderSync sets
    ``exact_top_p`` / ``sticky_nucleus_delta`` / ``sticky_nucleus_cap``)."""
    mcfg = (config or {}).get("megatron_cfg") or {}
    p = mcfg.get("exact_top_p")
    delta = mcfg.get("sticky_nucleus_delta")
    configure(
        None if p is None else float(p),
        0.0 if delta is None else float(delta),
        int(mcfg.get("sticky_nucleus_cap") or 64),
    )
    _STATE["sticky"] = delta is not None


def sticky() -> bool:
    return bool(_STATE.get("sticky")) and active()


# --------------------------------------------------------------------------
# Sharded exact threshold (vocab-parallel rows)
# --------------------------------------------------------------------------


def sharded_mask(
    w_local: torch.Tensor,
    p: float,
    vocab_offset: int,
    all_reduce_sum: Callable[[torch.Tensor], torch.Tensor],
    all_gather_rows: Callable[[torch.Tensor], list[torch.Tensor]],
    rank: int,
) -> torch.Tensor:
    """Nucleus membership of this rank's vocab shard [R, V_local], equal to
    ``plain_mask`` of the gathered rows (any split, ties included).

    ``all_reduce_sum(x)`` returns the elementwise sum over the vocab-parallel
    group (int64, exact); ``all_gather_rows(x)`` the list of every rank's x in
    rank order; ``vocab_offset`` is unused by the rule (ties are broken by
    global id = rank order, then local index) but kept for the caller's
    bookkeeping. Integer bisection on the threshold weight:
    a = min{t >= 0 : C(t) < T}, C(t) = sum_{w > t} w.
    """
    del vocab_offset
    R = w_local.shape[0]
    W = all_reduce_sum(w_local.sum(-1))
    T = mass_threshold(W, p)
    hi = all_reduce_max_via_sum(w_local.amax(-1), all_gather_rows)  # C(max w) = 0 < T
    lo = torch.full_like(hi, -1)  # invariant: C(lo) >= T (C(-1) = W >= T), C(hi) < T
    for _ in range(FRAC_BITS + 2):
        if bool((hi - lo <= 1).all()):
            break
        mid = (lo + hi) // 2
        c = all_reduce_sum(torch.where(w_local > mid[:, None], w_local, 0).sum(-1))
        below = c < T
        hi = torch.where(below, mid, hi)
        lo = torch.where(below, lo, mid)
    a = hi  # [R]
    c_a = all_reduce_sum(torch.where(w_local > a[:, None], w_local, 0).sum(-1))
    tie = w_local == a[:, None]
    n_tie_local = tie.sum(-1)
    counts = all_gather_rows(n_tie_local)  # list over ranks of [R]
    before = torch.zeros(R, dtype=torch.int64, device=w_local.device)
    for r in range(rank):
        before += counts[r].to(before.device)
    a_pos = a.clamp_min(1)
    n_in = torch.div(T - c_a + a_pos - 1, a_pos, rounding_mode="floor")  # ceil((T - C(a)) / a)
    tie_rank = torch.cumsum(tie.to(torch.int64), -1) - 1 + before[:, None]
    return (w_local > a[:, None]) | (tie & (tie_rank < n_in[:, None]))


def all_reduce_max_via_sum(
    x: torch.Tensor, all_gather_rows: Callable[[torch.Tensor], list[torch.Tensor]]
) -> torch.Tensor:
    parts = all_gather_rows(x)
    out = parts[0].clone()
    for t in parts[1:]:
        out = torch.maximum(out, t.to(out.device))
    return out


# --------------------------------------------------------------------------
# Check counters (sticky rule), as sticky_routing.start_stats / pop_stats
# --------------------------------------------------------------------------


def start_stats() -> None:
    _STATS.clear()
    _STATS.update(rows=0.0, anchored=0.0, kept=0.0, differs=0.0)


def pop_stats() -> dict[str, float]:
    out = dict(_STATS)
    _STATS.clear()
    return out


def describe() -> str:
    if not active():
        return "nucleus off"
    return f"nucleus top_p={_STATE['top_p']} delta={_STATE['delta']} cap={_STATE['cap']}"


def log2_scale() -> float:
    return math.log(_SCALE)
