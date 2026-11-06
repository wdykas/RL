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
"""Position-keyed sampling for exact cross-iteration speculative rollouts.

Every token is a Gumbel-max sample whose noise is a deterministic function of
(trajectory seed, absolute position, token id):

    x = argmax_v  logit(v) + G(seed, position, v),   G = -log(-log U),

with U a hashed uniform. This is an exact sample from softmax(logit). Because the
noise depends only on the key, two policies emit the same token wherever their
races agree, so a draft produced under any weights can be checked position by
position: the emitted sequence is exactly what the target policy would produce
with the same keys, whatever produced the draft. Gumbel coupling matches the
optimal coupling at two-way forks, where nearly all disagreements occur.

One Triton kernel per call: each program reduces one row over the vocabulary
with the noise generated in registers, so there is no host sync and no
materialized noise.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _mix32(x):
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    return x ^ (x >> 16)


@triton.jit
def _uniform(cols, key):
    h = _mix32(cols * 0x9E3779B9 + key)
    return ((h >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)


@triton.jit
def _neg_log_exp53(cols, key):
    """-log(E) of a 53-bit exponential E = -log U, U on the midpoint grid (k + 1/2) / 2^53.

    Two 32-bit hash words per (key, column): 27 + 26 bits -> k in [0, 2^53). The
    fp32 24-bit uniform this replaces bounded the Gumbel noise to [-2.85, 17.33]
    (a token below exp(-20.18) x p_top was never sampled; redteam_1016 §6); here
    E >= 2^-54 and the noise range is 39.4 nats, i.e. a relative probability floor
    of ~1e-16 x p_top (the fp64 resolution of the logits' softmax itself). U near 1
    goes through log1p of its exact complement (no fp64 cancellation). fp64 math.
    """
    h1 = _mix32(cols * 0x9E3779B9 + key)
    h2 = _mix32((cols ^ h1) * 0x85EBCA77 + (key ^ 0x27D4EB2F))
    k = ((h1 >> 5).to(tl.int64) << 26) | (h2 >> 6).to(tl.int64)
    half = 4503599627370496  # 2^52
    # U = (k + 0.5) 2^-53 (exact for k < 2^52); 1 - U = (2^53 - k - 0.5) 2^-53 (exact for k >= 2^52).
    u_lo = (k.to(tl.float64) + 0.5) * 1.1102230246251565e-16
    t_hi = ((2 * half - k).to(tl.float64) - 0.5) * 1.1102230246251565e-16
    e = tl.where(k < half, -libdevice.log(u_lo), -libdevice.log1p(-t_hi))
    return -libdevice.log(e)






@triton.jit
def _bits53_kw(cols, key, key2, KEY64: tl.constexpr):
    """``_bits53`` with an optional second 32-bit key word (ts-lawfix, ``key64``).

    KEY64 False: exactly ``_bits53(cols, key)`` (same instructions). KEY64 True: h1 is re-mixed
    with ``key2`` and h2 takes ``key2``, so two rows share noise only if BOTH words match: the
    32-bit key collided for ~0.5 (request, position) pairs per 8 x 8k group (crit agent), and
    keys k' = k + 0x9E3779B9 d gave column-shifted copies of the h1 stream. Per row, h1 is still
    a bijection of the column (odd multiply, add, xor, mix32 are bijections on 32 bits)."""
    h1 = _mix32(cols * 0x9E3779B9 + key)
    if KEY64:
        h1 = _mix32(h1 ^ key2)
        h2 = _mix32((cols ^ h1) * 0x85EBCA77 + (key2 ^ 0x27D4EB2F))
    else:
        h2 = _mix32((cols ^ h1) * 0x85EBCA77 + (key ^ 0x27D4EB2F))
    return ((h1 >> 5).to(tl.int64) << 26) | (h2 >> 6).to(tl.int64)


@triton.jit
def _neg_log_exp53_kw(cols, key, key2, KEY64: tl.constexpr):
    """``_neg_log_exp53`` on ``_bits53_kw`` (fp64; identical to it when KEY64 is False)."""
    k = _bits53_kw(cols, key, key2, KEY64)
    half = 4503599627370496  # 2^52
    u_lo = (k.to(tl.float64) + 0.5) * 1.1102230246251565e-16
    t_hi = ((2 * half - k).to(tl.float64) - 0.5) * 1.1102230246251565e-16
    e = tl.where(k < half, -libdevice.log(u_lo), -libdevice.log1p(-t_hi))
    return -libdevice.log(e)


@triton.jit
def _neg_log_exp53_f32_kw(cols, key, key2, KEY64: tl.constexpr):
    """``_neg_log_exp53_f32`` on ``_bits53_kw`` (identical to it when KEY64 is False)."""
    k = _bits53_kw(cols, key, key2, KEY64)
    half = 4503599627370496  # 2^52
    u_lo = (k.to(tl.float32) + 0.5) * 1.1102230246251565e-16
    t_hi = ((2 * half - k).to(tl.float32) - 0.5) * 1.1102230246251565e-16
    e = tl.where(k < half, -libdevice.log(u_lo), -libdevice.log1p(-t_hi))
    return -libdevice.log(e)


@triton.jit
def _row_key(seed, pos, stream):
    lo = (seed & 0xFFFFFFFF).to(tl.uint32)
    hi = ((seed >> 32) & 0xFFFFFFFF).to(tl.uint32)
    k = _mix32(lo ^ stream)
    k = _mix32(k ^ hi)
    return _mix32(k ^ (pos & 0xFFFFFFFF).to(tl.uint32))


@triton.jit
def _keyed_gumbel_argmax_kernel(
    logits_ptr,
    stride,
    seed_ptr,
    pos_ptr,
    variant_ptr,
    eps_ptr,
    out_ptr,
    vocab,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    seed = tl.load(seed_ptr + row)
    pos = tl.load(pos_ptr + row)
    key = _row_key(seed, pos, 0x632BE5AB)
    jkey = _row_key(seed + tl.load(variant_ptr + row) * 0x51ED27, pos, 0xC6577B56)
    eps = tl.load(eps_ptr + row)
    best = tl.full([BLOCK], float("-inf"), tl.float64)
    best_idx = tl.zeros([BLOCK], tl.int32)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        lg = tl.load(
            logits_ptr + row.to(tl.int64) * stride + cols,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        c = cols.to(tl.uint32)
        # 53-bit Gumbel race in fp64 (``_neg_log_exp53``).
        score = lg.to(tl.float64) + _neg_log_exp53(c, key)
        # Draft variants only (eps > 0): keyed logistic jitter on the logits.
        uj = _uniform(c, jkey)
        score = score + (eps * (tl.log(uj) - tl.log(1.0 - uj))).to(tl.float64)
        score = tl.where(mask, score, float("-inf"))
        better = score > best
        best = tl.where(better, score, best)
        best_idx = tl.where(better, cols, best_idx)
    m = tl.max(best, axis=0)
    idx = tl.min(tl.where(best == m, best_idx, 2147483647), axis=0)
    tl.store(out_ptr + row, idx.to(tl.int64))


def keyed_sample_logits(
    logits: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    vocab_limit: int | None = None,
    jitter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Keyed Gumbel-max sample from softmax(``logits[:, :vocab_limit]``) per row.

    ``seeds``/``positions``: [n] int64 (position = index of the sampled token).
    ``jitter``: optional [n, 2] (draft variant id, eps); eps = 0 rows are exact
    samples, eps > 0 rows are perturbed drafts. Targets never pass jitter.
    """
    n, width = logits.shape
    vocab = width if vocab_limit is None else min(vocab_limit, width)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    dev = logits.device
    seeds = seeds.long().to(dev, non_blocking=True).contiguous()
    positions = positions.long().to(dev, non_blocking=True).contiguous()
    if jitter is None:
        variants = torch.zeros(n, dtype=torch.int64, device=dev)
        eps = torch.zeros(n, dtype=torch.float32, device=dev)
    else:
        jitter = jitter.to(dev, non_blocking=True)
        variants = jitter[:, 0].long().contiguous()
        eps = jitter[:, 1].float().contiguous()
    out = torch.empty(n, dtype=torch.int64, device=dev)
    if n:
        _keyed_gumbel_argmax_kernel[(n,)](
            logits,
            logits.stride(0),
            seeds,
            positions,
            variants,
            eps,
            out,
            vocab,
            BLOCK=2048,
        )
    return out


@triton.jit(do_not_specialize=["salt_lo", "salt_hi"])
def _wide_gumbel_argmax_kernel(
    logits_ptr,
    stride,
    rid_ptr,
    pos_ptr,
    temp_ptr,
    out_ptr,
    vocab,
    salt_lo,
    salt_hi,
    salt_ptr,
    BLOCK: tl.constexpr,
    KEY64: tl.constexpr = False,
    ROW_SALT: tl.constexpr = False,
):
    row = tl.program_id(0)
    rid = tl.load(rid_ptr + row).to(tl.int64)
    pos = tl.load(pos_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + row).to(tl.float32)
    if ROW_SALT:
        # Per-row salt (stable fresh-row keys, worker.install_wide_sampler): the same two 31-bit words the
        # host derives from a scalar salt, so a row whose salt equals the scalar one draws the same noise.
        srow = tl.load(salt_ptr + row).to(tl.int64)
        salt_lo = srow & 0x7FFFFFFF
        salt_hi = (srow >> 31) & 0x7FFFFFFF
    lo = (rid & 0xFFFFFFFF).to(tl.uint32)
    hi = ((rid >> 32) & 0xFFFFFFFF).to(tl.uint32)
    k = _mix32(lo ^ salt_lo.to(tl.uint32))
    k = _mix32(k ^ hi ^ salt_hi.to(tl.uint32))
    key = _mix32(k ^ (pos & 0xFFFFFFFF).to(tl.uint32) ^ 0x5BD1E995)
    # Second key word (KEY64): its own chain over (salt, request id, position), words swapped and
    # other constants, so (key, key2) coincide for two (request, position) pairs w.p. ~2^-64.
    k2 = _mix32(hi ^ salt_lo.to(tl.uint32) ^ 0x3C6EF372)
    k2 = _mix32(k2 ^ lo ^ salt_hi.to(tl.uint32) ^ 0xA54FF53A)
    key2 = _mix32(k2 ^ (pos & 0xFFFFFFFF).to(tl.uint32) ^ 0x1B873593)
    # Pass 1: fp32 scores (|error| < ~1e-5 vs fp64) -> approximate max.
    best32 = tl.full([BLOCK], float("-inf"), tl.float32)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        lg = tl.load(
            logits_ptr + row.to(tl.int64) * stride + cols, mask=mask, other=float("-inf")
        ).to(tl.float32)
        s32 = lg / temp + _neg_log_exp53_f32_kw(cols.to(tl.uint32), key, key2, KEY64)
        best32 = tl.maximum(best32, tl.where(mask, s32, float("-inf")))
    m32 = tl.max(best32, axis=0)
    # Pass 2: exact fp64 race among the entries within MARGIN of it (every entry outside
    # has an fp64 score below the approximate winner's, so the argmax is the fp64 argmax).
    best = tl.full([BLOCK], float("-inf"), tl.float64)
    best_idx = tl.zeros([BLOCK], tl.int32)
    for off in range(0, vocab, BLOCK):
        cols = off + tl.arange(0, BLOCK)
        mask = cols < vocab
        lg = tl.load(
            logits_ptr + row.to(tl.int64) * stride + cols, mask=mask, other=float("-inf")
        ).to(tl.float32)
        c = cols.to(tl.uint32)
        scaled = lg / temp
        cand = mask & (scaled + _neg_log_exp53_f32_kw(c, key, key2, KEY64) >= m32 - 1e-3)
        if tl.sum(cand.to(tl.int32), axis=0) > 0:
            # The engine sampler's fp32 temperature scaling (FlashInferSampling), then the race.
            score = scaled.to(tl.float64) + _neg_log_exp53_kw(c, key, key2, KEY64)
            score = tl.where(cand, score, float("-inf"))
            better = score > best
            best = tl.where(better, score, best)
            best_idx = tl.where(better, cols, best_idx)
    m = tl.max(best, axis=0)
    idx = tl.min(tl.where(best == m, best_idx, 2147483647), axis=0)
    tl.store(out_ptr + row, idx.to(tl.int64))


def wide_sample_logits(
    logits: torch.Tensor,
    request_ids: torch.Tensor,
    positions: torch.Tensor,
    temperature: torch.Tensor,
    salt: int,
    vocab_limit: int | None = None,
    out: torch.Tensor | None = None,
    key64: bool = True,
    row_salts: torch.Tensor | None = None,
) -> torch.Tensor:
    """53-bit Gumbel-max sample from softmax(logits / T) per row, keyed by (salt, request, position).

    ``row_salts`` (int64 [n], device): a salt per row instead of ``salt`` (a row with row_salts[i] == salt draws
    exactly what the scalar call draws). Used for rows keyed by a stable identity (worker.install_wide_sampler).

    ``key64`` (default True since ts-lawfix): the per-row noise key is two 32-bit words, so two
    (request, position) pairs share noise w.p. ~2^-64 instead of 2^-32 (~0.5 coinciding pairs per
    8 x 8k group before); False = the previous 32-bit-key kernel, bitwise. Same law per token
    (a 53-bit hashed exponential race either way).

    The engine sampler of ThunderSync's block mode (``install_wide_sampler``): a row's
    noise is a function of (``salt``, ``request_ids[i]``, ``positions[i]``, column) only,
    so a token's draw does not depend on the other rows in the batch, their order or
    the engine's launch history (batch-invariant, reproducible), and its law is
    softmax_fp64(fp32 logits / T) down to relative probabilities ~1e-16 of the top token
    (the fp32 / 24-bit samplers stopped at 6e-8 / 1.7e-9). Device tensors only (no host
    sync); one Triton program per row, two passes over the vocabulary: fp32 race (53-bit draws,
    |error| < 1e-5) for the approximate max, then the exact fp64 race over the entries within
    1e-3 of it (usually one), so the result is the fp64 argmax. Cost per token O(V) hashes +
    fp32 logs: ~1.4 us per token at V=131072 on GB200 (256 rows: 363 us vs FlashInfer 346 us).
    """
    n, width = logits.shape
    vocab = width if vocab_limit is None else min(vocab_limit, width)
    if logits.stride(-1) != 1:
        logits = logits.contiguous()
    if out is None:
        out = torch.empty(n, dtype=torch.int64, device=logits.device)
    if n:
        _wide_gumbel_argmax_kernel[(n,)](
            logits,
            logits.stride(0),
            request_ids.contiguous(),
            positions.contiguous(),
            temperature.float().contiguous(),
            out,
            vocab,
            int(salt) & 0x7FFFFFFF,
            (int(salt) >> 31) & 0x7FFFFFFF,
            row_salts.contiguous() if row_salts is not None else request_ids,
            BLOCK=2048,
            KEY64=bool(key64),
            ROW_SALT=row_salts is not None,
            num_warps=8,  # GB200, V=131072, 256 rows: 363 us (FlashInfer's fp32 race: 346 us)
        )
    return out
