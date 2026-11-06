# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (see keyed_sampling.py).
"""Full keyed-race scores of the wide sampler S (keyed_sampling_s), for checks and negative controls only.

score[r, c] = logits[r, c] / T_r + G(salt, key id r, position r, column c), with G the 53-bit keyed Gumbel noise the
sampler races (its fp32 expression, ``_neg_log_exp53_f32_kw``). The sampler's draw is the race's argmax (fp64 on the
near-maximum columns); this module exposes the whole race so a gate can name the runner-up of the SAME race (the token
a keyed sampler would have emitted had the draw lost) or draw from a residual (the race with one column removed).
Not on any production path: tree_spec uses it only under THUNDERSYNC_TREE_DEBUG_RUNNERUP (CHECKS) and for the
negative controls tree_accept_topk / tree_accept_by_draft_prob.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from thundersync_rl.keyed_sampling import _neg_log_exp53_f32_kw
from thundersync_rl.keyed_sampling_s import _keys

BLK = 2048


@triton.jit(do_not_specialize=["salt_lo", "salt_hi"])
def _scores_kernel(logits_ptr, stride, cstride, rid_ptr, pos_ptr, temp_ptr, out_ptr, ostride, vocab, salt_lo, salt_hi,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    b = tl.program_id(1)
    key, key2 = _keys(rid_ptr, pos_ptr, row, salt_lo, salt_hi)
    temp = tl.load(temp_ptr + row).to(tl.float32)
    cols = b * BLOCK + tl.arange(0, BLOCK)
    mask = cols < vocab
    lg = tl.load(logits_ptr + row.to(tl.int64) * stride + cols.to(tl.int64) * cstride, mask=mask,
                 other=float("-inf")).to(tl.float32)
    s32 = lg / temp + _neg_log_exp53_f32_kw(cols.to(tl.uint32), key, key2, True)
    tl.store(out_ptr + row.to(tl.int64) * ostride + cols, tl.where(mask, s32, float("-inf")), mask=mask)


def keyed_scores(logits: torch.Tensor, key_ids: torch.Tensor, positions: torch.Tensor, temperature: torch.Tensor,
                 salt: int) -> torch.Tensor:
    """[n, V] fp32 race scores (allocates; checks only)."""
    n, vocab = logits.shape
    out = torch.empty(n, vocab, dtype=torch.float32, device=logits.device)
    if n == 0:
        return out
    temp = temperature.float().contiguous()
    salt_lo, salt_hi = int(salt) & 0x7FFFFFFF, (int(salt) >> 31) & 0x7FFFFFFF
    _scores_kernel[(n, triton.cdiv(vocab, BLK))](
        logits, logits.stride(0), logits.stride(1), key_ids.contiguous(), positions.contiguous(), temp, out,
        out.stride(0), vocab, salt_lo, salt_hi, BLOCK=BLK, num_warps=4)
    return out


def keyed_uniform(key_ids: torch.Tensor, positions: torch.Tensor, salt: int) -> torch.Tensor:
    """A keyed uniform in [0, 1) per (key id, position), independent of the race's columns (negative controls)."""
    h = key_ids.to(torch.int64) * 0x5DEECE66D + positions.to(torch.int64) * 0x2545F491 + int(salt) * 0x9E3779B1
    h = h ^ (h >> 29)
    h = h * 0x5851F42D
    h = h ^ (h >> 31)
    return ((h & ((1 << 53) - 1)).double() / float(1 << 53)).float()
