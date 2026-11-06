# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (see keyed_sampling.py).
# Kernel-agent candidate S (runs/bi_speed/bi_candidate/sampler.py, sha256 2aa19e680fe63cac), imports re-pointed to thundersync_rl.
"""S: keyed 53-bit Gumbel-max sampler, bitwise equal to oracle_snapshot.keyed_sampling.wide_sample_logits(key64=True).

Today's kernel: one program per row, two passes over the vocabulary (fp32 race -> approximate max m32; exact fp64
race over the columns with s32 >= m32 - 1e-3, ties to the lowest column; all -inf -> column 0).

Here:
  K1 (rows x blocks): one program per 2048-column block computes the fp32 scores s32 (the oracle's expressions) and
     writes the block maximum (maxnum, as tl.maximum / tl.max: NaN only if the whole block is NaN).
  K2 (rows): m32 = maxnum of the block maxima (= the oracle's m32: maxnum is order independent up to the sign of
     zero, which no comparison sees). Only blocks whose maximum reaches m32 - 1e-3 can hold a candidate; those are
     recomputed and raced in fp64 exactly as the oracle's pass 2, keeping (max score, lowest column among equal
     scores). The result is the oracle's: the fp64 argmax over the candidate set, lowest column on ties, column 0 when
     no candidate has a score above -inf.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from thundersync_rl.keyed_sampling import _mix32, _neg_log_exp53_f32_kw, _neg_log_exp53_kw

BLK = 2048


@triton.jit
def _keys(rid_ptr, pos_ptr, row, salt_lo, salt_hi):
    rid = tl.load(rid_ptr + row).to(tl.int64)
    pos = tl.load(pos_ptr + row).to(tl.int64)
    lo = (rid & 0xFFFFFFFF).to(tl.uint32)
    hi = ((rid >> 32) & 0xFFFFFFFF).to(tl.uint32)
    k = _mix32(lo ^ salt_lo.to(tl.uint32))
    k = _mix32(k ^ hi ^ salt_hi.to(tl.uint32))
    key = _mix32(k ^ (pos & 0xFFFFFFFF).to(tl.uint32) ^ 0x5BD1E995)
    k2 = _mix32(hi ^ salt_lo.to(tl.uint32) ^ 0x3C6EF372)
    k2 = _mix32(k2 ^ lo ^ salt_hi.to(tl.uint32) ^ 0xA54FF53A)
    key2 = _mix32(k2 ^ (pos & 0xFFFFFFFF).to(tl.uint32) ^ 0x1B873593)
    return key, key2


@triton.jit(do_not_specialize=["salt_lo", "salt_hi"])
def _block_max_kernel(logits_ptr, stride, cstride, rid_ptr, pos_ptr, temp_ptr, lm_ptr, vocab, nblk, salt_lo, salt_hi,
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
    best32 = tl.maximum(tl.full([BLOCK], float("-inf"), tl.float32), tl.where(mask, s32, float("-inf")))
    tl.store(lm_ptr + row.to(tl.int64) * nblk + b, tl.max(best32, axis=0))


@triton.jit(do_not_specialize=["salt_lo", "salt_hi"])
def _race_kernel(logits_ptr, stride, cstride, rid_ptr, pos_ptr, temp_ptr, lm_ptr, out_ptr, vocab, nblk, salt_lo,
                 salt_hi, BLOCK: tl.constexpr, NB_POW2: tl.constexpr):
    row = tl.program_id(0)
    key, key2 = _keys(rid_ptr, pos_ptr, row, salt_lo, salt_hi)
    temp = tl.load(temp_ptr + row).to(tl.float32)
    bs = tl.arange(0, NB_POW2)
    lm = tl.load(lm_ptr + row.to(tl.int64) * nblk + bs, mask=bs < nblk, other=float("-inf"))
    m32 = tl.max(lm, axis=0)
    thr = m32 - 1e-3
    best = tl.full([], float("-inf"), tl.float64)
    best_idx = tl.full([], 0, tl.int32)
    for b in range(0, nblk):
        lmb = tl.load(lm_ptr + row.to(tl.int64) * nblk + b)
        if lmb >= thr:
            cols = b * BLOCK + tl.arange(0, BLOCK)
            mask = cols < vocab
            lg = tl.load(logits_ptr + row.to(tl.int64) * stride + cols.to(tl.int64) * cstride, mask=mask,
                         other=float("-inf")).to(tl.float32)
            c = cols.to(tl.uint32)
            scaled = lg / temp
            cand = mask & (scaled + _neg_log_exp53_f32_kw(c, key, key2, True) >= m32 - 1e-3)
            score = scaled.to(tl.float64) + _neg_log_exp53_kw(c, key, key2, True)
            score = tl.where(cand, score, float("-inf"))
            mb = tl.max(score, axis=0)
            ib = tl.min(tl.where(score == mb, cols, 2147483647), axis=0)
            # blocks in increasing column order: a later block wins only with a strictly larger score
            better = mb > best
            best_idx = tl.where(better, ib, best_idx)
            best = tl.where(better, mb, best)
    tl.store(out_ptr + row, best_idx.to(tl.int64))


def workspace_bytes(rows: int, vocab: int) -> int:
    return int(rows) * triton.cdiv(int(vocab), BLK) * 4


def wide_sample(logits, request_ids, positions, temperature, salt, *, vocab_limit=None, out, workspace) -> None:
    n, width = logits.shape
    if n == 0:
        return
    vocab = width if vocab_limit is None else min(vocab_limit, width)
    nblk = triton.cdiv(vocab, BLK)
    lm = workspace[: n * nblk * 4].view(torch.float32)
    rid = request_ids if request_ids.is_contiguous() else request_ids.contiguous()
    pos = positions if positions.is_contiguous() else positions.contiguous()
    temp = temperature if (temperature.dtype == torch.float32 and temperature.is_contiguous()) else \
        temperature.float().contiguous()
    salt_lo, salt_hi = int(salt) & 0x7FFFFFFF, (int(salt) >> 31) & 0x7FFFFFFF
    if nblk:
        _block_max_kernel[(n, nblk)](logits, logits.stride(0), logits.stride(1), rid, pos, temp, lm, vocab, nblk,
                                     salt_lo, salt_hi, BLOCK=BLK, num_warps=4)
    _race_kernel[(n,)](logits, logits.stride(0), logits.stride(1), rid, pos, temp, lm, out, vocab, nblk, salt_lo,
                       salt_hi, BLOCK=BLK, NB_POW2=max(triton.next_power_of_2(max(nblk, 1)), 2), num_warps=8)
