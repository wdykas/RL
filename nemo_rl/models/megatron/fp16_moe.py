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
"""MoE experts under the fp16 GEMM forward (``forward_gemm_precision: fp16``).

Training (``TEGroupedMLP``, AllToAll dispatcher) and the Megatron engine
(``InferenceGroupedMLP``, all-gather dispatcher) compute every routed token's
expert output with the same functions:

- ``grouped_gemm``: one Triton kernel, fp16 x fp16 products accumulated in fp32,
  fp32 output, fixed tile shape and K loop. Rows are grouped by expert and each
  expert's rows padded to ``BLOCK_M``; a row's result depends only on the row and
  its expert's weight (batch invariant), and the launch shape is static (CUDA
  graph safe). The bf16 expert weights are read in place and converted to fp16
  in registers (exact in fp16's normal range); no weight copies.
- ``swiglu_probs``: SiLU(gate) * up * prob in fp32 (one Triton kernel); non-gated
  squared-ReLU experts (Nemotron-H) use ``relu2_probs``: ReLU(h)^2 * prob in fp32
  (eager elementwise ops). ``expert_activation`` picks one from the model config.
- combine: the engine sums a token's local-expert outputs in fp64 in ascending
  expert order and rounds the rank partial to fp32, exactly as Megatron's
  batch-invariant training unpermute (``transformer/moe/batch_invariant.py``)
  does; ranks are then summed in fp32 in rank order (MCore's
  ``ordered_reduce_scatter_v`` in the engine, the same chain in training).
- the router GEMM gives fp32 logits from the fp32 hidden state through
  ``fp16_gemm.fp32_gemm`` (every product exact, batch invariant) on both sides.

The engine all-gathers the fp16 token rows (bit-cast to bf16 for MCore's NVLS
all-gather buffers, a pure copy); training dispatches the fp32 rows (their
gradients stay fp32). Both round a row to fp16 once, at the first expert GEMM.
Backward: dgrad / wgrad per expert on bf16 casts with fp32 accumulation.
"""

from __future__ import annotations

import collections
import contextlib
import os
from typing import Any, Iterator, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

# Tile shapes do not change a row's bits: tl.dot accumulates the K dimension in
# order into one fp32 accumulator, and products of fp16 operands are exact, so the
# result depends only on the row, its expert's weight and K order. Measured bitwise
# identical over BLOCK_M 16-128, BLOCK_N 32-256, BLOCK_K 32-128, 2-8 warps and 2-4
# stages (research/thundersync_rl/DESIGN.md section 14b), and bitwise equal to cuBLAS
# without split-K. The configuration is picked per call from the expected rows per
# expert. Tuned at 500B-MoE expert shapes (8 local experts, hidden 7168, expert FFN
# 2048; thundersync_work/runs/moe_decode_sweep.json): fc1 decode 252 -> 168 us.
TileConfig = tuple  # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_CONFIGS: tuple[tuple[int, TileConfig], ...] = (
    (16, (16, 64, 128, 4, 4)),  # decode
    (32, (32, 64, 128, 4, 4)),
    (64, (64, 32, 128, 4, 4)),
    (1 << 62, (128, 64, 64, 4, 4)),  # engine prefill (training uses cuBLAS)
)
BLOCK_M = 16  # row alignment of the engine's decode buckets (smallest tile)


def select_config(rows_per_expert: float) -> TileConfig:
    """Tile configuration for an expected number of rows per expert."""
    for bound, cfg in _CONFIGS:
        if rows_per_expert <= bound:
            return cfg
    return _CONFIGS[-1][1]


if HAVE_TRITON:

    @triton.jit
    def _grouped_gemm_fp16w_kernel(
        x_ptr,  # [M_pad, K] fp16, rows grouped by expert, segments BLOCK_M-aligned
        w_ptr,  # [E, N, K] fp16 expert weights
        offs_ptr,  # [E] inclusive BLOCK_M-aligned row offsets of the expert segments
        out_ptr,  # [M_pad, N] fp32
        M,
        N,
        K,
        E,
        E_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # Same per-row math as _grouped_gemm_kernel (exact fp16 products, K accumulated
        # in order in fp32), with both operands fed through TMA descriptors. Persistent
        # over the live row tiles: program p computes M tiles p, p + P, ... below
        # offs[E - 1] (read on device, so the launch shape is static and the work is
        # proportional to the live rows, whatever the buffer's padded size).
        pid_n = tl.program_id(1)
        idx = tl.arange(0, E_POW2)
        ends = tl.load(offs_ptr + idx, mask=idx < E, other=2147483647).to(tl.int64)
        live_tiles = tl.load(offs_ptr + E - 1).to(tl.int32) // BLOCK_M
        xd = tl.make_tensor_descriptor(
            x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
        )
        wd = tl.make_tensor_descriptor(
            w_ptr, shape=[E * N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K]
        )
        od = tl.make_tensor_descriptor(
            out_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_M, BLOCK_N]
        )
        # num_stages=1: the K loop inside is pipelined; pipelining this loop as well would
        # keep the next tile's loads in shared memory during the epilogue (over the limit
        # for the 128 x 256 prefill tile).
        for pid_m in tl.range(
            tl.program_id(0), live_tiles, tl.num_programs(0), num_stages=1
        ):
            start = pid_m * BLOCK_M
            expert = tl.sum((ends <= start).to(tl.int32), axis=0)
            w_row = expert * N + pid_n * BLOCK_N
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k0 in range(0, K, BLOCK_K):
                acc = tl.dot(xd.load([start, k0]), wd.load([w_row, k0]).T, acc)
            od.store([start, pid_n * BLOCK_N], acc)

    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr,  # [M_pad, K] fp16, rows grouped by expert, segments BLOCK_M-aligned
        w_ptr,  # base pointer of the first expert's [N, K] weight
        w_off_ptr,  # [E] int64 element offset of each expert's weight from w_ptr
        offs_ptr,  # [E] inclusive BLOCK_M-aligned row offsets of the expert segments
        out_ptr,  # [M_pad, N] fp32
        N,
        K,
        stride_xm,
        stride_om,
        E,
        E_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        start = pid_m * BLOCK_M
        idx = tl.arange(0, E_POW2)
        ends = tl.load(offs_ptr + idx, mask=idx < E, other=2147483647).to(tl.int64)
        expert = tl.sum((ends <= start).to(tl.int32), axis=0)
        if expert >= E:
            return
        w_base = w_ptr + tl.load(w_off_ptr + expert)
        rm = (start + tl.arange(0, BLOCK_M)).to(tl.int64)
        rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            kk = k0 + rk
            a = tl.load(
                x_ptr + rm[:, None] * stride_xm + kk[None, :],
                mask=kk[None, :] < K,
                other=0.0,
            )
            b = tl.load(
                w_base + rn[None, :].to(tl.int64) * K + kk[:, None],
                mask=(rn[None, :] < N) & (kk[:, None] < K),
                other=0.0,
            ).to(tl.float16)
            acc = tl.dot(a, b, acc)
        tl.store(
            out_ptr + rm[:, None] * stride_om + rn[None, :],
            acc,
            mask=rn[None, :] < N,
        )

    @triton.jit
    def _swiglu_probs_kernel(
        h_ptr,  # [rows, 2F] fp32 (gate | up)
        p_ptr,  # [rows] fp32
        out_ptr,  # [rows, F] fp32 or fp16
        F,
        rows,  # rows of the buffer
        valid_ptr,  # scalar int32/int64: rows from here on are skipped (HAS_VALID)
        p_scale,  # fp32 factor on the probability (HAS_SCALE)
        BLOCK: tl.constexpr,
        HAS_VALID: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        if HAS_VALID:
            end = tl.load(valid_ptr).to(tl.int64)
        else:  # one row per program (grid = rows)
            end = tl.program_id(0).to(tl.int64) * 0 + rows
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        for row in tl.range(tl.program_id(0).to(tl.int64), end, tl.num_programs(0)):
            gate = tl.load(h_ptr + row * 2 * F + cols, mask=mask, other=0.0)
            up = tl.load(h_ptr + row * 2 * F + F + cols, mask=mask, other=0.0)
            p = tl.load(p_ptr + row)
            if HAS_SCALE:
                p = p * p_scale
            value = (gate / (1.0 + tl.exp(-gate))) * up * p
            tl.store(
                out_ptr + row * F + cols, value.to(out_ptr.dtype.element_ty), mask=mask
            )

    @triton.jit
    def _relu2_probs_kernel(
        h_ptr,  # [rows, F] fp32
        p_ptr,  # [rows] fp32
        out_ptr,  # [rows, F] fp32 or fp16
        F,
        valid_ptr,  # scalar int32/int64: rows from here on are skipped
        p_scale,  # fp32 factor on the probability (HAS_SCALE)
        BLOCK: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        # _relu2_eager's ops in its order: relu, (r * r), * p in fp32, then one cast.
        # Persistent over the rows below valid_ptr's value.
        end = tl.load(valid_ptr).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        for row in tl.range(tl.program_id(0).to(tl.int64), end, tl.num_programs(0)):
            h = tl.load(h_ptr + row * F + cols, mask=mask, other=0.0)
            p = tl.load(p_ptr + row)
            if HAS_SCALE:
                p = p * p_scale
            r = tl.maximum(h, 0.0)
            value = (r * r) * p
            tl.store(
                out_ptr + row * F + cols, value.to(out_ptr.dtype.element_ty), mask=mask
            )

    @triton.jit
    def _unpermute_fp64_kernel(
        expert_out_ptr,  # [rows, H] fp32 expert outputs (probability-weighted)
        inverse_map_ptr,  # [num_tokens, num_local_experts] int32 row or -1
        valid_tokens_ptr,  # scalar int32
        output_ptr,  # [num_tokens, H] fp32
        hidden_dim,
        y_scale,  # fp32 factor on every expert row (HAS_SCALE)
        num_local_experts: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_SCALE: tl.constexpr,
    ):
        # Persistent over the valid tokens; rows from valid_tokens on are not written.
        valid_tokens = tl.load(valid_tokens_ptr).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offsets < hidden_dim
        for tok in tl.range(
            tl.program_id(0).to(tl.int64), valid_tokens, tl.num_programs(0)
        ):
            acc = tl.zeros([BLOCK_H], dtype=tl.float64)
            for lid in tl.range(0, num_local_experts):
                pos = tl.load(inverse_map_ptr + tok * num_local_experts + lid)
                if pos >= 0:
                    vals = tl.load(
                        expert_out_ptr + pos.to(tl.int64) * hidden_dim + offsets,
                        mask=mask_h,
                    )
                    if HAS_SCALE:
                        vals = vals * y_scale
                    acc += vals.to(tl.float64)
            tl.store(
                output_ptr + tok * hidden_dim + offsets,
                acc.to(tl.float32),
                mask=mask_h,
            )

    @triton.jit
    def _permute_live_kernel(
        hidden_ptr,  # [max_tokens, H] 16-bit token rows
        probs_ptr,  # [max_tokens, TOPK] fp32
        routing_ptr,  # [max_tokens, TOPK] int64 global expert ids (-1: none)
        out_hidden_ptr,  # [rows, H] permuted token rows
        out_probs_ptr,  # [rows] fp32 permuted probabilities
        inverse_map_ptr,  # [max_tokens, E] int32 permuted row of (token, local expert) or -1
        counters_ptr,  # [E] int32 exclusive segment offsets, advanced atomically
        valid_tokens_ptr,  # scalar int32
        hidden_dim,
        local_expert_start,
        TOPK: tl.constexpr,
        E: tl.constexpr,
        E_POW2: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        # Persistent over the valid tokens: each token's local (token, expert) pairs get
        # a row of their expert's segment; its inverse-map row is written whole (-1
        # where the token does not route), so the map needs no separate fill.
        valid_tokens = tl.load(valid_tokens_ptr).to(tl.int64)
        lids = tl.arange(0, E_POW2)
        for tok in tl.range(
            tl.program_id(0).to(tl.int64), valid_tokens, tl.num_programs(0)
        ):
            inv = tl.full([E_POW2], -1, tl.int32)
            for k in tl.static_range(TOPK):
                lid = (tl.load(routing_ptr + tok * TOPK + k) - local_expert_start).to(
                    tl.int32
                )
                if (lid >= 0) & (lid < E):
                    pos = tl.atomic_add(counters_ptr + lid, 1)
                    for h in tl.range(0, hidden_dim, BLOCK_H):
                        o = h + tl.arange(0, BLOCK_H)
                        m = o < hidden_dim
                        tl.store(
                            out_hidden_ptr + pos.to(tl.int64) * hidden_dim + o,
                            tl.load(hidden_ptr + tok * hidden_dim + o, mask=m),
                            mask=m,
                        )
                    tl.store(out_probs_ptr + pos, tl.load(probs_ptr + tok * TOPK + k))
                    inv = tl.where(lids == lid, pos, inv)
            tl.store(inverse_map_ptr + tok * E + lids, inv, mask=lids < E)


if HAVE_TRITON:

    @triton.jit
    def _segment_tile(counts_ptr, t, E, E_POW2: tl.constexpr, BLOCK_M: tl.constexpr):
        """(expert, first row, end row) of row tile ``t`` over unpadded expert segments.

        Segment e holds counts[e] rows from sum(counts[:e]); its rows are cut into
        ceil(counts[e] / BLOCK_M) tiles, numbered in expert order. expert >= E: no tile.
        """
        idx = tl.arange(0, E_POW2)
        c = tl.load(counts_ptr + idx, mask=idx < E, other=0).to(tl.int64)
        tiles = (c + BLOCK_M - 1) // BLOCK_M
        tile_end = tl.cumsum(tiles, axis=0)
        row_end = tl.cumsum(c, axis=0)
        expert = tl.sum((tile_end <= t).to(tl.int32), axis=0)
        sel = idx == expert
        first_tile = tl.sum(tl.where(sel, tile_end - tiles, 0), axis=0)
        seg_end = tl.sum(tl.where(sel, row_end, 0), axis=0)
        seg_start = seg_end - tl.sum(tl.where(sel, c, 0), axis=0)
        return expert, seg_start + (t - first_tile) * BLOCK_M, seg_end

    @triton.jit
    def _segment_gemm_kernel(
        a_ptr,  # [M, R] rows grouped by expert (unpadded), 16-bit
        w_ptr,  # base pointer of the first expert's weight
        w_off_ptr,  # [E] int64 element offset of each expert's weight from w_ptr
        counts_ptr,  # [E] int32 rows per expert
        out_ptr,  # [M, C] fp32
        C,
        R,
        stride_wr,  # weight element stride along the reduction dimension
        stride_wc,  # weight element stride along the output dimension
        E,
        E_POW2: tl.constexpr,
        B_FP16: tl.constexpr,  # convert weight elements to fp16 (else keep the dtype)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        # out[r] = a[r] @ B_e for the rows r of expert e, B_e[j, c] = w_e[j * stride_wr +
        # c * stride_wc]: the forward (B = W_e^T, operands fp16) and dgrad (B = W_e,
        # operands bf16) of the training expert GEMMs. Exact products, the reduction
        # accumulated in order into one fp32 accumulator per row, as _grouped_gemm_kernel
        # (bitwise equal to cuBLAS without split-K). Rows outside the tile's segment are
        # masked (loaded as 0, not stored), so no padded copies of the rows are needed.
        expert, row0, seg_end = _segment_tile(
            counts_ptr, tl.program_id(0), E, E_POW2, BLOCK_M
        )
        if expert >= E:
            return
        w_base = w_ptr + tl.load(w_off_ptr + expert)
        rm = row0 + tl.arange(0, BLOCK_M).to(tl.int64)
        rn = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.arange(0, BLOCK_K)
        row_ok = rm < seg_end
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, R, BLOCK_K):
            kk = k0 + rk
            a = tl.load(
                a_ptr + rm[:, None] * R + kk[None, :],
                mask=row_ok[:, None] & (kk[None, :] < R),
                other=0.0,
            )
            b = tl.load(
                w_base
                + kk[:, None].to(tl.int64) * stride_wr
                + rn[None, :].to(tl.int64) * stride_wc,
                mask=(rn[None, :] < C) & (kk[:, None] < R),
                other=0.0,
            )
            if B_FP16:
                b = b.to(tl.float16)
            acc = tl.dot(a, b, acc)
        tl.store(
            out_ptr + rm[:, None] * C + rn[None, :],
            acc,
            mask=row_ok[:, None] & (rn[None, :] < C),
        )

    @triton.jit
    def _segment_wgrad_kernel(
        g_ptr,  # [M, N] bf16 output gradients, rows grouped by expert (unpadded)
        x_ptr,  # [M, K] bf16 inputs, same rows
        counts_ptr,  # [E] int32 rows per expert
        out_ptr,  # [E, N, K] fp32
        N,
        K,
        E,
        E_POW2: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        BLOCK_R: tl.constexpr,
    ):
        # out[e] = g_e^T @ x_e: the reduction runs over expert e's rows in order, one
        # fp32 accumulator per element (an expert without rows gets zeros).
        e = tl.program_id(2)
        idx = tl.arange(0, E_POW2)
        c = tl.load(counts_ptr + idx, mask=idx < E, other=0).to(tl.int64)
        row_end = tl.cumsum(c, axis=0)
        sel = idx == e
        end = tl.sum(tl.where(sel, row_end, 0), axis=0)
        start = end - tl.sum(tl.where(sel, c, 0), axis=0)
        rn = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
        rk = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
        rr = tl.arange(0, BLOCK_R)
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for r0 in range(start, end, BLOCK_R):
            r = (r0 + rr).to(tl.int64)
            ok = r < end
            gt = tl.load(
                g_ptr + r[None, :] * N + rn[:, None],
                mask=ok[None, :] & (rn[:, None] < N),
                other=0.0,
            )
            xb = tl.load(
                x_ptr + r[:, None] * K + rk[None, :],
                mask=ok[:, None] & (rk[None, :] < K),
                other=0.0,
            )
            acc = tl.dot(gt, xb, acc)
        if end - start == 1:
            # One row: cuBLAS forms the outer product with plain fp32 multiplies
            # (a zero keeps its sign); the MMA path above would return +0.0.
            gv = tl.load(g_ptr + start * N + rn, mask=rn < N, other=0.0)
            xv = tl.load(x_ptr + start * K + rk, mask=rk < K, other=0.0)
            acc = gv.to(tl.float32)[:, None] * xv.to(tl.float32)[None, :]
        tl.store(
            out_ptr
            + e.to(tl.int64) * N * K
            + rn[:, None].to(tl.int64) * K
            + rk[None, :],
            acc,
            mask=(rn[:, None] < N) & (rk[None, :] < K),
        )


def _weight_table(weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(base weight, int64 element offsets of every expert weight from it)."""
    base = weights[0]
    for w in weights:
        assert w.is_contiguous() and w.dtype == base.dtype and w.shape == base.shape
    size = base.element_size()
    offs = [(w.data_ptr() - base.data_ptr()) // size for w in weights]
    return base, torch.tensor(offs, dtype=torch.int64, device=base.device)


def grouped_gemm(
    x16: torch.Tensor,
    weights: list[torch.Tensor] | None,
    offsets: torch.Tensor,
    config: TileConfig,
    *,
    stacked: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-invariant grouped GEMM: x16 [M_pad, K] fp16 @ W_e.T -> fp32 [M_pad, N].

    ``offsets``: [E] inclusive row offsets of the expert segments, each a
    multiple of ``config``'s BLOCK_M (rows past the last offset are skipped).
    Weights: ``weights`` (one [N, K] tensor per expert, any addresses) or
    ``stacked`` ([E, N, K] contiguous; its offset table is built on device, so
    the call is CUDA-graph safe).
    """
    ws = list(weights) if weights is not None else list(stacked)
    if not x16.is_cuda:
        return _grouped_gemm_reference(x16, ws, offsets)
    bm, bn, bk, warps, stages = config
    m_pad, k = x16.shape
    assert m_pad % bm == 0 and x16.dtype == torch.float16 and x16.is_contiguous()
    if stacked is not None:
        e, n, _ = stacked.shape
        w_offs = torch.arange(e, device=stacked.device, dtype=torch.int64) * (n * k)
        base = stacked
    else:
        base, w_offs = _weight_table(ws)
        e, n = len(ws), base.shape[0]
    out = torch.empty((m_pad, n), dtype=torch.float32, device=x16.device)
    grid = (m_pad // bm, triton.cdiv(n, bn))
    _grouped_gemm_kernel[grid](
        x16,
        base,
        w_offs,
        offsets,
        out,
        n,
        k,
        x16.stride(0),
        out.stride(0),
        e,
        E_POW2=triton.next_power_of_2(e),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


# Engine tiles for grouped_gemm_fp16w, tuned at 500B-MoE expert shapes (8 local
# experts, hidden 7168, expert FFN 2048; thundersync_work/runs/moe_tma_sweep.json):
# fc1 / fc2 at 4 rows per expert 76 / 47 us, at 256 rows 135 / 85 us (bf16 cuBLAS
# grouped GEMM: 110 / 58 and 107 / 58 us). Every tile measured bitwise equal to cuBLAS.
_ENGINE_CONFIGS: tuple[tuple[int, TileConfig], ...] = (
    (16, (16, 256, 128, 4, 4)),  # decode
    (64, (64, 128, 64, 4, 4)),
    (1 << 62, (128, 256, 64, 8, 4)),
)


def select_engine_config(rows_per_expert: float) -> TileConfig:
    """Engine tile configuration for an expected number of rows per expert."""
    for bound, cfg in _ENGINE_CONFIGS:
        if rows_per_expert <= bound:
            return cfg
    return _ENGINE_CONFIGS[-1][1]


# Engine MoE kernels run persistent grids: a program loops over its share of the live
# work (rows / tokens below a device-side count), so the launch shape is static (CUDA
# graph safe) and the work is proportional to the live rows. The grid is the smallest
# of the buffer's padded work, a host hint of the live work (the CUDA-graph bucket
# bounds it: bucket tokens x EP x top-k) and _PROGRAMS_PER_SM programs per SM. The
# grid never changes a row's arithmetic, only which program computes it.
_PROGRAMS_PER_SM = 8
_NUM_SMS: dict[int, int] = {}


def _persistent_programs(
    work: int, hint: Optional[int], device: torch.device, per_program: int = 1
) -> int:
    """Grid size over ``work`` items: min(work, hint, cap), at least 1.

    ``per_program``: launched programs per work item along other grid axes (the
    cap counts all of them).
    """
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    cap = max(1, _PROGRAMS_PER_SM * _NUM_SMS[idx] // per_program)
    n = min(work, cap) if hint is None else min(work, hint, cap)
    return max(1, n)


def live_rows_bound(
    tokens: int, topk: int, num_local_experts: int, block_m: int
) -> int:
    """Upper bound of the permuted rows (``offsets[-1]``) for ``tokens`` gathered tokens.

    Each token routes to at most min(topk, local experts) local experts, and each
    non-empty expert segment is padded by fewer than ``block_m`` rows.
    """
    pairs = tokens * min(topk, num_local_experts)
    return pairs + min(num_local_experts, pairs) * (block_m - 1)


_ALLOCATOR_SET = False


def grouped_gemm_fp16w(
    x16: torch.Tensor,
    w16: torch.Tensor,
    offsets: torch.Tensor,
    config: TileConfig,
    rows_hint: Optional[int] = None,
) -> torch.Tensor:
    """Batch-invariant grouped GEMM on fp16 expert weights: x16 [M_pad, K] @ W_e.T -> fp32.

    ``w16``: [E, N, K] fp16 contiguous. ``offsets``: [E] inclusive row offsets, each a
    multiple of ``config``'s BLOCK_M; rows from ``offsets[-1]`` on are not computed
    (left uninitialized). Bitwise equal to ``grouped_gemm`` and to cuBLAS without
    split-K; TMA-fed, CUDA-graph safe. ``rows_hint`` (host int, expected upper bound
    of ``offsets[-1]``) only sizes the persistent grid; any live row count is
    computed correctly whatever the hint.
    """
    global _ALLOCATOR_SET
    if not _ALLOCATOR_SET:  # on-device TMA descriptors need scratch memory
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device="cuda"
            )
        )
        _ALLOCATOR_SET = True
    bm, bn, bk, warps, stages = config
    m_pad, k = x16.shape
    e, n, _ = w16.shape
    assert m_pad % bm == 0 and x16.dtype == torch.float16 and x16.is_contiguous()
    assert w16.dtype == torch.float16 and w16.is_contiguous()
    out = torch.empty((m_pad, n), dtype=torch.float32, device=x16.device)
    n_tiles = triton.cdiv(n, bn)
    m_programs = _persistent_programs(
        m_pad // bm,
        None if rows_hint is None else triton.cdiv(rows_hint, bm),
        x16.device,
        per_program=n_tiles,
    )
    _grouped_gemm_fp16w_kernel[(m_programs, n_tiles)](
        x16,
        w16,
        offsets,
        out,
        m_pad,
        n,
        k,
        e,
        E_POW2=triton.next_power_of_2(e),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _ensure_fp16_expert_weights(module: Any) -> None:
    """Hold the engine's expert weights as fp16 [E, N, K] buffers.

    Like MCore's ``_build_concatenated_weights`` (each expert parameter's ``.data``
    becomes a view into one contiguous buffer, so refit's ``copy_`` into the
    parameter lands there), but the buffer is fp16: same bytes as bf16, and
    bf16 -> fp16 is exact in fp16's normal range, the same conversion the training
    path and the register-converting kernel apply.
    """
    if getattr(module, "_fp16_expert_weights", False):
        return
    for name, linear in (
        ("_fc1_weight", module.linear_fc1),
        ("_fc2_weight", module.linear_fc2),
    ):
        params = [
            getattr(linear, f"weight{i}") for i in range(module.num_local_experts)
        ]
        buf = torch.empty(
            (len(params), *params[0].shape),
            dtype=torch.float16,
            device=params[0].device,
        )
        for i, p in enumerate(params):
            buf[i].copy_(p.data)
            p.data = buf[i]
        if name in module._buffers:
            module._buffers[name] = buf
        else:
            module.register_buffer(name, buf, persistent=False)
    module._concatenated_weights_built = True
    module._fp16_expert_weights = True


def _grouped_gemm_reference(x16, weights, offsets):
    """CPU stand-in for tests: the same per-row products, fp32 accumulation."""
    n = weights[0].shape[0]
    out = torch.zeros((x16.shape[0], n), dtype=torch.float32)
    start = 0
    for w, end in zip(weights, offsets.tolist()):
        out[start:end] = x16[start:end].float() @ w.to(torch.float16).float().t()
        start = end
    return out


def _swiglu_eager(h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    gate, up = torch.chunk(h.float(), 2, dim=-1)
    return (gate / (1.0 + torch.exp(-gate))) * up * probs.float().unsqueeze(-1)


def swiglu_probs_fwd(
    h: torch.Tensor,
    probs: torch.Tensor,
    out_dtype=torch.float32,
    valid_rows=None,
    rows_hint: Optional[int] = None,
    prob_scale: Optional[float] = None,
):
    """SiLU(gate) * up * prob in fp32 (one Triton kernel on CUDA), cast to out_dtype.

    ``valid_rows`` (engine, device scalar): rows from it on are not computed
    (left uninitialized); ``rows_hint`` / ``prob_scale``: see ``relu2_probs_fwd``.
    """
    if not h.is_cuda:
        p = probs if prob_scale is None else probs.float() * prob_scale
        return _swiglu_eager(h, p).to(out_dtype)
    rows, two_f = h.shape
    f = two_f // 2
    out = torch.empty((rows, f), dtype=out_dtype, device=h.device)
    block = min(triton.next_power_of_2(f), 1024)
    col_blocks = triton.cdiv(f, block)
    programs = (
        rows
        if valid_rows is None
        else _persistent_programs(rows, rows_hint, h.device, col_blocks)
    )
    _swiglu_probs_kernel[(programs, col_blocks)](
        h.contiguous(),
        probs.float().contiguous(),
        out,
        f,
        rows,
        out if valid_rows is None else valid_rows,
        1.0 if prob_scale is None else prob_scale,
        BLOCK=block,
        HAS_VALID=valid_rows is not None,
        HAS_SCALE=prob_scale is not None,
    )
    return out


class SwigluProbs(torch.autograd.Function):
    """Training: the engine's kernel forward (fp32 out), eager fp32 backward."""

    @staticmethod
    def forward(ctx, h, probs):  # noqa: D102
        ctx.save_for_backward(h, probs)
        return swiglu_probs_fwd(h, probs, torch.float32)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        h, probs = ctx.saved_tensors
        with torch.enable_grad():
            h_ = h.detach().float().requires_grad_()
            p_ = probs.detach().float().requires_grad_()
            y = _swiglu_eager(h_, p_)
            dh, dp = torch.autograd.grad(y, (h_, p_), grad.float())
        return dh.to(h.dtype), dp.to(probs.dtype)


def swiglu_probs(h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Autograd-aware SiLU(gate) * up * prob, fp32 output."""
    return SwigluProbs.apply(h, probs.reshape(-1))


def _relu2_eager(h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    r = torch.relu(h.float())
    return (r * r) * probs.float().unsqueeze(-1)


def relu2_probs_fwd(
    h: torch.Tensor,
    probs: torch.Tensor,
    out_dtype=torch.float32,
    valid_rows=None,
    rows_hint: Optional[int] = None,
    prob_scale: Optional[float] = None,
):
    """ReLU(h)^2 * prob in fp32 (non-gated squared-ReLU experts, Nemotron-H), cast to out_dtype.

    A row's value depends only on that row (batch invariant). Training runs the
    eager ops. ``valid_rows`` (the engine, a device scalar: the permuted rows'
    end, ``offsets[-1]``): one Triton kernel with the eager ops' order and
    rounding (bitwise the same values) over the rows below ``valid_rows`` only,
    on a persistent grid sized by ``rows_hint`` (host bound of the live rows; a
    smaller hint loops more, it never skips rows). The engine's CUDA-graph buffers
    hold the worst-case row count under expert parallelism (every EP rank's
    tokens x top-k); rows past ``valid_rows`` are never read (the grouped GEMM
    stops at ``offsets[-1]``, unpermute reads mapped rows), so they are left
    uninitialized. ``prob_scale``: the probability is first multiplied by it in
    fp32 (the down-projection prescale), as ``probs * prob_scale`` would.
    """
    if valid_rows is None or not h.is_cuda:
        p = probs if prob_scale is None else probs.float() * prob_scale
        return _relu2_eager(h, p).to(out_dtype)
    rows, f = h.shape
    out = torch.empty((rows, f), dtype=out_dtype, device=h.device)
    block = min(triton.next_power_of_2(f), 1024)
    col_blocks = triton.cdiv(f, block)
    _relu2_probs_kernel[
        (_persistent_programs(rows, rows_hint, h.device, col_blocks), col_blocks)
    ](
        h.contiguous(),
        probs.reshape(-1).float().contiguous(),
        out,
        f,
        valid_rows,
        1.0 if prob_scale is None else prob_scale,
        BLOCK=block,
        HAS_SCALE=prob_scale is not None,
    )
    return out


class Relu2Probs(torch.autograd.Function):
    """Training: the engine's forward (fp32 out), eager fp32 backward."""

    @staticmethod
    def forward(ctx, h, probs):  # noqa: D102
        ctx.save_for_backward(h, probs)
        return relu2_probs_fwd(h, probs, torch.float32)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        h, probs = ctx.saved_tensors
        with torch.enable_grad():
            h_ = h.detach().float().requires_grad_()
            p_ = probs.detach().float().requires_grad_()
            y = _relu2_eager(h_, p_)
            dh, dp = torch.autograd.grad(y, (h_, p_), grad.float())
        return dh.to(h.dtype), dp.to(probs.dtype)


def relu2_probs(h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """Autograd-aware ReLU(h)^2 * prob, fp32 output."""
    return Relu2Probs.apply(h, probs.reshape(-1))


def expert_activation(config: Any) -> str:
    """'swiglu' (gated SiLU) or 'relu2' (non-gated squared ReLU); ValueError otherwise."""
    act = getattr(getattr(config, "activation_func", None), "__name__", "silu")
    gated = bool(getattr(config, "gated_linear_unit", False))
    if gated and act == "silu":
        return "swiglu"
    if not gated and act == "squared_relu":
        return "relu2"
    raise ValueError(
        f"fp16 MoE experts: activation_func={act} gated_linear_unit={gated} not built yet"
    )


def _act_probs(
    module: Any,
    h,
    probs,
    out_dtype=None,
    valid_rows=None,
    rows_hint: Optional[int] = None,
    prob_scale: Optional[float] = None,
):
    """The expert activation times the routing probability (fp32 math).

    ``valid_rows``: the engine's permuted-row end (device scalar); rows past it
    are padding and are not computed. ``rows_hint`` / ``prob_scale``: see
    ``relu2_probs_fwd``.
    """
    cfg = getattr(module, "config", None)
    kind = "swiglu" if cfg is None else expert_activation(cfg)
    if out_dtype is None:  # training: autograd
        return swiglu_probs(h, probs) if kind == "swiglu" else relu2_probs(h, probs)
    fwd = swiglu_probs_fwd if kind == "swiglu" else relu2_probs_fwd
    return fwd(
        h,
        probs,
        out_dtype,
        valid_rows=valid_rows,
        rows_hint=rows_hint,
        prob_scale=prob_scale,
    )


def unpermute_fp64(
    expert_output: torch.Tensor,
    inverse_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    tokens_hint: Optional[int] = None,
    y_scale: Optional[float] = None,
) -> torch.Tensor:
    """Per-token sum of local-expert rows in fp64, ascending local expert, fp32 out.

    Rows from ``valid_tokens`` on are not written. ``tokens_hint`` (host bound of
    ``valid_tokens``) only sizes the persistent grid. ``y_scale``: every expert
    row is first multiplied by it in fp32, as ``expert_output * y_scale`` would.
    """
    _, hidden_dim = expert_output.shape
    num_tokens, num_local_experts = inverse_map.shape
    if out is None:
        out = torch.empty(
            num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device
        )
    block_h = min(triton.next_power_of_2(hidden_dim), 1024)
    h_blocks = triton.cdiv(hidden_dim, block_h)
    grid = (
        _persistent_programs(num_tokens, tokens_hint, expert_output.device, h_blocks),
        h_blocks,
    )
    _unpermute_fp64_kernel[grid](
        expert_output,
        inverse_map,
        valid_tokens,
        out,
        hidden_dim,
        1.0 if y_scale is None else y_scale,
        num_local_experts,
        BLOCK_H=block_h,
        HAS_SCALE=y_scale is not None,
    )
    return out


def permute_live(
    x16: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    valid_tokens: torch.Tensor,
    block_m: int,
    tokens_hint: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group the valid tokens' local (token, expert) pairs by expert (CUDA-graph safe).

    MCore's ``permute_tokens(..., alignment=block_m, row_alignment=block_m,
    return_batch_invariant_inverse_map=True)`` contract, with the work bounded by
    the valid tokens instead of the buffer: the same counts and BLOCK_M-aligned
    segment offsets (MCore's kernels), the same output buffer shapes, and one
    persistent kernel that copies the rows and writes the inverse-map rows of the
    valid tokens only (rows of tokens >= ``valid_tokens`` are never read). A
    pair's row within its expert segment is claimed atomically, as in MCore: the
    order inside a segment is arbitrary and the outputs do not depend on it (the
    grouped GEMM is row-independent and unpermute sums through the inverse map
    in expert order).

    Returns (permuted rows [R, H], permuted probs [R] fp32, inclusive offsets [E]
    int32, inverse map [max_tokens, E] int32).
    """
    from megatron.core.inference.moe.permute import (
        compute_expert_offsets,
        compute_local_tokens_per_expert,
    )

    max_tokens, hidden_dim = x16.shape
    topk = probs.shape[1]
    counts = compute_local_tokens_per_expert(
        routing_map,
        local_expert_start,
        num_local_experts,
        valid_tokens,
        persistent=True,
    )
    exclusive, inclusive = compute_expert_offsets(counts, alignment=block_m)
    rows = max_tokens * min(topk, num_local_experts) + block_m * num_local_experts
    rows = triton.cdiv(rows, block_m) * block_m
    xp = torch.empty((rows, hidden_dim), dtype=x16.dtype, device=x16.device)
    pp = torch.empty(rows, dtype=probs.dtype, device=x16.device)
    inverse_map = torch.empty(
        (max_tokens, num_local_experts), dtype=torch.int32, device=x16.device
    )
    _permute_live_kernel[(_persistent_programs(max_tokens, tokens_hint, x16.device),)](
        x16,
        probs,
        routing_map,
        xp,
        pp,
        inverse_map,
        exclusive,
        valid_tokens,
        hidden_dim,
        local_expert_start,
        TOPK=topk,
        E=num_local_experts,
        E_POW2=triton.next_power_of_2(num_local_experts),
        BLOCK_H=min(triton.next_power_of_2(hidden_dim), 1024),
    )
    return xp, pp, inclusive, inverse_map


# ---------------------------------------------------------------------------
# Training: autograd grouped linear over host-side expert splits.
# ---------------------------------------------------------------------------


def _aligned_rows(m_splits: list[int], block_m: int, device) -> tuple:
    """(destination row of every input row, inclusive aligned offsets, padded rows)."""
    dest, offs = [], []
    start = 0
    for m in m_splits:
        dest.append(torch.arange(start, start + m, dtype=torch.int64))
        start += -(-m // block_m) * block_m
        offs.append(start)
    dest_t = torch.cat(dest) if dest else torch.zeros(0, dtype=torch.int64)
    m_pad = max(start, block_m)
    return dest_t.to(device), torch.tensor(offs, dtype=torch.int64).to(device), m_pad


# Training expert GEMMs: one Triton launch per grouped linear (``segment``) or one
# cuBLAS GEMM per expert (``cublas``). Both give the same bits (exact 16-bit
# products, the reduction accumulated in order in fp32, no split-K); they differ in
# cost. cuBLAS costs ~20-60 us of host time per GEMM (PyTorch re-parses
# CUBLAS_WORKSPACE_CONFIG on every call under the batch-invariant :0:0 setting), so
# at few rows per expert a forward over many local experts is launch-bound (Nano
# EP1: 128 experts x 2 linears x #MoE layers GEMMs); at many rows per expert
# cuBLAS's tiles are faster on the GPU. ``auto`` takes the segment kernel below
# _SEGMENT_MAX_ROWS rows per expert (crossover ~1000-1500 rows per expert at Nano
# expert shapes, thundersync_work/research/callcost/moe_bench_seg*.json).
# NEMO_RL_FP16_MOE_TRAIN_GEMM overrides it (auto | segment | cublas).
_SEGMENT_MAX_ROWS = 1024
# (rows per expert bound, (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages))
_SEGMENT_CONFIGS: tuple[tuple[int, TileConfig], ...] = (
    (16, (16, 64, 64, 4, 3)),
    (64, (32, 64, 64, 4, 3)),
    (256, (64, 64, 64, 4, 3)),
    (1 << 62, (128, 128, 64, 8, 3)),
)
# wgrad tiles (BLOCK_N, BLOCK_K, BLOCK_R, num_warps, num_stages) by rows per expert.
_WGRAD_CONFIGS: tuple[tuple[int, tuple], ...] = (
    (256, (64, 64, 32, 4, 3)),
    (1 << 62, (128, 128, 64, 8, 3)),
)


def _train_gemm_kind(rows: int, experts: int) -> str:
    mode = os.environ.get("NEMO_RL_FP16_MOE_TRAIN_GEMM", "auto")
    if mode in ("segment", "cublas"):
        return mode
    return "segment" if rows <= _SEGMENT_MAX_ROWS * max(experts, 1) else "cublas"


def _segment_config(rows: int, experts: int) -> TileConfig:
    per = rows / max(experts, 1)
    for bound, cfg in _SEGMENT_CONFIGS:
        if per <= bound:
            return cfg
    return _SEGMENT_CONFIGS[-1][1]


def _segment_gemm(
    a: torch.Tensor,
    weights: tuple[torch.Tensor, ...],
    counts: torch.Tensor,
    transpose: bool,
    b_fp16: bool,
) -> torch.Tensor:
    """out[r] = a[r] @ (W_e^T if transpose else W_e) for the rows r of expert e (fp32).

    ``a`` [M, R] 16-bit contiguous, rows grouped by expert, ``counts`` [E] int32 on
    a's device (rows per expert, summing to M). ``b_fp16``: round the weight to fp16
    in registers (the forward); else the weight's dtype (dgrad on bf16 weights).
    """
    base, w_offs = _weight_table(list(weights))
    n, k = base.shape  # each weight [N, K]
    m, r = a.shape
    cols, (s_r, s_c) = (n, (1, k)) if transpose else (k, (k, 1))
    assert r == (k if transpose else n) and a.is_contiguous()
    e = len(weights)
    bm, bn, bk, warps, stages = _segment_config(m, e)
    out = torch.empty((m, cols), dtype=torch.float32, device=a.device)
    if m == 0:
        return out
    grid = (triton.cdiv(m, bm) + e, triton.cdiv(cols, bn))
    _segment_gemm_kernel[grid](
        a,
        base,
        w_offs,
        counts,
        out,
        cols,
        r,
        s_r,
        s_c,
        e,
        E_POW2=triton.next_power_of_2(e),
        B_FP16=b_fp16,
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _segment_wgrad(
    g: torch.Tensor, x: torch.Tensor, counts: torch.Tensor, e: int
) -> torch.Tensor:
    """[E, N, K] fp32: g_e^T @ x_e per expert (``g`` [M, N], ``x`` [M, K] bf16)."""
    m, n = g.shape
    k = x.shape[1]
    out = torch.empty((e, n, k), dtype=torch.float32, device=g.device)
    per = m / max(e, 1)
    bn, bk, br, warps, stages = next(c for bound, c in _WGRAD_CONFIGS if per <= bound)
    _segment_wgrad_kernel[(triton.cdiv(n, bn), triton.cdiv(k, bk), e)](
        g,
        x,
        counts,
        out,
        n,
        k,
        e,
        E_POW2=triton.next_power_of_2(e),
        BLOCK_N=bn,
        BLOCK_K=bk,
        BLOCK_R=br,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def _grouped_backward_reference(ctx, g, x16, weights):
    """Per-expert backward (CPU tests): dgrad / wgrad of each expert on bf16 casts."""
    dx = torch.empty(x16.shape, dtype=torch.float32, device=x16.device)
    wgrads = []
    start = 0
    for w, m in zip(weights, ctx.m_splits):
        g_e = g[start : start + m]
        x_e = x16[start : start + m].to(torch.bfloat16)
        if ctx.needs_input_grad[0]:
            dx[start : start + m] = _mm32(g_e, w.to(torch.bfloat16))
        wgrads.append(_mm32(g_e.t(), x_e).to(w.dtype))
        start += m
    dx = dx.to(ctx.x_dtype).view(ctx.x_shape) if ctx.needs_input_grad[0] else None
    return (dx, None, *wgrads)


class GroupedFp16Linear(torch.autograd.Function):
    """y[rows of expert e] = fp16(x) @ fp16(W_e).T in fp32 (the shared kernel).

    Backward: per expert, dgrad = g @ W_e and wgrad = g.T @ x on bf16 casts,
    fp32 accumulation; dgrad in x's dtype, wgrad in the weight dtype.
    """

    @staticmethod
    def forward(ctx, x, m_splits, *weights):  # noqa: D102
        k = x.shape[-1]
        x16 = x.reshape(-1, k).to(torch.float16)
        splits = list(m_splits)
        counts = None
        if x16.is_cuda and _train_gemm_kind(x16.shape[0], len(splits)) == "segment":
            counts = torch.tensor(splits, dtype=torch.int32).to(
                x16.device, non_blocking=True
            )
            out = _segment_gemm(x16.contiguous(), weights, counts, True, True)
        elif x16.is_cuda:
            out = _expert_gemm_cublas(x16, splits, weights)
        else:
            cfg = select_config(sum(splits) / max(len(splits), 1))
            dest, offs, m_pad = _aligned_rows(splits, cfg[0], x.device)
            x_pad = torch.zeros((m_pad, k), dtype=torch.float16, device=x.device)
            x_pad.index_copy_(0, dest, x16)
            out = grouped_gemm(x_pad, list(weights), offs, cfg).index_select(0, dest)
        ctx.save_for_backward(x16, *weights)
        ctx.m_splits = splits
        ctx.counts = counts  # device rows per expert: the segment-kernel backward
        ctx.x_dtype = x.dtype
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        x16, *weights = ctx.saved_tensors
        splits = ctx.m_splits
        g = grad.reshape(-1, weights[0].shape[0]).to(torch.bfloat16)
        if not g.is_cuda:
            return _grouped_backward_reference(ctx, g, x16, weights)
        w0 = weights[0]
        same = all(w.dtype == w0.dtype and w.shape == w0.shape for w in weights)
        if ctx.counts is not None and same and w0.dtype == torch.bfloat16:
            # One dgrad and one wgrad launch for all experts (same bits as below).
            dx = None
            if ctx.needs_input_grad[0]:
                dx = _segment_gemm(g.contiguous(), weights, ctx.counts, False, False)
                dx = dx.to(ctx.x_dtype).view(ctx.x_shape)
            wg32 = _segment_wgrad(
                g.contiguous(), x16.to(torch.bfloat16), ctx.counts, len(weights)
            )
            return (dx, None, *wg32.to(w0.dtype).unbind(0))
        # Per expert: the same cuBLAS GEMMs as before, on views of whole-tensor
        # casts and into views of whole-tensor outputs (one bf16 cast of x, one
        # bf16 cast of all wgrads), so the host issues one GEMM per expert and
        # nothing else. Bitwise the same as casting / rounding expert by expert.
        x_bf = x16.to(torch.bfloat16)
        g_e, x_e = g.split(splits), x_bf.split(splits)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = torch.empty(x16.shape, dtype=torch.float32, device=x16.device)
            for gi, w, di in zip(g_e, weights, dx.split(splits)):
                if gi.shape[0]:
                    torch.mm(gi, w.to(torch.bfloat16), out_dtype=torch.float32, out=di)
            dx = dx.to(ctx.x_dtype).view(ctx.x_shape)
        if not same:
            return (
                dx,
                None,
                *(
                    _mm32(gi.t(), xi).to(w.dtype)
                    for gi, xi, w in zip(g_e, x_e, weights)
                ),
            )
        wg32 = torch.empty(
            (len(weights), *w0.shape), dtype=torch.float32, device=g.device
        )
        for gi, xi, wi in zip(g_e, x_e, wg32.unbind(0)):
            if gi.shape[0]:
                torch.mm(gi.t(), xi, out_dtype=torch.float32, out=wi)
            else:
                wi.zero_()  # an expert without rows: zero gradient
        return (dx, None, *wg32.to(w0.dtype).unbind(0))


# fp16 copies of the training model's expert weights for the cuBLAS path (cuBLAS needs both operands in
# fp16; the Triton kernel converts in registers instead), stacked [E, N, K] per expert
# linear for the grouped GEMM, cast just in time and kept for the most recent
# _CACHE_LINEARS expert linears only: fc1 + fc2 of the two most recent MoE layers, so a
# rank holds at most two layers' fp16 expert weights (Nemotron-3-Nano EP=8: 2 x 0.3 GiB)
# instead of 2 B per local expert parameter (7.4 GB at EP=8, 29.4 GB at EP=2). A layer's
# fc1 and fc2 are cast once per forward (the fp16 values are the same, so the GEMMs are
# bitwise the same); evicted buffers are reused. Valid until the weights change: the
# policy worker calls invalidate_weight_cache() after every optimizer step.
# ``set_weight_cache(False)`` casts one expert at a time instead (no stacked buffer).
_CACHE_LINEARS = 4
_WEIGHT_CACHE: "collections.OrderedDict[tuple[int, ...], tuple[tuple[int, ...], torch.Tensor]]" = collections.OrderedDict()
# Sticky rounding's anchor forward runs on transient theta_n weights (streamed
# from host): those must not enter the cache (ids and pointers get reused).
# ``fc2_input``: sticky rounding's hook on the expert fc2 input rows.
_HOOKS: dict[str, Any] = {"cache": True, "fc2_input": None}


def set_fc2_input_hook(hook: Optional[Any]) -> None:
    """Install (None: remove) sticky rounding's expert fc2-input hook."""
    _HOOKS["fc2_input"] = hook


def set_weight_cache(enabled: bool) -> None:
    """Process default for stacking fp16 expert weights (see ``_WEIGHT_CACHE``)."""
    _HOOKS["cache"] = enabled
    if not enabled:
        _WEIGHT_CACHE.clear()


@contextlib.contextmanager
def weight_cache_disabled() -> Iterator[None]:
    """Expert GEMMs inside the block convert their weights without caching."""
    prev = _HOOKS["cache"]
    _HOOKS["cache"] = False
    try:
        yield
    finally:
        _HOOKS["cache"] = prev


def invalidate_weight_cache() -> None:
    """Drop cached fp16 expert weights (call after the weights change)."""
    _WEIGHT_CACHE.clear()


def _stacked_fp16(weights: tuple[torch.Tensor, ...]) -> torch.Tensor:
    key = tuple(id(w) for w in weights)
    ptrs = tuple(w.data_ptr() for w in weights)
    hit = _WEIGHT_CACHE.get(key)
    if hit is not None and hit[0] == ptrs:
        _WEIGHT_CACHE.move_to_end(key)
        return hit[1]
    _WEIGHT_CACHE.pop(key, None)
    shape = (len(weights), *weights[0].shape)
    w16 = None
    if len(_WEIGHT_CACHE) >= _CACHE_LINEARS:
        # Reuse the least recent buffer when it fits (same stream: its readers
        # are ordered before this cast).
        _, (_, old) = _WEIGHT_CACHE.popitem(last=False)
        if old.shape == shape and old.device == weights[0].device:
            w16 = old
    if w16 is None:
        w16 = torch.empty(shape, dtype=torch.float16, device=weights[0].device)
    # One multi-tensor cast (~E/4 launches) instead of one copy per expert.
    torch._foreach_copy_(list(w16.unbind(0)), [w.detach() for w in weights])
    _WEIGHT_CACHE[key] = (ptrs, w16)
    return w16


def _expert_gemm_cublas(
    x16: torch.Tensor, m_splits: list[int], weights: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    """Training-side expert GEMMs on cuBLAS: fp16 x fp16(W_e).T -> fp32, one GEMM per expert.

    Bitwise equal to ``grouped_gemm`` (the engine's Triton kernel) for every row:
    both form exact fp16 products and accumulate K serially in fp32, and cuBLAS
    without split-K (te_native workspace starvation) does not depend on M. At
    500B-MoE expert shapes (8 experts, K 7168, 1024 rows each) the GPU time is
    0.30 ms vs 1.21 ms for the Triton kernel. (Transformer Engine's grouped GEMM
    gives the same bits at 0.35 ms but syncs the host every call.)
    """
    # Uncached (transient weights): one expert's fp16 copy at a time.
    w16 = _stacked_fp16(weights) if _HOOKS["cache"] else None
    m_total, n = x16.shape[0], weights[0].shape[0]
    out = torch.empty((m_total, n), dtype=torch.float32, device=x16.device)
    ws = w16.unbind(0) if w16 is not None else weights
    for x_e, o_e, w in zip(x16.split(m_splits), out.split(m_splits), ws):
        if x_e.shape[0]:
            if w16 is None:
                w = w.detach().to(torch.float16)
            torch.mm(x_e, w.t(), out_dtype=torch.float32, out=o_e)
    return out


def _mm32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.is_cuda:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def grouped_linear(x: torch.Tensor, m_splits: list[int], weights: list[torch.Tensor]):
    """Autograd-aware grouped expert linear (training)."""
    return GroupedFp16Linear.apply(x, list(m_splits), *weights)


def _expert_weights(linear: Any) -> list[torch.Tensor]:
    return [getattr(linear, f"weight{i}") for i in range(linear.num_gemms)]


# ---------------------------------------------------------------------------
# Patched module forwards.
# ---------------------------------------------------------------------------


def _down_scale() -> int:
    from nemo_rl.models.megatron import fp16_gemm

    return fp16_gemm._STATE["down_scale"]


def training_experts_forward(
    module: Any, x, tokens_per_expert, permuted_probs
) -> tuple:
    """TEGroupedMLP forward (training and the training model's no-grad forwards)."""
    if isinstance(tokens_per_expert, torch.Tensor):
        tokens_per_expert = tokens_per_expert.tolist()
    m_splits = [int(m) for m in tokens_per_expert]
    s = _down_scale()
    probs = permuted_probs.reshape(-1)
    if s:  # down-projection prescale: a * 2^-s via the routing probabilities
        probs = probs * (2.0**-s)
    h = grouped_linear(x, m_splits, _expert_weights(module.linear_fc1))
    a = _act_probs(module, h, probs)
    if _HOOKS["fc2_input"] is not None:
        a = _HOOKS["fc2_input"](module, a, m_splits)
    y = grouped_linear(a, m_splits, _expert_weights(module.linear_fc2))
    if s:
        y = y * (2.0**s)
    return y, None


def engine_experts(
    x16: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    local_expert_start: int,
    ep_size: int,
    activation: str,
    tokens_hint: Optional[int] = None,
    down_scale: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The engine's routed-expert output of the gathered tokens (CUDA-graph safe).

    ``x16`` [T, K] fp16 / ``probs`` [T, topk] fp32 / ``routing_map`` [T, topk]:
    the gathered tokens, of which the first ``valid_tokens`` (device scalar) are
    live. ``fc1_weight`` / ``fc2_weight``: this rank's [E, N, K] fp16 experts.
    Returns [T, H] fp32 (rows >= ``valid_tokens`` unwritten): per token, the sum of
    its local experts' prob-weighted outputs in fp64, ascending expert, rounded to
    fp32. ``tokens_hint`` (host): expected upper bound of ``valid_tokens``, the
    CUDA-graph bucket's tokens x EP (None: the buffer's T). It picks the tile and
    sizes the persistent grids; no row's bits depend on it (any tile gives the same
    bits, and every kernel loops to its device-side live count).
    """
    num_local_experts = fc1_weight.shape[0]
    topk = routing_map.shape[1]
    hint = x16.shape[0] if tokens_hint is None else min(tokens_hint, x16.shape[0])
    # Expected rows per local expert: the live tokens x top-k spread over every expert.
    cfg = select_engine_config(hint * topk / (num_local_experts * ep_size))
    rows_hint = live_rows_bound(hint, topk, num_local_experts, cfg[0])
    xp, pp, offs, inverse_map = permute_live(
        x16,
        probs,
        routing_map,
        local_expert_start,
        num_local_experts,
        valid_tokens,
        cfg[0],
        hint,
    )
    # down_scale s: the prescale as training_experts_forward applies it (probs x 2^-s
    # before the activation, y x 2^s after fc2), inside the kernels, on live rows only.
    h = grouped_gemm_fp16w(xp, fc1_weight, offs, cfg, rows_hint)
    act = relu2_probs_fwd if activation == "relu2" else swiglu_probs_fwd
    a = act(
        h,
        pp,
        torch.float16,
        valid_rows=offs[-1:],
        rows_hint=rows_hint,
        prob_scale=2.0**-down_scale if down_scale else None,
    )
    y = grouped_gemm_fp16w(a, fc2_weight, offs, cfg, rows_hint)
    return unpermute_fp64(
        y,
        inverse_map,
        valid_tokens,
        out,
        tokens_hint=hint,
        y_scale=2.0**down_scale if down_scale else None,
    )


def inference_experts_forward(
    module: Any, hidden_states, permuted_probs, routing_map
) -> tuple:
    """InferenceGroupedMLP forward in the engine (CUDA-graph safe)."""
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    _ensure_fp16_expert_weights(module)
    x16 = (
        hidden_states.view(torch.float16)
        if hidden_states.dtype == torch.bfloat16
        else (hidden_states.to(torch.float16))
    )
    cfg = getattr(module, "config", None)
    out = None
    if module._nvls_dispatcher:
        out = NVLSAllGatherVDispatcher._get_rsv_tensor()
    y = engine_experts(
        x16,
        permuted_probs,
        routing_map,
        InferenceAllGatherDispatcherBase._valid_tokens(),
        module._fc1_weight,
        module._fc2_weight,
        module.ep_group.rank() * module.num_local_experts,
        module.ep_group.size(),
        "swiglu" if cfg is None else expert_activation(cfg),
        # The dispatcher's host estimate of the gathered tokens: this rank's
        # CUDA-graph bucket (or eager token count) x EP.
        tokens_hint=InferenceAllGatherDispatcherBase._get_host_valid_tokens_estimate(),
        down_scale=_down_scale(),
        out=out,
    )
    return y, None


def validate_model(model_cfg: Any) -> list[str]:
    """MoE settings the fp16 expert path covers; returns what is unsupported."""
    bad = []
    if not getattr(model_cfg, "moe_grouped_gemm", False):
        bad.append("moe_grouped_gemm=False")
    try:
        expert_activation(model_cfg)
    except ValueError:
        act = getattr(model_cfg, "activation_func", None)
        bad.append(
            f"expert activation {getattr(act, '__name__', act)} with "
            f"gated_linear_unit={getattr(model_cfg, 'gated_linear_unit', False)}"
        )
    if getattr(model_cfg, "moe_mlp_glu_interleave_size", None):
        bad.append("moe_mlp_glu_interleave_size")
    if getattr(model_cfg, "add_bias_linear", False):
        bad.append("expert biases")
    # Shared experts (Nemotron-H) are a dense MLP on the fp16 GEMM path (TE linears),
    # added to the routed output in fp32: allowed; engine exactness not measured yet.
    if getattr(model_cfg, "moe_apply_probs_on_input", False):
        bad.append("moe_apply_probs_on_input")
    if getattr(model_cfg, "moe_latent_size", None):
        bad.append("moe_latent_size")
    if getattr(model_cfg, "use_te_activation_func", False):
        bad.append("use_te_activation_func")
    if (getattr(model_cfg, "expert_tensor_parallel_size", 1) or 1) > 1:
        bad.append("expert_tensor_parallel_size>1")
    if getattr(model_cfg, "moe_token_dispatcher_type", "alltoall") != "alltoall":
        bad.append(f"moe_token_dispatcher_type={model_cfg.moe_token_dispatcher_type}")
    return bad
