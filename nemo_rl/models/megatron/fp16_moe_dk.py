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
"""The engine's routed-expert path for squared-ReLU experts (Nemotron-H): same bits, fewer bytes and launches.

``fp16_moe.engine_experts`` runs every relu2 call (decode and prefill) here; swiglu experts and the fc2 hi/lo
precision raise keep the generic chain there. Four launches per MoE layer plus a memset:

- ``dk_route``: count + place (``_dk_count_kernel``, ``_dk_place_kernel``): the live local (token, slot) pairs get a
  row of their expert's BLOCK_M-aligned segment (MCore ``compute_expert_offsets`` offsets), a row -> token map and a
  [tokens, topk] slot map; no row copy. Pairs of probability 0 (sparse cross-fade) are dropped.
- ``dk_grouped_gemm`` (``_dk_gemm_kernel``), fc1 and fc2:
  - a 1-D persistent grid over the live (row tile, column tile) work items;
  - optionally the operands swapped (``SWAP``): the MMA computes W_tile @ x_tile.T, the expert's output columns on
    the MMA's M side and the few rows of a decode step on its N side;
  - fc1 prologue gathers the token rows through the row -> token map (``GATHER``); fc1 epilogue applies
    ReLU(h)^2 * p in ``_relu2_probs_kernel``'s op order and casts once to fp16 (the fp32 fc1 output never reaches HBM).
- ``dk_unpermute_slots`` (``_dk_unpermute_slots_kernel``): the fp64 combine in ascending local expert over the
  token's computed slots (LIMBS 0 / 1 / 2, ROUND16, y_scale as ``unpermute_fp64``).

Bits: every output element is the same function of the same operands as the generic chain
(``permute_live -> grouped_gemm_fp16w -> relu2_probs_fwd -> grouped_gemm_fp16w -> unpermute_fp64``).
- GEMM: exact fp16 products, K accumulated in fp32 in ascending order in MMA k-groups, no split-K. The tile and the
  operand swap only change which program / MMA operand slot computes an element: every tile of the research/dk1
  sweeps (Super and Ultra shapes) is bitwise equal to ``grouped_gemm_fp16w`` and to per-expert cuBLAS (the learner).
- Epilogue: relu, r * r, * p (two multiplies, nothing to contract into an FMA), one cast.
- Combine: identical fp64 sum order (ascending local expert).
- A pair's row inside its segment is claimed atomically; no value depends on it.
CUDA graphs: static launch shapes (the host hint only sizes grids), device-side live counts, no host sync.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

if HAVE_TRITON:
    from nemo_rl.models.megatron.ep_invariant_combine import _finalize as _eic_finalize
    from nemo_rl.models.megatron.ep_invariant_combine import _limbs as _eic_limbs


# (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, SWAP, programs per SM)
DkConfig = tuple
EPI_NONE, EPI_RELU2 = 0, 1


if HAVE_TRITON:

    @triton.jit
    def _dk_gemm_kernel(
        x_ptr,  # [M_pad, K] fp16, rows grouped by expert, segments BLOCK_M-aligned
        w_ptr,  # [E, N, K] fp16
        offs_ptr,  # [E] inclusive BLOCK_M-aligned row offsets
        out_ptr,  # [M_pad, N] fp32 (EPI_NONE) or fp16 / fp32 (EPI_RELU2)
        p_ptr,  # [M_pad] fp32 permuted probabilities (EPI_RELU2)
        src_ptr,  # [M_pad] int32 token row of each permuted row, -1 = padding (GATHER)
        p_scale,
        M,  # rows of x (GATHER: the token rows)

        N,  # output columns: weight rows [n_off, n_off + N) of each expert
        K,
        E,
        W_N,  # weight rows per expert (N = W_N, n_off = 0 without a column slice)
        n_off,
        E_POW2: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        SWAP: tl.constexpr,
        EPI: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        GATHER: tl.constexpr,
    ):
        idx = tl.arange(0, E_POW2)
        ends = tl.load(offs_ptr + idx, mask=idx < E, other=2147483647).to(tl.int64)
        live_tiles = tl.load(offs_ptr + E - 1).to(tl.int32) // BLOCK_M
        n_tiles = tl.cdiv(N, BLOCK_N)
        work = live_tiles * n_tiles
        xd = tl.make_tensor_descriptor(
            x_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_M, BLOCK_K]
        )
        wd = tl.make_tensor_descriptor(
            w_ptr, shape=[E * W_N, K], strides=[K, 1], block_shape=[BLOCK_N, BLOCK_K]
        )
        rm0 = tl.arange(0, BLOCK_M)
        rn0 = tl.arange(0, BLOCK_N)
        rk0 = tl.arange(0, BLOCK_K)
        for wi in tl.range(tl.program_id(0), work, tl.num_programs(0), num_stages=1):
            pid_m = wi // n_tiles
            pid_n = wi - pid_m * n_tiles
            start = pid_m * BLOCK_M
            expert = tl.sum((ends <= start).to(tl.int32), axis=0)
            # A tile past the slice's last column reads the next weight rows (or TMA zeros): those columns are
            # masked at the store; every stored column is the same dot product as without the slice.
            w_row = expert * W_N + n_off + pid_n * BLOCK_N
            if GATHER:  # the permute folded into the prologue: x rows read from the token rows
                src = tl.load(src_ptr + start + rm0)
                x_rows = x_ptr + tl.maximum(src, 0).to(tl.int64)[:, None] * K
                x_ok = (src >= 0)[:, None]
            # Any K: a ragged last K tile reads zeros past K (TMA out-of-bounds fill for W and the TMA-fed x, masked
            # loads for gathered x), the same zero-filled tail as the generic kernel's TMA tiles; the zero products add
            # nothing (bitwise vs the generic kernel and cuBLAS at every expert shape: test_fp16_moe_dk.py).
            if SWAP:
                acc_t = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    if GATHER:
                        xt = tl.load(
                            x_rows + (k0 + rk0)[None, :], mask=x_ok & ((k0 + rk0)[None, :] < K), other=0.0
                        )
                    else:
                        xt = xd.load([start, k0])
                    acc_t = tl.dot(wd.load([w_row, k0]), xt.T, acc_t)
                acc = tl.trans(acc_t)
            else:
                acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for k0 in range(0, K, BLOCK_K):
                    if GATHER:
                        xt = tl.load(
                            x_rows + (k0 + rk0)[None, :], mask=x_ok & ((k0 + rk0)[None, :] < K), other=0.0
                        )
                    else:
                        xt = xd.load([start, k0])
                    acc = tl.dot(xt, wd.load([w_row, k0]).T, acc)
            rm = (start + rm0).to(tl.int64)
            rn = pid_n * BLOCK_N + rn0
            if EPI == 1:  # _relu2_probs_kernel's ops in its order: relu, (r * r), * p, one cast
                p = tl.load(p_ptr + rm)
                if HAS_SCALE:
                    p = p * p_scale
                r = tl.maximum(acc, 0.0)
                val = (r * r) * p[:, None]
            else:
                val = acc
            tl.store(
                out_ptr + rm[:, None] * N + rn[None, :],
                val.to(out_ptr.dtype.element_ty),
                mask=rn[None, :] < N,
            )

    @triton.jit
    def _dk_count_kernel(
        routing_ptr,  # [max_tokens, TOPK] global expert ids (-1: none)
        probs_ptr,  # [max_tokens, TOPK] fp32
        valid_ptr,  # scalar int32: live tokens
        state_ptr,  # [2E + 1] int32, zeroed: counts [E] | cursors [E] | programs done [1]
        offs_ptr,  # [E] int32 out: inclusive ALIGN-aligned segment offsets (MCore compute_expert_offsets)
        src_ptr,  # [R] int32 out: -1 below offs[-1] (the placement kernel then writes the live rows)
        local_start,
        E,
        TOPK: tl.constexpr,
        E_POW2: tl.constexpr,
        ALIGN: tl.constexpr,
        BLOCK: tl.constexpr,
        SKIP_ZERO: tl.constexpr,
    ):
        # Count the live local pairs (grid-strided blocks of pairs); the last program to finish turns the counts into
        # the aligned prefix offsets and the placement cursors, and marks the used rows as padding.
        pairs = tl.load(valid_ptr).to(tl.int32) * TOPK
        for b in tl.range(tl.program_id(0) * BLOCK, pairs, tl.num_programs(0) * BLOCK):
            o = b + tl.arange(0, BLOCK)
            m = o < pairs
            lid = (tl.load(routing_ptr + o, mask=m, other=-1) - local_start).to(tl.int32)
            ok = m & (lid >= 0) & (lid < E)
            if SKIP_ZERO:
                ok = ok & (tl.load(probs_ptr + o, mask=m, other=0.0) != 0)
            tl.atomic_add(state_ptr + lid, tl.full([BLOCK], 1, tl.int32), mask=ok)
        tl.debug_barrier()
        ticket = tl.atomic_add(state_ptr + 2 * E, 1)
        if ticket == tl.num_programs(0) - 1:
            lids = tl.arange(0, E_POW2)
            lmask = lids < E
            c = tl.atomic_add(state_ptr + lids, tl.zeros([E_POW2], tl.int32), mask=lmask)
            c = tl.where(lmask, c, 0)
            al = tl.where(c > 0, ((c + ALIGN - 1) // ALIGN) * ALIGN, 0)
            inc = tl.cumsum(al, axis=0)
            tl.store(offs_ptr + lids, inc, mask=lmask)
            tl.store(state_ptr + E + lids, inc - al, mask=lmask)
            n_used = tl.sum(al, axis=0)
            for b in range(0, n_used, BLOCK):
                o = b + tl.arange(0, BLOCK)
                tl.store(src_ptr + o, tl.full([BLOCK], -1, tl.int32), mask=o < n_used)

    @triton.jit
    def _dk_place_kernel(
        routing_ptr,
        probs_ptr,
        valid_ptr,
        state_ptr,  # cursors at state_ptr + E (exclusive offsets, advanced atomically)
        src_ptr,  # [R] int32 out: token of each live permuted row
        pp_ptr,  # [R] fp32 out: permuted probabilities
        slot_ptr,  # [max_tokens, TOPK] int32 out: permuted row of each live (token, slot), -1 = not computed here
        local_start,
        E,
        TOPK: tl.constexpr,
        BLOCK: tl.constexpr,
        SKIP_ZERO: tl.constexpr,
    ):
        # A pair's row inside its segment is claimed atomically: any order gives the same bits (the GEMM is
        # row-independent and the combine sums in expert order).
        pairs = tl.load(valid_ptr).to(tl.int32) * TOPK
        for b in tl.range(tl.program_id(0) * BLOCK, pairs, tl.num_programs(0) * BLOCK):
            o = b + tl.arange(0, BLOCK)
            m = o < pairs
            lid = (tl.load(routing_ptr + o, mask=m, other=-1) - local_start).to(tl.int32)
            p = tl.load(probs_ptr + o, mask=m, other=0.0)
            ok = m & (lid >= 0) & (lid < E)
            if SKIP_ZERO:
                ok = ok & (p != 0)
            pos = tl.atomic_add(state_ptr + E + lid, tl.full([BLOCK], 1, tl.int32), mask=ok)
            tl.store(src_ptr + pos, o // TOPK, mask=ok)
            tl.store(pp_ptr + pos, p, mask=ok)
            tl.store(slot_ptr + o, tl.where(ok, pos, -1), mask=m)

    @triton.jit
    def _dk_unpermute_slots_kernel(
        expert_out_ptr,  # [rows, H] fp32
        slot_ptr,  # [max_tokens, TOPK] int32 permuted row or -1
        routing_ptr,  # [max_tokens, TOPK] global expert ids
        valid_tokens_ptr,
        output_ptr,  # this call's first output column (a column window of the token rows)
        hidden_dim,  # columns of expert_out (= the window's width)
        out_stride,  # output row stride (elements)
        lo_off,  # LIMBS 1: offset of a row's lo limbs from its hi limbs (elements)
        y_scale,
        local_start,
        TOPK: tl.constexpr,
        K_POW2: tl.constexpr,
        BLOCK_H: tl.constexpr,
        HAS_SCALE: tl.constexpr,
        LIMBS: tl.constexpr,
        ROUND16: tl.constexpr,
    ):
        # _unpermute_fp64_kernel's sum (ascending local expert, fp64) over the token's computed slots. Two slots of
        # one expert (not produced by the router): the later slot is the one summed, as the inverse map keeps it.
        valid_tokens = tl.load(valid_tokens_ptr).to(tl.int64)
        offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offsets < hidden_dim
        ks = tl.arange(0, K_POW2)
        mk = ks < TOPK
        for tok in tl.range(tl.program_id(0).to(tl.int64), valid_tokens, tl.num_programs(0)):
            pos = tl.load(slot_ptr + tok * TOPK + ks, mask=mk, other=-1)
            lid = (tl.load(routing_ptr + tok * TOPK + ks, mask=mk, other=0) - local_start).to(tl.int32)
            hit = pos >= 0
            n_hit = tl.sum(hit.to(tl.int32), axis=0)
            acc = tl.zeros([BLOCK_H], dtype=tl.float64)
            acc_lo = tl.zeros([BLOCK_H], dtype=tl.float64)
            prev = -1
            for _i in tl.range(0, n_hit):
                cur = tl.min(tl.where(hit & (lid > prev), lid, 2147483647), axis=0)
                if cur < 2147483647:
                    kk = tl.max(tl.where(hit & (lid == cur), ks, -1), axis=0)
                    row = tl.sum(tl.where(ks == kk, pos, 0), axis=0)
                    vals = tl.load(expert_out_ptr + row.to(tl.int64) * hidden_dim + offsets, mask=mask_h)
                    if ROUND16:
                        vals = vals.to(tl.float16).to(tl.float32)
                    if HAS_SCALE:
                        vals = vals * y_scale
                    if LIMBS == 0:
                        acc += vals.to(tl.float64)
                    else:
                        hi, lo = _eic_limbs(vals)
                        acc += hi
                        acc_lo += lo
                    prev = cur
            if LIMBS == 0:
                tl.store(output_ptr + tok * out_stride + offsets, acc.to(tl.float32), mask=mask_h)
            elif LIMBS == 1:
                tl.store(output_ptr + tok * out_stride + offsets, acc, mask=mask_h)
                tl.store(output_ptr + tok * out_stride + lo_off + offsets, acc_lo, mask=mask_h)
            else:
                tl.store(output_ptr + tok * out_stride + offsets, _eic_finalize(acc, acc_lo), mask=mask_h)


# Tiles by expected live rows per local expert, rpe = gathered tokens x topk / (local experts x EP) (bound, fc1 config,
# fc2 config); fc1 and fc2 share BLOCK_M (the segments are aligned to it). A deterministic function of the call's
# shape; any entry gives the same bits. From the research/dk1 sweeps (bitwise-checked, every tile): Super (h 1024,
# ffn 2688; 128 / 64 local experts) and Ultra (h 2048, ffn 5120; 32 / 16 local experts), 1 row to prefill-sized
# calls. Per point the pick is within ~3% of the swept best except Ultra fc2 at rpe 18.75 (+10%), and faster than
# the generic chain's tiles at every swept point (decode and prefill; DESIGN §20.D1/D3).
_DK_TABLE: tuple[tuple[float, DkConfig, DkConfig], ...] = (
    (1.0, (16, 64, 128, 4, 5, False, 2), (16, 64, 128, 4, 5, False, 2)),  # ~1 row / expert: few experts touched
    (20.0, (16, 128, 128, 4, 4, False, 2), (16, 64, 128, 4, 5, False, 2)),
    (50.0, (32, 128, 64, 4, 5, True, 2), (32, 128, 64, 4, 5, True, 2)),
    (110.0, (64, 128, 64, 4, 4, False, 2), (64, 128, 64, 4, 4, False, 2)),
    (1 << 62, (128, 256, 64, 8, 4, False, 1), (128, 256, 64, 8, 4, False, 1)),  # prefill
)
_OVERRIDE: dict[str, DkConfig] = {}




def select_dk_configs(rows_per_expert: float) -> tuple[DkConfig, DkConfig]:
    """(fc1, fc2) tiles at an expected number of live rows per local expert (never changes bits)."""
    if _OVERRIDE:
        return _OVERRIDE["fc1"], _OVERRIDE["fc2"]
    for bound, c1, c2 in _DK_TABLE:
        if rows_per_expert <= bound:
            return c1, c2
    return _DK_TABLE[-1][1], _DK_TABLE[-1][2]


_NUM_SMS: dict[int, int] = {}


def _num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _NUM_SMS:
        _NUM_SMS[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _NUM_SMS[idx]


def dk_grouped_gemm(
    x16: torch.Tensor,
    w16: torch.Tensor,
    offsets: torch.Tensor,
    config: DkConfig,
    rows_hint: Optional[int] = None,
    probs: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.float32,
    prob_scale: Optional[float] = None,
    src: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    n_slice: Optional[tuple[int, int]] = None,
) -> torch.Tensor:
    """x16 [M_pad, K] @ W_e.T per expert segment; with ``probs``: ReLU(.)^2 * probs, cast to ``out_dtype``.

    ``src`` (gather): x16 is the [T, K] token rows and permuted row r reads token ``src[r]`` (-1: padding, zeros);
    the output has ``src.shape[0]`` rows. ``out``: a contiguous [M_pad, N] output to write (a workspace).
    ``n_slice`` (n0, width): only output columns [n0, n0 + width) (weight rows n0.. of each expert); bitwise the same
    columns of the full product.

    Without ``probs``: bitwise ``fp16_moe.grouped_gemm_fp16w``. With ``probs``: bitwise
    ``relu2_probs_fwd(grouped_gemm_fp16w(...), probs, out_dtype, valid_rows=offsets[-1:], prob_scale=...)``.
    Rows from ``offsets[-1]`` on are not written. ``rows_hint`` only sizes the persistent grid.
    """
    from nemo_rl.models.megatron.fp16_moe import _set_triton_allocator

    _set_triton_allocator()
    bm, bn, bk, warps, stages, swap, per_sm = config
    m_x, k = x16.shape
    m_pad = m_x if src is None else src.shape[0]
    e, w_n, _ = w16.shape
    n_off, n = (0, w_n) if n_slice is None else n_slice
    assert 0 <= n_off and n_off + n <= w_n
    assert m_pad % bm == 0 and x16.dtype == torch.float16 and x16.is_contiguous()
    assert w16.dtype == torch.float16 and w16.is_contiguous()
    epi = EPI_NONE if probs is None else EPI_RELU2
    if epi == EPI_NONE:
        out_dtype = torch.float32
    if out is None:
        out = torch.empty((m_pad, n), dtype=out_dtype, device=x16.device)
    assert out.shape == (m_pad, n) and out.dtype == out_dtype and out.is_contiguous()
    n_tiles = triton.cdiv(n, bn)
    live = m_pad if rows_hint is None else min(m_pad, rows_hint)
    work = max(1, triton.cdiv(live, bm) * n_tiles)
    grid = (max(1, min(work, per_sm * _num_sms(x16.device))),)
    _dk_gemm_kernel[grid](
        x16,
        w16,
        offsets,
        out,
        out if probs is None else probs,
        offsets if src is None else src,
        1.0 if prob_scale is None else prob_scale,
        m_x,
        n,
        k,
        e,
        w_n,
        n_off,
        E_POW2=triton.next_power_of_2(e),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=bk,
        SWAP=bool(swap),
        EPI=epi,
        HAS_SCALE=prob_scale is not None,
        GATHER=src is not None,
        num_warps=warps,
        num_stages=stages,
    )
    return out


def dk_route(
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    valid_tokens: torch.Tensor,
    local_expert_start: int,
    num_local_experts: int,
    block_m: int,
    skip_zero_probs: bool = False,
    tokens_hint: Optional[int] = None,
    rows_buf: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Routing of the live (token, slot) pairs to BLOCK_M-aligned local-expert segments (no row copy).

    Returns (src [R] int32 token per permuted row (-1 padding), permuted probs [R] fp32, inclusive offsets [E]
    int32 (MCore ``compute_expert_offsets`` of the local counts), slot map [max_tokens, topk] int32 (permuted row
    of each pair, -1 when not computed here)). ``skip_zero_probs``: pairs of probability 0 are not computed (the
    sparse cross-fade engine dispatch: their +0 adds nothing to the combine). A memset and two launches.
    ``rows_buf`` (src [R'] int32, probs [R'] fp32, R' >= R a BLOCK_M multiple): write the permuted rows there.
    """
    max_tokens, topk = routing_map.shape
    e = num_local_experts
    rows = max_tokens * min(topk, e) + block_m * e
    rows = triton.cdiv(rows, block_m) * block_m
    dev = routing_map.device
    if rows_buf is None:
        src = torch.empty(rows, dtype=torch.int32, device=dev)
        pp = torch.empty(rows, dtype=torch.float32, device=dev)
    else:
        src, pp = rows_buf
        assert src.shape[0] >= rows and src.shape[0] % block_m == 0 and pp.shape[0] == src.shape[0]
    offs = torch.empty(e, dtype=torch.int32, device=dev)
    state = torch.zeros(2 * e + 1, dtype=torch.int32, device=dev)
    slots = torch.empty((max_tokens, topk), dtype=torch.int32, device=dev)
    probs = probs.float() if probs.dtype != torch.float32 else probs
    block = 512
    hint = max_tokens if tokens_hint is None else min(tokens_hint, max_tokens)
    grid = (max(1, min(triton.cdiv(hint * topk, block), 2 * _num_sms(dev))),)
    _dk_count_kernel[grid](
        routing_map, probs, valid_tokens, state, offs, src, local_expert_start, e,
        TOPK=topk, E_POW2=triton.next_power_of_2(e), ALIGN=block_m, BLOCK=block, SKIP_ZERO=skip_zero_probs,
    )
    _dk_place_kernel[grid](
        routing_map, probs, valid_tokens, state, src, pp, slots, local_expert_start, e,
        TOPK=topk, BLOCK=block, SKIP_ZERO=skip_zero_probs,
    )
    return src, pp, offs, slots


def dk_unpermute_slots(
    expert_output: torch.Tensor,
    slots: torch.Tensor,
    routing_map: torch.Tensor,
    local_expert_start: int,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    tokens_hint: Optional[int] = None,
    y_scale: Optional[float] = None,
    limbs: int = 0,
    round16: bool = False,
    col0: int = 0,
    full_hidden: Optional[int] = None,
) -> torch.Tensor:
    """``unpermute_fp64`` (same bits, same output contract) from the slot map of ``dk_route``.

    Column window: ``expert_output`` holds columns [col0, col0 + width) of the expert outputs of a ``full_hidden``
    wide row (default: all of them); they are written to the same columns of ``out`` ([T, full_hidden], LIMBS 1:
    [T, 2 x full_hidden], hi | lo), every element the same per-column sum.
    """
    from nemo_rl.models.megatron.fp16_moe import _persistent_programs

    _, hidden_dim = expert_output.shape
    full = hidden_dim if full_hidden is None else full_hidden
    assert 0 <= col0 and col0 + hidden_dim <= full
    num_tokens, topk = slots.shape
    if out is None:
        out = torch.empty(
            num_tokens,
            2 * full if limbs == 1 else full,
            dtype=torch.float64 if limbs == 1 else torch.float32,
            device=expert_output.device,
        )
    out_stride = 2 * full if limbs == 1 else full
    assert out.is_contiguous() and out.shape[1] == out_stride
    block_h = min(triton.next_power_of_2(hidden_dim), 1024)
    h_blocks = triton.cdiv(hidden_dim, block_h)
    grid = (_persistent_programs(num_tokens, tokens_hint, expert_output.device, h_blocks), h_blocks)
    _dk_unpermute_slots_kernel[grid](
        expert_output,
        slots,
        routing_map,
        valid_tokens,
        out.view(-1)[col0:],
        hidden_dim,
        out_stride,
        full,
        1.0 if y_scale is None else y_scale,
        local_expert_start,
        TOPK=topk,
        K_POW2=triton.next_power_of_2(topk),
        BLOCK_H=block_h,
        HAS_SCALE=y_scale is not None,
        LIMBS=limbs,
        ROUND16=round16,
    )
    return out


# The engine's expert row workspace, one per (device, ffn, hidden): {src, pp, a, y (flat), rows, chunk, reserved}.
# Outgrown unreserved sets stay alive (graphs may hold them).
_ROW_WS: dict[tuple, dict] = {}
_ROW_WS_OUTGROWN: list = []
_WS_BLOCK_M = 128  # the largest BLOCK_M of _DK_TABLE: its row capacity covers every tile's


def workspace_rows(tokens: int, topk: int, num_local_experts: int) -> int:
    """Row capacity for ``tokens`` gathered tokens: min(top-k, local experts) pairs a token + a BLOCK_M per segment."""
    rows = tokens * min(topk, num_local_experts) + _WS_BLOCK_M * num_local_experts
    return triton.cdiv(rows, _WS_BLOCK_M) * _WS_BLOCK_M


def fc2_column_chunk(hidden: int, ffn: int) -> int:
    """Columns per fc2 pass when the workspace stays resident: a pass's fp32 output is no larger than the fc1 fp16
    output (4 B x chunk <= 2 B x ffn), in 64-column units. Nano (hidden 2688, ffn 1856): 3 passes of 896; Super /
    Ultra (latent 1024 / 2048, ffn 2688 / 5120): one pass."""
    passes = -(-4 * hidden // (2 * ffn))
    return min(hidden, -(-(-(-hidden // passes)) // 64) * 64)


def workspace_bytes(rows: int, ffn: int, chunk: int) -> int:
    return rows * (8 + 2 * ffn + 4 * chunk)


def _allocate_row_workspace(device: torch.device, rows: int, ffn: int, chunk: int, reserved: bool) -> dict:
    return dict(
        src=torch.empty(rows, dtype=torch.int32, device=device),
        pp=torch.empty(rows, dtype=torch.float32, device=device),
        a=torch.empty((rows, ffn), dtype=torch.float16, device=device),
        y=torch.empty(rows * chunk, dtype=torch.float32, device=device),
        rows=rows,
        chunk=chunk,
        reserved=reserved,
    )


def _ws_key(device: torch.device, ffn: int, hidden: int) -> tuple:
    return (device.index if device.index is not None else torch.cuda.current_device(), ffn, hidden)


def reserve_row_workspace(
    device: torch.device,
    tokens: int,
    topk: int,
    num_local_experts: int,
    ffn: int,
    hidden: int,
    colocated: bool,
) -> tuple[int, int]:
    """Allocate the engine's row workspace for ``tokens`` gathered tokens before any graph is captured.

    fc2 runs in one pass (fc2 output [R, H]) unless the workspace would stay resident through a colocated learner's
    training: then in ``fc2_column_chunk`` passes (+14..25 us / MoE layer at Nano decode). It is released
    during training when it is allocated in the engine-scratch region and that region is functional
    (``memory_saver.engine_scratch_region`` / ``check_engine_scratch_released``: torch_memory_saver, the TMS=1 overlay):
    pausing it must return its physical memory, checked here once, else the engine refuses to start. Returns
    (bytes, fc2 columns per pass).
    """
    from nemo_rl.models.megatron import memory_saver as ms

    rows = workspace_rows(tokens, topk, num_local_experts)
    released = ms.HAVE_TORCH_MEMORY_SAVER
    chunk = hidden if (released or not colocated) else fc2_column_chunk(hidden, ffn)
    key = _ws_key(device, ffn, hidden)
    old = _ROW_WS.get(key)
    if old is None or old["rows"] < rows or old["chunk"] != chunk:
        if old is not None:
            _ROW_WS_OUTGROWN.append(old)
        with ms.engine_scratch_region():
            _ROW_WS[key] = _allocate_row_workspace(device, rows, ffn, chunk, reserved=True)
        if released:
            ms.check_engine_scratch_released(workspace_bytes(rows, ffn, chunk))
    return workspace_bytes(rows, ffn, chunk), chunk


def row_workspace(device: torch.device, rows: int, ffn: int, hidden: int) -> tuple:
    """(src [R] int32, probs [R] fp32, fc1 output [R, F] fp16, fc2 output flat [R x chunk] fp32, chunk): the first
    ``rows`` rows of the shared workspace, one set for every MoE layer, graph bucket and eager step (the engine's
    forwards run one at a time on its stream) instead of a worst-case set allocated per call (in every capture's
    private pool and again in every eager step).

    A reserved workspace (``reserve_row_workspace``, the engine) never grows: a call needing more rows is an error
    (growth would sit outside the engine-scratch region, resident through training). Unreserved (unit tests, other
    callers): allocated or grown by an eager call with column passes (outgrown sets stay alive: graphs may hold
    them); a CUDA-graph capture must find it large enough.
    """
    key = _ws_key(device, ffn, hidden)
    ws = _ROW_WS.get(key)
    if ws is None or ws["rows"] < rows:
        if ws is not None and ws["reserved"]:
            raise RuntimeError(
                f"engine expert row workspace reserved for {ws['rows']} rows, a call needs {rows}: the reservation "
                "(reserve_engine_row_workspace) must cover the dispatcher's gathered buffer"
            )
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"engine expert row workspace ({0 if ws is None else ws['rows']} rows) is smaller than a captured "
                f"call's {rows} rows: reserve_row_workspace before capturing"
            )
        if ws is not None:
            _ROW_WS_OUTGROWN.append(ws)
        ws = _allocate_row_workspace(device, rows, ffn, fc2_column_chunk(hidden, ffn), reserved=False)
        _ROW_WS[key] = ws
    return ws["src"][:rows], ws["pp"][:rows], ws["a"][:rows], ws["y"], ws["chunk"]


def release_row_workspaces() -> None:
    """Drop every workspace reference (process teardown, inside a torch_memory_saver region)."""
    _ROW_WS.clear()
    _ROW_WS_OUTGROWN.clear()


def reserve_engine_row_workspace(model: Any, context: Any, colocated: bool) -> tuple[int, int]:
    """Reserve the row workspace of an engine whose routed experts run on fp16_moe_dk (no-op otherwise).

    Gathered tokens per call, the buffer the dispatcher hands the experts: the NVLS all-gather's
    round_up(max_tokens) / TP x EP (its per-rank worst case x EP), else round_up(max_tokens) x EP.
    Returns (reserved bytes, fc2 columns per pass); (0, 0): not a dk engine.
    """
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP
    from megatron.core.transformer.moe.token_dispatcher_inference import NVLSAllGatherVDispatcher

    from nemo_rl.models.megatron import fp16_gemm
    from nemo_rl.models.megatron import fp16_moe as fm
    from nemo_rl.models.megatron import precision_raise as pr

    cfg = getattr(model, "config", None)
    if cfg is None or not getattr(cfg, "num_moe_experts", None) or not fp16_gemm.enabled():
        return 0, 0
    if fm.expert_activation(cfg) != "relu2" or pr.on(pr.ROUTED_FC2_HILO):
        return 0, 0
    experts = next((m for m in model.modules() if isinstance(m, InferenceGroupedMLP)), None)
    if experts is None:
        return 0, 0
    ep = experts.ep_group.size()
    if getattr(context, "_nvls_dispatcher", False):
        tokens = NVLSAllGatherVDispatcher._per_rank_worst_case_token_count * ep
    else:
        tokens = context.round_up_tokens(context.max_tokens) * ep
    hidden = getattr(cfg, "moe_latent_size", None) or cfg.hidden_size
    return reserve_row_workspace(
        torch.device("cuda", torch.cuda.current_device()),
        tokens,
        cfg.moe_router_topk,
        cfg.num_moe_experts // ep,
        cfg.moe_ffn_hidden_size,
        hidden,
        colocated,
    )


def engine_experts_dk(
    x16: torch.Tensor,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
    local_expert_start: int,
    ep_size: int,
    tokens_hint: Optional[int] = None,
    down_scale: int = 0,
    out: Optional[torch.Tensor] = None,
    limbs: bool = False,
    payload_fp16: bool = False,
    skip_zero_probs: bool = False,
) -> torch.Tensor:
    """``fp16_moe.engine_experts`` for relu2 experts without the fc2 hi/lo raise (same arguments, same bits).

    Rows: x16 is the gathered buffer (NVLS: max_tokens / TP x EP rows). A step's live tokens are bounded only by it,
    in CUDA graphs too: the NVLS dispatcher lets EP ranks run different buckets and eager steps side by side
    (``match_ep_token_counts`` False), so a rank replaying a small decode graph receives its peers' prefill tokens.
    The row capacity R = T_buf x min(top-k, local experts) + segment padding is therefore the worst case; it is held
    once (``row_workspace``: one set for every layer, graph and eager step), R x (8 + 2 F + 4 chunk) bytes per rank,
    where every capture's private pool and every eager step allocated a full set (pools of 48.6 GiB and a 20.69 GiB
    init OOM in the Nano 16k wvs arm). fc2 runs in one pass (chunk = H) when the workspace is released during
    training or the engine is not colocated, else in column passes (``reserve_row_workspace``); each pass's columns
    are combined into the same output columns (per-column fp64 sums: same bits). Full Nano 16k TP1 EP8 cross-fade
    top-14 (R = 2,066,432): 27.85 GiB single pass, 14.06 GiB in 3 passes.
    """
    from nemo_rl.models.megatron import fp16_moe as fm

    num_local_experts = fc1_weight.shape[0]
    topk = routing_map.shape[1]
    hint = x16.shape[0] if tokens_hint is None else min(tokens_hint, x16.shape[0])
    c1, c2 = select_dk_configs(hint * topk / (num_local_experts * ep_size))
    bm = c1[0]
    assert _WS_BLOCK_M % bm == 0
    rows_hint = fm.live_rows_bound(hint, topk, num_local_experts, bm)
    rows = workspace_rows(routing_map.shape[0], topk, num_local_experts)
    ffn, hidden = fc1_weight.shape[1], fc2_weight.shape[1]
    ws_src, ws_pp, ws_a, ws_y, chunk = row_workspace(x16.device, rows, ffn, hidden)
    src, pp, offs, slots = dk_route(
        routing_map, probs, valid_tokens, local_expert_start, num_local_experts, bm, skip_zero_probs, hint,
        rows_buf=(ws_src, ws_pp),
    )
    a = dk_grouped_gemm(
        x16,
        fc1_weight,
        offs,
        c1,
        rows_hint,
        probs=pp,
        out_dtype=torch.float16,
        prob_scale=2.0**-down_scale if down_scale else None,
        src=src,
        out=ws_a,
    )
    lmode = (1 if ep_size > 1 else 2) if limbs else 0
    if out is None:
        out = torch.empty(
            slots.shape[0],
            2 * hidden if lmode == 1 else hidden,
            dtype=torch.float64 if lmode == 1 else torch.float32,
            device=x16.device,
        )
    for col0 in range(0, hidden, chunk):  # one pass unless the workspace stays resident (reserve_row_workspace)
        width = min(chunk, hidden - col0)
        y = dk_grouped_gemm(
            a, fc2_weight, offs, c2, rows_hint, out=ws_y[: rows * width].view(rows, width), n_slice=(col0, width)
        )
        dk_unpermute_slots(
            y,
            slots,
            routing_map,
            local_expert_start,
            valid_tokens,
            out=out,
            tokens_hint=hint,
            y_scale=2.0**down_scale if down_scale else None,
            limbs=lmode,
            round16=payload_fp16,
            col0=col0,
            full_hidden=hidden,
        )
    return out


def set_config_override(fc1: Optional[DkConfig], fc2: Optional[DkConfig]) -> None:
    """Tests / research/dk1 sweeps only: force both tiles (same BLOCK_M), or None to use the table."""
    _OVERRIDE.clear()
    if fc1 is not None:
        assert fc2 is not None and fc1[0] == fc2[0]
        _OVERRIDE["fc1"], _OVERRIDE["fc2"] = tuple(fc1), tuple(fc2)
