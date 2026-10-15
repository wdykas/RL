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
"""EP-invariant MoE combine (``policy.megatron_cfg.moe_ep_invariant_combine``).

The batch-invariant combine of the fp16 MoE path sums a token's expert rows per EP
rank (fp64, rounded to an fp32 rank partial) and then adds the rank partials in
rank order. The rounding points move with the EP size, so the same token gets
different bits at EP 1, 2 and 4 (measured: runs_1node/np-v2-lanes-ep,
thundersync_work/research/ep_invariant).

This combine makes the token's output a function of its k expert rows only:

- Every expert row value v (fp32; the probability-weighted, down-scaled fp16 x fp16
  GEMM output) maps to two fp64 integer limbs, ``hi = floor(v)`` and
  ``lo = rint((v - hi) * 2^48)``. Under the fp16 GEMM forward v is a multiple of
  2^-48 (products of two fp16 values are), so hi + lo * 2^-48 == v exactly.
- Limbs are summed in fp64. Every partial sum is an integer below 2^53 (checked at
  setup: |v| < 65504^2 * K_fc2 * 2^s, k <= 32), so every addition is exact, and
  integer addition is associative and commutative: any grouping of a token's rows
  (per local expert, per rank, any reduction order or collective) gives the same
  two sums.
- ``finalize`` turns the two exact sums into fp32: hi + lo * 2^-48 in fp64 with
  round-to-odd (TwoSum), then fp64 -> fp32 round-to-nearest, which is the correctly
  rounded fp32 value of the exact sum of the token's expert rows.
- Non-finite rows go into ``hi`` (lo = 0): IEEE inf / nan addition is order-free.

So each EP rank still sends one partial per token (now two fp64 limbs, 16 B per
element instead of 4 B), the engine's reduce-scatter may sum in any order, and the
learner's AlltoAll unpermute (which already receives all k rows of a token) sums
limbs instead of rank partials. Same bits at every EP size, on both sides.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

try:  # MCore's symmetric-memory barrier (engine NVLS combine only)
    from megatron.core.inference.communication.torch_symm_triton.barrier import (
        symm_mem_sync,
    )
    from megatron.core.inference.communication.torch_symm_triton.utils import (
        sync_threads,
    )

    HAVE_SYMM = HAVE_TRITON
except ImportError:  # pragma: no cover
    HAVE_SYMM = False

FRAC_BITS = 48  # fp16 x fp16 products are multiples of 2^-48
_STATE: dict[str, Any] = {"enabled": False, "logged": False}


def enabled() -> bool:
    """Whether this process uses the EP-invariant combine."""
    return bool(_STATE["enabled"])


def set_enabled(flag: bool) -> None:
    """Turn the EP-invariant combine on or off for this process (before the model runs)."""
    _STATE["enabled"] = bool(flag)


def check_exact_range(topk: int, fc2_k: int, down_scale: int = 0) -> list[str]:
    """Why the limb sums could round for this MoE shape (empty: every sum is exact).

    hi sums: sum of |floor(v)| over a token's k rows < 2^53 with |v| < 65504^2 * K * 2^s;
    lo sums: k rows each <= 2^48 -> k <= 32.
    """
    bad = []
    if topk > 32:
        bad.append(f"moe_router_topk={topk} > 32 (lo limb sums would exceed 2^53)")
    hi_bits = 2 * math.log2(65504.0) + math.log2(max(fc2_k, 1)) + down_scale + math.log2(max(topk, 1))
    if hi_bits >= 53:
        bad.append(
            f"expert fc2 K={fc2_k}, down_proj_prescale_log2={down_scale}, top-k={topk}: "
            f"|sum| bound 2^{hi_bits:.1f} >= 2^53 (hi limb sums could round)"
        )
    return bad


if HAVE_TRITON:

    @triton.jit
    def _limbs(v):
        """fp32 values -> (hi, lo) fp64 integer limbs, v == hi + lo * 2^-48."""
        v64 = v.to(tl.float64)
        finite = tl.abs(v64) < float("inf")  # False for inf and nan
        hi = libdevice.floor(v64)
        lo = libdevice.rint((v64 - hi) * 281474976710656.0)  # 2^48
        hi = tl.where(finite, hi, v64)
        lo = tl.where(finite, lo, 0.0)
        return hi, lo

    @triton.jit
    def _finalize(hi, lo):
        """Exact limb sums -> fp32 (correctly rounded hi + lo * 2^-48)."""
        hi = hi.to(tl.float64)
        b = lo.to(tl.float64) * 3.552713678800501e-15  # 2^-48, exact
        s = hi + b
        bb = s - hi
        err = (hi - (s - bb)) + (b - bb)
        # Round to odd: a correctly rounded fp32 from the fp64 sum (53 >= 24 + 2).
        sb = s.to(tl.int64, bitcast=True)
        nudge = (err != 0.0) & ((sb & 1) == 0)
        up = (err > 0.0) == (s > 0.0)  # err pulls |s| up
        sb = tl.where(nudge, tl.where(up, sb + 1, sb - 1), sb)
        s = sb.to(tl.float64, bitcast=True)
        finite = tl.abs(hi) < float("inf")
        return tl.where(finite, s, hi).to(tl.float32)

    @triton.jit
    def _finalize_kernel(limbs_ptr, out_ptr, n_rows, H, BLOCK_H: tl.constexpr):
        # limbs [n_rows, 2H] fp64 (hi | lo) -> out [n_rows, H] fp32
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        m = cols < H
        hi = tl.load(limbs_ptr + row * 2 * H + cols, mask=m, other=0.0)
        lo = tl.load(limbs_ptr + row * 2 * H + H + cols, mask=m, other=0.0)
        tl.store(out_ptr + row * H + cols, _finalize(hi, lo), mask=m)

    @triton.jit
    def _to_limbs_kernel(x_ptr, limbs_ptr, H, BLOCK_H: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        m = cols < H
        hi, lo = _limbs(tl.load(x_ptr + row * H + cols, mask=m, other=0.0))
        tl.store(limbs_ptr + row * 2 * H + cols, hi, mask=m)
        tl.store(limbs_ptr + row * 2 * H + H + cols, lo, mask=m)

    @triton.jit
    def _unpermute_rows_fwd_kernel(
        x_ptr,  # [rows, H] fp32 probability-weighted expert rows (AlltoAll combine output)
        rows_ptr,  # [T, K] int64 row of each top-k slot, -1: none
        experts_ptr,  # [T, K] int64 expert of each slot, -1: none
        out_ptr,  # [T, H] fp32
        H,
        num_experts,
        K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H
        acc_hi = tl.zeros([BLOCK_H], dtype=tl.float64)
        acc_lo = tl.zeros([BLOCK_H], dtype=tl.float64)
        for j in tl.static_range(K):
            row = tl.load(rows_ptr + tok * K + j)
            ex = tl.load(experts_ptr + tok * K + j)
            valid = (row >= 0) & (ex >= 0) & (ex < num_experts)
            if valid:
                hi, lo = _limbs(tl.load(x_ptr + row * H + cols, mask=mask, other=0.0))
                acc_hi += hi
                acc_lo += lo
        tl.store(out_ptr + tok * H + cols, _finalize(acc_hi, acc_lo), mask=mask)

    @triton.jit
    def _unpermute_rows_bwd_kernel(
        g_ptr, rows_ptr, experts_ptr, dx_ptr, H, num_experts, K: tl.constexpr, BLOCK_H: tl.constexpr
    ):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H
        g = tl.load(g_ptr + tok * H + cols, mask=mask, other=0.0) + 0.0
        for j in tl.static_range(K):
            row = tl.load(rows_ptr + tok * K + j)
            ex = tl.load(experts_ptr + tok * K + j)
            valid = (row >= 0) & (ex >= 0) & (ex < num_experts)
            tl.store(dx_ptr + tl.maximum(row, 0) * H + cols, g, mask=mask & valid)


if HAVE_SYMM:

    @triton.jit
    def _limbs_reduce_scatter_v_kernel(
        local_ptr,  # [local_tokens, H] fp32 output
        buffer_ptrs_dev,
        signal_pad_ptrs,
        local_tokens,
        rank_token_offset_ptr,
        ep_max_tokens_ptr,
        HIDDEN_SIZE: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
        RANK: tl.constexpr,
        WORLD_SIZE: tl.constexpr,
    ):
        """Sum every rank's (hi, lo) limb rows of this rank's tokens (exact), then finalize."""
        pid = tl.program_id(axis=0)
        ep_max_tokens = tl.load(ep_max_tokens_ptr)
        if pid >= ep_max_tokens:
            return
        symm_mem_sync(
            signal_pad_ptrs,
            None,
            RANK,
            WORLD_SIZE,
            hasPreviousMemAccess=False,
            hasSubsequentMemAccess=True,
        )
        sync_threads()
        tid = tl.arange(0, BLOCK_SIZE)
        rank_token_offset = tl.load(rank_token_offset_ptr)
        buffer_ptrs = buffer_ptrs_dev.to(tl.pointer_type(tl.uint64))
        for token_offset in range(pid, local_tokens, tl.num_programs(axis=0)):
            global_token = (rank_token_offset + token_offset).to(tl.int64)
            for channel_offset in range(0, HIDDEN_SIZE, BLOCK_SIZE):
                offsets = channel_offset + tid
                mask = offsets < HIDDEN_SIZE
                acc_hi = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
                acc_lo = tl.zeros([BLOCK_SIZE], dtype=tl.float64)
                for src_rank in tl.range(0, WORLD_SIZE):
                    peer = tl.load(buffer_ptrs + src_rank).to(tl.pointer_type(tl.float64))
                    base = peer + global_token * (2 * HIDDEN_SIZE)
                    acc_hi += tl.load(base + offsets, mask=mask, other=0.0)
                    acc_lo += tl.load(base + HIDDEN_SIZE + offsets, mask=mask, other=0.0)
                tl.store(
                    local_ptr + token_offset * HIDDEN_SIZE + offsets,
                    _finalize(acc_hi, acc_lo),
                    mask=mask,
                )


def _block(h: int) -> int:
    return min(triton.next_power_of_2(h), 1024)


def to_limbs(x: torch.Tensor) -> torch.Tensor:
    """[N, H] fp32 -> [N, 2H] fp64 (hi | lo)."""
    x = x.contiguous().float()
    n, h = x.shape
    out = torch.empty(n, 2 * h, dtype=torch.float64, device=x.device)
    if n:
        _to_limbs_kernel[(n, triton.cdiv(h, _block(h)))](x, out, h, BLOCK_H=_block(h))
    return out


def finalize_limbs(limbs: torch.Tensor, out: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[N, 2H] fp64 limb sums -> [N, H] fp32."""
    limbs = limbs.contiguous()
    n, h2 = limbs.shape
    h = h2 // 2
    if out is None:
        out = torch.empty(n, h, dtype=torch.float32, device=limbs.device)
    if n:
        _finalize_kernel[(n, triton.cdiv(h, _block(h)))](limbs, out, n, h, BLOCK_H=_block(h))
    return out


class _UnpermuteRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, rows, experts, num_tokens, num_experts):  # noqa: D102
        x = x.contiguous()
        h = x.shape[1]
        k = rows.shape[1]
        out = torch.empty((num_tokens, h), dtype=torch.float32, device=x.device)
        if num_tokens:
            _unpermute_rows_fwd_kernel[(num_tokens, triton.cdiv(h, _block(h)))](
                x, rows, experts, out, h, num_experts, K=k, BLOCK_H=_block(h)
            )
        ctx.save_for_backward(rows, experts)
        ctx.num_rows = x.shape[0]
        ctx.num_experts = num_experts
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        rows, experts = ctx.saved_tensors
        g = grad_out.to(torch.float32).contiguous()
        n, h = g.shape
        dx = torch.zeros((ctx.num_rows, h), dtype=torch.float32, device=g.device)
        if n:
            _unpermute_rows_bwd_kernel[(n, triton.cdiv(h, _block(h)))](
                g, rows, experts, dx, h, ctx.num_experts, K=rows.shape[1], BLOCK_H=_block(h)
            )
        return dx, None, None, None, None


def unpermute_rows(
    permuted_tokens: torch.Tensor,
    inverse_map: torch.Tensor,
    num_tokens: int,
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    """Learner combine: [T, H] fp32 = each token's slot rows summed through the limbs.

    ``inverse_map``: Megatron's [2, T, K] (row, global expert) batch-invariant map. The
    result does not depend on the EP size (no rank grouping) or on slot order.
    Backward: each row gets its token's output gradient (as Megatron's unpermute).
    """
    rows = inverse_map[0].to(torch.int64).contiguous()
    experts = inverse_map[1].to(torch.int64).contiguous()
    ne = int(num_experts) if num_experts is not None else (1 << 62)
    return _UnpermuteRows.apply(permuted_tokens.float(), rows, experts, int(num_tokens), ne)


def limbs_reduce_scatter_v(
    output: torch.Tensor,
    symm_mem_hdl: Any,
    rank_token_offset: torch.Tensor,
    ep_max_tokens: torch.Tensor,
    per_rank_max_tokens: int,
    max_num_blocks: int = 128,
) -> torch.Tensor:
    """Engine (NVLS) combine: output [local_tokens, H] fp32 from every rank's limb rows.

    Each rank's symmetric buffer holds [global_max, 2H] fp64 limb partials at offset 0
    (``engine_limb_buffer``). The sum over ranks is exact, so its order does not matter.
    """
    h = output.shape[1]
    block = _block(h)
    _limbs_reduce_scatter_v_kernel[(min(per_rank_max_tokens, max_num_blocks), 1, 1)](
        output,
        symm_mem_hdl.buffer_ptrs_dev,
        symm_mem_hdl.signal_pad_ptrs_dev,
        local_tokens=output.shape[0],
        rank_token_offset_ptr=rank_token_offset,
        ep_max_tokens_ptr=ep_max_tokens,
        HIDDEN_SIZE=h,
        BLOCK_SIZE=block,
        RANK=symm_mem_hdl.rank,
        WORLD_SIZE=symm_mem_hdl.world_size,
        num_warps=max(1, block // 32),
    )
    return output


def engine_limb_buffer(dispatcher_cls: Any, hidden_size: int) -> Optional[torch.Tensor]:
    """The NVLS dispatcher's symmetric RSV memory viewed as [global_max, 2H] fp64 limbs."""
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager

    if dispatcher_cls._symm_rsv is None or dispatcher_cls._symm_rsv["tensor"] is None:
        return None
    buf = SymmetricMemoryManager.get_buffer("ep_rsv").symm_buffer
    global_max = dispatcher_cls._symm_rsv["tensor"].shape[0]
    nbytes = global_max * 2 * hidden_size * 8
    assert buf.numel() >= nbytes, "ep_rsv was not sized for limbs (install before the engine)"
    return buf[:nbytes].view(torch.float64).view(global_max, 2 * hidden_size)


def install_engine_buffers() -> None:
    """Size the NVLS dispatcher's RSV symmetric buffer for fp64 limbs (4x the fp32 rows).

    The buffer is created here under MCore's name before MCore asks for it, so MCore's
    fp32 view and the limb view share one allocation (no second buffer).
    """
    from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
    from megatron.core.transformer.moe import token_dispatcher_inference as tdi

    cls = tdi.NVLSAllGatherVDispatcher
    if getattr(cls.allocate_buffers, "_ts_limbs", False):
        return
    orig = cls.allocate_buffers.__func__

    def allocate_buffers(klass, per_rank_worst_case_token_count, topk, hidden_size, ep_group):
        if enabled() and not SymmetricMemoryManager.is_initialized("ep_rsv"):
            from megatron.core.utils import get_pg_size

            global_max = per_rank_worst_case_token_count * get_pg_size(ep_group)
            nbytes = global_max * 2 * hidden_size * 8
            SymmetricMemoryManager.get_buffer(
                "ep_rsv", process_group=ep_group, size_mb=max(1, -(-nbytes // (1 << 20)))
            )
        return orig(klass, per_rank_worst_case_token_count, topk, hidden_size, ep_group)

    allocate_buffers._ts_limbs = True  # type: ignore[attr-defined]
    cls.allocate_buffers = classmethod(allocate_buffers)


def log_once(msg: str) -> None:
    """Print one line the first time the combine runs in this process."""
    if not _STATE["logged"]:
        _STATE["logged"] = True
        print(f"[ep_invariant_combine] {msg}", flush=True)
