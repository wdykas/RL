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
"""Fused batch-invariant MoE unpermute for the AlltoAll combine output (training).

Megatron's ``moe/batch_invariant.unpermute`` (used by the AlltoAll dispatcher under
``batch_invariant_mode``) mirrors the inference engine's summation tree: per token, for
every EP rank in ascending order, the token's routed rows held by that rank's experts
are summed in fp64 in top-k slot order (``probs`` is None: the rows are already
probability-weighted), the rank partial is rounded to fp32, and the partials are added
in rank order in fp32 (``ordered`` collective; fp64 then one final rounding for
``multimem``). It does so with EP x top-k full-size ``index_select`` passes in eager
PyTorch; autograd then builds one dense [rows, hidden] gradient per (rank, slot). Time
and gradient memory grow with EP x top-k.

``fused_unpermute`` computes the same sums in one Triton kernel. Per token the K slots
are visited in (rank, slot) order (a per-token sort of K keys), so each rank's fp64
partial adds the same values in the same order, starting from +0.0, and is rounded and
added to the fp32 (or fp64) accumulator in ascending rank order. The eager version also
adds +0.0 for every slot outside the rank and adds the +0.0 partial of a rank with no
slots; neither changes a value (a sum of finite terms started at +0.0 is never -0.0
under round-to-nearest), so skipping them keeps the result bitwise identical.
Backward: d out[t] / d row = 1 for the row of each of t's slots, so the gradient of a
row is the token's output gradient. The eager backward accumulates it into zeros
(+0.0 + g, which turns -0.0 into +0.0); the scatter kernel writes g + 0.0. Rows no slot
refers to get 0. Cost O(T * K * H) forward and backward, independent of EP; memory:
the output, the input gradient and [T, K] int32 maps.

Enabled by ``install()`` (``policy.megatron_cfg.fused_bi_unpermute``); other cases
(``probs`` given, a non-fp32 input, CPU) fall back to Megatron's function.
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

_STATE: dict[str, Any] = {"orig": None, "logged": False}


if HAVE_TRITON:

    @triton.jit
    def _fused_unpermute_fwd_kernel(
        x_ptr,  # [rows, H] fp32 combine output (probability-weighted expert rows)
        rows_ptr,  # [T, K] int32 row of each slot, slots sorted by (rank, slot); -1: none
        rank_ptr,  # [T, K] int32 EP rank of each sorted slot (EP for none)
        out_ptr,  # [T, H] fp32
        H,
        K: tl.constexpr,
        OUT_FP64: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H
        acc_dtype = tl.float64 if OUT_FP64 else tl.float32
        out = tl.zeros([BLOCK_H], dtype=acc_dtype)
        part = tl.zeros([BLOCK_H], dtype=tl.float64)
        cur = tl.load(rank_ptr + tok * K)
        for j in tl.static_range(K):
            r = tl.load(rank_ptr + tok * K + j)
            row = tl.load(rows_ptr + tok * K + j).to(tl.int64)
            changed = r != cur
            # Flush the previous rank's fp64 partial (rounded to fp32) in rank order.
            flushed = part.to(tl.float32).to(acc_dtype)
            out = tl.where(changed, out + flushed, out)
            part = tl.where(changed, tl.zeros_like(part), part)
            cur = r
            valid = row >= 0
            vals = tl.load(
                x_ptr + tl.maximum(row, 0) * H + cols, mask=mask & valid, other=0.0
            )
            part = tl.where(valid, part + vals.to(tl.float64), part)
        out = out + part.to(tl.float32).to(acc_dtype)
        tl.store(out_ptr + tok * H + cols, out.to(tl.float32), mask=mask)

    @triton.jit
    def _fused_unpermute_bwd_kernel(
        g_ptr,  # [T, H] fp32 output gradient
        rows_ptr,  # [T, K] int32 (any slot order); -1: none
        dx_ptr,  # [rows, H] fp32, zero-initialized
        H,
        K: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        tok = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        mask = cols < H
        g = tl.load(g_ptr + tok * H + cols, mask=mask, other=0.0) + 0.0
        for j in tl.static_range(K):
            row = tl.load(rows_ptr + tok * K + j).to(tl.int64)
            tl.store(dx_ptr + tl.maximum(row, 0) * H + cols, g, mask=mask & (row >= 0))


def slot_order(
    inverse_map: torch.Tensor, num_experts: int, ep_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """(rows, ranks) [T, K] int32 with each token's slots sorted by (EP rank, slot).

    ``inverse_map``: Megatron's [2, T, K] (row, expert) map
    (``build_inverse_permutation_map``); a slot is used when its row is >= 0 and its
    expert is in [0, num_experts) (rank = expert // (num_experts / ep_size)); unused
    slots sort last with row -1 and rank ``ep_size``.
    """
    rows, experts = inverse_map[0], inverse_map[1]
    k = rows.shape[1]
    valid = (rows >= 0) & (experts >= 0) & (experts < num_experts)
    per_rank = num_experts // ep_size
    rank = torch.where(
        valid, torch.div(experts, per_rank, rounding_mode="floor"), ep_size
    )
    slot = torch.arange(k, device=rows.device)
    order = torch.argsort(rank * k + slot, dim=1)
    rows_sorted = torch.where(valid, rows, -1).gather(1, order)
    return rows_sorted.to(torch.int32).contiguous(), rank.gather(1, order).to(
        torch.int32
    ).contiguous()


class _FusedUnpermute(torch.autograd.Function):
    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx: Any,
        permuted_tokens: torch.Tensor,
        rows: torch.Tensor,
        ranks: torch.Tensor,
        num_tokens: int,
        out_fp64: bool,
    ) -> torch.Tensor:
        x = permuted_tokens.contiguous()
        hidden = x.shape[1]
        k = rows.shape[1]
        out = torch.empty((num_tokens, hidden), dtype=torch.float32, device=x.device)
        block = min(triton.next_power_of_2(hidden), 1024)
        grid = (num_tokens, triton.cdiv(hidden, block))
        if num_tokens:
            _fused_unpermute_fwd_kernel[grid](
                x, rows, ranks, out, hidden, K=k, OUT_FP64=out_fp64, BLOCK_H=block
            )
        ctx.save_for_backward(rows)
        ctx.num_rows = x.shape[0]
        return out

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple:
        (rows,) = ctx.saved_tensors
        g = grad_out.to(torch.float32).contiguous()
        num_tokens, hidden = g.shape
        dx = torch.zeros((ctx.num_rows, hidden), dtype=torch.float32, device=g.device)
        block = min(triton.next_power_of_2(hidden), 1024)
        if num_tokens:
            _fused_unpermute_bwd_kernel[(num_tokens, triton.cdiv(hidden, block))](
                g, rows, dx, hidden, K=rows.shape[1], BLOCK_H=block
            )
        return dx, None, None, None, None


def fused_unpermute(
    permuted_tokens: torch.Tensor,
    restore_shape: torch.Size,
    *,
    probs: Optional[torch.Tensor],
    num_experts: int,
    inverse_map: torch.Tensor,
    ep_size: Optional[int] = None,
    cross_rank_fp64: Optional[bool] = None,
) -> torch.Tensor:
    """Drop-in for ``megatron.core.transformer.moe.batch_invariant.unpermute`` (bitwise)."""
    if (
        probs is not None
        or not HAVE_TRITON
        or not permuted_tokens.is_cuda
        or permuted_tokens.dtype != torch.float32
        or permuted_tokens.dim() != 2
    ):
        return _orig_unpermute()(
            permuted_tokens,
            restore_shape,
            probs=probs,
            num_experts=num_experts,
            inverse_map=inverse_map,
        )
    if ep_size is None:
        from megatron.core import parallel_state

        ep_size = parallel_state.get_expert_model_parallel_world_size() or 1
    if cross_rank_fp64 is None:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            get_batch_invariant_collective,
        )

        cross_rank_fp64 = get_batch_invariant_collective() == "multimem"
    assert num_experts % ep_size == 0, "batch-invariant MoE expects contiguous EP shards"
    if not _STATE["logged"]:
        _STATE["logged"] = True
        print(
            f"[bi_unpermute] fused batch-invariant unpermute: ep={ep_size} "
            f"tokens={int(restore_shape[0])} topk={inverse_map.shape[-1]} "
            f"hidden={permuted_tokens.shape[1]} cross_rank_fp64={bool(cross_rank_fp64)}",
            flush=True,
        )
    with torch.no_grad():
        rows, ranks = slot_order(inverse_map, num_experts, ep_size)
    out = _FusedUnpermute.apply(
        permuted_tokens, rows, ranks, int(restore_shape[0]), bool(cross_rank_fp64)
    )
    return out.view(restore_shape)


def _orig_unpermute() -> Any:
    if _STATE["orig"] is None:
        from megatron.core.transformer.moe import batch_invariant

        _STATE["orig"] = batch_invariant.unpermute
    return _STATE["orig"]


def install() -> None:
    """Route Megatron's batch-invariant AlltoAll unpermute through ``fused_unpermute``."""
    from megatron.core.transformer.moe import moe_utils

    if getattr(moe_utils.batch_invariant_unpermute, "_ts_fused", False):
        return
    _STATE["orig"] = moe_utils.batch_invariant_unpermute
    moe_utils.batch_invariant_unpermute = fused_unpermute
    fused_unpermute._ts_fused = True  # type: ignore[attr-defined]
