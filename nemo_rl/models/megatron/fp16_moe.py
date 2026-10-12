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
- ``swiglu_probs``: SiLU(gate) * up * prob in fp32 (eager elementwise ops).
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

from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

BLOCK_M = 64
BLOCK_N = 128
BLOCK_K = 64
NUM_WARPS = 4
NUM_STAGES = 3

_TL_DTYPES: dict = {}


if HAVE_TRITON:

    @triton.jit
    def _grouped_gemm_kernel(
        x_ptr,  # [M_pad, K] fp16, rows grouped by expert, segments BLOCK_M-aligned
        w_ptr,  # base pointer of the first expert's [N, K] weight
        w_off_ptr,  # [E] int64 element offset of each expert's weight from w_ptr
        tile_expert_ptr,  # [M_pad / BLOCK_M] int64 expert of each row tile, -1 unused
        out_ptr,  # [M_pad, N] fp32
        N,
        K,
        stride_xm,
        stride_om,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        expert = tl.load(tile_expert_ptr + pid_m)
        if expert < 0:
            return
        w_base = w_ptr + tl.load(w_off_ptr + expert)
        rm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)).to(tl.int64)
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
    def _unpermute_fp64_kernel(
        expert_out_ptr,  # [rows, H] fp32 expert outputs (probability-weighted)
        inverse_map_ptr,  # [num_tokens, num_local_experts] int32 row or -1
        valid_tokens_ptr,  # scalar int32
        output_ptr,  # [num_tokens, H] fp32
        hidden_dim,
        num_local_experts: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        tok = tl.program_id(0)
        block_h = tl.program_id(1)
        valid_tokens = tl.load(valid_tokens_ptr)
        offsets = block_h * BLOCK_H + tl.arange(0, BLOCK_H)
        mask_h = offsets < hidden_dim
        acc = tl.zeros([BLOCK_H], dtype=tl.float64)
        if tok < valid_tokens:
            tok_i64 = tok.to(tl.int64)
            for lid in tl.range(0, num_local_experts):
                pos = tl.load(inverse_map_ptr + tok_i64 * num_local_experts + lid)
                if pos >= 0:
                    vals = tl.load(
                        expert_out_ptr + pos.to(tl.int64) * hidden_dim + offsets, mask=mask_h
                    )
                    acc += vals.to(tl.float64)
            tl.store(output_ptr + tok_i64 * hidden_dim + offsets, acc.to(tl.float32), mask=mask_h)


def _launch_grouped_gemm(
    x16: torch.Tensor,
    w_base: torch.Tensor,
    w_offsets: torch.Tensor,
    tile_expert: torch.Tensor,
    n: int,
) -> torch.Tensor:
    m_pad, k = x16.shape
    assert m_pad % BLOCK_M == 0 and x16.dtype == torch.float16 and x16.is_contiguous()
    out = torch.empty((m_pad, n), dtype=torch.float32, device=x16.device)
    grid = (m_pad // BLOCK_M, triton.cdiv(n, BLOCK_N))
    _grouped_gemm_kernel[grid](
        x16,
        w_base,
        w_offsets,
        tile_expert,
        out,
        n,
        k,
        x16.stride(0),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=NUM_WARPS,
        num_stages=NUM_STAGES,
    )
    return out


def _weight_table(weights: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """(base weight, int64 element offsets of every expert weight from it)."""
    base = weights[0]
    for w in weights:
        assert w.is_contiguous() and w.dtype == base.dtype and w.shape == base.shape
    size = base.element_size()
    offs = [(w.data_ptr() - base.data_ptr()) // size for w in weights]
    return base, torch.tensor(offs, dtype=torch.int64, device=base.device)


def tile_experts_from_offsets(inclusive_offsets: torch.Tensor, num_tiles: int) -> torch.Tensor:
    """Expert of each BLOCK_M row tile from BLOCK_M-aligned inclusive offsets (on device)."""
    starts = torch.arange(num_tiles, device=inclusive_offsets.device, dtype=torch.int64) * BLOCK_M
    expert = torch.searchsorted(inclusive_offsets.to(torch.int64), starts, right=True)
    n = inclusive_offsets.numel()
    return torch.where(expert < n, expert, torch.full_like(expert, -1))


def grouped_gemm(
    x16: torch.Tensor,
    weights: list[torch.Tensor] | None,
    tile_expert: torch.Tensor,
    *,
    stacked: torch.Tensor | None = None,
) -> torch.Tensor:
    """Batch-invariant grouped GEMM: x16 [M_pad, K] fp16 @ W_e.T -> fp32 [M_pad, N].

    Weights: ``weights`` (one [N, K] tensor per expert, any addresses) or
    ``stacked`` ([E, N, K] contiguous; CUDA-graph safe: its offset table is
    built on device).
    """
    if not x16.is_cuda:
        return _grouped_gemm_reference(
            x16, list(weights) if weights is not None else list(stacked), tile_expert
        )
    if stacked is not None:
        e, n, k = stacked.shape
        offs = torch.arange(e, device=stacked.device, dtype=torch.int64) * (n * k)
        base = stacked
    else:
        base, offs = _weight_table(weights)
        n = base.shape[0]
    return _launch_grouped_gemm(x16, base, offs, tile_expert, n)


def _grouped_gemm_reference(x16, weights, tile_expert):
    """CPU stand-in for tests: the same per-row products, fp32 accumulation."""
    n = weights[0].shape[0]
    out = torch.zeros((x16.shape[0], n), dtype=torch.float32)
    for t, e in enumerate(tile_expert.tolist()):
        if e < 0:
            continue
        w = weights[e].to(torch.float16).float()
        rows = slice(t * BLOCK_M, (t + 1) * BLOCK_M)
        out[rows] = x16[rows].float() @ w.t()
    return out


def swiglu_probs(h: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """SiLU(gate) * up * prob in fp32. h [rows, 2F] (gate | up), probs [rows]."""
    h = h.float()
    gate, up = torch.chunk(h, 2, dim=-1)
    return (gate / (1.0 + torch.exp(-gate))) * up * probs.float().unsqueeze(-1)


def unpermute_fp64(
    expert_output: torch.Tensor,
    inverse_map: torch.Tensor,
    valid_tokens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-token sum of local-expert rows in fp64, ascending local expert, fp32 out."""
    _, hidden_dim = expert_output.shape
    num_tokens, num_local_experts = inverse_map.shape
    if out is None:
        out = torch.empty(num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device)
    block_h = min(triton.next_power_of_2(hidden_dim), 1024)
    grid = (num_tokens, triton.cdiv(hidden_dim, block_h))
    _unpermute_fp64_kernel[grid](
        expert_output,
        inverse_map,
        valid_tokens,
        out,
        hidden_dim,
        num_local_experts,
        BLOCK_H=block_h,
    )
    return out


# ---------------------------------------------------------------------------
# Training: autograd grouped linear over host-side expert splits.
# ---------------------------------------------------------------------------


def _aligned_rows(m_splits: list[int], device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(destination row of every input row, tile experts, padded row count)."""
    dest, tiles = [], []
    start = 0
    for e, m in enumerate(m_splits):
        dest.append(torch.arange(start, start + m, dtype=torch.int64))
        n_tiles = -(-m // BLOCK_M)
        tiles += [e] * n_tiles
        start += n_tiles * BLOCK_M
    dest_t = torch.cat(dest) if dest else torch.zeros(0, dtype=torch.int64)
    tiles_t = torch.tensor(tiles if tiles else [-1], dtype=torch.int64)
    m_pad = max(start, BLOCK_M)
    return dest_t.to(device), tiles_t.to(device), m_pad


class GroupedFp16Linear(torch.autograd.Function):
    """y[rows of expert e] = fp16(x) @ fp16(W_e).T in fp32 (the shared kernel).

    Backward: per expert, dgrad = g @ W_e and wgrad = g.T @ x on bf16 casts,
    fp32 accumulation; dgrad in x's dtype, wgrad in the weight dtype.
    """

    @staticmethod
    def forward(ctx, x, m_splits, *weights):  # noqa: D102
        k = x.shape[-1]
        x16 = x.reshape(-1, k).to(torch.float16)
        dest, tiles, m_pad = _aligned_rows(list(m_splits), x.device)
        x_pad = torch.zeros((m_pad, k), dtype=torch.float16, device=x.device)
        x_pad.index_copy_(0, dest, x16)
        out = grouped_gemm(x_pad, list(weights), tiles).index_select(0, dest)
        ctx.save_for_backward(x16, *weights)
        ctx.m_splits = list(m_splits)
        ctx.x_dtype = x.dtype
        ctx.x_shape = x.shape
        return out

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        x16, *weights = ctx.saved_tensors
        g = grad.reshape(-1, weights[0].shape[0]).to(torch.bfloat16)
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


def training_experts_forward(module: Any, x, tokens_per_expert, permuted_probs) -> tuple:
    """TEGroupedMLP forward (training and the training model's no-grad forwards)."""
    if isinstance(tokens_per_expert, torch.Tensor):
        tokens_per_expert = tokens_per_expert.tolist()
    m_splits = [int(m) for m in tokens_per_expert]
    h = grouped_linear(x, m_splits, _expert_weights(module.linear_fc1))
    a = swiglu_probs(h, permuted_probs.reshape(-1))
    y = grouped_linear(a, m_splits, _expert_weights(module.linear_fc2))
    return y, None


def inference_experts_forward(module: Any, hidden_states, permuted_probs, routing_map) -> tuple:
    """InferenceGroupedMLP forward in the engine (CUDA-graph safe)."""
    from megatron.core.inference.moe.permute import permute_tokens
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    if not module._concatenated_weights_built:
        module._build_concatenated_weights()
        module._concatenated_weights_built = True
    x16 = hidden_states.view(torch.float16) if hidden_states.dtype == torch.bfloat16 else (
        hidden_states.to(torch.float16)
    )
    valid_tokens = InferenceAllGatherDispatcherBase._valid_tokens()
    local_expert_start = module.ep_group.rank() * module.num_local_experts
    xp, pp, _pmap, offs, inverse_map = permute_tokens(
        x16,
        permuted_probs,
        routing_map,
        local_expert_start,
        module.num_local_experts,
        valid_tokens,
        alignment=BLOCK_M,
        return_batch_invariant_inverse_map=True,
        row_alignment=BLOCK_M,
    )
    tiles = tile_experts_from_offsets(offs, xp.shape[0] // BLOCK_M)
    h = grouped_gemm(xp, None, tiles, stacked=module._fc1_weight)
    a = swiglu_probs(h, pp)
    y = grouped_gemm(a.to(torch.float16), None, tiles, stacked=module._fc2_weight)
    out = None
    if module._nvls_dispatcher:
        out = NVLSAllGatherVDispatcher._get_rsv_tensor()
    return unpermute_fp64(y, inverse_map, valid_tokens, out), None


def validate_model(model_cfg: Any) -> list[str]:
    """MoE settings the fp16 expert path covers; returns what is unsupported."""
    bad = []
    if not getattr(model_cfg, "moe_grouped_gemm", False):
        bad.append("moe_grouped_gemm=False")
    if not getattr(model_cfg, "gated_linear_unit", False):
        bad.append("non-gated expert MLP")
    if getattr(model_cfg, "moe_mlp_glu_interleave_size", None):
        bad.append("moe_mlp_glu_interleave_size")
    if getattr(model_cfg, "add_bias_linear", False):
        bad.append("expert biases")
    if getattr(model_cfg, "moe_shared_expert_intermediate_size", None):
        bad.append("shared experts")
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
    act = getattr(model_cfg, "activation_func", None)
    if act is not None and getattr(act, "__name__", "") != "silu":
        bad.append(f"activation_func={getattr(act, '__name__', act)}")
    return bad
