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

# Tile shapes do not change a row's bits: tl.dot accumulates the K dimension in
# order into one fp32 accumulator, and products of fp16 operands are exact, so the
# result depends only on the row, its expert's weight and K order. Measured bitwise
# identical over BLOCK_M 16-128, BLOCK_N 32-256, BLOCK_K 32-128, 2-8 warps and 2-4
# stages (research/thundersync_rl/DESIGN.md section 14b). The configuration is picked
# per call from the expected rows per expert: small tiles for decode, large for
# prefill and training.
TileConfig = tuple  # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages)
_CONFIGS: tuple[tuple[int, TileConfig], ...] = (
    (32, (16, 64, 64, 4, 3)),  # decode: up to ~32 rows per expert
    (192, (128, 64, 128, 8, 3)),
    (1 << 62, (128, 64, 64, 4, 3)),
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
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = cols < F
        gate = tl.load(h_ptr + row * 2 * F + cols, mask=mask, other=0.0)
        up = tl.load(h_ptr + row * 2 * F + F + cols, mask=mask, other=0.0)
        p = tl.load(p_ptr + row)
        value = (gate / (1.0 + tl.exp(-gate))) * up * p
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
                        expert_out_ptr + pos.to(tl.int64) * hidden_dim + offsets,
                        mask=mask_h,
                    )
                    acc += vals.to(tl.float64)
            tl.store(
                output_ptr + tok_i64 * hidden_dim + offsets,
                acc.to(tl.float32),
                mask=mask_h,
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


def swiglu_probs_fwd(h: torch.Tensor, probs: torch.Tensor, out_dtype=torch.float32):
    """SiLU(gate) * up * prob in fp32 (one Triton kernel on CUDA), cast to out_dtype."""
    if not h.is_cuda:
        return _swiglu_eager(h, probs).to(out_dtype)
    rows, two_f = h.shape
    f = two_f // 2
    out = torch.empty((rows, f), dtype=out_dtype, device=h.device)
    block = min(triton.next_power_of_2(f), 1024)
    _swiglu_probs_kernel[(rows, triton.cdiv(f, block))](
        h.contiguous(), probs.float().contiguous(), out, f, BLOCK=block
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
        out = torch.empty(
            num_tokens, hidden_dim, dtype=torch.float32, device=expert_output.device
        )
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
        cfg = select_config(sum(splits) / max(len(splits), 1))
        dest, offs, m_pad = _aligned_rows(splits, cfg[0], x.device)
        x_pad = torch.zeros((m_pad, k), dtype=torch.float16, device=x.device)
        x_pad.index_copy_(0, dest, x16)
        out = grouped_gemm(x_pad, list(weights), offs, cfg).index_select(0, dest)
        ctx.save_for_backward(x16, *weights)
        ctx.m_splits = splits
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
    a = swiglu_probs(h, probs)
    y = grouped_linear(a, m_splits, _expert_weights(module.linear_fc2))
    if s:
        y = y * (2.0**s)
    return y, None


def inference_experts_forward(
    module: Any, hidden_states, permuted_probs, routing_map
) -> tuple:
    """InferenceGroupedMLP forward in the engine (CUDA-graph safe)."""
    from megatron.core.inference.moe.permute import permute_tokens
    from megatron.core.transformer.moe.token_dispatcher_inference import (
        InferenceAllGatherDispatcherBase,
        NVLSAllGatherVDispatcher,
    )

    if not module._concatenated_weights_built:
        module._build_concatenated_weights()
        module._concatenated_weights_built = True
    x16 = (
        hidden_states.view(torch.float16)
        if hidden_states.dtype == torch.bfloat16
        else (hidden_states.to(torch.float16))
    )
    valid_tokens = InferenceAllGatherDispatcherBase._valid_tokens()
    local_expert_start = module.ep_group.rank() * module.num_local_experts
    # Expected rows per local expert: tokens x top-k spread over every expert.
    topk = routing_map.shape[1]
    rows_per_expert = (
        x16.shape[0] * topk / (module.num_local_experts * module.ep_group.size())
    )
    cfg = select_config(rows_per_expert)
    xp, pp, _pmap, offs, inverse_map = permute_tokens(
        x16,
        permuted_probs,
        routing_map,
        local_expert_start,
        module.num_local_experts,
        valid_tokens,
        alignment=cfg[0],
        return_batch_invariant_inverse_map=True,
        row_alignment=cfg[0],
    )
    s = _down_scale()
    if s:  # same prescale as training_experts_forward
        pp = pp * (2.0**-s)
    h = grouped_gemm(xp, None, offs, cfg, stacked=module._fc1_weight)
    a = swiglu_probs_fwd(h, pp, torch.float16)
    y = grouped_gemm(a, None, offs, cfg, stacked=module._fc2_weight)
    if s:
        y = y * (2.0**s)
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
