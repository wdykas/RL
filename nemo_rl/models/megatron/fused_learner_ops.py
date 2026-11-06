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
"""Bitwise-equal fusions of the learner's fp32 data movement (fp16-GEMM recipe, training).

Each fusion performs the same IEEE operations on the same values as the op sequence it
replaces, so logits, log-probs, loss and every gradient bit are unchanged (unit tests:
tests/unit/models/megatron/test_fused_learner_ops.py; plus a model-level bitwise probe).

1. **wgrad -> main_grad** (``accumulate_wgrad``). The recipe's weight gradient is an fp32
   GEMM output rounded to the weight dtype (bf16) and returned to autograd; Megatron DDP's
   hook then adds that bf16 tensor into the fp32 ``main_grad`` (``main_grad.add_(grad)``:
   fp32 + float(bf16), round to nearest): a cast kernel plus a mixed-dtype add per parameter
   tensor (full Nemotron-3-Nano EP4, 2x8192: AccumulateGrad 34.6 ms -> 1.7 ms, research/learnfuse/
   full-ep4-device-*). Here one kernel per GEMM (per expert group) reads the fp32
   wgrad, rounds it to bf16, widens it back and adds it into ``main_grad`` (the same two
   roundings), then the backward returns an uninitialized dummy and marks the parameter
   ``grad_added_to_main_grad`` (Megatron's gradient-accumulation-fusion contract: the DDP
   hook skips its add). Only for weights used once per backward (TE linears, expert
   linears), fp32 ``main_grad``, bf16 weights, only when the running backward executes the
   weight's AccumulateGrad node (not under ``autograd.grad`` w.r.t. other inputs), and
   never inside a CUDA-graph capture or Megatron's capture flag (graph_capture.py adds
   into main_grad itself).

Per rank cost: no buffers (the dummy gradient is a caching-allocator block of the bf16
gradient's size, as before), no communication; cost is proportional to the parameters a
rank holds, independent of TP/PP/EP/DP size and of the token count.

``NEMO_RL_LEARNER_FUSED=0`` turns every fusion here off (A/B).
"""

from __future__ import annotations

import os
from typing import Sequence

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

ENABLED = {"on": os.environ.get("NEMO_RL_LEARNER_FUSED", "1") != "0"}
# Parameters whose wgrad went into main_grad here (learner_graphs must not read their
# grad_added_to_main_grad flag as TE's fused accumulation).
FLAG = "_learnfuse_wgrad"
_ANNOUNCED = {"wgrad": False}


def enabled() -> bool:
    return ENABLED["on"] and HAVE_TRITON


if HAVE_TRITON:

    @triton.jit
    def _accum_bf16_kernel(src_ptr, dst_ptrs, n, BLOCK: tl.constexpr):
        """dst_e += float(bf16(src[e])) for e = program_id(1) (round-to-nearest-even, IEEE add)."""
        e = tl.program_id(1).to(tl.int64)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        dst = tl.load(dst_ptrs + e).to(tl.pointer_type(tl.float32))
        w = tl.load(src_ptr + e * n + offs, mask=mask, other=0.0)
        r = w.to(tl.bfloat16).to(tl.float32)
        m = tl.load(dst + offs, mask=mask, other=0.0)
        s = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=r,r,r", [m, r], dtype=tl.float32, is_pure=True, pack=1
        )
        tl.store(dst + offs, s, mask=mask)


def _capturing() -> bool:
    if torch.cuda.is_current_stream_capturing():
        return True
    try:
        from megatron.core.transformer.cuda_graphs import is_graph_capturing

        return bool(is_graph_capturing())
    except ImportError:  # pragma: no cover
        return False


def wgrad_fp32() -> bool:
    """fp32 weight-gradient accumulation (``NEMO_RL_WGRAD_FP32=1``; on by default under BI context parallelism).

    Megatron's contract without gradient-accumulation fusion rounds each call's weight gradient to the parameter
    dtype (bf16) before DDP adds it into the fp32 ``main_grad``. Under context parallelism every rank holds a
    partial (its tokens) of each weight gradient, so that rounding happens once per rank instead of once per
    micro-batch, and the CP sum of bf16-rounded partials differs from CP 1 by up to a bf16 ulp of each partial
    (sparse experts: 0.85 relative measured). Here the fp32 GEMM output goes straight into
    ``main_grad`` (no bf16 rounding) -- also inside learner CUDA-graph captures, where the add is captured.
    """
    import os

    v = os.environ.get("NEMO_RL_WGRAD_FP32")
    if v is not None:
        return v == "1"
    try:
        from nemo_rl.models.megatron import bi_context_parallel

        return bi_context_parallel.size() > 1
    except ImportError:  # pragma: no cover
        return False


def wgrad_fusable(weights: Sequence[torch.Tensor], wgrad: torch.Tensor) -> bool:
    """Whether ``accumulate_wgrad`` applies (every weight bf16 with an fp32 main_grad of its shape)."""
    fp32 = wgrad_fp32()
    if not ((enabled() or fp32) and wgrad.is_cuda and wgrad.dtype == torch.float32) or (
        _capturing() and not fp32
    ):
        return False
    for w in weights:
        mg = getattr(w, "main_grad", None)
        if (
            mg is None
            or w.dtype != torch.bfloat16
            or mg.dtype != torch.float32
            or mg.shape != w.shape
            or not mg.is_contiguous()
            or not hasattr(w, "grad_added_to_main_grad")
            # weights with several uses per backward (tied embedding / head: Megatron marks
            # them) get their bf16 gradients summed by autograd before the hook: not fusable
            or getattr(w, "zero_out_wgrad", False)
            or getattr(w, "shared_embedding", False)
            or getattr(w, "shared", False)
            or not _will_accumulate(w)
        ):
            return False
    return True


def _will_accumulate(w: torch.Tensor) -> bool:
    """Whether this backward reaches w's AccumulateGrad node (False e.g. under
    ``torch.autograd.grad`` with respect to other inputs: then nothing may go to main_grad)."""
    try:
        from torch.autograd.graph import get_gradient_edge

        return bool(torch._C._will_engine_execute_node(get_gradient_edge(w).node))
    except (RuntimeError, AttributeError, ImportError):  # outside a backward / no grad edge
        return False


_PTRS: dict = {}


def _ptr_table(weights: Sequence[torch.Tensor], device) -> torch.Tensor:
    ptrs = tuple(w.main_grad.data_ptr() for w in weights)
    t = _PTRS.get(ptrs)
    if t is None:
        if len(_PTRS) > 4096:
            _PTRS.clear()
        t = torch.tensor(ptrs, dtype=torch.int64).to(device, non_blocking=True)
        _PTRS[ptrs] = t
    return t


def accumulate_wgrad(wgrad: torch.Tensor, weights: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """main_grad_e += float(bf16(wgrad[e])) for each weight; returns autograd's dummy grads.

    ``wgrad``: fp32, contiguous, [len(weights), *w.shape] (or w.shape for one weight).
    Bitwise the same as returning ``wgrad.to(bf16)`` and letting Megatron DDP's hook run
    ``main_grad.add_(grad)``. Call only after ``wgrad_fusable``.
    """
    if wgrad_fp32():
        # main_grad_e += wgrad[e] in fp32 (no bf16 rounding). During a learner-graph capture the add is part
        # of the backward graph and the weight gets no gradient (graph_capture adds nothing for it); eagerly,
        # Megatron's dummy gradient keeps DDP's hook bookkeeping (it skips the add).
        capturing = _capturing()
        wgrad = wgrad.contiguous()
        out = []
        for e, w in enumerate(weights):
            g = wgrad[e] if wgrad.dim() == w.dim() + 1 else wgrad
            w.main_grad.add_(g.view(w.main_grad.shape))
            if not capturing:
                w.grad_added_to_main_grad = True
                setattr(w, FLAG, True)
                out.append(torch.empty(w.shape, dtype=w.dtype, device=wgrad.device))
            else:
                out.append(None)
        return out
    if not _ANNOUNCED["wgrad"]:  # run evidence for the rules (run_rules log.learner_fused); host only, once
        _ANNOUNCED["wgrad"] = True
        print("[learner_fused] wgrad -> main_grad fused accumulate active (NEMO_RL_LEARNER_FUSED=1)", flush=True)
    wgrad = wgrad.contiguous()
    n = weights[0].numel()
    block = 4096
    _accum_bf16_kernel[(triton.cdiv(n, block), len(weights))](
        wgrad, _ptr_table(weights, wgrad.device), n, BLOCK=block, num_warps=8
    )
    out = []
    for w in weights:
        w.grad_added_to_main_grad = True
        setattr(w, FLAG, True)
        # Megatron's contract: a dummy gradient (never read; the DDP hook skips the add).
        out.append(torch.empty(w.shape, dtype=w.dtype, device=wgrad.device))
    return out


# ---------------------------------------------------------------------------
# 2. fp32 RMSNorm backward (fp16_gemm.RMSNormFp32.backward), elementwise parts fused.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _mul(a, b):
        return tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _sub(a, b):
        return tl.inline_asm_elementwise(
            "sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _add(a, b):
        return tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1
        )

    @triton.jit
    def _rms_bwd_p_kernel(g_ptr, x_ptr, w_ptr, p_ptr, D, ZC: tl.constexpr, BLOCK: tl.constexpr):
        """p = (g * w) * x  (the row-sum operand of RMSNormFp32.backward)."""
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        g = tl.load(g_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        if ZC:
            w = _add(w, 1.0)
        tl.store(p_ptr + row * D + offs, _mul(_mul(g, w), x), mask=m)

    @triton.jit
    def _rms_bwd_dx_kernel(
        g_ptr, x_ptr, w_ptr, r_ptr, s_ptr, dx_ptr, q_ptr, D, inv_d, ZC: tl.constexpr, BLOCK: tl.constexpr
    ):
        """dx = g * (w * r) - ((r * r * r) * x) * (s * inv_d);  q = (g * x) * r  (dw's column-sum operand)."""
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = offs < D
        g = tl.load(g_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + row * D + offs, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=m, other=0.0).to(tl.float32)
        if ZC:
            w = _add(w, 1.0)
        r = tl.load(r_ptr + row)
        s = tl.load(s_ptr + row)
        r3 = _mul(_mul(r, r), r)
        sd = _mul(s, inv_d)
        dx = _sub(_mul(g, _mul(w, r)), _mul(_mul(r3, x), sd))
        tl.store(dx_ptr + row * D + offs, dx, mask=m)
        tl.store(q_ptr + row * D + offs, _mul(_mul(g, x), r), mask=m)


def rms_bwd_supported(grad: torch.Tensor, x: torch.Tensor, weight: torch.Tensor, r: torch.Tensor) -> bool:
    return (
        enabled()
        and ENABLED.get("norm", True)
        and grad.is_cuda
        and x.dtype == torch.float32
        and grad.dtype == torch.float32
        and r.dtype == torch.float32
        and weight.dim() == 1
        and x.shape[-1] == weight.shape[0]
        and grad.shape == x.shape
        and r.numel() * x.shape[-1] == x.numel()
    )


def rms_norm_fp32_backward(grad, x, weight, r, zero_centered_gamma: bool):
    """(dx, dw) bitwise ``fp16_gemm.RMSNormFp32.backward``: same fp32 ops in the same order;
    the two reductions (row sum of (g*w)*x, column sum of (g*x)*r) stay torch's kernels on
    the same operands. 2 elementwise kernels instead of 11."""
    d = x.shape[-1]
    g2 = grad.reshape(-1, d).contiguous()
    x2 = x.reshape(-1, d).contiguous()
    w = weight.contiguous()
    rows = x2.shape[0]
    block = min(triton.next_power_of_2(d), 4096)
    grid = (rows, triton.cdiv(d, block))
    zc = bool(zero_centered_gamma)
    p = torch.empty_like(x2)
    if rows:
        _rms_bwd_p_kernel[grid](g2, x2, w, p, d, ZC=zc, BLOCK=block)
    s = p.sum(-1, keepdim=True)
    del p
    # torch divides by a host scalar as a multiplication by its fp32 reciprocal
    inv_d = float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(d), dtype=torch.float32))
    dx = torch.empty_like(x2)
    q = torch.empty_like(x2)
    if rows:
        _rms_bwd_dx_kernel[grid](
            g2, x2, w, r.reshape(-1).contiguous(), s.reshape(-1), dx, q, d, inv_d, ZC=zc, BLOCK=block
        )
    dw = q.sum(0).to(weight.dtype)
    return dx.view(x.shape), dw
