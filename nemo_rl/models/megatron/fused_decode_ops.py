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
"""Single-kernel versions of small op sequences on the engine's decode path (bitwise equal).

A full-Nano decode step is ~2k small kernels; at few rows each costs its launch/drain
latency, not its bytes. The kernels here replace an op-by-op sequence with one kernel that
performs the same IEEE fp32 operations in the same order, so the outputs are bitwise the
sequence's (unit-tested against it: tests/unit/models/megatron/test_fused_decode_ops.py).

- ``rms_norm_fp32_nograd``: ``fp16_gemm.RMSNormFp32.forward`` (x.float(); w.float() [+1];
  x*x; Megatron-Core's batch-invariant ``mean_dim``; + eps; rsqrt; (x * r) * w): x*x and
  mean_dim are kept (MCore's own kernel does the reduction), the rest is one kernel: 3 kernels
  per call instead of 7.

Per rank cost: none beyond the replaced ops (no buffers, no communication); the kernels are
row-local, so they are independent of TP/PP/EP and of the batch size.

``THUNDERSYNC_FUSED_DECODE_OPS=0`` turns all of them off (A/B).
"""

from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
    
    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

ENABLED = {"on": os.environ.get("THUNDERSYNC_FUSED_DECODE_OPS", "1") != "0"}
_MAX_N = 1 << 16

if HAVE_TRITON:

    @triton.jit
    def _rms_apply_kernel(
        x_ptr,
        ms_ptr,
        w_ptr,
        y_ptr,
        stride_x,
        stride_y,
        N,
        eps,
        ZERO_CENTERED: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """r = rsqrt(ms + eps); y = (x * r) * w (w + 1 when zero-centered), fp32.

        ms comes from Megatron-Core's own ``mean_dim`` (the reduction is not re-implemented:
        a Triton copy of it got a different reduction layout, hence order, inside a fused kernel).
        """
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        r = tl.math.rsqrt(tl.load(ms_ptr + row) + eps)
        v = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        if ZERO_CENTERED:
            w = w + 1.0
        tl.store(y_ptr + row * stride_y + offs, (v * r) * w, mask=mask)


def rms_norm_fp32_supported(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> bool:
    # eps >= fp32's smallest normal keeps rsqrt's argument normal (Triton's rsqrt flushes
    # subnormal inputs to zero; torch.rsqrt does not).
    return (
        ENABLED["on"]
        and eps >= 1.1754943508222875e-38
        and HAVE_TRITON
        and x.is_cuda
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and weight.dim() == 1
        and weight.is_contiguous()
        and x.shape[-1] == weight.shape[0]
        and 0 < x.shape[-1] <= _MAX_N
        and x.stride(-1) == 1
    )


def rms_norm_fp32_nograd(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered_gamma: bool
) -> torch.Tensor:
    """fp32 RMSNorm of x's rows (fp32 output), bitwise ``RMSNormFp32.forward``; no autograd."""
    n = x.shape[-1]
    x2 = x.reshape(-1, n)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty(x2.shape, dtype=torch.float32, device=x.device)
    m = x2.shape[0]
    if m:
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import mean_dim

        x32 = x2.float()
        ms = mean_dim(x32 * x32, dim=-1, keepdim=True)  # as RMSNormFp32.forward
        _rms_apply_kernel[(m, triton.cdiv(n, 1024))](
            x2, ms, weight, y, x2.stride(0), y.stride(0), n, eps,
            ZERO_CENTERED=bool(zero_centered_gamma), BLOCK_SIZE=1024,
        )
    return y.view(*x.shape[:-1], n)


# ---------------------------------------------------------------------------
# fp32-input GEMM (router, LM head): x = hi + lo / 2^11 split, one GEMM, combine.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _hilo_split_kernel(
        x_ptr,
        out_ptr,
        stride_x,
        M,
        K,
        SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out[:M] = fp16(x); out[M:] = fp16((x - float(fp16(x))) * SCALE)  (fp16_gemm.fp32_gemm's split)."""
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < K
        x = tl.load(x_ptr + row * stride_x + offs, mask=mask, other=0.0).to(tl.float32)
        hi = x.to(tl.float16)
        lo = ((x - hi.to(tl.float32)) * SCALE).to(tl.float16)
        tl.store(out_ptr + row * K + offs, hi, mask=mask)
        tl.store(out_ptr + (row + M) * K + offs, lo, mask=mask)

    @triton.jit
    def _hilo_combine_kernel(
        both_ptr,
        out_ptr,
        M,
        N,
        stride_out,
        SCALE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """out = both[:M] + both[M:] / SCALE (IEEE div.rn then add.rn, as torch's div_ + add)."""
        row = tl.program_id(0).to(tl.int64)
        offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        h = tl.load(both_ptr + row * N + offs, mask=mask, other=0.0)
        lo = tl.load(both_ptr + (row + M) * N + offs, mask=mask, other=0.0)
        q = tl.inline_asm_elementwise(
            "div.rn.f32 $0, $1, $2;", "=r,r,r", [lo, tl.full(lo.shape, SCALE, tl.float32)],
            dtype=tl.float32, is_pure=True, pack=1,
        )
        s = tl.inline_asm_elementwise(
            "add.rn.f32 $0, $1, $2;", "=r,r,r", [h, q], dtype=tl.float32, is_pure=True, pack=1,
        )
        tl.store(out_ptr + row * stride_out + offs, s, mask=mask)


def fp32_gemm_supported(x: torch.Tensor, w16t: torch.Tensor) -> bool:
    return (
        ENABLED["on"]
        and HAVE_TRITON
        and x.is_cuda
        and x.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and x.stride(-1) == 1
        and w16t.dtype == torch.float16
    )


def fp32_gemm_chunk(x32: torch.Tensor, w16t: torch.Tensor, out: torch.Tensor, scale: float) -> None:
    """One ``_fp32_gemm_chunked`` step: out = hi @ w + (lo @ w) / scale, bitwise; 3 kernels instead of 9."""
    m, k = x32.shape
    n = w16t.shape[1]
    both16 = torch.empty((2 * m, k), dtype=torch.float16, device=x32.device)
    _hilo_split_kernel[(m, triton.cdiv(k, 1024))](x32, both16, x32.stride(0), m, k, SCALE=scale, BLOCK=1024)
    both = torch.mm(both16, w16t, out_dtype=torch.float32)
    assert out.stride(-1) == 1
    _hilo_combine_kernel[(m, triton.cdiv(n, 1024))](both, out, m, n, out.stride(0), SCALE=scale, BLOCK=1024)
