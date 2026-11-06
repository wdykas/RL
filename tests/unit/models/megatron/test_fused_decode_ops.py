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
"""Fused decode ops are bitwise the op-by-op sequences they replace."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@cuda
def test_triton_rsqrt_matches_torch_rsqrt():
    """Every 97th positive normal fp32 bit pattern and inf: tl.math.rsqrt == torch.rsqrt.

    The norm takes rsqrt(mean + eps) with eps > 0, always a normal number (Triton flushes
    subnormal inputs, torch does not; never reached here).
    """
    import triton
    import triton.language as tl

    @triton.jit
    def k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        o = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = o < n
        tl.store(y_ptr + o, tl.math.rsqrt(tl.load(x_ptr + o, mask=m, other=1.0)), mask=m)

    step = 97
    for lo in range(0x00800000, 0x7F800001, step << 24):
        bits = torch.arange(lo, min(lo + (step << 24), 0x7F800001), step, device="cuda", dtype=torch.int64)
        x = bits.to(torch.int32).view(torch.float32)
        y = torch.empty_like(x)
        k[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024)
        ref = torch.rsqrt(x)
        same = (y.view(torch.int32) == ref.view(torch.int32)) | (torch.isnan(y) & torch.isnan(ref))
        assert bool(same.all()), x[~same][:8]


@cuda
@pytest.mark.parametrize("n", [128, 1000, 1024, 2688, 3712, 4096, 7168])
@pytest.mark.parametrize("rows", [1, 8, 64, 513])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("w_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("zc", [False, True])
def test_rms_norm_fp32_fused_bitwise(n, rows, x_dtype, w_dtype, zc):
    from nemo_rl.models.megatron import fused_decode_ops as fdo
    from nemo_rl.models.megatron.fp16_gemm import RMSNormFp32, rms_norm_fp32

    g = torch.Generator(device="cuda").manual_seed(n * 7919 + rows)
    scale = torch.logspace(-4, 4, rows, device="cuda").unsqueeze(1)
    x = (torch.randn(rows, n, device="cuda", generator=g) * scale).to(x_dtype)
    w = (torch.randn(n, device="cuda", generator=g) * 0.5 + 1.0).to(w_dtype)
    eps = 1e-5
    with torch.no_grad():
        ref = RMSNormFp32.apply(x, w, eps, zc)
        assert fdo.rms_norm_fp32_supported(x, w)
        new = rms_norm_fp32(x, w, eps, zc)
    assert new.dtype == ref.dtype == torch.float32 and new.shape == ref.shape
    assert torch.equal(new.view(torch.int32), ref.view(torch.int32))


@cuda
def test_rms_norm_fp32_fused_3d_strided_and_graph():
    from nemo_rl.models.megatron.fp16_gemm import RMSNormFp32, rms_norm_fp32

    g = torch.Generator(device="cuda").manual_seed(3)
    big = torch.randn(4, 3, 2 * 2688, device="cuda", generator=g)
    x = big[..., :2688]  # row stride 2 * 2688, last dim contiguous
    w = torch.randn(2688, device="cuda", generator=g).to(torch.bfloat16)
    with torch.no_grad():
        ref = RMSNormFp32.apply(x, w, 1e-5, False)
        assert torch.equal(rms_norm_fp32(x, w, 1e-5, False), ref)
        xs = x.contiguous().clone()
        rms_norm_fp32(xs, w, 1e-5, False)  # warm up
        cg = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cg):
            out = rms_norm_fp32(xs, w, 1e-5, False)
        for i in range(3):
            xs.copy_(torch.randn(xs.shape, device="cuda", generator=g) * 10**i)
            cg.replay()
            assert torch.equal(out, RMSNormFp32.apply(xs, w, 1e-5, False))


@cuda
def test_rms_norm_fp32_grad_path_unchanged():
    """With autograd the custom Function (and its backward) still runs."""
    from nemo_rl.models.megatron.fp16_gemm import rms_norm_fp32

    x = torch.randn(5, 256, device="cuda", requires_grad=True)
    w = torch.randn(256, device="cuda", requires_grad=True)
    y = rms_norm_fp32(x, w, 1e-5, False)
    assert y.grad_fn is not None and "RMSNormFp32" in type(y.grad_fn).__name__


@cuda
@pytest.mark.parametrize("m", [1, 3, 8, 64, 257])
@pytest.mark.parametrize("k,n", [(2688, 128), (2688, 4096), (1000, 77)])
@pytest.mark.parametrize("x_dtype", [torch.float32, torch.bfloat16])
def test_fp32_gemm_fused_split_combine_bitwise(m, k, n, x_dtype):
    from nemo_rl.models.megatron import fused_decode_ops as fdo
    from nemo_rl.models.megatron.fp16_gemm import _fp32_gemm_chunked

    g = torch.Generator(device="cuda").manual_seed(m * 31 + k + n)
    scale = torch.logspace(-30, 4.5, m, device="cuda").unsqueeze(1)  # incl. tiny rows and fp16 overflow
    x = (torch.randn(m, k, device="cuda", generator=g) * scale).to(x_dtype)
    x[0, :3] = torch.tensor([0.0, -0.0, 1e-40], device="cuda").to(x_dtype)
    w = (torch.randn(n, k, device="cuda", generator=g) * 0.05).to(torch.float16)
    try:
        fdo.ENABLED["on"] = False
        ref = _fp32_gemm_chunked(x, w)
        fdo.ENABLED["on"] = True
        assert fdo.fp32_gemm_supported(x.reshape(-1, k), w.t())
        new = _fp32_gemm_chunked(x, w)
    finally:
        fdo.ENABLED["on"] = True
    same = (new.view(torch.int32) == ref.view(torch.int32)) | (torch.isnan(new) & torch.isnan(ref))
    assert bool(same.all())
