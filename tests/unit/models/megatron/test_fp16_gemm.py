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
"""CPU tests of the fp16 GEMM forward (nemo_rl/models/megatron/fp16_gemm.py)."""

import pytest
import torch

from nemo_rl.models.megatron import fp16_gemm as fg


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setitem(fg._STATE, "enabled", True)


def _reference(x, w):
    """V8b reference: fp16 operands, exact products, fp64-accurate sum rounded to fp32."""
    x16 = x.to(torch.float16).double()
    w16 = w.to(torch.float16).double()
    return (x16 @ w16.t()).float()


def test_fp16_gemm_dtype_shape_and_values():
    torch.manual_seed(0)
    x = torch.randn(3, 5, 64) * 4
    w = (torch.randn(32, 64) * 0.05).to(torch.bfloat16)
    y = fg.fp16_gemm(x, w)
    assert y.dtype == torch.float32 and y.shape == (3, 5, 32)
    torch.testing.assert_close(y, _reference(x, w), rtol=1e-5, atol=1e-5)


def test_fp16_gemm_rounds_input_to_fp16_not_bf16():
    # 1 + 2^-9 is exact in fp16, rounds to 1 in bf16.
    x = torch.full((1, 4), 1.0 + 2.0**-9)
    w = torch.ones(1, 4, dtype=torch.bfloat16)
    assert fg.fp16_gemm(x, w).item() == 4 * (1.0 + 2.0**-9)


def test_bf16_weight_cast_to_fp16_is_exact_in_range():
    w = (torch.randn(1000) * 10).to(torch.bfloat16)
    w = w[(w.abs() >= 6.2e-5) & (w.abs() <= 65504)]
    assert torch.equal(w.to(torch.float16).float(), w.float())


def test_fp16_linear_forward_matches_gemm_and_backward_is_bf16():
    torch.manual_seed(2)
    x = torch.randn(6, 2, 32, requires_grad=True)
    w = (torch.randn(16, 32) * 0.1).to(torch.bfloat16).requires_grad_()
    y = fg.fp16_linear(x, w)
    assert torch.equal(y.detach(), fg.fp16_gemm(x.detach(), w.detach()))
    g = torch.randn_like(y)
    y.backward(g)
    assert x.grad.dtype == torch.float32 and w.grad.dtype == torch.bfloat16
    g_b = g.reshape(-1, 16).to(torch.bfloat16).float()
    dx = (g_b @ w.detach().float()).view_as(x)
    dw = g_b.t() @ x.detach().reshape(-1, 32).to(torch.float16).to(torch.bfloat16).float()
    torch.testing.assert_close(x.grad, dx)
    torch.testing.assert_close(w.grad.float(), dw.to(torch.bfloat16).float())


def test_fp16_linear_no_grad_forward():
    x = torch.randn(4, 8)
    w = torch.randn(3, 8, dtype=torch.bfloat16)
    with torch.inference_mode():
        y = fg.fp16_linear(x, w)
    assert torch.equal(y, fg.fp16_gemm(x, w))


def test_rms_norm_fp32():
    torch.manual_seed(3)
    x = torch.randn(5, 64) * 3
    w = torch.randn(64).to(torch.bfloat16)
    y = fg.rms_norm_fp32(x, w, 1e-6)
    ref = x.double() * torch.rsqrt(x.double().pow(2).mean(-1, keepdim=True) + 1e-6)
    assert y.dtype == torch.float32
    torch.testing.assert_close(y, (ref * w.double()).float(), rtol=1e-5, atol=1e-6)


def test_attention_grad_scaling_roundtrip():
    """fp16 region boundary: forward exact, backward scaled into fp16 and back."""
    box = fg._GradScaleBox()
    q = torch.randn(4, 8, requires_grad=True)
    q16 = fg.ToFp16Scaled.apply(q, box)
    assert q16.dtype == torch.float16
    out16 = q16 * 2  # stand-in for attention (linear in its gradient)
    out = fg.FromFp16Scaled.apply(out16, box)
    assert out.dtype == torch.float32 and torch.equal(out, out16.float())
    g = torch.full_like(out, 3e-9)  # underflows fp16 (min subnormal 6e-8) unscaled
    out.backward(g)
    assert box.scale is not None and box.scale.item() > 1
    torch.testing.assert_close(q.grad, torch.full_like(q, 6e-9), rtol=1e-3, atol=0)


def test_head_helpers_fp32_head(enabled):
    assert fg.head_precision() == "fp32"
    assert fg.head_input_dtype(torch.bfloat16) == torch.float32
    h = torch.randn(3, 16)
    w = torch.randn(5, 16).to(torch.bfloat16)
    ref = (h.double() @ w.double().t()).float()
    torch.testing.assert_close(fg.head_matmul(h, w), ref, rtol=1e-6, atol=1e-6)
    assert torch.equal(fg.head_matmul(h, w), fg.fp32_gemm(h, w))


def test_head_helpers_fp16_head(enabled, monkeypatch):
    monkeypatch.setitem(fg._STATE, "head", "fp16")
    assert fg.head_input_dtype(torch.bfloat16) == torch.float16
    h = torch.randn(3, 16)
    w = torch.randn(5, 16).to(torch.bfloat16)
    assert torch.equal(fg.head_matmul(h, w), fg.fp16_gemm(h, w))
    # The fp16 hidden copy gives the same logits as the fp32 head input.
    assert torch.equal(fg.head_matmul(h.to(torch.float16), w), fg.fp16_gemm(h, w))


def test_fp32_head_linear_keeps_fp32_input():
    x = torch.full((1, 4), 1.0 + 2.0**-20)  # not representable in fp16
    w = torch.ones(1, 4, dtype=torch.bfloat16)
    y = fg.fp16_linear(x, w, precision="fp32")
    assert y.item() == 4 * (1.0 + 2.0**-20)


def test_head_helpers_disabled():
    assert not fg.enabled()
    assert fg.head_input_dtype(torch.bfloat16) == torch.bfloat16


def test_active_requires_install_and_16bit_weight(enabled):
    assert fg._active(torch.zeros(1, dtype=torch.bfloat16))
    assert not fg._active(torch.zeros(1, dtype=torch.float32))
    assert not fg._active(None)


def _cfg(**kw):
    base = {"forward_gemm_precision": "fp16", "fp32_residual_connection": True}
    base.update(kw)
    return base


def test_validate_config():
    assert fg.validate_config({}) == []
    assert fg.validate_config(_cfg(), _cfg()) == []
    assert fg.validate_config(_cfg(forward_gemm_precision="fp8"))
    assert any(
        "fp32_residual_connection" in e
        for e in fg.validate_config(_cfg(fp32_residual_connection=False))
    )
    assert any(
        "same in training and generation" in e
        for e in fg.validate_config(_cfg(), _cfg(forward_gemm_precision=None))
    )
    assert fg.validate_config(_cfg(gradient_accumulation_fusion=True))
    assert fg.validate_config(_cfg(fp8_cfg={"enabled": True}))
    assert fg.validate_config(_cfg(lm_head_precision="bf16"))
    assert fg.validate_config({"lm_head_precision": "fp32"})
    assert fg.validate_config(_cfg(lm_head_precision="fp32"), _cfg()) != []
    assert fg.validate_config(_cfg(lm_head_precision="fp32"), _cfg(lm_head_precision="fp32")) == []


def test_validate_model_rejects_moe():
    class Cfg:
        num_moe_experts = 8
        normalization = "RMSNorm"

    with pytest.raises(ValueError, match="MoE"):
        fg.validate_model(Cfg())

    class Dense:
        num_moe_experts = None
        normalization = "RMSNorm"

    fg.validate_model(Dense())
