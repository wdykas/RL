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
"""Fused learner ops are bitwise the op sequences they replace."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _hard_values(shape, gen):
    """Normal values over many binades plus subnormals, +-0, ties for bf16 rounding."""
    n = 1
    for s in shape:
        n *= s
    x = torch.randn(n, generator=gen) * torch.exp2(torch.randint(-140, 60, (n,), generator=gen).float())
    bits = x.view(torch.int32)
    # every 7th value an exact bf16 tie (low 16 bits 0x8000), every 11th a subnormal
    bits[::7] = (bits[::7] & ~0xFFFF) | 0x8000
    bits[3::11] = torch.randint(1, 1 << 23, bits[3::11].shape, generator=gen, dtype=torch.int32)
    x = bits.view(torch.float32)
    x[5::13] = 0.0
    x[6::13] = -0.0
    return x.view(shape)


def _params(shapes, gen):
    ps = []
    for s in shapes:
        p = torch.nn.Parameter(torch.zeros(s, dtype=torch.bfloat16, device="cuda"))
        p.main_grad = _hard_values(s, gen).cuda()
        p.grad_added_to_main_grad = False
        ps.append(p)
    return ps


@cuda
@pytest.mark.parametrize("experts", [1, 3, 32])
def test_accumulate_wgrad_matches_ddp_hook(experts):
    from nemo_rl.models.megatron import fused_learner_ops as flo

    gen = torch.Generator().manual_seed(experts)
    shape = (96, 133)  # not a multiple of the block
    ws = _params([shape] * experts, gen)
    wg = _hard_values((experts, *shape), gen).cuda()
    ref = [p.main_grad.clone() for p in ws]
    for r, g in zip(ref, wg.to(torch.bfloat16).unbind(0)):
        r.add_(g)  # Megatron DDP hook: main_grad.add_(param.grad)
    dummies = flo.accumulate_wgrad(wg if experts > 1 else wg[0], ws)
    for p, r, d in zip(ws, ref, dummies):
        assert torch.equal(p.main_grad.view(torch.int32), r.view(torch.int32))
        assert p.grad_added_to_main_grad and d.shape == p.shape and d.dtype == p.dtype


@cuda
def test_fp16_linear_backward_fused_equals_unfused(monkeypatch):
    """Fp16Linear: main_grad after DDP's hook (unfused) == main_grad written by the fused backward."""
    from nemo_rl.models.megatron import fp16_gemm as fg
    from nemo_rl.models.megatron import fused_learner_ops as flo

    gen = torch.Generator().manual_seed(0)
    x = torch.randn(300, 256, generator=gen).cuda().requires_grad_()
    out = {}
    for fused in (False, True):
        monkeypatch.setitem(flo.ENABLED, "on", fused)
        (w,) = _params([(192, 256)], torch.Generator().manual_seed(1))
        w.data.copy_(torch.randn(192, 256, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16))
        y = fg.fp16_linear(x, w, fuse_wgrad=True)
        gy = torch.randn(y.shape, generator=torch.Generator().manual_seed(3)).cuda()
        (gx,) = torch.autograd.grad(y, (x,), gy, retain_graph=True)
        y.backward(gy)
        if not w.grad_added_to_main_grad:
            w.main_grad.add_(w.grad)
        out[fused] = (w.main_grad.clone(), gx)
    assert torch.equal(out[False][0].view(torch.int32), out[True][0].view(torch.int32))
    assert torch.equal(out[False][1], out[True][1])


@cuda
def test_not_fusable_outside_backward_or_without_main_grad():
    from nemo_rl.models.megatron import fused_learner_ops as flo

    w = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16, device="cuda"))
    assert not flo.wgrad_fusable((w,), torch.zeros(4, 4, device="cuda"))
    (p,) = _params([(4, 4)], torch.Generator().manual_seed(0))
    assert not flo.wgrad_fusable((p,), torch.zeros(4, 4, device="cuda"))  # no backward running


@cuda
@pytest.mark.parametrize("shape", [(37, 2, 2688), (5, 128), (300, 1, 4096), (3, 5000)])
@pytest.mark.parametrize("zc", [False, True])
def test_rmsnorm_backward_fused_equals_eager(monkeypatch, shape, zc):
    from nemo_rl.models.megatron import fp16_gemm as fg
    from nemo_rl.models.megatron import fused_learner_ops as flo

    gen = torch.Generator().manual_seed(len(shape) * 7 + int(zc))
    x0 = (torch.randn(shape, generator=gen) * torch.exp2(torch.randint(-8, 8, shape[:-1] + (1,), generator=gen).float())).cuda()
    w0 = (torch.randn(shape[-1], generator=gen) * 0.3 + (0.0 if zc else 1.0)).to(torch.bfloat16).cuda()
    gy = torch.randn(shape, generator=gen).cuda()
    out = {}
    for fused in (False, True):
        monkeypatch.setitem(flo.ENABLED, "on", fused)
        x = x0.clone().requires_grad_()
        w = w0.clone().requires_grad_()
        y = fg.RMSNormFp32.apply(x, w, 1e-5, zc)
        dx, dw = torch.autograd.grad(y, (x, w), gy)
        out[fused] = (y, dx, dw)
    for a, b in zip(out[False], out[True]):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert torch.equal(a.view(torch.int16 if a.element_size() == 2 else torch.int32),
                           b.view(torch.int16 if b.element_size() == 2 else torch.int32))
