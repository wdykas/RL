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
"""CPU tests of the fp16 MoE expert path (nemo_rl/models/megatron/fp16_moe.py)."""

import pytest
import torch

from nemo_rl.models.megatron import fp16_moe as fm


def _experts(e, n, k, seed=0):
    torch.manual_seed(seed)
    return [(torch.randn(n, k) * 0.05).to(torch.bfloat16) for _ in range(e)]


def test_select_config_small_tiles_for_decode():
    assert fm.select_config(4)[0] == 16
    assert fm.select_config(1000)[0] == 128


def test_aligned_rows():
    dest, offs, m_pad = fm._aligned_rows([3, 0, 17], 16, "cpu")
    assert m_pad == 48
    assert offs.tolist() == [16, 16, 48]
    assert dest.tolist() == [0, 1, 2] + list(range(16, 33))


def test_grouped_linear_matches_per_expert_fp16_reference():
    w = _experts(3, 16, 32)
    splits = [5, 0, 7]
    x = torch.randn(12, 32) * 2
    y = fm.grouped_linear(x, splits, w)
    assert y.dtype == torch.float32
    ref, s = [], 0
    for we, m in zip(w, splits):
        ref.append(
            x[s : s + m].to(torch.float16).double() @ we.to(torch.float16).double().t()
        )
        s += m
    torch.testing.assert_close(y.double(), torch.cat(ref), rtol=1e-5, atol=1e-5)


def test_grouped_linear_backward_bf16():
    w = [t.requires_grad_() for t in _experts(2, 8, 16, seed=1)]
    splits = [3, 4]
    x = torch.randn(7, 16, requires_grad=True)
    g = torch.randn(7, 8)
    fm.grouped_linear(x, splits, w).backward(g)
    gb = g.to(torch.bfloat16).float()
    dx = torch.cat([gb[:3] @ w[0].detach().float(), gb[3:] @ w[1].detach().float()])
    torch.testing.assert_close(x.grad, dx)
    xb = x.detach().to(torch.float16).to(torch.bfloat16).float()
    torch.testing.assert_close(
        w[1].grad.float(), (gb[3:].t() @ xb[3:]).to(torch.bfloat16).float()
    )
    assert w[0].grad.dtype == torch.bfloat16


def test_swiglu_probs():
    h = torch.randn(4, 6, requires_grad=True)
    p = torch.rand(4)
    out = fm.swiglu_probs(h, p)
    out.sum().backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()
    h = h.detach()
    gate, up = h[:, :3], h[:, 3:]
    torch.testing.assert_close(out, torch.nn.functional.silu(gate) * up * p[:, None])


def test_validate_model():
    class Cfg:
        moe_grouped_gemm = True
        gated_linear_unit = True
        moe_mlp_glu_interleave_size = None
        add_bias_linear = False
        moe_shared_expert_intermediate_size = None
        moe_apply_probs_on_input = False
        moe_latent_size = None
        use_te_activation_func = False
        expert_tensor_parallel_size = 1
        moe_token_dispatcher_type = "alltoall"
        activation_func = torch.nn.functional.silu

    assert fm.validate_model(Cfg()) == []
    # shared experts (dense MLP on the fp16 GEMM path) are allowed
    Cfg.moe_shared_expert_intermediate_size = 64
    assert fm.validate_model(Cfg()) == []
    # non-gated squared-ReLU experts (Nemotron-H) are built; gated squared ReLU is not
    from megatron.core.activations import squared_relu

    Cfg.activation_func = staticmethod(squared_relu)
    Cfg.gated_linear_unit = False
    assert fm.validate_model(Cfg()) == []
    Cfg.gated_linear_unit = True
    assert len(fm.validate_model(Cfg())) == 1
    Cfg.gated_linear_unit = False
    Cfg.moe_latent_size = 1024
    assert fm.validate_model(Cfg()) == ["moe_latent_size"]


def test_expert_down_proj_prescale_matches(monkeypatch):
    """Training expert forward: the prescale (via routing probs) only perturbs subnormals.

    x * 2^-s rounds to fp16 exactly as x does unless it lands in fp16's subnormal
    range (|x| < 6.1e-5 * 2^s), so outputs agree to within those tiny inputs.
    """
    from types import SimpleNamespace

    from nemo_rl.models.megatron import fp16_gemm

    torch.manual_seed(7)
    k, f, experts = 32, 24, 3

    def linear(rows, cols):
        ws = {
            f"weight{i}": (torch.randn(rows, cols) * 0.1).to(torch.bfloat16)
            for i in range(experts)
        }
        return SimpleNamespace(num_gemms=experts, **ws)

    module = SimpleNamespace(linear_fc1=linear(2 * f, k), linear_fc2=linear(k, f))
    x = torch.randn(10, k) * 3
    tokens = torch.tensor([4, 0, 6])
    probs = torch.rand(10)
    outs = []
    for s in (0, 4):
        monkeypatch.setitem(fp16_gemm._STATE, "down_scale", s)
        outs.append(fm.training_experts_forward(module, x, tokens, probs)[0])
    diff = (outs[0] - outs[1]).abs().max().item()
    print(f"prescale 2^-4 max |diff| {diff:.3e}")
    torch.testing.assert_close(outs[0], outs[1], rtol=0, atol=1e-4)


needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
def test_training_cublas_path_bitwise_equals_engine_kernel():
    """Training (per-expert cuBLAS) and engine (Triton) expert GEMMs give identical bits."""
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ws = [
        (torch.randn(96, 256, device=dev) * 0.02).to(torch.bfloat16) for _ in range(4)
    ]
    splits = [3, 0, 70, 19]
    x = torch.randn(sum(splits), 256, device=dev)
    fm.invalidate_weight_cache()
    y = fm.grouped_linear(x, splits, ws)
    for cfg in fm._CONFIGS:
        bm = cfg[1][0]
        dest, offs, m_pad = fm._aligned_rows(splits, bm, dev)
        xp = torch.zeros(m_pad, 256, device=dev, dtype=torch.float16)
        xp.index_copy_(0, dest, x.half())
        ye = fm.grouped_gemm(xp, ws, offs, cfg[1]).index_select(0, dest)
        assert torch.equal(y, ye), cfg


@needs_cuda
def test_fp16_weight_cache_invalidation():
    dev = torch.device("cuda")
    w = (torch.randn(32, 64, device=dev) * 0.02).to(torch.bfloat16)
    x = torch.randn(8, 64, device=dev)
    fm.invalidate_weight_cache()
    y0 = fm.grouped_linear(x, [8], [w])
    w.data.mul_(2)  # an optimizer-style in-place update (no version bump via .data)
    torch.testing.assert_close(
        fm.grouped_linear(x, [8], [w]), y0
    )  # stale until invalidated
    fm.invalidate_weight_cache()
    torch.testing.assert_close(fm.grouped_linear(x, [8], [w]), 2 * y0)
    fm.invalidate_weight_cache()


@needs_cuda
def test_fp16w_engine_kernel_bitwise_equals_cublas_and_triton():
    torch.manual_seed(0)
    dev = torch.device("cuda")
    e, n, k = 4, 256, 512
    w = (torch.randn(e, n, k, device=dev) * 0.02).to(torch.bfloat16)
    w16 = w.to(torch.float16)
    for _, cfg in fm._ENGINE_CONFIGS:
        bm = cfg[0]
        per = 2 * bm
        x = (torch.randn(per * e, k, device=dev) * 0.5).half()
        offs = torch.arange(1, e + 1, device=dev, dtype=torch.int32) * per
        y = fm.grouped_gemm_fp16w(x, w16, offs, cfg)
        ref = torch.cat(
            [
                torch.mm(
                    x[i * per : (i + 1) * per], w16[i].t(), out_dtype=torch.float32
                )
                for i in range(e)
            ]
        )
        assert torch.equal(y, ref), cfg
        assert torch.equal(
            y,
            fm.grouped_gemm(
                x,
                None,
                offs,
                fm._CONFIGS[0][1] if bm == 16 else (bm, 64, 64, 4, 3),
                stacked=w,
            ),
        ), cfg


@needs_cuda
def test_engine_expert_weights_become_fp16_views():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(3):
                self.register_parameter(
                    f"weight{i}",
                    torch.nn.Parameter(
                        torch.randn(8, 16, device="cuda").to(torch.bfloat16)
                    ),
                )

    class Experts(torch.nn.Module):
        num_local_experts = 3

        def __init__(self):
            super().__init__()
            self.linear_fc1, self.linear_fc2 = Linear(), Linear()
            self._concatenated_weights_built = False

    m = Experts()
    before = m.linear_fc1.weight1.detach().clone()
    fm._ensure_fp16_expert_weights(m)
    assert m._fc1_weight.dtype == torch.float16 and m._fc1_weight.shape == (3, 8, 16)
    assert torch.equal(m._fc1_weight[1], before.to(torch.float16))
    # A refit-style copy into the parameter lands in the fp16 buffer.
    new = torch.randn(8, 16, device="cuda").to(torch.bfloat16)
    with torch.no_grad():
        m.linear_fc1.weight1.copy_(new)
    assert torch.equal(m._fc1_weight[1], new.to(torch.float16))
