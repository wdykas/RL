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
    return [(torch.randn(n, k) * 0.05).to(torch.float16) for _ in range(e)]


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


def test_grouped_linear_backward_bf16_math():
    w = [t.requires_grad_() for t in _experts(2, 8, 16, seed=1)]
    splits = [3, 4]
    x = torch.randn(7, 16, requires_grad=True)
    g = torch.randn(7, 8)
    fm.grouped_linear(x, splits, w).backward(g)
    gb = g.to(torch.bfloat16).float()
    wb = [t.detach().to(torch.bfloat16).float() for t in w]  # dgrad: bf16(W)
    dx = torch.cat([gb[:3] @ wb[0], gb[3:] @ wb[1]])
    torch.testing.assert_close(x.grad, dx)
    xb = x.detach().to(torch.float16).to(torch.bfloat16).float()
    torch.testing.assert_close(
        w[1].grad.float(), (gb[3:].t() @ xb[3:]).to(torch.bfloat16).to(torch.float16).float()
    )
    assert w[0].grad.dtype == torch.float16


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
    # latent MoE (Nemotron Super / Ultra) is built; shared-expert overlap is refused
    # with it (MCore asserts in training) and the norm before the up-projection is
    # not tested yet
    Cfg.moe_latent_size = 1024
    assert fm.validate_model(Cfg()) == []
    Cfg.moe_shared_expert_overlap = True
    assert len(fm.validate_model(Cfg())) == 1
    Cfg.moe_shared_expert_overlap = False
    Cfg.moe_use_norm_before_up_proj = True
    assert len(fm.validate_model(Cfg())) == 1


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
            f"weight{i}": (torch.randn(rows, cols) * 0.1).to(torch.float16)
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
@pytest.mark.parametrize("device_counts", [False, True])
def test_training_forward_bitwise_equal_engine_kernel(device_counts):
    """Training (te_gt on fp16 expert params) and engine (Triton) expert GEMMs give identical bits."""
    if not fm.te_gt_supported():
        pytest.skip("needs SM100 + cuBLASLt 13.5.1 + TE")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ws = [(torch.randn(96, 256, device=dev) * 0.02).to(torch.float16) for _ in range(4)]
    splits = [3, 0, 70, 19]
    x = torch.randn(sum(splits), 256, device=dev)
    m = torch.tensor(splits, device=dev) if device_counts else splits
    y = fm.grouped_linear(x, m, ws)
    for cfg in fm._CONFIGS:
        bm = cfg[1][0]
        dest, offs, m_pad = fm._aligned_rows(splits, bm, dev)
        xp = torch.zeros(m_pad, 256, device=dev, dtype=torch.float16)
        xp.index_copy_(0, dest, x.half())
        ye = fm.grouped_gemm(xp, ws, offs, cfg[1]).index_select(0, dest)
        assert torch.equal(y, ye), cfg


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





@needs_cuda
@pytest.mark.parametrize("tma", [True, False])
@pytest.mark.parametrize("splits", [[3, 0, 70, 19, 1, 0], [0, 0, 5, 0, 0, 0], [40] * 6])
def test_training_dgrad_wgrad_kernels_equal_per_expert_cublas(monkeypatch, splits, tma):
    """Segment dgrad on fp16 W == per-expert cuBLAS on bf16(W); segment wgrad == per-expert cuBLAS (bits)."""
    if not tma:
        monkeypatch.setattr(fm, "_tma_ok", lambda *a: False)
    torch.manual_seed(0)
    dev = torch.device("cuda")
    ws = [(torch.randn(96, 200, device=dev) * 0.02).to(torch.float16) for _ in splits]
    m = sum(splits)
    g = (torch.randn(m, 96, device=dev) * 1e-3).to(torch.bfloat16)
    g[:, ::3] = 0.0
    x = torch.randn(m, 200, device=dev).to(torch.bfloat16)
    counts = torch.tensor(splits, dtype=torch.int32, device=dev)
    dx = fm._segment_dgrad(g, tuple(ws), counts)
    wg = fm._segment_wgrad(g, x, counts, len(ws))
    s0 = 0
    for e, (w, n) in enumerate(zip(ws, splits)):
        if n:
            ref = torch.mm(g[s0 : s0 + n], w.to(torch.bfloat16), out_dtype=torch.float32)
            assert torch.equal(dx[s0 : s0 + n].view(torch.int32), ref.view(torch.int32)), e
        refw = torch.mm(g[s0 : s0 + n].t(), x[s0 : s0 + n], out_dtype=torch.float32)
        assert torch.equal(wg[e].view(torch.int32), refw.view(torch.int32)), e
        s0 += n


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton kernel)")
def test_relu2_probs_bwd_live_block_reduced_dprobs():
    """relu2_probs_bwd_live: dh bitwise autograd of _relu2_eager on rows < live; dprobs (block-reduced in the kernel,
    no rows x F product buffer) within fp32 accuracy of autograd's row sum (colo lead 11-04: the 12.3 GiB buffer)."""
    from nemo_rl.models.megatron.fp16_moe import _relu2_eager, relu2_probs_bwd_live

    torch.manual_seed(0)
    rows, f, live = 300, 2688, 257
    h = torch.randn(rows, f, device="cuda", dtype=torch.float32)
    p = torch.rand(rows, 1, device="cuda", dtype=torch.float32)
    g = torch.randn(rows, f, device="cuda", dtype=torch.float32)
    h_ref = h[:live].clone()
    dh, dp = relu2_probs_bwd_live(h, p, g, torch.tensor([live], device="cuda", dtype=torch.int64))
    assert dh.data_ptr() == h.data_ptr()  # dh is written over the dead activation (no rows x F buffer)
    h_ = h_ref.requires_grad_()
    p_ = p[:live].reshape(-1).clone().requires_grad_()
    y = _relu2_eager(h_, p_)
    rdh, rdp = torch.autograd.grad(y, (h_, p_), g[:live])
    assert torch.equal(dh[:live], rdh)
    torch.testing.assert_close(dp.reshape(-1)[:live], rdp, rtol=1e-5, atol=1e-4)


@needs_cuda
def test_nograd_relu2_written_over_fc1_output_bitwise():
    """No-grad expert forward (verification / log-probs): the squared-ReLU activation is written over fc1's output
    (no second rows x F fp32 buffer; 21.09 GiB per MoE layer at Ultra-L27 TP8 EP32 before the T_rank fix) and the
    expert output is bitwise the grad-enabled path's (_ActProbsLive), on the live rows of a capacity-sized buffer."""
    from types import SimpleNamespace

    if not fm.te_gt_supported():
        pytest.skip("needs SM100 + cuBLASLt 13.5.1 + TE")
    torch.manual_seed(0)
    dev = torch.device("cuda")
    # relu2_probs_fwd(out=h): same bits as a fresh buffer, written over h.
    for rows, f, live in ((300, 2688, 257), (64, 5120, 1), (1024, 1024, 1024)):
        h = torch.randn(rows, f, device=dev)
        p = torch.rand(rows, 1, device=dev)
        n = torch.tensor(live, device=dev)
        ref = fm.relu2_probs_fwd(h, p, torch.float32, valid_rows=n)
        out = fm.relu2_probs_fwd(h, p, torch.float32, valid_rows=n, out=h)
        assert out.data_ptr() == h.data_ptr()
        assert torch.equal(out[:live], ref[:live])

    def squared_relu(x):  # noqa: D401 (expert_activation reads the name)
        return torch.relu(x) ** 2

    k, f, experts = 256, 384, 4
    cfg = SimpleNamespace(activation_func=squared_relu, gated_linear_unit=False)

    def linear(n_out, n_in):
        ws = {f"weight{i}": (torch.randn(n_out, n_in, device=dev) * 0.05).to(torch.float16) for i in range(experts)}
        return SimpleNamespace(num_gemms=experts, **ws)

    module = SimpleNamespace(config=cfg, linear_fc1=linear(f, k), linear_fc2=linear(k, f))
    counts = torch.tensor([37, 0, 211, 5], device=dev, dtype=torch.int32)
    live, cap = int(counts.sum()), 600  # rows past live: capacity padding (never read)
    x = torch.randn(cap, k, device=dev)
    probs = torch.rand(cap, device=dev)
    with torch.enable_grad():
        y_old = fm.training_experts_forward(module, x, counts, probs)[0].detach()
    with torch.no_grad():
        y_new = fm.training_experts_forward(module, x, counts, probs)[0]
    assert torch.equal(y_new[:live], y_old[:live])
