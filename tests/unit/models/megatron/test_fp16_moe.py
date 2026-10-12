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

import torch

from nemo_rl.models.megatron import fp16_moe as fm


def _experts(e, n, k, seed=0):
    torch.manual_seed(seed)
    return [(torch.randn(n, k) * 0.05).to(torch.bfloat16) for _ in range(e)]


def test_tile_experts_from_offsets():
    b = fm.BLOCK_M
    offs = torch.tensor([b, b, 3 * b, 4 * b], dtype=torch.int32)  # expert 1 empty
    tiles = fm.tile_experts_from_offsets(offs, 6)
    assert tiles.tolist() == [0, 2, 2, 3, -1, -1]


def test_aligned_rows():
    dest, tiles, m_pad = fm._aligned_rows([3, 0, fm.BLOCK_M + 1], "cpu")
    b = fm.BLOCK_M
    assert m_pad == 3 * b
    assert tiles.tolist() == [0, 2, 2]
    assert dest.tolist() == [0, 1, 2] + list(range(b, 2 * b + 1))


def test_grouped_linear_matches_per_expert_fp16_reference():
    w = _experts(3, 16, 32)
    splits = [5, 0, 7]
    x = torch.randn(12, 32) * 2
    y = fm.grouped_linear(x, splits, w)
    assert y.dtype == torch.float32
    ref, s = [], 0
    for we, m in zip(w, splits):
        ref.append(x[s : s + m].to(torch.float16).double() @ we.to(torch.float16).double().t())
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
    torch.testing.assert_close(w[1].grad.float(), (gb[3:].t() @ xb[3:]).to(torch.bfloat16).float())
    assert w[0].grad.dtype == torch.bfloat16


def test_swiglu_probs():
    h = torch.randn(4, 6)
    p = torch.rand(4)
    out = fm.swiglu_probs(h, p)
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
    Cfg.moe_shared_expert_intermediate_size = 64
    assert fm.validate_model(Cfg()) == ["shared experts"]
