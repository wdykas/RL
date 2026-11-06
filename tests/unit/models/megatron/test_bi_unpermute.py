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
"""Fused batch-invariant unpermute == Megatron's moe/batch_invariant.unpermute, bit for bit.

EP is emulated: Megatron's function only reads the EP world size to split the expert
range into ranks, which the test sets; the inverse map comes from Megatron's own
``permute`` on a skewed (imbalanced) top-k routing map.
"""

import pytest
import torch

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int32)


def _case(tokens, experts, topk, hidden, seed):
    from megatron.core.transformer.moe.moe_utils import permute

    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = torch.device("cuda")
    # Skewed scores: low expert ids (the first EP ranks) get most tokens.
    bias = torch.linspace(3.0, 0.0, experts, device=dev)
    scores = torch.randn(tokens, experts, generator=g, device=dev) + bias
    idx = scores.topk(topk, dim=1).indices
    routing_map = torch.zeros(tokens, experts, dtype=torch.bool, device=dev)
    routing_map.scatter_(1, idx, True)
    hidden_states = torch.randn(tokens, hidden, generator=g, device=dev)
    _, _, _, inverse_map, _ = permute(
        hidden_states,
        routing_map,
        num_out_tokens=tokens * topk,
        return_batch_invariant_inverse_map=True,
    )
    rows = tokens * topk
    # Expert outputs with a wide exponent range (rounding matters), signed zeros and
    # exact cancellations.
    scale = torch.exp2(torch.randint(-20, 20, (rows, 1), generator=g, device=dev).float())
    x = torch.randn(rows, hidden, generator=g, device=dev) * scale
    x[:: max(1, rows // 7), : hidden // 3] = -0.0
    x[1 :: max(2, rows // 5)] = -x[0]
    grad = torch.randn(tokens, hidden, generator=g, device=dev)
    grad[:: max(1, tokens // 4)] = -0.0
    return x, inverse_map, grad


def _run(fn, x, grad, **kw):
    x = x.clone().requires_grad_()
    out = fn(x, torch.Size((grad.shape[0], x.shape[1])), probs=None, **kw)
    (dx,) = torch.autograd.grad(out, x, grad)
    return out.detach(), dx


@needs_cuda
@pytest.mark.parametrize("collective", ["ordered", "multimem"])
@pytest.mark.parametrize(
    "ep,experts,topk,hidden",
    [
        (1, 128, 6, 2688),
        (2, 128, 1, 2048),
        (2, 128, 6, 2688),
        (2, 128, 8, 2048),
        (4, 128, 2, 4096),
        (4, 128, 8, 2048),
        (8, 128, 6, 2688),
        (8, 128, 8, 2048),
        (8, 512, 22, 4096),
        (64, 512, 22, 8192),
        (64, 512, 1, 2048),
    ],
)
def test_fused_unpermute_is_bitwise_megatron(monkeypatch, collective, ep, experts, topk, hidden):
    from megatron.core.transformer.moe import batch_invariant as mbi

    from nemo_rl.models.megatron import bi_unpermute

    monkeypatch.setattr(
        mbi.parallel_state, "get_expert_model_parallel_world_size", lambda: ep
    )
    monkeypatch.setattr(mbi, "get_batch_invariant_collective", lambda: collective)
    for tokens, seed in ((37, 0), (256, 1)):
        x, inverse_map, grad = _case(tokens, experts, topk, hidden, seed)
        ref_out, ref_dx = _run(
            mbi.unpermute, x, grad, num_experts=experts, inverse_map=inverse_map
        )
        out, dx = _run(
            bi_unpermute.fused_unpermute,
            x,
            grad,
            num_experts=experts,
            inverse_map=inverse_map,
            ep_size=ep,
            cross_rank_fp64=collective == "multimem",
        )
        assert out.dtype == ref_out.dtype and out.shape == ref_out.shape
        assert torch.equal(_bits(out), _bits(ref_out)), (tokens, seed)
        assert torch.equal(_bits(dx), _bits(ref_dx)), (tokens, seed)


@needs_cuda
def test_install_routes_moe_utils_and_falls_back_with_probs(monkeypatch):
    from megatron.core.transformer.moe import batch_invariant as mbi
    from megatron.core.transformer.moe import moe_utils

    from nemo_rl.models.megatron import bi_unpermute

    monkeypatch.setattr(moe_utils, "batch_invariant_unpermute", mbi.unpermute)
    monkeypatch.setitem(bi_unpermute._STATE, "orig", None)
    bi_unpermute.install()
    assert moe_utils.batch_invariant_unpermute is bi_unpermute.fused_unpermute
    bi_unpermute.install()  # idempotent
    assert bi_unpermute._STATE["orig"] is mbi.unpermute
    # probs given: Megatron's function.
    monkeypatch.setattr(mbi.parallel_state, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mbi, "get_batch_invariant_collective", lambda: "ordered")
    x, inverse_map, _ = _case(16, 8, 2, 64, 3)
    probs = torch.rand(16, 8, device="cuda")
    ref = mbi.unpermute(x, torch.Size((16, 64)), probs=probs, num_experts=8, inverse_map=inverse_map)
    out = bi_unpermute.fused_unpermute(
        x, torch.Size((16, 64)), probs=probs, num_experts=8, inverse_map=inverse_map
    )
    assert torch.equal(_bits(out), _bits(ref))
