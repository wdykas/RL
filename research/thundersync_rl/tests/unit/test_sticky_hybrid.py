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
"""Sticky routing on Nemotron-H style routers and hybrid layer patterns.

- sigmoid scores + aux-free expert bias (selection on sigmoid + bias, probabilities
  from the unbiased sigmoid, normalized over top-k, routed scaling factor), with and
  without group-limited top-k: sticky with anchors == the plain top-k is bitwise the
  plain routing (sparse training path and dense engine path, sorted and unsorted
  top-k, exact score ties);
- the rule acts on the biased selection scores;
- MoE layer numbering of hybrid patterns ('E' positions; '|' and '/' MTP parts);
- squared-ReLU experts on the fp16 expert path (Nemotron-H).
"""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.models.megatron import sticky_routing as sr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

E, K, SCALE = 128, 6, 2.5  # Nemotron-3-Nano router


@pytest.fixture
def sticky_on():
    sr.install()
    sr.set_delta(0.03)
    yield 0.03
    sr.set_delta(None)


def _replay():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    r = RouterReplay()
    RouterReplay.global_router_replay_instances.remove(r)
    return r


def _route(logits, bias, replay, dense, groups):
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    ng, gk = groups
    return topk_routing_with_score_function(
        logits,
        K,
        num_groups=ng,
        group_topk=gk,
        scaling_factor=SCALE,
        score_function="sigmoid",
        expert_bias=bias,
        router_replay=replay,
        dense_output=dense,
    )


def _logits(seed, T=512, ties=False, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    logits = 2.0 * torch.randn(T, E, generator=g)
    if ties:  # coarse grid: many exactly equal scores, inside and at the top-k boundary
        logits = (logits * 4).round() / 4
    bias = 0.02 * torch.randn(E, generator=g)
    if ties:
        bias = torch.zeros(E)
    return logits.to(device), bias.to(device)


GROUPS = [(None, None), (1, 1), (8, 4)]  # none / Nemotron-H (n_group 1) / DeepSeek-style


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
@pytest.mark.parametrize("groups", GROUPS)
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("grad", [False, True])  # no-grad MCore top-k is unsorted (without BI)
@pytest.mark.parametrize("ties", [False, True])
def test_anchors_equal_plain_topk_is_bitwise_plain(sticky_on, device, groups, dense, grad, ties):
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    logits, bias = _logits(7 + int(ties), ties=ties, device=device)
    with torch.set_grad_enabled(grad):
        rec = _replay()
        rec.set_router_replay_action(RouterReplayAction.RECORD)
        plain = _route(logits, bias, rec, dense, groups)
        anchors = rec.recorded_topk_idx.clone()
        r = _replay()
        r.set_target_indices(anchors)
        r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        sticky = _route(logits, bias, r, dense, groups)
        unrouted = _route(logits, bias, None, dense, groups)
    for a, b, c in zip(plain, sticky, unrouted):
        assert torch.equal(a, b)
        assert torch.equal(a, c)
    # anchors given in another order (e.g. canonicalized by expert id) as well
    with torch.set_grad_enabled(grad):
        r = _replay()
        r.set_target_indices(anchors.sort(1).values)
        r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        sticky2 = _route(logits, bias, r, dense, groups)
    for a, b in zip(plain, sticky2):
        assert torch.equal(a, b)


@pytest.mark.parametrize("groups", [(None, None), (1, 1)])
def test_rule_acts_on_biased_selection_scores(sticky_on, groups):
    """Keep test on sigmoid(logits) + bias; probabilities from the unbiased sigmoid."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    logits, bias = _logits(11)
    g = torch.Generator().manual_seed(12)
    drafter = logits + 0.05 * torch.randn(logits.shape, generator=g)  # theta_n
    sel_n = torch.sigmoid(drafter) + bias
    anchors = torch.topk(sel_n, K, dim=1).indices
    r = _replay()
    r.set_target_indices(anchors)
    r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    probs, idx = _route(logits, bias, r, True, groups)
    sel = torch.sigmoid(logits) + bias
    pv, pi = torch.topk(sel, K, dim=1)
    _, want, keep = sr.sticky_select(sel, anchors, pv, pi, sticky_on)
    assert torch.equal(idx, want)
    assert 0 < int(keep.sum()) and bool((want.sort(1).values != pi.sort(1).values).any())
    s = torch.sigmoid(logits).gather(1, want)
    assert torch.equal(probs, (s / (s.sum(-1, keepdim=True) + 1e-20)) * SCALE)


def test_hybrid_moe_layer_numbers():
    from nemo_rl.models.megatron.router_replay import (
        _global_moe_layer_numbers,
        router_replay_dimensions,
    )

    nano = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
    cfg = SimpleNamespace(num_layers=52, hybrid_layer_pattern=nano, moe_router_topk=6)
    layers = _global_moe_layer_numbers(cfg)
    assert len(layers) == nano.count("E") == 23
    assert layers[:3] == [2, 4, 7] and all(nano[i - 1] == "E" for i in layers)
    assert router_replay_dimensions(cfg) == (23, 6)
    # PP separators and MTP depths do not shift main-decoder layer numbers
    piped = SimpleNamespace(num_layers=6, hybrid_layer_pattern="ME*|EM-/*E/*E")
    assert _global_moe_layer_numbers(piped) == [2, 4]
    legacy = SimpleNamespace(num_layers=4, hybrid_override_pattern="MEME")
    assert _global_moe_layer_numbers(legacy) == [2, 4]
    # non-hybrid models keep moe_layer_freq semantics
    assert _global_moe_layer_numbers(SimpleNamespace(num_layers=4, moe_layer_freq=2)) == [1, 3]
    assert _global_moe_layer_numbers(
        SimpleNamespace(num_layers=4, moe_layer_freq=1, hybrid_layer_pattern=None)
    ) == [1, 2, 3, 4]


def test_expert_activation_selection():
    from megatron.core.activations import squared_relu

    from nemo_rl.models.megatron import fp16_moe

    silu = torch.nn.functional.silu
    assert fp16_moe.expert_activation(
        SimpleNamespace(activation_func=silu, gated_linear_unit=True)
    ) == "swiglu"
    assert fp16_moe.expert_activation(
        SimpleNamespace(activation_func=squared_relu, gated_linear_unit=False)
    ) == "relu2"
    with pytest.raises(ValueError):
        fp16_moe.expert_activation(SimpleNamespace(activation_func=squared_relu, gated_linear_unit=True))
    nemotron = SimpleNamespace(
        moe_grouped_gemm=True,
        activation_func=squared_relu,
        gated_linear_unit=False,
        moe_shared_expert_intermediate_size=3712,
        moe_token_dispatcher_type="alltoall",
    )
    assert fp16_moe.validate_model(nemotron) == []
    assert fp16_moe.validate_model(SimpleNamespace(**{**vars(nemotron), "moe_latent_size": 1024})) == [
        "moe_latent_size"
    ]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_relu2_probs_train_equals_engine_and_grad(device):
    from nemo_rl.models.megatron import fp16_moe

    g = torch.Generator().manual_seed(3)
    h = (4 * torch.randn(97, 1856, generator=g)).to(device)
    p = torch.rand(97, generator=g).to(device)
    train = fp16_moe.relu2_probs(h, p)
    engine = fp16_moe.relu2_probs_fwd(h, p, torch.float16)
    assert train.dtype == torch.float32
    # the training value rounds to fp16 at the next GEMM exactly as the engine's cast
    assert torch.equal(train.to(torch.float16), engine)
    # batch invariance: a row's value does not depend on the other rows
    assert torch.equal(fp16_moe.relu2_probs_fwd(h[5:9], p[5:9]), train[5:9])
    hh = h.clone().requires_grad_()
    pp = p.clone().requires_grad_()
    fp16_moe.relu2_probs(hh, pp).sum().backward()
    want_h = 2 * torch.relu(h) * p[:, None]
    want_p = (torch.relu(h) ** 2).sum(-1)
    torch.testing.assert_close(hh.grad, want_h)
    torch.testing.assert_close(pp.grad, want_p, rtol=1e-5, atol=1e-3)
