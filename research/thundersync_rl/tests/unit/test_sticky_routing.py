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
"""Sticky MoE routing: the selection rule, Megatron RouterReplay integration,
batch invariance of the rule, and backward replay (nemo_rl sticky_routing.py)."""

import pytest
import torch

from nemo_rl.models.megatron import sticky_routing as sr

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _topk(scores, topk, num_groups=None, group_topk=None):
    return torch.topk(scores, k=topk, dim=1, sorted=True)


def _reference(scores, anchors, k, delta):
    """Row-by-row statement of the rule (Python loops)."""
    out = []
    for s, a in zip(scores.tolist(), anchors.tolist()):
        plain = sorted(range(len(s)), key=lambda e: (-s[e], e))[:k]
        valid = all(x >= 0 for x in a) and len(set(a)) == len(a)
        if valid:
            min_in = min(s[e] for e in a)
            max_out = max((s[e] for e in range(len(s)) if e not in a), default=float("-inf"))
            if min_in >= max_out - delta:
                out.append(sorted(a, key=lambda e: -s[e]))
                continue
        out.append(None)  # plain
    return out


@pytest.mark.parametrize("k,E,delta", [(1, 4, 0.1), (2, 8, 0.05), (8, 128, 0.03)])
def test_rule_matches_reference(k, E, delta):
    g = torch.Generator().manual_seed(k * 1000 + E)
    T = 400
    scores = torch.randn(T, E, generator=g)
    plain_vals, plain_idx = _topk(scores, k)
    # Anchors: a perturbed plain top-k (near ties), random sets, invalid rows.
    pert = scores + 0.05 * torch.randn(T, E, generator=g)
    anchors = torch.topk(pert, k, dim=1).indices
    anchors[::7] = torch.randperm(E, generator=g)[:k]
    anchors[3::11] = -1
    if k > 1:
        anchors[5::13, 1] = anchors[5::13, 0]  # repeated id: invalid -> self
    vals, idx, keep = sr.sticky_select(scores, anchors, plain_vals, plain_idx, delta)
    assert torch.equal(vals, scores.gather(1, idx))  # values are the scores' bits
    ref = _reference(scores, anchors, k, delta)
    for t, r in enumerate(ref):
        if r is None:
            assert not keep[t]
            assert idx[t].tolist() == plain_idx[t].tolist()
        else:
            assert keep[t]
            assert set(idx[t].tolist()) == set(r)
            # ordered by the current scores, descending
            assert scores[t, idx[t]].tolist() == sorted(scores[t, idx[t]].tolist(), reverse=True)
    assert 0 < int(keep.sum()) < T


def test_diagonal_identity_is_bitwise_plain():
    """F(theta; plain_topk(theta)) is the plain forward, bit for bit."""
    g = torch.Generator().manual_seed(0)
    scores = torch.randn(1000, 64, generator=g)
    pv, pi = _topk(scores, 8)
    vals, idx, keep = sr.sticky_select(scores, pi, pv, pi, 0.03)
    assert bool(keep.all()) and torch.equal(idx, pi) and torch.equal(vals, pv)
    # and with the anchors in a different order
    perm = pi[:, torch.randperm(8, generator=g)]
    vals, idx, _ = sr.sticky_select(scores, perm, pv, pi, 0.03)
    assert torch.equal(idx, pi) and torch.equal(vals, pv)


def test_delta_zero_is_plain_unless_anchor_is_top():
    g = torch.Generator().manual_seed(1)
    scores = torch.randn(500, 16, generator=g)
    pv, pi = _topk(scores, 2)
    anchors = torch.topk(scores + 0.3 * torch.randn(500, 16, generator=g), 2).indices
    _, idx, _ = sr.sticky_select(scores, anchors, pv, pi, 0.0)
    assert torch.equal(idx.sort(1).values, pi.sort(1).values)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
def test_rule_is_batch_invariant(device):
    """A token's selection does not depend on the other rows (train == engine)."""
    g = torch.Generator().manual_seed(2)
    scores = torch.randn(257, 128, generator=g).to(device)
    anchors = torch.topk(scores + 0.02 * torch.randn(257, 128, generator=g).to(device), 8).indices
    full = sr.sticky_topk(scores, anchors, 8, 0.03, _topk)
    for rows in (slice(0, 1), slice(5, 6), slice(3, 70), slice(100, 257)):
        part = sr.sticky_topk(scores[rows], anchors[rows], 8, 0.03, _topk)
        for a, b in zip(full, part):
            assert torch.equal(a[rows], b)


# --------------------------------------------------------------------------
# Megatron RouterReplay integration (MCore topk_routing_with_score_function)
# --------------------------------------------------------------------------


@pytest.fixture
def sticky_on():
    sr.install()
    sr.set_delta(0.05)
    yield 0.05
    sr.set_delta(None)


def _replay():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    r = RouterReplay()
    RouterReplay.global_router_replay_instances.remove(r)
    return r


def _route(logits, topk, replay, dense=False):
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    return topk_routing_with_score_function(
        logits, topk, router_replay=replay, dense_output=dense
    )


def test_replay_forward_applies_the_rule_and_backward_replays_it(sticky_on):
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    g = torch.Generator().manual_seed(3)
    logits = torch.randn(300, 16, generator=g)
    anchors = torch.topk(logits + 0.04 * torch.randn(300, 16, generator=g), 2).indices
    anchors[:10] = -1
    r = _replay()
    r.set_target_indices(anchors)
    r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    probs, rmap = _route(logits, 2, r)
    pv, pi = _topk(logits, 2)
    _, want, keep = sr.sticky_select(logits, anchors, pv, pi, sticky_on)
    assert torch.equal(rmap, torch.zeros_like(rmap).scatter(1, want, True))
    assert not bool(keep[:10].any()) and bool((want.sort(1).values != pi.sort(1).values).any())
    # softmax over the selected logits, like plain routing
    sel = logits.gather(1, want).softmax(-1)
    assert torch.equal(probs.gather(1, want), sel.to(probs.dtype))
    # REPLAY_BACKWARD (activation recompute) replays the effective routes.
    assert torch.equal(r.replay_backward_list[-1], want)
    r.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    probs2, rmap2 = _route(logits, 2, r)
    assert torch.equal(rmap2, rmap) and torch.equal(probs2, probs)


def test_sparse_training_path_equals_dense_engine_path(sticky_on):
    """Training (routing map) and engine (dense_output) select the same experts
    with the same probabilities under the rule."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    g = torch.Generator().manual_seed(4)
    logits = torch.randn(64, 32, generator=g)
    anchors = torch.topk(logits + 0.05 * torch.randn(64, 32, generator=g), 4).indices
    outs = []
    for dense in (False, True):
        r = _replay()
        r.set_target_indices(anchors)
        r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
        outs.append(_route(logits, 4, r, dense=dense))
    (sp, smap), (dp, didx) = outs
    assert torch.equal(smap, torch.zeros_like(smap).scatter(1, didx, True))
    assert torch.equal(sp.gather(1, didx), dp)


def test_all_self_anchors_equal_plain_routing(sticky_on):
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    g = torch.Generator().manual_seed(5)
    logits = torch.randn(50, 8, generator=g)
    plain = _route(logits, 2, None)
    r = _replay()
    r.set_target_indices(torch.full((50, 2), -1, dtype=torch.long))
    r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    out = _route(logits, 2, r)
    assert all(torch.equal(a, b) for a, b in zip(plain, out))


def test_padding_rows_of_zeros_are_self(sticky_on):
    """Training batches pad routed_experts with zeros: repeated ids = no anchor."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    logits = torch.randn(4, 8)
    r = _replay()
    r.set_target_indices(torch.zeros(4, 2, dtype=torch.long))
    r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    assert torch.equal(_route(logits, 2, r)[1], _route(logits, 2, None)[1])


def test_delta_none_keeps_megatron_replay_semantics():
    """Sticky off: REPLAY_FORWARD replays the target exactly (R3 behaviour)."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    sr.install()
    sr.set_delta(None)
    logits = torch.randn(20, 8)
    target = torch.stack([torch.randperm(8)[:2] for _ in range(20)])
    r = _replay()
    r.set_target_indices(target)
    r.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _, rmap = _route(logits, 2, r)
    assert torch.equal(rmap, torch.zeros_like(rmap).scatter(1, target, True))


def test_record_into_static_buffer_inference_only(monkeypatch):
    """Engine RECORD: plain top-k copied into the static buffer, inference only."""
    from megatron.core.inference import utils as inf_utils
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    sr.install()
    buf = torch.full((16, 3, 2), -1, dtype=torch.int16)
    r = _replay()
    r.set_static_buffer(buf[:, 1, :])
    r._nrl_record_inference_only = True
    r.set_router_replay_action(RouterReplayAction.RECORD)
    logits = torch.randn(5, 8)
    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: False)
    _route(logits, 2, r)
    assert bool((buf == -1).all())  # a training-shaped forward does not record
    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)
    _, rmap = _route(logits, 2, r)
    rec = buf[:5, 1].long()
    assert torch.equal(rec, _topk(logits, 2)[1])
    assert bool((buf[:, 0] == -1).all()) and bool((buf[5:] == -1).all())


def test_router_replay_flag_and_model_flag_follow_config():
    cfg = {"megatron_cfg": {"sticky_routing_delta": 0.03}}
    assert sr.sticky_routing_enabled(cfg) and sr.sticky_routing_delta(cfg) == 0.03
    assert not sr.sticky_routing_enabled({"megatron_cfg": {}})
    assert not sr.sticky_routing_enabled({"megatron_cfg": {"sticky_routing_delta": None}})
    assert sr.configure(cfg) == 0.03 and sr.get_delta() == 0.03
    sr.configure({"megatron_cfg": {}})
    assert sr.get_delta() is None


def test_stats_count_kept_and_changed_routes(sticky_on):
    sr.start_stats()
    g = torch.Generator().manual_seed(6)
    scores = torch.randn(100, 16, generator=g)
    anchors = torch.topk(scores + 0.2 * torch.randn(100, 16, generator=g), 2).indices
    sr.sticky_topk(scores, anchors, 2, sticky_on, _topk)
    st = sr.pop_stats()
    assert st["token_layers"] == 100 and 0 < st["anchor_kept"] <= 1
    assert 0 < st["differs_from_plain"] <= st["anchor_kept"]
