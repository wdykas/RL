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
"""Sticky routing anchor horizon (DESIGN.md section 14i): the engine's anchor
apply (per-token buffer from per-request pages, the router rule against it,
the forward wrapper on a fake engine incl. expert-parallel dummy steps) and the
driver's horizon (theta_{k-1} work ends at the horizon, the refit engine
continues with the anchors). No Ray, no GPUs needed."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import EOD, _spec as make_spec
from test_sticky_driver import _Engine
from thundersync_rl.routes import (
    NO_ANCHOR,
    EngineAnchors,
    anchored_draft_length,
    fetch_anchors,
    merge_route_refs,
    put_routes,
)

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron import sticky_routing as sr

devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _rows(n, L, K, base):
    return (base + torch.arange(n * L * K)).view(n, L, K).to(torch.int16)


# --------------------------------------------------------------------------
# EngineAnchors: per-token buffer from per-request pages
# --------------------------------------------------------------------------


@pytest.mark.parametrize("device", devices)
def test_engine_anchors_fill_by_request_and_position(device):
    L, K = 2, 3
    rows = {"a": _rows(5, L, K, 0), "b": _rows(3, L, K, 1000)}  # c = 5 and 3
    ea = EngineAnchors(16, L, K, device, rows.get, page=2, pages=2)
    ia, ib = ea.anchor_id("a"), ea.anchor_id("b")  # 3 + 2 pages: pool grows
    assert ea.anchor_id("x") == -1 and ia != ib
    assert ea.pool_bytes == 8 * 2 * L * K * 2  # 2 -> 4 -> 8 pages of 2 positions
    # prefill: a at positions 0..6 (5, 6 beyond its horizon), b at 1..3, an
    # untagged request, b again at 0
    aid = np.array([ia] * 7 + [ib] * 3 + [-1, ib])
    pos = np.array([0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 0, 0])
    assert ea.fill(aid, pos) == 5 + 2 + 1
    buf = ea.buffer.cpu()
    assert torch.equal(buf[:5], rows["a"]) and bool((buf[5:7] == NO_ANCHOR).all())
    assert torch.equal(buf[7:9], rows["b"][1:3]) and bool((buf[9:11] == NO_ANCHOR).all())
    assert torch.equal(buf[11], rows["b"][0]) and bool((buf[12:] == NO_ANCHOR).all())
    # decode beyond the horizon: every row self again
    assert ea.fill(np.array([ia, ib]), np.array([7, 4])) == 0
    assert bool((ea.buffer == NO_ANCHOR).all())
    # release and reuse: pages and id come back, other requests keep theirs
    ea.release("a")
    rows["c"] = _rows(4, L, K, 2000)
    ic = ea.anchor_id("c")
    assert ic == ia
    ea.fill(np.array([ic, ib]), np.array([3, 2]))
    assert torch.equal(ea.buffer[0].cpu(), rows["c"][3])
    assert torch.equal(ea.buffer[1].cpu(), rows["b"][2])
    ea.clear()
    assert bool((ea.buffer == NO_ANCHOR).all())


def test_horizon_draft_length():
    # P = 2, c = 5: tokens need positions < min(P + m - 1, c) logged
    assert anchored_draft_length(3, 2, 10) == 2
    assert anchored_draft_length(3, 2, 10, horizon=5) == 2
    assert anchored_draft_length(5, 2, 10, horizon=5) == 10  # all of c logged
    assert anchored_draft_length(4, 2, 10, horizon=5) == 3


# --------------------------------------------------------------------------
# Router: engine anchor apply == learner sticky replay, bitwise
# --------------------------------------------------------------------------


def _replay():
    from megatron.core.transformer.moe.router_replay import RouterReplay

    return RouterReplay()


def _route(logits, topk, replay):
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function

    return topk_routing_with_score_function(logits, topk, router_replay=replay)


@pytest.fixture
def sticky_on():
    sr.install()
    sr.set_delta(0.05)
    yield 0.05
    sr.set_delta(None)


@pytest.mark.parametrize("device", devices)
@pytest.mark.parametrize("sp_rank", [0, 1])
def test_engine_apply_equals_learner_replay_bitwise(sticky_on, monkeypatch, device, sp_rank):
    """The engine router (RECORD + anchor buffer) selects bitwise what the
    learner (REPLAY_FORWARD with the same anchors) selects, with the same
    probabilities; rows without anchors are plain; the record is the plain
    top-k (the drafter's log)."""
    from megatron.core.inference import utils as inf_utils
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)
    g = torch.Generator().manual_seed(7)
    T, E, K = 96, 16, 2
    logits = torch.randn(T, E, generator=g).to(device)
    anchors = torch.topk(logits.cpu() + 0.06 * torch.randn(T, E, generator=g), K).indices
    anchors[::5] = NO_ANCHOR  # self rows (no anchor, or beyond the horizon)
    anchors = anchors.to(torch.int16).to(device)
    # the step's per-token buffer; with sequence parallelism rank r routes rows [rT, (r+1)T)
    buf = torch.full((2 * T, 3, K), NO_ANCHOR, dtype=torch.int16, device=device)
    buf[sp_rank * T : (sp_rank + 1) * T, 1] = anchors
    rec = torch.full((2 * T, K), -1, dtype=torch.int64, device=device)
    eng = _replay()
    eng._nrl_record_inference_only = True
    eng.set_static_buffer(rec)
    setattr(eng, sr.APPLY_ATTR, (buf[:, 1, :], sp_rank))
    eng.set_router_replay_action(RouterReplayAction.RECORD)
    probs_e, map_e = _route(logits, K, eng)

    lrn = _replay()
    lrn.set_target_indices(anchors.long())
    lrn.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    probs_l, map_l = _route(logits, K, lrn)
    assert torch.equal(map_e, map_l) and torch.equal(probs_e, probs_l)
    plain_p, plain_map = _route(logits, K, None)
    assert torch.equal(map_e[::5], plain_map[::5])
    assert not torch.equal(map_e, plain_map)  # the rule changed some route
    assert torch.equal(rec[:T].cpu(), torch.topk(logits.cpu(), K).indices)


def test_engine_apply_all_self_is_plain(sticky_on, monkeypatch):
    from megatron.core.inference import utils as inf_utils
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)
    logits = torch.randn(40, 8)
    eng = _replay()
    setattr(eng, sr.APPLY_ATTR, (torch.full((64, 2), NO_ANCHOR, dtype=torch.int16), 0))
    eng.set_router_replay_action(RouterReplayAction.RECORD)
    out, plain = _route(logits, 2, eng), _route(logits, 2, None)
    assert all(torch.equal(a, b) for a, b in zip(out, plain))


def test_horizon_config_reaches_workers():
    cfg = {"megatron_cfg": {"sticky_routing_delta": 0.03, "sticky_anchor_horizon": 64}}
    try:
        sr.configure(cfg)
        assert sr.get_anchor_horizon() == 64
        sr.configure({"megatron_cfg": {"sticky_anchor_horizon": 64}})  # sticky off
        assert sr.get_anchor_horizon() is None
    finally:
        sr.set_delta(None)


# --------------------------------------------------------------------------
# Worker: forward wrapper on a fake engine (async-safe, EP dummy steps)
# --------------------------------------------------------------------------


def _worker():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    return object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)


def test_engine_anchor_apply_on_a_fake_engine(monkeypatch):
    """Registered RouteRefs are fetched (this stage's layers) when the tagged
    request first appears, the buffer holds its anchors below c in token
    order before each forward launches, rows beyond c and of untagged
    requests are self, a dummy step leaves every row self, and pages are
    released once the request left the engine."""
    import thundersync_rl.worker as worker_mod
    from megatron.core import parallel_state as ps
    from megatron.core import utils as mcore_utils

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
    monkeypatch.setattr(worker_mod.ray, "get", lambda r: r)
    monkeypatch.setattr(mcore_utils, "get_model_config", lambda m: SimpleNamespace(moe_router_topk=2))
    monkeypatch.setattr(worker_mod.torch.cuda, "current_device", lambda: 0)
    L, K = 2, 2
    # Two pipeline stages logged the draft; this rank holds layers 3 and 5.
    other = torch.full((4, 1, K), 9, dtype=torch.int16)
    mine = _rows(4, L, K, 0)
    refs = merge_route_refs(
        [([1], 0, 4, put_routes(other, lambda t: t)), ([3, 5], 0, 4, put_routes(mine, lambda t: t))],
        4,
        [1, 3, 5],
    )
    w = _worker()
    w._gen_model = lambda: None
    w._route_record = {"replays": [SimpleNamespace() for _ in range(L)], "layers": [3, 5], "sequence_parallel": False}
    sr.set_delta(0.03)
    sr._STATE["horizon"] = 2
    seen = []

    def forward_logits(input_ids, position_ids):
        seen.append(w._anchor_apply["anchors"].buffer[: ctx.active_token_count].clone())

    controller = SimpleNamespace(
        _dynamic_step_forward_logits=forward_logits,
        _dynamic_step_context_init=lambda **kw: (None, None, None),
    )
    ctx = SimpleNamespace(
        max_tokens=8,
        active_token_count=0,
        token_to_request_idx=torch.zeros(8, dtype=torch.int32),
        token_to_pos_ids=torch.zeros(8, dtype=torch.long),
        request_ids=torch.tensor([21, 22, -1], dtype=torch.int32),
        request_in_prefill_status_tensor=torch.tensor([1, 1, 0], dtype=torch.int32),
        total_request_count=2,
    )

    def req(key):
        sp = SimpleNamespace() if key is None else SimpleNamespace(ts_anchor_key=key)
        return SimpleNamespace(record=[SimpleNamespace(sampling_params=sp)])

    engine = SimpleNamespace(requests={21: req("h0"), 22: req(None)})
    try:
        monkeypatch.setattr(worker_mod.torch, "device", lambda *a: "cpu")
        w._install_anchor_apply(ctx)
        assert all(sr.APPLY_ATTR in vars(r) for r in w._route_record["replays"])
        w._wrap_anchor_apply(engine, controller, ctx)
        w.register_engine_anchors({"h0": refs})

        def step(n, slots, pos, dummy=False):
            ctx.active_token_count = n
            ctx.token_to_request_idx[:n] = torch.tensor(slots)
            ctx.token_to_pos_ids[:n] = torch.tensor(pos)
            controller._dynamic_step_context_init(is_dummy_forward=dummy)
            controller._dynamic_step_forward_logits(None, None)

        # prefill: request 21 (slot 0) at positions 0..3 (c = 4), 22 at 0..1
        step(6, [0, 0, 0, 0, 1, 1], [0, 1, 2, 3, 0, 1])
        assert torch.equal(seen[-1][:4], mine) and bool((seen[-1][4:] == NO_ANCHOR).all())
        step(2, [0, 1], [4, 2])  # decode beyond the horizon
        assert bool((seen[-1] == NO_ANCHOR).all())
        # eviction re-prefill of 21 below c, then an EP dummy step: MCore
        # keeps the last step's ids/positions, but the rows must be self
        step(3, [0, 0, 0], [1, 2, 3])
        assert torch.equal(seen[-1], mine[1:4])
        step(3, [0, 0, 0], [1, 2, 3], dummy=True)
        assert bool((seen[-1] == NO_ANCHOR).all())
        st = w.engine_anchor_stats()
        assert st["loaded_requests"] == 1.0 and st["applied_tokens"] == 4 + 3
        # the request leaves the engine: its pages are released by the sweep
        del engine.requests[21]
        free0 = len(w._anchor_apply["anchors"]._free)
        for _ in range(64):
            step(1, [1], [3])
        assert len(w._anchor_apply["anchors"]._free) == free0 + 1
        assert w._anchor_apply["anchors"].keys() == []
    finally:
        sr.set_delta(None)
        sr._STATE["horizon"] = None


def test_requests_are_tagged_with_their_anchor_key(monkeypatch):
    sps = [SimpleNamespace(attrs={}, add_attributes=None) for _ in range(2)]
    for sp in sps:
        sp.add_attributes = sp.attrs.update
    w = _worker()
    cls = type(w)
    base = next(c for c in cls.__mro__ if c.__name__ == "MegatronGenerationMixin")
    monkeypatch.setattr(base, "_prepare_data_for_generation", lambda self, data, greedy=False: (["p", "q"], [None, None], sps))
    data = BatchedDataDict({"input_ids": torch.zeros(2, 3), "ts_anchor_key": ["h3", None]})
    w._prepare_data_for_generation(data)
    assert sps[0].attrs == {"ts_anchor_key": "h3"} and sps[1].attrs == {}


# --------------------------------------------------------------------------
# Driver: theta_{k-1} work ends at the horizon, the refit engine continues
# --------------------------------------------------------------------------


def test_horizon_bounds_theta_old_work_and_hands_plans_to_the_engine():
    """H = 2. Prompt [1, 2] (c = 4): the draft [4, 5, 6] has positions 0..3
    logged, so it is verified against anchors cut to c and, whole-accepted,
    needs no theta_{k-1} continuation. Prompt [3] (c = 3): rejected at 0 -> 9;
    a continuation of at most H + 1 tokens logs positions 0..2 and stops.
    Both plans go to the refit engine with their anchors registered."""
    engine = _Engine(
        [
            [
                ([4, 5, 6], False, [-0.1] * 3, 4),
                ([7, 8], False, [-0.2] * 2, 2),
            ],
            [([11, 12], True, [-0.5] * 2, 3)],  # from [3, 9]: positions 0..2
        ]
    )
    calls, registered = [], []
    verify_results = [{0: (3, 99), 1: (0, 9)}, {0: (2, 99)}]

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, anchors, **_):
        calls.append(dict(rows=[r.tolist() for r in rows], anchors=anchors))
        return [
            (i, {"accepted": a, "next": y, "logprobs": [-1.0] * (a + 1), "draft_logq": engine_logq[i][:a]})
            for i, (a, y) in verify_results.pop(0).items()
        ]

    handlers = engine.handlers()
    handlers["register_engine_anchors"] = lambda entries: registered.append(entries) or len(entries)
    handlers["engine_anchor_stats"] = lambda: {"applied_tokens": 5.0}
    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=handlers,
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        anchor_horizon=2,
    )
    s.current_step = 0
    drafts = asyncio.run(s.draft_streaming([torch.tensor([1, 2]), torch.tensor([3])]))()
    assert drafts[0][0][8].length == 4  # cut to c, self beyond
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    first, second = calls
    assert [a.length for a in first["anchors"]] == [4, 3]
    # one continuation (the rejected rollout), budget H + 1 - len(prefix) = 2
    assert len(engine.started) == 2 and engine.started[1][:2] == ([[3, 9]], [2])
    assert second["rows"] == [[3, 9, 11, 12]] and second["anchors"][0].length == 3
    plans = {k: v[0] for k, v in s.plans.items()}
    p0, p1 = plans[(1, 2)], plans[(3,)]
    assert not p0.complete and p0.prefix == [4, 5, 6] and p0.anchors.complete == 4
    assert not p1.complete and p1.prefix == [9, 11, 12] and p1.anchors.complete == 3
    (entries,) = registered
    assert set(entries) == {p0.engine_key, p1.engine_key}
    assert entries[p1.engine_key].length == 3
    assert s.stats["spec/horizon_plans"] == 2.0 and s.stats["spec/anchor_applied_tokens"] == 5.0
    # The rollout continues on the refit engine with the anchor key; training
    # claims the anchors (self beyond c).
    datum = BatchedDataDict({"input_ids": torch.tensor([[3]]), "input_lengths": torch.tensor([1])})
    _, out = asyncio.run(s._one(0, datum))
    req = s.base.requests[-1]
    assert req["ts_anchor_key"] == [p1.engine_key]
    assert req["input_ids"][0].tolist() == [3, 9, 11, 12]
    toks = out["output_ids"][0]
    assert toks.tolist() == [3, 9, 11, 12, 9, EOD]
    a = s.claim_anchors(torch.tensor([3]), toks)
    assert a is p1.anchors and fetch_anchors(a, [1], 1, 6, lambda t: t)[3:].eq(-1).all()


def test_horizon_config_validation():
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    assert ThunderSyncConfig().sticky_anchor_horizon is None
    assert ThunderSyncConfig(sticky_anchor_horizon=128).sticky_anchor_horizon == 128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("E,K", [(8, 1), (8, 2), (16, 3), (128, 8), (512, 22), (60, 7)])
def test_fused_rule_is_bitwise_the_reference(E, K):
    """The engine's one-kernel rule returns bitwise what the learner's torch
    rule returns: near ties (scores on a coarse grid), invalid anchor rows
    (negative ids, repeats), anchors kept and rejected."""
    from nemo_rl.models.megatron.sticky_routing_fused import sticky_select_fused

    g = torch.Generator().manual_seed(E * 100 + K)
    T = 3000
    scores = (torch.randn(T, E, generator=g) * 4).round() / 4  # many exact ties
    noisy = scores + 0.3 * torch.randn(T, E, generator=g)
    anchors = torch.topk(noisy, K).indices
    anchors[::7, 0] = -1  # self rows
    if K > 1:
        anchors[3::11, 1] = anchors[3::11, 0]  # repeated ids: self
    anchors = anchors.to(torch.int16)
    scores, anchors = scores.cuda(), anchors.cuda()
    pv, pi = torch.topk(scores, K, dim=1)
    for delta in (0.0, 0.03, 0.3):
        ref = sr.sticky_select(scores, anchors, pv, pi, delta)
        out = sticky_select_fused(scores, anchors, pv, pi, delta)
        assert torch.equal(out[0], ref[0]) and torch.equal(out[1], ref[1])
        assert torch.equal(out[2], ref[2])
    assert bool(ref[2].any()) and not bool(ref[2].all())


@pytest.mark.parametrize("device", devices)
def test_engine_apply_nemotron_sigmoid_bias_router(sticky_on, monkeypatch, device):
    """Nemotron-3 router (sigmoid scores, expert bias selects, top-6 of 128):
    the rule acts on the selection scores (sigmoid + bias), probabilities are
    the sigmoid scores of the effective experts; engine apply == learner."""
    from megatron.core.inference import utils as inf_utils
    from megatron.core.transformer.moe.moe_utils import topk_routing_with_score_function
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)
    g = torch.Generator().manual_seed(11)
    T, E, K = 80, 128, 6
    logits = torch.randn(T, E, generator=g).to(device)
    bias = (0.05 * torch.randn(E, generator=g)).to(device)
    sel = torch.sigmoid(logits.float().cpu()) + bias.cpu()
    anchors = torch.topk(sel + 0.02 * torch.randn(T, E, generator=g), K).indices
    anchors[::4] = NO_ANCHOR
    anchors = anchors.to(torch.int16).to(device)

    def route(replay):
        return topk_routing_with_score_function(
            logits, K, score_function="sigmoid", expert_bias=bias, router_replay=replay,
            scaling_factor=2.5,
        )

    eng = _replay()
    setattr(eng, sr.APPLY_ATTR, (anchors, 0))
    eng.set_router_replay_action(RouterReplayAction.RECORD)
    lrn = _replay()
    lrn.set_target_indices(anchors.long())
    lrn.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    (pe, me), (pl, ml), (pp, mp) = route(eng), route(lrn), route(None)
    assert torch.equal(me, ml) and torch.equal(pe, pl)
    assert torch.equal(me[::4], mp[::4]) and not torch.equal(me, mp)
