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
"""Sticky routing in the speculative driver, the training data, the config
validation and the worker (no Ray, no GPUs)."""

import asyncio
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_speculative_driver import EOD, _spec as make_spec

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _routes(key_prompt_len, n_pos, val=1):
    """pop_draft_routes payload: one layer (number 1), top-1, positions 0..n_pos-1."""
    pos = torch.arange(n_pos)
    return ([1], pos, torch.full((n_pos, 1, 1), val, dtype=torch.int16))


class _Engine:
    """Fake generation rank 0 + route/hidden logs for two drafting rounds."""

    def __init__(self, rounds):
        self.rounds = rounds  # list of [(tokens, finished, logq, n_route_pos)]
        self.started = []
        self.cur = None

    def start(self, prompts, budget, interval, hidden_keys=None):
        self.started.append((prompts, budget, hidden_keys))
        self.cur = (self.rounds.pop(0), hidden_keys)

    def collect(self, records=None):
        return [(t, f, lq) for t, f, lq, _ in self.cur[0]]

    def pop_hidden(self, keys, lengths):
        return {k: torch.ones(n, 4) for k, n in zip(keys, lengths) if n}

    def pop_routes(self, keys):
        out = {}
        for k, (_, _, _, n_pos) in zip(keys, self.cur[0]):
            if n_pos:
                out[k] = _routes(None, n_pos, val=len(self.started))
        return out

    def handlers(self):
        return {
            "install_draft_hidden_log": lambda: True,
            "install_draft_route_log": lambda: True,
            "start_drafts": self.start,
            "draft_request_ids": lambda: {},
            "collect_drafts": self.collect,
            "pop_draft_hidden": self.pop_hidden,
            "pop_draft_routes": self.pop_routes,
            "drafts_done": lambda: [True] * len(self.cur[0]),
        }


def test_sticky_drafts_verify_and_continue_to_completion():
    engine = _Engine(
        [
            # Drafting round (iteration k-1), prompts [1,2], [3], [8]:
            [
                ([4, 5, 6], False, [-0.1] * 3, 4),  # cut; routes 0..P+g-2 = 0..3
                ([7, EOD], True, [-0.2] * 2, 2),  # finished; routes 0..1
                ([5], False, [-0.3], 0),  # no routes logged: dropped
            ],
            # Continuation round 1 (engine still at theta_{k-1}):
            [
                ([10, EOD], True, [-0.4] * 2, 6),  # from [1,2,4,5,6]: routes 0..5
                ([11, EOD], True, [-0.5] * 2, 3),  # from [3,9]: routes 0..2
            ],
        ]
    )
    calls = []
    verify_results = [
        {0: (3, 99), 1: (0, 9)},  # whole draft (no bonus); reject at 0 -> 9
        {0: (2, 99), 1: (1, EOD)},  # whole and finished; reject at 1 -> EOD
    ]

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, anchors, **_):
        calls.append(
            dict(rows=[r.tolist() for r in rows], plens=prompt_lens, segs=segments, anchors=anchors)
        )
        res = verify_results.pop(0)
        return [
            (
                i,
                {
                    "accepted": a,
                    "next": y,
                    "logprobs": [-1.0 - 0.1 * j for j in range(a + 1)],
                    "draft_logq": engine_logq[i][:a],
                },
            )
            for i, (a, y) in res.items()
        ]

    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
    )
    s.current_step = 0
    prompts = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([8])]
    drafts = asyncio.run(s.draft_streaming(prompts))()
    # The draft without logged routes has no anchors: dropped before verification.
    assert drafts[2][0][8] is None
    assert drafts[0][0][8].shape[0] == 5 and int(drafts[0][0][8][4, 0, 0]) == -1
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    # Verification forwards got the anchors; the dropped draft never reached it.
    first, second = calls
    assert first["rows"] == [[1, 2, 4, 5, 6], [3, 7, EOD]]
    assert [a.shape[0] for a in first["anchors"]] == [5, 3]
    # Continuation round drafted from the kept prefixes with theta_{k-1}.
    prompts1, budgets1, _ = engine.started[1]
    assert prompts1 == [[1, 2, 4, 5, 6], [3, 9]] and budgets1 == [13, 15]
    assert second["rows"] == [[1, 2, 4, 5, 6, 10, EOD], [3, 9, 11, EOD]]
    assert second["plens"] == [5, 2] and second["segs"] == [[(0, 0, 2)], [(0, 0, 2)]]
    # Anchors spliced: new log over positions 0..P'+g'-2, nothing after.
    a0, a1 = second["anchors"]
    assert a0.shape[0] == 7 and bool((a0[:6] == 2).all()) and int(a0[6, 0, 0]) == -1
    assert a1.shape[0] == 4 and bool((a1[:3] == 2).all())
    plans = {k: v[0] for k, v in s.plans.items()}
    p0, p1 = plans[(1, 2)], plans[(3,)]
    assert p0.complete and p0.prefix == [4, 5, 6, 10, EOD]
    assert p0.prefix_logprobs == [-1.0, -1.1, -1.2, -1.0, -1.1]  # no bonus kept
    assert p0.anchors.shape[0] == 7
    # [3] rejected at 0 (-> 9), then [11] kept and EOD residual.
    assert p1.complete and p1.prefix == [9, 11, EOD] and p1.anchors.shape[0] == 3
    assert s.stats["spec/sticky_rounds"] == 1.0
    assert (8,) not in s.plans
    # The rollout returns the plan; training claims its anchors.
    datum = BatchedDataDict(
        {"input_ids": torch.tensor([[1, 2]]), "input_lengths": torch.tensor([2])}
    )
    _, out = asyncio.run(s._one(0, datum))
    toks = out["output_ids"][0]
    assert toks.tolist() == [1, 2, 4, 5, 6, 10, EOD]
    assert s.claim_anchors(torch.tensor([1, 2]), toks) is p0.anchors
    assert s.claim_anchors(torch.tensor([1, 2]), toks) is None


def test_sticky_round_retries_unlogged_continuations():
    engine = _Engine(
        [
            [([4, 5], False, [-0.1] * 2, 3)],
            [([6, EOD], True, [-0.2] * 2, 0)],  # routes lost: retry
            [([6, EOD], True, [-0.2] * 2, 5)],
        ]
    )
    results = [{0: (2, 0)}, {0: (2, 0)}]

    def verify(rows, anchors, **_):
        r = results.pop(0)
        return [(0, {"accepted": a, "next": y, "logprobs": [-1.0] * (a + 1)}) for a, y in r.values()]

    s = make_spec(
        {"verify_drafts_block_engine": verify},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
    )
    drafts = asyncio.run(s.draft_streaming([torch.tensor([1, 2])]))()
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    (plan,) = s.plans[(1, 2)]
    assert plan.complete and plan.prefix == [4, 5, 6, EOD]
    assert s.stats["spec/sticky_rounds"] == 2.0 and s.stats["spec/sticky_cont_retries"] == 1.0


def test_incomplete_sticky_plan_never_reaches_plain_decoding():
    from thundersync_rl.speculative import Plan

    s = make_spec(
        {},
        base_handlers={"install_draft_hidden_log": lambda: True, "install_draft_route_log": lambda: True},
        verify_source="engine",
        sticky_delta=0.03,
    )
    s.plans[(1,)].append(Plan(-1, [4], [-0.1], anchors=torch.zeros(2, 1, 1)))
    datum = BatchedDataDict({"input_ids": torch.tensor([[1]]), "input_lengths": torch.tensor([1])})
    with pytest.raises(RuntimeError, match="incomplete"):
        asyncio.run(s._one(0, datum))


def test_sticky_requires_engine_route_log():
    with pytest.raises(RuntimeError, match="cannot log"):
        make_spec(
            {},
            base_handlers={
                "install_draft_hidden_log": lambda: True,
                "install_draft_route_log": lambda: False,
            },
            verify_source="engine",
            sticky_delta=0.03,
        )
    with pytest.raises(ValueError, match="verify_source"):
        make_spec({}, sticky_delta=0.03)


# --------------------------------------------------------------------------
# Training data: routed_experts per row
# --------------------------------------------------------------------------


def test_chunk_data_carries_anchors_and_self_rows():
    from thundersync_rl.grpo_loop import build_chunk_data

    tok = SimpleNamespace(pad_token_id=0)
    cfg = SimpleNamespace(policy={"make_sequence_length_divisible_by": 4})

    def traj(ids, anchors):
        return SimpleNamespace(
            payload={
                "message_log": [
                    {"role": "user", "token_ids": torch.tensor(ids[:2])},
                    {"role": "assistant", "token_ids": torch.tensor(ids[2:]), "generation_logprobs": torch.zeros(len(ids) - 2)},
                ],
                "loss_multiplier": 1.0,
                "anchors": anchors,
            }
        )

    a = torch.arange(5 * 2 * 3, dtype=torch.int16).view(5, 2, 3)
    chunk = SimpleNamespace(
        trajectories=[traj([1, 2, 3, 4, 5], a), traj([1, 2, 7], None)],
        group=None,
        advantages=[1.0, -1.0],
        padding=False,
    )
    data = build_chunk_data(chunk, tok, cfg, 1)
    r = data["routed_experts"]
    assert r.shape == (2, data["input_ids"].shape[1], 2, 3)
    assert torch.equal(r[0, :5], a) and bool((r[0, 5:] == -1).all())
    assert bool((r[1] == -1).all())
    chunk.trajectories = [traj([1, 2, 7], None)]
    chunk.advantages = [1.0]
    assert "routed_experts" not in build_chunk_data(chunk, tok, cfg, 1)


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def _master(**ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        verify_on="learner",
        draft_lookahead=1,
        sticky_routing=True,
    )
    base.update(ts)
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        grpo=SimpleNamespace(max_rollout_turns=1),
        policy={
            "megatron_cfg": {"expert_model_parallel_size": 1},
            "generation": {"mcore_generation_config": {"enable_prefix_caching": False}},
        },
    )


def test_sticky_config_default_and_valid():
    from thundersync_rl.grpo_loop import (
        ThunderSyncConfig,
        configure_sticky_policy,
        sticky_delta,
        validate_sticky_routing,
    )

    assert ThunderSyncConfig().sticky_routing_delta == 0.03
    assert sticky_delta(ThunderSyncConfig()) is None  # off unless enabled
    m = _master()
    validate_sticky_routing(m)
    configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["sticky_routing_delta"] == 0.03
    m = _master(sticky_routing_delta=None)
    configure_sticky_policy(m)
    assert m.policy["megatron_cfg"]["sticky_routing_delta"] is None


@pytest.mark.parametrize(
    "ts,turns,cache,match",
    [
        ({"verify_source": "scorer"}, 1, False, "verify_source"),
        ({"draft_lookahead": 3}, 1, False, "draft_lookahead"),
        ({"tail_lookup": True}, 1, False, "tail_lookup"),
        ({"speculate": False}, 1, False, "speculate"),
        ({"sticky_routing_delta": -1.0}, 1, False, ">= 0"),
        ({}, 2, False, "max_rollout_turns"),
        ({}, 1, True, "prefix_caching"),
    ],
)
def test_sticky_config_rejects(ts, turns, cache, match):
    from thundersync_rl.grpo_loop import validate_sticky_routing

    m = _master(**ts)
    m.grpo.max_rollout_turns = turns
    m.policy["generation"]["mcore_generation_config"]["enable_prefix_caching"] = cache
    with pytest.raises(ValueError, match=match):
        validate_sticky_routing(m)


# --------------------------------------------------------------------------
# Worker: anchored training forwards, engine route log
# --------------------------------------------------------------------------


def _worker():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    return object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)


def test_worker_anchor_batch_and_route_fill(monkeypatch):
    from nemo_rl.models.megatron import router_replay, sticky_routing

    w = _worker()
    w._router_replay_enabled = True
    w.model = SimpleNamespace(config=SimpleNamespace())
    monkeypatch.setattr(router_replay, "router_replay_dimensions", lambda cfg: (3, 2))
    import megatron.core.utils as mcu

    monkeypatch.setattr(mcu, "get_model_config", lambda m: m.config)
    sticky_routing.set_delta(0.03)
    try:
        a = torch.ones(4, 3, 2, dtype=torch.int16)
        out = w._anchor_batch([torch.zeros(5), torch.zeros(3)], [a, None], 8)
        assert out.shape == (2, 8, 3, 2)
        assert bool((out[0, :4] == 1).all()) and bool((out[0, 4:] == -1).all())
        assert bool((out[1] == -1).all())
        d = BatchedDataDict({"input_ids": torch.zeros(2, 6, dtype=torch.long)})
        assert w._with_routes(d)["routed_experts"].shape == (2, 6, 3, 2)
    finally:
        sticky_routing.set_delta(None)
    assert w._anchor_batch([torch.zeros(5)], None, 8) is None


def test_engine_route_log_on_a_fake_engine(monkeypatch):
    """install_draft_route_log: routes of tagged tokens by absolute position,
    snapshotted at forward launch, read after the forward from where MCore's
    context had each router record them (its buffer on CUDA-graph steps,
    ``recorded_topk_idx`` on eager steps). Reading a buffer bound once at
    engine build lost every route on GPU (all drafts truncated to 0)."""
    from megatron.core import parallel_state as ps

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)

    def req(key):
        sp = SimpleNamespace() if key is None else SimpleNamespace(ts_hidden_key=key)
        return SimpleNamespace(record=[SimpleNamespace(sampling_params=sp)])

    L, K = 2, 2
    mcore = torch.zeros((16, L, K), dtype=torch.int32)
    replays = [SimpleNamespace(static_buffer=None, recorded_topk_idx=None) for _ in range(L)]
    ctx = SimpleNamespace(
        active_token_count=0,
        padded_active_token_count=0,
        token_to_request_idx=torch.zeros(16, dtype=torch.int32),
        token_to_pos_ids=torch.zeros(16, dtype=torch.long),
        request_ids=torch.tensor([11, 12, -1, -1], dtype=torch.int32),
    )

    def forward_logits(input_ids, position_ids):
        n = ctx.active_token_count
        step = forward_logits.step
        forward_logits.step += 1
        graph = step % 2 == 1  # eager prefill, graph decode, eager, ...
        for j, r in enumerate(replays):
            top = (100 * step + 10 * j + torch.arange(n)).view(n, 1).expand(n, K)
            if graph:  # MCore binds its buffer; the replay leaves recorded_topk_idx stale
                r.static_buffer = mcore[:, j]
                r.static_buffer[:n].copy_(top)
            else:  # no buffer: record_indices keeps the eager top-k
                r.static_buffer = None
                r.recorded_topk_idx = top.clone()

    forward_logits.step = 0
    controller = SimpleNamespace(_dynamic_step_forward_logits=forward_logits)
    w = _worker()
    w.dynamic_inference_engine = SimpleNamespace(
        requests={11: req("d|0"), 12: req(None)}, controller=controller
    )
    w.inference_context = ctx
    w._route_record = {"replays": replays, "layers": [1, 3], "sequence_parallel": False}
    assert w.install_draft_route_log() is True
    # eager prefill: request 11 (slot 0) 3 prompt tokens, request 12 (slot 1) 2
    ctx.active_token_count = 5
    ctx.token_to_request_idx[:5] = torch.tensor([0, 0, 0, 1, 1])
    ctx.token_to_pos_ids[:5] = torch.tensor([0, 1, 2, 0, 1])
    controller._dynamic_step_forward_logits(None, None)
    # graph decode, eager decode (one token each)
    for pos in (3, 4):
        ctx.active_token_count = 2
        ctx.token_to_request_idx[:2] = torch.tensor([0, 1])
        ctx.token_to_pos_ids[:2] = torch.tensor([pos, pos - 1])
        controller._dynamic_step_forward_logits(None, None)
    out = w.pop_draft_routes(["d|0", "x"])
    assert list(out) == ["d|0"]
    layers, pos, routes = out["d|0"]
    assert layers == [1, 3] and pos.tolist() == [0, 1, 2, 3, 4]
    assert routes[:, 0, 0].tolist() == [0, 1, 2, 100, 200]
    assert routes[:, 1, 1].tolist() == [10, 11, 12, 110, 210]
