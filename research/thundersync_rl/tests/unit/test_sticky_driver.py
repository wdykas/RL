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
    """pop_draft_routes payload: one layer (number 1), top-1, positions
    0..n_pos-1 logged, as blocks of 2 positions (the "refs" are the tensors:
    ``ray.get`` is the identity here)."""
    from thundersync_rl.routes import put_routes

    dense = torch.full((n_pos, 1, 1), val, dtype=torch.int16)
    return ([1], n_pos, put_routes(dense, lambda t: t, 2))


def _dense(refs, width=None):
    """A ``RouteRefs``' anchors [length, 1, 1] (layer 1, top-1)."""
    from thundersync_rl.routes import fetch_anchors

    n = refs.length if width is None else width
    return fetch_anchors(refs, [1], 1, n, lambda t: t)


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
            "install_draft_route_log": lambda: [1],
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
    assert drafts[0][0][8].length == 5 and drafts[0][0][8].complete == 4
    assert int(_dense(drafts[0][0][8])[4, 0, 0]) == -1
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    # Verification forwards got the anchors; the dropped draft never reached it.
    first, second = calls
    assert first["rows"] == [[1, 2, 4, 5, 6], [3, 7, EOD]]
    assert [a.length for a in first["anchors"]] == [5, 3]
    # Continuation round drafted from the kept prefixes with theta_{k-1}.
    prompts1, budgets1, _ = engine.started[1]
    assert prompts1 == [[1, 2, 4, 5, 6], [3, 9]] and budgets1 == [13, 15]
    assert second["rows"] == [[1, 2, 4, 5, 6, 10, EOD], [3, 9, 11, EOD]]
    assert second["plens"] == [5, 2] and second["segs"] == [[(0, 0, 2)], [(0, 0, 2)]]
    # Anchors spliced: new log over positions 0..P'+g'-2, nothing after.
    a0, a1 = map(_dense, second["anchors"])
    assert a0.shape[0] == 7 and bool((a0[:6] == 2).all()) and int(a0[6, 0, 0]) == -1
    assert a1.shape[0] == 4 and bool((a1[:3] == 2).all())
    plans = {k: v[0] for k, v in s.plans.items()}
    p0, p1 = plans[(1, 2)], plans[(3,)]
    assert p0.complete and p0.prefix == [4, 5, 6, 10, EOD]
    assert p0.prefix_logprobs == [-1.0, -1.1, -1.2, -1.0, -1.1]  # no bonus kept
    assert p0.anchors.length == 7
    # [3] rejected at 0 (-> 9), then [11] kept and EOD residual.
    assert p1.complete and p1.prefix == [9, 11, EOD] and p1.anchors.length == 3
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
    # Every route block of the drafts and the round is freed after step 1
    # trained (freed by the driver: the learner fetched them by handle).
    freed = []
    s._free_refs = freed.extend
    s.release_routes(0)
    assert freed == []
    s.release_routes(1)
    assert len(freed) == 3 + 5 and not s._route_blocks  # drafts, round 1


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
    from thundersync_rl.routes import RouteRefs
    from thundersync_rl.speculative import Plan

    s = make_spec(
        {},
        base_handlers={"install_draft_hidden_log": lambda: True, "install_draft_route_log": lambda: [1]},
        verify_source="engine",
        sticky_delta=0.03,
    )
    s.plans[(1,)].append(Plan(-1, [4], [-0.1], anchors=RouteRefs([], 2, 2)))
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


def test_chunk_data_carries_route_handles():
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

    from thundersync_rl.routes import RouteRefs

    a = RouteRefs([], 5, 5)  # handles only: no route data on the driver
    chunk = SimpleNamespace(
        trajectories=[traj([1, 2, 3, 4, 5], a), traj([1, 2, 7], None)],
        group=None,
        advantages=[1.0, -1.0],
        padding=False,
    )
    data = build_chunk_data(chunk, tok, cfg, 1)
    assert data["route_refs"] == [a, None] and "routed_experts" not in data
    chunk.trajectories = [traj([1, 2, 7], None)]
    chunk.advantages = [1.0]
    assert "route_refs" not in build_chunk_data(chunk, tok, cfg, 1)


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


def test_worker_anchor_batch_fetches_its_stage_layers(monkeypatch):
    """The learner materializes only its pipeline stage's MoE layers (here 3
    and 5 of 1, 3, 5) and fetches only the blocks that hold them."""
    import thundersync_rl.worker as worker_mod
    from thundersync_rl.routes import merge_route_refs, put_routes

    from nemo_rl.models.megatron import sticky_routing

    fetched = []
    monkeypatch.setattr(worker_mod.ray, "get", lambda r: fetched.append(r) or r)
    w = _worker()
    w._router_replay_enabled = True
    w._stage_layers = ([3, 5], 2)
    s1 = torch.full((4, 1, 2), 7, dtype=torch.int16)  # stage 1: layer 1
    s2 = torch.arange(4 * 2 * 2, dtype=torch.int16).view(4, 2, 2)  # layers 3, 5
    refs = merge_route_refs(
        [([1], 4, put_routes(s1, lambda t: t)), ([3, 5], 4, put_routes(s2, lambda t: t))],
        4,
        [1, 3, 5],
    )
    sticky_routing.set_delta(0.03)
    try:
        out = w._anchor_batch([refs, None], 2, 8)
        assert out.shape == (2, 8, 2, 2)
        assert torch.equal(out[0, :4], s2) and bool((out[0, 4:] == -1).all())
        assert bool((out[1] == -1).all())
        assert len(fetched) == 1 and fetched[0] is not s1
        d = BatchedDataDict(
            {"input_ids": torch.zeros(2, 6, dtype=torch.long), "route_refs": [None, refs]}
        )
        r = w._with_routes(d)
        assert "route_refs" not in r and r["routed_experts"].shape == (2, 6, 2, 2)
        assert torch.equal(r["routed_experts"][1, :4], s2)
        d = BatchedDataDict({"input_ids": torch.zeros(2, 6, dtype=torch.long)})
        assert bool((w._with_routes(d)["routed_experts"] == -1).all())
    finally:
        sticky_routing.set_delta(None)
    assert w._anchor_batch(None, 1, 8) is None


def test_stage_local_routed_experts_layout():
    """routed_experts with one column per MoE layer of this pipeline stage."""
    from nemo_rl.models.megatron.router_replay import _payload_indices_for_moe_layers

    idx = _payload_indices_for_moe_layers(
        global_moe_layers=[1, 2, 3, 4],
        num_payload_layers=2,
        total_num_layers=4,
        local_moe_layers=[4, 3],
    )
    assert idx == {3: 0, 4: 1}
    with pytest.raises(ValueError, match="stage-local"):
        _payload_indices_for_moe_layers(
            global_moe_layers=[1, 2, 3, 4], num_payload_layers=3, total_num_layers=4
        )


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
        request_in_prefill_status_tensor=torch.tensor([1, 1, 0, 0], dtype=torch.int32),
        total_request_count=2,
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
    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(worker_mod.ray, "put", lambda t: t)
    assert w.install_draft_route_log() == [1, 3]
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
    layers, complete, blocks = out["d|0"]
    assert layers == [1, 3] and complete == 5
    ((routes, lo, hi, nbytes),) = blocks  # put into the object store
    assert (lo, hi, nbytes) == (0, 5, 5 * L * K * 2)
    assert routes[:, 0, 0].tolist() == [0, 1, 2, 100, 200]
    assert routes[:, 1, 1].tolist() == [10, 11, 12, 110, 210]


# --------------------------------------------------------------------------
# Route truncation: hidden rows follow the kept tokens
# --------------------------------------------------------------------------


class _LoggingEngine(_Engine):
    """Hidden log like ``HiddenLog.pop``: rows 0..min(n, logged)-1 of a draft."""

    def __init__(self, rounds, hidden_rows):
        super().__init__(rounds)
        self.hidden_rows = list(hidden_rows)  # per round: logged rows per draft
        self.popped = []

    def pop_hidden(self, keys, lengths):
        logged = self.hidden_rows.pop(0)
        self.popped.append(list(lengths))
        return {
            k: torch.ones(min(n, m), 4)
            for k, n, m in zip(keys, lengths, logged)
            if min(n, m)
        }


def test_route_truncated_drafts_keep_hidden_rows_matching_tokens():
    from thundersync_rl.engine_q import hidden_num_rows

    engine = _LoggingEngine(
        [
            # prompt [1, 2]: routes of positions 0..2 only -> m = 2 of 4 tokens;
            # prompt [3]: no decode route -> m = 1 of 3; prompt [8, 9]: prompt
            # routes incomplete (position 0 only) -> m = 0 of 2.
            [
                ([4, 5, 6, 7], False, [-0.1] * 4, 3),
                ([7, 8, 9], False, [-0.2] * 3, 1),
                ([5, 6], False, [-0.3] * 2, 1),
            ],
            # continuation round 1 from [1,2,4,5], [3,7], [8,9]:
            # routes 0..4 -> m = 2 of 3; complete; routes 0..1 -> m = 1 of 2
            [
                ([10, 11, 12], False, [-0.4] * 3, 5),
                ([9, EOD], True, [-0.5] * 2, 3),
                ([4, EOD], True, [-0.5] * 2, 2),
            ],
            # round 2 from [1,2,4,5,10,11] and [8,9,4]
            [([EOD], True, [-0.6], 6), ([EOD], True, [-0.6], 3)],
        ],
        hidden_rows=[[4, 3, 2], [3, 2, 2], [1, 1]],  # every hidden row was logged
    )
    seen = []

    def verify(rows, prompt_lens, engine_logq, hidden, anchors, **_):
        out = []
        for i, (r, plen) in enumerate(zip(rows, prompt_lens)):
            g = r.numel() - plen
            # what verify_drafts_block_engine asserts
            assert len(engine_logq[i]) == g
            assert hidden_num_rows(hidden[i]) == g, (hidden_num_rows(hidden[i]), g)
            seen.append(r.tolist())
            res = {"accepted": g, "next": 99, "logprobs": [-1.0] * (g + 1)}
            out.append((i, {**res, "draft_logq": engine_logq[i]}))
        return out

    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
    )
    s.current_step = 0
    prompts = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([8, 9])]
    drafts = asyncio.run(s.draft_streaming(prompts))()
    toks = [d[0][2] for d in drafts]
    assert toks == [[4, 5], [7], []]
    # hidden popped only for the anchored tokens, and rows == tokens
    assert engine.popped[0] == [2, 1, 0]
    assert [hidden_num_rows(d[0][7]) for d in drafts] == [2, 1, 0]
    assert [len(d[0][6]) for d in drafts] == [2, 1, 0]
    assert s.stats["spec/route_truncated_drafts"] == 3.0
    assert s.stats["spec/route_truncated_tokens"] == 2.0 + 2.0 + 2.0
    # Verification of the cut drafts, then continuation rounds (cut by routes
    # too): hidden rows must match the tokens everywhere.
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    assert engine.popped[1:] == [[2, 2, 1], [1, 1]]
    assert seen[:3] == [[1, 2, 4, 5], [3, 7], [8, 9]]
    assert seen[3:6] == [[1, 2, 4, 5, 10, 11], [3, 7, 9, EOD], [8, 9, 4]]
    plans = {k: v[0] for k, v in s.plans.items()}
    assert plans[(1, 2)].complete and plans[(1, 2)].prefix == [4, 5, 10, 11, EOD]
    assert plans[(8, 9)].complete and plans[(8, 9)].prefix == [4, EOD]
    assert plans[(3,)].prefix == [7, 9, EOD]


@pytest.mark.parametrize("same_routes", [True, False])
def test_route_check_against_an_engine_prefill(monkeypatch, same_routes):
    """THUNDERSYNC_ENGINE_Q_CHECK=1: anchors vs the engine's re-prefill of the row."""
    monkeypatch.setenv("THUNDERSYNC_ENGINE_Q_CHECK", "1")

    class Engine(_Engine):
        def pop_routes(self, keys):
            val = 1 if same_routes or len(self.started) == 1 else 2
            return {
                k: _routes(None, n_pos, val=val)
                for k, (_, _, _, n_pos) in zip(keys, self.cur[0])
                if n_pos
            }

    engine = Engine(
        [
            [([4, 5, 6], False, [-0.1] * 3, 4), ([7], False, [-0.2], 1)],
            # route-check prefill of [1,2,4,5,6] and [3,7] (+1 token each)
            [([0], False, [-0.3], 5), ([0], False, [-0.3], 2)],
        ]
    )

    def verify(rows, **_):  # reject at 0 with EOS: every rollout completes
        return [(i, {"accepted": 0, "next": EOD, "logprobs": [-1.0]}) for i in range(len(rows))]

    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=engine.handlers(),
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
    )
    freed = []
    s._free_refs = freed.extend
    prompts = [torch.tensor([1, 2]), torch.tensor([3])]
    drafts = asyncio.run(s.draft_streaming(prompts))()
    s.current_step = 1
    asyncio.run(s.verify_sticky(drafts))
    prompts_chk, budgets_chk, _ = engine.started[1]
    assert prompts_chk == [[1, 2, 4, 5, 6], [3, 7]] and budgets_chk == [1, 1]
    assert s.stats["spec/route_check_tokens"] == 4.0 + 1.0
    assert s.stats["spec/route_check_exact"] == (5.0 if same_routes else 0.0)
    assert len(freed) == 3 + 1  # the check's own route blocks, freed at once


# --------------------------------------------------------------------------
# Engine: async scheduling, MCore bookkeeping, rebuilt engines
# --------------------------------------------------------------------------


def _fake_engine_base():
    """An engine base class with MCore's own async-scheduling validation."""
    from megatron.core.inference.config import AsyncScheduleMode
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    class Base:
        _validate_async_sched_support_for_config = (
            DynamicInferenceEngine._validate_async_sched_support_for_config
        )

        def __init__(self, controller, context):
            self.controller = controller
            self.context = context
            self.num_speculative_tokens = 0
            self._validate_async_sched_support_for_config()

    def controller():
        cfg = SimpleNamespace(moe_enable_routing_replay=True)
        return SimpleNamespace(
            inference_wrapped_model=SimpleNamespace(model=SimpleNamespace(config=cfg)),
            num_mtp_depths=0,
            _router_record_bookkeeping=lambda: "mcore routes",
            _dynamic_step_forward_logits=lambda *a: None,
        )

    ctx = SimpleNamespace(config=SimpleNamespace(async_sched_mode=AsyncScheduleMode.ASYNC))
    return Base, controller, ctx


def test_engine_with_route_log_runs_async_without_mcore_bookkeeping():
    """MCore rejects routing replay under async scheduling; with our route
    record it is allowed and MCore's own per-step bookkeeping is dropped.
    Without it, MCore's check still applies."""
    Base, make_controller, ctx = _fake_engine_base()
    w = _worker()
    w._install_hidden_copy = lambda c: None
    w._install_route_record = lambda c: None
    with pytest.raises(ValueError, match="routing replay"):
        w._engine_class(Base)(make_controller(), ctx)
    w._route_record = {"replays": [], "layers": [1], "sequence_parallel": False}
    c = make_controller()
    eng = w._engine_class(Base)(c, ctx)
    assert eng.controller._router_record_bookkeeping() is None
    assert c.inference_wrapped_model.model.config.moe_enable_routing_replay is True


def test_rebuilt_engine_keeps_logging_into_the_route_log(monkeypatch):
    """An engine built after the route log was installed (rebuild after a
    refit or offload) gets its forward wrapped and logs into the same log."""
    from megatron.core import parallel_state as ps

    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
    monkeypatch.setattr(worker_mod.ray, "put", lambda t: t)
    Base, make_controller, ctx = _fake_engine_base()
    replay = SimpleNamespace(static_buffer=None, recorded_topk_idx=None)
    w = _worker()
    w._install_hidden_copy = lambda c: None
    w._install_route_record = lambda c: None
    w._route_record = {"replays": [replay], "layers": [1], "sequence_parallel": False}
    sp = SimpleNamespace(ts_hidden_key="d|0")
    req = SimpleNamespace(record=[SimpleNamespace(sampling_params=sp)])

    def engine(controller):
        ctx2 = SimpleNamespace(
            config=ctx.config,
            active_token_count=1,
            total_request_count=1,
            request_ids=torch.tensor([5], dtype=torch.int32),
            request_in_prefill_status_tensor=torch.tensor([1], dtype=torch.int32),
            token_to_request_idx=torch.zeros(1, dtype=torch.int32),
            token_to_pos_ids=torch.zeros(1, dtype=torch.long),
        )
        eng = w._engine_class(Base)(controller, ctx2)
        eng.requests = {5: req}
        w.dynamic_inference_engine, w.inference_context = eng, ctx2
        return eng, ctx2

    eng1, ctx1 = engine(make_controller())
    assert w.install_draft_route_log() == [1]
    replay.recorded_topk_idx = torch.tensor([[3]])
    eng1.controller._dynamic_step_forward_logits()
    eng2, ctx2 = engine(make_controller())  # rebuilt: new controller and context
    ctx2.token_to_pos_ids[0] = 1
    replay.recorded_topk_idx = torch.tensor([[4]])
    eng2.controller._dynamic_step_forward_logits()
    assert w.install_draft_route_log() == [1]  # idempotent
    eng2.controller._dynamic_step_forward_logits()
    (layers, complete, ((routes, lo, hi, _),)) = w.pop_draft_routes(["d|0"])["d|0"]
    assert complete == 2 and routes[:, 0, 0].tolist() == [3, 4]


def test_route_record_rearms_new_router_instances(monkeypatch):
    """A rebuilt generation model has new RouterReplay instances: they are
    armed (RECORD, inference-only) and the route log reads them."""
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    import nemo_rl.models.megatron.router_replay as rr
    from nemo_rl.models.megatron import sticky_routing

    class Replay:
        def set_router_replay_action(self, a):
            self.action = a

    models = {"m1": [(Replay(), 1), (Replay(), 3)], "m2": [(Replay(), 1), (Replay(), 3)]}
    monkeypatch.setattr(rr, "_router_replay_instances_for_model", lambda m: models[m])
    monkeypatch.setattr(sticky_routing, "install", lambda: None)
    import megatron.core.utils as mcu

    monkeypatch.setattr(mcu, "get_model_config", lambda m: SimpleNamespace())
    w = _worker()
    ctx = SimpleNamespace(moe_enable_routing_replay=True)
    sticky_routing.set_delta(0.03)
    try:
        w._gen_model = lambda: "m1"
        w._install_route_record(ctx)
        first = w._route_record
        w._install_route_record(ctx)
        assert w._route_record is first  # same instances: unchanged
        w._gen_model = lambda: "m2"
        w._install_route_record(ctx)
    finally:
        sticky_routing.set_delta(None)
    assert w._route_record["replays"] == [r for r, _ in models["m2"]]
    assert all(r.action == RouterReplayAction.RECORD for r, _ in models["m2"])
    assert all(r._nrl_record_inference_only for r, _ in models["m2"])
