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
"""Risk-first streaming verify0 (DESIGN 14z): schedule, growing canonical order,
decisions equal to the one-shot verify0, driver end to end (no Ray, no GPUs)."""

import asyncio
import random
from collections import Counter, defaultdict

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.block_verification import NULL_TOKEN
from thundersync_rl.null_bv import NullPolicy, envelopes
from thundersync_rl.speculative import Plan, SpeculativeGeneration
from thundersync_rl.streaming import StreamPlanner, Trajectory
from thundersync_rl.verify0_stream import (
    PromptCounts,
    StreamOrder,
    TailFeed,
    Verify0Risk,
    length_bucket,
    schedule_chunks,
)


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


# --------------------------------------------------------------------------
# Schedule: counts and past steps only
# --------------------------------------------------------------------------


def _prompts(lengths_per_prompt, plen=10):
    out, first = [], 0
    for p, lens in enumerate(lengths_per_prompt):
        out.append(PromptCounts(key=(p,), first=first, lengths=tuple(lens), tokens=sum(plen + n for n in lens)))
        first += len(lens)
    return out


def test_schedule_is_riskiest_first_whole_prompts_and_sized_by_tokens():
    risk = Verify0Risk()
    prompts = _prompts([[100, 120], [8000, 50], [3000, 3000], [10, 20], [8192, 8192]])
    chunks = schedule_chunks(prompts, risk, first_tokens=8000, chunk_tokens=10000)
    flat = [k for c in chunks for k in c]
    assert sorted(flat) == [(p,) for p in range(5)]  # every prompt once, never split
    # Riskiest = longest drafts under the constant-hazard prior.
    assert flat[:2] == [(4,), (1,)]
    assert flat[-1] == (3,)
    # The first chunk closes at >= 8000 tokens, later ones at >= 10000.
    tok = {p.key: p.tokens for p in prompts}
    assert sum(tok[k] for k in chunks[0]) >= 8000
    for c in chunks[1:-1]:
        assert sum(tok[k] for k in c) >= 10000


def test_schedule_drafting_order_control():
    risk = Verify0Risk()
    prompts = _prompts([[100], [8000], [3000]])
    assert schedule_chunks(prompts, risk, 1, 1, "drafting") == [[(0,)], [(1,)], [(2,)]]
    assert schedule_chunks(prompts, risk, 1, 1) == [[(1,)], [(2,)], [(0,)]]
    with pytest.raises(ValueError):
        schedule_chunks(prompts, risk, 1, 1, "content")


def test_risk_learns_remainders_from_past_tails():
    r = Verify0Risk()
    base = r.row_risk(6000)
    # Past step: drafts of ~6k rejected early, then decoded to the 8k cap: remainders ~8k.
    r.update([(6000, 10, 8192)] * 4 + [(6000, 6000, 6000)] * 4)
    assert r.remainder(6000) == pytest.approx((4 * 8182 + 2 * 6000) / 6)
    assert r.row_risk(6000) > base


def test_schedule_ties_keep_drafting_order_and_is_deterministic():
    risk = Verify0Risk()
    prompts = _prompts([[500]] * 6)
    a = schedule_chunks(prompts, risk, 1, 1)
    b = schedule_chunks(list(prompts), risk, 1, 1)
    assert a == b == [[(p,)] for p in range(6)]


def test_schedule_reads_counts_only():
    """Two steps whose drafts have the same counts but different tokens get the
    same schedule: the schedule's inputs are PromptCounts and the risk state."""
    risk = Verify0Risk()
    risk.update([(100, 100), (5000, 10), (5000, 5000), (8192, 0)])
    counts = _prompts([[4096, 10], [300], [8192], [9000]])
    rng = random.Random(0)
    # Tokens differ arbitrarily; only the counts reach the schedule.
    drafts_a = [[[rng.randrange(50) for _ in range(n)] for n in p.lengths] for p in counts]
    drafts_b = [[[rng.randrange(50) for _ in range(n)] for n in p.lengths] for p in counts]
    sched = []
    for drafts in (drafts_a, drafts_b):
        pc = [
            PromptCounts(key=p.key, first=p.first, lengths=tuple(len(d) for d in dd), tokens=p.tokens)
            for p, dd in zip(counts, drafts)
        ]
        sched.append(schedule_chunks(pc, risk, 5000, 5000))
    assert sched[0] == sched[1]


def test_risk_uses_past_steps_and_shrinks_to_prior():
    r = Verify0Risk()
    p0 = r.whole_rate(5000)
    assert p0 == pytest.approx(np.exp(-2e-5 * 5000))
    r.update([(5000, 5000)] * 6 + [(5000, 3)] * 2)  # bucket 4: 6 of 8 whole
    assert r.whole_rate(5000) == pytest.approx((6 + 4 * np.exp(-0.1)) / (8 + 4))
    assert r.whole_rate(100) == pytest.approx(np.exp(-2e-5 * 100))  # other bucket untouched
    assert length_bucket(8191) == 4 and length_bucket(8192) == 5 and r.steps == 1


# --------------------------------------------------------------------------
# Growing canonical order and the planner
# --------------------------------------------------------------------------


def _planner(groups, G=2, dp=2):
    pl = StreamPlanner(
        lambda rewards: [r - sum(rewards) / len(rewards) for r in rewards],
        dp_size=dp,
        max_chunk_trajectories=4,
        max_open_groups=0,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    for g in groups:
        pl.register_group(g, G)
    return pl


def _drain(pl, flush=True):
    out = []
    while pl.has_work(flush):
        d = pl.next_dispatch(flush)
        out.append([[tuple((t.group, t.index) for t in c.trajectories) for c in cs] for cs in d.per_rank])
    return out


def test_stream_order_segments_tails_and_fresh():
    # 6 groups; keys a, b, c, d, a (duplicate prompt), e (never drafted).
    keys = ["a", "b", "c", "d", "a", "e"]
    need = Counter({k: 2 * c for k, c in Counter(keys).items()})
    o = StreamOrder(keys, need, groups_per_block=2)
    assert o.decided({"c": (2, 0), "a": (4, 1)}) == [2]  # a has a tail
    assert o.decided({"b": (2, 0)}) == [1]
    assert o.decided({"d": (1, 0)}) == []  # one plan for 2 rows: a fresh row
    o.close()
    assert o.order == [2, 1, 0, 4, 3, 5]
    assert o.tail_groups == [0, 4] and o.fresh == [3, 5]
    assert o.blocks == [1, 1, 2, 2]
    with pytest.raises(RuntimeError):
        o.decided({"x": (1, 0)})


def test_stream_order_fresh_keys_never_ready():
    # "b" decodes at theta_k whatever its plans (debug fresh rows): never in a
    # decided segment, listed in ``fresh`` at close, after the tail groups.
    keys = ["a", "b", "c"]
    need = Counter({k: 2 for k in keys})
    o = StreamOrder(keys, need, groups_per_block=2, fresh_keys=["b"])
    assert o.decided({"a": (2, 0), "b": (2, 0)}) == [0]
    assert o.decided({"c": (2, 1)}) == []
    o.close()
    assert o.order == [0, 2, 1]
    assert o.tail_groups == [2] and o.fresh == [1]


@pytest.mark.parametrize("seed", range(4))
def test_stream_fresh_covers_every_row_decoded_at_theta_k(seed, monkeypatch):
    """Regression (v0s-xlanes at 6fc1ae9c, max|dlogp| 1.66): with cohort lanes the
    cohort lane takes theta_k only if ``StreamOrder.fresh`` is non-empty. Forced
    fresh rows (THUNDERSYNC_DEBUG_FRESH_MOD) had full plans, so ``fresh`` was empty,
    the lane kept its drafting weights and those rows were sampled from them.
    Every group with a rollout that ``_one`` decodes at theta_k must be fresh."""
    from thundersync_rl.grpo_loop import stream_fresh_keys

    monkeypatch.setenv("THUNDERSYNC_DEBUG_FRESH_MOD", "3")
    rng = random.Random(seed)
    G = 4
    keys = [tuple(rng.randrange(1, 50) for _ in range(rng.randrange(2, 6))) for _ in range(12)]
    keys[0] = (3, 3)  # at least one forced-fresh prompt
    need = Counter({k: G * c for k, c in Counter(keys).items()})
    o = StreamOrder(keys, need, groups_per_block=2, fresh_keys=stream_fresh_keys(keys))
    info = {}
    for k in dict.fromkeys(keys):
        # Every drafted prompt decided with enough plans, some short, some with tails.
        n = need[k] - (1 if rng.random() < 0.2 else 0)
        info[k] = (n, 1 if rng.random() < 0.2 else 0)
    o.decided(info)
    o.close()
    decodes_fresh = {
        p for p, k in enumerate(keys) if spec_mod._debug_forced_fresh(k) or info[k][0] < need[k]
    }
    assert 0 in decodes_fresh
    assert decodes_fresh == set(o.fresh)
    # The lanes rule (grpo_loop ``tails_then_refit``): a fresh row means a theta_k refit.
    assert bool(o.fresh)


def test_rollout_at_theta_k_raises_when_the_refit_was_skipped():
    s = SpeculativeGeneration.__new__(SpeculativeGeneration)
    s._check_fresh_weights((1, 2))  # default: the engine holds theta_k
    s.fresh_at_current = False
    with pytest.raises(RuntimeError, match="drafting weights"):
        s._check_fresh_weights((1, 2))


def test_planner_follows_a_growing_order_like_the_final_one():
    """The same final order given at once (set_order) or grown with the decisions
    (begin/extend/close) dispatches the same chunks, whatever the arrival order."""
    keys = [f"k{p}" for p in range(6)]
    need = Counter({k: 2 for k in keys})
    rows = [
        Trajectory(group=(0, p), index=2 * p + j, reward=float(j), payload=None, num_tokens=10 + p)
        for p in range(6)
        for j in range(2)
    ]
    decisions = [{"k4": (2, 0), "k1": (2, 1)}, {"k0": (2, 0), "k5": (2, 0)}, {"k2": (2, 0), "k3": (1, 0)}]
    ref = StreamOrder(keys, need, groups_per_block=2)
    for d in decisions:
        ref.decided(d)
    ref.close()
    pl = _planner([(0, p) for p in range(6)])
    pl.set_order([(0, p) for p in ref.order], ref.blocks)
    for t in rows:
        pl.add(t)
    want = _drain(pl)
    for seed in range(4):
        arrival = list(rows)
        random.Random(seed).shuffle(arrival)
        o = StreamOrder(keys, need, groups_per_block=2)
        pl = _planner([(0, p) for p in range(6)])
        o.attach(pl, 0)
        got, steps = [], list(decisions) + [None]
        # Interleave decisions and arrivals: a group's rows arrive after its decision.
        for d in steps:
            if d is None:
                o.close()
                ready = rows
            else:
                o.decided(d)
                ready = [t for t in rows if f"k{t.group[1]}" in o.info]
            for t in [t for t in arrival if t in ready and t.index not in pl.groups[t.group].rewards]:
                pl.add(t)
                got.extend(_drain(pl, flush=False))
        got.extend(_drain(pl))
        assert pl.all_done()
        assert got == want, seed


def test_planner_growing_order_rejects_bad_extensions():
    pl = _planner([(0, 0), (0, 1)])
    with pytest.raises(RuntimeError):
        pl.extend_order([(0, 0)], [1])
    pl.begin_order()
    pl.extend_order([(0, 1)], [1])
    with pytest.raises(ValueError):
        pl.extend_order([(0, 1)], [1])
    with pytest.raises(ValueError):
        pl.close_order()
    pl.extend_order([(0, 0)], [1])
    pl.close_order()


# --------------------------------------------------------------------------
# ⊥-BV: streamed decisions equal the one-shot verify0's, for any schedule
# --------------------------------------------------------------------------


def _stub_spec(policy, grid_by_row, drop_is=False):
    s = object.__new__(SpeculativeGeneration)
    s._null_policy, s.stats = policy, {}
    s.null_bv_drop_is = drop_is
    s.banded_is_band = None
    s.plans = defaultdict(list)
    s._release_hidden = lambda ds: list(ds)
    s._stop_streams = lambda streams: None
    s._tail_cv = None

    async def verify_block_async(flat, rows, null_envelopes=None, band=None, **_):
        await asyncio.sleep(random.random() * 1e-3)  # results arrive in any timing
        return {j: grid_by_row[i] for j, (i, _) in enumerate(flat)}

    s._verify_block_async = verify_block_async

    def verify_block(flat, rows, null_envelopes=None, band=None):
        return {j: grid_by_row[i] for j, (i, _) in enumerate(flat)}

    s._verify_block = verify_block

    async def continue_sticky(pending, trace=None, ended=(), feed=None, **_):
        while feed.open:
            feed.take()
            await asyncio.sleep(0)

    s._continue_sticky = continue_sticky

    def null_plan(d, res):
        p = Plan(-1, list(d[2][: res["accepted"]]), complete=res["accepted"] == len(d[2]))
        p.verified = (len(d[2]), res["accepted"])
        return p

    s._null_plan = null_plan
    return s


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("window", [1, 2, 3])
def test_streamed_null_decisions_equal_one_shot(seed, window):
    rng = np.random.default_rng(seed)
    policy = NullPolicy(logratio_var=0.2, ref_len=3)
    ms = envelopes(policy)
    n_prompts = 7
    groups, flat, rows = [], [], []
    for p in rng.permutation(n_prompts).tolist():  # pools interleaved in drafting order
        groups += [f"g{p}"] * int(rng.integers(2, 6))
    order = rng.permutation(len(groups))
    groups = [groups[i] for i in sorted(order)]
    slots = {g: int(rng.integers(1, 4)) for g in set(groups)}
    for i, g in enumerate(groups):
        prompt = torch.tensor([100 + int(g[1:])])
        draft = [1] * int(rng.integers(1, 9))
        flat.append((i, (prompt, -1, draft, True, f"h{i}", None)))
        rows.append(torch.cat([prompt, torch.tensor(draft)]))
    grid = {
        i: {
            "null_grid": {M: (int(rng.integers(0, 3)), 0 if rng.random() < 1 / M else NULL_TOKEN, -1.0) for M in ms},
            "token_logprobs": [-1.0] * 3,
            "draft_logq": [-1.0] * 3,
        }
        for i in range(len(groups))
    }
    ref = _stub_spec(policy, grid)
    want = ref._verify_null(flat, rows, groups, slots)
    # Streamed: random past risk, small chunks, any window.
    s = _stub_spec(policy, grid)
    captured = {}
    s._finish_verify_null = lambda flat_, out, *a: captured.update(out=out, published=a[-1])
    risk = Verify0Risk()
    risk.update([(int(rng.integers(1, 9)), int(rng.integers(0, 9))) for _ in range(20)])
    names = {}
    v0 = dict(
        trace=spec_mod.TailTrace(),
        originals=None,
        orphans=[],
        drafts=None,
        flat=flat,
        rows=rows,
        keys=[spec_mod._key(d[0]) for _, d in flat],
        by_key={},
        groups=groups,
        slots=slots,
        null=True,
        dropped=0,
        n_empty=0,
        window=window,
        risk=risk,
    )
    for i, (_, d) in enumerate(flat):
        v0["by_key"].setdefault(spec_mod._key(d[0]), []).append(i)
    pcs = [
        PromptCounts(key=k, first=idx[0], lengths=tuple(len(flat[i][1][2]) for i in idx), tokens=sum(int(rows[i].numel()) for i in idx))
        for k, idx in v0["by_key"].items()
    ]
    v0["schedule"] = schedule_chunks(pcs, risk, int(rng.integers(1, 12)), int(rng.integers(1, 12)))
    s._plan_cv = asyncio.Condition()
    s._pending = defaultdict(int)
    s._tail_plans, s._tail_counts = set(), {}
    decided = []
    asyncio.run(s.run_stream(v0, on_decided=lambda info: decided.append(sorted(info))))
    out = captured["out"]
    assert [out[i].get("null_m") for i in range(len(groups))] == [want[i].get("null_m") for i in range(len(groups))]
    assert [out[i].get("null") for i in range(len(groups))] == [want[i].get("null") for i in range(len(groups))]
    # Every prompt decided exactly once; plans published in drafting order per prompt.
    assert sorted(k for d in decided for k in d) == sorted(v0["by_key"])
    for k, idx in v0["by_key"].items():
        outs = [i for i in idx if out[i].get("null") == "out"]
        assert [p.verified for p in s.plans[k]] == [(len(flat[i][1][2]), out[i]["accepted"]) for i in outs]


def test_streamed_call_composition_is_independent_of_result_timing():
    """Which rows share a call follows the schedule and the decisions only."""
    policy = NullPolicy(fixed_m=1.25)
    ms = envelopes(policy)
    groups = [f"g{p}" for p in range(4) for _ in range(6)]
    slots = {f"g{p}": 2 for p in range(4)}
    flat, rows = [], []
    for i, g in enumerate(groups):
        prompt = torch.tensor([100 + int(g[1:])])
        flat.append((i, (prompt, -1, [1, 1], True, f"h{i}", None)))
        rows.append(torch.cat([prompt, torch.tensor([1, 1])]))
    rng = np.random.default_rng(3)
    grid = {
        i: {"null_grid": {M: (2, 0 if rng.random() < 0.4 else NULL_TOKEN, -1.0) for M in ms}, "token_logprobs": [-1.0] * 2, "draft_logq": [-1.0] * 2}
        for i in range(len(groups))
    }
    seen = []
    for trial in range(4):
        random.seed(trial)
        s = _stub_spec(policy, grid)
        s._finish_verify_null = lambda *a: None
        calls = []
        inner = s._verify_block_async

        async def record(flat_, rows_, **kw):
            calls.append([i for i, _ in flat_])
            return await inner(flat_, rows_, **kw)

        s._verify_block_async = record
        by_key = defaultdict(list)
        for i, (_, d) in enumerate(flat):
            by_key[spec_mod._key(d[0])].append(i)
        v0 = dict(
            trace=spec_mod.TailTrace(), originals=None, orphans=[], drafts=None, flat=flat, rows=rows,
            keys=None, by_key=dict(by_key), groups=groups, slots=slots, null=True, dropped=0, n_empty=0,
            window=2, risk=Verify0Risk(), schedule=[[k] for k in by_key],
        )
        s._plan_cv, s._pending, s._tail_plans, s._tail_counts = asyncio.Condition(), defaultdict(int), set(), {}
        asyncio.run(s.run_stream(v0))
        seen.append(calls)
    assert all(c == seen[0] for c in seen)
    assert len(seen[0]) > 4  # follow-up waves rode in later calls


def test_tail_feed():
    f = TailFeed()
    assert f.open
    f.put([("k", 1)])
    f.close()
    assert f.open and f.take() == [("k", 1)] and not f.open
    assert f.admitted == [("k", 1)]
    with pytest.raises(RuntimeError):
        f.put([("k", 2)])


# --------------------------------------------------------------------------
# Driver end to end on the fake engine: streamed == one-shot plans
# --------------------------------------------------------------------------


def _engine_and_spec(make_spec, engine_cls, rounds, outcome):
    from test_sticky_driver import EOD  # noqa: F401

    engine = engine_cls(rounds)

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, anchors, **_):
        out = []
        for i, r in enumerate(rows):
            a, y = outcome[tuple(r.tolist())]
            out.append(
                (i, {"accepted": a, "next": y, "logprobs": [-1.0 - 0.1 * j for j in range(a + 1)], "draft_logq": engine_logq[i][:a]})
            )
        return out

    handlers = {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1}
    handlers["ordered_call"] = lambda seq, method, **kw: handlers[method](**kw)
    s = make_spec(handlers, base_handlers=engine.handlers(), verify_source="engine", hidden_transport="driver", sticky_delta=0.03)
    wg = s.learner_policy.worker_group

    def run(name, **kw):  # the fake's ``method`` parameter clashes with ordered_call's
        wg.calls.append((name, kw))
        return [handlers[name](**kw)]

    wg.run_all_workers_single_data = run
    return engine, s


@pytest.mark.parametrize("window", [1, 2])
def test_streamed_sticky_plans_equal_verify_sticky(window, monkeypatch):
    import itertools

    from test_speculative_driver import _spec as make_spec

    # Ordered learner calls are numbered by a process-wide counter: use a private one.
    monkeypatch.setattr(spec_mod, "_LEARNER_SEQ", itertools.count())
    from test_sticky_driver import EOD, _Engine

    def rounds():
        return [
            [
                ([4, 5, 6], False, [-0.1] * 3, 4, [10, EOD]),
                ([7, EOD], True, [-0.2] * 2, 2),
                ([5], False, [-0.3], 0),
            ],
            [([11, EOD], True, [-0.5] * 2, 3)],
        ]

    # Outcomes keyed by the verified row (content), so call grouping cannot matter.
    outcome = {
        (1, 2, 4, 5, 6): (3, 99),
        (3, 7, EOD): (0, 9),
        (1, 2, 4, 5, 6, 10, EOD): (2, 99),
        (3, 9, 11, EOD): (1, EOD),
    }
    prompts = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([8])]
    results = []
    for stream in (False, True):
        engine, s = _engine_and_spec(make_spec, _Engine, rounds(), outcome)
        s.current_step = 0
        drafts = asyncio.run(s.draft_streaming(prompts))()
        s.current_step = 1
        if not stream:
            asyncio.run(s.verify_sticky(drafts))
        else:
            decided = []
            v0 = s.prepare_stream(drafts, Verify0Risk(), 1, 1, window)
            assert v0["schedule"] == [[(1, 2)], [(3,)]]  # longest draft first

            class _Bag(set):  # the fake's results are lists, not object refs
                def update(self, *a):
                    pass

                def difference_update(self, *a):
                    pass

            async def go():
                s.begin_streaming_tails()
                s._learner_inflight = _Bag()
                try:
                    await s.complete_stream(v0, lambda info: decided.append(dict(info)))
                finally:
                    s.end_streaming_tails()

            asyncio.run(go())
            # (1, 2): whole-accepted, unfinished -> a tail; (3,): rejected -> a tail.
            assert decided == [{(1, 2): (1, 1)}, {(3,): (1, 1)}]
            assert s.stats["v0s/calls"] == 2.0 and s.stats["v0s/chunks"] == 2.0
        plans = {k: [(p.prefix, p.complete, p.anchors.length) for p in v] for k, v in s.plans.items()}
        results.append((plans, len(engine.started), s.stats["spec/sticky_rounds"] > 0, s.stats["spec/full_accept"]))
    assert results[0] == results[1]
    assert results[1][0][(1, 2)] == [([4, 5, 6, 10, EOD], True, 7)]
    assert results[1][0][(3,)] == [([9, 11, EOD], True, 3)]


def test_rollout_waits_for_its_prompts_decision():
    """A drafted prompt's rollout takes its plan only once its call is decided;
    a prompt never drafted takes a fresh plan at once (and waits for theta_k)."""
    s = SpeculativeGeneration.__new__(SpeculativeGeneration)
    s.plans = defaultdict(list)
    s._pending = defaultdict(int)
    s._pending[(1,)] = 1
    s._plan_cv = asyncio.Condition()
    s._verify_error = None

    async def go():
        got = []

        async def take(key):
            async with s._plan_cv:
                while not s.plans.get(key) and s._pending.get(key, 0) > 0:
                    await s._plan_cv.wait()
            got.append((key, bool(s.plans.get(key))))

        t1 = asyncio.create_task(take((1,)))
        t2 = asyncio.create_task(take((2,)))
        await asyncio.sleep(0.01)
        assert got == [((2,), False)]
        s.plans[(1,)].append(Plan(-1, [5], complete=True))
        async with s._plan_cv:
            s._pending[(1,)] = 0
            s._plan_cv.notify_all()
        await asyncio.gather(t1, t2)
        return got

    assert asyncio.run(go()) == [((2,), False), ((1,), True)]


def _master(**ts):
    from test_sticky_driver import _master as m

    return m(**ts)


def test_verify0_stream_config_validation():
    from thundersync_rl.grpo_loop import validate_verify0_stream

    ok = _master(verify0_stream=True)
    validate_verify0_stream(ok)
    # Lanes: allowed again (forced-fresh rows now refit the cohort lane; v0s-xlanes).
    validate_verify0_stream(_master(verify0_stream=True, cohort_lanes=2))
    for bad, match in (
        (dict(verify0_stream=True, null_bv_prompt_drop=True), "null_bv_prompt_drop"),
        (dict(verify0_stream=True, tail_excluded_baseline=True), "tail_excluded_baseline"),
        (dict(verify0_stream=True, verify0_stream_window=0), "window"),
        (dict(verify0_stream=True, stream_training_over_tails=False), "stream_training_over_tails"),
    ):
        with pytest.raises(ValueError, match=match):
            validate_verify0_stream(_master(**bad))
