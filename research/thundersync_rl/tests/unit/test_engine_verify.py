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
"""Engine-side verification (verify_on=engine, DESIGN 14ac): CPU pieces."""

import asyncio
from collections import defaultdict

import numpy as np
import pytest
import torch
from thundersync_rl.engine_q import verify_engine_q
from thundersync_rl.run_state import SeedStream
from thundersync_rl.engine_verify import (
    CaptureStore,
    HeadLogprobRows,
    HiddenRecompute,
    NeedQ,
    capture_selection,
    row_seed,
    verify_rows_two_pass,
)


def test_capture_store_positions_and_completeness():
    st = CaptureStore()
    st.begin("a", 4, 8, 3, torch.float32, "cpu")  # positions 4..7
    rows = torch.arange(12, dtype=torch.float32).view(4, 3)
    st.put("a", np.array([2, 3, 4, 5]), rows)  # 2, 3 outside: dropped
    assert not st.complete("a")
    st.put("a", np.array([6, 7]), rows[:2] + 100)
    assert st.complete("a")
    out = st.pop("a")
    assert torch.equal(out[0], rows[2]) and torch.equal(out[1], rows[3])
    assert torch.equal(out[2], rows[0] + 100)
    assert st.pop("a") is None


def test_capture_selection_keeps_tagged_window_only():
    # Slots: 0 untagged, 1 key 0 (lo 2, hi 5), 2 key 1 (lo 0, hi 1).
    slot_kid = np.array([-1, 0, 1])
    lo, hi = np.array([2, 0]), np.array([5, 1])
    slots = np.array([0, 1, 1, 1, 1, 1, 2, 2])
    pos = np.array([9, 0, 1, 2, 3, 4, 0, 1])
    sel, kid, p = capture_selection(slots, pos, slot_kid, lo, hi)
    assert sel.tolist() == [3, 4, 5, 6]
    assert kid.tolist() == [0, 0, 0, 1]
    assert p.tolist() == [2, 3, 4, 0]


def test_head_logprob_rows_chunked_equals_full():
    torch.manual_seed(0)
    h = torch.randn(37, 16)
    w = torch.randn(50, 16)
    head = lambda x: x @ w.t()  # noqa: E731
    rows = HeadLogprobRows(h, head, 50, chunk_rows=8)
    full = torch.log_softmax(h @ w.t(), -1)
    toks = torch.randint(0, 50, (36,))
    got = rows.token_logprobs(toks)
    ref = full[torch.arange(36), toks]
    assert torch.allclose(got, ref, atol=1e-6)
    idx = torch.tensor([3, 30])
    assert torch.allclose(rows(idx), full[idx], atol=1e-6)


def test_two_pass_reproduces_draws_and_decisions():
    """A row decided after a q recompute uses the same random stream."""
    torch.manual_seed(1)
    V, g = 20, 12
    q_full = torch.log_softmax(torch.randn(g + 1, V), -1)
    p_full = torch.log_softmax(torch.randn(g + 1, V) * 2, -1)
    draft = torch.multinomial(q_full[:g].exp(), 1)[:, 0]
    lq = q_full[torch.arange(g), draft]

    def decide(it, q_rows):
        gen = torch.Generator().manual_seed(row_seed(7, it["key"]))
        src = q_rows if q_rows is not None else it["own"]
        tau, y, _, _ = verify_engine_q(p_full, draft, lq, src, gen)
        return {"accepted": tau, "next": y}

    def raise_need(idx):
        raise NeedQ()

    items = [{"key": "a", "own": raise_need}]
    calls = []

    def recompute(need):
        calls.append([it["key"] for it in need])
        return {it["key"]: (lambda idx: q_full[idx]) for it in need}

    out, n = verify_rows_two_pass(items, decide, recompute)
    direct = decide({"key": "a", "own": lambda idx: q_full[idx]}, None)
    assert out == [("a", direct)]
    assert n == (1 if calls else 0)


def test_row_seed_is_independent_of_order():
    assert row_seed(3, "x") == row_seed(3, "x")
    assert row_seed(3, "x") != row_seed(4, "x")
    assert row_seed(3, "x") != row_seed(3, "y")


def test_hidden_recompute_counts_rows():
    from thundersync_rl.engine_q import hidden_nbytes, hidden_num_rows

    h = HiddenRecompute(17)
    assert hidden_num_rows(h) == 17 and hidden_nbytes(h) == 0


def _planner(groups, G=2):
    from thundersync_rl.streaming import StreamPlanner

    pl = StreamPlanner(
        lambda rewards: [r - sum(rewards) / len(rewards) for r in rewards],
        dp_size=1,
        max_chunk_trajectories=8,
        max_open_groups=0,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    for g in groups:
        pl.register_group(g, G)
    return pl


def test_incremental_order_dispatches_only_appended_blocks():
    from thundersync_rl.streaming import Trajectory

    groups = [(0, p) for p in range(3)]
    pl = _planner(groups)
    pl.begin_order()
    for g in groups:
        for i in range(2):
            pl.add(Trajectory(group=g, index=i, reward=float(i), payload=None, num_tokens=4))
    assert not pl.has_work()  # nothing ordered yet
    pl.extend_order([(0, 2)], [1])
    d = pl.next_dispatch()
    assert {t.group for cs in d.per_rank for c in cs for t in c.trajectories} == {(0, 2)}
    assert not pl.has_work() and not pl.order_complete()
    pl.extend_order([(0, 0), (0, 1)], [2])
    assert pl.order_complete()
    d = pl.next_dispatch()
    assert {t.group for cs in d.per_rank for c in cs for t in c.trajectories} == {(0, 0), (0, 1)}
    with pytest.raises(ValueError):
        pl.extend_order([(0, 1)], [1])


def test_classify_groups_and_feed():
    from thundersync_rl.grpo_loop import OrderFeed, block_chunks, classify_groups

    G = 2
    row_keys = [(1,), (1,), (2,), (2,), (3,), (3,)]
    info = {(1,): (2, True), (2,): (2, False), (3,): (1, True)}
    cls = classify_groups(row_keys, G, [0, 1, 2], info)
    assert cls == {0: "ready", 1: "tail", 2: "fresh"}
    assert block_chunks([5, 6, 7], 2) == [2, 1]
    pl = _planner([(4, p) for p in range(3)])
    feed = OrderFeed(4)
    feed.extend([1], [1])  # before attach: buffered
    feed.attach(pl)
    feed.extend([0, 2], [2])
    assert pl.order == [(4, 1), (4, 0), (4, 2)] and pl.order_complete()


def test_verify_sticky_blocks_publishes_per_block(monkeypatch):
    """Rollouts of block b are served after block b's verification, not after all."""
    from thundersync_rl.speculative import Plan, SpeculativeGeneration

    spec = SpeculativeGeneration.__new__(SpeculativeGeneration)
    spec.stats = {}
    spec.plans = defaultdict(list)
    spec.null_bv = False
    spec.verify_source = "engine"
    spec.sticky_delta = 0.03
    spec.tail_lookup = False
    spec.banded_is_band = None
    spec._null_slots = {}
    spec.current_step = 1
    spec._tail_cv = None
    spec._row_claims = None
    order = []

    def mk(key, n):
        return (torch.tensor(key), -1, [7] * n, True, f"k{key}", [(0, 0, n)], [0.0] * n,
                HiddenRecompute(n), object(), None)

    drafts = [[mk((1,), 3)], [mk((2,), 3)]]

    def fake_verify_block(flat, rows, envs=None, band=None):
        order.append(("verify", [tuple(d[0].tolist()) for _, d in flat]))
        return {j: {"accepted": len(d[2]), "next": -1, "logprobs": [0.0] * len(d[2]),
                    "draft_logq": [0.0] * len(d[2])} for j, (_, d) in enumerate(flat)}

    def fake_plan(d, r):
        return Plan(-1, list(d[2]), [0.0] * len(d[2]), complete=True)

    async def fake_continue(pending, trace=None, ended=(), incoming=None):
        while True:
            x = await incoming.get()
            if x is None:
                return

    spec._verify_block = fake_verify_block
    spec._plan = fake_plan
    spec._plan_tokens = fake_plan
    spec._continue_sticky = fake_continue
    spec._engine_q_filter = lambda d: (d, 0)
    spec._stop_streams = lambda s: None
    spec._record_siblings = lambda flat: None
    spec._count_ages = lambda *a: None
    spec._age_stats = lambda *a: None
    spec._band_stats = lambda: None
    spec._longest_stats = lambda *a: None

    async def main():
        spec.prepare_verify_blocks([[(1,)], [(2,)]])
        seen = []

        def on_block(b, info):
            seen.append((b, dict(info)))

        async def waiter(key):
            async with spec._plan_cv:
                while not spec.plans.get(key) and spec._pending.get(key, 0) > 0:
                    await spec._plan_cv.wait()
            order.append(("served", key))

        tasks = [asyncio.create_task(waiter((2,))), asyncio.create_task(waiter((1,)))]
        await spec.verify_sticky_blocks(drafts, [[(1,)], [(2,)]], on_block)
        await asyncio.gather(*tasks)
        return seen

    seen = asyncio.run(main())
    assert [b for b, _ in seen] == [0, 1]
    assert seen[0][1] == {(1,): (1, True)}
    assert order.index(("verify", [(1,)])) < order.index(("verify", [(2,)]))
    assert ("served", (1,)) in order


def test_batched_token_logprobs_match_per_row():
    from thundersync_rl.engine_verify import batched_token_logprobs

    torch.manual_seed(2)
    w = torch.randn(40, 8)
    head = lambda x: x @ w.t()  # noqa: E731
    hs = [torch.randn(n + 1, 8) for n in (5, 0, 9)]
    toks = [torch.randint(0, 40, (n,)) for n in (5, 0, 9)]
    got = batched_token_logprobs(hs, toks, head, 40, chunk_rows=4)
    for h, t, g in zip(hs, toks, got):
        ref = HeadLogprobRows(h, head, 40, chunk_rows=3).token_logprobs(t)
        assert g.shape == t.shape and torch.allclose(g, ref, atol=1e-6)


def _rowwise_head(W):
    # One GEMM per row: row-invariant by construction (the engine's batch-invariant
    # head GEMM is; plain CPU/GPU GEMMs need not be).
    def head(h):
        return torch.cat([h[i : i + 1].float() @ W.t() for i in range(h.shape[0])])

    return head


@pytest.mark.parametrize(
    "dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_decide_whole_rows_matches_decide_row(dev):
    """Batched whole-draft decisions == the per-row decision, bitwise; the rest is left."""
    from thundersync_rl.engine_q import gathered_log_softmax, no_gather
    from thundersync_rl.engine_verify import batched_token_logprobs, decide_row, decide_whole_rows

    torch.manual_seed(3)
    d, V = 16, 64
    W = torch.randn(V, d, device=dev)
    Wq = W + 0.05 * torch.randn(V, d, device=dev)
    p_head, q_head = _rowwise_head(W), _rowwise_head(Wq)
    specs = []
    for i in range(48):
        g = [1, 3, 9, 20, 33][i % 5]
        env = [None, (1.0,), (1.0, 1.3), (1.0, 1.15, 1.5)][i % 4]
        band = 2.0 if i % 11 == 7 else None
        same_q = i % 3 == 0  # q == p: whole acceptance likely
        specs.append((f"r{i}", g, env, band, same_q))
    rows, ref = [], {}
    hid = {}
    for key, g, env, band, same_q in specs:
        h = torch.randn(g + 1, d, device=dev)
        lq_rows = gathered_log_softmax((p_head if same_q else q_head)(h[:g]), no_gather, V)
        draft = torch.multinomial(lq_rows.exp(), 1)[:, 0]
        lq = lq_rows.gather(-1, draft.view(-1, 1))[:, 0].contiguous()
        hid[key] = (h, draft, lq, lq_rows, env, band)
    lpx_all = dict(
        zip(
            hid,
            batched_token_logprobs(
                [v[0] for v in hid.values()], [v[1] for v in hid.values()], p_head, V, 7
            ),
        )
    )
    seed = 12345
    for key, (h, draft, lq, lq_rows, env, band) in hid.items():
        rows.append(
            {"key": key, "hidden": h, "draft": draft, "lpx": lpx_all[key], "lq": lq,
             "envelopes": env, "band": band}
        )
        p_lp = HeadLogprobRows(h, p_head, V, 5)
        p_lp.cached_token_logprobs = lpx_all[key]
        res, _, _ = decide_row(p_lp, draft, lq, lambda idx, r=lq_rows: r[idx], seed, key, env, band)
        ref[key] = res
    fast, slow = decide_whole_rows(rows, p_head, V, seed, chunk_rows=5)
    assert set(fast) | set(slow) == set(hid) and not set(fast) & set(slow)
    n_whole = 0
    for key, (h, draft, lq, lq_rows, env, band) in hid.items():
        g = draft.numel()
        r = ref[key]
        if env is not None:
            whole = all(t == g for t, _, _ in r["null_grid"].values())
        else:
            whole = r["accepted"] == g
        if band is not None:
            assert key in slow
            continue
        assert (key in fast) == whole, key
        if whole:
            n_whole += 1
            assert fast[key] == r, key
    assert n_whole >= 8 and len(slow) >= 8


@pytest.mark.parametrize("slots_a,slots_b,n", [(8, 8, 12), (8, 4, 12), (16, 16, 24), (2, 1, 3)])
def test_null_first_wave_is_verify_null_first_call(slots_a, slots_b, n):
    """The prefetched first wave is exactly the rows of _verify_null's first call."""
    from thundersync_rl.null_bv import NullPolicy
    from thundersync_rl.speculative import SpeculativeGeneration

    spec = SpeculativeGeneration.__new__(SpeculativeGeneration)
    spec.stats = {}
    spec._null_policy = NullPolicy()
    spec.null_bv_drop_is = False
    spec._release_hidden = lambda ds: None
    groups = ["a"] * n + ["b"] * n
    flat = [(i, ("draft", i)) for i in range(2 * n)]
    rows = [torch.zeros(3) for _ in flat]
    calls = []

    def fake_verify_block(sub_flat, sub_rows, null_envelopes=None, band=None):
        calls.append([e[0] for e in sub_flat])
        # Every second row is BOT at every envelope: more waves follow.
        return {
            j: {
                "null_grid": {m: (2, -1 if e[0] % 2 else 1, 0.0) for m in envs},
                "token_logprobs": [0.0, 0.0],
                "draft_logq": [0.0, 0.0],
            }
            for j, (e, envs) in enumerate(zip(sub_flat, null_envelopes))
        }

    spec._verify_block = fake_verify_block
    slots = {"a": slots_a, "b": slots_b}
    first = spec._null_first_wave(groups, slots)
    spec._verify_null(flat, rows, groups, slots)
    assert calls and calls[0] == first


def test_verify_sticky_blocks_prefetches_next_block(monkeypatch):
    """With engine_verify_prefetch, block b+1 is sent before block b is decided, in block order."""
    from thundersync_rl.speculative import Plan, SpeculativeGeneration

    spec = SpeculativeGeneration.__new__(SpeculativeGeneration)
    spec.stats = {}
    spec.plans = defaultdict(list)
    spec.null_bv = False
    spec.verify_source = "engine"
    spec.verify_on = "engine"
    spec.engine_verify_prefetch = True
    spec.sticky_delta = 0.03
    spec.tail_lookup = False
    spec.banded_is_band = None
    spec._null_slots = {}
    spec.current_step = 1
    spec._tail_cv = None
    spec._row_claims = None
    order = []

    def mk(key, n):
        return (torch.tensor(key), -1, [7] * n, True, f"k{key}", [(0, 0, n)], [0.0] * n,
                HiddenRecompute(n), object(), None)

    drafts = [[mk((1,), 3)], [mk((2,), 3)], [mk((3,), 3)]]

    def fake_verify_block(flat, rows, envs=None, band=None):
        order.append(("verify", [tuple(d[0].tolist()) for _, d in flat]))
        return {j: {"accepted": len(d[2]), "next": -1, "logprobs": [0.0] * len(d[2]),
                    "draft_logq": [0.0] * len(d[2])} for j, (_, d) in enumerate(flat)}

    def fake_prefetch(flat, rows):
        order.append(("prefetch", [tuple(d[0].tolist()) for _, d in flat]))

    def fake_plan(d, r):
        return Plan(-1, list(d[2]), [0.0] * len(d[2]), complete=True)

    async def fake_continue(pending, trace=None, ended=(), incoming=None):
        while True:
            x = await incoming.get()
            if x is None:
                return

    spec._verify_block = fake_verify_block
    spec._ev_prefetch = fake_prefetch
    spec._ev_drop_prefetch = lambda: order.append(("drop",))
    spec._plan = fake_plan
    spec._plan_tokens = fake_plan
    spec._continue_sticky = fake_continue
    spec._engine_q_filter = lambda d: (d, 0)
    spec._stop_streams = lambda s: None
    spec._record_siblings = lambda flat: None
    spec._count_ages = lambda *a: None
    spec._age_stats = lambda *a: None
    spec._band_stats = lambda: None
    spec._longest_stats = lambda *a: None
    blocks = [[(1,)], [(2,)], [(3,)]]

    async def main():
        spec.prepare_verify_blocks(blocks)
        await spec.verify_sticky_blocks(drafts, blocks, lambda b, info: None)

    asyncio.run(main())
    assert order == [
        ("prefetch", [(1,)]),
        ("prefetch", [(2,)]),
        ("verify", [(1,)]),
        ("prefetch", [(3,)]),
        ("verify", [(2,)]),
        ("verify", [(3,)]),
        ("drop",),
    ]


def test_engine_verify_rows_takes_prefetch_and_drops_unused(monkeypatch):
    """A prefetched call is used by the same rows (one prefill, anchors once); others are dropped."""
    import itertools

    from thundersync_rl import speculative as sp
    from thundersync_rl.speculative import EngineVerifier, SpeculativeGeneration

    log = []

    class Call:
        def __init__(self, name, fn):
            self.name, self.fn = name, fn

        def remote(self, *a, **k):
            log.append(self.name)
            return self.fn(*a, **k)

    class Worker:
        def __init__(self):
            self.anchors = {}
            self.captured = set()
            self.run_verify_requests = Call("prefill", self._prefill)
            self.engine_verify_ready = Call("ready", lambda keys: [k for k in keys if k in self.captured])
            self.engine_verify_decide = Call("decide", self._decide)
            self.engine_verify_drop = Call("drop", lambda keys: self.captured.difference_update(keys))
            self.add_engine_anchors = Call("anchors+", lambda e: self.anchors.update(e))
            self.drop_engine_anchors = Call(
                "anchors-", lambda keys: [self.anchors.pop(k, None) for k in keys]
            )

        def _prefill(self, rows, keys, plens, akeys):
            assert all(isinstance(r, np.ndarray) for r in rows)
            self.captured.update(keys)
            return {"requests": float(len(rows))}

        def _decide(self, items, seed, vocab_limit, q_lane, check):
            out = []
            for it in items:
                assert it["tokens"].dtype == np.int64 and it["logq"].dtype == np.float32
                self.captured.discard(it["key"])
                out.append((it["key"], {"accepted": len(it["tokens"]), "env": it.get("envelopes")}))
            return out

    class Group:
        def __init__(self, w):
            self.workers = [w]

    monkeypatch.setattr(sp.ray, "get", lambda x: x)
    w = Worker()
    spec = SpeculativeGeneration.__new__(SpeculativeGeneration)
    spec.stats = {}
    spec.engine_verifier = EngineVerifier(Group(w))
    spec.current_step = 1
    spec._verify_seeds = SeedStream(42, "verify/0", lambda: spec.current_step)
    spec._ev_ids = itertools.count()
    spec.engine_max_len = None
    spec.sticky_delta = 0.03
    spec.vocab_limit = 10
    import threading

    spec._ev_lock = threading.Lock()

    def mk(i, n):
        d = (torch.tensor([1, 2]), -1, [3] * n, True, f"k{i}", [(0, 0, n)], [-0.5] * n,
             HiddenRecompute(n), ("anchors", i), None)
        return (i, d)

    flat_a = [mk(0, 3), mk(1, 4)]
    flat_b = [mk(2, 5)]
    rows_a = [torch.cat([d[0], torch.tensor(d[2])]) for _, d in flat_a]
    rows_b = [torch.cat([d[0], torch.tensor(d[2])]) for _, d in flat_b]
    spec._ev_prefetch(flat_a, rows_a)
    spec._ev_prefetch(flat_b, rows_b)
    assert log.count("prefill") == 2 and log.count("anchors+") == 2
    log.clear()
    out = spec._engine_verify_rows(flat_a, rows_a, null_envelopes=[(1.0,), (1.0, 1.3)])
    assert log.count("prefill") == 0 and log.count("anchors+") == 0  # taken from the prefetch
    assert log.count("anchors-") == 1
    assert out[0]["accepted"] == 3 and out[1]["accepted"] == 4
    assert out[1]["env"] == [1.0, 1.3]
    assert spec.stats["spec/engine_verify_prefetched_rows"] == 2.0
    # Rows that were not prefetched: a normal call (prefill sent now).
    log.clear()
    flat_c = [mk(3, 2)]
    spec._engine_verify_rows(flat_c, [torch.cat([flat_c[0][1][0], torch.tensor(flat_c[0][1][2])])])
    assert log.count("prefill") == 1
    # The unused prefetch (flat_b) is dropped with its anchors.
    log.clear()
    spec._ev_drop_prefetch()
    assert log == ["drop", "anchors-"]
    assert not w.captured and not w.anchors
    assert spec.stats["spec/engine_verify_prefetch_dropped"] == 1


def test_ev_lane_heads_orders_by_data_not_submission():
    """Lane head handoff on asyncio actors: every export sees its stash.

    The engine workers are asyncio actors (some methods are ``async``), which
    run calls in the order their arguments resolve, not the order they are
    sent. ``_ev_lane_heads`` must order stash -> export -> import by data
    dependencies; with submission order only, the export ran first, shipped
    None and the next step's verifier had no q head (ev4/ev5 crash at step 1).
    """
    import types

    import ray
    from thundersync_rl.grpo_loop import _ev_lane_heads

    @ray.remote
    class Head:
        def __init__(self):
            self.stash: dict = {}

        async def ping(self):  # makes this an asyncio actor, as the workers are
            return 0

        def _lm_head(self):
            return True

        def stash_head(self, version, keep):
            self.stash[version] = torch.full((2,), float(version))
            for v in list(self.stash):
                if v not in keep and v != version:
                    del self.stash[v]
            return len(self.stash)

        def head_stash_cpu(self, version, after=None):
            w = self.stash.get(version)
            if w is None:
                raise RuntimeError(f"no stash of {version} (have {sorted(self.stash)})")
            return w

        def import_head_stash(self, version, weight, keep, after=None):
            if weight is None:
                raise RuntimeError(f"no shard of {version}")
            self.stash[version] = weight
            for v in list(self.stash):
                if v not in keep and v != version:
                    del self.stash[v]
            return len(self.stash)

        def versions(self):
            return sorted(self.stash)

    def lane(workers):
        wg = types.SimpleNamespace(workers=workers)
        return types.SimpleNamespace(
            generation=types.SimpleNamespace(_policy=types.SimpleNamespace(worker_group=wg)),
            spec=types.SimpleNamespace(engine_verifier=None),
        )

    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        a = lane([Head.remote() for _ in range(2)])
        b = lane([Head.remote() for _ in range(2)])
        cfg = types.SimpleNamespace(hidden_transport="recompute")
        pending: list = []
        for step in range(12):
            # Cohort lane x verifies `step` on lane y (refit to theta_step).
            x, y = (a, b) if step % 2 == 0 else (b, a)
            _ev_lane_heads(x, y, step, pending, cfg)
            ev = x.spec.engine_verifier
            ray.get(list(ev.wait_refs))
            ys = y.generation._policy.worker_group.workers
            for v in ray.get([w.versions.remote() for w in ys]):
                # The verifier of cohort `step` holds theta_step and the drafter's head.
                assert step in v and (step == 0 or step - 1 in v), (step, v)
        ray.get(pending)
    finally:
        if started:
            ray.shutdown()
