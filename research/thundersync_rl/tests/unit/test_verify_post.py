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
"""Verify host time (PERF_RECONCILE_1102 item 1): one ⊥-BV wave per pool and chunk calls posted ahead.

- ``_NullPoolRun(one_wave=True)`` forwards every draft of the pool in one wave; with fixed per-row outcomes the rows
  the count process takes are the lazy walk's (any multiplier), in drafting order.
- ``verify_post`` / ``verify_collect``: chunks posted together, collected in order, give the plans (here: the rows,
  envelopes and grid results the finish sees) of sequential ``verify`` calls bitwise, and every post reaches the
  learner before the first collect waits.
"""

import random

import pytest
import torch

from thundersync_rl import speculative as sp
from thundersync_rl.null_bv import NullPolicy
from thundersync_rl.speculative import SpeculativeGeneration, _NullPoolRun


def _walk(run, outcome):
    took, nf, waves = [], 0, 0
    while True:
        batch = run.next_wave()
        if not batch:
            break
        waves += 1
        nf += len(batch)
        for i in batch:
            m = run.state.next_m()
            grid = {e: (*outcome(i, e), -1.0) for e in run.env[i]}
            run.decide(i, {"null_grid": grid, "token_logprobs": [-1.0] * 10, "draft_logq": [-1.0] * 10}, False, 10)
            if m is not None and grid[m][1] >= 0:
                took.append(i)
    run.close()
    return took, nf, waves


@pytest.mark.parametrize("seed", range(80))
def test_one_wave_takes_the_lazy_walks_rows(monkeypatch, seed):
    rng = random.Random(77 + seed)
    policy = NullPolicy(fixed_m=rng.choice([1.0, 1.25, 1.5, 2.0]))
    n, slots = rng.randint(1, 16), rng.randint(1, 8)
    u = {i: rng.random() for i in range(n)}
    outcome = lambda i, m: (5, 1) if u[i] < 1.0 / m else (0, -1)  # noqa: E731
    monkeypatch.setenv("THUNDERSYNC_NULL_WAVE_MULT", "1")
    lazy = _walk(_NullPoolRun(list(range(n)), slots, policy), outcome)
    one = _walk(_NullPoolRun(list(range(n)), slots, policy, one_wave=True), outcome)
    assert one[0] == lazy[0]  # the same rows taken, in drafting order
    assert one[2] == 1 and one[1] == n  # one wave, every draft forwarded


def _draft(key, n, prompt):
    return (torch.tensor([1, 2, prompt]), -1, list(range(1, n + 1)), True, f"k{key}", [(0, 0, n)])


def _spec(monkeypatch, log):
    s = object.__new__(SpeculativeGeneration)
    s.fused_verify_train = s.verify_in_train = False
    s.verify_on, s.null_bv, s.null_bv_drop_is = "learner", True, False
    s.current_step, s._null_slots, s.stats, s.plans = 5, {}, {}, {}
    s._null_policy = NullPolicy(fixed_m=1.25)
    s.null_one_wave = True
    s._engine_q_filter = lambda drafts: (drafts, [])
    s._record_siblings = lambda flat: None
    s._release_hidden = lambda drafts: None
    s._v0_kw = s._v0_cap_kw = lambda: {}
    s._v0_settle = s._vit_place = s._count_tails = lambda *a, **k: None

    def grid(d, envs):  # a deterministic result per row (its key) and envelope
        h = sum(map(ord, d[4]))
        return {"null_grid": {e: ((h % 5), (h % 3) - 1 if e > 1.0 else 1, -1.0) for e in envs},
                "token_logprobs": [-1.0] * 4, "draft_logq": [-1.0] * 4}

    def submit(flat, rows, null_envelopes=None, **kw):
        log.append(("submit", [d[4] for _, d in flat]))
        return [(0, ([tuple(e) for e in null_envelopes], [d for _, d in flat]))]  # [(first row, refs)]

    def results(flat, got):
        out = {}
        for a, (envs, ds) in got:
            for j, (d, e) in enumerate(zip(ds, envs)):
                out[a + j] = grid(d, e)
        return out

    def block(flat, rows, null_envelopes=None, **kw):
        log.append(("block", [d[4] for _, d in flat]))
        return {j: grid(d, tuple(e)) for j, ((_, d), e) in enumerate(zip(flat, null_envelopes))}

    def finish(flat, by_row, groups, slots, dropped, n_empty):
        log.append(("finish", [(d[4], by_row[r].get("null"), by_row[r].get("null_m")) for r, (_, d) in enumerate(flat)]))

    s._submit_verify_block, s._verify_block_results, s._verify_block = submit, results, block
    s._finish_verify_null = finish
    monkeypatch.setattr(sp.ray, "get", lambda refs: (log.append(("get", len(refs[1]))), refs)[1])
    return s


def _chunks():
    rng = random.Random(3)
    out, key = [], 0
    for _c in range(3):
        pools = []
        for _p in range(4):
            prompt = 100 + key  # one prompt per pool (a pool is every draft of one prompt)
            pools.append([_draft(key + j, rng.randint(2, 6), prompt) for j in range(rng.randint(1, 4))])
            key += 10
        out.append(pools)
    return out


def test_post_collect_equals_sequential_verify(monkeypatch):
    chunks = _chunks()
    slots = lambda pools: {tuple(p[0][0].tolist()): 2 for p in pools}  # noqa: E731
    seq_log = []
    a = _spec(monkeypatch, seq_log)
    for pools in chunks:
        a._null_slots[a.current_step] = slots(pools)
        a.verify(pools)
    post_log = []
    b = _spec(monkeypatch, post_log)
    hs = []
    for pools in chunks:
        b._null_slots[b.current_step] = slots(pools)
        hs.append(b.verify_post(pools))
    for h in hs:
        b.verify_collect(h)
    fin = lambda log: [x for x in log if x[0] == "finish"]  # noqa: E731
    assert fin(post_log) == fin(seq_log) and len(fin(seq_log)) == 3
    # posted ahead: every chunk's call is submitted before the first result is awaited
    kinds = [x[0] for x in post_log]
    assert kinds[:3] == ["submit"] * 3 and "get" not in kinds[:3]


def test_posted_replies_come_from_the_mailbox_and_rank_errors_surface(monkeypatch):
    """Posted calls return through the ReplyMailbox (the learner actors are async Ray actors; a synchronous call's
    reply waited behind later posted calls' bodies, 9 s at Super). await_posted_replies returns the mailbox's
    per-rank results; a rank whose call raised surfaces its error instead of waiting forever."""
    from thundersync_rl import speculative as sp

    class Take:  # the mailbox's take() ref: ready after `after` waits
        def __init__(self, after, value):
            self.after, self.value = after, value

    class Mailbox:
        def __init__(self, take):
            self.take = type("M", (), {"remote": lambda _s, key, n: take})()

    class Boom(Exception):
        pass

    def fake_wait(items, num_returns=1):
        take = items[0]
        if take.after <= 0:
            return [take], items[1:]
        take.after -= 1
        return [items[1]], [take] + items[2:]

    def fake_get(x):
        if isinstance(x, Take):
            return x.value
        if isinstance(x, list):
            return [f"result-{r}" for r in x]
        if x == "bad":
            raise Boom("rank 1 failed")
        return None

    monkeypatch.setattr(sp.ray, "wait", fake_wait)
    monkeypatch.setattr(sp.ray, "get", fake_get)
    ok = Take(1, ["r0", "r1"])  # one rank's call ref finishes (None) before the mailbox has both posts
    assert sp.await_posted_replies(["c0", "c1"], Mailbox(ok), "k") == ["result-r0", "result-r1"]
    bad = Take(5, None)  # rank 1 raised: its post never comes
    with pytest.raises(Boom):
        sp.await_posted_replies(["bad", "c1"], Mailbox(bad), "k")
