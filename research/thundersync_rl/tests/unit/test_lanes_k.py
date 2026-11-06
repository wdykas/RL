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
"""Lanes-k bookkeeping (no torch, no Ray): ts depth k (la2.Lookahead2(depth=k) on k CohortLanes) and the async-k
schedule (async_lanes.AsyncSchedule / AsyncLedger).

- Depth 2 reproduces lookahead 2 exactly: the 056a8d3d Lookahead2 (copied verbatim below as the reference) and
  Lookahead2(depth=2) make the same decisions on the same driver simulation (random fresh rows, boundaries, adaptive
  switches), so the generalized class is LA2 bitwise in schedule.
- Depth k: every cohort drafts and completes on lane t % k at one version (t - k early, t - 1 after step 0 / a
  boundary), a pinned lane is never refit, at most k versions live, the heads kept cover every pending cohort.
- Async-k: version / lane / age per cohort (k = 1 is the async1 schedule: cohort t at theta_{t-1}, cohort 0 at
  theta_0), and an event simulation of the driver's rule (refit + next start as soon as the cohort is complete and the
  step has begun, between dispatches): ledger invariants, k cohorts in flight, steady-state ages, and no lane waits
  for anything but its own cohort, theta_j and one in-flight dispatch.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest
from thundersync_rl.async_lanes import AsyncLedger, AsyncSchedule
from thundersync_rl.la2 import Lookahead2
from thundersync_rl.lanes import CohortLanes, Lane

# ---------------------------------------------------------------------------------------------------------------
# Reference: Lookahead2 exactly as on the frozen final tree 056a8d3d (la2.py), before the depth generalization.


@dataclass
class _Lookahead2Ref:
    pending: dict[int, Any] = field(default_factory=dict)
    versions: dict[int, int] = field(default_factory=dict)
    starts: dict[int, float] = field(default_factory=dict)

    def take(self, target: int) -> Optional[Any]:
        self.starts.pop(target, None)
        return self.pending.pop(target, None)

    def may_start(self, step, draft_limit, have_batch, lane_fresh, lane_cohort) -> bool:
        target = step + 2
        return (
            target < draft_limit
            and have_batch
            and not lane_fresh
            and lane_cohort is None
            and target not in self.pending
        )

    def started(self, target: int, version: int, collector: Any, t: float) -> None:
        if target in self.pending:
            raise RuntimeError(f"lookahead 2: cohort {target} already drafting")
        if version != target - 2:
            raise RuntimeError(
                f"lookahead 2: cohort {target} must draft on theta_{target - 2}, got theta_{version}"
            )
        self.pending[target] = collector
        self.versions[target] = version
        self.starts[target] = t

    def head_keep(self, step: int) -> list[int]:
        return [step - 1, step]

    def leftover(self) -> list[int]:
        return sorted(self.pending)


# ---------------------------------------------------------------------------------------------------------------
# Driver simulation (the rules of grpo_loop's streaming lanes loop with Lookahead2(depth)), real CohortLanes pinning.


def _simulate(la, depth, n_steps, seed, boundaries=(), p_fresh=0.0, want=None, upcoming_len=None):
    """Returns (drafted: cohort -> (lane, version), trace of every decision, heads per step)."""
    rng = random.Random(seed)
    lanes = [Lane(i, None, version=("theta", 0)) for i in range(depth)]
    sched = CohortLanes(lanes)
    drafted, trace, heads_by_step = {}, [], {}
    heads = set()
    have = upcoming_len if upcoming_len is not None else depth

    def draft_limit(step):
        b = [x for x in boundaries if x >= step + 1]
        return min([n_steps] + b)

    for step in range(n_steps):
        x, y = sched.cohort_lane(step), sched.drafting_lane(step)
        live = {ln.version for ln in lanes}
        assert len(live) <= depth
        early = la.take(step + 1)
        if early is not None:
            ev = step + 1 - depth
            assert y.cohort == step + 1 and y.cohort_version == ("theta", ev)
            trace.append(("early", step + 1, early))
        elif step + 1 < draft_limit(step):
            sched.refit(y, ("theta", step), lambda: None)
            sched.pin(y, step + 1)
            drafted[step + 1] = (y.index, step)
            trace.append(("la1", step + 1, y.index, step))
        if step in drafted:
            lane_i, ver = drafted[step]
            assert x.index == lane_i == step % depth
            sched.check(x, step)
            assert x.cohort_version == ("theta", ver)
            assert ver in heads or ver == step  # q's head of the drafting version was kept
            sched.unpin(x, step)
        fresh = step not in drafted or rng.random() < p_fresh
        go = la.may_start(step, draft_limit(step), step + depth < n_steps and have >= depth, fresh, x.cohort)
        if want is not None:
            go = want(step) and go
        trace.append(("decide", step, bool(fresh), bool(go)))
        if fresh or go:
            sched.refit(x, ("theta", step), lambda: None)
        if go:
            sched.pin(x, step + depth)
            la.started(step + depth, step, f"e{step + depth}", float(step))
            drafted[step + depth] = (x.index, step)
        heads = set(la.head_keep(step))
        heads_by_step[step] = sorted(heads)
        for t in range(step + 1, step + depth + 1):
            if t in drafted:
                assert drafted[t][1] in heads, (step, t, drafted[t], heads)
        if (step + 1) in boundaries:
            assert all(ln.cohort is None or ln.cohort == step + 1 for ln in lanes)
            assert la.leftover() == []
    return drafted, trace, heads_by_step


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("p_fresh", [0.0, 0.3])
@pytest.mark.parametrize("boundaries", [(), (7, 15), (3, 4, 11)])
def test_depth2_is_la2_bitwise_in_schedule(seed, p_fresh, boundaries):
    a = _simulate(_Lookahead2Ref(), 2, 24, seed, boundaries, p_fresh)
    b = _simulate(Lookahead2(depth=2), 2, 24, seed, boundaries, p_fresh)
    assert a == b
    assert Lookahead2() == Lookahead2(depth=2)  # the default is depth 2


@pytest.mark.parametrize("seed", range(4))
def test_depth2_is_la2_under_adaptive_switches(seed):
    rng = random.Random(100 + seed)
    modes = [rng.random() < 0.6 for _ in range(30)]
    want = lambda s: modes[s]  # noqa: E731  (AdaptiveLookahead.decide outcome per step)
    a = _simulate(_Lookahead2Ref(), 2, 30, seed, (12,), 0.2, want)
    b = _simulate(Lookahead2(depth=2), 2, 30, seed, (12,), 0.2, want)
    assert a == b


def test_depth2_api_matches_reference():
    ref, new = _Lookahead2Ref(), Lookahead2()
    for step in range(6):
        assert ref.head_keep(step) == new.head_keep(step)
    for args in [(3, 10, True, False, None), (3, 5, True, False, None), (3, 10, False, False, None),
                 (3, 10, True, True, None), (3, 10, True, False, 3)]:
        assert ref.may_start(*args) == new.may_start(*args)
    with pytest.raises(RuntimeError, match="must draft on theta_4"):
        new.started(6, 3, "c", 0.0)
    assert new.keep_previous() == 1
    with pytest.raises(ValueError, match="depth must be >= 2"):
        Lookahead2(depth=1)


@pytest.mark.parametrize("depth", [3, 4])
@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("p_fresh", [0.0, 0.25])
def test_depth_k_schedule_invariants(depth, seed, p_fresh):
    n = 30
    bnd = (11, 20)
    drafted, _, heads = _simulate(Lookahead2(depth=depth), depth, n, seed, bnd, p_fresh)
    lags = {c: c - v for c, (_, v) in drafted.items()}
    assert set(lags.values()) <= {1, depth}
    assert all(lane == c % depth for c, (lane, _) in drafted.items())
    if p_fresh == 0.0:
        # lag 1 only where no early start was possible: the step c - depth had no drafted cohort (step 0 decodes
        # fresh rows) or did not exist, or a boundary lies in (c - depth, c]
        for c, lag in lags.items():
            if lag == 1:
                assert (c - depth) not in drafted or c - depth < 0 or any(c - depth < b <= c for b in bnd), (c, lags)
        # steady state: every cohort past the fill and away from boundaries has age depth (cohort b after a boundary
        # decodes fresh rows like cohort 0, and b + 1 .. b + depth fall back to lookahead 1)
        steady = [c for c in lags if c > depth and not any(b <= c <= b + depth for b in bnd)]
        assert steady and all(lags[c] == depth for c in steady), lags
    assert all(len(h) == depth for h in heads.values())


def test_depth_k_head_keep_and_versions():
    la = Lookahead2(depth=4)
    assert la.head_keep(10) == [7, 8, 9, 10] and la.keep_previous() == 3
    assert la.may_start(3, 10, True, False, None)  # target 7
    assert not la.may_start(3, 7, True, False, None)  # 7 would cross the boundary at 7
    la.started(7, 3, "c7", 0.0)
    with pytest.raises(RuntimeError, match="lookahead 4: cohort 8 must draft on theta_4"):
        la.started(8, 3, "c", 0.0)
    assert la.take(7) == "c7" and la.leftover() == []


def test_depth_k_needs_k_batches_ahead():
    # with only 2 batches ahead the cohort lane cannot start cohort s + 3 (grpo_loop keeps max(2, depth) ahead)
    drafted, _, _ = _simulate(Lookahead2(depth=3), 3, 12, 0, (), 0.0, upcoming_len=2)
    assert all(c - v == 1 for c, (_, v) in drafted.items())


# ---------------------------------------------------------------------------------------------------------------
# Async-k schedule


def test_async_schedule_k1_is_async1():
    s = AsyncSchedule(1, 10)
    assert s.initial_cohorts() == [0]
    assert [s.version_of(c) for c in range(6)] == [0, 0, 1, 2, 3, 4]  # async1: cohort c at theta_{c-1}, 0 at 0
    assert [s.age(c) for c in range(6)] == [0, 1, 1, 1, 1, 1]
    assert [s.next_cohort(t) for t in (0, 8, 9)] == [1, 9, None]
    assert all(s.lane_of(c) == 0 for c in range(6))


@pytest.mark.parametrize("k", [2, 3, 4])
def test_async_schedule_k(k):
    s = AsyncSchedule(k, 16)
    assert s.initial_cohorts() == list(range(k))
    for c in range(16):
        assert s.lane_of(c) == c % k
        assert s.version_of(c) == max(0, c - k)
        assert s.age(c) == (c if c < k else k)
    assert s.next_cohort(16 - k - 1) == 15 and s.next_cohort(16 - k) is None
    with pytest.raises(ValueError):
        AsyncSchedule(0, 4)


def test_async_ledger_refuses_bad_moves():
    led = AsyncLedger.build(AsyncSchedule(2, 10))
    assert led.start(0, 0) == 0 and led.start(1, 1) == 0
    with pytest.raises(RuntimeError, match="still generates cohort 0"):
        led.refit(0, 1)
    with pytest.raises(RuntimeError, match="belongs on lane 0"):
        led.start(1, 2)
    led.finish(0)
    assert led.refit(0, 0) is False  # theta_0 already there
    led.start(0, 2)  # cohort 2 at theta_0 (age 2)
    led.finish(1)
    with pytest.raises(RuntimeError, match="must be generated at theta_1"):
        led.start(1, 3)  # lane 1 still at theta_0
    assert led.refit(1, 1) is True
    led.start(1, 3)
    with pytest.raises(RuntimeError, match="already started"):
        led.start(1, 3)
    led.finish(3)
    with pytest.raises(RuntimeError, match="go back"):
        led.refit(1, 0)


def _async_sim(k, n_steps, seed, learner_s=(20.0, 40.0), gen_s=(30.0, 120.0), dispatch_s=4.0):
    """Event simulation of train_async_lanes' rule. Per step j: training of cohort j streams its rows (modelled as
    a learner time after the cohort completes or after the step starts, whichever is later, chopped into
    dispatches); the side task, once cohort j is complete, waits for the in-flight dispatch (gate), refits lane j %
    k to theta_j and starts cohort j + k. Returns per-step records and the ledger."""
    rng = random.Random(seed)
    s = AsyncSchedule(k, n_steps)
    led = AsyncLedger.build(s)
    done_at: dict[int, float] = {}
    started_at: dict[int, float] = {}
    rec = []

    def start(c, t):
        led.start(s.lane_of(c), c, t)
        started_at[c] = t
        done_at[c] = t + rng.uniform(*gen_s)

    t = 0.0
    for c in s.initial_cohorts():
        start(c, t)
    for j in range(n_steps):
        top = t
        d = done_at[j]
        # training: streams rows; ends one learner-time after max(top, the cohort's completion) (pessimistic: the
        # cohort's last row gates the last dispatch)
        train = rng.uniform(*learner_s)
        end = max(top + train, d + dispatch_s)
        # side: refit at max(top, d) + the in-flight dispatch's remainder (<= dispatch_s)
        ready = max(top, d)
        gate_wait = 0.0 if ready == top else rng.uniform(0.0, dispatch_s)
        t_refit = ready + gate_wait
        led.finish(j, d)
        nxt = s.next_cohort(j)
        if nxt is not None:
            led.refit(s.lane_of(j), j)
            start(nxt, t_refit + 1.0)  # 1 s refit
            end = max(end, t_refit + 1.0)  # the side task ends before the optimizer step
        in_flight = len(led.in_flight())
        rec.append(dict(step=j, top=top, end=end, cohort_done=d, ready=ready, refit_at=t_refit, in_flight=in_flight,
                        version=led.started[j][1], lane=led.started[j][0]))
        t = end
    return rec, led, started_at, done_at, s


@pytest.mark.parametrize("k", [1, 2, 4])
@pytest.mark.parametrize("seed", range(3))
def test_async_driver_rule_sim(k, seed):
    n = 16
    rec, led, started_at, done_at, s = _async_sim(k, n, seed)
    for r in rec:
        j = r["step"]
        assert r["version"] == s.version_of(j) and r["lane"] == j % k
        # trained at step j with theta_j: age k after the fill
        assert j - r["version"] == min(j, k)
        # cohort j + k starts no earlier than theta_j exists (top of step j) and cohort j is complete
        if j + k < n:
            assert started_at[j + k] >= max(r["top"], done_at[j])
            # ... and no later than that plus one dispatch and the refit (no other wait)
            assert started_at[j + k] <= max(r["top"], done_at[j]) + 4.0 + 1.0 + 1e-9
    # k cohorts in flight while every lane has a next cohort (steps < n - k)
    for r in rec[: n - k]:
        assert r["in_flight"] == k  # cohorts j + 1 .. j + k
    assert led.in_flight() == []
    assert sorted(led.started) == list(range(n))
    # at most k versions on the lanes at any time, one per lane
    assert len(led.versions()) == k


@pytest.mark.parametrize("k", [2, 4])
def test_async_overlap_beats_serial(k):
    # engine-bound regime (generation >> learner): the k cohorts' generation windows overlap in time (cohort j + 1
    # starts before cohort j completes), which is what makes async-k faster than async1 when decode-bound.
    rec, _, started_at, done_at, _ = _async_sim(k, 20, 0, learner_s=(10.0, 12.0), gen_s=(80.0, 100.0))
    assert sum(started_at[j + 1] < done_at[j] for j in range(5, 15)) >= 5
    # and the steady step time is ~ generation time / k, not the generation time (async1)
    steady = [r["end"] - r["top"] for r in rec[5:15]]
    assert sum(steady) / len(steady) < 0.75 * 80.0
