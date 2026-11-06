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
"""Cohort-lane lookahead 2 (la2.py).

- Bookkeeping: early starts only on a free cohort lane, inside the draft limit, at theta_{target-2}.
- Schedule invariants over many steps (random fresh rows / boundaries), with the real CohortLanes pinning:
  every cohort drafts and completes on lane target % 2 at one version (target-2 early, target-1 fallback),
  a pinned lane is never refit, at most two versions are live, heads kept cover every pending cohort.
- draft_streaming(target=k+2): row keys / ⊥-BV slots registered for the target, version tag = current step.
- Exactness by enumeration: a draft from an older drafter (lag 2), truncated at a length fixed by another
  cohort's outcome (its lane's tail end) and verified with continuation from p, is exactly p; a cut that
  reads the draft's own later tokens is biased (non-vacuous control).
"""

from __future__ import annotations

import asyncio
import itertools
import math
import random
from types import SimpleNamespace

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_lanes import _master as _lane_master
from test_null_bv import _model, _rows, _seq_prob
from thundersync_rl.block_verification import output_distribution_given_draft
from thundersync_rl.la2 import Lookahead2, cohort_version
from thundersync_rl.lanes import CohortLanes, Lane

# ---------------------------------------------------------------------------
# Bookkeeping


def test_may_start_rules():
    la = Lookahead2()
    assert la.may_start(3, draft_limit=10, have_batch=True, lane_fresh=False, lane_cohort=None)
    assert not la.may_start(3, draft_limit=5, have_batch=True, lane_fresh=False, lane_cohort=None)  # 5 crosses
    assert la.may_start(3, draft_limit=6, have_batch=True, lane_fresh=False, lane_cohort=None)
    assert not la.may_start(3, draft_limit=10, have_batch=False, lane_fresh=False, lane_cohort=None)
    assert not la.may_start(3, draft_limit=10, have_batch=True, lane_fresh=True, lane_cohort=None)
    assert not la.may_start(3, draft_limit=10, have_batch=True, lane_fresh=False, lane_cohort=3)
    la.started(5, 3, "c5", 1.0)
    assert not la.may_start(3, draft_limit=10, have_batch=True, lane_fresh=False, lane_cohort=None)
    with pytest.raises(RuntimeError, match="must draft on theta_4"):
        Lookahead2().started(6, 3, "c", 0.0)
    assert la.take(5) == "c5" and la.take(5) is None and la.leftover() == []


def test_cohort_version():
    assert cohort_version(None, 7) == 6
    assert cohort_version(("theta", 5), 7) == 5
    with pytest.raises(RuntimeError, match="current weights only"):
        cohort_version(("theta_hat", 7), 7)


# ---------------------------------------------------------------------------
# Schedule invariants (driver rules of grpo_loop's streaming lanes loop, real pinning)


def _simulate(n_steps, seed, boundaries=(), p_fresh=0.0, want=None):
    """Mirror of the loop: top of step (early collector or LA1 start), cohort final -> unpin, early start of
    step+2 on the freed lane at theta_step, cut at the end of the step. Returns per-cohort (lane, version)."""
    rng = random.Random(seed)
    lanes = [Lane(0, None, version=("theta", 0)), Lane(1, None, version=("theta", 0))]
    sched = CohortLanes(lanes)
    la = Lookahead2()
    drafted = {}  # cohort -> (lane, version)
    collectors = {}  # cohort -> collector at the step's deadline
    heads = set()

    def draft_limit(step):
        b = [x for x in boundaries if x >= step + 1]
        return min([n_steps] + b)

    for step in range(n_steps):
        x, y = sched.cohort_lane(step), sched.drafting_lane(step)
        live = {ln.version for ln in lanes}
        assert len(live) <= 2
        # top of the step
        early = la.take(step + 1)
        if early is not None:
            assert y.cohort == step + 1 and y.cohort_version == ("theta", step - 1)
            collectors[step + 1] = early
        elif step + 1 < draft_limit(step):
            sched.refit(y, ("theta", step), lambda: None)
            sched.pin(y, step + 1)
            drafted[step + 1] = (y.index, step)
            collectors[step + 1] = f"c{step + 1}"
        # cohort step completes on its lane at its drafting version
        if step in drafted:
            lane_i, ver = drafted[step]
            assert x.index == lane_i == step % 2
            sched.check(x, step)
            assert x.cohort_version == ("theta", ver)
            assert ver in heads or ver == step  # q's head of the drafting version was kept
            sched.unpin(x, step)
        fresh = step not in drafted or rng.random() < p_fresh
        go = la.may_start(step, draft_limit(step), step + 2 < n_steps, fresh, x.cohort)
        if want is not None:  # phase-adaptive lookahead (scheduling only): decided every step
            go = want(step) and go
        if fresh or go:
            sched.refit(x, ("theta", step), lambda: None)
        if go:
            sched.pin(x, step + 2)
            la.started(step + 2, step, f"e{step + 2}", float(step))
            drafted[step + 2] = (x.index, step)
        # deadline: the next cohort's drafts are cut; heads kept at the end of the step
        heads = set(la.head_keep(step))
        if step + 1 in drafted:
            assert drafted[step + 1][1] in heads
        if step + 2 in drafted:
            assert drafted[step + 2][1] in heads
        if (step + 1) in boundaries:
            assert all(ln.cohort is None or ln.cohort == step + 1 for ln in lanes)
            assert la.leftover() == []
    return drafted


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("p_fresh", [0.0, 0.3])
def test_schedule_invariants(seed, p_fresh):
    d = _simulate(24, seed, boundaries=(7, 15), p_fresh=p_fresh)
    lags = {c: c - v for c, (_, v) in d.items()}
    assert set(lags.values()) <= {1, 2}
    if p_fresh == 0.0:
        # lag 1 only where no early start was possible: step c-2 had no drafted cohort (step 0 and the step
        # after a boundary decode fresh rows) or a boundary lies in (c-2, c]
        for c, lag in lags.items():
            if lag == 1:
                assert (c - 2) not in d or any(c - 1 <= b <= c for b in (7, 15)), (c, lags)
        assert sum(v == 2 for v in lags.values()) >= len(lags) // 2, lags
    assert all(lane == c % 2 for c, (lane, _) in d.items())


def test_validation():
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_cohort_lanes

    assert ThunderSyncConfig().cohort_lane_lookahead == 1
    ok = dict(cohort_lane_lookahead=2, stream_training_over_tails=True, verify_on="learner")
    validate_cohort_lanes(_lane_master(**ok))
    for bad, msg in [
        (dict(stream_training_over_tails=False), "stream_training_over_tails"),
        (dict(draft_weights="momentum"), "draft_weights must be current"),
        (dict(verify_on="engine"), "verify_on=learner"),
    ]:
        with pytest.raises(ValueError, match=msg):
            validate_cohort_lanes(_lane_master(**{**ok, **bad}))


# ---------------------------------------------------------------------------
# draft_streaming targets


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def test_draft_streaming_target(_no_ray):
    from test_draft_capacity import _fit_spec

    started = []
    s = _fit_spec(100, started)
    s.current_step = 3
    prompts = [torch.tensor([1]), torch.tensor([1]), torch.tensor([2]), torch.tensor([2])]
    asyncio.run(s.draft_streaming(prompts, null_slots={(1,): 2, (2,): 2}, target=5))
    assert 5 in s._null_slots and 4 not in s._null_slots
    with pytest.raises(ValueError, match="later step"):
        asyncio.run(s.draft_streaming(prompts, null_slots={(1,): 2, (2,): 2}, target=3))
    asyncio.run(s.draft_streaming(prompts, null_slots={(1,): 2, (2,): 2}))  # default: current + 1
    assert 4 in s._null_slots


# ---------------------------------------------------------------------------
# Exactness by enumeration

V, L = 3, 3


def _trunc_outcome_law(P, Q, x, n):
    """Law of the output when the draft x (from Q) is cut to its first n tokens and verified by BV against
    P (M = 1); every position after the decision is decoded from P (the continuation)."""
    out: dict = {}
    if n == 0:
        for z in itertools.product(range(V), repeat=L):
            out[z] = out.get(z, 0.0) + _seq_prob(P, z)
        return out
    xs = list(x[:n])
    lp = _rows(P, xs, n + 1, torch.float64) if n < L else _rows(P, xs, L, torch.float64)
    if n == L:
        lp = torch.cat([lp, lp[-1:]])  # bonus row at the cap (never used)
    lq = _rows(Q, xs, n, torch.float64)
    for tau, probs in output_distribution_given_draft(lp, lq, torch.tensor(xs), 1.0):
        probs = probs.double().numpy()
        if tau == L:
            out[tuple(xs)] = out.get(tuple(xs), 0.0) + probs.sum()
            continue
        for y, py in enumerate(probs[:V]):
            if py == 0:
                continue
            pre = xs[:tau] + [y]
            for tail in itertools.product(range(V), repeat=L - len(pre)):
                z = tuple(pre) + tail
                pz = py
                for t in range(len(pre), L):
                    pz *= P[z[:t]][z[t]]
                out[z] = out.get(z, 0.0) + pz
    return out


def _law(P, Q, cut_law):
    """Output law with the cut length drawn by cut_law(x) -> {n: prob} (x = the draft's own tokens)."""
    law: dict = {}
    for x in itertools.product(range(V), repeat=L):
        qx = _seq_prob(Q, x)
        for n, pn in cut_law(x).items():
            for z, v in _trunc_outcome_law(P, Q, x, n).items():
                law[z] = law.get(z, 0.0) + qx * pn * v
    return law


def _tv(law, P):
    return 0.5 * sum(abs(law.get(z, 0.0) - _seq_prob(P, z)) for z in itertools.product(range(V), repeat=L))


def test_lag2_draft_with_start_set_by_another_cohort_is_exact():
    theta = [_model(V, L, s) for s in (11, 12, 13)]  # theta_k, theta_{k+1}, theta_{k+2}
    P, Q2, Q1 = theta[2], theta[0], theta[1]
    # The start time of cohort k+2's drafts = cohort k's tail end, a function of cohort k's own samples
    # (an independent draw); with a fixed deadline it sets how many tokens each draft has at the cut.
    start_law = {0.0: 0.2, 0.4: 0.5, 0.9: 0.3}
    cut = lambda x: {min(L, int((1.0 - s) * 4)): ps for s, ps in start_law.items()}  # noqa: E731
    assert _tv(_law(P, Q2, cut), P) < 1e-12  # lag 2
    assert _tv(_law(P, Q1, cut), P) < 1e-12  # lag 1, same construction
    # Non-vacuous: drafts are often cut and often rejected.
    assert _tv(_law(P, Q2, lambda x: {L: 1.0}), P) < 1e-12
    lens = {n for d in [cut(None)] for n in d}
    assert lens == {0, 2, 3}


def test_cut_reading_the_own_draft_is_biased():
    P, Q = _model(V, L, 13), _model(V, L, 11)
    # A cut that is a stopping time of the draft (reads only tokens before it) is still exact; a cut that
    # reads the draft's tokens after it (e.g. wall-clock affected by how the row continues) is not.
    stop = lambda x: {1: 1.0} if x[0] == 0 else {L: 1.0}  # noqa: E731
    assert _tv(_law(P, Q, stop), P) < 1e-12
    future = lambda x: {1: 1.0} if x[L - 1] == 0 else {L: 1.0}  # noqa: E731
    assert _tv(_law(P, Q, future), P) > 1e-3
