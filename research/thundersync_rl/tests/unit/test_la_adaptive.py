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
"""Phase-adaptive lookahead (la2.AdaptiveLookahead, thundersync.cohort_lane_lookahead_adaptive).

- Policy: hysteresis on the forecast age-2 v; forecast from the calibrator's fresh per-age EMAs.
- Schedule invariants with switches anywhere (incl. next to checkpoint boundaries): every cohort drafts on lane
  c % 2 at lag 1 or 2, heads kept, boundaries quiescent; lag 2 exactly where the policy wanted it and an early
  start was possible.
- Exactness by enumeration: the drafter version (lag 1 / lag 2) and the cut chosen from another cohort's
  independent outcome leave the output law exactly p.
- Resume across a switch: the policy state round-trips through the specs' policy state, so a resumed run
  makes the same choices as the uninterrupted one.
"""
from __future__ import annotations

import random
from types import SimpleNamespace

import pytest
from test_la2 import L, _law, _model, _simulate, _tv, V
from thundersync_rl.la2 import AdaptiveLookahead
from thundersync_rl.null_bv import LogRatioCalibration


def _mom(v, n=1000):
    return [(n, 0.0, n * v)]


def test_hysteresis():
    p = AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5)
    seq = [5e-6, 1.5e-5, 2.5e-5, 1.5e-5, 1.2e-5, 9e-6, 1.9e-5, 3e-5]
    got = [p.decide(v) for v in seq]
    assert got == [True, True, False, False, False, True, True, False]
    assert p.switches == 3 and p.mode == 1
    with pytest.raises(ValueError):
        AdaptiveLookahead(v_hi=1e-5, v_lo=2e-5)


def test_forecast_uses_fresh_age_estimates():
    cal = LogRatioCalibration(prior=8e-7, decay=0.0)
    p = AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5, age_ratio=4.0)
    assert p.forecast(cal, 0) == pytest.approx(8e-7)  # nothing measured: the prior
    cal.update(_mom(6e-6), age=1, step=3)
    assert p.forecast(cal, 4) == pytest.approx(2.4e-5)  # age-1 only: ratio x age-1
    cal.update(_mom(3e-5), age=2, step=5)
    assert p.forecast(cal, 6) == pytest.approx(3e-5)  # fresh age-2 wins
    assert p.forecast(cal, 20) == pytest.approx(cal.estimate())  # all stale: pooled


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("p_fresh", [0.0, 0.3])
def test_schedule_invariants_with_switches(seed, p_fresh):
    rng = random.Random(100 + seed)
    pol = AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5)
    wanted = {}

    def want(step):
        wanted[step] = pol.decide(rng.choice([5e-6, 1.5e-5, 3e-5]))
        return wanted[step]

    boundaries = (7, 8, 15)  # a boundary right after another and switches next to them
    d = _simulate(24, seed, boundaries=boundaries, p_fresh=p_fresh, want=want)
    lags = {c: c - v for c, (_, v) in d.items()}
    assert set(lags.values()) <= {1, 2}
    assert all(lane == c % 2 for c, (lane, _) in d.items())
    for c, lag in lags.items():
        if lag == 2:
            assert wanted[c - 2], (c, lags, wanted)  # lag 2 only where the policy asked for it
    assert pol.switches >= 1  # non-vacuous: the run did switch


def test_switch_chosen_from_another_cohorts_outcome_is_exact():
    theta = [_model(V, L, s) for s in (11, 12, 13)]  # theta_k, theta_{k+1}, theta_{k+2}
    P, Q2, Q1 = theta[2], theta[0], theta[1]
    # The lookahead choice and the cut both come from an earlier cohort's (independent) outcome:
    # u ~ that outcome; lag 2 with one cut law, lag 1 with another.
    outcome = {0: 0.3, 1: 0.45, 2: 0.25}
    pick = {0: (Q2, lambda x: {L: 1.0}), 1: (Q1, lambda x: {1: 0.5, L: 0.5}), 2: (Q2, lambda x: {0: 0.2, 2: 0.8})}
    mix = {}
    for u, pu in outcome.items():
        Q, cut = pick[u]
        for y, py in _law(P, Q, cut).items():
            mix[y] = mix.get(y, 0.0) + pu * py
    assert _tv(mix, P) < 1e-12
    # Non-vacuous: lag 2 drafts differ from p (rejections happen) under every choice.
    import itertools

    from test_null_bv import _seq_prob

    for Q in (Q1, Q2):
        assert _tv({x: _seq_prob(Q, x) for x in itertools.product(range(V), repeat=L)}, P) > 1e-3


def test_policy_state_round_trip_across_a_switch():
    from thundersync_rl.speculative import SpeculativeGeneration

    ps = SpeculativeGeneration._POLICY_STATE
    seq = [5e-6, 3e-5, 1.5e-5, 1.5e-5, 8e-6, 3e-5, 1.2e-5]
    ref = AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5)
    ref_out = [ref.decide(v) for v in seq]
    a = SimpleNamespace(_POLICY_STATE=ps, _la_policy=AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5))
    head = [a._la_policy.decide(v) for v in seq[:3]]  # in lookahead-1 mode at the save (switched at step 1)
    assert a._la_policy.mode == 1
    state = SpeculativeGeneration.policy_state(a)
    b = SimpleNamespace(_POLICY_STATE=ps, _la_policy=AdaptiveLookahead(v_hi=2e-5, v_lo=1e-5))
    SpeculativeGeneration.load_policy_state(b, state)
    tail = [b._la_policy.decide(v) for v in seq[3:]]
    assert head + tail == ref_out and b._la_policy.switches == ref.switches


def test_validation():
    from test_lanes import _master as _lane_master
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_cohort_lanes

    assert ThunderSyncConfig().cohort_lane_lookahead_adaptive is False
    ok = dict(cohort_lane_lookahead=2, stream_training_over_tails=True, verify_on="learner",
              cohort_lane_lookahead_adaptive=True, null_bv=True, null_bv_calibrate=True)
    validate_cohort_lanes(_lane_master(**ok))
    for bad, msg in [
        (dict(cohort_lane_lookahead=1), "needs cohort_lane_lookahead=2"),
        (dict(null_bv_calibrate=False), "null_bv_calibrate"),
        (dict(lookahead_adaptive_v_lo=3e-5), "v_lo must be"),
    ]:
        with pytest.raises(ValueError, match=msg):
            validate_cohort_lanes(_lane_master(**{**ok, **bad}))
