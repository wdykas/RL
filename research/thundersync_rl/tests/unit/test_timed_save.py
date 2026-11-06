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
"""Time-triggered checkpoints (checkpointing.checkpoint_must_save_by) on the ThunderSync loop (run_state.TimedSave).

The loop is modelled step by step: during step s a cohort-lookahead-L run drafts cohorts s+1..s+L, each only below
``Boundaries.draft_limit(s)`` (la2.Lookahead2.may_start: target s+2 < draft_limit; LA1 / the LA2 fallback: s+1). A
checkpoint before step b is quiescent when no cohort >= b has been drafted (grpo_loop.save_boundary raises "lane ...
pinned to cohort b" otherwise). Regression: a Nano long run (ts_scale, LA2) crashed at its save-by save before step 110
because the save was requested at s+1 after step s while cohort s+1 was already drafted during step s-1.
"""

import inspect
from types import SimpleNamespace

import pytest
from thundersync_rl import grpo_loop
from thundersync_rl.run_state import Boundaries, TimedSave, cohort_lookahead


def _run(lookahead, due_after, *, request_lookahead=None, max_steps=40, save_period=20, align=None):
    """Simulate the loop. Returns (saves, timed, violations): checkpoint boundaries taken, the time-triggered
    boundary, and non-quiescent saves as (boundary, drafted cohorts >= boundary)."""
    bounds = Boundaries(max_steps=max_steps, save_period=save_period)
    clock = {"next": 0}
    timed = TimedSave(
        bounds,
        lookahead if request_lookahead is None else request_lookahead,
        lambda: clock["next"] - 1 >= due_after,
        align=align,
    )
    drafted: set[int] = set()
    saves, violations = [], []
    step = 0
    while step < max_steps:
        lim = bounds.draft_limit(step)
        for target in range(step + 1, step + lookahead + 1):
            if target < lim:
                drafted.add(target)
        drafted.discard(step)  # this step trained its cohort
        step += 1
        clock["next"] = step
        timed.after_step(step)
        if bounds.saves_after(step - 1) or step >= max_steps:
            ahead = sorted(c for c in drafted if c >= step)
            if ahead:
                violations.append((step, ahead))
            saves.append(step)
    return saves, timed.boundary, violations


@pytest.mark.parametrize("lookahead", [1, 2])
@pytest.mark.parametrize("due_after", [3, 7, 12, 25, 33])
def test_save_by_mid_run_is_quiescent(lookahead, due_after):
    saves, timed, violations = _run(lookahead, due_after)
    assert violations == []
    periodic = [b for b in range(due_after + 2, due_after + 1 + lookahead) if b % 20 == 0]
    assert timed == (periodic[0] if periodic else due_after + 1 + lookahead)
    assert timed in saves


def test_la2_regression_nano_long200_step_110():
    # long200 (ts_scale, LA2, save_period 20): the deadline was due after step 108. Requesting at s + 1 = 110
    # found cohort 110 drafted during step 108 -> 'lane 0 pinned to cohort 110'. Negative control: the old
    # request offset (lookahead 1) on an LA2 run reproduces exactly that.
    saves, timed, violations = _run(2, 108, request_lookahead=1, max_steps=200)
    assert violations == [(110, [110])]
    saves, timed, violations = _run(2, 108, max_steps=200)
    assert violations == [] and timed == 111 and 111 in saves


def test_la1_offset_unchanged():
    saves, timed, violations = _run(1, 108, max_steps=200)
    assert violations == [] and timed == 110


def test_earlier_periodic_boundary_is_used():
    # Due after step 18 at LA2: s + 1 + 2 = 21, but the periodic save before step 20 comes first and is
    # quiescent by construction: stop there, no extra checkpoint requested.
    bounds = Boundaries(max_steps=40, save_period=20)
    timed = TimedSave(bounds, 2, lambda: True)
    assert timed.after_step(19) == 20 and bounds.requested == set()
    saves, timed_b, violations = _run(2, 18)
    assert violations == [] and timed_b == 20 and saves == [20, 40]


def test_fires_once_and_not_on_a_boundary():
    bounds = Boundaries(max_steps=40, save_period=10)
    calls = []
    timed = TimedSave(bounds, 2, lambda: calls.append(1) or True)
    assert timed.after_step(10) is None and calls == []  # step 9 ended at a boundary: deadline not consumed
    assert timed.after_step(11) == 13 and bounds.requested == {13}
    assert timed.after_step(12) is None and len(calls) == 1


def test_anchor_epochs_align_the_boundary():
    bounds = Boundaries(max_steps=60, save_period=0)
    timed = TimedSave(bounds, 1, lambda: True, align=lambda b: -(-b // 4) * 4)
    assert timed.after_step(10) == 12 and bounds.requested == {12}


def test_cohort_lookahead_from_the_lane_config():
    assert cohort_lookahead(SimpleNamespace(cohort_lane_lookahead=2), lanes_on=True) == 2
    assert cohort_lookahead(SimpleNamespace(cohort_lane_lookahead=1), lanes_on=True) == 1
    assert cohort_lookahead(SimpleNamespace(cohort_lane_lookahead=2), lanes_on=False) == 1
    assert cohort_lookahead(SimpleNamespace(), lanes_on=True) == 1


def test_loop_requests_through_timed_save():
    src = inspect.getsource(grpo_loop.thundersync_grpo_train)
    assert "bounds.requested.add" not in src  # every time-triggered request goes through TimedSave
    assert "cohort_lookahead(ts_cfg, sched is not None)" in src


# ---------------------------------------------------------------------------
# The run stops at the time-triggered checkpoint
# ---------------------------------------------------------------------------


def _run_until_stop(lookahead, due_after, max_steps=40, save_period=20):
    """As _run, but the loop breaks after the time-triggered save (grpo_loop). Returns (saves, last step run)."""
    bounds = Boundaries(max_steps=max_steps, save_period=save_period)
    clock = {"next": 0}
    timed = TimedSave(bounds, lookahead, lambda: clock["next"] - 1 >= due_after)
    saves, step = [], 0
    while step < max_steps:
        step += 1
        clock["next"] = step
        timed.after_step(step)
        last = step >= max_steps
        if bounds.saves_after(step - 1) or last:
            saves.append(step)
            if not last and timed.stops_after(step):
                break
    return saves, step - 1


@pytest.mark.parametrize("lookahead", [1, 2])
@pytest.mark.parametrize("due_after", [3, 7, 18, 25, 33])
def test_run_stops_at_the_time_triggered_checkpoint(lookahead, due_after):
    saves, last_run = _run_until_stop(lookahead, due_after)
    periodic = [b for b in range(due_after + 2, due_after + 1 + lookahead) if b % 20 == 0]
    stop = periodic[0] if periodic else due_after + 1 + lookahead
    assert saves[-1] == stop and last_run == stop - 1  # the checkpoint is the run's last state
    assert all(b % 20 == 0 for b in saves[:-1])


def test_no_stop_without_a_deadline_or_at_periodic_saves():
    saves, last_run = _run_until_stop(2, due_after=10**9)
    assert saves == [20, 40] and last_run == 39
    timed = TimedSave(Boundaries(max_steps=40, save_period=20), 2, lambda: False)
    assert not any(timed.stops_after(b) for b in range(41))


def test_loop_breaks_after_the_time_triggered_checkpoint():
    src = inspect.getsource(grpo_loop.thundersync_grpo_train)
    i = src.index("save_boundary(step, val_metrics, final=last)")
    j = src.index("timed_save.stops_after(step)", i)
    assert src.index("break", j) < src.index("THUNDERSYNC_DEBUG_STOP_AT", i)
