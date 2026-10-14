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
"""Cohort lanes (lanes.py): every cohort is drafted, continued and verified on
the version that drafted it, on its own lane (fake engines, no Ray, no GPUs)."""

import itertools
from types import SimpleNamespace

import pytest
from thundersync_rl.lanes import CohortLanes, Lane, LaneStepOps


class FakeFleet:
    """Two fake engines; rows record the weights they were decoded with.

    The learner's version is ``theta_step``. A refit copies the learner's live
    weights (``self.live``, theta_step or theta_hat_{step+1} while a momentum
    stage has them loaded) into the lane. Drafting starts rows on a lane;
    completing a cohort decodes its continuations on the lane that holds it
    and checks them against the drafts' version, as sticky anchors require.
    """

    def __init__(self, momentum: bool, fresh_steps=(0,)):
        self.momentum = momentum
        self.fresh_steps = set(fresh_steps)
        self.engine = {0: ("theta", 0), 1: ("theta", 0)}
        self.live = None
        self.step = 0
        self.events: list[tuple] = []
        self.rows: dict[int, dict] = {}  # cohort -> what its rows were sampled with
        self.tick = itertools.count()

    def clock(self):
        return float(next(self.tick))

    def ops(self, lanes: CohortLanes, step: int, draft_next: bool) -> LaneStepOps:
        self.step = step
        self.live = ("theta", step)

        def drafter_version(t):
            return ("theta_hat", t) if self.momentum else ("theta", t - 1)

        def put_drafter(lane, target, refit):
            if self.momentum:
                self.live = ("theta_hat", target)  # learner loads theta_hat
                try:
                    refit()
                finally:
                    self.live = ("theta", self.step)  # restore theta_k
            else:
                refit()

        def refit_lane(lane):
            self.engine[lane.index] = self.live
            self.events.append(("refit", step, lane.index, self.live))

        def start_drafts(lane, target):
            self.rows[target] = {
                "draft_lane": lane.index,
                "draft_version": self.engine[lane.index],
            }
            self.events.append(("draft", step, lane.index, target))
            return lambda: ("drafts", target)

        def complete_cohort(lane, drafts):
            assert drafts == ("drafts", step)
            rec = self.rows[step]
            rec["cont_lane"] = lane.index
            rec["cont_version"] = self.engine[lane.index]
            self.events.append(("complete", step, lane.index))

        def needs_current(lane):
            return step in self.fresh_steps

        def run_step(lane, collector):
            fresh = step in self.fresh_steps
            self.rows.setdefault(step, {})["rollout_lane"] = lane.index
            if fresh:
                self.rows[step]["fresh_version"] = self.engine[lane.index]
            self.events.append(("rollouts", step, lane.index))
            return {"_drafts": collector() if collector is not None else None}

        return LaneStepOps(
            drafter_version=drafter_version,
            put_drafter=put_drafter,
            refit_lane=refit_lane,
            start_drafts=start_drafts,
            complete_cohort=complete_cohort,
            needs_current=needs_current,
            run_step=run_step,
        )


def _run(fleet: FakeFleet, steps: int):
    lanes = CohortLanes(
        [Lane(i, None, version=("theta", 0)) for i in range(2)], clock=fleet.clock
    )
    drafts, out = None, []
    for step in range(steps):
        m, drafts = lanes.step(
            step, drafts, step + 1 < steps, fleet.ops(lanes, step, step + 1 < steps)
        )
        out.append(m)
    return lanes, out


@pytest.mark.parametrize("momentum", [False, True])
def test_each_cohort_continues_on_its_drafting_lane_and_version(momentum):
    fleet = FakeFleet(momentum, fresh_steps=(0, 3))
    _run(fleet, 6)
    for t in range(1, 6):
        rec = fleet.rows[t]
        want = ("theta_hat", t) if momentum else ("theta", t - 1)
        assert rec["draft_lane"] == t % 2  # roles alternate
        assert rec["draft_version"] == want
        # Continuations (sticky anchors) on the same lane and the same weights.
        assert rec["cont_lane"] == rec["draft_lane"]
        assert rec["cont_version"] == want
        assert rec["rollout_lane"] == rec["draft_lane"]
    # Fresh rows (no complete plan) decode at theta_step on the cohort lane.
    assert fleet.rows[0]["fresh_version"] == ("theta", 0)
    assert fleet.rows[3]["fresh_version"] == ("theta", 3)
    assert "fresh_version" not in fleet.rows[2]


def test_next_cohort_drafts_before_this_cohort_completes():
    fleet = FakeFleet(momentum=True, fresh_steps=())
    lanes, metrics = _run(fleet, 4)
    for step in range(1, 3):
        ev = [e for e in fleet.events if e[1] == step]
        kinds = [e[0] for e in ev]
        # Drafter refit and drafts of cohort step+1 on the other lane come first.
        assert kinds[:4] == ["refit", "draft", "complete", "rollouts"]
        assert ev[0][2] == ev[1][2] == (step + 1) % 2
        assert ev[2][2] == step % 2
        assert metrics[step]["lanes/overlap_s"] > 0
    # Without fresh rows every lane refits once per two steps (one per step).
    assert [e[2] for e in fleet.events if e[0] == "refit"] == [1, 0, 1]


def test_current_drafts_skip_the_step0_refit():
    fleet = FakeFleet(momentum=False, fresh_steps=(0,))
    _run(fleet, 3)
    # theta_0 is on both lanes after setup: lane 1 drafts cohort 1 unrefit.
    refits = [e for e in fleet.events if e[0] == "refit"]
    # Step 1: lane 0 (done with cohort 0) takes theta_1
    # for cohort 2; the last step drafts nothing.
    assert refits == [("refit", 1, 0, ("theta", 1))]


def test_refit_of_a_pinned_lane_raises():
    lanes = CohortLanes([Lane(i, None, version=("theta", 0)) for i in range(2)])
    lane = lanes.lanes[1]
    lanes.pin(lane, 1)
    with pytest.raises(RuntimeError, match="still serves cohort 1"):
        lanes.refit(lane, ("theta", 1), lambda: None)
    lanes.unpin(lane, 1)
    called = []
    lanes.refit(lane, ("theta", 1), lambda: called.append(1))
    assert called == [1] and lane.version == ("theta", 1)


def test_cohort_must_complete_on_its_version():
    lanes = CohortLanes([Lane(i, None, version=("theta", 0)) for i in range(2)])
    lane = lanes.lanes[0]
    lanes.pin(lane, 3)
    lane.version = ("theta", 3)  # weights moved behind the scheduler's back
    with pytest.raises(RuntimeError, match="drafting version"):
        lanes.check(lane, 3)
    with pytest.raises(RuntimeError, match="drafting version"):
        lanes.check(lanes.lanes[1], 3)  # the other lane never drafted it


def test_fresh_rows_cannot_refit_before_the_cohort_completes():
    """A fresh-row refit while the cohort lane is pinned would move its tails."""
    fleet = FakeFleet(momentum=False, fresh_steps=(1,))
    lanes = CohortLanes(
        [Lane(i, None, version=("theta", 0)) for i in range(2)], clock=fleet.clock
    )
    _, drafts = lanes.step(0, None, True, fleet.ops(lanes, 0, True))
    ops = fleet.ops(lanes, 1, True)

    def complete_and_refit(lane, d):
        lanes.refit(lane, ("theta", 1), lambda: None)

    ops.complete_cohort = complete_and_refit
    with pytest.raises(RuntimeError, match="still serves cohort 1"):
        lanes.step(1, drafts, True, ops)


def test_exactly_two_lanes():
    with pytest.raises(ValueError, match="exactly 2 lanes"):
        CohortLanes([Lane(0, None)])


def test_refit_group_bookkeeping():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    W = ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    w = SimpleNamespace(model_update_group="g0")
    W.keep_refit_group(w, 0)
    assert w.model_update_group is None  # init_collective will not release it
    w.model_update_group = "g1"
    W.keep_refit_group(w, 1)
    W.select_refit_group(w, 0)
    assert w.model_update_group == "g0"
    W.select_refit_group(w, 1)
    assert w.model_update_group == "g1"
    with pytest.raises(RuntimeError, match="no refit communicator"):
        W.keep_refit_group(SimpleNamespace(model_update_group=None), 0)


def _master(gen=None, **ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        draft_lookahead=1,
        sticky_routing=True,
        cohort_lanes=2,
    )
    base.update(ts)
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        policy={"generation": {"backend": "megatron", "refit_transport": None, **(gen or {})}},
    )


def test_validate_cohort_lanes_accepts_sticky_lanes():
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_cohort_lanes

    assert ThunderSyncConfig().cohort_lanes == 1
    validate_cohort_lanes(_master())
    validate_cohort_lanes(_master(draft_weights="momentum"))
    validate_cohort_lanes(_master(cohort_lanes=1, sticky_routing=False))


@pytest.mark.parametrize(
    "gen,ts,msg",
    [
        ({}, {"cohort_lanes": 3}, "1 or 2"),
        ({}, {"sticky_routing": False}, "sticky_routing"),
        ({"refit_transport": "mcore"}, {}, "refit_transport"),
        ({"backend": "vllm"}, {}, "megatron"),
        ({}, {"draft_after_verify": True}, "draft_after_verify"),
    ],
)
def test_validate_cohort_lanes_rejects(gen, ts, msg):
    from thundersync_rl.grpo_loop import validate_cohort_lanes

    with pytest.raises(ValueError, match=msg):
        validate_cohort_lanes(_master(gen, **ts))


def _fingerprinted(fleet: FakeFleet, lanes: CohortLanes, step: int, draft_next: bool):
    ops = fleet.ops(lanes, step, draft_next)
    ops.fingerprint = lambda lane: hash(fleet.engine[lane.index])  # read from the engine
    return ops


@pytest.mark.parametrize("momentum", [False, True])
def test_engine_fingerprint_held_per_cohort_and_moves(momentum):
    fleet = FakeFleet(momentum, fresh_steps=(0,))
    lanes = CohortLanes(
        [Lane(i, None, version=("theta", 0)) for i in range(2)], clock=fleet.clock
    )
    drafts, out = None, []
    for step in range(5):
        ops = _fingerprinted(fleet, lanes, step, step + 1 < 5)
        m, drafts = lanes.step(step, drafts, step + 1 < 5, ops)
        out.append(m)
    assert "lanes/fingerprint" not in out[0]  # cohort 0 is fresh rows, no drafts
    for t in range(1, 5):
        want = ("theta_hat", t) if momentum else ("theta", t - 1)
        assert out[t]["lanes/fingerprint"] == hash(want)
    assert [m["lanes/fingerprint_moved"] for m in out[1:]] == [0.0, 1.0, 1.0, 1.0]


def test_engine_weights_moving_under_a_pinned_cohort_raise():
    """The engine's own weights changed (not via CohortLanes.refit): caught at completion."""
    fleet = FakeFleet(momentum=False, fresh_steps=(0,))
    lanes = CohortLanes(
        [Lane(i, None, version=("theta", 0)) for i in range(2)], clock=fleet.clock
    )
    _, drafts = lanes.step(0, None, True, _fingerprinted(fleet, lanes, 0, True))
    ops = _fingerprinted(fleet, lanes, 1, True)
    complete = ops.complete_cohort

    def complete_after_stray_refit(lane, d):
        fleet.engine[lane.index] = ("theta", 1)  # bypasses the scheduler
        complete(lane, d)

    ops.complete_cohort = complete_after_stray_refit
    with pytest.raises(RuntimeError, match="engine weights changed"):
        lanes.step(1, drafts, True, ops)
