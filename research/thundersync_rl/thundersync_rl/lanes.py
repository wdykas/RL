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
"""Cohort lanes: two weight versions in the engine fleet (DESIGN.md section 14j).

The engine fleet is split into lanes, each a separate Megatron inference engine
(one MegatronGeneration over its own GPUs, one or more full TP/PP/EP replicas)
holding one weight version at a time. At step k:

* the *cohort lane* (``k % n``) holds the drafting weights D_k of cohort k
  (theta_{k-1}, or theta_hat_k with momentum drafts). It verifies cohort k's
  drafts and decodes their continuation streams on D_k (sticky anchors), then,
  only if some rollout of step k has no complete plan, refits theta_k for them;
* the *drafting lane* (``(k + 1) % n``) refits the drafter D_{k+1} of cohort
  k+1 at the top of step k and starts drafting it right away, while the cohort
  lane is still busy with cohort k's tails.

Roles alternate every step. A cohort is pinned to its lane from the start of
its drafts until its last continuation is verified (``CohortLanes.pin`` /
``unpin``); refitting a pinned lane raises, so a cohort's drafts,
continuations and anchors all come from the one version that drafted it.

Each lane has its own refit communicator (packed collective, or the native
m2n refit group under ``refit_transport: mcore``, refit_m2n.py): the learner keeps
one per lane (``ThunderSyncMegatronPolicyWorker.keep_refit_group`` /
``select_refit_group``) and refits only the lane being changed, through the
same bucketed collective as a single engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# A weight version as the driver names it: ("theta", k) = the learner's theta_k,
# ("theta_hat", k) = the momentum prediction of theta_k made from step k-1's
# optimizer state.
Version = tuple[str, int]


@dataclass
class Lane:
    """One engine instance of the fleet and the cohort it serves."""

    index: int
    generation: Any
    spec: Any = None
    # Weights the lane's engine holds (None before the first refit).
    version: Optional[Version] = None
    # Cohort (target step) whose drafts / continuations run on this lane, and
    # the version they were started on.
    cohort: Optional[int] = None
    cohort_version: Optional[Version] = None
    # Engine weight fingerprint taken when the cohort was pinned (``fingerprint`` op).
    cohort_fingerprint: Optional[float] = None
    refits: int = 0


@dataclass
class LaneStepOps:
    """The engine/learner operations of one lane step (real or fake).

    ``drafter_version(target)``: the weights cohort ``target`` is drafted with
    (``("theta", target - 1)``, or ``("theta_hat", target)`` with momentum).
    ``put_drafter(lane, target, refit)``: put them on ``lane``; ``refit()``
    refits the lane from the learner's live weights (momentum: the learner
    loads theta_hat around it and restores theta_k after).
    ``refit_lane(lane)``: refit ``lane`` from the learner's live weights.
    ``start_drafts(lane, target)``: start drafting cohort ``target`` on
    ``lane`` without waiting; returns the collector called at the deadline.
    ``complete_cohort(lane, drafts)``: verify the drafts and decode + verify
    every continuation on ``lane`` (``SpeculativeGeneration.verify_sticky``).
    ``needs_current(lane)``: some rollout of this step has no complete plan
    (it is decoded at theta_k on ``lane``).
    ``run_step(lane, collector)``: this step's rollouts (served from
    ``lane``'s plans) and training; the collector is called at the deadline;
    returns the step metrics with the next cohort's drafts under ``"_drafts"``.
    ``fingerprint(lane)`` (optional, debug): a checksum of the weights the
    lane's engine ranks hold, read from the engine itself. Taken when a cohort
    is pinned and again when it completes; they must be equal (the cohort's
    drafts, continuations and anchors ran on one set of weights; raises
    otherwise). ``lanes/fingerprint_moved`` reports whether it differs from
    the previous cohort's (the lanes really hold different versions).
    """

    drafter_version: Callable[[int], Version]
    put_drafter: Callable[[Lane, int, Callable[[], None]], None]
    refit_lane: Callable[[Lane], None]
    start_drafts: Callable[[Lane, int], Any]
    complete_cohort: Callable[[Lane, Any], None]
    needs_current: Callable[[Lane], bool]
    run_step: Callable[[Lane, Optional[Any]], dict[str, Any]]
    fingerprint: Optional[Callable[[Lane], float]] = None


@dataclass
class CohortLanes:
    """Lane roles, the cohort pinning invariant and the lane timeline."""

    lanes: list[Lane]
    clock: Callable[[], float] = time.perf_counter
    history: list[dict[str, float]] = field(default_factory=list)
    # Fingerprint of the last cohort that completed (``LaneStepOps.fingerprint``).
    last_fingerprint: Optional[float] = None

    def __post_init__(self):
        if len(self.lanes) != 2:
            raise ValueError(
                f"cohort lanes: exactly 2 lanes are built (lookahead 1), got {len(self.lanes)}"
            )

    def cohort_lane(self, step: int) -> Lane:
        """The lane holding cohort ``step`` (drafted during step - 1)."""
        return self.lanes[step % len(self.lanes)]

    def drafting_lane(self, step: int) -> Lane:
        """The lane that drafts cohort ``step + 1`` during ``step``."""
        return self.lanes[(step + 1) % len(self.lanes)]

    # -- the exactness invariant ------------------------------------------------

    def refit(self, lane: Lane, version: Version, do_refit: Callable[[], Any]) -> None:
        """Refit ``lane`` to ``version``; refused while it serves a cohort.

        A pinned lane still decodes or verifies continuations of its cohort,
        whose anchors must stay the drafting version's routes: moving its
        weights would put those continuations on another version.
        """
        if lane.cohort is not None:
            raise RuntimeError(
                f"lane {lane.index} still serves cohort {lane.cohort} "
                f"(drafted at {lane.cohort_version}); refitting it to {version} "
                "would move that cohort's continuations off their drafting version"
            )
        if lane.version == version:
            return  # the same weights (e.g. theta_0 at step 0)
        do_refit()
        lane.version = version
        lane.refits += 1

    def pin(self, lane: Lane, cohort: int) -> None:
        """Cohort ``cohort`` starts drafting on ``lane`` at its current version."""
        if lane.cohort is not None:
            raise RuntimeError(
                f"lane {lane.index} already serves cohort {lane.cohort}; "
                f"cannot start cohort {cohort} there"
            )
        if lane.version is None:
            raise RuntimeError(f"lane {lane.index} holds no weights yet")
        lane.cohort, lane.cohort_version = cohort, lane.version

    def check(self, lane: Lane, cohort: int) -> None:
        """``lane`` holds cohort ``cohort`` at the version that drafted it."""
        if lane.cohort != cohort or lane.version != lane.cohort_version:
            raise RuntimeError(
                f"cohort {cohort} must complete on the lane that drafted it at its "
                f"drafting version; lane {lane.index} serves cohort {lane.cohort} "
                f"(drafted at {lane.cohort_version}) and holds {lane.version}"
            )

    def unpin(self, lane: Lane, cohort: int) -> None:
        """Every rollout of cohort ``cohort`` is complete: ``lane`` is free."""
        self.check(lane, cohort)
        lane.cohort = lane.cohort_version = lane.cohort_fingerprint = None

    def _check_fingerprint(self, lane: Lane, cohort: int, now: float) -> dict[str, float]:
        """The engine of ``lane`` still holds the weights that drafted ``cohort``."""
        start = lane.cohort_fingerprint
        if now != start:
            raise RuntimeError(
                f"cohort {cohort}: lane {lane.index}'s engine weights changed while it "
                f"served the cohort (fingerprint {start!r} at its drafts, {now!r} at "
                "completion)"
            )
        prev, self.last_fingerprint = self.last_fingerprint, now
        return {
            "lanes/fingerprint": now,
            # 1: this cohort ran on other weights than the previous cohort.
            "lanes/fingerprint_moved": float(prev is not None and now != prev),
        }

    # -- one step -----------------------------------------------------------------

    def step(
        self,
        step: int,
        drafts: Optional[Any],
        draft_next: bool,
        ops: LaneStepOps,
    ) -> tuple[dict[str, Any], Optional[Any]]:
        """Run step ``step`` on the lanes; returns (metrics, cohort step+1 drafts).

        Order: (1) the drafting lane takes cohort step+1's drafter and starts
        drafting; (2) the cohort lane completes cohort ``step`` (verification
        and continuation streams at its drafting version) — concurrently with
        (1)'s drafts, which run on the other lane's GPUs; (3) only if a rollout
        of this step has no complete plan, the cohort lane refits theta_step
        for it; (4) rollouts + training, drafts cut at the deadline.
        """
        x, y = self.cohort_lane(step), self.drafting_lane(step)
        t0 = self.clock()
        collector = None
        if draft_next:
            self.refit(
                y,
                ops.drafter_version(step + 1),
                lambda: ops.put_drafter(y, step + 1, lambda: ops.refit_lane(y)),
            )
            self.pin(y, step + 1)
            if ops.fingerprint is not None:
                y.cohort_fingerprint = ops.fingerprint(y)
            collector = ops.start_drafts(y, step + 1)
        t_draft = self.clock()
        if drafts is not None:
            self.check(x, step)
            ops.complete_cohort(x, drafts)
        checked = {}
        if x.cohort == step:
            if ops.fingerprint is not None and x.cohort_fingerprint is not None:
                checked = self._check_fingerprint(x, step, ops.fingerprint(x))
            self.unpin(x, step)
        t_tail = self.clock()
        fresh = ops.needs_current(x)
        if fresh:
            self.refit(x, ("theta", step), lambda: ops.refit_lane(x))
        t_fresh = self.clock()
        metrics = ops.run_step(x, collector)
        t_end = self.clock()
        next_drafts = metrics.pop("_drafts", None)
        timeline = {
            # Seconds from the top of the step.
            "lanes/draft_start_s": (t_draft - t0) if draft_next else float("nan"),
            "lanes/tail_end_s": t_tail - t0,
            # Drafting of cohort step+1 that ran while cohort step's tails were
            # still decoding (> 0: overlap; a single-version engine is <= 0).
            "lanes/overlap_s": (t_tail - t_draft) if draft_next else 0.0,
            # The cohort lane after its tails (and fresh rows) until the step
            # ends; the drafting lane before its drafts start.
            "lanes/cohort_lane_idle_s": t_end - (t_fresh if fresh else t_tail),
            "lanes/drafting_lane_idle_s": (t_draft - t0) if draft_next else t_end - t0,
            "lanes/fresh_refit": float(fresh),
            "lanes/cohort_lane": float(x.index),
            **checked,
        }
        span = max(t_end - t0, 1e-9)
        timeline["lanes/idle_frac"] = (
            timeline["lanes/cohort_lane_idle_s"] + timeline["lanes/drafting_lane_idle_s"]
        ) / (len(self.lanes) * span)
        self.history.append(timeline)
        metrics.update(timeline)
        return metrics, next_drafts


def build_engine_lanes(
    policy: Any,
    generation: Any,
    train_cluster: Any,
    inference_cluster: Any,
    policy_config: dict[str, Any],
    tokenizer: Any,
    num_lanes: int = 2,
    start_step: int = 0,
) -> list[Lane]:
    """Lanes over the setup's engine plus ``num_lanes - 1`` engines of its shape.

    ``generation`` (lane 0) is the engine ``grpo.setup`` built on
    ``inference_cluster`` (one lane's GPUs) with its refit communicator and
    theta_0. Every further lane gets the same placement shape on GPUs not
    used yet, its own MegatronGeneration (same layout and config, hence the
    same batch-invariant forward), its own refit communicator to the learner
    (the learner keeps one per lane, ``keep_refit_group``) and theta_0 through
    it. Per rank nothing changes: each engine rank holds one weight version.
    """
    import ray

    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
    from nemo_rl.models.generation.megatron import MegatronGeneration

    from thundersync_rl.refit_m2n import make_lane_synchronizer

    learners = policy.worker_group
    ray.get(learners.run_all_workers_single_data("keep_refit_group", lane=0))
    # Every lane starts on the learner's weights: theta_0, or theta_start on a resume.
    lanes = [Lane(0, generation, version=("theta", start_step))]
    for i in range(1, num_lanes):
        cluster = RayVirtualCluster(
            name=f"grpo_inference_lane{i}",
            bundle_ct_per_node_list=list(inference_cluster._bundle_ct_per_node_list),
            use_gpus=True,
            num_gpus_per_node=inference_cluster.num_gpus_per_node,
            max_colocated_worker_groups=1,
            port_range_low=inference_cluster.port_range_low,
            port_range_high=inference_cluster.port_range_high,
        )
        gen = MegatronGeneration(
            config=policy_config,
            tokenizer=tokenizer,
            cluster=cluster,
            name_prefix=f"megatron_generation_lane{i}",
            skip_weight_load=True,
        )
        sync = make_lane_synchronizer(policy, gen, train_cluster, cluster)
        gen.weight_synchronizer = sync
        if gen.uses_native_refit:
            # The learner's cached plan (same key for every lane) would skip
            # the plan build this lane's engine ranks run collectively.
            ray.get(learners.run_all_workers_single_data("clear_refit_plan_cache"))
        sync.init_communicator()
        ray.get(learners.run_all_workers_single_data("keep_refit_group", lane=i))
        ray.get(learners.run_all_workers_single_data("select_refit_group", lane=i))
        # The skip-load engine starts here, on its final weight buffers.
        sync.sync_weights()
        lanes.append(Lane(i, gen, version=("theta", start_step)))
    ray.get(learners.run_all_workers_single_data("select_refit_group", lane=0))
    return lanes


def refit_lane(policy: Any, lane: Lane) -> None:
    """Refit ``lane`` alone from the learner's live weights (its communicator)."""
    import ray

    from nemo_rl.algorithms.grpo import refit_policy_generation

    ray.get(
        policy.worker_group.run_all_workers_single_data(
            "select_refit_group", lane=lane.index
        )
    )
    refit_policy_generation(policy, lane.generation, False)
