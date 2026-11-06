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
"""Two-period drafting on cohort lanes (``thundersync.cohort_lane_lookahead: 2``).

Lookahead 1 (today): cohort k+1 drafts on lane (k+1) % 2 from the top of step k (theta_k) to the cut at the end
of step k, a window of one step period. A full-Nano 8k row needs ~160 s at 192 rows/rank on a 4-GPU lane against a
~97 s period, so drafts are cut, the next step continues them before its training, and step lengths alternate
.

Lookahead 2: lane k % 2 starts drafting cohort k+2 as soon as its own cohort k is final (verify0 and every
continuation done), with the learner's live weights theta_k, and the drafts run until the cut at the end of step
k+1: a window of ~2 periods minus cohort k's tail time. Per lane the timeline is

    ... [tails of cohort k on D_k] [refit theta_k] [drafts of cohort k+2 on theta_k ......... cut] [tails of k+2] ...
                                    ^ step k, mid-step                            end of step k+1 ^

Versions: lane k % 2 holds D_k until cohort k is final, then theta_k (= D_{k+2}); the other lane holds D_{k+1}. Two
versions in the fleet, one per GPU, as with lookahead 1. theta_k is the learner's live weights during step k (the
optimizer step is at the end), so the refit is the same mid-step refit lanes already use for fresh rows.

Exactness: D_{k+2} = theta_k is fixed before theta_{k+2}'s gradient exists; the drafts are verified against
p_{k+2} with their logged q (any proposal fixed before the target is valid); sticky anchors are D_{k+2}'s plain
routes; continuations decode on D_{k+2} (the lane keeps it until cohort k+2 is final). The drafting start time
depends on cohort k's content only (when its tails end), the cut is the usual deadline: neither reads a draft of
cohort k+2, so the truncation is independent of the drafts' own draws (the lookahead-1 argument).

Fallback: a cohort whose drafting lane could not start it early (step 0, a cohort lane with fresh rows, the step
after a boundary) is drafted at the top of the step before it with theta_{k-1}, exactly as lookahead 1.

Depth k (``cohort_lane_lookahead: k`` on ``cohort_lanes: k``, lanes-k; research/lanesk/README.md): the same rule with
k lanes. Lane k' % k starts cohort k'+k with theta_k' once its own cohort k' is final, and the drafts run until the
cut at the end of step k'+k-1 (~k periods minus cohort k''s tail time). One weight version per lane GPU set, so k
versions in the fleet: theta_{k'-k+1} .. theta_k' (the learner keeps those k drafting heads). The exactness argument
is unchanged (the drafter is fixed before theta_{k'+k}'s gradient exists; the cut reads no draft of its cohort).
Depth 2 is lookahead 2 above, bitwise (``Lookahead2(depth=2)`` is the default).

This module holds the driver-side bookkeeping only (no engine, no learner); grpo_loop does the calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Lookahead2:
    """Pending early-started cohorts and the rules that decide them.

    ``pending[t]`` = the collector of cohort ``t``'s drafts (started early on lane ``t % depth``), cut at the end
    of step ``t - 1``. ``versions[t]`` = the learner step whose weights drafted cohort ``t``. ``depth``: how many
    steps ahead the cohort lane of step s starts a cohort (s + depth); 2 = lookahead 2.
    """

    depth: int = 2
    pending: dict[int, Any] = field(default_factory=dict)
    versions: dict[int, int] = field(default_factory=dict)
    starts: dict[int, float] = field(default_factory=dict)

    def __post_init__(self):
        if int(self.depth) < 2:
            raise ValueError(f"lookahead depth must be >= 2 (got {self.depth}); depth 1 is lookahead 1 (no early start)")
        self.depth = int(self.depth)

    def take(self, target: int) -> Optional[Any]:
        """The collector of cohort ``target`` if it was started early (else None: start it at the top)."""
        self.starts.pop(target, None)
        return self.pending.pop(target, None)

    def may_start(
        self,
        step: int,
        draft_limit: int,
        have_batch: bool,
        lane_fresh: bool,
        lane_cohort: Optional[int],
    ) -> bool:
        """Start cohort step+depth on the cohort lane of ``step`` now (its cohort is final)?

        Not across a checkpoint / validation boundary (``draft_limit``), not without its batch, not while the
        lane decodes fresh rows of this step (they would share the engine with the drafts; conservative), and
        only on a free lane.
        """
        target = step + self.depth
        return (
            target < draft_limit
            and have_batch
            and not lane_fresh
            and lane_cohort is None
            and target not in self.pending
        )

    def started(self, target: int, version: int, collector: Any, t: float) -> None:
        if target in self.pending:
            raise RuntimeError(f"lookahead {self.depth}: cohort {target} already drafting")
        if version != target - self.depth:
            raise RuntimeError(
                f"lookahead {self.depth}: cohort {target} must draft on theta_{target - self.depth}, "
                f"got theta_{version}"
            )
        self.pending[target] = collector
        self.versions[target] = version
        self.starts[target] = t

    def head_keep(self, step: int) -> list[int]:
        """Drafting versions whose LM head the learner must keep at the end of ``step``.

        theta_step drafts cohort step+depth (early) or step+1 (fallback); theta_{step-j} drafted cohort
        step+depth-j when it was started early (j = 1 .. depth-1). Cohort step is verified by now.
        """
        return list(range(step - self.depth + 1, step + 1))

    def keep_previous(self) -> int:
        """How many versions before the current one the learner keeps at the end of a step (``head_keep``)."""
        return self.depth - 1

    def leftover(self) -> list[int]:
        return sorted(self.pending)


def cohort_version(lane_cohort_version: Optional[tuple[str, int]], step: int) -> int:
    """The learner step whose weights drafted cohort ``step`` (continuation pieces carry it as q's version)."""
    if lane_cohort_version is None:
        return step - 1
    kind, v = lane_cohort_version
    if kind != "theta":
        raise RuntimeError(f"lookahead 2 drafts with current weights only, cohort {step} drafted at {kind}_{v}")
    return int(v)


@dataclass
class AdaptiveLookahead:
    """Phase-adaptive lookahead (``thundersync.cohort_lane_lookahead_adaptive``): per cohort, start it early
    (lookahead 2, drafts two versions old) or let it fall back to lookahead 1 (one version old), from the forecast
    per-token Var log p/q of age-2 drafts.

    Measured (long200, research/longrun/README.md): at low v lookahead 2 is faster (its two-period window cuts
    fewer drafts); at high v the age-2 drafts' late fails dominate and lookahead 1 is faster. The forecast is
    the ⊥-BV calibrator's per-age estimate (past steps only): the fresh age-2 EMA, else ``age_ratio`` x the fresh
    age-1 EMA, else the pooled EMA. Hysteresis: switch to lookahead 1 above ``v_hi``, back to 2 below ``v_lo``.

    Exactness: the choice is scheduling only. It reads earlier cohorts' verification statistics, never the
    drafts of the cohort it schedules; both outcomes are the existing lookahead-2 early start and its
    lookahead-1 fallback (the path every cohort after a checkpoint boundary already takes).
    """

    v_hi: float
    v_lo: float
    age_ratio: float = 3.5
    stale_steps: int = 5
    mode: int = 2
    last_forecast: float = 0.0
    switches: int = 0

    def __post_init__(self):
        if not (0.0 < self.v_lo <= self.v_hi):
            raise ValueError(f"adaptive lookahead: need 0 < v_lo <= v_hi (got {self.v_lo}, {self.v_hi})")
        if not self.age_ratio >= 1.0:
            raise ValueError(f"adaptive lookahead: age_ratio must be >= 1 (got {self.age_ratio})")

    def forecast(self, calib: Any, step: int) -> float:
        """Age-2-equivalent v for the next cohorts from the calibrator's state after ``step``'s verification."""
        by_age = getattr(calib, "by_age", None) or {}
        age_step = getattr(calib, "age_step", None) or {}

        def fresh(a: int) -> bool:
            return a in by_age and 0 <= step - age_step.get(a, -10**9) <= self.stale_steps

        if fresh(2):
            return float(by_age[2])
        if fresh(1):
            return float(self.age_ratio * by_age[1])
        return float(calib.estimate()) if calib is not None else 0.0

    def decide(self, v2: float) -> bool:
        """Update the mode from the forecast; True = start the next cohort early (lookahead 2)."""
        self.last_forecast = float(v2)
        new = self.mode
        if self.mode == 2 and v2 > self.v_hi:
            new = 1
        elif self.mode == 1 and v2 < self.v_lo:
            new = 2
        if new != self.mode:
            self.switches += 1
            self.mode = new
        return self.mode == 2
