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
"""Anchored deep pipeline, first build AEP-1E (``thundersync.anchor_epoch``).

One engine holds one weight version, the anchor theta_a (theta_0 in AEP-1E), for the whole epoch and is never refit.
Cohort c (the rollouts trained at step c) is drafted on theta_a from step max(0, c - lambda) to the cut at the end of
step c - 1: lambda cohorts are drafted at once. Cohort c's policy is the sticky F(theta_c; A_theta_a): the anchors at
every position are theta_a's plain routes on the row's own prefix. At step c the learner (theta_c) verifies the drafts
(⊥-BV, verify-in-train for complete groups, packed verify0 for the rest); rows needing more tokens continue on the
engine, which still holds theta_a, with the learner verifying each piece against F(theta_c; A_theta_a). Slots a pool
cannot fill and forced-fresh rows are continuations from the empty prefix (the engine never holds theta_c, so it never
samples p_c directly). Cohort 0 is sampled by the engine at theta_0 = the engine's version (plain routing).

Rules (each checked here, raising on a violation; exactness toy research/aep/aep_toy.py, tests/unit/test_aep.py):
- R1 (toy C2): the anchor version of a cohort is a function of the cohort index only (``anchor``); drafts carry it as
  their version tag at submission and verification refuses a draft whose tag differs (``check_drafts``).
- R2 (toy C1): a cohort's tails decode only on its anchor version: the engine must hold ``anchor(cohort)`` whenever
  a continuation of cohort c runs (``check_tail_version``); a refit while any cohort of the epoch is live is refused
  (``refuse_refit``).
- R3: the learner keeps theta_a's LM head (q rows) for the whole epoch (``head_keep``).
- R4: a cohort's drafts are cut to a per-cohort budget fixed by the config and the engine capacity before any draft
  of it decodes, so every submitted draft is admitted at once (no queueing, presence fixed) and the
  continuations of the cohort being completed never wait for a slot (``draft_budget``).
Epoch switches (long runs): epochs are P = E + 1 steps long and start at the config-fixed steps
w, w + P, w + 2P, ... (w = anchor_start). An epoch's last cohort is trained at its last step, and no cohort of the next
epoch is drafted before its start (``targets`` stops at the epoch end), so every epoch start is a quiescent boundary:
``switch`` re-anchors there (raw: theta_s; momentum: Psi_h(theta_s, m_s, v_s)), refusing if a cohort is in flight. The
anchor of cohort c is the start of the epoch containing c -- a function of the cohort index only (R1); a tail of an
old epoch's cohort on the new anchor is refused by ``check_tail_version`` (R2). Checkpoints / validation happen only at
epoch starts (``Boundaries`` with offset w and periods that are multiples of P), so a resume starts a fresh epoch
from the restored state: nothing in flight needs saving.
Version eviction (heads kept on the learner, logs on the engine): a version is evicted only when no row of any
cohort referencing it is drafting, tailing or pending verification (``live_versions``); under AEP that is the anchor
of every cohort in flight plus the cohort being trained.

Driver-side bookkeeping only (no engine, no learner, nothing per token); grpo_loop / speculative do the calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


class AnchorEpochError(RuntimeError):
    """A violated AEP rule (R1-R4) or an AEP state this build does not cover."""


@dataclass
class AnchorEpoch:
    """Schedule and invariants of one anchor epoch.

    ``lam``: cohorts drafted at once (thundersync.draft_lookahead under anchor_epoch). ``epoch``: E, the cohorts
    anchored at ``anchor0`` are anchor0 + 1 .. anchor0 + E. ``reserve_rows``: engine slots kept free besides one
    cohort's outputs (debug route-check prefills).
    """

    lam: int
    epoch: int
    anchor0: int = 0
    reserve_rows: int = 0
    # "raw": the anchor is theta_anchor0 (the engine holds the anchor step's target: cohort anchor0 is sampled
    # plainly). "momentum": theta_hat = Psi_h(theta_anchor0, m, v) (h zero-gradient AdamW steps of the optimizer state
    # at the anchor step; staged once at its top): the engine never holds a target, so cohort anchor0 is itself an
    # anchored cohort (age 0), drafted on theta_hat and verified against F(theta_anchor0; A_theta_hat).
    kind: str = "raw"
    h: int = 1
    # The weight version the engine holds (learner step whose theta it is).
    engine_version: int = 0
    # First epoch start w (anchor_start, absolute step); epochs start at w + i * (epoch + 1). Default: anchor0.
    start: Optional[int] = None
    switches: int = 0
    pending: dict[int, Any] = field(default_factory=dict)  # target -> collector (drafts running)
    drafted_at: dict[int, int] = field(default_factory=dict)  # target -> step its drafts started
    tags: dict[int, int] = field(default_factory=dict)  # target -> version tag its drafts carry
    budgets: dict[int, Optional[int]] = field(default_factory=dict)  # target -> drafts submitted cap
    consumed: set = field(default_factory=set)

    def __post_init__(self):
        if self.lam < 1:
            raise AnchorEpochError(f"anchor_epoch: lookahead lambda must be >= 1 (got {self.lam})")
        if self.epoch < 1:
            raise AnchorEpochError(f"anchor_epoch: epoch length must be >= 1 (got {self.epoch})")
        if self.kind not in ("raw", "momentum") or int(self.h) < 1:
            raise AnchorEpochError(f"anchor_epoch: anchor kind {self.kind!r} / h {self.h} not built")
        self.start = int(self.anchor0 if self.start is None else self.start)
        if not self.is_epoch_start(self.anchor0):
            raise AnchorEpochError(
                f"anchor_epoch: step {self.anchor0} is not an epoch start (epochs start at {self.start} + "
                f"k x {self.period}); a run resumes only at an epoch start"
            )
        self.engine_version = self.anchor0

    # -- epochs ---------------------------------------------------------------------------------------------------
    @property
    def period(self) -> int:
        """Steps per epoch: the anchor step plus ``epoch`` anchored cohorts."""
        return int(self.epoch) + 1

    def epoch_start(self, cohort: int) -> int:
        c = int(cohort)
        if c < self.start:
            raise AnchorEpochError(f"anchor_epoch: cohort {c} precedes the first epoch (anchor step {self.start})")
        return self.start + (c - self.start) // self.period * self.period

    def epoch_index(self, cohort: int) -> int:
        return (self.epoch_start(cohort) - self.start) // self.period

    def is_epoch_start(self, step: int) -> bool:
        step = int(step)
        return step >= self.start and (step - self.start) % self.period == 0

    def next_epoch_start(self, step: int) -> int:
        """The first epoch start >= ``step``."""
        step = int(step)
        if step <= self.start:
            return self.start
        return self.start + -(-(step - self.start) // self.period) * self.period

    def switch(self, step: int) -> None:
        """Re-anchor at epoch start ``step`` (quiescent: no cohort in flight). R2 for the old epoch's tails."""
        step = int(step)
        if not self.is_epoch_start(step):
            raise AnchorEpochError(f"anchor_epoch: switch at step {step}, which is not an epoch start")
        if self.pending:
            raise AnchorEpochError(
                f"anchor_epoch (R2): switch at step {step} with cohorts {self.in_flight()} in flight (their tails "
                f"would continue on the new anchor)"
            )
        if step != self.anchor0:
            self.switches += 1
        self.anchor0 = self.engine_version = step
        for d in (self.drafted_at, self.tags, self.budgets):
            for c in [c for c in d if c < step]:
                del d[c]
        self.consumed = {c for c in self.consumed if c >= step}

    def live_versions(self, step: int) -> list[int]:
        """Versions some row still needs (drafting, tailing or pending verification): never evict these.

        The anchors of every cohort in flight and of cohort ``step`` (being trained). A general rule (segment
        versions too): a version is evictable only when no row of any cohort referencing it is live.
        """
        out = {self.anchor(c) for c in list(self.pending) + [int(step)] if int(c) >= self.start}
        return sorted(v for v in out if v is not None)

    # -- R1: the schedule ----------------------------------------------------------------------------------------
    def anchor(self, cohort: int) -> Optional[int]:
        """The anchor (= drafting) version of ``cohort``: a function of the cohort index only.

        None for cohorts <= anchor0 (sampled by the engine at its own version, plain routing).
        """
        cohort = int(cohort)
        s = self.epoch_start(cohort)
        if cohort == s and self.kind == "raw":
            return None
        return s

    def age(self, cohort: int) -> int:
        """Routing lag of ``cohort``: steps between its anchor and its target (0 for engine-sampled cohorts)."""
        a = self.anchor(cohort)
        return 0 if a is None else int(cohort) - a

    def engine_holds_target(self, step: int) -> bool:
        """Whether the engine's weights are step's target (only the raw anchor step): rows may then decode plainly."""
        return self.kind == "raw" and int(step) == self.anchor0

    def anchored(self, cohort: int) -> bool:
        return int(cohort) >= self.anchor0 and self.anchor(cohort) is not None

    def targets(self, step: int, draft_limit: int, batches_ahead: int) -> list[int]:
        """Cohorts whose drafts start at ``step``: cohorts of (step, step + lam] not started yet, at most lam in flight.

        Raw anchor at step 0: cohorts 1..lam, then step + lam at every later step. Momentum anchor at step w: the
        anchor step's own cohort w too (cohorts w..w+lam-1; cohort w is cut once its drafts finish, inside step w).
        Not across a boundary (``draft_limit``), not past the epoch, not without its batch (``batches_ahead``
        batches after this step's; the anchor step's own batch is always there).
        """
        out = []
        first = step if (step == self.anchor0 and self.kind == "momentum") else step + 1
        for t in range(first, step + 1 + self.lam):
            if len(self.pending) + len(out) >= self.lam:
                break
            if t in self.pending or t in self.consumed or t >= draft_limit:
                continue
            if (t > step and t - step - 1 >= batches_ahead) or t > self.anchor0 + self.epoch:
                continue
            out.append(t)
        return out

    # -- R4: admission ---------------------------------------------------------------------------------------------
    def draft_budget(self, capacity: int, outputs: int) -> int:
        """Drafts one cohort may submit: (capacity - outputs - reserve) // lam, at least its primaries.

        ``capacity``: requests the engine runs at once (max_requests x engine DP; 0 = unknown raises).
        ``outputs``: rows of one cohort (its primaries, prompts x G). At any time the engine holds at most lam
        drafting cohorts plus the outputs of the cohort being completed (each output slot has at most one live
        stream), so with this budget nothing ever queues. A function of the config and a constant only.
        """
        if not capacity:
            raise AnchorEpochError(
                "anchor_epoch: the engine's request capacity is unknown (engine not initialized?): the per-cohort "
                "draft budget cannot be fixed, and without it drafts of lambda cohorts could queue (R4)"
            )
        d = (int(capacity) - int(outputs) - int(self.reserve_rows)) // self.lam
        if d < outputs:
            raise AnchorEpochError(
                f"anchor_epoch: engine capacity {capacity} (max_requests x engine DP) cannot hold lambda={self.lam} "
                f"cohorts of >= {outputs} drafts plus one cohort's {outputs} outputs (+{self.reserve_rows} reserved): "
                f"needs >= {(self.lam + 1) * outputs + self.reserve_rows}; raise "
                "policy.generation.mcore_generation_config.max_requests or lower lambda"
            )
        return d

    def started(self, target: int, step: int, tag: int, collector: Any, budget: Optional[int]) -> None:
        if target in self.pending or target in self.consumed:
            raise AnchorEpochError(f"anchor_epoch: cohort {target} already drafted")
        want = self.anchor(target)
        if tag != want:
            raise AnchorEpochError(
                f"anchor_epoch (R1): cohort {target} drafts must carry its anchor version {want}, got {tag}"
            )
        if self.engine_version != want:
            raise AnchorEpochError(
                f"anchor_epoch: cohort {target} would draft on version {self.engine_version}, its anchor is {want}"
            )
        self.pending[target] = collector
        self.drafted_at[target] = int(step)
        self.tags[target] = int(tag)
        self.budgets[target] = budget

    def take(self, target: int) -> Optional[Any]:
        """The collector of cohort ``target`` (its deadline is this step's end), or None if never started."""
        c = self.pending.pop(target, None)
        if c is not None:
            self.consumed.add(target)
        return c

    def in_flight(self) -> list[int]:
        return sorted(self.pending)

    # -- R1 / R2 checks at verification and tails -----------------------------------------------------------------
    def check_drafts(self, cohort: int, versions) -> None:
        """Every draft of ``cohort`` carries the cohort's scheduled anchor version (R1)."""
        want = self.anchor(cohort)
        bad = sorted({int(v) for v in versions if int(v) != want})
        if bad:
            raise AnchorEpochError(
                f"anchor_epoch (R1): cohort {cohort} has drafts of versions {bad}; its anchor is {want} (a row's "
                "anchor version must be a function of the cohort index, never chosen per row)"
            )

    def check_tail_version(self, cohort: int, engine_version: int) -> None:
        """Continuations of ``cohort`` decode on its anchor version (R2)."""
        want = self.anchor(cohort)
        if want is not None and int(engine_version) != want:
            raise AnchorEpochError(
                f"anchor_epoch (R2): a continuation of cohort {cohort} would decode on version {engine_version}; "
                f"its anchor version is {want} (tails never continue with another epoch's anchors)"
            )

    def refuse_refit(self, why: str) -> None:
        """Any engine refit inside the epoch (R2: the engine must keep theta_a while a cohort is live)."""
        raise AnchorEpochError(
            f"anchor_epoch (R2): engine refit requested ({why}) while the engine holds anchor theta_{self.anchor0} "
            f"for cohorts {self.in_flight() or 'of the epoch'}; the engine is re-anchored only at epoch starts"
        )

    # -- R3 ---------------------------------------------------------------------------------------------------------
    def head_keep(self) -> list[int]:
        """Drafting versions whose LM head the learner keeps (q rows): the anchor, for the whole epoch."""
        return [self.anchor0]

    def leftover(self) -> list[int]:
        return sorted(self.pending)
