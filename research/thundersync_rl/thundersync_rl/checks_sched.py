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
"""Exactness-check schedule for long runs and per-step decomposition (ts-nano-scale).

Checks are environment switches read at call time in the driver and in every worker
(THUNDERSYNC_MISMATCH_CHECK, _ENGINE_Q_CHECK, _ROUTE_CHECK_ROWS, _LAW_CHECK, _ENGINE_PREFILL_CHECK,
_TRAIN_LP_CHECK, ...). ``THUNDERSYNC_CHECK_EVERY=k`` keeps the full set (the environment the run
started with) on the steps with ``step % k == THUNDERSYNC_CHECK_PHASE`` (default k // 2) and
switches to the cheap set ``THUNDERSYNC_CHECK_OFF_ENV`` ("K=V,K=V"; empty: every check off) on the
others, in the driver and in every learner / engine worker (``set_check_env``) before the step's
first call. Which steps are checked is a function of the step number only (never of outcomes), so
each checked step is a valid sample of the run's law.

Per rank: one RPC per worker per step with a dict of <= 12 strings; nothing per token.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

# Switches the schedule owns. Anything else in the environment is left alone.
CHECK_VARS = (
    "THUNDERSYNC_MISMATCH_CHECK",
    "THUNDERSYNC_MISMATCH_CHECK_ROWS",
    "THUNDERSYNC_ENGINE_Q_CHECK",
    "THUNDERSYNC_ROUTE_CHECK_ROWS",
    "THUNDERSYNC_LAW_CHECK",
    "THUNDERSYNC_EMISSION_LAW_EVERY",
    "THUNDERSYNC_ENGINE_PREFILL_CHECK",
    "THUNDERSYNC_ENGINE_PREFILL_CHECK_ROWS",
    "THUNDERSYNC_ENGINE_PREFILL_CHECK_TOKENS",
    "THUNDERSYNC_TRAIN_LP_CHECK",
)


def parse_env_list(s: str) -> dict[str, str]:
    out = {}
    for item in (s or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        k, _, v = item.partition("=")
        k = k.strip()
        if k not in CHECK_VARS:
            raise ValueError(f"THUNDERSYNC_CHECK_OFF_ENV: {k!r} is not a check switch ({', '.join(CHECK_VARS)})")
        out[k] = v.strip()
    return out


def apply_env(env: dict[str, Optional[str]], environ: Any = None) -> None:
    """Set (value) / unset (None) every switch of ``env`` in ``environ`` (default os.environ)."""
    environ = os.environ if environ is None else environ
    for k, v in env.items():
        if v is None:
            environ.pop(k, None)
        else:
            environ[k] = v


class CheckSchedule:
    """Full checks on 1 step in ``every``; the cheap set elsewhere (module docstring)."""

    def __init__(self, every: int, phase: Optional[int] = None, full: Optional[dict] = None,
                 off: Optional[dict] = None):
        if every < 1:
            raise ValueError("THUNDERSYNC_CHECK_EVERY must be >= 1")
        self.every = int(every)
        self.phase = int(every // 2 if phase is None else phase) % self.every
        self.full = dict(full or {})
        self.off = dict(off or {})
        self.current: Optional[bool] = None

    @classmethod
    def from_env(cls, environ: Any = None) -> Optional["CheckSchedule"]:
        environ = os.environ if environ is None else environ
        every = int(environ.get("THUNDERSYNC_CHECK_EVERY", "0") or 0)
        if every <= 1:
            return None
        phase = environ.get("THUNDERSYNC_CHECK_PHASE")
        full = {k: environ.get(k) for k in CHECK_VARS}
        return cls(every, None if phase in (None, "") else int(phase), full,
                   parse_env_list(environ.get("THUNDERSYNC_CHECK_OFF_ENV", "")))

    def is_full(self, step: int) -> bool:
        return step % self.every == self.phase

    def env_for(self, step: int) -> dict[str, Optional[str]]:
        if self.is_full(step):
            return dict(self.full)
        return {k: self.off.get(k) for k in CHECK_VARS}

    def apply(self, step: int, setters: Iterable[Callable[[dict], Any]] = ()) -> bool:
        """Switch the driver (and every worker through ``setters``) to the step's set; returns is_full."""
        full = self.is_full(step)
        if full != self.current:
            env = self.env_for(step)
            apply_env(env)
            for setter in setters:
                setter(env)
            self.current = full
        return full


# Timed phases of the main loop that run at the top of the step, before ``_run_one_step`` (outside
# ``time/step_total``): verify0 / sticky verification, the deferred draft handout, the lagged baseline's
# first rollouts. The refit and the cut are timed inside the step body on the streaming paths.
STEP_TOP_KEYS = ("time/verify", "time/draft_handout_wait", "time/first_rollout")


def step_decomposition(
    m: dict[str, Any], outside_keys: Iterable[str] = STEP_TOP_KEYS, verify0_in_body: bool = False
) -> dict[str, float]:
    """Critical-path split of one step's wall time (``time/iteration``) into disjoint parts.

    Step body (time/step_total) = rollout window (time/rollout: step start -> last rollout) + post-rollout
    (time/post_rollout_to_step: training after the last rollout, incl. the debug checks time/mismatch_check
    and the finish time/finish_step). The rollout window splits into verify0 (tail/verify0_s) and the rest
    (tails / decode) only when verify0 runs inside the body (``verify0_in_body``: streaming verify0);
    otherwise verify0 runs at the top of the step, inside time/verify, and decomp/verify0_s is 0.
    Outside the body (iteration - time/step_total): ``outside_keys`` are the caller's timed phases that run
    there (default STEP_TOP_KEYS; the caller adds time/refit where the refit runs there);
    ``decomp/step_top_s`` is their sum (clipped at the outside time). ``decomp/unattributed_s`` is the
    outside time they do not name plus any body time outside rollout + post (clipped at 0);
    ``decomp/attributed_frac`` = 1 - unattributed / iteration.
    """
    it = m.get("time/iteration")
    if not isinstance(it, (int, float)) or it <= 0:
        return {}
    g = lambda k: float(m.get(k) or 0.0)  # noqa: E731
    roll, post, body = g("time/rollout"), g("time/post_rollout_to_step"), g("time/step_total")
    v0 = min(g("tail/verify0_s"), roll) if verify0_in_body else 0.0
    chk = min(g("time/mismatch_check"), post)
    outside = max(0.0, it - body)
    top = min(sum(g(k) for k in dict.fromkeys(outside_keys)), outside)
    unattr = (outside - top) + max(0.0, body - roll - post)
    return {
        "decomp/verify0_s": v0,
        "decomp/rollout_rest_s": roll - v0,
        "decomp/post_train_s": post - chk,
        "decomp/checks_s": chk,
        "decomp/outside_s": outside,
        "decomp/step_top_s": top,
        "decomp/learner_train_busy_s": g("learner/busy_stream_accumulate_s"),
        "decomp/learner_verify_busy_s": g("learner/busy_posted_call_s"),
        "decomp/unattributed_s": unattr,
        "decomp/attributed_frac": max(0.0, 1.0 - unattr / it),
    }
