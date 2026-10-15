"""Drafting-window pacing: hold a fitted target window at the draft cut.

The drafting window of cohort k+1 is the time from its draft start (top of
step k) to the cut (end of step k's training), so it equals step k's length.
A step whose cohort was fully drafted is fast, which leaves the next cohort a
window shorter than a full-length draft: that cohort is cut a little short
(every stream at the same progress), pays a whole continuation phase, and is
slow, which leaves a long window again. Steps alternate fast/slow
(research/agent_window_pace.md).

``WindowPacer`` breaks the loop: the cut waits until ``t_start + W`` where ``W``
is the expected time for every draft to reach the cap, fitted online from
earlier cuts, and is released earlier once no draft is running. It never
waits when the deadline is more than ``max_wait_frac * W`` away (the cut then
happens now, as without pacing: a partial wait would not avoid the
continuation phase).

Exactness: the deadline is a function of earlier steps' timings and draft
lengths only. The early release depends on the current cohort through one
count (unfinished drafts) and only when it is zero: the drafts handed out are
then pathwise the ones a cut at the deadline would hand out (a finished draft
does not change), so the verified cohort is that of a fixed-deadline cut.

Cost: driver-only scalar state; during a wait, one O(1) count call to the
drafting engine's rank 0 per poll (``count_unfinished_drafts``), independent of
model size and parallelism.
"""

from __future__ import annotations

import time
from typing import Callable, Optional


class WindowPacer:
    def __init__(
        self,
        margin: float = 0.05,
        max_wait_frac: float = 0.3,
        init_s: Optional[float] = None,
        decay: float = 0.5,
        poll_s: float = 0.02,
        clock: Callable[[], float] = time.perf_counter,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if margin < 0 or max_wait_frac < 0 or not 0 < decay <= 1 or (init_s is not None and init_s <= 0):
            raise ValueError(
                "WindowPacer: margin, max_wait_frac >= 0, 0 < decay <= 1, init_s > 0 or None"
            )
        self.margin = margin
        self.max_wait_frac = max_wait_frac
        self.decay = decay
        self.poll_s = poll_s
        self.clock = clock
        self.sleep = sleep
        # Fitted seconds from draft start until every draft reaches the cap.
        # ``init_s``: a prior (config, not measured) so pacing can act before
        # the first cut-short cohort; later cuts refit it as usual.
        self.full_window_s: Optional[float] = init_s

    def target_s(self) -> Optional[float]:
        if self.full_window_s is None:
            return None
        return self.full_window_s * (1.0 + self.margin)

    def hold(self, t_start: float, unfinished: Callable[[], int]) -> dict[str, float]:
        """Wait (if worth it) until the target window or until no draft runs.

        Called at the cut; ``t_start`` is the cohort's draft start. Returns
        stats (``pace/*``).
        """
        now = self.clock()
        target = self.target_s()
        stats = {"pace/target_s": float(target or 0.0), "pace/wait_s": 0.0,
                 "pace/released": 0.0, "pace/deadline": 0.0, "pace/skipped": 0.0}
        if target is None:
            return stats
        deadline = t_start + target
        if deadline <= now:
            return stats
        if deadline - now > self.max_wait_frac * target:
            stats["pace/skipped"] = 1.0
            return stats
        while True:
            if unfinished() == 0:
                stats["pace/released"] = 1.0
                break
            t = self.clock()
            if t >= deadline:
                stats["pace/deadline"] = 1.0
                break
            self.sleep(min(self.poll_s, deadline - t))
        stats["pace/wait_s"] = self.clock() - now
        return stats

    def observe(
        self,
        elapsed_s: float,
        running_lens: list[int],
        cap: int,
        all_done: bool,
    ) -> None:
        """One cut: seconds since draft start and the unfinished drafts' lengths.

        ``all_done``: no draft was running at the cut; ``elapsed_s`` is then an
        upper bound of the full window (no fit unless it lowers the estimate).
        Otherwise the full window is extrapolated: the longest running draft
        gives the per-token rate (it ran the whole window), the shortest the
        tokens still to decode (later admission waves start late).
        """
        if elapsed_s <= 0:
            return
        if all_done or not running_lens:
            if self.full_window_s is not None and elapsed_s < self.full_window_s:
                self._blend(elapsed_s)
            return
        longest = max(running_lens)
        if longest <= 0:
            return
        per_token = elapsed_s / longest
        est = elapsed_s + max(0, cap - min(running_lens)) * per_token
        if self.full_window_s is None or est >= self.full_window_s:
            # Late is expensive (a continuation phase), early nearly free (the
            # release): move up at once, down slowly.
            self.full_window_s = est
        else:
            self._blend(est)

    def _blend(self, x: float) -> None:
        self.full_window_s = (1.0 - self.decay) * self.full_window_s + self.decay * x


def share_window_pacer(specs) -> None:
    """One pacer for every lane: the full drafting window is a property of the
    drafter (engine, cohort size, cap), not of a lane. Past-only: lanes cut one
    after another, each step's deadline uses earlier cuts.
    """
    shared = next((s._pacer for s in specs if getattr(s, "_pacer", None) is not None), None)
    for s in specs:
        if getattr(s, "_pacer", None) is not None:
            s._pacer = shared
