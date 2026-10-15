"""Rank-0 driver timeline of the post-refit tail (sticky routing): measurement only.

One ``TailTrace`` per ``verify_sticky`` call (one step). Times are seconds from
the top of ``verify_sticky``; the trace holds O(rows + rounds + rejections)
Python records and never touches tokens beyond lengths, so its cost does not
grow with the model or the parallel layout (no collective, no GPU work).

What it answers (research/agent_tail.md):
  * when each rollout's plan completes (the "could train now" time of its group),
  * where the first rejection of each row sits and when it is detected:
    verify0 rejections are found when the drafts are verified at the step top,
    continuation rejections when their piece is verified (``_continue_sticky``),
  * how many decoded tokens a rejection wasted (decoded past the rejected
    position by the time it was detected; a lower bound, the stream keeps
    decoding while the piece is verified),
  * per continuation round: live streams, ready pieces, verification seconds.

Output (``summary``): scalar ``tail/*`` metrics plus three lists for offline
analysis, ``timeline/tail_rows`` [group, drafted, v0_accepted, v0_rejected,
cont_rejects, cont_tokens, final_len, t_complete], ``timeline/tail_rounds``
[t, live, ready, verify_s, rejects] and ``timeline/tail_rejects`` [source
(0 verify0, 1 continuation), position, t_detect, wasted_tokens, group].
"""

from __future__ import annotations

import itertools
import os
import time
from typing import Any, Hashable, Optional

# Debug (``THUNDERSYNC_TAIL_DUMP=<dir>``): the token contents of every drafted row
# of the step -- prompt, draft, final rollout, each rejection's position and dead
# tokens -- written by ``summary`` to <dir>/tail_<pid>_<n>.pt for offline replays
# (research/tail_speed/lookup_replay.py: lookup speculation on real tail rows).
# Rank-0 driver memory: one copy of the step's drafted tokens (int32).
_DUMP_SEQ = itertools.count()


def _q(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1) + 0.5))]


class TailTrace:
    def __init__(self, t0: Optional[float] = None):
        self.t0 = time.perf_counter() if t0 is None else t0
        self.marks: dict[str, float] = {}
        self._rows: dict[int, dict[str, Any]] = {}  # id(plan) -> record
        self._groups: dict[Hashable, int] = {}
        self.rounds: list[list[float]] = []
        self.rejects: list[list[float]] = []
        self.dump = os.environ.get("THUNDERSYNC_TAIL_DUMP") or None
        self._tok: dict[int, dict[str, Any]] = {}  # id(plan) -> tokens (dump only)

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def mark(self, name: str) -> None:
        self.marks[name] = self.now()

    def _group(self, key: Hashable) -> int:
        return self._groups.setdefault(key, len(self._groups))

    def verified(self, key: Hashable, plan, drafted: int, accepted: int) -> None:
        """A plan made by verify0 from a draft of ``drafted`` tokens (``accepted`` kept)."""
        g = self._group(key)
        rejected = accepted < drafted
        t = self.now()
        self._rows[id(plan)] = dict(
            group=g, drafted=drafted, v0_acc=accepted, v0_rej=int(rejected),
            cont_rej=0, cont_tok=0, final=len(plan.prefix),
            done=t if plan.complete else None,
        )
        if rejected:
            self.rejects.append([0, accepted, t, drafted - accepted, g])
        if self.dump:
            draft = list(getattr(plan, "draft_tokens", None) or [])
            self._tok[id(plan)] = dict(
                key=list(key), plan=plan, draft=draft, group=g,
                rejects=[(0, accepted, draft[accepted:])] if rejected else [],
            )

    def piece(
        self, plan, start: int, tokens: int, accepted: int, decoded: int, toks=None
    ) -> None:
        """A verified continuation piece: tokens [start, start + tokens) of its stream.

        ``decoded``: the stream's length when the piece was taken, so a
        rejection had ``decoded - (start + accepted)`` tokens decoded past it.
        """
        r = self._rows.get(id(plan))
        if r is None:  # a plan verify0 did not see (cannot happen today): track it now
            r = self._rows[id(plan)] = dict(
                group=-1, drafted=-1, v0_acc=-1, v0_rej=0, cont_rej=0, cont_tok=0,
                final=0, done=None,
            )
        r["cont_tok"] += tokens
        if accepted < tokens:
            r["cont_rej"] += 1
            pos = len(plan.prefix) - 1  # the residual token just appended
            self.rejects.append([1, pos, self.now(), max(0, decoded - start - accepted), r["group"]])
            if self.dump and id(plan) in self._tok and toks is not None:
                self._tok[id(plan)]["rejects"].append((1, pos, list(toks)[accepted:]))

    def completed(self, plan) -> None:
        r = self._rows.get(id(plan))
        if r is not None and r["done"] is None:
            r["done"] = self.now()
            r["final"] = len(plan.prefix)

    def round(self, live: int, ready: int, verify_s: float, rejects: int) -> None:
        self.rounds.append([self.now(), live, ready, verify_s, rejects])

    def summary(self) -> dict[str, Any]:
        rows = list(self._rows.values())
        end = self.now()
        for r in rows:  # plans the loop left incomplete (served later): count them at the end
            if r["done"] is None:
                r["done"] = end
        done = [r["done"] for r in rows]
        by_group: dict[int, float] = {}
        for r in rows:
            by_group[r["group"]] = max(by_group.get(r["group"], 0.0), r["done"])
        out: dict[str, Any] = {f"tail/{k}_s": v for k, v in self.marks.items()}
        out["tail/end_s"] = max(done, default=self.marks.get("verify0", 0.0))
        out["tail/rows"] = float(len(rows))
        out["tail/v0_rejected_rows"] = float(sum(r["v0_rej"] for r in rows))
        out["tail/cont_rejects"] = float(sum(r["cont_rej"] for r in rows))
        out["tail/rows_done_q50_s"] = _q(done, 0.5)
        out["tail/rows_done_q90_s"] = _q(done, 0.9)
        out["tail/group_done_q50_s"] = _q(list(by_group.values()), 0.5)
        out["tail/group_done_q90_s"] = _q(list(by_group.values()), 0.9)
        v0 = [r for r in self.rejects if r[0] == 0]
        cont = [r for r in self.rejects if r[0] == 1]
        out["tail/v0_wasted_tokens_max"] = float(max((r[3] for r in v0), default=0))
        out["tail/cont_wasted_tokens_mean"] = (
            sum(r[3] for r in cont) / len(cont) if cont else 0.0
        )
        if rows:
            s = max(rows, key=lambda r: r["done"])
            out.update(
                {
                    "tail/slowest_drafted": float(s["drafted"]),
                    "tail/slowest_v0_accepted": float(s["v0_acc"]),
                    "tail/slowest_v0_rejected": float(s["v0_rej"]),
                    "tail/slowest_cont_rejects": float(s["cont_rej"]),
                    "tail/slowest_cont_tokens": float(s["cont_tok"]),
                    "tail/slowest_final_len": float(s["final"]),
                }
            )
        out["timeline/tail_rows"] = [
            [r["group"], r["drafted"], r["v0_acc"], r["v0_rej"], r["cont_rej"], r["cont_tok"],
             r["final"], round(r["done"], 4)]
            for r in rows
        ]
        out["timeline/tail_rounds"] = [[round(x, 4) for x in r] for r in self.rounds]
        out["timeline/tail_rejects"] = [[round(x, 4) for x in r] for r in self.rejects]
        if self.dump:
            self._write_dump()
        return out

    def _write_dump(self) -> None:
        import torch

        def t(x):
            return torch.tensor(list(x), dtype=torch.int32)

        rows = []
        for pid, e in self._tok.items():
            r = self._rows.get(pid, {})
            rows.append(
                dict(
                    key=t(e["key"]), draft=t(e["draft"]), final=t(e["plan"].prefix),
                    group=e["group"], done=r.get("done"),
                    rejects=[(src, int(pos), t(dead)) for src, pos, dead in e["rejects"]],
                )
            )
        os.makedirs(self.dump, exist_ok=True)
        torch.save(
            {"rows": rows, "rounds": self.rounds, "marks": self.marks},
            os.path.join(self.dump, f"tail_{os.getpid()}_{next(_DUMP_SEQ):04d}.pt"),
        )
        self._tok.clear()
