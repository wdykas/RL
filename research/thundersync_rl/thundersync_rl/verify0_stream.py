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
"""Risk-first streaming verify0 (``thundersync.verify0_stream``, DESIGN 14z).

verify0 (block verification of the cohort's drafts at theta_{k+1}) is split
into calls over whole prompts, riskiest first. Each prompt's plans are
published as soon as its pool is decided, so a rejected row's continuation
starts at its own decision and training of decided groups is dispatched
between the verification calls.

Exactness rules this module enforces (the argument and the enumeration are in
DESIGN 14z and ``research/v0s/v0s_enum.py``):

* The schedule (which prompts go into which call, in which order) is a
  function of counts (draft lengths, drafts per prompt) and of earlier steps'
  verification results only (``Verify0Risk`` is updated after a step's
  verify0 has decided every pool). It never reads this step's draft tokens,
  log-ratios or decisions.
* A prompt's pool is never split across calls and its drafts keep drafting
  order: ⊥-BV takes the first outputs in that order, so a content-dependent
  order inside a pool would change the output law.
* Every drafted row is decided whatever the order (no budget, deadline or
  other order-dependent switch to a different sampler), so order only moves
  latency. Each call draws a fresh verification seed.
* The canonical training order (``StreamOrder``) is a function of the
  schedule and of the decisions: ready groups of call 1, of call 2, ..., then
  the groups with tails, then groups with rows never drafted.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Iterable, Optional, Sequence

# Draft-length bucket upper edges (tokens); the last bucket is open.
LENGTH_EDGES: tuple[int, ...] = (256, 1024, 2048, 4096, 8192)


def length_bucket(n: int) -> int:
    for b, edge in enumerate(LENGTH_EDGES):
        if n < edge:
            return b
    return len(LENGTH_EDGES)


@dataclass
class Verify0Risk:
    """Past-only predictor of a draft's tail work by length bucket.

    ``whole_rate(L)`` = (whole_b + w * exp(-hazard * L)) / (n_b + w): the past
    steps' whole-acceptance fraction of decided drafts in L's length bucket,
    shrunk toward a constant-hazard prior (hazard ~1-2e-5 per token at 4-8k
    measured on full Nano; high_gap_tokens §0). ``remainder(L)``: the past
    steps' mean tokens decoded after a verify0 rejection (final length -
    accepted) in L's bucket, shrunk toward L. ``row_risk`` = P(reject) x
    remainder. Updated only with finished steps, so the schedule of step k
    reads steps < k only. Shared by the lanes (one object).
    """

    hazard: float = 2e-5
    prior_weight: float = 4.0
    remainder_prior_weight: float = 2.0
    n: list[int] = field(default_factory=lambda: [0] * (len(LENGTH_EDGES) + 1))
    whole: list[int] = field(default_factory=lambda: [0] * (len(LENGTH_EDGES) + 1))
    rejected: list[int] = field(default_factory=lambda: [0] * (len(LENGTH_EDGES) + 1))
    remainder_sum: list[float] = field(default_factory=lambda: [0.0] * (len(LENGTH_EDGES) + 1))
    steps: int = 0

    def whole_rate(self, length: int) -> float:
        b = length_bucket(length)
        prior = math.exp(-self.hazard * max(length, 0))
        return (self.whole[b] + self.prior_weight * prior) / (self.n[b] + self.prior_weight)

    def remainder(self, length: int) -> float:
        """Mean tokens decoded after a verify0 rejection of a draft this long (past steps)."""
        b = length_bucket(length)
        w = self.remainder_prior_weight
        return (self.remainder_sum[b] + w * length) / (self.rejected[b] + w)

    def row_risk(self, length: int) -> float:
        """Expected tail tokens of this draft: P(reject) x remainder after a rejection."""
        return (1.0 - self.whole_rate(length)) * self.remainder(length)

    def prompt_score(self, lengths: Sequence[int]) -> float:
        """A prompt's priority: expected tokens to redo over its drafts.

        A sum, not a max: at an 8k cap most prompts have a capped draft, so the
        max ties almost everywhere; the sum ranks prompts with more long drafts
        (more chances of an early rejection with a cap-long remainder) first.
        """
        return sum(self.row_risk(n) for n in lengths)

    def update(self, rows: Sequence[tuple]) -> None:
        """Fold in one finished step: (drafted, accepted[, final length]) per decided output.

        The final length (after the row's tail) sets the remainder statistics of
        rejected rows; rows without it count for the acceptance rate only.
        """
        for row in rows:
            drafted, accepted = int(row[0]), int(row[1])
            b = length_bucket(drafted)
            self.n[b] += 1
            if accepted >= drafted:
                self.whole[b] += 1
            elif len(row) > 2 and row[2] is not None:
                self.rejected[b] += 1
                self.remainder_sum[b] += max(0, int(row[2]) - accepted)
        self.steps += 1

    def snapshot(self) -> dict[str, float]:
        out = {"v0s/risk_steps": float(self.steps)}
        for b in range(len(self.n)):
            out[f"v0s/risk_whole_b{b}"] = self.whole_rate(
                (LENGTH_EDGES[b] - 1) if b < len(LENGTH_EDGES) else LENGTH_EDGES[-1]
            )
        return out


@dataclass(frozen=True)
class PromptCounts:
    """What the schedule may read about one prompt: counts only."""

    key: Hashable
    first: int  # drafting-order index of the prompt's first draft
    lengths: tuple[int, ...]  # drafted tokens of each draft, drafting order
    tokens: int  # verification forward tokens (prompt + draft) of all its drafts


def schedule_chunks(
    prompts: Sequence[PromptCounts],
    risk: Verify0Risk,
    first_tokens: int,
    chunk_tokens: int,
    order: str = "risk",
) -> list[list[Hashable]]:
    """Verification calls of verify0: lists of whole prompts, riskiest first.

    Prompts are sorted by (-risk score, drafting index) (``order="drafting"``:
    by drafting index only, the control) and cut into chunks of
    at least ``first_tokens`` (the first) / ``chunk_tokens`` (the others)
    forward tokens. A function of ``PromptCounts`` (counts) and ``risk`` (past
    steps) only.
    """
    if first_tokens <= 0 or chunk_tokens <= 0:
        raise ValueError("verify0_stream chunk sizes must be positive")
    if order not in ("risk", "drafting"):
        raise ValueError(f"verify0_stream_order must be risk or drafting (got {order!r})")
    ranked = sorted(
        prompts,
        key=lambda p: (-risk.prompt_score(p.lengths) if order == "risk" else 0.0, p.first),
    )
    chunks: list[list[Hashable]] = []
    cur: list[Hashable] = []
    tok = 0
    for p in ranked:
        cur.append(p.key)
        tok += p.tokens
        if tok >= (first_tokens if not chunks else chunk_tokens):
            chunks.append(cur)
            cur, tok = [], 0
    if cur:
        chunks.append(cur)
    return chunks


class StreamOrder:
    """The canonical training order of a streamed verify0, grown per decision.

    ``row_keys[p]``: prompt key of group p; ``need[key]``: rollouts of that key
    in the batch. ``decided(info)`` (info[key] = (plans, tail plans) at the
    key's decision) appends the newly decided groups that are ready (enough
    plans, none needing a continuation), in group index order; ``close()``
    appends the groups with tails, then the groups with a rollout never
    drafted (decoded at theta_k), each in group index order. Blocks are
    ``groups_per_block`` groups within each appended segment, or one segment
    per append with ``block_tokens`` (the planner cuts it by tokens). The
    order is a function of the schedule (counts, past) and of the decisions.

    ``fresh_keys``: prompts whose rollouts decode at theta_k whatever their
    plans (``THUNDERSYNC_DEBUG_FRESH_MOD`` rows ignore their drafts). Their
    groups are never ready at a decision and are listed in ``fresh``: with
    cohort lanes ``fresh`` decides whether the cohort lane takes theta_k, so
    every group with a row decoded at theta_k must be in it (v0s-xlanes at
    6fc1ae9c decoded these rows on the drafting weights).
    """

    def __init__(
        self,
        row_keys: Sequence[Hashable],
        need: Counter,
        groups_per_block: int,
        block_tokens: Optional[int] = None,
        fresh_keys: Iterable[Hashable] = (),
    ):
        self.row_keys = list(row_keys)
        self.need = need
        self.fresh_keys = frozenset(fresh_keys)
        self.bg = max(1, int(groups_per_block))
        self.block_tokens = block_tokens
        self.order: list[int] = []
        self.blocks: list[int] = []
        self.tail_groups: list[int] = []
        self.fresh: list[int] = []
        self.info: dict[Hashable, tuple[int, int]] = {}
        self.closed = False
        self._segments: list[int] = []  # sizes of the appended segments
        self._planners: list[tuple[Any, int]] = []
        self.wake: Optional[Callable[[], None]] = None

    def attach(self, planner, step: int) -> None:
        planner.begin_order(self.block_tokens)
        self._planners.append((planner, step))
        at = 0
        for b in self._segments:
            planner.extend_order([(step, p) for p in self.order[at : at + b]], self._split(b))
            at += b
        if self.closed:
            planner.close_order()

    def _split(self, n: int) -> list[int]:
        if self.block_tokens:
            return [n]
        return [min(self.bg, n - i) for i in range(0, n, self.bg)]

    def _extend(self, seg: list[int]) -> None:
        if not seg:
            return
        self._segments.append(len(seg))
        self.order.extend(seg)
        self.blocks.extend(self._split(len(seg)))
        for planner, step in self._planners:
            planner.extend_order([(step, p) for p in seg], self._split(len(seg)))
        if self.wake is not None:
            self.wake()

    def _ready(self, key) -> bool:
        n, tails = self.info.get(key, (0, 0))
        return n >= self.need[key] and tails == 0 and key not in self.fresh_keys

    def _fresh(self, key) -> bool:
        """A rollout of ``key`` decodes at theta_k (too few plans, or a fresh key)."""
        return key in self.fresh_keys or self.info.get(key, (0, 0))[0] < self.need[key]

    def decided(self, info: dict[Hashable, tuple[int, int]]) -> list[int]:
        if self.closed:
            raise RuntimeError("StreamOrder: a decision after close")
        dup = set(info) & set(self.info)
        if dup:
            raise RuntimeError(f"StreamOrder: prompts decided twice: {sorted(map(repr, dup))[:3]}")
        self.info.update(info)
        seg = [p for p, k in enumerate(self.row_keys) if k in info and self._ready(k)]
        self._extend(seg)
        return seg

    def close(self) -> None:
        if self.closed:
            return
        listed = set(self.order)
        rest = [p for p in range(len(self.row_keys)) if p not in listed]
        self.fresh = [p for p in rest if self._fresh(self.row_keys[p])]
        fresh = set(self.fresh)
        self.tail_groups[:] = [p for p in rest if p not in fresh]
        self._extend(list(self.tail_groups))
        self._extend(list(self.fresh))
        if sorted(self.order) != list(range(len(self.row_keys))):
            raise RuntimeError("StreamOrder: the order does not list every group once")
        self.closed = True
        for planner, _ in self._planners:
            planner.close_order()
        if self.wake is not None:
            self.wake()


class TailFeed:
    """Plans handed to the running continuation rounds as their pools are decided."""

    def __init__(self):
        self.items: list[tuple[Hashable, Any]] = []
        self.closed = False
        self.admitted: list[tuple[Hashable, Any]] = []

    def put(self, items) -> None:
        if self.closed:
            raise RuntimeError("TailFeed: put after close")
        self.items.extend(items)

    def take(self) -> list[tuple[Hashable, Any]]:
        out, self.items = self.items, []
        self.admitted.extend(out)
        return out

    def close(self) -> None:
        self.closed = True

    @property
    def open(self) -> bool:
        return bool(self.items) or not self.closed
