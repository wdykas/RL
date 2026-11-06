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
"""Risk-first streaming verify0 (``thundersync.verify0_stream``).

verify0 (block verification of the cohort's drafts at theta_{k+1}) is split
into calls over whole prompts, riskiest first. Each prompt's plans are
published as soon as its pool is decided, so a rejected row's continuation
starts at its own decision and training of decided groups is dispatched
between the verification calls.

Exactness rules this module enforces (the argument and the enumeration are  and ``research/v0s/v0s_enum.py``):

* The schedule (which prompts go into which call, in which order) is a
  function of counts (draft lengths, drafts per prompt) and of earlier steps'
  verification results only (``Verify0Risk`` is updated after a step's
  verify0 has decided every pool). It never reads this step's draft tokens,
  log-ratios or decisions. Exception (F3, ``order="surprisal"``): with
  per-row seeds (THUNDERSYNC_ROW_SEEDS=1) every row's decision is a function
  of its p, its q and its own seed only, whatever call or position it rides in,
  so the schedule may also read q-side content (the drafted tokens' engine
  log-probs); it still never reads p, log-ratios or decisions
  (``research/v0s/f3_enum.py``).
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
from dataclasses import dataclass, field
from typing import Hashable, Sequence

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
    # q-side content score (order "surprisal", F3): max over the prompt's drafts of sum(-log q of the drafted
    # tokens). Reading content is allowed only with per-row seeds (THUNDERSYNC_ROW_SEEDS=1), under which a row's
    # decision does not depend on its call or position, so the order moves latency only.
    score: float = 0.0


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
    if order not in ("risk", "drafting", "surprisal"):
        raise ValueError(f"verify0_stream_order must be risk, drafting or surprisal (got {order!r})")
    if order == "surprisal":
        from thundersync_rl.speculative import row_seeds_enabled

        if not row_seeds_enabled():
            raise ValueError("verify0_stream_order=surprisal reads draft content: it needs THUNDERSYNC_ROW_SEEDS=1")

    def key(p):
        if order == "risk":
            return (-risk.prompt_score(p.lengths), p.first)
        if order == "surprisal":
            return (-p.score, p.first)
        return (0.0, p.first)

    ranked = sorted(prompts, key=key)
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




