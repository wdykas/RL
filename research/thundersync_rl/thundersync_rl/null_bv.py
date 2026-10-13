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
"""Pooled null-augmented block verification (⊥-BV) of a prompt's drafts.

Per prompt: G output slots (the GRPO group, fixed size) and N = round(K G)
drafts X_1..X_N from q, in a fixed index order chosen before verification
(the drafting order). Draft j is block-verified against

    p_hat_j = p / M_j  (+)  (1 - 1/M_j) * BOT        (BOT only at position 1)

(``block_verification``, ``null_m``). It yields an exact p-sample with
probability exactly 1/M_j (whole draft, or kept prefix + residual token + a p
tail decode), or BOT (discarded, nothing sampled). Successes fill the slots in
index order; once all G are filled the remaining drafts are not verified.
Slots still empty when the drafts run out are decoded fresh from p.

Exactness [derived; ``tests/unit/test_null_bv.py`` enumerates it]: given the
past, a draft's non-BOT output is p-distributed and independent of the past,
with success probability 1/M_j. M_j and the stopping rule depend only on the
counts (drafts left d, slots left s), never on draft contents, so the
successes taken in index order are iid p (optional sampling), and the G slots
are G iid p-samples. Forbidden: reordering drafts or choosing M_j by content
(length, reward, acceptance).

M_j policy: a dynamic program over (d, s) minimising the expected
critical-path cost sum over outputs of P(tail decode) * E[tail / cap] plus a
fresh full decode per unfilled slot, with the per-success tail-decode
probability from the Brownian model of log w (``agent_multi_draft.md``
section 2.3, validated on real rows): log w_t under p is Brownian with drift
+a^2/2 and variance a^2 = L v over a draft of length L. M = 1 whenever
d <= s (never risk a fresh decode while drafts suffice).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

# Envelopes the DP may pick (1.0 = plain block verification).
M_GRID = (1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0)


def _sf(x: float) -> float:
    """Standard normal survival function."""
    return 0.5 * math.erfc(x / math.sqrt(2.0))


def _pmax(m: float, a: float) -> float:
    """P(max_t log w_t > m), log w Brownian with drift a^2/2, variance a^2 (m >= 0)."""
    h = a * a / 2
    # e^m sf(.) underflows gracefully; guard the product for large m.
    tail = _sf((m + h) / a)
    return _sf((m - h) / a) + (math.exp(m) * tail if tail > 0 else 0.0)


def _integrate(f, lo: float, hi: float, n: int = 2000) -> float:
    """Composite Simpson rule on [lo, hi] (n even)."""
    hstep = (hi - lo) / n
    s = f(lo) + f(hi)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(lo + i * hstep)
    return s * hstep / 3


def tail_decode_prob(M: float, a: float) -> float:
    """Per-success probability that the output needs a tail decode.

    M = 1 (plain BV): 1 - E_p[1/W*]; M > 1: E_p[(1 - M/W*)+], W* = max_t w_t.
    """
    if a <= 0:
        return 0.0
    lm = math.log(M)
    hi = lm + 50 * a + 5
    if M == 1.0:
        return _integrate(lambda m: _pmax(m, a) * math.exp(-m), 0.0, hi)
    return _integrate(lambda m: _pmax(m, a) * M * math.exp(-m), lm, hi)


@dataclass
class NullPolicy:
    """M_j as a function of (drafts left d, slots left s) only.

    ``logratio_var``: per-token Var log p/q (v); ``ref_len``: draft length the
    cost model uses (the rollout cap). ``fixed_m``: constant envelope instead of
    the DP (still M = 1 when d <= s).
    """

    logratio_var: float = 8e-7
    ref_len: int = 32768
    fixed_m: Optional[float] = None
    grid: tuple[float, ...] = M_GRID
    _table: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    _value: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    _cost: Optional[dict[float, float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.fixed_m is not None and not self.fixed_m >= 1.0:
            raise ValueError(f"null_bv fixed M must be >= 1 (got {self.fixed_m})")
        if any(not m >= 1.0 for m in self.grid):
            raise ValueError(f"null_bv M grid must be >= 1 (got {self.grid})")

    def costs(self) -> dict[float, float]:
        """Expected critical-path cost of one verified draft at envelope M
        (units: one full fresh decode = 1)."""
        if self._cost is None:
            a = math.sqrt(max(self.ref_len * self.logratio_var, 0.0))
            # Tail length given a tail decode: BV rejections are front-loaded
            # (E[1 - tau/g | fail] ~ 0.6); ⊥-BV ones sit late (tau > g/2: ~0.25).
            self._cost = {
                M: tail_decode_prob(M, a) * (0.6 if M == 1.0 else 0.25) / M
                for M in self.grid
            }
        return self._cost

    def value(self, d: int, s: int) -> float:
        if s <= 0:
            return 0.0
        if d <= 0:
            return float(s)
        self.m(d, s)
        return self._value[(d, s)]

    def m(self, d: int, s: int) -> float:
        """Envelope for the next draft with ``d`` drafts (this one included) and
        ``s`` empty slots left."""
        if s <= 0:
            raise ValueError("no slot left: the draft must not be verified")
        if d <= s:
            return 1.0
        if self.fixed_m is not None:
            return float(self.fixed_m)
        key = (d, s)
        if key not in self._table:
            # Fill bottom-up so the recursion depth stays bounded.
            for dd in range(1, d + 1):
                for ss in range(1, s + 1):
                    if (dd, ss) in self._table:
                        continue
                    best, bm = math.inf, 1.0
                    for M in (1.0,) if dd <= ss else self.grid:
                        val = (
                            self.costs()[M]
                            + self.value(dd - 1, ss - 1) / M
                            + (1 - 1 / M) * self.value(dd - 1, ss)
                        )
                        if val < best:
                            best, bm = val, M
                    self._table[(dd, ss)] = bm
                    self._value[(dd, ss)] = best
        return self._table[key]


def drafts_per_prompt(k: float, group_size: int) -> int:
    """N = max(G, round(K G)) drafts per GRPO group."""
    if not k >= 1.0:
        raise ValueError(f"null_bv_drafts_per_output must be >= 1 (got {k})")
    return max(group_size, int(round(k * group_size)))


class NullGroup:
    """Sequential ⊥-BV state of one prompt: drafts in index order, G slots.

    ``next_m()`` -> envelope for the next draft (None: slots full, stop);
    ``record(output: bool)`` after its verification.
    """

    def __init__(self, n_drafts: int, slots: int, policy: NullPolicy):
        self.d = n_drafts
        self.s = slots
        self.policy = policy
        self.verified = 0
        self.outputs = 0
        self.bots = 0

    def next_m(self) -> Optional[float]:
        if self.s <= 0 or self.d <= 0:
            return None
        return self.policy.m(self.d, self.s)

    def record(self, output: bool) -> None:
        self.d -= 1
        self.verified += 1
        if output:
            self.s -= 1
            self.outputs += 1
        else:
            self.bots += 1
