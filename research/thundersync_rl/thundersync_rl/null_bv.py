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
(length, reward, acceptance) of the row itself, of later rows, or of any
pool-wide statistic over the other drafts (e.g. M_j from the mean length of
all other drafts: biased, z = +208 in agent_tail_exact_schemes.md section 3.4,
because the other drafts' envelopes then depend on X_j, which decides whether
j is reached). Earlier rows' outcomes may enter (the counts do); this module
uses counts only, and ``NullPolicy.m`` accepts nothing but the two counts.

Opt-in variants (off by default; ``agent_tail.md``):
  * prompt drop (``NullPolicy.prompt_drop``, constant M = ``fixed_m``): M
    never falls to 1; once a pool cannot fill its slots (d < s) the prompt is
    dropped (``NullGroup.dropped``) and a spare prompt takes its place. The
    drop is a function of success counts only, whose law (Bernoulli(1/M) per
    draft given the past) is the same for every prompt, so the kept prompts
    are an independent thinning and their groups are G iid p-samples. It
    shifts which prompts train (like DAPO's dynamic-sampling filter) only
    through the spare order, never through the samples' contents.
  * drop-IS (``record(output, tail_dropped=True)``): an output that needs a
    tail decode does not fill a slot; kept rows carry the importance weight
    max(1, W*/M) (W* = max prefix ratio). Unbiased gradient, not exact
    samples: opt-in only, like banded IS.

M_j policy: a dynamic program over (d, s) minimising the expected
critical-path cost sum over outputs of P(tail decode) * E[tail / cap] plus a
fresh full decode per unfilled slot, with the per-success tail-decode
probability from the Brownian model of log w (``agent_multi_draft.md``
section 2.3, validated on real rows): log w_t under p is Brownian with drift
+a^2/2 and variance a^2 = L v over a draft of length L. M = 1 whenever
d <= s (never risk a fresh decode while drafts suffice).

Calibration (``thundersync.null_bv_calibrate``): v is a moving average of the
per-token Var log p/q measured on EARLIER steps' verified drafts
(``LogRatioCalibration``; workers return each row's token count, sum and sum
of squares of log p/q: 3 numbers per row, nothing per token). A step's policy
is built from it before any of the step's drafts is verified, so within the
step v is a constant determined by the past, like a hyperparameter, and M_j
remains a function of the counts only: the argument above holds unchanged.
Estimating v from the step's own rows would make M_j depend on the drafts'
contents (forbidden, section 3.4 above). The same holds for ``verify_cost``,
the critical-path cost of verifying one draft (a BOT draft costs a
verification too), measured as earlier steps' verification seconds per
verified draft over a full decode (earlier drafts' per-token latency times
the cap): it adapts to wherever verification runs (learner, engine, fused
with training). ``tail_weight`` scales the tail cost
against a fresh decode (a constant of the cost model). v drifts over training
as the policy sharpens, so a fixed v leaves M near 1 when tails are not rare.

Placement (scale): verification forwards are placed on data-parallel ranks
by tokens (``speculative.lpt_owners``), independent of prompts. Each rank verifies its
rows for every M the policy can pick (``envelopes``) with one set of random
draws per row, and the driver walks each prompt's drafts in index order
(``decide_pools``), taking the outcome at the M the counts choose. Exactness
is unchanged: a row's draws are independent of every other row, and the M
used for it is a function of the earlier rows' outcomes only, so the outcome
taken has its M's law given the past; how the outcomes of one row at
different M are coupled is irrelevant (only one is ever used).
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
    the DP (still M = 1 when d <= s). ``prompt_drop`` (needs ``fixed_m``):
    M = ``fixed_m`` at every count, and the pool is dropped once d < s.
    """

    logratio_var: float = 8e-7
    ref_len: int = 32768
    fixed_m: Optional[float] = None
    prompt_drop: bool = False
    # Critical-path weight of a tail decode against a fresh decode of a slot.
    tail_weight: float = 1.0
    # Critical-path cost of verifying one draft (any outcome), in fresh
    # decodes: a BOT outcome is not free, the next draft is verified too.
    verify_cost: float = 0.0
    grid: tuple[float, ...] = M_GRID
    _table: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    _value: dict[tuple[int, int], float] = field(default_factory=dict, repr=False)
    _cost: Optional[dict[float, float]] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.fixed_m is not None and not self.fixed_m >= 1.0:
            raise ValueError(f"null_bv fixed M must be >= 1 (got {self.fixed_m})")
        if any(not m >= 1.0 for m in self.grid):
            raise ValueError(f"null_bv M grid must be >= 1 (got {self.grid})")
        if not self.verify_cost >= 0.0:
            raise ValueError(f"null_bv verify cost must be >= 0 (got {self.verify_cost})")
        if self.prompt_drop and self.fixed_m is None:
            raise ValueError("null_bv prompt drop needs a constant envelope (fixed_m)")

    def drop(self, d: int, s: int) -> bool:
        """Whether a pool with ``d`` drafts and ``s`` empty slots left is dropped."""
        return self.prompt_drop and d < s

    def costs(self) -> dict[float, float]:
        """Expected critical-path cost of one verified draft at envelope M
        (units: one full fresh decode = 1)."""
        if self._cost is None:
            a = math.sqrt(max(self.ref_len * self.logratio_var, 0.0))
            # Tail length given a tail decode: BV rejections are front-loaded
            # (E[1 - tau/g | fail] ~ 0.6); ⊥-BV ones sit late (tau > g/2: ~0.25).
            self._cost = {
                M: self.tail_weight * tail_decode_prob(M, a) * (0.6 if M == 1.0 else 0.25) / M
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
        ``s`` empty slots left. Counts only: never a function of draft contents."""
        if type(d) is not int or type(s) is not int:
            raise TypeError("null_bv envelopes depend on the counts (d, s) only")
        if s <= 0:
            raise ValueError("no slot left: the draft must not be verified")
        if self.prompt_drop:
            if d < s:
                raise ValueError("a pool with fewer drafts than slots is dropped")
            return float(self.fixed_m)
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
                            self.verify_cost
                            + self.costs()[M]
                            + self.value(dd - 1, ss - 1) / M
                            + (1 - 1 / M) * self.value(dd - 1, ss)
                        )
                        if val < best:
                            best, bm = val, M
                    self._table[(dd, ss)] = bm
                    self._value[(dd, ss)] = best
        return self._table[key]


class LogRatioCalibration:
    """Running estimate of the per-token Var log p/q that the M_j policy models.

    Exactness: a step's NullPolicy is built from ``estimate()`` BEFORE its
    drafts are verified, from the moments of earlier steps' verified drafts
    only (``update`` after a step's verification). Given the past, v is a
    constant for the whole step, like a hyperparameter: M_j is still a
    function of the counts (d, s) alone within the step, never of the
    contents of this step's drafts, so the argument in the module docstring
    (successes in index order are iid p) is unchanged. Using this step's own
    rows would be forbidden (section 3.4 of agent_tail_exact_schemes.md).

    ``decay``: weight of the past in the exponential moving average of the
    pooled per-step estimates (v drifts over training as the policy sharpens).
    """

    def __init__(self, prior: float, decay: float = 0.5):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"calibration decay must be in [0, 1) (got {decay})")
        self.prior, self.decay = float(prior), float(decay)
        self.value: Optional[float] = None
        self.last_step: Optional[float] = None
        # Verification seconds per verified draft and engine seconds per
        # decoded token (one stream), both measured on earlier steps: the
        # verify cost in fresh decodes is their ratio over the rollout cap.
        self.verify_row_s: Optional[float] = None
        self.token_s: Optional[float] = None

    def estimate(self) -> float:
        return self.prior if self.value is None else self.value

    def _ema(self, old: Optional[float], new: float) -> float:
        return new if old is None else self.decay * old + (1 - self.decay) * new

    def update_verify(self, seconds: float, rows: int) -> None:
        """One step's verification wall time over the drafts it verified."""
        if rows > 0 and seconds > 0:
            self.verify_row_s = self._ema(self.verify_row_s, seconds / rows)

    def update_decode(self, seconds: float, tokens: int) -> None:
        """Wall time a still-running draft took to decode ``tokens`` tokens."""
        if tokens > 0 and seconds > 0:
            self.token_s = self._ema(self.token_s, seconds / tokens)

    def verify_cost(self, cap: int) -> float:
        """Critical-path cost of verifying one draft, in fresh full decodes (0 until measured)."""
        if self.verify_row_s is None or self.token_s is None or cap <= 0:
            return 0.0
        return self.verify_row_s / (self.token_s * cap)

    def update(self, moments: list[tuple[int, float, float]]) -> None:
        """Fold in one step's per-row (tokens, sum log p/q, sum (log p/q)^2)."""
        n = sum(m[0] for m in moments)
        if n < 2:
            return
        s = sum(m[1] for m in moments)
        sq = sum(m[2] for m in moments)
        v = max(sq / n - (s / n) ** 2, 0.0)
        self.last_step = v
        self.value = self._ema(self.value, v)


def drafts_per_prompt(k: float, group_size: int) -> int:
    """N = max(G, round(K G)) drafts per GRPO group."""
    if not k >= 1.0:
        raise ValueError(f"null_bv_drafts_per_output must be >= 1 (got {k})")
    return max(group_size, int(round(k * group_size)))


class NullGroup:
    """Sequential ⊥-BV state of one prompt: drafts in index order, G slots.

    ``next_m()`` -> envelope for the next draft (None: slots full, or the
    pool dropped (``dropped``), stop); ``record(output)`` after its
    verification. ``tail_dropped`` (drop-IS): an output that needs a tail
    decode is discarded and does not fill a slot.
    """

    def __init__(self, n_drafts: int, slots: int, policy: NullPolicy):
        self.d = n_drafts
        self.s = slots
        self.policy = policy
        self.verified = 0
        self.outputs = 0
        self.bots = 0
        self.tail_drops = 0
        self.dropped = False

    def next_m(self) -> Optional[float]:
        if self.dropped or self.s <= 0:
            return None
        if self.policy.drop(self.d, self.s):
            self.dropped = True
            return None
        if self.d <= 0:
            return None
        return self.policy.m(self.d, self.s)

    def record(self, output: bool, tail_dropped: bool = False) -> None:
        self.d -= 1
        self.verified += 1
        if output and not tail_dropped:
            self.s -= 1
            self.outputs += 1
        elif output:
            self.tail_drops += 1
        else:
            self.bots += 1


def envelopes(policy: NullPolicy) -> tuple[float, ...]:
    """Every M ``policy.m`` can return: 1 (d <= s) and the grid or the fixed M
    (prompt drop: the fixed M only)."""
    if policy.prompt_drop:
        return (float(policy.fixed_m),)
    ms = {1.0} | ({float(policy.fixed_m)} if policy.fixed_m is not None else set(policy.grid))
    return tuple(sorted(ms))


def row_envelopes(policy: NullPolicy, n_drafts: int, slots: int, j: int) -> tuple[float, ...]:
    """Every envelope draft ``j`` (0-based, drafting order) of a pool of ``n_drafts``
    drafts and ``slots`` slots can be decided at: ``policy.m(d, s)`` with d = n_drafts - j
    drafts left and s from max(1, slots - j) to slots slots left. A function of the
    counts only (exogenous), so verifying a row at just these envelopes is as exact
    as the full ``envelopes(policy)`` grid; it saves the scans at unreachable M.
    """
    d = n_drafts - j
    out = set()
    for s in range(max(1, slots - j), slots + 1):
        if d <= 0 or policy.drop(d, s):
            continue
        out.add(float(policy.m(d, s)))
    return tuple(sorted(out)) or (float(envelopes(policy)[0]),)


def decide_pools(
    groups: list[str],
    slots: dict[str, int],
    outcomes: list[dict[float, bool]],
    policy: NullPolicy,
    dropped: Optional[set[str]] = None,
) -> list[Optional[float]]:
    """The envelope each row is decided at, None for rows left unused.

    ``groups[i]``: row i's prompt pool; rows of a pool are taken in index
    order. ``outcomes[i][M]``: whether row i yields an output (not BOT) when
    verified at envelope M (every M of ``envelopes(policy)``). The sequential
    count process of ``NullGroup``: M from (drafts left, slots left), stop
    when the slots are full. ``dropped`` (prompt drop) collects the dropped pools.
    """
    count: dict[str, int] = {}
    for g in groups:
        count[g] = count.get(g, 0) + 1
    state = {g: NullGroup(n, int(slots[g]), policy) for g, n in count.items()}
    out: list[Optional[float]] = []
    for g, oc in zip(groups, outcomes):
        m = state[g].next_m()
        out.append(m)
        if m is not None:
            state[g].record(oc[m])
    if dropped is not None:
        for g, st in state.items():
            st.next_m()  # a pool short after its last draft is dropped too
            if st.dropped:
                dropped.add(g)
    return out
