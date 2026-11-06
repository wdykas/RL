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
"""Exact tail test of law_check.law_test: sum(-log t) ~ Gamma(n, 1) under H0.

The normal approximation of mean(-log t) over-rejected a perfect sampler at small n (P(reject) 2.2e-3 instead of
2.5e-5 at n = 3). lhab-tsb2-r3 step 3 (a1a5a160, Super-L8): the emission-law residual test had n = 3, one token at
t in (1e-5, 1e-4], p = 5.1e-5 -> FAIL; the exact p is 1.4e-2.
"""

import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from thundersync_rl import law_check  # noqa: E402


def _stats(ts):
    """[N_STATS] stats (no moments) of the given t = 1 - u values."""
    st = [0.0] * law_check.N_STATS
    for t in ts:
        u = 1.0 - t
        if t <= 1.0 / law_check.N_BULK:
            b = law_check.N_BULK - 1
            for k, thr in enumerate(law_check.TAIL_T):
                if t <= thr:
                    b = law_check.N_BULK + k
        else:
            b = min(int(u * law_check.N_BULK), law_check.N_BULK - 1)
        st[b] += 1
        st[law_check.N_BINS + 1] += u
        st[law_check.N_BINS + 2] += -math.log(max(t, 1e-300))
    st[law_check.N_BINS] = float(len(ts))
    return st


def _poisson_ref(x, k):
    """(P(G <= x), P(G > x)) of G ~ Gamma(k, 1), integer k: P(G > x) = P(Poisson(x) < k)."""
    lt = [-x + j * math.log(x) - math.lgamma(j + 1) for j in range(int(k) + int(3 * x) + 200)]
    m = max(lt)
    q = math.fsum(math.exp(v - m) for v in lt[: int(k)])
    p = math.fsum(math.exp(v - m) for v in lt[int(k) :])
    return p / (p + q), q / (p + q)


def test_lhab_tsb2_r3_step3_residual_is_not_a_fail():
    st = [0.0] * law_check.N_STATS
    st[22] = st[32] = st[66] = 1.0  # recorded bins (history.json law_emit_residual/stats, step 3)
    st[law_check.N_BINS : law_check.N_BINS + 3] = [3.0, 1.8635, 10.556]
    r = law_check.law_test(st)
    assert abs(r["z_tail"] - 4.36) < 0.01  # the reported statistic is unchanged
    assert abs(r["p_tail"] - 2 * 1.75e-3) < 1e-4  # 2 x P(Gamma(3, 1) > 10.556)
    assert r["p"] > 1e-2 and r["p"] > law_check.alpha()


def test_gamma_tails_match_the_poisson_sum():
    for k in (1, 2, 3, 5, 10, 30, 100, 1000, 10000):
        for z in (-6, -4, -2, 0, 2, 4, 6, 10):
            x = k + z * math.sqrt(k)
            if x <= 0:
                continue
            lo, hi = law_check._gamma_cdf_sf(x, k)
            rlo, rhi = _poisson_ref(x, k)
            for a, b in ((lo, rlo), (hi, rhi)):
                if b > 1e-250:
                    assert abs(a - b) <= 2e-9 * b + 1e-300 or abs(a - b) / b < 1e-6, (k, z, a, b)
    # Wilson-Hilferty above k = 1e4 agrees with the exact branch at the switch
    for z in (-5, 5):
        x = 1e4 + z * 100
        assert abs(law_check._gamma_cdf_sf(x, 10001)[1 if z > 0 else 0] / _poisson_ref(x, 10001)[1 if z > 0 else 0] - 1) < 5e-3


def test_tail_component_is_calibrated_at_small_n():
    # H0 (perfect sampler): P(p_tail < a) <= a at n = 1, 3, 10 (the normal approximation gave 0.2 at a = 0.04, n = 1).
    rng = random.Random(7)
    a, reps = 0.04, 20000
    for n in (1, 3, 10):
        rej = sum(
            law_check.law_test(_stats([1.0 - rng.random() for _ in range(n)]))["p_tail"] < a for _ in range(reps)
        )
        sd = math.sqrt(a * (1 - a) / reps)
        assert rej / reps < a + 4 * sd, (n, rej / reps)
        assert rej / reps > a - 4 * sd, (n, rej / reps)  # exact, not merely conservative


def test_small_n_power_on_impossible_and_far_tail_tokens():
    # A token the law gives zero mass (t = 0: the residual_from_p / null_m_1 controls at n = 1-4) still fails at n = 1.
    assert law_check.law_test(_stats([0.0]))["p"] < 1e-100
    assert law_check.law_test(_stats([0.0, 0.4, 0.7]))["p"] < 1e-100
    # Two tokens at t = 1e-7 among 3 (probability ~3e-14 under H0) fail; one token at t = 1e-4 among 3 does not.
    assert law_check.law_test(_stats([1e-7, 1e-7, 0.5]))["p"] < 1e-8
    assert law_check.law_test(_stats([1e-4, 0.5, 0.35]))["p"] > 1e-3
