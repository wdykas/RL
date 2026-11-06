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
"""Exact block verification under integer laws (exact_bv.py).

1. Law identity (Fractions, no sampling): drafts enumerated from the integer q law, BV output
   law from ``output_law_fraction``, continuation to a fixed length: the sequence law is the
   integer p law EXACTLY (plain, ⊥-BV, top-p restricted laws with support flips). Negative
   control: a logged q that is not the sampled q breaks the identity.
2. The sampler implements that law: ``verify_exact`` frequencies vs the exact law (chi-square),
   fast path and forced exact slow path, residual rejection included.
3. Outcomes are a deterministic function of (key, integer laws): the same (tau, y) whether
   log q is logged in fp32, fp64, or recomputed, and with every decision forced through the
   exact slow path (floating point never decides an outcome).
4. exact_bernoulli agrees with the exact comparison of the generated eta.
"""

import itertools
import os
import random
import sys
from collections import Counter
from fractions import Fraction

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from thundersync_rl import exact_bv as xb  # noqa: E402

from nemo_rl.models.megatron import exact_int as xi  # noqa: E402
from nemo_rl.models.megatron import exact_nucleus  # noqa: E402

V = 3


def _row(seed, prefix, zero_frac=0.25, hi=12):
    rng = random.Random(hash((seed, tuple(prefix))) & 0xFFFFFFFF)
    w = [0 if rng.random() < zero_frac else rng.randrange(1, hi) for _ in range(V)]
    if sum(w) == 0:
        w[rng.randrange(V)] = 5
    return w


def _law(w):
    s = sum(w)
    return [Fraction(x, s) for x in w]


def _run_block(prefix, g, p_of, q_of, M=Fraction(1), q_logged_of=None):
    """Exact law of the emitted tokens (tuple; ('BOT',) for BOT) of one BV block from prefix."""
    out = Counter()
    q_logged_of = q_logged_of or q_of
    for draft in itertools.product(range(V), repeat=g):
        pr = Fraction(1)
        for i, x in enumerate(draft):
            pr *= _law(q_of(prefix + list(draft[:i])))[x]
        if pr == 0:
            continue
        wp = [p_of(prefix + list(draft[:i])) for i in range(g + 1)]
        wq = [q_logged_of(prefix + list(draft[:i])) for i in range(g)]
        for (tau, y), pt in xb.output_law_fraction(wp, wq, list(draft), M).items():
            emitted = ("BOT",) if y == -1 else tuple(draft[:tau]) + (y,)
            out[emitted] += pr * pt
    return out


def _sequence_law(L, g, p_of, q_of, M=Fraction(1), q_logged_of=None):
    """Law of the first L tokens: one block with envelope M from the root, then plain BV."""
    final = Counter()
    stack = [((), Fraction(1), True)]
    while stack:
        s, pr, first = stack.pop()
        if len(s) >= L:
            final[s[:L]] += pr
            continue
        gg = min(g, L - len(s))
        for em, pe in _run_block(list(s), gg, p_of, q_of, M if first else Fraction(1), q_logged_of).items():
            if em == ("BOT",):
                final["BOT"] += pr * pe
            else:
                stack.append((s + em, pr * pe, False))
    return final


def _p_sequence_law(L, p_of):
    out = {}
    for seq in itertools.product(range(V), repeat=L):
        pr = Fraction(1)
        for t in range(L):
            pr *= _law(p_of(list(seq[:t])))[seq[t]]
        if pr:
            out[seq] = pr
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("g", [1, 2, 3])
def test_integer_law_identity_plain(seed, g):
    p_of = lambda pre: _row(seed, pre)  # noqa: E731
    q_of = lambda pre: _row(seed + 100, pre)  # noqa: E731
    L = 3
    got = _sequence_law(L, g, p_of, q_of)
    want = _p_sequence_law(L, p_of)
    assert dict(got) == want  # exact Fraction equality


@pytest.mark.parametrize("M", [Fraction(7, 4), Fraction(3), Fraction(5, 2)])
def test_integer_law_identity_null_bv(M):
    p_of = lambda pre: _row(5, pre)  # noqa: E731
    q_of = lambda pre: _row(105, pre)  # noqa: E731
    L = 3
    got = _sequence_law(L, 2, p_of, q_of, M)
    bot = got.pop("BOT")
    assert bot == 1 - 1 / M
    want = _p_sequence_law(L, p_of)
    assert {k: v / (1 - bot) for k, v in got.items()} == want


def test_integer_law_identity_top_p_supports():
    """Top-p: both laws restricted to their own exact nucleus (same integers): support
    flips (drafted token outside the target nucleus) are rejected with probability 1."""

    def nuc(seed):
        def f(pre):
            w = _row(seed, pre, zero_frac=0.0, hi=30)
            keep = exact_nucleus.plain_mask(torch.tensor([w], dtype=torch.int64), 0.7)[0].tolist()
            return [x if k else 0 for x, k in zip(w, keep)]

        return f

    p_of, q_of = nuc(11), nuc(111)
    flips = sum(
        1
        for pre in itertools.product(range(V), repeat=2)
        for v in range(V)
        if q_of(list(pre))[v] > 0 and p_of(list(pre))[v] == 0
    )
    assert flips > 0
    got = _sequence_law(3, 2, p_of, q_of)
    assert dict(got) == _p_sequence_law(3, p_of)


def test_identity_breaks_when_logged_q_is_not_sampled_q():
    """Negative control: drafts sampled from q, verified with a different logged q."""
    p_of = lambda pre: _row(0, pre)  # noqa: E731
    q_of = lambda pre: _row(100, pre, zero_frac=0.0)  # noqa: E731
    q_wrong = lambda pre: [x + (1 if i == 0 else 0) for i, x in enumerate(q_of(pre))]  # noqa: E731
    got = _sequence_law(3, 2, p_of, q_of, q_logged_of=q_wrong)
    assert dict(got) != _p_sequence_law(3, p_of)


# ----------------------------------------------------------------------------
# The sampler (verify_exact) implements the exact law
# ----------------------------------------------------------------------------


def _rows_from_weights(ws):
    """fp32 log-prob rows whose integer weights are close to ws (any positive rows work:
    the test compares against the integer law of the rows themselves)."""
    t = torch.tensor(ws, dtype=torch.float64)
    lp = torch.log(t / t.sum(-1, keepdim=True)).float()
    return lp


def _case(seed, g, Vc=5):
    rng = np.random.default_rng(seed)
    pw = rng.integers(1, 20, size=(g + 1, Vc)).astype(float)
    qw = rng.integers(1, 20, size=(g, Vc)).astype(float)
    pw[rng.random(pw.shape) < 0.2] = 0.0
    pw[:, 0] += 1
    qw[:, 0] += 1
    P = _rows_from_weights(pw)
    Q = _rows_from_weights(qw)
    draft = torch.tensor([int(np.argmax(qw[i])) if i % 2 else 0 for i in range(g)])
    return P, Q, draft


def _exact_law(P, Q, draft, M=1.0):
    wp = xi.int_weights(P).tolist()
    wq = xi.int_weights(Q).tolist()
    return xb.output_law_fraction(wp, wq, draft.tolist(), Fraction(*float(M).as_integer_ratio()))


def _chi2_p(counts: Counter, law: dict, n: int, support_ok: bool = False) -> float:
    from scipy.stats import chi2

    keys = list(law)
    exp = np.array([float(law[k]) * n for k in keys])
    obs = np.array([counts.get(k, 0) for k in keys], dtype=float)
    extra = sum(c for k, c in counts.items() if k not in law)
    if support_ok and extra:
        return 0.0
    assert extra == 0, f"outcomes outside the support: {[(k, c) for k, c in counts.items() if k not in law]}"
    big = exp >= 5
    o = np.append(obs[big], obs[~big].sum())
    e = np.append(exp[big], exp[~big].sum())
    keep = e > 0
    stat = float(((o[keep] - e[keep]) ** 2 / e[keep]).sum())
    return float(chi2.sf(stat, max(1, int(keep.sum()) - 1)))


@pytest.mark.parametrize("M", [1.0, 1.75])
@pytest.mark.parametrize("slow", [False, True])
def test_sampler_frequencies_match_exact_law(M, slow, monkeypatch):
    monkeypatch.setattr(xb, "FORCE_SLOW", slow)
    P, Q, draft = _case(3, 3)
    law = _exact_law(P, Q, draft, M)
    n = 3000 if slow else 12000
    cnt = Counter()
    for k in range(n):
        tau, y, _, _ = xb.verify_exact(P, Q, draft, null_m=M, key=1_000_003 * k + 17)
        cnt[(tau, y)] += 1
    p = _chi2_p(cnt, law, n)
    assert p > 1e-4, (p, cnt, {k: float(v) for k, v in law.items()})
    # the residual path is exercised (tau < g with a vocabulary draw)
    assert sum(c for (t, y), c in cnt.items() if t < draft.numel() and y >= 0) > n // 20


def test_negative_control_residual_from_p(monkeypatch):
    P, Q, draft = _case(3, 3)
    law = _exact_law(P, Q, draft)
    monkeypatch.setenv("THUNDERSYNC_DEBUG_LAW_BUG", "residual_from_p")
    n = 12000
    cnt = Counter()
    for k in range(n):
        tau, y, _, _ = xb.verify_exact(P, Q, draft, key=1_000_003 * k + 17)
        cnt[(tau, y)] += 1
    assert _chi2_p(cnt, law, n, support_ok=True) < 1e-6


@pytest.mark.parametrize("seed", range(6))
def test_outcome_is_a_function_of_key_and_integer_laws(seed, monkeypatch):
    """Same (tau, y) for every precision of the logged log q and with every decision
    forced through the exact path: the floating point only decides how fast."""
    g = 6
    P, Q, draft = _case(seed, g, Vc=7)
    W, wx = xi.row_stats(Q, draft)
    lam64 = xi.lam(wx, W)
    for key in range(seed * 50, seed * 50 + 50):
        ref = xb.verify_exact(P, Q, draft, lqx=None, key=key)
        for lq in (lam64, lam64.float()):
            got = xb.verify_exact(P, Q, draft, lqx=lq, key=key)
            assert got[:2] == ref[:2] and torch.equal(got[2], ref[2])
        monkeypatch.setattr(xb, "FORCE_SLOW", True)
        got = xb.verify_exact(P, Q, draft, lqx=lam64.float(), key=key)
        monkeypatch.setattr(xb, "FORCE_SLOW", False)
        assert got[:2] == ref[:2]


def test_grid_marginals_equal_single_envelope():
    P, Q, draft = _case(4, 3)
    for key in range(40):
        out, lam_p, _ = xb.verify_exact_grid(P, Q, draft, None, None, (1.0, 1.5, 3.0), key=key)
        for M, (tau, y, lp) in out.items():
            t1, y1, lp1, _ = xb.verify_exact(P, Q, draft, null_m=M, key=key)
            assert (tau, y) == (t1, y1)
            if y >= 0:
                assert lp == float(lp1[-1])
        assert torch.equal(lam_p, xb.p_token_lambda(P, draft))


def test_logp_is_integer_law_lambda():
    P, Q, draft = _case(6, 4)
    tau, y, lp, _ = xb.verify_exact(P, Q, draft, key=5)
    kept = torch.cat([draft[:tau], torch.tensor([y])])
    ref = xi.int_logprob(P[: tau + 1], kept).float()
    assert torch.equal(lp, ref)


def test_p_zero_draft_token_is_rejected():
    P, Q, draft = _case(7, 3)
    P = P.clone()
    P[1, int(draft[1])] = float("-inf")  # p(x_1) = 0 exactly
    P[1] = torch.log_softmax(P[1], -1)
    for key in range(200):
        tau, y, _, _ = xb.verify_exact(P, Q, draft, key=key)
        assert tau <= 1


def test_exact_bernoulli_matches_exact_comparison(monkeypatch):
    rng = random.Random(9)
    for slow in (False, True):
        monkeypatch.setattr(xb, "FORCE_SLOW", slow)
        for _ in range(300):
            a = Fraction(rng.randrange(1, 10**6), rng.randrange(10**6, 3 * 10**6))
            key = rng.getrandbits(62)
            words = lambda k, key=key: xi.word_py(key, 0, 0, xi.TEST, k)  # noqa: E731
            K, bits = xb._eta_block_interval(words, 3)
            eta_lo, eta_hi = Fraction(K, 1 << bits), Fraction(K + 1, 1 << bits)
            truth = eta_hi <= a if (eta_hi <= a or eta_lo >= a) else None
            fa = float(a)
            got = xb.exact_bernoulli(words, (fa * (1 - 1e-15), fa * (1 + 1e-15)), lambda F, a=a: (a, a))
            if truth is not None:
                assert got == truth
    # adversarial: eta's first block straddles a (forces the slow path), decided by block 2
    a = Fraction(1, 3)
    k0 = int(a * (1 << 53))  # first block = floor(a 2^53): eta in [k0, k0 + 1) 2^-53 contains a

    for _ in range(50):
        blocks = [k0] + [rng.getrandbits(53) for _ in range(8)]

        def words(k, blocks=blocks):
            return blocks[k] << 11

        K, bits = xb._eta_block_interval(words, 2)
        got = xb.exact_bernoulli(words, (float(a) - 1e-12, float(a) + 1e-12), lambda F: (a, a))
        assert got == (Fraction(K + 1, 1 << bits) <= a)


# ----------------------------------------------------------------------------
# Negative control residual_from_p under the integer law (ei-xforce-bug-res, 10-20)
#
# exact_bv injects the bug at EVERY stop position, including the ⊥-BV root (tau = 0, M > 1), where
# the correct emission is [(p/M - q)+, BOT 1 - 1/M]. There the bug = "BOT replaced by a fresh draw
# from p", which is itself an exact sampler of the row law p up to the root residual mass rho: the
# row law under the root bug is p + rho p - R (R = correct non-BOT root-residual outputs, mass rho),
# so TV <= rho and the bug is INERT on the row law when (p/M - q)+ = 0 (M q >= p) at the root.
# The trained-token PIT (law test) only sees the row law; the emission-law tests (conditional on
# (draft, tau)) see the missing BOT column and the non-root residual bias.
# ----------------------------------------------------------------------------


def _bug_block_law(wp, wq, draft, M, positions):
    """(tau, y) law of one exact_bv block with residual_from_p at the stop positions in
    ``positions`` (the stop-index law is unchanged; the emission there is p, never BOT)."""
    good = xb.output_law_fraction(wp, wq, draft, M)
    g = len(draft)
    ptau = Counter()
    for (t, _), pr in good.items():
        ptau[t] += pr
    out = Counter()
    for (t, y), pr in good.items():
        if not (t < g and t in positions):
            out[(t, y)] += pr
    for t in positions:
        if t < g and ptau[t]:
            for v, x in enumerate(_law(wp[t])):
                if x:
                    out[(t, v)] += ptau[t] * x
    return out


def _run_block_fn(prefix, g, p_of, q_of, M, positions):
    out = Counter()
    for draft in itertools.product(range(V), repeat=g):
        pr = Fraction(1)
        for i, x in enumerate(draft):
            pr *= _law(q_of(prefix + list(draft[:i])))[x]
        if pr == 0:
            continue
        wp = [p_of(prefix + list(draft[:i])) for i in range(g + 1)]
        wq = [q_of(prefix + list(draft[:i])) for i in range(g)]
        for (tau, y), pt in _bug_block_law(wp, wq, list(draft), M, positions).items():
            emitted = ("BOT",) if y == -1 else tuple(draft[:tau]) + (y,)
            out[emitted] += pr * pt
    return out


def _sequence_law_bug(L, g, p_of, q_of, M, first_positions, later_positions):
    final = Counter()
    stack = [((), Fraction(1), True)]
    while stack:
        s, pr, first = stack.pop()
        if len(s) >= L:
            final[s[:L]] += pr
            continue
        gg = min(g, L - len(s))
        pos = first_positions if first else later_positions
        for em, pe in _run_block_fn(list(s), gg, p_of, q_of, M if first else Fraction(1), pos).items():
            if em == ("BOT",):
                final["BOT"] += pr * pe
            else:
                stack.append((s + em, pr * pe, False))
    return final


def _root_residual_mass(g, p_of, q_of, M):
    """rho = P(stop at the ⊥-BV root with a vocabulary (non-BOT) emission), correct verifier."""
    rho = Fraction(0)
    for draft in itertools.product(range(V), repeat=g):
        pr = Fraction(1)
        for i, x in enumerate(draft):
            pr *= _law(q_of(list(draft[:i])))[x]
        if pr == 0:
            continue
        wp = [p_of(list(draft[:i])) for i in range(g + 1)]
        wq = [q_of(list(draft[:i])) for i in range(g)]
        rho += pr * sum(v for (t, y), v in xb.output_law_fraction(wp, wq, list(draft), M).items() if t == 0 and y >= 0)
    return rho


@pytest.mark.parametrize("M", [Fraction(7, 4), Fraction(3)])
def test_root_residual_from_p_is_inert_on_the_row_law(M):
    """[E] Bug only at the ⊥-BV root, M q >= p there: the row law is p EXACTLY and BOT never
    occurs (correct: P(BOT) = 1 - 1/M). The trained-token law test cannot see this part."""
    p_of = lambda pre: _row(21, pre, zero_frac=0.0)  # noqa: E731
    q_of = lambda pre: p_of(pre) if not pre else _row(121, pre)  # noqa: E731  (q = p at the root)
    L, g = 3, 2
    assert _root_residual_mass(g, p_of, q_of, M) == 0
    bug = _sequence_law_bug(L, g, p_of, q_of, M, {0}, set())
    assert "BOT" not in bug
    assert dict(bug) == _p_sequence_law(L, p_of)
    good = _sequence_law(L, g, p_of, q_of, M)
    assert good["BOT"] == 1 - 1 / M


@pytest.mark.parametrize("seed", [3, 4])
def test_root_residual_from_p_bias_is_bounded_by_root_residual_mass(seed):
    """[E] General rows: TV(row law under the root-only bug, p) <= rho (root residual mass)."""
    M = Fraction(7, 4)
    p_of = lambda pre: _row(seed, pre, zero_frac=0.0)  # noqa: E731
    q_of = lambda pre: _row(seed + 100, pre, zero_frac=0.0)  # noqa: E731
    L, g = 3, 2
    rho = _root_residual_mass(g, p_of, q_of, M)
    bug = _sequence_law_bug(L, g, p_of, q_of, M, {0}, set())
    want = _p_sequence_law(L, p_of)
    tv = sum(abs(bug.get(k, 0) - want.get(k, 0)) for k in set(bug) | set(want)) / 2
    assert 0 < tv <= rho


@pytest.mark.parametrize("M", [Fraction(1), Fraction(7, 4)])
def test_residual_from_p_everywhere_biases_the_row_law(M):
    """[E] The bug as injected (every stop position, every block) changes the row law."""
    p_of = lambda pre: _row(21, pre, zero_frac=0.0)  # noqa: E731
    q_of = lambda pre: p_of(pre) if not pre else _row(121, pre)  # noqa: E731
    L, g = 3, 2
    allpos = set(range(g))
    bug = _sequence_law_bug(L, g, p_of, q_of, M, allpos, allpos)
    bug.pop("BOT", None)
    tot = sum(bug.values())
    assert {k: v / tot for k, v in bug.items()} != _p_sequence_law(L, p_of)


def _emission_pits(P, Q, draft, M, n, seed=0):
    """Emission-law PITs (law_check reference laws, as worker._emission_law / _null_emission_pits)
    of verify_exact_grid outcomes at envelope M."""
    from thundersync_rl import law_check

    g = draft.numel()
    lpx = P[torch.arange(g), draft]
    lqx = Q[torch.arange(g), draft]
    us = []
    for k in range(n):
        out, _, _ = xb.verify_exact_grid(P, Q, draft, None, None, (float(M),), key=7_000_003 * k + seed)
        tau, y, _ = out[float(M)]
        lq_row = None if tau == g else Q[tau]
        if float(M) == 1.0:
            dist = law_check.reference_emission(P[tau], lq_row, lpx, lqx, tau)
            idx = y
        else:
            dist = law_check.reference_null_emission(P[tau], lq_row, lpx, lqx, tau, float(M))
            idx = y if y >= 0 else dist.numel() - 1
        us.append(law_check.emission_pit(dist, idx, law_check.hash_uniform("emit", seed, k, tau))[0])
    return us


@pytest.mark.parametrize("M", [1.0, 1.75])
@pytest.mark.parametrize("root_q_equals_p", [False, True])
@pytest.mark.parametrize("bug", ["residual_from_p", "null_m_1"])
def test_emission_law_check_catches_residual_from_p_on_exact_bv(M, root_q_equals_p, bug, monkeypatch):
    """[T] The emission-law PIT on exact_bv outcomes: uniform for the exact verifier, caught
    (KS p < 1e-6) under residual_from_p, including the root-only-inert case (q = p at the root)."""
    from scipy.stats import kstest

    if bug == "null_m_1" and (M == 1.0 or root_q_equals_p):
        pytest.skip("null_m_1 acts at the ⊥-BV root only (M > 1) and needs a root residual (q != p)")
    # case 1 stops at the root w.p. 0.77 at M 1.75 (case 3: never; it exercises later positions)
    P, Q, draft = _case(1 if M != 1.0 else 3, 3)
    if root_q_equals_p:
        Q = Q.clone()
        Q[0] = P[0]
    n = 3000
    good = _emission_pits(P, Q, draft, M, n)
    assert kstest(good, "uniform").pvalue > 1e-4
    monkeypatch.setenv("THUNDERSYNC_DEBUG_LAW_BUG", bug)
    bad = _emission_pits(P, Q, draft, M, n)
    assert kstest(bad, "uniform").pvalue < 1e-6


# ----------------------------------------------------------------------------
# Batched stop index (exactint-final 10-20: full-Nano verify0 352 s vs 28 s wide = per-position h)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1.0, 1.75, 3.0])
@pytest.mark.parametrize("seed", range(4))
def test_batched_stop_index_equals_per_position_scan(M, seed, monkeypatch):
    g = 40
    P, Q, draft = _case(seed, g, Vc=9)
    for key in range(seed * 100, seed * 100 + 60):
        v1 = xb._Verify(P, Q, draft, None, key)
        v2 = xb._Verify(P, Q, draft, None, key)
        assert v1.stop_index(M)[0] == v2.stop_index_ref(M)[0]
    monkeypatch.setattr(xb, "FORCE_SLOW", True)
    for key in range(5):
        v1 = xb._Verify(P, Q, draft, None, key)
        v2 = xb._Verify(P, Q, draft, None, key)
        assert v1.stop_index(M)[0] == v2.stop_index_ref(M)[0]


@pytest.mark.parametrize("M", [1.0, 1.75])
def test_batched_h_bounds_contain_the_exact_h(M):
    P, Q, draft = _case(2, 12, Vc=9)
    v = xb._Verify(P, Q, draft, None, 3)
    b, b_lo, b_hi = v.b_bounds(M)
    idx = np.arange(1, 12)
    h_lo, h_hi = v.h_bounds_vec(idx, b, b_lo, b_hi)
    ref = v.h_bounds(idx.tolist(), b, b_lo, b_hi)
    for j, i in enumerate(idx.tolist()):
        lo, hi = v.h_exact(M, i)(256)
        assert h_lo[j] <= float(lo) and float(hi) <= h_hi[j]
        assert abs(h_lo[j] - ref[j][0]) <= 1e-12 and abs(h_hi[j] - ref[j][1]) <= 1e-12


def test_residual_mass_matches_reference_rows():
    from nemo_rl.models.megatron import exact_int as xi_

    P, Q, _ = _case(5, 6, Vc=11)
    Wp, _ = xi_.row_stats(P[:6], None)
    b = torch.tensor([1.0, 0.5, 0.9, 0.1, 1.0, 0.75], dtype=torch.float64)
    Wq, r = xi_.residual_mass(P[:6], Q[:6], Wp, b)
    for k in range(6):
        wp, wq = xi_.int_weights(P[k]), xi_.int_weights(Q[k])
        assert int(Wq[k]) == int(wq.sum())
        want = sum(max(Fraction(0), Fraction(b[k].item()) * Fraction(int(x), int(Wp[k])) - Fraction(int(y), int(wq.sum())))
                   for x, y in zip(wp.tolist(), wq.tolist()))
        assert abs(float(r[k]) - float(want)) <= 1e-15
