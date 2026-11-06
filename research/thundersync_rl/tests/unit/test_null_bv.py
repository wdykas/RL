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
"""Pooled null-augmented block verification (⊥-BV): exactness and plumbing.

* exact enumeration on a toy autoregressive model: per-draft law (BOT with
  probability 1 - 1/M, otherwise exactly p) and the joint law of the G slots of
  the pooled process (iid p), for DP, fixed and count-adaptive policies;
* Monte Carlo through the sampler the worker runs (``verify_engine_q`` with
  ``null_m``) and the worker's group state (``NullGroup``), chi-square against
  p x p, and a deliberately biased control (M chosen from the draft content);
* eager / lazy / engine-q samplers pathwise identical with ``null_m``;
* DP policy, config validation, driver plans and statistics, the worker's
  per-group decisions.
"""

from __future__ import annotations

import asyncio
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.block_verification import (
    NULL_TOKEN,
    block_verify,
    block_verify_lazy,
    output_distribution_given_draft,
)
from thundersync_rl.engine_q import verify_engine_q, verify_engine_q_grid
from thundersync_rl.null_bv import NullGroup, NullPolicy, drafts_per_prompt, envelopes, tail_decode_prob

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# --------------------------------------------------------------------------
# Toy autoregressive models (fixed length L, vocabulary V)
# --------------------------------------------------------------------------


def _model(V, L, seed, conc=1.0):
    rng = np.random.default_rng(seed)
    return {
        pre: rng.dirichlet(np.full(V, conc))
        for t in range(L + 1)
        for pre in itertools.product(range(V), repeat=t)
    }


def _seq_prob(m, x):
    return math.prod(m[tuple(x[:t])][x[t]] for t in range(len(x)))


def _rows(m, x, n, dtype=torch.float32):
    """log-prob rows [n, V] at prefixes x[:0..n-1] (fp32 as the verifier gets
    them; fp64 for the exact enumeration, so the rows are exactly the model)."""
    return torch.tensor(np.log(np.stack([m[tuple(x[:t])] for t in range(n)])), dtype=dtype)


def _draft_outcome_law(P, Q, x, M, V, L):
    """Exact law of one ⊥-BV verification of draft x (implementation's law):
    {sequence: prob} plus "BOT"; tails after a rejection decoded from p."""
    lp = _rows(P, x, L + 1, torch.float64)
    lq = _rows(Q, x, L, torch.float64)
    out: dict = {}
    for tau, probs in output_distribution_given_draft(lp, lq, torch.tensor(x), M):
        probs = probs.double().numpy()
        if tau == L:
            # Finished draft (length cap): the emitted bonus is never used.
            out[tuple(x)] = out.get(tuple(x), 0.0) + probs.sum()
            continue
        for y, py in enumerate(probs):
            if py == 0:
                continue
            if y == V:  # BOT column (root residual only)
                assert tau == 0 and M > 1
                out["BOT"] = out.get("BOT", 0.0) + py
                continue
            pre = list(x[:tau]) + [y]
            for tail in itertools.product(range(V), repeat=L - len(pre)):
                z = tuple(pre) + tail
                pz = py
                for t in range(len(pre), L):
                    pz *= P[z[:t]][z[t]]
                out[z] = out.get(z, 0.0) + pz
    return out


def _per_draft_law(P, Q, M, V, L):
    law: dict = {}
    for x in itertools.product(range(V), repeat=L):
        qx = _seq_prob(Q, x)
        for k, v in _draft_outcome_law(P, Q, x, M, V, L).items():
            law[k] = law.get(k, 0.0) + qx * v
    return law


class _RulePolicy(NullPolicy):
    """Count-adaptive envelope M = d / s (capped), M = 1 when d <= s."""

    def m(self, d, s):
        return 1.0 if d <= s else min(3.0, d / s)


def _group_law(per_draft, policy, N, G, seqs, p1):
    """Exact joint law of the G ordered slots of the pooled process."""
    memo = {}

    def law(d, s):
        if (d, s) in memo:
            return memo[(d, s)]
        if s == 0:
            res = {(): 1.0}
        elif d == 0:
            res = {
                zs: math.prod(p1[z] for z in zs)
                for zs in itertools.product(seqs, repeat=s)
            }
        else:
            M = policy.m(d, s)
            pd = per_draft[M]
            res = {}
            for k, v in law(d - 1, s).items():
                res[k] = res.get(k, 0.0) + pd.get("BOT", 0.0) * v
            rest = law(d - 1, s - 1)
            for z in seqs:
                pz = pd.get(z, 0.0)
                if pz == 0:
                    continue
                for k, v in rest.items():
                    kk = (z,) + k
                    res[kk] = res.get(kk, 0.0) + pz * v
        memo[(d, s)] = res
        return res

    return law(N, G)


def test_per_draft_law_is_p_over_m_plus_bot():
    V, L = 3, 3
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    for M in (1.0, 1.25, 1.7, 3.0):
        law = _per_draft_law(P, Q, M, V, L)
        assert abs(law.get("BOT", 0.0) - (1 - 1 / M)) < 1e-12
        tv = 0.5 * sum(abs(law.get(z, 0.0) - _seq_prob(P, z) / M) for z in seqs)
        assert tv < 1e-12, (M, tv)


@pytest.mark.parametrize(
    "policy",
    [
        NullPolicy(logratio_var=0.2, ref_len=3),  # DP (mixes M values)
        NullPolicy(fixed_m=2.5),
        _RulePolicy(),
    ],
    ids=["dp", "fixed2.5", "d_over_s"],
)
def test_pooled_slots_are_iid_p_by_enumeration(policy):
    V, L, N, G = 3, 3, 5, 2
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    p1 = {z: _seq_prob(P, z) for z in seqs}
    tvq = 0.5 * sum(abs(p1[z] - _seq_prob(Q, z)) for z in seqs)
    assert tvq > 0.3  # p and q far apart: rejections at the root are common
    Ms = {policy.m(d, s) for d in range(1, N + 1) for s in range(1, G + 1)}
    per_draft = {M: _per_draft_law(P, Q, M, V, L) for M in Ms}
    joint = _group_law(per_draft, policy, N, G, seqs, p1)
    tv = 0.5 * sum(
        abs(joint.get(zs, 0.0) - math.prod(p1[z] for z in zs))
        for zs in itertools.product(seqs, repeat=G)
    )
    assert tv < 1e-9, tv
    if not isinstance(policy, _RulePolicy) and policy.fixed_m is None:
        assert len(Ms) > 2, Ms  # the DP really varies M


def _coupled_outcomes(P, Q, V, L, Ms):
    """Joint law of one row's outcomes at every M in ``Ms``: {(o_M, ...): prob}.

    The outcomes at the different M are coupled comonotonically through one
    shared uniform (inverse CDFs in a fixed outcome order), as the grid
    sampler couples them through shared draws.
    """
    order = ["BOT"] + list(itertools.product(range(V), repeat=L))
    joint: dict = {}
    for x in itertools.product(range(V), repeat=L):
        qx = _seq_prob(Q, x)
        cdfs = []
        for M in Ms:
            law = _draft_outcome_law(P, Q, x, M, V, L)
            acc, pts = 0.0, []
            for o in order:
                if law.get(o, 0.0) > 0:
                    acc += law[o]
                    pts.append((acc, o))
            assert abs(acc - 1.0) < 1e-12
            pts[-1] = (1.0, pts[-1][1])  # rounding: every CDF ends at 1
            cdfs.append(pts)
        cuts = sorted({c for pts in cdfs for c, _ in pts} | {0.0})
        for lo, hi in zip(cuts, cuts[1:]):
            mid = 0.5 * (lo + hi)
            vec = tuple(next(o for c, o in pts if mid <= c) for pts in cdfs)
            joint[vec] = joint.get(vec, 0.0) + qx * (hi - lo)
    return joint




def test_outcome_based_envelope_choice_is_biased():
    """Control: taking a row's outcome at the M that yields an output (a
    choice that depends on the row's own draws) is biased."""
    V, L, N, G = 2, 2, 3, 2
    P, Q = _model(V, L, 7, conc=0.5), _model(V, L, 8, conc=0.5)
    seqs = list(itertools.product(range(V), repeat=L))
    p1 = {z: _seq_prob(P, z) for z in seqs}
    Ms = (1.0, 2.0, 3.0)
    row = list(_coupled_outcomes(P, Q, V, L, Ms).items())
    joint: dict = {}
    for combo in itertools.product(row, repeat=N):
        pr = math.prod(v for _, v in combo)
        outs = []
        for vec, _ in combo:
            if len(outs) < G:
                # prefer the largest M whose outcome is not BOT (forbidden)
                ok = [o for o in reversed(vec) if o != "BOT"]
                if ok:
                    outs.append(ok[0])
        for tail in itertools.product(seqs, repeat=G - len(outs)):
            key = tuple(outs) + tail
            joint[key] = joint.get(key, 0.0) + pr * math.prod(p1[z] for z in tail)
    tv = 0.5 * sum(
        abs(joint.get(zs, 0.0) - math.prod(p1[z] for z in zs))
        for zs in itertools.product(seqs, repeat=G)
    )
    assert tv > 1e-3, tv


def test_grid_sampler_marginals_match_each_envelope():
    """``verify_engine_q_grid``: at every M, the outcome law of the M sampler
    (Monte Carlo, chi-square per M against the exact per-draft law)."""
    V, L, n = 3, 2, 6000
    P, Q = _model(V, L, 3), _model(V, L, 4)
    Ms = (1.0, 1.5, 3.0)
    x = (1, 2)
    lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
    gen = torch.Generator().manual_seed(0)
    keys = {M: {} for M in Ms}
    for _ in range(n):
        grid, _, _ = verify_engine_q_grid(
            lp, torch.tensor(x), lq[torch.arange(L), torch.tensor(x)], lambda i: lq[i], gen, Ms
        )
        for M, (tau, y, _) in grid.items():
            k = (tau, y)
            keys[M][k] = keys[M].get(k, 0) + 1
    for M in Ms:
        exact = {}
        lp64, lq64 = _rows(P, x, L + 1, torch.float64), _rows(Q, x, L, torch.float64)
        for tau, probs in output_distribution_given_draft(lp64, lq64, torch.tensor(x), M):
            for y, py in enumerate(probs.double().tolist()):
                if py > 0:
                    yy = NULL_TOKEN if y == V else y
                    exact[(tau, yy)] = exact.get((tau, yy), 0.0) + py
        chi2 = sum((keys[M].get(k, 0) - n * v) ** 2 / (n * v) for k, v in exact.items())
        dof = len(exact) - 1
        assert (chi2 - dof) / math.sqrt(2 * dof) < 4, (M, chi2, dof)
        assert set(keys[M]) <= set(exact)


def test_lpt_owners_balance_tokens():
    from thundersync_rl.speculative import lpt_owners

    lengths = [64000, 1000, 1000, 30000, 30000, 4000, 500, 62000]
    owners = lpt_owners(lengths, 4)
    load = [sum(n for n, r in zip(lengths, owners) if r == k) for k in range(4)]
    assert max(load) == 64000 and sorted(load)[0] >= 30000
    assert lpt_owners([5, 5], 1) == [0, 0]


def test_content_dependent_m_is_biased_by_enumeration():
    """Control: M chosen from the draft's first token breaks exactness."""
    V, L = 3, 3
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    out: dict = {}
    for x in seqs:
        M = 1.0 if x[0] == 0 else 2.0  # forbidden: depends on X_j
        for k, v in _draft_outcome_law(P, Q, x, M, V, L).items():
            out[k] = out.get(k, 0.0) + _seq_prob(Q, x) * v
    ok = 1 - out.get("BOT", 0.0)
    tv = 0.5 * sum(abs(out.get(z, 0.0) / ok - _seq_prob(P, z)) for z in seqs)
    assert tv > 1e-3, tv


# --------------------------------------------------------------------------
# Monte Carlo through the worker's sampler and group state
# --------------------------------------------------------------------------


def _sample(m, rng, pre, L):
    z = list(pre)
    while len(z) < L:
        z.append(int(rng.choice(len(m[()]), p=m[tuple(z)])))
    return tuple(z)


def _mc_counts(P, Q, V, L, N, G, policy, n, seed, prefer_whole=False):
    """Joint counts of the G slots from ``verify_engine_q`` + ``NullGroup``.

    ``prefer_whole`` (biased control, outcome-based selection): verify every
    draft with plain BV and fill the slots with whole-accepted outputs first.
    """
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    seqs = list(itertools.product(range(V), repeat=L))
    idx = {z: i for i, z in enumerate(seqs)}
    cnt = np.zeros((len(seqs),) * G)
    for _ in range(n):
        grp, outs = NullGroup(N, G, policy), []
        if prefer_whole:
            whole, part = [], []
            for _ in range(N):
                x = _sample(Q, rng, (), L)
                lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
                tau, y, _, _ = verify_engine_q(
                    lp, torch.tensor(x), lq[torch.arange(L), torch.tensor(x)], lambda i: lq[i], gen
                )
                if tau == L:
                    whole.append(x)
                else:
                    part.append(_sample(P, rng, list(x[:tau]) + [y], L))
            outs = (whole + part)[:G]
            cnt[tuple(idx[z] for z in outs)] += 1
            continue
        while True:
            M = grp.next_m()
            if M is None:
                break
            x = _sample(Q, rng, (), L)
            lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
            tau, y, _, _ = verify_engine_q(
                lp, torch.tensor(x), lq[torch.arange(L), torch.tensor(x)], lambda i: lq[i], gen, M
            )
            grp.record(y != NULL_TOKEN)
            if y == NULL_TOKEN:
                continue
            outs.append(x if tau == L else _sample(P, rng, list(x[:tau]) + [y], L))
        while len(outs) < G:  # drafts ran out: fresh decodes
            outs.append(_sample(P, rng, (), L))
        cnt[tuple(idx[z] for z in outs)] += 1
    return cnt, np.array([_seq_prob(P, z) for z in seqs])


def _chi2_z(cnt, p1):
    n = cnt.sum()
    exp = n * np.einsum("i,j->ij", p1, p1)
    chi2 = ((cnt - exp) ** 2 / exp).sum()
    dof = cnt.size - 1
    return (chi2 - dof) / math.sqrt(2 * dof)


def test_sampler_slots_match_p_by_chi_square_and_control_does_not():
    V, L, N, G, n = 2, 3, 4, 2, 12000
    P, Q = _model(V, L, 3), _model(V, L, 4)
    policy = _RulePolicy()
    cnt, p1 = _mc_counts(P, Q, V, L, N, G, policy, n, seed=1)
    z = _chi2_z(cnt, p1)
    assert abs(z) < 4, z
    bad, _ = _mc_counts(P, Q, V, L, N, G, policy, n // 3, seed=2, prefer_whole=True)
    zb = _chi2_z(bad, p1)
    assert zb > 20, zb


# --------------------------------------------------------------------------
# Samplers pathwise identical with null_m
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1.0, 1.3, 2.0, 4.0])
def test_eager_lazy_engine_q_identical_with_null(M):
    torch.manual_seed(0)
    V = 40
    n_bot = 0
    for trial in range(60):
        g = 1 + trial % 9
        lp = torch.log_softmax(torch.randn(g + 1, V) * 2, -1)
        lq = torch.log_softmax(lp[:g] + torch.randn(g, V) * (0.3 + trial % 3), -1)
        draft = torch.multinomial(lq.exp(), 1).flatten()
        outs, states = [], []
        for fn in ("eager", "lazy", "engine"):
            gen = torch.Generator().manual_seed(trial)
            if fn == "eager":
                tau, y = block_verify(lp, lq, draft, gen, null_m=M)
            elif fn == "lazy":
                tau, y = block_verify_lazy(lp, lq, draft, gen, null_m=M)
            else:
                tau, y, logp, _ = verify_engine_q(
                    lp, draft, lq[torch.arange(g), draft], lambda i: lq[i], gen, M
                )
                assert logp.numel() == (tau if y == NULL_TOKEN else tau + 1)
            outs.append((tau, y))
            states.append(gen.get_state())
        assert outs[0] == outs[1] == outs[2], outs
        assert all(torch.equal(states[0], s) for s in states[1:])
        if outs[0][1] == NULL_TOKEN:
            assert M > 1 and outs[0][0] == 0
            n_bot += 1
    if M >= 2:
        assert n_bot > 0


def test_null_m_one_is_plain_block_verification():
    torch.manual_seed(1)
    V, g = 30, 6
    lp = torch.log_softmax(torch.randn(g + 1, V), -1)
    lq = torch.log_softmax(torch.randn(g, V), -1)
    draft = torch.multinomial(lq.exp(), 1).flatten()
    a = output_distribution_given_draft(lp, lq, draft)
    b = output_distribution_given_draft(lp, lq, draft, 1.0)
    assert all(i == j and torch.equal(x, y) for (i, x), (j, y) in zip(a, b))


def test_null_rejects_empty_drafts_and_m_below_one():
    lp = torch.log_softmax(torch.randn(1, 5), -1)
    with pytest.raises(ValueError, match="at least one token"):
        block_verify(lp, lp[:0], torch.zeros(0, dtype=torch.long), null_m=2.0)
    lp = torch.log_softmax(torch.randn(3, 5), -1)
    with pytest.raises(ValueError, match=">= 1"):
        block_verify(lp, lp[:2], torch.zeros(2, dtype=torch.long), null_m=0.5)


# --------------------------------------------------------------------------
# Policy
# --------------------------------------------------------------------------


def test_tail_decode_model_matches_the_report():
    # Brownian model of log w, v = 8.1e-7.
    a32, a64 = math.sqrt(32768 * 8.1e-7), math.sqrt(65536 * 8.1e-7)
    assert abs((1 - tail_decode_prob(1.0, a32)) - 0.8765) < 2e-3
    assert abs(tail_decode_prob(1.5, a32) / 8.1e-4 - 1) < 0.05
    assert abs(tail_decode_prob(2.0, a64) / 2.3e-4 - 1) < 0.05


def test_dp_policy_counts_only():
    pol = NullPolicy(logratio_var=8e-7, ref_len=32768)
    assert pol.m(24, 16) == 1.3  # report: G = 16, N = 24 starts at 1.3
    for d in range(1, 30):
        for s in range(1, 17):
            m = pol.m(d, s)
            assert m >= 1.0
            if d <= s:
                assert m == 1.0
    assert NullPolicy(fixed_m=2.0).m(5, 2) == 2.0
    assert NullPolicy(fixed_m=2.0).m(2, 2) == 1.0
    with pytest.raises(ValueError):
        NullPolicy(fixed_m=0.9)
    assert drafts_per_prompt(1.5, 16) == 24 and drafts_per_prompt(1.0, 8) == 8
    assert drafts_per_prompt(1.2, 2) == 2
    with pytest.raises(ValueError):
        drafts_per_prompt(0.5, 8)


def test_null_group_stops_when_slots_fill():
    grp = NullGroup(5, 2, NullPolicy(fixed_m=2.0))
    assert grp.next_m() == 2.0
    grp.record(False)
    assert grp.next_m() == 2.0
    grp.record(True)
    assert grp.next_m() == 2.0  # 3 drafts left for 1 slot
    grp.record(True)
    assert grp.next_m() is None and grp.verified == 3 and grp.bots == 1


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------








# --------------------------------------------------------------------------
# Driver: K G drafts -> per-prompt pools -> plans for the first G outputs
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Worker: per-group sequential decisions with the real sampler
# --------------------------------------------------------------------------


@cuda
def test_worker_verifies_every_row_at_every_envelope(monkeypatch):
    """The worker's grid results equal the reference grid sampler on the same
    generator stream, rows in the worker's order (its rows by length)."""
    from test_engine_q import _engine_drafts, _engine_worker, _hidden, _logits, _lsm

    w, emb, heads = _engine_worker(monkeypatch)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads, n=20, seed=5)
    keep = [i for i in range(20) if rows[i].numel() > plens[i]]
    rows, plens, segs, lqs, hids, keys = (
        [x[i] for i in keep] for x in (rows, plens, segs, lqs, hids, keys)
    )
    Ms = (1.0, 1.5, 2.0)
    res = dict(
        w.verify_drafts_block_engine(
            rows=rows,
            prompt_lens=plens,
            segments=segs,
            engine_logq=lqs,
            hidden=hids,
            vocab_limit=1 << 30,
            seed=11,
            keys=keys,
            batch_tokens=64,
            null_envelopes=[list(Ms)] * len(rows),
            row_ranks=[0] * len(rows),
        )
    )
    assert sorted(res) == list(range(len(rows)))
    gen = torch.Generator(device="cuda").manual_seed(11 * 1_000_003)
    for i in sorted(range(len(rows)), key=lambda i: rows[i].numel()):
        n, plen = rows[i].numel(), plens[i]
        h_all = _hidden(rows[i].cuda(), emb)
        lp = _lsm(_logits(h_all[plen - 1 : n], heads[-1]).float())
        vers = [v for v, a, b in segs[i] for _ in range(a, b)]
        lq = torch.stack(
            [_lsm(_logits(h_all[plen - 1 + t], heads[v]).float()) for t, v in enumerate(vers)]
        )
        draft = rows[i][plen:].cuda()
        grid, lpx, _ = verify_engine_q_grid(
            lp, draft, torch.tensor(lqs[i], device="cuda"), lambda j: lq[j], gen, Ms
        )
        assert res[i]["null_grid"] == grid, i
        tau = max(t for t, _, _ in grid.values())
        assert res[i]["token_logprobs"] == lpx[:tau].tolist()

def test_logratio_calibration_uses_past_steps_only():
    from thundersync_rl.null_bv import LogRatioCalibration

    cal = LogRatioCalibration(prior=8e-7, decay=0.5)
    assert cal.estimate() == 8e-7  # no step seen yet: the prior
    # One step: tokens with log p/q in {-0.1, 0.1}: variance 0.01.
    cal.update([(2, 0.0, 0.02), (2, 0.0, 0.02)])
    assert cal.estimate() == pytest.approx(0.01)
    cal.update([(4, 0.0, 0.0)])  # a step with variance 0: halfway (decay 0.5)
    assert cal.estimate() == pytest.approx(0.005)
    cal.update([(1, 0.3, 0.09)])  # fewer than 2 tokens: ignored
    assert cal.estimate() == pytest.approx(0.005)
    with pytest.raises(ValueError):
        LogRatioCalibration(prior=1e-6, decay=1.0)


def test_larger_logratio_var_and_tail_weight_raise_the_envelope():
    from thundersync_rl.null_bv import NullPolicy

    base = NullPolicy(logratio_var=8e-7, ref_len=1024)
    calibrated = NullPolicy(logratio_var=1e-4, ref_len=1024)
    assert base.m(12, 8) < calibrated.m(12, 8)
    for v in (1e-5, 1e-4):
        plain = NullPolicy(logratio_var=v, ref_len=1024)
        weighted = NullPolicy(logratio_var=v, ref_len=1024, tail_weight=4.0)
        assert all(weighted.costs()[m] == 4.0 * plain.costs()[m] for m in plain.grid)
        assert weighted.m(12, 8) >= plain.m(12, 8)
    # Still counts only, and M = 1 once drafts no longer exceed slots.
    assert weighted.m(8, 8) == 1.0


def test_verify_cost_is_measured_from_past_steps_and_lowers_the_envelope():
    from thundersync_rl.null_bv import LogRatioCalibration, NullPolicy

    cal = LogRatioCalibration(prior=1e-4, decay=0.0)
    assert cal.verify_cost(1024) == 0.0  # nothing measured yet
    cal.update_verify(seconds=1.6, rows=160)  # 10 ms per verified draft
    assert cal.verify_cost(1024) == 0.0  # decode latency still unknown
    cal.update_decode(seconds=5.12, tokens=1024)  # 5 ms per token: a full decode is 5.12 s
    assert cal.verify_cost(1024) == pytest.approx(0.01 / 5.12)
    free = NullPolicy(logratio_var=1e-4, ref_len=1024)
    costly = NullPolicy(logratio_var=1e-4, ref_len=1024, verify_cost=0.2)
    assert costly.m(12, 8) <= free.m(12, 8)
    assert costly.m(12, 8) < free.m(12, 8) or free.m(12, 8) == 1.0
    with pytest.raises(ValueError):
        NullPolicy(verify_cost=-1.0)




def _mom(v, n=1000):
    """One row's (tokens, sum, sum of squares) with mean 0 and variance v."""
    return [(n, 0.0, n * v)]


def test_logratio_calibration_separates_draft_ages():
    """long200 (LA2): steps 101-102 verify age-1 drafts (v ~3x lower) after the step-100 save boundary. The age-2
    step 103 must not be calibrated from them (pooled EMA: 1.3e-5 for a true 5e-5)."""
    from thundersync_rl.null_bv import LogRatioCalibration

    cal = LogRatioCalibration(prior=8e-7, decay=0.5)
    for step in range(95, 100):
        cal.update(_mom(2e-5), age=2, step=step)
    cal.update(_mom(7e-6), age=1, step=101)
    cal.update(_mom(7e-6), age=1, step=102)
    pooled = cal.estimate()
    assert pooled == pytest.approx(0.25 * 2e-5 + 0.75 * 7e-6)
    assert cal.estimate(age=2, step=103) == pytest.approx(2e-5)  # age-2 EMA, 4 steps old: fresh
    assert cal.estimate(age=1, step=103) == pytest.approx(7e-6)
    # An age with no recent update falls back to the most recently updated age.
    assert cal.estimate(age=3, step=103) == pytest.approx(7e-6)
    # All ages stale: the pooled EMA. No age given: the pooled EMA (legacy callers).
    assert cal.estimate(age=2, step=200) == pytest.approx(pooled)
    assert cal.estimate() == pytest.approx(pooled)


def test_draft_age_follows_the_cohort_version():
    from types import SimpleNamespace

    from thundersync_rl.speculative import SpeculativeGeneration

    s = SimpleNamespace(current_step=12)
    assert SpeculativeGeneration._draft_age(s) == 1  # lookahead 1 / fallback: theta_{k-1}
    s.cohort_draft_version = 10
    assert SpeculativeGeneration._draft_age(s) == 2  # lookahead 2 early cohort: theta_{k-2}


# ---- VG-1.4: walk every draft at a constant M, group = the non-⊥ outputs (research/colo/TAIL_FLOOR.md §4.5) ----


def test_walk_all_policy_is_constant_m_at_every_count():
    from thundersync_rl.null_bv import NullGroup, NullPolicy, decide_pools, envelopes, row_envelopes

    pol = NullPolicy(fixed_m=1.4, walk_all=True)
    assert envelopes(pol) == (1.4,)
    for d in range(1, 9):
        for s in range(1, 9):
            assert pol.m(d, s) == 1.4  # no M = 1 fallback, also when d <= s
    assert all(row_envelopes(pol, 8, 8, j) == (1.4,) for j in range(8))
    # slots = N: every draft is decided (no stopping rule), whatever the outcomes.
    for pattern in itertools.product([False, True], repeat=8):
        out = decide_pools(["a"] * 8, {"a": 8}, [{1.4: o} for o in pattern], pol)
        assert out == [1.4] * 8
    st = NullGroup(8, 8, pol)
    for o in (True, False, True):
        assert st.next_m() == 1.4
        st.record(o)
    assert (st.outputs, st.bots, st.dropped) == (2, 1, False)
    with pytest.raises(ValueError):
        NullPolicy(walk_all=True)  # needs a constant envelope


def _bot_bv_one_token(p, q, m, rng):
    """Exact single-token ⊥-BV against p/M (+) (1 - 1/M)⊥: returns (draft, output or -1 for ⊥)."""
    x = int(rng.choice(len(q), p=q))
    if rng.random() < min(1.0, p[x] / (m * q[x])):
        return x, x
    res = np.maximum(p / m - q, 0.0)
    w = np.append(res, 1.0 - 1.0 / m)
    y = int(rng.choice(len(w), p=w / w.sum()))
    return x, (-1 if y == len(p) else y)


def test_variable_group_rows_are_iid_p_and_content_selection_is_not():
    """VG: kept rows (all non-⊥ outputs of groups of size >= 2) are p-distributed; dropping each group's lowest-p
    output (the vg_by_logp negative) is not."""
    rng = np.random.default_rng(7)
    p = np.array([0.5, 0.3, 0.15, 0.05])
    q = np.array([0.35, 0.35, 0.2, 0.1])
    n, m = 8, 1.4
    kept = np.zeros(4)
    neg = np.zeros(4)
    sizes = []
    for _ in range(20000):
        outs = [y for _, y in (_bot_bv_one_token(p, q, m, rng) for _ in range(n)) if y >= 0]
        sizes.append(len(outs))
        if len(outs) >= 2:
            for y in outs:
                kept[y] += 1
        if len(outs) >= 3:
            outs.remove(min(outs, key=lambda y: p[y]))
            for y in outs:
                neg[y] += 1
    assert abs(np.mean(sizes) - n / m) < 0.05  # size ~ Bin(N, 1/M)

    def chi2(c):
        e = c.sum() * p
        return float(((c - e) ** 2 / e).sum())

    assert chi2(kept) < 16.3  # 3 dof, p > 1e-3
    assert chi2(neg) > 100.0


@pytest.mark.parametrize("seed", range(12))
def test_spare_pools_wait_for_a_drop_and_select_as_verifying_every_pool(seed):
    """XPR-S lazy waves (``_null_wave``): under prompt drop with spare slots, a pool's rows are forwarded only if it
    fits the selection walk (earlier deciding pools counted undropped), so the prompts the slot selection keeps
    (``_select_prompts`` / ``null_bv.select_pools``) and every kept row's decision are those of verifying every pool,
    at fewer forwarded rows. Covers a duplicated prompt (one pool of 2 x G slots) and a prompt without drafts."""
    from thundersync_rl.null_bv import select_pools
    from thundersync_rl.speculative import SpeculativeGeneration

    rng = np.random.default_rng(seed)
    m, g, n, prompts, spares = 1.25, 4, 6, 10, 3
    policy = NullPolicy(fixed_m=m, prompt_drop=True)
    keys = [(k,) for k in range(prompts)]
    dup, empty = (keys[int(k)] for k in rng.choice(prompts, size=2, replace=False))
    all_slots = {k: 2 * g if k == dup else g for k in keys}
    names = {k: f"g{k[0]}" for k in keys if k != empty}
    groups = [names[k] for k in keys if k in names for _ in range(2 * n if k == dup else n)]
    p_bot = 0.0 if seed % 4 == 0 else float(rng.uniform(0.1, 0.45))  # none / some pools drop
    grid = [{m: (0, NULL_TOKEN if rng.random() < p_bot else 0, -1.0)} for _ in groups]

    def run(spare_prompts):
        # Slim's ⊥-BV wave loop is verify_post / verify_collect: drive it with the first wave of verify_post.
        s = object.__new__(SpeculativeGeneration)
        s._null_policy, s.stats, s.plans = policy, {}, {}
        s.null_bv_drop_is = False
        s.null_bv_spare_prompts, s.null_bv_spare_slots = spare_prompts, spares * g
        s._null_names, s._null_all_slots = names, all_slots
        s._release_hidden = lambda ds: list(ds)
        s._v0_kw = s._v0_cap_kw = lambda: {}
        s._vprof_add = lambda key, v: None
        s._v0_settle = s._vit_place = s._count_tails = lambda: None
        out = {}
        s._finish_verify_null = lambda flat, by_row, *a: out.update(by_row)
        forwarded = []

        def verify_block(flat, rows, null_envelopes, **kw):
            forwarded.extend(i for i, _ in flat)
            return {j: {"null_grid": grid[i], "token_logprobs": [], "draft_logq": []} for j, (i, _) in enumerate(flat)}

        s._verify_block = verify_block
        flat = [(i, (None, None, [0]) + (None,) * 5) for i in range(len(groups))]  # one drafted token (empty drafts take the coin)
        slots = {names[k]: all_slots[k] for k in names}
        pools = s._null_pools(flat, groups, slots)
        h = dict(flat=flat, rows=[None] * len(groups), groups=groups, slots=slots, dropped=[], n_empty=0, pools=pools,
                 batch=s._null_wave(pools, names, all_slots), names=names, all_slots=all_slots, t_post=0.0, futs=None)
        s.verify_collect(h)
        inv = {v: k for k, v in names.items()}
        s.null_dropped_keys = {inv[x] for x in s._null_dropped}
        s.null_bv_spare_prompts = spares
        SpeculativeGeneration._select_prompts(s)
        return out, s.null_selected_keys, forwarded, s.null_dropped_keys

    every, kept_every, fwd_every, dropped = run(0)
    lazy, kept_lazy, fwd_lazy, _ = run(spares)
    train = sum(all_slots.values()) - spares * g
    assert kept_every == select_pools(keys, all_slots, train, dropped)
    assert kept_lazy == kept_every
    kept_groups = {names[k] for k in kept_every if k in names}
    rows = [i for i, x in enumerate(groups) if x in kept_groups]
    assert [lazy[i].get("null_m") for i in rows] == [every[i].get("null_m") for i in rows]
    assert set(fwd_lazy) <= set(fwd_every)
    # Without drops, only the pools of the first walk (no spare) are forwarded.
    if not dropped:
        used, active = 0, set()
        for k in keys:
            if used + all_slots[k] <= train:
                used += all_slots[k]
                active |= {names[k]} if k in names else set()
        assert {groups[i] for i in fwd_lazy} <= active and len(fwd_lazy) < len(fwd_every)


def test_select_pools_fits_slots_in_order_then_fills_from_dropped():
    """Count-only XPR-S selection: undropped pools in order while their slots fit (a pool that does not fit is
    skipped, a later smaller one can still fit), then a shortfall from the dropped ones in order."""
    from thundersync_rl.null_bv import select_pools

    order = ["a", "b", "c", "d", "e"]
    slots = {"a": 8, "b": 16, "c": 8, "d": 8, "e": 8}
    assert select_pools(order, slots, 24, set()) == ["a", "b"]
    assert select_pools(order, slots, 24, {"b"}) == ["a", "c", "d"]
    assert select_pools(order, slots, 16, {"a"}) == ["b"]
    assert select_pools(order, slots, 32, {"c", "d", "e"}) == ["a", "b", "c"]


def test_null_empty_walked_keys():
    """Metric names (from ts-colo 3e5fc420; slim has no streaming driver test): the per-verify count of empty drafts
    is spec/null_empty_walked, the run total spec/null_empty_walked_total; spec/null_empty_dropped is gone."""
    import inspect

    from thundersync_rl.speculative import SpeculativeGeneration, _NullPoolRun

    s = object.__new__(SpeculativeGeneration)
    s.stats, s.current_step, s._empty_coin_seed = {}, 3, (1, "t")
    pol = NullPolicy(fixed_m=2.0)
    pools = {"g": _NullPoolRun([0, 1], 2, pol)}
    flat = [(i, (torch.tensor([1, 2]), None, [])) for i in range(2)]
    s._null_empty_results(pools, ["g", "g"], flat, [0, 1])
    s._null_empty_results(pools, ["g", "g"], flat, [1])
    assert s.stats["spec/null_empty_walked_total"] == 3.0
    src = inspect.getsource(SpeculativeGeneration._finish_verify_null)
    assert '"spec/null_empty_walked"' in src and "null_empty_dropped" not in src
