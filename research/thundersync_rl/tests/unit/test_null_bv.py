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
from thundersync_rl.null_bv import (
    NullGroup,
    NullPolicy,
    decide_pools,
    drafts_per_prompt,
    envelopes,
    tail_decode_prob,
)

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


@pytest.mark.parametrize(
    "policy", [NullPolicy(fixed_m=2.0), _RulePolicy()], ids=["fixed2", "d_over_s"]
)
def test_driver_decisions_over_coupled_grid_outcomes_are_iid_p(policy):
    """Exactness of grid verification + driver decisions, by enumeration.

    Every row is verified at every envelope with one set of draws (outcomes
    coupled across M) and ``decide_pools`` takes each row's outcome at the M
    the counts give; the G slots must be iid p for any coupling.
    """
    V, L, N, G = 2, 2, 3, 2
    P, Q = _model(V, L, 7, conc=0.5), _model(V, L, 8, conc=0.5)
    seqs = list(itertools.product(range(V), repeat=L))
    p1 = {z: _seq_prob(P, z) for z in seqs}
    Ms = envelopes(policy) if policy.fixed_m is not None else (1.0, 1.5, 2.0, 3.0)
    row = list(_coupled_outcomes(P, Q, V, L, Ms).items())
    assert abs(sum(v for _, v in row) - 1.0) < 1e-12
    joint: dict = {}
    for combo in itertools.product(row, repeat=N):
        pr = math.prod(v for _, v in combo)
        outcomes = [{M: vec[k] != "BOT" for k, M in enumerate(Ms)} for vec, _ in combo]
        chosen = decide_pools(["a"] * N, {"a": G}, outcomes, policy)
        outs = tuple(
            vec[Ms.index(m)]
            for (vec, _), m in zip(combo, chosen)
            if m is not None and vec[Ms.index(m)] != "BOT"
        )
        fill = G - len(outs)  # drafts ran out: fresh p decodes
        for tail in itertools.product(seqs, repeat=fill):
            key = outs + tail
            joint[key] = joint.get(key, 0.0) + pr * math.prod(p1[z] for z in tail)
    tv = 0.5 * sum(
        abs(joint.get(zs, 0.0) - math.prod(p1[z] for z in zs))
        for zs in itertools.product(seqs, repeat=G)
    )
    assert tv < 1e-9, tv


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
    # agent_multi_draft.md section 2.3 (Brownian model, v = 8.1e-7).
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


def _master(**ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        verify_on="learner",
        draft_lookahead=1,
        null_bv=True,
    )
    base.update(ts)
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        grpo=SimpleNamespace(max_rollout_turns=1),
        policy={"generation": {"max_new_tokens": 2048}},
    )


def test_null_config_default_off_and_policy_args():
    from thundersync_rl.grpo_loop import ThunderSyncConfig, null_bv_policy, validate_null_bv

    c = ThunderSyncConfig()
    assert c.null_bv is False and c.null_bv_drafts_per_output == 1.5
    validate_null_bv(_master())
    assert null_bv_policy(_master(null_bv_fixed_m=1.5)) == {
        "logratio_var": 8e-7,
        "ref_len": 2048,
        "fixed_m": 1.5,
        "prompt_drop": False,
    }


@pytest.mark.parametrize(
    "ts,match",
    [
        ({"verify_source": "scorer"}, "verify_source"),
        ({"verify_on": "both"}, "verify_on"),
        ({"draft_lookahead": 3}, "draft_lookahead"),
        ({"speculate": False}, "speculate"),
        ({"fused_verify_train": True}, "fused_verify_train"),
        ({"null_bv_drafts_per_output": 0.5}, "drafts_per_output"),
        ({"null_bv_fixed_m": 0.5}, "fixed_m"),
    ],
)
def test_null_config_rejects(ts, match):
    from thundersync_rl.grpo_loop import validate_null_bv

    with pytest.raises(ValueError, match=match):
        validate_null_bv(_master(**ts))


# --------------------------------------------------------------------------
# Driver: K G drafts -> per-prompt pools -> plans for the first G outputs
# --------------------------------------------------------------------------


def test_driver_pools_drafts_and_fills_slots_in_order(monkeypatch):
    from test_speculative_driver import EOD, _spec as make_spec

    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    # Two prompts, G = 2, N = 3 drafts each (K = 1.5); prompt B's 2nd draft is
    # empty (cut before its first token) and is dropped from its pool.
    pa, pb = [1, 2], [3]
    drafted = [
        ([4, 5, EOD], True, [-0.1] * 3),  # A0: BOT
        ([6, EOD], True, [-0.1] * 2),  # A1: whole
        ([7, 8], False, [-0.1] * 2),  # A2: rejected at 1 -> 9
        ([10, EOD], True, [-0.1] * 2),  # B0: whole
        ([], False, []),  # B1: empty -> dropped
        ([11, 12, 13], False, [-0.1] * 3),  # B2: scripted BOT (see below)
    ]
    started = []
    calls = []
    # Scripted outcomes by row key: (accepted, next); next = -1 is BOT.
    script = {"1:0": (0, -1), "1:1": (2, 0), "1:2": (1, 9), "1:3": (2, 0), "1:5": (0, -1)}

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, **kw):
        # Every row at every envelope (the script: same outcome at each M).
        calls.append(kw)
        return [
            (
                i,
                {
                    "null_grid": {
                        M: (script[k][0], script[k][1], None if script[k][1] < 0 else -1.0)
                        for M in kw["null_envelopes"][i]  # this row's envelopes
                    },
                    "token_logprobs": [-1.0] * script[k][0],
                    "draft_logq": engine_logq[i][: script[k][0]],
                },
            )
            for i, k in enumerate(keys)
        ]

    engine = {
        "install_draft_hidden_log": lambda **_: True,
        "start_drafts": lambda prompts, budget, interval, hidden_keys=None: started.append(
            (prompts, hidden_keys)
        ),
        "draft_request_ids": lambda: {},
        "collect_drafts": lambda records=None: drafted,
        "pop_draft_hidden": lambda keys, lengths: {
            k: torch.ones(n, 4) for k, n in zip(keys, lengths) if n
        },
    }
    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=engine,
        verify_source="engine",
        hidden_transport="driver",
        null_bv=True,
        null_bv_policy={"fixed_m": 2.0},
    )
    s.current_step = 0
    prompts = [torch.tensor(pa)] * 3 + [torch.tensor(pb)] * 3
    collect = asyncio.run(
        s.draft_streaming(prompts, null_slots={tuple(pa): 2, tuple(pb): 2})
    )
    drafts = collect()
    s.current_step = 1
    s.verify(drafts)
    (kw,) = calls
    # Every row verified at the envelopes its pool index can be decided at
    # (row_envelopes: A0 d=3,s=2 -> 2; A1 d=2,s in 1..2 -> 2 or 1; A2 and the
    # B rows -> 1; B1 empty, dropped), on the ranks the driver placed it on.
    assert kw["null_envelopes"] == [[2.0], [1.0, 2.0], [1.0], [1.0], [1.0]]
    assert kw["row_ranks"] == [0] * 5
    plans_a = s.plans[tuple(pa)]
    assert [p.prefix for p in plans_a] == [[6, EOD], [7, 9]]
    assert plans_a[0].complete and not plans_a[1].complete
    # B: 2 drafts for 2 slots -> M = 1 for both (BOT is then impossible; the
    # fake scripts B2 as BOT only to exercise the bookkeeping): B0 whole, one
    # slot left empty -> fresh decode.
    plans_b = s.plans[tuple(pb)]
    assert [p.prefix for p in plans_b] == [[10, EOD]]
    st = s.stats
    assert st["spec/null_slots"] == 4.0 and st["spec/null_outputs"] == 3.0
    assert st["spec/null_fresh_slots"] == 1.0
    assert st["spec/null_empty_dropped"] == 1.0
    assert st["spec/null_bot_frac"] == pytest.approx(2 / 5)
    assert st["spec/null_tail_frac"] == pytest.approx(1 / 3)
    assert st["spec/null_drafts_per_output"] == pytest.approx(5 / 3)
    # Rollouts: two of prompt B; the second has no plan and decodes fresh.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    datum = BatchedDataDict({"input_ids": torch.tensor([pb]), "input_lengths": torch.tensor([1])})
    _, o1 = asyncio.run(s._one(0, datum))
    _, o2 = asyncio.run(s._one(1, datum))
    assert o1["output_ids"][0].tolist() == [3, 10, EOD]
    assert o2["output_ids"][0].tolist() == [3, 9, EOD]  # fake engine's fresh tail


# --------------------------------------------------------------------------
# Worker: per-group sequential decisions with the real sampler
# --------------------------------------------------------------------------


@cuda
def test_worker_verifies_every_row_at_every_envelope(monkeypatch):
    """The worker's grid results equal the reference grid sampler on the same
    generator stream, rows in the worker's order (its rows by length)."""
    from test_engine_q import _engine_drafts, _engine_worker, _hidden, _logits

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
        lp = torch.log_softmax(_logits(h_all[plen - 1 : n], heads[-1]).float(), -1)
        vers = [v for v, a, b in segs[i] for _ in range(a, b)]
        lq = torch.stack(
            [torch.log_softmax(_logits(h_all[plen - 1 + t], heads[v]).float(), -1) for t, v in enumerate(vers)]
        )
        draft = rows[i][plen:].cuda()
        grid, lpx, _ = verify_engine_q_grid(
            lp, draft, torch.tensor(lqs[i], device="cuda"), lambda j: lq[j], gen, Ms
        )
        assert res[i]["null_grid"] == grid, i
        tau = max(t for t, _, _ in grid.values())
        assert res[i]["token_logprobs"] == lpx[:tau].tolist()

def test_driver_forwards_only_rows_that_can_be_used(monkeypatch):
    """Waves: per prompt the next ceil(s * M) drafts are forwarded; drafts past
    the filled slots are never forwarded and their hidden states are freed."""
    from test_speculative_driver import EOD, _spec as make_spec

    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    N, G = 6, 2
    drafted = [([4 + j, EOD], True, [-0.1] * 2) for j in range(N)]
    sent = []

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, **kw):
        sent.append(list(keys))
        return [
            (
                i,
                {
                    "null_grid": {M: (2, 0, -1.0) for M in kw["null_envelopes"][i]},
                    "token_logprobs": [-1.0, -1.0],
                    "draft_logq": engine_logq[i][:2],
                },
            )
            for i in range(len(keys))
        ]

    engine = {
        "install_draft_hidden_log": lambda **_: True,
        "start_drafts": lambda prompts, budget, interval, hidden_keys=None: None,
        "draft_request_ids": lambda: {},
        "collect_drafts": lambda records=None: drafted,
        "pop_draft_hidden": lambda keys, lengths: {k: torch.ones(n, 4) for k, n in zip(keys, lengths) if n},
    }
    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers=engine,
        verify_source="engine",
        hidden_transport="driver",
        null_bv=True,
        null_bv_policy={"fixed_m": 2.0},
    )
    freed = []
    s._free_hidden = lambda hs: freed.extend(hs)
    s.current_step = 0
    collect = asyncio.run(s.draft_streaming([torch.tensor([1])] * N, null_slots={(1,): G}))
    drafts = collect()
    s.current_step = 1
    s.verify(drafts)
    # one wave of ceil(2 * 2.0) = 4 drafts; both slots filled by the first two
    assert sent == [["1:0", "1:1", "1:2", "1:3"]]
    assert [p.prefix for p in s.plans[(1,)]] == [[4, EOD], [5, EOD]]
    assert s.stats["spec/null_skipped_drafts"] == 2.0 and s.stats["spec/null_waves"] == 1.0
    assert s.stats["spec/null_unused_drafts"] == 4.0  # 2 forwarded + 2 never forwarded
    assert len(freed) >= 2


@pytest.mark.parametrize("seed", range(5))
def test_wave_decisions_equal_the_one_shot_process(seed):
    """``_verify_null`` (waves, incremental decisions) takes every row at the
    envelope ``decide_pools`` gives it over the same outcomes."""
    from thundersync_rl.speculative import SpeculativeGeneration

    rng = np.random.default_rng(seed)
    policy = NullPolicy(logratio_var=0.2, ref_len=3)
    ms = envelopes(policy)
    groups = [f"g{int(k)}" for k in rng.integers(0, 3, size=24)]
    slots = {g: int(rng.integers(1, 5)) for g in set(groups)}
    grid = [{M: (0, 0 if rng.random() < 1 / M else NULL_TOKEN, -1.0) for M in ms} for _ in groups]
    s = object.__new__(SpeculativeGeneration)
    s._null_policy, s.stats = policy, {}
    s.null_bv_drop_is = False
    s._release_hidden = lambda ds: list(ds)

    def verify_block(flat, rows, null_envelopes):
        return {
            j: {"null_grid": grid[i], "token_logprobs": [], "draft_logq": []}
            for j, (i, _) in enumerate(flat)
        }

    s._verify_block = verify_block
    flat = [(i, (None,) * 8) for i in range(len(groups))]
    out = s._verify_null(flat, [None] * len(groups), groups, slots)
    want = decide_pools(groups, slots, [{M: o[1] >= 0 for M, o in g.items()} for g in grid], policy)
    assert [out[i].get("null_m") for i in range(len(groups))] == want
