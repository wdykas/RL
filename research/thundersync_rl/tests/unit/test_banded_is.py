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
"""Banded importance sampling (``thundersync.banded_is_band``): exactness and plumbing.

* exact enumeration on a toy autoregressive model: every output keeps the draft
  through the band exit T, its weight w is a function of the output alone, and
  E[w 1{Y = y}] = p(y) (the law of Y is p / w), alone and composed with ⊥-BV
  (non-BOT outcomes: E[w 1{Y = y}; out] = p(y) / M, BOT with probability 1 - 1/M);
* GRPO groups by exact enumeration: per-row weights reproduce the exact-sampling
  expectation for RLOO advantages; with the leave-one-out std normalisation
  (NeMo-RL's default) they give the same direction with a different scale, and
  the group's product of weights reproduces it exactly;
* eager / lazy / engine-q samplers pathwise identical with ``band``;
* config validation, the per-row loss weight in the training batch, the driver's
  plans, claims and statistics.
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
    band_exit,
    banded_log_weight,
    block_verify,
    block_verify_lazy,
    output_distribution_given_draft,
)
from thundersync_rl.engine_q import verify_engine_q

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


def _rows(m, x, n, dtype=torch.float64):
    return torch.tensor(np.log(np.stack([m[tuple(x[:t])] for t in range(n)])), dtype=dtype)


def _outcomes(P, Q, x, band, M=1.0):
    """Exact outcomes of banded verification of draft x: [(output or "BOT",
    prob, log w)]; tails after a rejection decoded from p."""
    V, L = len(P[()]), len(x)
    lp, lq = _rows(P, x, L + 1), _rows(Q, x, L)
    ar = torch.arange(L)
    lr = lp[ar, torch.tensor(x)] - lq[ar, torch.tensor(x)]
    t_exit = band_exit(lr, band)
    out = []
    for tau, probs in output_distribution_given_draft(lp, lq, torch.tensor(x), M, band):
        # fp64 rounding leaves |P| ~ 1e-17 on impossible outcomes (1 - h, h = 1).
        probs = probs.double().numpy()
        probs[np.abs(probs) < 1e-14] = 0.0
        if probs.sum() == 0:
            continue
        if tau == L:  # length cap: the bonus token is never used
            log_w, _ = banded_log_weight(lr, band)
            out.append((tuple(x), probs.sum(), log_w))
            continue
        for y, py in enumerate(probs):
            if py == 0:
                continue
            if y == V:
                assert tau == 0 and M > 1
                out.append(("BOT", py, None))
                continue
            # Every output keeps the draft through the band exit.
            assert t_exit is not None and tau >= t_exit, (tau, t_exit)
            log_w, exited = banded_log_weight(lr[:tau], band)
            assert exited
            pre = list(x[:tau]) + [y]
            for tail in itertools.product(range(V), repeat=L - len(pre)):
                z = tuple(pre) + tail
                pz = py * math.prod(P[z[:t]][z[t]] for t in range(len(pre), L))
                out.append((z, pz, log_w))
    return out


def _banded_law(P, Q, band, M=1.0):
    """{y: (P(Y = y), w(y))} and P(BOT) over drafts x ~ Q; asserts w is a
    function of the output."""
    V, L = len(P[()]), max(len(k) for k in P)
    law, w_of, bot = {}, {}, 0.0
    for x in itertools.product(range(V), repeat=L):
        qx = _seq_prob(Q, x)
        for z, pz, log_w in _outcomes(P, Q, x, band, M):
            if z == "BOT":
                bot += qx * pz
                continue
            law[z] = law.get(z, 0.0) + qx * pz
            w = math.exp(log_w)
            assert abs(w_of.setdefault(z, w) - w) < 1e-12 * w, (z, w_of[z], w)
    return {z: (law[z], w_of[z]) for z in law}, bot


# --------------------------------------------------------------------------
# Exactness of the per-draft law
# --------------------------------------------------------------------------


@pytest.mark.parametrize("band", [0.15, 0.5, 1.5, 1e9])
@pytest.mark.parametrize("M", [1.0, 1.7])
def test_weighted_law_is_p_by_enumeration(band, M):
    V, L = 3, 3
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    law, bot = _banded_law(P, Q, band, M)
    assert abs(bot - (1 - 1 / M)) < 1e-12
    err = max(abs(law.get(z, (0.0, 1.0))[0] * law.get(z, (0.0, 1.0))[1] - _seq_prob(P, z) / M) for z in seqs)
    assert err < 1e-12, err
    # Rows are exact in expectation, not in law: unweighted, Y is far from p.
    tv = 0.5 * sum(abs(law.get(z, (0.0, 1.0))[0] * M - _seq_prob(P, z)) for z in seqs)
    if band < 1e9:
        assert 1e-3 < tv, tv


def test_band_zero_limit_is_plain_block_verification():
    """A band below every |S_1| exits at the first token: plain BV, w = 1."""
    V, L = 3, 3
    P, Q = _model(V, L, 7), _model(V, L, 8)
    law, _ = _banded_law(P, Q, 1e-12)
    for z, (pz, w) in law.items():
        # w = p/q of the first token; the law is p / w (first token from q).
        assert abs(pz * w - _seq_prob(P, z)) < 1e-12


# --------------------------------------------------------------------------
# GRPO groups: per-row weights vs the exact-sampling expectation
# --------------------------------------------------------------------------


def _score(P, y, idx):
    """d log p(y) / d logits (softmax parametrisation, one logit vector per prefix)."""
    s = np.zeros(len(idx) * len(P[()]))
    for t in range(len(y)):
        k = idx[y[:t]] * len(P[()])
        s[k : k + len(P[()])] -= P[y[:t]]
        s[k + y[t]] += 1.0
    return s


def _advantages(r, std_norm):
    """GRPO advantages as NeMo-RL computes them with use_leave_one_out_baseline:
    A_i = (R_i - mean_{j!=i} R_j) / (std_{j!=i} + 1e-6), the division only when
    the other rewards vary (std_norm = normalize_rewards)."""
    G = len(r)
    out = []
    for i in range(G):
        o = np.delete(r, i)
        a = r[i] - o.mean()
        if std_norm and o.max() > o.min():
            a /= o.std(ddof=1) + 1e-6
        out.append(a)
    return np.array(out)


def _group_expectations(P, law, R, G, std_norm):
    """(exact, per-row weights, group-product weights) expectations of
    sum_i A_i s(y_i) over one prompt's group of G rows."""
    seqs = sorted(law)
    idx = {pre: k for k, pre in enumerate(sorted({z[:t] for z in seqs for t in range(len(z))}))}
    S = {z: _score(P, z, idx) for z in seqs}
    p = {z: _seq_prob(P, z) for z in seqs}
    e_exact = e_row = e_grp = 0.0
    for ys in itertools.product(seqs, repeat=G):
        A = _advantages(np.array([R[z] for z in ys]), std_norm)
        g = sum(a * S[z] for a, z in zip(A, ys))
        e_exact = e_exact + math.prod(p[z] for z in ys) * g
        m = math.prod(law[z][0] for z in ys)
        w = [law[z][1] for z in ys]
        e_row = e_row + m * sum(wi * a * S[z] for wi, a, z in zip(w, A, ys))
        e_grp = e_grp + m * math.prod(w) * g
    return e_exact, e_row, e_grp


@pytest.mark.parametrize("std_norm", [False, True])
def test_group_advantages_by_enumeration(std_norm):
    V, L, G = 2, 3, 3
    P, Q = _model(V, L, 3), _model(V, L, 4)
    law, _ = _banded_law(P, Q, 0.6)
    rng = np.random.default_rng(0)
    R = {z: float(rng.integers(0, 2)) for z in law}
    e_exact, e_row, e_grp = _group_expectations(P, law, R, G, std_norm)
    scale = np.abs(e_exact).max()
    assert scale > 1e-3
    # The group's product of weights reweights the whole group: always exact.
    assert np.abs(e_grp - e_exact).max() < 1e-12 * max(1.0, scale)
    cos = e_row @ e_exact / np.linalg.norm(e_row) / np.linalg.norm(e_exact)
    if not std_norm:
        # RLOO: the baseline is independent of the row, so the per-row weight
        # is exact (both equal G * grad J).
        assert np.abs(e_row - e_exact).max() < 1e-12 * max(1.0, scale)
    else:
        # A_i = a(Y_-i) R_i + c(Y_-i): per-row weights give E_m[a] grad J
        # instead of E_p[a] grad J, the same direction with another scale.
        assert cos > 1 - 1e-12
        assert np.abs(e_row - e_exact).max() > 1e-4 * scale


# --------------------------------------------------------------------------
# Samplers pathwise identical with band
# --------------------------------------------------------------------------


@pytest.mark.parametrize("M", [1.0, 2.0])
@pytest.mark.parametrize("band", [0.05, 0.4, 3.0])
def test_eager_lazy_engine_q_identical_with_band(band, M):
    torch.manual_seed(0)
    V = 40
    n_exit = n_rej = 0
    for trial in range(60):
        g = 1 + trial % 9
        lp = torch.log_softmax(torch.randn(g + 1, V) * 2, -1)
        lq = torch.log_softmax(lp[:g] + torch.randn(g, V) * (0.1 + 0.2 * (trial % 3)), -1)
        draft = torch.multinomial(lq.exp(), 1).flatten()
        lr = lp[torch.arange(g), draft] - lq[torch.arange(g), draft]
        t_exit = band_exit(lr, band)
        outs, states = [], []
        for fn in ("eager", "lazy", "engine"):
            gen = torch.Generator().manual_seed(trial)
            if fn == "eager":
                tau, y = block_verify(lp, lq, draft, gen, null_m=M, band=band)
            elif fn == "lazy":
                tau, y = block_verify_lazy(lp, lq, draft, gen, null_m=M, band=band)
            else:
                tau, y, logp, _ = verify_engine_q(
                    lp, draft, lq[torch.arange(g), draft], lambda i: lq[i], gen, M, band
                )
            outs.append((tau, y))
            states.append(gen.get_state())
        assert outs[0] == outs[1] == outs[2], outs
        assert all(torch.equal(states[0], s) for s in states[1:])
        tau, y = outs[0]
        if y == NULL_TOKEN:
            assert M > 1 and tau == 0
            continue
        # Outputs keep the draft through the exit; the kept prefix gives S_T.
        assert tau >= (g if t_exit is None else t_exit)
        log_w, exited = banded_log_weight(lr[:tau], band)
        assert exited == (t_exit is not None)
        s = torch.cumsum(lr.double(), 0)
        assert log_w == float(s[(g if t_exit is None else t_exit) - 1])
        n_exit += exited
        n_rej += tau < g
    if band < 3.0:
        assert n_exit > 0 and n_rej > 0


def test_no_band_is_bitwise_the_plain_path():
    torch.manual_seed(1)
    V, g = 30, 6
    lp = torch.log_softmax(torch.randn(g + 1, V), -1)
    lq = torch.log_softmax(torch.randn(g, V), -1)
    draft = torch.multinomial(lq.exp(), 1).flatten()
    a = output_distribution_given_draft(lp, lq, draft)
    b = output_distribution_given_draft(lp, lq, draft, 1.0, None)
    assert all(i == j and torch.equal(x, y) for (i, x), (j, y) in zip(a, b))
    # A band no prefix leaves: whole draft accepted, tau = g always.
    c = output_distribution_given_draft(lp, lq, draft, band=1e9)
    assert float(c[-1][1].sum()) == 1.0


# --------------------------------------------------------------------------
# Config, loss weight, driver
# --------------------------------------------------------------------------


def _master(**ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        banded_is_band=0.5,
    )
    base.update(ts)
    return SimpleNamespace(thundersync=ThunderSyncConfig(**base))


def test_band_config_default_off_and_validation():
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_banded_is

    assert ThunderSyncConfig().banded_is_band is None
    validate_banded_is(_master())
    validate_banded_is(_master(banded_is_band=None, verify_source="scorer"))
    # Composes with sticky routing and momentum draft weights.
    validate_banded_is(_master(sticky_routing=True, draft_weights="momentum"))
    for ts, match in [
        ({"banded_is_band": 0.0}, "finite and > 0"),
        ({"banded_is_band": math.inf}, "finite and > 0"),
        ({"speculate": False}, "speculate"),
        ({"verify_source": "scorer"}, "verify_source"),
        ({"fused_verify_train": True}, "fused_verify_train"),
        # Pooled ⊥-BV (section 14d) verifies on an envelope grid that is not banded.
        ({"null_bv": True}, "null_bv must be false"),
    ]:
        with pytest.raises(ValueError, match=match):
            validate_banded_is(_master(**ts))


def test_chunk_advantages_carry_the_row_weight():
    from thundersync_rl.grpo_loop import build_chunk_data

    tok = SimpleNamespace(pad_token_id=0)
    cfg = SimpleNamespace(policy={"make_sequence_length_divisible_by": 4})

    def traj(ids, **extra):
        return SimpleNamespace(
            payload={
                "message_log": [
                    {"role": "user", "token_ids": torch.tensor(ids[:2])},
                    {
                        "role": "assistant",
                        "token_ids": torch.tensor(ids[2:]),
                        "generation_logprobs": torch.zeros(len(ids) - 2),
                    },
                ],
                "loss_multiplier": 1.0,
                **extra,
            }
        )

    rows = [traj([1, 2, 3, 4], is_weight=1.25), traj([1, 2, 7])]
    final = SimpleNamespace(trajectories=rows, group=None, advantages=[2.0, -1.0], padding=False)
    adv = build_chunk_data(final, tok, cfg, 1)["advantages"]
    assert torch.equal(adv[:, 0], torch.tensor([2.5, -1.0]))
    # Open-group bucket chunk: advantage 1 times the weight, scaled later.
    bucket = SimpleNamespace(trajectories=rows, group=(0, 0), advantages=None, padding=False)
    adv = build_chunk_data(bucket, tok, cfg, 1)["advantages"]
    assert torch.equal(adv[:, 0], torch.tensor([1.25, 1.0]))


def test_driver_plans_claims_and_stats(monkeypatch):
    from test_speculative_driver import EOD, _spec as make_spec

    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)
    calls = []

    def verify(rows, prompt_lens, keys, engine_logq, **kw):
        calls.append(kw)
        # Row 0: whole draft, never left the band; row 1: exit, then rejected.
        res = [
            {"accepted": 2, "next": 5, "logprobs": [-1.0] * 3, "draft_logq": [-1.0] * 2,
             "band_log_weight": 0.1, "band_exited": False},
            {"accepted": 1, "next": 9, "logprobs": [-1.0] * 2, "draft_logq": [-1.0],
             "band_log_weight": -0.7, "band_exited": True},
        ]
        return list(enumerate(res[: len(rows)]))

    s = make_spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        base_handlers={"install_draft_hidden_log": lambda: True},
        verify_source="engine",
        hidden_transport="driver",
        banded_is_band=0.5,
    )
    prompt = torch.tensor([1, 2])
    lq = [-1.0, -1.0]
    hid = torch.ones(2, 4)
    drafts = [
        [(prompt, -1, [6, EOD], True, "k0", [(0, 0, 2)], lq, hid)],
        [(prompt, -1, [7, 8], False, "k1", [(0, 0, 2)], lq, hid)],
    ]
    s.verify(drafts)
    assert calls[0]["band"] == 0.5
    plans = s.plans[tuple(prompt.tolist())]
    assert [p.log_weight for p in plans] == [0.1, -0.7]
    st = s.stats
    w = [math.exp(0.1), math.exp(-0.7)]
    assert st["spec/band_exit_frac"] == 0.5
    assert st["spec/band_ess"] == pytest.approx(sum(w) ** 2 / (2 * sum(x * x for x in w)))
    # Rollouts: complete plan, then continuation of [7, 9]; claims by prefix.
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    datum = BatchedDataDict({"input_ids": prompt.view(1, -1), "input_lengths": torch.tensor([2])})
    _, o1 = asyncio.run(s._one(0, datum))
    _, o2 = asyncio.run(s._one(1, datum))
    assert s.claim_log_weight(prompt, o2["output_ids"][0]) == -0.7
    assert s.claim_log_weight(prompt, o1["output_ids"][0]) == 0.1
    assert s.claim_log_weight(prompt, o1["output_ids"][0]) == 0.0  # claimed once
    # Continuation rounds (sticky) are verified without the band.
    s._verify_block([(0, drafts[0][0])], [torch.tensor([1, 2, 6, EOD])])
    assert "band" not in calls[-1]


def test_worker_reports_band_weights(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from test_engine_q import _engine_drafts, _engine_worker

    w, emb, heads = _engine_worker(monkeypatch)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads, n=12, seed=3)
    kw = dict(rows=rows, prompt_lens=plens, segments=segs, engine_logq=lqs, hidden=hids,
              vocab_limit=1 << 30, seed=7, keys=keys, batch_tokens=64)
    plain = dict(w.verify_drafts_block_engine(**kw))
    banded = dict(w.verify_drafts_block_engine(**kw, band=1e9))
    for i in range(12):
        assert "band_log_weight" not in plain[i]
        g = rows[i].numel() - plens[i]
        r = banded[i]
        # A band no prefix leaves: whole drafts, w = W_g.
        assert r["accepted"] == g and not r["band_exited"]
        lr = torch.tensor(r["logprobs"][:g]) - torch.tensor(lqs[i])
        assert r["band_log_weight"] == pytest.approx(float(lr.double().sum()), abs=1e-12)
