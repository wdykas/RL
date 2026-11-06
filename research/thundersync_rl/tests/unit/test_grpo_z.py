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
"""GRPO-Z (thundersync_rl.grpo_z; FIRSTPRINCIPLES_D Lemma Z): exact enumeration on toy bandits, stock equivalence when
every draft is accepted, the negatives that must break it, and the config validator.

Toy: one-token responses over V outcomes, policy p = softmax(theta), drafting law q = softmax(theta + delta), G slots
per group. Each slot draws X ~ q and the exact row Y by the optimal single-draft coupling (Y = X w.p. min(1, p/q), else
Y ~ (p - q)+ / TV), so Y ~ p and (X_i, Y_i) are independent across slots. Expectations are exact sums over all
(V x V)^G joint outcomes, accumulated in float64 over float32 advantages (the production dtype): equalities hold to
float32 rounding (~1e-7 relative, tolerance 2e-6), negatives miss by > 1e-4.
"""

import itertools

import pytest
import torch

from thundersync_rl import grpo_z


def _coupling(p, q):
    """Joint law P(X = x, Y = y) of the single-draft coupling, [V, V]."""
    V = len(p)
    acc = torch.minimum(torch.ones(V, dtype=torch.float64), p / q)
    res = torch.clamp(p - q, min=0.0)
    tv = float(res.sum())
    J = torch.zeros(V, V, dtype=torch.float64)
    for x in range(V):
        J[x, x] += q[x] * acc[x]
        if tv > 0:
            J[x] += q[x] * (1 - acc[x]) * res / tv
    return J


def _expected_grad(theta, delta, R, G, adv_fn):
    """E[sum_i A_i grad_theta log p(Y_i)] by exact enumeration; adv_fn(final [G], draft [G]) -> A [G]."""
    p = torch.softmax(theta, 0)
    q = torch.softmax(theta + delta, 0)
    J = _coupling(p, q)
    assert torch.allclose(J.sum(0), p, atol=1e-15)  # Y ~ p exactly
    V = len(p)
    pairs = [(x, y) for x in range(V) for y in range(V) if J[x, y] > 0]
    g = torch.zeros(V, dtype=torch.float64)
    total = 0.0
    for combo in itertools.product(pairs, repeat=G):
        w = 1.0
        for x, y in combo:
            w *= float(J[x, y])
        # float32 rewards, as the data plane carries them (stock's LOO helper is float32-only); the expectation is
        # accumulated in float64, so the equalities below hold to ~1e-7 relative
        xs = torch.tensor([R[x] for x, _ in combo], dtype=torch.float32)
        ys = torch.tensor([R[y] for _, y in combo], dtype=torch.float32)
        A = adv_fn(ys, xs)
        for i, (_, y) in enumerate(combo):
            score = -p.clone()
            score[y] += 1.0
            g += w * float(A[i]) * score
        total += w
    assert abs(total - 1.0) < 1e-12
    return g, p


def _grad_J(p, R):
    V = len(p)
    gj = torch.zeros(V, dtype=torch.float64)
    for y in range(V):
        s = -p.clone()
        s[y] += 1.0
        gj += p[y] * R[y] * s
    return gj


TOYS = [
    # theta, delta (draft drift), rewards
    (torch.tensor([0.3, -0.2, 0.1], dtype=torch.float64), torch.tensor([0.25, -0.1, 0.0], dtype=torch.float64),
     [1.0, 0.0, 0.3]),
    (torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64), torch.tensor([-0.4, 0.3, 0.2], dtype=torch.float64),
     [0.0, 1.0, 1.0]),
    (torch.tensor([0.0, 0.5], dtype=torch.float64), torch.tensor([0.7, -0.7], dtype=torch.float64), [1.0, 0.0]),
]


def _gid(G):
    return torch.zeros(G, dtype=torch.long)


@pytest.mark.parametrize("toy", range(len(TOYS)))
def test_loo_c_is_exactly_the_on_policy_gradient(toy):
    theta, delta, R = TOYS[toy]
    G, s = 3, 0.7
    gz, p = _expected_grad(theta, delta, R, G,
                           lambda y, x: grpo_z.advantages(_gid(G), y, x, "loo_c", step_scale=s))
    gj = _grad_J(p, R)
    # sum over G rows: each row contributes grad J / s
    assert torch.allclose(gz, G * gj / s, atol=2e-6 * float((G * gj / s).abs().max()), rtol=0), (gz, G * gj / s)
    # and equals on-policy LOO-c (baseline from Y_-i) exactly
    gon, _ = _expected_grad(theta, delta, R, G,
                            lambda y, x: grpo_z.advantages(_gid(G), y, y, "loo_c", step_scale=s))
    assert torch.allclose(gz, gon, atol=2e-6 * float(gon.abs().max()), rtol=0)


@pytest.mark.parametrize("toy", range(len(TOYS)))
def test_loo_std_is_the_exact_direction_with_a_prompt_weight(toy):
    theta, delta, R = TOYS[toy]
    G = 3
    gz, p = _expected_grad(theta, delta, R, G, lambda y, x: grpo_z.advantages(_gid(G), y, x, "loo_std"))
    gj = _grad_J(p, R)
    # E[A_i h_i] = E[w(X_-i)] grad J with w = 1 / (std + eps) (or 1 for a trivial sibling set)
    q = torch.softmax(theta + delta, 0)
    Ew = 0.0
    for xs in itertools.product(range(len(p)), repeat=G - 1):
        pr = 1.0
        for x in xs:
            pr *= float(q[x])
        r = torch.tensor([R[x] for x in xs], dtype=torch.float64)
        if bool((r == r[0]).all()):
            w = 1.0
        else:
            w = 1.0 / (float(r.std(unbiased=True)) + grpo_z.EPS)
        Ew += pr * w
    ref = G * Ew * gj
    assert torch.allclose(gz, ref, atol=2e-6 * float(ref.abs().max()), rtol=0), (gz, ref)


@pytest.mark.parametrize("toy", range(len(TOYS)))
def test_negative_own_draft_in_baseline_is_biased(toy):
    """The row's own draft reward in its baseline (b_i over all X, incl. X_i = Y_i on acceptance) breaks Lemma Z."""
    theta, delta, R = TOYS[toy]
    G, s = 3, 0.7

    def own_in_baseline(y, x):
        return (y - x.mean()) / s

    gneg, p = _expected_grad(theta, delta, R, G, own_in_baseline)
    gj = _grad_J(p, R)
    # best scalar fit c * gJ still leaves a residual: not the gradient of J (direction or scale is off)
    c = float(gneg @ gj / (gj @ gj))
    resid = float((gneg - c * gj).norm() / gneg.norm())
    scale_err = abs(c - G / s) / (G / s)
    assert resid > 1e-4 or scale_err > 1e-3, (resid, scale_err)


def test_negative_batch_scale_including_the_row_is_biased():
    """loo_c with a scale computed from the CURRENT rows' rewards (incl. Y_i) is not the on-policy gradient."""
    theta, delta, R = TOYS[0]
    G = 3

    def cur_scale(y, x):
        s = float(y.std(unbiased=True)) + 0.5
        return grpo_z.advantages(_gid(G), y, x, "loo_c", step_scale=s)

    gneg, p = _expected_grad(theta, delta, R, G, cur_scale)
    gj = _grad_J(p, R)
    c = float(gneg @ gj / (gj @ gj))
    resid = float((gneg - c * gj).norm() / gneg.norm())
    assert resid > 1e-4


def test_all_accepted_loo_std_is_bitwise_stock():
    from nemo_rl.algorithms.advantage_estimator import AdvEstimatorConfig, GRPOAdvantageEstimator

    gen = torch.Generator().manual_seed(0)
    G, P = 8, 64
    gid = torch.arange(P).repeat_interleave(G)
    for kind in ("binary", "real", "trivial_mix"):
        if kind == "binary":
            r = (torch.rand(P * G, generator=gen) < 0.4).float()
        elif kind == "real":
            r = torch.rand(P * G, generator=gen)
        else:
            r = (torch.rand(P * G, generator=gen) < 0.5).float()
            r[: 3 * G] = 1.0  # groups with no variation
            r[3 * G: 4 * G] = 0.0
            r[4 * G] = 1.0  # one row differs from 7 equal siblings: LOO-trivial for it, not for the others
        est = GRPOAdvantageEstimator(AdvEstimatorConfig(), loss_config=None)
        mask = torch.ones(P * G, 5)
        stock = est.compute_advantage(gid.reshape(-1, 1), r, mask)[:, 0]
        z = grpo_z.advantages(gid, r, r.clone(), "loo_std")
        assert torch.equal(stock, z), (kind, (stock - z).abs().max())


def test_rejected_sibling_changes_only_the_others():
    """A row's own final reward moves only its own advantage; a sibling's FINAL reward moves nothing (draft-based)."""
    gid = torch.zeros(4, dtype=torch.long)
    x = torch.tensor([1.0, 0.0, 0.0, 1.0])
    y = x.clone()
    a0 = grpo_z.advantages(gid, y, x, "loo_std")
    y2 = y.clone()
    y2[1] = 1.0  # row 1 was rejected and its continuation scored 1
    a1 = grpo_z.advantages(gid, y2, x, "loo_std")
    assert torch.equal(a0[[0, 2, 3]], a1[[0, 2, 3]])
    assert not torch.equal(a0[1], a1[1])


def test_next_step_scale_uses_final_rewards_only():
    gid = torch.arange(2).repeat_interleave(4)
    r = torch.tensor([1.0, 0, 0, 1, 0, 0, 0, 0])
    s = grpo_z.next_step_scale(gid, r)
    assert abs(s - float(torch.tensor([1.0, 0, 0, 1], dtype=torch.float64).std(unbiased=True)) / 2) < 1e-12
    assert grpo_z.next_step_scale(gid, torch.zeros(8)) == 1e-3


def _cfg(**over):
    c = {
        "thundersync": {"mode": "nolanes", "grpo_z": "loo_c"},
        "grpo": {"num_generations_per_prompt": 8, "adv_estimator": {"name": "grpo", "use_leave_one_out_baseline": True,
                                                                    "normalize_rewards": True}},
    }
    for k, v in over.items():
        cur = c
        parts = k.split(".")
        for p_ in parts[:-1]:
            cur = cur.setdefault(p_, {})
        cur[parts[-1]] = v
    return c


def test_validator():
    assert grpo_z.config_problems(_cfg()) == []
    assert grpo_z.config_problems(_cfg(**{"thundersync.grpo_z": "off"})) == []
    assert grpo_z.config_problems(_cfg(**{"thundersync.grpo_z": "loo_std"})) == []
    assert grpo_z.config_problems(_cfg(**{"thundersync.grpo_z": "bogus"}))
    assert grpo_z.config_problems(_cfg(**{"grpo.adv_estimator.use_leave_one_out_baseline": False}))
    assert grpo_z.config_problems(_cfg(**{"thundersync.grpo_z": "loo_std",
                                          "grpo.adv_estimator.normalize_rewards": False}))
    assert grpo_z.config_problems(_cfg(**{"grpo.num_generations_per_prompt": 1}))
    assert grpo_z.config_problems(_cfg(**{"grpo.use_dynamic_sampling": True}))
    assert grpo_z.config_problems(_cfg(**{"thundersync.mode": "colo"}))
    assert grpo_z.config_problems(_cfg(**{"grpo.adv_estimator.name": "gdpo"}))
    with pytest.raises(ValueError):
        grpo_z.advantages(torch.zeros(2), torch.zeros(2), torch.zeros(2), "loo_c", step_scale=None)


def test_group_row_advantages_matches_the_vector_form_bitwise():
    """N = G, sources = slots: the per-row form is ``advantages`` on the group (both modes), with rejected rows."""
    gen = torch.Generator().manual_seed(1)
    G = 8
    for trial in range(20):
        x = (torch.rand(G, generator=gen) < 0.5).float()
        y = x.clone()
        flip = torch.rand(G, generator=gen) < 0.3  # rejected rows whose continuation scored differently
        y[flip] = 1.0 - y[flip]
        for mode, sc in (("loo_std", None), ("loo_c", 0.37)):
            vec = grpo_z.advantages(_gid(G), y, x, mode, step_scale=sc)
            rows = grpo_z.group_row_advantages(y.tolist(), list(range(G)), x.tolist(), mode, step_scale=sc)
            assert torch.equal(vec, torch.tensor(rows, dtype=torch.float32)), (trial, mode)


def test_group_row_advantages_pool_shapes():
    """N > G drafts (spare drafts) and a fresh slot: every row's baseline is the mean of all OTHER drafts."""
    x = [1.0, 0.0, 0.0, 1.0, 1.0]  # 5 drafts for G = 4 slots
    y = [1.0, 0.0, 1.0, 0.0]
    src = [0, 2, 3, None]  # draft 1 was ⊥ (skipped), draft 4 spare, slot 3 fresh
    a = grpo_z.group_row_advantages(y, src, x, "loo_c", step_scale=1.0)
    exp = [1.0 - (0 + 0 + 1 + 1) / 4, 0.0 - (1 + 0 + 1 + 1) / 4, 1.0 - (1 + 0 + 0 + 1) / 4, 0.0 - sum(x) / 5]
    assert all(abs(u - v) < 1e-6 for u, v in zip(a, exp)), (a, exp)
    with pytest.raises(ValueError):
        grpo_z.group_row_advantages([1.0], [7], x, "loo_c", step_scale=1.0)
