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
"""IS-RLOO at lag K (thundersync_rl/isspr.py, ``weight: row`` / ``auto``, ``length_normalizer: constant``).

Toy bandit by exact enumeration, with mixed-policy rows at lag K = 1..4:
- a tabular softmax policy over {0, 1, EOS}, rows of 1..T tokens (EOS ends a row; variable lengths, so the length
  normalizer matters), groups of G rows, reward correlated with length;
- versions theta_{n-a} = theta_n + a D (drift D); token t of row j is sampled from theta_{n - age(j, t)}, ages falling
  along the row from up to K to 0 (the row spans several wakes; tokens of age 0 were decoded after the last wake, from
  theta_n's recomputed state). Row j's log ratio is s_j = sum_t (log p_t - log q_t) over all its tokens (age-0 terms
  are exactly 0), as IsRows computes it.
Estimands, each against the exact on-policy value:
- IS-RLOO (row weights, RLOO advantage, constant normalizer L_c): E_q[sum_j w_j (R_j - mean_{-j} R) score_j / L_c] =
  G grad J / L_c exactly, at every K;
- the group weight is exact for any group statistic at every K (the K = 1 arm of ``auto``);
- row weights with the per-row token mean 1 / l_j, or with GRPO's LOO std, are NOT the on-policy value (why the
  validator pins RLOO + constant);
- the ``is_weight`` column (train_coefficients: w l / L_c) under the per-row token-mean loss is the constant-normalizer
  row term;
- the clip's bias is exactly the logged excess; equal versions (lr 0) give s = 0 exactly; the stale-state negative
  (post-wake tokens not from theta_n's state) is biased.
Not covered here: the timing coupling R1 (a row's pause point depending on its siblings' content). Rows here have
content-independent ages, so siblings are independent of Y_j under q; test_isspr's coupled enumeration shows RIS + RLOO
keeps a cross term there (0.038 vs max |grad| 1.18) which the group weight does not.
"""

from __future__ import annotations

import functools
import itertools
import math

import numpy as np
import pytest

from thundersync_rl import isspr

EOS, VOC, T, G = 2, 3, 3, 3
L_C = T  # the constant normalizer (max_new_tokens of the toy)
CTXS = [c for n in range(T) for c in itertools.product((0, 1), repeat=n)]  # EOS ends a row: no context holds it
CIDX = {c: i for i, c in enumerate(CTXS)}
ROWS = [r for n in range(1, T + 1) for r in itertools.product(range(VOC), repeat=n)
        if all(x != EOS for x in r[:-1]) and (n == T or r[-1] == EOS)]


def _table(seed: int, scale: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=(len(CTXS), VOC))


P_TABLE = _table(1, 1.0)
DRIFT = _table(2, 0.35)
STALE_TABLE = _table(3, 1.0)


def _version(age: int) -> np.ndarray:
    return P_TABLE + age * DRIFT


def _logp(theta: np.ndarray, ctx: tuple) -> np.ndarray:
    z = theta[CIDX[ctx]]
    return z - (z.max() + math.log(np.exp(z - z.max()).sum()))


def _lps(theta_of_t, row: tuple) -> list[float]:
    return [float(_logp(theta_of_t(t), row[:t])[row[t]]) for t in range(len(row))]


def _age(k: int, j: int, t: int) -> int:
    """Content-independent ages: row j's first token is K (even j) or K - 1 (odd j) versions old, younger along the
    row, and a full-length row's last token is post-wake (age 0). K = 4: ages 4, 2, 0 and 3, 1, 0."""
    return max(0, k - j % 2 - (t * k) // (T - 1))


@functools.lru_cache(maxsize=None)
def _q_lps(k: int, j: int, row: tuple) -> tuple[float, ...]:
    return tuple(_lps(lambda t: _version(_age(k, j, t)), row))


@functools.lru_cache(maxsize=None)
def _p_lps(row: tuple) -> tuple[float, ...]:
    return tuple(_lps(lambda t: P_TABLE, row))


def _score(row: tuple) -> np.ndarray:
    g = np.zeros_like(P_TABLE)
    for t in range(len(row)):
        ctx = row[:t]
        g[CIDX[ctx]] -= np.exp(_logp(P_TABLE, ctx))
        g[CIDX[ctx], row[t]] += 1.0
    return g.ravel()


def _reward(row: tuple) -> float:
    return float(len(row)) + 0.7 * float(row[0] == 1)


SCORE = {r: _score(r) for r in ROWS}
REWARD = {r: _reward(r) for r in ROWS}
P_ROW = {r: math.exp(math.fsum(_p_lps(r))) for r in ROWS}


def _grad_j() -> np.ndarray:
    """grad_theta E_p[R] of the toy, exactly."""
    return sum(P_ROW[r] * REWARD[r] * SCORE[r] for r in ROWS)


def _rloo(r: np.ndarray, i: int) -> float:
    return float(r[i] - (r.sum() - r[i]) / (len(r) - 1))


def _loo_std(r: np.ndarray, i: int) -> float:
    """Stock NeMo-RL: (R_i - LOO mean) / (LOO std with Bessel + 1e-6); unnormalized when the siblings are all equal."""
    others = np.delete(r, i)
    sd = float(others.std(ddof=1))
    return _rloo(r, i) / (sd + 1e-6) if sd > 0 else _rloo(r, i)


def _const(row: tuple) -> float:
    return 1.0 / L_C


def _rowmean(row: tuple) -> float:
    return 1.0 / len(row)


def _expect(k: int | None, weight: str, adv, norm, clip: float = math.inf, stale: bool = False) -> np.ndarray:
    """E over groups of sum_j coef_j adv_j norm(Y_j) score_j. k None: on-policy (every row from p, coef 1). Else row j
    from the lag-k mixed law, coef the clipped row (``row``) or group (``group``) weight. stale: tokens of age 0 come
    from STALE_TABLE (not theta_n's state), the weight still counts only the older tokens (the negative)."""
    def law(j, row):
        if k is None:
            return P_ROW[row]
        if not stale:
            return math.exp(math.fsum(_q_lps(k, j, row)))
        th = [STALE_TABLE if stale and _age(k, j, t) == 0 else _version(_age(k, j, t)) for t in range(len(row))]
        return math.exp(math.fsum(_lps(lambda t: th[t], row)))

    total = np.zeros(P_TABLE.size)
    for rows in itertools.product(ROWS, repeat=G):
        pr = math.prod(law(j, rows[j]) for j in range(G))
        if pr == 0.0:
            continue
        r = np.array([REWARD[x] for x in rows])
        if k is None:
            coef = [1.0] * G
        else:
            s = []
            for j, x in enumerate(rows):
                pre = [t for t in range(len(x)) if _age(k, j, t) > 0] if stale else range(len(x))
                p, q = _p_lps(x), _q_lps(k, j, x)
                s.append(isspr.row_log_ratio([p[t] for t in pre], [q[t] for t in pre]))
            if weight == "row":
                coef = isspr.clipped_weights(s, clip)
            else:
                coef = isspr.clipped_weights([math.fsum(s)], clip) * G
        total += pr * sum(coef[j] * adv(r, j) * norm(rows[j]) * SCORE[rows[j]] for j in range(G))
    return total


LAGS = (1, 2, 3, 4)


def test_is_rloo_unbiased_for_the_on_policy_gradient():
    exact = _grad_j()
    assert np.abs(exact).max() > 0.05, "the toy gradient must be non-trivial"
    for k in LAGS:
        got = _expect(k, "row", _rloo, _const)
        np.testing.assert_allclose(got, G * exact / L_C, rtol=0, atol=1e-12)
        # ... and it is the on-policy RLOO estimator's expectation
        np.testing.assert_allclose(_expect(None, "row", _rloo, _const), G * exact / L_C, rtol=0, atol=1e-12)


def test_group_weight_exact_for_any_group_statistic():
    for k in LAGS:
        for adv, norm in ((_rloo, _const), (_rloo, _rowmean), (_loo_std, _rowmean)):
            np.testing.assert_allclose(_expect(k, "group", adv, norm), _expect(None, "group", adv, norm), rtol=0,
                                       atol=1e-12)


def test_row_weights_biased_without_the_pinned_objective():
    for k in LAGS:
        # per-row token mean: E_p[score / l] != 0, so the baseline term keeps (E_q - E_p)[mean_-j R] E_p[score / l]
        on = _expect(None, "row", _rloo, _rowmean)
        bias = float(np.abs(_expect(k, "row", _rloo, _rowmean) - on).max())
        assert bias > 1e-4, bias
        # GRPO's LOO std: 1 / std_-j is a sibling statistic left at q's law (prompt-scale bias)
        on_std = _expect(None, "row", _loo_std, _const)
        bias_std = float(np.abs(_expect(k, "row", _loo_std, _const) - on_std).max())
        assert bias_std > 1e-4, bias_std
        print(f"[is-rloo toy] K={k} row-weight bias: per-row 1/l {bias:.3g}, LOO std {bias_std:.3g} "
              f"(max |G grad J / L_c| {float(np.abs(G * _grad_j() / L_C).max()):.3g})")


def test_clip_bias_is_exactly_the_logged_excess():
    for k in LAGS:
        c = 1.01
        clipped = _expect(k, "row", _rloo, _const, clip=c)
        raw = _expect(k, "row", _rloo, _const)

        total = np.zeros(P_TABLE.size)
        for rows in itertools.product(ROWS, repeat=G):
            q = [math.exp(math.fsum(_q_lps(k, j, x))) for j, x in enumerate(rows)]
            s = [isspr.row_log_ratio(_p_lps(x), _q_lps(k, j, x)) for j, x in enumerate(rows)]
            r = np.array([REWARD[x] for x in rows])
            total += math.prod(q) * sum(max(0.0, math.exp(s[j]) - c) * _rloo(r, j) * _const(rows[j]) * SCORE[rows[j]]
                                        for j in range(G))
        assert np.abs(raw - clipped).max() > 1e-4, "C must bind in this toy"
        np.testing.assert_allclose(clipped, raw - total, rtol=0, atol=1e-12)


def test_negative_stale_state_is_biased():
    for k in LAGS:
        on = _expect(None, "row", _rloo, _const)
        got = _expect(k, "row", _rloo, _const, stale=True)
        assert np.abs(got - on).max() > 1e-3, np.abs(got - on).max()


def test_lr0_gives_s_exactly_zero():
    for row in ROWS:
        lp = _p_lps(row)
        assert isspr.row_log_ratio(lp, list(lp)) == 0.0


def test_row_weight_second_moment_below_group():
    """E_q[W_g^2] = prod_j E_q[w_j^2] >= each E_q[w_j^2] (independent rows): the group's variance factor dominates."""
    for k in LAGS:
        m2 = []
        for j in range(G):
            m2.append(math.fsum(math.exp(math.fsum(_q_lps(k, j, x))) * math.exp(2 * isspr.row_log_ratio(
                _p_lps(x), _q_lps(k, j, x))) for x in ROWS))
        assert all(x >= 1.0 - 1e-12 for x in m2)
        assert math.prod(m2) >= max(m2) - 1e-12
        print(f"[is-rloo toy] K={k} row 1/E[w^2] {', '.join(f'{1 / x:.3f}' for x in m2)}; group {1 / math.prod(m2):.3f}")


# -- the is_weight column ----------------------------------------------------------------------------------------------


def _cfg(**kw):
    return isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", **kw},
                                          "policy": {"generation": {"max_new_tokens": L_C}}})


def test_train_coefficients_make_the_row_mean_loss_constant_normalized():
    cfg = _cfg(weight="row", length_normalizer="constant")
    w, lengths = [1.0, 0.5, 2.0], [1, 3, 2]
    coef = isspr.train_coefficients(w, lengths, cfg)
    for c, wi, n in zip(coef, w, lengths):
        assert math.isclose(c / n, wi / L_C, rel_tol=1e-15)  # per-row mean loss x coefficient = w / L_c per token
    assert isspr.train_coefficients(w, lengths, _cfg()) == w  # length_normalizer row: the weight itself
    with pytest.raises(RuntimeError):
        isspr.train_coefficients([1.0], [L_C + 1], cfg)
    with pytest.raises(RuntimeError):
        isspr.train_coefficients([1.0, 1.0], [1], cfg)


# -- config mode -------------------------------------------------------------------------------------------------------


def _full(weight="auto", norm="constant", lag=2, std=False, loo=True, name="grpo", mnt=8192):
    return {"thundersync": {"mode": "isspr", "weight": weight, "length_normalizer": norm},
            "async_rl": {"sampler": {"max_lookahead_versions": lag}},
            "grpo": {"adv_estimator": {"name": name, "normalize_rewards": std, "use_leave_one_out_baseline": loo}},
            "policy": {"generation": {"max_new_tokens": mnt}}}


def test_auto_resolves_group_at_lag_1_and_row_from_lag_2():
    for lag, unit in ((0, "group"), (1, "group"), (2, "row"), (3, "row"), (4, "row")):
        cfg = isspr.IsSprConfig.from_config(_full(lag=lag))
        assert cfg.unit == unit and cfg.lag == lag and cfg.max_new_tokens == 8192
    assert isspr.IsSprConfig.from_config(_full(weight="group", lag=4)).unit == "group"
    with pytest.raises(ValueError):  # resolved fields are not keys
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "lag": 2}})
    with pytest.raises(ValueError):
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "length_normalizer": "token"}})


def test_estimator_validator():
    assert isspr.estimator_problems(_full()) == []
    assert isspr.estimator_problems(_full(weight="row", lag=1)) == []
    assert isspr.estimator_problems(_full(weight="group", norm="row", std=True)) == []  # stock group arm
    assert len(isspr.estimator_problems(_full(std=True))) == 1
    assert len(isspr.estimator_problems(_full(loo=False))) == 1
    assert len(isspr.estimator_problems(_full(norm="row"))) == 1
    assert len(isspr.estimator_problems(_full(name="gdpo"))) == 1
    assert len(isspr.estimator_problems(_full(weight="row", std=True, norm="row"))) == 2
    assert len(isspr.estimator_problems(_full(mnt=0))) == 1
    # fail closed: the estimator reads only grpo.adv_estimator, a missing normalize_rewards is its default True, the
    # flat grpo keys do not count (REVIEW batch 41 V1)
    flat = _full()
    flat["grpo"] = {"normalize_rewards": False, "use_leave_one_out_baseline": True}
    assert len(isspr.estimator_problems(flat)) == 1


def test_stage_setup_requires_the_rloo_estimator():
    from types import SimpleNamespace

    def stage(**est):
        return SimpleNamespace(algo=SimpleNamespace(adv_estimator=SimpleNamespace(name="grpo", **est)))

    auto = isspr.IsSprConfig.from_config(_full(lag=2))
    isspr.require_rloo(stage(normalize_rewards=False, use_leave_one_out_baseline=True), auto)
    with pytest.raises(RuntimeError):
        isspr.require_rloo(stage(normalize_rewards=True, use_leave_one_out_baseline=True), auto)
    with pytest.raises(RuntimeError):
        isspr.require_rloo(stage(normalize_rewards=False, use_leave_one_out_baseline=False), auto)
    group = isspr.IsSprConfig.from_config(_full(weight="group", norm="row"))
    isspr.require_rloo(stage(normalize_rewards=True, use_leave_one_out_baseline=True), group)  # group: any estimator
