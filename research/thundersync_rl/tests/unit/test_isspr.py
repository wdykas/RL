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
"""IS-SPR (thundersync_rl/isspr.py).

- Unbiasedness by exact enumeration: a toy tabular policy, groups of G rows, each row sampled from q before its wake
  point and from p after it. Each test names its estimand:
  - group weights W_g: E_q[W_g ghat] = E_p[ghat], the on-policy GRPO gradient (LOO mean + std), unclipped; the clip's
    bias is exactly -E_q[(e^s - C)^+ ghat];
  - row weights (RIS) with RLOO advantages (LOO mean, no std), constant normalizer: = the on-policy gradient exactly;
  - row weights with the stock LOO-std advantage: = Theorem R's estimand sum_i E_{R_-i ~ q} E_{Y_i ~ p}[f_i] exactly
    (FIRSTPRINCIPLES_C §3.2), and NOT the on-policy LOO-std gradient (an O(dp) prompt-scale difference).
- Negative: stale state (post-wake tokens not from p, weight over pre-wake tokens only) is biased under both.
- The weight log (eps_hat, ESS, var).
- The loss: is_weight scales each row's gradient by W; the sequence-level normalizer gives a row the same gradient
  whatever the other rows' token counts (token-level does not).
- The row loop against exact SPR's toy engine: generation log-probs are theta_target's for every token, s is
  sum(p - q) over the tokens sampled before the certifying wake, finished rows are rescored (D1), lag 2, the stale-KV
  negative keeps q.
"""

from __future__ import annotations

import asyncio
import itertools
import math

import numpy as np
import pytest
import torch

from thundersync_rl import isspr

# -- toy enumeration -------------------------------------------------------------------------------------------------

T, VOC, G = 3, 2, 3
CTXS = [c for n in range(T) for c in itertools.product(range(VOC), repeat=n)]
CIDX = {c: i for i, c in enumerate(CTXS)}


def _table(seed: int, scale: float) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, scale, size=(len(CTXS), VOC))


def _logp(theta: np.ndarray, ctx: tuple) -> np.ndarray:
    z = theta[CIDX[ctx]]
    return z - (z.max() + math.log(np.exp(z - z.max()).sum()))


def _row_lps(theta: np.ndarray, row: tuple) -> list[float]:
    return [float(_logp(theta, row[:t])[row[t]]) for t in range(T)]


def _score(theta: np.ndarray, row: tuple) -> np.ndarray:
    """grad_theta log p_theta(row) of the tabular softmax policy."""
    g = np.zeros_like(theta)
    for t in range(T):
        ctx = row[:t]
        g[CIDX[ctx]] -= np.exp(_logp(theta, ctx))
        g[CIDX[ctx], row[t]] += 1.0
    return g.ravel()


def _reward(row: tuple) -> float:
    return float(sum(row)) + 1.5 * float(row[0] == row[-1])


def _grpo_ghat(theta: np.ndarray, rows: tuple) -> np.ndarray:
    """GRPO gradient estimate of a group: leave-one-out baseline, std-normalized (the trained configuration)."""
    r = np.array([_reward(x) for x in rows])
    loo = (r.sum() - r) / (len(r) - 1)
    a = (r - loo) / (r.std() + 1e-6)
    return sum(a[i] * _score(theta, rows[i]) for i in range(len(rows)))


def _row_prob(row: tuple, wake: int, before: np.ndarray, after: np.ndarray) -> float:
    """Probability of a row whose tokens before ``wake`` come from ``before``, the rest from ``after``."""
    return math.exp(sum(float(_logp(before if t < wake else after, row[:t])[row[t]]) for t in range(T)))


ROWS = list(itertools.product(range(VOC), repeat=T))
WAKES = (1, 2, 3)  # per-row wake points: content-independent (the stand-down is set by the previous batch)


def _expect(sampler, weight) -> np.ndarray:
    """Sum over all groups of prob(group) * weight(group) * ghat(group); ``sampler(i, row)`` -> row prob."""
    p = _table(1, 1.0)
    total = np.zeros(p.size)
    for rows in itertools.product(ROWS, repeat=G):
        pr = math.prod(sampler(i, rows[i]) for i in range(G))
        total += pr * weight(rows) * _grpo_ghat(p, rows)
    return total


def _s_rows(rows, p, q) -> list[float]:
    return [isspr.row_log_ratio(_row_lps(p, x)[:w], _row_lps(q, x)[:w]) for x, w in zip(rows, WAKES)]


P_TABLE, Q_TABLE, STALE_TABLE = _table(1, 1.0), _table(1, 1.0) + _table(2, 0.6), _table(3, 1.0)


def _on_policy() -> np.ndarray:
    return _expect(lambda i, x: _row_prob(x, 0, P_TABLE, P_TABLE), lambda rows: 1.0)


def test_group_weight_unbiased_for_grpo_by_enumeration():
    exact = _on_policy()
    big = 1e300
    got = _expect(lambda i, x: _row_prob(x, WAKES[i], Q_TABLE, P_TABLE),
                  lambda rows: isspr.clipped_weights([math.fsum(_s_rows(rows, P_TABLE, Q_TABLE))], big)[0])
    assert np.abs(exact).max() > 0.05, "the toy gradient must be non-trivial"
    np.testing.assert_allclose(got, exact, rtol=0, atol=1e-12)


def test_clip_bias_is_exactly_the_logged_excess():
    exact = _on_policy()
    c = 1.3
    sampler = lambda i, x: _row_prob(x, WAKES[i], Q_TABLE, P_TABLE)  # noqa: E731
    clipped = _expect(sampler, lambda rows: isspr.clipped_weights([math.fsum(_s_rows(rows, P_TABLE, Q_TABLE))], c)[0])
    excess = _expect(sampler, lambda rows: max(0.0, math.exp(math.fsum(_s_rows(rows, P_TABLE, Q_TABLE))) - c))
    assert np.abs(exact - clipped).max() > 1e-3, "C must bind in this toy"
    np.testing.assert_allclose(clipped, exact - excess, rtol=0, atol=1e-12)


def _adv_rloo(r: np.ndarray, i: int) -> float:
    return float(r[i] - (r.sum() - r[i]) / (len(r) - 1))


def _adv_loo_std(r: np.ndarray, i: int) -> float:
    return _adv_rloo(r, i) / (float(r.std()) + 1e-6)


def _row_terms(adv, row_law, weighted: bool) -> np.ndarray:
    """sum over groups and rows i of prob * (w_i if weighted) * adv_i * score_i, row j drawn from row_law(i, j, x)."""
    total = np.zeros(P_TABLE.size)
    for rows in itertools.product(ROWS, repeat=G):
        r = np.array([_reward(x) for x in rows])
        w = [math.exp(x) for x in _s_rows(rows, P_TABLE, Q_TABLE)]
        for i in range(G):
            pr = math.prod(row_law(i, j, rows[j]) for j in range(G))
            total += pr * (w[i] if weighted else 1.0) * adv(r, i) * _score(P_TABLE, rows[i])
    return total


def _q(i, j, x):
    return _row_prob(x, WAKES[j], Q_TABLE, P_TABLE)


def _p(i, j, x):
    return _row_prob(x, 0, P_TABLE, P_TABLE)


def _theorem_r(i, j, x):  # row i from p, its siblings from q
    return _p(i, j, x) if j == i else _q(i, j, x)


def test_row_weights_unbiased_for_rloo_by_enumeration():
    exact = _row_terms(_adv_rloo, _p, weighted=False)
    got = _row_terms(_adv_rloo, _q, weighted=True)
    assert np.abs(exact).max() > 0.05
    np.testing.assert_allclose(got, exact, rtol=0, atol=1e-12)


def test_row_weights_with_loo_std_give_theorem_r_estimand():
    estimand = _row_terms(_adv_loo_std, _theorem_r, weighted=False)
    got = _row_terms(_adv_loo_std, _q, weighted=True)
    np.testing.assert_allclose(got, estimand, rtol=0, atol=1e-12)
    # ... which is not the on-policy LOO-std gradient: the siblings' reward law is q's (O(dp) prompt scale)
    on_policy = _row_terms(_adv_loo_std, _p, weighted=False)
    np.testing.assert_allclose(on_policy, _on_policy(), rtol=0, atol=1e-12)
    assert np.abs(got - on_policy).max() > 1e-2, np.abs(got - on_policy).max()


def _coupled_wake(i: int, rows: tuple) -> int:
    """Shared-engine timing (REVIEW batch 20, R1): row i's pause point depends on a sibling's content (its first token,
    sampled at time 0 before any wake) -- e.g. a sibling that will run long keeps the engine busy."""
    return 1 if rows[(i + 1) % G][0] == 0 else 2


def _coupled(weight_of, adv) -> tuple[np.ndarray, float]:
    """Exact E_q[sum_i weight_of(i, rows, wakes) adv_i score_i] and E_q[mean_i (w_i - 1) R_i] under the coupled law."""
    total, shift = np.zeros(P_TABLE.size), 0.0
    for rows in itertools.product(ROWS, repeat=G):
        wakes = [_coupled_wake(i, rows) for i in range(G)]
        pr = math.prod(_row_prob(rows[i], wakes[i], Q_TABLE, P_TABLE) for i in range(G))
        s = [isspr.row_log_ratio(_row_lps(P_TABLE, x)[:w], _row_lps(Q_TABLE, x)[:w]) for x, w in zip(rows, wakes)]
        r = np.array([_reward(x) for x in rows])
        total += pr * sum(weight_of(i, s) * adv(r, i) * _score(P_TABLE, rows[i]) for i in range(G))
        shift += pr * float(np.mean([(math.exp(s[i]) - 1.0) * r[i] for i in range(G)]))
    return total, shift


def test_timing_coupling_group_exact_row_cross_term():
    on_rloo = _row_terms(_adv_rloo, _p, weighted=False)
    group, _ = _coupled(lambda i, s: math.exp(math.fsum(s)), _adv_rloo)
    np.testing.assert_allclose(group, on_rloo, rtol=0, atol=1e-12)  # group weights: exact under any timing
    group_std, _ = _coupled(lambda i, s: math.exp(math.fsum(s)), _adv_loo_std)
    np.testing.assert_allclose(group_std, _on_policy(), rtol=0, atol=1e-12)
    row, dp = _coupled(lambda i, s: math.exp(s[i]), _adv_rloo)
    cross = float(np.abs(row - on_rloo).max())
    print(f"[isspr R1] RIS + RLOO under coupled timing: max |E - on-policy| = {cross:.4g} "
          f"(on-policy max |grad| {float(np.abs(on_rloo).max()):.4g}); pooled reward shift dp = {abs(dp):.4g}")
    assert cross > 1e-6, "the coupled toy must expose RIS's cross term"


def test_reward_shift():
    assert isspr.reward_shift([0.0, 0.0], [1.0, 0.0]) == 0.0
    assert math.isclose(isspr.reward_shift([math.log(2.0), math.log(0.5)], [1.0, 1.0]), 0.25)
    with pytest.raises(RuntimeError):
        isspr.reward_shift([0.0], [1.0, 0.0])


def test_negative_stale_state_is_biased():
    # Skip the recompute: post-wake tokens come from stale state (here: another conditional law), the weight only
    # sees the pre-wake tokens. Biased; with the recompute (post-wake tokens from p) the same weight is exact.
    exact = _on_policy()
    got = _expect(lambda i, x: _row_prob(x, WAKES[i], Q_TABLE, STALE_TABLE),
                  lambda rows: math.exp(math.fsum(_s_rows(rows, P_TABLE, Q_TABLE))))
    assert np.abs(got - exact).max() > 1e-2, np.abs(got - exact).max()


# -- the weight log --------------------------------------------------------------------------------------------------


def test_clipped_weights_refuse_nan():
    assert isspr.clipped_weights([0.0, 10.0], 5.0) == [1.0, 5.0]
    with pytest.raises(RuntimeError, match="non-finite"):
        isspr.clipped_weights([float("nan")], 5.0)


def test_weight_stats():
    st = isspr.weight_stats([0.0] * 64, 5.0)
    assert st["ess"] == 1.0 and st["ess_clip"] == 1.0 and st["eps_hat"] == 0.0 and st["s_var"] == 0.0
    assert st["clipped"] == 0.0 and st["w_mean"] == 1.0
    s = [math.log(10.0), 0.0, 0.0, 0.0]
    st = isspr.weight_stats(s, 5.0)
    assert math.isclose(st["eps_hat"], 5.0 / 4)
    assert math.isclose(st["ess"], 13.0**2 / (4 * 103.0))
    assert math.isclose(st["ess_clip"], 8.0**2 / (4 * 28.0))
    assert st["clipped"] == 1.0 and math.isclose(st["w_mean"], 2.0)
    assert isspr.clipped_weights([800.0, -800.0], 5.0) == [5.0, 0.0]
    # far-off-policy steps (every e^{s_g} underflows) still log finite numbers
    st = isspr.weight_stats([-900.0, -901.0, -950.0], 5.0)
    assert 0.0 < st["ess"] < 1.0 and st["ess"] == st["ess_clip"] and st["w_mean"] == 0.0 and st["eps_hat"] == 0.0
    # lognormal s (variance v): ESS -> exp(-v), the theory's number
    rng = np.random.default_rng(0)
    v = 0.25
    st = isspr.weight_stats((rng.normal(-v / 2, math.sqrt(v), 200000)).tolist(), 5.0)
    assert abs(st["ess"] - math.exp(-v)) < 0.01 and abs(st["s_var"] - v) < 0.01


def test_age_stats():
    c = [isspr.IsCert(version=3, key=0, tokens=10, s=-0.002, rescored_tokens=10, first_version=2),
         isspr.IsCert(version=3, key=1, tokens=10, s=-0.006, rescored_tokens=20, first_version=2),
         isspr.IsCert(version=3, key=2, tokens=5, s=0.0, rescored_tokens=0, first_version=3)]
    st = isspr.age_stats(3, c)
    assert st[0] == (1, 0.0) and st[1][0] == 2 and math.isclose(st[1][1], 0.008 / 30)


def test_token_stats():
    r = [np.zeros(999), np.array([5.0])]
    st = isspr.token_stats(r)
    assert st["tokens"] == 1000 and st["abs_top0.001"] == 1.0 and st["sq_top0.001"] == 1.0
    st = isspr.token_stats([np.full(1000, -0.01)])
    assert math.isclose(st["abs_top0.01"], 0.01) and math.isclose(st["q"][3], -0.01)
    assert isspr.token_stats([]) == {"tokens": 0}


def test_row_log_ratio():
    assert isspr.row_log_ratio([-1.0, -2.0, -0.5], [-1.0, -2.0, -0.5]) == 0.0
    assert isspr.row_log_ratio([-1.0, -2.0], [-1.5, -1.0]) == -0.5
    with pytest.raises(RuntimeError):
        isspr.row_log_ratio([-1.0], [-1.0, -2.0])
    # a post-wake token copied into both p and q at -inf / NaN must raise, not give s = NaN (seen at step 6 of a run)
    with pytest.raises(RuntimeError, match="position 1"):
        isspr.row_log_ratio([-1.0, float("-inf")], [-1.0, float("-inf")])
    with pytest.raises(RuntimeError, match="position 0"):
        isspr.row_log_ratio([float("nan")], [float("nan")])


# -- the loss --------------------------------------------------------------------------------------------------------


def _loss_fn(token_level: bool):
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn

    return ClippedPGLossFn(ClippedPGLossConfig(
        token_level_loss=token_level, force_on_policy_ratio=True, reference_policy_kl_penalty=0.0,
        use_importance_sampling_correction=True, ratio_clip_min=0.2, ratio_clip_max=0.28, metrics_level="minimal"))


def _row_grads(lengths: list[int], weights=None, token_level=False) -> list[torch.Tensor]:
    """d loss / d curr_logprobs of every row (its valid tokens) for rows of the given generated lengths."""
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    b, s = len(lengths), max(lengths) + 1
    tm = torch.zeros(b, s)
    for i, n in enumerate(lengths):
        tm[i, 1:n + 1] = 1.0
    lp = torch.full((b, s), -1.0)
    data = BatchedDataDict({
        "input_ids": torch.zeros(b, s, dtype=torch.long), "token_mask": tm, "sample_mask": torch.ones(b),
        "advantages": torch.arange(1, b + 1, dtype=torch.float32).unsqueeze(-1).expand(b, s).clone(),
        "generation_logprobs": lp.clone(), "prev_logprobs": lp.clone(),
    })
    if weights is not None:
        data["is_weight"] = torch.tensor(weights, dtype=torch.float32)
    curr = lp[:, 1:].clone().requires_grad_(True)
    loss, _ = _loss_fn(token_level)(curr, data, global_valid_seqs=torch.tensor(float(b)),
                                    global_valid_toks=(tm[:, 1:]).sum())
    loss.backward()
    return [curr.grad[i, :n].clone() for i, n in enumerate(lengths)]


def test_loss_is_weight_scales_row_gradients():
    lengths = [3, 5, 2, 4]
    w = [1.0, 0.5, 5.0, 2.25]
    base = _row_grads(lengths)
    got = _row_grads(lengths, weights=w)
    for g0, g1, wi in zip(base, got, w):
        assert torch.allclose(g1, g0 * wi, rtol=1e-6, atol=0), (g0, g1, wi)
    # the per-token value: -A_r W_r / (n_r N) under the sequence-level normalizer
    for i, (g, n) in enumerate(zip(got, lengths)):
        assert torch.allclose(g, torch.full((n,), -(i + 1) * w[i] / (n * len(lengths)))), (g, n)


def test_loss_normalizer_constant_across_token_counts():
    # Row 0's gradient must not depend on the other rows' token counts (a step with more tokens): constant
    # normalizer. Token-level loss (the stock setting) fails this, so the group terms would not add up.
    a = _row_grads([4, 2, 3, 2])
    b = _row_grads([4, 900, 700, 1000])
    assert torch.equal(a[0], b[0])
    ta = _row_grads([4, 2, 3, 2], token_level=True)
    tb = _row_grads([4, 900, 700, 1000], token_level=True)
    assert not torch.allclose(ta[0], tb[0])


# -- config ----------------------------------------------------------------------------------------------------------


def test_config_block():
    cfg = isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "clip_c": 5.0}})
    assert cfg.clip_c == 5.0 and cfg.seed_stream == "isspr" and cfg.weight == "group"
    assert isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "weight": "row"}}).weight == "row"
    cf = isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "crossfade_routing_m": 0.01,
                                                         "crossfade_routing_extra": 8}})
    assert cf.crossfade_routing_m == 0.01 and cf.crossfade_routing_sparse and cf.crossfade_routing_sparse_engine
    with pytest.raises(ValueError):
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "weight": "token"}})
    with pytest.raises(ValueError):  # exact SPR's fork dump is not an IS-SPR key
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "row_dump_dir": "/tmp/x"}})
    with pytest.raises(ValueError):
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "clip_c": 1.0}})
    with pytest.raises(ValueError):
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "tau": 1}})
    with pytest.raises(ValueError):
        isspr.IsSprConfig.from_config({"thundersync": {"mode": "xspr"}})


def test_loss_problems():
    def probs(loss):
        return [x for x in isspr.config_problems({"loss_fn": loss}) if x.startswith("loss_fn")]

    ok = {"token_level_loss": False, "force_on_policy_ratio": True, "reference_policy_kl_penalty": 0}
    assert probs(ok) == []
    assert len(probs({**ok, "token_level_loss": True})) == 1
    assert len(probs({**ok, "force_on_policy_ratio": False})) == 1
    assert len(probs({**ok, "reference_policy_kl_penalty": 0.01})) == 1
    assert len(probs({**ok, "truncated_importance_sampling_ratio": 2.0})) == 1

    def gen_probs(gen):
        return [x for x in isspr.config_problems({"policy": {"generation": gen}}) if "temperature" in x or
                "num_speculative_tokens" in x]

    assert gen_probs({"temperature": 1.0, "mcore_generation_config": {"num_speculative_tokens": 0}}) == []
    assert len(gen_probs({"temperature": 0.7})) == 1
    assert len(gen_probs({"mcore_generation_config": {"num_speculative_tokens": 2}})) == 1


def test_learner_resident_only_where_a_mode_sets_it():
    """xspr / isspr offload before the refit as stock / SPR-BI (the K = 4 isspr arms OOMed at the TMS resume with
    the grad buffers resident); colo / nolanes set megatron_cfg.resident_learner and keep them."""
    import thundersync_rl.worker as w

    calls = []
    stock = w.MegatronPolicyWorkerImpl.offload_before_refit
    w.MegatronPolicyWorkerImpl.offload_before_refit = lambda self: calls.append("stock")
    try:
        cls = w.ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class  # the class behind the Ray actor

        def worker(**mcfg):
            obj = object.__new__(cls)  # no constructor: only the attributes it reads
            obj.__dict__.update(is_generation_colocated=True, cfg={"megatron_cfg": mcfg},
                                finalize_async_save=lambda: calls.append("finalize"))
            return obj

        cls.offload_before_refit(worker(engine_verify=True))
        assert calls == ["stock"]
        calls.clear()
        cls.offload_before_refit(worker(resident_learner=True))
        assert calls == ["finalize"]
    finally:
        w.MegatronPolicyWorkerImpl.offload_before_refit = stock


def test_advantages_must_be_group_local():
    from types import SimpleNamespace

    def stage_cfg(name, shardable, is_ppo=False):
        return SimpleNamespace(is_ppo=is_ppo, shardable=shardable,
                               algo=SimpleNamespace(adv_estimator=SimpleNamespace(name=name)))

    isspr.require_group_local(stage_cfg("grpo", True))
    with pytest.raises(RuntimeError):
        isspr.require_group_local(stage_cfg("reinforce_plus_plus", False))
    with pytest.raises(RuntimeError):
        isspr.require_group_local(stage_cfg("gae", False, is_ppo=True))


def _fake_sc(shardable=True, rloo=False):
    from types import SimpleNamespace

    impl = SimpleNamespace(_generate_response=None)
    est = SimpleNamespace(name="grpo", normalize_rewards=not rloo, use_leave_one_out_baseline=True)
    return SimpleNamespace(
        _algo_cfg=SimpleNamespace(seed=42, num_generations_per_prompt=8),
        _advantage_stage_config=SimpleNamespace(is_ppo=False, shardable=shardable,
                                                algo=SimpleNamespace(adv_estimator=est)),
        _rollout_manager=SimpleNamespace(_impl=impl, weight_version=lambda: 0, nl_record_sink=None),
    )


def test_stage_builds_with_its_row_loop_and_negative():
    import os
    from types import SimpleNamespace

    from thundersync_rl.isspr_stage import IsSprStage

    # LoggerConfig is a pydantic model: attribute access only
    mc = SimpleNamespace(policy={"generation": {"max_new_tokens": 64}}, logger=SimpleNamespace(log_dir="/tmp/isspr-test"))
    cfg = isspr.IsSprConfig()
    old = os.environ.pop("THUNDERSYNC_NL_NEG", None)
    try:
        sc = _fake_sc()
        st = IsSprStage(sc, mc, cfg)
        assert isinstance(st.rows, isspr.IsRows) and sc._rollout_manager._impl.xspr is st.rows
        assert st.wake_cut and st.rows.neg == "" and isspr.IS_WEIGHT_FIELD in st.extra_train_fields
        os.environ["THUNDERSYNC_NL_NEG"] = isspr.NEG_STALE_KV
        st = IsSprStage(_fake_sc(), mc, cfg)
        assert not st.wake_cut and st.rows.neg == isspr.NEG_STALE_KV
        with pytest.raises(RuntimeError):
            IsSprStage(_fake_sc(shardable=False), mc, cfg)
    finally:
        os.environ.pop("THUNDERSYNC_NL_NEG", None)
        if old is not None:
            os.environ["THUNDERSYNC_NL_NEG"] = old


# -- the row loop on exact SPR's toy engine --------------------------------------------------------------------------


def _run(world, target, n_rows=6, neg="", start_version=0, void_first_check=False, learner_s=False):
    """``void_first_check``: each row's first check request comes back cut mid-prefill (chunked prefill across a
    wake: no result, no new token), as the engine voids it. ``learner_s``: the row unit's loop (no rescore)."""
    from test_xspr import PROMPT, FakeImpl

    from nemo_rl.experience.rollout_manager import ROLLOUT_TARGET_STEP

    world.version = start_version
    rows = isspr.IsRows(seed=7, stream="t", max_new_tokens=world.cap, version_fn=lambda: world.version, neg=neg)
    rows.learner_s = learner_s
    q_of: dict[int, dict[int, float]] = {}  # key -> {position: log-prob the token was decoded with}

    class Recording(FakeImpl):
        voided: set = set()

        async def _generate_response(self, message_log, stop_strings, native_generation_data=None, forced=None):
            data = forced["generation_data"]
            if void_first_check and data.get("xspr_check") and data["xspr_key"] not in self.voided:
                self.voided.add(data["xspr_key"])
                await self.w.step()
                msg = self._msg(list(forced["prefix"]), list(forced["prefix_logprobs"]))
                msg["xspr"] = {"cut": 1, "check": None, "wake_at": []}
                return msg, torch.tensor([len(message_log[0]["token_ids"])]), {}
            msg, in_len, m = await super()._generate_response(message_log, stop_strings, native_generation_data, forced)
            if not (data.get("xspr_check") and not data.get("xspr_continue")):  # check-only: no kept token
                q = q_of.setdefault(int(data["xspr_key"]), {})
                for t in range(len(forced["prefix"]), len(msg["token_ids"])):
                    q[t] = float(msg["generation_logprobs"][t])
            return msg, in_len, m

    impl = Recording(world)

    async def one(j):
        ROLLOUT_TARGET_STEP.set(target)
        msg, _, _ = await rows.respond(impl, [{"token_ids": PROMPT}], None, None, {"idx": 11}, j)
        return msg

    async def drive():
        task = asyncio.ensure_future(asyncio.gather(*(one(j) for j in range(n_rows))))
        while not task.done():
            await world.step()
            if world.tick > 10000:
                task.cancel()
                raise AssertionError("rows never finished")
        return task.result()

    msgs = asyncio.run(drive())
    out = []
    for j, m in enumerate(msgs):
        key = rows.key_of(target, 11, j)
        out.append((m["token_ids"].tolist(), m["generation_logprobs"].tolist(), rows.pop_cert(target, 11, j),
                    q_of.get(key, {})))
    return world, out


def _lp(world, v, ctx, tok):
    from test_xspr import log_softmax

    return float(np.float32(log_softmax(world.logits(v, ctx))[tok]))


def _check_rows(world, results, target):
    """Every token's generation log-prob is theta_target's (D2); s = sum over tokens of (p_target - q), q the log-prob
    the engine decoded the token with."""
    from test_xspr import PROMPT

    for toks, lps, cert, q_at in results:
        assert cert.version == target
        assert len(cert.log_ratios) == cert.rescored_tokens and math.isclose(
            float(cert.log_ratios.sum()), cert.s, abs_tol=1e-9)
        p = [_lp(world, target, PROMPT + toks[:t], toks[t]) for t in range(len(toks))]
        q = [q_at[t] for t in range(len(toks))]
        assert np.array_equal(np.float32(lps), np.float32(p))
        assert math.isclose(cert.s, math.fsum(float(a) - float(b) for a, b in zip(np.float32(p), q)), abs_tol=1e-9)


def test_rows_without_a_wake_have_weight_one():
    from test_xspr import World

    w, res = _run(World(wakes={}), target=0)
    _check_rows(w, res, 0)
    assert all(c.rescores == 0 and c.s == 0.0 for _, _, c, _ in res)


def test_rows_across_a_wake_rescored_and_resumed():
    from test_xspr import World

    w, res = _run(World(wakes={20: 1}), target=1)
    _check_rows(w, res, 1)
    assert any(c.rescores for _, _, c, _ in res) and any(c.s != 0.0 for _, _, c, _ in res)
    assert all(c.first_version == 0 for _, _, c, _ in res)


def test_check_cut_mid_prefill_is_redone():
    from test_xspr import World

    for wakes in ({20: 1}, {400: 1}):  # resumed rows and rows finished before the wake
        w, res = _run(World(wakes=wakes), target=1, void_first_check=True)
        _check_rows(w, res, 1)
        assert any(c.rescores for _, _, c, _ in res)


def test_rows_finished_before_the_wake_are_rescored():
    from test_xspr import World

    w, res = _run(World(wakes={400: 1}), target=1)
    _check_rows(w, res, 1)
    assert all(c.rescores == 1 and c.rescored_tokens == len(t) for t, _, c, _ in res)


def test_lag_two():
    from test_xspr import World

    w, res = _run(World(wakes={6: 1, 30: 2}), target=2, n_rows=8)
    _check_rows(w, res, 2)
    assert max(c.rescores for _, _, c, _ in res) >= 2


def test_negative_stale_kv_keeps_q():
    from test_xspr import World

    class NoCut(World):  # the engine's wake cut off: paused rows resume where they were
        async def step(self):
            await asyncio.sleep(0)
            self.tick += 1
            if self.tick in self.wakes:
                self.version = self.wakes[self.tick]

    from test_xspr import PROMPT

    w, res = _run(NoCut(wakes={20: 1}), target=1, neg=isspr.NEG_STALE_KV)
    assert all(c.rescores == 0 and c.s == 0.0 and c.version == 0 for _, _, c, _ in res)
    # what the sentinel sees: generation log-probs that are not theta_1's
    p1 = [np.float32([_lp(w, 1, PROMPT + t[:i], t[i]) for i in range(len(t))]) for t, _, _, _ in res]
    off = sum(int(not np.array_equal(np.float32(lps), p)) for (_, lps, _, _), p in zip(res, p1))
    assert off > 0


# -- row weights from the learner (learner s) ------------------------------------------------------------------------


def test_old_after_from_labels_and_wake_marks():
    # a segment that ends below theta_target leaves every token so far old
    assert isspr._old_after(0, 0, 7, 0, {"wake_at": []}, 1) == 7
    assert isspr._old_after(3, 5, 9, 0, {"wake_at": [2, 6]}, 3) == 9
    # sent at theta_target: nothing new is old
    assert isspr._old_after(4, 4, 9, 1, {"wake_at": []}, 1) == 4
    # in flight across the wake that installed theta_target (no cut): its tokens before the mark are old
    assert isspr._old_after(0, 0, 9, 0, {"wake_at": [5]}, 1) == 5
    assert isspr._old_after(0, 3, 9, 0, {"wake_at": [1, 4]}, 2) == 7
    with pytest.raises(RuntimeError, match="version 2"):
        isspr._old_after(0, 0, 9, 1, {"wake_at": [5]}, 1)


def _learner_rows(world, target, **kw):
    """Rows of the learner-s loop: q kept, no rescore, and the tokens after old_tokens sampled under theta_target from
    theta_target's state (the toy decodes from the context: exactly theta_target's log-probs)."""
    from test_xspr import PROMPT

    w, res = _run(world, target, learner_s=True, **kw)
    for toks, lps, cert, q_at in res:
        assert cert.version == target and cert.rescores == 0 and cert.only_tokens == 0 and math.isnan(cert.s)
        assert lps == [q_at[t] for t in range(len(toks))]  # the generation log-probs stay q
        assert 0 <= cert.old_tokens <= len(toks)
        for t in range(cert.old_tokens, len(toks)):
            assert np.float32(lps[t]) == np.float32(_lp(w, target, PROMPT + toks[:t], toks[t]))
    return w, res


def test_learner_s_finished_rows_not_rescored_cut_rows_resumed():
    from test_xspr import World

    _, res = _learner_rows(World(wakes={400: 1}), target=1)  # every row finished under theta_0
    assert all(c.old_tokens == len(t) and c.resume_tokens == 0 for t, _, c, _ in res)
    _, res = _learner_rows(World(wakes={20: 1}), target=1)  # rows cut at the wake resume by a plain prefill
    assert any(c.resume_tokens > 0 for _, _, c, _ in res)
    assert any(0 < c.old_tokens < len(t) for t, _, c, _ in res)
    _, res = _learner_rows(World(wakes={6: 1, 30: 2}), target=2, n_rows=8)
    assert any(c.resume_tokens > 0 for _, _, c, _ in res)


def test_learner_s_without_the_wake_cut_marks_the_new_tokens():
    # The stale-KV negative's engine (no wake cut): a row in flight across the wake keeps decoding; the wake mark
    # puts its post-wake tokens inside the sentinel (where the real engine's stale state differs from the learner).
    from test_xspr import PROMPT, World

    w, res = _run(World(wakes={20: 1}, cut=False), target=1, neg=isspr.NEG_STALE_KV, learner_s=True)
    straddled = [(t, c) for t, _, c, _ in res if 0 < c.old_tokens < len(t)]
    assert straddled
    for toks, c in straddled:
        assert c.rescores == 0 and c.resume_tokens == 0


def test_mark_wake_appends_generated_counts():
    from thundersync_rl.nolanes_engine import KEY_RID_ATTR
    from thundersync_rl.xspr_engine import WAKE_AT_ATTR, mark_wake

    class SP:
        def add_attributes(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    from types import SimpleNamespace

    class Rec(SimpleNamespace):
        def __getitem__(self, i):
            return self.requests[i]

    def entry(keyed, gens):
        sp = SP()
        if keyed:
            setattr(sp, KEY_RID_ATTR, 1)
        return SimpleNamespace(record=Rec(requests=[SimpleNamespace(sampling_params=sp, generated_tokens=[0] * g)
                                                    for g in gens])), sp

    (a, spa), (b, spb), (c, spc) = entry(True, [3, 2]), entry(True, []), entry(False, [4])
    engine = SimpleNamespace(requests={1: a, 2: b, 3: c})
    assert mark_wake(engine) == 2
    assert mark_wake(engine) == 2
    assert getattr(spa, WAKE_AT_ATTR) == [5, 5] and getattr(spb, WAKE_AT_ATTR) == [0, 0]
    assert not hasattr(spc, WAKE_AT_ATTR)


def _row_data(lengths, old, gen_shift=0.0, audit_shift=None):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    b, s = len(lengths), max(lengths) + 1
    tm = torch.zeros(b, s)
    for i, n in enumerate(lengths):
        tm[i, 1:n + 1] = 1.0
    lp = torch.full((b, s), -1.0)
    gen = lp.clone()
    for i, n in enumerate(lengths):  # the first old[i] tokens were sampled under older weights: log q differs
        gen[i, 1:1 + old[i]] = -1.0 - gen_shift * (i + 1)
    data = BatchedDataDict({
        "input_ids": torch.zeros(b, s, dtype=torch.long), "token_mask": tm, "sample_mask": torch.ones(b),
        "advantages": torch.arange(1, b + 1, dtype=torch.float32).unsqueeze(-1).expand(b, s).clone(),
        "generation_logprobs": gen, "prev_logprobs": lp.clone(),
        "is_weight": torch.ones(b), "is_old_tokens": torch.tensor(old, dtype=torch.int64),
    })
    if audit_shift is not None:
        audit = lp.clone()
        audit[0, 1] += audit_shift
        data["is_audit_logprobs"] = audit
    return data, lp, tm


def test_is_row_weights_from_the_forward():
    from nemo_rl.algorithms.loss.loss_functions import is_row_weights

    data, lp, tm = _row_data([3, 5, 2], old=[2, 0, 2], gen_shift=0.25, audit_shift=0.0)
    curr = lp[:, 1:]
    w, fresh, m = is_row_weights(curr, data["generation_logprobs"][:, 1:], tm[:, 1:], data["is_old_tokens"], 5.0,
                                 data["is_audit_logprobs"])
    s = torch.tensor([2 * 0.25, 0.0, 2 * 0.75], dtype=torch.float64)  # sum (log p - log q) over the old tokens
    assert torch.allclose(w, torch.clamp(torch.exp(s), max=5.0))
    assert fresh.sum().item() == (3 - 2) + 5 + 0  # the sentinel's tokens: those sampled under these weights
    assert m["is_rows"] == 3 and m["is_old_tok"] == 4 and m["is_clipped"] == 0
    assert math.isclose(m["is_s_sum"], 2.0) and m["is_audit_tok"] == 10 and m["is_audit_inexact"] == 0
    # clip, and an audit that differs in one token
    data, lp, tm = _row_data([3], old=[3], gen_shift=1.0, audit_shift=1e-3)
    w, _, m = is_row_weights(lp[:, 1:], data["generation_logprobs"][:, 1:], tm[:, 1:], data["is_old_tokens"], 5.0,
                             data["is_audit_logprobs"])
    assert w.item() == 5.0 and m["is_clipped"] == 1 and math.isclose(m["is_excess_sum"], math.exp(3.0) - 5.0)
    assert m["is_audit_inexact"] == 1
    # lr 0: the forward is the sampling law: s = 0, w = 1
    data, lp, tm = _row_data([4, 2], old=[4, 1])
    w, _, m = is_row_weights(lp[:, 1:], data["generation_logprobs"][:, 1:], tm[:, 1:], data["is_old_tokens"], 5.0)
    assert torch.equal(w, torch.ones(2, dtype=torch.float64)) and m["is_s2_sum"] == 0.0 and m["is_nonfinite"] == 0
    # a non-finite log-prob is counted (the stage raises on it), never mapped to a weight
    curr = lp[:, 1:].clone()
    curr[1, 0] = float("nan")
    _, _, m = is_row_weights(curr, data["generation_logprobs"][:, 1:], tm[:, 1:], data["is_old_tokens"], 5.0)
    assert m["is_nonfinite"] == 1


def test_loss_learner_row_weights_scale_gradients_and_narrow_the_sentinel():
    from nemo_rl.algorithms.loss.loss_functions import ClippedPGLossConfig, ClippedPGLossFn

    def fn(clip):
        return ClippedPGLossFn(ClippedPGLossConfig(
            token_level_loss=False, force_on_policy_ratio=True, reference_policy_kl_penalty=0.0,
            use_importance_sampling_correction=True, ratio_clip_min=0.2, ratio_clip_max=0.28, metrics_level="minimal",
            train_lp_check=True, is_row_clip=clip))

    lengths, old = [3, 5, 2, 4], [2, 0, 1, 4]
    data, lp, tm = _row_data(lengths, old, gen_shift=0.2)
    grads = []
    for clip in (None, 5.0):
        d = data.copy() if hasattr(data, "copy") else data
        curr = lp[:, 1:].clone().requires_grad_(True)
        loss, metrics = fn(clip)(curr, d, global_valid_seqs=torch.tensor(4.0), global_valid_toks=tm[:, 1:].sum())
        loss.backward()
        grads.append(curr.grad.clone())
        if clip is None:
            assert metrics["train_gen_lp_inexact_tokens"] == sum(old)  # old tokens' q differs from the forward
        else:
            assert metrics["train_gen_lp_inexact_tokens"] == 0 and metrics["train_gen_lp_tokens"] == sum(
                n - o for n, o in zip(lengths, old))
    s = [0.2 * (i + 1) * o for i, o in enumerate(old)]
    for i, n in enumerate(lengths):
        assert torch.allclose(grads[1][i, :n], grads[0][i, :n] * min(5.0, math.exp(s[i])), rtol=1e-6, atol=0)


def test_stage_learner_s_fields():
    import os
    from types import SimpleNamespace

    from thundersync_rl.isspr_stage import IsSprStage

    mc = SimpleNamespace(policy={"generation": {"max_new_tokens": 64}}, logger=SimpleNamespace(log_dir="/tmp/isspr-test"))
    old = os.environ.pop("THUNDERSYNC_NL_NEG", None)
    try:
        cfg = isspr.IsSprConfig.from_config({"thundersync": {"mode": "isspr", "weight": "auto",
                                                            "length_normalizer": "constant"},
                                             "async_rl": {"sampler": {"max_lookahead_versions": 2}},
                                             "policy": {"generation": {"max_new_tokens": 64}}})
        st = IsSprStage(_fake_sc(rloo=True), mc, cfg)
        assert st.learner_s and st.rows.learner_s
        assert {isspr.IS_WEIGHT_FIELD, isspr.IS_OLD_FIELD} <= set(st.extra_train_fields)
        st = IsSprStage(_fake_sc(), mc, isspr.IsSprConfig())  # group unit: the engine rescore
        assert not st.learner_s and not st.rows.learner_s and st.extra_train_fields == (isspr.IS_WEIGHT_FIELD,)
    finally:
        if old is not None:
            os.environ["THUNDERSYNC_NL_NEG"] = old


def test_nonfinite_learner_step_dumps_its_rows():
    import json
    import os
    import tempfile
    from types import SimpleNamespace

    from thundersync_rl.isspr_stage import IsSprStage

    d = tempfile.mkdtemp()
    mc = SimpleNamespace(policy={"generation": {"max_new_tokens": 64}}, logger=SimpleNamespace(log_dir=d))
    old = os.environ.pop("THUNDERSYNC_NL_NEG", None)
    try:
        st = IsSprStage(_fake_sc(), mc, isspr.IsSprConfig())
    finally:
        if old is not None:
            os.environ["THUNDERSYNC_NL_NEG"] = old
    comp = SimpleNamespace(message_log=[{"role": "user", "token_ids": torch.tensor([1, 2])},
                                        {"role": "assistant", "token_ids": torch.tensor([5, 6, 7]),
                                         "generation_logprobs": torch.tensor([-0.5, -1.0, -2.0])}])
    rec = SimpleNamespace(prompt_idx=11, completions=[comp, comp])
    st._step_rows = (3, [({"message_log": [{"token_ids": torch.tensor([1, 2])}]}, rec)])
    try:
        st.check_train_lp({"grad_norm": float("nan"), "loss": 0.1}, 3)
        raise AssertionError("a non-finite step must raise")
    except RuntimeError as e:
        assert "non-finite (grad_norm)" in str(e)
        path = str(e).split("dumped to ")[-1].split(" for")[0]
    rows = [json.loads(l) for l in open(path)]
    assert len(rows) == 2 and rows[1]["sample"] == 1 and rows[0]["token_ids"] == [5, 6, 7]
    assert rows[0]["prompt_token_ids"] == [1, 2] and rows[0]["key"] == st.rows.key_of(3, 11, 0)


def test_learner_rows_counted_once_and_cp_refused():
    import os
    import tempfile
    from types import SimpleNamespace

    from thundersync_rl.isspr_stage import IsSprStage

    mc = SimpleNamespace(policy={"generation": {"max_new_tokens": 64}},
                         logger=SimpleNamespace(log_dir=tempfile.mkdtemp()))
    full = {"thundersync": {"mode": "isspr", "weight": "auto", "length_normalizer": "constant"},
            "async_rl": {"sampler": {"max_lookahead_versions": 2}}, "policy": {"generation": {"max_new_tokens": 64}}}
    old = os.environ.pop("THUNDERSYNC_NL_NEG", None)
    try:
        st = IsSprStage(_fake_sc(rloo=True), mc, isspr.IsSprConfig.from_config(full))
    finally:
        if old is not None:
            os.environ["THUNDERSYNC_NL_NEG"] = old
    comp = SimpleNamespace(message_log=[{"role": "assistant", "token_ids": torch.tensor([5, 6, 7]),
                                         "generation_logprobs": torch.zeros(3)}])
    st._step_rows = (2, [({"message_log": []}, SimpleNamespace(prompt_idx=1, completions=[comp, comp]))])
    m = {"is_rows": 2.0, "is_s_sum": 0.0, "is_s2_sum": 0.0, "is_w_sum": 2.0, "is_w2_sum": 2.0, "is_e_sum": 2.0,
         "is_e2_sum": 2.0, "is_excess_sum": 0.0, "is_clipped": 0.0, "is_nonfinite": 0.0, "is_old_tok": 0.0,
         "train_gen_lp_tokens": 6.0, "train_gen_lp_inexact_tokens": 0.0, "train_gen_lp_abs_diff_max": 0.0}
    st.check_train_lp(dict(m), 2)
    with pytest.raises(RuntimeError, match="weighted 4 rows, the step trained 2"):
        st.check_train_lp(dict(m, is_rows=4.0), 2)  # a TP / CP duplicate
    cp = dict(full, policy={"generation": {"max_new_tokens": 64}, "megatron_cfg": {"context_parallel_size": 2}})
    assert any("context_parallel_size" in x for x in isspr.estimator_problems(cp))
    grp = dict(cp, async_rl={"sampler": {"max_lookahead_versions": 1}})  # auto at K 1: group, no row weights
    assert not any("context_parallel_size" in x for x in isspr.estimator_problems(grp))
