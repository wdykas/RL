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
"""IS-SPR (``thundersync.mode=isspr``; docs/colo/THEORY.md, "IS-SPR"): stock colocated partial
rollout with exact state at every wake and an importance weight per GRPO group (default) or per row (RIS, opt-in).

Rule.
1. Stock SPR: in_order sampler at lag K, rows paused in the engine at the stand-down. No check, no truncation, no
   re-decode.
2. At every wake every row that ran under older weights is rescored under theta_n by one engine prefill (the exact-SPR
   recompute layer, thundersync_rl/xspr_engine.py): paused rows are cut and resumed on prompt + their tokens, so their
   KV / SSM state is theta_n's; rows that finished earlier are rescored once theta_target is installed (D1). The
   prefill's log p_n replace the generation log-probs (D2): training never sees an older version's value.
3. Row r's log likelihood ratio against theta_target is s_r = sum over its tokens of (log p_n - log q), q the
   log-prob the token was sampled with. Tokens sampled after the last rescore are theta_target samples from
   theta_target state, so their term is exactly 0: s_r needs no pre-wake mask. A group's is s_g = sum_r s_r.
4. The weight, detached, scales the advantages (loss_functions.py, field ``is_weight``). The loss is the per-row token
   mean, then the mean over the batch's rows (token_level_loss false): a constant normalizer, so the batch loss is a
   sum of row terms.
   - ``weight: group`` (default): W_g = min(C, exp(s_g)) on all of g's rows; E_q[W_g ghat(X_g)] = E_p[ghat(Y_g)] for
     any group statistic ghat, whatever the engine timing does to the pause points (the whole group is p's).
   - ``weight: row`` (RIS, opt-in; research/colo/FIRSTPRINCIPLES_C.md §3.2): w_r = min(C, exp(s_r)) on row r. Row r's
     term is f_r(Y_r; R_-r). IF the siblings are independent of Y_r under q, E_q[w_r f_r] = E_{R_-r ~ q} E_{Y_r ~ p}
     [f_r] (Theorem R): exactly on-policy for RLOO / LOO-mean advantages with a constant normalizer; for the stock
     LOO-std advantage the prompt-reweighted gradient sum_x c(p_x^q) grad J_x (U', an O(dp) prompt scale; dp_hat =
     |mean_r (e^{s_r} - 1) R_r| is logged). The condition fails in colocated SPR (REVIEW.md batch 20, R1): a group's
     rows share one engine, so a row's pause point depends on its siblings' lengths, and the siblings' version mix on
     the row's content; then even RLOO keeps a timing-coupling cross term (sized in test_isspr's coupled enumeration).
     Both units' ESS / var and every s_r are logged in either mode.
   - ``weight: auto`` (IS-RLOO, user 11-06): group at lag K <= 1, row at lag K >= 2 (EARLY_LATE.md: the group product's
     ESS falls to 0.2-0.57 with 4-26 % clip mass from K = 2; row weights hold to K = 2-3). See "IS-RLOO" below.
   Either way minus the clip's share, E_q[(e^s - C)^+ f], estimated without bias by eps_hat = mean (e^s - C)^+ and
   logged every step with mean / var of s and the ESS. Not exact in law.
IS-RLOO (``weight: row`` or ``auto``; FIRSTPRINCIPLES_C §3.2; tests/unit/test_isspr_rloo.py). Row weights are exact
only for a statistic whose dependence on row r's content is through its own score with a baseline that does not involve
Y_r, and only if that score has mean 0 under p. The validator therefore pins the objective whenever row weights can be
used (``row``, ``auto``):
   - advantage A_r = R_r - mean_{j != r} R_j (RLOO: grpo.adv_estimator use_leave_one_out_baseline true,
     normalize_rewards false). GRPO's std is DROPPED, not replaced by the leave-one-out std: 1 / std_{-r} is a function
     of the siblings, which the row weight leaves at q's law, so with it E_q[w_r f_r] = sum_x E_q[1 / std_{-r}] grad J_x
     against on-policy's E_p[1 / std_{-r}]: a per-prompt scale bias of O(dp) (2-5 % at p = 0.05-0.1 for dp = 0.01),
     and no estimand is exactly the on-policy one. Without std the estimator is G grad J / L_c exactly.
   - constant length normalizer (``length_normalizer: constant``): row r's loss is sum_t (...) / L_c with
     L_c = policy.generation.max_new_tokens, a constant. The per-row token mean (token_level_loss false) divides by
     l_r, a function of Y_r: then h_r = score_r / l_r has E_p[h_r] != 0 and the baseline term leaves
     (E_q[mean_{-r} R] - E_p[mean_{-r} R]) E_p[h_r], an O(dp) bias. Implemented without a loss change: the loss keeps
     the per-row token mean and the ``is_weight`` column carries w_r * l_r / L_c (``train_coefficients``); l_r is the
     row's trained token count. The weight log (ESS, var, eps_hat) is of w alone.
   With K = 1 under ``auto`` the group weight is used with the same objective (the group product is exact for it).
   Residual (R1, REVIEW batch 20): rows of a group share one engine, so a row's pause point can depend on its siblings'
   content; then E_q[w_r f_r] keeps a timing-coupling cross term (test_isspr's coupled enumeration: 0.038 vs max |grad|
   1.18), which the group weight does not have. Its size at Nano is being simulated (CPU); dp_hat is logged.
Conditions (validated): q is the sampling law (temperature 1, one token per step), ghat is group-local (a
shard-invariant advantage estimator), the loss is a sum of row terms (constant normalizer, ratio forced to 1, no KL).

What the checks cover: p (every trained token's log-prob is theta_target's) is checked bitwise by the learner sentinel
every step and by the CHECKS rescore audit. q is the engine's decode log-prob of the sampled token, the same value the
exact-SPR T1 / P1 checks pin to the engine prefill and the learner bitwise; no IS-SPR check recomputes q under the
version that sampled it (that version's weights are gone after the wake).

Row weights from the learner (11-06, lead-approved; next/isspr-learner-s.md). With the row unit, s_r needs no engine
rescore: the learner's training forward at theta_target computes log p_target of every trained token, bitwise the
engine's under batch invariance, and w_r = min(C, e^{s_r}) depends on row r alone, so the loss forms it from its own
detached log-probs (``is_old_tokens``; loss_functions.py). The engine then re-prefills only rows that continue (their
KV / SSM state, a plain resume with no check); a row that finished under older weights is not touched again, and its
generation log-probs stay q. The learner sentinel covers the tokens sampled under theta_target (each row's suffix after
its ``old_tokens``, from the version labels and the engine's per-request wake marks, WAKE_AT_ATTR), so the stale-KV
negative still fails there. The group unit (K <= 1 under auto) keeps the engine rescore: W_g needs every row of the
group before any row's weighted loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal, Optional, Sequence

import numpy as np

from thundersync_rl import xspr

MODE = "isspr"
IS_WEIGHT_FIELD = "is_weight"
IS_OLD_FIELD = "is_old_tokens"  # learner s: each row's leading tokens not sampled under the trained weights
IS_AUDIT_FIELD = "is_audit_logprobs"  # learner s, CHECKS: the engine's from-scratch rescore of every token
NEG_STALE_KV = "isspr_stale_kv"  # no wake cut / rescore: paused rows resume on stale state, generation lps = q
NEG_STALE_CACHE = "isspr_stale_cache"  # rescore requests hit the previous weight epoch's prefix-cache KV
NEG_NO_PREFILL_CUT = "isspr_no_prefill_cut"  # the wake cut skips requests part-way through a chunked prefill
NEGATIVES = (NEG_STALE_KV, NEG_STALE_CACHE, NEG_NO_PREFILL_CUT)
# ExactRows._segment check mode: rescore the row's tokens and keep decoding from the recomputed state, whatever the
# keyed check finds (one prefill; IS-SPR keeps every token).
RESCORE_RESUME = "continue_always"


@dataclass
class IsSprConfig:
    """``thundersync:`` block of an SC config in isspr mode."""

    mode: Literal["isspr"] = MODE
    seed_stream: str = "isspr"
    # The importance-weighted unit: "group" (W_g = min(C, e^{s_g}); the on-policy gradient of any group statistic up to
    # the logged clip), "row" (IS-RLOO: w_r = min(C, e^{s_r}) with the RLOO objective the validator pins) or "auto"
    # (IS-RLOO at lag K: group at K <= 1, row at K >= 2, same objective). Module docstring.
    weight: Literal["row", "group", "auto"] = "group"
    # Row loss normalizer: "row" (stock per-row token mean, 1 / l_r) or "constant" (1 / L_c, L_c =
    # policy.generation.max_new_tokens; folded into the is_weight column). Row weights require "constant".
    length_normalizer: Literal["row", "constant"] = "row"
    # Recompute every row's KV / SSM state under theta_n at each wake (cut and resume). With row weights from the
    # learner (unit row) it is not needed for unbiasedness: w_r = prod p_target / q with q the law each token was
    # actually drawn with, stale state or not; it only makes the post-wake tokens' log ratio 0. False: the engine
    # runs as SPR-BI's (no wake cut, no resume prefill). Group unit: required.
    wake_recompute: bool = True
    # The clip C (TAIL_FLOOR §5.5: 5).
    clip_c: float = 5.0
    # Rescore positions whose logits exist at once (each [V] fp32).
    check_chunk_rows: int = 256
    # Cross-fade routing (LEVERS.md; exact SPR's / nolanes' keys and rule, nolanes.crossfade_problems): learner and
    # engine route identically, so it narrows version divergence without touching exactness. m None needs a reason.
    crossfade_routing_m: Optional[float] = None
    crossfade_routing_m_profile: Any = None
    crossfade_routing_extra: int = 8
    crossfade_routing_ramp_stopgrad: bool = True
    crossfade_routing_sparse: bool = True
    crossfade_routing_sparse_engine: bool = True
    crossfade_off_reason: Optional[str] = None
    # Exact SPR's fork dump (XsprStage reads it at init): IS-SPR keeps every token, so it has no forks. Not a key.
    row_dump_dir: ClassVar[None] = None
    # Resolved from the rest of the config by from_config (not keys): the sampler's lag K and L_c.
    lag: int = field(default=0, init=False)
    max_new_tokens: int = field(default=0, init=False)

    @property
    def unit(self) -> str:
        """The weighted unit actually used: ``auto`` resolves to group at lag <= 1 and row at lag >= 2."""
        if self.weight == "auto":
            return "row" if self.lag >= 2 else "group"
        return self.weight

    @classmethod
    def from_config(cls, config: Any) -> "IsSprConfig":
        block = xspr._get(config, "thundersync") or {}
        block = dict(block)
        known = {f.name for f in fields(cls) if f.init}
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"thundersync (mode=isspr): unknown keys {unknown} (known: {sorted(known)})")
        out = cls(**block)
        if out.mode != MODE:
            raise ValueError(f"thundersync.mode must be {MODE!r} (got {out.mode!r})")
        if out.weight not in ("row", "group", "auto"):
            raise ValueError(f"thundersync.weight must be 'row', 'group' or 'auto' (got {out.weight!r})")
        if out.length_normalizer not in ("row", "constant"):
            raise ValueError(
                f"thundersync.length_normalizer must be 'row' or 'constant' (got {out.length_normalizer!r})")
        out.lag = _lag(config)
        out.max_new_tokens = int(xspr._get(config, "policy.generation.max_new_tokens", 0) or 0)
        if not float(out.clip_c) > 1.0:
            raise ValueError("thundersync.clip_c must be > 1")
        if int(out.check_chunk_rows) < 1:
            raise ValueError("thundersync.check_chunk_rows must be >= 1")
        return out


def config_problems(config: Any) -> list[str]:
    """Everything exact SPR's recompute layer needs, plus the loss the weight is unbiased in."""
    p = xspr.config_problems(config)
    loss = xspr._get(config, "loss_fn") or {}
    if xspr._get(loss, "token_level_loss", True):
        p.append("loss_fn.token_level_loss must be false (constant normalizer: row mean, then mean over the rows)")
    if not xspr._get(loss, "force_on_policy_ratio", False):
        p.append("loss_fn.force_on_policy_ratio must be true (one update per batch: no PPO clip on W_g A)")
    if float(xspr._get(loss, "reference_policy_kl_penalty", 0.0) or 0.0) != 0.0:
        p.append("loss_fn.reference_policy_kl_penalty must be 0 (W_g scales the advantages only)")
    if xspr._get(loss, "truncated_importance_sampling_ratio") is not None:
        p.append("loss_fn.truncated_importance_sampling_ratio must be null (W_g is the only off-policy weight)")
    if xspr._get(config, "grpo.overlong_filtering", False):
        p.append("grpo.overlong_filtering must be false (a dropped row changes the group statistic)")
    p += estimator_problems(config)
    gen = xspr._get(config, "policy.generation") or {}
    if float(xspr._get(gen, "temperature", 1.0)) != 1.0:
        p.append("policy.generation.temperature must be 1 (q and p are raw log-probs: the law the engine samples from "
                 "only at temperature 1)")
    if int(xspr._get(gen, "mcore_generation_config.num_speculative_tokens", 0) or 0) != 0:
        p.append("policy.generation.mcore_generation_config.num_speculative_tokens must be 0 (q is the law of one "
                 "sampled token)")
    return p


def _lag(config: Any) -> int:
    return int(xspr._get(config, "async_rl.sampler.max_lookahead_versions", 0) or 0)


def estimator_problems(config: Any) -> list[str]:
    """IS-RLOO's objective (module docstring): with row weights possible (weight row / auto) the advantage is RLOO
    (LOO mean, no std) and the row normalizer is a constant; a constant normalizer needs L_c."""
    ts = xspr._get(config, "thundersync") or {}
    weight = xspr._get(ts, "weight", "group")
    norm = xspr._get(ts, "length_normalizer", "row")
    p: list[str] = []
    if weight in ("row", "auto"):
        why = f"thundersync.weight {weight} (row weights are exact only for RLOO with a constant normalizer)"
        adv = xspr._get(config, "grpo.adv_estimator") or {}
        name = xspr._get(adv, "name", "grpo")
        if name != "grpo":
            p.append(f"grpo.adv_estimator.name must be grpo (got {name!r}): {why}")
        # the estimator reads only this block (AdvEstimatorConfig defaults: both true), never the flat grpo.* keys
        loo = xspr._get(adv, "use_leave_one_out_baseline", True)
        std = xspr._get(adv, "normalize_rewards", True)
        if not loo:
            p.append(f"grpo.adv_estimator.use_leave_one_out_baseline must be true: {why}")
        if std:
            p.append(f"grpo.adv_estimator.normalize_rewards must be false (GRPO's std dropped): {why}")
        if norm != "constant":
            p.append(f"thundersync.length_normalizer must be constant: {why}")
    row_unit = weight == "row" or (weight == "auto" and _lag(config) >= 2)
    if not xspr._get(ts, "wake_recompute", True) and not row_unit:
        p.append("thundersync.wake_recompute false needs row weights from the learner (weight row, or auto at lag >= 2): "
                 "the group unit's engine rescore reads theta_target state")
    cp = int(xspr._get(config, "policy.megatron_cfg.context_parallel_size", 1) or 1)
    if row_unit and cp > 1:
        # Row weights are formed in the loss from each row's log-probs (loss_functions.is_row_weights): the
        # Megatron loss path all-gathers them over CP (full rows on every rank), but no CP = 2 run has shown s_r
        # bitwise equal to CP = 1 yet (REVIEW batch 50 L1).
        p.append(f"row weights (thundersync.weight {weight}) need policy.megatron_cfg.context_parallel_size 1 "
                 f"(got {cp}) until a CP = 2 run shows the learner's s_r bitwise equal to CP = 1")
    if norm == "constant" and int(xspr._get(config, "policy.generation.max_new_tokens", 0) or 0) <= 0:
        p.append("policy.generation.max_new_tokens must be set (> 0): it is L_c of length_normalizer constant")
    return p


def train_coefficients(w: Sequence[float], lengths: Sequence[int], cfg: "IsSprConfig") -> list[float]:
    """The is_weight column: the clipped weight w_r of each trained row, times l_r / L_c under the constant normalizer
    (the loss divides row r by its token count l_r; this turns it into 1 / L_c). l_r must be in [1, L_c]."""
    if len(w) != len(lengths):
        raise RuntimeError(f"[isspr] {len(w)} weights for {len(lengths)} rows")
    if cfg.length_normalizer == "row":
        return [float(x) for x in w]
    lc = int(cfg.max_new_tokens)
    bad = [n for n in lengths if not 1 <= int(n) <= lc]
    if bad:
        raise RuntimeError(f"[isspr] trained token counts {bad[:5]} outside [1, L_c = {lc}]")
    return [float(x) * int(n) / lc for x, n in zip(w, lengths)]


def require_group_local(advantage_stage_config: Any) -> None:
    """W_g is unbiased for a statistic of group g's rows only: the advantage estimator must be group-local
    (shard-invariant, as the stage's sharding rule decides); a batch-level normalization or a cross-group baseline
    would mix in other groups' unweighted rows."""
    if advantage_stage_config.is_ppo or not advantage_stage_config.shardable:
        raise RuntimeError(f"[isspr] the advantage estimator {advantage_stage_config.algo.adv_estimator.name!r} is not "
                           "group-local (needs a shard-invariant GRPO estimator)")


def require_rloo(advantage_stage_config: Any, cfg: "IsSprConfig") -> None:
    """Row weights (weight row / auto) are exact only for RLOO: check the estimator the advantage stage was built
    from (the resolved AdvEstimatorConfig), not the YAML the validator saw."""
    if cfg.weight == "group":
        return
    est = advantage_stage_config.algo.adv_estimator
    if est.name != "grpo" or est.normalize_rewards or not est.use_leave_one_out_baseline:
        raise RuntimeError(f"[isspr] weight {cfg.weight} needs the RLOO advantage; the advantage stage runs {est.name!r} "
                           f"with normalize_rewards={est.normalize_rewards} "
                           f"use_leave_one_out_baseline={est.use_leave_one_out_baseline}")


def validate_config(config: Any) -> IsSprConfig:
    problems = config_problems(config)
    if problems:
        raise ValueError("thundersync.mode=isspr: " + "; ".join(problems) + ".")
    return IsSprConfig.from_config(config)


def configure_driver(config: dict) -> IsSprConfig:
    """Driver side, before setup_single_controller: validate, then exact SPR's worker setup (keyed sampler, engine
    rescore, train-lp sentinel)."""
    from thundersync_rl import nolanes

    cfg = validate_config(config)
    nolanes.derive_layout_fields(config)
    xspr.configure_worker(config, cfg)
    if cfg.unit == "row":
        # Row weights from the learner's forward (module docstring; loss_functions.is_row_weights).
        config.setdefault("loss_fn", {})["is_row_clip"] = float(cfg.clip_c)
    neg = nolanes.negative()
    lag = int(xspr._get(config, "async_rl.sampler.max_lookahead_versions", 0) or 0)
    print(f"[isspr] mode=isspr lag={lag} weight={cfg.weight} unit={cfg.unit} length_normalizer={cfg.length_normalizer} "
          f"L_c={cfg.max_new_tokens} clip_c={cfg.clip_c} s={'learner' if cfg.unit == 'row' else 'engine'} "
          f"crossfade_m={cfg.crossfade_routing_m} "
          f"check_chunk_rows={cfg.check_chunk_rows} "
          f"checks={int(nolanes.checks_on())}" + (f" MUTATION {neg}" if neg else ""), flush=True)
    return cfg


# -- the weight ------------------------------------------------------------------------------------------------------


def row_log_ratio(p: Sequence[float], q: Sequence[float]) -> float:
    """sum_t (log p_t - log q_t) of one row, in float64 (exactly 0 where p == q). A sampled token's log-prob must be
    finite under both laws (a token sampled at log q = -inf, or a NaN, means the logged q is not the law it was drawn
    from): raises, naming the first such position."""
    if len(p) != len(q):
        raise RuntimeError(f"[isspr] {len(p)} rescored log-probs for {len(q)} sampled tokens")
    bad = next((t for t, (a, b) in enumerate(zip(p, q)) if not (math.isfinite(a) and math.isfinite(b))), None)
    if bad is not None:
        raise RuntimeError(f"[isspr] non-finite log-prob of a sampled token at position {bad} of {len(p)}: "
                           f"log p {p[bad]!r}, log q {q[bad]!r}")
    return math.fsum(float(a) - float(b) for a, b in zip(p, q))


def clipped_weights(s: Sequence[float], clip_c: float) -> list[float]:
    """min(C, exp(s)) of each log ratio. A non-finite s raises (min(NaN, 700) is NaN's first argument: it would map
    to C silently)."""
    bad = [x for x in s if not math.isfinite(float(x))]
    if bad:
        raise RuntimeError(f"[isspr] non-finite log ratio {bad[0]!r} ({len(bad)} of {len(s)})")
    return [min(float(clip_c), math.exp(min(float(x), 700.0))) for x in s]


def weight_stats(s: Sequence[float], clip_c: float) -> dict[str, float]:
    """Per-step log of the weights: eps_hat = mean_g (e^{s_g} - C)^+ (unbiased estimate of the clip's bias factor),
    mean / var of s_g, the normalized ESS (sum w)^2 / (n sum w^2) of the raw weights e^{s_g} (theory's number,
    TAIL_FLOOR §5.5) and of the clipped ones, the min / max s_g and the clipped-group count."""
    n = len(s)
    if n == 0:
        raise RuntimeError("[isspr] no groups")
    xs = [float(x) for x in s]
    mean = math.fsum(xs) / n
    var = math.fsum((x - mean) ** 2 for x in xs) / n
    clipped = clipped_weights(xs, clip_c)
    excess = [max(0.0, math.exp(min(x, 700.0)) - clip_c) for x in xs]
    return {
        "groups": float(n),
        "s_mean": mean,
        "s_var": var,
        "s_min": min(xs),
        "s_max": max(xs),
        "ess": _ess(xs),
        "ess_clip": _ess([min(x, math.log(clip_c)) for x in xs]),
        "eps_hat": math.fsum(excess) / n,
        "clipped": float(sum(1 for x in xs if math.exp(min(x, 700.0)) > clip_c)),
        "w_mean": math.fsum(clipped) / n,
    }


def _ess(log_w: list[float]) -> float:
    """(sum w)^2 / (n sum w^2) of the weights exp(log_w), shifted by the max (scale-free; no underflow to 0 / 0)."""
    top = max(log_w)
    w = [math.exp(x - top) for x in log_w]
    return math.fsum(w) ** 2 / (len(w) * math.fsum(x * x for x in w))


def age_stats(target: int, certs: Sequence["IsCert"]) -> dict[int, tuple[int, float]]:
    """Per token age a = target - (first version label): {a: (rows, KL per rescored token)}, the KL estimated by
    -sum(s) / sum(rescored tokens) (E_q[s] = -KL(q || p) of the row's pre-wake tokens)."""
    out: dict[int, tuple[int, float]] = {}
    for a in sorted({int(target) - c.first_version for c in certs}):
        cs = [c for c in certs if int(target) - c.first_version == a]
        n = sum(c.rescored_tokens for c in cs)
        out[a] = (len(cs), -math.fsum(c.s for c in cs) / n if n else 0.0)
    return out


def reward_shift(s_rows: Sequence[float], rewards: Sequence[float]) -> float:
    """dp_hat = |mean_r (e^{s_r} - 1) R_r|: E_q[(w - 1) R] = E_p[R] - E_q[R], the pooled reward shift the RIS prompt
    scale error is O() of (FIRSTPRINCIPLES_C §3.2)."""
    if len(s_rows) != len(rewards) or not s_rows:
        raise RuntimeError(f"[isspr] {len(s_rows)} row log ratios for {len(rewards)} rewards")
    return abs(math.fsum((math.exp(min(float(x), 700.0)) - 1.0) * float(r) for x, r in zip(s_rows, rewards))
               / len(s_rows))


TOKEN_QUANTILES = (0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)


def token_stats(log_ratios: Sequence[Any]) -> dict[str, Any]:
    """Distribution of the per-token log ratios r = log p - log q of the step's rescored tokens: quantiles of r, and the
    share of sum |r| and of sum r^2 carried by the top 0.1 % / 1 % of tokens by |r| (how concentrated the divergence
    is: KL >> TV only when a few tokens carry large |r|)."""
    r = np.concatenate([np.asarray(x, dtype=np.float64) for x in log_ratios]) if log_ratios else np.zeros(0)
    if r.size == 0:
        return {"tokens": 0}
    a = np.sort(np.abs(r))[::-1]
    out: dict[str, Any] = {"tokens": int(r.size), "q": np.quantile(r, TOKEN_QUANTILES).tolist()}
    for frac in (0.001, 0.01):
        k = max(1, int(round(frac * r.size)))
        out[f"abs_top{frac:g}"] = float(a[:k].sum() / a.sum()) if a.sum() > 0 else 0.0
        out[f"sq_top{frac:g}"] = float((a[:k] ** 2).sum() / (a ** 2).sum()) if a.sum() > 0 else 0.0
    return out


# -- the row loop ----------------------------------------------------------------------------------------------------


@dataclass
class IsCert:
    """A finished row: the version its log-probs are theta's of, its log ratio, what the recompute cost."""

    version: int
    key: int
    tokens: int
    s: float = 0.0
    rescores: int = 0
    rescored_tokens: int = 0  # tokens of the last rescore (sampled before the certifying wake)
    first_version: int = -1  # version label of the row's first request: its oldest tokens are at most this old
    log_ratios: Any = field(default=None, repr=False)  # float64 [rescored_tokens]: p - q of the rescored tokens
    prefill_tokens: int = 0  # rescore + resume prefill tokens over the row's life
    only_tokens: int = 0  # tokens re-prefilled only to rescore a finished row ("only" checks)
    resume_tokens: int = 0  # tokens re-prefilled to resume a cut row (its KV / SSM state)
    old_tokens: int = 0  # leading tokens not sampled under theta_target (learner s: outside the sentinel)


def _old_after(old: int, start: int, end: int, label: int, info: dict, target: int, recompute: bool = True) -> int:
    """The row's leading tokens not sampled under theta_target after a segment that sampled tokens [start, end) from
    version ``label`` (conservative: at most the version that ran it). The engine marked the segment's generated
    count at every wake it was in flight across (``wake_at``; one version per wake): tokens after the last mark are
    version label + marks'. A segment that ends below theta_target leaves every token so far old."""
    marks = info.get("wake_at") or []
    last = int(label) + len(marks)
    if last > int(target):
        raise RuntimeError(f"[isspr] a segment of a batch-{target} row ran under version {last}")
    if last < int(target) or (marks and not recompute):
        return end  # below theta_target, or (no wake recompute) on state older than theta_target after its wake
    return max(old, start + max(marks)) if marks else old


class IsRows(xspr.ExactRows):
    """The row loop of IS-SPR (AsyncRolloutImpl._run_single_rollout calls ``respond`` for each row). Exact SPR's
    engine primitives (``_segment``: decode / rescore requests with the row's key), without its keyed check."""

    # Row unit: s_r from the learner's forward (module docstring); set by the stage from IsSprConfig.unit.
    learner_s = False
    # IsSprConfig.wake_recompute (set by the stage): False marks a segment's post-wake tokens as old too.
    wake_recompute = True

    async def respond(self, impl, message_log, stop_strings, native, input_sample, traj_idx: int):
        """One row of batch ``target``, its log-probs theta_target's, its s recorded for the group weight."""
        import torch

        from nemo_rl.experience.rollout_manager import ROLLOUT_TARGET_STEP

        if stop_strings:
            raise RuntimeError("[isspr] stop strings are not supported (validator)")
        target = ROLLOUT_TARGET_STEP.get()
        if target is None:
            raise RuntimeError("[isspr] a row without a target step (in_order sampler)")
        idx = xspr._plain(input_sample["idx"])
        key = self.key_of(target, idx, traj_idx)
        self.ledger.open(key)
        cert = IsCert(version=-1, key=key, tokens=0)
        msg, in_len, metrics, info, label = await self._segment(impl, message_log, native, key, [], [])
        cert.first_version = int(label)
        tokens = [int(t) for t in msg["token_ids"].tolist()]
        q = [float(x) for x in msg["generation_logprobs"].tolist()]
        p = list(q)
        old = _old_after(0, 0, len(tokens), label, info, target, self.wake_recompute)
        self.ledger.draw(key, 0, len(tokens))
        cut = bool(info["cut"])
        stale = self.neg == NEG_STALE_KV
        while (cut or label != target) and not stale:
            if not cut:
                if self.learner_s:
                    break  # finished under older weights: its log p_target come from the learner's forward
                # Finished under older weights (D1): rescored once theta_target is installed.
                await self.wait_version(target)
            n = len(tokens)
            if not (cut or n):
                label = target  # nothing sampled before the wake
                continue
            # One request: a cut row is rescored and resumed by the same prefill (its state is then the new
            # weights'); a finished row is only rescored. A row cut before its first token just decodes. Learner s:
            # a cut row is resumed by a plain prefill (no check: its s is the learner's).
            mode = "" if self.learner_s else (RESCORE_RESUME if cut else "only") if n else ""
            res, in_len, metrics, info, checked = await self._segment(impl, message_log, native, key, tokens, p,
                                                                        check=mode)
            if mode and n and info["check"] is None:
                continue  # cut mid-prefill by a wake (chunked prefill): nothing rescored, redone under the new weights
            label = checked
            if n:
                if mode:
                    p = [float(x) for x in info["check"]["lps"]]
                    if len(p) != n:
                        raise RuntimeError(f"[isspr] rescore of key {key}: {len(p)} log-probs for {n} tokens")
                    cert.rescores += 1
                    cert.rescored_tokens = n
                if cut:
                    cert.resume_tokens += n
                else:
                    cert.only_tokens += n
                cert.prefill_tokens += int(in_len) + n
            if not cut:
                continue
            msg = res
            new = [int(t) for t in msg["token_ids"].tolist()]
            if new[:n] != tokens:
                raise RuntimeError(f"[isspr] key {key}: the resumed row does not start with its {n} kept tokens")
            new_lps = [float(x) for x in msg["generation_logprobs"].tolist()]
            q += new_lps[n:]
            p += new_lps[n:]
            old = _old_after(old, n, len(new), label, info, target, self.wake_recompute)
            self.ledger.draw(key, n, len(new))
            tokens = new
            cut = bool(info["cut"])
        self.ledger.close(key)
        cert.old_tokens = old
        if self.learner_s:
            # q stays the row's generation log-probs; s_r, w_r and their log are the learner's (loss metrics).
            cert.version, cert.tokens, cert.s = int(target), len(tokens), float("nan")
            self.certs[(int(target), idx, int(traj_idx))] = cert
            out = dict(msg)
            out["token_ids"] = torch.tensor(tokens, dtype=msg["token_ids"].dtype)
            out["generation_logprobs"] = torch.tensor(q, dtype=torch.float32)
            out["content"] = impl._tokenizer.decode(out["token_ids"], skip_special_tokens=True)
            return out, in_len, metrics
        try:
            cert.s = row_log_ratio(p, q)
        except RuntimeError as e:
            path = self.dump_row(key, "nonfinite_ratio",
                                 [int(t) for m in message_log for t in m["token_ids"]], tokens, {"p": p, "q": q},
                                 target=int(target), prompt_idx=str(idx), sample=int(traj_idx))
            raise RuntimeError(f"{e}; row key {key}, target {target}, prompt {idx}, sample {traj_idx}, "
                               f"{cert.rescored_tokens} rescored tokens, {cert.rescores} rescores; row dumped to "
                               f"{path or '(no dump dir)'}") from e
        cert.log_ratios = np.asarray(p[: cert.rescored_tokens], dtype=np.float64) - np.asarray(
            q[: cert.rescored_tokens], dtype=np.float64)
        cert.version, cert.tokens = int(label), len(tokens)
        self.certs[(int(target), idx, int(traj_idx))] = cert
        out = dict(msg)
        out["token_ids"] = torch.tensor(tokens, dtype=msg["token_ids"].dtype)
        out["generation_logprobs"] = torch.tensor(p, dtype=torch.float32)
        out["content"] = impl._tokenizer.decode(out["token_ids"], skip_special_tokens=True)
        return out, in_len, metrics


def negative() -> str:
    from thundersync_rl import nolanes

    v = nolanes.negative()
    return v if v in NEGATIVES else ""
