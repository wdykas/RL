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
   Either way minus the clip's share, E_q[(e^s - C)^+ f], estimated without bias by eps_hat = mean (e^s - C)^+ and
   logged every step with mean / var of s and the ESS. Not exact in law.
Conditions (validated): q is the sampling law (temperature 1, one token per step), ghat is group-local (a
shard-invariant advantage estimator), the loss is a sum of row terms (constant normalizer, ratio forced to 1, no KL).

What the checks cover: p (every trained token's log-prob is theta_target's) is checked bitwise by the learner sentinel
every step and by the CHECKS rescore audit. q is the engine's decode log-prob of the sampled token, the same value the
exact-SPR T1 / P1 checks pin to the engine prefill and the learner bitwise; no IS-SPR check recomputes q under the
version that sampled it (that version's weights are gone after the wake).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar, Literal, Optional, Sequence

import numpy as np

from thundersync_rl import xspr

MODE = "isspr"
IS_WEIGHT_FIELD = "is_weight"
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
    # The importance-weighted unit: "group" (W_g = min(C, e^{s_g}); the approved estimator: the on-policy GRPO
    # gradient up to the logged clip) or "row" (RIS, opt-in: w_r = min(C, e^{s_r}); estimand in the module docstring).
    weight: Literal["row", "group"] = "group"
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

    @classmethod
    def from_config(cls, config: Any) -> "IsSprConfig":
        block = xspr._get(config, "thundersync") or {}
        block = dict(block)
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(block) - known)
        if unknown:
            raise ValueError(f"thundersync (mode=isspr): unknown keys {unknown} (known: {sorted(known)})")
        out = cls(**block)
        if out.mode != MODE:
            raise ValueError(f"thundersync.mode must be {MODE!r} (got {out.mode!r})")
        if out.weight not in ("row", "group"):
            raise ValueError(f"thundersync.weight must be 'row' or 'group' (got {out.weight!r})")
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
    gen = xspr._get(config, "policy.generation") or {}
    if float(xspr._get(gen, "temperature", 1.0)) != 1.0:
        p.append("policy.generation.temperature must be 1 (q and p are raw log-probs: the law the engine samples from "
                 "only at temperature 1)")
    if int(xspr._get(gen, "mcore_generation_config.num_speculative_tokens", 0) or 0) != 0:
        p.append("policy.generation.mcore_generation_config.num_speculative_tokens must be 0 (q is the law of one "
                 "sampled token)")
    return p


def require_group_local(advantage_stage_config: Any) -> None:
    """W_g is unbiased for a statistic of group g's rows only: the advantage estimator must be group-local
    (shard-invariant, as the stage's sharding rule decides); a batch-level normalization or a cross-group baseline
    would mix in other groups' unweighted rows."""
    if advantage_stage_config.is_ppo or not advantage_stage_config.shardable:
        raise RuntimeError(f"[isspr] the advantage estimator {advantage_stage_config.algo.adv_estimator.name!r} is not "
                           "group-local (needs a shard-invariant GRPO estimator)")


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
    neg = nolanes.negative()
    lag = int(xspr._get(config, "async_rl.sampler.max_lookahead_versions", 0) or 0)
    print(f"[isspr] mode=isspr lag={lag} weight={cfg.weight} clip_c={cfg.clip_c} crossfade_m={cfg.crossfade_routing_m} "
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
    """min(C, exp(s)) of each log ratio."""
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


class IsRows(xspr.ExactRows):
    """The row loop of IS-SPR (AsyncRolloutImpl._run_single_rollout calls ``respond`` for each row). Exact SPR's
    engine primitives (``_segment``: decode / rescore requests with the row's key), without its keyed check."""

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
        self.ledger.draw(key, 0, len(tokens))
        cut = bool(info["cut"])
        stale = self.neg == NEG_STALE_KV
        while (cut or label != target) and not stale:
            if not cut:
                # Finished under older weights (D1): rescored once theta_target is installed.
                await self.wait_version(target)
            n = len(tokens)
            if not (cut or n):
                label = target  # nothing sampled before the wake
                continue
            # One request: a cut row is rescored and resumed by the same prefill (its state is then the new
            # weights'); a finished row is only rescored. A row cut before its first token just decodes.
            mode = (RESCORE_RESUME if cut else "only") if n else ""
            res, in_len, metrics, info, checked = await self._segment(impl, message_log, native, key, tokens, p,
                                                                        check=mode)
            if n and info["check"] is None:
                continue  # cut mid-prefill by a wake (chunked prefill): nothing rescored, redone under the new weights
            label = checked
            if n:
                p = [float(x) for x in info["check"]["lps"]]
                if len(p) != n:
                    raise RuntimeError(f"[isspr] rescore of key {key}: {len(p)} log-probs for {n} tokens")
                cert.rescores += 1
                cert.rescored_tokens = n
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
            self.ledger.draw(key, n, len(new))
            tokens = new
            cut = bool(info["cut"])
        self.ledger.close(key)
        try:
            cert.s = row_log_ratio(p, q)
        except RuntimeError as e:
            raise RuntimeError(f"{e}; row key {key}, target {target}, prompt {idx}, sample {traj_idx}, "
                               f"{cert.rescored_tokens} rescored tokens, {cert.rescores} rescores") from e
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
