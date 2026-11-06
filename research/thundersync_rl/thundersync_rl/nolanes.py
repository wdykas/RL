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
"""ThunderSync without lanes (``thundersync.mode=nolanes``; docs/colo/THEORY.md).

Stock NeMo-RL async GRPO (the SingleController loop: in_order sampler with lag k, one generation fleet refitted to
theta_k at every step in ``persist`` mode, requests kept in flight across the refit) plus three things:

1. theta-only MoE routing (cross-fade, ``nemo_rl/models/megatron/crossfade_routing.py``);
2. the engine logs, per sampled token, the weight version of the forward it was sampled from and the LM-head input
   (``worker.nl_engine_setup``), so the proposal q of every token is known bitwise (E1);
3. before training step k, every rollout of batch k is block-verified (⊥-BV) on the learner against theta_k
   (``VerifyStage``); rejected rows and unfilled slots are finished by sampling from theta_k on the same engine, as
   NEW requests whose prefix is re-prefilled at theta_k (E8).

Exactness conditions (DESIGN §1 E1-E6, review R23 E7-E8) and where each is enforced:
- E1 q is the sampled law: engine log q + head input + the version tag of the FORWARD (worker); engine-q check.
- E2 the target is pi_{theta_k} alone: validator refuses sticky routing / router replay; cross-fade or dense.
- E3 verification options depend on counts only: ``_NullPoolRun`` (counts), drafting order = sample index.
- E4 / R4 fresh keys: continuations are new engine requests (new request id); the engine refuses a reused
  (request id, position) noise key (``KeyRegistry``); verification coins come from the verifier's own seeds.
- E5 target tokens bitwise: batch-invariant engine == learner (validator + mismatch / train-lp checks).
- E6 / E9 provenance: only verified output rows reach the loss, advantages, critic or any controller
  (``check_provenance``; validator refuses consumers of draft statistics).
- E7 generation_logprobs of every output token = the verified p (accepted draft tokens, residual / bonus token) or
  the theta_k engine log-prob (continuations, == p under BI): the stock IS correction sees ratio 1.
- E8 continuations re-prefill at theta_k: new requests; prefix caching only on an engine whose prefix cache is scoped
  by weight version (``mcore_prefix_cache_weight_scoped``).
- R2 theta_c never depends on cohort c's drafts: one chunk per step, rows in a fixed (dispatch) order.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

MODE = "nolanes"
WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"
# Row tag of a verified output row (E6 / E9 provenance): the target version it was verified against.
VERIFIED_TAG = "nl_verified_version"
# Assistant-message / output key of the engine's draft reference ({"key": hidden key, "rid": request id}).
DRAFT_REF = "ts_draft"

# Negative controls (THUNDERSYNC_NL_NEG; each must FAIL a check, research/nolanes2 X1-neg).
NEGATIVES = (
    "wrong_tag",  # E1: every token's version tag = the dispatch version (mixed rows rebuilt with the wrong head)
    "tag_off_by_one",  # E1c: tokens of the first engine steps after a refit keep the previous version's tag
    "content_m",  # E3: the envelope M of a draft chosen from its own first token
    "key_reuse",  # E4: a continuation reuses the rejected draft's request id as its noise key
    "engine_q_genlp",  # E7: accepted draft tokens keep the engine q as their generation log-prob
    "cf_engine_m",  # E5/E2: the engine routes with twice the learner's cross-fade band (target tokens off p)
    "carry_unverified",  # colocated partial rollout: carried (stale-version) draft tokens trained without verification
    "spare_by_content",  # XPR-S: the trained prompts are the P pools with the fewest continued rows (selection on content)
    "spare_tail_drop",  # XPR-S: an output that needs a tail does not fill a slot (the next draft does), no IS weight
    "spare_by_logp",  # XPR-S: the trained prompts are the P pools whose drafts have the highest mean log-prob (content)
    "vg_by_logp",  # VG: each group drops its non-⊥ output with the lowest first-token log-prob (selection on content)
    # exact SPR mutations (thundersync_rl/xspr.py NEGATIVES; each must fail T1 or T3)
    "xspr_skip_check",
    "xspr_new_only",
    "xspr_skip_finished",
    "xspr_fresh_keys",
    "xspr_keep_lp",
    "xspr_state_prefix_end",
    "xspr_no_wake_cut",
    # IS-SPR mutation (thundersync_rl/isspr.py NEGATIVES; must fail the sentinel and the rescore audit)
    "isspr_stale_kv",
    "isspr_stale_cache",
    "isspr_no_prefill_cut",
)


def negative() -> str:
    """The active negative control ('' = none); unknown names are refused."""
    v = os.environ.get("THUNDERSYNC_NL_NEG", "").strip()
    if v and v not in NEGATIVES:
        raise ValueError(f"THUNDERSYNC_NL_NEG={v!r}: unknown negative control (known: {NEGATIVES})")
    return v


def checks_on() -> bool:
    """CHECKS=1 exactness checks of the verify stage (law / mismatch forward of every output row)."""
    return os.environ.get("THUNDERSYNC_NL_CHECKS", "0") == "1"


def _get(d: Any, path: str, default: Any = None) -> Any:
    cur = d
    for part in path.split("."):
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(part, None)
        else:
            cur = getattr(cur, part, None)
    return default if cur is None else cur


@dataclass
class NolanesConfig:
    """``thundersync:`` block of an SC config in nolanes mode (every key optional except ``mode``)."""

    mode: Literal["nolanes"] = "nolanes"
    # Drafts per output: N = max(G, round(K G)) drafts per prompt (null_bv.drafts_per_prompt).
    k_drafts: float = 1.0
    # theta-only cross-fade routing (None: dense model / stock top-k, refused for MoE by validate_config's caller).
    # Base band s and the per-layer profile (routing_crux, full-depth Nano 10-30): m = s x scale over the MoE layers'
    # relative depth; 0.3 for the first 6/23, 1.0 to 18/23, 0.5 after (2.3x sticky's v, ~1 extra expert / token-layer).
    crossfade_routing_m: Optional[float] = 0.03
    crossfade_routing_m_profile: Optional[list] = field(
        default_factory=lambda: [[0.0, 0.2608, 0.3], [0.2608, 0.7826, 1.0], [0.7826, 1.0, 0.5]]
    )
    crossfade_routing_extra: int = 4
    crossfade_routing_ramp_stopgrad: bool = True
    crossfade_routing_sparse: bool = False
    # Engine expert GEMMs skip zero-weight cross-fade slots (routing_crux K-RC1 prototype; same bits).
    crossfade_routing_sparse_engine: bool = True
    # Running without cross-fade (crossfade_routing_m null) needs a stated reason (crossfade_problems).
    crossfade_off_reason: Optional[str] = None
    # Share of verification on the learner (ES "x"); x < 1 (engine prompt-logprob prefill share) is not built yet.
    verify_split_x: float = 1.0
    # Continuations first on one engine replica (B2). Engine DP 1 needs nothing; DP > 1 pinning is not built yet.
    tail_priority: bool = True
    # Streaming continuation start: the step's pools are verified in this many calls (contiguous runs of the fixed R2
    # pool order, balanced by draft count) and each call's rejected / unfinished rows start decoding at theta_k on the
    # engine while the learner verifies the next call. Row seeds make every decision independent of the call (F3) and
    # the BI forward makes every p independent of the batch, so the outputs are those of one call; 1 = one call.
    verify_stream_chunks: int = 1
    # Padded tokens per learner verify forward. Packing only (BI forward: a row's logits do not depend on its batch).
    # None (the default) derives the rule: the training micro-batch, MBS x max_total_sequence_length (verify forwards
    # pack rows like training does, halving the per-forward fixed cost at Super MBS 2), capped by the MoE
    # device-dispatch capacity NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS when device dispatch is on (structural review L17).
    verify_batch_tokens: Optional[int] = None
    null_bv_logratio_var: float = 8e-7
    null_bv_fixed_m: Optional[float] = None
    null_bv_tail_weight: float = 1.0
    engine_sampler: Literal["wide"] = "wide"
    # F3: per-row verification seeds (a row's decision never depends on the call / rank it is verified in) and the
    # lazy ⊥-BV pool walk (waves of ceil(s * min(M, mult)) rows). Driver-side env of the reused verify0 code.
    row_seeds: bool = True
    # Reuse the verify pass as the prev-logprob pass: every output token's generation log-prob is theta_k's own
    # (verified p / BI engine decode), so prev_logprobs := generation_logprobs and the stock forward is skipped.
    # CHECKS=1 proves the equality on every token (nolanes mismatch line).
    prev_logprobs_from_verify: bool = True
    wide_sampler_key64: bool = True
    seed_stream: str = "nl0"
    # Spare prompts (XPR-S; research/colo/TAIL_FLOOR.md Theorem S): the step admits num_prompts_per_step = P + S prompt
    # groups, each pool walked by ⊥-BV at the constant envelope null_bv_fixed_m with prompt drop (a pool that can no
    # longer fill its G slots is dropped: a function of ⊥ counts only, whose law is the same for every prompt), and the
    # first P undropped pools in the fixed order train; the other groups are discarded unseen by the learner. The kept
    # groups are an independent thinning of the admitted ones, so each is G iid theta_k samples (null_bv.py, prompt
    # drop). 0 = off (every admitted group trains; a short pool decodes fresh slots).
    spare_prompts: int = 0
    # Verify at the wake (colocated; research/colo/MODEL.md v2 §13): at the stand-down that ends phase k the
    # lookahead drafts still decoding are cut where they stand (finished with their tokens so far, unfinished), so
    # batch k+1 is complete at the stand-down; after training, the stage verifies it against theta_{k+1} at the
    # start of phase k+1 and its continuations (rejected rows from tau, unfinished accepted rows from their end)
    # decode inside phase k+1 alongside the next batch's drafting: no separate continuation phase, no second wake.
    # The cut point is a stopping time of each draft (the stand-down time is a function of the previous batch's
    # rows and of the engine's iteration timing, never of the draft's later tokens), so block verification of the
    # cut prefix is exact (MODEL.md v2 §13; the async design's deadline cut, Corollary 1).
    wake_verify: bool = False
    # Variable group size (VG-1.4; research/colo/TAIL_FLOOR.md §4.5): every one of a group's N drafts is ⊥-BV-verified
    # at the constant envelope null_bv_fixed_m (walk_all: no M = 1 fallback, no stopping rule), the group is all its
    # non-⊥ outputs (size Bin(N, 1/M)), and a group with fewer than 2 outputs is dropped. Group sizes and drops are
    # functions of the ⊥ pattern alone (independent of the outputs' contents), so the trained rows are iid theta_k;
    # the GRPO baseline is per group over its own size (leave-one-out, keyed on the group id).
    variable_group: bool = False

    @classmethod
    def from_config(cls, config: Any) -> "NolanesConfig":
        raw = _get(config, "thundersync") or {}
        if not isinstance(raw, dict):
            raw = dict(raw)
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"thundersync (mode=nolanes): unknown keys {unknown} (known: {sorted(known)})")
        out = cls(**raw)
        if out.mode != MODE:
            raise ValueError(f"thundersync.mode must be {MODE!r} (got {out.mode!r})")
        return out


def apply_driver_env(cfg: "NolanesConfig") -> None:
    """The verify0 code reads the F3 row-seed switch from the driver's environment (speculative.row_seeds_enabled):
    set it from the config in the process that runs the verify stage. ⊥-BV waves are ceil(s x M) rows by rule
    (``null_wave_size``): the lazy-wave multiplier override is cleared."""
    os.environ["THUNDERSYNC_ROW_SEEDS"] = "1" if cfg.row_seeds else "0"
    os.environ.pop("THUNDERSYNC_NULL_WAVE_MULT", None)


def enabled(config: Any) -> bool:
    return _get(config, "thundersync.mode") == MODE


def drafts_per_prompt(cfg: NolanesConfig, g: int) -> int:
    from thundersync_rl.null_bv import drafts_per_prompt as _n

    return _n(float(cfg.k_drafts), int(g))


# V8b exact-numerics recipe (ab_base.yaml): learner and engine (the engine Policy is a deep copy of policy.*).
V8B_LEAVES = {
    "policy.megatron_cfg.forward_gemm_precision": "fp16",
    "policy.megatron_cfg.fp32_residual_connection": True,
    "policy.megatron_cfg.moe_router_dtype": "fp32",
    "policy.megatron_cfg.defer_fp32_logits": True,
    "policy.megatron_cfg.batch_invariant_mode": True,
}


def mcore_prefix_cache_weight_scoped() -> bool:
    """Whether the installed Megatron-Core scopes the prefix cache by weight version: ``resume()`` bumps a weight epoch
    salted into every block hash and re-salts every request that holds no KV yet
    (``DynamicInferenceEngine._resalt_unprefilled_requests``; an older engine with only ``_weight_scoped_salt`` kept a
    waiting request's old salt). False if Megatron-Core is not importable here (refuse rather than assume)."""
    try:
        from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine
    except ImportError:
        return False
    return callable(getattr(DynamicInferenceEngine, "_resalt_unprefilled_requests", None))


def config_problems(config: Any) -> list[str]:
    """Every reason the config is not a valid nolanes run (empty: valid). Pure function of the config dict."""
    p: list[str] = []
    try:
        cfg = NolanesConfig.from_config(config)
    except (TypeError, ValueError) as e:
        return [str(e)]
    pol = _get(config, "policy") or {}
    gen = _get(pol, "generation") or {}
    mcore = _get(gen, "mcore_generation_config") or {}
    mcfg = _get(pol, "megatron_cfg") or {}
    async_rl = _get(config, "async_rl") or {}
    grpo = _get(config, "grpo") or {}
    # Backend, layout, refit mode.
    if not _get(mcfg, "enabled", False):
        p.append("policy.megatron_cfg.enabled must be true (Megatron learner)")
    if _get(gen, "backend") != "megatron":
        p.append("policy.generation.backend must be megatron (the engine logs head inputs and version tags)")
    if _get(gen, "colocated.enabled", False):
        # Colocated exact partial rollout (docs/colo/THEORY.md, "XPR"): one fleet drafts and
        # trains. The in_order lag-K loop pauses the lookahead batches at every stand-down and resumes them after the
        # wake (a carried row = a draft whose tokens span versions); at its train step a batch is verified against
        # theta_k with the engine stood down, the engine is woken on the same weights for the continuations, and the
        # loop trains after them. Streamed verify chunks need continuations to start while the learner verifies,
        # which the stood-down colocated engine cannot.
        if int(cfg.verify_stream_chunks) != 1:
            p.append("thundersync.verify_stream_chunks must be 1 colocated (no continuation starts before the wake)")
        if _get(pol, "offload_optimizer_for_refit", True):
            p.append("policy.offload_optimizer_for_refit must be false colocated (resident learner: MODEL.md v2 §11)")
        if cfg.wake_verify:
            # The cut ends phase k only once batch k+1 is fully dispatched: with the loop's post-step wake deferred,
            # a group of batch k+1 that is not in flight at the stand-down would wait for a wake only the next
            # stage gives (a hang). So the lookahead batch must be admissible as a whole.
            nps = int(_get(grpo, "num_prompts_per_step", 0) or 0)
            lag = _get(async_rl, "sampler.max_lookahead_versions")
            if lag is None or int(lag) != 1:
                p.append(f"thundersync.wake_verify needs async_rl.sampler.max_lookahead_versions = 1 (got {lag})")
            for key in ("max_inflight_prompts", "max_buffered_rollouts"):
                v = _get(async_rl, key)
                if v is None or int(v) < 2 * max(nps, 1):
                    p.append(f"thundersync.wake_verify needs async_rl.{key} >= 2 x num_prompts_per_step "
                             f"({v} < {2 * nps}): the lookahead batch must be in flight whole at the cut")
    elif cfg.wake_verify:
        p.append("thundersync.wake_verify needs a colocated engine (policy.generation.colocated.enabled)")
    p += kv_mode_problems(mcore)
    if _get(async_rl, "recompute_kv_cache_after_weight_updates", True):
        p.append("async_rl.recompute_kv_cache_after_weight_updates must be false (persist refit)")
    # User rule 11-02: the MoE router (gate weights) and its expert bias are ALWAYS frozen. Hard error otherwise.
    p += router_frozen_problems(config)
    if int(cfg.spare_prompts) < 0:
        p.append("thundersync.spare_prompts must be >= 0")
    if int(cfg.spare_prompts) > 0:
        if cfg.null_bv_fixed_m is None or not float(cfg.null_bv_fixed_m) > 1.0:
            p.append("thundersync.spare_prompts needs a constant envelope thundersync.null_bv_fixed_m > 1 (prompt drop)")
        g = int(_get(grpo, "num_generations_per_prompt", 0) or 0)
        gbs = int(_get(pol, "train_global_batch_size", 0) or 0)
        n_adm = int(_get(grpo, "num_prompts_per_step", 0) or 0)
        if g and gbs and (n_adm - int(cfg.spare_prompts)) * g != gbs:
            p.append(f"thundersync.spare_prompts={cfg.spare_prompts}: grpo.num_prompts_per_step ({n_adm}) must be the "
                     f"trained prompts policy.train_global_batch_size / G ({gbs} / {g}) plus the spares")
        if int(cfg.verify_stream_chunks) != 1:
            p.append("thundersync.spare_prompts is built for one verify call per step (verify_stream_chunks 1)")
    if cfg.variable_group:
        if cfg.null_bv_fixed_m is None or not float(cfg.null_bv_fixed_m) > 1.0:
            p.append("thundersync.variable_group needs a constant envelope thundersync.null_bv_fixed_m > 1")
        if int(cfg.spare_prompts):
            p.append("thundersync.variable_group and spare_prompts are exclusive (VG trains every group of size >= 2)")
        if float(cfg.k_drafts) != 1.0:
            p.append("thundersync.variable_group: k_drafts must be 1 (N = grpo.num_generations_per_prompt drafts)")
        if int(cfg.verify_stream_chunks) != 1:
            p.append("thundersync.variable_group is built for one verify call per step (verify_stream_chunks 1)")
    # A continuation re-prefills prompt + kept prefix in ONE engine step (chunked prefill is off): the engine's token
    # budget must cover the longest sequence (DESIGN §18.3 lowered it to max_model_len at Super).
    mt, mseq = _get(mcore, "max_tokens"), _get(pol, "max_total_sequence_length")
    if (not _get(mcore, "enable_chunked_prefill", False) and mt is not None and mseq is not None
            and not (isinstance(mt, int) and isinstance(mseq, int))):
        p.append(f"policy.generation.mcore_generation_config.max_tokens ({mt!r}) and max_total_sequence_length "
                 f"({mseq!r}) must be resolved integers (R-structural L22)")
    elif (not _get(mcore, "enable_chunked_prefill", False) and isinstance(mt, int) and isinstance(mseq, int)
            and mt < mseq):
        p.append(f"policy.generation.mcore_generation_config.max_tokens={mt} < max_total_sequence_length={mseq} "
                 "(a continuation's re-prefill must fit one engine step without chunked prefill)")
    # E8: no prefix-cache hits across versions. Every weight install (refit, colocated wake) is an engine suspend /
    # resume; an engine that salts its prefix-cache block hashes with a weight epoch bumped at every resume (and
    # re-salts every request without KV at the resume) makes blocks of different versions unmatchable. Invariant
    # relied on: no target row (continuation / fresh row) spans a weight install -- target rows are new requests
    # launched after the wake / refit that installs theta_k, and VerifyStage.run returns batch k only after every one
    # of them finished, i.e. before training step k and the next install. So a target row's prefill can only match
    # blocks computed at theta_k. A draft spanning an install keeps its old salt (MCore) and its q is logged per
    # forward (E1).
    if _get(mcore, "enable_prefix_caching", False) and not mcore_prefix_cache_weight_scoped():
        p.append(
            "policy.generation.mcore_generation_config.enable_prefix_caching must be false (E8: a continuation "
            "must re-prefill its prefix at theta_k; the installed Megatron-Core does not scope the prefix cache "
            "by weight version)"
        )
    # Sampler, chunking, drops (R2, MEDIUM-4).
    sampler = _get(async_rl, "sampler") or {}
    if _get(sampler, "name", "in_order") != "in_order":
        p.append("async_rl.sampler.name must be in_order")
    if int(_get(sampler, "max_lookahead_versions", 1) or 0) < 1:
        p.append("async_rl.sampler.max_lookahead_versions must be >= 1")
    if _get(async_rl, "trajectory_streaming") is not None:
        p.append("async_rl.trajectory_streaming must be null (verification decides whole groups)")
    prompts = int(_get(grpo, "num_prompts_per_step", 0) or 0)
    if int(_get(async_rl, "min_groups_for_streaming_train", 0) or 0) != prompts:
        p.append(
            "async_rl.min_groups_for_streaming_train must equal grpo.num_prompts_per_step (R2: one chunk per step, "
            "trained in a fixed order, so theta_{k+1} never depends on engine timing)"
        )
    rf = _get(async_rl, "rollout_failure") or {}
    if int(_get(rf, "max_skipped_prompts", 0) or 0) != 0 or int(_get(rf, "max_consecutive_dropped_prompts", 0) or 0) != 0:
        p.append(
            "async_rl.rollout_failure.max_skipped_prompts / max_consecutive_dropped_prompts must be 0 (MEDIUM-4: "
            "every dispatched group is trained at its target step or the run fails)"
        )
    if int(_get(grpo, "max_rollout_turns", 1) or 1) != 1:
        p.append("grpo.max_rollout_turns must be 1 (multi-turn continuation is not built yet)")
    # E6 / E9: nothing but verified output rows may reach the learner or a controller.
    if _get(grpo, "use_dynamic_sampling", False):
        p.append("grpo.use_dynamic_sampling must be false (E6: it filters groups by draft rewards before verification)")
    adv = str(_get(config, "algorithm") or _get(grpo, "adv_estimator.name") or "")
    if "ppo" in adv.lower() or _get(config, "critic") is not None:
        p.append("PPO / a critic is not built in nolanes mode yet (E9 provenance hook is in place)")
    # E2: theta-only routing.
    if _get(pol, "router_replay.enabled", False):
        p.append("policy.router_replay.enabled must be false (E2: R3 replays the drafter's routes)")
    p += crossfade_problems(cfg, mcfg)
    # E5: batch-invariant numerics (V8b recipe) on learner and engine (one policy config for both).
    for leaf, want in V8B_LEAVES.items():
        got = _get(config, leaf)
        if got != want:
            p.append(f"{leaf} must be {want!r} (V8b exact-numerics recipe; got {got!r})")
    if _get(pol, "sequence_packing.enabled", False) and checks_on():
        p.append("policy.sequence_packing.enabled must be false under THUNDERSYNC_NL_CHECKS (law check)")
    # Sampler law: the wide keyed race samples the full softmax; filters fall back to the engine kernel.
    if _get(gen, "top_k") not in (None, -1, 0) or _get(gen, "top_p") not in (None, 1.0):
        p.append("policy.generation.top_k / top_p must be off (the wide sampler draws the full softmax)")
    # R29 refusals (until each is built and reviewed).
    temp = _get(gen, "temperature", 1.0)
    if temp is not None and float(temp) != 1.0:
        p.append("policy.generation.temperature must be 1.0 (the verifier's p and the stock loss are untempered)")
    if _get(gen, "stop_strings") or _get(config, "data.default.stop_strings"):
        p.append("stop strings are refused (a stop-string cut is not part of the verified token law)")
    if int(_get(rf, "max_data_attempts_per_prompt", 2) or 1) != 1 or int(_get(rf, "max_infra_attempts_per_prompt", 5) or 1) != 1:
        p.append(
            "async_rl.rollout_failure.max_data_attempts_per_prompt / max_infra_attempts_per_prompt must be 1 (a retried "
            "group is selected by its scores or its timing)"
        )
    for leaf in ("native.generation_timeout_s", "native.env_timeout_s", "nemo_gym.rollout_timeout_s"):
        if _get(rf, leaf) is not None:
            p.append(f"async_rl.rollout_failure.{leaf} must be null (a deadline selects groups by length)")
    if _get(config, "ppo") is not None:
        p.append("a top-level ppo: block is refused (PPO in nolanes mode is not built yet)")
    if _get(async_rl, "generation_fleet_health.enabled", False) or _get(async_rl, "generation_fleet_health.restart_dead_shards", False):
        p.append("async_rl.generation_fleet_health must be off (restarts / shard retirement: engine keys and versions)")
    if _get(async_rl, "generation_router.enabled", False):
        p.append("async_rl.generation_router.enabled must be false (one generation shard; keys are not shard-salted)")
    if _get(config, "checkpointing.enabled", False) or _get(config, "rollout_checkpointing.snapshot_attempt_interval_s"):
        p.append("checkpointing / rollout checkpointing is refused (resume re-keys the engine and loses records)")
    if _get(config, "env.should_use_nemo_gym", False) or _get(config, "token_capture.enabled", False):
        p.append("NeMo-Gym rollouts and token capture are refused (single-turn native rollouts only)")
    try:
        tp = int(_get(mcfg, "tensor_model_parallel_size", 1) or 1)
    except (TypeError, ValueError):
        tp = 1
    if tp >= 3 and str(_get(mcfg, "env_vars.NEMO_RL_TP_REDUCE", os.environ.get("NEMO_RL_TP_REDUCE", "auto"))) == "stock":
        p.append("NEMO_RL_TP_REDUCE=stock at TP >= 3 is refused (row reductions must be rank ordered on both sides)")
    own = driver_owned_env(config)
    if own:
        p.append(f"{own} are set by configure_driver only; remove them")
    dbg = debug_env(config)
    if dbg and not checks_on():
        p.append(f"debug / negative-control variables {dbg} need THUNDERSYNC_NL_CHECKS=1")
    if not float(cfg.k_drafts) >= 1.0:
        p.append(f"thundersync.k_drafts must be >= 1 (got {cfg.k_drafts})")
    if float(cfg.verify_split_x) != 1.0:
        p.append("thundersync.verify_split_x < 1 (engine prompt-logprob share) is not built yet")
    if int(cfg.verify_stream_chunks) < 1:
        p.append(f"thundersync.verify_stream_chunks must be >= 1 (got {cfg.verify_stream_chunks})")
    if int(cfg.verify_stream_chunks) > 1 and not cfg.row_seeds:
        p.append("thundersync.verify_stream_chunks > 1 needs row_seeds (a decision must not depend on its call)")
    if cfg.verify_batch_tokens is not None:
        vbt = int(cfg.verify_batch_tokens)
        envs0 = mcfg.get("env_vars") or {}
        mb = int(_get(pol, "train_micro_batch_size") or 1) * int(_get(pol, "max_total_sequence_length") or 0)
        if vbt < 1:
            p.append(f"thundersync.verify_batch_tokens={vbt} must be >= 1")
        elif str(envs0.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "0")) == "1" and mb > 0 and vbt > mb:
            # Per-call receive caps (CALL_CAP) are sized for a training micro-batch: a larger verify call overflows
            # and reruns (3d36c629).
            p.append(f"thundersync.verify_batch_tokens={vbt} exceeds the training micro-batch {mb} "
                     "(MBS x max_total_sequence_length) under NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP=1")
    # The effective verify budget must fit the MoE device-dispatch capacity, which covers the largest call
    # (device_dispatch.py; an overflow reruns the whole call on Megatron's dispatcher every time). R36.
    envs = mcfg.get("env_vars") or {}
    eff = effective_verify_batch_tokens(config, cfg)
    if eff is not None and str(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH", "0")) == "1":
        cap = int(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS", "20480"))
        if eff > cap:
            p.append(f"verify batch tokens {eff} exceed NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS={cap} "
                     "(thundersync.verify_batch_tokens)")
    return p


def effective_verify_batch_tokens(config: Any, cfg: "NolanesConfig") -> Optional[int]:
    """thundersync.verify_batch_tokens, else the rule: MBS x max_total_sequence_length (the training micro-batch),
    capped by NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS under device dispatch. None when neither is available."""
    if cfg.verify_batch_tokens is not None:
        return int(cfg.verify_batch_tokens)
    pol = _get(config, "policy") or {}
    mbs, mseq = _get(pol, "train_micro_batch_size"), _get(pol, "max_total_sequence_length")
    if not isinstance(mbs, int) or not isinstance(mseq, int):
        return None
    v = mbs * mseq
    envs = _get(pol, "megatron_cfg.env_vars") or {}
    if str(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH", "0")) == "1":
        v = min(v, int(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS", "20480")))
    return v


DEBUG_ENV_PREFIXES = ("THUNDERSYNC_NL_NEG", "THUNDERSYNC_NL_DEBUG_", "THUNDERSYNC_DEBUG_")
# Debug / negative-control knobs outside the prefixes (R29 item 8 / R30 / R32 item 6 cross-check of every env the tree
# reads that perturbs numerics or disables a check): refused without THUNDERSYNC_NL_CHECKS=1.
DEBUG_ENV_NAMES = frozenset({
    "NEMO_RL_FP16_GEMM_DEBUG",  # fp16 GEMM debug path
    "THUNDERSYNC_HIDDEN_DEBUG",  # hidden-log transport debug
    "THUNDERSYNC_FAST_BWD_GRAD_CHECK",  # R30 (fast backward; not on this branch, refused if set)
    "THUNDERSYNC_LEARNER_FAST_BWD_ALT",  # R30
    "THUNDERSYNC_VWALL_CHECK",  # gate check: every DS2 row re-run on the old path (doubles the row tails)
    "NEMO_RL_DEBUG_KV_LIVE_DROP_ONE",  # kv_live_offload negative control: one live KV block is not saved / restored
})
# Owned by configure_driver: never set by hand, CHECKS or not.
DRIVER_OWNED_ENV: tuple[str, ...] = ()


def kv_mode_problems(mcore: Any) -> list[str]:
    """The engine's KV / state must survive the refit with its addresses (requests stay in flight across it, CUDA
    graphs captured the buffer): persist, or offload (released to a CPU backup during training and restored at the
    same addresses; needs torch_memory_saver, the TMS=1 overlay)."""
    mode = _get(mcore, "kv_cache_management_mode")
    if mode == "persist":
        return []
    if mode == "offload":
        try:
            import torch_memory_saver  # noqa: F401
        except ImportError:
            return ["kv_cache_management_mode offload needs torch_memory_saver (see docs/colo/HANDOFF.md, overlays)"]
        from nemo_rl.models.megatron.memory_saver import tms_version_problem

        problem = tms_version_problem()
        return [f"kv_cache_management_mode offload: {problem}"] if problem else []
    return [
        "policy.generation.mcore_generation_config.kv_cache_management_mode must be persist or offload, "
        f"not {mode!r}"
    ]


def router_frozen_problems(config: Any) -> list[str]:
    """The router must be frozen (user rule 11-02): gate weights non-trainable (freeze_moe_router) and no expert-bias
    update (moe_router_bias_update_rate exactly 0, set explicitly)."""
    mcfg = _get(config, "policy.megatron_cfg") or {}
    p: list[str] = []
    if _get(mcfg, "freeze_moe_router") is not True:
        p.append("policy.megatron_cfg.freeze_moe_router must be true (the MoE router gate weights are always frozen)")
    rate = _get(mcfg, "moe_router_bias_update_rate")
    if rate is None or float(rate) != 0.0:
        p.append(f"policy.megatron_cfg.moe_router_bias_update_rate must be 0 (the router expert bias is always frozen; "
                 f"got {rate})")
    return p


def router_frozen_line(config: Any) -> str:
    """The preflight line every run prints (and run_nl_arm.sh requires); raises when the router is not frozen."""
    p = router_frozen_problems(config)
    if p:
        raise ValueError("router not frozen: " + "; ".join(p))
    return "router: frozen (weights+bias)"


def debug_env(config: Any) -> list[str]:
    """Debug / negative-control variables set in this process or in the workers' env_vars."""
    def is_debug(k: str) -> bool:
        return k.startswith(DEBUG_ENV_PREFIXES) or k in DEBUG_ENV_NAMES

    names = {k for k, v in os.environ.items() if v not in ("", "0") and is_debug(k)}
    ev = _get(config, "policy.megatron_cfg.env_vars") or {}
    names |= {k for k, v in dict(ev).items() if str(v) not in ("", "0") and is_debug(k)}
    return sorted(names)


def driver_owned_env(config: Any) -> list[str]:
    """DRIVER_OWNED_ENV names set in this process or the workers' env_vars before configure_driver."""
    ev = dict(_get(config, "policy.megatron_cfg.env_vars") or {})
    return sorted({k for k in DRIVER_OWNED_ENV if os.environ.get(k, "") not in ("", "0") or str(ev.get(k, "")) not in ("", "0")})


def config_warnings(config: Any) -> list[str]:
    """Valid but costly settings (printed loudly by validate_config)."""
    w: list[str] = []  # (the expert-bias rate warning is gone: router_frozen_problems refuses any rate != 0)
    return w


def validate_config(config: Any) -> NolanesConfig:
    for w in config_warnings(config):
        print(f"[nolanes] WARNING: {w}", flush=True)
    problems = config_problems(config)
    if problems:
        raise ValueError("thundersync.mode=nolanes: " + "; ".join(problems) + ".")
    return NolanesConfig.from_config(config)


def derive_layout_fields(config: dict) -> None:
    """Megatron fields that follow from the parallel layouts, set in place (configure_driver, and the launcher's DRY
    preflight before it runs the batch-invariance validators, so both see the config the workers get).

    EP-invariant MoE combine <=> engine EP != learner EP: exact limb sums only where the EP layouts differ; with matched
    EP the engine and learner combine the same expert rows in the same order. Derived here, not in launchers."""
    pol0 = config.get("policy") or {}
    lep = int(_get(pol0, "megatron_cfg.expert_model_parallel_size") or 1)
    gep = int(_get(pol0, "generation.mcore_generation_config.expert_model_parallel_size") or lep)
    pol0.setdefault("megatron_cfg", {})["moe_ep_invariant_combine"] = gep != lep


def crossfade_problems(cfg: Any, mcfg: Any) -> list[str]:
    """Cross-fade routing of a speculative / verified / exact mode (nolanes, xspr, IS-SPR, ...; ``cfg`` has the
    crossfade_routing_* fields and ``crossfade_off_reason``). Required: without it a one-step policy change flips MoE
    expert choices and the draft / target distributions drift apart (Nano whole-row BV acceptance 43-47 % off vs
    85-89 % on, 11-05). Off only with a stated ``crossfade_off_reason`` (logged by ``apply_crossfade``)."""
    p: list[str] = []
    m = getattr(cfg, "crossfade_routing_m", None)
    if m is None:
        if not str(getattr(cfg, "crossfade_off_reason", None) or "").strip():
            p.append("cross-fade routing must be on (thundersync.crossfade_routing_m); to run without it set "
                     "thundersync.crossfade_off_reason to the reason")
        return p
    if not m > 0:
        p.append(f"thundersync.crossfade_routing_m must be > 0 (got {m})")
    if int(cfg.crossfade_routing_extra) < 1:
        p.append("thundersync.crossfade_routing_extra must be >= 1")
    if cfg.crossfade_routing_sparse and str(_get(mcfg, "env_vars.NEMO_RL_MOE_DEVICE_DISPATCH", "0")) != "1":
        p.append(
            "thundersync.crossfade_routing_sparse needs policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH=1 "
            "(Megatron's AlltoAll dispatcher fails on the sparse routing map)"
        )
    if _get(mcfg, "moe_router_fusion", False):
        p.append("policy.megatron_cfg.moe_router_fusion must be false (the fused TE top-k bypasses cross-fade)")
    return p


def apply_crossfade(cfg: Any, mcfg: dict) -> None:
    """Cross-fade routing into ``policy.megatron_cfg`` (one config for learner and engine)."""
    m = getattr(cfg, "crossfade_routing_m", None)
    mcfg["crossfade_routing_m"] = None if m is None else float(m)
    mcfg["crossfade_routing_extra"] = int(cfg.crossfade_routing_extra)
    mcfg["crossfade_routing_m_profile"] = None if cfg.crossfade_routing_m_profile is None else [
        [float(x) for x in r] for r in cfg.crossfade_routing_m_profile
    ]
    mcfg["crossfade_routing_sparse"] = bool(cfg.crossfade_routing_sparse)
    mcfg["crossfade_routing_sparse_engine"] = bool(cfg.crossfade_routing_sparse_engine)
    mcfg["crossfade_routing_ramp_stopgrad"] = bool(cfg.crossfade_routing_ramp_stopgrad)
    if m is None:
        print(f"[crossfade] OFF: {cfg.crossfade_off_reason}", flush=True)


def configure_driver(config: dict) -> NolanesConfig:
    """Driver side, before setup_single_controller: validate, select the ThunderSync worker class (learner and the
    non-colocated engine Policy, a deep copy of policy.*), and hand cross-fade routing to every worker."""
    cfg = validate_config(config)
    # The verify packing rule, resolved once on the driver (the stage only sees a reduced master config).
    vbt = effective_verify_batch_tokens(config, cfg)
    if vbt is None:
        raise ValueError("thundersync.verify_batch_tokens: no rule applies (set it, or policy.train_micro_batch_size "
                         "and policy.max_total_sequence_length)")
    config["thundersync"]["verify_batch_tokens"] = vbt
    derive_layout_fields(config)
    cfg = NolanesConfig.from_config(config)
    from nemo_rl.distributed.ray_actor_environment_registry import ACTOR_ENVIRONMENT_REGISTRY
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    ACTOR_ENVIRONMENT_REGISTRY[WORKER_FQN] = PY_EXECUTABLES.MCORE
    config["policy"]["worker_extension_cls_fqn"] = WORKER_FQN
    mcfg = config["policy"]["megatron_cfg"]
    apply_crossfade(cfg, mcfg)
    # Always-on sentinel (R29 item 4): the loss reports the TRAINING forward's log-probs against every trained
    # token's generation log-prob (train_gen_lp_*; force_on_policy_ratio would otherwise hide a difference); the
    # SC loop raises on any inexact token (VerifyStage.check_train_lp). A loss config field.
    config.setdefault("loss_fn", {})["train_lp_check"] = True
    neg = negative()
    if neg == "cf_engine_m":
        # Negative control: the engine (inference mode only) routes with 2 x the band (crossfade_routing.py).
        mcfg.setdefault("env_vars", {})
        mcfg["env_vars"] = dict(mcfg["env_vars"] or {}, THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE="2")
    print(
        f"[nolanes] mode=nolanes K={cfg.k_drafts} cf m={cfg.crossfade_routing_m} profile={cfg.crossfade_routing_m_profile} extra={cfg.crossfade_routing_extra} "
        f"ramp_stopgrad={cfg.crossfade_routing_ramp_stopgrad} lag="
        f"{_get(config, 'async_rl.sampler.max_lookahead_versions')} checks={int(checks_on())}"
        + (f" NEGATIVE CONTROL {neg}" if neg else ""),
        flush=True,
    )
    return cfg


# ---------------------------------------------------------------------------------------------------------------
# E6 / E9 provenance: every row the learner (loss, advantages, critic, controllers) sees is a verified output row.


class ProvenanceError(RuntimeError):
    pass


def check_provenance(tags: list[dict[str, Any]], version: int, where: str) -> int:
    """Raise unless every row tag carries ``VERIFIED_TAG == version``; returns the rows checked (counter)."""
    bad = [i for i, t in enumerate(tags) if (t or {}).get(VERIFIED_TAG) != version]
    if bad:
        raise ProvenanceError(
            f"[nolanes] {where}: {len(bad)} of {len(tags)} rows are not verified outputs of version {version} "
            f"(E6/E9: drafts, ⊥ rows or rejected prefixes must never reach the learner); first {bad[:5]}"
        )
    return len(tags)


# ---------------------------------------------------------------------------------------------------------------
# Drafts from rollout records (single-turn).


@dataclass
class DraftRow:
    """One drafted rollout of a prompt group, as the rollout pump produced it."""

    group_id: str
    index: int  # sample index in the group (drafting order: fixed at dispatch, never by content or timing)
    prompt: list[int]
    tokens: list[int]
    logq: list[float]  # engine log q of each token (the sampler's law)
    key: Optional[str]  # engine hidden-log key (None: not logged -> dropped from the pool, decoded fresh)
    rid: Optional[int]  # engine request id (noise key of its tokens)
    finished: bool  # EOS / stop or the length cap (the record is complete)


def _assistant(message_log: list[dict]) -> dict:
    asst = [m for m in message_log if m.get("role") == "assistant"]
    if len(asst) != 1:
        raise ValueError(f"nolanes: single-turn rollouts only (found {len(asst)} assistant messages)")
    return asst[0]


def prompt_tokens(prompt_messages: list[dict]) -> list[int]:
    out: list[int] = []
    for m in prompt_messages:
        out += [int(t) for t in m["token_ids"].tolist()]
    return out


def drafts_of_record(group_id: str, prompt_messages: list[dict], record: Any) -> list[DraftRow]:
    prompt = prompt_tokens(prompt_messages)
    rows = []
    for i, c in enumerate(record.completions):
        a = _assistant(c.message_log)
        ref = a.get(DRAFT_REF) or {}
        toks = [int(t) for t in a["token_ids"].tolist()]
        lq = a.get("generation_logprobs")
        lq = [float(x) for x in lq.tolist()] if lq is not None else []
        rows.append(
            DraftRow(
                group_id=group_id,
                index=i,
                prompt=prompt,
                tokens=toks,
                logq=lq,
                key=ref.get("key"),
                rid=ref.get("rid"),
                # A draft cut at a stand-down (wake_verify) ends unfinished: its verified prefix is continued.
                finished=not bool(ref.get("cut", 0)),
            )
        )
    return rows


def version_segments(versions) -> list[tuple[int, int, int]]:
    """Per-token versions -> [(version, start, end)] runs (the ``segments`` draft field).

    Vectorized (the driver runs this for every draft of the step before the first verify call: ~640 drafts x ~5k
    tokens at Super-L22 took ~1 s per step as a per-token Python loop). Same runs as the loop it replaces
    (test_version_segments_matches_the_reference_loop)."""
    import numpy as np

    a = np.asarray(versions, dtype=np.int64).reshape(-1)
    if a.size == 0:
        return []
    cut = np.flatnonzero(a[1:] != a[:-1]) + 1
    starts = np.concatenate(([0], cut)).tolist()
    ends = np.concatenate((cut, [a.size])).tolist()
    vals = a[starts].tolist()
    return [(int(v), int(s), int(e)) for v, s, e in zip(vals, starts, ends)]


def shift_segments(segs: list[tuple[int, int, int]], how: str, dispatch_version: int) -> list[tuple[int, int, int]]:
    """Negative controls on the version tags (E1): ``wrong_tag`` = every token at the dispatch version;
    ``tag_off_by_one`` = the first token after each version change keeps the previous version."""
    if how == "wrong_tag":
        n = segs[-1][2] if segs else 0
        return [(int(dispatch_version), 0, n)] if n else []
    if how == "tag_off_by_one":
        vers: list[int] = []
        for v, a, b in segs:
            vers += [v] * (b - a)
        out = list(vers)
        for i in range(1, len(vers)):
            if vers[i] != vers[i - 1]:
                out[i] = vers[i - 1]
        return version_segments(out)
    return segs


def content_m(draft_tokens: list[int], envs: tuple[float, ...], logq0_is_mode: bool) -> Optional[float]:
    """E3 negative: the envelope of a draft chosen from its OWN first token (the forbidden dependence): the largest
    envelope when its first token is the engine's mode token, else the smallest."""
    if not envs or not draft_tokens:
        return None
    return max(envs) if logq0_is_mode else min(envs)


# ---------------------------------------------------------------------------------------------------------------
# Rows after verification.


@dataclass
class OutputRow:
    """One trained row of a group after verification: a verified draft or a continuation / fresh decode spec."""

    slot: int
    prefix: list[int]  # tokens fixed by verification ([] for a fresh slot)
    prefix_logprobs: list[float]  # verified log p of each prefix token (E7)
    complete: bool  # nothing left to decode
    draft: Optional[DraftRow] = None  # the draft it came from (None: fresh slot)
    kind: str = "fresh"  # whole | continued | fresh


def outputs_of_group(plan_drafts: list[DraftRow], plans: list[Any], g: int) -> list[OutputRow]:
    """The group's G output rows: verified plans (drafting order) in the first slots, then fresh slots."""
    if len(plans) > g:
        raise RuntimeError(f"nolanes: {len(plans)} verified outputs for {g} slots")
    out: list[OutputRow] = []
    for slot, (plan, d) in enumerate(zip(plans, plan_drafts)):
        lps = list(plan.prefix_logprobs)
        if len(lps) != len(plan.prefix):
            raise RuntimeError(
                f"nolanes: plan of draft {d.group_id}/{d.index} has {len(lps)} log-probs for {len(plan.prefix)} tokens"
            )
        whole = bool(plan.complete) and list(plan.prefix) == list(d.tokens)
        out.append(
            OutputRow(
                slot=slot,
                prefix=list(plan.prefix),
                prefix_logprobs=lps,
                complete=bool(plan.complete),
                draft=d,
                kind="whole" if whole else "continued",
            )
        )
    for slot in range(len(out), g):
        out.append(OutputRow(slot=slot, prefix=[], prefix_logprobs=[], complete=False))
    return out


def accept_unverified(rows: list[OutputRow]) -> int:
    """NEGATIVE CONTROL (carry_unverified): every drafted row is trained as drafted, whatever its verification said:
    all its tokens with the engine log q as generation log-prob. Returns the number of tokens it changed."""
    n = 0
    for o in rows:
        d = o.draft
        if d is None:
            continue
        keep = 0
        while keep < min(len(o.prefix), len(d.tokens)) and o.prefix[keep] == d.tokens[keep]:
            keep += 1
        n += len(d.tokens) - keep
        o.prefix, o.prefix_logprobs = list(d.tokens), list(d.logq)
        o.complete, o.kind = bool(d.finished), "whole"
    return n


def keep_engine_q(rows: list[OutputRow]) -> int:
    """NEGATIVE CONTROL (E7): accepted draft tokens keep the engine log q as generation log-prob. Returns the
    number of tokens changed."""
    n = 0
    for o in rows:
        if o.draft is None:
            continue
        for j in range(min(len(o.prefix), len(o.draft.tokens))):
            if o.prefix[j] != o.draft.tokens[j]:
                break
            if o.prefix_logprobs[j] != o.draft.logq[j]:
                o.prefix_logprobs[j] = o.draft.logq[j]
                n += 1
    return n


def continuation_budget(prompt_len: int, prefix_len: int, max_new_tokens: int, max_seq_len: Optional[int]) -> int:
    """Tokens a continuation may still generate: the row's cap (max_new_tokens, or the total-length cap) minus the
    fixed prefix. A function of lengths only."""
    cap = max_new_tokens if max_seq_len is None else max(0, min(max_new_tokens, max_seq_len - prompt_len))
    return max(0, cap - prefix_len)


def stream_chunks(keys: list, sizes: list[int], n: int) -> list[list]:
    """Contiguous runs of ``keys`` (the fixed R2 pool order) for streamed verification: at most ``n`` runs, each
    closing once its draft count reaches the next 1/n of the total. A function of the order and the draft counts only
    (never content or timing); every key in exactly one run, order kept."""
    if len(keys) != len(sizes):
        raise ValueError("stream_chunks: keys and sizes differ in length")
    n = max(1, int(n))
    total = sum(sizes)
    out: list[list] = []
    cur: list = []
    acc = 0
    for k, sz in zip(keys, sizes):
        cur.append(k)
        acc += sz
        if len(out) < n - 1 and acc * n >= total * (len(out) + 1):
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


# spec/null_* statistics that are counts over a call's pools (summed over streamed calls); waves and run totals
# (``*_total``) take the max; the rest are ratios, weighted by the call's drafts.
NULL_COUNT_STATS = frozenset(
    "spec/null_" + k
    for k in (
        "skipped_drafts", "drafts", "slots", "outputs",
        "fresh_slots", "unused_drafts", "empty_walked", "live_groups",
    )
)
# Fractions of a call's live groups: merged weighted by live groups (summing them exceeded 1 with several calls, R36).
NULL_GROUP_FRACTIONS = frozenset(("spec/null_groups_fully_drafted", "spec/null_groups_whole_accepted"))


def merge_call_stats(calls: list[dict]) -> dict:
    """One step's spec/null_* statistics from its streamed verify calls (``calls``: each call's spec/null_* stats)."""
    if len(calls) == 1:
        return dict(calls[0])
    out: dict = {}
    w = [float(c.get("spec/null_drafts", 0.0)) for c in calls]
    for k in sorted({k for c in calls for k in c}):
        vals = [(c[k], wi) for c, wi in zip(calls, w) if k in c]
        if k in NULL_COUNT_STATS:
            out[k] = float(sum(v for v, _ in vals))
        elif k in NULL_GROUP_FRACTIONS:
            lw = [(c[k], float(c.get("spec/null_live_groups", 0.0))) for c in calls if k in c]
            n = sum(x for _, x in lw)
            out[k] = float(sum(v * x for v, x in lw) / n) if n > 0 else float(lw[-1][0])
        elif k in ("spec/null_waves", "spec/null_longest_len_cut") or k.endswith("_total"):
            # run totals (one cumulative counter read after each call): the last call's value is the largest
            out[k] = float(max(v for v, _ in vals))
        else:
            ws = sum(wi for _, wi in vals)
            out[k] = float(sum(v * wi for v, wi in vals) / ws) if ws > 0 else float(vals[-1][0])
    return out


REJECT_BINS = (16, 128, 512, 2048, 4096)
# Critical-path probe: a continued row rejected below this many accepted tokens has a long tail to decode ("early" in
# StepStats.crit first_early_launch). Not the early_rejects count (< EARLY_REJECT_POS), which feeds the CHECKS law.
EARLY_TAIL_TOKENS = 512
EARLY_REJECT_POS = 8
REJECT_LABELS = ("<16", "16-128", "128-512", "512-2k", "2k-4k", ">=4k")


def reject_bin(tau: int) -> int:
    for i, b in enumerate(REJECT_BINS):
        if tau < b:
            return i
    return len(REJECT_BINS)


def early_reject_tail(
    rejections: int, scanned: int, rows: int, early: int, window: int = EARLY_REJECT_POS, slack: float = 3.0
) -> float:
    """P(Poisson(slack * expected) >= early), expected = (rejections / scanned positions) * window * rows."""
    if early <= 0:
        return 1.0
    lam = slack * (rejections / max(scanned, 1)) * window * max(rows, 0)
    term, cdf = math.exp(-lam), 0.0
    for k in range(early):
        cdf += term
        term *= lam / (k + 1)
    return max(0.0, 1.0 - cdf)


@dataclass
class StepStats:
    rows: int = 0
    drafts: int = 0
    whole: int = 0
    continued: int = 0
    fresh: int = 0
    dropped_unlogged: int = 0
    mixed_version_rows: int = 0
    mixed_continued: int = 0  # continued rows whose draft was mixed-version (gate coverage, R35 2(e))
    draft_tokens: int = 0
    accepted_tokens: int = 0
    continuation_tokens: int = 0
    neg_tokens: int = 0
    # Rejection position (accepted tokens before the residual) of every continued row, by bins of REJECT_BINS.
    reject_hist: list = field(default_factory=lambda: [0] * (len(REJECT_BINS) + 1))
    early_rejects: int = 0  # rejections at positions < EARLY_REJECT_POS
    arrival_inversions: int = -1  # train overlap: group pairs trained out of the fixed order (-1: overlap off)
    train_calls: int = 0  # learner train calls this step (groups ready together share one call)
    crit: dict = field(default_factory=dict)  # critical-path probes (first early rejection, longest tail, ms/token)
    times: dict[str, float] = field(default_factory=dict)
    vprof: dict = field(default_factory=dict)  # verify-wall timing (driver + slowest-rank call seconds; logging only)

    def line(self, step: int) -> str:
        t = " ".join(f"{k}={v:.2f}s" for k, v in self.times.items())
        return (
            f"[nolanes verify] step={step} rows={self.rows} drafts={self.drafts} whole={self.whole} "
            f"continued={self.continued} fresh={self.fresh} unlogged={self.dropped_unlogged} "
            f"mixed_version_rows={self.mixed_version_rows} mixed_continued={self.mixed_continued} accepted_frac="
            f"{self.accepted_tokens / max(self.draft_tokens, 1):.4f} cont_tokens={self.continuation_tokens} "
            f"neg_tokens={self.neg_tokens} reject_pos_hist={dict(zip(REJECT_LABELS, self.reject_hist))} early_rejects={self.early_rejects} arrival_inversions={self.arrival_inversions} train_calls={self.train_calls} {t} crit={self.crit}"
            f" vprof={self.vprof}"
        )




def finite(x: float) -> bool:
    return math.isfinite(x)


def now() -> float:
    return time.perf_counter()


def call_idle(calls: list[dict], t_start: float) -> dict:
    """Verify-wall timing (logging only): from the per-call {rank: (entry, exit)} wall-clock pairs of one step's verify
    calls, each rank's idle seconds between its consecutive calls and before its first call (from ``t_start``, the
    verify stage's start), as max / min over ranks; and the last call's exit relative to ``t_start``."""
    if not calls:
        return {}
    ranks = sorted({r for c in calls for r in c})
    idle, first = [], []
    for r in ranks:
        spans = sorted(c[r] for c in calls if r in c)
        first.append(max(0.0, spans[0][0] - t_start))
        idle.append(sum(max(0.0, b[0] - a[1]) for a, b in zip(spans, spans[1:])))
    last = max(c[r][1] for c in calls for r in c)
    return {
        "rank_idle_between_max": round(max(idle), 3),
        "rank_idle_between_min": round(min(idle), 3),
        "rank_first_entry_max": round(max(first), 3),
        "learner_end_rel": round(last - t_start, 3),
    }
