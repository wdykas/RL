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
"""Zero-staleness GRPO with gradient streaming (ThunderSyncRL).

Per logical batch k (all rollouts sampled from theta_k):

1. Open a train step on the learner (zero grads, null mcore sync hooks).
2. Launch every rollout on the (non-colocated) Megatron inference engine.
3. As each trajectory's reward arrives, hand it to the ``StreamPlanner``;
   whenever the learner is idle, dispatch all ready trajectories as backward
   chunks. Open groups accumulate per-(group, reward) gradient sums; closed
   groups get their final advantage. The learner never waits on a group.
4. Barrier: once every rollout is back and every chunk is backpropagated,
   fold the accumulators, apply 1/N, DP-reduce, clip, take one optimizer step.
5. Refit the inference engine with theta_{k+1}.

Every gradient in update k is evaluated at theta_k, so the update is identical
to batch-synchronous GRPO; only the backward work moves into the rollout tail.
``thundersync.granularity=group`` defers each trajectory until its group
finishes; ``granularity=batch`` defers everything until all rollouts finish
(the batch-synchronous baseline). Same code path, identical update.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import os
import time
import warnings
from types import SimpleNamespace
from typing import Any, Callable, Literal, Optional

import numpy as np
import ray
import torch
from pydantic import BaseModel

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _clip_grpo_advantages,
    _create_advantage_estimator,
    add_grpo_token_loss_masks_and_generation_logprobs,
    refit_policy_generation,
)
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import run_sample_multi_turn_rollout
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.logger import Logger
from thundersync_rl.fused_train import FusedRef
from thundersync_rl.null_bv import drafts_per_prompt
from thundersync_rl.routes import RouteRefs
from thundersync_rl.speculative import SpeculativeGeneration, cohort_draft
from thundersync_rl.streaming import (
    Chunk,
    StreamingLearner,
    StreamPlanner,
    Trajectory,
    pad_batch_to_multiple,
)


class ThunderSyncConfig(BaseModel, extra="allow"):
    """Gradient-streaming settings (the ``thundersync:`` YAML block)."""

    # When backward work may start:
    #   "trajectory": as soon as each trajectory's reward arrives (ThunderSyncRL);
    #   "group": once the trajectory's whole group has finished;
    #   "batch": after every rollout of the batch (batch-synchronous baseline).
    # All three produce the identical update; only the overlap differs.
    granularity: Literal["trajectory", "group", "batch"] = "trajectory"
    # Where per-group gradient accumulators live: "cuda", or "cpu" for pinned
    # host memory (the paper's placement; trades H2D/D2H traffic for HBM).
    storage_device: Literal["cuda", "cpu"] = "cuda"
    # Distinct reward values kept per open group before its buckets collapse
    # to the affine (G1, G2) form. 2 is exact for binary rewards under any
    # group-wise estimator; >2-valued rewards need either more buckets or an
    # estimator that is affine in the reward (shared mean/std).
    max_buckets_per_group: int = 2
    # Cap on groups holding streamed gradient buckets at once; bounds
    # accumulator memory to ~max_buckets_per_group * max_open_groups fp32
    # copies of the per-rank gradient. Other open groups wait for their close.
    max_open_groups: int = 4
    # Upper bound on trajectories per backward chunk (bounds activation memory).
    max_chunk_trajectories: int = 8
    # Pipeline parallelism (DESIGN.md section 2, "Streaming under pipeline
    # parallelism"). Each learner rank runs all chunks of a dispatch that share
    # a gradient destination (all final chunks; one open-group bucket) as ONE
    # pipeline schedule call, paying the (PP - 1)-slot fill/drain bubble once
    # per dispatch instead of once per chunk. False: one call per chunk.
    merge_pipeline_calls: bool = True
    # While rollouts are running, hold rows of closed groups back until they
    # make at least this many micro-batches per learner DP rank, so a
    # dispatch's pipeline call amortizes its bubble ((PP-1)/(m+PP-1) for m
    # micro-batches). Always flushed once the step's rollouts are done, so the
    # tail never waits. None: 2 * PP when PP > 1, else 0 (no holding).
    min_pipeline_microbatches: Optional[int] = None
    # Exact cross-iteration speculative rollouts (speculative.py, DESIGN.md):
    # while iteration k runs, the inference engine drafts the rollouts of the next
    # ``draft_lookahead`` iterations; after the optimizer step every draft is
    # block-verified against the new weights and only rejected or unfinished
    # remainders are decoded. Defaults below are the validated configuration
    # (Qwen2.5-Math-1.5B, 2 inference + 2 training GPUs); also set
    # NVIDIA_TF32_OVERRIDE=1 for TF32 GEMMs in the fp32 scorer.
    speculate: bool = False
    # Max new tokens per draft (the rollout length cap is the natural value).
    draft_budget: int = 4096
    # Start drafting once this fraction of the iteration's rollouts finished
    # (0: at the first finished rollout). Starting before any rollout finishes
    # is slower: drafts then compete with the iteration's own rollouts (1.5B,
    # 100 steps: 3.17 vs 3.01 s mean, 3.10 vs 2.55 s median).
    draft_start_frac: float = 0.0
    # Keyed mode only: top-k of the head scored per position.
    spec_head_k: int = 64
    # Keyed mode only: drafts per trajectory sharing its keys; variants > 0
    # jitter the head log-probs by keyed logistic noise of scale ``variant_eps``.
    draft_variants: int = 1
    variant_eps: float = 0.02
    # "block": randomized block verification (Sun et al. 2024; ~99% of draft
    # tokens kept). "keyed": position-keyed Gumbel sampling, exactly on-policy
    # but token-level (~55% kept).
    verify_mode: Literal["block", "keyed"] = "block"
    # Scorer for p and q: "fp32" (a Megatron fp32 copy of the policy, which
    # removes activation-rounding noise from p/q; 4 B/param per shard) or
    # "model" (the training dtypes; half the memory, ~90% kept).
    verify_precision: Literal["model", "fp32"] = "fp32"
    # Lazy, logit-free block verification: residual masses only where the
    # backward scan needs them, and final hidden states [T, d] instead of
    # [T, V] logits (rows recomputed on demand). Same decisions as the eager
    # rule given the same logits; off until validated on GPU.
    verify_lazy: bool = False
    # Where q (the drafts' sampling law) and p come from (DESIGN.md section 14,
    # engine_q.py). "scorer": a scorer copy of the policy (verify_precision)
    # and a whole-model weight stash per pending version. "engine" (batch
    # invariance only): log q of each drafted token is the engine's own
    # log-prob, full q rows are rebuilt on demand from the engine's logged
    # final hidden states and an LM-head-only stash per pending version, and p
    # is the training model's forward. No scorer, no whole-model stash; exact.
    # Requires speculate, verify_mode=block, verify_on=learner,
    # batch_invariant_mode, PP=EP=1, no fused_verify_train and no MTP
    # speculative decoding in the engine. Falls back to "scorer" when the
    # engine cannot log hidden states.
    verify_source: Literal["scorer", "engine"] = "scorer"
    # verify_source=engine: how the engine's hidden states (d x 2 B per drafted
    # token) reach the learner. "object_store": the generation worker puts each
    # draft's rows into the Ray object store once (256-row blocks) and only
    # handles pass through the driver, which frees them after verification; the verifying learner rank fetches a block only
    # when its lazy scan needs a q row in it (fully accepted drafts fetch
    # nothing). "driver": every draft's rows go generation worker -> driver ->
    # each learner worker (the earlier path, kept for A/B).
    hidden_transport: Literal["object_store", "driver"] = "object_store"
    # Iterations drafted concurrently. A draft cut off at one deadline resumes
    # from its prefix, under the newer weights, in the next iteration.
    draft_lookahead: int = 3
    # Block mode: verification chunks, each publishing its plans when done.
    verify_chunks: int = 4
    # Tokens per scorer forward batch (bounds logits memory; lower it for large
    # vocabularies or when verification shares the learner GPUs, e.g. 4096 at 4B).
    verify_batch_tokens: int = 16384
    # Exact lookup speculative decoding of continuations after a rejection: pass the
    # rejected draft's tail and the prompt's other drafts as engine lookup
    # references. Needs the engine's lookup method (mcore_generation_config
    # speculative_method="lookup", num_speculative_tokens).
    tail_lookup: bool = False
    tail_lookup_max_refs: int = 8
    # Verify the longest drafts first so likely stragglers continue earliest.
    verify_longest_first: bool = True
    # Worker groups that run weight stashes and verification: "learner",
    # "inference" or "both" (chunks round-robin over both pools). With large
    # models on few learner GPUs, "inference" keeps scorer memory off the learner.
    verify_on: Literal["learner", "inference", "both"] = "both"
    # Run verification concurrently with the rollouts (rollouts start as their
    # plans are published) also when verify_on != "learner".
    verify_overlap: bool = True
    # Chunks per pool in the round-robin, in verify_on order (learner, inference).
    verify_pool_weights: Optional[list[int]] = None
    # Start drafting when this iteration's verification finishes, instead of at
    # draft_start_frac of completed rollouts.
    draft_after_verify: bool = False
    # EXPERIMENTAL, untested on GPU (DESIGN.md section 14, fused_train.py):
    # fused verification + training under batch-invariant kernels. The learner
    # computes p with the training model at theta_{k+1} (bitwise the training
    # forward under BI) with autograd; drafts that verification turns into
    # finished rollouts keep that graph and are trained from it, saving one
    # forward per such trajectory. Requires speculate, verify_mode=block,
    # verify_on=learner, verify_precision=model,
    # policy.megatron_cfg.batch_invariant_mode, pipeline parallel size 1,
    # generation temperature 1 and no fused linear log-probs. Off by default.
    fused_verify_train: bool = False
    # Fused mode: tokens of retained verification graphs per learner rank
    # (bounds their activation memory, about one training micro-batch of this
    # many tokens). Drafts over the budget are trained the regular way.
    fused_max_retained_tokens: int = 16384
    # Sticky MoE routing anchored to the drafter (nemo_rl/models/megatron/
    # sticky_routing.py, routes.py, DESIGN.md section 14c). The policy of step
    # k is F(theta_k; A): at each token and MoE layer keep the anchor expert
    # set A when its weakest router score is within ``sticky_routing_delta``
    # of the best excluded one (router-logit units for softmax routers), else
    # plain top-k. A = the plain theta_{k-1} forward's routes on the rollout's
    # own prefix (drafted rollouts; up to the rollout cap), self for rollouts
    # never drafted. Removes routing-flip noise from p/q (acceptance) at the
    # price of a one-version routing lag. Requires an MoE model, speculate,
    # verify_mode=block, verify_source=engine, draft_lookahead=1, single-turn
    # rollouts, no tail_lookup and no engine prefix caching.
    sticky_routing: bool = False
    # 0.03 saturates on Qwen3-30B-A3B (full 48 layers, real GPU numerics: same
    # per-token variance as 0.1 and pure replay; 0.003 keeps 66% vs 73% of
    # 32k drafts; ACCEPTANCE_LEDGER section T). None disables sticky routing.
    sticky_routing_delta: Optional[float] = 0.03
    # Pooled null-augmented block verification (⊥-BV; null_bv.py, DESIGN.md
    # section 14d). The engine drafts N = max(G, round(K G)) drafts per GRPO
    # group (K = ``null_bv_drafts_per_output``); each is block-verified
    # against p/M (+) (1 - 1/M) BOT (BOT only at the first position), in
    # drafting order, until the group's G slots hold the first non-BOT outputs;
    # leftover slots decode fresh. The G outputs are iid samples of the
    # current policy (exact); rejections in the first half of long drafts
    # practically vanish at M >= 1.25, at K x the engine draft work. G stays
    # fixed. M_j = DP over (drafts left, slots left) with the Brownian
    # tail-decode model at ``null_bv_logratio_var`` (per-token Var log p/q)
    # and drafts of the rollout cap, or ``null_bv_fixed_m``. Requires
    # speculate, verify_mode=block, verify_source=engine, verify_on=learner,
    # draft_lookahead=1, no fused_verify_train.
    null_bv: bool = False
    null_bv_drafts_per_output: float = 1.5
    null_bv_logratio_var: float = 8e-7
    null_bv_fixed_m: Optional[float] = None


def resolve_min_pipeline_microbatches(ts_cfg: ThunderSyncConfig, pp: int) -> int:
    """``thundersync.min_pipeline_microbatches`` with its PP-dependent default."""
    if ts_cfg.min_pipeline_microbatches is not None:
        if ts_cfg.min_pipeline_microbatches < 0:
            raise ValueError("thundersync.min_pipeline_microbatches must be >= 0")
        return ts_cfg.min_pipeline_microbatches
    return 2 * pp if pp > 1 else 0


def new_cohort_targets(
    step: int,
    lookahead: int,
    max_steps: int,
    existing,
    per_step: int = 2,
) -> list[int]:
    """Iterations whose draft cohorts start at ``step``.

    The nearest iterations in the lookahead window without a cohort, at most
    ``per_step`` per step: warm-up ramps up instead of launching every cohort
    at once, and every iteration after the first gets drafts.
    """
    window = range(step + 1, min(step + 1 + lookahead, max_steps))
    return [t for t in window if t not in existing][:per_step]


class ThunderSyncMasterConfig(MasterConfig):
    thundersync: ThunderSyncConfig = ThunderSyncConfig()


def validate_config(master_config: ThunderSyncMasterConfig) -> None:
    """Fail fast on settings that would break exact equivalence."""
    loss_cfg = master_config.loss_fn
    assert loss_cfg.reference_policy_kl_penalty == 0.0, (
        "gradient streaming supports kl=0 (the reference KL term would need the "
        "reference policy on the learner)"
    )
    assert loss_cfg.force_on_policy_ratio, (
        "set loss_fn.force_on_policy_ratio=true: every gradient is taken at "
        "theta_k, so prev_logprobs == current logprobs"
    )
    # Importance-sampling correction for the inference/training numerical
    # mismatch (loss_fn.use_importance_sampling_correction, optionally
    # truncated) is supported: its weights exp(curr - generation_logprobs) are
    # detached per-token constants, so the loss stays linear in the advantage
    # and the per-(group, reward) buckets remain exact.
    assert not loss_cfg.positive_example_nll_weight
    assert master_config.grpo.num_generations_per_prompt > 1
    assert not master_config.grpo.use_dynamic_sampling
    assert not master_config.grpo.reward_scaling.enabled
    assert not master_config.grpo.reward_shaping.enabled
    assert not master_config.policy["generation"]["colocated"]["enabled"], (
        "gradient streaming needs rollout and learner on separate GPUs"
    )
    assert not master_config.policy["dynamic_batching"]["enabled"]
    assert not master_config.policy["sequence_packing"]["enabled"]
    assert master_config.policy["megatron_cfg"]["enabled"]
    ddp = master_config.policy["megatron_cfg"]["distributed_data_parallel_config"]
    assert ddp["grad_reduce_in_fp32"], (
        "per-trajectory gradients are moved out of the grad buffer; keep it fp32"
    )
    # Per-group advantages are computed as each group closes; estimators with
    # batch-global statistics would silently become per-group.
    assert master_config.grpo.adv_estimator.name == "grpo", (
        "gradient streaming supports grpo.adv_estimator.name='grpo' (per-group "
        f"advantages), got {master_config.grpo.adv_estimator.name!r}"
    )
    assert not getattr(loss_cfg, "seq_logprob_error_in_loss", False), (
        "loss_fn.seq_logprob_error_in_loss renormalizes over surviving sequences, "
        "which the streamed per-chunk loss does not see"
    )
    # Per-sample advantage overrides and reward penalties of the synchronous
    # trainer are not applied by the streaming path.
    grpo_cfg = master_config.grpo
    for key in ("invalid_tool_call_advantage", "malformed_thinking_advantage"):
        assert getattr(grpo_cfg, key, None) is None, (
            f"grpo.{key} is not supported with gradient streaming"
        )
    penalty_cfg = getattr(grpo_cfg, "reward_penalties", None) or {}
    if not isinstance(penalty_cfg, dict):
        penalty_cfg = penalty_cfg.model_dump()
    penalties = [k for k, v in penalty_cfg.items() if k.startswith("penalize_") and v]
    assert not penalties, f"grpo.reward_penalties not supported: {penalties}"
    # The router's load-balancing loss and expert-bias update depend on which
    # rows share a micro-batch, which streaming changes.
    megatron_cfg = master_config.policy["megatron_cfg"]
    assert megatron_cfg.get("moe_router_load_balancing_type", "none") in (
        "none",
        ["none"],
    ), (
        "gradient streaming needs policy.megatron_cfg.moe_router_load_balancing_type='none'"
    )
    assert not megatron_cfg.get("moe_router_bias_update_rate", 0.0), (
        "gradient streaming needs policy.megatron_cfg.moe_router_bias_update_rate=0"
    )
    validate_speculation(master_config.thundersync, master_config.policy)
    validate_expert_parallel(master_config.thundersync, master_config.policy)
    validate_fused_verify_train(master_config.thundersync, master_config.policy)
    validate_engine_verify_source(master_config.thundersync, master_config.policy)
    validate_sticky_routing(master_config)
    validate_null_bv(master_config)


def sticky_delta(ts_cfg: ThunderSyncConfig) -> Optional[float]:
    """The sticky routing threshold in effect (None: sticky routing off)."""
    if not ts_cfg.sticky_routing or ts_cfg.sticky_routing_delta is None:
        return None
    return float(ts_cfg.sticky_routing_delta)


def validate_sticky_routing(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings under which the sticky policy is not exactly sampled.

    The anchors must be an exogenous function of each rollout's prefix (plain
    theta_{k-1} routes up to the rollout cap), engine and learner must apply
    the same rule bitwise (batch invariance, engine-sourced q), and every
    token of a drafted rollout must be sampled from F(theta_k; A): drafted
    tokens by block verification, the rest by theta_{k-1} continuation
    rounds before the engine refits (``SpeculativeGeneration.verify_sticky``).
    """
    ts_cfg = master_config.thundersync
    delta = sticky_delta(ts_cfg)
    if delta is None:
        return
    policy_cfg = master_config.policy
    mcore_gen = policy_cfg["generation"].get("mcore_generation_config") or {}
    problems = []
    if delta < 0:
        problems.append(f"thundersync.sticky_routing_delta must be >= 0 (got {delta})")
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block" or ts_cfg.verify_source != "engine":
        problems.append(
            "thundersync.verify_mode='block' and verify_source='engine' are required "
            "(the engine logs the drafter's routes with its hidden states, and p must "
            "be the training forward with the same anchors)"
        )
    if ts_cfg.draft_lookahead != 1:
        problems.append(
            "thundersync.draft_lookahead must be 1 (drafts spanning versions would "
            "need the routes of the latest drafter version on older segments; not "
            "built yet)"
        )
    if ts_cfg.tail_lookup:
        problems.append(
            "thundersync.tail_lookup must be false (continuations are theta_{k-1} "
            "drafting rounds, not engine lookup decoding)"
        )
    if (master_config.grpo.max_rollout_turns or 1) != 1:
        problems.append(
            "grpo.max_rollout_turns must be 1 (a later turn would be decoded on a "
            "prefix whose anchors the engine cannot apply yet: not built yet)"
        )
    if mcore_gen.get("enable_prefix_caching"):
        problems.append(
            "policy.generation.mcore_generation_config.enable_prefix_caching must be "
            "false (cached prompt positions are not re-routed, so their routes would "
            "not be logged)"
        )
    if mcore_gen.get("num_speculative_tokens"):
        problems.append("in-engine speculative decoding must be off")
    if (policy_cfg["megatron_cfg"].get("expert_model_parallel_size") or 1) < 1:
        problems.append("invalid expert_model_parallel_size")
    if (policy_cfg.get("router_replay") or {}).get("enabled"):
        problems.append("policy.router_replay.enabled (R3) must be false")
    if problems:
        raise ValueError("thundersync.sticky_routing: " + "; ".join(problems) + ".")


def validate_null_bv(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings under which pooled ⊥-BV is not built (or not exact)."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.null_bv:
        return
    problems = []
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block" or ts_cfg.verify_source != "engine":
        problems.append(
            "thundersync.verify_mode='block' and verify_source='engine' are required "
            "(the scorer path is not built yet)"
        )
    if ts_cfg.verify_on != "learner":
        problems.append("thundersync.verify_on must be 'learner' (one worker decides a pool)")
    if ts_cfg.draft_lookahead != 1:
        problems.append(
            "thundersync.draft_lookahead must be 1 (multi-step cohort pools are not "
            "built yet)"
        )
    if ts_cfg.fused_verify_train:
        problems.append("thundersync.fused_verify_train must be false (not built yet)")
    if ts_cfg.tail_lookup:
        problems.append("thundersync.tail_lookup must be false (not built yet)")
    if not ts_cfg.null_bv_drafts_per_output >= 1.0:
        problems.append("thundersync.null_bv_drafts_per_output must be >= 1")
    if ts_cfg.null_bv_fixed_m is not None and not ts_cfg.null_bv_fixed_m >= 1.0:
        problems.append("thundersync.null_bv_fixed_m must be >= 1")
    if not ts_cfg.null_bv_logratio_var >= 0:
        problems.append("thundersync.null_bv_logratio_var must be >= 0")
    if problems:
        raise ValueError("thundersync.null_bv: " + "; ".join(problems) + ".")


def null_bv_policy(master_config: Any) -> dict[str, Any]:
    """``NullPolicy`` arguments: the DP's cost model uses drafts of the rollout cap."""
    ts_cfg = master_config.thundersync
    return {
        "logratio_var": float(ts_cfg.null_bv_logratio_var),
        "ref_len": int(master_config.policy["generation"]["max_new_tokens"]),
        "fixed_m": ts_cfg.null_bv_fixed_m,
    }


def configure_sticky_policy(master_config: Any) -> None:
    """Hand the sticky threshold to the Megatron workers (before they are built).

    Workers read ``policy.megatron_cfg.sticky_routing_delta``: it enables
    MCore's RouterReplay instances in the model (``moe_enable_routing_replay``)
    and the sticky rule (``nemo_rl.models.megatron.sticky_routing``).
    """
    delta = sticky_delta(master_config.thundersync)
    master_config.policy["megatron_cfg"]["sticky_routing_delta"] = delta


def validate_engine_verify_source(
    ts_cfg: ThunderSyncConfig, policy_cfg: PolicyConfig
) -> None:
    """Reject settings under which engine-sourced q is not the drafts' law.

    q is taken from the engine (its log-probs, and rows rebuilt from its
    hidden states with the training model's LM head) and p from the training
    model on the learner, so both must run the same numerics (batch-invariant
    kernels). Any tensor, pipeline and expert parallelism works when training
    and generation use the same layout (batch invariance enforces it): the
    learner's LM-head shard is then the engine's, and q rows are gathered over
    the vocabulary as the engine gathers its logits. At TP>1 the engine must
    materialize only last-token logits, so the head input it logs is gathered
    over sequence parallelism. PP>1 (forward-only pipeline schedule, last
    stage verifies) and EP>1 (equal forward counts over the expert-parallel
    group) are unit-tested only: a warning says so. Virtual (interleaved)
    pipeline stages are not implemented.
    """
    if ts_cfg.verify_source != "engine":
        return
    megatron_cfg = policy_cfg["megatron_cfg"]
    gen_cfg = policy_cfg["generation"]
    mcore_gen = gen_cfg.get("mcore_generation_config") or {}
    problems = []
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block":
        problems.append(
            f"thundersync.verify_mode must be 'block' (got {ts_cfg.verify_mode!r})"
        )
    if ts_cfg.verify_on != "learner":
        problems.append(
            "thundersync.verify_on must be 'learner' (p is the training model's "
            f"forward; got {ts_cfg.verify_on!r})"
        )
    if ts_cfg.fused_verify_train:
        problems.append("thundersync.fused_verify_train must be false")
    if not megatron_cfg.get("batch_invariant_mode"):
        problems.append(
            "policy.megatron_cfg.batch_invariant_mode must be true (the engine's "
            "log-probs and hidden states are the training forward's only under "
            "batch invariance)"
        )
    if (megatron_cfg.get("virtual_pipeline_model_parallel_size") or 1) > 1:
        problems.append(
            "policy.megatron_cfg.virtual_pipeline_model_parallel_size > 1 is not "
            "implemented (the verification forward runs the non-interleaved "
            "pipeline schedule on one model chunk)"
        )
    for key, why in (
        ("expert_model_parallel_size", "the expert all-to-alls and GEMM shapes"),
    ):
        train_v = megatron_cfg.get(key, 1)
        gen_v = mcore_gen.get(key)
        if gen_v is not None and gen_v != train_v:
            problems.append(
                f"policy.generation.mcore_generation_config.{key} ({gen_v}) must "
                f"equal policy.megatron_cfg.{key} ({train_v}): batch invariance "
                f"needs the same layout ({why})"
            )
    gen_tp = mcore_gen.get("tensor_model_parallel_size")
    train_tp = megatron_cfg.get("tensor_model_parallel_size", 1)
    if gen_tp is not None and gen_tp != train_tp:
        problems.append(
            "policy.generation.mcore_generation_config.tensor_model_parallel_size "
            f"({gen_tp}) must equal policy.megatron_cfg.tensor_model_parallel_size "
            f"({train_tp}): the LM-head shards must match the engine's"
        )
    if (
        max(train_tp, gen_tp or 1) > 1
        and mcore_gen.get("materialize_only_last_token_logits") is False
    ):
        problems.append(
            "policy.generation.mcore_generation_config."
            "materialize_only_last_token_logits must be true at tensor parallel "
            "size > 1 (the engine then gathers the sequence-parallel head input "
            "that is logged)"
        )
    if gen_cfg.get("backend") not in (None, "megatron"):
        problems.append("policy.generation.backend must be 'megatron'")
    if mcore_gen.get("num_speculative_tokens"):
        problems.append(
            "policy.generation.mcore_generation_config.num_speculative_tokens must "
            "be 0 (one sampled token per engine forward)"
        )
    if mcore_gen.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        problems.append(
            "policy.generation.mcore_generation_config.logprobs_mode must be "
            "'raw_logprobs' (log q of the sampling distribution)"
        )
    if problems:
        raise ValueError(
            "thundersync.verify_source='engine': " + "; ".join(problems) + "."
        )
    untested = [
        f"{key}={megatron_cfg[key]}"
        for key in ("pipeline_model_parallel_size", "expert_model_parallel_size")
        if (megatron_cfg.get(key) or 1) > 1
    ]
    if untested:
        warnings.warn(
            "thundersync.verify_source='engine' with "
            + ", ".join(untested)
            + " is untested end to end (unit-tested only; GPU validation: "
            "research/thundersync_rl/scripts/bi_tp/engine_q_pp.sh, engine_q_moe.sh)",
            UserWarning,
            stacklevel=2,
        )


def validate_fused_verify_train(
    ts_cfg: ThunderSyncConfig, policy_cfg: PolicyConfig
) -> None:
    """Reject settings under which fused verification + training is not exact.

    The fused gradient equals the regular one only if the verification p
    forward IS the training forward: the training model (learner, training
    dtypes) under batch-invariant kernels, unscaled logits, logits for every
    position, and a single pipeline stage holding the retained graph.
    """
    if not ts_cfg.fused_verify_train:
        return
    megatron_cfg = policy_cfg["megatron_cfg"]
    problems = []
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block":
        problems.append(
            f"thundersync.verify_mode must be 'block' (got {ts_cfg.verify_mode!r})"
        )
    if ts_cfg.verify_on != "learner":
        problems.append(
            f"thundersync.verify_on must be 'learner' (got {ts_cfg.verify_on!r})"
        )
    if ts_cfg.verify_precision != "model":
        problems.append(
            "thundersync.verify_precision must be 'model' so p and q use the "
            f"training numerics (got {ts_cfg.verify_precision!r})"
        )
    if ts_cfg.fused_max_retained_tokens <= 0:
        problems.append("thundersync.fused_max_retained_tokens must be positive")
    if not megatron_cfg.get("batch_invariant_mode"):
        problems.append(
            "policy.megatron_cfg.batch_invariant_mode must be true (the "
            "verification forward must be bitwise the training forward)"
        )
    if megatron_cfg["pipeline_model_parallel_size"] != 1:
        problems.append("policy.megatron_cfg.pipeline_model_parallel_size must be 1")
    if megatron_cfg["tensor_model_parallel_size"] != 1:
        # Verification reads p from the training forward's logits, which are
        # vocab-sharded at TP>1 (no gather on that path).
        problems.append(
            "policy.megatron_cfg.tensor_model_parallel_size must be 1 (the "
            "training forward's logits are vocab-sharded at TP>1)"
        )
    if megatron_cfg["expert_model_parallel_size"] != 1:
        problems.append(
            "policy.megatron_cfg.expert_model_parallel_size must be 1 (fused "
            "forwards run per rank, not in expert-parallel lockstep)"
        )
    if megatron_cfg.get("use_fused_linear_logprobs"):
        problems.append(
            "policy.megatron_cfg.use_fused_linear_logprobs must be false "
            "(verification needs full logits)"
        )
    if policy_cfg["generation"]["temperature"] != 1.0:
        problems.append(
            "policy.generation.temperature must be 1.0 (verification uses "
            "unscaled logits)"
        )
    if problems:
        raise ValueError(
            "thundersync.fused_verify_train=true: " + "; ".join(problems) + "."
        )


def validate_speculation(ts_cfg: ThunderSyncConfig, policy_cfg: PolicyConfig) -> None:
    """Reject settings speculative verification does not reproduce or support.

    Verification compares raw (temperature 1, untruncated) logits of the
    drafting and current weights, and drafts run without stop strings, so the
    rollout law matches the configured sampler only for plain sampling.
    """
    if not ts_cfg.speculate:
        return
    gen = policy_cfg["generation"]
    problems = []
    if gen["temperature"] != 1.0:
        problems.append(
            f"policy.generation.temperature must be 1.0 (got {gen['temperature']})"
        )
    if gen.get("top_p") not in (None, 1.0):
        problems.append(f"policy.generation.top_p must be 1.0 (got {gen['top_p']})")
    if gen.get("top_k") not in (None, -1, 0):
        problems.append(f"policy.generation.top_k must be unset (got {gen['top_k']})")
    if gen.get("stop_strings"):
        problems.append(
            "policy.generation.stop_strings must be unset (drafts ignore them)"
        )
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    inference = merged_inference_megatron_cfg(policy_cfg)
    if ts_cfg.verify_on != "learner" and (
        inference["tensor_model_parallel_size"] > 1
        or inference["pipeline_model_parallel_size"] > 1
    ):
        # The scorer's collectives would share the engine's TP/PP communicators
        # with the running engine, which nothing pauses (as under EP).
        problems.append(
            "thundersync.verify_on must be 'learner' with inference TP or PP > 1"
        )
    megatron_cfg = policy_cfg["megatron_cfg"]
    if ts_cfg.verify_mode == "keyed" and (
        megatron_cfg["tensor_model_parallel_size"] > 1
        or megatron_cfg["pipeline_model_parallel_size"] > 1
    ):
        # Keyed verification calls the training model directly: vocab-sharded
        # logits under TP, no pipeline schedule under PP.
        problems.append("thundersync.verify_mode='keyed' requires learner TP=PP=1")
    if problems:
        raise ValueError("thundersync.speculate=true: " + "; ".join(problems) + ".")


def validate_expert_parallel(
    ts_cfg: ThunderSyncConfig, policy_cfg: PolicyConfig
) -> None:
    """Reject settings that are inexact or deadlock under expert parallelism.

    An open-group bucket is scaled by its group's advantage after the fact,
    which assumes a rank's gradient comes only from its own rows. Expert weights
    also receive gradient from tokens other EP ranks route to them (rows of
    other groups), so their buckets would be mis-scaled. Final chunks carry
    their advantages in the loss and stay exact.
    """
    if (
        policy_cfg["megatron_cfg"]["expert_model_parallel_size"] > 1
        and ts_cfg.granularity == "trajectory"
        and ts_cfg.max_open_groups > 0
    ):
        raise ValueError(
            "expert_model_parallel_size > 1 requires thundersync.granularity="
            "'group' (or max_open_groups=0): open-group gradient buckets are not "
            "exact under expert parallelism"
        )
    from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg

    if (
        ts_cfg.speculate
        and ts_cfg.verify_on != "learner"
        and merged_inference_megatron_cfg(policy_cfg)["expert_model_parallel_size"] > 1
    ):
        # The inference engine steps its EP ranks together; a generation worker
        # blocked in a verification forward leaves its peers' engine all-to-all
        # unmatched while its own scorer all-to-all waits (deadlock).
        raise ValueError(
            "speculation with inference expert parallelism requires "
            f"thundersync.verify_on='learner' (got {ts_cfg.verify_on!r})"
        )


def make_group_advantage_fn(master_config: MasterConfig):
    """Per-group advantages from the exact estimator the sync trainer uses."""
    estimator = _create_advantage_estimator(master_config)

    def fn(rewards: list[float]) -> list[float]:
        r = torch.tensor(rewards, dtype=torch.float32)
        m = len(rewards)
        adv = estimator.compute_advantage(
            prompt_ids=torch.zeros(m, 1, dtype=torch.long),
            rewards=r,
            mask=torch.ones(m, 1),
        )
        adv = _clip_grpo_advantages(adv, master_config.grpo)
        return adv[:, 0].tolist()

    return fn


def build_chunk_data(
    chunk: Chunk,
    tokenizer,
    master_config: ThunderSyncMasterConfig,
    pad_to_multiple: int,
) -> BatchedDataDict:
    """Flatten a chunk's message logs into a ClippedPG training batch.

    A padding chunk keeps its template's tokens but is fully masked out
    (``sample_mask = token_mask = advantages = 0``): it runs a forward/backward
    that adds neither gradient nor normalization count.
    """
    message_logs = [t.payload["message_log"] for t in chunk.trajectories]
    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=master_config.policy[
            "make_sequence_length_divisible_by"
        ],
    )
    n, s = flat["token_ids"].shape
    if chunk.group is None:
        adv = torch.tensor(chunk.advantages, dtype=torch.float32)
    else:
        adv = torch.ones(n, dtype=torch.float32)
    data = BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "prev_logprobs": flat["generation_logprobs"],
            "reference_policy_logprobs": torch.zeros_like(flat["generation_logprobs"]),
            "token_mask": flat["token_loss_mask"],
            "sample_mask": torch.tensor(
                [float(t.payload["loss_multiplier"]) for t in chunk.trajectories]
            ),
            "advantages": adv.unsqueeze(-1).expand(n, s).clone(),
        }
    )
    anchors = [t.payload.get("anchors") for t in chunk.trajectories]
    if any(a is not None for a in anchors):
        # Sticky routing: object-store handles of each drafted rollout's
        # routes (row positions from 0), None for undrafted rows. Each learner
        # rank fetches its pipeline stage's layers (worker ``_with_routes``);
        # undrafted rows and padding get anchor = self there.
        data["route_refs"] = anchors
    if chunk.padding:
        for k in ("sample_mask", "token_mask", "advantages"):
            data[k] = torch.zeros_like(data[k])
    return pad_batch_to_multiple(data, pad_to_multiple)


def _run_workers(policy: Policy, method: str, **kwargs) -> list[Any]:
    return ray.get(policy.worker_group.run_all_workers_single_data(method, **kwargs))


def train_gen_mismatch(trajectories, learner, tokenizer, master_config) -> dict:
    """Debug (THUNDERSYNC_MISMATCH_CHECK): rollout log-probs vs the learner's.

    Called before the optimizer step, so both sides use theta_k: any difference
    is train/generation numerics (zero with batch-invariant kernels, NeMo-RL
    PR 3208) plus, for speculative rollouts, the verification scorer's numerics.
    """
    data = build_chunk_data(
        SimpleNamespace(
            trajectories=trajectories,
            group=None,
            advantages=[0.0] * len(trajectories),
            padding=False,
        ),
        tokenizer,
        master_config,
        learner.dp_size * master_config.policy["logprob_batch_size"],
    )
    train_lp = learner.policy.get_logprobs(data)["logprobs"]
    mask = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).bool()
    mask[:, 0] = False  # no prediction for the first token
    diff = (train_lp - data["generation_logprobs"])[mask]
    return {
        "mismatch/max_abs_logprob_diff": float(diff.abs().max()),
        "mismatch/mean_abs_logprob_diff": float(diff.abs().mean()),
        "mismatch/exact_token_frac": float((diff == 0).float().mean()),
        # k3 estimator of KL(gen || train), as gen_kl_error in the GRPO loss.
        "mismatch/gen_kl": float((diff.exp() - 1 - diff).mean()),
    }


def _initial_sample_state(batch: BatchedDataDict, i: int) -> dict[str, Any]:
    return {
        "message_log": batch["message_log"][i],
        "extra_env_info": batch["extra_env_info"][i],
        "task_name": batch["task_name"][i],
        "stop_strings": batch.get("stop_strings", [None] * batch.size)[i],
        "idx": batch.get("idx", list(range(batch.size)))[i],
    }


async def _run_one_step(
    *,
    step: int,
    batch: BatchedDataDict,
    learner: StreamingLearner,
    policy_generation: GenerationInterface,
    tokenizer,
    task_to_env: dict[str, EnvironmentInterface],
    loss_fn: LossFunction,
    master_config: ThunderSyncMasterConfig,
    ts_cfg: ThunderSyncConfig,
    adv_fn,
    drafter=None,
    before_finish=None,
    pre_rollout=None,
    claim_fused: Optional[
        Callable[[torch.Tensor, torch.Tensor], Optional[FusedRef]]
    ] = None,
    claim_anchors: Optional[
        Callable[[torch.Tensor, torch.Tensor], Optional[RouteRefs]]
    ] = None,
) -> dict[str, Any]:
    G = master_config.grpo.num_generations_per_prompt
    repeated = batch.repeat_interleave(G)
    n = repeated.size
    mbs = master_config.policy["train_micro_batch_size"]

    # Only trajectory granularity streams open groups; group and batch
    # granularity are the same planner with no open-group buckets.
    max_open_groups = (
        ts_cfg.max_open_groups if ts_cfg.granularity == "trajectory" else 0
    )
    # Expert parallelism runs a token all-to-all across DP ranks in every
    # forward/backward, so all ranks must run the same number per dispatch;
    # the worker then also trains zero-advantage rows (no skipping).
    expert_parallel = (
        master_config.policy["megatron_cfg"]["expert_model_parallel_size"] > 1
    )
    pp = (
        master_config.policy["megatron_cfg"].get("pipeline_model_parallel_size", 1) or 1
    )
    min_pipeline_microbatches = resolve_min_pipeline_microbatches(ts_cfg, pp)
    planner = StreamPlanner(
        adv_fn,
        dp_size=learner.dp_size,
        max_chunk_trajectories=ts_cfg.max_chunk_trajectories,
        max_open_groups=max_open_groups,
        micro_batch_size=mbs,
        equal_rank_microbatches=expert_parallel,
        skip_zero_advantage_chunks=(
            bool(getattr(loss_fn, "skip_zero_advantage_rows", False))
            and not expert_parallel
        ),
        min_pipeline_microbatches=min_pipeline_microbatches,
    )
    for p in range(n // G):
        planner.register_group((step, p), G)

    learner.begin(
        loss_fn,
        gbs=n,
        mbs=mbs,
        storage_device=ts_cfg.storage_device,
        max_buckets_per_group=ts_cfg.max_buckets_per_group,
        max_open_groups=max_open_groups,
        merge_pipeline_calls=ts_cfg.merge_pipeline_calls,
    )

    t_start = time.perf_counter()
    wake = asyncio.Event()
    rollouts_done = False
    rewards = [0.0] * n
    sample_metrics: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]
    learner_busy = 0.0
    # Pipeline schedule calls of the busiest rank, summed over dispatches.
    pipeline_calls = 0
    num_dispatches = 0
    # Timeline for offline analysis: (t, group, tokens, reward) per arrival and
    # (t_start, t_end, n_trajectories, n_tokens, n_bucket_chunks) per dispatch.
    arrivals: list[tuple[float, int, int, float]] = []
    dispatches: list[tuple[float, float, int, int, int]] = []
    last_rollout_t = None
    all_trajectories: list[Trajectory] = []
    # Fused verification + training: trajectories handed a retained forward,
    # and what the learner did with them.
    fused_counts = collections.Counter()

    async def rollout(i: int):
        state, metrics = await run_sample_multi_turn_rollout(
            sample_idx=i,
            initial_sample_state=_initial_sample_state(repeated, i),
            policy_generation=policy_generation,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=master_config.policy["max_total_sequence_length"],
            max_rollout_turns=master_config.grpo.max_rollout_turns,
            greedy=False,
        )
        return i, state, metrics

    async def learner_loop():
        nonlocal learner_busy, num_dispatches, pipeline_calls
        while True:
            await wake.wait()
            wake.clear()
            # Batch-synchronous baseline: hold everything until the batch closes.
            # Under PP, rows of closed groups are held until they fill the
            # pipeline (min_pipeline_microbatches) unless the rollouts are done.
            while planner.has_work(flush=rollouts_done) and (
                ts_cfg.granularity != "batch" or rollouts_done
            ):
                d = planner.next_dispatch(flush=rollouts_done)
                per_rank = [
                    [
                        {
                            # Pad only chunks larger than a micro-batch.
                            "data": build_chunk_data(
                                c,
                                tokenizer,
                                master_config,
                                1 if len(c.trajectories) <= mbs else mbs,
                            ),
                            "group": c.group,
                            "reward": c.reward,
                            "fused_key": None if c.fused is None else c.fused.key,
                        }
                        for c in chunks
                    ]
                    for chunks in d.per_rank
                ]
                t0 = time.perf_counter()
                rank_results = await asyncio.gather(*learner.submit(per_rank, d.closes))
                t1 = time.perf_counter()
                pipeline_calls += max(
                    (r.get("num_calls", 0) for r in rank_results), default=0
                )
                for r in rank_results:
                    for k in ("fused_trained", "fused_tokens", "fused_fallback"):
                        fused_counts[k] += r[k]
                learner_busy += t1 - t0
                chunks = [c for cs in d.per_rank for c in cs if not c.padding]
                dispatches.append(
                    (
                        t0 - t_start,
                        t1 - t_start,
                        d.num_trajectories,
                        sum(c.num_tokens for c in chunks),
                        sum(1 for c in chunks if c.group is not None),
                    )
                )
                num_dispatches += 1
            if rollouts_done and not planner.has_work():
                return

    verify_task = (
        asyncio.create_task(pre_rollout()) if pre_rollout is not None else None
    )
    learner_task = asyncio.create_task(learner_loop())
    draft_task = None
    n_done = 0
    after_verify = (
        ts_cfg.draft_after_verify and verify_task is not None and drafter is not None
    )
    if after_verify:
        # Draft only once this iteration's verification is done, so verification
        # runs uncontended on the GPUs it shares with the drafting engine.
        def _start_drafts(_task):
            nonlocal draft_task
            if draft_task is None:
                draft_task = asyncio.create_task(drafter())

        verify_task.add_done_callback(_start_drafts)
    try:
        for fut in asyncio.as_completed([rollout(i) for i in range(n)]):
            i, state, metrics = await fut
            n_done += 1
            if (
                drafter is not None
                and draft_task is None
                and not after_verify
                and n_done >= ts_cfg.draft_start_frac * n
            ):
                draft_task = asyncio.create_task(drafter())
            if learner_task.done():
                # Surface learner failures now instead of after the last rollout.
                learner_task.result()
            last_rollout_t = time.perf_counter()
            reward = float(state["total_reward"])
            rewards[i] = reward
            sample_metrics[i] = metrics
            loss_mult = float(repeated["loss_multiplier"][i])
            if master_config.grpo.overlong_filtering and metrics["truncated"]:
                loss_mult = 0.0
            ntok = sum(int(m["token_ids"].numel()) for m in state["message_log"])
            arrivals.append((last_rollout_t - t_start, i // G, ntok, reward))
            fused = None
            if claim_fused is not None:
                fused = claim_fused(
                    torch.cat([m["token_ids"] for m in repeated["message_log"][i]]),
                    torch.cat([m["token_ids"] for m in state["message_log"]]),
                )
                fused_counts["claimed"] += fused is not None
            payload = {
                "message_log": state["message_log"],
                "loss_multiplier": loss_mult,
            }
            if claim_anchors is not None:
                payload["anchors"] = claim_anchors(
                    torch.cat([m["token_ids"] for m in repeated["message_log"][i]]),
                    torch.cat([m["token_ids"] for m in state["message_log"]]),
                )
            traj = Trajectory(
                group=(step, i // G),
                index=i,
                reward=reward,
                payload=payload,
                num_tokens=ntok,
                fused=fused,
            )
            all_trajectories.append(traj)
            planner.add(traj)
            wake.set()
        rollouts_done = True
        wake.set()
        await learner_task
        if verify_task is not None:
            await verify_task
        t_wait = time.perf_counter()
        drafts = await draft_task if draft_task is not None else None
        if callable(drafts):  # streaming drafts: cut them off at this deadline
            drafts = drafts()
        draft_wait = time.perf_counter() - t_wait
        t_score = time.perf_counter()
        if before_finish is not None and drafts is not None:
            before_finish(drafts)  # learner still holds theta_k
        draft_score = time.perf_counter() - t_score
    except BaseException:
        if draft_task is not None:
            draft_task.cancel()
        learner_task.cancel()
        learner.abort()
        raise
    assert planner.all_done()
    t_rollout_end = last_rollout_t
    mismatch = (
        train_gen_mismatch(all_trajectories, learner, tokenizer, master_config)
        if os.environ.get("THUNDERSYNC_MISMATCH_CHECK")
        else {}
    )
    t_finish = time.perf_counter()
    results = learner.finish()
    t_end = time.perf_counter()
    if mismatch:
        # Weights after this step's update: the exactness checks are only
        # meaningful if they change every step (``mismatch/param_checksum``).
        mismatch["mismatch/param_checksum"] = sum(
            _run_workers(learner.policy, "model_param_checksum")
        )

    # No "loss" metric: bucket chunks are backpropagated with advantage 1, so
    # the per-chunk loss values the worker sums are not the GRPO loss.
    out = {
        "reward": float(np.mean(rewards)),
        "grad_norm": float(torch.as_tensor(results["grad_norm"]).float().mean()),
        "mean_gen_tokens": float(
            np.mean([m["assistant_tokens"] for m in sample_metrics])
        ),
        "max_gen_tokens": float(
            np.max([m["assistant_tokens"] for m in sample_metrics])
        ),
        "time/rollout": t_rollout_end - t_start,
        "time/post_rollout_to_step": t_end - t_rollout_end,
        # Barrier only: write-back, 1/N, DP reduce, clip, optimizer step.
        "time/finish_step": t_end - t_finish,
        "time/step_total": t_end - t_start,
        "learner/busy_s": learner_busy,
        "learner/dispatches": num_dispatches,
        "learner/pipeline_calls": pipeline_calls,
        "learner/peak_open_buffers": results["stream_peak_open_buffers"],
        "timeline/arrivals": arrivals,
        "timeline/dispatches": dispatches,
        "timeline/step_end": t_end - t_start,
        "time/draft_wait": draft_wait,
        "time/draft_score": draft_score,
        "_drafts": drafts,
        **mismatch,
    }
    if claim_fused is not None:
        # Trajectories trained from their verification forward; each saved
        # one training forward of its row (fused/forward_tokens_saved tokens).
        out["fused/trajectories"] = fused_counts["fused_trained"]
        out["fused/forward_tokens_saved"] = fused_counts["fused_tokens"]
        out["fused/claimed"] = fused_counts["claimed"]
        out["fused/fallbacks"] = fused_counts["fused_fallback"]
        out["fused/unclaimed"] = results["stream_fused_unclaimed"]
    return out


def thundersync_grpo_train(
    policy: Policy,
    policy_generation: GenerationInterface,
    dataloader,
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: ThunderSyncMasterConfig,
    logger: Optional[Logger] = None,
) -> list[dict[str, Any]]:
    validate_config(master_config)
    ts_cfg = master_config.thundersync
    adv_fn = make_group_advantage_fn(master_config)
    learner = StreamingLearner(policy)
    max_steps = master_config.grpo.max_num_steps
    history = []
    stale = False  # setup() already performed the initial refit
    step = 0
    spec = None
    if ts_cfg.speculate:
        policy_generation.prepare_for_generation()
        spec = SpeculativeGeneration(
            policy_generation,
            policy,
            vocab_limit=len(tokenizer),
            head_k=ts_cfg.spec_head_k,
            draft_budget=ts_cfg.draft_budget,
            max_new_tokens=master_config.policy["generation"]["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            draft_variants=ts_cfg.draft_variants,
            variant_eps=ts_cfg.variant_eps,
            verify_mode=ts_cfg.verify_mode,
            verify_precision=ts_cfg.verify_precision,
            verify_lazy=ts_cfg.verify_lazy,
            verify_batch_tokens=ts_cfg.verify_batch_tokens,
            tail_lookup=ts_cfg.tail_lookup,
            tail_lookup_max_refs=ts_cfg.tail_lookup_max_refs,
            longest_first=ts_cfg.verify_longest_first,
            verify_on=ts_cfg.verify_on,
            pool_weights=ts_cfg.verify_pool_weights,
            fused_verify_train=ts_cfg.fused_verify_train,
            fused_max_retained_tokens=ts_cfg.fused_max_retained_tokens,
            verify_source=ts_cfg.verify_source,
            hidden_transport=ts_cfg.hidden_transport,
            sticky_delta=sticky_delta(ts_cfg),
            null_bv=ts_cfg.null_bv,
            null_bv_policy=null_bv_policy(master_config) if ts_cfg.null_bv else None,
        )
    sticky = spec is not None and spec.sticky_delta is not None
    G = master_config.grpo.num_generations_per_prompt
    if spec is not None and os.environ.get("THUNDERSYNC_SPEC_SELFTEST"):
        first = next(iter(dataloader))
        prompts = [
            torch.cat([m["token_ids"] for m in first["message_log"][i]])
            for i in range(4)
        ]
        spec.draft_budget, budget = 256, spec.draft_budget
        print("[spec selftest]", asyncio.run(spec.self_test(prompts)), flush=True)
        spec.draft_budget = budget
    batches = iter(dataloader)
    upcoming = collections.deque(
        b for b in itertools.islice(batches, max(1, ts_cfg.draft_lookahead) + 1)
    )
    drafts = None
    cohort_mode = (
        spec is not None
        and ts_cfg.verify_mode == "block"
        and ts_cfg.draft_lookahead > 1
    )
    while upcoming:
        batch = upcoming.popleft()
        nb = next(batches, None)
        if nb is not None:
            upcoming.append(nb)
        next_batch = upcoming[0] if upcoming else None
        if step >= max_steps:
            break
        t0 = time.perf_counter()
        t_verify = 0.0
        if spec is not None:
            spec.current_step = step
        if sticky and drafts is not None:
            # Sticky routing: verify and complete every drafted rollout while
            # the engine still holds theta_{step-1}, whose plain forward
            # supplies the anchors of continuation positions; refit after.
            t1 = time.perf_counter()
            policy_generation.prepare_for_generation()
            asyncio.run(spec.verify_sticky(drafts))
            policy_generation.finish_generation()
            t_verify = time.perf_counter() - t1
            drafts = None
        t_r = time.perf_counter()
        if stale:
            refit_policy_generation(policy, policy_generation, False)
        else:
            policy_generation.prepare_for_generation()
        t_refit = time.perf_counter() - t_r
        pre_rollout = None
        chunked = (
            spec is not None
            and ts_cfg.verify_mode == "block"
            and (ts_cfg.verify_on == "learner" or ts_cfg.verify_overlap)
            # ⊥-BV pools are decided in one call before the rollouts start.
            and not ts_cfg.null_bv
        )
        if cohort_mode and chunked:
            cohort = spec.cohorts.pop(step, None)
            if cohort is not None:
                cdrafts = [[cohort_draft(d)] for d in cohort]
                pre_rollout = lambda c=cdrafts: spec.verify_chunked(
                    c, ts_cfg.verify_chunks
                )  # noqa: E731
        elif chunked and drafts is not None:
            pre_rollout = lambda d=drafts: spec.verify_chunked(d, ts_cfg.verify_chunks)  # noqa: E731
        elif cohort_mode:
            t1 = time.perf_counter()
            spec.verify_cohort(step)
            t_verify = time.perf_counter() - t1
        elif spec is not None and drafts is not None:
            t1 = time.perf_counter()
            spec.verify(drafts)
            t_verify = time.perf_counter() - t1
        drafter = None

        def _prompts(b):
            r = b.repeat_interleave(G)
            return [
                torch.cat([m["token_ids"] for m in r["message_log"][i]])
                for i in range(r.size)
            ]

        if cohort_mode:
            ahead = list(upcoming)[: ts_cfg.draft_lookahead]
            targets = new_cohort_targets(
                step, len(ahead), max_steps, existing=spec.cohorts.keys()
            )
            new = {t: _prompts(ahead[t - step - 1]) for t in targets}
            drafter = lambda new=new, step=step: spec.draft_cohorts(step, new)  # noqa: E731
        elif spec is not None and next_batch is not None and step + 1 < max_steps:
            null_slots = None
            if ts_cfg.null_bv:
                # K G drafts per group, in a fixed (drafting) order; the next
                # step requests G rollouts per batch entry.
                n_drafts = drafts_per_prompt(ts_cfg.null_bv_drafts_per_output, G)
                nxt = next_batch.repeat_interleave(n_drafts)
                null_slots = collections.Counter()
                for i in range(next_batch.size):
                    p = torch.cat([m["token_ids"] for m in next_batch["message_log"][i]])
                    null_slots[tuple(int(t) for t in p.tolist())] += G
                null_slots = dict(null_slots)
            else:
                nxt = next_batch.repeat_interleave(G)
            next_prompts = [
                torch.cat([m["token_ids"] for m in nxt["message_log"][i]])
                for i in range(nxt.size)
            ]
            drafter = (  # noqa: E731
                (
                    lambda prompts=next_prompts, slots=null_slots: spec.draft_streaming(
                        prompts, null_slots=slots
                    )
                )
                if ts_cfg.verify_mode == "block"
                else (lambda prompts=next_prompts: spec.draft(prompts))
            )
        policy.prepare_for_training()
        metrics = asyncio.run(
            _run_one_step(
                step=step,
                batch=batch,
                learner=learner,
                policy_generation=spec if spec is not None else policy_generation,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                loss_fn=loss_fn,
                master_config=master_config,
                ts_cfg=ts_cfg,
                adv_fn=adv_fn,
                drafter=drafter,
                before_finish=spec.score_prev if spec is not None else None,
                pre_rollout=pre_rollout,
                claim_fused=spec.claim_fused
                if spec is not None and ts_cfg.fused_verify_train
                else None,
                claim_anchors=spec.claim_anchors if sticky else None,
            )
        )
        drafts = metrics.pop("_drafts")
        if sticky:
            # This step's training has replayed every anchor targeted at it.
            spec.release_routes(step)
        metrics["time/verify"] = t_verify
        if spec is not None:
            metrics.update(spec.stats)
            spec.stats = {}
        policy_generation.finish_generation()
        stale = True
        metrics["time/refit"] = t_refit
        metrics["time/iteration"] = time.perf_counter() - t0
        history.append(metrics)
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        print(
            f"[thundersync step {step}] "
            + " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in scalars.items()
            ),
            flush=True,
        )
        if logger is not None:
            logger.log_metrics(scalars, step + 1, prefix="thundersync")
        step += 1
    return history
