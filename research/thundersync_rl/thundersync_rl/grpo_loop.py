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
import contextlib
import itertools
import math
import os
import time
import zlib
import warnings
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Literal, Optional

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
from thundersync_rl.lanes import CohortLanes, Lane, LaneStepOps, refit_lane
from thundersync_rl.multi_turn import PrefixGuardGeneration, TurnEnvironment
from thundersync_rl.null_bv import drafts_per_prompt
from thundersync_rl.routes import RouteRefs
from thundersync_rl.run_state import (
    Boundaries,
    PromptStream,
    check_fingerprint,
    config_fingerprint,
    load_state,
    save_state,
)
from thundersync_rl.sentinel import WeightSentinel
from thundersync_rl.speculative import SpeculativeGeneration, cohort_draft
from thundersync_rl.streaming import (
    Chunk,
    StreamingLearner,
    StreamPlanner,
    Trajectory,
    pad_batch_to_multiple,
    tail_excluded_advantages,
)


class ThunderSyncConfig(BaseModel, extra="allow"):
    """Gradient-streaming settings (the ``thundersync:`` YAML block)."""

    # When backward work may start:
    #   "trajectory": as soon as each trajectory's reward arrives (ThunderSyncRL);
    #   "group": once the trajectory's whole group has finished;
    #   "batch": after every rollout of the batch (batch-synchronous baseline).
    # All three produce the identical update; only the overlap differs.
    granularity: Literal["trajectory", "group", "batch"] = "trajectory"
    # Rollout staleness, for the async baseline on THIS learner path (A/B kit arm
    # "async1"): 0 = exact on-policy (every step's rollouts sampled at theta_k);
    # 1 = the rollouts trained at step k were sampled at theta_{k-1}: the engine
    # generates step k+1's batch at theta_k while the learner trains step k, with no
    # verification (NeMo-RL async GRPO with max_trajectory_age_steps=1). Off-policy:
    # pair it with loss_fn.use_importance_sampling_correction (weights
    # exp(curr - generation_logprobs) from the training forward itself).
    rollout_lag: Literal[0, 1] = 0
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
    # Sticky fp16 rounding (nemo_rl/models/megatron/sticky_rounding.py,
    # DESIGN.md section 14g; opt-in, not part of the default recipe). At every
    # fp16 rounding site of the fp16-GEMM recipe the policy of step k rounds
    # x to the anchor a (the plain theta_{k-1} forward's fp16 value on the same
    # prefix) when a is one of x's two fp16 neighbours, else to nearest; same
    # anchor schedule as sticky routing (drafted rollouts anchored, others
    # self). The learner recomputes the anchors: every policy forward runs a
    # theta_{k-1} forward in lockstep, its weights streamed per layer from a
    # host copy of each rank's shard. Requires sticky_routing,
    # forward_gemm_precision=fp16, pipeline parallel size 1 (not built yet),
    # no activation recompute or context parallelism, fused_verify_train false.
    sticky_rounding: bool = False
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
    # Opt-in (agent_tail.md; agent_tail_exact_schemes.md section 3.2): constant
    # envelope null_bv_fixed_m at every count (never the forced M = 1), and a
    # prompt whose pool can no longer fill its G slots is DROPPED. The drop
    # depends on success counts only (law the same for every prompt), so the
    # kept groups are exact; the dataloader batch carries
    # null_bv_spare_prompts extra prompts and the step trains on the first
    # (batch - spares) prompts that were not dropped, in batch order (dropped
    # prompts decode fresh only if too few survive). It shifts the prompt mix
    # through the spare order, like DAPO's dynamic-sampling filter.
    # grpo.num_prompts_per_step is the trained count; the loop re-batches the
    # data stream into batches of num_prompts_per_step + spares prompts.
    null_bv_prompt_drop: bool = False
    null_bv_spare_prompts: int = 0
    # Opt-in, NOT exact samples (unbiased gradient, like banded IS): a ⊥-BV
    # output rejected inside its draft is discarded like BOT (no tail decode)
    # and every kept row's advantage is weighted by max(1, W*/M), W* = the
    # draft's max prefix ratio p/q. Logs spec/drop_is_ess and the weighted
    # fraction. Requires null_bv.
    null_bv_drop_is: bool = False
    # Opt-in estimator change: each row's baseline (and std) is computed over
    # the group's OTHER rows that needed no continuation after verification
    # (all other rows if none), so a group's tail-free rows can train before
    # its tails return; the gradient stays unbiased (streaming.
    # tail_excluded_advantages). Requires speculation and group or batch
    # granularity. Default: the standard GRPO estimator.
    tail_excluded_baseline: bool = False
    # Sticky routing: continuations at theta_{k-1} are block-verified every this
    # many decoded tokens while the engine keeps decoding them; a rejection
    # restarts the continuation from the new prefix, so it wastes at most the
    # tokens decoded past the verified piece (SpeculativeGeneration
    # ._continue_sticky). Fixed count: the cut is exogenous. Below the rollout
    # cap, so a rejection is found before the stream decodes its whole
    # remainder (2048 > a 1024 cap re-decoded the remainder: 1.2-1.9x plain
    # decode per kept token, DESIGN.md section 14k).
    sticky_continuation_chunk_tokens: int = 128
    # Upper bound on tokens per training micro-batch (rows x
    # policy.max_total_sequence_length): rows per micro-batch =
    # min(policy.train_micro_batch_size, this // max_total_sequence_length), >= 1.
    # Bounds activation memory per learner rank by tokens, not rows (an 8 x 10k
    # micro-batch OOMed Nano L8). None: rows only.
    train_micro_batch_tokens: Optional[int] = 32768
    # Continuation piece size grows with the row's generated position:
    # max(chunk_tokens, growth * position) (SpeculativeGeneration._piece_tokens).
    # Each piece costs one forward of the whole row on the learner, so fixed
    # pieces are O(L^2 / chunk) per continued row; growth 1.0 keeps <= 7 pieces
    # per 8k row. 0: fixed pieces.
    sticky_continuation_chunk_growth: float = 1.0
    # Learner prefix-state cache (DESIGN.md section 14p; nemo_rl/models/megatron/
    # prefix_state_cache.py). Each continuation piece is verified by a forward
    # of positions [last Mamba chunk boundary before the piece, end of piece)
    # from the row's cached attention K/V and Mamba conv/SSM state, instead of
    # the whole row: per continued row O(L) learner work instead of
    # O(L^2 / piece). Bitwise the whole-row forward's logits (batch
    # invariance). The driver keeps a row on the learner rank that holds its
    # state. Per learner rank: device residency gpu_gb (colder rows wait in
    # pinned host memory), total max_gb (least recently verified rows are
    # dropped beyond it: slower, same bits). Requires sticky_routing,
    # verify_source=engine, verify_on=learner, no sticky_rounding, no context
    # parallelism, a model without RoPE (Nemotron-H; RoPE offsets not built
    # yet). Off until validated.
    learner_prefix_cache: bool = False
    learner_prefix_cache_gpu_gb: float = 8.0
    learner_prefix_cache_max_gb: float = 32.0
    # Sticky, one engine: serve and train every rollout whose plan is complete
    # while the continuation rounds of the other rows still run (both are
    # learner-side at theta_k; DESIGN.md section 14k). False: training starts
    # after the last tail (the refit and the rollouts follow verify_sticky).
    stream_training_over_tails: bool = True
    # With stream_training_over_tails: train groups with no pending tail first,
    # then the groups whose rows still decode tails, one group per dispatch in a
    # canonical order fixed after the drafts' verification ((tail?, group
    # index)), so the accumulated gradient is bitwise independent of arrival
    # times (StreamPlanner.set_order). False: dispatch as rows arrive.
    canonical_group_order: bool = True
    # Canonical order: groups per training dispatch (blocks formed within each
    # class -- ready now, pending tails, never drafted -- in group index order;
    # a function of the data only). Each learner call has a fixed cost
    # (~0.35 s at Nano L8 EP2), so one group per call made training 2x slower.
    canonical_block_groups: int = 4
    # Measurement only (research/agent_tail.md): the drafting deadline waits
    # until every draft finished (EOS or the cap), so each step verifies whole
    # rollouts and its post-refit tail is the rejection tail alone. Exact (no
    # draft is cut, so none is truncated by content); serializes drafting
    # after training, so never for step-time runs. Requires sticky_routing.
    draft_until_complete: bool = False
    # Sticky routing anchor horizon H (generated positions; DESIGN.md section
    # 14i). None: anchors up to the rollout cap (every drafted rollout is
    # completed at theta_{k-1} before the refit). Set: anchors only at
    # positions < c = prompt + min(H, cap), self beyond; theta_{k-1} work per
    # drafted rollout ends once its anchors below c are logged, and the refit
    # engine samples the rest from p_k directly, re-applying the anchors with
    # the sticky rule in its forwards (engine anchor apply). Bounds theta_{k-1}
    # work (and the refit wait) by H tokens per rollout.
    sticky_anchor_horizon: Optional[int] = None
    # Weights the engine drafts the next step's rollouts with (draft_weights.py,
    # DESIGN.md section 14f). "current": theta_k. "momentum": theta_hat_{k+1} =
    # the AdamW step of the optimizer state (theta_k, m_k, v_k) with gradient 0,
    # rounded to the parameter dtype: a closer proposal q (per-token Var log p/q
    # ~10-15x lower offline, ACCEPTANCE_LEDGER section T), verified exactly as
    # before (q is the engine's logged law). Staged at the top of step k (the
    # state exists right after step k-1's update): the learner writes it into
    # its parameters shard by shard, the engine receives it through the normal
    # refit and keeps a host copy, theta_k is restored bitwise and refit for
    # the rollouts; once the step's rollouts are done every engine rank copies
    # the staged weights back (drafting starts then, not at draft_start_frac,
    # and does not wait for the learner). The engine keeps theta_hat until the
    # next refit (sticky anchors and continuation rounds use it). Requires speculate,
    # verify_mode=block, verify_source=engine, verify_on=learner,
    # draft_lookahead=1, no draft_after_verify, no fused_verify_train,
    # Megatron's distributed optimizer (Adam; with or without the
    # precision-aware optimizer).
    draft_weights: Literal["current", "momentum"] = "current"
    # Cohort lanes (lanes.py, DESIGN.md section 14j): 2 splits the engine fleet
    # into two engines of policy.generation.colocated.resources GPUs each
    # (run_thundersync_grpo.py builds the second), each holding one weight
    # version. At step k the lane of cohort k keeps its drafting weights and
    # completes cohort k's continuation streams, while the other lane refits
    # the drafter of cohort k+1 and drafts it from the top of the step; roles
    # alternate. 1: one engine (single version). Requires sticky routing
    # (the lanes carry the continuation anchors; without it the remainder is
    # plain theta_k decoding, not built on lanes yet) and the packed
    # collective refit (policy.generation.refit_transport null).
    cohort_lanes: int = 1
    # Banded importance sampling (block_verification.py, DESIGN.md section
    # 14e). None (default): off, every row is an exact sample of the current
    # policy. b > 0: a draft is verified against q while its prefix log-ratio
    # S_t = log p/q stays inside |S_t| < b and against p after the exit T; the
    # row is trained with loss weight w = exp(S_T) (|log w| < b + one token's
    # |log p/q|). Rows are then exact in expectation (E[w f(Y)] = E_p f), not
    # in law: the gradient stays unbiased, rejections before T vanish, the
    # weights cost 1/ESS in gradient variance (``spec/band_ess``). With
    # grpo.normalize_rewards the leave-one-out std makes each prompt's scale
    # an expectation over its other rows, which the weights do not reproduce
    # (same per-prompt direction, scale off by O(TV(p, q)); see DESIGN).
    # Requires speculate, verify_mode=block, verify_source=engine, no
    # fused_verify_train, no null_bv (the pooled ⊥-BV grid is not banded). Composes
    # with sticky routing (continuation rounds are plain BV), sticky rounding
    # and momentum draft weights (q is whatever the engine drafted with).
    banded_is_band: Optional[float] = None
    # Always-on engine-vs-learner weight sentinel (sentinel.py, DESIGN.md
    # section 14s): after every refit (theta_k, theta_hat_{k+1}, the initial
    # weights and validation refits) each engine rank's weights are digested
    # and compared bitwise with the learner's shards of the same tensors under
    # the engine's storage dtype; the momentum host stage is re-checked when it
    # is loaded, the stashed LM head against the live one, and expert_bias
    # buffers must never change. A mismatch is an error naming the tensors.
    # O(params / rank) reads per refit, no gather.
    weight_sentinel: bool = True


def train_micro_batch_rows(master_config: "ThunderSyncMasterConfig") -> int:
    """Rows per training micro-batch: policy.train_micro_batch_size, bounded by tokens.

    ``thundersync.train_micro_batch_tokens`` caps rows x the padded row length
    (policy.max_total_sequence_length), so a micro-batch's activations (and the
    LM head's chunked buffers) are bounded by tokens whatever the rollout cap.
    A function of the config only: every rank and the planner agree, and the
    micro-batch boundaries do not depend on timing or batch contents.
    """
    rows = int(master_config.policy["train_micro_batch_size"])
    cap = master_config.thundersync.train_micro_batch_tokens
    if cap:
        rows = min(rows, max(1, int(cap) // int(master_config.policy["max_total_sequence_length"])))
    return rows


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
    data_cfg = getattr(master_config, "data", None) or {}
    max_in = data_cfg.get("max_input_seq_length") if isinstance(data_cfg, dict) else None
    max_total = master_config.policy["max_total_sequence_length"]
    if max_in is not None and max_in > max_total:
        # A prompt between the two passes the data filter (which masks prompts
        # >= max_input_seq_length) but overflows the engine's total-length cap.
        raise ValueError(
            f"data.max_input_seq_length ({max_in}) must not exceed "
            f"policy.max_total_sequence_length ({max_total})"
        )
    if os.environ.get("THUNDERSYNC_DEBUG_ORDER_REPLAY") and not (
        master_config.policy["megatron_cfg"].get("deterministic_training", False)
    ):
        # The replay compares gradient digests of the same rows in other orders;
        # without reproducible backward kernels identical orders already differ.
        raise ValueError(
            "THUNDERSYNC_DEBUG_ORDER_REPLAY requires "
            "policy.megatron_cfg.deterministic_training=true"
        )
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
    assert not (master_config.thundersync.rollout_lag and master_config.thundersync.speculate), (
        "thundersync.rollout_lag=1 is the unverified async baseline; it excludes speculate"
    )
    validate_speculation(master_config.thundersync, master_config.policy)
    validate_expert_parallel(master_config.thundersync, master_config.policy)
    validate_fused_verify_train(master_config.thundersync, master_config.policy)
    validate_engine_verify_source(master_config.thundersync, master_config.policy)
    validate_sticky_routing(master_config)
    validate_sticky_rounding(master_config)
    validate_null_bv(master_config)
    validate_tail_tools(master_config)
    validate_learner_prefix_cache(master_config)
    if master_config.thundersync.draft_until_complete and not (
        master_config.thundersync.sticky_routing
    ):
        raise ValueError(
            "thundersync.draft_until_complete requires sticky_routing (the only "
            "path that keeps draft streams running past the deadline)"
        )
    validate_draft_weights(master_config)
    validate_cohort_lanes(master_config)
    validate_banded_is(master_config)
    validate_run_control(master_config)


def _section(master_config: Any, name: str) -> dict[str, Any]:
    part = getattr(master_config, name, None) or {}
    if not isinstance(part, dict):
        part = part.model_dump() if hasattr(part, "model_dump") else dict(vars(part))
    return part


def validate_run_control(master_config: Any) -> None:
    """Checkpointing, validation and epochs as the ThunderSync loop runs them (DESIGN 14s)."""
    ts_cfg = master_config.thundersync
    ckpt = _section(master_config, "checkpointing")
    grpo = master_config.grpo
    data = _section(master_config, "data")
    problems = []
    val_on = (grpo.val_period or 0) > 0 or grpo.val_at_start or grpo.val_at_end
    if ckpt.get("enabled"):
        if not ckpt.get("save_optimizer", True):
            problems.append(
                "checkpointing.save_optimizer must be true (exact resume needs the "
                "optimizer state; the momentum draft is a function of it)"
            )
        if (ckpt.get("save_period") or 0) <= 0:
            problems.append("checkpointing.save_period must be > 0")
        if ts_cfg.rollout_lag:
            problems.append(
                "checkpointing with thundersync.rollout_lag=1 is not built yet (the "
                "lagged rollouts in flight would have to be saved)"
            )
        if data.get("use_multiple_dataloader"):
            problems.append("checkpointing with data.use_multiple_dataloader is not built yet")
        if ckpt.get("ft_save_period"):
            problems.append(
                "checkpointing.ft_save_period is not built yet (use save_period; "
                "checkpoints are quiescent step boundaries)"
            )
    if (ckpt.get("enabled") or val_on) and ts_cfg.cohort_lanes > 1 and not (
        ts_cfg.stream_training_over_tails
    ):
        problems.append(
            "checkpoints and validation on the non-streaming lanes loop "
            "(stream_training_over_tails=false) are not built yet"
        )
    if val_on and ts_cfg.rollout_lag:
        problems.append("validation with thundersync.rollout_lag=1 is not built yet")
    if (grpo.max_num_epochs or 0) < 1:
        problems.append("grpo.max_num_epochs must be >= 1")
    if problems:
        raise ValueError("thundersync run control: " + "; ".join(problems) + ".")


def check_multi_turn_template(master_config: Any, tokenizer: Any) -> None:
    """Multi-turn rows must be prefix extensions: no chat template re-rendering of history.

    The Nemotron chat template's ``truncate_history_thinking`` (default True)
    strips the reasoning of assistant turns before the last user turn whenever
    the history is rendered. The rollout loops tokenize incrementally (each
    observation alone, ids concatenated; ``PrefixGuardGeneration`` asserts it
    per turn), but any path that renders the history would silently change
    earlier turns: refuse templates that can do it unless it is disabled.
    """
    if (master_config.grpo.max_rollout_turns or 1) <= 1:
        return
    template = getattr(tokenizer, "chat_template", None) or ""
    kwargs = (master_config.policy.get("tokenizer") or {}).get("chat_template_kwargs") or {}
    if "truncate_history_thinking" in template and kwargs.get("truncate_history_thinking") is not False:
        raise ValueError(
            "multi-turn rollouts (grpo.max_rollout_turns > 1) with a chat template that "
            "truncates history thinking by default: set policy.tokenizer."
            "chat_template_kwargs.truncate_history_thinking=false (re-rendered history "
            "would not be a prefix of the sampled row)"
        )


def validate_learner_prefix_cache(master_config: ThunderSyncMasterConfig) -> None:
    """Settings the learner prefix-state cache is built for (DESIGN 14p)."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.learner_prefix_cache:
        return
    megatron_cfg = master_config.policy["megatron_cfg"]
    problems = []
    if not ts_cfg.sticky_routing:
        problems.append("it caches continuation pieces: needs sticky_routing")
    if ts_cfg.verify_source != "engine":
        problems.append("needs verify_source=engine")
    if ts_cfg.verify_on != "learner":
        problems.append("needs verify_on=learner")
    if ts_cfg.sticky_rounding:
        problems.append("sticky_rounding's theta_n anchor forward is not cached (not built yet)")
    if not megatron_cfg.get("batch_invariant_mode"):
        problems.append("needs policy.megatron_cfg.batch_invariant_mode")
    if megatron_cfg.get("context_parallel_size", 1) != 1:
        problems.append("context parallelism not built yet")
    if ts_cfg.learner_prefix_cache_max_gb <= 0 or ts_cfg.learner_prefix_cache_gpu_gb < 0:
        problems.append("needs learner_prefix_cache_max_gb > 0 and gpu_gb >= 0")
    if problems:
        raise ValueError("thundersync.learner_prefix_cache: " + "; ".join(problems) + ".")


def validate_cohort_lanes(master_config: ThunderSyncMasterConfig) -> None:
    """Settings under which two engine lanes keep every cohort on its version."""
    ts_cfg = master_config.thundersync
    if ts_cfg.cohort_lanes == 1:
        return
    gen = master_config.policy["generation"]
    problems = []
    if ts_cfg.cohort_lanes != 2:
        problems.append(f"cohort_lanes must be 1 or 2 (got {ts_cfg.cohort_lanes})")
    if sticky_delta(ts_cfg) is None:
        problems.append(
            "thundersync.sticky_routing must be on (lanes complete each cohort's "
            "continuations on its drafting weights; the non-sticky remainder path "
            "is not built on lanes yet)"
        )
    if gen.get("backend") != "megatron" or gen.get("refit_transport") is not None:
        problems.append(
            "policy.generation.backend must be megatron with refit_transport null "
            "(one packed-collective refit communicator per lane)"
        )
    if ts_cfg.draft_after_verify or ts_cfg.fused_verify_train:
        problems.append("draft_after_verify and fused_verify_train must be off")
    grpo = getattr(master_config, "grpo", None)
    if (getattr(grpo, "max_rollout_turns", 1) or 1) > 1 or ts_cfg.rollout_lag:
        problems.append("multi-turn rollouts and rollout_lag are not built on lanes yet")
    if ts_cfg.draft_lookahead != 1:
        problems.append("draft_lookahead must be 1 (two lanes hold two cohorts)")
    if problems:
        raise ValueError("thundersync.cohort_lanes: " + "; ".join(problems) + ".")


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
    # grpo.max_rollout_turns > 1: every turn of a drafted rollout is sampled
    # at the drafting weights before the refit (scheme A, multi_turn.py);
    # undrafted rollouts are "self" everywhere and take the plain path.
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
    horizon = ts_cfg.sticky_anchor_horizon
    if horizon is not None and int(horizon) < 0:
        problems.append(f"thundersync.sticky_anchor_horizon must be >= 0 (got {horizon})")
    multi_turn = (master_config.grpo.max_rollout_turns or 1) > 1
    if ts_cfg.stream_training_over_tails and multi_turn:
        # Not built yet: a streamed rollout waits for its plan to complete in
        # the continuation rounds; multi-turn plans complete through
        # environment steps (replay_row).
        problems.append(
            "multi-turn rollouts run with thundersync.stream_training_over_tails=false "
            "(streaming over their tails is not built yet)"
        )
    if problems:
        raise ValueError("thundersync.sticky_routing: " + "; ".join(problems) + ".")


def validate_sticky_rounding(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings under which sticky rounding is not built (or not exact).

    The anchor schedule and the continuation rounds are sticky routing's, so it
    must be on; the anchor forward runs in lockstep inside the training
    model's forward (no pipeline stages, no recompute, no context sharding yet).
    """
    ts_cfg = master_config.thundersync
    if not ts_cfg.sticky_rounding:
        return
    mcfg = master_config.policy["megatron_cfg"]
    problems = []
    if sticky_delta(ts_cfg) is None:
        problems.append(
            "thundersync.sticky_routing must be on (its anchor schedule and theta_{k-1} "
            "continuation rounds define where roundings are anchored)"
        )
    if mcfg.get("forward_gemm_precision") != "fp16":
        problems.append("policy.megatron_cfg.forward_gemm_precision must be 'fp16'")
    if (mcfg.get("pipeline_model_parallel_size") or 1) != 1:
        problems.append(
            "pipeline_model_parallel_size must be 1 (the anchor residual stream "
            "travelling with the stage activation is not built yet)"
        )
    if (mcfg.get("context_parallel_size") or 1) != 1:
        problems.append("context_parallel_size must be 1 (not built yet)")
    if mcfg.get("activation_checkpointing"):
        problems.append(
            "activation_checkpointing must be false (recompute would need the "
            "anchors again: not built yet)"
        )
    if mcfg.get("moe_token_dispatcher_type", "alltoall") not in ("alltoall", "allgather"):
        problems.append("moe_token_dispatcher_type must be 'alltoall' or 'allgather'")
    if mcfg.get("moe_shared_expert_overlap"):
        problems.append("moe_shared_expert_overlap must be false")
    if ts_cfg.fused_verify_train:
        problems.append("thundersync.fused_verify_train must be false (not built yet)")
    if ts_cfg.draft_weights != "current":
        problems.append(
            "thundersync.draft_weights must be 'current' (the anchor store keeps "
            "theta_{k-1}; storing the drafting weights theta_hat_k instead is not "
            "built yet)"
        )
    if problems:
        raise ValueError("thundersync.sticky_rounding: " + "; ".join(problems) + ".")


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
    if ts_cfg.null_bv_prompt_drop and ts_cfg.null_bv_fixed_m is None:
        problems.append("thundersync.null_bv_prompt_drop needs null_bv_fixed_m (a constant envelope)")
    if ts_cfg.null_bv_spare_prompts < 0:
        problems.append("thundersync.null_bv_spare_prompts must be >= 0")
    if ts_cfg.null_bv_spare_prompts and not ts_cfg.null_bv_prompt_drop:
        problems.append("thundersync.null_bv_spare_prompts needs null_bv_prompt_drop")
    if problems:
        raise ValueError("thundersync.null_bv: " + "; ".join(problems) + ".")


def validate_tail_tools(master_config: ThunderSyncMasterConfig) -> None:
    """Opt-in tail tools that need other features (agent_tail.md)."""
    ts_cfg = master_config.thundersync
    problems = []
    if (ts_cfg.null_bv_prompt_drop or ts_cfg.null_bv_drop_is) and not ts_cfg.null_bv:
        problems.append("null_bv_prompt_drop / null_bv_drop_is need thundersync.null_bv")
    if ts_cfg.tail_excluded_baseline:
        if not ts_cfg.speculate:
            problems.append("tail_excluded_baseline needs thundersync.speculate")
        if ts_cfg.granularity == "trajectory":
            problems.append("tail_excluded_baseline needs granularity group or batch")
    if problems:
        raise ValueError("thundersync: " + "; ".join(problems) + ".")


def validate_draft_weights(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings under which momentum draft weights are not built.

    q must stay the engine's logged law with the drafting LM head stashed on
    the learner (engine-sourced q), every draft must belong to one version
    (lookahead 1), and the engine may switch weights only once the step's own
    rollouts (sampled at theta_k) are done.
    """
    ts_cfg = master_config.thundersync
    if ts_cfg.draft_weights == "current":
        return
    if ts_cfg.draft_weights != "momentum":
        raise ValueError(f"unknown thundersync.draft_weights {ts_cfg.draft_weights!r}")
    megatron_cfg = master_config.policy["megatron_cfg"]
    opt_cfg = megatron_cfg.get("optimizer") or {}
    problems = []
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block" or ts_cfg.verify_source != "engine":
        problems.append(
            "thundersync.verify_mode='block' and verify_source='engine' are required "
            "(the scorer path recomputes q from theta_k)"
        )
    if ts_cfg.verify_on != "learner":
        problems.append("thundersync.verify_on must be 'learner'")
    if ts_cfg.draft_lookahead != 1:
        problems.append(
            "thundersync.draft_lookahead must be 1 (a two-step momentum draft is not "
            "built yet)"
        )
    if ts_cfg.draft_after_verify:
        problems.append(
            "thundersync.draft_after_verify must be false (drafting starts after the "
            "step's rollouts)"
        )
    if ts_cfg.fused_verify_train:
        problems.append("thundersync.fused_verify_train must be false (not built yet)")
    if not opt_cfg.get("use_distributed_optimizer", False):
        problems.append(
            "policy.megatron_cfg.optimizer.use_distributed_optimizer must be true"
        )
    if opt_cfg.get("optimizer", "adam") != "adam":
        problems.append("policy.megatron_cfg.optimizer.optimizer must be 'adam'")
    if megatron_cfg.get("moe_router_bias_update_rate") and megatron_cfg.get(
        "moe_router_enable_expert_bias", True
    ):
        # Independent of the streaming-wide rejection in validate_config: the
        # momentum stage stages parameters only, not the expert_bias buffer.
        problems.append(
            "MoE expert-bias updates (moe_router_bias_update_rate > 0) are not "
            "supported: the momentum stage does not stage the expert_bias buffer "
            "(not built yet)"
        )
    if problems:
        raise ValueError("thundersync.draft_weights=momentum: " + "; ".join(problems) + ".")


def validate_banded_is(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings under which banded importance sampling is not built."""
    ts_cfg = master_config.thundersync
    band = ts_cfg.banded_is_band
    if band is None:
        return
    problems = []
    if not (0 < band < math.inf):
        problems.append(f"the band must be finite and > 0 (got {band})")
    if not ts_cfg.speculate:
        problems.append("thundersync.speculate must be true")
    if ts_cfg.verify_mode != "block" or ts_cfg.verify_source != "engine":
        problems.append(
            "thundersync.verify_mode='block' and verify_source='engine' are required "
            "(the scorer path is not built yet)"
        )
    if ts_cfg.fused_verify_train:
        problems.append("thundersync.fused_verify_train must be false (not built yet)")
    if ts_cfg.null_bv:
        problems.append(
            "thundersync.null_bv must be false: pooled ⊥-BV verifies every row on a "
            "grid of envelopes and decides on the driver, and that grid is not banded "
            "(not built; the two are alternatives, DESIGN.md section 14e)"
        )
    if problems:
        raise ValueError("thundersync.banded_is_band: " + "; ".join(problems) + ".")


def null_bv_policy(master_config: Any) -> dict[str, Any]:
    """``NullPolicy`` arguments: the DP's cost model uses drafts of the rollout cap."""
    ts_cfg = master_config.thundersync
    return {
        "logratio_var": float(ts_cfg.null_bv_logratio_var),
        "ref_len": int(master_config.policy["generation"]["max_new_tokens"]),
        "fixed_m": ts_cfg.null_bv_fixed_m,
        "prompt_drop": bool(ts_cfg.null_bv_prompt_drop),
    }


def configure_sticky_policy(master_config: Any) -> None:
    """Hand the sticky threshold to the Megatron workers (before they are built).

    Workers read ``policy.megatron_cfg.sticky_routing_delta``: it enables
    MCore's RouterReplay instances in the model (``moe_enable_routing_replay``)
    and the sticky rule (``nemo_rl.models.megatron.sticky_routing``).
    """
    delta = sticky_delta(master_config.thundersync)
    master_config.policy["megatron_cfg"]["sticky_routing_delta"] = delta
    master_config.policy["megatron_cfg"]["sticky_rounding"] = bool(
        master_config.thundersync.sticky_rounding
    )
    # The generation workers apply anchors in their forwards (engine anchor
    # apply) only with an anchor horizon.
    master_config.policy["megatron_cfg"]["sticky_anchor_horizon"] = (
        master_config.thundersync.sticky_anchor_horizon if delta is not None else None
    )


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
        # Measured (runs_1node/np-v2-lanes-ep, Nano L8, learner EP2, engine EP1,
        # THUNDERSYNC_ALLOW_EP_MISMATCH=1): rows the engine samples plainly (step 0,
        # fresh / undrafted rows) differ from the learner forward (max |dlogp| 0.89,
        # exact_frac 2.5e-4); drafted rows are exact because their log-probs come
        # from the learner's verification. So the engine's EP must equal the
        # learner's until the MoE combine is EP-invariant (per-rank fp32 partials
        # summed in rank order vs one rank's fixed-order sum). The override stays
        # for debugging that fix.
        allow = os.environ.get("THUNDERSYNC_ALLOW_EP_MISMATCH") == "1"
        if gen_v is not None and gen_v != train_v and not allow:
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


def rebatch(batches, size: int):
    """Re-chunk a stream of prompt batches into batches of ``size`` prompts (in order)."""
    buf, n = [], 0
    for b in batches:
        buf.append(b)
        n += b.size
        while n >= size:
            whole = BatchedDataDict.from_batches(buf)
            yield whole.slice(0, size)
            rest = whole.slice(size, n) if n > size else None
            buf, n = ([rest] if rest is not None else []), n - size


def select_null_prompts(batch: BatchedDataDict, spec, spares: int) -> BatchedDataDict:
    """⊥-BV prompt drop: the first (batch - spares) prompts whose pools were not dropped.

    Batch order is fixed before drafting and the drop is a function of success
    counts only, so the kept prompts are an independent thinning of the batch.
    If too few survive, dropped prompts fill the rest in batch order and decode
    fresh (no plan: exact p-samples).
    """
    selected = None if spec is None else spec.null_selected_keys
    if selected is None:  # no verified pools this step: the first prompts
        return batch.select_indices(list(range(batch.size - spares)))
    keys = [
        tuple(int(t) for t in torch.cat([m["token_ids"] for m in batch["message_log"][i]]).tolist())
        for i in range(batch.size)
    ]
    want = set(selected)
    rows = [i for i, k in enumerate(keys) if k in want]
    if len(rows) != batch.size - spares:
        raise RuntimeError("null_bv prompt drop: selected prompts do not match the batch")
    return batch.select_indices(rows)


def make_tail_excluded_fn(master_config: MasterConfig):
    """Per-row advantage against the other tail-free rows (opt-in estimator)."""
    normalize = bool(master_config.grpo.normalize_rewards)

    def fn(reward: float, others: list[float]) -> float:
        adv = tail_excluded_advantages(reward, others, normalize)
        return float(_clip_grpo_advantages(torch.tensor([[adv]]), master_config.grpo)[0, 0])

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

    Banded IS: a row's loss weight ``payload["is_weight"]`` multiplies its
    advantage (a detached per-row constant, so open-group buckets stay exact).
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
    # Drop-IS / banded IS (opt-in): a row's importance weight multiplies its
    # advantage (a detached per-row constant).
    adv = adv * torch.tensor(
        [float(t.payload.get("is_weight", 1.0)) for t in chunk.trajectories]
    )
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
        # Bitwise digest of the learner's log-probs of the trained tokens.
        "mismatch/train_lp_crc": zlib.crc32(train_lp[mask].float().numpy().tobytes()),
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


async def _generate_rollouts(
    batch: BatchedDataDict,
    policy_generation: GenerationInterface,
    tokenizer,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: ThunderSyncMasterConfig,
) -> tuple[list[tuple[int, dict, dict]], float]:
    """Every rollout of ``batch`` (G per prompt) at the engine's current weights.

    Returns ``(i, final_state, metrics)`` per rollout and the wall time.
    """
    repeated = batch.repeat_interleave(master_config.grpo.num_generations_per_prompt)
    t0 = time.perf_counter()

    async def one(i: int):
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

    out = await asyncio.gather(*(one(i) for i in range(repeated.size)))
    return list(out), time.perf_counter() - t0


async def _ready(x):
    return x


class KeyedFreshGeneration:
    """Debug (``THUNDERSYNC_DEBUG_KEYED_FRESH=1``, speculation off): reproducible rollouts.

    Every turn request of row ``row`` at step ``step`` carries
    ``noise_seed = hash(grpo.seed, "fresh", step, row, turn)``; the engine's
    keyed sampler (``install_keyed_sampler``, full vocabulary) then draws each
    token by Gumbel-max with noise keyed by (seed, position): an exact sample of
    the engine's distribution that does not depend on which requests share a
    batch. With batch-invariant kernels and deterministic training, two runs
    (or a run and its resume) then produce bitwise the same rollouts, which the
    kill-and-resume exactness test needs (the engine's own sampler draws from
    one generator over the whole batch: not reproducible run to run).
    """

    def __init__(self, inner: Any, base_seed: int, step: int, row: int) -> None:
        self._inner, self._key = inner, (int(base_seed), "fresh", int(step), int(row))
        self._turn = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate_async(self, data, greedy: bool = False, **kwargs):
        from thundersync_rl.run_state import derive_seed

        data["noise_seed"] = torch.tensor([derive_seed(*self._key, self._turn)])
        self._turn += 1
        async for out in self._inner.generate_async(data, greedy=greedy, **kwargs):
            yield out


def keyed_fresh_enabled(master_config: Any) -> bool:
    if not os.environ.get("THUNDERSYNC_DEBUG_KEYED_FRESH"):
        return False
    if master_config.thundersync.speculate:
        raise ValueError("THUNDERSYNC_DEBUG_KEYED_FRESH needs thundersync.speculate=false")
    return True


def install_keyed_fresh(generation: Any) -> None:
    """Install the keyed sampler on every engine rank (full vocabulary, as the logged law)."""
    ray.get(
        generation._policy.worker_group.run_all_workers_single_data(
            "install_keyed_sampler", vocab_limit=1 << 30, head_k=1
        )
    )


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
    replay_row: Optional[
        Callable[
            [int], Awaitable[Optional[tuple[dict, dict, Optional[RouteRefs]]]]
        ]
    ] = None,
    draft_swap: Optional[Callable[[], None]] = None,
    draft_start_frac: Optional[float] = None,
    drafts_after_pre_rollout: bool = False,
    group_order: Optional[list[int]] = None,
    group_blocks: Optional[list[int]] = None,
    tail_groups: Optional[list[int]] = None,
    ordered_learner: Optional[Any] = None,
    claim_row: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
    tail_count: Optional[Callable[[torch.Tensor], int]] = None,
    claim_log_weight: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    stale_rollouts: Optional[list[tuple[int, dict, dict]]] = None,
    generate_next: Optional[Callable[[], Any]] = None,
) -> dict[str, Any]:
    """One step: rollouts, streaming training and the next step's drafts.

    ``drafts_after_pre_rollout``: the drafter also waits for ``pre_rollout``
    (sticky streaming over the tails: the continuation rounds and the refit
    that puts the drafter's weights on the engine). ``group_order``: dispatch
    the step's groups (prompt indices) one at a time in this order
    (``StreamPlanner.set_order``). ``tail_groups``: groups with pending tails,
    logged with each group's completion and dispatch times.
    ``ordered_learner`` (the spec while streaming over the tails): training
    dispatches are numbered learner calls (``learner_seq``) that every learner
    rank runs in the same order, interleaved with the tails' verification
    forwards (asynchronous Ray actors do not keep submission order when
    several calls wait; ``ordered_call`` does). ``stale_rollouts``
    (rollout_lag=1): this batch's rollouts, already generated at theta_{k-1},
    are trained instead of sampling new ones, while ``generate_next()``
    samples the next batch at theta_k on the engine concurrently (returned as
    ``_next_rollouts``).
    """
    G = master_config.grpo.num_generations_per_prompt
    keyed_fresh = keyed_fresh_enabled(master_config)
    if keyed_fresh and group_order is None:
        # Reproducible runs also need a reproducible gradient order: dispatch
        # the groups in index order (StreamPlanner.set_order), not by arrival.
        group_order = list(range(batch.size))
        bg = max(1, int(ts_cfg.canonical_block_groups))
        group_blocks = [min(bg, len(group_order) - j) for j in range(0, len(group_order), bg)]
    # Streaming over the tails: training dispatches are ordered learner calls
    # (SpeculativeGeneration.learner_seq), interleaved with verification.
    ordered = ordered_learner
    repeated = batch.repeat_interleave(G)
    n = repeated.size
    mbs = train_micro_batch_rows(master_config)

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
    def make_planner(order, block_sizes=None):
        pl = StreamPlanner(
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
            tail_excluded_baseline=(
                make_tail_excluded_fn(master_config) if ts_cfg.tail_excluded_baseline else None
            ),
        )
        for p in range(n // G):
            prompt = torch.cat([m["token_ids"] for m in repeated["message_log"][p * G]])
            pl.register_group(
                (step, p),
                G,
                tails=tail_count(prompt) if ts_cfg.tail_excluded_baseline and tail_count else 0,
            )
        if order is not None:
            pl.set_order([(step, p) for p in order], block_sizes)
        return pl

    planner = make_planner(group_order, group_blocks)

    begin_kw = dict(
        gbs=n,
        mbs=mbs,
        storage_device=ts_cfg.storage_device,
        max_buckets_per_group=ts_cfg.max_buckets_per_group,
        max_open_groups=max_open_groups,
        merge_pipeline_calls=ts_cfg.merge_pipeline_calls,
    )
    learner.begin(loss_fn, **begin_kw)
    # Debug (THUNDERSYNC_DEBUG_ORDER_REPLAY): train only after the rollouts and
    # first replay them in other arrival orders (``_order_replay``).
    order_replay = bool(os.environ.get("THUNDERSYNC_DEBUG_ORDER_REPLAY")) and (
        group_order is not None
    )
    replay_digests: dict[str, Any] = {}
    # =params: also name the parameters whose gradients differ (canonical_a vs
    # canonical_a_again).
    per_param_replay = os.environ.get("THUNDERSYNC_DEBUG_ORDER_REPLAY") == "params"
    replay_params: dict[str, Any] = {}
    dispatch_sigs: dict[str, list] = {"real": []}

    def dispatch_sig(d):
        return tuple(
            tuple((c.padding, tuple((t.group[1], t.index) for t in c.trajectories)) for c in cs)
            for cs in d.per_rank
        )

    def grad_digests():
        return [tuple(x) for x in _run_workers(learner.policy, "grad_digest")]

    def replay(key, trajs, order, seed, block_sizes=None):
        """Train ``trajs`` (arrival order shuffled by ``seed``) on a fresh step; digest it."""
        import random as _random

        learner.abort()
        learner.begin(loss_fn, **begin_kw)
        pl = make_planner(order, block_sizes)
        arrival = list(trajs)
        _random.Random(seed).shuffle(arrival)
        sigs = dispatch_sigs.setdefault(key, [])
        for t in arrival:
            pl.add(t)
            while pl.has_work(flush=False):
                d = pl.next_dispatch(flush=False)
                sigs.append(dispatch_sig(d))
                ray.get(learner.submit(_build_dispatch(d), d.closes))
        while pl.has_work():
            d = pl.next_dispatch()
            sigs.append(dispatch_sig(d))
            ray.get(learner.submit(_build_dispatch(d), d.closes))
        out = grad_digests()
        if per_param_replay:
            replay_params[key] = _run_workers(learner.policy, "grad_param_digests")
            if key.startswith("canonical"):
                _run_workers(learner.policy, "grad_param_snapshot", tag=key)
        learner.abort()
        learner.begin(loss_fn, **begin_kw)
        return out

    def _build_dispatch(d):
        # Pad only chunks larger than a micro-batch.
        return [
            [
                {
                    "data": build_chunk_data(
                        c, tokenizer, master_config, 1 if len(c.trajectories) <= mbs else mbs
                    ),
                    "group": c.group,
                    "reward": c.reward,
                    "fused_key": None if c.fused is None else c.fused.key,
                }
                for c in chunks
            ]
            for chunks in d.per_rank
        ]

    t_start = time.perf_counter()
    wake = asyncio.Event()
    rollouts_done = False
    rewards = [0.0] * n
    is_weights = [1.0] * n  # banded IS loss weights
    sample_metrics: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]
    learner_busy = 0.0
    # Pipeline schedule calls of the busiest rank, summed over dispatches.
    pipeline_calls = 0
    num_dispatches = 0
    # Timeline for offline analysis: (t, group, tokens, reward) per arrival and
    # (t_start, t_end, n_trajectories, n_tokens, n_bucket_chunks) per dispatch.
    arrivals: list[tuple[float, int, int, float]] = []
    dispatches: list[tuple[float, float, int, int, int]] = []
    # (group, dispatch start, dispatch end) for every group a dispatch trains.
    group_dispatch: list[tuple[int, float, float]] = []
    last_rollout_t = None
    all_trajectories: list[Trajectory] = []
    # Fused verification + training: trajectories handed a retained forward,
    # and what the learner did with them.
    fused_counts = collections.Counter()

    replayed = [0]  # drafted multi-turn rollouts rebuilt from their records

    async def rollout(i: int):
        drafted = await replay_row(i) if replay_row is not None else None
        if drafted is not None:
            # Multi-turn rollout sampled turn by turn before the refit (sticky
            # routing, scheme A): rebuilt from its recorded turns, the
            # environment is not stepped again.
            replayed[0] += 1
            return i, *drafted
        # Multi-turn: every turn's context must extend the previous turn's
        # tokens (row-is-a-prefix; always on, an error otherwise).
        multi = (master_config.grpo.max_rollout_turns or 1) > 1
        gen = policy_generation
        if keyed_fresh:
            gen = KeyedFreshGeneration(gen, master_config.grpo.seed, step, i)
        if multi:
            gen = PrefixGuardGeneration(gen, i)
        state, metrics = await run_sample_multi_turn_rollout(
            sample_idx=i,
            initial_sample_state=_initial_sample_state(repeated, i),
            policy_generation=gen,
            tokenizer=tokenizer,
            task_to_env=task_to_env,
            max_seq_len=master_config.policy["max_total_sequence_length"],
            max_rollout_turns=master_config.grpo.max_rollout_turns,
            greedy=False,
        )
        if multi:
            gen.final_check(state["message_log"])
        return i, state, metrics, None

    async def learner_loop():
        nonlocal learner_busy, num_dispatches, pipeline_calls
        while True:
            await wake.wait()
            wake.clear()
            # Batch-synchronous baseline: hold everything until the batch closes.
            # Under PP, rows of closed groups are held until they fill the
            # pipeline (min_pipeline_microbatches) unless the rollouts are done.
            while planner.has_work(flush=rollouts_done) and (
                (ts_cfg.granularity != "batch" and not order_replay) or rollouts_done
            ):
                d = planner.next_dispatch(flush=rollouts_done)
                if per_param_replay:
                    dispatch_sigs["real"].append(dispatch_sig(d))
                per_rank = _build_dispatch(d)
                t0 = time.perf_counter()
                seq = None if ordered is None else ordered.learner_seq()
                refs = learner.submit(per_rank, d.closes, seq=seq)
                if seq is not None:
                    ordered.learner_track(refs)
                if rollouts_done and not planner.has_work():
                    last_sent.set()
                rank_results = await asyncio.gather(*refs)
                if seq is not None:
                    ordered._learner_inflight.difference_update(refs)
                t1 = time.perf_counter()
                pipeline_calls += max(
                    (r.get("num_calls", 0) for r in rank_results), default=0
                )
                for r in rank_results:
                    for k in ("fused_trained", "fused_tokens", "fused_fallback"):
                        fused_counts[k] += r[k]
                learner_busy += t1 - t0
                chunks = [c for cs in d.per_rank for c in cs if not c.padding]
                for gid in sorted({t.group[1] for c in chunks for t in c.trajectories}):
                    group_dispatch.append((gid, t0 - t_start, t1 - t_start))
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
                last_sent.set()
                return

    # The step's last training dispatch is on the learner (sticky streaming:
    # the next cohort's drafts are cut meanwhile, ``cut_drafts``).
    last_sent = asyncio.Event()
    verify_task = (
        asyncio.create_task(pre_rollout()) if pre_rollout is not None else None
    )
    if draft_start_frac is None:
        draft_start_frac = ts_cfg.draft_start_frac
    if draft_swap is not None and drafter is not None:
        # Draft weights other than theta_k (staged on the engine at the top of
        # the step): the engine loads them once every rollout of this step
        # (sampled at theta_k) is done; engine-local, the learner keeps training.
        draft_start_frac = 1.0
        draft_after = drafter

        async def drafter():
            await asyncio.to_thread(draft_swap)
            return await draft_after()

    if drafts_after_pre_rollout and verify_task is not None and drafter is not None:
        draft_when_ready = drafter

        async def drafter():
            nonlocal draft_started
            await asyncio.shield(verify_task)
            draft_started = time.perf_counter()  # the drafter's weights are in
            return await draft_when_ready()

    draft_started = None
    learner_task = asyncio.create_task(learner_loop())
    next_task = (
        asyncio.create_task(generate_next()) if generate_next is not None else None
    )
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
            nonlocal draft_started
            if draft_task is None:
                draft_started = time.perf_counter()
                draft_task = asyncio.create_task(drafter())

        verify_task.add_done_callback(_start_drafts)
    cut_task = None
    try:
        sources = (
            asyncio.as_completed([rollout(i) for i in range(n)])
            if stale_rollouts is None
            else [_ready((*r, None)) for r in stale_rollouts]
        )
        for fut in sources:
            i, state, metrics, anchors = await fut
            n_done += 1
            if (
                drafter is not None
                and draft_task is None
                and not after_verify
                and n_done >= draft_start_frac * n
            ):
                draft_started = time.perf_counter()
                draft_task = asyncio.create_task(drafter())
            if learner_task.done():
                # Surface learner failures now instead of after the last rollout.
                learner_task.result()
            last_rollout_t = time.perf_counter()
            reward = float(state["total_reward"])
            if os.environ.get("THUNDERSYNC_DEBUG_PARITY_REWARD"):
                # Debug: a reward that is a fixed function of the rollout's tokens
                # (crc32 parity), so truncated models with zero task reward still
                # train on nonzero advantages (determinism checks need them).
                toks = torch.cat([m["token_ids"] for m in state["message_log"]])
                reward = float(zlib.crc32(toks.to(torch.int64).numpy().tobytes()) & 1)
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
            if anchors is not None:
                payload["anchors"] = anchors
            elif claim_anchors is not None:
                payload["anchors"] = claim_anchors(
                    torch.cat([m["token_ids"] for m in repeated["message_log"][i]]),
                    torch.cat([m["token_ids"] for m in state["message_log"]]),
                )
            info = None
            if claim_row is not None:
                info = claim_row(
                    torch.cat([m["token_ids"] for m in repeated["message_log"][i]]),
                    torch.cat([m["token_ids"] for m in state["message_log"]]),
                )
                if info is not None and info.log_weight:
                    payload["is_weight"] = math.exp(info.log_weight)
            if claim_log_weight is not None:
                is_weights[i] = math.exp(
                    claim_log_weight(
                        torch.cat([m["token_ids"] for m in repeated["message_log"][i]]),
                        torch.cat([m["token_ids"] for m in state["message_log"]]),
                    )
                )
                payload["is_weight"] = is_weights[i]
            traj = Trajectory(
                group=(step, i // G),
                index=i,
                reward=reward,
                payload=payload,
                num_tokens=ntok,
                fused=fused,
                tail=bool(info is not None and info.tail),
            )
            all_trajectories.append(traj)
            planner.add(traj)
            wake.set()
        if order_replay:
            # Same trajectories, two other arrival orders under the canonical
            # order (must equal the real step bitwise) and one with arrival
            # order dispatch (control: expected to differ).
            if ordered is not None:
                await ordered.learner_drain()
            trajs = all_trajectories
            for key, order, seed, blocks in (
                ("canonical_a", group_order, 1, group_blocks),
                ("canonical_b", group_order, 2, group_blocks),
                ("canonical_a_again", group_order, 1, group_blocks),
                ("arrival", None, 3, None),
            ):
                replay_digests[key] = replay(key, trajs, order, seed, blocks)
        rollouts_done = True
        wake.set()
        if ordered is not None and draft_task is not None:
            # Sticky streams go on decoding after the cut (their tokens become
            # continuation pieces), so the cut time only moves tokens between
            # the next verify0 and its continuation rounds: cut while the last
            # training dispatch runs instead of after it.
            async def cut_drafts():
                await last_sent.wait()
                t = time.perf_counter()
                d = await draft_task
                if callable(d):
                    d = await asyncio.to_thread(d)
                return d, time.perf_counter() - t

            cut_task = asyncio.create_task(cut_drafts())
        await learner_task
        t_next = time.perf_counter()
        next_rollouts = await next_task if next_task is not None else None
        next_wait = time.perf_counter() - t_next
        if verify_task is not None:
            await verify_task
        t_wait = time.perf_counter()
        if cut_task is not None:
            drafts, draft_cut_s = await cut_task
        else:
            drafts = await draft_task if draft_task is not None else None
            if callable(drafts):  # streaming drafts: cut them off at this deadline
                drafts = drafts()
            draft_cut_s = None
        draft_wait = time.perf_counter() - t_wait
        t_score = time.perf_counter()
        if before_finish is not None and drafts is not None:
            before_finish(drafts)  # learner still holds theta_k
        draft_score = time.perf_counter() - t_score
    except BaseException:
        if next_task is not None:
            next_task.cancel()
        if draft_task is not None:
            draft_task.cancel()
        if cut_task is not None:
            cut_task.cancel()
        learner_task.cancel()
        learner.abort()
        raise
    assert planner.all_done()
    if order_replay:
        real = grad_digests()
        if per_param_replay:
            _run_workers(learner.policy, "grad_param_snapshot", tag="real")
            for k in ("canonical_a", "canonical_b", "canonical_a_again"):
                diffs = _run_workers(
                    learner.policy, "grad_param_compare", a=k, b="real", drop=False
                )
                for r, diff in enumerate(diffs):
                    print(f"[order replay] step={step} rank={r} {k}_vs_real={diff}", flush=True)
                sig = dispatch_sigs.get(k)
                first = next(
                    (i for i, (x, y) in enumerate(zip(sig, dispatch_sigs["real"])) if x != y),
                    None,
                )
                print(
                    f"[order replay] step={step} {k} dispatches={len(sig)} "
                    f"real={len(dispatch_sigs['real'])} first_differing={first}"
                    + ("" if first is None else f" {k}={sig[first]} real={dispatch_sigs['real'][first]}"),
                    flush=True,
                )
            _run_workers(learner.policy, "grad_param_compare", a="real", b="real")
        bad = []
        for k, v in replay_digests.items():
            same = v == real
            print(f"[order replay] step={step} {k} same_as_real={same} real={real} {k}={v}", flush=True)
            if k.startswith("canonical") and not same:
                bad.append(k)
        # canonical_a_again repeats canonical_a's exact order: if even it differs,
        # the training kernels themselves are not deterministic.
        print(
            f"[order replay] step={step} canonical_a==canonical_a_again="
            f"{replay_digests['canonical_a'] == replay_digests['canonical_a_again']}",
            flush=True,
        )
        if per_param_replay:
            pairs = zip(replay_params["canonical_a"], replay_params["canonical_a_again"])
            for r, (a, b) in enumerate(pairs):
                diff = [k for k in a if a[k] != b.get(k)]
                print(
                    f"[order replay] step={step} rank={r} params_differ={len(diff)}/{len(a)} "
                    f"{diff}",
                    flush=True,
                )
        if bad:
            raise RuntimeError(f"order replay: {bad} gradient differs from the real step")
        replay_metrics = {
            f"debug/order_replay_{k}_same": float(v == real) for k, v in replay_digests.items()
        }
    t_rollout_end = last_rollout_t
    t_check = time.perf_counter()
    mismatch = (
        train_gen_mismatch(all_trajectories, learner, tokenizer, master_config)
        if os.environ.get("THUNDERSYNC_MISMATCH_CHECK")
        else {}
    )
    if mismatch:
        # Debug forward on the step's critical path: reported so step-time
        # comparisons can subtract it (``time/mismatch_check``).
        mismatch["time/mismatch_check"] = time.perf_counter() - t_check
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
    # Rollout-token entropy estimate E[-log pi(token)] from the rollouts' own
    # log-probs (the same definition for every A/B arm; on-policy it equals
    # the stock loss's approx_entropy).
    nll = n_tok = 0.0
    for t in all_trajectories:
        for m in t.payload["message_log"]:
            lp = m.get("generation_logprobs") if m.get("role") == "assistant" else None
            if lp is not None and len(lp):
                nll -= float(torch.as_tensor(lp).float().sum())
                n_tok += len(lp)
    out = {
        "reward": float(np.mean(rewards)),
        "rollout_entropy": nll / max(n_tok, 1),
        # Bitwise digest of the trained rows (tokens and their stored
        # log-probs), row order fixed by row index.
        "digest/rollouts": _rollout_crc(all_trajectories),
        "grad_norm": float(torch.as_tensor(results["grad_norm"]).float().mean()),
        "mean_gen_tokens": float(
            np.mean([m["assistant_tokens"] for m in sample_metrics])
        ),
        "max_gen_tokens": float(
            np.max([m["assistant_tokens"] for m in sample_metrics])
        ),
        "mean_turns": float(np.mean([m["turn_count"] for m in sample_metrics])),
        "max_turns": float(np.max([m["turn_count"] for m in sample_metrics])),
        "time/rollout": t_rollout_end - t_start,
        "time/post_rollout_to_step": t_end - t_rollout_end,
        # Barrier only: write-back, 1/N, DP reduce, clip, optimizer step.
        "time/finish_step": t_end - t_finish,
        "time/step_total": t_end - t_start,
        "learner/busy_s": learner_busy,
        # Non-vacuity of timing comparisons: rows whose GRPO advantage is
        # nonzero (all-zero-advantage rows may be skipped by the learner, which
        # makes training free when every reward is equal).
        "train/rows": float(len(all_trajectories)),
        "train/nonzero_adv_rows": float(
            sum(1 for t in all_trajectories if planner._advantage(t) != 0.0)
        ),
        "learner/dispatches": num_dispatches,
        "learner/pipeline_calls": pipeline_calls,
        "learner/peak_open_buffers": results["stream_peak_open_buffers"],
        # Rank 0's peak allocated / reserved memory over the step's dispatches.
        "learner/mem_peak_gb": results.get("learner_mem_peak_gb", float("nan")),
        "learner/mem_reserved_gb": results.get("learner_mem_reserved_gb", float("nan")),
        # Seconds rank 0 spent releasing cached blocks before the optimizer step
        # (0 when reserved memory was below the pressure threshold).
        "learner/empty_cache_s": results.get("learner_empty_cache_s", float("nan")),
        "timeline/arrivals": arrivals,
        "timeline/dispatches": dispatches,
        "timeline/step_end": t_end - t_start,
        "timeline/group_dispatch": group_dispatch,
        "timeline/tail_groups": list(tail_groups or []),
        "time/draft_wait": draft_wait,
        # When this step's drafter started, from the start of the rollouts.
        "time/draft_start": (
            draft_started - t_start if draft_started is not None else float("nan")
        ),
        "time/draft_score": draft_score,
        # Overlapped cut: its own duration (draft_wait is what it left exposed).
        "time/draft_cut": draft_cut_s if draft_cut_s is not None else draft_wait,
        "_drafts": drafts,
        **mismatch,
        **(replay_metrics if order_replay else {}),
    }
    if replay_row is not None:
        out["spec/replayed_multi_turn_rows"] = float(replayed[0])
    if next_rollouts is not None:
        # rollout_lag=1: the next batch, sampled at theta_k during this step.
        out["_next_rollouts"], out["time/next_rollout"] = next_rollouts
        out["time/next_rollout_wait"] = next_wait
    if claim_log_weight is not None:
        # Unbiased for the policy's mean reward (plain "reward" is the mean
        # under the rows' own law p / w).
        out["reward_is"] = float(np.mean(np.multiply(is_weights, rewards)))
    if claim_fused is not None:
        # Trajectories trained from their verification forward; each saved
        # one training forward of its row (fused/forward_tokens_saved tokens).
        out["fused/trajectories"] = fused_counts["fused_trained"]
        out["fused/forward_tokens_saved"] = fused_counts["fused_tokens"]
        out["fused/claimed"] = fused_counts["claimed"]
        out["fused/fallbacks"] = fused_counts["fused_fallback"]
        out["fused/unclaimed"] = results["stream_fused_unclaimed"]
    return out


def _rollout_crc(trajectories) -> int:
    crc = 0
    for t in sorted(trajectories, key=lambda t: t.index):
        for m in t.payload["message_log"]:
            crc = zlib.crc32(m["token_ids"].to(torch.int64).numpy().tobytes(), crc)
            lp = m.get("generation_logprobs")
            if lp is not None:
                crc = zlib.crc32(torch.as_tensor(lp).float().numpy().tobytes(), crc)
    return crc


def next_draft_prompts(
    next_batch: BatchedDataDict, ts_cfg: ThunderSyncConfig, G: int
) -> tuple[list[torch.Tensor], Optional[dict[tuple[int, ...], int]]]:
    """Prompts of the next step's drafts and, with ⊥-BV, its slots per prompt."""
    null_slots = None
    if ts_cfg.null_bv:
        # K G drafts per group, in a fixed (drafting) order; the next step
        # requests G rollouts per batch entry.
        n_drafts = drafts_per_prompt(ts_cfg.null_bv_drafts_per_output, G)
        nxt = next_batch.repeat_interleave(n_drafts)
        # Submission order: every group's first G drafts (the ones its slots
        # take unless some are rejected), then the extras. The engine admits
        # drafts in this order, so with request capacity for the primaries
        # the extras decode only in what is left of the window. Each pool's
        # own order (draft index within the group) is unchanged.
        primary = [i * n_drafts + j for i in range(next_batch.size) for j in range(G)]
        extra = [i * n_drafts + j for i in range(next_batch.size) for j in range(G, n_drafts)]
        nxt = nxt.select_indices(primary + extra)
        counts = collections.Counter()
        for i in range(next_batch.size):
            p = torch.cat([m["token_ids"] for m in next_batch["message_log"][i]])
            counts[tuple(int(t) for t in p.tolist())] += G
        null_slots = dict(counts)
    else:
        nxt = next_batch.repeat_interleave(G)
    prompts = [
        torch.cat([m["token_ids"] for m in nxt["message_log"][i]])
        for i in range(nxt.size)
    ]
    return prompts, null_slots


def make_speculative_generation(
    generation: GenerationInterface,
    policy: Policy,
    tokenizer,
    master_config: ThunderSyncMasterConfig,
    stream: int = 0,
) -> SpeculativeGeneration:
    """The speculative wrapper of one engine (one per lane with cohort lanes).

    ``stream``: seed stream of this wrapper (the lane index): verification
    randomness is hash(grpo.seed, stream, step, counter) (``run_state``).
    """
    ts_cfg = master_config.thundersync
    return SpeculativeGeneration(
        generation,
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
        null_bv_drop_is=ts_cfg.null_bv_drop_is,
        null_bv_spare_prompts=ts_cfg.null_bv_spare_prompts,
        sticky_chunk_tokens=ts_cfg.sticky_continuation_chunk_tokens,
        sticky_chunk_growth=ts_cfg.sticky_continuation_chunk_growth,
        draft_until_complete=ts_cfg.draft_until_complete,
        draft_weights=ts_cfg.draft_weights,
        banded_is_band=ts_cfg.banded_is_band,
        max_seq_len=master_config.policy["max_total_sequence_length"],
        anchor_horizon=(
            ts_cfg.sticky_anchor_horizon if sticky_delta(ts_cfg) is not None else None
        ),
        learner_prefix_cache=(
            {
                "gpu_gb": ts_cfg.learner_prefix_cache_gpu_gb,
                "max_gb": ts_cfg.learner_prefix_cache_max_gb,
            }
            if ts_cfg.learner_prefix_cache
            else None
        ),
        seed=int(master_config.grpo.seed),
        seed_stream=str(stream),
    )


def thundersync_grpo_train(
    policy: Policy,
    policy_generation: GenerationInterface,
    dataloader,
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: ThunderSyncMasterConfig,
    logger: Optional[Logger] = None,
    on_step: Optional[Callable[[list[dict[str, Any]]], None]] = None,
    lanes: Optional[list[Lane]] = None,
    checkpointer: Optional[Any] = None,
    grpo_save_state: Optional[Any] = None,
    val_dataloader: Optional[Any] = None,
    val_task_to_env: Optional[dict[str, EnvironmentInterface]] = None,
    resume_path: Optional[str] = None,
) -> list[dict[str, Any]]:
    """``on_step(history)`` runs after every completed step (e.g. to persist the
    per-step history, so a run cut short still has every finished step's metrics).
    ``lanes``: the engine lanes (``lanes.build_engine_lanes``) when
    ``thundersync.cohort_lanes`` is 2; ``policy_generation`` is then lane 0.

    Run control (DESIGN.md section 14s, ``run_state.py``): batches come from
    ``grpo.max_num_epochs`` epochs of ``dataloader`` until ``grpo.max_num_steps``;
    checkpoints (``checkpointer``, NeMo-RL's, with ``checkpointing.enabled``)
    and validation (``val_period`` / ``val_at_start`` / ``val_at_end``, plain
    routing) run at quiescent step boundaries; ``resume_path``: the checkpoint
    this run continues from (``run_state.resolve_resume``; setup() already
    loaded its weights and optimizer)."""
    validate_config(master_config)
    ts_cfg = master_config.thundersync
    adv_fn = make_group_advantage_fn(master_config)
    learner = StreamingLearner(policy)
    if (ts_cfg.cohort_lanes > 1) != (lanes is not None):
        raise ValueError(
            f"thundersync.cohort_lanes={ts_cfg.cohort_lanes} but "
            f"{'no' if lanes is None else len(lanes)} engine lanes were built"
        )
    ckpt_cfg = _section(master_config, "checkpointing")
    ckpt_on = checkpointer is not None and bool(ckpt_cfg.get("enabled"))
    resume = load_state(resume_path) if resume_path is not None else None
    if resume is not None:
        if not ckpt_on:
            raise RuntimeError("resume_path given but checkpointing is not enabled")
        check_fingerprint(resume["fingerprint"], config_fingerprint(master_config))
    if lanes is not None and not ts_cfg.stream_training_over_tails:
        if resume is not None:
            raise RuntimeError("resume on the non-streaming lanes loop is not built yet")
        return _train_on_lanes(
            policy, lanes, learner, adv_fn, dataloader, tokenizer, loss_fn,
            task_to_env, master_config, logger, on_step,
        )
    max_steps = master_config.grpo.max_num_steps
    grpo_cfg = master_config.grpo
    bounds = Boundaries(
        max_steps=max_steps,
        save_period=int(ckpt_cfg.get("save_period") or 0) if ckpt_on else 0,
        val_period=int(grpo_cfg.val_period or 0) if val_dataloader is not None else 0,
        val_start_at=int(getattr(grpo_cfg, "val_start_at", 0) or 0),
    )
    timeout = None
    if ckpt_on and ckpt_cfg.get("checkpoint_must_save_by"):
        from nemo_rl.utils.timer import TimeoutChecker

        timeout = TimeoutChecker(timeout=ckpt_cfg["checkpoint_must_save_by"], fit_last_save_time=True)
        timeout.start_iterations()
    history = []
    stale = False  # setup() already performed the initial refit
    step = int(resume["next_step"]) if resume is not None else 0
    spec = None
    sched = None
    sentinel = WeightSentinel(policy, enabled=ts_cfg.weight_sentinel)
    if lanes is not None:
        # Cohort lanes on the streaming loop (DESIGN.md section 14j): one
        # SpeculativeGeneration per lane; each step's cohort runs on lane
        # step % 2 (verification, tails, fresh rows, streaming training) while
        # the other lane drafts the next cohort from the top of the step.
        for lane in lanes:
            lane.generation.prepare_for_generation()
            lane.spec = make_speculative_generation(
                lane.generation, policy, tokenizer, master_config, stream=lane.index
            )
        sched = CohortLanes(lanes)
        spec = lanes[0].spec
    elif ts_cfg.speculate:
        policy_generation.prepare_for_generation()
        spec = make_speculative_generation(
            policy_generation, policy, tokenizer, master_config
        )
    specs = [lane.spec for lane in lanes] if lanes is not None else ([spec] if spec else [])
    for sp in specs:
        sp.sentinel = sentinel
    if resume is not None:
        seeds = resume.get("seeds") or {}
        for i, sp in enumerate(specs):
            if str(i) in seeds:
                sp.load_seed_state(seeds[str(i)])
        if lanes is not None:
            for lane in lanes:
                # Every lane holds theta_start now (setup's refit / the lane build).
                lane.version, lane.cohort = ("theta", step), None
        print(
            f"[thundersync] resumed from {resume_path} at step {step} "
            f"(epoch {resume['data']['epoch']})",
            flush=True,
        )
    if keyed_fresh_enabled(master_config):
        for gen in ([lane.generation for lane in lanes] if lanes is not None else [policy_generation]):
            install_keyed_fresh(gen)
    # The engines' initial weights (setup's refit, lane builds, or a resume)
    # must already be the learner's: checked before step 0 like every refit.
    for gen in ([lane.generation for lane in lanes] if lanes is not None else [policy_generation]):
        sentinel.check(gen, f"initial weights (step {step})")
    check_weights = bool(os.environ.get("THUNDERSYNC_MISMATCH_CHECK"))

    def lane_fp(lane):
        # Debug: the weights the lane's engine ranks hold, read from the engine.
        return sum(
            ray.get(
                lane.generation._policy.worker_group.run_all_workers_single_data(
                    "engine_weight_checksum"
                )
            )
        )
    sticky = spec is not None and spec.sticky_delta is not None
    G = master_config.grpo.num_generations_per_prompt
    multi_turn = sticky and (master_config.grpo.max_rollout_turns or 1) > 1
    if multi_turn:
        # Drafted rollouts finish every turn at the drafting weights; the
        # driver steps the environment at each verified turn end.
        spec.turn_env = TurnEnvironment(
            tokenizer,
            task_to_env,
            master_config.policy["max_total_sequence_length"],
            master_config.grpo.max_rollout_turns,
        )
    if spec is not None and os.environ.get("THUNDERSYNC_SPEC_SELFTEST") and resume is None:
        # Debug only; never on a resume (it would consume the loaded data position).
        first = next(iter(dataloader))
        prompts = [
            torch.cat([m["token_ids"] for m in first["message_log"][i]])
            for i in range(4)
        ]
        spec.draft_budget, budget = 256, spec.draft_budget
        print("[spec selftest]", asyncio.run(spec.self_test(prompts)), flush=True)
        spec.draft_budget = budget
    # Every epoch of the data (grpo.max_num_epochs), each batch tagged with the
    # stream state that yields it (the checkpoint's data position). Prompt drop:
    # every step's batch carries the spares (trained prompts stay
    # grpo.num_prompts_per_step).
    stream = PromptStream(
        dataloader,
        max_epochs=int(grpo_cfg.max_num_epochs or 1),
        rebatch_size=(
            master_config.grpo.num_prompts_per_step + ts_cfg.null_bv_spare_prompts
            if ts_cfg.null_bv_prompt_drop and ts_cfg.null_bv_spare_prompts
            else None
        ),
        state=resume["data"] if resume is not None else None,
    )
    upcoming = collections.deque(
        itertools.islice(stream, max(1, ts_cfg.draft_lookahead) + 1)
    )

    def run_validation(at_step: int) -> dict[str, Any]:
        """Validation of theta_{at_step} at a quiescent boundary, plain routing.

        The engine holds no drafting weights or plans here (the step before a
        boundary drafted nothing); eval requests go to the raw engine (never the
        speculative wrapper), carry no anchors, and the engine applies anchors
        only to requests that carry them, so routing is plain top-k.
        """
        nonlocal stale
        from nemo_rl.algorithms.grpo import validate

        problems = [p for sp in specs for p in sp.boundary_state_problems()]
        if problems:
            raise RuntimeError(f"validation at step {at_step}: engine not quiescent: {problems}")
        if sched is not None:
            gen = sched.cohort_lane(at_step).generation
            lane = sched.cohort_lane(at_step)
            sched.refit(lane, ("theta", at_step), lambda: refit_lane(policy, lane))
            lane.generation.prepare_for_generation()
        else:
            gen = policy_generation
            refit_policy_generation(policy, gen, False)
            stale = False
        sentinel.check(gen, f"validation refit (theta_{at_step})")
        t = time.perf_counter()
        val_metrics, val_timings = validate(
            gen, val_dataloader, tokenizer, val_task_to_env, step=at_step,
            master_config=master_config, logger=logger,
        )
        gen.finish_generation()
        val_metrics = dict(val_metrics)
        val_metrics["time_s"] = time.perf_counter() - t
        print(f"[thundersync validation step {at_step}] {val_metrics}", flush=True)
        if logger is not None:
            logger.log_metrics(val_metrics, at_step, prefix="validation")
            logger.log_metrics(val_timings, at_step, prefix="timing/validation")
        return val_metrics

    def save_boundary(next_step: int, val_metrics, final: bool) -> str:
        """Checkpoint at the quiescent boundary before step ``next_step``."""
        problems = [p for sp in specs for p in sp.boundary_state_problems()]
        if sched is not None:
            problems += [f"lane {ln.index} pinned to cohort {ln.cohort}" for ln in lanes if ln.cohort is not None]
        if problems:
            raise RuntimeError(f"checkpoint before step {next_step}: boundary not quiescent: {problems}")
        t = time.perf_counter()
        data_state = upcoming[0].state if upcoming else stream.state()
        if grpo_save_state is not None:
            grpo_save_state.total_steps = next_step
            grpo_save_state.current_step = next_step
            grpo_save_state.current_epoch = int(data_state["epoch"])
            grpo_save_state.consumed_samples = next_step * int(master_config.grpo.num_prompts_per_step)
            if val_metrics and "accuracy" in val_metrics:
                grpo_save_state.val_reward = val_metrics["accuracy"]
            metric = ckpt_cfg.get("metric_name")
            if metric:
                prefix, name = metric.split(":", 1)
                src = val_metrics if prefix == "val" else (history[-1] if history else {})
                if src and name in src:
                    setattr(grpo_save_state, metric, src[name])
        info = vars(grpo_save_state) if grpo_save_state is not None else {"total_steps": next_step}
        checkpointer.finalize_pending()
        path = checkpointer.init_tmp_checkpoint(next_step, info, master_config)
        policy.save_checkpoint(
            weights_path=os.path.join(path, "policy", "weights"),
            optimizer_path=os.path.join(path, "policy", "optimizer"),
            tokenizer_path=os.path.join(path, "policy", "tokenizer"),
            is_final_checkpoint=final,
        )
        if data_state.get("dl") is not None:
            # NeMo-RL's own resume reads it too (setup's load_dataloader_state).
            torch.save(data_state["dl"], os.path.join(path, "train_dataloader.pt"))
        state = {
            "next_step": next_step,
            "data": data_state,
            "seeds": {str(i): sp.seed_state() for i, sp in enumerate(specs)},
            "lanes": (
                [{"index": ln.index, "version": ln.version} for ln in lanes]
                if lanes is not None
                else None
            ),
            "fingerprint": config_fingerprint(master_config),
        }
        if os.environ.get("THUNDERSYNC_RESUME_DIGEST"):
            state["digest"] = resume_digest(policy, upcoming, stream, specs)
            print(f"[resume digest] save next_step={next_step} {state['digest']}", flush=True)
        save_state(path, state)
        checkpointer.begin_finalization(path, wait_fn=policy.finalize_async_save)
        secs = time.perf_counter() - t
        print(f"[thundersync] checkpoint {path} (next step {next_step}) in {secs:.1f} s", flush=True)
        return path

    if resume is not None and os.environ.get("THUNDERSYNC_RESUME_DIGEST"):
        d = resume_digest(policy, upcoming, stream, specs)
        print(f"[resume digest] load next_step={step} {d}", flush=True)
        saved = resume.get("digest")
        if saved is not None and saved != d:
            raise RuntimeError(f"resume digest differs: saved {saved} loaded {d}")
    if step == 0 and grpo_cfg.val_at_start and val_dataloader is not None:
        run_validation(0)
    drafts = None
    cohort_mode = (
        spec is not None
        and ts_cfg.verify_mode == "block"
        and ts_cfg.draft_lookahead > 1
    )
    lag = ts_cfg.rollout_lag
    pending = None  # rollout_lag=1: rollouts of the next step's batch
    def _prompts(b):
        r = b.repeat_interleave(G)
        return [
            torch.cat([m["token_ids"] for m in r["message_log"][i]])
            for i in range(r.size)
        ]

    while upcoming:
        if step >= max_steps:
            break
        batch = upcoming.popleft().batch
        nb = next(stream, None)
        if nb is not None:
            upcoming.append(nb)
        next_batch = upcoming[0].batch if upcoming else None
        # Drafts made in this step may target steps < draft_limit only: none
        # crosses a checkpoint / validation boundary (run_state.Boundaries).
        draft_limit = bounds.draft_limit(step)
        t0 = time.perf_counter()
        t_verify = 0.0
        t_first = 0.0
        lane_x = lane_y = lane_drafts = None
        lane_stats: dict[str, float] = {}
        if sched is not None:
            lane_x, lane_y = sched.cohort_lane(step), sched.drafting_lane(step)
            spec, policy_generation = lane_x.spec, lane_x.generation
            for lane in lanes:
                lane.spec.current_step = step
            if next_batch is not None and step + 1 < draft_limit:
                # The drafting lane takes cohort step+1's drafter (theta_step, or
                # theta_hat_{step+1}: learner load, refit of this lane only,
                # learner restore) and starts drafting before this step's
                # verification; nothing else is in flight on the learner here.
                momentum = spec.draft_weights != "current"
                version = ("theta_hat", step + 1) if momentum else ("theta", step)

                def put_drafter(y=lane_y, momentum=momentum):
                    if momentum:
                        y.spec.stage_draft_weights(
                            lambda: refit_lane(policy, y), stage_on_engine=False
                        )
                    else:
                        refit_lane(policy, y)
                        sentinel.check(y.generation, f"drafter theta_{step} (lane {y.index})")

                sched.refit(lane_y, version, put_drafter)
                sched.pin(lane_y, step + 1)
                if check_weights:
                    lane_y.cohort_fingerprint = lane_fp(lane_y)
                next_prompts, null_slots = next_draft_prompts(next_batch, ts_cfg, G)
                lane_drafts = asyncio.run(
                    lane_y.spec.draft_streaming(next_prompts, null_slots=null_slots)
                )
                lane_stats["lanes/draft_start_s"] = time.perf_counter() - t0
        if lag and pending is None:
            # Step 0 of the lagged baseline: its rollouts come from theta_0.
            t1 = time.perf_counter()
            policy_generation.prepare_for_generation()
            pending, _ = asyncio.run(
                _generate_rollouts(
                    batch, policy_generation, tokenizer, task_to_env, master_config
                )
            )
            t_first = time.perf_counter() - t1
        if spec is not None:
            spec.current_step = step
        tails = None  # pending plans whose continuation rounds overlap training
        if sticky and drafts is not None:
            # Sticky routing: verify and complete every drafted rollout while
            # the engine still holds the drafting weights (theta_{step-1}, or
            # theta_hat_step with momentum draft weights), whose plain forward
            # supplies the anchors of continuation positions; refit after.
            t1 = time.perf_counter()
            policy_generation.prepare_for_generation()
            if multi_turn:
                rows = batch.repeat_interleave(G)
                spec.bind_rows(
                    [_initial_sample_state(rows, i) for i in range(rows.size)]
                )
            if ts_cfg.stream_training_over_tails:
                # Only the drafts' verification here; the continuation rounds
                # run inside the step, under the training of complete groups.
                tails = asyncio.run(spec.begin_sticky(drafts))
            else:
                asyncio.run(spec.verify_sticky(drafts))
                policy_generation.finish_generation()
            t_verify = time.perf_counter() - t1
            drafts = None
        if ts_cfg.null_bv_prompt_drop and (sticky or drafts is None):
            batch = select_null_prompts(batch, spec, ts_cfg.null_bv_spare_prompts)
        staged = (
            spec is not None
            and sched is None  # lanes stage the drafter on the drafting lane
            and spec.draft_weights != "current"
            and next_batch is not None
            and step + 1 < draft_limit
        )
        # Lanes: rows without a plan (never drafted, or dropped) decode at
        # theta_step on the cohort lane after its tails; decided before any
        # rollout takes a plan.
        lane_fresh = sched is not None and not spec.plans_cover(
            _prompts(batch), pending_ok=tails is not None
        )
        # Sticky: every drafted rollout is (or, streaming, will be) complete.
        # When every rollout of this step is served by a complete plan the
        # engine decodes nothing at theta_k, so it takes theta_hat_{step+1}
        # directly and drafts from the start of the step (no theta_k refit,
        # no host stage).
        draft_now = (
            staged
            and sticky
            and spec.plans_cover(_prompts(batch), pending_ok=tails is not None)
        )
        refit_s = []

        def refit_for_step(staged=staged, draft_now=draft_now):
            if sched is not None:
                # The cohort lane is free once its cohort's tails are done; it
                # takes theta_step only if some row decodes fresh.
                t_r = time.perf_counter()
                lane_stats["lanes/tail_end_s"] = t_r - t0
                if "lanes/draft_start_s" in lane_stats:
                    # Drafting of the next cohort that ran under this cohort's tails.
                    lane_stats["lanes/overlap_s"] = t_r - t0 - lane_stats["lanes/draft_start_s"]
                if lane_x.cohort == step:
                    if check_weights and lane_x.cohort_fingerprint is not None:
                        lane_stats.update(
                            sched._check_fingerprint(lane_x, step, lane_fp(lane_x))
                        )
                    sched.unpin(lane_x, step)
                if lane_fresh:
                    sched.refit(lane_x, ("theta", step), lambda: refit_lane(policy, lane_x))
                    sentinel.check(lane_x.generation, f"theta_{step} (lane {lane_x.index})")
                else:
                    lane_x.generation.prepare_for_generation()
                refit_s.append(time.perf_counter() - t_r)
                lane_stats["lanes/fresh_refit"] = float(lane_fresh)
                return
            if staged:
                # theta_hat_{step+1} needs only the state after the last
                # update; the learner and the engine are idle (streaming: the
                # actor queue orders it between training micro-batches).
                spec.stage_draft_weights(
                    lambda: refit_policy_generation(policy, policy_generation, False),
                    stage_on_engine=not draft_now,
                )
            t_r = time.perf_counter()
            if (stale or staged) and not draft_now:
                refit_policy_generation(policy, policy_generation, False)
                sentinel.check(policy_generation, f"theta_{step}")
            else:
                policy_generation.prepare_for_generation()
            refit_s.append(time.perf_counter() - t_r)

        tails_s = []
        group_order = group_blocks = tail_groups = None
        if tails is None:
            refit_for_step()
        else:
            # Groups whose rows still decode tails (a pending plan of their
            # prompt), fixed by the drafts' verification, not by timing.
            # Incomplete after verify0: continuation rounds on the drafting
            # version, or (anchor horizon) decoded by the refit engine at theta_k.
            pending_keys = {
                key for key, queue in spec.plans.items() for plan in queue if not plan.complete
            }
            row_keys = [tuple(x.tolist()) for x in _prompts(batch)]
            tail_groups = [
                p for p in range(len(row_keys) // G) if row_keys[p * G] in pending_keys
            ]
            if ts_cfg.canonical_group_order:
                # Ready now, then after their tails, then rows never drafted
                # (decoded at theta_k once the tails are done and it is refit).
                n_plans = collections.Counter()
                for key, queue in spec.plans.items():
                    n_plans[key] += len(queue)
                need = collections.Counter(row_keys)
                fresh = {
                    p for p in range(len(row_keys) // G)
                    if n_plans[row_keys[p * G]] < need[row_keys[p * G]]
                }
                late = set(tail_groups) | fresh
                classes = [
                    [p for p in range(len(row_keys) // G) if p not in late],
                    [p for p in tail_groups if p not in fresh],
                    sorted(fresh),
                ]
                group_order = [p for c in classes for p in c]
                # Blocks of canonical_block_groups within each class.
                bg = max(1, int(ts_cfg.canonical_block_groups))
                group_blocks = [
                    min(bg, len(c) - i) for c in classes for i in range(0, len(c), bg)
                ]

            async def tails_then_refit(pending=tails):
                # Continuation rounds on the drafting weights, then theta_k:
                # rollouts never drafted and the next cohort's drafts wait for it.
                t = time.perf_counter()
                try:
                    await spec.complete_tails(pending)
                    policy_generation.finish_generation()
                    tails_s.append(time.perf_counter() - t)
                    # The momentum stage and the refit call the learner: no
                    # training dispatch may be in flight meanwhile.
                    # The momentum stage and the refit are unordered learner
                    # calls: every ordered call sent so far finishes first, and
                    # none is sent while they run (they block the event loop).
                    await spec.learner_drain()
                    refit_for_step()
                except BaseException as exc:
                    spec.weights_ready(error=exc)
                    raise
                spec.weights_ready()
        pre_rollout = None if tails is None else tails_then_refit
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
            if ts_cfg.null_bv_prompt_drop:
                batch = select_null_prompts(batch, spec, ts_cfg.null_bv_spare_prompts)
        drafter = None

        if cohort_mode:
            ahead = list(upcoming)[: ts_cfg.draft_lookahead]
            targets = new_cohort_targets(
                step, len(ahead), draft_limit, existing=spec.cohorts.keys()
            )
            new = {t: _prompts(ahead[t - step - 1].batch) for t in targets}
            drafter = lambda new=new, step=step: spec.draft_cohorts(step, new)  # noqa: E731
        elif spec is not None and next_batch is not None and step + 1 < draft_limit:
            next_prompts, null_slots = next_draft_prompts(next_batch, ts_cfg, G)
            drafter = (  # noqa: E731
                (
                    lambda prompts=next_prompts, slots=null_slots: spec.draft_streaming(
                        prompts, null_slots=slots
                    )
                )
                if ts_cfg.verify_mode == "block"
                else (lambda prompts=next_prompts: spec.draft(prompts))
            )
        if sched is not None:
            # The next cohort's drafts already run on the drafting lane.
            if lane_drafts is not None:

                async def drafter(collector=lane_drafts):
                    return collector

            else:
                drafter = None
        policy.prepare_for_training()
        metrics = asyncio.run(
            _run_streaming_tails(spec, tails is not None)(
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
                before_finish=(
                    lane_y.spec.score_prev
                    if sched is not None
                    else (spec.score_prev if spec is not None else None)
                ),
                pre_rollout=pre_rollout,
                claim_fused=spec.claim_fused
                if spec is not None and ts_cfg.fused_verify_train
                else None,
                claim_anchors=spec.claim_anchors if sticky else None,
                claim_row=spec.claim_row if spec is not None else None,
                tail_count=spec.tail_count if spec is not None else None,
                replay_row=spec.replay_row if multi_turn else None,
                draft_swap=(
                    spec.load_staged_draft_weights if staged and not draft_now else None
                ),
                draft_start_frac=0.0 if (draft_now or sched is not None) else None,
                drafts_after_pre_rollout=tails is not None and sched is None,
                group_order=group_order,
                group_blocks=group_blocks,
                tail_groups=tail_groups,
                ordered_learner=spec if tails is not None else None,
                claim_log_weight=spec.claim_log_weight
                if spec is not None and ts_cfg.banded_is_band is not None
                else None,
                stale_rollouts=pending,
                generate_next=(
                    (
                        lambda b=next_batch: _generate_rollouts(
                            b, policy_generation, tokenizer, task_to_env, master_config
                        )
                    )
                    if lag and next_batch is not None and step + 1 < max_steps
                    else None
                ),
            ))
        )
        t_refit = refit_s[0]
        if tails is not None:
            # Every learner rank ran the same ordered calls in the same order.
            digests = ray.get(
                policy.worker_group.run_all_workers_single_data("ordered_digest")
            )
            if len(set(map(tuple, digests))) != 1:
                raise RuntimeError(f"learner ranks ran different call sequences: {digests}")
            metrics["learner/ordered_calls"] = float(digests[0][0])
            busy = ray.get(policy.worker_group.run_all_workers_single_data("ordered_busy"))
            for method in sorted({m for b in busy for m in b}):
                # Max over ranks: learner seconds in training vs piece verification.
                metrics[f"learner/busy_{method}_s"] = max(b.get(method, 0.0) for b in busy)
        if tails_s:
            # Continuation rounds, overlapped with this step's training.
            metrics["time/tails"] = tails_s[0]
        drafts = metrics.pop("_drafts")
        if lag:
            pending = metrics.pop("_next_rollouts", None)
            metrics["time/first_rollout"] = t_first
        if sticky:
            # This step's training has replayed every anchor targeted at it.
            spec.release_routes(step)
        metrics["time/verify"] = t_verify
        if spec is not None:
            metrics.update(spec.stats)
            spec.stats = {}
        if sched is not None:
            metrics.update(lane_y.spec.stats)
            lane_y.spec.stats = {}
            lane_stats["lanes/cohort_lane"] = float(lane_x.index)
            lane_stats["lanes/refits"] = float(sum(lane.refits for lane in lanes))
            metrics.update(lane_stats)
        policy_generation.finish_generation()
        stale = True
        metrics["time/refit"] = t_refit
        metrics["time/iteration"] = time.perf_counter() - t0
        if os.environ.get("THUNDERSYNC_ENGINE_PROF"):
            # Debug: engine step timing of this step (rank 0 logged; all printed).
            per_rank = ray.get(
                policy_generation._policy.worker_group.run_all_workers_single_data(
                    "engine_step_stats"
                )
            )
            for r, st in enumerate(per_rank):
                print(f"[engine prof] step={step} rank={r} {st}", flush=True)
            metrics.update(per_rank[0] or {})
        # Plans left by this step were verified against its policy only.
        metrics["spec/plans_dropped"] = float(sum(sp.end_step() for sp in specs))
        if os.environ.get("THUNDERSYNC_STEP_DIGEST"):
            # Debug (resume tests): bitwise digests of the weights after this
            # step's update (per rank, folded) and of this step's rollouts.
            ranks = _run_workers(policy, "resume_state_digest")
            metrics["digest/params"] = "|".join(r["params"] for r in ranks)
            metrics["digest/optimizer"] = "|".join(str(r.get("optimizer")) for r in ranks)
        metrics.update(sentinel.pop_stats())
        metrics["step"] = step
        metrics["data/epoch"] = float(stream.epoch if not upcoming else upcoming[0].state["epoch"])
        history.append(metrics)
        _log_step(step, metrics, logger)
        if on_step is not None:
            on_step(history)
        step += 1
        if timeout is not None:
            timeout.mark_iteration()
            if not bounds.is_boundary(step - 1) and timeout.check_save():
                # The next step drafts nothing past itself; save after it.
                bounds.requested.add(step + 1)
        last = step >= max_steps or not upcoming
        val_metrics = None
        if val_dataloader is not None and (
            bounds.validates_after(step - 1) or (last and grpo_cfg.val_at_end)
        ):
            val_metrics = run_validation(step)
        if ckpt_on and (bounds.saves_after(step - 1) or last):
            save_boundary(step, val_metrics, final=last)
    if checkpointer is not None:
        checkpointer.shutdown()
    return history


def resume_digest(policy: Policy, upcoming, stream, specs) -> dict[str, Any]:
    """Debug (THUNDERSYNC_RESUME_DIGEST): the boundary state, bitwise, for resume tests.

    Learner parameter, optimizer-state and momentum-draft digests (per rank,
    no gather), the next batch's prompt tokens, the data position and the seed
    counters. Equal at a save and at the resume that loads it.
    """
    per_rank = _run_workers(policy, "resume_state_digest")
    nxt = upcoming[0].batch if upcoming else None
    prompt_crc = None
    if nxt is not None:
        prompt_crc = zlib.crc32(
            b"".join(
                torch.cat([m["token_ids"] for m in nxt["message_log"][i]]).numpy().tobytes()
                for i in range(nxt.size)
            )
        )
    return {
        "ranks": per_rank,
        "next_prompts_crc": prompt_crc,
        "epoch": upcoming[0].state["epoch"] if upcoming else stream.epoch,
        "seeds": [sp.seed_state() for sp in specs],
    }


def _run_streaming_tails(spec, on: bool):
    """Run a step coroutine with ``spec``'s streaming-over-tails state (if ``on``)."""

    async def run(coro):
        if not on:
            return await coro
        spec.begin_streaming_tails()
        try:
            return await coro
        finally:
            spec.end_streaming_tails()

    return run


def _log_step(step: int, metrics: dict[str, Any], logger: Optional[Logger]) -> None:
    scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    print(
        f"[thundersync step {step}] "
        + " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in scalars.items()
        ),
        flush=True,
    )
    print(
        f"[thundersync step {step} done] {time.strftime('%Y-%m-%dT%H:%M:%S')}",
        flush=True,
    )
    if logger is not None:
        logger.log_metrics(scalars, step + 1, prefix="thundersync")


def _train_on_lanes(
    policy: Policy,
    lanes: list[Lane],
    learner: StreamingLearner,
    adv_fn,
    dataloader,
    tokenizer,
    loss_fn: LossFunction,
    task_to_env: dict[str, EnvironmentInterface],
    master_config: ThunderSyncMasterConfig,
    logger: Optional[Logger],
    on_step: Optional[Callable[[list[dict[str, Any]]], None]] = None,
) -> list[dict[str, Any]]:
    """Training loop on cohort lanes (lanes.py, DESIGN.md section 14j).

    Same algorithm as ``thundersync_grpo_train`` with sticky routing and
    lookahead 1; only where and when the engine work runs differs: cohort
    step+1 is drafted on the other lane from the top of step ``step``
    (``CohortLanes.step``), not after cohort ``step``'s continuations.
    """
    ts_cfg = master_config.thundersync
    G = master_config.grpo.num_generations_per_prompt
    max_steps = master_config.grpo.max_num_steps
    momentum = ts_cfg.draft_weights != "current"
    sentinel = WeightSentinel(policy, enabled=ts_cfg.weight_sentinel)
    for lane in lanes:
        lane.generation.prepare_for_generation()
        lane.spec = make_speculative_generation(
            lane.generation, policy, tokenizer, master_config, stream=lane.index
        )
        lane.spec.sentinel = sentinel
        sentinel.check(lane.generation, f"initial weights (lane {lane.index})")
    sched = CohortLanes(lanes)
    check_weights = bool(os.environ.get("THUNDERSYNC_MISMATCH_CHECK"))

    def checked_refit(lane):
        refit_lane(policy, lane)
        sentinel.check(lane.generation, f"refit lane {lane.index}")

    def lane_fingerprint(lane):
        # Debug: the weights the lane's engine ranks hold, read from the engine.
        return sum(
            ray.get(
                lane.generation._policy.worker_group.run_all_workers_single_data(
                    "engine_weight_checksum"
                )
            )
        )

    batches = (t.batch for t in PromptStream(dataloader, int(master_config.grpo.max_num_epochs or 1)))
    batch, next_batch = next(batches, None), next(batches, None)
    drafts = None
    history = []
    step = 0
    while batch is not None and step < max_steps:
        t0 = time.perf_counter()
        x, y = sched.cohort_lane(step), sched.drafting_lane(step)
        for lane in lanes:
            lane.spec.current_step = step
        times = collections.Counter()

        def timed(key, fn):
            t = time.perf_counter()
            try:
                return fn()
            finally:
                times[key] += time.perf_counter() - t

        def put_drafter(lane, target, refit):
            if momentum:
                # theta_hat_{step+1} from (theta_step, m, v): learner load,
                # refit of this lane alone, learner restore; no host stage.
                lane.spec.stage_draft_weights(refit, stage_on_engine=False)
            else:
                refit()

        def start_drafts(lane, target, nb=next_batch):
            prompts, slots = next_draft_prompts(nb, ts_cfg, G)
            return asyncio.run(lane.spec.draft_streaming(prompts, null_slots=slots))

        def complete_cohort(lane, cohort_drafts):
            lane.generation.prepare_for_generation()
            timed("time/verify", lambda: asyncio.run(lane.spec.verify_sticky(cohort_drafts)))

        def run_step(lane, collector, b=batch, y=y):
            async def drafter():
                return collector

            policy.prepare_for_training()
            return asyncio.run(
                _run_one_step(
                    step=step,
                    batch=b,
                    learner=learner,
                    policy_generation=lane.spec,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    loss_fn=loss_fn,
                    master_config=master_config,
                    ts_cfg=ts_cfg,
                    adv_fn=adv_fn,
                    drafter=drafter if collector is not None else None,
                    before_finish=y.spec.score_prev,
                    claim_anchors=lane.spec.claim_anchors,
                    draft_start_frac=0.0,
                )
            )

        def needs_current(lane, b=batch):
            # Rollouts without a complete plan decode at theta_step on the lane.
            prompts = [
                torch.cat([m["token_ids"] for m in r])
                for r in b.repeat_interleave(G)["message_log"]
            ]
            return not lane.spec.plans_cover(prompts)

        ops = LaneStepOps(
            drafter_version=lambda t: ("theta_hat", t) if momentum else ("theta", t - 1),
            put_drafter=put_drafter,
            refit_lane=lambda lane: timed("time/refit", lambda: checked_refit(lane)),
            start_drafts=start_drafts,
            complete_cohort=complete_cohort,
            needs_current=needs_current,
            run_step=run_step,
            fingerprint=lane_fingerprint if check_weights else None,
        )
        draft_next = next_batch is not None and step + 1 < max_steps
        metrics, drafts = sched.step(step, drafts, draft_next, ops)
        # This step's training has replayed every anchor targeted at it.
        x.spec.release_routes(step)
        metrics["spec/plans_dropped"] = float(sum(lane.spec.end_step() for lane in lanes))
        metrics.update(sentinel.pop_stats())
        metrics["time/verify"] = times["time/verify"]
        metrics["time/refit"] = times["time/refit"]
        for lane in (x, y):
            metrics.update(lane.spec.stats)
            lane.spec.stats = {}
        metrics["lanes/refits"] = float(sum(lane.refits for lane in lanes))
        metrics["time/iteration"] = time.perf_counter() - t0
        history.append(metrics)
        _log_step(step, metrics, logger)
        if on_step is not None:
            on_step(history)
        batch, next_batch = next_batch, next(batches, None)
        step += 1
    return history
