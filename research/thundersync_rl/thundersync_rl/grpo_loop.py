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
import concurrent.futures
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
from thundersync_rl.health import annotate as step_health_annotate, strict as health_strict
from thundersync_rl import leak_probe
from thundersync_rl.aep import AnchorEpoch
from thundersync_rl.async_lanes import bounded_side_hold, side_hold_limit_s
from thundersync_rl.checks_sched import STEP_TOP_KEYS, CheckSchedule, step_decomposition
from thundersync_rl.la2 import AdaptiveLookahead, Lookahead2, cohort_version
from thundersync_rl.lanes import CohortLanes, Lane, LaneStepOps, refit_lane
from thundersync_rl.speculative import EngineVerifier
from thundersync_rl.multi_turn import PrefixGuardGeneration, TurnEnvironment
from thundersync_rl.null_bv import drafts_per_prompt
from thundersync_rl.routes import RouteRefs
from thundersync_rl.run_state import (
    Boundaries,
    PromptStream,
    TimedSave,
    cohort_lookahead,
    check_fingerprint,
    config_fingerprint,
    load_state,
    save_state,
)
from thundersync_rl.sentinel import WeightSentinel
from thundersync_rl import lookahead_draft
from thundersync_rl.speculative import SpeculativeGeneration, _debug_forced_fresh, cohort_draft
from thundersync_rl.verify0_stream import StreamOrder, Verify0Risk
from thundersync_rl import vit
from thundersync_rl.window_pace import share_window_pacer
from thundersync_rl.streaming import (
    Chunk,
    StreamingLearner,
    StreamPlanner,
    Trajectory,
    microbatch_counts,
    pad_batch_to_multiple,
    tail_excluded_advantages,
)
from thundersync_rl.envflag import env_flag


class ThunderSyncConfig(BaseModel, extra="allow"):
    """Gradient-streaming settings (the ``thundersync:`` YAML block)."""

    # Driver-process environment set at startup from the config (driver_env.py): the env-var switches of a
    # configuration (configs/paper/*.yaml) so one config file gives the behavior. Empty = nothing set.
    driver_env: dict[str, str] = {}
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
    # k >= 1 with ``async_lanes`` (lanes-k async, async_lanes.py): the rollouts trained at step k were sampled at
    # theta_{k-lag} on engine lane k % lag (fixed age = lag; the first lag cohorts at theta_0).
    rollout_lag: int = 0
    # Lanes-k async baseline (async_lanes.py, research/lanesk/README.md; opt-in, off = today's paths): the engine
    # fleet is split into ``cohort_lanes`` = ``rollout_lag`` = k engines, one weight version each; lane j % k
    # generates cohort j + k from theta_j as soon as its cohort j is complete (mid-step refit between learner
    # dispatches), so k batches are in flight and training streams each cohort's rows as they finish. Same
    # learner path, loss and IS correction as rollout_lag=1 (async1); speculate must be off. k = 1 runs the
    # async1 version schedule on this path (one lane = the setup engine): the a1 equivalence gate.
    async_lanes: bool = False
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
    # Deal the rows of each training dispatch over every learner DP rank, one
    # micro-batch per unit (StreamPlanner deal_rows): each rank
    # runs ceil(rows / (DP * MBS)) micro-batches, so under expert parallelism
    # no rank pads to a busier one with all-masked fillers. False: whole
    # max_chunk_trajectories-row chunks per rank (filler fraction ~1 - rows /
    # (DP * max_chunk_trajectories) per dispatch). Placement only: fp32 order.
    deal_rows: bool = True
    # Pipeline parallelism. Each learner rank runs all chunks of a dispatch that share
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
    # Exact cross-iteration speculative rollouts (speculative.py):
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
    # Where q (the drafts' sampling law) and p come from
    # (engine_q.py). "scorer": a scorer copy of the policy (verify_precision)
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
    # "recompute" (verify_on=engine with cohort lanes): no per-token hidden
    # states are logged; q rows a lazy scan needs are rebuilt by a prefill of
    # the draft on the lane that drafted it (it holds the drafting weights while
    # the cohort's tails run).
    hidden_transport: Literal["object_store", "driver", "recompute"] = "object_store"
    # verify_on=engine: groups per verification block. Blocks are verified in
    # group index order on the engine while the learner trains the groups of
    # earlier blocks (canonical order: each block's ready groups, then the
    # groups with tails, then the groups with fresh rows). None: canonical_block_groups.
    engine_verify_block_groups: Optional[int] = None
    # verify_on=engine: groups in the FIRST verification block (the learner waits
    # for it: pipeline fill). None: engine_verify_block_groups. Blocks are a
    # function of the group index only, so the training order stays a function
    # of the verification results.
    engine_verify_first_block_groups: Optional[int] = None
    # verify_on=engine: submit the next block's first verification wave to the
    # engine (prefill + head-input capture) while the current block is decided,
    # so prefill and decisions overlap. Timing only (the rows, keys' randomness
    # and decisions are those of the unpipelined path). Per verifying rank: the
    # captured rows of up to two blocks in flight ((g+1) d b_h bytes per row).
    engine_verify_prefetch: bool = False
    # Iterations drafted concurrently. A draft cut off at one deadline resumes
    # from its prefix, under the newer weights, in the next iteration.
    draft_lookahead: int = 3
    # Block mode: verification chunks, each publishing its plans when done.
    verify_chunks: int = 4
    # Tokens per scorer forward batch (bounds logits memory; lower it for large
    # vocabularies or when verification shares the learner GPUs, e.g. 4096 at 4B).
    verify_batch_tokens: int = 16384
    # verify0 only (the step-top block verification of every draft, one-shot or
    # streamed): padded tokens per learner forward. Whole 8k drafts
    # go one per 16384-token forward and the learner forward is launch-bound
    # (~0.4 s per call at full Nano), so packing more rows per forward cuts
    # verify0. Packing only (same rows, order and random stream on every rank;
    # batch-invariant kernels): exact for any value. Per-rank cost: one no-grad
    # forward's transient activations for this many tokens (the fp16-head
    # verification head is chunked, 4096 tokens; at TP > 1 the head is
    # materialized: + tokens x vocab/TP x 4 B). None: verify_batch_tokens.
    verify0_batch_tokens: Optional[int] = None
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
    # "engine" (engine_verify.py): verification of drafts and
    # continuation pieces runs on an engine holding theta_k (a prefill of the
    # row with the drafts' anchors applied, p from the engine's own LM head on
    # its captured head inputs); the learner only trains. Needs
    # verify_source=engine; with sticky routing it needs cohort_lanes=2 and
    # draft_weights=current (the drafting lane holds theta_k for the step).
    verify_on: Literal["learner", "inference", "both", "engine"] = "both"
    # Run verification concurrently with the rollouts (rollouts start as their
    # plans are published) also when verify_on != "learner".
    verify_overlap: bool = True
    # Chunks per pool in the round-robin, in verify_on order (learner, inference).
    verify_pool_weights: Optional[list[int]] = None
    # Start drafting when this iteration's verification finishes, instead of at
    # draft_start_frac of completed rollouts.
    draft_after_verify: bool = False
    # EXPERIMENTAL, untested on GPU (fused_train.py):
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
    # sticky_routing.py, routes.py). The policy of step
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
    # Exact top-p (nemo_rl/models/megatron/exact_nucleus.py). The policy is softmax restricted to the canonical nucleus of
    # policy.generation.top_p (integer-mass sort, ties by id), masked bit for bit
    # the same way by the engine sampler and every learner log-prob (training,
    # verification p and q rows). Block verification against the truncated
    # target is exact (tokens outside the target nucleus: p = 0, rejected).
    # Requires batch_invariant_mode, temperature 1, top_k unset, KL 0.
    exact_top_p: bool = False
    # Sticky nucleus (opt-in, needs exact_top_p + sticky_routing): the target's
    # nucleus is the drafter's logged nucleus A when A is delta-feasible under
    # the target (mass(A) >= P - delta and mass(A) - min_A < P + delta, mass
    # units), else the plain nucleus; A has at most ``sticky_nucleus_cap`` ids
    # (a larger drafter nucleus is "self"). Removes nucleus-boundary flips
    # (full Nano offline: whole@32k .56 plain -> .89).
    sticky_nucleus: bool = False
    sticky_nucleus_delta: float = 0.01
    sticky_nucleus_cap: int = 64
    # 0.03 saturates on Qwen3-30B-A3B (full 48 layers, real GPU numerics: same
    # per-token variance as 0.1 and pure replay; 0.003 keeps 66% vs 73% of
    # 32k drafts; ACCEPTANCE_LEDGER section T). None disables sticky routing.
    sticky_routing_delta: Optional[float] = 0.03
    # Cross-fade MoE routing (nemo_rl/models/megatron/crossfade_routing.py; opt-in, exclusive with
    # sticky_routing): a θ-only rule, continuous in the router scores, so a rejected or cut row continues as a
    # plain sample of the current weights on the one engine (no drafter anchors, no lanes;
    # the research notes, summarized in docs/colo/). None: off. m is the band width in selection-score
    # units (Super .003 or .01, Nano .01: research/nolanes/gates); extra = candidates beyond the stock top-k
    # (moe_router_topk becomes k + extra: fixed-width dispatch, zero weight outside the band).
    crossfade_routing_m: Optional[float] = None
    crossfade_routing_extra: int = 4
    # Learner sparse dispatch (nolanes P2): the learner routes only the nonzero-weight cross-fade slots (the engine
    # keeps the fixed k + extra width); the same bits, fewer expert rows and all-to-all bytes (c/k -> e/k).
    crossfade_routing_sparse: bool = False
    # Learner gradient ignores the cross-fade ramp (c, tau): the sticky-like estimator; forward / sampling unchanged.
    crossfade_routing_ramp_stopgrad: bool = False
    # One engine, θ-only routing (cross-fade / plain): start drafting the next cohort (on theta_k) at the top of the
    # step instead of after the first finished rollout (nolanes P2: a longer draft window; the start time is exogenous,
    # the cut stays the step deadline).
    draft_at_step_top: bool = False
    # Sticky fp16 rounding (nemo_rl/models/megatron/sticky_rounding.py;
    # opt-in, not part of the default recipe). At every
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
    # Pooled null-augmented block verification (⊥-BV; null_bv.py). The engine drafts N = max(G, round(K G)) drafts per GRPO
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
    # Calibrated M_j policy: the DP's per-token Var log p/q is a moving
    # average (weight null_bv_calibration_decay on the past) of the variance
    # measured on EARLIER steps' verified drafts (null_bv_logratio_var until
    # the first step; v drifts as the policy sharpens). Exact: within a step v
    # is fixed before its drafts are verified, so M_j still depends on the
    # counts only (null_bv.LogRatioCalibration). Logs spec/null_v_used and
    # spec/logratio_var_step. null_bv_tail_weight: critical-path cost of a
    # tail decode relative to a fresh decode of a slot (DP cost model).
    # ⊥-BV draft admission order: "group" (whole pools, group by group;
    # default: at max_requests = primaries the early groups finish fully
    # drafted, 0.56-0.69 of groups at Nano L8 1k vs 0.31-0.50) or
    # "primaries_first" (every group's G primaries, then the extras).
    # "rounds": the primaries, then extra 1 of every group, extra 2 of every
    # group, ... (the extras a BOT calls for are the early ones of each pool).
    # "fit": the "rounds" order cut to the drafting engine's
    # concurrent capacity (max_requests x engine DP) before submission, so
    # every submitted draft runs from the top of the window: no queued drafts,
    # no undrafted slots (the 74-88 s fresh decodes after the tails), and
    # which drafts exist is fixed by counts and capacity before drafting.
    # Exact either way (counts and batch order only); logged:
    # spec/null_groups_fully_drafted, spec/null_groups_whole_accepted,
    # draft/submitted, draft/fit_cut_extras.
    # "auto" (default since ts-integ5): "fit" with cohort lanes
    # (cohort_lanes=2) and ⊥-BV unless null_bv_prompt_drop (fit is not built
    # with it), else "group". Queued admission (group, primaries_first, and
    # rounds whenever drafts exceed capacity) lets a later draft of a pool
    # decide an earlier one's presence through EP step timing (enumeration TV
    # 0.04-0.07, test_draft_capacity.py); fit fixes presence before drafting.
    # group / primaries_first / rounds stay selectable.
    draft_admission: str = "auto"
    null_bv_calibrate: bool = False
    null_bv_calibration_decay: float = 0.5
    null_bv_tail_weight: float = 1.0
    # Opt-in: constant
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
    # decode per kept token).
    sticky_continuation_chunk_tokens: int = 128
    # Upper bound on tokens per training micro-batch (rows x
    # policy.max_total_sequence_length): rows per micro-batch =
    # min(policy.train_micro_batch_size, this // max_total_sequence_length), >= 1.
    # Bounds activation memory per learner rank by tokens, not rows (an 8 x 10k
    # micro-batch OOMed Nano L8). None: rows only.
    train_micro_batch_tokens: Optional[int] = 32768
    # Memory budget of that cap (mem_budget.py): with train_mem_bytes_per_token set, startup raises unless
    # rows x max_total_sequence_length <= floor((device - nontorch - margin - fixed) / bytes_per_token), with
    # device = the smallest learner rank's memory and the other constants measured for the model / layout
    # (learner/mem_peak_gb vs learner/max_mb_tokens; learner/mem_nontorch_gb after a checkpoint save). Each
    # step logs learner/mem_cap_fit_tokens recomputed from its own peak. None: no check.
    train_mem_bytes_per_token: Optional[float] = None
    train_mem_fixed_gb: Optional[float] = None
    train_mem_nontorch_gb: Optional[float] = None
    train_mem_margin_gb: float = 5.0
    # Continuation piece size grows with the row's generated position:
    # max(chunk_tokens, growth * position) (SpeculativeGeneration._piece_tokens).
    # Each piece costs one forward of the whole row on the learner, so fixed
    # pieces are O(L^2 / chunk) per continued row; growth 1.0 keeps <= 7 pieces
    # per 8k row. 0: fixed pieces.
    sticky_continuation_chunk_growth: float = 1.0
    # Run cap: once a continuation piece of a row is rejected, its later pieces are at most
    # max(128, ceil(f * decided tokens / rejections)) tokens (f mean accepted runs; SpeculativeGeneration.
    # _piece_tokens). Without it, large chunks (16384: every piece the whole remainder) and growth cost a
    # whole-remainder decode + verification per rejection, quadratic in the row at low acceptance (a forced
    # lr 3e-4 Super-L8 step never finished). Exact (decided history only). 0: off.
    sticky_continuation_run_factor: float = 4.0
    # Continuation piece verifications sent while a training dispatch runs are
    # run at that dispatch's next micro-batch boundary instead of after it
    # (verify_yield.py). Exact: training runs the same
    # micro-batches with the same loss scaling, verification results do not
    # depend on when they run; only timing changes. Learner path while
    # streaming over the tails; pipeline parallel size 1 (PP > 1: posted calls
    # run in order).
    verify_yield: bool = False
    # Carried-walk continuation verification (carried_walk.py).
    # "every": every piece is its own block (decided at its end; the walk restarts).
    # "row": the pieces of a stream form one block, decided only at the stream's end
    # (the one-block law). "dip": also decided at a piece end where the walk has
    # dipped by -log b >= sticky_continuation_decision_dip. Forward pieces keep the
    # chunk/growth sizes above; decisions are a separate schedule. All exact.
    sticky_continuation_decision: str = "every"
    sticky_continuation_decision_dip: float = 0.1
    # Continuation pieces in flight per stream: the next piece is cut and sent
    # while earlier ones are verified (assuming they are accepted whole; a
    # rejection discards the later results). 1: one at a time (the earlier
    # behaviour). Bounds a stream's stall on verification latency.
    sticky_pieces_in_flight: int = 1
    # Learner prefix-state cache (nemo_rl/models/megatron/
    # prefix_state_cache.py). Each continuation piece is verified by a forward
    # of positions [last Mamba chunk boundary before the piece, end of piece)
    # from the row's cached attention K/V and Mamba conv/SSM state, instead of
    # the whole row: per continued row O(L) learner work instead of
    # O(L^2 / piece). Bitwise the whole-row forward's logits (batch
    # invariance). The driver keeps a row on the learner rank that holds its
    # state. Per learner rank: device residency gpu_gb (colder rows wait in
    # pinned host memory), total budget at most max_gb: size "auto" = this
    # step's continued rows per rank x one row's bytes at the row cap; policy
    # "pin" admits a row only if its reservation fits and never evicts (LRU
    # thrashes under cyclic rounds: every access misses); uncached rows
    # forward whole (slower, same bits). min_hit_frac > 0 raises when the hit
    # fraction of re-verified rows falls below it (a warning is always
    # printed below 0.9 / on denials). Requires sticky_routing,
    # verify_source=engine, verify_on=learner, no sticky_rounding, no context
    # parallelism, a model without RoPE (Nemotron-H; RoPE offsets not built
    # yet). Off until validated.
    learner_prefix_cache: bool = False
    learner_prefix_cache_gpu_gb: float = 8.0
    learner_prefix_cache_max_gb: float = 32.0
    learner_prefix_cache_size: str = "auto"
    learner_prefix_cache_policy: str = "pin"
    learner_prefix_cache_min_hit_frac: float = 0.0
    # verify0 capture (ts-v0cache): verify0's whole-row learner forwards keep
    # the rows' states (from the prompt's chunk boundary on), so a continued row's first
    # piece starts from its cached boundary instead of re-forwarding prompt + prefix
    # (~40% of verified tokens at full Nano 8k). Captures are admitted against
    # learner_prefix_cache_max_gb before the decision, then the continuing rows are kept
    # (trimmed to one boundary) and the rest released. Same bits (the same hooked forward a
    # first piece runs; THUNDERSYNC_PREFIX_CACHE_CHECK checks it). Needs learner_prefix_cache;
    # not with multi-turn rollouts or the anchor horizon. Off until the A/B decides.
    learner_prefix_cache_verify0: bool = False
    # Sticky, one engine: serve and train every rollout whose plan is complete
    # while the continuation rounds of the other rows still run (both are
    # learner-side at theta_k). False: training starts
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
    # With deal_rows a block is raised to at least ceil(DP / G) groups so every
    # learner DP rank gets a row (canonical_groups_per_block).
    canonical_block_groups: int = 4
    # Canonical order of the TAIL class (groups with a pending continuation after
    # verify0). "index": group index order (the slowest row's group
    # blocks every later tail block: at full Nano 8k 7-9 tail dispatches, 20-41 s,
    # ran after the chain ended, bh-ab). "kept": by the fewest tokens kept at the
    # verify0 decision among the group's tail plans, most kept first (least
    # remaining decode), ties by index. A function of the verification results
    # only, so the accumulated gradient stays bitwise independent of arrival
    # times; it changes which groups share a dispatch (summation order), not
    # what is trained.
    tail_group_order: str = "index"
    # Set: canonical blocks are cut by tokens instead (a block ends at the
    # first group at which its rows reach this many tokens, within each class
    # of the canonical order), so a dispatch's size does not depend on the
    # response length; canonical_block_groups is then unused.
    canonical_block_tokens: Optional[int] = None
    # Risk-first streaming verify0 (opt-in). verify0 runs inside
    # the step as ordered learner calls over whole prompts, riskiest first (a
    # past-only prediction: draft length bucket x earlier steps' whole
    # acceptance), and each prompt's plans are published at its decision:
    # its tails start then and its ready groups train between the calls.
    # Calls are cut by forward tokens (first call smaller, so the riskiest
    # tails start early); ``window`` calls are queued on the learner at once.
    verify0_stream: bool = False
    verify0_stream_first_chunk_tokens: int = 262144
    verify0_stream_chunk_tokens: int = 524288
    verify0_stream_window: int = 2
    # "risk" (past-only risk score, riskiest first) | "drafting" (drafting order; the control).
    verify0_stream_order: str = "risk"
    # Measurement only: the drafting deadline waits
    # until every draft finished (EOS or the cap), so each step verifies whole
    # rollouts and its post-refit tail is the rejection tail alone. Exact (no
    # draft is cut, so none is truncated by content); serializes drafting
    # after training, so never for step-time runs. Requires sticky_routing.
    draft_until_complete: bool = False
    # Drafting-window pacing (window_pace.py):
    # the draft cut waits until a target window fitted from earlier cuts
    # (the expected time for every draft to reach the cap, x (1 + margin)),
    # released once no draft runs, and never when that deadline is more than
    # max_wait_frac x target away. Damps the fast/slow step alternation (a
    # fast step leaves the next cohort a window just short of full drafts).
    # Exact: the deadline uses earlier steps' timings and lengths only.
    # Requires sticky_routing (streams run past the cut).
    draft_window_pace: bool = False
    draft_window_pace_margin: float = 0.05
    draft_window_pace_max_wait_frac: float = 0.3
    # Prior full window (s) before any cut is observed; None: no pacing until
    # a cohort is cut short once.
    draft_window_pace_init_s: Optional[float] = None
    # Sticky draft cut: wait up to this many seconds after training ends for the
    # drafts to finish (the cut is at that fixed time, or earlier once all are
    # complete: exact). 0: cut when training ends.
    draft_cut_slack_s: float = 0.0
    # Sticky routing anchor horizon H (generated positions). None: anchors up to the rollout cap (every drafted rollout is
    # completed at theta_{k-1} before the refit). Set: anchors only at
    # positions < c = prompt + min(H, cap), self beyond; theta_{k-1} work per
    # drafted rollout ends once its anchors below c are logged, and the refit
    # engine samples the rest from p_k directly, re-applying the anchors with
    # the sticky rule in its forwards (engine anchor apply). Bounds theta_{k-1}
    # work (and the refit wait) by H tokens per rollout.
    sticky_anchor_horizon: Optional[int] = None
    # Weights the engine drafts the next step's rollouts with (draft_weights.py). "current": theta_k. "momentum": theta_hat_{k+1} =
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
    # Cohort lanes (lanes.py): 2 splits the engine fleet
    # into two engines of policy.generation.colocated.resources GPUs each
    # (run_thundersync_grpo.py builds the second), each holding one weight
    # version. At step k the lane of cohort k keeps its drafting weights and
    # completes cohort k's continuation streams, while the other lane refits
    # the drafter of cohort k+1 and drafts it from the top of the step; roles
    # alternate. 1: one engine (single version). Requires sticky routing
    # (the lanes carry the continuation anchors; without it the remainder is
    # plain theta_k decoding, not built on lanes yet) and refit_transport null
    # (packed collective) or mcore (native m2n refit, refit_m2n.py).
    cohort_lanes: int = 1
    # Cohort-lane lookahead (la2.py). 1: cohort k+1 drafts
    # on its lane from the top of step k to the cut at the end of step k (one
    # period). 2: the cohort lane of step k, once its cohort is final, refits
    # theta_k (the learner's live weights) and drafts cohort k+2 until the cut at
    # the end of step k+1 (~two periods); cohorts it could not start early fall
    # back to lookahead 1. Requires cohort_lanes=2, stream_training_over_tails,
    # draft_weights=current, verify_on=learner. k > 2 (depth k, lanes-k; la2.py): cohort_lanes=k, the cohort lane
    # of step k' drafts cohort k'+k from theta_k'; same requirements (adaptive lookahead and X5 not built at k > 2).
    cohort_lane_lookahead: int = 1
    # Phase-adaptive lookahead (la2.AdaptiveLookahead; needs cohort_lane_lookahead=2 and null_bv_calibrate):
    # each cohort starts early (lookahead 2) while the forecast age-2 v stays below the hysteresis band and falls
    # back to lookahead 1 above it (switch to 1 above v_hi, back to 2 below v_lo). Scheduling only (exact).
    cohort_lane_lookahead_adaptive: bool = False
    lookahead_adaptive_v_hi: float = 2e-5
    lookahead_adaptive_v_lo: float = 1e-5
    lookahead_adaptive_age_ratio: float = 3.5
    # Anchored deep pipeline, first build AEP-1E (aep.py; opt-in). Set (E >= 1): one engine is
    # loaded with theta_0 and never refit for the run; cohorts 1..E are anchored at theta_0 (routing lag = age up to
    # E), and draft_lookahead = lambda cohorts are drafted on it at once (cohort c from step max(0, c - lambda) to
    # the cut at the end of step c - 1). Target of cohort c: sticky F(theta_c; A_theta_0). Verification: VIT for
    # complete groups (verify_in_train), packed verify0 for the rest; tails continue on theta_0 with learner piece
    # verification; slots a pool cannot fill decode as continuations from the empty prefix. The learner keeps
    # theta_0's LM head. Needs one engine (cohort_lanes 1), sticky routing, ⊥-BV with draft_admission fit,
    # verify_source engine / verify_on learner, streaming over the tails, max_num_steps <= E + 1 (epoch switches
    # are not built), no checkpoint / validation boundaries inside the run. None: off.
    anchor_epoch: Optional[int] = None
    # Anchor weights: "raw" = theta_w itself. "momentum" (opt-in) = Psi_h(theta_w, m_w, v_w), h =
    # anchor_momentum_steps zero-gradient AdamW steps of the optimizer state at the anchor step w (needs w >= 1: at
    # step 0 the moments are zero). The anchor step's own cohort is then anchored too (age 0): the engine never holds
    # a target. anchor_start = w: steps 0..w-1 run the regular one-engine path, step w-1 drafts nothing (quiescent
    # switch), the anchor is put on the engine once at the top of step w.
    anchor_weights: Literal["raw", "momentum"] = "raw"
    anchor_start: int = 0
    anchor_momentum_steps: int = 1
    # Engine slots kept free besides one cohort's outputs when the per-cohort draft budget is computed
    # (aep.AnchorEpoch.draft_budget; debug route-check prefills use up to THUNDERSYNC_ROUTE_CHECK_ROWS).
    anchor_reserve_rows: int = 32
    # Short EP-lockstep fillers. An all-masked filler
    # micro-batch (Chunk.padding: sample_mask = token_mask = advantages = 0) only keeps
    # every rank's forward/backward count equal; it adds exactly zero gradient and no
    # normalization count. False: it is the dispatch's shortest row at full length
    # (in-loop full Nano 8k: 1 x 8448 tokens, 1.5-2.0 s per filler). True: its first
    # policy.make_sequence_length_divisible_by tokens only, and no route anchors
    # (anchor = self). Real rows' log-probs and every gradient are unchanged (the
    # filler's contribution is zero at any length; the forward is batch-invariant).
    short_fillers: bool = False
    # Engine sampler of drafts, fresh rows and continuations (verify_mode block):
    # "wide" (default) = 53-bit keyed Gumbel race in fp64 (keyed_sampling.wide_sample_logits,
    # key = (seed stream, request id, position): batch-invariant, law exact to ~1e-16
    # relative); "engine" = the engine's own fp32 sampler (FlashInfer: no token below
    # ~6e-8 x p_top is drawn; redteam_1016 §6). keyed mode always uses its keyed kernel.
    # "exact_int" (opt-in) = the policy IS the integer law w / W of the fp32
    # log-softmax row (w = floor(exp(l) 2^52), nemo_rl/models/megatron/exact_int.py), sampled
    # exactly (keyed rejection + integer prefix sums), logged and trained as
    # log(w_x) - log(W), and block-verified exactly (thundersync_rl/exact_bv.py).
    engine_sampler: Literal["wide", "engine", "exact_int"] = "wide"
    # Engine sampler of the arms WITHOUT speculation (sync, async1), whose rows decode through
    # plain generate (ported from ts-syncuni 29cca55f): "engine" (default, the
    # engine's FlashInfer fp32 sampler, as before) or "wide" (the 53-bit keyed race of
    # engine_sampler, installed on every engine rank at loop start). Fair baselines in the
    # A/B: ab_async1.yaml / ab_sync.yaml set "wide" on ts-nano-scale. Speculative arms use
    # engine_sampler.
    # "exact_int": the integer law of engine_sampler=exact_int on the plain path (sync / async1:
    # engine sampling + logged lambda; the learner trains lambda), so a baseline can run the
    # same policy definition as the speculative arms.
    plain_engine_sampler: Literal["wide", "engine", "exact_int"] = "engine"
    # Noise key of the "wide" sampler (both engine_sampler and plain_engine_sampler): True (default
    # since ts-lawfix) = two 32-bit words per (request, position), coincidences ~2^-64 per pair;
    # False = the previous single 32-bit word (~0.5 coinciding (row, position) pairs per 8 x 8k
    # group: those two tokens shared their Gumbel noise), bitwise the old kernel.
    wide_sampler_key64: bool = True
    # Banded importance sampling (block_verification.py). None (default): off, every row is an exact sample of the current
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
    # Always-on engine-vs-learner weight sentinel (sentinel.py): after every refit (theta_k, theta_hat_{k+1}, the initial
    # weights and validation refits) each engine rank's weights are digested
    # and compared bitwise with the learner's shards of the same tensors under
    # the engine's storage dtype; the momentum host stage is re-checked when it
    # is loaded, the stashed LM head against the live one, and expert_bias
    # buffers must never change. A mismatch is an error naming the tensors.
    # O(params / rank) reads per refit, no gather.
    weight_sentinel: bool = True
    # Verify-in-train (vit.py; opt-in). A prompt whose G
    # output drafts are all finished at the top of the step skips verify0: its
    # rows are trained first, one micro-batch per group (never mixing groups),
    # and each row's block-verification decision is read from that training
    # forward. A group with any rejection leaves no gradient (its micro-batch's
    # loss is zeroed; every rank still runs the backward) and is verified the
    # regular way with the same per-row random streams, then trained after its
    # tails in canonical order. Requires speculate, null_bv, verify_mode=block,
    # verify_source=engine, verify_on=learner, sticky routing, streaming over
    # the tails with the canonical order, group granularity, PP = 1 and
    # G <= the training micro-batch rows.
    verify_in_train: bool = False
    # Verify-in-train gate: a group goes to VIT only if its predicted probability
    # of being accepted whole (past steps' per-length counts, vit.PassPredictor,
    # last vit_pass_window steps) is at least vit_min_pass_prob, and, when it
    # spans several micro-batches, its retained forwards (estimated per rank from
    # the measured bytes/token, initially vit_bytes_per_token) fit vit_checkpoint_gb.
    vit_min_pass_prob: float = 0.5
    vit_pass_window: int = 4
    vit_checkpoint_gb: float = 16.0
    vit_bytes_per_token: float = 1.0e6


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


def canonical_groups_per_block(
    ts_cfg: "ThunderSyncConfig",
    dp_size: int,
    G: int,
    micro_batch_size: int = 1,
    min_pipeline_microbatches: int = 0,
) -> int:
    """Groups per canonical training dispatch.

    ``canonical_block_groups``, and with ``deal_rows`` at least enough groups
    that a block's rows reach every learner DP rank (``ceil(dp_size / G)``:
    fewer rows than ranks leave ranks to all-masked fillers under expert
    parallelism) and, under pipeline parallelism, give every rank
    ``min_pipeline_microbatches`` micro-batches per dispatch (one pipeline
    call per dispatch pays the (PP - 1)-slot fill/drain once; the canonical
    order does not hold rows back the way the arrival-order planner does). A
    function of the config and the layout only, so the blocks stay canonical.
    """
    asked = int(ts_cfg.canonical_block_groups)
    bg = max(1, asked)
    if ts_cfg.deal_rows:
        G = max(1, int(G))
        bg = max(
            bg,
            math.ceil(dp_size / G),
            math.ceil(int(min_pipeline_microbatches) * dp_size * int(micro_batch_size) / G),
        )
    if bg != asked and (asked, bg) not in _BG_LOGGED:
        _BG_LOGGED.add((asked, bg))
        print(
            f"[thundersync] canonical_block_groups={asked} raised to {bg} (deal_rows: DP {dp_size}, "
            f"G {G}, MBS {micro_batch_size}, pipeline micro-batches {min_pipeline_microbatches})",
            flush=True,
        )
    return bg


_BG_LOGGED: set = set()


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


def validate_verify0_batch_tokens(master_config: ThunderSyncMasterConfig) -> None:
    """``thundersync.verify0_batch_tokens``: a positive token count or None.

    At TP > 1 the verification head is materialized per forward ([tokens,
    vocab/TP] fp32); the per-rank bytes are printed, not capped (memory only,
    no exactness effect).
    """
    v = master_config.thundersync.verify0_batch_tokens
    if v is not None and int(v) < 1:
        raise ValueError(f"thundersync.verify0_batch_tokens must be >= 1 or null (got {v})")
    # Learner device MoE dispatch (LGRAPHS=dd|1|2, EP > 1): its symmetric workspace holds
    # T_cap tokens per rank per MoE call and a larger call raises in device_dispatch._Plan
    # (on that rank only: its EP peers then wait in the exchange). A verify forward packs
    # up to its batch budget, so the budget must fit T_cap. Config check
    # only: no effect on any value.
    mcfg = master_config.policy.get("megatron_cfg") or {}
    envs = mcfg.get("env_vars") or {}
    if (
        str(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH", "0")) == "1"
        and int(mcfg.get("expert_model_parallel_size", 1) or 1) > 1
    ):
        t_cap = int(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS", 20480))
        for name, budget in (
            ("verify0_batch_tokens", v),
            ("verify_batch_tokens", master_config.thundersync.verify_batch_tokens),
        ):
            if budget is not None and int(budget) > t_cap:
                raise ValueError(
                    f"thundersync.{name}={budget} > the learner's device MoE dispatch token cap "
                    f"NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS={t_cap} (EP > 1): a verify forward of that size "
                    f"cannot use the symmetric exchange; set {name} <= {t_cap} or raise the cap "
                    f"(workspace grows linearly with it)"
                )
        # The learner workers inherit the driver's process env as well as megatron_cfg.env_vars: either sets it.
        if "1" in (
            str(envs.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "0")),
            os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP", "0"),
        ):
            # Per-call receive rows (device_dispatch.call_rows) have a fixed floor (R_cap / 8), so a call's
            # headroom over its mean receive load shrinks as the call grows. A verify forward larger than the
            # training micro-batch overflows where training does not and is run twice (device pass, then the
            # whole call again on Megatron's NCCL dispatcher; exact, but measured on 90 of 95 verify forwards
            # per rank at Ultra-L27 with 2-row verify0 forwards at MBS 1: verify0 ~2x training-forward cost per
            # token). Config check only: no effect on any value.
            mb_tokens = train_micro_batch_rows(master_config) * int(
                master_config.policy["max_total_sequence_length"]
            )
            # NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD = f (fraction of R_cap): verify forwards (no grad) get
            # a floor of f x R_cap rows instead of R_cap / 8, i.e. the training micro-batch's headroom over its
            # mean load at up to 8 f micro-batches of tokens.
            ng = float(
                envs.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD")
                or os.environ.get("NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP_NOGRAD")
                or 0
            )
            mb_tokens = int(mb_tokens * max(1.0, 8.0 * ng))
            for name, budget in (
                ("verify0_batch_tokens", v if v is not None else master_config.thundersync.verify_batch_tokens),
                ("verify_batch_tokens", master_config.thundersync.verify_batch_tokens),
            ):
                if budget is not None and int(budget) > mb_tokens:
                    raise ValueError(
                        f"thundersync.{name}={budget} > the training micro-batch ({mb_tokens} tokens = "
                        f"train_micro_batch_rows x max_total_sequence_length) with "
                        f"NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP=1: verify forwards larger than a training "
                        f"micro-batch overflow the per-call MoE receive cap and rerun on Megatron's dispatcher; "
                        f"set {name} <= {mb_tokens}"
                    )
    if v is None:
        return
    tp = int(mcfg.get("tensor_model_parallel_size", 1) or 1)
    if tp > 1:
        vocab = int(master_config.policy.get("vocab_size", 131072) or 131072)
        gb = int(v) * vocab / tp * 4 / 2**30
        print(
            f"[thundersync] verify0_batch_tokens={v} at TP={tp}: materialized verification head "
            f"~{gb:.1f} GiB per rank per forward (plus the forward's transient activations)",
            flush=True,
        )


def validate_known_keys(master_config: ThunderSyncMasterConfig) -> None:
    """Reject ``thundersync.*`` keys this tree does not declare.

    ``ThunderSyncConfig`` allows extra keys, and Hydra ``++`` adds any key, so a
    misspelled or other-branch knob (e.g. ``++thundersync.remainder_lookup`` on a
    tree without it) used to run silently as the default. Every key the code
    reads is a declared field, so an extra key is never consumed.
    THUNDERSYNC_ALLOW_UNKNOWN_KEYS=1 downgrades the error to a warning.
    """
    ts_cfg = getattr(master_config, "thundersync", None)
    extra = sorted((getattr(ts_cfg, "model_extra", None) or {}).keys())
    if not extra:
        return
    msg = (
        f"unknown thundersync config keys {extra}: not declared by ThunderSyncConfig on "
        "this tree, so they would be ignored (misspelled, or a knob from another branch)"
    )
    if os.environ.get("THUNDERSYNC_ALLOW_UNKNOWN_KEYS") == "1":
        print(f"[thundersync] WARNING {msg}", flush=True)
        return
    raise ValueError(msg + "; THUNDERSYNC_ALLOW_UNKNOWN_KEYS=1 to run anyway")


def effective_engine_sampler(ts_cfg: Any) -> str:
    """The sampler the engines of this run actually use: speculative arms run engine_sampler
    (verify_mode keyed: its own keyed kernel), arms without speculation (sync, async1) run
    plain_engine_sampler. The learner's policy definition (policy.megatron_cfg.exact_int_law)
    and every exact_int validator follow this, never engine_sampler alone."""
    if getattr(ts_cfg, "speculate", False):
        if getattr(ts_cfg, "verify_mode", "block") == "keyed":
            return "keyed"
        return str(getattr(ts_cfg, "engine_sampler", "wide"))
    return str(getattr(ts_cfg, "plain_engine_sampler", "engine"))


def install_plain_exact_int_sampler(generation: Any, master_config: Any, segment: int = 0) -> None:
    """``plain_engine_sampler=exact_int``: the exact integer-law sampler + logged lambda on every
    engine rank of ``generation`` (sync / async1; speculative arms install theirs in
    SpeculativeGeneration). ``segment`` as install_plain_wide_sampler (resume keys)."""
    import zlib as _zlib

    seed = int(master_config.grpo.seed)
    seg = f"/seg{int(segment)}" if int(segment) else ""
    salt = _zlib.crc32(f"exact_int/{seed}/plain{seg}".encode()) | ((seed & 0x7FFFFFFF) << 32)
    ok = ray.get(generation._policy.worker_group.run_all_workers_single_data("install_exact_int_sampler", salt=salt))
    if not any(x is not None for x in ok):
        raise RuntimeError("plain_engine_sampler=exact_int: no engine rank installed the sampler")


def install_plain_wide_sampler(generation: Any, master_config: Any, segment: int = 0, lane: int = 0) -> None:
    """``plain_engine_sampler=wide``: the 53-bit keyed sampler on every engine rank of
    ``generation`` (speculative arms install theirs in SpeculativeGeneration). ``segment``:
    the first step of this process (a resumed run must not reuse the saving run's keys); 0 keeps ts-syncuni's salt.
    ``lane`` (async_lanes): engine lanes number their requests independently, so each lane > 0 gets its own salt
    (lane 0 keeps the single-engine salt)."""
    import zlib as _zlib

    seed = int(master_config.grpo.seed)
    seg = f"/seg{int(segment)}" if int(segment) else ""
    seg += f"/lane{int(lane)}" if int(lane) else ""
    salt = _zlib.crc32(f"wide/{seed}/plain{seg}".encode()) | ((seed & 0x7FFFFFFF) << 32)
    ok = ray.get(
        generation._policy.worker_group.run_all_workers_single_data(
            "install_wide_sampler", salt=salt,
            key64=bool(getattr(getattr(master_config, "thundersync", None), "wide_sampler_key64", True)),
        )
    )
    if not any(x is not None for x in ok):
        raise RuntimeError("plain_engine_sampler=wide: no engine rank installed the sampler")


def validate_precision_guards(policy: Any) -> None:
    """Configs under which the fp16 GEMM recipe silently diverges between learner and engine.

    Each used to run without an error (silent_bug_sweep.md, latent dtype items).
    """
    mcfg = dict(policy.get("megatron_cfg") or {})
    gen = dict(policy.get("generation") or {})
    gcfg = dict(gen.get("mcore_generation_config") or {})
    if mcfg.get("forward_gemm_precision") is None:
        return
    problems = []
    for side, cfg in (("megatron_cfg", mcfg), ("mcore_generation_config", gcfg)):
        if cfg.get("moe_shared_expert_overlap"):
            # NVLS dispatch then hands genuine bf16 tensors to the fp16 bit-cast view.
            problems.append(f"{side}.moe_shared_expert_overlap=true (fp16 bit-cast dispatch)")
        if cfg.get("moe_use_grouped_gemm_for_shared_expert") or cfg.get("use_grouped_gemm_for_shared_expert"):
            problems.append(f"{side}: grouped-GEMM shared experts are not covered by the fp16 patch")
    if gen.get("refit_transport") == "nccl_reshard":
        problems.append(
            "generation.refit_transport=nccl_reshard byte-copies into the engine's fp16 expert "
            "storage (the raw-bf16-bits class); use refit_transport=mcore"
        )
    if mcfg.get("batch_invariant_mode") or gcfg.get("batch_invariant_mode"):
        for k in ("mamba_inference_ssm_states_dtype", "mamba_inference_conv_states_dtype"):
            v = gcfg.get(k)
            if v is not None and str(v).replace("torch.", "") not in ("float32", "fp32"):
                problems.append(
                    f"mcore_generation_config.{k}={v} (the learner keeps fp32 states; the override "
                    "bypasses MCore's batch-invariance fp32 check)"
                )
    if problems:
        raise ValueError("fp16 GEMM recipe: " + "; ".join(problems) + ".")


def validate_config(master_config: ThunderSyncMasterConfig) -> None:
    """Fail fast on settings that would break exact equivalence."""
    validate_known_keys(master_config)
    ts_cfg0 = master_config.thundersync
    if int(ts_cfg0.sticky_pieces_in_flight) < 1:
        raise ValueError(f"thundersync.sticky_pieces_in_flight must be >= 1 (got {ts_cfg0.sticky_pieces_in_flight})")
    validate_verify0_batch_tokens(master_config)
    if float(ts_cfg0.draft_cut_slack_s) < 0:
        raise ValueError(f"thundersync.draft_cut_slack_s must be >= 0 (got {ts_cfg0.draft_cut_slack_s})")
    validate_precision_guards(master_config.policy)
    asked = int(master_config.policy["train_micro_batch_size"])
    eff = train_micro_batch_rows(master_config)
    if eff != asked:
        # The token cap cut the requested rows per micro-batch (the 32768 // 10240 = 3
        # class); runs labelled "MBS k" otherwise train at a different MBS unannounced.
        print(
            f"[thundersync] WARNING policy.train_micro_batch_size={asked} is cut to {eff} rows by "
            f"thundersync.train_micro_batch_tokens={ts_cfg0.train_micro_batch_tokens} at "
            f"max_total_sequence_length={master_config.policy['max_total_sequence_length']}",
            flush=True,
        )
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
    if env_flag("THUNDERSYNC_DEBUG_ORDER_REPLAY") and not (
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
    validate_async_lanes(master_config)
    validate_speculation(master_config.thundersync, master_config.policy)
    validate_expert_parallel(master_config.thundersync, master_config.policy)
    validate_fused_verify_train(master_config.thundersync, master_config.policy)
    validate_engine_verify_source(master_config.thundersync, master_config.policy)
    validate_engine_verify(master_config)
    validate_sticky_routing(master_config)
    validate_crossfade_routing(master_config)
    validate_draft_at_step_top(master_config)
    validate_exact_int(master_config)
    validate_exact_top_p(master_config)
    validate_sticky_rounding(master_config)
    validate_null_bv(master_config)
    validate_tail_tools(master_config)
    validate_learner_prefix_cache(master_config)
    validate_carried_walk(master_config)
    if master_config.thundersync.draft_until_complete and not (
        master_config.thundersync.sticky_routing
    ):
        raise ValueError(
            "thundersync.draft_until_complete requires sticky_routing (the only "
            "path that keeps draft streams running past the deadline)"
        )
    if master_config.thundersync.draft_window_pace and not (
        master_config.thundersync.sticky_routing
    ):
        raise ValueError(
            "thundersync.draft_window_pace requires sticky_routing (the only "
            "path whose drafts keep running to the deferred cut)"
        )
    ts = master_config.thundersync
    if int(ts.sticky_pieces_in_flight) > 1 and ts.sticky_continuation_decision != "every":
        raise ValueError(
            "thundersync.sticky_pieces_in_flight > 1 needs sticky_continuation_decision="
            "'every' (a carried walk advances one piece at a time; not built together)"
        )
    if float(ts.draft_cut_slack_s) > 0 and ts.draft_window_pace:
        raise ValueError(
            "thundersync.draft_cut_slack_s and draft_window_pace both set the draft cut: "
            "use one"
        )
    validate_draft_weights(master_config)
    validate_cohort_lanes(master_config)
    validate_anchor_epoch(master_config)
    validate_banded_is(master_config)
    validate_run_control(master_config)
    validate_verify_in_train(master_config)
    validate_verify_yield(master_config)
    if getattr(ts, "tail_group_order", "index") not in ("index", "kept"):
        raise ValueError(
            f"thundersync.tail_group_order must be index or kept (got {ts.tail_group_order!r})"
        )
    validate_verify0_stream(master_config)
    if ts.null_bv and getattr(ts, "draft_admission", "auto") == "auto":
        print(f"[thundersync] draft_admission=auto -> {resolve_draft_admission(ts)}", flush=True)


def stream_fresh_keys(row_keys) -> list:
    """Prompt keys whose rollouts decode at theta_k whatever their plans.

    The rule ``SpeculativeGeneration._one`` applies (``_debug_forced_fresh``):
    streaming verify0's ``StreamOrder`` must list these groups as fresh, or a
    cohort lane skips its theta_k refit and they decode on the drafting weights.
    """
    return sorted(k for k in set(row_keys) if _debug_forced_fresh(k))


def validate_verify0_stream(master_config: ThunderSyncMasterConfig) -> None:
    """Settings risk-first streaming verify0 is built for."""
    ts_cfg = master_config.thundersync
    if not getattr(ts_cfg, "verify0_stream", False):
        return
    problems = []
    if sticky_delta(ts_cfg) is None:
        problems.append("needs sticky_routing (the streaming-over-tails path)")
    if not (ts_cfg.stream_training_over_tails and ts_cfg.canonical_group_order):
        problems.append("needs stream_training_over_tails and canonical_group_order")
    if ts_cfg.verify_mode != "block" or ts_cfg.verify_on != "learner":
        problems.append("needs verify_mode=block and verify_on=learner")
    if (master_config.grpo.max_rollout_turns or 1) > 1:
        problems.append("multi-turn rollouts are not built")
    for name in (
        # Prompt drop selects the trained prompts after every pool is decided.
        "null_bv_prompt_drop",
        "verify_in_train",
        "fused_verify_train",
        # The tail-excluded baseline needs every group's tail count at step start.
        "tail_excluded_baseline",
    ):
        if getattr(ts_cfg, name, False):
            problems.append(f"{name} must be off (not built with verify0_stream)")
    if ts_cfg.sticky_anchor_horizon is not None:
        problems.append("the anchor horizon is not built with verify0_stream")
    if min(
        ts_cfg.verify0_stream_first_chunk_tokens,
        ts_cfg.verify0_stream_chunk_tokens,
        ts_cfg.verify0_stream_window,
    ) < 1:
        problems.append("chunk tokens and window must be >= 1")
    if ts_cfg.verify0_stream_order not in ("risk", "drafting", "surprisal"):
        problems.append("verify0_stream_order must be risk, drafting or surprisal")
    if ts_cfg.verify0_stream_order == "surprisal":
        from thundersync_rl.envflag import env_flag

        if not env_flag("THUNDERSYNC_ROW_SEEDS"):
            problems.append("verify0_stream_order=surprisal reads draft content and needs THUNDERSYNC_ROW_SEEDS=1")
    if problems:
        raise ValueError("thundersync.verify0_stream: " + "; ".join(problems) + ".")


def _section(master_config: Any, name: str) -> dict[str, Any]:
    part = getattr(master_config, name, None) or {}
    if not isinstance(part, dict):
        part = part.model_dump() if hasattr(part, "model_dump") else dict(vars(part))
    return part


def _learner_expandable_segments_on() -> bool:
    return os.environ.get("THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS", "1") != "0"


def validate_run_control(master_config: Any) -> None:
    """Checkpointing, validation and epochs as the ThunderSync loop runs them."""
    ts_cfg = master_config.thundersync
    ckpt = _section(master_config, "checkpointing")
    grpo = master_config.grpo
    data = _section(master_config, "data")
    problems = []
    val_on = (grpo.val_period or 0) > 0 or grpo.val_at_start or grpo.val_at_end
    if ckpt.get("enabled"):
        mcfg = (getattr(master_config, "policy", None) or {}).get("megatron_cfg") or {}
        if (mcfg.get("checkpoint") or {}).get("async_save") and _learner_expandable_segments_on():
            # ns-resume (ts-nano-scale): the persistent async-save worker receives the learner's
            # CUDA tensors through CUDA IPC; memory from expandable segments (learner default since
            # 14an) is shared with pidfd_getfd, which the container refuses ("pidfd_getfd: Bad
            # file descriptor"): the worker died at the first checkpoint and the run hung.
            problems.append(
                "policy.megatron_cfg.checkpoint.async_save must be false with learner expandable "
                "segments (THUNDERSYNC_LEARNER_EXPANDABLE_SEGMENTS, default on): CUDA IPC of "
                "expandable-segment memory to the async-save worker fails (pidfd_getfd)"
            )
        if not ckpt.get("save_optimizer", True):
            problems.append(
                "checkpointing.save_optimizer must be true (exact resume needs the "
                "optimizer state; the momentum draft is a function of it)"
            )
        if (ckpt.get("save_period") or 0) <= 0:
            problems.append("checkpointing.save_period must be > 0")
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


def validate_engine_verify(master_config: ThunderSyncMasterConfig) -> None:
    """Settings engine-side verification (``verify_on=engine``) is built for."""
    ts_cfg = master_config.thundersync
    if ts_cfg.hidden_transport == "recompute" and not (
        (ts_cfg.verify_on == "engine" and ts_cfg.cohort_lanes == 2)
        or (ts_cfg.verify_on == "learner" and ts_cfg.anchor_epoch is not None)
    ):
        raise ValueError(
            "thundersync.hidden_transport=recompute needs verify_on=engine and cohort_lanes=2 (q rows are "
            "rebuilt on the lane that holds the drafting weights), or anchor epochs with verify_on=learner (q rows "
            "rebuilt on the anchor engine, which holds the drafting version for the whole epoch; DESIGN 14bd)"
        )
    if ts_cfg.verify_on != "engine":
        return
    mcfg = master_config.policy["megatron_cfg"]
    gen = master_config.policy["generation"]
    mcore_gen = gen.get("mcore_generation_config") or {}
    problems = []
    if ts_cfg.verify_source != "engine" or ts_cfg.verify_mode != "block":
        problems.append("needs verify_source=engine and verify_mode=block")
    if not mcfg.get("batch_invariant_mode"):
        problems.append("needs policy.megatron_cfg.batch_invariant_mode (engine prefill == training forward)")
    tp = max(
        int(mcfg.get("tensor_model_parallel_size") or 1),
        int(mcore_gen.get("tensor_model_parallel_size") or 1),
    )
    if tp > 1:
        problems.append(
            "tensor parallel size 1 only (not built yet at TP>1: the decision's vocab "
            "all-gather would run on the engine's TP group outside the engine's step order)"
        )
    if mcore_gen.get("materialize_only_last_token_logits") is False:
        problems.append("needs mcore_generation_config.materialize_only_last_token_logits=true")
    if mcore_gen.get("enable_chunked_prefill"):
        problems.append(
            "needs mcore_generation_config.enable_chunked_prefill=false (a row's prefill in "
            "one forward; chunk-boundary-aligned splits are not built yet)"
        )
    sticky = sticky_delta(ts_cfg) is not None
    if sticky and ts_cfg.cohort_lanes != 2:
        problems.append(
            "with sticky routing it needs cohort_lanes=2: one engine holds the drafting "
            "weights D_k for the cohort's continuations until they end, and theta_k only "
            "after, so no engine holds theta_k while verification is needed"
        )
    if ts_cfg.draft_weights != "current":
        problems.append(
            "needs draft_weights=current (with lanes the verifying lane holds theta_k, "
            "which is also the next cohort's drafter; momentum would need theta_k and "
            "theta_hat_{k+1} on one lane at once, and its head stash lives on the "
            "learner: not built)"
        )
    if ts_cfg.sticky_anchor_horizon is not None:
        problems.append("sticky_anchor_horizon must be unset (not built together)")
    if ts_cfg.learner_prefix_cache or ts_cfg.fused_verify_train:
        problems.append("learner_prefix_cache and fused_verify_train must be off (learner-side tools)")
    if ts_cfg.null_bv_prompt_drop:
        problems.append("null_bv_prompt_drop is not built with engine verification yet")
    if (master_config.grpo.max_rollout_turns or 1) > 1:
        problems.append("multi-turn rollouts are not built with engine verification yet")
    if ts_cfg.sticky_continuation_decision != "every":
        problems.append(
            "needs sticky_continuation_decision='every' (carried-walk decisions are "
            "learner-side; not built on the engine yet)"
        )
    if problems:
        raise ValueError("thundersync.verify_on=engine: " + "; ".join(problems) + ".")


def validate_learner_prefix_cache(master_config: ThunderSyncMasterConfig) -> None:
    """Settings the learner prefix-state cache is built for."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.learner_prefix_cache:
        if getattr(ts_cfg, "learner_prefix_cache_verify0", False):
            raise ValueError(
                "thundersync.learner_prefix_cache_verify0 needs thundersync.learner_prefix_cache"
            )
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
    if getattr(ts_cfg, "learner_prefix_cache_size", "auto") not in ("auto", "fixed"):
        problems.append("learner_prefix_cache_size must be 'auto' or 'fixed'")
    if getattr(ts_cfg, "learner_prefix_cache_policy", "pin") not in ("pin", "lru"):
        problems.append("learner_prefix_cache_policy must be 'pin' or 'lru'")
    if getattr(ts_cfg, "learner_prefix_cache_verify0", False):
        if (master_config.grpo.max_rollout_turns or 1) > 1:
            problems.append("learner_prefix_cache_verify0 is not built for multi-turn rollouts")
        if ts_cfg.sticky_anchor_horizon is not None:
            problems.append("learner_prefix_cache_verify0 is not built with sticky_anchor_horizon")
    if problems:
        raise ValueError("thundersync.learner_prefix_cache: " + "; ".join(problems) + ".")


def validate_verify_yield(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings verify_yield is not built for (verify_yield.py)."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.verify_yield:
        return
    problems = []
    if sticky_delta(ts_cfg) is None or not ts_cfg.stream_training_over_tails:
        problems.append("needs sticky_routing and stream_training_over_tails (ordered learner calls)")
    if ts_cfg.verify_on != "learner":
        problems.append("needs verify_on=learner (continuation pieces verified by the learner)")
    if int((master_config.policy.get("megatron_cfg") or {}).get("pipeline_model_parallel_size", 1) or 1) > 1:
        # Used to print once and silently run posted verifications in order.
        problems.append("boundary yields are not built for pipeline_model_parallel_size > 1")
    if problems:
        raise ValueError("thundersync.verify_yield: " + "; ".join(problems) + ".")


def validate_verify_in_train(master_config: ThunderSyncMasterConfig) -> None:
    """Reject settings verify-in-train is not built for (vit.py)."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.verify_in_train:
        return
    mcfg = master_config.policy["megatron_cfg"]
    G = master_config.grpo.num_generations_per_prompt
    problems = []
    if not (ts_cfg.speculate and ts_cfg.null_bv and ts_cfg.verify_mode == "block"):
        problems.append("needs speculate, null_bv (pooled drafts) and verify_mode=block")
    if ts_cfg.verify_source != "engine" or ts_cfg.verify_on != "learner":
        problems.append("needs verify_source=engine and verify_on=learner")
    if sticky_delta(ts_cfg) is None:
        problems.append("needs sticky_routing (the streaming-over-tails path)")
    if not (ts_cfg.stream_training_over_tails and ts_cfg.canonical_group_order):
        problems.append("needs stream_training_over_tails and canonical_group_order")
    if ts_cfg.granularity != "group":
        problems.append("needs granularity=group")
    if (mcfg.get("pipeline_model_parallel_size", 1) or 1) > 1:
        problems.append("pipeline parallelism is not built")
    if (mcfg.get("context_parallel_size", 1) or 1) > 1:
        problems.append("context parallelism is not built")
    if not 0.0 <= ts_cfg.vit_min_pass_prob <= 1.0 or ts_cfg.vit_checkpoint_gb < 0:
        problems.append("vit_min_pass_prob must be in [0, 1] and vit_checkpoint_gb >= 0")
    if (master_config.grpo.max_rollout_turns or 1) > 1:
        problems.append("multi-turn rollouts are not built")
    for name in (
        "null_bv_prompt_drop",
        "null_bv_drop_is",
        "fused_verify_train",
        "tail_excluded_baseline",
        "sticky_rounding",
    ):
        if getattr(ts_cfg, name, False):
            problems.append(f"{name} must be off (not built with verify_in_train)")
    if ts_cfg.banded_is_band is not None or ts_cfg.sticky_anchor_horizon is not None:
        problems.append("banded IS and the anchor horizon are not built with verify_in_train")
    if env_flag("THUNDERSYNC_DEBUG_ORDER_REPLAY"):
        problems.append("THUNDERSYNC_DEBUG_ORDER_REPLAY does not replay VIT-trained groups")
    if master_config.policy.get("sequence_packing", {}).get("enabled", False):
        problems.append("sequence packing is not built")
    if problems:
        raise ValueError("thundersync.verify_in_train: " + "; ".join(problems) + ".")


def validate_async_lanes(master_config: Any) -> None:
    """Lanes-k async baseline (``thundersync.async_lanes``, async_lanes.py) and the rollout_lag range."""
    ts_cfg = master_config.thundersync
    lag = int(ts_cfg.rollout_lag)
    on = bool(getattr(ts_cfg, "async_lanes", False))
    problems = []
    if lag < 0:
        problems.append(f"rollout_lag must be >= 0 (got {lag})")
    if lag > 1 and not on:
        problems.append(
            f"rollout_lag={lag} needs async_lanes=true (one engine holds one weight version: lag > 1 with k "
            "batches in flight runs on k engine lanes)"
        )
    if on:
        if lag < 1:
            problems.append("async_lanes needs rollout_lag >= 1 (the fixed age k)")
        if ts_cfg.speculate:
            problems.append("async_lanes is the unverified async baseline; it excludes speculate")
        if int(ts_cfg.cohort_lanes) != max(1, lag):
            problems.append(
                f"async_lanes: cohort_lanes must equal rollout_lag (one cohort in flight per lane; got "
                f"cohort_lanes={ts_cfg.cohort_lanes}, rollout_lag={lag})"
            )
        gen = (getattr(master_config, "policy", None) or {}).get("generation") or {}
        if gen.get("backend") != "megatron" or gen.get("refit_transport") not in (None, "mcore"):
            problems.append(
                "async_lanes: policy.generation.backend must be megatron with refit_transport null or mcore "
                "(one refit communicator per lane)"
            )
        grpo = getattr(master_config, "grpo", None)
        if (getattr(grpo, "max_rollout_turns", 1) or 1) > 1:
            problems.append("async_lanes: multi-turn rollouts are not built yet")
        # Checkpoints / resume: built (async_lanes.train_async_lanes; a resume refills the pipeline at its step).
        if getattr(ts_cfg, "plain_engine_sampler", "engine") == "exact_int":
            problems.append("async_lanes: plain_engine_sampler=exact_int is not built yet (use wide)")
        if ts_cfg.granularity == "batch":
            problems.append("async_lanes streams each cohort's rows into training: granularity must be "
                            "trajectory or group")
    if problems:
        raise ValueError("thundersync.async_lanes: " + "; ".join(problems) + ".")


def validate_cohort_lanes(master_config: ThunderSyncMasterConfig) -> None:
    """Settings under which the engine lanes (2, or k at depth k) keep every cohort on its version."""
    ts_cfg = master_config.thundersync
    if ts_cfg.cohort_lanes == 1 or getattr(ts_cfg, "async_lanes", False):
        # async_lanes: validate_async_lanes (plain generation, no speculation on the lanes)
        return
    gen = master_config.policy["generation"]
    problems = []
    la_k = int(getattr(ts_cfg, "cohort_lane_lookahead", 1) or 1)
    if ts_cfg.cohort_lanes < 2 or (ts_cfg.cohort_lanes > 2 and la_k != ts_cfg.cohort_lanes):
        problems.append(
            f"cohort_lanes must be 1 or 2, or k > 2 with cohort_lane_lookahead = k (depth k, one cohort per lane; "
            f"got cohort_lanes={ts_cfg.cohort_lanes}, cohort_lane_lookahead={la_k})"
        )
    if sticky_delta(ts_cfg) is None:
        problems.append(
            "thundersync.sticky_routing must be on (lanes complete each cohort's "
            "continuations on its drafting weights; the non-sticky remainder path "
            "is not built on lanes yet)"
        )
    if gen.get("backend") != "megatron" or gen.get("refit_transport") not in (
        None,
        "mcore",
    ):
        problems.append(
            "policy.generation.backend must be megatron with refit_transport null "
            "(packed collective) or mcore (native m2n refit); one refit "
            "communicator per lane"
        )
    if ts_cfg.draft_after_verify or ts_cfg.fused_verify_train:
        problems.append("draft_after_verify and fused_verify_train must be off")
    if not ts_cfg.stream_training_over_tails and (
        ts_cfg.verify0_stream or ts_cfg.verify_yield or ts_cfg.tail_group_order != "index"
    ):
        # The non-streaming lanes loop (_train_on_lanes) does not implement these:
        # they used to be ignored there without an error.
        problems.append(
            "verify0_stream / verify_yield / tail_group_order need stream_training_over_tails "
            "(the non-streaming lanes loop ignores them)"
        )
    grpo = getattr(master_config, "grpo", None)
    if (getattr(grpo, "max_rollout_turns", 1) or 1) > 1 or ts_cfg.rollout_lag:
        problems.append("multi-turn rollouts and rollout_lag are not built on lanes yet")
    if ts_cfg.draft_lookahead != 1:
        problems.append("draft_lookahead must be 1 (two lanes hold two cohorts)")
    la = int(getattr(ts_cfg, "cohort_lane_lookahead", 1) or 1)
    if la < 1 or la > max(2, ts_cfg.cohort_lanes) or (la > 2 and la != ts_cfg.cohort_lanes):
        problems.append(
            f"cohort_lane_lookahead must be 1 or 2, or k = cohort_lanes (depth k) (got {la}, cohort_lanes="
            f"{ts_cfg.cohort_lanes})"
        )
    if la >= 2:
        if not ts_cfg.stream_training_over_tails:
            problems.append(f"cohort_lane_lookahead={la} needs stream_training_over_tails")
        if ts_cfg.draft_weights != "current":
            problems.append(
                f"cohort_lane_lookahead={la} drafts with the learner's live theta_k mid-step: "
                "draft_weights must be current (a two-step momentum draft is not built)"
            )
        if ts_cfg.verify_on != "learner":
            problems.append(
                f"cohort_lane_lookahead={la} needs verify_on=learner (engine verification uses the "
                "drafting lane as the theta_k lane)"
            )
        if ts_cfg.null_bv_prompt_drop or ts_cfg.draft_window_pace:
            problems.append(f"cohort_lane_lookahead={la}: null_bv_prompt_drop / draft_window_pace not built")
    if la > 2:
        if lookahead_draft.alphas() is not None:
            problems.append(f"cohort_lane_lookahead={la}: lookahead drafts (X5) are not built at depth > 2")
        grpo_c = getattr(master_config, "grpo", None)
        val_on = grpo_c is not None and (
            int(getattr(grpo_c, "val_period", 0) or 0) > 0
            or getattr(grpo_c, "val_at_start", False)
            or getattr(grpo_c, "val_at_end", False)
        )
        if val_on or _section(master_config, "checkpointing").get("enabled"):
            # The boundary rules are generic (draft_limit, the lookahead-1 fallback for the k - 1 cohorts after a
            # boundary), but no run has exercised them at depth > 2: refused until one does.
            problems.append(
                f"cohort_lane_lookahead={la}: checkpoint / validation boundaries are untested at depth > 2"
            )
    if getattr(ts_cfg, "cohort_lane_lookahead_adaptive", False):
        if la != 2:
            problems.append("cohort_lane_lookahead_adaptive needs cohort_lane_lookahead=2 (it chooses per cohort)")
        if not (ts_cfg.null_bv and ts_cfg.null_bv_calibrate):
            problems.append("cohort_lane_lookahead_adaptive forecasts v from the ⊥-BV calibrator (null_bv_calibrate)")
        if not (0 < float(ts_cfg.lookahead_adaptive_v_lo) <= float(ts_cfg.lookahead_adaptive_v_hi)):
            problems.append("lookahead_adaptive_v_lo must be > 0 and <= lookahead_adaptive_v_hi")
    if problems:
        raise ValueError("thundersync.cohort_lanes: " + "; ".join(problems) + ".")


def aep_lambda(ts_cfg) -> Optional[int]:
    """Cohorts drafted at once under ``anchor_epoch`` (= draft_lookahead), None when AEP is off."""
    if getattr(ts_cfg, "anchor_epoch", None) is None:
        return None
    return int(ts_cfg.draft_lookahead)


def validate_anchor_epoch(master_config: ThunderSyncMasterConfig) -> None:
    """Settings AEP-1E (``thundersync.anchor_epoch``, aep.py) is built and exact for."""
    ts_cfg = master_config.thundersync
    if ts_cfg.anchor_epoch is None:
        if ts_cfg.anchor_weights != "raw" or int(ts_cfg.anchor_start) or int(ts_cfg.anchor_momentum_steps) != 1:
            raise ValueError("thundersync.anchor_weights / anchor_start / anchor_momentum_steps need anchor_epoch")
        return
    problems = []
    E, lam = int(ts_cfg.anchor_epoch), int(ts_cfg.draft_lookahead)
    if E < 1 or lam < 1:
        problems.append(f"anchor_epoch ({E}) and draft_lookahead ({lam}) must be >= 1")
    w, h = int(ts_cfg.anchor_start), int(ts_cfg.anchor_momentum_steps)
    if w < 0 or h < 1:
        problems.append(f"anchor_start ({w}) must be >= 0 and anchor_momentum_steps ({h}) >= 1")
    if ts_cfg.anchor_weights == "momentum":
        if w < 1:
            problems.append(
                "anchor_weights=momentum needs anchor_start >= 1 (at step 0 the optimizer moments are zero: the "
                "extrapolation would be theta_0 up to weight decay)"
            )
        opt_cfg = (master_config.policy.get("megatron_cfg") or {}).get("optimizer") or {}
        if not opt_cfg.get("use_distributed_optimizer", False) or opt_cfg.get("optimizer", "adam") != "adam":
            problems.append("anchor_weights=momentum needs Megatron's distributed Adam optimizer (the momentum stage)")
    elif h != 1:
        problems.append("anchor_momentum_steps applies to anchor_weights=momentum only")
    if int(ts_cfg.anchor_reserve_rows) < 0:
        problems.append("anchor_reserve_rows must be >= 0")
    if not (ts_cfg.speculate and ts_cfg.verify_mode == "block"):
        problems.append("needs speculate and verify_mode=block")
    if sticky_delta(ts_cfg) is None:
        problems.append("needs sticky_routing (the policy is F(theta_k; A_theta_a))")
    if ts_cfg.verify_source != "engine" or ts_cfg.verify_on != "learner":
        problems.append("needs verify_source=engine and verify_on=learner (theta_k exists only on the learner)")
    if not ts_cfg.null_bv:
        problems.append("needs null_bv (pooled drafts; unfilled slots become continuations)")
    if ts_cfg.null_bv and resolve_draft_admission(ts_cfg) != "fit":
        problems.append(
            "needs draft_admission=fit (auto resolves to fit): every submitted draft must be admitted at once "
            "(presence fixed before drafting, DESIGN 14ai)"
        )
    if not (ts_cfg.stream_training_over_tails and ts_cfg.canonical_group_order):
        problems.append("needs stream_training_over_tails and canonical_group_order")
    if int(ts_cfg.cohort_lanes) != 1:
        problems.append("needs one engine (cohort_lanes 1): the engine holds the anchor only")
    if ts_cfg.draft_weights != "current":
        problems.append("draft_weights must be current (anchor_weights selects the anchor)")
    for name in (
        "verify0_stream",
        "draft_window_pace",
        "draft_until_complete",
        "null_bv_prompt_drop",
        "null_bv_drop_is",
        "fused_verify_train",
        "tail_lookup",
        "draft_after_verify",
        "sticky_rounding",
        "rollout_lag",
    ):
        if getattr(ts_cfg, name, False):
            problems.append(f"{name} is not built with anchor_epoch")
    if ts_cfg.sticky_anchor_horizon is not None or ts_cfg.banded_is_band is not None:
        problems.append("the anchor horizon and banded IS are not built with anchor_epoch")
    # ts-nano-scale: ts-topp / ts-exact-int were merged next to AEP-1E without a run of the
    # pair (the theta_0 head stash rebuilds q rows on the anchor path). Untested, so refused
    # until an L8 CHECKS=1 arm runs the combination.
    if getattr(ts_cfg, "exact_top_p", False) or getattr(ts_cfg, "sticky_nucleus", False):
        problems.append("exact_top_p / sticky_nucleus with anchor_epoch is untested (not validated together)")
    if effective_engine_sampler(ts_cfg) == "exact_int":
        problems.append(
            "engine_sampler=exact_int with anchor_epoch is untested (not validated together; "
            "set thundersync.engine_sampler=wide)"
        )
    grpo = master_config.grpo
    if (getattr(grpo, "max_rollout_turns", 1) or 1) > 1:
        problems.append("multi-turn rollouts are not built with anchor_epoch")
    # Epoch switches: epochs of P = E + 1 steps start at w + k P; runs may span any number of them.
    # Checkpoint / validation boundaries must be epoch starts (quiescent): periods that are multiples of P (the loop
    # offsets them by w); a timeout-requested save is moved to the next epoch start.
    P = E + 1
    vp = int(getattr(grpo, "val_period", 0) or 0)
    if vp > 0 and vp % P:
        problems.append(
            f"grpo.val_period={vp} must be a multiple of the epoch length anchor_epoch + 1 = {P} (validation refits "
            "the engine: only at epoch starts)"
        )
    ck = _section(master_config, "checkpointing")
    sp = int(ck.get("save_period") or 0)
    if ck.get("enabled") and w % max(sp, 1) and ck.get("keep_top_k") is not None and not ck.get("ft_keep_latest_k"):
        problems.append(
            f"checkpointing: boundaries at anchor_start + k x save_period ({w} + k x {sp}) are not multiples of "
            "save_period, which NeMo-RL's keep_top_k retention counts as periodic (it would delete all but the "
            "latest): set checkpointing.ft_keep_latest_k (recency retention) or keep_top_k null"
        )
    if ck.get("enabled") and sp % P:
        problems.append(
            f"checkpointing.save_period={sp} must be a multiple of the epoch length anchor_epoch + 1 = {P} "
            "(checkpoints are taken at epoch starts, where no cohort is in flight)"
        )
    if problems:
        raise ValueError("thundersync.anchor_epoch: " + "; ".join(problems) + ".")


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
    if ts_cfg.draft_lookahead != 1 and aep_lambda(ts_cfg) is None:
        problems.append(
            "thundersync.draft_lookahead must be 1 unless anchor_epoch is set (drafts spanning versions would "
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
    if float(ts_cfg.sticky_continuation_run_factor) < 0:
        problems.append(
            f"thundersync.sticky_continuation_run_factor must be >= 0 (got {ts_cfg.sticky_continuation_run_factor})"
        )
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
    if problems:
        raise ValueError("thundersync.sticky_rounding: " + "; ".join(problems) + ".")


def validate_carried_walk(master_config: ThunderSyncMasterConfig) -> None:
    """Carried-walk decisions apply to sticky continuation rounds."""
    from thundersync_rl.carried_walk import DecisionRule

    ts_cfg = master_config.thundersync
    rule = DecisionRule(
        ts_cfg.sticky_continuation_decision, float(ts_cfg.sticky_continuation_decision_dip)
    )
    if rule.mode == "every":
        return
    problems = []
    if not ts_cfg.sticky_routing:
        problems.append("thundersync.sticky_routing must be true (continuation rounds)")
    if ts_cfg.verify_source != "engine" or ts_cfg.verify_on != "learner":
        problems.append(
            "thundersync.verify_source='engine' and verify_on='learner' are required "
            "(engine placement: same functions, wiring not built on this branch)"
        )
    if ts_cfg.banded_is_band is not None:
        problems.append("thundersync.banded_is_band does not compose with carried blocks")
    if problems:
        raise ValueError("sticky_continuation_decision: " + "; ".join(problems))


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
    if ts_cfg.verify_on not in ("learner", "engine"):
        problems.append(
            "thundersync.verify_on must be 'learner' or 'engine' (one worker decides a pool)"
        )
    if ts_cfg.draft_lookahead != 1 and aep_lambda(ts_cfg) is None:
        problems.append(
            "thundersync.draft_lookahead must be 1 unless anchor_epoch is set (multi-step cohort pools are not "
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
    admission = resolve_draft_admission(ts_cfg)
    if admission not in DRAFT_ADMISSIONS:
        problems.append(
            "thundersync.draft_admission must be auto or one of " + ", ".join(DRAFT_ADMISSIONS)
        )
    if admission == "fit" and ts_cfg.null_bv_prompt_drop:
        problems.append(
            "thundersync.draft_admission=fit with null_bv_prompt_drop is not built yet "
            "(pools of unequal size change which prompts can be dropped)"
        )
    if not 0.0 <= ts_cfg.null_bv_calibration_decay < 1.0:
        problems.append("thundersync.null_bv_calibration_decay must be in [0, 1)")
    if not ts_cfg.null_bv_tail_weight > 0:
        problems.append("thundersync.null_bv_tail_weight must be > 0")
    if ts_cfg.null_bv_prompt_drop and ts_cfg.null_bv_fixed_m is None:
        problems.append("thundersync.null_bv_prompt_drop needs null_bv_fixed_m (a constant envelope)")
    if ts_cfg.null_bv_spare_prompts < 0:
        problems.append("thundersync.null_bv_spare_prompts must be >= 0")
    if ts_cfg.null_bv_spare_prompts and not ts_cfg.null_bv_prompt_drop:
        problems.append("thundersync.null_bv_spare_prompts needs null_bv_prompt_drop")
    if problems:
        raise ValueError("thundersync.null_bv: " + "; ".join(problems) + ".")


def validate_tail_tools(master_config: ThunderSyncMasterConfig) -> None:
    """Opt-in tail tools that need other features."""
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
        "tail_weight": float(ts_cfg.null_bv_tail_weight),
        "calibrate": bool(ts_cfg.null_bv_calibrate),
        "calibration_decay": float(ts_cfg.null_bv_calibration_decay),
    }


def validate_exact_int(master_config: ThunderSyncMasterConfig) -> None:
    """Settings the exact integer law (engine_sampler=exact_int) is built for."""
    ts = master_config.thundersync
    if effective_engine_sampler(ts) != "exact_int":
        return
    policy_cfg = master_config.policy
    gen = policy_cfg["generation"]
    mcfg = policy_cfg["megatron_cfg"]
    problems = []
    if gen["temperature"] != 1.0:
        problems.append("policy.generation.temperature must be 1.0")
    if gen.get("top_p") not in (None, 1.0) or gen.get("top_k") not in (None, -1, 0):
        problems.append(
            "top_p / top_k must be off (the exact top-p composition, exact_int.law_weights, is "
            "built and enumerated but not wired into the engine sampler / learner / verifier; "
            "for top_p < 1 set thundersync.engine_sampler=wide (+ exact_top_p), and "
            "plain_engine_sampler=wide on sync / async1)"
        )
    if gen.get("backend") != "megatron" or not mcfg.get("batch_invariant_mode"):
        problems.append("the Megatron engine with policy.megatron_cfg.batch_invariant_mode is required")
    mcore_gen = gen.get("mcore_generation_config") or {}
    if mcore_gen.get("num_speculative_tokens"):
        problems.append("in-engine speculative decoding must be off")
    if mcore_gen.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        problems.append("mcore_generation_config.logprobs_mode must be raw_logprobs")
    if master_config.loss_fn.reference_policy_kl_penalty:
        problems.append("loss_fn.reference_policy_kl_penalty must be 0")
    if not ts.speculate:
        if keyed_fresh_enabled(master_config):
            problems.append("debug keyed fresh sampling replaces the plain sampler (set plain_engine_sampler=wide)")
    if ts.speculate:
        if ts.verify_mode != "block" or ts.verify_source != "engine":
            problems.append("verify_mode='block' and verify_source='engine' are required")
        if ts.verify_on not in ("learner",):
            problems.append("verify_on must be 'learner' (engine-side exact verification not built)")
        if ts.banded_is_band is not None:
            problems.append("banded_is_band is not built with exact_int")
        if ts.fused_verify_train:
            problems.append("fused_verify_train is not built with exact_int")
        if getattr(ts, "sticky_continuation_decision", "every") != "every":
            problems.append("sticky_continuation_decision must be 'every' (carried walk not built)")
        if ts.sticky_anchor_horizon is not None:
            problems.append("sticky_anchor_horizon (engine-side anchor verification) not built")
    if problems:
        raise ValueError("thundersync.engine_sampler=exact_int: " + "; ".join(problems) + ".")


def validate_exact_top_p(master_config: ThunderSyncMasterConfig) -> None:
    """Settings the exact top-p policy is built for."""
    ts = master_config.thundersync
    if not ts.exact_top_p:
        if ts.sticky_nucleus:
            raise ValueError("thundersync.sticky_nucleus requires thundersync.exact_top_p")
        return
    policy_cfg = master_config.policy
    gen = policy_cfg["generation"]
    mcfg = policy_cfg["megatron_cfg"]
    problems = []
    if effective_engine_sampler(ts) == "exact_int":
        # ts-nano-scale: the composition (lambda from exact_int.law_weights restricted to
        # the nucleus) is built and enumerated in exact_int but not
        # wired through the engine wrapper / learner chunk path; refused until it is.
        problems.append(
            "engine_sampler=exact_int with exact_top_p is not wired yet (set "
            "thundersync.engine_sampler=wide, or plain_engine_sampler=wide on sync / async1)"
        )
    top_p = gen.get("top_p")
    if top_p is None or not (0.0 < float(top_p) < 1.0):
        problems.append(f"policy.generation.top_p must be in (0, 1) (got {top_p})")
    if gen.get("top_k") not in (None, -1, 0):
        problems.append(f"policy.generation.top_k must be unset (got {gen.get('top_k')})")
    if gen["temperature"] != 1.0:
        problems.append("policy.generation.temperature must be 1.0")
    if gen.get("backend") != "megatron" or not mcfg.get("batch_invariant_mode"):
        problems.append(
            "the Megatron engine with policy.megatron_cfg.batch_invariant_mode is required "
            "(engine and learner mask the nucleus of bitwise-equal rows)"
        )
    mcore_gen = gen.get("mcore_generation_config") or {}
    if mcore_gen.get("num_speculative_tokens"):
        problems.append("in-engine speculative decoding must be off")
    if mcore_gen.get("logprobs_mode", "raw_logprobs") != "raw_logprobs":
        problems.append("mcore_generation_config.logprobs_mode must be raw_logprobs")
    if master_config.loss_fn.reference_policy_kl_penalty:
        problems.append("loss_fn.reference_policy_kl_penalty must be 0 (reference truncation not built)")
    if ts.speculate:
        if ts.verify_mode != "block" or ts.verify_source != "engine":
            problems.append("thundersync.verify_mode='block' and verify_source='engine' are required")
        if ts.verify_on == "engine":
            problems.append("thundersync.verify_on='engine' is untested with top-p (use learner)")
        if ts.banded_is_band is not None:
            problems.append("thundersync.banded_is_band is not built with top-p")
        if ts.fused_verify_train:
            problems.append("thundersync.fused_verify_train is not built with top-p")
    if ts.sticky_nucleus:
        if not ts.sticky_routing:
            problems.append(
                "thundersync.sticky_nucleus requires sticky_routing (the theta_{k-1} "
                "continuation rounds that make drafter anchors exogenous)"
            )
        if not (0.0 <= float(ts.sticky_nucleus_delta) < float(top_p or 0.0)):
            problems.append("thundersync.sticky_nucleus_delta must be in [0, top_p)")
        if int(ts.sticky_nucleus_cap) < 1:
            problems.append("thundersync.sticky_nucleus_cap must be >= 1")
        if ts.sticky_anchor_horizon is not None:
            problems.append(
                "thundersync.sticky_anchor_horizon: engine-side nucleus anchors are not built"
            )
        if ts.learner_prefix_cache:
            problems.append("thundersync.learner_prefix_cache is untested with sticky_nucleus")
        if int(mcfg.get("context_parallel_size") or 1) > 1:
            problems.append(
                "context_parallel_size > 1: CP-sharded nucleus anchors are not built yet"
            )
        if os.environ.get("NEMO_RL_LEARNER_GRAPHS_TRAIN") == "1":
            problems.append(
                "NEMO_RL_LEARNER_GRAPHS_TRAIN=1: training graphs would replay the capture "
                "call's nucleus anchors (not built; LGRAPHS<=1)"
            )
    if problems:
        raise ValueError("thundersync.exact_top_p: " + "; ".join(problems) + ".")


def validate_crossfade_routing(master_config: ThunderSyncMasterConfig) -> None:
    """Settings the θ-only cross-fade policy is built for."""
    ts_cfg = master_config.thundersync
    m = ts_cfg.crossfade_routing_m
    if m is None:
        return
    problems = []
    if m <= 0:
        problems.append(f"thundersync.crossfade_routing_m must be > 0 (got {m})")
    if int(ts_cfg.crossfade_routing_extra) < (0 if env_flag("THUNDERSYNC_DEBUG_CF_ALLOW_EXTRA0") else 1):
        problems.append(f"thundersync.crossfade_routing_extra must be >= 1 (got {ts_cfg.crossfade_routing_extra})")
    if sticky_delta(ts_cfg) is not None:
        problems.append("thundersync.sticky_routing must be false (cross-fade replaces the sticky anchors)")
    if int(getattr(ts_cfg, "cohort_lanes", 1) or 1) != 1:
        problems.append("thundersync.cohort_lanes must be 1 (one engine, one version: no lanes are needed)")
    if ts_cfg.sticky_rounding:
        problems.append("thundersync.sticky_rounding is not built with cross-fade routing")
    if master_config.policy["megatron_cfg"].get("moe_router_fusion"):
        problems.append("policy.megatron_cfg.moe_router_fusion must be false (the fused TE top-k bypasses the rule)")
    if getattr(ts_cfg, "crossfade_routing_sparse", False) and master_config.policy["megatron_cfg"].get("moe_permute_fusion"):
        problems.append("crossfade_routing_sparse needs moe_permute_fusion=false (the argsort permute with padding rows)")
    if (master_config.policy.get("router_replay") or {}).get("enabled"):
        problems.append("policy.router_replay.enabled (R3) must be false")
    if problems:
        raise ValueError("thundersync.crossfade_routing: " + "; ".join(problems) + ".")


def validate_draft_at_step_top(master_config: ThunderSyncMasterConfig) -> None:
    """draft_at_step_top needs the engine to hold theta_k from the top: one engine, θ-only routing, current drafts."""
    ts_cfg = master_config.thundersync
    if not ts_cfg.draft_at_step_top:
        return
    problems = []
    if sticky_delta(ts_cfg) is not None:
        problems.append("sticky routing keeps D_k on the engine until the tails are done")
    if int(getattr(ts_cfg, "cohort_lanes", 1) or 1) != 1:
        problems.append("cohort_lanes must be 1")
    if ts_cfg.draft_weights != "current":
        problems.append("draft_weights must be current (staged weights swap in after the rollouts)")
    if ts_cfg.draft_after_verify:
        problems.append("draft_after_verify contradicts it")
    if problems:
        raise ValueError("thundersync.draft_at_step_top: " + "; ".join(problems) + ".")


def configure_sticky_policy(master_config: Any) -> None:
    """Hand the sticky threshold to the Megatron workers (before they are built).

    Workers read ``policy.megatron_cfg.sticky_routing_delta``: it enables
    MCore's RouterReplay instances in the model (``moe_enable_routing_replay``)
    and the sticky rule (``nemo_rl.models.megatron.sticky_routing``).
    """
    delta = sticky_delta(master_config.thundersync)
    master_config.policy["megatron_cfg"]["sticky_routing_delta"] = delta
    # Cross-fade routing (crossfade_routing.configure in every worker's _apply_moe_config).
    cf_m = master_config.thundersync.crossfade_routing_m
    master_config.policy["megatron_cfg"]["crossfade_routing_m"] = None if cf_m is None else float(cf_m)
    master_config.policy["megatron_cfg"]["crossfade_routing_extra"] = int(master_config.thundersync.crossfade_routing_extra)
    master_config.policy["megatron_cfg"]["crossfade_routing_sparse"] = bool(master_config.thundersync.crossfade_routing_sparse)
    master_config.policy["megatron_cfg"]["crossfade_routing_ramp_stopgrad"] = bool(
        getattr(master_config.thundersync, "crossfade_routing_ramp_stopgrad", False))
    # Exact integer law (exact_int.configure_from_policy in every worker).
    # The learner's policy is the integer law iff the engines of this run sample it.
    master_config.policy["megatron_cfg"]["exact_int_law"] = (
        effective_engine_sampler(master_config.thundersync) == "exact_int"
    )
    # Exact top-p (exact_nucleus.configure_from_policy in every worker).
    ts = master_config.thundersync
    top_p = master_config.policy["generation"].get("top_p")
    master_config.policy["megatron_cfg"]["exact_top_p"] = (
        float(top_p) if ts.exact_top_p and top_p is not None else None
    )
    master_config.policy["megatron_cfg"]["sticky_nucleus_delta"] = (
        float(ts.sticky_nucleus_delta) if ts.exact_top_p and ts.sticky_nucleus else None
    )
    master_config.policy["megatron_cfg"]["sticky_nucleus_cap"] = int(ts.sticky_nucleus_cap)
    master_config.policy["megatron_cfg"]["sticky_rounding"] = bool(
        master_config.thundersync.sticky_rounding
    )
    # The generation workers apply anchors in their forwards (engine anchor
    # apply) with an anchor horizon or engine-side verification.
    master_config.policy["megatron_cfg"]["sticky_anchor_horizon"] = (
        master_config.thundersync.sticky_anchor_horizon if delta is not None else None
    )
    # Engine-side verification (or its prefill check): the engines capture head
    # inputs of tagged prefills (worker ``_install_verify_capture``).
    master_config.policy["megatron_cfg"]["engine_verify"] = bool(
        master_config.thundersync.verify_on == "engine"
        or os.environ.get("THUNDERSYNC_ENGINE_PREFILL_CHECK") == "1"
        # q-row recompute on the anchor engine captures prefill head inputs the same way.
        or master_config.thundersync.hidden_transport == "recompute"
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
    if ts_cfg.verify_on not in ("learner", "engine"):
        problems.append(
            "thundersync.verify_on must be 'learner' or 'engine' (p is the training "
            f"forward or, bitwise the same, the engine's prefill; got {ts_cfg.verify_on!r})"
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
        # The MoE combine is EP-dependent unless policy.megatron_cfg.
        # moe_ep_invariant_combine is on: runs_1node/np-v2-lanes-ep (learner EP2,
        # engine EP1, old combine) had plainly sampled rows off by up to 0.89 in
        # log-prob; with the EP-invariant combine runs_1node/epinv-lanes-ep (same
        # layout, forced fresh rows every step) is exact 4/4 steps and
        # epinv-lanes-ep-off (control) fails. THUNDERSYNC_ALLOW_EP_MISMATCH=1
        # without the combine is a debug override (inexact).
        allow = bool(megatron_cfg.get("moe_ep_invariant_combine", False)) or (
            os.environ.get("THUNDERSYNC_ALLOW_EP_MISMATCH") == "1"
        )
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
    if gen.get("top_p") not in (None, 1.0) and not ts_cfg.exact_top_p:
        problems.append(
            f"policy.generation.top_p must be 1.0 (got {gen['top_p']}); "
            "thundersync.exact_top_p=true runs the exact truncated policy"
        )
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
        and ts_cfg.verify_on not in ("learner", "engine")
        and merged_inference_megatron_cfg(policy_cfg)["expert_model_parallel_size"] > 1
    ):
        # The inference engine steps its EP ranks together; a generation worker
        # blocked in a verification forward leaves its peers' engine all-to-all
        # unmatched while its own scorer all-to-all waits (deadlock). Not
        # verify_on=engine: its forwards are engine requests (run
        # by the engine loop, EP ranks in step) and its decisions are rank-local
        # head GEMMs on a side stream (TP 1), so no rank blocks the engine loop.
        raise ValueError(
            "speculation with inference expert parallelism requires "
            f"thundersync.verify_on='learner' or 'engine' (got {ts_cfg.verify_on!r})"
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
        ts_cfg = getattr(master_config, "thundersync", None)
        if getattr(ts_cfg, "short_fillers", False):
            data = short_filler(data, int(master_config.policy["make_sequence_length_divisible_by"]))
    return pad_batch_to_multiple(data, pad_to_multiple)


def short_filler(data: BatchedDataDict, length: int) -> BatchedDataDict:
    """An all-masked filler cut to its first ``length`` tokens (``short_fillers``).

    Only for fully masked rows (sample_mask = token_mask = advantages = 0): they add
    zero gradient at any length, so the cut changes no gradient and no other row's
    log-probs (batch-invariant forward). Route anchors are dropped (anchor = self).
    """
    assert not bool(data["sample_mask"].any()) and not bool(data["token_mask"].any()), (
        "short_filler: only all-masked filler rows may be cut"
    )
    length = max(1, int(length))
    s = data["input_ids"].shape[1]
    if s <= length and "route_refs" not in data:
        return data
    out = BatchedDataDict()
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[1] == s:
            out[k] = v[:, : min(s, length)].clone()
        elif k == "input_lengths":
            out[k] = v.clamp(max=length)
        elif k == "route_refs":
            out[k] = [None] * len(v)
        else:
            out[k] = v
    return out


def _run_workers(policy: Policy, method: str, **kwargs) -> list[Any]:
    return ray.get(policy.worker_group.run_all_workers_single_data(method, **kwargs))


def _cap_fit(results: dict) -> float:
    from thundersync_rl import mem_budget

    fit = mem_budget.measured_fit(
        results.get("learner_mem_peak_gb", float("nan")),
        results.get("learner_max_mb_tokens", 0.0),
        results.get("learner_mem_nontorch_gb", float("nan")),
    )
    return float("nan") if fit is None else fit


def check_train_mem_cap(policy: Policy, master_config: Any) -> None:
    """Startup rule (mem_budget): the training micro-batch token cap fits the smallest learner rank's memory."""
    from thundersync_rl import mem_budget

    b = mem_budget.budget(master_config.thundersync)
    if b is None:
        return
    device_gb = min(_run_workers(policy, "device_total_gb"))
    row = int(master_config.policy["max_total_sequence_length"])
    cap = train_micro_batch_rows(master_config) * row
    fit = mem_budget.check_cap(cap, row, device_gb, b)
    print(f"[mem budget] PASS micro-batch cap {cap} tokens <= fit {fit} (device {device_gb:.2f} GiB)", flush=True)


def check_learner_layout(policy: Policy, verify_yield_on: bool) -> None:
    """Startup run rule: every TP x CP and ETP x EP group of the learner lies in one
    verify-yield domain (when verify_yield is on) and sees one EP lockstep schedule; raises ``RunCheckError``
    otherwise. One small report per learner rank (its group rank tuples), checked on the driver."""
    from thundersync_rl.run_checks import check_layout

    reports = _run_workers(policy, "layout_ids")
    if not verify_yield_on:
        reports = [{**r, "yield_domain": None} for r in reports]
    check_layout(reports)
    doms = {repr(r.get("yield_domain")) for r in reports}
    print(
        f"[layout check] PASS learner ranks={len(reports)} yield domains={len(doms)} "
        f"ep_groups={len({r['ep'] for r in reports if r.get('ep') is not None})}",
        flush=True,
    )


def train_gen_mismatch(trajectories, learner, tokenizer, master_config, step: int = 0) -> dict:
    """Debug (THUNDERSYNC_MISMATCH_CHECK): rollout log-probs vs the learner's.

    Called before the optimizer step, so both sides use theta_k: any difference
    is train/generation numerics (zero with batch-invariant kernels, NeMo-RL
    PR 3208) plus, for speculative rollouts, the verification scorer's numerics.

    ``THUNDERSYNC_MISMATCH_CHECK_ROWS=n`` (production subset): the
    forward (and the law test in it) runs on n rows chosen by a hash of (grpo.seed, step, row
    index) only -- never by content -- instead of every row.
    """
    n_rows = int(os.environ.get("THUNDERSYNC_MISMATCH_CHECK_ROWS", "0") or 0)
    if 0 < n_rows < len(trajectories):
        seed = int(master_config.grpo.seed)
        trajectories = sorted(
            trajectories,
            key=lambda t: (zlib.crc32(f"mm/{seed}/{step}/{t.index}".encode()), t.index),
        )[:n_rows]
        trajectories = sorted(trajectories, key=lambda t: t.index)
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
    from thundersync_rl import law_check

    law = law_check.enabled()
    out_lp = (
        learner.policy.get_logprobs(
            data, law_check={"seed": int(master_config.grpo.seed) * 7919 + int(step)}
        )
        if law
        else learner.policy.get_logprobs(data)
    )
    train_lp = out_lp["logprobs"]
    law_metrics = {}
    if law:
        # Randomized PIT of every trained token under the learner's p (law_check.py):
        # the sampling LAW, which the value checks below cannot see.
        both = out_lp["law_stats"].double().sum(0)
        stats = both[: law_check.N_STATS]
        print(law_check.format_line(step, stats), flush=True)
        law_metrics = law_check.history_metrics(stats)
        # PIT uniforms keyed per data-parallel replica (ts-lawfix): marks runs whose law/* is valid at DP > 1.
        law_metrics["law/pit_dp_keyed"] = 1.0
        legacy = both[law_check.N_STATS :]
        if legacy.numel() == law_check.N_STATS and float(legacy[law_check.N_BINS]) > 0:
            # THUNDERSYNC_LAW_LEGACY_SHADOW=1: the same tokens under the pre-fix grid (diagnostic, never a verdict).
            print(law_check.format_line(step, legacy).replace("[law check]", "[law check legacy-grid]")
                  .replace(" FAIL", " (legacy: no verdict)").replace(" ok", " (legacy: no verdict)"), flush=True)
            law_metrics.update({k: v for k, v in law_check.history_metrics(legacy, prefix="law_legacy/").items()
                                if not k.endswith("/fail")})
        # Emission law at the verifier: every plain-BV residual / bonus token vs the BV law given its
        # draft and tau, recomputed from the definition (full power per residual draw: the marginal
        # test above sees a residual-law bug only in proportion to the residual draws among all tokens).
        st_null = law_check.pop_null_emission()
        if st_null is not None:
            print(law_check.format_line(step, st_null).replace("[law check]", "[emission law nullbv]"), flush=True)
            law_metrics.update(law_check.history_metrics(st_null, prefix="law_emit_nullbv/"))
        em = [x for x in _run_workers(learner.policy, "pop_emission_law") if x]
        for kind in ("residual", "bonus"):
            st = torch.tensor([x[kind] for x in em], dtype=torch.float64).sum(0) if em else None
            if st is not None and st[law_check.N_BINS] > 0:
                print(law_check.format_line(step, st).replace("[law check]", f"[emission law {kind}]"), flush=True)
                law_metrics.update(law_check.history_metrics(st, prefix=f"law_emit_{kind}/"))
    mask = (data["token_mask"] * data["sample_mask"].unsqueeze(-1)).bool()
    mask[:, 0] = False  # no prediction for the first token
    diff = (train_lp - data["generation_logprobs"])[mask]
    if os.environ.get("THUNDERSYNC_MISMATCH_ROW_REPORT") == "1" and bool((diff != 0).any()):
        # Debug: where the learner and the engine disagree (per row: masked tokens, exact tokens, first inexact
        # position, max |d|), for a failing check only.
        d_all = (train_lp - data["generation_logprobs"]).abs().masked_fill(~mask, 0)
        for r in range(min(int(mask.shape[0]), 64)):
            m = mask[r]
            bad = (d_all[r] != 0) & m
            first = int(bad.nonzero()[0]) if bool(bad.any()) else -1
            print(f"[mismatch row] step={step} row={r} len={int(data['input_lengths'][r])} masked={int(m.sum())} "
                  f"exact={int((m & ~bad).sum())} first_inexact={first} first_masked="
                  f"{int(m.nonzero()[0]) if bool(m.any()) else -1} max={float(d_all[r].max()):.3g}", flush=True)
    return {
        # Bitwise digest of the learner's log-probs of the trained tokens.
        "mismatch/train_lp_crc": zlib.crc32(train_lp[mask].float().numpy().tobytes()),
        "mismatch/max_abs_logprob_diff": float(diff.abs().max()),
        "mismatch/mean_abs_logprob_diff": float(diff.abs().mean()),
        "mismatch/exact_token_frac": float((diff == 0).float().mean()),
        # k3 estimator of KL(gen || train), as gen_kl_error in the GRPO loss.
        "mismatch/gen_kl": float((diff.exp() - 1 - diff).mean()),
        **law_metrics,
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
    key_step: Optional[int] = None,
) -> tuple[list[tuple[int, dict, dict]], float]:
    """Every rollout of ``batch`` (G per prompt) at the engine's current weights.

    Returns ``(i, final_state, metrics)`` per rollout and the wall time. ``key_step``: the step these rollouts
    are trained at; under THUNDERSYNC_DEBUG_KEYED_FRESH every row's noise is keyed by (seed, key_step, row), so the
    lagged rollouts are reproducible across a kill and resume (async1 resume test).
    """
    repeated = batch.repeat_interleave(master_config.grpo.num_generations_per_prompt)
    t0 = time.perf_counter()
    keyed = key_step is not None and keyed_fresh_enabled(master_config)

    async def one(i: int):
        gen = (
            KeyedFreshGeneration(policy_generation, master_config.grpo.seed, key_step, i)
            if keyed
            else policy_generation
        )
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
    if not env_flag("THUNDERSYNC_DEBUG_KEYED_FRESH"):
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


class OrderFeed:
    """A step's canonical training order, extended as engine verification blocks finish.

    ``extend(groups, sizes)`` appends groups (batch indices) to the planner's
    incremental order (``StreamPlanner.extend_order``) and wakes the learner
    loop; calls before ``attach`` are buffered. ``done()`` marks the order
    final (every group listed).
    """

    def __init__(self, step: int, block_tokens: Optional[int] = None):
        self.step = step
        self.block_tokens = block_tokens
        self.planner = None
        self.wake: Optional[asyncio.Event] = None
        self._buffer: list[tuple[list[int], list[int]]] = []
        self.final = False

    def attach(self, planner) -> None:
        self.planner = planner
        planner.begin_order(self.block_tokens)
        for groups, sizes in self._buffer:
            planner.extend_order([(self.step, p) for p in groups], sizes)
        self._buffer = []

    def extend(self, groups: list[int], sizes: list[int]) -> None:
        if not groups:
            return
        if self.planner is None:
            self._buffer.append((list(groups), list(sizes)))
            return
        self.planner.extend_order([(self.step, p) for p in groups], sizes)
        if self.wake is not None:
            self.wake.set()

    def done(self) -> None:
        self.final = True
        if self.wake is not None:
            self.wake.set()


def classify_groups(
    row_keys: list[tuple[int, ...]], G: int, groups: list[int], info: dict
) -> dict[int, str]:
    """Class of each group after verification: "ready", "tail" or "fresh".

    ``info[key] = (plans, all complete)`` of the prompt's published plans.
    fresh: fewer plans than rows (some row decodes at theta_k after the
    refit); tail: some plan still needs continuation rounds; else ready. A
    function of the verification results only.
    """
    need = collections.Counter(row_keys)
    out = {}
    for p in groups:
        key = row_keys[p * G]
        n_plans, complete = info.get(key, (0, True))
        if n_plans < need[key]:
            out[p] = "fresh"
        elif not complete:
            out[p] = "tail"
        else:
            out[p] = "ready"
    return out


def order_tail_groups(
    groups: list[int], row_keys: list, G: int, kept: Callable[[Any], Optional[int]], how: str
) -> list[int]:
    """The tail class in canonical order (``thundersync.tail_group_order``).

    ``kept(key)``: fewest tokens kept at verify0 among the prompt's tail plans
    (``SpeculativeGeneration.tail_kept``; None: no recorded tail plan, listed
    first). "kept": most kept first (least left to decode), ties by group index.
    """
    if how == "index":
        return list(groups)
    if how != "kept":
        raise ValueError(f"thundersync.tail_group_order must be index or kept (got {how!r})")

    def rank(p):
        k = kept(row_keys[p * G])
        return (-(k if k is not None else 1 << 40), p)

    return sorted(groups, key=rank)


def block_chunks(groups: list[int], b: int) -> list[int]:
    """Sizes of consecutive blocks of at most ``b`` groups."""
    b = max(1, int(b))
    return [min(b, len(groups) - i) for i in range(0, len(groups), b)]


def cancel_on_failure(task: asyncio.Task, target: Optional[asyncio.Task]):
    """Cancel ``target`` as soon as ``task`` fails; returns (errors, callback).

    ``errors`` receives ``task``'s exception (the caller re-raises it in place
    of the CancelledError); remove ``callback`` from ``task`` once ``target``
    no longer needs it.
    """
    errors: list[BaseException] = []

    def callback(t: asyncio.Task) -> None:
        if t.cancelled() or t.exception() is None or errors:
            return
        errors.append(t.exception())
        if target is not None and not target.done():
            target.cancel()

    task.add_done_callback(callback)
    return errors, callback


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
    order_stream: Optional[StreamOrder] = None,
    order_feed: Optional[OrderFeed] = None,
    ordered_learner: Optional[Any] = None,
    cut_while_training: Optional[bool] = None,
    claim_row: Optional[Callable[[torch.Tensor, torch.Tensor], Any]] = None,
    tail_count: Optional[Callable[[torch.Tensor], int]] = None,
    claim_log_weight: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    stale_rollouts: Optional[list[tuple[int, dict, dict]]] = None,
    generate_next: Optional[Callable[[], Any]] = None,
    vit_spec: Optional[Any] = None,
    rollout_source: Optional[Callable[[], list[Awaitable]]] = None,
    side_task: Optional[Callable[[asyncio.Lock], Awaitable[Any]]] = None,
    side_hold_rows: Optional[int] = None,
    side_dispatch_rows: Optional[int] = None,
) -> dict[str, Any]:
    """One step: rollouts, streaming training and the next step's drafts.

    ``vit_spec`` (verify-in-train, ``vit.py``): the spec holding this step's VIT
    groups; their training calls go first, before any planner dispatch.

    ``drafts_after_pre_rollout``: the drafter also waits for ``pre_rollout``
    (sticky streaming over the tails: the continuation rounds and the refit
    that puts the drafter's weights on the engine). ``group_order``: dispatch
    the step's groups (prompt indices) one at a time in this order
    (``StreamPlanner.set_order``). ``tail_groups``: groups with pending tails,
    logged with each group's completion and dispatch times.
    ``order_stream`` (streaming verify0): ``group_order`` /
    ``group_blocks`` are its lists, which grow while the step runs; the
    planner follows them as they grow (``StreamPlanner.extend_order``).
    ``ordered_learner`` (the spec while streaming over the tails, or while
    chunked verification overlaps training): training dispatches are numbered
    learner calls (``learner_seq``) that every learner rank runs in the same
    order, interleaved with the verification forwards (asynchronous Ray actors
    do not keep submission order when several calls wait; ``ordered_call``
    does). ``cut_while_training`` (default: with ``ordered_learner``; sticky
    streaming): the next drafts are cut while the last dispatch trains.
    ``stale_rollouts``
    (rollout_lag=1): this batch's rollouts, already generated at theta_{k-1},
    are trained instead of sampling new ones, while ``generate_next``
    samples the next batch at theta_k on the engine concurrently (returned as
    ``_next_rollouts``).
    ``rollout_source`` (async_lanes): this batch's rollouts come from these awaitables, each resolving to
    ``(i, final_state, metrics, None)`` as its row finishes on an engine lane (generated in an earlier step at an
    older version); training streams them as they arrive. ``side_task(gate)`` (async_lanes): a coroutine run
    beside the step; while it holds ``gate`` no training dispatch is on the learner (the learner loop takes the
    gate around each dispatch), so it may call the learner (a lane refit from the live theta_k). It finishes
    before the optimizer step; its result is returned as ``_side``. ``side_hold_rows``: once at most this many rows of
    the batch are still outstanding, the learner starts no new dispatch until the side task is done (its lane refit
    then starts at the cohort's completion instead of after an in-flight dispatch; ``learner/side_hold_s``).
    ``side_dispatch_rows``: while the side task is pending, each training dispatch takes at most this many rows (whole
    groups), so the side task's wait for the gate stays short; afterwards the planner's dispatches are uncapped.
    """
    G = master_config.grpo.num_generations_per_prompt
    keyed_fresh = keyed_fresh_enabled(master_config)
    if keyed_fresh and group_order is None:
        # Reproducible runs also need a reproducible gradient order: dispatch
        # the groups in index order (StreamPlanner.set_order), not by arrival.
        group_order = list(range(batch.size))
        bg = canonical_groups_per_block(
            ts_cfg,
            learner.dp_size,
            G,
            train_micro_batch_rows(master_config),
            resolve_min_pipeline_microbatches(
                ts_cfg,
                master_config.policy["megatron_cfg"].get("pipeline_model_parallel_size", 1) or 1,
            ),
        )
        group_blocks = [min(bg, len(group_order) - j) for j in range(0, len(group_order), bg)]
    # Streaming over the tails: training dispatches are ordered learner calls
    # (SpeculativeGeneration.learner_seq), interleaved with verification.
    ordered = ordered_learner
    if cut_while_training is None:
        cut_while_training = ordered is not None
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
            # Under expert parallelism only with dealt rows: the planner then
            # marks zero-advantage chunks count-only and pads the live forwards
            # to equal per-rank counts, so the worker skips exactly those chunks
            # on every rank (StreamPlanner.count_only).
            skip_zero_advantage_chunks=(
                bool(getattr(loss_fn, "skip_zero_advantage_rows", False))
                and (not expert_parallel or ts_cfg.deal_rows)
            ),
            min_pipeline_microbatches=min_pipeline_microbatches,
            tail_excluded_baseline=(
                make_tail_excluded_fn(master_config) if ts_cfg.tail_excluded_baseline else None
            ),
            deal_rows=ts_cfg.deal_rows,
        )
        for p in range(n // G):
            prompt = torch.cat([m["token_ids"] for m in repeated["message_log"][p * G]])
            pl.register_group(
                (step, p),
                G,
                tails=tail_count(prompt) if ts_cfg.tail_excluded_baseline and tail_count else 0,
            )
        if order_stream is not None and order is order_stream.order and not order_stream.closed:
            # Streaming verify0: the order grows with the decisions.
            order_stream.attach(pl, step)
        elif order is not None:
            pl.set_order(
                [(step, p) for p in order],
                block_sizes,
                block_tokens=ts_cfg.canonical_block_tokens,
            )
        return pl

    planner = make_planner(group_order, group_blocks)
    if order_feed is not None:
        # Engine verification: the order grows as verification blocks finish.
        order_feed.attach(planner)

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
    order_replay = env_flag("THUNDERSYNC_DEBUG_ORDER_REPLAY") and (
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
                    # Expert parallelism: the worker counts this chunk for
                    # normalization and runs no forward (planner decision, the
                    # same on every rank). Without EP the worker skips
                    # zero-advantage rows itself (loss_fn.skip_zero_advantage_rows).
                    "count_only": expert_parallel and planner.count_only(c),
                }
                for c in chunks
            ]
            for chunks in d.per_rank
        ]

    t_start = time.perf_counter()
    wake = asyncio.Event()
    if order_stream is not None:
        order_stream.wake = wake.set
    if order_feed is not None:
        order_feed.wake = wake
    rollouts_done = False
    rewards = [0.0] * n
    is_weights = [1.0] * n  # banded IS loss weights
    sample_metrics: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]
    learner_busy = 0.0
    # Pipeline schedule calls of the busiest rank, summed over dispatches.
    pipeline_calls = 0
    num_dispatches = 0
    # Timeline for offline analysis: (t, group, tokens, reward) per arrival and
    # (t_start, t_end, n_trajectories, n_tokens, n_bucket_chunks, real / filler
    # micro-batches, micro-batches per rank, busiest rank's tokens) per dispatch.
    arrivals: list[tuple[float, int, int, float]] = []
    dispatches: list[tuple] = []
    # Planner micro-batches over the step (``microbatch_counts``).
    mb_totals = {"real": 0, "filler": 0, "skipped": 0, "per_rank": 0, "filler_heavy": 0}
    # (group, dispatch start, dispatch end) for every group a dispatch trains.
    group_dispatch: list[tuple[int, float, float]] = []
    last_rollout_t = None
    all_trajectories: list[Trajectory] = []
    # Fused verification + training: trajectories handed a retained forward,
    # and what the learner did with them.
    fused_counts = collections.Counter()

    replayed = [0]  # drafted multi-turn rollouts rebuilt from their records
    stable_fresh = (
        isinstance(policy_generation, SpeculativeGeneration)
        and getattr(policy_generation, "engine_sampler", None) == "wide"
    )

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
        if keyed_fresh or stable_fresh:
            # stable_fresh: the speculative wide sampler keys this row's fresh (never drafted) decode by
            # hash(seed, "fresh", step, row, turn) instead of the engine request id, so the first step after a
            # checkpoint (all rows fresh) reproduces across a resume (worker.install_wide_sampler).
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

    # Verify-in-train: this step's groups trained (and verified) by their own calls.
    vit_groups = vit_spec.take_vit_groups() if vit_spec is not None else []
    vit_submitted = asyncio.Event()
    if not vit_groups:
        vit_submitted.set()
    vit_stats: dict[str, float] = {}

    def _row_reward(state) -> float:
        reward = float(state["total_reward"])
        if env_flag("THUNDERSYNC_DEBUG_PARITY_REWARD"):
            toks = torch.cat([m["token_ids"] for m in state["message_log"]])
            reward = float(zlib.crc32(toks.to(torch.int64).numpy().tobytes()) & 1)
        return reward

    async def vit_run():
        """Train the VIT groups first; decide each from its training forward."""
        nonlocal learner_busy
        t0 = time.perf_counter()
        try:
            key_to_p = {}
            for p in range(n // G):
                k = tuple(
                    torch.cat([m["token_ids"] for m in repeated["message_log"][p * G]]).tolist()
                )
                key_to_p.setdefault(k, p)
            groups = sorted(((key_to_p[vg.key], vg) for vg in vit_groups), key=lambda x: x[0])

            async def stub(i, d):
                g = len(d[2])
                gen = vit.StubGeneration(
                    getattr(policy_generation, "cfg", None),
                    d[2],
                    [float(x) for x in d[6][:g]],
                )
                return await run_sample_multi_turn_rollout(
                    sample_idx=i,
                    initial_sample_state=_initial_sample_state(repeated, i),
                    policy_generation=gen,
                    tokenizer=tokenizer,
                    task_to_env=task_to_env,
                    max_seq_len=master_config.policy["max_total_sequence_length"],
                    max_rollout_turns=master_config.grpo.max_rollout_turns,
                    greedy=False,
                )

            outs = await asyncio.gather(
                *[stub(p * G + j, vg.drafts[j]) for p, vg in groups for j in range(G)]
            )
            vit_stats["vit/stub_s"] = time.perf_counter() - t0
            entries = []
            for gi, (p, vg) in enumerate(groups):
                trajs, rows = [], []
                for j in range(G):
                    state, metrics = outs[gi * G + j]
                    i = p * G + j
                    d = vg.drafts[j]
                    g = len(d[2])
                    loss_mult = float(repeated["loss_multiplier"][i])
                    if master_config.grpo.overlong_filtering and metrics["truncated"]:
                        loss_mult = 0.0
                    a, ln = vit.assistant_span(state["message_log"])
                    toks = torch.cat([m["token_ids"] for m in state["message_log"]])
                    if ln != g or toks[a : a + g].tolist() != list(d[2]):
                        raise RuntimeError(
                            "verify_in_train: the reward rollout's assistant tokens are not the draft"
                        )
                    trajs.append(
                        Trajectory(
                            group=(step, p),
                            index=i,
                            reward=_row_reward(state),
                            payload={
                                "message_log": state["message_log"],
                                "loss_multiplier": loss_mult,
                                "anchors": d[8].truncate(d[0].numel() + g),
                            },
                            num_tokens=int(toks.numel()),
                        )
                    )
                    rows.append(
                        {
                            "start": a,
                            "len": g,
                            "logq": [float(x) for x in d[6][:g]],
                            "seed": vg.seeds[j],
                            "m": vg.ms[j],
                            "row": j,
                        }
                    )
                adv = adv_fn([t.reward for t in trajs])
                entries.append((p, vg, trajs, rows, adv, gi))

            def make_item(trajs, rows, adv, gi, spread):
                data = build_chunk_data(
                    Chunk(
                        group=None,
                        reward=None,
                        trajectories=trajs,
                        advantages=adv,
                        fused=None,
                        padding=False,
                    ),
                    tokenizer,
                    master_config,
                    1,
                )
                vit.add_vit_fields(data, rows)
                return {
                    "data": data,
                    "group": None,
                    "reward": None,
                    "fused_key": None,
                    "vit": True,
                    "vit_spread": spread,
                    "vit_group": gi,
                }

            def pad_item(item):
                data = BatchedDataDict(
                    {k: (v.clone() if isinstance(v, torch.Tensor) else list(v) if isinstance(v, list) else v)
                     for k, v in item["data"].items()}
                )
                for k in ("sample_mask", "token_mask", "advantages"):
                    data[k] = torch.zeros_like(data[k])
                return {**item, "data": data, "vit_group": -1}

            dp = learner.dp_size
            spread_rows = vit_spec.vit_spread_rows()

            def submit(per_rank, what):
                seq = None if ordered is None else ordered.learner_seq()
                refs = learner.submit(per_rank, [], seq=seq)
                if seq is not None:
                    ordered.learner_track(refs)
                sent.append((what, refs, time.perf_counter()))

            def flush(blk):
                # One micro-batch per group, groups placed on ranks by tokens in group order.
                if not blk:
                    return
                per_rank = [[] for _ in range(dp)]
                load = [0] * dp
                for e in blk:
                    r = load.index(min(load))
                    per_rank[r].append(make_item(e[2], e[3], e[4], e[5], False))
                    load[r] += sum(t.num_tokens for t in e[2])
                if expert_parallel:
                    target = max(len(x) for x in per_rank)
                    for x in per_rank:
                        while len(x) < target:
                            x.append(pad_item(next(y for z in per_rank for y in z)))
                submit(per_rank, list(blk))

            # Canonical: groups in group order; single-micro-batch groups in blocks of
            # canonical_block_groups, a group larger than one micro-batch alone, spread
            # over every rank (vit.plan_group_layout) and decided after all its forwards.
            bg = max(1, int(ts_cfg.canonical_block_groups))
            sent = []
            blk = []
            n_spread = 0
            for e in entries:
                p, vg, trajs, rows, adv, gi = e
                if spread_rows is None or G <= spread_rows:
                    blk.append(e)
                    if len(blk) == bg:
                        flush(blk)
                        blk = []
                    continue
                flush(blk)
                blk = []
                n_spread += 1
                layout = vit.plan_group_layout(
                    [t.num_tokens for t in trajs], dp, int(spread_rows)
                )
                shortest = min(range(G), key=lambda j: (trajs[j].num_tokens, j))
                per_rank = []
                for slots in layout:
                    items = []
                    for slot in slots:
                        if slot is None:
                            items.append(
                                pad_item(make_item([trajs[shortest]], [rows[shortest]], [adv[shortest]], gi, True))
                            )
                        else:
                            items.append(
                                make_item(
                                    [trajs[j] for j in slot],
                                    [rows[j] for j in slot],
                                    [adv[j] for j in slot],
                                    gi,
                                    True,
                                )
                            )
                    per_rank.append(items)
                submit(per_rank, [e])
            flush(blk)
            vit_submitted.set()
            failed = []
            n_pass = 0
            bpt = []
            for blk_e, refs, t_sub in sent:
                rank_results = await asyncio.gather(*refs)
                if ordered is not None:
                    ordered._learner_inflight.difference_update(refs)
                learner_busy += time.perf_counter() - t_sub
                dec: dict[int, tuple[bool, dict]] = {}
                for rr in rank_results:
                    for e in rr.get("vit", []):
                        if e["group"] < 0:
                            continue
                        if "retained_bytes_per_token" in e:
                            bpt.append(float(e["retained_bytes_per_token"]))
                        prev = dec.get(e["group"])
                        if prev is not None and prev[0] != e["pass"]:
                            raise RuntimeError("verify_in_train: ranks disagree on a group")
                        rows_lp = {} if prev is None else prev[1]
                        rows_lp.update({j: lpx for j, _, lpx in e["rows"]})
                        dec[e["group"]] = (e["pass"], rows_lp)
                for p, vg, trajs, rows, adv, gi in blk_e:
                    ok, rows_lp = dec[gi]
                    if ok:
                        planner.mark_pretrained((step, p), [t.reward for t in trajs])
                        await vit_spec.vit_pass(vg, rows_lp)
                        n_pass += 1
                    else:
                        failed.append(vg)
            if bpt:
                vit_spec.vit_observe_bytes_per_token(max(bpt))
                vit_stats["vit/retained_bytes_per_token"] = max(bpt)
            vit_stats["vit/spread_groups"] = float(n_spread)
            vit_stats["vit/train_s"] = time.perf_counter() - t0
            pending_tails = await vit_spec.vit_fallback(failed)
            vit_stats.update(
                {
                    "vit/groups": float(len(entries)),
                    "vit/passed_groups": float(n_pass),
                    "vit/failed_groups": float(len(failed)),
                    "vit/fallback_tails": float(len(pending_tails)),
                    "vit/rows_trained": float(G * n_pass),
                    "vit/s": time.perf_counter() - t0,
                }
            )
            vit_spec.vit_finish(pending_tails)
            wake.set()
        except BaseException as exc:
            vit_submitted.set()
            vit_spec.vit_finish([], error=exc)
            await vit_spec._notify_tails()
            raise

    vit_task = asyncio.create_task(vit_run()) if vit_groups else None

    # async_lanes: the side task's learner calls (a lane refit) run between dispatches, never during one.
    gate = asyncio.Lock() if side_task is not None else None
    side_done = asyncio.Event() if side_task is not None else None
    arrived = [0]  # rows of this batch arrived so far (side_hold_rows)
    side_hold = [0.0]
    # Bounded hold: learner seconds and rows of this step's timed dispatches (the hold limit's per-row rate), and
    # whether a hold already expired this step (then no further holds: the dispatch cap bounds the refit's wait).
    side_rate = [0.0, 0]
    side_hold_expired = [False]

    async def learner_loop():
        nonlocal learner_busy, num_dispatches, pipeline_calls
        # Verify-in-train calls are the step's first gradient contributions.
        await vit_submitted.wait()
        while True:
            await wake.wait()
            wake.clear()
            # Batch-synchronous baseline: hold everything until the batch closes.
            # Under PP, rows of closed groups are held until they fill the
            # pipeline (min_pipeline_microbatches) unless the rollouts are done.
            while planner.has_work(flush=rollouts_done) and (
                (ts_cfg.granularity != "batch" and not order_replay) or rollouts_done
            ):
                if side_dispatch_rows is not None and side_done is not None:
                    planner.max_dispatch_rows = None if side_done.is_set() else int(side_dispatch_rows)
                d = planner.next_dispatch(flush=rollouts_done)
                if per_param_replay:
                    dispatch_sigs["real"].append(dispatch_sig(d))
                per_rank = _build_dispatch(d)
                if (
                    side_hold_rows is not None
                    and side_done is not None
                    and not side_done.is_set()
                    and not side_hold_expired[0]
                    and arrived[0] >= n - side_hold_rows
                ):
                    # The cohort is nearly complete: keep the learner free for the lane refit, but no longer than
                    # this dispatch would take (async_lanes.side_hold_limit_s). A long cohort tail would otherwise
                    # idle the learner (Super B1024 a1l: ~35 s/step); past the limit the refit waits behind at most
                    # one capped dispatch (side_dispatch_rows).
                    waited, expired = await bounded_side_hold(
                        side_done, side_hold_limit_s(d.num_trajectories, side_rate[0], side_rate[1])
                    )
                    side_hold[0] += waited
                    side_hold_expired[0] = expired
                if gate is not None:
                    await gate.acquire()
                try:
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
                finally:
                    if gate is not None:
                        gate.release()
                pipeline_calls += max(
                    (r.get("num_calls", 0) for r in rank_results), default=0
                )
                for r in rank_results:
                    for k in ("fused_trained", "fused_tokens", "fused_fallback"):
                        fused_counts[k] += r[k]
                learner_busy += t1 - t0
                side_rate[0] += t1 - t0
                side_rate[1] += d.num_trajectories
                chunks = [c for cs in d.per_rank for c in cs if not c.padding]
                for gid in sorted({t.group[1] for c in chunks for t in c.trajectories}):
                    group_dispatch.append((gid, t0 - t_start, t1 - t_start))
                mb = microbatch_counts(d, mbs, planner.count_only)
                for k in ("real", "filler", "skipped", "per_rank"):
                    mb_totals[k] += mb[k]
                if mb["filler"] > 0.25 * (mb["real"] + mb["filler"]):
                    # Validator: more than a quarter of the dispatch's forwards
                    # are all-masked fillers (rows < DP, or deal_rows off).
                    mb_totals["filler_heavy"] += 1
                    if mb_totals["filler_heavy"] == 1:
                        print(
                            f"[thundersync] step {step}: dispatch of {d.num_trajectories} rows "
                            f"ran {mb['filler']} filler / {mb['real']} real micro-batches over "
                            f"{learner.dp_size} learner DP ranks (deal_rows={ts_cfg.deal_rows})",
                            flush=True,
                        )
                # (start, end, rows, tokens, bucket chunks, real micro-batches,
                # filler micro-batches (both summed over ranks), micro-batches per
                # rank, real tokens on the busiest rank)
                dispatches.append(
                    (
                        t0 - t_start,
                        t1 - t_start,
                        d.num_trajectories,
                        sum(c.num_tokens for c in chunks),
                        sum(1 for c in chunks if c.group is not None),
                        mb["real"],
                        mb["filler"],
                        mb["per_rank"],
                        mb["max_rank_tokens"],
                    )
                )
                num_dispatches += 1
            if rollouts_done and not planner.has_work():
                if not planner.order_complete():
                    # A failed verification leaves its groups unordered: name it.
                    cause = (
                        verify_task.exception()
                        if verify_task is not None
                        and verify_task.done()
                        and not verify_task.cancelled()
                        else None
                    )
                    raise RuntimeError(
                        "canonical order incomplete after the last rollout "
                        f"({len(planner.order)} of {len(planner.groups)} groups)"
                        + (f"; verification failed: {cause!r}" if cause is not None else "")
                    ) from cause
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
    # Fail fast on a learner error. Before, a failed dispatch (e.g. CUDA OOM on
    # one rank) surfaced only when the next rollout finished; rollouts wait on
    # verification, which is queued behind the peers stuck in the failed
    # dispatch's collectives, so the run sat until the 600 s NCCL watchdog.
    # Cancel this coroutine instead; the handler below aborts the learner and
    # re-raises the learner's own error.
    learner_error, _on_learner_done = cancel_on_failure(
        learner_task, asyncio.current_task()
    )
    next_task = (
        asyncio.create_task(generate_next()) if generate_next is not None else None
    )
    side = asyncio.create_task(side_task(gate)) if side_task is not None else None
    if side is not None:
        side.add_done_callback(lambda _t: side_done.set())
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
    if (
        ts_cfg.draft_at_step_top
        and drafter is not None
        and draft_task is None
        and not after_verify
        and draft_swap is None
        and stale_rollouts is None
    ):
        draft_started = time.perf_counter()
        draft_task = asyncio.create_task(drafter())
    try:
        if rollout_source is not None:
            sources = asyncio.as_completed(rollout_source())
        else:
            sources = (
                asyncio.as_completed([rollout(i) for i in range(n)])
                if stale_rollouts is None
                else [_ready((*r, None)) for r in stale_rollouts]
            )
        for fut in sources:
            i, state, metrics, anchors = await fut
            n_done += 1
            arrived[0] = n_done
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
            if env_flag("THUNDERSYNC_DEBUG_PARITY_REWARD"):
                # Debug: a reward that is a fixed function of the rollout's tokens
                # (crc32 parity), so truncated models with zero task reward still
                # train on nonzero advantages (determinism checks need them).
                toks = torch.cat([m["token_ids"] for m in state["message_log"]])
                reward = float(zlib.crc32(toks.to(torch.int64).numpy().tobytes()) & 1)
            zero_every = int(os.environ.get("THUNDERSYNC_DEBUG_ZERO_ADV_GROUPS", "0") or 0)
            if zero_every > 0 and (i // G) % zero_every == 0:
                # Debug: every zero_every-th group gets a constant reward (all-zero
                # advantage), so zero-advantage skipping is exercised
                # under the parity reward (which almost never ties a group).
                reward = 0.0
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
            if stale_rollouts is None and planner.has_work(flush=False):
                # Completed rollouts (served from complete plans) resolve
                # without suspending this loop: yield so the learner loop
                # sends the dispatch now, not after the last row is served.
                # Not with stale_rollouts (rollout_lag=1): every row is already
                # in hand, and yielding here sent the first group alone (an
                # 8-row dispatch costing a whole 8-micro-batch-per-rank call
                # under EP at full Nano, 10.6-14.2 s/step in async1-*-ab2).
                await asyncio.sleep(0)
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
        if vit_task is not None:
            await vit_task
        rollouts_done = True
        wake.set()
        if cut_while_training and draft_task is not None:
            # Sticky streams go on decoding after the cut (their tokens become
            # continuation pieces), so the cut time only moves tokens between
            # the next verify0 and its continuation rounds. A deferrable cut
            # is one snapshot call: drafts run until training ends and are
            # handed out at the next step's top (``_resolve_drafts``, beside
            # the drafting lane's refit). Otherwise cut while the last
            # training dispatch runs.
            async def cut_drafts():
                d = await draft_task
                deferrable = callable(d) and getattr(d, "deferrable", False)
                await (learner_task if deferrable else last_sent.wait())
                t = time.perf_counter()
                if callable(d):
                    d = await asyncio.to_thread(d, True) if deferrable else await asyncio.to_thread(d)
                return d, time.perf_counter() - t

            cut_task = asyncio.create_task(cut_drafts())
        await learner_task
        # async_lanes: the side task (lane refit + next cohort start) ends before the optimizer step.
        side_result = await side if side is not None else None
        t_next = time.perf_counter()
        next_rollouts = await next_task if next_task is not None else None
        next_wait = time.perf_counter() - t_next
        if verify_task is not None:
            await verify_task
        t_wait = time.perf_counter()
        if cut_task is not None:
            drafts, draft_cut_s = await cut_task
            if callable(drafts) and os.environ.get("THUNDERSYNC_EARLY_HANDOUT", "0") == "1":
                # The deferred cut fixed every draft's length at the deadline; handing the drafts out (the
                # drafting lane's route blocks and hidden rows for those lengths) only reads logs. Start it now,
                # beside the optimizer step and the step's bookkeeping, instead of at the next step's top where
                # the learner waits for it (time/draft_handout_wait): same cut, same bytes.
                drafts = _EARLY_HANDOUT_POOL.submit(drafts).result
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
    except BaseException as exc:
        if next_task is not None:
            next_task.cancel()
        if side is not None:
            side.cancel()
        if draft_task is not None:
            draft_task.cancel()
        if cut_task is not None:
            cut_task.cancel()
        learner_task.cancel()
        learner.abort()
        if learner_error and isinstance(exc, asyncio.CancelledError):
            raise learner_error[0] from None
        raise
    learner_task.remove_done_callback(_on_learner_done)
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
    if ordered is not None:
        # The debug checks below and the optimizer step are unordered learner
        # calls: every numbered call has finished first.
        await ordered.learner_drain()
    t_rollout_end = last_rollout_t
    t_check = time.perf_counter()
    mismatch = (
        # Row order fixed by row index (not arrival), so the learner log-prob crc
        # is comparable across runs.
        train_gen_mismatch(
            sorted(all_trajectories, key=lambda t: t.index), learner, tokenizer, master_config,
            step=step,
        )
        if env_flag("THUNDERSYNC_MISMATCH_CHECK")
        else {}
    )
    grad_digests_pre_step = None
    if env_flag("THUNDERSYNC_STEP_DIGEST"):
        # Debug (resume tests): crc of every parameter's accumulated gradient on
        # every learner rank before the step's DP reduction and update, to locate
        # nondeterministic gradients between two runs of the same rollouts.
        grad_digests_pre_step = _run_workers(learner.policy, "grad_param_digests")
    if mismatch:
        # Debug forward on the step's critical path: reported so step-time
        # comparisons can subtract it (``time/mismatch_check``).
        mismatch["time/mismatch_check"] = time.perf_counter() - t_check
    t_finish = time.perf_counter()
    results = learner.finish()
    t_end = time.perf_counter()
    audit = results.get("row_audit") if isinstance(results, dict) else None
    if audit and (
        audit["rows_received"] != audit["rows_sent"]
        or audit["tokens_received"] != audit["tokens_sent"]
        or audit["digest_received"] != audit["digest_sent"]
    ):
        # Every row dealt to the learner must reach exactly one DP replica (research/tpfix:
        # dealing into world-size lists dropped half the rows at TP2).
        raise RuntimeError(f"learner row audit: dealt rows did not all reach the learner: {audit}")
    step_counts = results.get("step_counts") if isinstance(results, dict) else None
    if step_counts and step_counts["violations"]:
        # Run rule: every valid token dealt is in the DP-reduced loss denominator
        # (minus verify-in-train rejections), and TP / PP / CP peers agree on their counts.
        from thundersync_rl.run_checks import RunCheckError

        raise RunCheckError(f"learner count audit: {step_counts['violations']}")
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
        **vit_stats,
        # Non-vacuity of timing comparisons: rows whose GRPO advantage is
        # nonzero (all-zero-advantage rows may be skipped by the learner, which
        # makes training free when every reward is equal).
        "train/rows": float(len(all_trajectories)),
        # Rows the loss counts (loss_multiplier > 0); run rule 3: learner/rows_sent == train/rows_counted +
        # learner/rows_uncounted.
        "train/rows_counted": float(
            sum(1 for t in all_trajectories if float(t.payload.get("loss_multiplier", 1.0)) > 0)
        ),
        "train/nonzero_adv_rows": float(
            sum(1 for t in all_trajectories if planner._advantage(t) != 0.0)
        ),
        "learner/dispatches": num_dispatches,
        # Micro-batches of the step's planner dispatches, summed over learner DP
        # ranks: real ones and all-masked fillers (equal per-rank counts under
        # EP); per_rank = what one rank runs (sum over dispatches of the max).
        "learner/real_microbatches": float(mb_totals["real"]),
        "learner/filler_microbatches": float(mb_totals["filler"]),
        # Zero-advantage micro-batches counted for normalization, never run.
        "learner/skipped_microbatches": float(mb_totals["skipped"]),
        "learner/microbatches_per_rank": float(mb_totals["per_rank"]),
        "learner/filler_frac": float(mb_totals["filler"])
        / max(1.0, float(mb_totals["real"] + mb_totals["filler"])),
        "learner/filler_heavy_dispatches": float(mb_totals["filler_heavy"]),
        "learner/pipeline_calls": pipeline_calls,
        "learner/peak_open_buffers": results["stream_peak_open_buffers"],
        # Peak allocated / reserved memory over the step's dispatches, max over learner ranks
        # (rank 0 only before).
        "learner/mem_peak_gb": results.get("learner_mem_peak_gb", float("nan")),
        "learner/mem_reserved_gb": results.get("learner_mem_reserved_gb", float("nan")),
        "learner/prefix_pool_dropped": results.get("learner_prefix_pool_dropped", float("nan")),
        # Allocator peaks over the same window (worker.allocator_stats): reserved,
        # free-but-stranded inside partly used segments, memory outside PyTorch,
        # allocation retries (cumulative).
        "learner/mem_reserved_peak_gb": results.get("learner_mem_reserved_peak_gb", float("nan")),
        "learner/mem_frag_peak_gb": results.get("learner_mem_frag_peak_gb", float("nan")),
        "learner/mem_nontorch_gb": results.get("learner_mem_nontorch_gb", float("nan")),
        "learner/mem_alloc_retries": results.get("learner_mem_alloc_retries", float("nan")),
        "learner/mem_device_allocs": results.get("learner_mem_device_allocs", float("nan")),
        "learner/mem_device_frees": results.get("learner_mem_device_frees", float("nan")),
        # MoE device-dispatch capacity overflows that reran a learner call on Megatron's dispatcher (cumulative,
        # max over ranks; every EP rank reruns together). Each one runs that forward twice.
        "learner/moe_capacity_reruns": results.get("learner_moe_capacity_reruns", float("nan")),
        # Max over learner ranks (mem_* above), the tightest rank's headroom, the largest micro-batch
        # (rows x padded length) and the rank with the peak.
        "learner/mem_headroom_gb": results.get("learner_mem_headroom_gb", float("nan")),
        "learner/max_mb_tokens": results.get("learner_max_mb_tokens", float("nan")),
        "learner/mem_peak_rank": results.get("learner_mem_peak_rank", float("nan")),
        # Largest micro-batch that fits by this step's own peak (mem_budget.measured_fit; nan when off).
        "learner/mem_cap_fit_tokens": _cap_fit(results),
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
        # Rows / valid tokens dealt to the learner vs received by its DP replicas (always on;
        # streaming.row_audit_merge). Digests only under THUNDERSYNC_ROW_AUDIT=1.
        **({f"learner/{k}": float(v) for k, v in audit.items()} if audit else {}),
        # DP-reduced loss denominators (run rule 2; learner/tokens_sent == global_valid_toks + tokens_uncounted).
        **(
            {f"learner/{k}": float(v) for k, v in step_counts.items() if k != "violations"}
            if step_counts
            else {}
        ),
        # Training forward (fwd+bwd path, graphs, capacity reruns, packing) vs the rollout
        # log-probs (THUNDERSYNC_TRAIN_LP_CHECK; the ratio is forced to 1 in the loss).
        **(
            {f"trainlp/{k}": float(v) for k, v in results["train_lp_check"].items()}
            if isinstance(results, dict) and results.get("train_lp_check")
            else {}
        ),
        **(replay_metrics if order_replay else {}),
        # Optimizer-shard phase-in + Adam update statistics (drift_log.py; all-reduced on the learner).
        **(
            {k: float(v) for k, v in (results.get("drift") or {}).items()}
            if isinstance(results, dict)
            else {}
        ),
    }
    if grad_digests_pre_step is not None:
        out["digest/grads"] = grad_digests_pre_step
    if replay_row is not None:
        out["spec/replayed_multi_turn_rows"] = float(replayed[0])
    if side_task is not None:
        out["_side"] = side_result
        out["learner/side_hold_s"] = side_hold[0]
        out["learner/side_hold_expired"] = float(side_hold_expired[0])
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


DRAFT_ADMISSIONS = ("group", "primaries_first", "rounds", "fit")


def resolve_draft_admission(ts_cfg) -> str:
    """The ⊥-BV draft admission a config runs (``draft_admission: auto`` resolved).

    auto -> "fit" with cohort lanes and ⊥-BV (unless null_bv_prompt_drop, with
    which fit is not built), else "group". A pure function of the
    config: the same value at validation, at draft ordering and in the engine.
    """
    a = getattr(ts_cfg, "draft_admission", "auto")
    if a != "auto":
        return a
    lanes = int(getattr(ts_cfg, "cohort_lanes", 1) or 1) > 1
    # Anchor epochs: lambda cohorts share the engine; every draft must be admitted at once.
    lanes = lanes or getattr(ts_cfg, "anchor_epoch", None) is not None
    if lanes and getattr(ts_cfg, "null_bv", False) and not getattr(ts_cfg, "null_bv_prompt_drop", False):
        return "fit"
    return "group"


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
        # Submission (= engine admission) order, a function of the counts and
        # the batch order only; each pool's own order (draft index within the
        # group) is unchanged either way.
        admission = resolve_draft_admission(ts_cfg)
        if admission in ("rounds", "fit"):
            # Primaries, then the extras by their index in the pool across
            # groups: extra G of every group, then extra G + 1, ... Each
            # pool's own order (primaries, then its extras by index) is kept.
            # "fit" cuts this list to the drafting engine's capacity at
            # submission (SpeculativeGeneration.draft_streaming): the extras
            # left out are the last rounds, a function of counts only.
            primary = [i * n_drafts + j for i in range(next_batch.size) for j in range(G)]
            extra = [i * n_drafts + j for j in range(G, n_drafts) for i in range(next_batch.size)]
            nxt = nxt.select_indices(primary + extra)
        elif admission == "primaries_first":
            # Every group's first G drafts (the ones its slots take unless
            # some are BOT), then the extras, which decode in what is left of
            # the window: the most rows drafted, but the extras a BOT calls
            # for are the late ones, about one per group.
            primary = [i * n_drafts + j for i in range(next_batch.size) for j in range(G)]
            extra = [i * n_drafts + j for i in range(next_batch.size) for j in range(G, n_drafts)]
            nxt = nxt.select_indices(primary + extra)
        # "group": whole pools in batch order, so whatever does not fit the
        # engine's capacity is whole late groups, and the early groups are
        # fully drafted, extras included.
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


def share_null_calibration(specs: list[SpeculativeGeneration]) -> None:
    """One ⊥-BV calibration for every lane: v and the verify cost are properties
    of the drafter and the verifier, not of a lane. Still past-only: steps are
    verified one after another, so each step's policy sees earlier steps only.
    """
    shared = next((s._null_calib for s in specs if s._null_calib is not None), None)
    for s in specs:
        if s._null_calib is not None:
            s._null_calib = shared


def make_speculative_generation(
    generation: GenerationInterface,
    policy: Policy,
    tokenizer,
    master_config: ThunderSyncMasterConfig,
    stream: int = 0,
    run_start_step: int = 0,
) -> SpeculativeGeneration:
    """The speculative wrapper of one engine (one per lane with cohort lanes).

    ``stream``: seed stream of this wrapper (the lane index): verification
    randomness is hash(grpo.seed, stream, step, counter) (``run_state``).
    """
    spec = _make_speculative_generation(
        generation, policy, tokenizer, master_config, stream, run_start_step
    )
    spec.pieces_in_flight = max(1, int(master_config.thundersync.sticky_pieces_in_flight))
    spec.draft_cut_slack_s = max(0.0, float(master_config.thundersync.draft_cut_slack_s))
    spec.engine_verify_prefetch = bool(master_config.thundersync.engine_verify_prefetch)
    v0b = master_config.thundersync.verify0_batch_tokens
    spec.verify0_batch_tokens = int(v0b) if v0b else None
    return spec


def _make_speculative_generation(
    generation: GenerationInterface,
    policy: Policy,
    tokenizer,
    master_config: ThunderSyncMasterConfig,
    stream: int,
    run_start_step: int = 0,
) -> SpeculativeGeneration:
    ts_cfg = master_config.thundersync
    spec = SpeculativeGeneration(
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
        draft_admission=resolve_draft_admission(ts_cfg),
        sticky_chunk_tokens=ts_cfg.sticky_continuation_chunk_tokens,
        sticky_chunk_growth=ts_cfg.sticky_continuation_chunk_growth,
        sticky_run_factor=ts_cfg.sticky_continuation_run_factor,
        sticky_decision=ts_cfg.sticky_continuation_decision,
        sticky_decision_dip=ts_cfg.sticky_continuation_decision_dip,
        draft_until_complete=ts_cfg.draft_until_complete,
        window_pace=(
            dict(
                margin=ts_cfg.draft_window_pace_margin,
                max_wait_frac=ts_cfg.draft_window_pace_max_wait_frac,
                init_s=ts_cfg.draft_window_pace_init_s,
            )
            if ts_cfg.draft_window_pace
            else None
        ),
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
                "size": ts_cfg.learner_prefix_cache_size,
                "policy": ts_cfg.learner_prefix_cache_policy,
                "row_cap_tokens": master_config.policy["max_total_sequence_length"],
                "min_hit_frac": ts_cfg.learner_prefix_cache_min_hit_frac,
            }
            if ts_cfg.learner_prefix_cache
            else None
        ),
        learner_prefix_cache_verify0=bool(
            ts_cfg.learner_prefix_cache and getattr(ts_cfg, "learner_prefix_cache_verify0", False)
        ),
        seed=int(master_config.grpo.seed),
        seed_stream=str(stream),
        engine_sampler=str(getattr(ts_cfg, "engine_sampler", "wide")),
        sampler_segment=int(run_start_step),
        wide_key64=bool(getattr(ts_cfg, "wide_sampler_key64", True)),
        engine_max_len=(
            master_config.policy["generation"].get("mcore_generation_config") or {}
        ).get("max_model_len"),
    )
    spec.verify_in_train = bool(ts_cfg.verify_in_train)
    spec.verify_yield = bool(ts_cfg.verify_yield)
    spec.vit_group_size = int(master_config.grpo.num_generations_per_prompt)
    spec.vit_mbs = int(train_micro_batch_rows(master_config))
    spec.vit_min_pass_prob = float(ts_cfg.vit_min_pass_prob)
    spec.vit_checkpoint_bytes = float(ts_cfg.vit_checkpoint_gb) * 2**30
    spec.vit_bytes_per_token = float(ts_cfg.vit_bytes_per_token)
    spec.vit_predictor = vit.PassPredictor(ts_cfg.vit_pass_window)
    return spec


def _ev_lane_heads(lane_x, lane_y, step: int, pending: list, ts_cfg) -> None:
    """Engine verification on lanes: LM-head stashes and the step's verifier.

    Lane y (refit to theta_step: the verifier of cohort ``step`` and the
    drafter of cohort step+1) stashes its own head as version ``step``; the
    head of version step-1 (cohort ``step``'s drafter, lane x) must be on lane
    y for q rows: it was sent last step (``pending``). Lane y's head is sent
    to lane x now, which verifies cohort step+1 next step (rank i -> rank i,
    the lanes have the same layout). Per rank: one V/TP x d x 2 B transfer per
    step through the object store, off the critical path except its first use.
    """
    gx = lane_x.generation._policy.worker_group
    gy = lane_y.generation._policy.worker_group
    # Nothing here waits; the first decision of the step waits for the stash
    # and last step's import (``EngineVerifier.wait_refs``). Their order on
    # lane y does not matter (both keep {step-1, step}). Every other order is
    # a Ray data dependency, not submission order: the workers are asyncio
    # actors, which run calls in the order their arguments resolve (an
    # export sent right after a stash whose arguments are object-store refs
    # ran first and shipped None, so the next step's verifier had no q head).
    #   export of `step` on y      after the stash of `step` on y;
    #   import of `step` into x    after its export (the argument), and after
    #                              last step's import (whose argument was x's export
    #                              of step-1, so x's stash is read before it is dropped).
    prev = list(pending) or [None] * len(gx.workers)
    pending.clear()
    stash = [wy.stash_head.remote(step, [step - 1, step]) for wy in gy.workers]
    wait = [r for r in prev if r is not None] + stash
    pending.extend(
        wx.import_head_stash.remote(
            step, wy.head_stash_cpu.remote(step, after=s), [step, step + 1], after=a
        )
        for wx, wy, s, a in zip(gx.workers, gy.workers, stash, prev)
    )
    lane_x.spec.engine_verifier = EngineVerifier(
        gy,
        q_lane=(
            [gx.workers[0], list(gx.workers)]
            if ts_cfg.hidden_transport == "recompute"
            else None
        ),
        wait_refs=wait,
    )


# Early draft handout (THUNDERSYNC_EARLY_HANDOUT=1; default off until a Super A/B): one worker thread that starts a deferred cut's handout
# right after the cut; the next step's top waits on its result.
_EARLY_HANDOUT_POOL = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="ts-early-handout")


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

    Run control (``run_state.py``): batches come from
    ``grpo.max_num_epochs`` epochs of ``dataloader`` until ``grpo.max_num_steps``;
    checkpoints (``checkpointer``, NeMo-RL's, with ``checkpointing.enabled``)
    and validation (``val_period`` / ``val_at_start`` / ``val_at_end``, plain
    routing) run at quiescent step boundaries; ``resume_path``: the checkpoint
    this run continues from (``run_state.resolve_resume``; setup already
    loaded its weights and optimizer)."""
    validate_config(master_config)
    ts_cfg = master_config.thundersync
    if getattr(ts_cfg, "async_lanes", False):
        # Lanes-k async baseline (async_lanes.py): its own loop over k engine lanes (k = 1: the setup engine).
        from thundersync_rl.async_lanes import train_async_lanes

        if lanes is None or len(lanes) != int(ts_cfg.rollout_lag):
            raise ValueError(
                f"thundersync.async_lanes with rollout_lag={ts_cfg.rollout_lag} needs that many engine lanes "
                f"(run_thundersync_grpo.py builds them), got {'none' if lanes is None else len(lanes)}"
            )
        if val_dataloader is not None and (
            (master_config.grpo.val_period or 0) > 0 or master_config.grpo.val_at_start or master_config.grpo.val_at_end
        ):
            raise ValueError("thundersync.async_lanes: validation is not built yet")
        return train_async_lanes(
            policy, lanes, dataloader, tokenizer, loss_fn, task_to_env, master_config, logger, on_step,
            checkpointer=checkpointer, grpo_save_state=grpo_save_state, resume_path=resume_path,
        )
    fp16_range_metrics(policy, master_config)  # turns the learner's fp16 range counter on
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
        # Anchor epochs: periodic boundaries at epoch starts anchor_start + k x period.
        offset=int(ts_cfg.anchor_start) if ts_cfg.anchor_epoch is not None else 0,
    )
    timeout = None
    if ckpt_on and ckpt_cfg.get("checkpoint_must_save_by"):
        from nemo_rl.utils.timer import TimeoutChecker

        timeout = TimeoutChecker(timeout=ckpt_cfg["checkpoint_must_save_by"], fit_last_save_time=True)
        timeout.start_iterations()
    history = []
    la2 = None  # cohort-lane lookahead 2 bookkeeping (lanes only)
    stale = False  # setup() already performed the initial refit
    step = int(resume["next_step"]) if resume is not None else 0
    spec = None
    sched = None
    sentinel = WeightSentinel(policy, enabled=ts_cfg.weight_sentinel)
    check_learner_layout(policy, bool(ts_cfg.verify_yield))
    check_train_mem_cap(policy, master_config)
    if lanes is not None:
        # Cohort lanes on the streaming loop: one
        # SpeculativeGeneration per lane; each step's cohort runs on lane
        # step % 2 (verification, tails, fresh rows, streaming training) while
        # the other lane drafts the next cohort from the top of the step.
        for lane in lanes:
            lane.generation.prepare_for_generation()
            lane.spec = make_speculative_generation(
                lane.generation, policy, tokenizer, master_config, stream=lane.index,
                run_start_step=step,
            )
        share_null_calibration([lane.spec for lane in lanes])
        share_window_pacer([lane.spec for lane in lanes])
        sched = CohortLanes(lanes)
        spec = lanes[0].spec
        if int(getattr(ts_cfg, "cohort_lane_lookahead", 1) or 1) >= 2:
            # Two-period drafting (la2.py); depth k > 2 on k lanes (lanes-k): k-period drafting.
            la2 = Lookahead2(depth=int(ts_cfg.cohort_lane_lookahead))
            for lane in lanes:
                # Versions before the current one whose drafting heads the learner keeps (1 at depth 2).
                lane.spec.la2_keep_previous = True if la2.depth == 2 else la2.keep_previous()
            if ts_cfg.cohort_lane_lookahead_adaptive:
                # One policy shared by both lanes, checkpointed with the specs' policy state (resume makes the
                # same choices as the uninterrupted run).
                la_policy = AdaptiveLookahead(
                    v_hi=float(ts_cfg.lookahead_adaptive_v_hi),
                    v_lo=float(ts_cfg.lookahead_adaptive_v_lo),
                    age_ratio=float(ts_cfg.lookahead_adaptive_age_ratio),
                )
                for lane in lanes:
                    lane.spec._la_policy = la_policy
    elif ts_cfg.speculate:
        policy_generation.prepare_for_generation()
        spec = make_speculative_generation(
            policy_generation, policy, tokenizer, master_config, run_start_step=step
        )
    specs = [lane.spec for lane in lanes] if lanes is not None else ([spec] if spec else [])
    for sp in specs:
        sp.sentinel = sentinel
    # Anchored deep pipeline AEP-1E (aep.py): one engine at theta_anchor0 for the whole run.
    aep = None
    q_broker = None  # hidden_transport=recompute under anchor epochs (q_proxy.py)
    if ts_cfg.anchor_epoch is not None:
        if lanes is not None or spec is None:
            raise ValueError("thundersync.anchor_epoch: one engine with speculation on")
        w = int(ts_cfg.anchor_start)
        # anchor_start is an absolute step; a resumed run starts at an epoch start (its checkpoint
        # boundary) or before w (the regular path then runs up to w). Its first anchor is staged from the restored
        # state exactly as the uninterrupted run staged it at that step.
        aep = AnchorEpoch(
            lam=int(ts_cfg.draft_lookahead),
            epoch=int(ts_cfg.anchor_epoch),
            anchor0=max(w, step),
            start=w,
            reserve_rows=int(ts_cfg.anchor_reserve_rows),
            kind=str(ts_cfg.anchor_weights),
            h=int(ts_cfg.anchor_momentum_steps),
        )
        # Steps before the anchor step run the regular one-engine path; ``aep_switch`` turns AEP on at it and
        # re-anchors at every later epoch start.
        if ts_cfg.hidden_transport == "recompute":
            # q rows rebuilt on the anchor engine on demand: learner ranks call engine rank 0, which
            # prefills and gathers the captured rows from its peers.
            from thundersync_rl.q_proxy import make_broker

            gw = policy_generation._policy.worker_group.workers
            q_broker = make_broker(gw[0], list(gw))
            ray.get([w.set_q_engine.remote(q_broker) for w in policy.worker_group.workers])
    # Streaming verify0: one past-only risk model for every lane.
    v0_risk = Verify0Risk()
    if resume is not None:
        seeds = resume.get("seeds") or {}
        for i, sp in enumerate(specs):
            if str(i) in seeds:
                sp.load_seed_state(seeds[str(i)])
        pstate = resume.get("policy_state") or {}
        for i, sp in enumerate(specs):
            if str(i) in pstate:
                sp.load_policy_state(pstate[str(i)])
        defaulted = policy_state_defaults(specs, pstate)
        if defaulted:
            print(
                "[resume] WARNING policy-state fields absent from the checkpoint (saved by an older tree), "
                f"left at their defaults: {', '.join(defaulted)}",
                flush=True,
            )
        saved_aep = resume.get("aep")
        if (saved_aep is None) != (ts_cfg.anchor_epoch is None):
            raise RuntimeError("resume: the checkpoint and this run disagree on anchor_epoch")
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
    if spec is None and lanes is None and getattr(ts_cfg, "plain_engine_sampler", "engine") == "wide":
        if keyed_fresh_enabled(master_config):
            print("[wide sampler] plain_engine_sampler=wide skipped: debug keyed fresh sampling is installed", flush=True)
        else:
            policy_generation.prepare_for_generation()  # engine up (a no-op when it serves)
            install_plain_wide_sampler(policy_generation, master_config, segment=step)
    if spec is None and lanes is None and getattr(ts_cfg, "plain_engine_sampler", "engine") == "exact_int":
        # validate_exact_int refuses keyed fresh sampling with it (the learner trains lambda).
        policy_generation.prepare_for_generation()
        install_plain_exact_int_sampler(policy_generation, master_config, segment=step)
    # Sampled checks (THUNDERSYNC_CHECK_EVERY, checks_sched.py): the per-refit weight fingerprints
    # stay off (the always-on weight sentinel checks every refit); the rest switches per step.
    check_sched = CheckSchedule.from_env()
    check_weights = env_flag("THUNDERSYNC_MISMATCH_CHECK") and check_sched is None

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
    aep_fp0 = None
    aep_stats: dict[str, float] = {}

    def aep_engine_fp() -> float:
        # Debug (CHECKS): the engine's weight checksum, read from the engine; constant over an epoch (R2).
        return sum(
            ray.get(
                policy_generation._policy.worker_group.run_all_workers_single_data(
                    "engine_weight_checksum"
                )
            )
        )

    def aep_switch(step: int) -> None:
        """Top of the anchor step (quiescent: the step before it drafted nothing): put the anchor on the engine once.

        raw: theta_step (the regular refit if the engine is stale). momentum: theta_hat = Psi_h(theta_step, m, v),
        staged through the momentum stage (learner loads it shard by shard, refit, its LM head stashed as version
        ``step``, sentinel checks, theta_step restored bitwise); the engine never holds theta_step.
        """
        nonlocal stale, aep_fp0
        if drafts is not None or spec.plans or aep.pending:
            raise RuntimeError(f"anchor_epoch: the anchor step {step} is not quiescent (drafts / plans left)")
        if spec.__dict__.get("_aep_streams_open"):
            # Epoch switch: every stream of the previous epoch must have ended (continuations of its
            # last cohort finished in its step, surplus drafts were stopped by verify0) before the engine is
            # re-anchored (R2: no tail continues on the new anchor).
            rank0 = spec.base._policy.worker_group.workers[0]
            t_wait = time.perf_counter()
            running = ray.get(rank0.draft_request_ids.remote())
            while running and time.perf_counter() - t_wait < 30.0:
                # An aborted stream's final frame may still be in flight (its done flag lags the abort).
                time.sleep(0.05)
                running = ray.get(rank0.draft_request_ids.remote())
            if running:
                raise RuntimeError(
                    f"anchor_epoch (R2): {len(running)} draft streams still decode on anchor "
                    f"theta_{aep.engine_version} at the switch to epoch start {step}"
                )
            # The next epoch's first cohort starts fresh rank-0 tables (start_drafts).
            spec._aep_streams_open = False
        aep.switch(step)
        spec.current_step = step
        engine_group = policy_generation._policy.worker_group
        if ts_cfg.hidden_transport == "recompute":
            ray.get(q_broker.set_version.remote(None))
        policy_generation.prepare_for_generation()
        if aep.kind == "momentum":
            spec.stage_draft_weights(
                lambda: refit_policy_generation(policy, policy_generation, False),
                stage_on_engine=False,
                steps=aep.h,
                kind="momentum",
            )
        elif stale:
            refit_policy_generation(policy, policy_generation, False)
            sentinel.check(policy_generation, f"anchor theta_{step}")
        stale = False
        policy_generation.prepare_for_generation()
        spec.aep = aep
        # q's version of every continuation piece: the anchor (the engine never holds another version).
        spec.cohort_draft_version = aep.anchor0
        if ts_cfg.hidden_transport == "recompute":
            # The version q recompute requests must find on the engine (R2 for rebuilt q rows).
            ray.get(q_broker.set_version.remote(aep.anchor0))
        if check_weights:
            aep_fp0 = aep_engine_fp()
        if os.environ.get("THUNDERSYNC_ANCHOR_DIGEST") == "1":
            # Debug (resume tests): the anchor the engine holds, per engine rank. A resumed run must
            # stage bitwise the anchor the uninterrupted run staged at the same epoch start.
            cks = ray.get(engine_group.run_all_workers_single_data("engine_weight_checksum"))
            aep_stats["aep/anchor_checksum"] = "|".join(repr(float(c)) for c in cks)
            print(f"[aep] anchor step={step} kind={aep.kind} checksum={aep_stats['aep/anchor_checksum']}", flush=True)

    def aep_start(step: int, rows_now: int) -> None:
        """Start every cohort of (step, step + lambda] not drafted yet, on the anchor (aep.py).

        ``rows_now``: rows of this step's cohort (fresh rows at the anchor step, else at most one live stream
        per output slot); with the budgets of the drafting cohorts they bound the engine's live rows (R4).
        """
        rank0 = spec.base._policy.worker_group.workers[0]
        cap = 0
        started = []
        for t in aep.targets(step, draft_limit, len(upcoming)):
            b = batch if t == step else upcoming[t - step - 1].batch  # t == step: momentum anchor cohort
            prompts, slots = next_draft_prompts(b, ts_cfg, G)
            outputs = int(sum(slots.values())) if slots else len(prompts)
            cap = spec._engine_capacity(rank0)
            budget = aep.draft_budget(cap, outputs)
            spec.aep_fit_cap = budget
            tag = aep.anchor(t)
            coll = asyncio.run(
                spec.draft_streaming(prompts, null_slots=slots, target=t, version=tag)
            )
            aep.started(t, step, tag, coll, budget)
            started.append(t)
        planned = sum(int(aep.budgets[k] or 0) for k in aep.pending) + int(rows_now)
        if cap and planned > cap:
            raise RuntimeError(
                f"anchor_epoch (R4): {planned} rows planned on an engine of capacity {cap}"
            )
        aep_stats.update(
            {
                "aep/in_flight": float(len(aep.in_flight())),
                "aep/started": float(len(started)),
                "aep/last_started_target": float(max(started, default=-1)),
                "aep/slots_planned_max": float(planned),
            }
        )
        if cap:
            aep_stats["aep/capacity"] = float(cap)
        if started:
            aep_stats["aep/draft_budget"] = float(aep.budgets[started[-1]])

    # Engine-side verification: lanes verify each cohort on the
    # drafting lane (it holds theta_step); one engine verifies on itself.
    ev_mode = spec is not None and ts_cfg.verify_on == "engine"
    # Lookahead drafts (X5, lookahead_draft.py): drafter-only lane refits send theta~ = theta_k + alpha D.
    la_alpha = lookahead_draft.alphas()
    la_on = la_alpha is not None
    if la_on:
        la_problems = []
        if sched is None:
            la_problems.append("needs cohort lanes (one engine both drafts and decodes target rows)")
        if spec is not None and spec.draft_weights != "current":
            la_problems.append(f"draft_weights={spec.draft_weights} (momentum drafts are their own lookahead)")
        if ts_cfg.verify_source != "engine":
            la_problems.append("needs verify_source=engine (the drafting head stash; full-weight stashes not built)")
        if ev_mode:
            la_problems.append("verify_on=engine is not built with lookahead drafts")
        if aep is not None:
            la_problems.append("anchor epochs are not built with lookahead drafts")
        if not master_config.policy["generation"].get("refit_transport") == "mcore":
            la_problems.append("needs refit_transport=mcore (the native refit's source-side transform)")
        if la_problems:
            raise ValueError(f"{lookahead_draft.ENV}={la_alpha}: " + "; ".join(la_problems))
        la_load = os.path.join(resume_path, "policy", "lookahead") if resume_path is not None else None
        la_init = ray.get(policy.worker_group.run_all_workers_single_data(
            "lookahead_init", alpha=list(la_alpha), load_dir=la_load, start_step=step
        ))
        for lane in lanes:
            lane.spec.lookahead_drafts = True
        print(
            f"[lookahead] alpha={la_alpha} resume={la_load} restored={[x['restored'] for x in la_init]} "
            f"pinned host per learner rank (2 buffers) {max(x['host_gib'] for x in la_init):.2f} GiB",
            flush=True,
        )
    ev_pipe = ev_mode and sticky and ts_cfg.stream_training_over_tails
    ev_head_import: list = []

    def _canonical_bg() -> int:
        """Groups per canonical dispatch block (deal_rows / PP aware; config and layout only)."""
        return canonical_groups_per_block(
            ts_cfg,
            learner.dp_size,
            master_config.grpo.num_generations_per_prompt,
            train_micro_batch_rows(master_config),
            resolve_min_pipeline_microbatches(
                ts_cfg,
                master_config.policy["megatron_cfg"].get("pipeline_model_parallel_size", 1)
                or 1,
            ),
        )
    # Premise check of engine-side verification on one engine (sticky, streaming).
    prefill_check = os.environ.get("THUNDERSYNC_ENGINE_PREFILL_CHECK") == "1"
    check_setters = []
    if check_sched is not None:
        _groups = [policy.worker_group] + (
            [lane.generation._policy.worker_group for lane in lanes]
            if lanes is not None
            else ([policy_generation._policy.worker_group] if policy_generation is not None else [])
        )
        check_setters = [
            (lambda env, g=g: ray.get(g.run_all_workers_single_data("set_check_env", env=env)))
            for g in _groups
        ]
        print(
            f"[checks] full exactness checks on steps with step % {check_sched.every} == "
            f"{check_sched.phase}; other steps: {check_sched.off or 'none'}",
            flush=True,
        )
    step_checks_full = None
    if ev_mode and sched is None:
        spec.engine_verifier = EngineVerifier(policy_generation._policy.worker_group)
        spec.ev_stash_on_engine = True
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
    if spec is not None and env_flag("THUNDERSYNC_SPEC_SELFTEST") and resume is None:
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
        itertools.islice(
            stream,
            max(
                max(1, ts_cfg.draft_lookahead) + 1,
                # Depth k (la2.py): the cohort lane of step s drafts cohort s + k from upcoming[k - 1].
                la2.depth if la2 is not None else 0,
            ),
        )
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
        if aep is not None and (aep.pending or not (aep.is_epoch_start(at_step) or at_step >= max_steps
                                                     or at_step <= aep.start)):
            # Validation refits the engine: only where no cohort is in flight (epoch starts, the run's end).
            aep.refuse_refit(f"validation at step {at_step}")
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
        if la2 is not None and la2.leftover():
            problems.append(f"lookahead-2 drafts for cohorts {la2.leftover()}")
        if aep is not None and aep.leftover():
            problems.append(f"anchor-epoch drafts for cohorts {aep.leftover()}")
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
        if aep is not None and next_step < max_steps and next_step > aep.start and not aep.is_epoch_start(next_step):
            raise RuntimeError(
                f"checkpoint before step {next_step}: not an epoch start of the anchor epochs (starts "
                f"{aep.start} + k x {aep.period}); a resume there would need the in-flight cohorts"
            )
        state = {
            "next_step": next_step,
            "data": data_state,
            "seeds": {str(i): sp.seed_state() for i, sp in enumerate(specs)},
            # Past-only policy state (⊥-BV calibration, VIT pass predictor, window pacer): exactness
            # does not depend on it, but with it a resumed run makes the uninterrupted run's choices.
            "policy_state": {str(i): sp.policy_state() for i, sp in enumerate(specs)},
            # rollout_lag=1 (async1): the next step's rollouts, sampled at theta_{next_step-1} during the last
            # step, persisted so the resumed run trains exactly what the uninterrupted run trains.
            "pending_rollouts": pending if lag else None,
            # Anchor epochs: the boundary is an epoch start (no cohort in flight); recorded for the resume check.
            "aep": (
                None
                if aep is None
                else {"start": aep.start, "period": aep.period, "kind": aep.kind, "h": aep.h,
                      "anchor0": aep.anchor0, "switches": aep.switches}
            ),
            "lanes": (
                [{"index": ln.index, "version": ln.version} for ln in lanes]
                if lanes is not None
                else None
            ),
            "fingerprint": config_fingerprint(master_config),
        }
        if la_on:
            ray.get(policy.worker_group.run_all_workers_single_data(
                "lookahead_save", path=os.path.join(path, "policy", "lookahead")
            ))
        if env_flag("THUNDERSYNC_RESUME_DIGEST"):
            state["digest"] = resume_digest(policy, upcoming, stream, specs, pending if lag else None, aep)
            print(f"[resume digest] save next_step={next_step} {state['digest']}", flush=True)
        save_state(path, state)
        checkpointer.begin_finalization(path, wait_fn=policy.finalize_async_save)
        secs = time.perf_counter() - t
        print(f"[thundersync] checkpoint {path} (next step {next_step}) in {secs:.1f} s", flush=True)
        return path

    if step == 0 and grpo_cfg.val_at_start and val_dataloader is not None:
        run_validation(0)
    drafts = None
    cohort_mode = (
        spec is not None
        and ts_cfg.verify_mode == "block"
        and ts_cfg.draft_lookahead > 1
        and aep is None
    )
    lag = ts_cfg.rollout_lag
    pending = None  # rollout_lag=1: rollouts of the next step's batch
    if lag and resume is not None:
        # Persisted at the boundary; None only for a checkpoint taken before any step ran.
        pending = resume.get("pending_rollouts")
        if pending is None and step > 0:
            raise RuntimeError(
                f"resume at step {step} with rollout_lag=1: the checkpoint holds no pending rollouts (saved by an "
                "older tree?); resuming would train fresh on-policy rollouts at this step"
            )
    if resume is not None and env_flag("THUNDERSYNC_RESUME_DIGEST"):
        d = resume_digest(
            policy, upcoming, stream, specs, pending if lag else None, aep,
            saved_policy_state=resume.get("policy_state") or {},
        )
        print(f"[resume digest] load next_step={step} {d}", flush=True)
        saved = resume.get("digest")
        if saved is not None and saved != d:
            raise RuntimeError(f"resume digest differs: saved {saved} loaded {d}")
    handout_pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="ts-handout")
    def _prompts(b):
        r = b.repeat_interleave(G)
        return [
            torch.cat([m["token_ids"] for m in r["message_log"][i]])
            for i in range(r.size)
        ]

    # Time-triggered save (checkpoint_must_save_by): requested past every cohort already drafted (the cohort
    # lookahead), then the run stops at that checkpoint.
    timed_save = None
    if timeout is not None:
        timed_save = TimedSave(
            bounds,
            cohort_lookahead(ts_cfg, sched is not None),
            timeout.check_save,
            align=aep.next_epoch_start if aep is not None else None,
        )
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
        aep_on = aep is not None and step >= aep.start
        if aep is not None and step < aep.start and ts_cfg.hidden_transport == "recompute":
            # Before the first anchor (regular one-engine path): the engine holds theta_{step-1}, the drafting
            # version of this step's cohort, until refit_for_step (q recompute checks it).
            ray.get(q_broker.set_version.remote(step - 1 if step > 0 else 0))
        if aep is not None and step < aep.start:
            # Quiescent before the anchor step: no cohort >= anchor0 is drafted on another version.
            draft_limit = min(draft_limit, aep.start)
        t0 = time.perf_counter()
        if check_sched is not None:
            step_checks_full = check_sched.apply(step, check_setters)
            prefill_check = os.environ.get("THUNDERSYNC_ENGINE_PREFILL_CHECK") == "1"
        t_verify = 0.0
        t_first = 0.0
        # A deferred cut (``cut_drafts``): hand the drafts out beside the
        # drafting lane's refit below (different lane, no learner call).
        handout = handout_pool.submit(drafts) if callable(drafts) else None
        lane_x = lane_y = lane_drafts = None
        lane_stats: dict[str, float] = {}
        if sched is not None:
            lane_x, lane_y = sched.cohort_lane(step), sched.drafting_lane(step)
            spec, policy_generation = lane_x.spec, lane_x.generation
            if prefill_check:
                # Every lanes step records its verified rows (checked at the end of the step).
                spec.prefill_check_begin()
            for lane in lanes:
                lane.spec.current_step = step
            # Lookahead 2: the continuation pieces of cohort step carry its drafting version.
            if la2 is not None:
                lane_x.spec.cohort_draft_version = (
                    cohort_version(lane_x.cohort_version, step) if lane_x.cohort == step else None
                )
            early_t = la2.starts.get(step + 1) if la2 is not None else None
            early = la2.take(step + 1) if la2 is not None else None
            if early is not None:
                # Cohort step+1 has drafted on lane_y since cohort step+1-depth was final (theta_{step+1-depth}).
                ev = step + 1 - la2.depth
                if lane_y.cohort != step + 1 or lane_y.cohort_version != ("theta", ev):
                    raise RuntimeError(
                        f"lookahead {la2.depth}: lane {lane_y.index} must hold cohort {step + 1} at theta_{ev}, "
                        f"holds cohort {lane_y.cohort} at {lane_y.cohort_version}"
                    )
                lane_drafts = early
                lane_stats["lanes/la2_early"] = 1.0
                lane_stats["lanes/draft_start_s"] = early_t - t0
            elif next_batch is not None and step + 1 < draft_limit:
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
                        refit_lane(policy, y, lookahead=la_on)
                        sentinel.check(y.generation, f"drafter theta_{step} (lane {y.index})", lookahead=la_on)

                sched.refit(lane_y, version, put_drafter, lookahead=la_on and not momentum)
                sched.pin(lane_y, step + 1)
                if check_weights:
                    lane_y.cohort_fingerprint = lane_fp(lane_y)
                if ev_mode:
                    _ev_lane_heads(lane_x, lane_y, step, ev_head_import, ts_cfg)
                bench_rows = int(os.environ.get("THUNDERSYNC_EV_PREFILL_BENCH", "0") or 0)
                bench = ev_mode and bench_rows > 0 and step in (1, 2)
                gy = lane_y.generation._policy.worker_group
                if bench:
                    lane_x.spec.engine_prefill_bench(
                        gy, _prompts(batch)[:bench_rows],
                        master_config.policy["generation"]["max_new_tokens"], "alone",
                    )
                next_prompts, null_slots = next_draft_prompts(next_batch, ts_cfg, G)
                lane_drafts = asyncio.run(
                    lane_y.spec.draft_streaming(next_prompts, null_slots=null_slots)
                )
                lane_stats["lanes/draft_start_s"] = time.perf_counter() - t0
                if bench:
                    time.sleep(1.0)  # drafts in full decode
                    lane_x.spec.engine_prefill_bench(
                        gy, _prompts(batch)[:bench_rows],
                        master_config.policy["generation"]["max_new_tokens"], "mixed",
                    )
            elif ev_mode and drafts is not None:
                # Last step: no next cohort, but the verifier lane must hold theta_step.
                sched.refit(lane_y, ("theta", step), lambda: refit_lane(policy, lane_y))
                _ev_lane_heads(lane_x, lane_y, step, ev_head_import, ts_cfg)
        if lag and pending is None:
            # Step 0 of the lagged baseline: its rollouts come from theta_0.
            t1 = time.perf_counter()
            policy_generation.prepare_for_generation()
            pending, _ = asyncio.run(
                _generate_rollouts(
                    batch, policy_generation, tokenizer, task_to_env, master_config, key_step=step
                )
            )
            t_first = time.perf_counter() - t1
        if spec is not None:
            spec.current_step = step
            spec.step_t0 = t0  # step clock for the one-engine tail/one_* metrics (speculative._one)
        handout_wait = 0.0
        if handout is not None:
            t1 = time.perf_counter()
            drafts = handout.result()
            handout_wait = time.perf_counter() - t1
        if aep_on:
            aep_stats = {}
            if aep.is_epoch_start(step):
                aep_switch(step)  # first anchor, or an epoch switch
            # Every rollout of an anchored cohort comes from a plan (drafts, or empty-prefix continuations).
            spec.aep_need = collections.Counter(tuple(x.tolist()) for x in _prompts(batch))
            if aep.is_epoch_start(step) and aep.kind == "momentum":
                # The anchor step's own cohort is anchored (age 0): drafted on theta_hat now, cut once every draft
                # finished (no truncation), verified below like every cohort.
                aep_start(step, 0)
                coll = aep.take(step)
                t1 = time.perf_counter()
                spec._aep_cut_complete = True
                try:
                    drafts = coll()
                finally:
                    spec._aep_cut_complete = False
                aep_stats["aep/anchor_cohort_draft_s"] = time.perf_counter() - t1
            if not aep.engine_holds_target(step) and drafts is None:
                raise RuntimeError(
                    f"anchor_epoch: cohort {step} has no drafts; its rows would decode on the anchor engine "
                    f"(version {aep.anchor0}), not at theta_{step}"
                )
        tails = None  # pending plans whose continuation rounds overlap training
        v0 = None  # streaming verify0: its calls run inside the step
        ev_drafts = ev_block_keys = None
        if ev_pipe and drafts is not None:
            # Engine verification pipelined against training: the drafts are
            # verified inside the step, block by block (verify_sticky_blocks).
            policy_generation.prepare_for_generation()
            ev_row_keys = [tuple(x.tolist()) for x in _prompts(batch)]
            ev_n_groups = len(ev_row_keys) // G
            ev_bg = max(1, int(ts_cfg.engine_verify_block_groups or _canonical_bg()))
            ev_b0 = max(1, int(ts_cfg.engine_verify_first_block_groups or ev_bg))
            ev_starts = [0] + list(range(ev_b0, ev_n_groups, ev_bg))
            ev_group_blocks = [
                list(range(a, min(b, ev_n_groups)))
                for a, b in zip(ev_starts, ev_starts[1:] + [ev_n_groups])
                if a < ev_n_groups
            ]
            ev_block_keys = [
                list(dict.fromkeys(ev_row_keys[p * G] for p in blk)) for blk in ev_group_blocks
            ]
            spec.prepare_verify_blocks(ev_block_keys)
            ev_drafts, drafts = drafts, None
            tails = []
        elif sticky and drafts is not None:
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
            if ts_cfg.stream_training_over_tails and ts_cfg.verify0_stream:
                # Only the schedule here (counts and past steps); verify0's
                # calls, the decisions and the tails run inside the step.
                v0 = spec.prepare_stream(
                    drafts,
                    v0_risk,
                    ts_cfg.verify0_stream_first_chunk_tokens,
                    ts_cfg.verify0_stream_chunk_tokens,
                    ts_cfg.verify0_stream_window,
                    ts_cfg.verify0_stream_order,
                )
                tails = []  # pending plans arrive with the decisions
                if prefill_check:
                    spec.prefill_check_begin()
            elif ts_cfg.stream_training_over_tails:
                bench_rows = int(os.environ.get("THUNDERSYNC_EV_PREFILL_BENCH", "0") or 0)
                if bench_rows and sched is None and step in (1, 2):
                    # One engine at the top of the step: its draft streams still
                    # decode (continuations), so this is the mixed case.
                    spec.engine_prefill_bench(
                        policy_generation._policy.worker_group, _prompts(batch)[:bench_rows],
                        master_config.policy["generation"]["max_new_tokens"], "mixed",
                    )
                # Only the drafts' verification here; the continuation rounds
                # run inside the step, under the training of complete groups.
                spec.vit_dp = learner.dp_size
                tails = asyncio.run(spec.begin_sticky(drafts))
                if prefill_check:
                    spec.prefill_check_begin()
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
            # Streaming verify0: unknown at the step top (taken as not covered).
            and v0 is None
            and spec.plans_cover(_prompts(batch), pending_ok=tails is not None)
        )
        refit_s = []
        la2_ready: list[int] = []
        # Engine prefill check (THUNDERSYNC_ENGINE_PREFILL_CHECK): only once the
        # cohort's engine holds theta_step (set by refit_for_step).
        pc_at_theta = [False]

        async def prefill_check_now():
            if sched is not None:
                # Lanes: at the end of the step (below), when the rollout loop has taken
                # every plan; the cohort lane keeps theta_step until the next step's refits.
                return
            if prefill_check and pc_at_theta[0]:
                await spec.engine_prefill_check()
            elif prefill_check:
                spec._prefill_check_rows = None
                spec.stats["spec/engine_prefill_check_skipped"] = 1.0

        async def la2_begin():
            # Lookahead 2: the cohort lane, now at theta_step, drafts cohort step+2
            # until the cut at the end of step+1 (la2.py); depth k: cohort step+k until the end of step+k-1.
            if not la2_ready:
                return
            target = la2_ready.pop()
            sched.pin(lane_x, target)
            if check_weights:
                lane_x.cohort_fingerprint = lane_fp(lane_x)
            la_prompts, la_slots = next_draft_prompts(upcoming[la2.depth - 1].batch, ts_cfg, G)
            coll = await lane_x.spec.draft_streaming(la_prompts, null_slots=la_slots, target=target)
            now = time.perf_counter()
            la2.started(target, step, coll, now)
            lane_stats["lanes/la2_next_start_s"] = now - t0

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
                la2_go = la2 is not None and la2.may_start(
                    step, draft_limit, len(upcoming) >= la2.depth, bool(lane_fresh), lane_x.cohort
                )
                la_pol = lane_x.spec.__dict__.get("_la_policy")
                if la2 is not None and la_pol is not None:
                    # Phase-adaptive lookahead: earlier cohorts' verification statistics only (scheduling).
                    want2 = la_pol.decide(la_pol.forecast(lane_x.spec.__dict__.get("_null_calib"), step))
                    forced = os.environ.get("THUNDERSYNC_DEBUG_LA_FORCE")
                    if forced:
                        # Debug (switch-boundary tests): lookahead per step from a list "2,1,1,2,..." (cycled).
                        modes = [int(m) for m in forced.split(",")]
                        want2 = modes[step % len(modes)] == 2
                        la_pol.mode = 2 if want2 else 1
                    la2_go = la2_go and want2
                    lane_stats["lanes/la_mode"] = float(la_pol.mode)
                    lane_stats["lanes/la_v2_forecast"] = la_pol.last_forecast
                    lane_stats["lanes/la_switches"] = float(la_pol.switches)
                if lane_fresh or la2_go:
                    # Fresh rows decode at theta_step; lookahead 2 drafts cohort
                    # step+2 with the same live weights (la2.py). Lookahead drafts: a lane
                    # without fresh rows only drafts, so it takes theta~ instead.
                    la_x = la_on and not lane_fresh
                    if la_on:
                        lane_stats["lookahead/alpha"] = lookahead_draft.alpha_at(la_alpha, step)
                        lane_stats["lookahead/lane_refit"] = float(la_x)
                    sched.refit(
                        lane_x, ("theta", step),
                        lambda: refit_lane(policy, lane_x, lookahead=la_x), lookahead=la_x,
                    )
                    sentinel.check(lane_x.generation, f"theta_{step} (lane {lane_x.index})", lookahead=la_x)
                    lane_x.generation.prepare_for_generation()
                    pc_at_theta[0] = not la_x
                    if la2_go:
                        la2_ready.append(step + la2.depth)
                else:
                    lane_x.generation.prepare_for_generation()
                # Without the refit the lane keeps the drafting weights: a
                # rollout that then decodes there raises (``_one``) instead of
                # sampling from them (v0s-xlanes at 6fc1ae9c).
                spec.fresh_at_current = bool(lane_fresh)
                refit_s.append(time.perf_counter() - t_r)
                lane_stats["lanes/fresh_refit"] = float(lane_fresh)
                return
            if aep_on:
                # Anchor epochs: the engine keeps theta_anchor0 for the whole run (R2); it decodes no row at
                # theta_step except at the anchor step itself (cohort 0, engine version = theta_step).
                policy_generation.prepare_for_generation()
                spec.fresh_at_current = aep.engine_holds_target(step)
                # Engine prefill check (engine p == learner p): only where the engine holds theta_step, i.e.
                # the anchor step; at every later step the engine holds theta_anchor0 != theta_step, so the
                # check is skipped (spec/engine_prefill_check_skipped) and engine-vs-learner rests on the
                # engine-q check (theta_0 head stash) and the route check.
                pc_at_theta[0] = aep.engine_holds_target(step)
                refit_s.append(0.0)
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
                if q_broker is not None:
                    ray.get(q_broker.set_version.remote(step))
            else:
                policy_generation.prepare_for_generation()
            # draft_now: the engine took theta_hat_{step+1}; nothing may decode at theta_k.
            if spec is not None:
                spec.fresh_at_current = not draft_now
            pc_at_theta[0] = not draft_now
            refit_s.append(time.perf_counter() - t_r)

        tails_s = []
        group_order = group_blocks = tail_groups = None
        v0_order = None
        order_feed = None
        if ev_drafts is not None:
            order_feed = OrderFeed(step, ts_cfg.canonical_block_tokens)
            tail_groups = []
            ev_info: dict = {}
            ev_fresh: list = []

            async def ev_then_refit(
                d=ev_drafts, keys=ev_block_keys, blocks=ev_group_blocks, row_keys=ev_row_keys
            ):
                nonlocal lane_fresh
                t = time.perf_counter()

                def on_block(b, info):
                    ev_info.update(info)
                    cls = classify_groups(row_keys, G, blocks[b], info)
                    ready = [p for p in blocks[b] if cls[p] == "ready"]
                    if ready:
                        order_feed.extend(ready, [len(ready)])

                def after_verify():
                    every = [p for blk in blocks for p in blk]
                    cls = classify_groups(row_keys, G, every, ev_info)
                    tails_g = order_tail_groups(
                        [p for p in every if cls[p] == "tail"],
                        row_keys,
                        G,
                        spec.tail_kept,
                        ts_cfg.tail_group_order,
                    )
                    fresh_g = [p for p in every if cls[p] == "fresh"]
                    ev_fresh.append(bool(fresh_g))
                    tail_groups.extend(tails_g)
                    if ts_cfg.canonical_block_tokens:
                        # Each class is one segment, cut by tokens as it completes.
                        order_feed.extend(tails_g, [len(tails_g)])
                        order_feed.extend(fresh_g, [len(fresh_g)])
                    else:
                        bg = _canonical_bg()
                        order_feed.extend(tails_g, block_chunks(tails_g, bg))
                        order_feed.extend(fresh_g, block_chunks(fresh_g, bg))
                    order_feed.done()

                try:
                    await spec.verify_sticky_blocks(d, keys, on_block, after_verify)
                    policy_generation.finish_generation()
                    tails_s.append(time.perf_counter() - t)
                    await spec.learner_drain()
                    # Rows never drafted need theta_k on the cohort lane (the
                    # plans may already be taken by served rollouts, so read
                    # the verification's classes, not spec.plans).
                    lane_fresh = sched is not None and bool(ev_fresh and ev_fresh[0])
                    refit_for_step()
                    await la2_begin()
                except BaseException as exc:
                    spec.weights_ready(error=exc)
                    raise
                spec.weights_ready()

        refit_at_top = False  # the refit runs here, outside the step body (step_decomposition)
        if ev_drafts is not None:
            pass
        elif tails is None:
            refit_for_step()
            refit_at_top = True
            if la2_ready:
                asyncio.run(la2_begin())
            if prefill_check and sched is None and spec is not None and spec.sticky_delta is None and pc_at_theta[0]:
                # One engine, θ-only routing (cross-fade / plain): the engine holds theta_step for the whole step
                # (drafts of the next cohort are drafted on it too), so every row of the step is recorded as it
                # finishes and checked at the end of the step (engine prefill p == the row's logged p: learner
                # verification p on drafted / residual tokens, engine decode p on continuation and fresh tokens).
                spec.prefill_check_begin()
        elif v0 is not None:
            # Streaming verify0: the canonical order grows with the decisions
            # (ready groups of each decided call, then the tail groups, then
            # groups with rows never drafted; StreamOrder), a function of the
            # schedule (counts, past steps) and the decisions.
            row_keys = [tuple(x.tolist()) for x in _prompts(batch)]
            v0_order = StreamOrder(
                [row_keys[p * G] for p in range(len(row_keys) // G)],
                collections.Counter(row_keys),
                canonical_groups_per_block(
                    ts_cfg,
                    learner.dp_size,
                    master_config.grpo.num_generations_per_prompt,
                    train_micro_batch_rows(master_config),
                    resolve_min_pipeline_microbatches(
                        ts_cfg,
                        master_config.policy["megatron_cfg"].get(
                            "pipeline_model_parallel_size", 1
                        )
                        or 1,
                    ),
                ),
                block_tokens=ts_cfg.canonical_block_tokens,
                # Rows that decode at theta_k whatever their plans (debug fresh
                # rows): their groups count as fresh, so the cohort lane takes
                # theta_k for them (lanes) and they train after the tails.
                fresh_keys=stream_fresh_keys(row_keys),
                tail_order=lambda tails, rk=row_keys: order_tail_groups(
                    tails, rk, G, spec.tail_kept, ts_cfg.tail_group_order
                ),
            )
            group_order, group_blocks = v0_order.order, v0_order.blocks
            tail_groups = v0_order.tail_groups

            async def tails_then_refit(v0=v0, order=v0_order):
                # verify0's calls and the continuation rounds on the drafting
                # weights, then theta_k: rollouts never drafted and the next
                # cohort's drafts wait for it.
                nonlocal lane_fresh
                t = time.perf_counter()
                try:
                    await spec.complete_stream(v0, order.decided, order.close)
                    policy_generation.finish_generation()
                    tails_s.append(time.perf_counter() - t)
                    await spec.learner_drain()
                    # Lanes: the cohort lane takes theta_k only if a row decodes fresh.
                    lane_fresh = sched is not None and bool(order.fresh)
                    refit_for_step()
                    await prefill_check_now()
                    await la2_begin()
                except BaseException as exc:
                    spec.weights_ready(error=exc)
                    raise
                spec.weights_ready()
        else:
            # Groups whose rows still decode tails (a pending plan of their
            # prompt), fixed by the drafts' verification, not by timing.
            # Incomplete after verify0: continuation rounds on the drafting
            # version, or (anchor horizon) decoded by the refit engine at theta_k.
            pending_keys = {
                key
                for key, queue in spec.plans.items()
                for plan in queue
                if not plan.complete and not plan.vit
            }
            vit_keys = {
                key for key, queue in spec.plans.items() for plan in queue if plan.vit
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
                # Verify-in-train groups: trained by their own calls before any
                # dispatch (passed ones leave the order); a failed one trains here,
                # after the groups ready now (its tails decode meanwhile).
                vit_groups = [
                    p for p in range(len(row_keys) // G) if row_keys[p * G] in vit_keys
                ]
                late = set(tail_groups) | fresh | set(vit_groups)
                classes = [
                    [p for p in range(len(row_keys) // G) if p not in late],
                    [p for p in vit_groups if p not in fresh],
                    order_tail_groups(
                        [p for p in tail_groups if p not in fresh],
                        row_keys,
                        G,
                        spec.tail_kept,
                        ts_cfg.tail_group_order,
                    ),
                    sorted(fresh),
                ]
                group_order = [p for c in classes for p in c]
                if ts_cfg.canonical_block_tokens:
                    # Each class is a segment; the planner cuts it into blocks
                    # of canonical_block_tokens tokens as its groups complete.
                    group_blocks = [len(c) for c in classes if c]
                else:
                    # Blocks of canonical_block_groups within each class.
                    bg = canonical_groups_per_block(
                        ts_cfg,
                        learner.dp_size,
                        master_config.grpo.num_generations_per_prompt,
                        train_micro_batch_rows(master_config),
                        resolve_min_pipeline_microbatches(
                            ts_cfg,
                            master_config.policy["megatron_cfg"].get(
                                "pipeline_model_parallel_size", 1
                            )
                            or 1,
                        ),
                    )
                    group_blocks = [
                        min(bg, len(c) - i) for c in classes for i in range(0, len(c), bg)
                    ]

            async def tails_then_refit(pending=tails):
                # Continuation rounds on the drafting weights, then theta_k:
                # rollouts never drafted and the next cohort's drafts wait for it.
                t = time.perf_counter()
                try:
                    await spec.complete_tails(pending)
                    # Verify-in-train: failed groups' tails also run on the drafting weights.
                    await spec.vit_complete_tails()
                    policy_generation.finish_generation()
                    tails_s.append(time.perf_counter() - t)
                    # The momentum stage and the refit call the learner: no
                    # training dispatch may be in flight meanwhile.
                    # The momentum stage and the refit are unordered learner
                    # calls: every ordered call sent so far finishes first, and
                    # none is sent while they run (they block the event loop).
                    await spec.learner_drain()
                    refit_for_step()
                    # Premise check: the engine now holds theta_k (= the learner's);
                    # before lookahead 2 starts drafting on it.
                    await prefill_check_now()
                    await la2_begin()
                except BaseException as exc:
                    spec.weights_ready(error=exc)
                    raise
                spec.weights_ready()
        if aep_on:
            # After verify0 stopped this cohort's surplus streams (k >= 1) or the engine is prepared (step 0).
            aep_start(step, len(_prompts(batch)))
        pre_rollout = (
            ev_then_refit
            if ev_drafts is not None
            else (None if tails is None else tails_then_refit)
        )
        chunked = (
            spec is not None
            and ts_cfg.verify_mode == "block"
            and (ts_cfg.verify_on == "learner" or ts_cfg.verify_overlap)
            # ⊥-BV pools are decided in one call before the rollouts start.
            and not ts_cfg.null_bv
        )
        # Chunked verification overlaps the step's training dispatches on the
        # learner: both are numbered learner calls then (``ordered_call``), as
        # while streaming over the tails, so every rank runs them in one order.
        ordered_chunks = False
        if cohort_mode and chunked:
            cohort = spec.cohorts.pop(step, None)
            if cohort is not None:
                cdrafts = [[cohort_draft(d)] for d in cohort]
                pre_rollout = lambda c=cdrafts: spec.verify_chunked(
                    c, ts_cfg.verify_chunks
                )  # noqa: E731
                ordered_chunks = tails is None
        elif chunked and drafts is not None:
            pre_rollout = lambda d=drafts: spec.verify_chunked(d, ts_cfg.verify_chunks)  # noqa: E731
            ordered_chunks = tails is None
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
        if aep_on:
            # Anchor epochs: cohort step+1 has drafted since step max(0, step+1-lambda); cut at this step's end.
            aep_coll = aep.take(step + 1)

            async def drafter(collector=aep_coll):
                return collector

            if aep_coll is None:
                drafter = None
        if sched is not None:
            # The next cohort's drafts already run on the drafting lane.
            if lane_drafts is not None:

                async def drafter(collector=lane_drafts):
                    return collector

            else:
                drafter = None
        if (
            step == 0
            and sched is None
            and spec is not None
            and int(os.environ.get("THUNDERSYNC_EV_PREFILL_BENCH", "0") or 0)
        ):
            # Idle engine (step 0, before its rollouts): prefill throughput alone.
            spec.engine_prefill_bench(
                policy_generation._policy.worker_group,
                _prompts(batch)[: int(os.environ["THUNDERSYNC_EV_PREFILL_BENCH"])],
                master_config.policy["generation"]["max_new_tokens"],
                "alone",
            )
        policy.prepare_for_training()
        metrics = asyncio.run(
            _run_streaming_tails(spec, tails is not None, ordered=ordered_chunks)(
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
                order_stream=v0_order,
                order_feed=order_feed,
                ordered_learner=spec if tails is not None or ordered_chunks else None,
                # Sticky streams: cut the next drafts while training ends.
                cut_while_training=tails is not None,
                vit_spec=spec if tails is not None and ts_cfg.verify_in_train else None,
                claim_log_weight=spec.claim_log_weight
                if spec is not None and ts_cfg.banded_is_band is not None
                else None,
                stale_rollouts=pending,
                generate_next=(
                    (
                        lambda b=next_batch, k=step + 1: _generate_rollouts(
                            b, policy_generation, tokenizer, task_to_env, master_config, key_step=k
                        )
                    )
                    if lag and next_batch is not None and step + 1 < max_steps
                    else None
                ),
            ))
        )
        if (
            prefill_check
            and sched is None
            and spec is not None
            and spec.sticky_delta is None
            and getattr(spec, "_prefill_check_rows", None) is not None
        ):
            if pc_at_theta[0] and not staged:
                asyncio.run(spec.engine_prefill_check())
                spec.stats["spec/engine_prefill_check_one_engine"] = 1.0
            else:
                spec._prefill_check_rows = None
                spec.stats["spec/engine_prefill_check_skipped"] = 1.0
        if prefill_check and sched is not None:
            # Engine prefill check on the cohort lane (it still holds theta_step if it took
            # it this step: fresh rows or lookahead 2); every row of the step is recorded now.
            if (pc_at_theta[0] or env_flag("THUNDERSYNC_DEBUG_PC_STALE_LANE")) and spec._prefill_check_rows:
                # (debug negative control THUNDERSYNC_DEBUG_PC_STALE_LANE: check on the cohort lane even when it
                # still holds the drafting version; the check must then report inexact tokens)
                asyncio.run(spec.engine_prefill_check())
                spec.stats["spec/engine_prefill_check_lane"] = 0.0
            elif (
                spec._prefill_check_rows and lane_y is not None and lane_y.version == ("theta", step)
                and not lane_y.lookahead
            ):
                # Lookahead 1 (or the fallback after a boundary): the cohort lane never takes theta_step
                # without fresh rows, but the drafting lane was refit to theta_step at the top of the step
                # (it drafts cohort step+1 on it). Check this step's rows on that engine: same weights as
                # the learner's verification, so engine p == learner p must hold there too.
                lane_y.spec._prefill_check_rows = spec._prefill_check_rows
                spec._prefill_check_rows = None
                asyncio.run(lane_y.spec.engine_prefill_check())
                for k in [k for k in lane_y.spec.stats if k.startswith("spec/engine_prefill_check")]:
                    spec.stats[k] = lane_y.spec.stats.pop(k)
                spec.stats["spec/engine_prefill_check_lane"] = 1.0
            else:
                spec._prefill_check_rows = None
                spec.stats["spec/engine_prefill_check_skipped"] = 1.0
        t_refit = refit_s[0]
        if tails is not None or ordered_chunks:
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
            if spec is not None and tails is not None:
                # verify_yield: verifications run at training boundaries this step.
                metrics.update(spec.yield_stats())
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
        if ev_drafts is not None:
            spec._plan_cv = None  # per-step gate of the pipelined verification
        metrics["time/verify"] = t_verify
        # Top of step waiting for the deferred draft handout (after the refit).
        metrics["time/draft_handout_wait"] = handout_wait
        if spec is not None:
            metrics.update(spec.stats)
            spec.stats = {}
        if aep_on:
            metrics.update(aep_stats)
            a = aep.anchor(step)
            metrics["aep/age"] = float(aep.age(step))
            metrics["aep/anchor_version"] = float(-1 if a is None else a)
            metrics["aep/engine_version"] = float(aep.engine_version)
            metrics["aep/momentum_anchor"] = float(aep.kind == "momentum")
            if ts_cfg.hidden_transport == "recompute":
                qs = ray.get(q_broker.pop_stats.remote())
                metrics.update({f"qrc/{k}": v for k, v in (qs or {}).items()})
            metrics["aep/epoch_index"] = float(aep.epoch_index(step))
            metrics["aep/switches"] = float(aep.switches)
            # Rows the anchor engine sampled without verification (only cohort 0 may have any; `_one` raises
            # otherwise, this is the run-level record).
            metrics["aep/engine_rows_after_anchor"] = (
                0.0 if aep.engine_holds_target(step) else float(metrics.get("exact/rows_engine_cont", 0.0) or 0.0)
            )
            # Coverage of fused verification (VIT): groups decided inside training / groups of the step.
            n_groups = max(1, len(_prompts(batch)) // G)
            metrics["aep/vit_group_frac"] = float(metrics.get("vit/groups", 0.0) or 0.0) / n_groups
            metrics["aep/vit_passed_group_frac"] = float(metrics.get("vit/passed_groups", 0.0) or 0.0) / n_groups
            # Per-age record of v (log p/q variance of this cohort's verified drafts) for the E2 comparison.
            if metrics.get("spec/logratio_var_step") is not None:
                metrics[f"aep/v_age{int(aep.age(step))}"] = float(metrics["spec/logratio_var_step"])
            if aep_fp0 is not None:
                moved = aep_engine_fp() != aep_fp0
                metrics["aep/engine_checksum_moved"] = float(moved)
                if moved:
                    raise RuntimeError(
                        f"anchor_epoch (R2): the engine's weights changed during step {step} "
                        f"(it must hold theta_{aep.anchor0} for the epoch)"
                    )
        if sched is not None:
            metrics.update(lane_y.spec.stats)
            lane_y.spec.stats = {}
            lane_stats["lanes/cohort_lane"] = float(lane_x.index)
            lane_stats["lanes/refits"] = float(sum(lane.refits for lane in lanes))
            metrics.update(lane_stats)
        if not (la2 is not None and lane_x is not None and lane_x.cohort == step + la2.depth):
            # (lookahead 2: the cohort lane already drafts cohort step+2 (depth k: step+k); keep it running)
            policy_generation.finish_generation()
        stale = True
        metrics["time/refit"] = t_refit
        metrics["time/iteration"] = time.perf_counter() - t0
        if env_flag("THUNDERSYNC_ENGINE_PROF"):
            # Debug: engine step timing of this step, every engine of every lane.
            gens = (
                [(f"lane{lane.index}", lane.generation) for lane in lanes]
                if sched is not None
                else [("", policy_generation)]
            )
            cohort = f"lane{lane_x.index}" if sched is not None else ""
            metrics.update(engine_prof_metrics(step, gens, cohort))
        # Plans left by this step were verified against its policy only.
        metrics["spec/plans_dropped"] = float(sum(sp.end_step() for sp in specs))
        # Long runs: free rank 0's per-draft entries of the cohorts trained now and old closed log
        # keys; then the leak / health probe (leak_probe.py).
        for sp in specs:
            for k, v in sp.release_consumed(step).items():
                metrics[k] = metrics.get(k, 0.0) + v
        if leak_probe.enabled():
            groups = {"learner": policy.worker_group}
            if lanes is not None:
                groups.update({f"engine{ln.index}": ln.generation._policy.worker_group for ln in lanes})
            elif policy_generation is not None and getattr(policy_generation, "_policy", None) is not None:
                groups["engine"] = policy_generation._policy.worker_group
            metrics.update(
                leak_probe.collect(
                    groups, specs, [os.environ.get("THUNDERSYNC_RUN_DIR")], history_len=len(history) + 1
                )
            )
        if env_flag("THUNDERSYNC_STEP_DIGEST"):
            # Debug (resume tests): bitwise digests of the weights after this
            # step's update (per rank, folded) and of this step's rollouts.
            ranks = _run_workers(policy, "resume_state_digest")
            metrics["digest/params"] = "|".join(r["params"] for r in ranks)
            metrics["digest/optimizer"] = "|".join(str(r.get("optimizer")) for r in ranks)
        metrics.update(sentinel.pop_stats())
        metrics.update(fp16_range_metrics(policy, master_config))
        _reforward_frac(metrics, policy)
        if sticky:
            metrics.update(anchor_missing_metrics(
                [lane.generation for lane in lanes] if sched is not None else [policy_generation]
            ))
        metrics["step"] = step
        metrics["data/epoch"] = float(stream.epoch if not upcoming else upcoming[0].state["epoch"])
        if step_checks_full is not None:
            metrics["checks/full"] = float(step_checks_full)
        metrics.update(
            step_decomposition(
                metrics,
                STEP_TOP_KEYS + (("time/refit",) if refit_at_top else ()),
                verify0_in_body=v0 is not None,
            )
        )
        history.append(metrics)
        _log_step(
            step, metrics, logger, history,
            master_config.policy.get("train_micro_batch_size"),
            {"verify_yield": bool(master_config.thundersync.verify_yield)},
        )
        if on_step is not None:
            on_step(history)
        if health_strict() and metrics.get("health/warn"):
            raise RuntimeError(f"THUNDERSYNC_HEALTH_STRICT: step {step} {metrics['health/flags']}")
        step += 1
        if timeout is not None:
            timeout.mark_iteration()
            b = timed_save.after_step(step)
            if b is not None:
                print(f"[thundersync] checkpoint_must_save_by due after step {step - 1}: time-triggered checkpoint "
                      f"before step {b} (cohort lookahead {timed_save.lookahead}), then the run stops", flush=True)
        last = step >= max_steps or not upcoming
        val_metrics = None
        if val_dataloader is not None and (
            bounds.validates_after(step - 1) or (last and grpo_cfg.val_at_end)
        ):
            val_metrics = run_validation(step)
        if ckpt_on and (bounds.saves_after(step - 1) or last):
            save_boundary(step, val_metrics, final=last)
            if not last and timed_save is not None and timed_save.stops_after(step):
                # The job's time is nearly up: end with this checkpoint as the run's last state (resubmitting
                # resumes here) instead of training steps the job limit would throw away.
                print(f"[thundersync] stopping after the time-triggered checkpoint before step {step}", flush=True)
                break
            stop_at = int(os.environ.get("THUNDERSYNC_DEBUG_STOP_AT", "0") or 0)
            if stop_at > 0 and step == stop_at and not last:
                # Debug (resume tests): die like a killed job right after a durable
                # checkpoint: no clean shutdown of the loop, workers or engines.
                checkpointer.shutdown()
                print(f"[thundersync] debug stop after step {step} (checkpoint durable)", flush=True)
                os._exit(0)
    if checkpointer is not None:
        checkpointer.shutdown()
    return history


def _rollouts_crc(pending) -> Optional[int]:
    """CRC of persisted lagged rollouts (rollout_lag=1): token ids, log-probs and rewards of every row."""
    if pending is None:
        return None
    import pickle

    h = 0
    for i, state, _ in sorted(pending, key=lambda x: x[0]):
        parts = [str(i).encode()]
        for m in state.get("message_log", []):
            for k in ("token_ids", "generation_logprobs"):
                v = m.get(k)
                if isinstance(v, torch.Tensor):
                    parts.append(v.detach().cpu().contiguous().numpy().tobytes())
        for k in ("total_reward",):
            v = state.get(k)
            parts.append(pickle.dumps(v.tolist() if isinstance(v, torch.Tensor) else v))
        h = zlib.crc32(b"".join(parts), h)
    return h


def _policy_state_crc(specs, saved: Optional[dict] = None) -> int:
    """crc of the specs' past-only policy state. ``saved`` (at a resume: the checkpoint's policy_state, by spec
    index): only the objects / fields the checkpoint holds count (each compared exactly), so a tree that added
    policy-state fields since the save still matches what it restored; the added fields keep their defaults and
    are reported by ``policy_state_defaults``."""
    states = [sp.policy_state() for sp in specs]
    if saved is not None:
        states = [
            {name: {f: v for f, v in fields.items() if f in (saved.get(str(i)) or {}).get(name, {})}
             for name, fields in st.items() if name in (saved.get(str(i)) or {})}
            for i, st in enumerate(states)
        ]
    return zlib.crc32(repr(states).encode())


def policy_state_defaults(specs, saved: dict) -> list[str]:
    """'spec<i>.<object>.<field>' of every policy-state field this tree has and the checkpoint does not (it stays at
    its default after the resume)."""
    out = []
    for i, sp in enumerate(specs):
        sv = saved.get(str(i)) or {}
        for name, fields in sp.policy_state().items():
            out += [f"spec{i}.{name}.{f}" for f in fields if f not in sv.get(name, {})]
    return out


def resume_digest(policy: Policy, upcoming, stream, specs, pending=None, aep=None, saved_policy_state=None) -> dict[str, Any]:
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
        # Past-only policy state, persisted async1 rollouts, the anchor-epoch schedule position.
        "policy_state_crc": _policy_state_crc(specs, saved_policy_state),
        "pending_rollouts_crc": _rollouts_crc(pending),
        "aep": None if aep is None else (aep.start, aep.period, aep.kind, aep.h),
    }


def _run_streaming_tails(spec, on: bool, ordered: bool = False):
    """Run a step coroutine with ``spec``'s streaming-over-tails state (if ``on``).

    ``ordered`` (not ``on``): only the ordered learner calls
    (``begin_ordered_learner``), for chunked verification beside training.
    """

    async def run(coro):
        if on:
            spec.begin_streaming_tails()
        elif ordered:
            spec.begin_ordered_learner()
        else:
            return await coro
        try:
            return await coro
        finally:
            if on:
                spec.end_streaming_tails()
            else:
                spec.end_ordered_learner()

    return run


def anchor_missing_metrics(gens) -> dict[str, float]:
    """``engine/anchor_missing``: tagged engine requests whose anchors were not registered.

    They raise unless THUNDERSYNC_ALLOW_MISSING_ANCHORS=1; cumulative per run, summed
    over every engine rank of ``gens``. One ray.get per engine at the step end.
    """
    total = 0.0
    for gen in gens:
        group = getattr(gen, "base", gen)._policy.worker_group
        for st in ray.get(group.run_all_workers_single_data("engine_anchor_stats")):
            total += float((st or {}).get("missing", 0.0))
    return {"engine/anchor_missing": total}


def _reforward_frac(metrics: dict, policy: Any) -> None:
    """spec/piece_prefix_reforward_frac = re-forward replica-seconds / (learner DP size x step seconds).

    Reopen trigger of the learner prefix-state cache (D-2026-10-19-coord-c83): >= 0.10.
    ``spec/piece_prefix_reforward_s`` sums one result per verified row, i.e. the seconds of the
    data-parallel replica that verified it (its TP / PP ranks spend the same seconds together), so the
    denominator is the model data-parallel size. It was the worker count, which under-reported the
    fraction by TP x PP (2x at Super TP2, 8x at TP8).
    """
    from thundersync_rl.speculative import data_parallel_size

    s = metrics.get("spec/piece_prefix_reforward_s")
    step = metrics.get("time/step_total")
    if s is None or not step:
        return
    try:
        n = max(1, data_parallel_size(policy.worker_group))
    except Exception:  # pragma: no cover
        n = 1
    metrics["spec/piece_prefix_reforward_frac"] = float(s) / (n * float(step))


def fp16_range_metrics(policy: Any, master_config: Any) -> dict[str, float]:
    """Learner fp16 range monitor (worker.fp16_range_stats), reduced over ranks.

    One ray.get per step at the step end (not on a hot path); {} without the
    fp16 GEMM recipe. The first call (loop start) only turns the counter on.
    """
    mcfg = master_config.policy.get("megatron_cfg") or {}
    if mcfg.get("forward_gemm_precision") is None:
        return {}
    per = ray.get(policy.worker_group.run_all_workers_single_data("fp16_range_stats"))
    out: dict[str, float] = {}
    for st in per:
        for k, v in (st or {}).items():
            out[k] = max(out.get(k, v), v) if k.endswith(("amax",)) else out.get(k, 0.0) + v
    return out


def engine_prof_metrics(step: int, gens, cohort: str = "") -> dict[str, float]:
    """``THUNDERSYNC_ENGINE_PROF``: per-step engine timing of every rank of every engine.

    ``gens``: [(label, generation)] (one per lane, label "" without lanes). Each
    engine's ranks are reduced by max (EP ranks step together; the slowest rank
    is what a row waits for); keys ``engine/<label>/...`` per lane, and the
    unprefixed ``engine/...`` keys are the ``cohort`` lane's (the one decoding
    this step's tails). Earlier trees logged engine rank 0 of lane 0 only, so
    lane 1 was never measured and DP replicas other than the first were dropped.
    """
    out: dict[str, float] = {}
    for label, gen in gens:
        group = getattr(gen, "base", gen)._policy.worker_group
        per_rank = ray.get(group.run_all_workers_single_data("engine_step_stats"))
        merged: dict[str, float] = {}
        for r, st in enumerate(per_rank):
            print(f"[engine prof] step={step} {label or 'engine'} rank={r} {st}", flush=True)
            for k, v in (st or {}).items():
                merged[k] = max(merged.get(k, v), v)
        for k, v in merged.items():
            out[k.replace("engine/", f"engine/{label}/", 1) if label else k] = v
            if label and label == cohort:
                out[k] = v
    return out


def _log_step(
    step: int,
    metrics: dict[str, Any],
    logger: Optional[Logger],
    history: Optional[list[dict[str, Any]]] = None,
    mbs: Optional[int] = None,
    expect: Optional[dict[str, Any]] = None,
) -> None:
    # One-line health summary (health.py): silent waste flagged per step.
    prev = history[-2] if history is not None and len(history) >= 2 else None
    health_line = step_health_annotate(step, metrics, prev, mbs, expect)
    scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    print(
        f"[thundersync step {step}] "
        + " ".join(
            f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
            for k, v in scalars.items()
        ),
        flush=True,
    )
    print(health_line, flush=True)
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
    """Training loop on cohort lanes (lanes.py).

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
    check_learner_layout(policy, bool(ts_cfg.verify_yield))
    check_train_mem_cap(policy, master_config)
    for lane in lanes:
        lane.generation.prepare_for_generation()
        lane.spec = make_speculative_generation(
            lane.generation, policy, tokenizer, master_config, stream=lane.index
        )
        lane.spec.sentinel = sentinel
        sentinel.check(lane.generation, f"initial weights (lane {lane.index})")
    share_null_calibration([lane.spec for lane in lanes])
    share_window_pacer([lane.spec for lane in lanes])
    sched = CohortLanes(lanes)
    check_weights = env_flag("THUNDERSYNC_MISMATCH_CHECK")

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
        metrics.update(fp16_range_metrics(policy, master_config))
        _reforward_frac(metrics, policy)
        metrics["time/verify"] = times["time/verify"]
        metrics["time/refit"] = times["time/refit"]
        for lane in (x, y):
            metrics.update(lane.spec.stats)
            lane.spec.stats = {}
        metrics["lanes/refits"] = float(sum(lane.refits for lane in lanes))
        metrics["time/iteration"] = time.perf_counter() - t0
        # Lanes: the cohort verification and every refit run outside the step body.
        metrics.update(step_decomposition(metrics, ("time/verify", "time/refit")))
        history.append(metrics)
        _log_step(
            step, metrics, logger, history,
            master_config.policy.get("train_micro_batch_size"),
            {"verify_yield": bool(master_config.thundersync.verify_yield)},
        )
        if on_step is not None:
            on_step(history)
        if health_strict() and metrics.get("health/warn"):
            raise RuntimeError(f"THUNDERSYNC_HEALTH_STRICT: step {step} {metrics['health/flags']}")
        batch, next_batch = next_batch, next(batches, None)
        step += 1
    return history
