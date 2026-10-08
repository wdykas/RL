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
import time
from typing import Any, Literal, Optional

import numpy as np
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
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.logger import Logger
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
    """Flatten a chunk's message logs into a ClippedPG training batch."""
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
    return pad_batch_to_multiple(data, pad_to_multiple)


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
) -> dict[str, Any]:
    G = master_config.grpo.num_generations_per_prompt
    repeated = batch.repeat_interleave(G)
    n = repeated.size
    mbs = master_config.policy["train_micro_batch_size"]

    planner = StreamPlanner(
        adv_fn,
        dp_size=learner.dp_size,
        max_chunk_trajectories=ts_cfg.max_chunk_trajectories,
        group_only=ts_cfg.granularity == "group",
        max_open_groups=ts_cfg.max_open_groups,
    )
    for p in range(n // G):
        planner.register_group((step, p), G)

    learner.begin(
        loss_fn,
        gbs=n,
        mbs=mbs,
        storage_device=ts_cfg.storage_device,
        max_buckets_per_group=ts_cfg.max_buckets_per_group,
        # Only trajectory granularity ever holds per-group buckets.
        max_open_groups=(
            ts_cfg.max_open_groups if ts_cfg.granularity == "trajectory" else 0
        ),
    )

    t_start = time.perf_counter()
    wake = asyncio.Event()
    rollouts_done = False
    rewards = [0.0] * n
    sample_metrics: list[dict[str, Any]] = [None] * n  # type: ignore[list-item]
    learner_busy = 0.0
    num_dispatches = 0
    # Timeline for offline analysis: (t, group, tokens, reward) per arrival and
    # (t_start, t_end, n_trajectories, n_tokens, n_bucket_chunks) per dispatch.
    arrivals: list[tuple[float, int, int, float]] = []
    dispatches: list[tuple[float, float, int, int, int]] = []
    last_rollout_t = None

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
        nonlocal learner_busy, num_dispatches
        while True:
            await wake.wait()
            wake.clear()
            # Batch-synchronous baseline: hold everything until the batch closes.
            while planner.has_work() and (
                ts_cfg.granularity != "batch" or rollouts_done
            ):
                d = planner.next_dispatch()
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
                        }
                        for c in chunks
                    ]
                    for chunks in d.per_rank
                ]
                t0 = time.perf_counter()
                await asyncio.gather(*learner.submit(per_rank, d.closes))
                t1 = time.perf_counter()
                learner_busy += t1 - t0
                chunks = [c for cs in d.per_rank for c in cs]
                dispatches.append(
                    (
                        t0 - t_start,
                        t1 - t_start,
                        sum(len(c.trajectories) for c in chunks),
                        sum(c.num_tokens for c in chunks),
                        sum(1 for c in chunks if c.group is not None),
                    )
                )
                num_dispatches += 1
            if rollouts_done and not planner.has_work():
                return

    learner_task = asyncio.create_task(learner_loop())
    try:
        for fut in asyncio.as_completed([rollout(i) for i in range(n)]):
            i, state, metrics = await fut
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
            planner.add(
                Trajectory(
                    group=(step, i // G),
                    index=i,
                    reward=reward,
                    payload={
                        "message_log": state["message_log"],
                        "loss_multiplier": loss_mult,
                    },
                    num_tokens=ntok,
                )
            )
            wake.set()
        rollouts_done = True
        wake.set()
        await learner_task
    except BaseException:
        learner_task.cancel()
        learner.abort()
        raise
    assert planner.all_done()
    t_rollout_end = last_rollout_t
    t_finish = time.perf_counter()
    results = learner.finish()
    t_end = time.perf_counter()

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
        "learner/peak_open_buffers": results["stream_peak_open_buffers"],
        "timeline/arrivals": arrivals,
        "timeline/dispatches": dispatches,
        "timeline/step_end": t_end - t_start,
    }
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
    for batch in dataloader:
        if step >= max_steps:
            break
        t0 = time.perf_counter()
        if stale:
            refit_policy_generation(policy, policy_generation, False)
        else:
            policy_generation.prepare_for_generation()
        t_refit = time.perf_counter() - t0
        policy.prepare_for_training()
        metrics = asyncio.run(
            _run_one_step(
                step=step,
                batch=batch,
                learner=learner,
                policy_generation=policy_generation,
                tokenizer=tokenizer,
                task_to_env=task_to_env,
                loss_fn=loss_fn,
                master_config=master_config,
                ts_cfg=ts_cfg,
                adv_fn=adv_fn,
            )
        )
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
