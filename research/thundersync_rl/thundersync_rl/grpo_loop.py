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
import os
import time
from types import SimpleNamespace
from typing import Any, Literal, Optional

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
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.logger import Logger
from thundersync_rl.speculative import SpeculativeGeneration
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
    # Experiment: score each batch under theta_k and theta_{k+1} (two extra
    # forward passes) and log how much of it cross-iteration speculative
    # rollouts would keep (see policy_drift_metrics).
    measure_policy_drift: bool = False
    # Exact cross-iteration speculative rollouts (speculative.py): draft the next
    # iteration's rollouts under the current weights once ``draft_start_frac`` of
    # this iteration's rollouts have finished (up to ``draft_budget`` tokens
    # each), verify them on the learner under the next weights, and decode only
    # the rejected remainders.
    speculate: bool = False
    draft_budget: int = 1024
    draft_start_frac: float = 0.5
    spec_head_k: int = 64
    # Drafts per trajectory sharing its keys; variants > 0 jitter the head
    # log-probs by keyed logistic noise of scale ``variant_eps`` to explore races
    # the target may flip. The longest verified prefix is kept (exact).
    draft_variants: int = 1
    variant_eps: float = 0.02
    # "block": randomized block verification (Sun et al. 2024) on the learner with
    # the drafts' q scored at theta_k before the optimizer step (highest
    # acceptance). "keyed": shared-noise Gumbel verification (no q needed).
    verify_mode: Literal["block", "keyed"] = "block"
    # Activations used to score p and q for block verification: "model" (the
    # learner's own bf16 forward) or "fp32" (HF fp32 copy with the same weights,
    # removing activation-rounding noise from the p/q ratio).
    verify_precision: Literal["model", "fp32", "tf32"] = "model"


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


def _run_workers(policy: Policy, method: str, **kwargs) -> list[Any]:
    return ray.get(
        policy.worker_group.run_all_workers_single_data(method, **kwargs)
    )


def policy_drift_metrics(
    lp_old: torch.Tensor,
    lp_new: torch.Tensor,
    mask: torch.Tensor,
    prefix: str,
) -> dict[str, Any]:
    """Acceptance statistics for drafts sampled from ``lp_old`` verified by ``lp_new``.

    Exact speculative sampling accepts draft token x with probability
    min(1, p_new(x) / p_old(x)). For each trajectory (row) this reports the
    expected accepted prefix length sum_t prod_{s<=t} alpha_s under token-level
    verification, and the probability of accepting the whole draft under token
    (prod alpha) and block verification (b_t = min(1, b_{t-1} r_t), Sun et al.).
    """
    out: dict[str, list[float]] = {"reject": [], "frac": [], "full": [], "block": []}
    lens = []
    # One sampled verification per row: (generated tokens, tokens accepted before
    # the first rejection), for offline schedule simulation.
    samples: list[tuple[int, int]] = []
    gen = torch.Generator().manual_seed(0)
    for i in range(mask.shape[0]):
        m = mask[i].bool()
        if not m.any():
            continue
        log_r = (lp_new[i] - lp_old[i])[m].double()
        alpha = torch.clamp(log_r, max=0.0)  # log min(1, r)
        surv = torch.cumsum(alpha, 0).exp()  # P(prefix through t accepted)
        log_b = torch.zeros(())
        for lr in log_r.tolist():
            log_b = min(0.0, float(log_b) + lr)
        n = int(m.sum())
        lens.append(n)
        u = torch.rand(n, generator=gen, dtype=torch.float64)
        rejected = torch.nonzero(u > alpha.exp()).flatten()
        samples.append((n, int(rejected[0]) if rejected.numel() else n))
        out["reject"].append(float((1 - alpha.exp()).mean()))
        out["frac"].append(float(surv.sum()) / n)
        out["full"].append(float(surv[-1]))
        out["block"].append(float(np.exp(log_b)))
    w = np.array(lens, dtype=np.float64)
    long = w >= np.percentile(w, 75)
    return {
        f"{prefix}/token_reject_rate": float(np.mean(out["reject"])),
        # Token-weighted: share of all generated tokens a draft would keep.
        f"{prefix}/accepted_token_frac": float(np.sum(np.array(out["frac"]) * w) / w.sum()),
        f"{prefix}/accepted_token_frac_longest25": float(
            np.sum((np.array(out["frac"]) * w)[long]) / w[long].sum()
        ),
        f"{prefix}/full_accept_token_verif": float(np.mean(out["full"])),
        f"{prefix}/full_accept_block_verif": float(np.mean(out["block"])),
        f"{prefix}/samples": samples,
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
    planner = StreamPlanner(
        adv_fn,
        dp_size=learner.dp_size,
        max_chunk_trajectories=ts_cfg.max_chunk_trajectories,
        max_open_groups=max_open_groups,
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
    all_trajectories: list[Trajectory] = []

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
    draft_task = None
    n_done = 0
    try:
        for fut in asyncio.as_completed([rollout(i) for i in range(n)]):
            i, state, metrics = await fut
            n_done += 1
            if drafter is not None and draft_task is None and (
                n_done >= ts_cfg.draft_start_frac * n
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
            traj = Trajectory(
                group=(step, i // G),
                index=i,
                reward=reward,
                payload={
                    "message_log": state["message_log"],
                    "loss_multiplier": loss_mult,
                },
                num_tokens=ntok,
            )
            all_trajectories.append(traj)
            planner.add(traj)
            wake.set()
        rollouts_done = True
        wake.set()
        await learner_task
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
    drift_data = lp_old = None
    if ts_cfg.measure_policy_drift:
        # theta_k: every chunk is backpropagated, the optimizer has not stepped.
        drift_data = build_chunk_data(
            SimpleNamespace(
                trajectories=all_trajectories,
                group=None,
                advantages=[0.0] * len(all_trajectories),
            ),
            tokenizer,
            master_config,
            learner.dp_size * master_config.policy["logprob_batch_size"],
        )
        lp_old = learner.policy.get_logprobs(drift_data)["logprobs"]
        _run_workers(learner.policy, "save_master_weights", tag="k")
        # Shared-randomness couplings: same per-(row, position) noise at theta_k
        # and theta_{k+1}; the emitted tokens agree where a draft would be kept.
        coupling_rows = [
            drift_data["input_ids"][i, : int(drift_data["input_lengths"][i])]
            for i in range(drift_data.size)
            if float(drift_data["sample_mask"][i]) > 0
        ]
        coupled_old = [
            x
            for x in _run_workers(
                learner.policy, "coupled_samples", rows=coupling_rows, seed=step
            )
            if x is not None
        ][0]
        # Multi-iteration lookahead: drafts from theta_{k-m} verified by theta_k.
        lag_lps = {}
        for m in range(1, 5):
            if step - m < 0:
                continue
            _run_workers(
                learner.policy, "load_master_combination", coeffs={f"hist{step - m}": 1.0}
            )
            lag_lps[m] = learner.policy.get_logprobs(drift_data)["logprobs"]
        if lag_lps:
            _run_workers(learner.policy, "load_master_combination", coeffs={"k": 1.0})
        # Optimizer-state forecast of theta_{k+1} (zero new gradient). Scored
        # now, then theta_k is restored bit-exactly before the real step.
        lp_adam = None
        if step > 0:
            _run_workers(learner.policy, "save_adam_forecast", tag="adam")
            _run_workers(learner.policy, "load_master_combination", coeffs={"adam": 1.0})
            lp_adam = learner.policy.get_logprobs(drift_data)["logprobs"]
            _run_workers(learner.policy, "load_master_combination", coeffs={"k": 1.0})
            assert torch.equal(
                learner.policy.get_logprobs(drift_data)["logprobs"], lp_old
            ), "theta_k restore failed"
        # Same weights, reversed row order (different batching): the training
        # engine's own batch-variance floor.
        rev = list(range(drift_data.size))[::-1]
        lp_old_rev = learner.policy.get_logprobs(drift_data.select_indices(rev))[
            "logprobs"
        ][rev]
    t_finish = time.perf_counter()
    results = learner.finish()
    t_end = time.perf_counter()
    drift: dict[str, float] = {}
    if drift_data is not None:
        lp_new = learner.policy.get_logprobs(drift_data)["logprobs"]
        mask = drift_data["token_mask"] * drift_data["sample_mask"].unsqueeze(-1)
        gen = drift_data["generation_logprobs"]
        # Pure policy drift (same training engine on both sides).
        drift.update(policy_drift_metrics(lp_old, lp_new, mask, "drift"))
        # Drafts as actually sampled (inference engine q) verified by theta_{k+1}
        # on the training engine: drift plus the engine mismatch.
        drift.update(policy_drift_metrics(gen, lp_new, mask, "drift_vs_gen"))
        drift.update(policy_drift_metrics(lp_old, lp_old_rev, mask, "batchvar"))
        coupled_new = [
            x
            for x in _run_workers(
                learner.policy, "coupled_samples", rows=coupling_rows, seed=step
            )
            if x is not None
        ][0]
        live = [
            i for i in range(drift_data.size) if float(drift_data["sample_mask"][i]) > 0
        ]
        # Position j predicts token j+1: compare where token j+1 is generated.
        gen_mask = torch.zeros(coupled_old.shape[:2], dtype=torch.bool)
        for r, i in enumerate(live):
            n = int(drift_data["input_lengths"][i])
            gen_mask[r, : n - 1] = drift_data["token_mask"][i, 1:n].bool()
        for c, name in enumerate(("crn_cdf_id", "crn_cdf_sorted", "crn_gumbel")):
            differ = (coupled_old[..., c] != coupled_new[..., c]) & gen_mask
            fracs, w = [], []
            for r in range(differ.shape[0]):
                pos = torch.nonzero(gen_mask[r]).flatten()
                if pos.numel() == 0:
                    continue
                d = differ[r, pos]
                first = int(torch.nonzero(d)[0]) if d.any() else pos.numel()
                fracs.append(first / pos.numel())
                w.append(pos.numel())
            w_ = np.array(w, float)
            drift[f"{name}/token_reject_rate"] = float(differ.sum() / gen_mask.sum())
            drift[f"{name}/accepted_token_frac"] = float(np.sum(np.array(fracs) * w_) / w_.sum())
            drift[f"{name}/full_accept"] = float(np.mean(np.array(fracs) == 1.0))
        # Forecast drafter: extrapolate the fp32 master weights along the last
        # update, theta_hat = theta_k + c (theta_k - theta_{k-1}), round to the
        # model dtype, and score the same tokens. Drafts would be sampled from
        # theta_hat; the importance weight q_hat/q_k re-targets the per-token
        # rejection estimate from theta_k's samples.
        _run_workers(learner.policy, "save_master_weights", tag="k1")
        if lp_adam is not None:
            # Parameter-space quality of the forecast (rank 0's shard).
            dist = _run_workers(
                learner.policy, "master_distances", ref="k1", others=["k", "adam"]
            )[0]
            for k_, v_ in dist.items():
                drift[f"param/{k_}"] = v_
            drift.update(policy_drift_metrics(lp_adam, lp_new, mask, "forecast_adam"))
            w = (lp_adam - lp_old).exp()
            rej = (1 - (lp_new - lp_adam).exp()).clamp(min=0)
            drift["forecast_adam/token_reject_rate_is"] = float(
                (w * rej * mask).sum() / mask.sum()
            )
        if step > 0:
            for c in (1.0,):
                _run_workers(
                    learner.policy,
                    "load_master_combination",
                    coeffs={"k": 1.0 + c, f"hist{step - 1}": -c},
                )
                lp_hat = learner.policy.get_logprobs(drift_data)["logprobs"]
                tag = f"forecast_c{c}"
                drift.update(policy_drift_metrics(lp_hat, lp_new, mask, tag))
                w = (lp_hat - lp_old).exp()
                rej = (1 - (lp_new - lp_hat).exp()).clamp(min=0)
                drift[f"{tag}/token_reject_rate_is"] = float(
                    (w * rej * mask).sum() / mask.sum()
                )
                drift.update(policy_drift_metrics(lp_old, lp_hat, mask, f"{tag}_vs_k"))
            _run_workers(learner.policy, "load_master_combination", coeffs={"k1": 1.0})
            lp_check = learner.policy.get_logprobs(drift_data)["logprobs"]
            assert torch.equal(lp_check * mask, lp_new * mask), "restore failed"
        mask_ = drift_data["token_mask"] * drift_data["sample_mask"].unsqueeze(-1)
        for m, lp_m in lag_lps.items():
            drift.update(policy_drift_metrics(lp_m, lp_old, mask_, f"lag{m}"))
        _run_workers(learner.policy, "rename_master_weights", src="k", dst=f"hist{step}")
        _run_workers(learner.policy, "drop_master_weights", tag=f"hist{step - 5}")
        # Mismatch floor: theta_k on both, different engines.
        drift.update(policy_drift_metrics(gen, lp_old, mask, "mismatch"))

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
        "time/draft_wait": draft_wait,
        "time/draft_score": draft_score,
        "_drafts": drafts,
        **drift,
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
        )
    G = master_config.grpo.num_generations_per_prompt
    if spec is not None and os.environ.get("THUNDERSYNC_SPEC_SELFTEST"):
        first = next(iter(dataloader))
        prompts = [torch.cat([m["token_ids"] for m in first["message_log"][i]]) for i in range(4)]
        spec.draft_budget, budget = 256, spec.draft_budget
        print("[spec selftest]", asyncio.run(spec.self_test(prompts)), flush=True)
        spec.draft_budget = budget
    batches = iter(dataloader)
    next_batch = next(batches, None)
    drafts = None
    while next_batch is not None:
        batch, next_batch = next_batch, next(batches, None)
        if step >= max_steps:
            break
        t0 = time.perf_counter()
        if stale:
            refit_policy_generation(policy, policy_generation, False)
        else:
            policy_generation.prepare_for_generation()
        t_refit = time.perf_counter() - t0
        t_verify = 0.0
        if spec is not None and drafts is not None:
            t1 = time.perf_counter()
            spec.verify(drafts)
            t_verify = time.perf_counter() - t1
        drafter = None
        if spec is not None and next_batch is not None and step + 1 < max_steps:
            nxt = next_batch.repeat_interleave(G)
            next_prompts = [
                torch.cat([m["token_ids"] for m in nxt["message_log"][i]])
                for i in range(nxt.size)
            ]
            drafter = (  # noqa: E731
                (lambda prompts=next_prompts: spec.draft_streaming(prompts))
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
            )
        )
        drafts = metrics.pop("_drafts")
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
