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
"""Megatron policy worker with gradient-streaming (ThunderSyncRL) methods.

Builds on the worker's split train-step API (``begin_train_step`` /
``train_microbatch`` / ``finish_train_step``), which already accumulates
un-normalized gradients locally under ``model.no_sync()`` and defers the 1/N
normalization, the DP reduction, clipping and the optimizer step to a single
barrier. On top of that we use the DDP ``grad_data`` buffers as scratch: every
streamed chunk's backward lands there and is immediately moved into a
per-(group, reward) accumulator, so its advantage can be applied once the
group closes. At the barrier, the folded batch gradient is written back into
the grad buffers and the regular ``finish_train_step`` runs unchanged.
"""

import time
from typing import Any

import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.algorithms.grad_streaming import GradStreamingSpec, StreamBucket

WORKER_FQN = "thundersync_rl.worker.ThunderSyncMegatronPolicyWorker"


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class ThunderSyncMegatronPolicyWorker(MegatronPolicyWorkerImpl):
    """Driver-friendly wrappers over the core split API's streaming mode."""

    def stream_begin_step(
        self,
        loss_fn: LossFunction,
        *,
        gbs: int,
        mbs: int,
        storage_device: str,
        max_buckets_per_group: int,
        max_open_groups: int,
    ) -> None:
        """Open an optimizer step whose gradient arrives in streamed chunks."""
        self.begin_train_step(
            loss_fn=loss_fn,
            gbs=gbs,
            mbs=mbs,
            grad_streaming=GradStreamingSpec(
                storage_device=storage_device,
                max_buckets_per_group=max_buckets_per_group,
                max_open_groups=max_open_groups,
            ),
        )
        self._stream_busy_s = 0.0

    def stream_accumulate(
        self,
        items: list[dict[str, Any]],
        closes: list[tuple[Any, dict[float, float]]],
    ) -> dict[str, Any]:
        """Backprop streamed chunks, then fold any groups that closed.

        Each item is ``{"data": BatchedDataDict, "group": gid | None,
        "reward": float | None}``; ``group=None`` items carry final advantages.
        """
        t0 = time.perf_counter()
        for item in items:
            bucket = (
                None
                if item["group"] is None
                else StreamBucket(group=item["group"], reward=item["reward"])
            )
            # Chunks smaller than a micro-batch run as one micro-batch of exactly
            # their rows (no dummy rows); larger chunks arrive padded to a
            # multiple of the configured size and use it. Keeping the configured
            # size bounds memory and matches the synchronous trainer's numerics
            # (Megatron fp32 gradients shift ~1e-4 with micro-batch size).
            state = self._train_step_state
            step_mbs = state["mbs"]
            state["mbs"] = min(item["data"].size, step_mbs)
            try:
                self.train_microbatch(item["data"], stream_bucket=bucket)
            finally:
                state["mbs"] = step_mbs
        if closes:
            self.close_stream_groups(closes)
        torch.cuda.synchronize()
        self._stream_busy_s += time.perf_counter() - t0
        acc = self._train_step_state["grad_stream"]
        return {"num_items": len(items), "open_buffers": acc.num_open_buffers()}

    def stream_finish_step(self) -> dict[str, Any]:
        """Barrier: write back, normalize, DP-reduce, clip, step."""
        peak = self._train_step_state["grad_stream"].peak_open_buffers
        results = self.finish_train_step()
        results["stream_peak_open_buffers"] = peak
        results["stream_learner_busy_s"] = self._stream_busy_s
        return results

    def get_flat_grads(self) -> torch.Tensor:
        """Return this rank's gradient buffers on CPU (for equivalence tests).

        Valid after a step with DP=1, where the buffers hold the full
        normalized gradient until the next step zeroes them.
        """
        buffers = self.model.buffers + self.model.expert_parallel_buffers
        return torch.cat([b.grad_data.detach().float().cpu() for b in buffers])

    def get_flat_params(self) -> dict[str, torch.Tensor]:
        """Return this rank's parameters on CPU (for equivalence tests)."""
        return {
            name: p.detach().float().cpu().clone()
            for name, p in self.model.named_parameters()
        }

    # ---- Cross-iteration speculation experiments (policy forecasting) ----
    # Snapshots and linear combinations of this rank's fp32 master-weight
    # shards. Requires the distributed optimizer without the precision-aware
    # optimizer, so master weights live in ``shard_fp32_from_float16_groups``.

    def _master_shards(self) -> list[torch.Tensor]:
        shards = []
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            assert not opt.config.use_precision_aware_optimizer, (
                "forecasting needs master weights in the dist optimizer; set "
                "policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false"
            )
            for group in opt.shard_fp32_from_float16_groups + opt.shard_fp32_groups:
                shards.extend(group)
        return shards

    def save_master_weights(self, tag: str) -> None:
        """Snapshot the fp32 master shards under ``tag``."""
        if not hasattr(self, "_master_snapshots"):
            self._master_snapshots: dict[str, list[torch.Tensor]] = {}
        self._master_snapshots[tag] = [s.detach().clone() for s in self._master_shards()]

    def rename_master_weights(self, src: str, dst: str) -> None:
        self._master_snapshots[dst] = self._master_snapshots.pop(src)

    @torch.no_grad()
    def load_master_combination(self, coeffs: dict[str, float]) -> None:
        """Set master weights to sum_tag coeff * snapshot[tag] and refresh the model.

        Copies master shards into the model's param buffers and all-gathers them
        across DP, exactly as the optimizer does after a step.
        """
        for i, shard in enumerate(self._master_shards()):
            acc = torch.zeros_like(shard)
            for tag, c in coeffs.items():
                acc.add_(self._master_snapshots[tag][i], alpha=c)
            shard.copy_(acc)
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            opt._copy_main_params_to_model_params()
            opt.start_param_sync_for_bucket_group_subset()
        torch.cuda.synchronize()

    @torch.no_grad()
    def save_adam_forecast(self, tag: str) -> dict[str, float]:
        """Snapshot the next Adam(W) step's weights assuming a zero gradient.

        theta_hat = theta - lr * (m_hat / (sqrt(v_hat) + eps) + wd * theta), with
        m, v decayed one step and bias-corrected at t + 1: the part of the next
        update the optimizer state already determines. Call before the step.
        """
        shards, n_missing = [], 0
        for opt in getattr(self.optimizer, "chained_optimizers", [self.optimizer]):
            inner = opt.optimizer
            for group in inner.param_groups:
                b1, b2 = group["betas"]
                for p in group["params"]:
                    st = inner.state.get(p, {})
                    if "exp_avg" not in st:
                        n_missing += 1
                        shards.append((p, p.detach().clone()))
                        continue
                    t = int(st.get("step", group.get("step", 0))) + 1
                    m_hat = st["exp_avg"].float() * b1 / (1 - b1**t)
                    v_hat = st["exp_avg_sq"].float() * b2 / (1 - b2**t)
                    upd = m_hat / (v_hat.sqrt() + group["eps"])
                    upd.add_(p, alpha=group["weight_decay"])
                    shards.append((p, p - group["lr"] * upd))
        # Order must match _master_shards(): map by identity.
        by_id = {id(p): hat for p, hat in shards}
        self._master_snapshots[tag] = [by_id[id(s)] for s in self._master_shards()]
        return {"params_without_state": float(n_missing)}

    @torch.no_grad()
    def master_distances(self, ref: str, others: list[str]) -> dict[str, float]:
        """fp32 L2 distance and bf16 disagreement count of snapshots vs ``ref``."""
        out: dict[str, float] = {}
        refs = self._master_snapshots[ref]
        out["numel"] = float(sum(r.numel() for r in refs))
        for tag in others:
            sq = flips = 0.0
            for a, b in zip(self._master_snapshots[tag], refs):
                sq += float((a - b).double().pow(2).sum())
                flips += float((a.bfloat16() != b.bfloat16()).sum())
            out[f"{tag}/l2"] = sq**0.5
            out[f"{tag}/bf16_flips"] = flips
        return out
