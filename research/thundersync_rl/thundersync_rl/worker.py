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
