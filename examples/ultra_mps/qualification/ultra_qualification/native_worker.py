# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Unscored Ultra qualification through the existing native worker extension."""

from dataclasses import asdict
import hashlib
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

import ray
import torch
from megatron.core.resharding.utils import named_refit_tensors

from nemo_rl.models.generation.megatron.megatron_worker import WORKER_CONCURRENCY_GROUPS
from nemo_rl.models.megatron import sm_partition
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from nemo_rl.models.policy.workers.megatron_policy_worker import (
    MegatronPolicyWorkerImpl,
)
from nemo_rl.telemetry.instrumentation import accepts_trace_context
from ultra_qualification.publication import PublicationGuard


class UltraAuditWorkerImpl(MegatronPolicyWorkerImpl):
    """Observe native storage, publication and cumulative allocation high-waters."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._audit_dir = Path(os.environ["NRL_ULTRA_AUDIT_DIR"])
        self._audit_dir.mkdir(parents=True, exist_ok=True)
        self._audit_guard = PublicationGuard()
        self._audit_optimizer_calls = 0
        self._audit_microbatch_calls = 0
        self._audit_sequence = 0
        self._audit_peak_allocated = 0
        self._audit_peak_reserved = 0
        self._audit_lock = threading.RLock()
        self._audit_clock_domain = (
            Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            os.readlink("/proc/self/ns/time"),
        )
        super().__init__(*args, **kwargs)
        if logging.getLogger(
            "nemo_rl.models.policy.workers.megatron_policy_worker"
        ).isEnabledFor(logging.DEBUG):
            raise ValueError(
                "Qualification cumulative peaks require native DEBUG resets off"
            )
        self._audit_concurrent = sm_partition.engine_sm_fraction(self.cfg) is not None
        if not self.is_refit_destination:
            config = self.model.ddp_config
            if (
                not config.overlap_param_gather
                or config.grad_reduce_in_fp32
                or config.overlap_grad_reduce
                or not config.use_distributed_optimizer
            ):
                raise ValueError(
                    "Ultra observer requires the reviewed BF16 DDP profile"
                )
            self._install_gather_observers()
        self._snapshot("constructed", catalog=True)

    def _buckets(self) -> list[Any]:
        return self.model.bucket_groups + self.model.expert_parallel_bucket_groups

    def _bucket_states(self) -> tuple[tuple[bool, bool], ...]:
        return tuple(
            (bucket.param_gather_dispatched, bucket.param_gather_handle is not None)
            for bucket in self._buckets()
        )

    def _parameters_outside_ddp_storage(self) -> list[str]:
        mismatches = []
        for group_index, group in enumerate(self._buckets()):
            for bucket_index, bucket in enumerate(group.buckets):
                if bucket.param_data is None:
                    raise RuntimeError("Qualification requires DDP-owned BF16 storage")
                for parameter, (start, _) in bucket.param_to_index.items():
                    expected = (
                        bucket.param_data.data_ptr()
                        + start * bucket.param_data.element_size()
                    )
                    if parameter.data_ptr() != expected:
                        mismatches.append(f"{group_index}:{bucket_index}:{start}")
        return mismatches

    def _install_gather_observers(self) -> None:
        for index, bucket in enumerate(self._buckets()):
            original = bucket.start_param_sync

            def observed(
                force_sync: bool = False,
                *,
                _bucket: Any = bucket,
                _original: Any = original,
                _index: int = index,
            ) -> Any:
                try:
                    self._audit_guard.before_gather(
                        force_sync=force_sync,
                        has_handle=_bucket.param_gather_handle is not None,
                    )
                except RuntimeError:
                    self._snapshot(
                        "rejected_postpublication_gather", bucket_index=_index
                    )
                    raise
                return _original(force_sync=force_sync)

            bucket.start_param_sync = observed

    def _snapshot(
        self,
        event: str,
        *,
        catalog: bool = False,
        generation_model: bool = False,
        **extra: Any,
    ) -> None:
        with self._audit_lock:
            self._snapshot_locked(
                event, catalog=catalog, generation_model=generation_model, **extra
            )

    def _snapshot_locked(
        self,
        event: str,
        *,
        catalog: bool,
        generation_model: bool,
        **extra: Any,
    ) -> None:
        is_engine_model = self.is_refit_destination or generation_model
        core = sm_partition.refit_core(
            self._gen_model() if is_engine_model else self.model
        )
        names = sm_partition.refit_names(core)
        tensors = list(named_refit_tensors(core))
        resolved = [names[name] for name, _ in tensors]
        if len(set(resolved)) != len(resolved):
            raise RuntimeError("Duplicate resolved refit tensor names")
        source = getattr(self, "_coresident_src", None)
        records = []
        for name, tensor in tensors:
            alias = None
            if source is not None:
                peer = source[name]
                alias = (
                    tensor.data_ptr() == peer.data_ptr()
                    and tensor.shape == peer.shape
                    and tensor.stride() == peer.stride()
                    and tensor.dtype == peer.dtype
                )
                if ".experts.linear_fc" in name and not alias:
                    raise RuntimeError(f"Expert storage is private: {name}")
            records.append(
                {
                    "name": name,
                    "resolved_name": names[name],
                    "shape": list(tensor.shape),
                    "stride": list(tensor.stride()),
                    "dtype": str(tensor.dtype),
                    "data_ptr": tensor.data_ptr(),
                    "storage_ptr": tensor.untyped_storage().data_ptr(),
                    "storage_offset": tensor.storage_offset(),
                    "bytes": tensor.numel() * tensor.element_size(),
                    "parameter": isinstance(tensor, torch.nn.Parameter),
                    "requires_grad": tensor.requires_grad,
                    "aliases_mapped_source": alias,
                }
            )
        role = "engine" if is_engine_model else "learner"
        experts = [row for row in records if ".experts.linear_fc" in row["name"]]
        if len(experts) != 784 or sum(row["bytes"] for row in experts) != 16441671680:
            raise RuntimeError("Unexpected Ultra expert parameter catalog")
        if any(
            not row["parameter"] or row["dtype"] != "torch.bfloat16" for row in experts
        ):
            raise RuntimeError("Qualification requires ordinary BF16 expert Parameters")
        stacks = []
        for name, module in core.named_modules():
            if hasattr(module, "_concatenated_weights_built"):
                stacks.append(
                    {"module": name, "built": module._concatenated_weights_built}
                )
        device = next(core.parameters()).device
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        self._audit_peak_allocated = max(
            self._audit_peak_allocated, torch.cuda.max_memory_allocated(device)
        )
        self._audit_peak_reserved = max(
            self._audit_peak_reserved, torch.cuda.max_memory_reserved(device)
        )
        self._audit_sequence += 1
        output = {
            "event": event,
            "sequence": self._audit_sequence,
            "hostname": socket.gethostname(),
            "monotonic_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            "clock_domain": self._audit_clock_domain,
            "rank": torch.distributed.get_rank(),
            "role": role,
            "process_role": "engine" if self.is_refit_destination else "learner",
            "pid": os.getpid(),
            "device_uuid": self.report_device_id(),
            "rank_coordinates": sm_partition.rank_coordinates(),
            "ep_coordinate": [
                core.pg_collection.ep.rank(),
                core.pg_collection.ep.size(),
            ],
            "cp_coordinate": [
                core.pg_collection.cp.rank(),
                core.pg_collection.cp.size(),
            ],
            "concurrent": self._audit_concurrent,
            "weight_epoch": getattr(
                getattr(self, "dynamic_inference_engine", None), "_weight_epoch", None
            ),
            "optimizer_calls": self._audit_optimizer_calls,
            "publication": asdict(self._audit_guard),
            "bucket_states": None if role == "engine" else self._bucket_states(),
            "catalog_count": len(records),
            "expert_matrices": len(experts),
            "expert_stacks": stacks,
            "catalog_storage_digest": hashlib.sha256(
                json.dumps(records, sort_keys=True).encode()
            ).hexdigest(),
            "expert_storage_digest": hashlib.sha256(
                json.dumps(
                    [
                        {
                            key: value
                            for key, value in row.items()
                            if key != "requires_grad"
                        }
                        for row in experts
                    ],
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "catalog": records if catalog else None,
            "memory": {
                "allocated": torch.cuda.memory_allocated(device),
                "reserved": torch.cuda.memory_reserved(device),
                "cumulative_peak_allocated": self._audit_peak_allocated,
                "cumulative_peak_reserved": self._audit_peak_reserved,
                "device_free": free_bytes,
                "device_total": total_bytes,
            },
            **extra,
        }
        path = self._audit_dir / (
            f"{socket.gethostname()}-{os.getpid()}-{self._audit_sequence:04d}-{event}.json"
        )
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"NRL_ULTRA_AUDIT {path}", flush=True)

    def open_coresident_refit_handles(self, *args: Any, **kwargs: Any) -> Any:
        result = super().open_coresident_refit_handles(*args, **kwargs)
        self._snapshot("bound", catalog=True, binding=result)
        return result

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> Any:
        result = super().prepare_for_generation(*args, **kwargs)
        self._snapshot("generation_prepared", catalog=True, generation_model=True)
        return result

    def begin_train_step(self, *args: Any, **kwargs: Any) -> Any:
        self._snapshot("train_begin_before")
        self._audit_microbatch_calls = 0
        result = super().begin_train_step(*args, **kwargs)
        self._snapshot("train_begin_after")
        return result

    @accepts_trace_context
    def train_microbatch_presharded(self, *args: Any, **kwargs: Any) -> Any:
        result = super().train_microbatch_presharded(*args, **kwargs)
        self._audit_microbatch_calls += 1
        if self._audit_microbatch_calls == 1:
            self._snapshot("first_microbatch_returned")
        return result

    def reset_peak_memory_stats(self) -> None:
        self._snapshot("before_native_peak_reset")
        super().reset_peak_memory_stats()

    def finish_train_step(self, *args: Any, **kwargs: Any) -> Any:
        self._audit_guard.before_optimizer()
        self._snapshot("optimizer_before")
        result = super().finish_train_step(*args, **kwargs)
        self._audit_optimizer_calls += 1
        self._snapshot("optimizer_returned")
        return result

    def sync_params_before_refit(self) -> None:
        super().sync_params_before_refit()
        states = self._bucket_states()
        if self._audit_concurrent:
            outside = self._parameters_outside_ddp_storage()
            if outside:
                raise RuntimeError(
                    f"Published parameters escaped DDP storage: {outside}"
                )
            if self._audit_optimizer_calls:
                self._audit_guard.publish(bucket_states=states)
            else:
                self._audit_guard.publish_initial(bucket_states=states)
        self._snapshot("parameters_published")

    def _coresident_copy_weights(self) -> Any:
        result = super()._coresident_copy_weights()
        core = sm_partition.refit_core(self._gen_model())
        biases = []
        for name, tensor in named_refit_tensors(core):
            if name.endswith("router.expert_bias"):
                source = self._coresident_src[name]
                private = tensor.data_ptr() != source.data_ptr()
                equal = torch.equal(tensor, source)
                biases.append({"name": name, "private": private, "equal": equal})
                if not private or not equal:
                    raise RuntimeError(
                        f"Private router bias publication failed: {name}"
                    )
        if len(biases) != 49:
            raise RuntimeError("Expected 49 private Ultra router bias buffers")
        self._snapshot("private_buffers_published", router_biases=biases)
        return result

    async def _coresident_pause(self) -> None:
        await super()._coresident_pause()
        self._snapshot(
            "reader_drained", weight_epoch=self.dynamic_inference_engine._weight_epoch
        )

    async def _coresident_unpause(self) -> None:
        self._snapshot(
            "reader_before_unpause",
            weight_epoch=self.dynamic_inference_engine._weight_epoch,
        )
        await super()._coresident_unpause()
        self._snapshot(
            "reader_unpaused", weight_epoch=self.dynamic_inference_engine._weight_epoch
        )


UltraAuditWorker = ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"),
    concurrency_groups=WORKER_CONCURRENCY_GROUPS,
)(UltraAuditWorkerImpl)
