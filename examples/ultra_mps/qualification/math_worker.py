# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Native Math qualification worker; original native computation unchanged."""

import json
import os
from pathlib import Path
import socket
import threading
import time

import ray
import torch
from ultra_qualification.native_worker import UltraAuditWorkerImpl
from nemo_rl.models.generation.megatron.megatron_worker import WORKER_CONCURRENCY_GROUPS
from nemo_rl.models.policy.utils import get_runtime_env_for_policy_worker
from mtp_fix_guard import verify_imports
from raw_logprob_guard import require_raw_logprob_lengths
from gpu_optimizer_inventory import inspect_gpu_optimizer
from host_telemetry import capture_host_memory


class MathAuditWorkerImpl(UltraAuditWorkerImpl):
    def __init__(self, *args, **kwargs):
        self._raw_payload_lock = threading.Lock()
        self._raw_payload_ordinal = 0
        root = Path(os.environ["ULTRA_DIRECT64_RUN_DIR"])
        identity = f"{socket.gethostname()}-{os.getpid()}"
        proof = verify_imports(
            Path(os.environ["NRL_ULTRA_MTP_FIX_SOURCE"]),
            Path(os.environ["NRL_ULTRA_MCORE_SOURCE"]),
        )
        proof.update({"hostname": socket.gethostname(), "pid": os.getpid()})
        destination = root / "mtp-import-guards"
        destination.mkdir(parents=True, exist_ok=True)
        (destination / f"{identity}.json").write_text(
            json.dumps(proof, indent=2) + "\n"
        )
        self._raw_payload_path = root / "raw-generation-payloads" / f"{identity}.jsonl"
        self._raw_payload_path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(*args, **kwargs)
        if not self.is_refit_destination:
            from megatron.core.optimizer.optimizer import ChainedOptimizer
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
            from transformer_engine.pytorch.optimizers import FusedAdam

            cfg = self.megatron_cfg.optimizer
            if (
                cfg.optimizer_cpu_offload is not False
                or cfg.optimizer_offload_fraction != 0.0
                or cfg.use_precision_aware_optimizer is not True
                or self.cfg["offload_optimizer_for_refit"]
                or self.cfg["offload_optimizer_for_logprob"]
            ):
                raise ValueError(
                    "Require fully GPU-resident native optimizer configuration"
                )
            inventory = inspect_gpu_optimizer(
                self.optimizer,
                chained_type=ChainedOptimizer,
                distributed_type=DistributedOptimizer,
                fused_type=FusedAdam,
            )
            inventory.update(
                {
                    "hostname": socket.gethostname(),
                    "pid": os.getpid(),
                    "rank": torch.distributed.get_rank(),
                    "use_precision_aware_optimizer": cfg.use_precision_aware_optimizer,
                }
            )
            destination = root / "gpu-optimizer-inventory"
            destination.mkdir(parents=True, exist_ok=True)
            (destination / f"{identity}.json").write_text(
                json.dumps(inventory, indent=2) + "\n"
            )

    def _snapshot_locked(self, *args, **kwargs):
        event = args[0] if args else kwargs.get("event")
        if (
            event in ("optimizer_returned", "generation_prepared")
            and not self.is_refit_destination
        ):
            from megatron.core.optimizer.optimizer import ChainedOptimizer
            from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
            from transformer_engine.pytorch.optimizers import FusedAdam

            inventory = inspect_gpu_optimizer(
                self.optimizer,
                chained_type=ChainedOptimizer,
                distributed_type=DistributedOptimizer,
                fused_type=FusedAdam,
            )
            if event == "optimizer_returned" or self._audit_optimizer_calls > 0:
                if any(
                    not all(
                        child["state_metadata"].get(key)
                        for key in ("exp_avg", "exp_avg_sq", "master_param")
                    )
                    for child in inventory["children"]
                ):
                    raise ValueError("Post-update GPU moments/remainder state missing")
            kwargs["gpu_optimizer_inventory"] = inventory
        stats = getattr(torch.cuda.memory, "host_memory_stats", None)
        kwargs["host_memory"] = capture_host_memory(
            host_stats_fn=stats, native_stats_requested=True
        )
        return super()._snapshot_locked(*args, **kwargs)

    def _parse_result_to_batched_data_dict(self, data, result):
        # Actual native direct _build_sampling_params always requests logprobs.
        # Guard BEFORE original pack; never replace/slice/pad original payloads.
        try:
            evidence = require_raw_logprob_lengths(result, requested=True)
        except ValueError as error:
            self._write_raw_record(
                {"status": "RAW_PAYLOAD_MISMATCH", "error": str(error)}
            )
            raise
        self._write_raw_record(
            {"status": "RAW_PAYLOAD_LENGTHS_PASS", "requests": evidence}
        )
        return super()._parse_result_to_batched_data_dict(data, result)

    def _write_raw_record(self, record):
        with self._raw_payload_lock:
            self._raw_payload_ordinal += 1
            record.update(
                {
                    "call_ordinal": self._raw_payload_ordinal,
                    "rank": torch.distributed.get_rank(),
                    "wall_ns": time.time_ns(),
                    "monotonic_ns": time.monotonic_ns(),
                }
            )
            with self._raw_payload_path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")


MathAuditWorker = ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"),
    concurrency_groups=WORKER_CONCURRENCY_GROUPS,
)(MathAuditWorkerImpl)
