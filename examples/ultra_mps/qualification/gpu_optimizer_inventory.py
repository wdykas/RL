# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Read-only native GPU optimizer metadata; lazy Adam state is not initialized here."""


def inspect_gpu_optimizer(optimizer, *, chained_type, distributed_type, fused_type):
    children = (
        optimizer.chained_optimizers
        if isinstance(optimizer, chained_type)
        else [optimizer]
    )
    rows = []
    for child in children:
        if not isinstance(child, distributed_type) or not isinstance(
            child.optimizer, fused_type
        ):
            raise TypeError("Require native DistributedOptimizer and TE FusedAdam")
        inner = child.optimizer
        flags = {
            "use_decoupled_grad": inner.use_decoupled_grad,
            "master_weights": inner.master_weights,
            "master_weight_dtype": str(inner.master_weight_dtype),
            "store_param_remainders": inner.store_param_remainders,
        }
        if flags != {
            "use_decoupled_grad": True,
            "master_weights": True,
            "master_weight_dtype": "torch.float32",
            "store_param_remainders": True,
        }:
            raise ValueError("Require actual native precision-aware TE optimizer flags")
        state_metadata = {}
        for state in inner.state.values():
            for name, value in state.items():
                if not hasattr(value, "device"):
                    continue
                if value.device.type != "cuda":
                    raise ValueError("Optimizer tensor state escaped GPU residency")
                expected_dtype = {
                    "exp_avg": "torch.float32",
                    "exp_avg_sq": "torch.float32",
                    "master_param": "torch.int16",
                }.get(name)
                if expected_dtype is not None and str(value.dtype) != expected_dtype:
                    raise ValueError(
                        "Native precision-aware optimizer state dtype differs"
                    )
                row = state_metadata.setdefault(
                    name,
                    {
                        "tensor_count": 0,
                        "elements": 0,
                        "storage_nbytes_observed_sum": 0,
                        "dtype": str(value.dtype),
                        "device": value.device.type,
                    },
                )
                row["tensor_count"] += 1
                row["elements"] += value.numel()
                row["storage_nbytes_observed_sum"] += value.untyped_storage().nbytes()
        params = [p for group in inner.param_groups for p in group["params"]]
        if not params or any(
            p.device.type != "cuda" or str(p.dtype) != "torch.bfloat16" for p in params
        ):
            raise ValueError("Require actual native CUDA BF16 owned shards")
        group_size = child.data_parallel_group.size()
        buffer_sizes = sorted(
            {buffer.data_parallel_group.size() for buffer in child.buffers}
        )
        if group_size not in (2, 16) or buffer_sizes != [group_size]:
            raise ValueError("Require denseDPxCP16 or expertDP2 optimizer group")
        rows.append(
            {
                "wrapper": type(child).__name__,
                "inner": type(inner).__name__,
                "owned_elements": sum(p.numel() for p in params),
                "owned_parameter_count": len(params),
                "parameter_dtype": "torch.bfloat16",
                "parameter_device": "cuda",
                "optimizer_group_size": group_size,
                "buffer_group_sizes": buffer_sizes,
                "initialized_state_entries": len(inner.state),
                "precision_flags": flags,
                "state_metadata": state_metadata,
            }
        )
    if not rows or {row["optimizer_group_size"] for row in rows} != {2, 16}:
        raise ValueError(
            "Require actual dense and expert native GPU optimizer children"
        )
    return {
        "status": "NATIVE_DISTRIBUTED_GPU_OPTIMIZER_PASS",
        "children": rows,
        "scope": "Metadata only. CUDA BF16 owned parameter shards, not FP32 masters. Lazy TE state may be empty; no allocation/state_dict or tensor reads. Actual updates/publication and state residency require native evidence.",
    }
