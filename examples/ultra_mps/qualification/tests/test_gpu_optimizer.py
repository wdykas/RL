# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Native GPU optimizer ownership and precision-state rejection."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpu_optimizer_inventory import inspect_gpu_optimizer


class Tensor:
    def __init__(self, dtype="torch.bfloat16", device="cuda"):
        self.dtype = dtype
        self.device = SimpleNamespace(type=device)

    def numel(self):
        return 12

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: 24)


class Fused:
    use_decoupled_grad = master_weights = store_param_remainders = True
    master_weight_dtype = "torch.float32"

    def __init__(self):
        self.param_groups = [{"params": [Tensor()]}]
        self.state = {}


class Distributed:
    def __init__(self, n):
        self.optimizer = Fused()
        self.data_parallel_group = SimpleNamespace(size=lambda: n)
        self.buffers = [SimpleNamespace(data_parallel_group=self.data_parallel_group)]


class Chained:
    def __init__(self):
        self.chained_optimizers = [Distributed(16), Distributed(2)]


def inspect(o):
    return inspect_gpu_optimizer(
        o, chained_type=Chained, distributed_type=Distributed, fused_type=Fused
    )


class GPUOptimizerTests(unittest.TestCase):
    def test_lazy_state_allowed_but_cpu_or_wrong_dtype_rejected(self):
        optimizer = Chained()
        self.assertEqual(
            inspect(optimizer)["children"][0]["initialized_state_entries"], 0
        )
        inner = optimizer.chained_optimizers[0].optimizer
        inner.state[object()] = {
            "exp_avg": Tensor("torch.float32"),
            "exp_avg_sq": Tensor("torch.float32"),
            "master_param": Tensor("torch.int16"),
        }
        self.assertEqual(
            inspect(optimizer)["children"][0]["state_metadata"]["master_param"][
                "elements"
            ],
            12,
        )
        for bad in (Tensor("torch.float32", "cpu"), Tensor("torch.float16")):
            inner.state[next(iter(inner.state))]["exp_avg"] = bad
            with self.assertRaises(ValueError):
                inspect(optimizer)

    def test_group_parameter_and_precision_flags_are_actual(self):
        for field in ("group", "parameter", "flag"):
            optimizer = Chained()
            child = optimizer.chained_optimizers[0]
            if field == "group":
                child.data_parallel_group = SimpleNamespace(size=lambda: 8)
            if field == "parameter":
                child.optimizer.param_groups[0]["params"] = [Tensor(device="cpu")]
            if field == "flag":
                child.optimizer.master_weights = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                inspect(optimizer)


if __name__ == "__main__":
    unittest.main()
