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
"""Portable CPU topology/dispatch tests; no Ray, CUDA or Slurm launch."""

import ast
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

PACK = Path(__file__).resolve().parents[1]


def load_local(name):
    spec = importlib.util.spec_from_file_location(
        "portable_test_" + name, PACK / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


topology = load_local("topology")
validate_nodes = topology.validate_nodes
validate_placement = topology.validate_placement
validate_rank_islands = topology.validate_rank_islands
parse_gpu_query = load_local("public_placement").parse_gpu_query


class PortTests(unittest.TestCase):
    def nodes(self):
        rows = []
        for i in range(32):
            domain = "12345678-1234-1234-1234-" + f"{i // 16 + 1:012d}"
            rows.append(
                {
                    "node_id": str(i),
                    "resources": {"GPU": 4, "nvlink_domain_" + domain: 1},
                    "gpus": [
                        {"gpu_uuid": f"GPU-{i}-{j}", "cluster_uuid": domain}
                        for j in range(4)
                    ],
                    "inherited_mps_keys": [],
                    "private_mps_pipe_exists": False,
                }
            )
        return rows

    def test_two_complete_native_islands(self):
        nodes = self.nodes()
        order = [str(i) for i in range(32) for _ in range(4)]
        self.assertEqual(validate_nodes(nodes)["reserved_gpus"], 128)
        self.assertEqual(
            len(validate_rank_islands(ordered_node_ids=order, probes=nodes)), 2
        )
        validate_placement(
            selected_node_ids=[str(i) for i in range(32)], actual_bundles=order
        )

    def test_cross_fabric_rank_order_rejected(self):
        nodes = self.nodes()
        order = [str(i) for i in range(32) for _ in range(4)]
        order[0], order[64] = order[64], order[0]
        with self.assertRaises(ValueError):
            validate_rank_islands(ordered_node_ids=order, probes=nodes)

    def test_physical_leak_duplicate_and_idle_capacity_rejected(self):
        for mutation in ("duplicate", "pipe", "env"):
            nodes = self.nodes()
            if mutation == "duplicate":
                nodes[1]["gpus"][0]["gpu_uuid"] = nodes[0]["gpus"][0]["gpu_uuid"]
            if mutation == "pipe":
                nodes[0]["private_mps_pipe_exists"] = True
            if mutation == "env":
                nodes[0]["inherited_mps_keys"] = ["CUDA_MPS_PIPE_DIRECTORY"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                validate_nodes(nodes)
        with self.assertRaises(ValueError):
            validate_placement(
                selected_node_ids=[str(i) for i in range(32)],
                actual_bundles=[str(i) for i in range(32) for _ in range(4)][:-1],
            )

    def test_actual_fabric_output_requires_registered_gpu(self):
        raw = "    GPU UUID : GPU-test\n    Fabric\n        State : Completed\n        Status : Success\n        ClusterUUID : 12345678-1234-1234-1234-123456789abc\n"
        self.assertEqual(
            parse_gpu_query(gpu_uuid="GPU-test", query=raw)["gpu_uuid"], "GPU-test"
        )
        with self.assertRaises(ValueError):
            parse_gpu_query(
                gpu_uuid="GPU-test", query=raw.replace("Success", "Failure")
            )

    def main_function(self, argv):
        tree = ast.parse((PACK / "entry.py").read_text())
        fn = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        registry = {
            "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": "native-default"
        }
        module = ModuleType("nemo_rl.distributed.ray_actor_environment_registry")
        module.ACTOR_ENVIRONMENT_REGISTRY = registry
        calls = []
        namespace = {
            "Path": Path,
            "os": os,
            "sys": SimpleNamespace(argv=argv),
            "install": lambda: calls.append("install"),
            "runpy": SimpleNamespace(
                run_path=lambda path, **kwargs: calls.append((path, kwargs))
            ),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])),
                "entry-main",
                "exec",
            ),
            namespace,
        )
        return namespace["main"], registry, calls, module

    def test_native_registry_override_and_repo_dispatch(self):
        argv = ["entry.py", "--config", "config.json"]
        main, registry, calls, module = self.main_function(argv)
        with (
            patch.dict(
                sys.modules,
                {"nemo_rl.distributed.ray_actor_environment_registry": module},
            ),
            patch.dict(
                os.environ,
                {
                    "ULTRA_POLICY_PYTHON": "/portable/policy/python",
                    "ULTRA_REPO_ROOT": "/portable/repo",
                },
            ),
        ):
            main()
        self.assertEqual(
            registry["math_worker.MathAuditWorker"], "/portable/policy/python"
        )
        self.assertEqual(
            calls,
            [
                "install",
                (
                    "/portable/repo/examples/run_grpo_single_controller.py",
                    {"run_name": "__main__"},
                ),
            ],
        )
        self.assertEqual(
            argv[-1], "++policy.worker_extension_cls_fqn=math_worker.MathAuditWorker"
        )

    def test_conflicting_worker_cli_rejected_before_native_dispatch(self):
        main, registry, calls, module = self.main_function(
            ["entry.py", "++policy.worker_extension_cls_fqn=other.Worker"]
        )
        with (
            patch.dict(
                sys.modules,
                {"nemo_rl.distributed.ray_actor_environment_registry": module},
            ),
            self.assertRaises(ValueError),
        ):
            main()
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
