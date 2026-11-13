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
"""Run native Ultra Math with verified placement and policy-only private MPS."""

import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from typing import Any

import ray
from ray.util.placement_group import placement_group_table
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
from public_placement import parse_gpu_query
from topology import validate_nodes, validate_placement, validate_rank_islands


def native_probe(pipe: str) -> dict[str, Any]:
    listing = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout
    gpu_queries = {}
    for uuid in listing.splitlines():
        query = subprocess.run(
            ["nvidia-smi", "-i", uuid.strip(), "-q"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        gpu_queries[uuid.strip()] = query.stdout
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "hostname": socket.gethostname().split(".")[0],
        "native_queries": gpu_queries,
        "private_mps_pipe_exists": bool(pipe) and Path(pipe).exists(),
        "inherited_mps_keys": sorted(
            k for k in os.environ if k.startswith("CUDA_MPS_")
        ),
    }


def start_selected_policy_mps(script: str, pipe: str) -> dict[str, Any]:
    """Start a private daemon only on a node selected by the ready policy PG."""
    log = (
        Path(os.environ["ULTRA_DIRECT64_RUN_DIR"])
        / "nodes"
        / socket.gethostname().split(".")[0]
        / "mps-start-command.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        subprocess.run(
            ["bash", script],
            env={**os.environ, "CUDA_MPS_PIPE_DIRECTORY": pipe},
            check=True,
            timeout=60,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    return {
        "node_id": ray.get_runtime_context().get_node_id(),
        "hostname": socket.gethostname().split(".")[0],
        "private_mps_pipe_exists": Path(pipe).exists(),
    }


def install() -> None:
    from nemo_rl.algorithms.single_controller_utils import setup
    from nemo_rl.distributed import virtual_cluster
    from nemo_rl.models.megatron import sm_partition

    original_init = virtual_cluster.init_ray
    original_build = setup._build_clusters
    output = Path(os.environ["ULTRA_DIRECT64_TOPOLOGY_OUTPUT"])
    pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")

    def clean_job_init(*args: Any, **kwargs: Any) -> Any:
        saved = {k: v for k, v in os.environ.items() if k.startswith("CUDA_MPS_")}
        for key in saved:
            os.environ.pop(key)
        try:
            return original_init(*args, **kwargs)
        finally:
            os.environ.update(saved)

    def checked_build(config: Any) -> tuple[Any, Any, Any]:
        evidence = {
            "status": "STARTED",
            "scope": "128GPU native topology only, no judge/sandbox/HTTP",
        }
        try:
            if (
                config.cluster.num_nodes,
                config.cluster.gpus_per_node,
                config.cluster.segment_size,
            ) != (32, 4, None):
                raise ValueError("Require 32 homogeneous 4-GPU nodes")
            generation = config.policy["generation"]
            if (
                generation["backend"] != "megatron"
                or generation["mcore_generation_config"]["expose_http_server"]
            ):
                raise ValueError(
                    "Only the native direct-generation Math path is supported"
                )
            if config.env["should_use_nemo_gym"] or "nemo_gym" in config.env:
                raise ValueError("No Gym/judge resources may be introduced")
            concurrent = sm_partition.engine_sm_fraction(config.policy) is not None
            if concurrent != (os.environ["ULTRA_DIRECT64_ARM"] == "concurrent"):
                raise ValueError("Runtime mode differs from chosen native arm")
            if concurrent and not pipe:
                raise ValueError("Concurrent arm requires private MPS directory")
            live = [node for node in ray.nodes() if node["Alive"]]
            remote = ray.remote(num_cpus=0.01)(native_probe)
            probes = ray.get(
                [
                    remote.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node["NodeID"], soft=False
                        ),
                        runtime_env={"py_executable": sys.executable},
                    ).remote(pipe)
                    for node in live
                ],
                timeout=180,
            )
            by_id = {node["NodeID"]: node for node in live}
            for probe in probes:
                probe["resources"] = by_id[probe["node_id"]]["Resources"]
                probe["gpus"] = [
                    parse_gpu_query(gpu_uuid=uuid, query=query)
                    for uuid, query in probe["native_queries"].items()
                ]
            evidence["nodes"] = probes
            evidence["physical"] = validate_nodes(probes)
            topology = virtual_cluster.get_ray_cluster_topology()
            constraints, _, _ = virtual_cluster.prepare_segment_topology(
                16, 32, topology=topology, role="native_math_policy"
            )
            if constraints is None or len(constraints) != 32:
                raise ValueError("Native segment selection did not return 32 nodes")
            constructor = setup.RayVirtualCluster

            def constrained(*args: Any, **kwargs: Any) -> Any:
                if (
                    kwargs.get("name")
                    not in ("sc_coresident_cluster", "sc_policy_cluster")
                    or kwargs.get("node_resource_constraints") is not None
                    or kwargs.get("segment_size") is not None
                ):
                    raise ValueError("Unexpected native cluster construction")
                kwargs["node_resource_constraints"] = constraints
                return constructor(*args, **kwargs)

            setup.RayVirtualCluster = constrained
            try:
                train, engine, teacher = original_build(config)
            finally:
                setup.RayVirtualCluster = constructor
            if train is not engine or teacher is not None:
                raise ValueError("Require one shared native 128-GPU cluster")
            tables = []
            bundles = []
            for pg in train.get_placement_groups():
                table = placement_group_table(pg)
                tables.append(table)
                if table["state"] != "CREATED":
                    raise ValueError("Native policy PG not ready")
                for idx, bundle in enumerate(pg.bundle_specs):
                    if bundle.get("GPU") != 1:
                        raise ValueError("One GPU per native actor bundle required")
                    bundles.append(table["bundles_to_node_id"][idx])
            if len(tables) != 1 or train._sorted_bundle_indices is None:
                raise ValueError(
                    "Require native unified PG and explicit sorted rank order"
                )
            order = list(train._sorted_bundle_indices)
            if sorted(order) != list(range(128)):
                raise ValueError("Native rank sort is not a 128-rank permutation")
            evidence["rank_islands"] = validate_rank_islands(
                ordered_node_ids=[bundles[index] for index in order], probes=probes
            )
            ids = list(by_id)
            validate_placement(selected_node_ids=ids, actual_bundles=bundles)
            if ray.available_resources().get("GPU", 0) != 0:
                raise ValueError("No unused/judge GPU allocation is allowed")
            evidence["placement_groups"] = tables
            for probe in probes:
                node_dir = output.parent / "nodes" / probe["hostname"]
                node_dir.mkdir(parents=True, exist_ok=True)
                (node_dir / "role.txt").write_text("policy\n")
                if not concurrent:
                    (node_dir / "mps-status.txt").write_text(
                        "NO_MPS_DAEMON_FOR_ALTERNATING_ARM\n"
                    )
            if concurrent:
                starter = ray.remote(num_cpus=0.01)(start_selected_policy_mps)
                starts = ray.get(
                    [
                        starter.options(
                            scheduling_strategy=NodeAffinitySchedulingStrategy(
                                nid, soft=False
                            ),
                            runtime_env={"py_executable": sys.executable},
                        ).remote(
                            str(Path(__file__).parent / "start_policy_mps.sh"), pipe
                        )
                        for nid in ids
                    ],
                    timeout=180,
                )
                if len(starts) != 32 or not all(
                    item["private_mps_pipe_exists"] for item in starts
                ):
                    raise ValueError("Private MPS startup incomplete")
                evidence["MPS_start"] = starts
            evidence["status"] = "PRE_MODEL_NATIVE_TOPOLOGY_PASS"
            print(
                "ULTRA_NATIVE_MATH_TOPOLOGY " + json.dumps(evidence["physical"]),
                flush=True,
            )
            return train, engine, teacher
        finally:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(evidence, indent=2, default=str) + "\n")

    virtual_cluster.init_ray = clean_job_init
    setup._build_clusters = checked_build


def main() -> None:
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )

    base = "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
    extension = "math_worker.MathAuditWorker"
    explicit_python = os.environ.get("ULTRA_POLICY_PYTHON")
    if explicit_python:
        ACTOR_ENVIRONMENT_REGISTRY[base] = explicit_python
    ACTOR_ENVIRONMENT_REGISTRY[extension] = ACTOR_ENVIRONMENT_REGISTRY[base]
    if any("worker_extension_cls_fqn=" in argument for argument in sys.argv):
        raise ValueError(
            "The portable qualification entry installs its own worker extension"
        )
    sys.argv.append("++policy.worker_extension_cls_fqn=" + extension)
    install()
    repo = Path(os.environ["ULTRA_REPO_ROOT"]).resolve()
    runpy.run_path(
        str(repo / "examples/run_grpo_single_controller.py"), run_name="__main__"
    )


if __name__ == "__main__":
    main()
