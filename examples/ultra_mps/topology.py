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
"""Native 128-GPU placement and contiguous EP64 fabric-island checks."""

from collections import Counter
from uuid import UUID


def validate_nodes(probes: list[dict]) -> dict:
    if len(probes) != 32:
        raise ValueError("Require 32 actual Ray nodes, no idle judge nodes")
    ids = set()
    uuids = []
    domains = set()
    for node in probes:
        if (
            node["node_id"] in ids
            or node["resources"].get("GPU") != 4
            or len(node["gpus"]) != 4
        ):
            raise ValueError("Duplicate node or wrong physical GPU geometry")
        ids.add(node["node_id"])
        if node["inherited_mps_keys"] or node["private_mps_pipe_exists"]:
            raise ValueError("MPS leaked before native PG selection")
        for gpu in node["gpus"]:
            if UUID(gpu["cluster_uuid"]).int == 0:
                raise ValueError("Missing native fabric identity")
            if node["resources"].get("nvlink_domain_" + gpu["cluster_uuid"], 0) != 1:
                raise ValueError("Native/Ray fabric domains disagree")
            domains.add(gpu["cluster_uuid"])
            uuids.append(gpu["gpu_uuid"])
    if len(set(uuids)) != 128:
        raise ValueError("Require 128 unique GPUs in two validated EP64 rank islands")
    return {
        "physical_gpu_uuids": sorted(uuids),
        "native_fabric_uuids": sorted(domains),
        "reserved_gpus": 128,
    }


def validate_placement(
    *, selected_node_ids: list[str], actual_bundles: list[str]
) -> None:
    if len(selected_node_ids) != 32 or len(set(selected_node_ids)) != 32:
        raise ValueError("Invalid selected policy node set")
    counts = Counter(actual_bundles)
    if set(counts) != set(selected_node_ids) or set(counts.values()) != {4}:
        raise ValueError("Native 128-GPU PG placement differs from physical allocation")


def validate_rank_islands(*, ordered_node_ids, probes):
    if len(ordered_node_ids) != 128:
        raise ValueError("Require 128 native sorted actor ranks")
    domains = {
        row["node_id"]: {gpu["cluster_uuid"] for gpu in row["gpus"]} for row in probes
    }
    islands = []
    for start in (0, 64):
        selected = ordered_node_ids[start : start + 64]
        if len(set(selected)) != 16 or set(Counter(selected).values()) != {4}:
            raise ValueError("Each EP64 island must occupy 16 full 4-GPU nodes")
        fabric = set().union(*(domains[node] for node in selected))
        if len(fabric) != 1:
            raise ValueError("Native sorted EP64 ranks cross NVLink domains")
        islands.append(
            {
                "first_rank": start,
                "last_rank": start + 63,
                "fabric_uuid": next(iter(fabric)),
                "node_ids": sorted(set(selected)),
            }
        )
    return islands
