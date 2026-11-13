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

"""Offline all-rank checks for unscored native Ultra observer records."""

import argparse
import json
from pathlib import Path
import re
from typing import Any


def check_catalog(record: dict[str, Any], *, shared: bool) -> None:
    """Check native catalog structure and local global-expert ownership."""
    catalog = record["catalog"]
    if not catalog or len({row["resolved_name"] for row in catalog}) != len(catalog):
        raise ValueError("Missing catalog or duplicate resolved names")
    experts = [row for row in catalog if ".experts.linear_fc" in row["name"]]
    if len(experts) != 784 or sum(row["bytes"] for row in experts) != 16441671680:
        raise ValueError("Expected 784 local expert matrices")
    ep_rank, ep_size = record["ep_coordinate"]
    if ep_size != 64 or not 0 <= ep_rank < 64:
        raise ValueError("Unexpected expert coordinate")
    projections: dict[tuple[str, str], set[int]] = {}
    mtp_modules = set()
    for row in experts:
        module, projection = row["name"].split(".experts.")
        projection_name = projection.split(".")[0]
        resolved = re.search(r"\.weight(\d+)$", row["resolved_name"])
        if resolved is None or projection_name not in ("linear_fc1", "linear_fc2"):
            raise ValueError("Unsupported expert parameter name")
        projections.setdefault((module, projection_name), set()).add(int(resolved[1]))
        shape = [5120, 2048] if projection_name == "linear_fc1" else [2048, 5120]
        if (
            row["shape"] != shape
            or row["dtype"] != "torch.bfloat16"
            or not row["parameter"]
        ):
            raise ValueError("Expert format differs from reviewed BF16 Parameters")
        if row["stride"] != [shape[1], 1]:
            raise ValueError("Expert matrix is not contiguous")
        if shared and row["aliases_mapped_source"] is not True:
            raise ValueError("Expert matrix does not alias its mapped source")
        if ".mtp." in f".{module}.":
            mtp_modules.add(module)
            if re.search(r"(?:^|\.)mtp\.layers\.0\.", module) is None:
                raise ValueError("MTP has an unexpected physical block index")
    if len(projections) != 98 or len({module for module, _ in projections}) != 49:
        raise ValueError("Expected 49 physical expert modules and 98 projections")
    if len(mtp_modules) != 1:
        raise ValueError("Expected one physical repeated MTP expert module")
    expected_ids = set(range(8 * ep_rank, 8 * ep_rank + 8))
    if any(ids != expected_ids for ids in projections.values()):
        raise ValueError("Expert global IDs differ from local EP ownership")
    if record["role"] == "engine":
        stacks = record["expert_stacks"]
        if len(stacks) != 49 or not all(stack["built"] for stack in stacks):
            raise ValueError("Expected 49 initialized native expert stack modules")


def validate(
    records: list[dict[str, Any]],
    *,
    concurrent: bool,
    updates: int,
    world_size: int = 64,
) -> dict:
    """Validate rank coverage and same-host lifecycle correlation only."""
    if updates < 1:
        raise ValueError("Positive completed update count required")
    processes: dict[tuple[str, int], list[dict]] = {}
    for row in records:
        if row["concurrent"] is not concurrent:
            raise ValueError("Mixed execution modes in one result directory")
        processes.setdefault((row["hostname"], row["pid"]), []).append(row)
    learners = {}
    engines = {}
    memory = []
    for identity, events in processes.items():
        events.sort(key=lambda row: row["sequence"])
        if [row["sequence"] for row in events] != list(range(1, len(events) + 1)):
            raise ValueError(f"Missing or duplicate process event sequence: {identity}")
        if any(
            a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(events, events[1:])
        ):
            raise ValueError("Process monotonic timestamps moved backwards")
        if any(
            type(row[clock]) is not int or row[clock] <= 0
            for row in events
            for clock in ("monotonic_ns", "wall_ns")
        ):
            raise ValueError("Missing native clock observations")
        if any(row["publication"]["rejected_dispatches"] for row in events):
            raise ValueError("A postpublication gather was rejected")
        rank = events[0]["rank"]
        if any(row["rank"] != rank for row in events):
            raise ValueError("One process reports inconsistent ranks")
        process_role = events[0]["process_role"]
        target = learners if process_role == "learner" else engines
        if rank in target:
            raise ValueError("Duplicate rank process")
        target[rank] = events
        for row in events:
            stats = row["memory"]
            if not 0 <= stats["allocated"] <= stats["cumulative_peak_allocated"]:
                raise ValueError("Invalid allocator highwater observation")
            if not 0 <= stats["reserved"] <= stats["cumulative_peak_reserved"]:
                raise ValueError("Invalid reserved highwater observation")
        memory.append(
            {
                "identity": identity,
                "rank": rank,
                "process_role": process_role,
                "peak_allocated": max(
                    row["memory"]["cumulative_peak_allocated"] for row in events
                ),
                "peak_reserved": max(
                    row["memory"]["cumulative_peak_reserved"] for row in events
                ),
            }
        )
    if set(learners) != set(range(world_size)) or set(engines) != (
        set(range(world_size)) if concurrent else set()
    ):
        raise ValueError(
            "Expected 64 learners and 64 concurrent engines or 64 shared alternating actors"
        )
    for rank, events in learners.items():
        for event in (
            "train_begin_before",
            "train_begin_after",
            "first_microbatch_returned",
            "optimizer_before",
        ):
            selected = [row for row in events if row["event"] == event]
            if [row["optimizer_calls"] for row in selected] != list(range(updates)):
                raise ValueError(f"Incomplete learner lifecycle {rank}: {event}")
        finished = [row for row in events if row["event"] == "optimizer_returned"]
        if [row["optimizer_calls"] for row in finished] != list(range(1, updates + 1)):
            raise ValueError("Incomplete optimizer lifecycle")
        initial = next(row for row in events if row["event"] == "constructed")
        check_catalog(initial, shared=False)
        generation_events = engines[rank] if concurrent else events
        prepared = [
            row for row in generation_events if row["event"] == "generation_prepared"
        ]
        if not prepared:
            raise ValueError("No prepared inference model evidence")
        for row in prepared:
            check_catalog(row, shared=concurrent)
            learner_signature = {
                item["resolved_name"]: (item["shape"], item["dtype"])
                for item in initial["catalog"]
            }
            engine_signature = {
                item["resolved_name"]: (item["shape"], item["dtype"])
                for item in row["catalog"]
            }
            if learner_signature != engine_signature:
                raise ValueError(
                    "Learner and inference refit name/shape/dtype catalogs differ"
                )
        if not concurrent:
            continue
        bound = [row for row in generation_events if row["event"] == "bound"]
        if len(bound) != 1:
            raise ValueError("Expected one IPC binding per engine")
        check_catalog(bound[0], shared=True)
        if (
            initial["device_uuid"] != bound[0]["device_uuid"]
            or initial["hostname"] != bound[0]["hostname"]
        ):
            raise ValueError("Learner and engine do not share a physical GPU/host")
        if initial["clock_domain"] != bound[0]["clock_domain"]:
            raise ValueError(
                "Same-host actors have different monotonic clock namespaces"
            )
        if initial["rank_coordinates"] != bound[0]["rank_coordinates"]:
            raise ValueError("TP/PP shard coordinates differ")
        if any(
            row["expert_storage_digest"] != bound[0]["expert_storage_digest"]
            for row in generation_events
            if row["sequence"] >= bound[0]["sequence"]
        ):
            raise ValueError("Expert storage changed after binding")
        for update in range(1, updates + 1):

            def one(rows: list[dict], event: str, field: str, value: int) -> dict:
                matches = [
                    row for row in rows if row["event"] == event and row[field] == value
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {event} at {field}={value}")
                return matches[0]

            drained = one(
                generation_events, "reader_drained", "weight_epoch", update - 1
            )
            before = one(events, "optimizer_before", "optimizer_calls", update - 1)
            published = one(events, "parameters_published", "optimizer_calls", update)
            copied = one(
                generation_events,
                "private_buffers_published",
                "weight_epoch",
                update - 1,
            )
            unpause = one(
                generation_events, "reader_before_unpause", "weight_epoch", update
            )
            resumed = one(generation_events, "reader_unpaused", "weight_epoch", update)
            chain = [drained, before, published, copied, unpause, resumed]
            if any(
                a["monotonic_ns"] > b["monotonic_ns"] for a, b in zip(chain, chain[1:])
            ):
                raise ValueError("Same-host publication/reader order violated")
            if not published["bucket_states"] or not all(
                dispatched and not handle
                for dispatched, handle in published["bucket_states"]
            ):
                raise ValueError("Publication retained incomplete gather state")
            biases = copied["router_biases"]
            if len(biases) != 49 or not all(
                row["private"] and row["equal"] for row in biases
            ):
                raise ValueError("Private router bias publication mismatch")
    return {
        "observer_checks_pass": True,
        "native_full_qualification_pass": False,
        "mode": "concurrent" if concurrent else "alternating",
        "updates": updates,
        "processes": len(processes),
        "allocator_highwaters": memory,
        "limits": [
            "Monotonic clocks compared only for same-host paired GPU processes; wall clocks are diagnostic only.",
            "No postpublication bucket gather observed is not an arbitrary GPU-write detector.",
            "External physical-GPU memory, actual long-prefill completion, graph/cache configuration, G16/reward/work/lag checks remain separate mandatory gates.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--mode", choices=("alternating", "concurrent"), required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--world-size", type=int, choices=(64, 128), default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = [json.loads(path.read_text()) for path in args.records.glob("*.json")]
    result = validate(
        records,
        concurrent=args.mode == "concurrent",
        updates=args.updates,
        world_size=args.world_size,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
