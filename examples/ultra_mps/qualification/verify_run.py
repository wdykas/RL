# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Read-only three-update native work, real GRPO signal and observer qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def read_events(paths: list[Path]) -> dict:
    # TensorBoard is required only by native postrun qualification.
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
    from tensorboard.util.tensor_util import make_ndarray

    rows = {}
    for path in paths:
        for event in EventFileLoader(str(path)).Load():
            for value in event.summary.value:
                if not value.tag.startswith(("train/", "timing/train/")):
                    continue
                if value.HasField("simple_value"):
                    scalar = float(value.simple_value)
                elif value.HasField("tensor"):
                    data = make_ndarray(value.tensor)
                    if data.size != 1:
                        continue
                    scalar = float(data.item())
                else:
                    continue
                key = (int(event.step), value.tag)
                if key in rows:
                    raise ValueError(f"Duplicate native completed-step metric: {key}")
                rows[key] = (float(event.wall_time), scalar)
    return rows


def validate_work(rows: dict) -> dict:
    boundary = "timing/train/total_step_time"
    if sorted(step for step, tag in rows if tag == boundary) != [1, 2, 3]:
        raise ValueError("Require exactly three native one-based completed updates")
    required = (
        "train/global_valid_seqs",
        "train/global_valid_toks",
        "train/loss",
        "train/grad_norm",
        "train/reward",
        "train/weight_version_lag",
        "train/advantages/min",
        "train/advantages/max",
        boundary,
    )
    waste = tuple(
        "train/" + name
        for name in (
            "dropped_prompt_groups",
            "evicted_stale_prompt_groups",
            "aborted_stale_inflight_groups",
            "replaced_prompt_groups",
            "promoted_prompt_groups",
        )
    )
    updates = []
    for step in (1, 2, 3):
        for key in required + waste:
            if (step, key) not in rows:
                raise ValueError(f"Missing native work/signal evidence {step}:{key}")
        values = {
            tag: value for (number, tag), (_, value) in rows.items() if number == step
        }
        if not all(
            math.isfinite(value) and math.isfinite(clock)
            for (number, _), (clock, value) in rows.items()
            if number == step
        ):
            raise ValueError("Nonfinite native metric or event clock")
        if (
            values["train/global_valid_seqs"] != 64
            or values["train/global_valid_toks"] <= 0
        ):
            raise ValueError("Incomplete64sequence cohort or no valid tokens")
        if values["train/weight_version_lag"] not in (0, 1) or any(
            values[key] != 0 for key in waste
        ):
            raise ValueError("Lag/waste/replacement contract failed")
        updates.append(
            {
                "native_step": step,
                "boundary_event_wall_time": rows[step, boundary][0],
                "metrics": values,
            }
        )
    if not all(
        b["boundary_event_wall_time"] > a["boundary_event_wall_time"]
        for a, b in zip(updates, updates[1:])
    ):
        raise ValueError("Nonmonotonic completed-update event clock")
    nonzero = [
        u["native_step"]
        for u in updates
        if u["metrics"]["train/grad_norm"] > 0
        and u["metrics"]["train/advantages/min"]
        < 0
        < u["metrics"]["train/advantages/max"]
    ]
    return {
        "native_work_pass": True,
        "nonzero_GRPO_steps": nonzero,
        "GRPO_signal_pass": len(nonzero) == 3,
        "updates": updates,
        "scope": "Functional work only, no steady-state throughput or model numerical-equivalence claim",
    }


def validate_raw_payload_evidence(run: Path, arm: str, source_manifest: Path) -> dict:
    source = json.loads(source_manifest.read_text())
    guards = sorted((run / "mtp-import-guards").glob("*.json"))
    expected = 256 if arm == "concurrent" else 128
    if len(guards) != expected:
        raise ValueError(f"Require {expected} actual worker MTP import guards")
    all_guards = [json.loads((run / "actual-mtp-import-guard.json").read_text())]
    all_guards.extend(json.loads(path.read_text()) for path in guards)
    identities = set()
    for index, guard in enumerate(all_guards):
        if (
            guard["status"] != "ACTUAL_MTP_MODULE_AND_CONTROLLER_IDENTITY_PASS"
            or guard["controller_uses_fixed_triton_helper"] is not True
        ):
            raise ValueError("Actual native MTP import/controller proof missing")
        for kind in ("pytorch", "triton"):
            if (
                guard["imports"][kind]["sha256"]
                != source["files"][f"mtp_utils_{kind}.py"]["fixed_sha256"]
            ):
                raise ValueError("Worker did not import exact reviewed fixed helper")
        if index:
            identity = (guard["hostname"], guard["pid"])
            if identity in identities:
                raise ValueError("Duplicate worker import identity")
            identities.add(identity)
    requests = 0
    seen_requests = set()
    raw_files = sorted((run / "raw-generation-payloads").glob("*.jsonl"))
    for path in raw_files:
        previous = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if (
                row["status"] != "RAW_PAYLOAD_LENGTHS_PASS"
                or row["call_ordinal"] != previous + 1
            ):
                raise ValueError("Raw payload failure/gap before native packing")
            previous = row["call_ordinal"]
            for request in row["requests"]:
                if (
                    request["raw_token_count"] is None
                    or request["raw_token_count"] < 0
                    or request["raw_token_count"] != request["raw_logprob_count"]
                ):
                    raise ValueError("Raw token/logprob length mismatch")
                identity = (path.name, request["request_id"])
                if request["request_id"] is None or identity in seen_requests:
                    raise ValueError("Missing/duplicate raw request identity")
                seen_requests.add(identity)
                requests += 1
    if requests < 192:
        raise ValueError(
            "Require raw prepack evidence for at least three64sequence cohorts"
        )
    return {
        "actual_worker_MTP_import_guards": len(guards),
        "raw_request_lengths_checked": requests,
        "raw_payload_files_SHA256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in raw_files
        },
        "scope": "Complete raw payload lengths, not token/logprob indexing or numerical equivalence proof",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--arm", choices=("alternating", "concurrent"), required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "mtp_fix/SOURCE.json",
    )
    args = parser.parse_args()
    files = sorted((args.run / "training-logs").rglob("events.out.tfevents.*"))
    output = args.run / "functional-qualification.json"
    report = {
        "status": "FAILED_OR_INCOMPLETE_NATIVE_EVIDENCE",
        "arm": args.arm,
        "reserved_gpus": 128,
        "no_performance_claim": True,
    }
    try:
        work = validate_work(read_events(files))
        inventory_paths = sorted((args.run / "gpu-optimizer-inventory").glob("*.json"))
        inventories = [json.loads(path.read_text()) for path in inventory_paths]
        if len(inventories) != 128 or {row["rank"] for row in inventories} != set(
            range(128)
        ):
            raise ValueError("Require all128 native GPU optimizer inventories")
        if any(
            row["status"] != "NATIVE_DISTRIBUTED_GPU_OPTIMIZER_PASS"
            for row in inventories
        ):
            raise ValueError("GPU-only optimizer ownership proof missing")
        report["actual_gpu_optimizer_inventories"] = len(inventories)
        state_records = []
        for path in (args.run / "observer").glob("*.json"):
            row = json.loads(path.read_text())
            if row.get("process_role") != "learner" or row.get("event") not in (
                "optimizer_returned",
                "generation_prepared",
            ):
                continue
            if row.get("optimizer_calls", 0) < 1:
                continue
            inventory = row.get("gpu_optimizer_inventory", {})
            if inventory.get("status") != "NATIVE_DISTRIBUTED_GPU_OPTIMIZER_PASS":
                raise ValueError("Post-update GPU optimizer state snapshot missing")
            for child in inventory["children"]:
                for key, dtype in (
                    ("exp_avg", "torch.float32"),
                    ("exp_avg_sq", "torch.float32"),
                    ("master_param", "torch.int16"),
                ):
                    state = child["state_metadata"].get(key)
                    if (
                        not state
                        or state["device"] != "cuda"
                        or state["dtype"] != dtype
                    ):
                        raise ValueError(
                            "Post-update optimizer state residency/dtype proof failed"
                        )
            state_records.append(row)
        if (
            sum(row["event"] == "optimizer_returned" for row in state_records)
            != 3 * 128
        ):
            raise ValueError("Require all128 GPU state snapshots after each3 updates")
        report["post_update_gpu_state_snapshots"] = len(state_records)

        payloads = validate_raw_payload_evidence(
            args.run,
            args.arm,
            args.source_manifest,
        )
        report["raw_payload_and_helper_evidence"] = payloads
        observer = json.loads((args.run / "observer-validation.json").read_text())
        if not observer["observer_checks_pass"]:
            raise ValueError("All-rank observer validation did not pass")
        topology = json.loads((args.run / "topology.json").read_text())
        if (
            topology["status"] != "PRE_MODEL_NATIVE_TOPOLOGY_PASS"
            or topology["physical"]["reserved_gpus"] != 128
        ):
            raise ValueError("Native128GPU topology gate missing")
        if len(topology.get("rank_islands", [])) != 2:
            raise ValueError("Missing two native EP64 rank-island checks")
        report.update(work)
        report["status"] = (
            "THREE_UPDATE_NATIVE_MATH_WORK_OBSERVER_PASS_REVIEW_REQUIRED"
            if work["GRPO_signal_pass"]
            else "WORK_PASS_GRPO_SIGNAL_INCONCLUSIVE"
        )
        report["events_SHA256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
        }
        report["UNKNOWN_external_checks"] = [
            "root allocation completion/cleanup and physical peak audit",
            "per-sibling group identity and actual MPS registration/caps",
            "numerical equivalence/convergence and any steady speed",
        ]
    except (ValueError, KeyError, FileNotFoundError) as error:
        report["error"] = str(error)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        raise
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if not report["GRPO_signal_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
