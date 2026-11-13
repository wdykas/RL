# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Read-only host allocator/process/cgroup telemetry; never resets allocator stats."""

import argparse
import json
import math
import os
from pathlib import Path
import resource
import socket
import time

ALIASES = ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_HIP_ALLOC_CONF", "PYTORCH_ALLOC_CONF")


def host_scope() -> dict:
    membership = Path("/proc/self/cgroup").read_text()
    mounts = [
        line
        for line in Path("/proc/self/mountinfo").read_text().splitlines()
        if " - cgroup" in line
    ]
    result = {
        "membership": membership,
        "cgroup_mountinfo": mounts,
        "scope": "Self process membership and mounted visibility; not assumed to be the Slurm job root",
        "visible_memory": [],
    }
    v2 = next(
        (
            line.split(":", 2)[2]
            for line in membership.splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if v2 is not None:
        for line in mounts:
            if " - cgroup2 " not in line:
                continue
            fields = line.split(" - ", 1)[0].split()
            root, mount = Path(fields[3]), Path(fields[4])
            row = {
                "membership_path": v2,
                "mount_root": str(root),
                "mount_point": str(mount),
            }
            try:
                relative = Path(v2).relative_to(root)
                candidate = mount / relative
                assert candidate.resolve().is_relative_to(mount.resolve())
                row["directory"] = str(candidate)
                row["files"] = {}
                for name in (
                    "memory.current",
                    "memory.max",
                    "memory.peak",
                    "memory.events",
                    "memory.stat",
                ):
                    try:
                        row["files"][name] = (candidate / name).read_text()
                    except OSError as error:
                        row["files"][name] = {"unavailable": str(error)}
            except (ValueError, AssertionError) as error:
                row["unavailable"] = str(error)
            # Literal mounted counters remain useful when namespace membership
            # cannot be resolved. Never infer these are job/self counters.
            visible = {
                "directory": str(mount),
                "scope": "UNKNOWN_SCOPE_LITERAL_MOUNTPOINT",
                "files": {},
            }
            for name in (
                "memory.current",
                "memory.max",
                "memory.peak",
                "memory.events",
                "memory.stat",
            ):
                try:
                    visible["files"][name] = (mount / name).read_text()
                except OSError as error:
                    visible["files"][name] = {"unavailable": str(error)}
            row["literal_mountpoint_counters"] = visible
            result["visible_memory"].append(row)
    result["process_status"] = [
        line
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.split(":", 1)[0]
        in {
            "VmRSS",
            "VmHWM",
            "VmPin",
            "VmLck",
            "Threads",
            "RssAnon",
            "RssFile",
            "RssShmem",
            "VmSize",
        }
    ]
    return result


def _json_metadata(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_metadata(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_metadata(v) for v in value]
    # Do not stringify unknown objects (especially tensors), which could read
    # device contents or synchronize. Native host statistics should be scalars.
    return {"unsupported_type": type(value).__name__}


def capture_host_memory(*, host_stats_fn=None, native_stats_requested=False):
    result = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "wall_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "maxrss_KiB_Linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "allocator_env": {key: os.environ.get(key) for key in ALIASES},
        "scope": "Read-only self/visible cgroup counters; no allocator reset or allocation; cgroup not presumed jobroot",
    }
    try:
        result["host_scope"] = host_scope()
    except OSError as error:
        result["host_scope"] = {"unavailable": str(error)}
    if host_stats_fn is not None:
        try:
            result["native_host_memory_stats"] = {
                "status": "AVAILABLE",
                "values": _json_metadata(host_stats_fn()),
            }
        except Exception as error:
            result["native_host_memory_stats"] = {
                "status": "UNAVAILABLE",
                "error": str(error),
            }
    else:
        result["native_host_memory_stats"] = {
            "status": "UNAVAILABLE"
            if native_stats_requested
            else "NOT_REQUESTED_NODE_MONITOR_NO_TORCH_IMPORT"
        }
    return result


def node_sample():
    result = capture_host_memory()
    result["process_scope"] = (
        "This monitor process RSS/HWM is not the sum of learner/engine RSS; use node meminfo and recorded cgroup membership/visibility."
    )
    try:
        result["node_meminfo"] = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.split(":", 1)[0]
            in {
                "MemTotal",
                "MemAvailable",
                "MemFree",
                "Buffers",
                "Cached",
                "Shmem",
                "Mlocked",
                "Unevictable",
                "SwapTotal",
                "SwapFree",
            }
        }
    except OSError as error:
        result["node_meminfo"] = {"unavailable": str(error)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ended", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    args = parser.parse_args()
    if not math.isfinite(args.interval_seconds) or args.interval_seconds < 1:
        raise ValueError("Host monitoring interval must be finite and at least1second")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while not args.ended.exists():
        with args.output.open("a") as stream:
            stream.write(json.dumps(node_sample(), allow_nan=False) + "\n")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
