# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Pure metadata/config regressions, not actual native parser or host-fit proof."""

import ast
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from host_telemetry import capture_host_memory, _json_metadata


class HostTelemetryTests(unittest.TestCase):
    def test_host_capture_reads_without_reset_or_tensor_stringification(self):
        calls = []

        def native_stats():
            calls.append("query")
            return {"allocated_bytes.current": 123, "reserved_bytes.current": 456}

        with patch(
            "host_telemetry.host_scope",
            return_value={"process_status": ["VmRSS: 7 kB", "VmHWM: 9 kB"]},
        ):
            result = capture_host_memory(host_stats_fn=native_stats)
        self.assertEqual(calls, ["query"])
        self.assertEqual(
            result["native_host_memory_stats"]["values"]["allocated_bytes.current"], 123
        )
        self.assertEqual(
            result["host_scope"]["process_status"], ["VmRSS: 7 kB", "VmHWM: 9 kB"]
        )

        class NoString:
            def __str__(self):
                raise AssertionError("Must not read/stringify unexpected values")

        self.assertEqual(_json_metadata(NoString()), {"unsupported_type": "NoString"})

    def test_actual_worker_snapshot_preserves_native_call_and_no_reset(self):
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "math_worker.py").read_text()
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        method = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name == "_snapshot_locked"
        )
        calls = []
        queries = []
        sentinel = object()

        class Native:
            def _snapshot_locked(self, *args, **kwargs):
                calls.append((args, kwargs))
                return sentinel

        def native_stats():
            queries.append("read_only_query")
            return {"allocated_bytes.current": 123}

        namespace = {
            "Native": Native,
            "capture_host_memory": capture_host_memory,
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    memory=SimpleNamespace(host_memory_stats=native_stats)
                )
            ),
        }
        reduced = ast.ClassDef(
            name="ActualWorker",
            bases=[ast.Name(id="Native", ctx=ast.Load())],
            keywords=[],
            body=[method],
            decorator_list=[],
        )
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[reduced], type_ignores=[])),
                "math_worker.py",
                "exec",
            ),
            namespace,
        )
        with patch("host_telemetry.host_scope", return_value={}):
            result = namespace["ActualWorker"]()._snapshot_locked(
                "optimizer_before", catalog=True, weight_epoch=1
            )
        self.assertIs(result, sentinel)
        self.assertEqual(queries, ["read_only_query"])
        args, kwargs = calls[0]
        self.assertEqual(args, ("optimizer_before",))
        self.assertTrue(kwargs["catalog"])
        self.assertEqual(kwargs["weight_epoch"], 1)
        self.assertEqual(
            kwargs["host_memory"]["native_host_memory_stats"]["status"], "AVAILABLE"
        )

    def test_unresolved_membership_retains_unknown_scope_mount_counters(self):
        from host_telemetry import host_scope

        def read(path):
            value = str(path)
            if value == "/proc/self/cgroup":
                return "0::/../../outside\n"
            if value == "/proc/self/mountinfo":
                return "1 0 0:1 / /sys/fs/cgroup rw - cgroup2 cgroup rw\n"
            if value == "/proc/self/status":
                return "RssAnon: 12 kB\nVmSize: 99 kB\n"
            if value.startswith("/sys/fs/cgroup/memory."):
                return "42\n"
            raise AssertionError(value)

        with patch.object(Path, "read_text", read):
            record = host_scope()
        row = record["visible_memory"][0]
        self.assertIn("unavailable", row)
        mounted = row["literal_mountpoint_counters"]
        self.assertEqual(mounted["scope"], "UNKNOWN_SCOPE_LITERAL_MOUNTPOINT")
        self.assertEqual(mounted["files"]["memory.current"], "42\n")
        self.assertIn("RssAnon: 12 kB", record["process_status"])

    def test_missing_native_stats_reports_unknown_not_fake_zero(self):
        with patch("host_telemetry.host_scope", side_effect=OSError("not mounted")):
            result = capture_host_memory()
        self.assertIn("unavailable", result["host_scope"])
        with patch("host_telemetry.host_scope", return_value={}):
            missing = capture_host_memory(native_stats_requested=True)
        self.assertEqual(missing["native_host_memory_stats"]["status"], "UNAVAILABLE")
        self.assertEqual(
            result["native_host_memory_stats"]["status"],
            "NOT_REQUESTED_NODE_MONITOR_NO_TORCH_IMPORT",
        )


if __name__ == "__main__":
    unittest.main()
