# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""CPU-only negative checks for prospective MTP bind/payload qualification."""

import ast
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mtp_fix_guard import verify_imports
from raw_logprob_guard import require_raw_logprob_lengths
from verify_run import validate_raw_payload_evidence

SOURCE = Path(__file__).resolve().parents[2] / "mtp_fix/SOURCE.json"


class MTPFixGuardTests(unittest.TestCase):
    def test_actual_import_paths_hashes_and_controller_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = {"files": {}}
            modules = {}
            for kind in ("pytorch", "triton"):
                name = f"mtp_utils_{kind}.py"
                path = root / name
                path.write_text(kind)
                metadata["files"][name] = {
                    "native_relative_path": name,
                    "fixed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                modules[kind] = SimpleNamespace(
                    __file__=str(path), prepare_next_forward_pass=object()
                )
            manifest = root / "SOURCE.json"
            manifest.write_text(json.dumps(metadata))
            controller = SimpleNamespace(
                prepare_next_forward_pass=modules["triton"].prepare_next_forward_pass
            )

            def imported(name):
                return (
                    controller
                    if name.endswith("text_generation_controller")
                    else modules[name.rsplit("_", 1)[1]]
                )

            with patch("mtp_fix_guard.importlib.import_module", side_effect=imported):
                self.assertEqual(
                    verify_imports(manifest, root)["status"],
                    "ACTUAL_MTP_MODULE_AND_CONTROLLER_IDENTITY_PASS",
                )
                controller.prepare_next_forward_pass = object()
                with self.assertRaisesRegex(ValueError, "controller"):
                    verify_imports(manifest, root)
                controller.prepare_next_forward_pass = modules[
                    "triton"
                ].prepare_next_forward_pass
                (root / "mtp_utils_pytorch.py").write_text("old bytes")
                with self.assertRaisesRegex(ValueError, "bytes differ"):
                    verify_imports(manifest, root)

    def test_actual_worker_method_rejects_before_native_pack(self):
        tree = ast.parse(
            (Path(__file__).resolve().parents[1] / "math_worker.py").read_text()
        )
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        parse = next(
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "_parse_result_to_batched_data_dict"
        )
        records = []
        native_calls = []
        sentinel = object()

        class Parent:
            def _write_raw_record(self, record):
                records.append(record)

            def _parse_result_to_batched_data_dict(self, data, result):
                native_calls.append((data, result))
                return sentinel

        reduced = ast.ClassDef(
            name="ActualGuard",
            bases=[ast.Name(id="Parent", ctx=ast.Load())],
            keywords=[],
            body=[parse],
            decorator_list=[],
        )
        namespace = {
            "Parent": Parent,
            "require_raw_logprob_lengths": require_raw_logprob_lengths,
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[reduced], type_ignores=[])),
                "math_worker.py",
                "exec",
            ),
            namespace,
        )
        worker = namespace["ActualGuard"]()
        bad = SimpleNamespace(
            request_id=1, generated_tokens=[1, 11], generated_log_probs=[]
        )
        with self.assertRaises(ValueError):
            worker._parse_result_to_batched_data_dict(None, [bad])
        self.assertEqual(native_calls, [])
        self.assertEqual(records[-1]["status"], "RAW_PAYLOAD_MISMATCH")
        good = SimpleNamespace(
            request_id=2, generated_tokens=[1, 11], generated_log_probs=[-1.0, -2.0]
        )
        self.assertIs(worker._parse_result_to_batched_data_dict(None, [good]), sentinel)
        self.assertEqual(len(native_calls), 1)
        self.assertEqual(records[-1]["status"], "RAW_PAYLOAD_LENGTHS_PASS")

    def test_postrun_evidence_cannot_pass_with_only_tensor_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                validate_raw_payload_evidence(root, "alternating", SOURCE)
            source = json.loads((SOURCE).read_text())
            proof = {
                "status": "ACTUAL_MTP_MODULE_AND_CONTROLLER_IDENTITY_PASS",
                "controller_uses_fixed_triton_helper": True,
                "imports": {
                    k: {"sha256": source["files"][f"mtp_utils_{k}.py"]["fixed_sha256"]}
                    for k in ("pytorch", "triton")
                },
            }
            (root / "actual-mtp-import-guard.json").write_text(json.dumps(proof))
            guards = root / "mtp-import-guards"
            guards.mkdir()
            for i in range(128):
                (guards / f"{i}.json").write_text(
                    json.dumps(dict(proof, hostname="fixture", pid=i))
                )
            rows = root / "raw-generation-payloads"
            rows.mkdir()
            raw = rows / "fixture.jsonl"
            event = {
                "status": "RAW_PAYLOAD_LENGTHS_PASS",
                "call_ordinal": 1,
                "requests": [
                    {"request_id": i, "raw_token_count": 4, "raw_logprob_count": 4}
                    for i in range(192)
                ],
            }
            raw.write_text(json.dumps(event) + "\n")
            self.assertEqual(
                validate_raw_payload_evidence(root, "alternating", SOURCE)[
                    "raw_request_lengths_checked"
                ],
                192,
            )
            event["requests"][-1]["raw_logprob_count"] = 3
            raw.write_text(json.dumps(event) + "\n")
            with self.assertRaisesRegex(ValueError, "length mismatch"):
                validate_raw_payload_evidence(root, "alternating", SOURCE)


if __name__ == "__main__":
    unittest.main()
