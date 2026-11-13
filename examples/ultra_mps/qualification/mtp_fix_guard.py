# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied.
"""Verify actual normal-module MTP helper bytes and controller identity."""

import argparse
import hashlib
import importlib
import json
from pathlib import Path


def verify_imports(source_manifest: Path, mcore_source: Path) -> dict:
    source = json.loads(source_manifest.read_text())
    modules = {}
    evidence = {}
    for kind in ("pytorch", "triton"):
        name = f"mtp_utils_{kind}.py"
        metadata = source["files"][name]
        module = importlib.import_module(
            f"megatron.core.inference.text_generation_controllers.mtp_utils_{kind}"
        )
        actual = Path(module.__file__)
        expected = mcore_source / metadata["native_relative_path"]
        if not actual.samefile(expected):
            raise ValueError(f"MTP helper imported from unexpected path: {actual}")
        digest = hashlib.sha256(actual.read_bytes()).hexdigest()
        if digest != metadata["fixed_sha256"]:
            raise ValueError(f"MTP helper bytes differ: {actual}")
        modules[kind] = module
        evidence[kind] = {"path": str(actual), "sha256": digest}
    controller = importlib.import_module(
        "megatron.core.inference.text_generation_controllers.text_generation_controller"
    )
    if (
        controller.prepare_next_forward_pass
        is not modules["triton"].prepare_next_forward_pass
    ):
        raise ValueError("Actual controller is not bound to fixed native Triton helper")
    return {
        "status": "ACTUAL_MTP_MODULE_AND_CONTROLLER_IDENTITY_PASS",
        "imports": evidence,
        "controller_uses_fixed_triton_helper": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--mcore-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify_imports(args.source_manifest, args.mcore_source)
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
