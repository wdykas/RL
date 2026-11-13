# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Submission contract checks with fake Git/Slurm, without requesting resources."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

EXAMPLE = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(tmp_path):
    repo = tmp_path / "repo"
    example = repo / "examples/ultra_mps"
    example.mkdir(parents=True)
    shutil.copy(EXAMPLE / "submit.sh", example / "submit.sh")
    tools = tmp_path / "bin"
    tools.mkdir()
    mock = (
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from pathlib import Path
if Path(sys.argv[0]).name == "git":
    if "diff" in sys.argv:
        sys.exit(int(os.environ.get("TEST_DIRTY_SOURCE", "0")))
    print("3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a" if sys.argv[2].endswith("mcore") else "7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e")
else:
    Path(os.environ["TEST_SUBMIT_RECORD"]).write_text(json.dumps({"args": sys.argv[1:], "arm": os.environ["ULTRA_DIRECT64_ARM"], "cwd": os.getcwd()}))
    print("12345")
"""
    )
    for name in ("git", "sbatch"):
        path = tools / name
        path.write_text(mock)
        path.chmod(0o755)
    for name in ("model", "mcore", "bridge"):
        (tmp_path / name).mkdir()
    (tmp_path / "model/config.json").write_text("{}")
    record = tmp_path / "submission.json"
    env = {
        key: value for key, value in os.environ.items() if not key.startswith("ULTRA_")
    }
    env.update(
        PATH=str(tools) + os.pathsep + env["PATH"],
        ULTRA_ACCOUNT="test-account",
        ULTRA_PARTITION="test-partition",
        ULTRA_CONTAINER="image.sqsh",
        ULTRA_SHARED_ROOT=str(tmp_path),
        ULTRA_MODEL_PATH=str(tmp_path / "model"),
        ULTRA_MCORE_ROOT=str(tmp_path / "mcore"),
        ULTRA_BRIDGE_ROOT=str(tmp_path / "bridge"),
        ULTRA_OUTPUT_ROOT=str(tmp_path / "outputs"),
        ULTRA_HF_HOME=str(tmp_path / "cache"),
        ULTRA_DRIVER_PYTHON="/opt/driver/bin/python",
        ULTRA_POLICY_PYTHON="/opt/policy/bin/python",
        ULTRA_PLACEMENT_ARGS="--segment=16",
        TEST_SUBMIT_RECORD=str(record),
    )
    return example / "submit.sh", env, record


def test_dry_run_never_submits(launcher):
    script, env, record = launcher
    result = subprocess.run(
        ["bash", str(script), "concurrent"], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "DRY RUN:" in result.stdout and "--segment=16" in result.stdout
    assert not record.exists()


def test_explicit_submission_keeps_geometry_and_mode(launcher):
    script, env, record = launcher
    result = subprocess.run(
        ["bash", str(script), "alternating", "--qos=test-qos"],
        env={**env, "ULTRA_SUBMIT": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(record.read_text())
    assert data["arm"] == "alternating"
    assert Path(data["cwd"]) == script.parents[2]
    for flag in (
        "--nodes=32",
        "--gres=gpu:4",
        "--segment=16",
        "--qos=test-qos",
        "--account=test-account",
    ):
        assert flag in data["args"]


@pytest.mark.parametrize(
    "change",
    [
        {"TEST_DIRTY_SOURCE": "1"},
        {"ULTRA_SUBMIT": "invalid"},
        {"ULTRA_MODEL_PATH": "/tmp/not-shared"},
        {"ULTRA_CONTAINER": "bad;command"},
    ],
)
def test_invalid_inputs_never_submit(launcher, change):
    script, env, record = launcher
    result = subprocess.run(
        ["bash", str(script), "concurrent"],
        env={**env, "ULTRA_SUBMIT": "1", **change},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not record.exists()
