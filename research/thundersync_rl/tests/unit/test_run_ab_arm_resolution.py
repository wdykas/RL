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
"""The gate / paper configs resolve identically through the tracked A/B arm launcher.

Every gate and paper run goes through ``research/thundersync_rl/scripts/ab/run_ab_arm.sh``: it turns the job's
environment (LGRAPHS, CHECKS, V0B, EPINV, PREFIX_CACHE, LANES, CKPT_DIR, ...) into the Hydra overrides and the
exported check / dispatch variables. Each fixture in ``data/run_ab_arm/<name>.cmd`` is the launcher assignment of a
real driver command (submit_ab.sh DRY_CMD_DIR: Super-L22 gate3 ts + async-1, the X5 alpha sweep, the Nano
acceptance-decay jitter and collapse runs). The launcher runs with ``timeout`` and ``python`` stubbed; the captured
argv of the training command and the exported THUNDERSYNC_* / NEMO_RL_* variables must equal
``<name>.golden``, which was produced by the launcher copy the runs used: the frozen final snapshot's
(RL-clean-scale-056a8d3d) for the gate / X5 / acceptance-decay commands, the lanes-k snapshot's
(RL-clean-scale-16166771) for the ``lanesk_*`` commands (async-1 on lanes, async-2, async-4, ThunderSync depth 4).
``paper_ts_*`` is the config route (CFG_FILE=configs/paper/ts.yaml) of the gate layout, produced by this tree.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = "research/thundersync_rl/scripts/ab/run_ab_arm.sh"
DATA = Path(__file__).resolve().parent / "data" / "run_ab_arm"
W = os.environ.get("TS_WORK", "/workspace/thundersync_work")
# NEMO_RL_SNAPSHOT_PINS names the tree's own pins file (absent in a branch checkout): not part of the config.
_ENV_SKIP = ("NEMO_RL_SNAPSHOT_PINS=",)

_TIMEOUT_STUB = """#!/bin/bash
shift
for a in "$@"; do printf 'ARG %s\\n' "$a"; done > "$AB_CAPTURE"
env | grep -E '^(THUNDERSYNC_|NEMO_RL_|NVIDIA_TF32_OVERRIDE=|RAY_DEDUP_LOGS=)' | sort | sed 's/^/ENV /' >> "$AB_CAPTURE"
exit 0
"""


def resolve(tree: Path, assign: str, tmp: Path) -> str:
    """Run ``tree``'s run_ab_arm.sh with the fixture's assignments; return the normalized capture."""
    stub = tmp / "bin"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "timeout").write_text(_TIMEOUT_STUB)
    (stub / "python").write_text("#!/bin/bash\nexit 0\n")
    for f in ("timeout", "python"):
        (stub / f).chmod(0o755)
    runs = os.path.relpath(tmp / "runs", W)
    cap = tmp / "capture.txt"
    assign = assign.replace("{W}", W).replace("{TREE}", str(tree)).replace("{RUNS}", runs)
    cmd = f"cd {shlex.quote(str(tree))} && export TS_WORK={shlex.quote(W)} PATH={shlex.quote(str(stub))}:$PATH AB_CAPTURE={shlex.quote(str(cap))} && {assign} bash {SCRIPT}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("THUNDERSYNC_", "NEMO_RL_"))}
    p = subprocess.run(["bash", "-c", cmd], env=env, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout + p.stderr
    out = cap.read_text()
    out = out.replace(f"{W}/{runs}", "{OUT_ROOT}").replace(str(tmp / "runs"), "{OUT_ROOT}").replace(str(tree), "{TREE}").replace(W, "{W}")
    return "".join(l + "\n" for l in out.splitlines() if not l.startswith(tuple("ENV " + s for s in _ENV_SKIP)))


CASES = sorted(p.stem for p in DATA.glob("*.cmd"))


@pytest.mark.skipif(not Path(W, "env_sbatch.sh").is_file(), reason="needs the ThunderSync harness (env_sbatch.sh)")
@pytest.mark.parametrize("name", CASES)
def test_gate_config_resolves_identically(name, tmp_path):
    got = resolve(ROOT, (DATA / f"{name}.cmd").read_text().strip(), tmp_path)
    assert got == (DATA / f"{name}.golden").read_text()


@pytest.mark.skipif(not Path(W, "env_sbatch.sh").is_file(), reason="needs the ThunderSync harness (env_sbatch.sh)")
def test_a_changed_switch_changes_the_resolution(tmp_path):
    """Negative control: the comparison sees a gate switch (learner graphs dd -> 1 adds the graph env and overrides)."""
    a = (DATA / "gate3_super_l22_ts.cmd").read_text().strip()
    assert "LGRAPHS=dd " in a
    got = resolve(ROOT, a.replace("LGRAPHS=dd ", "LGRAPHS=1 "), tmp_path)
    assert got != (DATA / "gate3_super_l22_ts.golden").read_text()
    assert "ENV NEMO_RL_LEARNER_GRAPHS=1" in got


def test_fixtures_present_and_cover_the_gate_switches():
    assert len(CASES) >= 5
    text = " ".join((DATA / f"{n}.cmd").read_text() for n in CASES)
    for knob in ("LGRAPHS=", "CHECKS=", "V0B=", "LANES=", "CKPT_DIR=", "PREFIX_CACHE="):
        assert knob in text, knob
    for n in CASES:
        g = (DATA / f"{n}.golden").read_text()
        assert "ARG --config" in g and ("ENV THUNDERSYNC_" in g or "ENV NEMO_RL_" in g), n


def test_launcher_is_tracked():
    p = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--error-unmatch", SCRIPT], capture_output=True, text=True)
    if p.returncode == 128 and "not a git repository" in p.stderr:
        pytest.skip("not a git checkout")
    assert p.returncode == 0, p.stderr
