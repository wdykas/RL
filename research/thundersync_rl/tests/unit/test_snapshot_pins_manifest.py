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
"""The branch's pinned dependency trees (configs/snapshot_pins.manifest) are tracked and match a snapshot's pins.

A frozen snapshot carries an untracked ``.snapshot_pins`` (absolute roots + commit + content hash) that
module_provenance enforces at startup. The manifest is the tracked record of the same pins: a snapshot whose
``.snapshot_pins`` names other commits or hashes than its own branch manifest is refused here.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
MANIFEST = ROOT / "research/thundersync_rl/configs/snapshot_pins.manifest"
PKGS = ("MEGATRON_LM", "MEGATRON_BRIDGE", "TE")


def parse(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k] = v
    return out


def test_manifest_is_complete():
    m = parse(MANIFEST)
    for p in PKGS:
        assert re.fullmatch(r"[0-9a-f]{64}", m[f"{p}_HASH"]), p
        assert m[f"{p}_DIR"], p
    for p in ("MEGATRON_LM", "MEGATRON_BRIDGE"):
        assert re.fullmatch(r"[0-9a-f]{40}", m[f"{p}_SHA"]), p
        assert m[f"{p}_DIR"].split("/")[0].startswith(("Megatron-LM", "Megatron-Bridge")), p


def mismatches(m: dict[str, str], s: dict[str, str]) -> list[str]:
    """Pins of a snapshot (s) that differ from the manifest (m)."""
    bad = [f"{p}_HASH" for p in PKGS if s.get(f"{p}_HASH") != m[f"{p}_HASH"]]
    bad += [f"{p}_ROOT" for p in PKGS if not s.get(f"{p}_ROOT", "").rstrip("/").endswith("/" + m[f"{p}_DIR"])]
    bad += [f"{p}_SHA" for p in ("MEGATRON_LM", "MEGATRON_BRIDGE") if s.get(f"{p}_SHA") != m[f"{p}_SHA"]]
    return bad


def test_snapshot_pins_match_the_manifest():
    pins = ROOT / ".snapshot_pins"
    if not pins.is_file():
        pytest.skip("not a frozen snapshot (no .snapshot_pins)")
    assert mismatches(parse(MANIFEST), parse(pins)) == []


def test_pinned_megatron_trees_are_at_their_commits():
    m = parse(MANIFEST)
    seen = 0
    for p in ("MEGATRON_LM", "MEGATRON_BRIDGE"):
        tree = ROOT.parent / m[f"{p}_DIR"].split("/")[0]
        if not tree.is_dir():
            continue
        head = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        assert head == m[f"{p}_SHA"], (p, head)
        seen += 1
    if not seen:
        pytest.skip("pinned trees are not next to this checkout")


def test_a_wrong_pin_is_caught():
    """Negative controls for the comparison: a matching pins file passes, each altered field is reported."""
    m = parse(MANIFEST)
    good = {k: v for k, v in m.items() if not k.endswith("_DIR")}
    good.update({f"{p}_ROOT": f"/somewhere/{m[f'{p}_DIR']}" for p in PKGS})
    assert mismatches(m, good) == []
    assert mismatches(m, dict(good, MEGATRON_LM_SHA="0" * 40)) == ["MEGATRON_LM_SHA"]
    assert mismatches(m, dict(good, TE_HASH="f" * 64)) == ["TE_HASH"]
    assert mismatches(m, dict(good, MEGATRON_BRIDGE_ROOT="/somewhere/Megatron-Bridge-other/src")) == ["MEGATRON_BRIDGE_ROOT"]
