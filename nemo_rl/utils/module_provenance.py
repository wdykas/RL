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
"""Which source tree every loaded copy of Megatron-LM, Megatron-Bridge, Transformer Engine and NeMo-RL came from.

The container image ships its own copies (``/opt/nemo-rl/3rdparty/...``, the venv's site-packages) and a launch puts
other copies first on ``PYTHONPATH`` (the run's tree, ``RL/3rdparty``, overlays). A process that ends up importing
modules of one package from two trees runs a mixture nobody tested. ``check_module_provenance`` groups every loaded
module of the watched packages by the root it was imported from, prints one ``[provenance]`` line per process and
raises when a package has modules from more than one root.

Snapshot pins: a frozen snapshot carries ``.snapshot_pins`` (KEY=VALUE lines written when it is frozen) naming the
exact Megatron-LM / Megatron-Bridge trees and commits and the Transformer Engine tree and content hash it runs with.
When the loaded ``nemo_rl`` comes from such a snapshot (or ``NEMO_RL_SNAPSHOT_PINS`` names a pins file), every
package must come from its pinned root, the git trees must be at the pinned commit with no tracked changes, and TE's
content hash must match; anything else raises.

``NEMO_RL_PROVENANCE_CHECK``: ``1`` (default) raise on a mix or a pin mismatch, ``warn`` print only, ``0`` off.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from typing import Optional

# package name -> the import path whose directory is the package root (module "a.b.c" lives at <root>/a/b/c.py)
WATCHED = ("megatron.core", "megatron.bridge", "transformer_engine", "nemo_rl")


def module_roots(modules: Optional[dict] = None) -> dict[str, dict[str, int]]:
    """{package: {root: number of loaded modules of that package imported from root}}."""
    modules = sys.modules if modules is None else modules
    out: dict[str, dict[str, int]] = {}
    for name, mod in list(modules.items()):
        pkg = next((p for p in WATCHED if name == p or name.startswith(p + ".")), None)
        path = getattr(mod, "__file__", None) if pkg else None
        if not path:
            continue
        rel = name.replace(".", os.sep)
        path = os.path.realpath(path)
        stem = path[: -len("/__init__.py")] if path.endswith(os.sep + "__init__.py") else os.path.splitext(path)[0]
        if not stem.endswith(rel):
            continue  # extension modules with platform suffixes etc.: rooted by their package's __init__
        root = stem[: -len(rel)].rstrip(os.sep)
        bucket = out.setdefault(pkg, {})
        bucket[root] = bucket.get(root, 0) + 1
    return out


PIN_KEYS = {  # package -> (root key, git commit key or None, content hash key)
    "megatron.core": ("MEGATRON_LM_ROOT", "MEGATRON_LM_SHA", "MEGATRON_LM_HASH"),
    "megatron.bridge": ("MEGATRON_BRIDGE_ROOT", "MEGATRON_BRIDGE_SHA", "MEGATRON_BRIDGE_HASH"),
    "transformer_engine": ("TE_ROOT", None, "TE_HASH"),
}


def read_pins(path: str) -> dict[str, str]:
    pins = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                pins[k.strip()] = v.strip()
    return pins


def pins_file(roots: dict[str, dict[str, int]]) -> Optional[str]:
    """The pins file of this process: NEMO_RL_SNAPSHOT_PINS, else <the loaded nemo_rl tree>/.snapshot_pins."""
    env = os.environ.get("NEMO_RL_SNAPSHOT_PINS")
    if env:
        return env
    for root in roots.get("nemo_rl", {}):
        cand = os.path.join(root, ".snapshot_pins")
        if os.path.isfile(cand):
            return cand
    return None


def tree_hash(root: str, package: str) -> str:
    """sha256 over (relative path, content) of every .py file under root/<package as a path>, sorted.

    The enforced pin: it needs no git in the container and sees any edit, tracked or not."""
    h = hashlib.sha256()
    base = os.path.join(root, *package.split("."))
    files = []
    for d, dirs, names in os.walk(base):
        dirs[:] = sorted(x for x in dirs if x != "__pycache__")
        files += [os.path.join(d, n) for n in names if n.endswith(".py")]
    for path in sorted(files):
        h.update(os.path.relpath(path, root).encode())
        with open(path, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


def _have_git() -> bool:
    import shutil

    return shutil.which("git") is not None


def _git(root: str, *args: str) -> str:
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, check=False).stdout.strip()


def pin_mismatches(roots: dict[str, dict[str, int]], pins: dict[str, str]) -> list[str]:
    """Every way the loaded trees differ from the pins (empty: all match)."""
    bad = []
    for pkg, (root_key, sha_key, hash_key) in PIN_KEYS.items():
        if root_key not in pins:
            continue
        want = os.path.realpath(pins[root_key])
        got = sorted(roots.get(pkg, {}))
        if pkg in roots and got != [want]:
            bad.append(f"{pkg} loaded from {got}, pinned {want}")
            continue
        if sha_key and sha_key in pins and _have_git():
            head = _git(want, "rev-parse", "HEAD")
            if head != pins[sha_key]:
                bad.append(f"{pkg} tree {want} is at {head or '?'}, pinned {pins[sha_key]}")
            dirty = _git(want, "status", "--porcelain", "--untracked-files=no")
            if dirty:
                bad.append(f"{pkg} tree {want} has tracked changes: {dirty.splitlines()[:3]}")
        if hash_key and hash_key in pins:
            got_hash = tree_hash(want, pkg)
            if got_hash != pins[hash_key]:
                bad.append(f"{pkg} tree {want} content hash {got_hash[:12]} != pinned {pins[hash_key][:12]}")
    return bad


def check_module_provenance(role: str, modules: Optional[dict] = None) -> dict[str, dict[str, int]]:
    """Print ``[provenance] <role> ...`` (one root per package) and raise if any package is loaded from two roots."""
    mode = os.environ.get("NEMO_RL_PROVENANCE_CHECK", "1")
    if mode == "0":
        return {}
    roots = module_roots(modules)
    parts = [f"{pkg}={','.join(sorted(r))}" for pkg, r in sorted(roots.items())]
    print(f"[provenance] {role} pid={os.getpid()} " + " ".join(parts), flush=True)
    problems = []
    mixed = {pkg: r for pkg, r in roots.items() if len(r) > 1}
    if mixed:
        detail = "; ".join(f"{pkg}: " + ", ".join(f"{root} ({n} modules)" for root, n in sorted(r.items())) for pkg, r in mixed.items())
        problems.append(f"modules of one package loaded from more than one source tree: {detail}")
    pf = pins_file(roots) if modules is None or os.environ.get("NEMO_RL_SNAPSHOT_PINS") else None
    if pf:
        bad = pin_mismatches(roots, read_pins(pf))
        if bad:
            problems.append(f"not the snapshot's pinned trees ({pf}): " + "; ".join(bad))
        else:
            print(f"[provenance] {role}: matches the snapshot pins {pf}", flush=True)
    if problems:
        msg = f"[provenance] {role}: " + " | ".join(problems)
        if mode == "warn":
            print(msg + " (NEMO_RL_PROVENANCE_CHECK=warn: continuing)", flush=True)
        else:
            raise RuntimeError(msg + " (fix PYTHONPATH / the pinned trees; NEMO_RL_PROVENANCE_CHECK=warn to continue)")
    return roots
