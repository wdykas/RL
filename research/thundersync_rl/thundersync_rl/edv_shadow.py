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
"""EDV shadow counter (DESIGN §22.5, step 1): one jsonl line per decided ⊥-BV draft, always on, log only.

Per draft decided at envelope M (the pool's counts): the row's final coin eta_g, b_g at M (and at every envelope the
row was scanned at), tau, the emitted kind (whole / continued / bot) and the row's per-token log p/q moments
(n, sum, sum of squares). Would-be EDV surprises at a threshold e are the rows with eta_g <= e and tau < g.

Nothing here feeds back: the record is built from values the verification already produced (``ds2.take_last_row``:
host copies of eta and b; ``null_lr_moments``), after the decision, and is only written to a file. The driver writes
``<run dir>/edv_shadow.jsonl`` (``run_dir_of`` the logger's base_log_dir), buffered, flushed at each ``set_step``.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

_PATH: Optional[str] = None
_STEP: int = -1
_BUF: list[str] = []
COUNTS = {"rows": 0, "missing": 0}


def configure(log_dir: Optional[str]) -> Optional[str]:
    """Set the jsonl path from logger.log_dir (``<run dir>/logs``); ``THUNDERSYNC_EDV_SHADOW_PATH`` overrides."""
    global _PATH
    env = os.environ.get("THUNDERSYNC_EDV_SHADOW_PATH", "").strip()
    if env:
        _PATH = env
    elif log_dir:
        _PATH = os.path.join(run_dir_of(str(log_dir)), "edv_shadow.jsonl")
    else:
        _PATH = None
    return _PATH


def run_dir_of(log_dir: str) -> str:
    """The run dir of a logger dir: ``<run>/logs`` or ``<run>/logs/exp_NNN`` -> ``<run>``; otherwise its parent."""
    d = os.path.abspath(log_dir.rstrip("/"))
    if os.path.basename(d).startswith("exp_"):
        d = os.path.dirname(d)
    return os.path.dirname(d)


def set_step(step: int) -> None:
    """Flush the previous step's lines and tag the next ones with ``step``."""
    global _STEP
    flush()
    _STEP = int(step)


def flush() -> None:
    if not _BUF or _PATH is None:
        _BUF.clear()
        return
    with open(_PATH, "a") as f:
        f.write("".join(_BUF))
    _BUF.clear()


def row_record(res: dict[str, Any], m: float) -> Optional[dict[str, Any]]:
    """The jsonl record of a draft decided at envelope ``m`` (None without a DS2 shadow record)."""
    sh = res.get("edv_shadow")
    if sh is None:
        return None
    tau, y, _ = res["null_grid"][m]
    g = int(sh["g"])
    kind = "bot" if y < 0 else ("whole" if tau == g else "continued")
    n, s1, s2 = res.get("null_lr_moments", (0, 0.0, 0.0))
    return {
        "step": _STEP,
        "key": sh.get("key"),
        "g": g,
        "m": float(m),
        "eta_g": float(sh["eta_g"]),
        "b_g": float(sh["b_g"][float(m)]),
        "b_g_env": {str(k): float(v) for k, v in sh["b_g"].items()},
        "tau": int(tau),
        "kind": kind,
        "whole": kind == "whole",
        "log_wstar": float(res.get("null_log_wstar", 0.0)),
        "lr_n": int(n),
        "lr_sum": float(s1),
        "lr_sumsq": float(s2),
        "p_zero": int(res.get("p_zero_tokens", 0)),
    }


def record(res: dict[str, Any], m: float) -> None:
    """Buffer the record of a decided draft (log only)."""
    rec = row_record(res, m)
    if rec is None:
        COUNTS["missing"] += 1
        return
    COUNTS["rows"] += 1
    if _PATH is not None:
        _BUF.append(json.dumps(rec, separators=(",", ":")) + "\n")
