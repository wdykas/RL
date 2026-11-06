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
"""Driver environment from the config (``thundersync.driver_env``), so one config file gives the paper behavior.

Several ThunderSync switches are environment variables read in the driver process (sentinel cadence, early
handout, verify0 hold / no-interleave, the check schedule, ...). ``thundersync.driver_env`` lists them in the config;
``apply`` sets them in the driver's ``os.environ`` at startup, before Ray, setup or any loop code reads them (every
reader is call-time, none at import). Worker processes get their variables through the existing
``policy.megatron_cfg.env_vars``; the config lists those separately, as the env-var launch does.

A variable that the launching shell already set to a different value is an error: the two launch routes must not
silently disagree. The same value is accepted. ``effective`` returns the ThunderSync / NeMo-RL variables of a process,
printed by the driver in both routes (config and env) so a run log shows the environment it ran with.
"""

from __future__ import annotations

import os
from typing import Mapping, MutableMapping, Optional

PREFIXES = ("THUNDERSYNC_", "NEMO_RL_")
# Per-run or harness variables that are not part of a configuration (run directory, interpreter selection).
NOT_CONFIG = ("THUNDERSYNC_RUN_DIR", "NEMO_RL_PY_EXECUTABLES_SYSTEM", "NEMO_RL_SNAPSHOT_PINS")


def apply(env: Optional[Mapping[str, object]], environ: Optional[MutableMapping[str, str]] = None) -> dict[str, str]:
    """Set ``env`` in ``environ`` (default ``os.environ``); refuse a conflicting value already set. Returns what was set."""
    environ = os.environ if environ is None else environ
    want = {str(k): str(v) for k, v in (env or {}).items()}
    conflicts = {k: (environ[k], v) for k, v in want.items() if k in environ and environ[k] != v}
    if conflicts:
        raise ValueError(
            "thundersync.driver_env conflicts with the launching environment (unset one of them): "
            + ", ".join(f"{k}: env {a!r} vs config {b!r}" for k, (a, b) in sorted(conflicts.items()))
        )
    for k, v in want.items():
        environ[k] = v
    return want


def effective(environ: Optional[Mapping[str, str]] = None) -> dict[str, str]:
    """The ThunderSync / NeMo-RL configuration variables of this process, sorted."""
    environ = os.environ if environ is None else environ
    return {k: environ[k] for k in sorted(environ) if k.startswith(PREFIXES) and k not in NOT_CONFIG}
