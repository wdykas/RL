"""On/off environment flags.

``bool(os.environ.get(name))`` treats every non-empty string as on, so ``NAME=0`` turned a flag ON (e.g.
THUNDERSYNC_SPEC_PROF=0 enabled the profiler and its per-call device syncs). ``env_flag`` reads "", "0", "false",
"no" and "off" (any case, surrounding spaces ignored) as off and any other value as on; unset gives ``default``.
"""

from __future__ import annotations

import os

_OFF = frozenset(("", "0", "false", "no", "off"))


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in _OFF
