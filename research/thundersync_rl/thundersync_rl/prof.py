"""Opt-in phase timers for verification profiling (``THUNDERSYNC_SPEC_PROF=1``).

``timed(name)`` synchronizes the current CUDA device around the block (so GPU
work is charged to the phase that launched it) and accumulates wall seconds in
``STATS``; with the variable unset it is a no-op. Phases must not nest unless
their names say so (``q_rows/...`` is a breakdown of ``q_rows``).
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from contextlib import contextmanager

import torch

STATS: defaultdict[str, float] = defaultdict(float)


def enabled() -> bool:
    return bool(os.environ.get("THUNDERSYNC_SPEC_PROF"))


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


@contextmanager
def timed(name: str):
    if not enabled():
        yield
        return
    _sync()
    t = time.perf_counter()
    try:
        yield
    finally:
        _sync()
        STATS[name] += time.perf_counter() - t


def pop() -> dict[str, float]:
    out = {k: round(v, 4) for k, v in STATS.items()}
    STATS.clear()
    return out
