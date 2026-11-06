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
"""Learner weight-sentinel digest off the training path (``THUNDERSYNC_SENTINEL_ASYNC=1``, opt-in).

The synchronous digest (``sentinel_weight_digest``) is one learner actor call: it waits for the GPU queue and
digests every parameter shard while no training dispatch can start (gate 2: 2.4 s per refit step on the learner).
Here the learner call only records a CUDA event on the current stream (after the refit's parameter read and every
earlier write) and hands the digest to one worker thread that runs it on a side stream ordered after that event.
Training forwards / backwards only read parameters, so they run meanwhile. Every parameter write
(optimizer step, draft-weight load / restore) first joins all digests in flight (``join_before_write``), so a
digest always reads the weights of the refit it checks. The driver collects the result later (``result``) and
compares it exactly as before: same digests, same comparison, the verdict arrives up to one step later.

Per rank: one thread, one stream, transient memory O(``chunk``) elements (int64 temporaries; 2**22 by default,
~100 MB), O(parameters / rank) reads, no collectives. Independent of model size, TP / PP / EP.
"""

from __future__ import annotations

import asyncio
import os
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import torch

ENV = "THUNDERSYNC_SENTINEL_ASYNC"
ASYNC_CHUNK = 1 << 22


def enabled() -> bool:
    return os.environ.get(ENV, "0") == "1"


class _State:
    def __init__(self) -> None:
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="sentinel-digest")
        self.stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        self.inflight: dict[int, Future] = {}


def _state(worker: Any) -> _State:
    st = worker.__dict__.get("_sentinel_async")
    if st is None:
        st = worker.__dict__["_sentinel_async"] = _State()
    return st


def begin(worker: Any, token: int, fn: Callable[[], Any]) -> None:
    """Start ``fn`` (a digest of this rank's live weights) on the side stream, ordered after all work queued on the
    current stream now. Digests run one at a time, in begin order."""
    st = _state(worker)
    if token in st.inflight:
        raise RuntimeError(f"sentinel digest {token} already in flight")
    ev = None
    dev = None
    if st.stream is not None:
        dev = torch.cuda.current_device()
        ev = torch.cuda.Event()
        ev.record()

    def run() -> tuple[Any, float]:
        t0 = time.perf_counter()
        if st.stream is None:
            return fn(), time.perf_counter() - t0
        torch.cuda.set_device(dev)
        with torch.cuda.stream(st.stream):
            st.stream.wait_event(ev)
            out = fn()
            st.stream.synchronize()
        return out, time.perf_counter() - t0

    st.inflight[token] = st.pool.submit(run)


def join_before_write(worker: Any, what: str) -> float:
    """Wait for every digest in flight (they read the parameters ``what`` is about to write). Results stay stored
    for the driver; a digest's exception is raised by ``result``, not here. Returns the wait (s)."""
    st = worker.__dict__.get("_sentinel_async")
    if st is None or not st.inflight:
        return 0.0
    t0 = time.perf_counter()
    for f in list(st.inflight.values()):
        try:
            f.result()
        except Exception:  # noqa: BLE001 - reported to the driver by result()
            pass
    waited = time.perf_counter() - t0
    worker.__dict__["_sentinel_async_join_s"] = worker.__dict__.get("_sentinel_async_join_s", 0.0) + waited
    return waited


async def result(worker: Any, token: int) -> tuple[Any, float, float]:
    """(digest, digest thread time s, time parameter writes waited for digests s) of digest ``token``; raises what
    the digest raised. Does not block the actor's event loop."""
    st = _state(worker)
    fut = st.inflight.get(token)
    if fut is None:
        raise RuntimeError(f"no sentinel digest {token} on this worker")
    try:
        out, secs = await asyncio.wrap_future(fut)
    finally:
        st.inflight.pop(token, None)
    waited = worker.__dict__.pop("_sentinel_async_join_s", 0.0)
    return out, secs, waited
