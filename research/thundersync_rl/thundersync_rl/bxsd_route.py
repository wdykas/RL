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
"""BX-SD lane routing: honour a per-request target DP rank in Megatron's DP inference coordinator (no MLM edit).

The engine's DataParallelInferenceCoordinator routes every request by load / prefix affinity. BX-SD places rows
(thundersync_rl.bxsd_lane) and puts the target DP rank on the request as the SamplingParams attribute
``ts_dp_rank`` (SamplingParams.serialize() is its __dict__, so the coordinator sees it in the request's
sampling_params dict). This module wraps the SUBMIT_REQUEST handler: a request with ``ts_dp_rank >= 0`` goes to the
engine whose identity is ``mp-coord-<dp_rank>`` (dynamic_engine.start_listening_to_data_parallel_coordinator);
every other request, and every piece of bookkeeping (pending counts, request -> rank map, prefix hashes), is the
stock handler's.

The coordinator runs in a process spawned with the 'spawn' context, so a monkeypatch in the engine process would be
lost. ``install()`` (engine process, before the coordinator starts) replaces the class's ``entrypoint`` with this
module's ``_entrypoint``; Process(target=...) pickles it by reference, the child imports this module (PYTHONPATH is
inherited), patches the handler table there, then runs the stock entrypoint.

Routing changes only WHICH engine replica computes a row; under plain BI every EP group of the bulk layout is
bitwise identical (FP_B_MEASURE §2.5), so outputs are unchanged. A target that names no registered engine falls back
to the stock choice and is counted (``misrouted``), never dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

DP_RANK_ATTR = "ts_dp_rank"
STATS = {"routed": 0, "misrouted": 0, "stock": 0}


def target_of(metadata: Any) -> int:
    """The request's target DP rank, or -1. metadata = [header, client_request_id, sampling_params, media_meta]."""
    try:
        sp = metadata[2]
    except (IndexError, TypeError):
        return -1
    if not isinstance(sp, dict):
        return -1
    v = sp.get(DP_RANK_ATTR, -1)
    return int(v) if v is not None else -1


def identity_of(dp_rank: int) -> bytes:
    return f"mp-coord-{int(dp_rank)}".encode("utf-8")


def routed(orig: Callable) -> Callable:
    """Wrap a SUBMIT_REQUEST handler(coordinator, sender_identity, metadata, bodies)."""

    def handler(coordinator, sender_identity, metadata, bodies):
        tgt = target_of(metadata)
        if tgt < 0:
            STATS["stock"] += 1
            return orig(coordinator, sender_identity, metadata, bodies)
        ident = identity_of(tgt)
        if ident not in coordinator.identity_to_rank_index:
            STATS["misrouted"] += 1
            logging.error("bxsd_route: target DP rank %d has no registered engine; stock routing used", tgt)
            return orig(coordinator, sender_identity, metadata, bodies)
        STATS["routed"] += 1
        # The stock handler picks the rank through get_best_data_parallel_rank (and retries it if a send fails);
        # an instance attribute shadows the method for this one call.
        coordinator.get_best_data_parallel_rank = lambda *_a, **_k: ident
        try:
            return orig(coordinator, sender_identity, metadata, bodies)
        finally:
            del coordinator.get_best_data_parallel_rank

    handler.__wrapped__ = orig
    handler._bxsd_routed = True
    return handler


def patch_handlers() -> None:
    """Coordinator process: wrap SUBMIT_REQUEST in the handler registry (the coordinator copies it at __init__)."""
    from megatron.core.inference.data_parallel_inference_coordinator import handlers as H
    from megatron.core.inference.headers import Headers

    cur = H.HANDLERS[Headers.SUBMIT_REQUEST]
    if not getattr(cur, "_bxsd_routed", False):
        H.HANDLERS[Headers.SUBMIT_REQUEST] = routed(cur)


def _entrypoint(*args, **kwargs):
    """Spawned coordinator process target: patch, then the stock entrypoint."""
    patch_handlers()
    from megatron.core.inference.data_parallel_inference_coordinator.coordinator import (
        DataParallelInferenceCoordinator,
    )

    ep = DataParallelInferenceCoordinator.__dict__["entrypoint"]
    fn = ep.__func__ if isinstance(ep, (classmethod, staticmethod)) else ep
    if fn is _entrypoint:
        raise RuntimeError("bxsd_route: the child sees the patched entrypoint (fork start method?)")
    return DataParallelInferenceCoordinator.entrypoint(*args, **kwargs)


def install() -> None:
    """Engine process (the DP coordinator rank), before start_listening_to_data_parallel_coordinator."""
    from megatron.core.inference.data_parallel_inference_coordinator.coordinator import (
        DataParallelInferenceCoordinator,
    )

    if DataParallelInferenceCoordinator.__dict__.get("_bxsd_orig_entrypoint") is None:
        DataParallelInferenceCoordinator._bxsd_orig_entrypoint = DataParallelInferenceCoordinator.__dict__["entrypoint"]
        DataParallelInferenceCoordinator.entrypoint = staticmethod(_entrypoint)
