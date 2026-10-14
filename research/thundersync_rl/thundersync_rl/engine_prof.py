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
"""Debug: the Megatron dynamic engine's step timing inside the ThunderSync loop.

``THUNDERSYNC_ENGINE_PROF=1``: every ``DynamicInferenceEngine.async_step`` of a
generation rank records (start, end, active rows, prefill rows); the driver
reads a per-step summary (``engine_step_stats``): wall time of decode-only steps
by active-row bucket (ms per token of each row), time inside steps vs between
steps (the engine loop's own overhead: EP consensus, idle sleeps, coordinator
polling) and step counts. ``THUNDERSYNC_ENGINE_TORCHPROF=<first>:<count>``
additionally runs the torch profiler over steps first..first+count-1 of each
rank and writes its kernel table and chrome trace under
``THUNDERSYNC_ENGINE_TORCHPROF_DIR`` (default /tmp).
"""

from __future__ import annotations

import os
import time
from typing import Any

_STEPS: list[tuple[float, float, int, int]] = []
_PROF: dict[str, Any] = {"n": 0, "prof": None}
_BUCKETS = ((1, 1), (2, 4), (5, 16), (17, 64), (65, 128), (129, 1 << 30))


def enabled() -> bool:
    return bool(os.environ.get("THUNDERSYNC_ENGINE_PROF"))


def install() -> None:
    """Wrap ``DynamicInferenceEngine.async_step`` (idempotent, process-wide)."""
    from megatron.core.inference.engines.dynamic_engine import DynamicInferenceEngine

    if getattr(DynamicInferenceEngine, "_ts_step_prof", False):
        return
    orig = DynamicInferenceEngine.async_step
    window = os.environ.get("THUNDERSYNC_ENGINE_TORCHPROF")
    first, count = (int(x) for x in window.split(":")) if window else (-1, 0)

    async def async_step(self):
        ctx = self.context
        rows = int(ctx.total_request_count - ctx.paused_request_count)
        prefill = int(getattr(ctx, "num_prefill_requests", 0))
        _torch_prof(first, count)
        t0 = time.perf_counter()
        ret = await orig(self)
        _STEPS.append((t0, time.perf_counter(), rows, prefill))
        return ret

    DynamicInferenceEngine.async_step = async_step
    DynamicInferenceEngine._ts_step_prof = True


def _torch_prof(first: int, count: int) -> None:
    n = _PROF["n"]
    _PROF["n"] = n + 1
    if first < 0:
        return
    if n == first:
        from torch.profiler import ProfilerActivity, profile

        _PROF["prof"] = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        _PROF["prof"].__enter__()
    elif n == first + count and _PROF["prof"] is not None:
        import torch

        prof, _PROF["prof"] = _PROF["prof"], None
        torch.cuda.synchronize()
        prof.__exit__(None, None, None)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        out = os.environ.get("THUNDERSYNC_ENGINE_TORCHPROF_DIR", "/tmp")
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"engine_rank{rank}_steps{first}-{first + count}.txt"), "w") as f:
            f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
            f.write("\n\n")
            f.write(prof.key_averages().table(sort_by="cpu_time_total", row_limit=40))
        prof.export_chrome_trace(
            os.path.join(out, f"engine_rank{rank}_steps{first}-{first + count}.json")
        )


def summary(reset: bool = True) -> dict[str, float]:
    """Engine step statistics since the last call (this rank)."""
    steps = list(_STEPS)
    if reset:
        _STEPS.clear()
    if not steps:
        return {}
    inside = sum(t1 - t0 for t0, t1, _, _ in steps)
    span = steps[-1][1] - steps[0][0]
    out = {
        "engine/steps": float(len(steps)),
        "engine/span_s": span,
        "engine/in_step_s": inside,
        # Between consecutive steps: loop overhead (consensus, sleeps, polling)
        # plus idle time with no work.
        "engine/between_steps_s": span - inside,
        "engine/prefill_steps": float(sum(1 for *_, p in steps if p > 0)),
    }
    # (step wall, start to next start) of decode-only steps; the last step has no next.
    cycles = [
        (t1 - t0, nxt[0] - t0, r, p) for (t0, t1, r, p), nxt in zip(steps, steps[1:])
    ]
    for lo, hi in _BUCKETS:
        sel = [(w, c) for w, c, r, p in cycles if lo <= r <= hi and p == 0]
        if sel:
            tag = f"{lo}-{hi}" if hi < 1 << 30 else f"{lo}+"
            out[f"engine/decode_ms_rows{tag}"] = 1e3 * sum(x for x, _ in sel) / len(sel)
            # Start-to-next-start: what a row waits per token.
            out[f"engine/cycle_ms_rows{tag}"] = 1e3 * sum(y for _, y in sel) / len(sel)
            out[f"engine/decode_steps_rows{tag}"] = float(len(sel))
    return out
