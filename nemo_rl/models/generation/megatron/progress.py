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
"""Generation progress log and stall watchdog for one Megatron dynamic-engine rank.

A daemon thread samples plain Python counters of this rank's ``DynamicInferenceEngine``
every ``interval_s`` seconds: no CUDA synchronization, no collective, O(1) per sample,
so the cost does not grow with the number of ranks or requests.

- Progress line (only on ranks with ``log=True``, by default global rank 0): active /
  paused / waiting / finished requests, generated tokens and tokens/s of THIS engine rank.
  Printed while the rank has work and once more after it drains.
- Stall watchdog (every rank, ``stall_timeout_s > 0``): while the engine is RUNNING with
  requests outstanding, if neither the finished-request count nor the generated-token
  count grows for ``stall_timeout_s``, dump every thread's stack (faulthandler) and exit
  the process non-zero, so a hung engine fails the job instead of idling the allocation.
  Each rank decides from its own counters; a collective hang freezes every rank holding
  requests, so all of them dump and exit.
"""

import faulthandler
import os
import sys
import threading
import time
from datetime import datetime
from typing import Any, Callable, Optional

STALL_EXIT_CODE = 75


def with_generated_token_counter(engine_cls: type) -> type:
    """``engine_cls`` (an MCore DynamicInferenceEngine) counting generated tokens.

    ``generated_token_count`` grows by one per request processed in a step (its sampled
    token), read from the length of the CPU-side id list ``post_process_requests``
    receives; no device access. Speculative (MTP) extra tokens are not counted.
    """

    class _CountingEngine(engine_cls):  # type: ignore[misc, valid-type]
        generated_token_count = 0

        def post_process_requests(self, request_ids, *args, **kwargs):
            self.generated_token_count += len(request_ids)
            return super().post_process_requests(request_ids, *args, **kwargs)

    _CountingEngine.__name__ = engine_cls.__name__
    _CountingEngine.__qualname__ = engine_cls.__qualname__
    return _CountingEngine


def _fmt(n: float) -> str:
    for div, unit in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"


class GenerationProgressMonitor:
    """See the module docstring. ``start()`` launches the daemon thread."""

    def __init__(
        self,
        engine: Any,
        rank: int,
        *,
        interval_s: float = 60.0,
        stall_timeout_s: float = 0.0,
        log: bool = True,
        clock: Callable[[], float] = time.monotonic,
        on_stall: Optional[Callable[[str], None]] = None,
    ):
        self.engine = engine
        self.rank = rank
        self.interval_s = interval_s
        self.stall_timeout_s = stall_timeout_s
        self.log = log
        self.clock = clock
        self.on_stall = on_stall or self._dump_and_exit
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last: Optional[tuple[float, int, int]] = None  # (t, finished, tokens)
        self._last_progress_t: Optional[float] = None
        self._drained_logged = True

    def counters(self) -> dict:
        """Snapshot of this rank's engine counters (plain attribute reads)."""
        e = self.engine
        ctx = e.context
        total = int(getattr(ctx, "total_request_count", 0))
        paused = int(getattr(ctx, "paused_request_count", 0))
        return {
            "requests": len(e.requests),
            "active": total - paused,
            "paused": paused,
            "waiting": len(e.waiting_request_ids),
            "finished": int(e.finished_request_count),
            "tokens": int(getattr(e, "generated_token_count", 0)),
            "steps": int(getattr(ctx, "step_count", 0)),
            "state": getattr(getattr(e, "state", None), "name", "?"),
        }

    def _busy(self, c: dict) -> bool:
        return c["requests"] > 0 and c["state"] == "RUNNING"

    def tick(self) -> Optional[str]:
        """One sample: returns the progress line printed (or None); may call on_stall."""
        now = self.clock()
        c = self.counters()
        if self._last is None:
            self._last = (now, c["finished"], c["tokens"])
            self._last_progress_t = now
            return None
        t0, f0, k0 = self._last
        # The engine's finished count restarts at 0 when it is reset; never go negative.
        d_fin = c["finished"] - f0 if c["finished"] >= f0 else c["finished"]
        d_tok = c["tokens"] - k0 if c["tokens"] >= k0 else c["tokens"]
        self._last = (now, c["finished"], c["tokens"])
        busy = self._busy(c)
        if d_fin > 0 or d_tok > 0 or not busy:
            self._last_progress_t = now
        line = None
        if self.log and (busy or not self._drained_logged):
            dt = max(now - t0, 1e-9)
            line = (
                f"[gen progress] rank {self.rank} | {datetime.now():%H:%M:%S} | "
                f"active {c['active']} (paused {c['paused']}) waiting {c['waiting']} | "
                f"finished {c['finished']} (+{d_fin}) | tokens {_fmt(c['tokens'])} "
                f"(+{_fmt(d_tok)}) | {d_tok / dt:.0f} tok/s | steps {c['steps']} | {c['state']}"
            )
            print(line, flush=True)
            self._drained_logged = not busy
        if (
            busy
            and self.stall_timeout_s > 0
            and now - self._last_progress_t >= self.stall_timeout_s
        ):
            self.on_stall(
                f"[gen watchdog] rank {self.rank}: no request finished and no token generated "
                f"for {now - self._last_progress_t:.0f} s with {c['requests']} requests "
                f"outstanding ({c})"
            )
        return line

    def _dump_and_exit(self, msg: str) -> None:
        print(msg + "; dumping all thread stacks and exiting", file=sys.stderr, flush=True)
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        sys.stdout.flush()
        os._exit(STALL_EXIT_CODE)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            self.tick()

    def start(self) -> "GenerationProgressMonitor":
        self._thread = threading.Thread(
            target=self._run, name="gen-progress", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
