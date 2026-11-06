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
"""Colocated wake / sleep ordering across ranks (regression for the post-wake illegal memory access).

Rank 0 drives the engines through the coordinator: its resume / unpause reaches EVERY rank's engine. Each rank's
local wake (inference weights onloaded and resharded, engine scratch mapped back) runs on that rank's main thread.
If rank 0 resumes the engines before every rank finished its local wake, a slower rank's engine steps on unmapped
engine scratch (TMS) or on weights still being written: Nano 8k SPR-BI on cdfeb3ad / 514231f7 died with a Warp MMU
Fault in fp16_moe_dk's _dk_count_kernel storing to the first bytes of the engine-scratch region
(core dump), 0.2-0.6 s after a wake, on the slower node's ranks. The fix (5a96a507) puts the default-group barrier
between the local wake and the coordinator resume.

Two ranks run the real ``MegatronGenerationMixin._wake`` / ``_sleep`` on their own main and engine-loop threads, with
a shared fake coordinator, a threading barrier for ``torch.distributed.barrier`` and rank 1's local wake delayed.
Each fake engine records a violation when it is resumed while its rank's scratch is still unmapped or its rank's
inference weights are not resharded yet, or when its rank's scratch is unmapped while it is not suspended. The
weights case drives the real ``prepare_for_generation`` (onload + reshard, then ``_wake``) with the reshard stubbed,
so a refactor that moves the reshard past the barrier fails here too.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

import nemo_rl.models.generation.megatron.megatron_worker as mw
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore

WORLD = 2


class _Cluster:
    """Per-rank scratch state, thread -> rank map, the engines, the barrier and the recorded violations."""

    def __init__(self):
        self.scratch_mapped = {r: False for r in range(WORLD)}
        self.weights_ready = {r: True for r in range(WORLD)}
        self.thread_rank: dict[int, int] = {}
        self.engines: dict[int, "_Engine"] = {}
        self.barrier = threading.Barrier(WORLD)
        self.violations: list[str] = []
        self.lock = threading.Lock()

    def rank(self) -> int:
        return self.thread_rank[threading.get_ident()]

    def violation(self, msg: str) -> None:
        with self.lock:
            self.violations.append(msg)


class _Engine:
    """Engine state as the coordinator drives it; its events are set on its own loop."""

    def __init__(self, cluster: _Cluster, rank: int, loop: asyncio.AbstractEventLoop):
        self.c, self.rank, self.loop = cluster, rank, loop
        self.state = mw.EngineState.SUSPENDED
        self.events: dict = {}

    def _event(self, state):
        if state not in self.events:
            self.events[state] = asyncio.Event()
        return self.events[state]

    # MCore's event semantics (DynamicInferenceEngine signal handling): PAUSE clears RUNNING, SUSPEND clears RESUMED,
    # RESUME clears SUSPENDED, UNPAUSE clears PAUSED; every other event stays set (a rank may wait for PAUSED after
    # rank 0 already sent the suspend, or for RESUMED after the unpause).
    _CLEARS = {
        "RESUMED": ("SUSPENDED",),
        "RUNNING": ("PAUSED",),
        "PAUSED": ("RUNNING",),
        "SUSPENDED": ("RESUMED",),
    }

    def _set(self, state):
        def go():
            self.state = state
            for name in self._CLEARS[state.name]:
                self._event(getattr(mw.EngineState, name)).clear()
            self._event(state).set()

        self.loop.call_soon_threadsafe(go)

    # Coordinator commands (broadcast to every engine).
    def _check_local_wake(self, what):
        if not self.c.scratch_mapped[self.rank]:
            self.c.violation(f"rank {self.rank}: engine {what} while its engine scratch is unmapped")
        if not self.c.weights_ready[self.rank]:
            self.c.violation(f"rank {self.rank}: engine {what} while its inference weights are being resharded")

    def resume(self):
        self._check_local_wake("resumed")
        self._set(mw.EngineState.RESUMED)

    def unpause(self):
        self._check_local_wake("runs")
        self._set(mw.EngineState.RUNNING)

    def pause(self):
        self._set(mw.EngineState.PAUSED)

    def suspend(self):
        self._set(mw.EngineState.SUSPENDED)

    async def wait_until(self, state):
        await self._event(state).wait()


class _Worker(MegatronGenerationMixin):
    def __init__(self, cluster: _Cluster, rank: int):
        self.c, self.rank = cluster, rank
        self._inference_engine_asleep = True
        self._inference_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        while self._thread.ident is None:
            time.sleep(0.001)
        cluster.thread_rank[self._thread.ident] = rank
        self.dynamic_inference_engine = _Engine(cluster, rank, self._inference_loop)
        cluster.engines[rank] = self.dynamic_inference_engine
        engines = cluster.engines
        self.inference_client = SimpleNamespace(
            resume_engines=lambda: [e.resume() for e in engines.values()],
            unpause_engines=lambda: [e.unpause() for e in engines.values()],
            pause_engines=lambda: [e.pause() for e in engines.values()],
            suspend_engines=lambda: [e.suspend() for e in engines.values()],
        )

    def _run_loop(self):
        asyncio.set_event_loop(self._inference_loop)
        self._inference_loop.run_forever()

    async def _client_send(self, fn) -> None:
        fn()


@pytest.fixture
def cluster(monkeypatch):
    c = _Cluster()

    def resume_engine_scratch():
        c.scratch_mapped[c.rank()] = True

    def pause_engine_scratch():
        r = c.rank()
        if c.engines[r].state != mw.EngineState.SUSPENDED:
            c.violation(f"rank {r}: engine scratch unmapped while its engine is {c.engines[r].state.name}")
        c.scratch_mapped[r] = False

    monkeypatch.setattr(mw, "resume_engine_scratch", resume_engine_scratch)
    monkeypatch.setattr(mw, "pause_engine_scratch", pause_engine_scratch)
    monkeypatch.setattr(mw.torch.distributed, "barrier", lambda *a, **k: c.barrier.wait(timeout=20))
    monkeypatch.setattr(mw.torch.distributed, "get_rank", lambda *a, **k: c.rank())
    return c


def _run_ranks(cluster, workers, fn, delay_rank: int, delay_s: float):
    errors = []

    def main(w):
        cluster.thread_rank[threading.get_ident()] = w.rank
        try:
            if w.rank == delay_rank:
                time.sleep(delay_s)  # this rank's local work (onload, reshard, move_model) is slower
            fn(w)
        except BaseException as e:  # noqa: BLE001 - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=main, args=(w,)) for w in workers]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a rank hung"
    assert not errors, errors


@pytest.mark.parametrize("slow_rank", [0, 1])
def test_no_engine_resumes_before_every_rank_finished_its_local_wake(cluster, slow_rank):
    workers = [_Worker(cluster, r) for r in range(WORLD)]
    _run_ranks(cluster, workers, lambda w: w._wake(), delay_rank=slow_rank, delay_s=0.3)
    assert cluster.violations == []
    assert all(e.state == mw.EngineState.RUNNING for e in cluster.engines.values())
    assert not any(w._inference_engine_asleep for w in workers)


@pytest.mark.parametrize("slow_rank", [0, 1])
def test_no_rank_unmaps_its_scratch_before_its_engine_is_suspended(cluster, slow_rank):
    workers = [_Worker(cluster, r) for r in range(WORLD)]
    for w in workers:
        w._inference_engine_asleep = False
        cluster.scratch_mapped[w.rank] = True
        w.dynamic_inference_engine._set(mw.EngineState.RUNNING)
    time.sleep(0.05)
    _run_ranks(cluster, workers, lambda w: w._sleep(), delay_rank=slow_rank, delay_s=0.3)
    assert cluster.violations == []
    assert all(e.state == mw.EngineState.SUSPENDED for e in cluster.engines.values())


class _InferenceModel:
    config = SimpleNamespace(flash_decode=True)

    def eval(self):
        return self


@pytest.mark.parametrize("slow_rank", [0, 1])
def test_no_engine_resumes_before_every_rank_resharded_its_weights(cluster, monkeypatch, slow_rank):
    """``prepare_for_generation`` reshards, then wakes: no engine may resume while any rank is still resharding."""
    monkeypatch.setattr(mw, "log_gpu_memory", lambda *a, **k: None)
    workers = [_Worker(cluster, r) for r in range(WORLD)]
    model = _InferenceModel()
    for w in workers:
        cluster.weights_ready[w.rank] = False
        w._inference_engine_initialized = True
        w.cfg = {"generation": {"mcore_generation_config": {"cuda_graph_impl": "none"}}}
        w._colocated_reshard_plan = None
        w.is_generation_colocated = True
        w.inference_model = model
        w._inference_model_and_media_parts = lambda m=model: (m, None)

        def reshard(w=w):
            if w.rank == slow_rank:
                time.sleep(0.3)  # this rank's onload / reshard is slower
            cluster.weights_ready[w.rank] = True

        w._reshard_into_inference_model = reshard
    _run_ranks(cluster, workers, lambda w: w.prepare_for_generation(), delay_rank=-1, delay_s=0.0)
    assert cluster.violations == []
    assert all(e.state == mw.EngineState.RUNNING for e in cluster.engines.values())
