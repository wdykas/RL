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
"""A generation request fails instead of hanging when the engine loop died (source: rl-determinism 90e78222).

The MCore engine loop exits on a fatal step error (e.g. an illegal memory access in a graph replay) without
resolving in-flight request futures; ``generate`` / ``generate_async`` waited on them forever."""

import asyncio
import concurrent.futures
import threading
import time
from types import SimpleNamespace

import pytest

import nemo_rl.models.generation.megatron.megatron_worker as mw
from nemo_rl.models.generation.megatron.megatron_worker import MegatronGenerationMixin

pytestmark = pytest.mark.mcore


class _Worker(MegatronGenerationMixin):
    def __init__(self, loop, loop_thread, engine_task):
        self.rank = 0
        self._inference_loop = loop
        self._inference_thread = loop_thread
        self.dynamic_inference_engine = SimpleNamespace(engine_loop_task=engine_task)

    def _prepare_data_for_generation(self, data, greedy):
        return [], [], None

    async def _generate_with_persistent_engine(self, prompts, mm, sampling):
        await asyncio.Event().wait()  # a request the dead engine never resolves

    def _parse_result_to_batched_data_dict(self, data, result):
        return result


def _loop_with_task(engine_dies: bool):
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def engine():
        if engine_dies:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")
        await asyncio.Event().wait()

    task = asyncio.run_coroutine_threadsafe(_create(loop, engine), loop).result()
    return loop, thread, task


async def _create(loop, engine):
    return loop.create_task(engine())


@pytest.fixture(autouse=True)
def _fast_poll(monkeypatch):
    monkeypatch.setattr(mw, "_ENGINE_LIVENESS_POLL_S", 0.2, raising=False)  # raising=False: also runs on unfixed trees


def test_dead_engine_detected_and_alive_engine_not():
    loop, thread, task = _loop_with_task(engine_dies=True)
    time.sleep(0.2)
    w = _Worker(loop, thread, task)
    fut = concurrent.futures.Future()
    with pytest.raises(RuntimeError, match="engine loop exited"):
        w._raise_if_inference_engine_dead(fut)
    assert fut.cancelled()
    loop2, thread2, task2 = _loop_with_task(engine_dies=False)
    _Worker(loop2, thread2, task2)._raise_if_inference_engine_dead(concurrent.futures.Future())  # no raise
    for lp in (loop, loop2):
        lp.call_soon_threadsafe(lp.stop)


def test_generate_fails_instead_of_hanging_when_the_engine_died():
    loop, thread, task = _loop_with_task(engine_dies=True)
    w = _Worker(loop, thread, task)
    box = {}

    def call():
        try:
            w.generate(data=None)
        except RuntimeError as exc:
            box["err"] = exc

    t = threading.Thread(target=call, daemon=True)
    t.start()
    t.join(15)
    assert not t.is_alive(), "generate() still waiting on a dead engine"
    assert "cannot complete" in str(box.get("err"))
    loop.call_soon_threadsafe(loop.stop)


def test_generate_async_fails_instead_of_hanging_when_the_engine_died():
    loop, thread, task = _loop_with_task(engine_dies=True)
    w = _Worker(loop, thread, task)
    data = SimpleNamespace(size=1, get_batch=lambda index, n: None)

    async def consume():
        async for _ in w.generate_async(data):
            pass

    with pytest.raises(RuntimeError, match="cannot complete"):
        asyncio.run(asyncio.wait_for(consume(), timeout=15))
    loop.call_soon_threadsafe(loop.stop)
