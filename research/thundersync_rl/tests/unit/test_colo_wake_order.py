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
"""Colocated wake order: the default-group barrier runs before the engine resumes, never after. A resumed engine
steps its paused requests at once (EP collectives); a barrier issued after the resume can interleave with those
collectives differently on different ranks and deadlock NCCL (Nano 8k SPR-BI)."""

import asyncio
import threading
from types import SimpleNamespace

import torch

from nemo_rl.models.generation.megatron import megatron_worker


def test_wake_barrier_precedes_engine_resume():
    events = []
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()

    async def wake_engine():
        events.append("resume")

    worker = SimpleNamespace(
        _inference_engine_asleep=True, _inference_loop=loop, _wake_engine=wake_engine, rank=0
    )
    saved = torch.distributed.barrier
    torch.distributed.barrier = lambda *a, **k: events.append("barrier")
    try:
        megatron_worker.MegatronGenerationMixin._wake(worker)
    finally:
        torch.distributed.barrier = saved
        loop.call_soon_threadsafe(loop.stop)
        thread.join()
    assert events == ["barrier", "resume"], events
    assert worker._inference_engine_asleep is False


if __name__ == "__main__":
    test_wake_barrier_precedes_engine_resume()
    print("PASS test_wake_barrier_precedes_engine_resume")
