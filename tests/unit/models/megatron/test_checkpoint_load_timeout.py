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
"""setup._checkpoint_load_collective_timeout: a broadcast whose peer is late by more
than the group's NCCL timeout (one rank still "reading its checkpoint share") survives
inside the load window and times out outside it; previous timeouts are restored."""

import datetime
import os
import socket
import time

import pytest
import torch
import torch.multiprocessing as mp

LATE_S = 12.0  # the slow rank's extra "storage read" time
GROUP_TIMEOUT = datetime.timedelta(seconds=4)

needs_2_gpus = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs"
)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank: int, port: int, use_window: bool, out) -> None:
    import torch.distributed as dist

    from nemo_rl.models.megatron.setup import _checkpoint_load_collective_timeout

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=2,
        init_method=f"tcp://127.0.0.1:{port}",
        timeout=GROUP_TIMEOUT,
        device_id=torch.device("cuda", rank),
    )
    dp = dist.new_group([0, 1], timeout=GROUP_TIMEOUT)
    x = torch.full((1024,), float(rank), device="cuda")
    dist.all_reduce(x, group=dp)  # metadata exchange stand-in: both ranks join
    torch.cuda.synchronize()
    ok = True
    try:
        window = (
            _checkpoint_load_collective_timeout(0.5)  # 30 s > LATE_S
            if use_window
            else _checkpoint_load_collective_timeout(None)
        )
        with window:
            if rank == 1:
                time.sleep(LATE_S)  # slow storage read
            work = dist.broadcast(x, src=0, group=dp, async_op=True)
            work.wait()
            torch.cuda.synchronize()
            dist.barrier()
        timeout_after = dp._get_backend(torch.device("cuda")).options._timeout
        ok = ok and timeout_after == GROUP_TIMEOUT
    except Exception:
        ok = False
    out.put((rank, ok))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def _run(use_window: bool) -> dict:
    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, port, use_window, out)) for r in (0, 1)]
    for p in procs:
        p.start()
    results = {}
    deadline = time.time() + 120
    while len(results) < 2 and time.time() < deadline:
        try:
            rank, ok = out.get(timeout=5)
            results[rank] = ok
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    return results


@needs_2_gpus
def test_late_rank_survives_inside_the_load_window():
    assert _run(use_window=True) == {0: True, 1: True}


@needs_2_gpus
def test_late_rank_times_out_without_the_window():
    """Non-vacuity: the same schedule fails at the group's own timeout."""
    results = _run(use_window=False)
    assert results.get(0) is not True
