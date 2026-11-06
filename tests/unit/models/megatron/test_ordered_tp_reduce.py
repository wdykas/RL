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
"""Ordered TP reduce-scatter (fp16_gemm.ordered_reduce_scatter_rows): CPU, gloo, 2-4 ranks.

A row's sum must be the rank-order left fold of its partials, bit for bit, whichever rank
owns the row and whatever the row count (the batch-invariance property the learner and the
engine rely on at TP >= 3).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_rl.models.megatron import fp16_gemm as fg


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        bad = 0
        for t, shift in ((4 * world, 0), (4 * world, 4), (12 * world, 4 * world + 3)):
            g = torch.Generator().manual_seed(7)
            parts = torch.randn(world, t, 5, generator=g) * torch.exp2(
                torch.randint(-20, 20, (world, t, 1), generator=g).float()
            )
            order = torch.roll(torch.arange(t), shift)
            y = fg.ordered_reduce_scatter_rows(parts[rank, order].contiguous(), dist.group.WORLD)
            full = [torch.empty_like(y) for _ in range(world)]
            dist.all_gather(full, y)
            full = torch.cat(full)
            back = torch.empty_like(full)
            back[order] = full
            ref = parts[0] + parts[1]
            for k in range(2, world):
                ref = ref + parts[k]
            bad += int((back.view(torch.int32) != ref.view(torch.int32)).sum())
        # ordered all-reduce == left fold everywhere
        os.environ["NEMO_RL_TP_REDUCE"] = "ordered"
        g = torch.Generator().manual_seed(9)
        parts = torch.randn(world, 2 * world, 3, generator=g)
        out = fg._all_reduce(parts[rank].clone(), dist.group.WORLD)
        ref = parts[0] + parts[1]
        for k in range(2, world):
            ref = ref + parts[k]
        bad += int((out.view(torch.int32) != ref.view(torch.int32)).sum())
        q.put((rank, bad))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 3, 4])
def test_ordered_reduce_scatter_is_rank_order_left_fold(world):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=120) for _ in range(world))
    for p in procs:
        p.join(timeout=60)
    assert res == {r: 0 for r in range(world)}


def test_tp_reduce_mode(monkeypatch):
    monkeypatch.delenv("NEMO_RL_TP_REDUCE", raising=False)
    assert fg.tp_reduce_mode(2) == "stock"
    assert fg.tp_reduce_mode(3) == "ordered"
    assert fg.tp_reduce_mode(8) == "ordered"
    monkeypatch.setenv("NEMO_RL_TP_REDUCE", "stock")
    assert fg.tp_reduce_mode(8) == "stock"
    monkeypatch.setenv("NEMO_RL_TP_REDUCE", "ordered")
    assert fg.tp_reduce_mode(2) == "ordered"
    monkeypatch.setenv("NEMO_RL_TP_REDUCE", "bogus")
    with pytest.raises(ValueError):
        fg.tp_reduce_mode(4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU (Triton fold kernel)")
@pytest.mark.parametrize("tp", [2, 3, 4, 8])
def test_fused_fold_matches_eager_left_fold(tp, monkeypatch):
    g = torch.Generator(device="cuda").manual_seed(tp)
    for rows, cols in ((1, 7), (8, 4096), (333, 2688), (2048, 1024)):
        parts = torch.randn(tp, rows, cols, device="cuda", generator=g) * torch.exp2(
            torch.randint(-30, 30, (tp, rows, 1), device="cuda", generator=g).float()
        )
        monkeypatch.setenv("NEMO_RL_TP_REDUCE_FUSED", "1")
        fused = fg._fold_rank_order(parts)
        monkeypatch.setenv("NEMO_RL_TP_REDUCE_FUSED", "0")
        eager = fg._fold_rank_order(parts)
        assert torch.equal(fused.view(torch.int32), eager.view(torch.int32))
