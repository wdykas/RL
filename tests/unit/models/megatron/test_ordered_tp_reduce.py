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
"""Ordered TP reductions (CPU, gloo): rank-order fp32 sums, token-count invariant."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("megatron.core")

from nemo_rl.models.megatron import ordered_tp_reduce as otr  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _partials(world: int, tokens: int, hidden: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    return torch.randn(world, tokens, hidden, generator=g).to(torch.bfloat16)


def _reference(parts: torch.Tensor) -> torch.Tensor:
    acc = parts[0].float()
    for r in range(1, parts.shape[0]):
        acc = acc + parts[r].float()
    return acc.to(parts.dtype)


def _worker(rank: int, world: int, port: int, errors) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group("gloo", rank=rank, world_size=world)
        group = dist.new_group(list(range(world)))
        otr.install()
        otr._ordered_groups = lambda: [group]  # stand-in for the TP group

        hidden, pool = 8, 16 * world
        parts = _partials(world, pool, hidden)
        ref = _reference(parts)
        mine = parts[rank]

        # Reduce-scatter through the patched torch.distributed entry point, for
        # several token counts: a token's bits never depend on the count.
        for tokens in (world, 4 * world, pool):
            out = torch.empty(tokens // world, hidden, dtype=torch.bfloat16)
            work = dist.reduce_scatter_tensor(
                out, mine[:tokens].contiguous(), group=group, async_op=True
            )
            work.wait()
            lo = rank * (tokens // world)
            assert torch.equal(out, ref[lo : lo + tokens // world]), tokens

        # All-reduce (numel not a multiple of world exercises the padding).
        for tokens in (1, 3, pool):
            t = mine[:tokens].clone()
            dist.all_reduce(t, group=group)
            assert torch.equal(t, ref[:tokens]), tokens

        # Non-sum ops keep the original collective.
        t = mine[:2].float().clone()
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
        assert torch.equal(t, parts[:, :2].float().amax(0))
        dist.destroy_process_group()
    except Exception as e:  # noqa: BLE001
        errors.put(f"rank {rank}: {type(e).__name__}: {e}")
        raise


@pytest.mark.parametrize("world", [3, 4])
def test_ordered_reductions_are_rank_order_sums(world):
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, errors)) for r in range(world)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    msgs = []
    while not errors.empty():
        msgs.append(errors.get())
    assert not msgs, msgs
    assert all(p.exitcode == 0 for p in procs)
