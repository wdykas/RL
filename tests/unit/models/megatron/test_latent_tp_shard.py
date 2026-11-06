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
"""TP-sharded latent projections (latent_tp_shard.py): CPU, gloo, 2 and 4 ranks.

Against the duplicated layer (fp16_gemm.fp16_linear on the full weight, every rank its own
sequence-parallel rows): weight-mode and rows-mode forwards bitwise; weight-mode dgrad bitwise;
the shard's main_grad == the shard rows of sum_r fp32(bf16(wgrad_r)) (rank-order fold, bitwise).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class _Lin(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone())


def _bits(a, b):
    return int((a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)).sum())


def _worker(rank, world, port, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from nemo_rl.models.megatron import fp16_gemm as fg
        from nemo_rl.models.megatron import latent_tp_shard as lts

        g = torch.Generator().manual_seed(5)
        n, k, rows = 8 * world, 24, 3
        w = (torch.randn(n, k, generator=g) * 0.1).to(torch.bfloat16)
        xs = torch.randn(world, rows, k, generator=g)  # each rank's SP rows
        gos = torch.randn(world, rows, n, generator=g)
        bad = {}
        # reference: duplicated layer on this rank's rows
        x = xs[rank].clone().requires_grad_(True)
        ref_w = torch.nn.Parameter(w.clone())
        out_ref = fg.fp16_linear(x, ref_w, fg.COMM_NONE, None)
        out_ref.backward(gos[rank])
        dx_ref = x.grad.clone()
        wg = ref_w.grad.float()  # bf16 per-rank wgrad (DDP adds it into an fp32 main_grad)
        allw = [torch.empty_like(wg) for _ in range(world)]
        dist.all_gather(allw, wg)
        tot = allw[0] + allw[1]
        for j in range(2, world):
            tot = tot + allw[j]
        r = n // world
        # sharded
        lin = _Lin(w)
        lts.shard_linear(lin, dist.group.WORLD)
        assert lin.weight.shape == (r, k) and lin.tensor_model_parallel and lin.partition_dim == 0
        lin.weight.main_grad = torch.zeros(r, k)
        lin.weight.grad_added_to_main_grad = False
        x2 = xs[rank].clone().requires_grad_(True)
        os.environ["NEMO_RL_LATENT_SHARD_MODE"] = "weight"
        out_w = lts.linear_forward(lin, x2)
        out_w.backward(gos[rank])
        bad["fwd_weight"] = _bits(out_w.detach(), out_ref.detach())
        bad["dgrad"] = _bits(x2.grad, dx_ref)
        bad["main_grad"] = _bits(lin.weight.main_grad, tot[rank * r:(rank + 1) * r])
        bad["flag"] = int(not lin.weight.grad_added_to_main_grad)
        os.environ["NEMO_RL_LATENT_SHARD_MODE"] = "rows"
        with torch.no_grad():
            out_r = lts.linear_forward(lin, xs[rank].clone())
        bad["fwd_rows"] = _bits(out_r, out_ref.detach())
        os.environ["NEMO_RL_LATENT_SHARD_MODE"] = "auto"
        with torch.no_grad():
            out_a = lts.linear_forward(lin, xs[rank].clone())
        bad["fwd_auto"] = _bits(out_a, out_ref.detach())
        q.put((rank, bad))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [2, 4])
def test_sharded_latent_projection_matches_duplicated(world):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    res = dict(q.get(timeout=180) for _ in range(world))
    for p in procs:
        p.join(timeout=60)
    for r in range(world):
        assert all(v == 0 for v in res[r].values()), (r, res[r])
