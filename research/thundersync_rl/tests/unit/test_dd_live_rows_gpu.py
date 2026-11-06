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
"""4-GPU symmetric exchange under sparse cross-fade routing (standalone: python test_dd_live_rows_gpu.py, spawns 4
ranks). A workspace sized by live rows (stock k per token) gives bitwise the outputs and gradients of the one sized
by every candidate slot (C), with no overflow; a source with every candidate live overflows its return rows: flagged
on every rank, no fault."""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD, E, K, C, T, H = 4, 16, 2, 6, 64, 40


def _routing(rank, all_live):
    g = torch.Generator().manual_seed(100 + rank)
    rows = []  # (expert, token)
    for t in range(T):
        n = C if all_live else K + int(torch.randint(0, C - K + 1, (1,), generator=g)) // 3
        for e in torch.randperm(E, generator=g)[:n].tolist():
            rows.append((e, t))
    rows.sort()  # permutation 1: expert major, token order within
    counts = torch.zeros(E, dtype=torch.int32)
    for e, _ in rows:
        counts[e] += 1
    tok = torch.randn(T, H, generator=g)
    x = torch.randn(T * C, H, generator=g)  # padding rows past the live ones: junk
    p = torch.rand(T * C, generator=g)
    for i, (e, t) in enumerate(rows):
        x[i] = tok[t] * (1 + e)
    return counts, x, p, len(rows)


def _run(dd, ws, counts, x, p):
    xs = x.cuda().requires_grad_()
    ps = p.cuda().requires_grad_()
    plan = dd._Plan(ws, counts.cuda(), xs.shape[0], sparse=True)
    xr, pr = dd._Dispatch.apply(xs, ps, plan, False)
    y = xr * (1.5 + pr.view(-1, 1))  # a row-wise expert op
    out = dd._Combine.apply(y, plan, False)
    g = torch.Generator(device="cuda").manual_seed(7)
    (out * torch.randn(out.shape, device="cuda", generator=g)).sum().backward()
    return out.detach(), xs.grad, ps.grad


def _worker(rank, port, q):
    try:
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS=str(T))
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", rank=rank, world_size=WORLD, device_id=torch.device("cuda", rank))
        import torch.distributed._symmetric_memory as symm_mem

        symm_mem.enable_symm_mem_for_group(dist.group.WORLD.group_name)
        from nemo_rl.models.megatron import device_dispatch as dd

        grp = dist.group.WORLD
        ws_c = dd._Workspace(grp, E, C, H, T, 1, row_k=C)  # every candidate slot (before)
        ws_k = dd._Workspace(grp, E, C, H, T, 1, row_k=K)  # live rows (after)
        assert ws_k.r_cap < ws_c.r_cap and ws_k.s_cap < ws_c.s_cap, (ws_k.r_cap, ws_c.r_cap, ws_k.s_cap, ws_c.s_cap)
        counts, x, p, live = _routing(rank, False)
        assert live <= ws_k.s_cap
        res = []
        for ws in (ws_c, ws_k):
            dd.overflowed(force=True)
            r = _run(dd, ws, counts, x, p)
            torch.cuda.synchronize()
            assert not dd.overflowed(force=True)
            res.append(r)
        for a, b, what in zip(res[0], res[1], ("output", "token grad", "prob grad")):
            assert torch.equal(a, b), f"rank {rank}: {what} differs"
        # rank 1 routes every candidate: 384 live rows > s_cap 256 -> overflow on every rank, no fault
        counts, x, p, live = _routing(rank, rank == 1)
        dd.overflowed(force=True)
        _run(dd, ws_k, counts, x, p)
        torch.cuda.synchronize()
        assert dd.overflowed(force=True), f"rank {rank}: source overflow not flagged"
        dist.barrier()
        q.put((rank, "ok"))
    except BaseException as exc:  # noqa: BLE001 - report to the parent
        q.put((rank, repr(exc)))
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_live_row_workspace_bitwise_and_source_overflow():
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=_worker, args=(r, port, q)) for r in range(WORLD)]
    for pr in procs:
        pr.start()
    got = dict(q.get(timeout=600) for _ in range(WORLD))
    for pr in procs:
        pr.join(60)
    assert all(v == "ok" for v in got.values()), got


if __name__ == "__main__":
    test_live_row_workspace_bitwise_and_source_overflow()
    print("PASS test_live_row_workspace_bitwise_and_source_overflow")
