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
"""DS2 at TP > 1 (torchrun --nproc_per_node=TP ds2_tp.py): every rank holds its vocabulary shard of the p and q
logits and runs ds2.grid_scan_sharded with real NCCL collectives. On every rank the outcomes, the drafted tokens' log p,
the positions evaluated and the generator state must equal the single-GPU TP 1 run on the full rows, bit for bit.
Also the gathered scan (block_verify_stats_lazy_grid) on the full rows. Prints "DS2 TP PASS" on success."""

import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(__file__))

from test_ds2 import ENVS, _case, _gathered, _lse  # noqa: E402

from thundersync_rl import ds2  # noqa: E402


def main() -> int:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    group = dist.group.WORLD
    cases = [
        dict(g=40, nblk=world), dict(g=600, nblk=world, sharp=0.03), dict(g=300, nblk=2 * world, sharp=0.05),
        dict(g=200, nblk=world, vlim_cut=1000, sharp=0.05), dict(g=64, nblk=world, accept_all=True),
        dict(g=500, nblk=world, sharp=3.0), dict(g=150, nblk=world, dtype=torch.bfloat16, sharp=0.05),
    ]
    bad = 0
    n = 0
    for ci, case in enumerate(cases):
        for seed in range(3):
            zp, zq, vlim, draft = _case(seed, **case)  # the same full rows on every rank (CPU generator)
            want, lpx_w, lqx, gen_w = _gathered(zp, zq, vlim, draft, 200 + seed, ENVS)
            width = zp.shape[1] // world
            off = rank * width
            sp, sq = zp[:, off : off + width].contiguous(), zq[:, off : off + width].contiguous()
            rows = ds2.ShardedRows(sp, vlim, group, off)
            lpx = rows.token_logprobs(draft)
            gen = torch.Generator(device="cuda").manual_seed(200 + seed)

            def q_local(idx, sq=sq):
                from nemo_rl.models.megatron import vocab_lse

                zz = sq[idx.to("cuda")].contiguous()
                return zz, vocab_lse.lse_sharded(zz, off, vlim, group)

            got, n_eval = ds2.grid_scan_sharded(rows, lpx, lqx, q_local, gen, ENVS, first_chunk=64, chunk=256)
            ok = got == want and torch.equal(lpx, lpx_w) and torch.equal(gen.get_state(), gen_w)
            lse_ok = torch.equal(rows.lse, _lse(zp, vlim))
            n += 1
            if not (ok and lse_ok):
                bad += 1
                print(f"[rank {rank}] MISMATCH case {ci} seed {seed}: got {got} want {want} lse_ok {lse_ok}", flush=True)
    t = torch.tensor([bad], device="cuda")
    dist.all_reduce(t)
    if rank == 0:
        print(f"TP {world}: {n} rows x {world} ranks, mismatches {int(t)}")
        if int(t) == 0:
            print("DS2 TP PASS", flush=True)
    dist.destroy_process_group()
    return 0 if int(t) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
