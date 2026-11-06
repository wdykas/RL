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
"""Momentum draft weights with sharded optimizer state (torchrun, 2+ ranks).

    torchrun --nproc_per_node=2 draft_weights_multirank.py --tp 2 --ep 1   # TP-sharded params
    torchrun --nproc_per_node=2 draft_weights_multirank.py --tp 1 --ep 2   # DP-sharded dense, EP experts

Every rank computes the draft of only the main shards it owns. Checks: (1) each owned
shard's slice of the parameter buffer == the harness formula on that shard's
(master, exp_avg, exp_avg_sq), rounded; (2) after the all-gather every bucket equals
the union of the owners' drafts (each rank's slices summed over the bucket's
data-parallel group); (3) restore gives back theta_n bitwise. Prints MULTIRANK OK.
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_draft_weights import _toy, _train_step, harness_mom  # noqa: E402

from thundersync_rl.draft_weights import (  # noqa: E402
    _master_chunks,
    _shards,
    load_draft,
    restore_training,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--pa", choices=["none", "fp32", "remainders"], default="none")
    args = ap.parse_args()
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    ps.initialize_model_parallel(args.tp, 1, expert_model_parallel_size=args.ep)
    model_parallel_cuda_manual_seed(0)
    # MoE when EP > 1 (all-bf16); TP > 1 runs the dense model (fp32 norms under
    # autocast: the aliased fp32 path; Megatron's MoE + TP needs sequence parallel).
    moe = args.ep > 1
    ddp, opt = _toy(
        moe, tp=args.tp, ep=args.ep, precision_aware=None if args.pa == "none" else args.pa
    )
    for s in range(args.steps):
        _train_step(ddp, opt, s, autocast=not moe)
    params = dict(ddp.module.named_parameters())
    before = {n: p.detach().clone() for n, p in params.items()}

    # Expected draft of every owned shard (harness formula), keyed by its buffer slice.
    expected = []
    for sh in _shards(opt):
        n = sh.target.numel()
        master = _master_chunks(sh)(slice(0, n))
        m, v = (
            sh.inner.get_unscaled_state(sh.key, k).view(-1)
            if args.pa != "none"
            else sh.inner.state[sh.key][k].view(-1)
            for k in ("exp_avg", "exp_avg_sq")
        )
        b1, b2 = sh.group["betas"]
        w = harness_mom(master, m, v, int(sh.group["step"]),
                        sh.group["lr"], b1, b2, sh.group["eps"], sh.group["weight_decay"])
        expected.append((sh.target, w.to(sh.target.dtype)))
    owned = sum(t.numel() for t, _ in expected)
    total = sum(p.numel() for p in params.values() if p.requires_grad)

    swap = load_draft(opt, "momentum", chunk=777)
    for target, w in expected:  # (1) owned slices
        assert torch.equal(target, w)
    # (2) gathered buckets == union of the owners' drafts
    members = getattr(opt, "chained_optimizers", [opt])
    nb = 0
    for o in members:
        group = o.data_parallel_group
        for buf in o.buffers:
            for bucket in buf.buckets:
                want = torch.zeros_like(bucket.param_data, dtype=torch.float64)
                flat = want.view(-1)
                for target, w in expected:
                    base = bucket.param_data.view(-1)
                    off = target.data_ptr() - base.data_ptr()
                    if 0 <= off < base.numel() * base.element_size():
                        i = off // base.element_size()
                        flat[i : i + w.numel()] = w.double()
                dist.all_reduce(want, group=group)
                assert torch.equal(want.to(bucket.param_data.dtype), bucket.param_data)
                nb += 1
    changed = swap.stats["draft/changed_elems"]
    assert changed > 0
    restore_training(swap)
    for n, p in params.items():
        assert torch.equal(p.detach(), before[n]), n
    print(f"rank {rank}: tp={args.tp} ep={args.ep} owned {owned}/{total} elems, "
          f"{nb} buckets, changed {int(changed)}, peak_extra "
          f"{swap.stats['draft/peak_extra_bytes'] / 2**20:.2f} MiB", flush=True)
    dist.barrier()
    if rank == 0:
        print("MULTIRANK OK", flush=True)
    ps.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
