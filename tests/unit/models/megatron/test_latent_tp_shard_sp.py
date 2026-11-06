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
"""TP-sharded latent MoE projections (nemo_rl/models/megatron/latent_tp_shard.py).

``_ShardedLinear.backward`` always sums the full-weight wgrad over TP (reduce-scatter). That equals the duplicated
layer's gradient only under sequence parallelism (each TP rank holds T/TP distinct rows; Megatron then all-reduces
the duplicated weight's grad over TP because TELinear marks it ``sequence_parallel``). Without SP every TP rank
holds the same rows, the duplicated weight's grad is NOT summed over TP, and the shard's grad comes out TP x too
large. Megatron requires SP only for MoE with TP > 1 *and* EP > 1, so TP > 1 / EP 1 / SP off builds and trains
silently with a TP-fold latent gradient. Real gloo collectives at TP 2 / 4 (one process per TP rank).
"""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _worker(rank, world, port, sp, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from nemo_rl.models.megatron import latent_tp_shard as lts

        g = dist.group.WORLD
        torch.manual_seed(0)
        T, K, N = 4 * world, 16, 8 * world  # tokens (all ranks), in, out (latent rows)
        x_full = torch.randn(T, K).to(torch.float16).float()
        w_full = torch.randn(N, K).to(torch.bfloat16)
        gout_full = torch.randn(T, N).to(torch.bfloat16).float()
        rows = N // world
        # this rank's rows: SP -> its T/TP slice; no SP -> the whole (replicated) batch
        sl = slice(rank * T // world, (rank + 1) * T // world) if sp else slice(0, T)
        x = x_full[sl].clone().requires_grad_(True)
        shard = torch.nn.Parameter(w_full[rank * rows : (rank + 1) * rows].clone())
        out = lts._ShardedLinear.apply(x, shard, g, None)
        out.backward(gout_full[sl])
        # duplicated layer's gradient as Megatron finalizes it: SP -> summed over TP (= over all T rows);
        # no SP -> this rank's own (identical on every rank) gradient over all T rows, NOT summed.
        ref_full = (gout_full.to(torch.bfloat16).float().t() @ x_full).to(torch.bfloat16).float()
        ref = ref_full[rank * rows : (rank + 1) * rows]
        got = shard.grad.float()
        q.put((rank, float((got - ref).abs().max()), float(ref.abs().max()), float(got.norm() / ref.norm())))
    finally:
        dist.destroy_process_group()


def _run(world, sp):
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    ps = [ctx.Process(target=_worker, args=(r, world, port, sp, q)) for r in range(world)]
    for p in ps:
        p.start()
    for p in ps:
        p.join(120)
    assert all(p.exitcode == 0 for p in ps), [p.exitcode for p in ps]
    return [q.get() for _ in range(world)]


@pytest.mark.parametrize("world", [2, 4])
def test_sharded_latent_wgrad_matches_duplicated_with_sp(world):
    for rank, err, scale, ratio in _run(world, sp=True):
        # same products, TP-ordered fp32 sum of bf16-rounded per-rank partials vs one fp32 sum: rounding only
        assert err <= 0.02 * scale, (rank, err, scale, ratio)


@pytest.mark.parametrize("world", [2, 4])
def test_sharded_latent_wgrad_without_sp_is_tp_fold(world):
    """Without SP the shard's grad is TP x the duplicated layer's (measured 1.999 / 3.999 before the guard). The mechanism is kept (rows mode / weight mode need SP to be equivalent); the fix rejects the
    configuration (test_config_rejects_latent_shard_without_sp)."""
    for rank, err, scale, ratio in _run(world, sp=False):
        assert abs(ratio - world) < 0.02 * world, f"TP {world} rank {rank}: shard grad / duplicated grad = {ratio:.3f}"


@pytest.mark.parametrize("tp,sp,ok", [(1, False, True), (2, True, True), (2, False, False), (4, False, False), (8, True, True)])
def test_config_rejects_latent_shard_without_sp(tp, sp, ok):
    """The fix guards the config (both the learner and the engine config pass through validate_config)."""
    from nemo_rl.models.megatron import fp16_gemm

    cfg = {
        "forward_gemm_precision": "fp16",
        "fp32_residual_connection": True,
        "moe_latent_tp_shard": True,
        "tensor_model_parallel_size": tp,
        "sequence_parallel": sp,
    }
    errs = [e for e in fp16_gemm.validate_config(cfg) if "moe_latent_tp_shard" in e]
    assert (not errs) == ok, errs
