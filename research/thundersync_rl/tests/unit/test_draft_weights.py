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
"""Momentum-predicted draft weights (thundersync_rl.draft_weights, DESIGN.md 14f)."""

import os

import pytest
import torch

from thundersync_rl.draft_weights import (
    load_draft,
    momentum_draft_master,
    restore_training,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def harness_mom(master, m, v, t, lr, b1, b2, eps, wd):
    """``adam_new(n, None, t + 1)[0]`` of research/sticky/adam_draft_gpu.py, verbatim."""
    t1 = t + 1
    w, mm, vv = master, m, v
    mm = b1 * mm
    vv = b2 * vv
    upd = (mm / (1 - b1**t1)) / ((vv / (1 - b2**t1)).sqrt() + eps)
    return w - lr * (upd + wd * w)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=cuda)])
@pytest.mark.parametrize("t", [0, 1, 7, 300])
def test_formula_matches_harness_bitwise(device, t):
    g = torch.Generator().manual_seed(t)
    w = torch.randn(4097, generator=g).to(device)
    m = (torch.randn(4097, generator=g) * 1e-3).to(device)
    v = (torch.rand(4097, generator=g) * 1e-6).to(device)
    if t == 0:  # before the first step the state is zero
        m.zero_()
        v.zero_()
    hp = dict(lr=1e-6, b1=0.9, b2=0.999, eps=1e-8, wd=0.01)
    ref = harness_mom(w, m, v, t, **hp)
    got = momentum_draft_master(
        w, m, v, lr=hp["lr"], beta1=hp["b1"], beta2=hp["b2"], eps=hp["eps"],
        weight_decay=hp["wd"], step=t + 1,
    )
    assert torch.equal(got, ref)
    assert torch.equal(got.bfloat16(), ref.bfloat16())
    if t > 0:
        assert not torch.equal(got, w)


@pytest.fixture(scope="module")
def mcore():
    """World-size-1 torch.distributed + Megatron model parallel state."""
    import torch.distributed as dist
    from megatron.core import parallel_state as ps
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    owns_pg = not dist.is_initialized()
    if owns_pg:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(29500 + os.getpid() % 1000))
        dist.init_process_group("nccl", rank=0, world_size=1)
    ps.initialize_model_parallel(1, 1)
    model_parallel_cuda_manual_seed(0)
    yield
    ps.destroy_model_parallel()
    if owns_pg:  # leave the process as found: other tests set up their own groups
        dist.destroy_process_group()


def _toy(moe: bool, tp: int = 1, ep: int = 1):
    """Tiny bf16 GPT (MoE: expert + dense DistOpts in a ChainedOptimizer) with
    Megatron DDP + DistributedOptimizer, as the learner builds it."""
    from megatron.core.distributed import (
        DistributedDataParallel,
        DistributedDataParallelConfig,
    )
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.transformer import TransformerConfig

    cfg = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        ffn_hidden_size=128,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        num_moe_experts=4 if moe else None,
        moe_router_topk=2,
        moe_grouped_gemm=False,
        moe_token_dispatcher_type="alltoall",
        add_bias_linear=False,
        tensor_model_parallel_size=tp,
        expert_model_parallel_size=ep,
    )
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_local_spec(
            num_experts=4 if moe else None, moe_grouped_gemm=False
        ),
        vocab_size=128,
        max_sequence_length=32,
    ).cuda()
    if moe:
        # All-bf16 model (Megatron's mixed-precision wrapper); the dense case keeps
        # fp32 norms (main shard aliases the parameter) and runs under autocast.
        from megatron.core.transformer.module import Float16Module

        model = Float16Module(cfg, model)
    ddp = DistributedDataParallel(
        cfg,
        DistributedDataParallelConfig(
            grad_reduce_in_fp32=True, use_distributed_optimizer=True
        ),
        model,
    )
    opt = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam", lr=1e-3, weight_decay=0.01, adam_beta1=0.9,
            adam_beta2=0.999, adam_eps=1e-8, bf16=True, use_distributed_optimizer=True,
        ),
        [ddp],
    )
    return ddp, opt


def _train_step(ddp, opt, seed, autocast):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.randint(0, 128, (8, 32), generator=g, device="cuda")
    pos = torch.arange(32, device="cuda")[None].expand(8, -1)
    ddp.zero_grad_buffer()
    opt.zero_grad()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        out = ddp(ids, pos, None)
    (out.float().square().mean()).backward()
    ddp.finish_grad_sync()
    ok, _, _ = opt.step()
    assert ok


@cuda
@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("steps", [0, 3])
@pytest.mark.parametrize("chunk", [None, 1000])
def test_distopt_draft_matches_harness_and_restores(mcore, moe, steps, chunk):
    ddp, opt = _toy(moe)
    for s in range(steps):
        _train_step(ddp, opt, s, autocast=not moe)
    params = dict(ddp.module.named_parameters())
    before = {n: p.detach().clone() for n, p in params.items()}
    # Reference: harness formula on every main param (world size 1: shard = param).
    members = getattr(opt, "chained_optimizers", [opt])
    ref = {}
    for o in members:
        inner = o.optimizer
        pairs = list(zip(o.shard_fp32_from_float16_groups, o.model_float16_groups))
        # fp32 model params (e.g. norms): the main shard aliases the parameter.
        pairs += list(zip(o.shard_fp32_groups, o.model_fp32_groups))
        for mains, models in pairs:
            for main, mp in zip(mains, models):
                grp = next(g for g in inner.param_groups if any(p is main for p in g["params"]))
                st = inner.state[main]
                m = st["exp_avg"] if st else torch.zeros_like(main)
                v = st["exp_avg_sq"] if st else torch.zeros_like(main)
                b1, b2 = grp["betas"]
                w = harness_mom(
                    main, m, v, int(grp.get("step", 0)), grp["lr"], b1, b2,
                    grp["eps"], grp["weight_decay"],
                )
                ref[id(mp)] = w.to(mp.dtype).view(mp.shape)
    assert len(ref) == sum(p.requires_grad for p in params.values())

    swap = load_draft(opt, "momentum", **({} if chunk is None else {"chunk": chunk}))
    for n, p in params.items():
        assert torch.equal(p.detach(), ref[id(p)]), n
    if steps:  # with momentum the draft differs from theta_n (at 0 only lr*wd moves it)
        assert swap.stats["draft/changed_elems"] > 0
    assert swap.stats["draft/shard_elems"] == sum(p.numel() for p in params.values())

    restore_training(swap)
    for n, p in params.items():
        assert torch.equal(p.detach(), before[n]), n


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs")
@pytest.mark.parametrize("tp,ep", [(2, 1), (1, 2)])
def test_distopt_draft_sharded_two_ranks(tp, ep):
    """TP- / EP+DP-sharded optimizer state on 2 ranks (draft_weights_multirank.py)."""
    import subprocess
    import sys

    here = os.path.dirname(os.path.abspath(__file__))
    port = 29600 + os.getpid() % 300
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--nproc_per_node=2",
         f"--master_port={port}", os.path.join(here, "draft_weights_multirank.py"),
         f"--tp={tp}", f"--ep={ep}"],
        capture_output=True, text=True, timeout=600,
    )
    assert out.returncode == 0 and "MULTIRANK OK" in out.stdout, out.stdout[-3000:] + out.stderr[-3000:]


def test_engine_stage_round_trip_keeps_parameter_storage():
    from thundersync_rl.draft_weights import EngineStage

    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    shared = torch.nn.ModuleList([model, model])  # chunks sharing parameters
    draft = [p.detach().clone() for p in model.parameters()]
    ptrs = [p.data_ptr() for p in model.parameters()]
    stage = EngineStage()
    st = stage.stage(shared, checksum=True)
    assert st["stage_bytes"] == sum(p.numel() * 4 for p in model.parameters())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)  # the rollout weights replace the draft weights
    ld = stage.load(shared, checksum=True)
    assert ld["checksum"] == st["checksum"]
    assert [p.data_ptr() for p in model.parameters()] == ptrs
    for p, d in zip(model.parameters(), draft):
        assert torch.equal(p, d)
    with pytest.raises(RuntimeError, match="not staged"):
        EngineStage().load(model)
