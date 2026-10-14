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
"""Momentum-predicted draft weights (``thundersync.draft_weights: momentum``).

The engine drafts the rollouts of step n+1 with

    theta_hat_{n+1} = AdamW step from (theta_n, m_n, v_n) with gradient 0,

rounded to the parameter storage dtype, instead of theta_n. It is a different
proposal q only: q stays the law the engine actually sampled (engine-sourced
q), so block verification against pi_{n+1} is unchanged (DESIGN.md 14f).

theta_hat is computed on the learner from the Megatron DistributedOptimizer
state, on the rank that owns each main-parameter shard (no gather of master or
moments), in chunks of ``CHUNK_ELEMS`` elements (bounded fp32 temporaries),
written into the DDP parameter buffer in place of cast(master) and all-gathered
exactly as after an optimizer step. The existing refit path then streams the
buffer to the engine, and the training weights are restored bitwise by the
optimizer's own main -> model copy + all-gather. No extra full copy of the
model is materialized.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

DRAFT_WEIGHTS = ("current", "momentum")


def momentum_draft_master(
    w: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
    weight_decay: float,
    step: int,
) -> torch.Tensor:
    """fp32 master after one AdamW step with gradient 0 at optimizer count ``step``.

    The expression (and its operation order) is the research harness's
    ``adam_new(n, None, t + 1)`` (``research/sticky/adam_draft_gpu.py``,
    ``draft_master_fn("mom")``), so the draft is bitwise the one the offline
    acceptance numbers were measured with. ``step`` is the count of the update
    being predicted (optimizer steps taken so far + 1). Does not mutate.
    """
    mm = beta1 * exp_avg
    vv = beta2 * exp_avg_sq
    upd = (mm / (1 - beta1**step)) / ((vv / (1 - beta2**step)).sqrt() + eps)
    return w - lr * (upd + weight_decay * w)


@dataclass
class _Shard:
    inner: torch.optim.Optimizer  # the Adam over main-parameter shards
    group: dict
    main: torch.Tensor  # fp32 main-parameter shard
    target: torch.Tensor  # this rank's slice of the DDP parameter buffer
    aliased: bool  # fp32 model param: the main shard *is* the buffer slice


def _dist_optimizers(optimizer) -> list:
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    members = getattr(optimizer, "chained_optimizers", [optimizer])
    for opt in members:
        if not isinstance(opt, DistributedOptimizer):
            raise NotImplementedError(
                "draft_weights=momentum: built for Megatron's DistributedOptimizer "
                f"(use_distributed_optimizer=true); got {type(opt).__name__}"
            )
        if opt.ddp_config.use_megatron_fsdp:
            raise NotImplementedError("draft_weights=momentum: Megatron-FSDP not built yet")
        if opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            raise NotImplementedError(
                "draft_weights=momentum: precision-aware optimizer not built yet "
                "(main params live inside TE FusedAdam)"
            )
        if opt.ddp_config.fp8_param_gather or opt.ddp_config.fp4_param_gather:
            raise NotImplementedError("draft_weights=momentum: fp8/fp4 params not built yet")
        if not getattr(opt.optimizer, "adam_w_mode", True):
            raise NotImplementedError(
                "draft_weights=momentum: the draft is an AdamW (decoupled weight decay) step"
            )
    return [opt for opt in members if not opt.is_stub_optimizer]


def _shards(optimizer) -> Iterator[_Shard]:
    """Every main-parameter shard of this rank with its parameter-buffer slice."""
    for opt in _dist_optimizers(optimizer):
        group_of = {
            id(p): g for g in opt.optimizer.param_groups for p in g["params"]
        }
        for mains, models, aliased in (
            (opt.shard_fp32_from_float16_groups, opt.model_float16_groups, False),
            (opt.shard_fp32_groups, opt.model_fp32_groups, True),
        ):
            for main_group, model_group in zip(mains, models):
                for main, model_param in zip(main_group, model_group):
                    world_range = opt._get_model_param_range_map(model_param)[
                        "gbuf_world_in_bucket"
                    ]
                    gbuf_index, _, bucket_id = opt.model_param_gbuf_map[model_param]
                    buf = opt.buffers[gbuf_index].buckets[bucket_id].param_data
                    yield _Shard(
                        inner=opt.optimizer,
                        group=group_of[id(main)],
                        main=main,
                        target=buf.view(-1)[world_range.start : world_range.end],
                        aliased=aliased,
                    )


def _adam_state(inner, main: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat (exp_avg, exp_avg_sq) of a main shard; zeros before the first step."""
    if not inner.state.get(main):
        z = main.new_zeros(()).expand(main.numel())  # flat zeros, no allocation
        return z, z
    get = getattr(inner, "get_unscaled_state", None)
    if get is not None:  # TE FusedAdam (handles fp16/bf16/fp8-stored states)
        m, v = get(main, "exp_avg"), get(main, "exp_avg_sq")
    else:
        m, v = inner.state[main]["exp_avg"], inner.state[main]["exp_avg_sq"]
    return m.view(-1), v.view(-1)


# Elements per draft computation: bounds the fp32 temporaries (about 5 x 4 B per
# element) independently of the largest parameter shard. The formula is
# elementwise, so chunking does not change a bit.
CHUNK_ELEMS = 1 << 24


def _draft_chunks(shard: _Shard, chunk: int) -> Iterator[tuple[slice, torch.Tensor]]:
    """(slice, rounded draft) of one main shard, ``chunk`` elements at a time."""
    g = shard.group
    beta1, beta2 = g["betas"]
    m, v = _adam_state(shard.inner, shard.main)
    w = shard.main.view(-1)
    for start in range(0, w.numel(), chunk):
        sl = slice(start, min(start + chunk, w.numel()))
        draft = momentum_draft_master(
            w[sl],
            m[sl].float(),  # no-op for fp32 states
            v[sl].float(),
            lr=g["lr"],
            beta1=beta1,
            beta2=beta2,
            eps=g["eps"],
            weight_decay=g["weight_decay"],
            step=int(g.get("step", 0)) + 1,
        )
        yield sl, draft.to(shard.target.dtype)


@dataclass
class DraftSwap:
    """Training weights temporarily replaced by draft weights (see ``load_draft``)."""

    optimizer: object
    # fp32 model params whose main shard is the buffer itself: original values.
    saved: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)


def _param_sync(optimizer) -> None:
    for chunk in optimizer.model_chunks:
        chunk.start_param_sync(force_sync=True)


@torch.no_grad()
def load_draft(optimizer, kind: str, chunk: int = CHUNK_ELEMS) -> DraftSwap:
    """Write draft weights ``kind`` into the model parameters (all ranks, collective).

    Per main shard: compute the fp32 draft, round it into the shard's slice of
    the parameter buffer (as the optimizer's main -> model copy does), then
    all-gather the buffers as after an optimizer step. Undo with
    ``restore_training``. Frozen parameters (no main shard) keep theta_n,
    which is their AdamW step too.
    """
    if kind != "momentum":
        raise ValueError(f"unknown draft weights {kind!r} (expected one of {DRAFT_WEIGHTS})")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    swap = DraftSwap(optimizer)
    changed = total = 0
    for shard in _shards(optimizer):
        if shard.aliased:
            swap.saved.append((shard.target, shard.target.clone()))
        target = shard.target.view(-1)
        for sl, new in _draft_chunks(shard, chunk):
            changed += (new != target[sl]).sum()  # on device: no sync per chunk
            total += new.numel()
            target[sl].copy_(new)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    _param_sync(optimizer)
    torch.cuda.synchronize()
    swap.stats = {
        "draft/compute_s": t1 - t0,
        "draft/gather_s": time.perf_counter() - t1,
        "draft/peak_extra_bytes": float(torch.cuda.max_memory_allocated() - base),
        "draft/changed_elems": float(changed),  # one sync, after the gather
        "draft/shard_elems": float(total),
    }
    return swap


@torch.no_grad()
def restore_training(swap: DraftSwap) -> dict[str, float]:
    """Put theta_n back, bitwise: the optimizer's own main -> model copy + all-gather."""
    t0 = time.perf_counter()
    for target, orig in swap.saved:
        target.copy_(orig)
    swap.saved.clear()
    swap.optimizer.quantize_and_sync_model_params_from_main_params()
    torch.cuda.synchronize()
    return {"draft/restore_s": time.perf_counter() - t0}
