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
q), so block verification against pi_{n+1} is unchanged.

theta_hat is computed on the learner from the Megatron DistributedOptimizer
state, on the rank that owns each main-parameter shard (no gather of master or
moments), in chunks of ``CHUNK_ELEMS`` elements (bounded fp32 temporaries),
written into the DDP parameter buffer in place of cast(master) and all-gathered
exactly as after an optimizer step. The existing refit path then streams the
buffer to the engine, and the training weights are restored bitwise by the
optimizer's own main -> model copy + all-gather. No extra full copy of the
model is materialized.

Precision-aware optimizer (``use_precision_aware_optimizer``): the masters live
in TE FusedAdam's state (fp32, fp16-scaled, or bf16 parameter + int16 remainder
bits); they are read per chunk the same way, and since TE writes the parameter
slice inside its step (no main -> model copy to redo), the owned slice is saved
before the draft is written (2 bytes per owned element) and copied back.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    steps: int = 1,
) -> torch.Tensor:
    """fp32 master after ``steps`` AdamW steps with gradient 0, the first at optimizer count ``step``.

    ``steps`` = h > 1 (anchor epochs: Psi_h): counts step .. step + h - 1, fp32 throughout, one final
    rounding by the caller -- verbatim the E2 harness's ``psi(n, st, L)`` (research/sticky_alt/adam_lag_alt.py).

    The expression (and its operation order) is the research harness's
    ``adam_new(n, None, t + 1)`` (``research/sticky/adam_draft_gpu.py``,
    ``draft_master_fn("mom")``), so the draft is bitwise the one the offline
    acceptance numbers were measured with. ``step`` is the count of the update
    being predicted (optimizer steps taken so far + 1). Does not mutate.
    """
    if int(steps) == 1:
        mm = beta1 * exp_avg
        vv = beta2 * exp_avg_sq
        upd = (mm / (1 - beta1**step)) / ((vv / (1 - beta2**step)).sqrt() + eps)
        return w - lr * (upd + weight_decay * w)
    if int(steps) < 1:
        raise ValueError(f"momentum draft: steps must be >= 1 (got {steps})")
    mm, vv = exp_avg, exp_avg_sq
    for j in range(int(steps)):
        t1 = step + j
        mm = beta1 * mm
        vv = beta2 * vv
        w = w - lr * ((mm / (1 - beta1**t1)) / ((vv / (1 - beta2**t1)).sqrt() + eps) + weight_decay * w)
    return w


@dataclass
class _Shard:
    inner: torch.optim.Optimizer  # the Adam over main-parameter shards
    group: dict
    key: torch.Tensor  # the inner optimizer's parameter (its state key)
    target: torch.Tensor  # this rank's slice of the DDP parameter buffer
    # True when the buffer slice must be saved to be restored bitwise: an fp32
    # model param (the main shard *is* the slice), or any param under the
    # precision-aware optimizer (TE FusedAdam writes the slice in its step; there
    # is no separate main -> model copy to redo).
    aliased: bool
    # fp32 main shard (default optimizer); None: masters inside TE FusedAdam.
    main: torch.Tensor | None = None


def master_from_remainders(param: torch.Tensor, remainder: torch.Tensor) -> torch.Tensor:
    """fp32 master of TE FusedAdam ``store_param_remainders`` (bf16 param + int16 low bits).

    TE keeps the high 16 bits in the bf16 parameter, rounded up when the low
    16 bits (the int16 remainder) have their top bit set (negative as int16),
    so the high half is ``param bits - (remainder < 0)``. Inverse of
    ``multi_tensor_adam_param_remainder``'s split; elementwise.
    """
    if param.dtype != torch.bfloat16 or remainder.dtype != torch.int16:
        raise TypeError(f"expected bf16 param + int16 remainder, got {param.dtype}, {remainder.dtype}")
    hi = param.view(torch.int16).to(torch.int32) - (remainder < 0).to(torch.int32)
    bits = (hi << 16) | (remainder.to(torch.int32) & 0xFFFF)
    return bits.view(torch.float32)


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
        if opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8 and not hasattr(
            opt.optimizer, "get_unscaled_state"
        ):
            raise NotImplementedError(
                "draft_weights=momentum: the precision-aware optimizer is built for "
                f"TE FusedAdam (master weights in its state); got {type(opt.optimizer).__name__}"
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
        if opt.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
            # The inner optimizer's params are the model-param shards (views of
            # the buffer); TE FusedAdam holds their masters.
            pairs = (
                (opt.shard_float16_groups, opt.model_float16_groups, True, False),
                (opt.shard_fp32_groups, opt.model_fp32_groups, True, False),
            )
        else:
            pairs = (
                (opt.shard_fp32_from_float16_groups, opt.model_float16_groups, False, True),
                (opt.shard_fp32_groups, opt.model_fp32_groups, True, True),
            )
        for keys, models, aliased, has_main in pairs:
            for key_group, model_group in zip(keys, models):
                for key, model_param in zip(key_group, model_group):
                    world_range = opt._get_model_param_range_map(model_param)[
                        "gbuf_world_in_bucket"
                    ]
                    gbuf_index, _, bucket_id = opt.model_param_gbuf_map[model_param]
                    buf = opt.buffers[gbuf_index].buckets[bucket_id].param_data
                    yield _Shard(
                        inner=opt.optimizer,
                        group=group_of[id(key)],
                        key=key,
                        target=buf.view(-1)[world_range.start : world_range.end],
                        aliased=aliased,
                        main=key if has_main else None,
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


def _master_chunks(shard: _Shard):
    """Callable slice -> fp32 master values of the shard (no full fp32 copy if avoidable)."""
    if shard.main is not None:
        w = shard.main.view(-1)
        return lambda sl: w[sl]
    state = shard.inner.state.get(shard.key)
    param = shard.key.view(-1)
    if not state or "master_param" not in state:
        # fp32 param (its own master), or before the first step: the master is
        # the parameter itself.
        return lambda sl: param[sl].float()
    master = state["master_param"]
    if master.dtype == torch.int16:  # store_param_remainders
        rem = master.view(-1)
        return lambda sl: master_from_remainders(param[sl], rem[sl])
    full = shard.inner.get_unscaled_state(shard.key, "master_param").view(-1)
    return lambda sl: full[sl].float()














