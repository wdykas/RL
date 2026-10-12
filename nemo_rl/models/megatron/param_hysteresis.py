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
"""Hysteresis rounding of fp32 master weights into bf16/fp16 model weights.

Megatron rounds every fp32 master weight to the nearest bf16 value after each
optimizer step. With small learning rates the master moves a tiny fraction of a
bf16 ulp per step, often with a noisy sign, so masters sitting near a rounding
midpoint flip the bf16 weight back and forth by a full ulp. Those flips change the
deployed policy far more than the update itself, which lowers exact speculative
acceptance between consecutive policies.

With hysteresis the model weight keeps its value until the master is at least
``tau`` ulps (of the current weight) away from it, then takes round(master).
``tau = 0.5`` is round-to-nearest; ``tau = 1`` changes a weight only after its
master has moved a full ulp away. The model weights stay a deterministic function
of the masters and the previous weights, and they are what training's forward and
the inference engine both use, so the policy remains exactly defined.

Requires the distributed optimizer with separately stored fp32 masters
(``use_precision_aware_optimizer=False``: with TE's precision-aware optimizer the
bf16 weight is part of the master's storage).
"""

from __future__ import annotations

import torch

_STATE: dict[str, float] = {}


@torch.no_grad()
def hysteresis_copy_(dst: torch.Tensor, main: torch.Tensor, tau: float) -> None:
    """``dst`` (low precision) <- round(main) where |main - dst| >= tau ulp(dst), else unchanged."""
    old = dst.float()
    main = main.float()
    # Spacing to the neighbour on main's side: 2^(e-1) * eps for |old| in
    # [2^(e-1), 2^e), halved when |old| is a power of two and main lies toward zero.
    mantissa, exponent = torch.frexp(old)
    ulp = torch.ldexp(torch.full_like(old, torch.finfo(dst.dtype).eps), exponent - 1)
    toward_zero = (main - old) * old < 0
    ulp = torch.where((mantissa.abs() == 0.5) & toward_zero, ulp / 2, ulp)
    move = ((main - old).abs() >= tau * ulp) | (old == 0)
    dst.copy_(torch.where(move, main.to(dst.dtype), dst))


def _copy_main_params_to_model_params(self) -> None:
    tau = _STATE["tau"]
    if self.is_stub_optimizer:
        return
    if (
        self.ddp_config.use_megatron_fsdp
        or self.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        or self.ddp_config.fp8_param_gather
        or getattr(self.ddp_config, "fp4_param_gather", False)
    ):
        raise RuntimeError(
            "bf16 parameter hysteresis supports the distributed optimizer with "
            "fp32 masters and bf16/fp16 parameters only."
        )

    def copy_groups(shard_main_groups, model_groups, hysteresis: bool) -> None:
        for shard_main_group, model_group in zip(shard_main_groups, model_groups):
            for shard_main_param, model_param in zip(shard_main_group, model_group):
                world_range = self._get_model_param_range_map(model_param)[
                    "gbuf_world_in_bucket"
                ]
                assert world_range.size == shard_main_param.nelement()
                gbuf_index, _, bucket_id = self.model_param_gbuf_map[model_param]
                buffer = self.buffers[gbuf_index].buckets[bucket_id].param_data
                shard = buffer.view(-1)[world_range.start : world_range.end]
                if hysteresis and shard.dtype in (torch.bfloat16, torch.float16):
                    hysteresis_copy_(shard, shard_main_param.view(-1), tau)
                else:
                    shard.data.copy_(shard_main_param)

    copy_groups(self.shard_fp32_from_float16_groups, self.model_float16_groups, True)
    copy_groups(self.shard_fp32_groups, self.model_fp32_groups, False)


def install(tau: float) -> None:
    """Use hysteresis rounding in Megatron's distributed optimizer (idempotent)."""
    if not 0.5 <= tau <= 1.0:
        raise ValueError(
            f"bf16 parameter hysteresis tau must be in [0.5, 1], got {tau}"
        )
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    _STATE["tau"] = tau
    DistributedOptimizer._copy_main_params_to_model_params = (
        _copy_main_params_to_model_params
    )
    print(f"[param_hysteresis] bf16 parameter hysteresis tau={tau}", flush=True)
