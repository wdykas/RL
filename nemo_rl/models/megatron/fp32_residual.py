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
"""Megatron inference support for ``fp32_residual_connection``.

With an fp32 residual stream, Transformer Engine's training ``LayerNormLinear``
casts its fp32 input to the activation dtype before the norm
(``cast_if_needed(inp, activation_dtype)``). MCore's inference-optimized fused
norm + linear passes the fp32 tensor straight to TE's ``rmsnorm_fwd`` with bf16
weights, for which TE has no kernel. ``install()`` makes the inference norm do
the same cast as training, so the two paths keep computing the same function.
"""

from __future__ import annotations

import torch

_INSTALLED: list[bool] = []


def install() -> None:
    """Cast the inference norm input to the norm weight's dtype (idempotent)."""
    if _INSTALLED:
        return
    from megatron.core.tensor_parallel import inference_layers

    inner = inference_layers._te_rms_norm_kernel

    def _te_rms_norm_kernel(x: torch.Tensor, weight: torch.Tensor, eps: float):
        if x.dtype != weight.dtype:
            x = x.to(weight.dtype)
        return inner(x=x, weight=weight, eps=eps)

    inference_layers._te_rms_norm_kernel = _te_rms_norm_kernel
    _INSTALLED.append(True)
