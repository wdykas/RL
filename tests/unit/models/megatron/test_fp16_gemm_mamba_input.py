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
"""fp16_gemm's Mamba pre-norm patch keeps the fp32 residual as the norm input on both Megatron-Core layouts."""

import sys
import types

import pytest
import torch

from nemo_rl.models.megatron import fp16_gemm


def _fake_mamba_layer_module(split_residual: bool):
    m = types.ModuleType("megatron.core.ssm.mamba_layer")
    m.apply_module = lambda mod: mod

    class MambaLayer:
        def __init__(self):
            self.config = types.SimpleNamespace(fp32_residual_connection=True, params_dtype=torch.bfloat16)
            self.norm = lambda x: x * 2
            self.mixer = types.SimpleNamespace(in_proj=types.SimpleNamespace(weight=object()))

        if split_residual:  # Megatron-LM main: the residual is prepared separately

            def _prepare_residual(self, h):
                return h.float()

            def _prepare_mixer_input(self, h):
                return self.norm(h.to(self.config.params_dtype))

        else:  # older Megatron-Core: (normalized input, residual)

            def _prepare_mixer_input(self, h):
                return self.norm(h.to(self.config.params_dtype)), h.float()

    m.MambaLayer = MambaLayer
    return m


@pytest.mark.parametrize("split_residual", [True, False])
def test_mamba_prenorm_patch_matches_the_installed_layout(monkeypatch, split_residual):
    mod = _fake_mamba_layer_module(split_residual)
    monkeypatch.setitem(sys.modules, "megatron.core.ssm.mamba_layer", mod)
    ssm = sys.modules.get("megatron.core.ssm")
    if ssm is not None:
        monkeypatch.setattr(ssm, "mamba_layer", mod, raising=False)
    monkeypatch.setattr(fp16_gemm, "_active", lambda w: w is not None)
    monkeypatch.setitem(fp16_gemm._STATE, "orig", dict(fp16_gemm._STATE.get("orig", {})))
    fp16_gemm._patch_mamba_layer_input()
    layer = mod.MambaLayer()
    x = torch.tensor([1.0 + 2**-12], dtype=torch.float32)  # not representable in bf16
    out = layer._prepare_mixer_input(x)
    if split_residual:
        assert torch.is_tensor(out) and out.dtype == torch.float32 and out.item() == 2 * x.item()
    else:
        normed, residual = out
        assert normed.dtype == torch.float32 and normed.item() == 2 * x.item() and residual is x
