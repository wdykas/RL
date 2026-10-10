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
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs CUDA (Triton)"
)


def _reference_attention(q, k, v):
    """fp64 causal attention on Megatron's [s, b, heads, dim] layout (GQA)."""
    s, b, h, d = q.shape
    rep = h // k.shape[2]
    qq, kk, vv = (t.permute(1, 2, 0, 3).double() for t in (q, k, v))
    kk = kk.repeat_interleave(rep, 1)
    vv = vv.repeat_interleave(rep, 1)
    scores = qq @ kk.transpose(-1, -2) / math.sqrt(d)
    mask = torch.triu(torch.ones(s, s, dtype=torch.bool, device=q.device), 1)
    out = scores.masked_fill(mask, float("-inf")).softmax(-1) @ vv
    return out.permute(2, 0, 1, 3).reshape(s, b, -1)


def test_sdpa_core_attention_matches_fp64_reference_with_gqa():
    from thundersync_rl.scorer_kernels import SDPACoreAttention

    torch.manual_seed(0)
    s, b, h, hk, d = 37, 3, 6, 2, 64
    q = torch.randn(s, b, h, d, device="cuda")
    k = torch.randn(s, b, hk, d, device="cuda")
    v = torch.randn(s, b, hk, d, device="cuda")
    out = SDPACoreAttention(softmax_scale=1 / math.sqrt(d))(q, k, v, None)
    assert out.shape == (s, b, h * d) and out.dtype == torch.float32
    torch.testing.assert_close(
        out.double(), _reference_attention(q, k, v), atol=1e-5, rtol=0
    )


def test_sdpa_core_attention_bf16_option_keeps_output_dtype():
    from thundersync_rl.scorer_kernels import SDPACoreAttention

    q = torch.randn(8, 2, 4, 64, device="cuda")
    attn = SDPACoreAttention(1 / 8, attn_dtype=torch.bfloat16)
    assert attn(q, q, q, None).dtype == torch.float32


class _LocalAttention(torch.nn.Module):
    """Stands in for Megatron's local DotProductAttention (keeps softmax_scale)."""

    def __init__(self, scale, window_size=None):
        super().__init__()
        self.softmax_scale = scale
        if window_size is not None:
            self.window_size = window_size


class _Layer(torch.nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core_attention = core


def test_use_sdpa_attention_keeps_each_modules_scale_and_skips_unsupported():
    from thundersync_rl.scorer_kernels import SDPACoreAttention, use_sdpa_attention

    class Sink(_LocalAttention):  # learnable attention sink: has parameters
        def __init__(self):
            super().__init__(0.1)
            self.softmax_offset = torch.nn.Parameter(torch.zeros(2))

    cfg = SimpleNamespace(softmax_scale=None, window_size=None, softmax_type="vanilla")
    model = torch.nn.Sequential(
        _Layer(_LocalAttention(0.25)),  # MLA/YaRN-style custom scale
        _Layer(_LocalAttention(0.125, window_size=(128, 0))),  # sliding window
        _Layer(Sink()),
        _Layer(torch.nn.Identity()),  # scale unknown
    )
    assert use_sdpa_attention(model, cfg) == 1
    assert use_sdpa_attention(model, cfg) == 0
    assert isinstance(model[0].core_attention, SDPACoreAttention)
    assert model[0].core_attention.softmax_scale == 0.25
    assert not any(isinstance(m.core_attention, SDPACoreAttention) for m in model[1:])
    windowed_cfg = SimpleNamespace(softmax_scale=None, window_size=(64, 0))
    assert (
        use_sdpa_attention(
            torch.nn.Sequential(_Layer(_LocalAttention(0.2))), windowed_cfg
        )
        == 0
    )


def test_fused_swiglu_patches_only_dense_mlp():
    from megatron.core.transformer.mlp import MLP
    from thundersync_rl.scorer_kernels import use_fused_swiglu

    cfg = SimpleNamespace(gated_linear_unit=True, activation_func=F.silu)

    class NotDense(torch.nn.Module):  # e.g. an MoE expert container
        def __init__(self):
            super().__init__()
            self.config, self.linear_fc1, self.linear_fc2 = cfg, None, None

    dense = MLP.__new__(MLP)
    torch.nn.Module.__init__(dense)
    dense.config = cfg
    model = torch.nn.Sequential(dense, NotDense())
    assert use_fused_swiglu(model) == 1
    assert "forward" in vars(dense) and "forward" not in vars(model[1])


@pytest.mark.parametrize("shape", [(5, 3, 2 * 1000), (7, 2 * 4096)])
def test_swiglu_matches_silu_times_linear(shape):
    from thundersync_rl.scorer_kernels import swiglu

    x = torch.randn(*shape, device="cuda")
    a, lin = torch.chunk(x, 2, dim=-1)
    torch.testing.assert_close(swiglu(x), F.silu(a) * lin, atol=1e-6, rtol=1e-6)


def test_selective_output_returns_logits_of_selected_positions_only():
    from thundersync_rl.scorer_kernels import (
        select_positions,
        supports_selection,
        use_selective_output,
    )

    class Out(torch.nn.Module):  # stands in for Megatron's ColumnParallelLinear
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(11, 4, device="cuda"))
            self.sequence_parallel = False

        def forward(self, input_, weight=None, runtime_gather_output=None):
            return input_ @ self.weight.t(), None

    model = SimpleNamespace(output_layer=Out())
    assert not supports_selection(model)
    assert use_selective_output(model) and supports_selection(model)
    hidden = torch.randn(6, 2, 4, device="cuda")  # [s, b, h]
    full, _ = model.output_layer(hidden)
    sel = (
        torch.tensor([0, 3, 5], device="cuda"),
        torch.tensor([1, 0, 1], device="cuda"),
    )
    with select_positions(model, sel):
        part, _ = model.output_layer(hidden)
    assert part.shape == (3, 1, 11)
    torch.testing.assert_close(part[:, 0], full[sel[0], sel[1]])
    after, _ = model.output_layer(hidden)  # selection ends with the block
    assert after.shape == full.shape
