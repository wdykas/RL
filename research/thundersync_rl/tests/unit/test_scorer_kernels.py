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
    out = SDPACoreAttention(torch.nn.Identity(), softmax_scale=None)(q, k, v, None)
    assert out.shape == (s, b, h * d) and out.dtype == torch.float32
    torch.testing.assert_close(
        out.double(), _reference_attention(q, k, v), atol=1e-5, rtol=0
    )


def test_sdpa_core_attention_bf16_option_keeps_output_dtype():
    from thundersync_rl.scorer_kernels import SDPACoreAttention

    q = torch.randn(8, 2, 4, 64, device="cuda")
    attn = SDPACoreAttention(torch.nn.Identity(), None, attn_dtype=torch.bfloat16)
    assert attn(q, q, q, None).dtype == torch.float32


def test_use_sdpa_attention_replaces_every_core_attention_once():
    from thundersync_rl.scorer_kernels import SDPACoreAttention, use_sdpa_attention

    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core_attention = torch.nn.Identity()

    model = torch.nn.Sequential(Layer(), Layer())
    assert use_sdpa_attention(model, SimpleNamespace(softmax_scale=None)) == 2
    assert use_sdpa_attention(model, SimpleNamespace(softmax_scale=None)) == 0
    assert all(isinstance(m.core_attention, SDPACoreAttention) for m in model)


@pytest.mark.parametrize("shape", [(5, 3, 2 * 1000), (7, 2 * 4096)])
def test_swiglu_matches_silu_times_linear(shape):
    from thundersync_rl.scorer_kernels import swiglu

    x = torch.randn(*shape, device="cuda")
    a, lin = torch.chunk(x, 2, dim=-1)
    torch.testing.assert_close(swiglu(x), F.silu(a) * lin, atol=1e-6, rtol=1e-6)


def test_selective_output_returns_logits_of_selected_positions_only():
    from thundersync_rl.scorer_kernels import use_selective_output

    class Out(torch.nn.Module):  # stands in for Megatron's ColumnParallelLinear
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(11, 4, device="cuda"))
            self.sequence_parallel = False

        def forward(self, input_, weight=None, runtime_gather_output=None):
            return input_ @ self.weight.t(), None

    model = SimpleNamespace(output_layer=Out())
    assert use_selective_output(model)
    hidden = torch.randn(6, 2, 4, device="cuda")  # [s, b, h]
    full, _ = model.output_layer(hidden)
    sel = (
        torch.tensor([0, 3, 5], device="cuda"),
        torch.tensor([1, 0, 1], device="cuda"),
    )
    model.output_layer.ts_select = sel
    part, _ = model.output_layer(hidden)
    assert part.shape == (3, 1, 11)
    torch.testing.assert_close(part[:, 0], full[sel[0], sel[1]])
