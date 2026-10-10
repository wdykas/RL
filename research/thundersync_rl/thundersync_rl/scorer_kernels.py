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
"""Fast fp32 forward pieces for the verification scorer (Megatron GPTModel copy).

Megatron/TE kernels are tuned for bf16; in fp32 the scorer falls back to
unfused paths. These replacements compute the same functions:

* ``swiglu``: one Triton pass over the fc1 output instead of strided silu,
  ``+ glu_linear_offset`` and mul kernels on non-contiguous halves.
* ``SDPACoreAttention``: causal attention through torch SDPA (fused fp32)
  instead of the unfused score-matrix path with TF32 score GEMMs.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _swiglu_kernel(x_ptr, out_ptr, h, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < h
    a = tl.load(x_ptr + row * 2 * h + cols, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(x_ptr + row * 2 * h + h + cols, mask=mask, other=0.0).to(tl.float32)
    y = a / (1.0 + tl.exp(-a)) * b
    tl.store(out_ptr + row * h + cols, y.to(out_ptr.dtype.element_ty), mask=mask)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    """silu(x[..., :h]) * x[..., h:] for a contiguous [..., 2h] tensor."""
    x = x.contiguous()
    h = x.shape[-1] // 2
    rows = x.numel() // (2 * h)
    out = torch.empty(*x.shape[:-1], h, dtype=x.dtype, device=x.device)
    if rows:
        block = 1024
        _swiglu_kernel[(rows, triton.cdiv(h, block))](x, out, h, BLOCK=block)
    return out


def use_fused_swiglu(model: torch.nn.Module) -> int:
    """Route every plain SwiGLU Megatron MLP of ``model`` through ``swiglu``."""
    n = 0
    for mod in list(model.modules()):
        cfg = getattr(mod, "config", None)
        if not (
            hasattr(mod, "linear_fc1")
            and hasattr(mod, "linear_fc2")
            and cfg is not None
        ):
            continue
        if not (
            cfg.gated_linear_unit
            and cfg.activation_func is F.silu
            and not getattr(cfg, "use_te_activation_func", False)
            and getattr(cfg, "activation_func_clamp_value", None) is None
            and getattr(cfg, "activation_func_tanh_clamp_scale", None) is None
            and not getattr(cfg, "glu_linear_offset", 0.0)
        ):
            continue

        def forward(hidden_states, per_token_scale=None, _m=mod, **kw):
            assert per_token_scale is None
            x, bias = _m.linear_fc1(hidden_states)
            if bias is not None:
                x = x + bias
            return _m.linear_fc2(swiglu(x))

        mod.forward = forward
        n += 1
    return n


class SDPACoreAttention(torch.nn.Module):
    """Causal core attention through torch SDPA.

    Inputs are Megatron's [s, b, heads, dim] (GQA: fewer kv heads); output is
    [s, b, heads * dim]. Rows are right-padded, so causal masking alone is exact
    for the real tokens.
    """

    def __init__(
        self, orig: torch.nn.Module, softmax_scale: float | None, attn_dtype=None
    ):
        super().__init__()
        self.orig = orig
        self.softmax_scale = softmax_scale
        # Optional lower-precision attention (e.g. bf16 flash) inside an fp32 model.
        self.attn_dtype = attn_dtype

    def forward(
        self,
        query,
        key,
        value,
        attention_mask=None,
        attn_mask_type=None,
        attention_bias=None,
        packed_seq_params=None,
        **kw,
    ):
        assert packed_seq_params is None and attention_bias is None
        q, k, v = (t.permute(1, 2, 0, 3) for t in (query, key, value))  # [b, h, s, d]
        if k.shape[1] != q.shape[1]:
            rep = q.shape[1] // k.shape[1]
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        dtype = q.dtype
        if self.attn_dtype is not None:
            q, k, v = (
                q.to(self.attn_dtype),
                k.to(self.attn_dtype),
                v.to(self.attn_dtype),
            )
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.softmax_scale
        )
        s, b = query.shape[0], query.shape[1]
        return out.to(dtype).permute(2, 0, 1, 3).reshape(s, b, -1)


def use_sdpa_attention(model: torch.nn.Module, config, attn_dtype=None) -> int:
    """Replace every core_attention submodule of ``model`` with SDPA; return count."""
    n = 0
    for mod in list(model.modules()):
        ca = getattr(mod, "core_attention", None)
        if isinstance(ca, torch.nn.Module) and not isinstance(ca, SDPACoreAttention):
            mod.core_attention = SDPACoreAttention(
                ca, getattr(config, "softmax_scale", None), attn_dtype
            )
            n += 1
    return n


def use_selective_output(model: torch.nn.Module) -> bool:
    """Let callers pick which hidden positions reach the LM head.

    Set ``model.output_layer.ts_select = (seq_idx, batch_idx)`` before a forward
    to get logits of shape [1, n, V] for just those positions (prompt tokens and
    positions outside a draft segment need no logits). Not available with
    sequence parallelism, where the output layer gathers sharded hidden states.
    """
    layer = getattr(model, "output_layer", None)
    if layer is None or getattr(layer, "sequence_parallel", False):
        return False
    orig_forward = layer.forward
    layer.ts_select = None

    def forward(input_, weight=None, runtime_gather_output=None, **kw):
        sel = layer.ts_select
        if sel is not None:
            input_ = input_[sel[0], sel[1]].unsqueeze(1)  # [n, 1, h]
        return orig_forward(
            input_, weight=weight, runtime_gather_output=runtime_gather_output, **kw
        )

    layer.forward = forward
    return True
