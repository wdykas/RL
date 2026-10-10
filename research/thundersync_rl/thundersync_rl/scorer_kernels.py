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
unfused paths. These replacements compute the same functions and are applied
only where that holds:

* ``swiglu``: one Triton pass over the fc1 output instead of strided silu,
  ``+ glu_linear_offset`` and mul kernels on non-contiguous halves (dense
  ``MLP`` modules only; MoE experts keep their own forward).
* ``SDPACoreAttention``: causal attention through torch SDPA (fused fp32)
  instead of the unfused score-matrix path with TF32 score GEMMs (plain causal
  softmax attention only: no sliding window, no attention sinks).
* ``select_positions``: run the LM head on chosen positions only.
"""

from __future__ import annotations

import contextlib

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
    """silu(x[..., :h]) * x[..., h:] for a [..., 2h] tensor."""
    x = x.contiguous()
    h = x.shape[-1] // 2
    rows = x.numel() // (2 * h)
    out = torch.empty(*x.shape[:-1], h, dtype=x.dtype, device=x.device)
    if rows:
        block = 1024
        _swiglu_kernel[(rows, triton.cdiv(h, block))](x, out, h, BLOCK=block)
    return out


def _plain_swiglu(cfg) -> bool:
    return bool(
        cfg.gated_linear_unit
        and cfg.activation_func is F.silu
        and not getattr(cfg, "use_te_activation_func", False)
        and getattr(cfg, "activation_func_clamp_value", None) is None
        and getattr(cfg, "activation_func_tanh_clamp_scale", None) is None
        and not getattr(cfg, "glu_linear_offset", 0.0)
    )


def use_fused_swiglu(model: torch.nn.Module) -> int:
    """Route every dense SwiGLU ``MLP`` of ``model`` through ``swiglu``; return count.

    Only modules whose type is exactly Megatron's ``MLP``: MoE experts
    (``TEGroupedMLP``, ``SequentialMLP``) and ``SharedExpertMLP`` have other call
    signatures and outputs, and keep their own forward.
    """
    from megatron.core.transformer.mlp import MLP

    n = 0
    for mod in list(model.modules()):
        if type(mod) is not MLP or not _plain_swiglu(mod.config):
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
    for the real tokens. ``softmax_scale`` must be the replaced module's own
    scale (it differs from 1/sqrt(dim) for e.g. MLA with YaRN).
    """

    def __init__(self, softmax_scale: float, attn_dtype: torch.dtype | None = None):
        super().__init__()
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
            q, k, v = (t.to(self.attn_dtype) for t in (q, k, v))
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, scale=self.softmax_scale
        )
        s, b = query.shape[0], query.shape[1]
        return out.to(dtype).permute(2, 0, 1, 3).reshape(s, b, -1)


def _replaceable_scale(core_attention: torch.nn.Module, config) -> float | None:
    """The module's softmax scale if it is plain causal softmax attention, else None.

    Megatron's local ``DotProductAttention`` keeps ``softmax_scale`` itself; TE's
    keeps it on its backends (``unfused_attention``). Modules with parameters
    (learnable attention sinks), a sliding window or a non-vanilla softmax are
    left alone.
    """
    if any(True for _ in core_attention.parameters()):
        return None
    if getattr(config, "window_size", None) is not None:
        return None
    window = getattr(core_attention, "window_size", None)
    if window is not None and tuple(window) not in ((-1, 0), (-1, -1)):
        return None
    softmax_type = getattr(
        core_attention, "softmax_type", getattr(config, "softmax_type", "vanilla")
    )
    if softmax_type != "vanilla":
        return None
    scale = getattr(core_attention, "softmax_scale", None)
    if scale is None:
        scale = getattr(
            getattr(core_attention, "unfused_attention", None), "softmax_scale", None
        )
    return None if scale is None else float(scale)


def use_sdpa_attention(model: torch.nn.Module, config, attn_dtype=None) -> int:
    """Replace each plain causal core_attention of ``model`` with SDPA; return count."""
    n = 0
    for mod in list(model.modules()):
        ca = getattr(mod, "core_attention", None)
        if not isinstance(ca, torch.nn.Module) or isinstance(ca, SDPACoreAttention):
            continue
        scale = _replaceable_scale(ca, config)
        if scale is not None:
            mod.core_attention = SDPACoreAttention(scale, attn_dtype)
            n += 1
    return n


def use_selective_output(model: torch.nn.Module) -> bool:
    """Patch ``model``'s LM head so ``select_positions`` can restrict it.

    Not available with sequence parallelism, where the output layer gathers
    sequence-sharded hidden states itself. Returns whether it was applied.
    """
    layer = getattr(model, "output_layer", None)
    if layer is None or getattr(layer, "sequence_parallel", False):
        return False
    orig_forward = layer.forward
    layer._ts_select = None

    def forward(input_, weight=None, runtime_gather_output=None, **kw):
        sel = layer._ts_select
        if sel is not None:
            input_ = input_[sel[0], sel[1]].unsqueeze(1)  # [n, 1, h]
        return orig_forward(
            input_, weight=weight, runtime_gather_output=runtime_gather_output, **kw
        )

    layer.forward = forward
    return True


def supports_selection(model: torch.nn.Module) -> bool:
    """Whether ``use_selective_output`` was applied to ``model``."""
    return hasattr(getattr(model, "output_layer", None), "_ts_select")


@contextlib.contextmanager
def select_positions(model: torch.nn.Module, select):
    """Restrict ``model``'s LM head to ``select = (seq_idx, batch_idx)`` in the block.

    Forwards inside the block return logits [1, n, V] for those positions only.
    """
    layer = model.output_layer
    prev = layer._ts_select
    layer._ts_select = select
    try:
        yield
    finally:
        layer._ts_select = prev
