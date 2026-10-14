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
"""fp16 GEMM forward numerics (``policy.megatron_cfg.forward_gemm_precision: fp16``).

The "V8b" forward of thundersync_work/research/bf16_ablation.md, computed by the
same functions in Megatron training and in the Megatron inference engine:

- every linear layer (attention QKV / output projection, MLP fc1 / fc2, LM head)
  casts its input and its bf16 weight to fp16 and runs one cuBLAS GEMM with fp32
  accumulation and an **fp32 output** (``fp16_gemm``). A bf16 weight is exactly
  representable in fp16 when |w| is in [6.1e-5, 65504]; weights, checkpoints,
  fp32 master weights and the optimizer are unchanged (bf16 / fp32);
- the residual stream is fp32 (``fp32_residual_connection``, required);
- RMSNorm (fused input norms, q/k norms, final norm) runs in fp32 on the fp32
  input with the batch-invariant row kernel and emits fp32; SwiGLU runs on the
  fp32 fc1 output. The next GEMM rounds its input to fp16;
- q / k / v are rounded to fp16 after the q/k norms, so RoPE, FlashAttention
  (fp16 probabilities in its PV product) and the engine's KV cache are fp16;
- the LM head (``lm_head_precision``, default "fp32") multiplies the fp32
  final-norm output by the head weight with fp32-input numerics (``fp32_gemm``;
  "fp16" uses the fp16 GEMM instead); logits are fp32 and the batch-invariant
  fp32 log-softmax is unchanged.

Tensor parallelism (sequence parallel): a norm output is rounded to fp16 before
its all-gather (same bytes as bf16 today); row-parallel GEMM outputs are
reduce-scattered in fp32 (2x the bytes of today's bf16 reduce-scatter). Both
sides call the same code, so their collectives match.

Backward (training only; only the forward has to match the engine): the
dgrad / wgrad GEMMs run on bf16 casts with fp32 accumulation. Gradients never
pass through fp16 except inside FlashAttention's backward, where the incoming
gradient is scaled by a power of two chosen from its amax (and unscaled on the
q / k / v side) so it does not underflow fp16's exponent range.

``install()`` patches the Megatron-Core / Transformer Engine classes from NeMo-RL
(the Megatron sources are shared and not edited); every patched method falls
back to the original unless ``install()`` ran in this process and the module's
weight is 16-bit (a separately built fp32 model keeps its own numerics).
Dense models only: MoE grouped GEMMs are not built yet.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

FP16_GEMM = "fp16"
SUPPORTED_PRECISIONS = (FP16_GEMM,)
# LM-head GEMM under the fp16 forward (``lm_head_precision``): "fp32" (default)
# multiplies the fp32 final-norm output by the bf16 head weight with fp32-input
# numerics (``fp32_gemm``: two exact fp16 GEMMs on a hi/lo split of the input);
# "fp16" uses the fp16 GEMM like every layer.
FP32_HEAD = "fp32"
HEAD_PRECISIONS = (FP32_HEAD, FP16_GEMM)

_STATE: dict[str, Any] = {
    "enabled": False,
    "head": FP32_HEAD,
    "orig": {},
    "down_scale": 0,
    # Sticky rounding (sticky_rounding.py, opt-in): fp16 cast of a rounding
    # site, site_cast(site, x) -> fp16, and the training MoE preprocess hook.
    # None: plain round-to-nearest.
    "site_cast": None,
    "moe_preprocess": None,
}


def set_sticky_hooks(site_cast: Optional[Any], moe_preprocess: Optional[Any]) -> None:
    """Install (or with None remove) sticky rounding's site hooks."""
    _STATE["site_cast"] = site_cast
    _STATE["moe_preprocess"] = moe_preprocess


def _to_fp16(x: torch.Tensor, site: Any) -> torch.Tensor:
    """fp16 cast of an activation at a rounding site (sticky rounding if installed)."""
    cast = _STATE["site_cast"]
    if cast is None or site is None:
        return x.to(torch.float16)
    return cast(site, x)

# Target |amax| of an attention-output gradient dO after scaling into fp16: 2^0.
# FlashAttention's dV = P^T dO sums dO over every query attending a key, so an
# attention-sink key gets up to L * amax(dO) (L = sequence length); 2^0 keeps it
# below fp16's max (65504) up to L = 64k, while gradients down to 2^-24 * amax
# remain representable (subnormal below 2^-14). A larger target (2^8) saturated
# dV of sink keys on Qwen3-1.7B at L = 2048 (grad norm 17 vs 0.4, then NaN).
_ATTN_GRAD_TARGET_LOG2 = 0.0
# NEMO_RL_FP16_GEMM_DEBUG=1: report fp16 attention gradients near fp16's max.
_DEBUG = os.environ.get("NEMO_RL_FP16_GEMM_DEBUG") == "1"


def enabled() -> bool:
    """Whether the fp16 GEMM forward is installed in this process."""
    return bool(_STATE["enabled"])


def _active(weight: Optional[torch.Tensor]) -> bool:
    return (
        _STATE["enabled"]
        and weight is not None
        and weight.dtype in (torch.bfloat16, torch.float16)
    )


# ---------------------------------------------------------------------------
# The shared GEMM.
# ---------------------------------------------------------------------------


def _mm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
    """A [M, K] @ b [K, N] with fp32 accumulation, emitted in ``out_dtype``.

    CUDA: one cuBLAS GEMM (``torch.mm(..., out_dtype=...)``; under the te_native
    batch-invariant backend cuBLASLt runs without workspace, i.e. without
    split-K). CPU (tests): fp32 matmul of the 16-bit operands, which is exact per
    product (11- or 8-bit significands) with fp32 accumulation.
    """
    if a.is_cuda:
        if a.dtype == out_dtype:
            return torch.mm(a, b)
        return torch.mm(a, b, out_dtype=out_dtype)
    return torch.mm(a.float(), b.float()).to(out_dtype)


def fp16_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` with fp16 operands, fp32 accumulation and fp32 output.

    x [..., K] (any float dtype; rounded to fp16), weight [N, K] (bf16 or fp16;
    rounded to fp16). Returns fp32 [..., N]. The single GEMM of the fp16 forward:
    the training linear layers, the inference linear layers, the LM head and
    ThunderSync's engine-q head recomputation all call it.
    """
    k = x.shape[-1]
    x2 = x.reshape(-1, k).to(torch.float16).contiguous()
    w16 = weight.to(torch.float16)
    out = _mm(x2, w16.t(), torch.float32)
    return out.view(*x.shape[:-1], weight.shape[0])


_SPLIT_SCALE = 2048.0  # 2^11: the fp16 significand width


def fp32_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``x @ weight.T`` for an fp32 x and a bf16 weight, fp32 output.

    x is split into two fp16 terms, x = hi + lo / 2^11 (hi = fp16(x),
    lo = fp16((x - hi) * 2^11); ~22 significant bits of x), and the weight is
    cast to fp16 (exact for a bf16 weight in fp16's normal range). Two fp16
    GEMMs with fp32 accumulation give every product exactly; the result is
    hi @ w.T + (lo @ w.T) / 2^11 in fp32.

    Why not one IEEE fp32 cuBLAS GEMM: measured on GB200, fp32 SGEMM rows at
    M = 1, 3, 8 differ from the same rows in a 4096-row GEMM (kernel choice
    changes the reduction), so it is not batch invariant for few decoding
    requests; and ``NVIDIA_TF32_OVERRIDE=1`` (set by the engine-q scripts)
    turns it into TF32 regardless of ``allow_tf32``. The fp16 GEMM is batch
    invariant at every M tested (1..4096) and runs on tensor cores.
    """
    x32 = x.float()
    hi = x32.to(torch.float16)
    lo = ((x32 - hi.float()) * _SPLIT_SCALE).to(torch.float16)
    return fp16_gemm(hi, weight) + fp16_gemm(lo, weight) / _SPLIT_SCALE


_GEMMS = {FP16_GEMM: fp16_gemm, FP32_HEAD: fp32_gemm}
_OPERAND_DTYPE = {FP16_GEMM: torch.float16, FP32_HEAD: torch.float32}


def head_precision() -> str:
    """Operand precision of the LM-head GEMM under the fp16 forward."""
    return _STATE["head"]


def head_input_dtype(weight_dtype: torch.dtype) -> torch.dtype:
    """Dtype that holds an LM-head input losslessly for the head GEMM.

    Under the fp16 forward: fp32 with the fp32 head (the final-norm output as
    is; twice the bytes of bf16), fp16 with the fp16 head (the GEMM rounds its
    input to fp16 first, so storing that rounding loses nothing). Otherwise the
    weight dtype, as before.
    """
    if not _STATE["enabled"]:
        return weight_dtype
    return _OPERAND_DTYPE[_STATE["head"]]


def head_matmul(h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """fp32 logits ``h @ weight.T`` exactly as the model's LM head computes them."""
    if _STATE["enabled"]:
        return _GEMMS[_STATE["head"]](h, weight)
    return h.float() @ weight.float().t()


# ---------------------------------------------------------------------------
# Autograd: fp16 forward, bf16 backward.
# ---------------------------------------------------------------------------

# Collectives fused into the linear function (sequence dimension = dim 0).
COMM_NONE = None
COMM_SP_GATHER = "sp_gather"  # all-gather input rows; bwd: reduce-scatter dgrad
COMM_SP_SCATTER = "sp_scatter"  # reduce-scatter output rows; bwd: all-gather grad
COMM_TP_COPY = "tp_copy"  # input replicated; bwd: all-reduce dgrad
COMM_TP_ALLREDUCE = "tp_allreduce"  # all-reduce output; bwd: identity


def _gather_rows(x: torch.Tensor, group: Any) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import _gather_along_first_dim

    return _gather_along_first_dim(x, group)


def _reduce_scatter_rows(x: torch.Tensor, group: Any) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import _reduce_scatter_along_first_dim

    return _reduce_scatter_along_first_dim(x, group)


def _all_reduce(x: torch.Tensor, group: Any) -> torch.Tensor:
    torch.distributed.all_reduce(x, group=group)
    return x


class Fp16Linear(torch.autograd.Function):
    """y = fp16_gemm(x, w) (or fp32_gemm for the fp32 head), optional TP collective.

    Forward: x and w rounded to fp16 (x before its all-gather), fp32 GEMM
    output, fp32 reduce-scatter / all-reduce. Backward: dgrad and wgrad GEMMs on
    bf16 casts with fp32 accumulation; dgrad returned in x's dtype, wgrad in the
    weight's dtype (Megatron DDP adds ``param.grad`` into the fp32 main grad).
    """

    @staticmethod
    def forward(ctx, x, weight, comm, group, precision=FP16_GEMM, scale_log2=0, site=None):  # noqa: D102
        # Down-projection prescale (``down_proj_prescale_log2``): x * 2^-s before
        # the fp16 cast and the fp32 output * 2^s; power-of-two scaling only
        # shifts exponents, so it widens fp16's range without changing rounding
        # (except for values that become fp16 subnormals).
        if scale_log2:
            x = x * (2.0**-scale_log2)
        if _OPERAND_DTYPE[precision] == torch.float16:
            x16 = _to_fp16(x, site)
        else:
            x16 = x.to(_OPERAND_DTYPE[precision])
        if comm == COMM_SP_GATHER:
            x16 = _gather_rows(x16, group)
        out = _GEMMS[precision](x16, weight)
        if scale_log2:
            out = out * (2.0**scale_log2)
        if comm == COMM_SP_SCATTER:
            out = _reduce_scatter_rows(out, group)
        elif comm == COMM_TP_ALLREDUCE:
            out = _all_reduce(out, group)
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            ctx.save_for_backward(x16, weight)
        ctx.x_dtype = x.dtype
        ctx.comm = comm
        ctx.group = group
        ctx.scale_log2 = scale_log2
        return out

    @staticmethod
    def backward(ctx, grad_out):  # noqa: D102
        x16, weight = ctx.saved_tensors
        g = grad_out
        if ctx.comm == COMM_SP_SCATTER:
            g = _gather_rows(g.contiguous(), ctx.group)
        n = weight.shape[0]
        g2 = g.reshape(-1, n).to(torch.bfloat16).contiguous()
        dgrad = wgrad = None
        if ctx.needs_input_grad[0]:
            w16 = weight.to(torch.bfloat16)
            dgrad = _mm(g2, w16, torch.float32).view(*x16.shape[:-1], weight.shape[1])
            if ctx.comm == COMM_SP_GATHER:
                dgrad = _reduce_scatter_rows(dgrad, ctx.group)
            elif ctx.comm == COMM_TP_COPY:
                dgrad = _all_reduce(dgrad, ctx.group)
            dgrad = dgrad.to(ctx.x_dtype)
        if ctx.needs_input_grad[1]:
            x2 = x16.reshape(-1, x16.shape[-1]).to(torch.bfloat16)
            wgrad = _mm(g2.t(), x2, torch.float32)
            if ctx.scale_log2:  # x16 holds x * 2^-s
                wgrad = wgrad * (2.0**ctx.scale_log2)
            wgrad = wgrad.to(weight.dtype)
        return dgrad, wgrad, None, None, None, None, None


def fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    comm: Optional[str] = COMM_NONE,
    group: Any = None,
    precision: str = FP16_GEMM,
    scale_log2: int = 0,
    site: Any = None,
) -> torch.Tensor:
    """Autograd-aware ``Fp16Linear``; plain forward under no_grad / inference.

    ``site``: key of the input's rounding site for sticky rounding (the module).
    """
    return Fp16Linear.apply(x, weight, comm, group, precision, scale_log2, site)


class RMSNormFp32(torch.autograd.Function):
    """y = x * rsqrt(mean(x^2) + eps) * w in fp32 (fp32 output).

    Forward on CUDA: the row mean is Megatron-Core's batch-invariant Triton
    ``mean_dim`` (one program per row, fixed reduction order), the same op
    sequence as MCore's ``BatchInvariantRMSNormFn``. Backward (fp32):
    dx = r * w * g - r^3 * x * sum(w * g * x) / D, dw = sum(g * x * r).
    (``BatchInvariantRMSNormFn.backward`` multiplies its second dx term by w
    once more, which is exact only for w == 1; measured on Qwen3-1.7B, whose
    q/k norm weights are far from 1, as a 100x wrong gradient norm. Fixed
    upstream by NVIDIA/Megatron-LM#6859; switch back to MCore's class once our
    Megatron-LM pin includes it.)
    """

    @staticmethod
    def forward(ctx, x, weight, eps, zero_centered_gamma):  # noqa: D102
        x32 = x.float()
        w32 = weight.float() + 1.0 if zero_centered_gamma else weight.float()
        sq = x32 * x32
        if x32.is_cuda:
            from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
                mean_dim,
            )

            ms = mean_dim(sq, dim=-1, keepdim=True)
        else:
            ms = sq.mean(-1, keepdim=True)
        rsigma = torch.rsqrt(ms + eps)
        ctx.save_for_backward(x, weight, rsigma)
        ctx.zero_centered_gamma = zero_centered_gamma
        return (x32 * rsigma) * w32

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        x, weight, r = ctx.saved_tensors
        g = grad.float()
        x32 = x.float()
        w32 = weight.float() + 1.0 if ctx.zero_centered_gamma else weight.float()
        d = x32.shape[-1]
        s = (g * w32 * x32).sum(-1, keepdim=True)
        dx = g * (w32 * r) - (r * r * r) * x32 * (s / d)
        dw = (g * x32 * r).reshape(-1, d).sum(0).to(weight.dtype)
        return dx.to(x.dtype), dw, None, None


def rms_norm_fp32(
    x: torch.Tensor, weight: torch.Tensor, eps: float, zero_centered_gamma: bool = False
) -> torch.Tensor:
    """RMSNorm of x in fp32 (fp32 output), weight upcast to fp32 (``RMSNormFp32``)."""
    return RMSNormFp32.apply(x, weight, eps, zero_centered_gamma)


class _GradScaleBox:
    """Scale shared by one attention forward's fp16 boundary ops."""

    __slots__ = ("scale",)

    def __init__(self) -> None:
        self.scale: Optional[torch.Tensor] = None


class ToFp16Scaled(torch.autograd.Function):
    """x -> fp16 (forward). Backward: grad / box.scale, in x's dtype."""

    @staticmethod
    def forward(ctx, x, box, site=None):  # noqa: D102
        ctx.x_dtype = x.dtype
        ctx.box = box
        return _to_fp16(x, site)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        if _DEBUG:
            amax = grad.abs().amax().item()
            if not amax < 32768:
                print(f"[fp16_gemm debug] fp16 attention grad amax={amax}", flush=True)
        g = grad.float()
        if ctx.box.scale is not None:
            g = g / ctx.box.scale
        return g.to(ctx.x_dtype), None, None


class FromFp16Scaled(torch.autograd.Function):
    """fp16 -> fp32 (exact, forward). Backward: grad * 2^e into fp16.

    2^e brings the gradient's amax to about 2^0 (see ``_ATTN_GRAD_TARGET_LOG2``)
    so FlashAttention's fp16 backward neither underflows nor overflows;
    ``ToFp16Scaled`` divides it out.
    """

    @staticmethod
    def forward(ctx, x16, box):  # noqa: D102
        ctx.box = box
        return x16.float()

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        g = grad.float()
        amax = g.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        exponent = torch.floor(_ATTN_GRAD_TARGET_LOG2 - torch.log2(amax)).clamp(-60, 60)
        scale = torch.exp2(exponent)
        ctx.box.scale = scale
        return (g * scale).to(torch.float16), None


# ---------------------------------------------------------------------------
# Megatron-Core / Transformer Engine patches.
# ---------------------------------------------------------------------------


def _tp(module: Any) -> tuple[int, Any]:
    group = getattr(module, "tp_group", None)
    if group is None:
        return 1, None
    size = torch.distributed.get_world_size(group)
    return size, group


def _te_linear_forward(module: Any, x: torch.Tensor, norm: bool):
    """Forward of a TE (or inference-optimized) linear under the fp16 GEMM.

    Returns (output, bias-or-None) as Megatron's TE wrappers do.
    """
    if getattr(module, "fp8", False) or getattr(
        module, "primary_weights_in_fp8", False
    ):
        raise NotImplementedError("forward_gemm_precision=fp16 does not support FP8.")
    weight = module.weight
    tp_size, group = _tp(module)
    sequence_parallel = (
        bool(getattr(module, "sequence_parallel", False)) and tp_size > 1
    )
    mode = getattr(module, "parallel_mode", None)
    if norm:
        if getattr(module, "normalization", "RMSNorm") != "RMSNorm":
            raise NotImplementedError(
                "forward_gemm_precision=fp16 supports RMSNorm only "
                f"(got {module.normalization})."
            )
        x = rms_norm_fp32(
            x,
            module.layer_norm_weight,
            module.eps,
            bool(getattr(module, "zero_centered_gamma", False)),
        )
    if mode == "column":
        comm = (
            COMM_SP_GATHER
            if sequence_parallel
            else (COMM_TP_COPY if tp_size > 1 else COMM_NONE)
        )
    elif mode == "row":
        comm = (
            COMM_SP_SCATTER
            if sequence_parallel
            else (COMM_TP_ALLREDUCE if tp_size > 1 else COMM_NONE)
        )
    else:
        comm = COMM_NONE
    scale = _STATE["down_scale"] if getattr(module, "_fp16_down_proj", False) else 0
    out = fp16_linear(x, weight, comm, group, scale_log2=scale, site=module)
    bias = getattr(module, "bias", None)
    has_bias = (
        bool(getattr(module, "use_bias", False)) and bias is not None and bias.numel()
    )
    if has_bias:
        if getattr(module, "te_return_bias", False):
            return out, bias
        out = out + bias.float()
    return out, None


def _make_te_forward(cls: type, norm: bool):
    orig = cls.__dict__.get("forward") or cls.forward
    _STATE["orig"][cls] = orig

    def forward(self, x, *args, **kwargs):
        if not _active(getattr(self, "weight", None)):
            return orig(self, x, *args, **kwargs)
        return _te_linear_forward(self, x, norm)

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.forward = forward


def _make_output_layer_forward(cls: type):
    """tensor_parallel.ColumnParallelLinear (the GPT LM head)."""
    orig = cls.__dict__["forward"]
    _STATE["orig"][cls] = orig

    def forward(self, input_, weight=None, runtime_gather_output=None):
        w = self.weight if weight is None else weight
        if not _active(w):
            return orig(
                self, input_, weight=weight, runtime_gather_output=runtime_gather_output
            )
        from megatron.core.tensor_parallel.mappings import (
            gather_from_tensor_model_parallel_region,
        )

        tp_size, group = _tp(self)
        if self.sequence_parallel and tp_size > 1:
            comm = COMM_SP_GATHER
        elif tp_size > 1 and not (
            getattr(self, "explicit_expert_comm", False)
            or getattr(self, "disable_grad_reduce", False)
        ):
            comm = COMM_TP_COPY
        else:
            comm = COMM_NONE
        out = fp16_linear(input_, w, comm, group, _STATE["head"], site=self)
        if self.bias is not None and not self.skip_bias_add:
            out = out + self.bias.float()
        gather = (
            self.gather_output
            if runtime_gather_output is None
            else runtime_gather_output
        )
        if gather and tp_size > 1:
            out = gather_from_tensor_model_parallel_region(out, group=group)
        return out, (self.bias if self.skip_bias_add else None)

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    cls.forward = forward


def _patch_te_rmsnorm() -> None:
    """TE RMSNorm modules (q/k norms, final norm): fp32 batch-invariant norm."""
    import transformer_engine.pytorch as te

    cls = te.RMSNorm
    orig = cls.forward
    _STATE["orig"][cls] = orig

    def forward(self, x, *args, **kwargs):
        if not _active(getattr(self, "weight", None)):
            return orig(self, x, *args, **kwargs)
        return rms_norm_fp32(
            x, self.weight, self.eps, bool(getattr(self, "zero_centered_gamma", False))
        )

    cls.forward = forward


def _cast_qkv(outputs: Any, box: Optional[_GradScaleBox], site: Any = None) -> Any:
    """q / k / v (and any other float output) to fp16; ``site``: the attention module."""

    def cast(t: Any, i: int) -> Any:
        if (
            not torch.is_tensor(t)
            or not t.is_floating_point()
            or t.dtype == torch.float16
        ):
            return t
        key = None if site is None else (site, i)
        if box is not None:
            return ToFp16Scaled.apply(t, box, key)
        return _to_fp16(t, key)

    if isinstance(outputs, tuple):
        return tuple(cast(t, i) for i, t in enumerate(outputs))
    return cast(outputs, 0)


def _patch_attention() -> None:
    """Q / k / v to fp16 after the q/k norms; scaled fp16 boundary for autograd."""
    from megatron.core.transformer.attention import Attention, SelfAttention

    orig_qkv = SelfAttention.get_query_key_value_tensors
    orig_post = Attention.forward_post_core_attn
    _STATE["orig"][(SelfAttention, "qkv")] = orig_qkv
    _STATE["orig"][(Attention, "post")] = orig_post

    def get_query_key_value_tensors(self, *args, **kwargs):
        outputs = orig_qkv(self, *args, **kwargs)
        if not _active(getattr(self.linear_qkv, "weight", None)):
            return outputs
        box = None
        if torch.is_grad_enabled():
            box = _GradScaleBox()
            self._fp16_gemm_box = box
        return _cast_qkv(outputs, box, self)

    def forward_post_core_attn(self, core_attn_out):
        box = self.__dict__.pop("_fp16_gemm_box", None)
        if (
            box is not None
            and torch.is_grad_enabled()
            and core_attn_out.dtype == torch.float16
            and core_attn_out.requires_grad
        ):
            core_attn_out = FromFp16Scaled.apply(core_attn_out, box)
        return orig_post(self, core_attn_out)

    SelfAttention.get_query_key_value_tensors = get_query_key_value_tensors
    Attention.forward_post_core_attn = forward_post_core_attn


def _patch_inference_pp_logits() -> None:
    """Engine PP > 1: keep last-stage logits fp32 and broadcast them in fp32.

    MCore's ``forward_pass_with_pipeline_parallel`` casts the last stage's
    logits to ``params_dtype`` and the controller broadcasts them to the other
    stages in ``params_dtype``; under the fp16 GEMM forward the logits are fp32
    (the training forward's), so both would round them to bf16.
    """
    from megatron.core.inference.model_inference_wrappers import (
        abstract_model_inference_wrapper as amw,
    )
    from megatron.core.inference.text_generation_controllers import (
        text_generation_controller as tgc,
    )

    cls = amw.AbstractModelInferenceWrapper
    orig_pp = cls.forward_pass_with_pipeline_parallel
    _STATE["orig"][(cls, "pp_forward")] = orig_pp

    def forward_pass_with_pipeline_parallel(self, *args, **kwargs):
        inner = self._forward
        held: dict[str, torch.Tensor] = {}

        def capture(*a, **k):
            out = inner(*a, **k)
            held["out"] = out
            return out

        self._forward = capture
        try:
            logits = orig_pp(self, *args, **kwargs)
        finally:
            del self._forward  # back to the class method
        out = held.get("out")
        if logits is not None and torch.is_tensor(out) and out.dtype == torch.float32:
            return out
        return logits

    cls.forward_pass_with_pipeline_parallel = forward_pass_with_pipeline_parallel

    orig_bcast = tgc.broadcast_from_last_pipeline_stage
    _STATE["orig"][(tgc, "broadcast")] = orig_bcast

    def broadcast_from_last_pipeline_stage(size, dtype, tensor=None, pp_group=None):
        # Every stage calls this with the same arguments; logits are fp32 here.
        if dtype in (torch.bfloat16, torch.float16):
            dtype = torch.float32
            if tensor is not None:
                tensor = tensor.float()
        return orig_bcast(size, dtype, tensor=tensor, pp_group=pp_group)

    tgc.broadcast_from_last_pipeline_stage = broadcast_from_last_pipeline_stage


def _patch_inference_buffers() -> None:
    """fp16 KV cache; fp32 logits buffer of the CUDA-graph text generation controller."""
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

    orig_alloc = DynamicInferenceContext._allocate_memory_buffer
    _STATE["orig"][(DynamicInferenceContext, "alloc")] = orig_alloc

    def _allocate_memory_buffer(self, *args, **kwargs):
        if getattr(self, "cache_mla_latent", False):
            return orig_alloc(self, *args, **kwargs)
        saved = self.params_dtype
        self.params_dtype = torch.float16
        try:
            return orig_alloc(self, *args, **kwargs)
        finally:
            self.params_dtype = saved

    DynamicInferenceContext._allocate_memory_buffer = _allocate_memory_buffer

    orig_init = TextGenerationController.__init__
    _STATE["orig"][(TextGenerationController, "init")] = orig_init

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        buf = getattr(self, "_all_logits_cuda", None)
        if torch.is_tensor(buf) and buf.dtype != torch.float32:
            self._all_logits_cuda = torch.zeros_like(buf, dtype=torch.float32)

    TextGenerationController.__init__ = __init__


def _patch_moe() -> None:
    """MoE: fp32 router logits, fp16 expert GEMMs with fp32 output, fp32 combine."""
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.transformer.moe import router as moe_router
    from megatron.core.transformer.moe import token_dispatcher_inference as tdi
    from megatron.core.transformer.moe.experts import InferenceGroupedMLP, TEGroupedMLP
    from megatron.core.transformer.moe.moe_layer import MoELayer

    from nemo_rl.models.megatron import fp16_moe

    # Router: fp32 logits from the fp32 hidden state, every product exact.
    orig_gating = moe_router.router_gating_linear
    _STATE["orig"][(moe_router, "router_gating_linear")] = orig_gating

    def router_gating_linear(inp, weight, bias, router_dtype):
        if not _active(weight):
            return orig_gating(inp, weight, bias, router_dtype)
        logits = fp16_linear(inp, weight, COMM_NONE, None, FP32_HEAD)
        if bias is not None:
            logits = logits + bias.float()
        return logits.to(router_dtype)

    moe_router.router_gating_linear = router_gating_linear

    # Engine dispatch: fp16 rows, bit-cast to bf16 for MCore's 16-bit all-gather
    # buffers (a copy, no arithmetic); training keeps fp32 rows (fp32 gradients).
    orig_preprocess = MoELayer.preprocess
    _STATE["orig"][(MoELayer, "preprocess")] = orig_preprocess

    def preprocess(self, hidden_states, probs, routing_map):
        if (
            _active(getattr(self.router, "weight", None))
            and InferenceMode.is_active()
            and hidden_states.dtype == torch.float32
        ):
            hidden_states = hidden_states.to(torch.float16).view(torch.bfloat16)
        elif _STATE["moe_preprocess"] is not None and not InferenceMode.is_active():
            # Sticky rounding: the dispatched copy's fc1-input rounding.
            hidden_states = _STATE["moe_preprocess"](
                self, hidden_states, probs, routing_map
            )
        return orig_preprocess(self, hidden_states, probs, routing_map)

    MoELayer.preprocess = preprocess

    orig_te_experts = TEGroupedMLP.forward
    _STATE["orig"][(TEGroupedMLP, "forward")] = orig_te_experts

    def te_experts_forward(self, x, tokens_per_expert, permuted_probs, *args, **kwargs):
        if not _active(getattr(self.linear_fc1, "weight0", None)) or args or kwargs:
            return orig_te_experts(
                self, x, tokens_per_expert, permuted_probs, *args, **kwargs
            )
        return fp16_moe.training_experts_forward(
            self, x, tokens_per_expert, permuted_probs
        )

    TEGroupedMLP.forward = te_experts_forward

    orig_inf_experts = InferenceGroupedMLP.forward
    _STATE["orig"][(InferenceGroupedMLP, "forward")] = orig_inf_experts

    def inf_experts_forward(
        self, x, tokens_per_expert, permuted_probs, routing_map=None, **kwargs
    ):
        if not InferenceMode.is_active() or not _active(
            getattr(self.linear_fc1, "weight0", None)
        ):
            return orig_inf_experts(
                self,
                x,
                tokens_per_expert,
                permuted_probs,
                routing_map=routing_map,
                **kwargs,
            )
        return fp16_moe.inference_experts_forward(self, x, permuted_probs, routing_map)

    InferenceGroupedMLP.forward = inf_experts_forward

    # Engine combine: keep the fp32 rank-ordered sum (MCore casts it to bf16).
    for cls, combine in (
        (tdi.NVLSAllGatherVDispatcher, _nvls_token_combine_fp32),
        (tdi.NCCLAllGatherDispatcher, _nccl_token_combine_fp32),
    ):
        orig = cls.token_combine
        _STATE["orig"][(cls, "token_combine")] = orig

        def token_combine(self, hidden_states, _orig=orig, _fp32=combine):
            if hidden_states.dtype != torch.float32:
                return _orig(self, hidden_states)
            return _fp32(self, hidden_states)

        cls.token_combine = token_combine


def _nvls_token_combine_fp32(self, hidden_states):
    """``NVLSAllGatherVDispatcher.token_combine`` without the final bf16 cast."""
    from megatron.core.inference.communication.torch_symm_triton import (
        multimem_reduce_scatter_v,
    )
    from megatron.core.inference.moe import batch_invariant
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        get_batch_invariant_collective,
    )

    if self.ep_size == 1:
        return hidden_states
    rsv = self.__class__._symm_rsv
    if hidden_states is not rsv["tensor"]:
        rsv["tensor"].copy_(hidden_states)
    output = torch.empty(
        self._local_tokens,
        hidden_states.shape[1],
        dtype=rsv["tensor"].dtype,
        device=hidden_states.device,
    )
    use_ordered = (
        batch_invariant.enabled() and get_batch_invariant_collective() == "ordered"
    )
    reduce_scatter_v = (
        batch_invariant.ordered_reduce_scatter_v
        if use_ordered
        else multimem_reduce_scatter_v
    )
    reduce_scatter_v(
        output,
        rsv["tensor"],
        rsv["handle"],
        rank_token_offset=self._rank_token_offset(),
        ep_max_tokens=self._ep_max_tokens(),
        per_rank_max_tokens=self._per_rank_worst_case_token_count,
    )
    return output


def _nccl_token_combine_fp32(self, hidden_states):
    """``NCCLAllGatherDispatcher.token_combine`` without the final bf16 cast.

    Its NCCL reduce-scatter sums EP partials in NCCL's order: rank-order (as in
    training) only up to EP=2; use the NVLS dispatcher (ordered) beyond that.
    """
    import torch.distributed as dist
    from megatron.core.tensor_parallel.mappings import (
        reduce_scatter_to_sequence_parallel_region,
    )
    from megatron.core.utils import get_pg_rank

    if self.ep_size == 1:
        return hidden_states
    if not self.__class__._use_allgather_v:
        return reduce_scatter_to_sequence_parallel_region(
            hidden_states, group=self.tp_ep_group
        )
    tokens_per_rank = self.__class__._local_tokens_per_rank
    max_tokens = max(tokens_per_rank)
    ep_rank = get_pg_rank(self.ep_group)
    padded = hidden_states.new_zeros(self.ep_size * max_tokens, hidden_states.shape[1])
    offset = 0
    for dst_rank, n_tokens in enumerate(tokens_per_rank):
        padded[dst_rank * max_tokens : dst_rank * max_tokens + n_tokens] = (
            hidden_states[offset : offset + n_tokens]
        )
        offset += n_tokens
    scattered = padded.new_empty(max_tokens, hidden_states.shape[1])
    dist.reduce_scatter_tensor(scattered, padded, group=self.ep_group)
    return scattered[: tokens_per_rank[ep_rank]]


def _tag_down_projections() -> None:
    """Mark every MLP's ``linear_fc2`` (dense down-projection) for the prescale."""
    from megatron.core.transformer.mlp import MLP

    orig = MLP.__init__
    _STATE["orig"][(MLP, "init")] = orig

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        fc2 = getattr(self, "linear_fc2", None)
        if fc2 is not None:
            fc2._fp16_down_proj = True

    MLP.__init__ = __init__


def install(
    lm_head_precision: Optional[str] = None,
    down_proj_prescale_log2: int = 0,
    expert_weight_cache: bool = True,
) -> None:
    """Install the fp16 GEMM forward for this process (idempotent).

    Call before the models are built and before the inference engine captures
    its CUDA graphs, in every training and generation worker.
    ``expert_weight_cache`` (``policy.megatron_cfg.fp16_expert_weight_cache``):
    the training model keeps fp16 copies of its local expert weights between
    optimizer steps (2 B per local expert parameter); False casts each expert's
    weight just in time in every forward instead (one expert's copy resident;
    same fp16 values, so bitwise the same outputs).
    """
    if _STATE["enabled"]:
        return
    from nemo_rl.models.megatron import fp16_moe

    fp16_moe.set_weight_cache(bool(expert_weight_cache))
    head = FP32_HEAD if lm_head_precision is None else lm_head_precision
    if head not in HEAD_PRECISIONS:
        raise ValueError(
            f"lm_head_precision must be one of {HEAD_PRECISIONS} (got {head!r})."
        )
    _STATE["head"] = head
    if not 0 <= int(down_proj_prescale_log2) <= 8:
        raise ValueError("down_proj_prescale_log2 must be in [0, 8].")
    _STATE["down_scale"] = int(down_proj_prescale_log2)
    from megatron.core.extensions import transformer_engine as mte
    from megatron.core.tensor_parallel import inference_layers as il
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    for cls, norm in (
        (mte.TELinear, False),
        (mte.TEColumnParallelLinear, False),
        (mte.TERowParallelLinear, False),
        (mte.TELayerNormColumnParallelLinear, True),
        (il.InferenceLinear, False),
        (il.InferenceColumnParallelLinear, False),
        (il.InferenceRowParallelLinear, False),
        (il.InferenceLayerNormColumnParallelLinear, True),
    ):
        _make_te_forward(cls, norm)
    _make_output_layer_forward(ColumnParallelLinear)
    _patch_te_rmsnorm()
    _patch_attention()
    _patch_inference_buffers()
    _patch_inference_pp_logits()
    _patch_moe()
    _tag_down_projections()
    _STATE["enabled"] = True
    print(
        "[fp16_gemm] forward_gemm_precision=fp16: fp16 GEMM operands, fp32 GEMM "
        f"outputs / norms / residual, fp16 attention and KV cache, {head} LM head",
        flush=True,
    )


def validate_config(
    megatron_cfg: dict[str, Any], inference_cfg: Optional[dict] = None
) -> list[str]:
    """Config violations of ``forward_gemm_precision`` (empty when unset)."""
    value = megatron_cfg.get("forward_gemm_precision")
    if value is None:
        if megatron_cfg.get("lm_head_precision") is not None:
            return [
                "policy.megatron_cfg.lm_head_precision requires "
                "forward_gemm_precision='fp16' (not built for the bf16 forward yet)."
            ]
        return []
    errors = []
    if value not in SUPPORTED_PRECISIONS:
        return [
            f"policy.megatron_cfg.forward_gemm_precision must be one of "
            f"{SUPPORTED_PRECISIONS} or unset (got {value!r})."
        ]
    if not megatron_cfg.get("fp32_residual_connection"):
        errors.append(
            "forward_gemm_precision='fp16' requires "
            "policy.megatron_cfg.fp32_residual_connection=true (fp32 residual stream)."
        )
    if (megatron_cfg.get("pipeline_model_parallel_size") or 1) > 1 and megatron_cfg.get(
        "pipeline_dtype"
    ) != "float32":
        errors.append(
            "forward_gemm_precision='fp16' with pipeline parallelism requires "
            "policy.megatron_cfg.pipeline_dtype='float32': stages exchange the fp32 "
            "residual stream, and Megatron's receive buffers use pipeline_dtype "
            f"(got {megatron_cfg.get('pipeline_dtype')!r})."
        )
    if megatron_cfg.get("gradient_accumulation_fusion"):
        errors.append(
            "forward_gemm_precision='fp16' requires "
            "policy.megatron_cfg.gradient_accumulation_fusion=false (its wgrad "
            "is returned as param.grad)."
        )
    if megatron_cfg.get("use_fused_linear_logprobs"):
        errors.append(
            "forward_gemm_precision='fp16' requires use_fused_linear_logprobs=false "
            "(the fused head does not use the fp16 GEMM)."
        )
    for key in ("fp8_cfg", "fp4_cfg"):
        cfg = megatron_cfg.get(key)
        enabled_flag = (
            cfg.get("enabled")
            if isinstance(cfg, dict)
            else getattr(cfg, "enabled", False)
        )
        if enabled_flag:
            errors.append(
                f"forward_gemm_precision='fp16' does not support {key}.enabled=true."
            )
    head = megatron_cfg.get("lm_head_precision")
    if head is not None and head not in HEAD_PRECISIONS:
        errors.append(
            f"policy.megatron_cfg.lm_head_precision must be one of {HEAD_PRECISIONS} "
            f"or unset (got {head!r})."
        )
    if inference_cfg is not None:
        for key in ("forward_gemm_precision", "lm_head_precision"):
            if inference_cfg.get(key) != megatron_cfg.get(key):
                errors.append(
                    f"{key} must be the same in training and generation "
                    f"(training: {megatron_cfg.get(key)!r}, generation: "
                    f"{inference_cfg.get(key)!r})."
                )
    return errors


def validate_model(model_cfg: Any) -> None:
    """Reject model architectures the fp16 GEMM forward does not cover yet."""
    unsupported = []
    if getattr(model_cfg, "num_moe_experts", None):
        from nemo_rl.models.megatron import fp16_moe

        unsupported += [f"MoE {x}" for x in fp16_moe.validate_model(model_cfg)]
    if getattr(model_cfg, "multi_latent_attention", False):
        unsupported.append("multi_latent_attention")
    if getattr(model_cfg, "normalization", "RMSNorm") != "RMSNorm":
        unsupported.append(f"normalization={model_cfg.normalization}")
    if getattr(model_cfg, "mtp_num_layers", None):
        unsupported.append("mtp_num_layers")
    if getattr(model_cfg, "attention_output_gate", False):
        unsupported.append("attention_output_gate")
    if unsupported:
        raise ValueError(
            "forward_gemm_precision='fp16' does not support: " + ", ".join(unsupported)
        )
