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

The fp16-operand forward, computed by the
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
    # Linear classes whose forward is the fp16 GEMM (``hold_fp16_weights``).
    "gemm_classes": (),
}


def set_sticky_hooks(site_cast: Optional[Any], moe_preprocess: Optional[Any]) -> None:
    """Install (or with None remove) sticky rounding's site hooks."""
    _STATE["site_cast"] = site_cast
    _STATE["moe_preprocess"] = moe_preprocess


# fp16 range monitor (silent_bug_sweep.md): activations above fp16's max (65504) become
# inf at the cast, on the learner and the engine alike (train == generation still holds
# bitwise, but the computed model changes silently). Off by default; a learner worker
# turns it on (``range_monitor(True)``). Counts accumulate on the device (no host sync)
# and are read once per step (``range_take``). Never inside a CUDA graph capture.
FP16_MAX = 65504.0
_RANGE: dict[str, Any] = {"on": False, "over": None, "checked": None}


def range_monitor(on: bool) -> None:
    _RANGE["on"] = bool(on)


# Elements per chunk of the torch fallback of range_count (CPU / non-contiguous inputs): the comparison's
# temporaries are per chunk (abs fp32 + bool mask + count_nonzero's int64 copy, ~13 B per element).
_RANGE_CHUNK_ELEMS = 1 << 22

_over_kernel = None


def _range_over_kernel():
    """Triton kernel: atomically add the count of |x| > FP16_MAX over a contiguous buffer (no temporaries)."""
    global _over_kernel
    if _over_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _k(x_ptr, n, out_ptr, thr, BLOCK: tl.constexpr):
            offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            c = tl.sum((tl.abs(v) > thr).to(tl.int32), axis=0)
            tl.atomic_add(out_ptr, c.to(tl.int64))

        _over_kernel = _k
    return _over_kernel


def range_count(x: torch.Tensor) -> None:
    """Count elements of ``x`` (about to be cast to fp16) outside fp16's finite range.

    Bounded memory: the count was ``count_nonzero(x.abs() > FP16_MAX)``, whose temporaries (an fp32 abs copy, a
    bool mask and count_nonzero's int64 copy of the mask, 2.25x an fp32 input) sat on top of the learner's
    training peak: at full Nano the MoE recompute's routed-expert activation (R_cap x 1856 fp32, 1.7 GiB) gave
    3.8 GiB of them, and the final50b resume OOMed allocating that int64 copy (3.40 GiB). Now one Triton pass with
    an integer atomic per block (CUDA, contiguous; exact count, order-independent), else a chunked torch count.
    """
    if not _RANGE["on"] or x.dtype == torch.float16 or not x.is_floating_point():
        return
    if x.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    if torch.is_inference_mode_enabled():
        # A colocated engine (inference mode) shares this process-wide monitor with the learner. The engine never
        # counts (as in the split layout, where the monitor is off in engine processes): a counter first created or
        # updated under inference mode is an inference tensor, and the learner's next in-place add outside it raises
        # "Inplace update to inference tensor outside InferenceMode" (ts-syncuni 458f498e).
        return
    with torch.no_grad():
        _range_counters(x.device)
        xd = x.detach()
        n = xd.numel()
        if n == 0:
            return
        if xd.is_cuda and xd.is_contiguous():
            block = 4096
            _range_over_kernel()[(triton_cdiv(n, block),)](xd, n, _RANGE["over"], FP16_MAX, BLOCK=block)
        else:
            flat = xd.reshape(-1) if xd.is_contiguous() else None
            chunks = flat.split(_RANGE_CHUNK_ELEMS) if flat is not None else _row_chunks(xd)
            for c in chunks:
                # Compared in fp32 (as the Triton pass): FP16_MAX is not a bf16 value (it rounds to 65536).
                _RANGE["over"] += torch.count_nonzero(c.float().abs() > FP16_MAX)
        _RANGE["checked"] += n


# The monitor counted in the cast itself (``cast_fp16_counted``): one pass reads x once, writes fp16(x) and adds the
# count. NEMO_RL_FP16_RANGE_FUSED=0 keeps the separate counting pass (A/B; same counts, same cast bits).
_RANGE["fused"] = os.environ.get("NEMO_RL_FP16_RANGE_FUSED", "1") != "0"
_cast_count_kernel = None


def _range_cast_kernel():
    """Triton kernel: y = fp16(x) over a contiguous buffer, plus range_count's two counters (integer atomics)."""
    global _cast_count_kernel
    if _cast_count_kernel is None:
        import triton
        import triton.language as tl

        @triton.jit
        def _cast_count(x_ptr, y_ptr, n, over_ptr, checked_ptr, thr, BLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
            m = offs < n
            # bf16 / fp32 -> fp32 is exact; fp32 -> fp16 rounds to nearest even (torch's .to(float16)).
            v = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
            tl.store(y_ptr + offs, v.to(tl.float16), mask=m)
            c = tl.sum((tl.abs(v) > thr).to(tl.int32), axis=0)
            if c > 0:
                tl.atomic_add(over_ptr, c.to(tl.int64))
            if pid == 0:
                tl.atomic_add(checked_ptr, tl.cast(n, tl.int64))

        _cast_count_kernel = _cast_count
    return _cast_count_kernel


def _range_counters(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if _RANGE["over"] is None or _RANGE["over"].device != device:
        with torch.inference_mode(False):  # never inference tensors (colocated engine, see range_count)
            _RANGE["over"] = torch.zeros((), dtype=torch.int64, device=device)
            _RANGE["checked"] = torch.zeros((), dtype=torch.int64, device=device)
    return _RANGE["over"], _RANGE["checked"]


def range_fusable(x: torch.Tensor) -> bool:
    """Whether ``x``'s range count can ride in its fp16 cast kernel (monitor on, CUDA, contiguous fp32 / bf16)."""
    return (
        _RANGE["on"]
        and _RANGE["fused"]
        and x.is_cuda
        and x.dtype in (torch.float32, torch.bfloat16)
        and x.is_contiguous()
        and x.numel() > 0
        and not torch.cuda.is_current_stream_capturing()
    )


def cast_fp16_counted(x: torch.Tensor, site: Any = None) -> torch.Tensor:
    """``range_count(x)`` then ``_to_fp16(x, site)``, in one kernel when ``range_fusable(x)``.

    The fused kernel reads x once (the counting pass was a second full read of every GEMM input: 73 launches per
    Super-L22 verify forward). Same counts (the same fp32 comparison, integer atomics: order-independent) and the
    same fp16 bits as ``x.to(torch.float16)``. Sticky rounding's site cast, CPU, non-contiguous, fp16 inputs, the
    monitor off and CUDA-graph capture take the unfused path (identical to before).
    """
    if (site is None or _STATE["site_cast"] is None) and range_fusable(x):
        with torch.no_grad():
            xd = x.detach()
            over, checked = _range_counters(xd.device)
            y = torch.empty(xd.shape, dtype=torch.float16, device=xd.device)
            n, block = xd.numel(), 4096
            _range_cast_kernel()[(triton_cdiv(n, block),)](xd, y, n, over, checked, FP16_MAX, BLOCK=block)
        return y
    range_count(x)
    return _to_fp16(x, site)


def triton_cdiv(a: int, b: int) -> int:
    return -(-a // b)


def _row_chunks(x: torch.Tensor):
    """Chunks of a non-contiguous tensor along dim 0 of about _RANGE_CHUNK_ELEMS elements (one row at least)."""
    if x.dim() == 0:
        return [x.reshape(1)]
    per_row = max(1, x[0].numel())
    return x.split(max(1, _RANGE_CHUNK_ELEMS // per_row), 0)


def range_take() -> dict[str, float]:
    """Counts since the last call (one host sync), then reset."""
    over, checked = _RANGE["over"], _RANGE["checked"]
    if over is None:
        return {"overflow": 0.0, "checked": 0.0}
    out = {"overflow": float(over.item()), "checked": float(checked.item())}
    over.zero_()
    checked.zero_()
    return out


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
    w16 = weight.to(torch.float16)  # no-op for engine weights held in fp16
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
    tracked = torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)
    if not x.is_cuda or tracked:
        x32 = x.float()
        hi = x32.to(torch.float16)
        lo = ((x32 - hi.float()) * _SPLIT_SCALE).to(torch.float16)
        # One GEMM on the stacked [hi; lo] rows: the fp16 GEMM is row-invariant
        # (the batch-invariance property both sides rely on), so each half equals
        # the separate GEMM bitwise, with one weight read instead of two.
        m = hi.numel() // hi.shape[-1]
        both = fp16_gemm(torch.cat([hi.reshape(m, -1), lo.reshape(m, -1)]), weight)
        out = both[:m] + both[m:] / _SPLIT_SCALE
        return out.view(*x.shape[:-1], weight.shape[0])
    return _fp32_gemm_chunked(x, weight)


# Rows of x per step of the chunked fp32-input GEMM. Each step runs one GEMM on
# the stacked [hi; lo] chunk (2x this many rows), so its temporaries are bounded
# by 2 x this many [N]-wide fp32 rows, whatever M is.
_FP32_GEMM_CHUNK_ROWS = 2048


def _fp32_gemm_chunked(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``fp32_gemm`` with one [M, N] fp32 buffer: the output.

    The unchunked expression holds hi @ w.T, lo @ w.T, its scaled copy and the
    sum at once: 3-4 [M, N] fp32 buffers, which for the LM head are logits-sized
    (3 x 9.3 GiB for 16384 tokens x 151936 vocab). Here each row chunk runs one
    GEMM on the stacked [hi; lo] rows (one weight read: the engine's decode
    steps, M = rows in flight, are one chunk), scales the lo half in place and
    writes hi + lo into the output: the same fp32 operations on the same values,
    so bitwise the same result (row results do not depend on M: cuBLAS without
    split-K under the batch-invariant backend, see ``_mm``).
    """
    k, n = x.shape[-1], weight.shape[0]
    x2 = x.reshape(-1, k)
    w16t = weight.to(torch.float16).t()  # no-op for engine weights held in fp16
    out = torch.empty((x2.shape[0], n), dtype=torch.float32, device=x.device)
    from nemo_rl.models.megatron import fused_decode_ops as fdo

    fused = fdo.fp32_gemm_supported(x2, w16t)
    for start in range(0, x2.shape[0], _FP32_GEMM_CHUNK_ROWS):
        if fused:  # same split / GEMM / combine in 3 kernels (bitwise; fused_decode_ops)
            rows = x2[start : start + _FP32_GEMM_CHUNK_ROWS]
            fdo.fp32_gemm_chunk(rows, w16t, out[start : start + rows.shape[0]], _SPLIT_SCALE)
            continue
        x32 = x2[start : start + _FP32_GEMM_CHUNK_ROWS].float()
        c = x32.shape[0]
        hi = x32.to(torch.float16)
        lo = ((x32 - hi.float()) * _SPLIT_SCALE).to(torch.float16)
        both = torch.mm(torch.cat([hi, lo]), w16t, out_dtype=torch.float32)
        torch.add(both[:c], both[c:].div_(_SPLIT_SCALE), out=out[start : start + c])
    return out.view(*x.shape[:-1], n)


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


def hold_fp16_weights(model: Any) -> int:
    """Store the dense GEMM weights of an inference-only model in fp16 (in place).

    Every recipe GEMM rounds its weight to fp16 (``fp16_gemm``); holding the weight in
    fp16 makes that cast a no-op, so the engine does it once per weight update instead
    of on every call (bitwise the same operands; the cast costs 4-8x the decode GEMM).
    Like the engine's fp16 expert buffers: each parameter's ``.data`` is replaced by its
    fp16 copy, so the parameter keeps its identity, the bf16 storage is freed (same
    bytes), and every later write through the parameter (refit ``copy_``, momentum
    stage / load) rounds bf16 -> fp16 exactly as the per-call cast did. Covers the
    patched linear classes, the untied LM head and the MoE router; not embeddings, norms
    or a tied head (the embedding lookup reads the same tensor). Call only on a model
    that never trains, before its CUDA graphs are captured; refits must write through
    ``copy_`` (not a dtype-strict raw receive). Returns the number of converted weights.
    """
    from megatron.core.transformer.moe.router import Router

    classes = tuple(_STATE["gemm_classes"]) + (Router,)
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = 0
    with torch.no_grad():
        for chunk in chunks:
            for module in chunk.modules():
                if not isinstance(module, classes):
                    continue
                w = module.__dict__.get("_parameters", {}).get("weight")
                if w is None or w.dtype != torch.bfloat16:
                    continue
                w.data = w.data.to(torch.float16)
                n += 1
    return n


# ---------------------------------------------------------------------------
# Autograd: fp16 forward, bf16 backward.
# ---------------------------------------------------------------------------

# Collectives fused into the linear function (sequence dimension = dim 0).
COMM_NONE = None
COMM_SP_GATHER = "sp_gather"  # all-gather input rows; bwd: reduce-scatter dgrad
COMM_SP_SCATTER = "sp_scatter"  # reduce-scatter output rows; bwd: all-gather grad
COMM_TP_COPY = "tp_copy"  # input replicated; bwd: all-reduce dgrad
COMM_TP_ALLREDUCE = "tp_allreduce"  # all-reduce output; bwd: identity


def _gather_rows(x: torch.Tensor, group: Any, view_ok: bool = False) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import _gather_along_first_dim

    if _ENGINE_SYMM_TP["on"] and not torch.is_grad_enabled():
        from nemo_rl.models.megatron import engine_symm_tp

        if engine_symm_tp.active_for(group):
            y = engine_symm_tp.gather_rows(x, group, view_ok=view_ok)
            if y is not None:
                return y
    return _gather_along_first_dim(x, group)


def tp_reduce_mode(tp_size: int) -> str:
    """Forward row-parallel fp32 reduction: "ordered" or "stock" (NCCL reduce-scatter).

    ``NEMO_RL_TP_REDUCE`` = ordered | stock | auto (default). auto: ordered at TP >= 3.
    At TP = 2 a token's sum is one commutative IEEE addition, so NCCL's reduce-scatter
    and the ordered sum give the same bits; at TP >= 3 NCCL's order for a row depends on
    which rank owns the row (ring rotation), i.e. on the row's position in the batch,
    which differs between the learner ([S, B] rows) and the engine (flat decode rows).
    """
    mode = os.environ.get("NEMO_RL_TP_REDUCE", "auto")
    if mode not in ("auto", "ordered", "stock"):
        raise ValueError(f"NEMO_RL_TP_REDUCE must be auto|ordered|stock (got {mode!r})")
    if mode == "auto":
        return "ordered" if tp_size >= 3 else "stock"
    return mode


_FOLD = {"kernel": None}


def _fold_kernel():
    """Triton: out = ((p_0 + p_1) + p_2) + ... in fp32, one pass (None without Triton)."""
    if _FOLD["kernel"] is not None:
        return _FOLD["kernel"] or None
    try:
        import triton
        import triton.language as tl
    except ImportError:  # pragma: no cover
        _FOLD["kernel"] = False
        return None

    @triton.jit
    def _fold(src, dst, n, TP: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.load(src + offs, mask=m)
        for j in tl.static_range(1, TP):
            acc = acc + tl.load(src + j * n + offs, mask=m)  # add.rn.f32, rank order
        tl.store(dst + offs, acc, mask=m)

    def run(parts: torch.Tensor) -> torch.Tensor:
        tp = parts.shape[0]
        out = torch.empty(parts.shape[1:], dtype=parts.dtype, device=parts.device)
        n = out.numel()
        block = 4096
        _fold[(triton.cdiv(n, block),)](parts, out, n, TP=tp, BLOCK=block, num_warps=8)
        return out

    _FOLD["kernel"] = run
    return run


def _fold_rank_order(parts: torch.Tensor) -> torch.Tensor:
    """parts [TP, ...] -> ((parts[0] + parts[1]) + parts[2]) + ... (bitwise the eager left fold)."""
    tp = parts.shape[0]
    if (
        parts.is_cuda
        and parts.dtype == torch.float32
        and parts.is_contiguous()
        and os.environ.get("NEMO_RL_TP_REDUCE_FUSED", "1") != "0"
    ):
        k = _fold_kernel()
        if k is not None:
            return k(parts)
    out = parts[0] + parts[1]
    for j in range(2, tp):
        out.add_(parts[j])
    return out


def ordered_reduce_scatter_rows(x: torch.Tensor, group: Any) -> torch.Tensor:
    """Reduce-scatter along dim 0 with a fixed summation order (rank 0, 1, ..., TP-1).

    One all-to-all of the partials (rank j receives every rank's slice j: the same
    bytes a ring reduce-scatter moves, (TP-1)/TP of the input per rank), then a left
    fold ((p_0 + p_1) + p_2) + ... in the input dtype. Every output row's bits depend
    only on the TP partials of that row, never on which rank owns it or on the row
    count, so the learner and the engine (and any batch layout) get the same sums.
    Per-rank cost: one receive buffer of the input's size (transient) + one fold pass over
    the T/TP owned rows (a Triton kernel reading the TP partials once; eager adds without
    Triton or with NEMO_RL_TP_REDUCE_FUSED=0, same bits). CUDA-graph capturable.
    """
    tp = torch.distributed.get_world_size(group)
    if tp == 1:
        return x
    x = x.contiguous()
    assert x.shape[0] % tp == 0, "first dimension must be divisible by TP"
    recv = torch.empty_like(x)
    torch.distributed.all_to_all_single(recv, x, group=group)
    parts = recv.view(tp, x.shape[0] // tp, *x.shape[1:])
    return _fold_rank_order(parts)


# Engine processes only (engine_symm_tp.install via enable_engine_symm_tp): one-shot ordered TP collectives.
_ENGINE_SYMM_TP = {"on": False}


def enable_engine_symm_tp(group: Any) -> bool:
    """Route this process's no-grad sequence-parallel row collectives through engine_symm_tp (same bits)."""
    from nemo_rl.models.megatron import engine_symm_tp

    _ENGINE_SYMM_TP["on"] = engine_symm_tp.install(group)
    return _ENGINE_SYMM_TP["on"]


def _reduce_scatter_rows(x: torch.Tensor, group: Any) -> torch.Tensor:
    from megatron.core.tensor_parallel.mappings import _reduce_scatter_along_first_dim

    if _ENGINE_SYMM_TP["on"] and not torch.is_grad_enabled():
        from nemo_rl.models.megatron import engine_symm_tp

        ws = torch.distributed.get_world_size(group)
        # The symmetric RS folds in rank order (= "ordered"); at TP >= 3 with NEMO_RL_TP_REDUCE=stock the learner
        # reduces in NCCL's order, so the engine must too (R27 review).
        if engine_symm_tp.active_for(group) and (ws <= 2 or tp_reduce_mode(ws) == "ordered"):
            y = engine_symm_tp.reduce_scatter_rows(x, group)
            if y is not None:
                return y

    if tp_reduce_mode(torch.distributed.get_world_size(group)) == "ordered":
        return ordered_reduce_scatter_rows(x, group)
    return _reduce_scatter_along_first_dim(x, group)


def _gather_rows_dim0(x: torch.Tensor, group: Any) -> torch.Tensor:
    x = x.contiguous()
    tp = torch.distributed.get_world_size(group)
    if x.is_cuda:
        out = torch.empty((tp * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
        torch.distributed.all_gather_into_tensor(out, x, group=group)
        return out
    parts = [torch.empty_like(x) for _ in range(tp)]
    torch.distributed.all_gather(parts, x, group=group)
    return torch.cat(parts)


def _all_reduce(x: torch.Tensor, group: Any) -> torch.Tensor:
    tp = torch.distributed.get_world_size(group)
    if tp_reduce_mode(tp) == "ordered" and x.shape[0] % tp == 0:
        # Ordered all-reduce = ordered reduce-scatter + all-gather (a copy).
        return _gather_rows_dim0(ordered_reduce_scatter_rows(x, group), group)
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
    def forward(
        ctx, x, weight, comm, group, precision=FP16_GEMM, scale_log2=0, site=None,
        fuse_wgrad=False,
    ):  # noqa: D102
        # Down-projection prescale (``down_proj_prescale_log2``): x * 2^-s before
        # the fp16 cast and the fp32 output * 2^s; power-of-two scaling only
        # shifts exponents, so it widens fp16's range without changing rounding
        # (except for values that become fp16 subnormals).
        if scale_log2:
            x = x * (2.0**-scale_log2)
        if _OPERAND_DTYPE[precision] == torch.float16:
            x16 = cast_fp16_counted(x, site)
        else:
            if site is not None:  # a raised hi/lo GEMM input: its hi part is fp16 too
                range_count(x)
            x16 = x.to(_OPERAND_DTYPE[precision])
        if comm == COMM_SP_GATHER:
            # Consumed by the GEMM below only: the engine's symmetric gather may hand back its staging.
            x16 = _gather_rows(x16, group, view_ok=not (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]))
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
        ctx.fuse_wgrad = fuse_wgrad
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
            from nemo_rl.models.megatron import fused_learner_ops as flo

            if (ctx.fuse_wgrad or flo.wgrad_fp32()) and flo.wgrad_fusable((weight,), wgrad):
                # main_grad += float(bf16(wgrad)) in one kernel (bitwise the DDP hook's add).
                wgrad = flo.accumulate_wgrad(wgrad, (weight,))[0]
            else:
                wgrad = wgrad.to(weight.dtype)
        return dgrad, wgrad, None, None, None, None, None, None


def fp16_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    comm: Optional[str] = COMM_NONE,
    group: Any = None,
    precision: str = FP16_GEMM,
    scale_log2: int = 0,
    site: Any = None,
    fuse_wgrad: bool = False,
) -> torch.Tensor:
    """Autograd-aware ``Fp16Linear``; plain forward under no_grad / inference.

    ``site``: key of the input's rounding site for sticky rounding (the module).
    ``fuse_wgrad``: the weight is used once per backward (a TE linear): its gradient may
    go straight into Megatron's fp32 ``main_grad`` (``fused_learner_ops``; same bits).
    """
    return Fp16Linear.apply(x, weight, comm, group, precision, scale_log2, site, fuse_wgrad)


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
        from nemo_rl.models.megatron import fused_learner_ops as flo

        if flo.wgrad_fp32() and getattr(weight, "main_grad", None) is not None:
            # fp32 weight-gradient accumulation (flo.wgrad_fp32): dw into main_grad without bf16 rounding.
            g = grad.float()
            x32 = x.float()
            w32 = weight.float() + 1.0 if ctx.zero_centered_gamma else weight.float()
            d = x32.shape[-1]
            s_ = (g * w32 * x32).sum(-1, keepdim=True)
            dx = g * (w32 * r) - (r * r * r) * x32 * (s_ / d)
            dw32 = (g * x32 * r).reshape(-1, d).sum(0)
            if flo.wgrad_fusable((weight,), dw32):
                return dx.to(x.dtype), flo.accumulate_wgrad(dw32, (weight,))[0], None, None
            return dx.to(x.dtype), dw32.to(weight.dtype), None, None
        if flo.rms_bwd_supported(grad, x, weight, r):  # same ops, 2 fused kernels (bitwise)
            dx, dw = flo.rms_norm_fp32_backward(grad, x, weight, r, ctx.zero_centered_gamma)
            return dx, dw, None, None
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
    """RMSNorm of x in fp32 (fp32 output), weight upcast to fp32 (``RMSNormFp32``).

    Without autograd (engine and verification forwards) one fused kernel computes the
    same op sequence bitwise (``fused_decode_ops.rms_norm_fp32_nograd``).
    """
    if not (torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad)):
        from nemo_rl.models.megatron import fused_decode_ops as fdo

        if fdo.rms_norm_fp32_supported(x, weight, eps):
            return fdo.rms_norm_fp32_nograd(x, weight, eps, zero_centered_gamma)
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
        from nemo_rl.models.megatron import bi_context_parallel as _cpx

        if _cpx.size() > 1:
            # Context parallel: the attention backward runs on whole-sequence head slices assembled from
            # every CP rank's dO chunk, so the scale must be one for the whole sequence (a per-chunk scale
            # mixed scales inside one head's dK / dV sums: qkv gradients off by ~9% vs CP 1).
            torch.distributed.all_reduce(amax, op=torch.distributed.ReduceOp.MAX, group=_cpx.group())
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
    if getattr(module, "_ts_latent_shard", None) is not None:
        # TP-sharded latent MoE projection (latent_tp_shard.py): the duplicated layer's bits.
        from nemo_rl.models.megatron import latent_tp_shard

        return latent_tp_shard.linear_forward(module, x), None
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
    # precision_raise.py: a tagged linear takes its input as an fp32 hi/lo split
    # (fp32_gemm, ~22 bits) instead of one fp16 rounding; same call on both sides.
    # fuse_wgrad (fused_learner_ops): the wgrad is computed from bf16 operands in both
    # precisions, so the fused main_grad accumulate applies to raised sites too.
    precision = FP32_HEAD if getattr(module, "_ts_hilo_input", False) else FP16_GEMM
    out = fp16_linear(
        x, weight, comm, group, precision, scale_log2=scale, site=module, fuse_wgrad=True
    )
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
        if (
            _STATE.get("deferred_head")
            and comm == COMM_NONE
            and _STATE["head"] == FP32_HEAD
            and (
                (torch.is_grad_enabled() and w.requires_grad)
                # Verification forwards (no autograd) that consume log-probs in
                # chunks (deferred_head.verification_logits); never for a batch.
                or (_STATE.get("deferred_head_nograd") and not torch.is_grad_enabled())
            )
            and self.bias is None
        ):
            # Training (TP=1): the loss needs only the target log-probs, so the
            # logits are formed chunk by chunk by the log-prob op instead of for
            # the whole micro-batch here (deferred_head.py).
            from nemo_rl.models.megatron.deferred_head import deferred_head_logits

            return deferred_head_logits(input_, w), None
        if (
            _STATE.get("deferred_head")
            and _STATE.get("deferred_head_nograd")
            and not torch.is_grad_enabled()
            and comm in (COMM_TP_COPY, COMM_SP_GATHER)
            and _STATE["head"] == FP32_HEAD
            and self.bias is None
            and not (self.gather_output if runtime_gather_output is None else runtime_gather_output)
        ):
            # Verification forward at TP > 1 (no autograd): the head input of this rank's vocabulary shard, exactly
            # as fp16_linear forms it (fp32 operand, sequence-parallel rows all-gathered), instead of the shard's
            # [tokens, V/TP] fp32 logits. A row's logits are fp32_gemm(x_row, w) later, bitwise (row-invariant GEMM;
            # the gather is a copy); the TP copy is the identity without autograd.
            from nemo_rl.models.megatron.deferred_head import deferred_head_logits

            x = input_.to(_OPERAND_DTYPE[FP32_HEAD])
            if comm == COMM_SP_GATHER:
                x = _gather_rows(x, group)
            return deferred_head_logits(x, w), None
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
    """Q / k / v (and any other float output) to fp16; ``site``: the attention module."""

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


def _patch_mamba_layer_input() -> None:
    """Mamba layers: keep the fp32 residual as the input of the fused pre-norm.

    Stock ``MambaLayer._prepare_mixer_input`` casts the layer input to
    ``config.params_dtype`` before ``in_proj``'s fused norm. With bf16 params
    that rounds the fp32 residual stream to bf16 (8-bit mantissa) at every
    Mamba layer, against this recipe's contract (fp32 norm on the fp32
    residual, as for attention and MoE layers). Training and engine both run
    ``_prepare_mixer_input``, so they stay identical; the patched layer feeds
    the fp32 residual to the patched fused norm + fp16 GEMM instead.
    """
    from megatron.core.ssm import mamba_layer as ml

    cls = ml.MambaLayer
    orig = cls._prepare_mixer_input
    _STATE["orig"][(cls, "prepare")] = orig
    # Megatron-LM main split the residual out (``_prepare_residual``): ``_prepare_mixer_input`` returns the
    # normalized branch input only. Older Megatron-Core returns ``(normalized input, residual)``.
    returns_residual = not hasattr(cls, "_prepare_residual")

    def _prepare_mixer_input(self, hidden_states):
        in_proj = getattr(getattr(self, "mixer", None), "in_proj", None)
        if (
            not _active(getattr(in_proj, "weight", None))
            or not self.config.fp32_residual_connection
            or hidden_states.dtype != torch.float32
        ):
            return orig(self, hidden_states)
        normed = ml.apply_module(self.norm)(hidden_states)
        return (normed, hidden_states) if returns_residual else normed

    cls._prepare_mixer_input = _prepare_mixer_input


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

    def dispatched_rows(layer, hidden_states, probs, routing_map):
        """The expert fc1 input rows in token space, before MCore's dispatch."""
        if (
            _active(getattr(layer.router, "weight", None))
            and InferenceMode.is_active()
            and hidden_states.dtype == torch.float32
        ):
            return hidden_states.to(torch.float16).view(torch.bfloat16)
        if _STATE["moe_preprocess"] is not None and not InferenceMode.is_active():
            # Sticky rounding: the dispatched copy's fc1-input rounding.
            return _STATE["moe_preprocess"](layer, hidden_states, probs, routing_map)
        return hidden_states

    def preprocess(self, hidden_states, probs, routing_map, *args, **kwargs):
        # *args / **kwargs: Megatron-LM main passes padding_mask and packed_seq_params through to the dispatcher.
        if getattr(self.config, "moe_latent_size", None):
            # Latent MoE (Nemotron Super / Ultra): MCore's preprocess projects the
            # hidden states to the latent (fc1_latent_proj: a "duplicated" TE /
            # inference linear, so the fp16 GEMM with fp32 output above) and then
            # dispatches the latent rows. The fc1 input of the experts is the
            # projection's output, so the engine's fp16 cast and sticky rounding
            # apply to it (a forward hook on the projection), not to the hidden.
            _latent_hook(self)
            self._fp16_latent_ctx = (probs, routing_map)
            try:
                return orig_preprocess(self, hidden_states, probs, routing_map, *args, **kwargs)
            finally:
                self._fp16_latent_ctx = None
        return orig_preprocess(
            self, dispatched_rows(self, hidden_states, probs, routing_map), probs, routing_map, *args, **kwargs
        )

    def _latent_hook(layer):
        if getattr(layer, "_fp16_latent_hooked", False):
            return

        def hook(module, args, output):
            ctx = getattr(layer, "_fp16_latent_ctx", None)
            if ctx is None:  # the projection called outside preprocess: no dispatch
                return output
            y, bias = output
            return dispatched_rows(layer, y, ctx[0], ctx[1]), bias

        layer.fc1_latent_proj.register_forward_hook(hook)
        layer._fp16_latent_hooked = True

    MoELayer.preprocess = preprocess

    orig_te_experts = TEGroupedMLP.forward
    _STATE["orig"][(TEGroupedMLP, "forward")] = orig_te_experts

    def te_experts_forward(self, x, tokens_per_expert, permuted_probs, *args, **kwargs):
        if not _active(getattr(self.linear_fc1, "weight0", None)):
            return orig_te_experts(
                self, x, tokens_per_expert, permuted_probs, *args, **kwargs
            )
        if args or kwargs:
            # Used to fall back to the stock bf16 experts silently while the
            # engine stays fp16 (train != generation numerics).
            raise NotImplementedError(
                "fp16 GEMM training experts: TEGroupedMLP.forward got extra arguments "
                f"({len(args)} positional, keys {sorted(kwargs)}) the fp16 path does not "
                "handle (e.g. NCCL-EP zero-copy buffers); use the alltoall dispatcher"
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
            if hidden_states.dtype == torch.float64:  # EP-invariant limb partials
                return _limb_token_combine(self, hidden_states)
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


def _limb_token_combine(self, limbs):
    """EP-invariant combine (ep_invariant_combine.py): exact limb sum over EP ranks -> fp32.

    ``limbs``: [global tokens, 2H] fp64 partials of this rank's experts. The cross-rank
    sum is exact, so the NVLS kernel and NCCL may add ranks in any order.
    """
    import torch.distributed as dist
    from megatron.core.transformer.moe import token_dispatcher_inference as tdi
    from megatron.core.utils import get_pg_rank

    from nemo_rl.models.megatron import ep_invariant_combine as eic

    hidden = limbs.shape[1] // 2
    if isinstance(self, tdi.NVLSAllGatherVDispatcher):
        cls = self.__class__
        buf = eic.engine_limb_buffer(cls, hidden)
        if limbs.data_ptr() != buf.data_ptr():
            buf[: limbs.shape[0]].copy_(limbs)
        output = torch.empty(
            self._local_tokens, hidden, dtype=torch.float32, device=limbs.device
        )
        return eic.limbs_reduce_scatter_v(
            output,
            cls._symm_rsv["handle"],
            rank_token_offset=self._rank_token_offset(),
            ep_max_tokens=self._ep_max_tokens(),
            per_rank_max_tokens=self._per_rank_worst_case_token_count,
        )
    # NCCL all-gather dispatcher: reduce-scatter the fp64 limbs (any order is exact).
    if not self.__class__._use_allgather_v:  # CUDA-graph path: equal counts per rank
        group = self.tp_ep_group
        scattered = limbs.new_empty(limbs.shape[0] // group.size(), limbs.shape[1])
        dist.reduce_scatter_tensor(scattered, limbs.contiguous(), group=group)
        return eic.finalize_limbs(scattered)
    tokens_per_rank = self.__class__._local_tokens_per_rank
    max_tokens = max(tokens_per_rank)
    ep_rank = get_pg_rank(self.ep_group)
    padded = limbs.new_zeros(self.ep_size * max_tokens, limbs.shape[1])
    offset = 0
    for dst_rank, n_tokens in enumerate(tokens_per_rank):
        padded[dst_rank * max_tokens : dst_rank * max_tokens + n_tokens] = limbs[
            offset : offset + n_tokens
        ]
        offset += n_tokens
    scattered = padded.new_empty(max_tokens, limbs.shape[1])
    dist.reduce_scatter_tensor(scattered, padded, group=self.ep_group)
    return eic.finalize_limbs(scattered[: tokens_per_rank[ep_rank]])


def _nccl_token_combine_fp32(self, hidden_states):
    """``NCCLAllGatherDispatcher.token_combine`` without the final bf16 cast.

    The partials are summed in rank order (``ordered_reduce_scatter_rows``: all-to-all +
    left fold) whenever the group has 3+ ranks; NCCL's reduce-scatter order depends on
    which rank owns a row there (at 2 ranks the one addition is commutative, same bits).
    """
    import torch.distributed as dist
    from megatron.core.tensor_parallel.mappings import (
        reduce_scatter_to_sequence_parallel_region,
    )
    from megatron.core.utils import get_pg_rank

    if self.ep_size == 1:
        return hidden_states
    if not self.__class__._use_allgather_v:
        group = self.tp_ep_group
        if tp_reduce_mode(dist.get_world_size(group)) == "ordered":
            return ordered_reduce_scatter_rows(hidden_states, group)
        return reduce_scatter_to_sequence_parallel_region(hidden_states, group=group)
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
    if tp_reduce_mode(self.ep_size) == "ordered":
        return ordered_reduce_scatter_rows(padded, self.ep_group)[: tokens_per_rank[ep_rank]]
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
    mamba_fp32_prenorm_input: bool = True,
) -> None:
    """Install the fp16 GEMM forward for this process (idempotent).

    Call before the models are built and before the inference engine captures
    its CUDA graphs, in every training and generation worker.
    ``mamba_fp32_prenorm_input`` (``policy.megatron_cfg.mamba_fp32_prenorm_input``,
    default True): Mamba layers feed the fp32 residual to their fused pre-norm
    (_patch_mamba_layer_input). False keeps stock MCore's cast to params_dtype
    (the pre-fix numerics; a control arm only). Must match between training and
    generation.
    """
    if _STATE["enabled"]:
        return
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
    _STATE["gemm_classes"] = (
        mte.TELinear,
        mte.TEColumnParallelLinear,
        mte.TERowParallelLinear,
        mte.TELayerNormColumnParallelLinear,
        il.InferenceLinear,
        il.InferenceColumnParallelLinear,
        il.InferenceRowParallelLinear,
        il.InferenceLayerNormColumnParallelLinear,
        ColumnParallelLinear,
    )
    _patch_te_rmsnorm()
    _patch_attention()
    if mamba_fp32_prenorm_input:
        _patch_mamba_layer_input()
    _patch_inference_buffers()
    _patch_inference_pp_logits()
    _patch_moe()
    _tag_down_projections()
    # Bridge mapping of the latent MoE projections (engine InferenceLinear at TP > 1; sharded ones).
    from nemo_rl.models.megatron import latent_tp_shard

    latent_tp_shard.patch_bridge_mapping()
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
        if megatron_cfg.get("moe_latent_tp_shard"):
            return [
                "policy.megatron_cfg.moe_latent_tp_shard requires "
                "forward_gemm_precision='fp16' (latent_tp_shard.py runs the recipe's GEMM)."
            ]
        if megatron_cfg.get("lm_head_precision") is not None:
            return [
                "policy.megatron_cfg.lm_head_precision requires "
                "forward_gemm_precision='fp16' (not built for the bf16 forward yet)."
            ]
        return []
    errors = []
    if (
        megatron_cfg.get("moe_latent_tp_shard")
        and (megatron_cfg.get("tensor_model_parallel_size") or 1) > 1
        and not megatron_cfg.get("sequence_parallel")
    ):
        # The shard's backward sums the full-weight wgrad over TP (latent_tp_shard._ShardedLinear),
        # which is the duplicated layer's gradient only when every TP rank holds distinct rows (SP).
        # Without SP the rows are replicated over TP and the latent gradient came out TP x too large.
        errors.append(
            "policy.megatron_cfg.moe_latent_tp_shard with tensor_model_parallel_size > 1 requires "
            "sequence_parallel=true (without it the latent projections' gradient is summed over "
            "replicated rows: TP x too large)."
        )
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


def validate_model(model_cfg: Any, ep_invariant_combine: bool = False) -> None:
    """Reject model architectures the fp16 GEMM forward does not cover yet."""
    unsupported = []
    if getattr(model_cfg, "num_moe_experts", None):
        from nemo_rl.models.megatron import fp16_moe

        unsupported += [f"MoE {x}" for x in fp16_moe.validate_model(model_cfg)]
        if ep_invariant_combine:
            from nemo_rl.models.megatron import ep_invariant_combine as eic

            unsupported += [
                f"moe_ep_invariant_combine: {x}"
                for x in eic.check_exact_range(
                    int(getattr(model_cfg, "moe_router_topk", 1) or 1),
                    int(
                        getattr(model_cfg, "moe_ffn_hidden_size", None)
                        or getattr(model_cfg, "ffn_hidden_size", 1)
                    ),
                    int(_STATE["down_scale"]),
                )
            ]
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
