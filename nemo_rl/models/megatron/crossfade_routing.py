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
"""Midpoint cross-fade MoE routing: a θ-only routing rule (no drafter anchors).

The policy routes each token at each MoE layer as a deterministic function of
the current weights' own router logits, continuous in the scores, so a draft
from θ_k verified against θ_{k+1} needs no route anchors and a rejected row can
continue as a plain sample of θ_{k+1} (the research notes, summarized in docs/colo/
BUILD_DESIGN.md §1, §5). Rule, per token (fp32 throughout, op order fixed):

    s = sigmoid(x);  t = s + b                       (x router logits, b expert bias)
    idx = the C = k + c largest t, ties to the lower expert id;  v = t[idx]
    tau = (v[k-1] + v[k]) * 0.5
    c_j = clamp((v_j - tau) / m + 0.5, 0, 1)
    w_j = s[idx_j] * c_j;  W = w_0 + w_1 + ... + w_{C-1}  (sequential)
    p_j = w_j / (W + 1e-20) * scaling_factor

The k stock experts always have c >= 0.5; the c extra candidates enter with
weight 0 outside the band. Dispatch is fixed-width: every token is routed to
its C candidates (zero-weight ones contribute exactly 0), so every static
shape in MCore (num_out_tokens = T x topk, inference buffers, CUDA graphs)
holds with ``moe_router_topk = k + c``. m -> 0 selects the stock top-k.

This module is the unfused reference (also the correctness oracle for the
fused kernel, the research notes, summarized in docs/colo/). Each step is a
row-wise elementwise op or an exact selection, so a row's bits depend only
on its own logits: batch invariant by construction, on the learner and the
inference engine alike.

Config (``policy.megatron_cfg``): ``crossfade_routing_m`` (None: off) and
``crossfade_routing_extra`` (c, default 4). ``_apply_moe_config`` sets
``moe_router_topk = k + c`` and calls ``install`` in every process that
builds a model (learner and engine).
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

_PATCH_ATTR = "_nrl_crossfade_routing_patch"

# Process-wide rule parameters (None: off). k = stock top-k, extra = c.
_STATE: dict[str, Any] = {
    "m": None, "k": None, "extra": None, "sparse": False, "ramp_stopgrad": False, "sparse_engine": False,
    # Per-layer band (crossfade_routing_m_profile): global layer number -> m; the router module running now.
    "layer_m": None, "cur_layer": None,
}
_TRACE_SEEN: dict[tuple, int] = {}


def _trace(event: str, **fields: Any) -> None:
    """THUNDERSYNC_DEBUG_CF_TRACE=<dir>: one JSON line per event to <dir>/cf_trace_<pid>.jsonl (debug only)."""
    d = os.environ.get("THUNDERSYNC_DEBUG_CF_TRACE")
    if not d:
        return
    import json
    import socket
    import traceback

    os.makedirs(d, exist_ok=True)
    rec = dict(event=event, pid=os.getpid(), host=socket.gethostname(), file=__file__,
               cvd=os.environ.get("CUDA_VISIBLE_DEVICES"), **fields)
    rec["stack"] = [f"{f.filename.split('/')[-1]}:{f.lineno}:{f.name}" for f in traceback.extract_stack()[-9:-2]]
    with open(os.path.join(d, f"cf_trace_{os.getpid()}.jsonl"), "a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def crossfade_m(config: Any) -> Optional[float]:
    """``policy.megatron_cfg.crossfade_routing_m`` (None: cross-fade routing off)."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    m = megatron_cfg.get("crossfade_routing_m")
    return None if m is None else float(m)


def crossfade_m_profile(config: Any) -> Optional[list[tuple[float, float, float]]]:
    """``policy.megatron_cfg.crossfade_routing_m_profile``: [[start, end, scale], ...] over the relative depth of the
    MoE layers (MoE ordinal j of n at depth j / n, start <= j / n < end; the last range includes 1). A layer's band is
    m x scale. None: the same m at every layer."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    prof = megatron_cfg.get("crossfade_routing_m_profile")
    if prof is None:
        return None
    out = [(float(a), float(b), float(c)) for a, b, c in prof]
    if not out or any(not c > 0 or not b > a for a, b, c in out):
        raise ValueError(f"crossfade_routing_m_profile: ranges [start, end, scale > 0] with end > start (got {prof})")
    return out


def layer_bands(m: float, profile, moe_layers: list[int]) -> dict[int, float]:
    """Global layer number -> band width for each MoE layer (``crossfade_m_profile``)."""
    n = len(moe_layers)
    out = {}
    for j, ln in enumerate(moe_layers):
        x = j / n
        hit = [c for a, b, c in profile if a <= x < b or (b >= 1.0 and x >= a)]
        if not hit:
            raise ValueError(f"crossfade_routing_m_profile does not cover MoE layer {j} of {n} (depth {x:.3f})")
        out[int(ln)] = float(m) * hit[0]
    return out


def moe_layer_numbers(model_cfg: Any) -> Optional[list[int]]:
    """1-based global layer numbers of the MoE layers of a hybrid model (pattern 'E'); None if not hybrid."""
    pattern = getattr(model_cfg, "hybrid_layer_pattern", None) or getattr(model_cfg, "hybrid_override_pattern", None)
    if not pattern:
        return None
    pattern = str(pattern).split("/")[0]
    return [i + 1 for i, ch in enumerate(pattern.replace("|", "")) if ch == "E"]


def crossfade_extra(config: Any) -> int:
    """``policy.megatron_cfg.crossfade_routing_extra``: candidates beyond the stock top-k."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    extra = megatron_cfg.get("crossfade_routing_extra")
    return 4 if extra is None else int(extra)


def crossfade_sparse(config: Any) -> bool:
    """``policy.megatron_cfg.crossfade_routing_sparse``: the learner dispatches only nonzero-weight slots."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    return bool(megatron_cfg.get("crossfade_routing_sparse") or False)


def crossfade_sparse_engine(config: Any) -> bool:
    """``policy.megatron_cfg.crossfade_routing_sparse_engine``: the engine's expert GEMMs skip zero-weight slots.

    The engine keeps the fixed-width [T, k + c] routing (static shapes, CUDA graphs); the expert path
    (``fp16_moe.inference_experts_forward``) maps a zero-weight slot's expert id to -1, which the live-row permute
    treats like a non-local expert. Same bits: such a slot contributes exactly +0 to its token's fp64 combine.
    """
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    return bool(megatron_cfg.get("crossfade_routing_sparse_engine") or False)


def engine_sparse_active() -> bool:
    """Cross-fade routing on and the engine zero-weight-slot skip enabled (this process)."""
    return _STATE.get("m") is not None and bool(_STATE.get("sparse_engine"))


def crossfade_ramp_stopgrad(config: Any) -> bool:
    """``policy.megatron_cfg.crossfade_routing_ramp_stopgrad``: the learner's gradient ignores the routing ramp.

    The forward (gates, log-probs, sampling) is unchanged bit for bit; only the backward drops the terms through the
    ramp c and the threshold tau, so d gates / d logits flows through the sigmoid scores of the selected candidates
    alone, as under stock / sticky routing, whose discrete selection carries no gradient. This is not the exact policy
    gradient of the cross-fade policy (that includes the ramp, slope 1/m): it is the sticky-like estimator. q40 / G1
    (10-28): the ramp path makes the RL gradient ~2.6x and slowed learning.
    """
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    return bool(megatron_cfg.get("crossfade_routing_ramp_stopgrad") or False)


def crossfade_enabled(config: Any) -> bool:
    return crossfade_m(config) is not None


def get_state() -> dict[str, Any]:
    return dict(_STATE)


def candidates(t: torch.Tensor, C: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The C largest selection scores per row, descending, ties to the lower id.

    A stable sort of -t keeps index order among equal values (exact; no
    arithmetic), so the result is deterministic and batch invariant.
    """
    vals, idx = torch.sort(-t, dim=1, stable=True)
    # Contiguous: [:, :C] of the [T, E] sort is a strided view, and the inference engine's MoE kernels (NVLS
    # dispatch, fused experts) read the dense [T, topk] routing tensors by raw pointer (a check_parallel_parity
    # run: a strided index tensor routed the engine to the wrong experts, 4 nats).
    return (-vals[:, :C]).contiguous(), idx[:, :C].contiguous()


def crossfade_route(
    logits: torch.Tensor,
    expert_bias: torch.Tensor,
    k: int,
    extra: int,
    m: float,
    scaling_factor: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gates [T, C] (fp32, zeros outside the band) and candidate ids [T, C] (int64).

    Differentiable in ``logits`` (the policy gradient flows through s, the ramp
    and tau). Reference op order: see the module docstring.
    """
    assert logits.dim() == 2, logits.shape
    E = logits.shape[1]
    C = min(E, k + extra)
    assert k >= 1 and C >= k, (k, extra, E)
    if _fused_ok(logits, k, C):
        return _CrossfadeFused.apply(logits.float().contiguous(), expert_bias, k, C, m, scaling_factor)
    s = torch.sigmoid(logits.float())
    t = s + expert_bias.float()
    # extra = 0 (debug width-k plumbing check): tau still needs the (k+1)-th score; only k columns are returned.
    stop = bool(_STATE.get("ramp_stopgrad"))
    v, idx = candidates(t.detach() if stop else t, max(C, k + 1))
    tau = (v[:, k - 1] + v[:, k]) * 0.5
    c = ((v - tau.unsqueeze(1)) * inv_m(m) + 0.5).clamp(0.0, 1.0)
    if stop:
        c = c.detach()  # same values: the forward is unchanged; the ramp / tau terms leave the gradient
    if os.environ.get("THUNDERSYNC_DEBUG_CF_STOCKWIDE"):
        # Debug only: stock top-C (every candidate at full weight, no zero gates) through the same widened dispatch.
        c = torch.ones_like(c)
    w = torch.gather(s, 1, idx) * c
    total = w[:, 0]
    for j in range(1, w.shape[1] if C > k else k):
        total = total + w[:, j]
    p = w / (total.unsqueeze(1) + 1e-20)
    if scaling_factor:
        p = p * scaling_factor
    return p[:, :C].contiguous(), idx[:, :C].contiguous()


def inv_m(m: float) -> float:
    """fp32(1 / m), rounded once from double: the ramp multiplies by it (the fused kernel does the same)."""
    import numpy as np

    return float(np.float32(1.0 / float(m)))


def _fused_ok(logits: torch.Tensor, k: int, C: int) -> bool:
    """The fused CUDA kernel (cf_router_fused, kernel task E) serves this call: CUDA, C <= 32, C > k, not disabled."""
    return (
        logits.is_cuda
        and C > k >= 1
        and C <= 32
        and os.environ.get("THUNDERSYNC_CF_FUSED", "1") != "0"
        and not os.environ.get("THUNDERSYNC_DEBUG_CF_STOCKWIDE")
    )


class _CrossfadeFused(torch.autograd.Function):
    """Fused forward (bitwise the reference above) and backward (d gates / d logits, fp32, deterministic)."""

    @staticmethod
    def forward(ctx, logits, expert_bias, k, C, m, scaling_factor):
        from nemo_rl.models.megatron import cf_router_fused as K

        T = logits.shape[0]
        probs = torch.empty(T, C, device=logits.device, dtype=torch.float32)
        idx = torch.empty(T, C, device=logits.device, dtype=torch.int64)
        mv, recip = K._m_arg(m)
        bias = expert_bias.detach().float().contiguous()
        K._mod().cf_forward(logits, bias, k, C, mv, recip, K._sf(scaling_factor), probs, idx, None, None, None,
                            K._sel(T, logits.shape[1]))
        ctx.save_for_backward(logits, bias)
        ctx.params = (k, C, mv, recip, K._sf(scaling_factor))
        ctx.m = m
        ctx.mark_non_differentiable(idx)
        return probs, idx

    @staticmethod
    def backward(ctx, grad_probs, grad_idx):
        from nemo_rl.models.megatron import cf_router_fused as K

        logits, bias = ctx.saved_tensors
        k, C, mv, recip, sf = ctx.params
        if _STATE.get("ramp_stopgrad"):
            return _ramp_stopgrad_grad(logits, bias, k, C, ctx.m, sf, grad_probs), None, None, None, None, None
        gx = torch.empty_like(logits)
        K._mod().cf_backward(logits, bias, grad_probs.float().contiguous(), k, C, mv, recip, sf, gx,
                             K._sel(logits.shape[0], logits.shape[1]))
        return gx, None, None, None, None, None


def _ramp_stopgrad_grad(logits, bias, k, C, m, sf, grad_probs):
    """d loss / d logits of the reference rule with the ramp c and tau detached (crossfade_routing_ramp_stopgrad).

    Recomputes s = sigmoid(x) and the candidates from the saved logits (the router is [T, E] fp32: small next to
    the layer) and backpropagates grad_probs through p = sf * s[idx] c / sum(s[idx] c) with c constant.
    """
    with torch.enable_grad():
        x = logits.detach().float().requires_grad_(True)
        s = torch.sigmoid(x)
        t = (s + bias.float()).detach()
        v, idx = candidates(t, max(C, k + 1))
        tau = (v[:, k - 1] + v[:, k]) * 0.5
        c = ((v - tau.unsqueeze(1)) * inv_m(m) + 0.5).clamp(0.0, 1.0)
        w = torch.gather(s, 1, idx) * c
        total = w[:, 0]
        for j in range(1, w.shape[1] if C > k else k):
            total = total + w[:, j]
        p = w / (total.unsqueeze(1) + 1e-20)
        if sf:
            p = p * sf
        (gx,) = torch.autograd.grad(p[:, :C], x, grad_probs.float())
    return gx




def configure(config: Any, model_cfg: Any) -> Optional[float]:
    """Set the rule from a policy config on a model provider config (before the model is built).

    With cross-fade on: records k = the provider's ``moe_router_topk``, sets
    ``moe_router_topk = k + extra`` (fixed-width dispatch) and installs the
    MCore patch. Idempotent per process: a second provider (the inference
    engine's) must carry the same k.
    """
    m = crossfade_m(config)
    _trace("configure", m=m, topk_in=getattr(model_cfg, "moe_router_topk", None),
           base=getattr(model_cfg, "_nrl_crossfade_base_topk", None), model_cfg_type=type(model_cfg).__name__)
    if m is None:
        _STATE.update(m=None, k=None, extra=None, sparse=False, ramp_stopgrad=False)
        return None
    if m <= 0:
        raise ValueError(f"crossfade_routing_m must be > 0 (got {m})")
    if os.environ.get("THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE", "") not in ("", "0") and not (
        os.environ.get("THUNDERSYNC_NL_CHECKS") == "1" or os.environ.get("THUNDERSYNC_NL_NEG") == "cf_engine_m"
    ):
        # Negative control only (engine band != learner band): refused outside CHECKS / the cf_engine_m negative, on
        # every path including the stock loop (no ThunderSync validator there).
        raise ValueError("THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE is a negative control: set THUNDERSYNC_NL_CHECKS=1 or "
                         "THUNDERSYNC_NL_NEG=cf_engine_m")
    extra = crossfade_extra(config)
    if extra < 0 or (extra == 0 and not os.environ.get("THUNDERSYNC_DEBUG_CF_ALLOW_EXTRA0")):
        raise ValueError(f"crossfade_routing_extra must be >= 1 (got {extra}; 0 only as a debug plumbing check)")
    if getattr(model_cfg, "moe_router_fusion", False):
        raise ValueError(
            "cross-fade routing requires moe_router_fusion=False: the fused TE top-k would bypass the rule"
        )
    k = getattr(model_cfg, "_nrl_crossfade_base_topk", None)
    if k is None:
        k = int(model_cfg.moe_router_topk)
        model_cfg._nrl_crossfade_base_topk = k
        model_cfg.moe_router_topk = k + extra
    if crossfade_sparse(config):
        from nemo_rl.models.megatron import device_dispatch

        if not device_dispatch.enabled():
            # The sparse routing map (zero-weight slots dropped, static T x topk capacity kept) is built for the
            # learner's device dispatch; Megatron's AlltoAll dispatcher splits by T x topk and fails on it
            # ("Split sizes doesn't match total dim 0 size").
            raise ValueError(
                "crossfade_routing_sparse needs the learner's device dispatch (NEMO_RL_MOE_DEVICE_DISPATCH=1)"
            )
    profile = crossfade_m_profile(config)
    layer_m = None
    if profile is not None:
        layers = moe_layer_numbers(model_cfg)
        if not layers:
            raise ValueError("crossfade_routing_m_profile needs a hybrid layer pattern with MoE ('E') layers")
        layer_m = layer_bands(m, profile, layers)
    if _STATE["k"] is not None and (_STATE["k"], _STATE["extra"], _STATE["m"], _STATE["layer_m"]) != (
        k, extra, m, layer_m
    ):
        raise ValueError(
            f"cross-fade routing configured twice with different rules: {(_STATE['k'], _STATE['extra'], _STATE['m'])} "
            f"vs {(k, extra, m)} (or a different per-layer profile)"
        )
    _STATE.update(m=m, k=k, extra=extra, sparse=crossfade_sparse(config), ramp_stopgrad=crossfade_ramp_stopgrad(config),
                  layer_m=layer_m, sparse_engine=crossfade_sparse_engine(config))
    # One line per model built (the learner's and, colocated, the engine's provider): the band that process routes with.
    bands = None if layer_m is None else [round(layer_m[ln], 6) for ln in sorted(layer_m)]
    print(f"[crossfade] configure provider={type(model_cfg).__name__} pid={os.getpid()} k={k} extra={extra} m={m} "
          f"layer_m={bands} sparse={_STATE['sparse']} sparse_engine={_STATE['sparse_engine']} "
          f"ramp_stopgrad={_STATE['ramp_stopgrad']}", flush=True)
    install()
    return m


def _wrap(orig):
    def topk_routing_with_score_function(
        logits: torch.Tensor,
        topk: int,
        use_pre_softmax: bool = False,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        scaling_factor: Optional[float] = None,
        score_function: str = "softmax",
        expert_bias: Optional[torch.Tensor] = None,
        fused: bool = False,
        router_replay: Any = None,
        dense_output: bool = False,
        precomputed_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        # kwargs: Megatron-LM main's qb_histogram / qb_bin_bounds (quantile balancing), topk_indices (fused dense
        # output buffer) and allow_batch_invariant_sigmoid_topk; forwarded to MCore's routing when cross-fade is off.
        m, k, extra = _STATE["m"], _STATE["k"], _STATE["extra"]
        if m is None:
            return orig(
                logits,
                topk,
                use_pre_softmax=use_pre_softmax,
                num_groups=num_groups,
                group_topk=group_topk,
                scaling_factor=scaling_factor,
                score_function=score_function,
                expert_bias=expert_bias,
                fused=fused,
                router_replay=router_replay,
                dense_output=dense_output,
                precomputed_indices=precomputed_indices,
                **kwargs,
            )
        if kwargs.get("qb_histogram") is not None or kwargs.get("topk_indices") is not None:
            raise NotImplementedError(
                "cross-fade routing: quantile-balancing histograms / a fused top-k index buffer are not built"
            )
        if score_function != "sigmoid" or expert_bias is None or use_pre_softmax:
            raise NotImplementedError(
                "cross-fade routing is built for sigmoid + expert-bias routers (Nemotron-H); got "
                f"score_function={score_function}, expert_bias={'set' if expert_bias is not None else None}, "
                f"use_pre_softmax={use_pre_softmax}"
            )
        if group_topk and group_topk != num_groups:
            raise NotImplementedError("cross-fade routing: group-limited top-k is not built")
        if precomputed_indices is not None:
            raise NotImplementedError("cross-fade routing: precomputed indices (quantile balancing) are not built")
        if topk != k + extra:
            raise RuntimeError(
                f"cross-fade routing: the router's topk {topk} != k + extra = {k} + {extra} (moe_router_topk not "
                "widened for this model: configure() did not run on its provider)"
            )
        layer_m = _STATE.get("layer_m")
        if layer_m is not None:
            # Per-layer band: the router module that is routing now (``_track_layer``).
            ln = _STATE.get("cur_layer")
            if ln not in layer_m:
                raise RuntimeError(f"cross-fade per-layer band: no band for layer {ln} (MoE layers {sorted(layer_m)})")
            m = layer_m[ln]
        scale = os.environ.get("THUNDERSYNC_DEBUG_CF_ENGINE_M_SCALE")
        if scale and _inference_active():
            # Negative test only: the engine routes with another band width than the learner; the mismatch /
            # prefill / train-lp checks must catch it.
            m = m * float(scale)
        probs, top_indices = crossfade_route(logits, expert_bias, k, extra, m, scaling_factor)
        probs = probs.type_as(logits)
        if os.environ.get("THUNDERSYNC_DEBUG_CF_TRACE"):
            key = (_inference_active(), dense_output)
            n = _TRACE_SEEN.get(key, 0)
            if n < 3 and not torch.compiler.is_compiling() and not (
                logits.is_cuda and torch.cuda.is_current_stream_capturing()
            ):
                _TRACE_SEEN[key] = n + 1
                _trace("route", inference=key[0], dense=dense_output, topk=topk, T=int(logits.shape[0]),
                       logits_dtype=str(logits.dtype), m=m, k=k,
                       idx0=top_indices[:1].tolist(), p0=[round(float(v), 6) for v in probs[:1].flatten().tolist()])
        if dense_output:
            return probs, top_indices
        num_tokens = logits.shape[0]
        if torch.are_deterministic_algorithms_enabled():
            routing_probs = torch.zeros_like(logits)
            rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
            routing_probs.index_put_((rows, top_indices), probs, accumulate=False)
            routing_map = torch.zeros_like(logits, dtype=logits.dtype)
            routing_map.index_put_(
                (rows, top_indices), torch.ones_like(probs, dtype=routing_map.dtype), accumulate=False
            )
            routing_map = routing_map.bool()
        else:
            routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
            routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()
        if _STATE.get("sparse") and not _inference_active():
            # Learner sparse dispatch: only the nonzero-weight slots are routed (their contribution is the same bits:
            # a zero-weight slot adds exactly +0 to the combine and gets a zero gradient). The dispatch keeps the
            # static T x topk capacity; device_dispatch treats the rest as padding rows (zero outputs / gradients).
            routing_map = routing_map & (routing_probs != 0)
        return routing_probs, routing_map

    setattr(topk_routing_with_score_function, _PATCH_ATTR, True)
    topk_routing_with_score_function.__wrapped__ = orig  # type: ignore[attr-defined]
    return topk_routing_with_score_function


def _track_layer(orig):
    """Router forward that records its layer number for the per-layer band (host side, also at graph capture)."""

    def forward(self, *args, **kwargs):
        _STATE["cur_layer"] = getattr(self, "layer_number", None)
        return orig(self, *args, **kwargs)

    setattr(forward, _PATCH_ATTR, True)
    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    return forward


def _inference_active() -> bool:
    try:
        from megatron.core.inference.utils import InferenceMode
    except ImportError:  # pragma: no cover
        return False
    return bool(InferenceMode.is_active())


def _wrap_inverse_map(orig):
    def build_inverse_permutation_map(routing_map, flat_sorted, sorted_indices, num_out_tokens):
        if not _STATE.get("sparse") or _STATE.get("m") is None:
            return orig(routing_map, flat_sorted, sorted_indices, num_out_tokens)
        # Sparse dispatch: the static T x topk permutation holds padding rows (False entries of the routing map past
        # the real count). Megatron's builder would give them the slot of the previous True entry and overwrite a
        # real slot; route them to a scratch slot instead. Real slots are unique and get the same (row, expert) as
        # Megatron's; unused slots stay -1 (batch-invariant unpermute skips them, their rows get zero gradients).
        num_tokens = routing_map.size(0)
        assert isinstance(num_out_tokens, int) and num_out_tokens % num_tokens == 0
        topk = num_out_tokens // num_tokens
        rm = routing_map.bool()
        row_ids = torch.arange(num_out_tokens, device=routing_map.device, dtype=torch.long)
        expert_ids = torch.div(flat_sorted, num_tokens, rounding_mode="floor").to(torch.long)
        token_ids = sorted_indices.to(torch.long)
        valid = rm[token_ids, expert_ids]
        row_slots = (rm.to(torch.long).cumsum(dim=1) - 1)[token_ids, expert_ids]
        scratch = num_tokens * topk
        linear = torch.where(valid, token_ids * topk + row_slots, torch.full_like(token_ids, scratch))
        inv_rows = torch.full((scratch + 1,), -1, device=routing_map.device, dtype=torch.long)
        inv_exp = torch.full((scratch + 1,), -1, device=routing_map.device, dtype=torch.long)
        inv_rows.scatter_(0, linear, row_ids)
        inv_exp.scatter_(0, linear, expert_ids)
        return torch.stack((inv_rows[:scratch].view(num_tokens, topk), inv_exp[:scratch].view(num_tokens, topk)), dim=0)

    setattr(build_inverse_permutation_map, _PATCH_ATTR, True)
    build_inverse_permutation_map.__wrapped__ = orig  # type: ignore[attr-defined]
    return build_inverse_permutation_map


def install() -> None:
    """Route MCore's top-k score function (the router module's and moe_utils') through the rule (idempotent)."""
    from megatron.core.transformer.moe import moe_utils
    from megatron.core.transformer.moe import router as mcore_router

    _trace("install", router_file=getattr(mcore_router, "__file__", None))
    if torch.cuda.is_available() and os.environ.get("THUNDERSYNC_CF_FUSED", "1") != "0":
        from nemo_rl.models.megatron import cf_router_fused

        cf_router_fused._mod()  # load_inline build on first use: here, not inside a graph capture
    for cls_name in ("TopKRouter", "InferenceTopKRouter"):
        cls = getattr(mcore_router, cls_name, None)
        if cls is not None and "forward" in cls.__dict__ and not getattr(cls.forward, _PATCH_ATTR, False):
            cls.forward = _track_layer(cls.forward)
    for mod in (moe_utils, mcore_router):
        fn = getattr(mod, "topk_routing_with_score_function")
        if getattr(fn, _PATCH_ATTR, False):
            continue
        setattr(mod, "topk_routing_with_score_function", _wrap(fn))
    inv = getattr(moe_utils, "build_batch_invariant_inverse_permutation_map", None)
    if inv is not None and not getattr(inv, _PATCH_ATTR, False):
        moe_utils.build_batch_invariant_inverse_permutation_map = _wrap_inverse_map(inv)


def installed() -> bool:
    try:
        from megatron.core.transformer.moe import router as mcore_router
    except ImportError:  # pragma: no cover
        return False
    return bool(getattr(mcore_router.topk_routing_with_score_function, _PATCH_ATTR, False))
