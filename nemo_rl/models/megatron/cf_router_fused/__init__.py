"""Fused cross-fade MoE router (cf_router.md): CUDA C++ (cf_router.cu, compiled with load_inline on first use).

Selection (all variants give the same bits; picked by shape, _sel):
- block per row (decode-size T, E > 128): 256 threads compute the row's s / t; an MSB-first 4 x 8-bit radix select
  over order-preserving 32-bit keys finds the C-th largest t; ties at it go to the lowest ids (block ballot ranks);
  the C winners are sorted (t descending, id ascending) by a 32-wide warp bitonic sort on (key, ~id);
- warp per row (large T): the same radix select within a warp, or C rounds of redux max (t) / min (id) for E <= 128.
Then one warp, lane j = slot j: tau, the ramp, w, the sequential sum W (all C values broadcast, added in slot order),
p, counts (integer atomics after a memset; skipped when the outputs are None). Every fp32 op is the reference's op
with explicit round-to-nearest intrinsics (no fma contraction; precise expf and IEEE divide in the sigmoid, as
torch.sigmoid). The Python-float band width enters as torch applies tensor / scalar: a multiply by fp32(1 / m), the
reciprocal computed in double and rounded once (measured: the IEEE divide and fp32(1) / fp32(m) differ).
"""

from __future__ import annotations

import os

import numpy as np
import torch

_RECIP = True  # torch's tensor / scalar = multiply by the host fp32 reciprocal (False: IEEE divide)
_RECIP_MODE = "d"  # "d": fp32(1 / m) rounded once from double (torch's bits); "f": fp32(1) / fp32(m)
_SEL = None  # selection: 0 = warp per row, C rounds of redux arg-max; 1 = warp per row, radix select; 2 = block
#              per row (256 threads), radix select. All give the same bits (measured); None = by shape (fastest
#              measured on GB200, cfk-dev5): E <= 128: 0; E > 128: 2 for T <= 2048 (per-row latency), else 1.
_BLOCK_T = 2048


def _sel(T, E):
    if _SEL is not None:
        return _SEL
    if E <= 128:
        return 0
    return 2 if T <= _BLOCK_T else 1
_MOD = None


def _mod():
    global _MOD
    if _MOD is None:
        from torch.utils.cpp_extension import load_inline

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cf_router.cu")) as f:
            src = f.read()
        _MOD = load_inline(
            name="cf_router_v5",
            cpp_sources=(
                "void cf_forward(torch::Tensor x, torch::Tensor bias, int64_t k, int64_t C, double m_or_inv, "
                "bool recip, double sf, torch::Tensor probs, torch::Tensor idx, c10::optional<torch::Tensor> mask, "
                "c10::optional<torch::Tensor> tpe, c10::optional<torch::Tensor> stock, int64_t sel);\n"
                "void cf_backward(torch::Tensor x, torch::Tensor bias, torch::Tensor g, int64_t k, int64_t C, "
                "double m_or_inv, bool recip, double sf, torch::Tensor gx, int64_t sel);\n"
                "void cf_phase(torch::Tensor x, torch::Tensor bias, int64_t C, int64_t phase, torch::Tensor out);"
            ),
            cuda_sources=src,
            functions=["cf_forward", "cf_backward", "cf_phase"],
            extra_cuda_cflags=["-O3", "--fmad=false"],
            verbose=False,
        )
    return _MOD


def _m_arg(m):
    if _RECIP:
        if _RECIP_MODE == "d":
            return float(np.float32(1.0 / float(m))), True
        return float(np.float32(1.0) / np.float32(m)), True
    return float(np.float32(m)), False


def _sf(scaling_factor):
    return float(np.float32(scaling_factor)) if scaling_factor else 0.0


def _ok(logits, k, extra):
    E = logits.shape[1]
    C = min(E, k + extra)
    return logits.is_cuda and logits.dtype == torch.float32 and logits.is_contiguous() and C <= 32 and C > k >= 1


def _route_torch(logits, bias, k, extra, m, scaling_factor):
    """The reference's PyTorch ops (fallback outside the kernel's range: C > 32, CPU, ...)."""
    E = logits.shape[1]
    C = min(E, k + extra)
    s = torch.sigmoid(logits.float())
    t = s + bias.float()
    vals, idx = torch.sort(-t, dim=1, stable=True)
    v, idx = -vals[:, :C], idx[:, :C]
    tau = (v[:, k - 1] + v[:, k]) * 0.5
    c = ((v - tau.unsqueeze(1)) / m + 0.5).clamp(0.0, 1.0)
    w = torch.gather(s, 1, idx) * c
    total = w[:, 0]
    for j in range(1, C):
        total = total + w[:, j]
    p = w / (total.unsqueeze(1) + 1e-20)
    if scaling_factor:
        p = p * scaling_factor
    return p, idx, c


def cf_route_forward(logits, bias, k, extra, m, scaling_factor, *, out_probs, out_idx, out_mask,
                     out_tokens_per_expert, out_stock_counts):
    if not _ok(logits, k, extra):
        p, idx, c = _route_torch(logits, bias, k, extra, m, scaling_factor)
        out_probs.copy_(p)
        out_idx.copy_(idx.to(out_idx.dtype))
        out_mask.copy_(c > 0)
        one = torch.ones(1, dtype=torch.int32, device=logits.device)
        out_tokens_per_expert.zero_().scatter_add_(0, idx.reshape(-1), one.expand(idx.numel()))
        out_stock_counts.zero_().scatter_add_(0, idx[:, :k].reshape(-1), one.expand(idx[:, :k].numel()))
        return
    C = min(logits.shape[1], k + extra)
    mv, recip = _m_arg(m)
    _mod().cf_forward(logits, bias.float().contiguous(), k, C, mv, recip, _sf(scaling_factor), out_probs, out_idx,
                      out_mask, out_tokens_per_expert, out_stock_counts, _sel(logits.shape[0], logits.shape[1]))


def cf_route_backward(logits, bias, k, extra, m, scaling_factor, grad_probs, *, out_grad_logits):
    if not _ok(logits, k, extra):
        x = logits.detach().float().requires_grad_(True)
        with torch.enable_grad():
            p, _, _ = _route_torch(x, bias, k, extra, m, scaling_factor)
            (g,) = torch.autograd.grad(p, x, grad_probs)
        out_grad_logits.copy_(g)
        return
    C = min(logits.shape[1], k + extra)
    mv, recip = _m_arg(m)
    _mod().cf_backward(logits, bias.float().contiguous(), grad_probs.float().contiguous(), k, C, mv, recip,
                       _sf(scaling_factor), out_grad_logits, _sel(logits.shape[0], logits.shape[1]))
