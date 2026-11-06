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
"""Canonical-block log-sum-exp: one log-prob definition for trainer, verifier, q rows and engine.

Opt-in (``NEMO_RL_VOCAB_LSE=canonical_block``, set through ``policy.megatron_cfg.env_vars`` so every learner
and engine worker sees it). With it on, the log-prob of column v of a logit row z is

    logp(v) = fp32(z_v) - LSE,   LSE = combine_j (m_j, s_j) over 4096-column vocabulary blocks in block order,

with each block's (max, sum exp) computed by one fixed-order Triton program and the combine done in fp64 in global
block order (kernel task D, ``kernel_tasks/vocablse/vocab_lse.md``; these are its reference kernels, same bits).
The value depends only on the row: not on the batch, and not on TP, because a TP rank owns whole blocks and the
partials of the shards concatenate to the partials of the full row. So a rank can compute LSE from its own shard
plus an all-gather of 8 bytes per block per row (``lse_sharded``) and get the bits of the full-row computation
(``lse_full``) that a gathered-logits consumer runs.

Off (default): nothing changes; every consumer keeps ``log_softmax`` of the gathered row.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 4096
_TILE = 1024
_ROWS = 64  # rows per program of the combine (one row per thread; any value gives the same bits)
_GRAD_COLS = 2048
ENV = "NEMO_RL_VOCAB_LSE"


def enabled() -> bool:
    v = os.environ.get(ENV, "")
    if v not in ("", "0", "off", "gathered", "canonical_block"):
        raise ValueError(f"{ENV}={v!r}: expected canonical_block or off")
    return v == "canonical_block"


@triton.jit
def _ld32(base, cols, g0, vlim):
    return tl.load(base + cols, mask=(g0 + cols) < vlim, other=float("-inf")).to(tl.float32)


@triton.jit
def _nan_to_ninf(x):
    return tl.where(x != x, float("-inf"), x)


@triton.jit
def _exp_tile(x, m, mfin):
    # where(live, exp(where(live, x - m, 0)), 0) with live = (x > -inf) & (x == x) & mfin: for x = -inf
    # (m finite) exp(x - m) = exp(-inf) = +0, the same +0.
    return tl.where((x == x) & mfin, tl.exp(x - m), 0.0)


@triton.jit
def _partials_kernel(Z, OUT, stride_z, voff, vlim, nb_local, BLOCK: tl.constexpr, TILE: tl.constexpr,
                     PIN4: tl.constexpr):
    """One program per (row, block), one pass: the block's four 1024-column tiles are loaded up front; the max
    and the NaN flag are reduced once (max is exact, so its order is free); s is the sequential sum of the four
    tiles' ``tl.sum`` (4 warps, 4-wide load layout): that order defines the bits (kernel task D)."""
    row = tl.program_id(0)
    j = tl.program_id(1)
    base = Z + row.to(tl.int64) * stride_z + j * BLOCK
    g0 = voff + j * BLOCK
    c = tl.arange(0, TILE)
    if PIN4:  # bf16 rows read in place with the fp32 rows' 4-wide load layout, hence the fp32 reduction tree
        c = tl.max_contiguous(c, 4)
    x0 = _ld32(base, c, g0, vlim)
    x1 = _ld32(base, c + TILE, g0, vlim)
    x2 = _ld32(base, c + 2 * TILE, g0, vlim)
    x3 = _ld32(base, c + 3 * TILE, g0, vlim)
    bad = (x0 != x0) | (x1 != x1) | (x2 != x2) | (x3 != x3)
    nan = tl.max(bad.to(tl.int32), 0)
    mx = tl.maximum(tl.maximum(_nan_to_ninf(x0), _nan_to_ninf(x1)), tl.maximum(_nan_to_ninf(x2), _nan_to_ninf(x3)))
    m = tl.max(mx, 0)
    mfin = (m > float("-inf")) & (m < float("inf"))
    s = tl.zeros([], tl.float32)
    s += tl.sum(_exp_tile(x0, m, mfin), 0)
    s += tl.sum(_exp_tile(x1, m, mfin), 0)
    s += tl.sum(_exp_tile(x2, m, mfin), 0)
    s += tl.sum(_exp_tile(x3, m, mfin), 0)
    s = tl.where(m == float("inf"), 1.0, s)  # a +inf column: m = +inf, s = 1
    m = tl.where(nan > 0, float("nan"), m)
    o = OUT + (row.to(tl.int64) * nb_local + j) * 2
    tl.store(o, m)
    tl.store(o + 1, s)


@triton.jit
def _mul_rn(a, b):
    """fp64 a * b rounded to nearest and never contracted into an fma (PTX mul.rn): plain mul / add leave ptxas
    free to fuse them depending on the schedule, which would make the bits depend on how the kernel is written."""
    return tl.inline_asm_elementwise("mul.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True,
                                     pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f64 $0, $1, $2;", "=d,d,d", [a, b], dtype=tl.float64, is_pure=True,
                                     pack=1)


@triton.jit
def _combine_kernel(P, OUT, n, nb, ROWS: tl.constexpr):
    """One thread per row: M = max_j m_j, then S = sum_j s_j exp(m_j - M) sequentially in block order (fp64)."""
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    rok = rows < n
    base = P + rows.to(tl.int64) * nb * 2
    M = tl.full([ROWS], float("-inf"), tl.float64)
    nan = tl.zeros([ROWS], tl.int32)
    for j in range(0, nb):
        mj = tl.load(base + 2 * j, mask=rok, other=0.0).to(tl.float64)
        nan += (mj != mj).to(tl.int32)
        M = tl.maximum(M, tl.where(mj != mj, float("-inf"), mj))
    S = tl.zeros([ROWS], tl.float64)
    for j in range(0, nb):
        mj = tl.load(base + 2 * j, mask=rok, other=0.0).to(tl.float64)
        sj = tl.load(base + 2 * j + 1, mask=rok, other=0.0).to(tl.float64)
        live = (mj > float("-inf")) & (mj == mj) & (M < float("inf"))
        S = _add_rn(S, tl.where(live, _mul_rn(sj, tl.exp(tl.where(live, mj - M, 0.0))), 0.0))
    lse = tl.where(M == float("-inf"), float("-inf"), tl.where(M == float("inf"), float("inf"), _add_rn(M, tl.log(S))))
    lse = tl.where(nan > 0, float("nan"), lse)
    tl.store(OUT + rows, lse.to(tl.float32), mask=rok)


@triton.jit
def _grad_kernel(Z, LSE, TOK, G, GZ, sz, sg, width, voff, COLS: tl.constexpr):
    """grad_z = g * (onehot - exp(fp32(z) - LSE)) on one shard row, as the eager chunk loop computes it:
    -(p * g), and g + (-(p * g)) at the owned token (fp fusion off at launch: no fma)."""
    row = tl.program_id(0)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    inb = cols < width
    z = tl.load(Z + row.to(tl.int64) * sz + cols, mask=inb, other=0.0).to(tl.float32)
    lse = tl.load(LSE + row)
    g = tl.load(G + row)
    tok = tl.load(TOK + row)
    v = -(libdevice.exp(z - lse) * g)
    v = tl.where(voff + cols == tok, g + v, v)
    tl.store(GZ + row.to(tl.int64) * sg + cols, v.to(GZ.dtype.element_ty), mask=inb)


def _rows(z: torch.Tensor) -> torch.Tensor:
    """[..., V] -> [n, V] with unit column stride (a view when possible)."""
    z2 = z.reshape(-1, z.shape[-1])
    if z2.stride(-1) != 1:
        z2 = z2.contiguous()
    return z2


def partials(z_local: torch.Tensor, vocab_offset: int, vocab_limit: int) -> torch.Tensor:
    """[n, ceil(V_local / BLOCK), 2] fp32 (max, sum exp) of this shard's canonical blocks.

    ``vocab_offset`` must be a multiple of BLOCK (whole blocks per shard). A shard narrower than a whole number
    of blocks (only the last shard, or a full row whose width is not a multiple of BLOCK) is read with columns
    past its width masked like columns >= ``vocab_limit``.
    """
    if vocab_offset % BLOCK:
        raise ValueError(f"vocab shard offset {vocab_offset} is not a multiple of {BLOCK}")
    z2 = _rows(z_local)
    # The definition is that of fp32(z). bf16 rows are read in place when their loads can take the 4-wide
    # layout an fp32 copy gets (16-byte-aligned base, row stride a multiple of 16): same reduction tree, same
    # bits, no fp32 copy. Anything else is cast first.
    pin4 = z2.dtype == torch.bfloat16 and z2.is_cuda and z2.data_ptr() % 16 == 0 and z2.stride(0) % 16 == 0
    if z2.dtype != torch.float32 and not pin4:
        z2 = z2.float()
    elif z2.dtype == torch.float32 and z2.is_cuda and (z2.data_ptr() % 16 or z2.stride(0) % 16):
        # A misaligned fp32 view (base or row stride not 16-byte aligned) would specialize the kernel to a
        # 1-wide load layout, a different reduction tree, other bits for the same values (batch invariance
        # broken by the view's alignment). Copy it to an aligned buffer: the bits of the same row aligned.
        z2 = z2.clone(memory_format=torch.contiguous_format)
    n, width = z2.shape
    nb = -(-width // BLOCK)
    if not z2.is_cuda:
        return _partials_host(z2, vocab_offset, vocab_limit)
    out = torch.empty(n, nb, 2, dtype=torch.float32, device=z2.device)
    if n:
        vlim = min(int(vocab_limit), vocab_offset + width)
        _partials_kernel[(n, nb)](z2, out, z2.stride(0), vocab_offset, vlim, nb, BLOCK=BLOCK, TILE=_TILE, PIN4=pin4,
                                  num_warps=4)
    return out


def _partials_host(z2: torch.Tensor, vocab_offset: int, vocab_limit: int) -> torch.Tensor:
    """Host (CPU tests only) partials with the same edge-case rules; not bitwise the GPU kernel."""
    n, width = z2.shape
    nb = -(-width // BLOCK)
    x = torch.full((n, nb * BLOCK), float("-inf"), dtype=torch.float32)
    x[:, :width] = z2.float()
    cols = torch.arange(nb * BLOCK) + vocab_offset
    x[:, cols >= vocab_limit] = float("-inf")
    x = x.view(n, nb, BLOCK)
    nan = torch.isnan(x).any(-1)
    pinf = (x == float("inf")).any(-1)
    xs = torch.where(torch.isnan(x), torch.full_like(x, float("-inf")), x)
    m = xs.max(-1).values
    live = torch.isfinite(m)[..., None] & (xs > float("-inf"))
    s = torch.where(live, torch.exp(torch.where(live, xs - m[..., None], torch.zeros_like(xs))), torch.zeros_like(xs)).sum(-1)
    s = torch.where(pinf, torch.ones_like(s), s)
    m = torch.where(nan, torch.full_like(m, float("nan")), m)
    return torch.stack([m, s], -1)


def _combine_host(parts: torch.Tensor) -> torch.Tensor:
    m, s = parts[..., 0].double(), parts[..., 1].double()
    nan = torch.isnan(m).any(-1)
    mm = torch.where(torch.isnan(m), torch.full_like(m, float("-inf")), m)
    M = mm.max(-1).values
    live = (mm > float("-inf")) & (M < float("inf"))[..., None]
    S = torch.where(live, s * torch.exp(torch.where(live, mm - M[..., None], torch.zeros_like(mm))), torch.zeros_like(mm)).sum(-1)
    lse = torch.where(M == float("-inf"), M, torch.where(M == float("inf"), M, M + torch.log(S)))
    lse = torch.where(nan, torch.full_like(lse, float("nan")), lse)
    return lse.float()


def combine(parts: torch.Tensor) -> torch.Tensor:
    """[n, nb, 2] partials in global block order -> [n] fp32 LSE."""
    if not parts.is_cuda:
        return _combine_host(parts)
    parts = parts.contiguous()
    n, nb, _ = parts.shape
    out = torch.empty(n, dtype=torch.float32, device=parts.device)
    if n:
        _combine_kernel[(triton.cdiv(n, _ROWS),)](parts, out, n, nb, ROWS=_ROWS, num_warps=2)
    return out


def lse_full(z: torch.Tensor, vocab_limit: Optional[int] = None) -> torch.Tensor:
    """Canonical LSE of full-vocabulary rows ``z`` [..., V] -> [...] fp32 (columns >= vocab_limit excluded)."""
    lead = z.shape[:-1]
    vl = int(z.shape[-1]) if vocab_limit is None else int(vocab_limit)
    return combine(partials(z, 0, vl)).reshape(lead)


def lse_sharded(
    z_local: torch.Tensor, vocab_offset: int, vocab_limit: int, group: Optional[torch.distributed.ProcessGroup]
) -> torch.Tensor:
    """Canonical LSE from this rank's vocab shard [..., V_local] plus an all-gather of the block partials over
    ``group`` (rank order = vocabulary order, whole blocks per rank). Bitwise ``lse_full`` of the gathered row."""
    lead = z_local.shape[:-1]
    p = partials(z_local, vocab_offset, vocab_limit)
    if group is not None and torch.distributed.get_world_size(group) > 1:
        if z_local.shape[-1] % BLOCK:
            raise ValueError(f"TP vocab shard width {z_local.shape[-1]} is not a multiple of {BLOCK}")
        parts = [torch.empty_like(p) for _ in range(torch.distributed.get_world_size(group))]
        torch.distributed.all_gather(parts, p, group=group)
        p = torch.cat(parts, 1)
    return combine(p).reshape(lead)


def logprob_rows(z: torch.Tensor, vocab_limit: Optional[int] = None) -> torch.Tensor:
    """fp32 log-prob rows ``fp32(z) - LSE`` of full-vocabulary rows, columns [0, vocab_limit) (the canonical
    replacement of ``log_softmax(z[..., :vocab_limit].float(), -1)``)."""
    vl = int(z.shape[-1]) if vocab_limit is None else int(vocab_limit)
    lse = lse_full(z, vl)
    return z[..., :vl].float() - lse.unsqueeze(-1)


def logprob_grad(
    local: torch.Tensor,
    target: torch.Tensor,
    grad_output: torch.Tensor,
    lse: torch.Tensor,
    vocab_start: int,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """d logp(target) / d z_local times ``grad_output`` on this rank's vocab shard, one fused pass:
    ``g * (onehot - exp(fp32(z) - LSE))`` in ``local``'s dtype (local [..., V_local], target / grad_output /
    lse [...]; ``out_dtype`` overrides the output dtype). Same elementwise ops as the eager chunk loop (``_canonical_sharded_grad``): fp32 exp, then
    ``-(p * g)`` and ``g + (-(p * g))`` at the owned token, rounded once to ``local``'s dtype."""
    width = int(local.shape[-1])
    if not local.is_cuda:
        # Host tensors (CPU tests): the eager form of the same elementwise ops.
        p = (local.float() - lse.to(torch.float32).unsqueeze(-1)).exp_()
        g = grad_output.to(torch.float32)
        v = p.mul_(g.unsqueeze(-1)).neg_()
        cols = torch.arange(width, device=local.device) + vocab_start
        hit = cols == target.to(torch.int64).unsqueeze(-1)
        v = torch.where(hit, g.unsqueeze(-1) + v, v)
        return v.to(out_dtype or local.dtype)
    z2 = _rows(local)
    n = z2.shape[0]
    out = torch.empty((n, width), dtype=out_dtype or local.dtype, device=local.device)
    if n:
        tok = target.reshape(-1).to(torch.int64).contiguous()
        g = grad_output.reshape(-1).to(torch.float32).contiguous()
        l = lse.reshape(-1).to(torch.float32).contiguous()
        _grad_kernel[(n, triton.cdiv(width, _GRAD_COLS))](
            z2, l, tok, g, out, z2.stride(0), out.stride(0), width, vocab_start, COLS=_GRAD_COLS, num_warps=8,
            enable_fp_fusion=False,
        )
    return out.view(local.shape)
