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
"""High-precision, batch-invariant causal attention (``precision_raise`` site ``attn_core_fp32``).

Wired into Megatron learner / engine by ``hp_attention_mcore.py``.

One Triton kernel computes attention for the training forward / engine prefill
and for engine decode on a paged KV cache, with the same per-row arithmetic, so
the output for token t is bitwise identical whichever path computed it and
whatever else is in the batch.

Numerics (all configurable through ``HPAttnConfig`` to measure trade-offs):

- q, K, V (fp32, after RoPE) are split per row (one token x one head, D values)
  into fp16 ``hi + lo`` after a power-of-two prescale chosen from the row's own
  amax (``split_rows``): x = (hi + lo) * inv_scale. Row-local => batch
  invariant; ~22 significant bits; lo is never subnormal for elements within
  ~2^-13 of the row amax.
- S = (qh.kh + [qh.kl + ql.kh (+ ql.kl)]) * q_inv * k_inv * scale*log2(e); the
  correction products go to a second fp32 accumulator added once per block.
- online softmax in fp32 (MUFU exp2); P'' = p * v_inv_j * 2^P_SHIFT (an exact
  power-of-two column scale that undoes V's per-row prescale) is split into
  Ph + Pl and O += Ph.Vh + [Ph.Vl + Pl.Vh (+ Pl.Vl)], fp32 accumulators.
- fixed KV block size (BLOCK_N keys) and block order. Keys are grouped into
  fixed chunks of CHUNK keys at absolute positions; each chunk runs the online
  softmax from scratch and chunk results are folded in ascending order with one
  shared fp32 fold. The training forward folds in-kernel; decode may run each
  chunk in its own program (split-KV for parallelism) and fold them in a second
  kernel: the same fold on the same bits, so splitting is numerically invisible
  (it can depend on the batch without breaking batch invariance).
- GQA packing: a tile is BLOCK_M = TM * G rows, row = slot * G + g, holding
  TM consecutive positions [t0, t0 + TM) (t0 % TM == 0) of one sequence for the
  G query heads of one KV head. Decode places its query at slot t % TM of the
  same tile shape (other rows zero), and the number of KV blocks a tile visits
  depends only on t0, so a row's instruction sequence and in-tile position are
  the same in prefill and decode.
- every kernel is compiled with FMA contraction off (``enable_fp_fusion=False``)
  so that elementwise rounding does not depend on code context.

KV cache: K and V as (hi, lo) fp16 planes [num_pages, page_size, H_kv, D]
plus fp32 inverse scales [num_pages, page_size, H_kv]: 2.02x the bytes of an
fp16 cache (1.01x when lo planes are disabled).

Backward (``HPAttention``): the forward defines the policy; the gradient is
FlashAttention's fp16 backward evaluated at the hi parts with the hp output and
log-sum-exp.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453


@dataclasses.dataclass(frozen=True)
class HPAttnConfig:
    """Numerics and tiling of the hp attention.

    Every field except ``num_stages`` is part of the numerics: training and
    engine must use the same value.
    """

    q_lo: bool = True
    k_lo: bool = True
    v_lo: bool = True
    p_lo: bool = True
    lo_lo: bool = False  # also the lo.lo products (4 MMAs per product)
    sep_acc: bool = True  # correction products in their own fp32 accumulators
    block_m: int = 64  # rows per tile = TM * G
    block_n: int = 64  # keys per KV block
    chunk: int = 1024  # keys per fold chunk (multiple of block_n)
    num_warps: int = 8
    # Decode tile (rows = query tokens x G heads). Not part of the numerics: a
    # row's bits are the same at block_m 64 / 4-8 warps and 16 / 1-8 warps
    # (measured, tests/unit/models/megatron/test_hp_attention.py); 32 differs.
    decode_block_m: int = 16
    decode_num_warps: int = 4
    num_stages: int = 2
    p_shift: int = 12  # P'' = p * v_inv * 2^p_shift; v_inv <= 2^(15 - p_shift)
    qk_emin: int = -40
    qk_emax: int = 40
    v_emin: int = -40
    # K / V stored as plain fp16 (fp16(k), fp16(v) exactly as the fp16 recipe rounds them; no lo
    # planes, no per-row scales: the cache is byte-for-byte today's fp16 cache). Needs k_lo = v_lo = False.
    kv_fp16: bool = False

    @property
    def v_emax(self) -> int:
        return 15 - self.p_shift

    @staticmethod
    def fp16() -> "HPAttnConfig":
        """The recipe's precision (fp16 q, K, V, P) in the same kernel."""
        return HPAttnConfig(q_lo=False, k_lo=False, v_lo=False, p_lo=False)


# ---------------------------------------------------------------------------
# Shared elementwise split (training and engine call the same function).
# ---------------------------------------------------------------------------


def pow2(e: torch.Tensor) -> torch.Tensor:
    """2^e as fp32 for integer e in [-126, 127], built from the exponent field."""
    return ((e.to(torch.int32) + 127) << 23).view(torch.float32)


def row_exponent(amax: torch.Tensor) -> torch.Tensor:
    """floor(log2(amax)) from the fp32 exponent field (-127 for 0 / subnormal)."""
    return ((amax.float().view(torch.int32) >> 23) & 255) - 127


def split_rows(
    x: torch.Tensor, emin: int, emax: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split rows: x [..., D] -> (hi, lo, inv_scale) with x ~= (hi + lo) * inv_scale.

    inv_scale = 2^clamp(floor(log2(amax|x_row|)), emin, emax), so the scaled row
    has amax in [1, 2) (unless clamped). hi = fp16(x / inv_scale),
    lo = fp16(x / inv_scale - hi). Every op is elementwise or an exact max over
    the row, so the bits of a row depend only on the row. Reference (torch)
    version of the fused ``_split_kernel`` (bitwise equal; tested).
    """
    x = x.float()
    amax = x.abs().amax(dim=-1)
    e = row_exponent(amax).clamp(emin, emax)
    xs = x * pow2(-e).unsqueeze(-1)
    hi = xs.to(torch.float16)
    lo = (xs - hi.float()).to(torch.float16)
    return hi, lo, pow2(e)


if HAVE_TRITON:

    @triton.jit
    def _split_kernel(
        x_ptr, dst_ptr, hi_ptr, lo_ptr, inv_ptr, emin, emax,
        H: tl.constexpr, D: tl.constexpr, BLOCK_H: tl.constexpr,
        HAS_DST: tl.constexpr, STORE_LO: tl.constexpr,
    ):  # fmt: skip
        """One token: x [H, D] fp32 -> (hi, lo) fp16 rows + fp32 inverse scales."""
        i = tl.program_id(0).to(tl.int64)
        h = tl.arange(0, BLOCK_H)
        d = tl.arange(0, D)
        hm = h < H
        x = tl.load(x_ptr + (i * H + h)[:, None] * D + d[None, :], mask=hm[:, None])
        amax = tl.max(tl.abs(x), 1)
        e = ((amax.to(tl.int32, bitcast=True) >> 23) & 255) - 127
        e = tl.minimum(tl.maximum(e, emin), emax)
        scale = ((127 - e) << 23).to(tl.float32, bitcast=True)
        inv = ((127 + e) << 23).to(tl.float32, bitcast=True)
        xs = x * scale[:, None]
        hi = xs.to(tl.float16)
        if HAS_DST:
            row = tl.load(dst_ptr + i).to(tl.int64)
        else:
            row = i
        off = (row * H + h)[:, None] * D + d[None, :]
        tl.store(hi_ptr + off, hi, mask=hm[:, None])
        if STORE_LO:
            lo = (xs - hi.to(tl.float32)).to(tl.float16)
            tl.store(lo_ptr + off, lo, mask=hm[:, None])
        tl.store(inv_ptr + row * H + h, inv, mask=hm)


def _split_into(x, emin, emax, hi, lo, inv, dst=None) -> None:
    """Fused split of x [n, H, D] fp32 into hi/lo/inv (rows ``dst`` or 0..n-1)."""
    n, h, d = x.shape
    if n == 0:
        return
    x = x.float().contiguous()
    _split_kernel[(n,)](
        x, dst if dst is not None else x, hi, lo if lo is not None else hi, inv,
        emin, emax, H=h, D=d, BLOCK_H=triton.next_power_of_2(h),
        HAS_DST=dst is not None, STORE_LO=lo is not None, num_warps=4,
    )  # fmt: skip


def split_rows_fused(x: torch.Tensor, emin: int, emax: int):
    """``split_rows`` for x [n, H, D] on CUDA in one kernel."""
    n, h, d = x.shape
    hi = torch.empty((n, h, d), dtype=torch.float16, device=x.device)
    lo = torch.empty_like(hi)
    inv = torch.empty((n, h), dtype=torch.float32, device=x.device)
    _split_into(x, emin, emax, hi, lo, inv)
    return hi, lo, inv


def split_q(q: torch.Tensor, cfg: HPAttnConfig):
    """Split queries: q [N, Hq, D] fp32 -> (qh, ql, q_inv)."""
    if q.is_cuda and HAVE_TRITON:
        return split_rows_fused(q, cfg.qk_emin, cfg.qk_emax)
    return split_rows(q, cfg.qk_emin, cfg.qk_emax)


# ---------------------------------------------------------------------------
# Paged KV cache.
# ---------------------------------------------------------------------------


class HPKVCache:
    """K, V as fp16 (hi, lo) planes + fp32 per-row inverse scales, paged."""

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_kv_heads: int,
        head_dim: int,
        cfg: HPAttnConfig,
        device="cuda",
    ) -> None:
        shape = (num_pages, page_size, num_kv_heads, head_dim)
        f16 = dict(dtype=torch.float16, device=device)
        self.cfg = cfg
        self.page_size = page_size
        self.kh = torch.zeros(shape, **f16)
        self.vh = torch.zeros(shape, **f16)
        self.kl = torch.zeros(shape, **f16) if cfg.k_lo else None
        self.vl = torch.zeros(shape, **f16) if cfg.v_lo else None
        assert not cfg.kv_fp16 or not (cfg.k_lo or cfg.v_lo), "kv_fp16 needs k_lo = v_lo = False"
        if cfg.kv_fp16:
            self.k_inv = self.v_inv = None
        else:
            self.k_inv = torch.ones(shape[:3], dtype=torch.float32, device=device)
            self.v_inv = torch.ones(shape[:3], dtype=torch.float32, device=device)

    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (self.kh, self.vh, self.kl, self.vl, self.k_inv, self.v_inv)
            if t is not None
        )

    def write(self, slots: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        """Store fp32 k, v [n, Hkv, D] at flat slots (page * page_size + offset)."""
        cfg = self.cfg
        if cfg.kv_fp16:
            write_fp16_kv(self.kh, self.vh, slots, k, v)
            return
        if k.is_cuda and HAVE_TRITON:
            slots = slots.to(torch.int64).contiguous()
            _split_into(
                k, cfg.qk_emin, cfg.qk_emax, self.kh, self.kl, self.k_inv, slots
            )
            _split_into(v, cfg.v_emin, cfg.v_emax, self.vh, self.vl, self.v_inv, slots)
            return
        kh, kl, ki = split_rows(k, cfg.qk_emin, cfg.qk_emax)
        vh, vl, vi = split_rows(v, cfg.v_emin, cfg.v_emax)
        slots = slots.long()
        hkv, d = self.kh.shape[2:]
        self.kh.view(-1, hkv, d)[slots] = kh
        self.vh.view(-1, hkv, d)[slots] = vh
        if self.kl is not None:
            self.kl.view(-1, hkv, d)[slots] = kl
        if self.vl is not None:
            self.vl.view(-1, hkv, d)[slots] = vl
        self.k_inv.view(-1, hkv)[slots] = ki
        self.v_inv.view(-1, hkv)[slots] = vi


def write_fp16_kv(kh, vh, slots, k, v) -> None:
    """kv_fp16 cache write: fp16(k), fp16(v) at flat slots (round-to-nearest, as the fp16 recipe)."""
    hkv, d = kh.shape[-2], kh.shape[-1]
    slots = slots.long()
    kh.view(-1, hkv, d)[slots] = k.reshape(-1, hkv, d).to(torch.float16)
    vh.view(-1, hkv, d)[slots] = v.reshape(-1, hkv, d).to(torch.float16)


def build_tiles(q_seq: torch.Tensor, q_pos: torch.Tensor, tm: int):
    """Group query tokens into tiles of TM aligned positions of one sequence.

    Returns (tile_seq [NT], tile_t0 [NT], tile_q [NT, TM] (query index or -1)).
    """
    q_seq = q_seq.long()
    q_pos = q_pos.long()
    key = q_seq * (1 << 40) + q_pos // tm
    uniq, inv = torch.unique(key, sorted=True, return_inverse=True)
    nt = uniq.numel()
    tile_seq = (uniq >> 40).to(torch.int32)
    tile_t0 = ((uniq & ((1 << 40) - 1)) * tm).to(torch.int32)
    tile_q = torch.full((nt, tm), -1, dtype=torch.int32, device=q_seq.device)
    tile_q[inv, q_pos % tm] = torch.arange(
        q_seq.numel(), dtype=torch.int32, device=q_seq.device
    )
    return tile_seq, tile_t0, tile_q


# ---------------------------------------------------------------------------
# Triton kernels.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _fold(M, L, O, m, l, o):
        """Fold one chunk (m, l, o) into the running (M, L, O). Shared by both paths."""
        mn = tl.maximum(M, m)
        a = tl.exp2(M - mn)
        b = tl.exp2(m - mn)
        O = O * a[:, None] + o * b[:, None]
        L = L * a + l * b
        return mn, L, O

    @triton.jit
    def _finalize(M, L, O, P_SHIFT: tl.constexpr):
        out = tl.math.div_rn(O * (1.0 / (2**P_SHIFT)), L[:, None])
        lse = (M + tl.log2(L)) * 0.6931471805599453
        return out, lse

    @triton.jit
    def _block(
        blk, m, l, acc, corr, qh, ql, q_inv, q_pos, seq, kvh, kv_len,
        kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride, qk_scale,
        HKV: tl.constexpr, D: tl.constexpr, PS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, MASK: tl.constexpr,
        Q_LO: tl.constexpr, K_LO: tl.constexpr, V_LO: tl.constexpr,
        P_LO: tl.constexpr, LO_LO: tl.constexpr, SEP_ACC: tl.constexpr,
        P_SHIFT: tl.constexpr, KV_FP16: tl.constexpr,
    ):  # fmt: skip
        """One KV block of the online softmax (the per-row arithmetic).

        A block of BLOCK_N keys lies inside one page (PS % BLOCK_N == 0), so it
        is one TMA box. MASK: the block may hold keys past a row's position or
        past the cached length (only blocks at or after the tile's first query
        position); masked keys are zeroed and their scores set to -inf, which
        leaves every other value bit-identical to the unmasked arithmetic.
        """
        k0 = blk * BLOCK_N
        page = tl.load(bt_ptr + seq * bt_stride + k0 // PS)
        slot0 = page * PS + k0 % PS
        kp = k0 + tl.arange(0, BLOCK_N)
        srow = (slot0 + tl.arange(0, BLOCK_N)).to(tl.int64) * HKV + kvh
        kh = kh_d.load([slot0, kvh * D])
        if KV_FP16:
            kinv = tl.full([BLOCK_N], 1.0, tl.float32)
        else:
            kinv = tl.load(kinv_ptr + srow)
        if MASK:
            kvalid = kp < kv_len
            kh = tl.where(kvalid[:, None], kh, 0.0)
            kinv = tl.where(kvalid, kinv, 1.0)
        s = tl.dot(qh, tl.trans(kh))
        if K_LO or Q_LO:
            if SEP_ACC:
                c = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            else:
                c = s
            if K_LO:
                kl = kl_d.load([slot0, kvh * D])
                if MASK:
                    kl = tl.where(kvalid[:, None], kl, 0.0)
                c = tl.dot(qh, tl.trans(kl), c)
            if Q_LO:
                c = tl.dot(ql, tl.trans(kh), c)
            if LO_LO and Q_LO and K_LO:
                c = tl.dot(ql, tl.trans(kl), c)
            if SEP_ACC:
                s = s + c
            else:
                s = c
        s = s * (q_inv[:, None] * kinv[None, :])
        s = s * qk_scale
        if MASK:
            s = tl.where(kp[None, :] <= q_pos[:, None], s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        alpha = tl.exp2(m - m_new)
        p = tl.exp2(s - m_new[:, None])
        # Row sum on the tensor core (p = p0h + p0l, ~22 bits, times ones): its
        # accumulation order is fixed per row, unlike tl.sum, whose order follows
        # the register layout (which changes with BLOCK_M / num_warps).
        p0h = p.to(tl.float16)
        p0l = (p - p0h.to(tl.float32)).to(tl.float16)
        ones = tl.full([BLOCK_N, 16], 1.0, tl.float16)
        l = l * alpha[:, None]
        l = tl.dot(p0h, ones, l)
        l = tl.dot(p0l, ones, l)
        if KV_FP16:
            vinv = tl.full([BLOCK_N], 1.0, tl.float32)
        else:
            vinv = tl.load(vinv_ptr + srow)
        vh = vh_d.load([slot0, kvh * D])
        if MASK:
            vinv = tl.where(kvalid, vinv, 1.0)
            vh = tl.where(kvalid[:, None], vh, 0.0)
        pp = p * (vinv * (2**P_SHIFT))[None, :]
        ph = pp.to(tl.float16)
        acc = acc * alpha[:, None]
        acc = tl.dot(ph, vh, acc)
        if V_LO or P_LO:
            if SEP_ACC:
                corr = corr * alpha[:, None]
                c2 = corr
            else:
                c2 = acc
            if V_LO:
                vl = vl_d.load([slot0, kvh * D])
                if MASK:
                    vl = tl.where(kvalid[:, None], vl, 0.0)
                c2 = tl.dot(ph, vl, c2)
            if P_LO:
                pl = (pp - ph.to(tl.float32)).to(tl.float16)
                c2 = tl.dot(pl, vh, c2)
                if LO_LO and V_LO:
                    c2 = tl.dot(pl, vl, c2)
            if SEP_ACC:
                corr = c2
            else:
                acc = c2
        return m_new, l, acc, corr

    @triton.jit
    def _attend_chunk(
        qh, ql, q_inv, q_pos, seq, kvh, kv_len, blk_lo, blk_hi, n_full,
        kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride, qk_scale,
        HKV: tl.constexpr, D: tl.constexpr, PS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
        Q_LO: tl.constexpr, K_LO: tl.constexpr, V_LO: tl.constexpr,
        P_LO: tl.constexpr, LO_LO: tl.constexpr, SEP_ACC: tl.constexpr,
        P_SHIFT: tl.constexpr, KV_FP16: tl.constexpr,
    ):  # fmt: skip
        """Online softmax over KV blocks [blk_lo, blk_hi) from a fresh state.

        Blocks below ``n_full`` (which depends only on the tile's t0, so it is
        the same in prefill and decode) need no masks.
        """
        m = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l = tl.zeros([BLOCK_M, 16], tl.float32)  # 16 identical columns
        acc = tl.zeros([BLOCK_M, D], tl.float32)
        corr = tl.zeros([BLOCK_M, D], tl.float32)
        mid = tl.minimum(tl.maximum(n_full, blk_lo), blk_hi)
        for blk in range(blk_lo, mid):
            m, l, acc, corr = _block(
                blk, m, l, acc, corr, qh, ql, q_inv, q_pos, seq, kvh, kv_len,
                kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride,
                qk_scale, HKV, D, PS, BLOCK_M, BLOCK_N, False,
                Q_LO, K_LO, V_LO, P_LO, LO_LO, SEP_ACC, P_SHIFT, KV_FP16,
            )  # fmt: skip
        for blk in range(mid, blk_hi):
            m, l, acc, corr = _block(
                blk, m, l, acc, corr, qh, ql, q_inv, q_pos, seq, kvh, kv_len,
                kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride,
                qk_scale, HKV, D, PS, BLOCK_M, BLOCK_N, True,
                Q_LO, K_LO, V_LO, P_LO, LO_LO, SEP_ACC, P_SHIFT, KV_FP16,
            )  # fmt: skip
        # Always materialise acc + corr (corr may be all zero). Measured: when the
        # chunk output is the raw PV accumulator, the 16-row (mma.sync) and
        # 64-row (tcgen05) tiles differ in ~8% of outputs (~2^-18 rel.); with
        # this add they are bitwise equal. Root cause in the Triton lowering not
        # identified (open issue ).
        acc = acc + corr
        return m, tl.max(l, 1), acc

    @triton.jit
    def _hp_attn_kernel(
        qh_ptr, ql_ptr, qinv_ptr, kh_ptr, kl_ptr, vh_ptr, vl_ptr, kinv_ptr,
        vinv_ptr, n_slots, bt_ptr, bt_stride, kv_len_ptr, tile_seq_ptr,
        tile_t0_ptr, tile_q_ptr, out_ptr, lse_ptr, part_o_ptr, part_m_ptr,
        part_l_ptr, nc_max, qk_scale,
        HQ: tl.constexpr, HKV: tl.constexpr, D: tl.constexpr, G: tl.constexpr,
        TM: tl.constexpr, QA: tl.constexpr, PS: tl.constexpr, BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr, CHUNK_BLOCKS: tl.constexpr, SPLIT: tl.constexpr,
        Q_LO: tl.constexpr, K_LO: tl.constexpr, V_LO: tl.constexpr,
        P_LO: tl.constexpr, LO_LO: tl.constexpr, SEP_ACC: tl.constexpr,
        P_SHIFT: tl.constexpr, KV_FP16: tl.constexpr,
    ):  # fmt: skip
        tile = tl.program_id(0)
        kvh = tl.program_id(1)
        cid = tl.program_id(2)
        seq = tl.load(tile_seq_ptr + tile)
        t0 = tl.load(tile_t0_ptr + tile)
        kv_len = tl.load(kv_len_ptr + seq)
        r = tl.arange(0, BLOCK_M)
        slot = r // G
        head = kvh * G + r % G
        qidx = tl.load(tile_q_ptr + tile * TM + slot)
        qvalid = qidx >= 0
        offs_d = tl.arange(0, D)
        qrow = qidx.to(tl.int64) * HQ + head
        q_off = qrow[:, None] * D + offs_d[None, :]
        qh = tl.load(qh_ptr + q_off, mask=qvalid[:, None], other=0.0)
        if Q_LO:
            ql = tl.load(ql_ptr + q_off, mask=qvalid[:, None], other=0.0)
        else:
            ql = qh
        q_inv = tl.load(qinv_ptr + qrow, mask=qvalid, other=1.0)
        q_pos = t0 + slot
        tq = t0 - t0 % QA  # position quantum: the visited blocks depend on tq only
        last_blk = (tq + QA - 1) // BLOCK_N
        n_full = (tq + 1) // BLOCK_N  # blocks whose keys are all <= tq
        n_chunks = last_blk // CHUNK_BLOCKS + 1
        kh_d = tl.make_tensor_descriptor(
            kh_ptr,
            shape=[n_slots, HKV * D],
            strides=[HKV * D, 1],
            block_shape=[BLOCK_N, D],
        )
        kl_d = tl.make_tensor_descriptor(
            kl_ptr,
            shape=[n_slots, HKV * D],
            strides=[HKV * D, 1],
            block_shape=[BLOCK_N, D],
        )
        vh_d = tl.make_tensor_descriptor(
            vh_ptr,
            shape=[n_slots, HKV * D],
            strides=[HKV * D, 1],
            block_shape=[BLOCK_N, D],
        )
        vl_d = tl.make_tensor_descriptor(
            vl_ptr,
            shape=[n_slots, HKV * D],
            strides=[HKV * D, 1],
            block_shape=[BLOCK_N, D],
        )
        if SPLIT:
            blk_lo = cid * CHUNK_BLOCKS
            blk_hi = tl.minimum(blk_lo + CHUNK_BLOCKS, last_blk + 1)
            m, l, o = _attend_chunk(
                qh, ql, q_inv, q_pos, seq, kvh, kv_len, blk_lo, blk_hi, n_full,
                kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride,
                qk_scale, HKV, D, PS, BLOCK_M, BLOCK_N,
                Q_LO, K_LO, V_LO, P_LO, LO_LO, SEP_ACC, P_SHIFT, KV_FP16,
            )  # fmt: skip
            pmask = qvalid & (cid < n_chunks)
            prow = qrow * nc_max + cid
            tl.store(part_m_ptr + prow, m, mask=pmask)
            tl.store(part_l_ptr + prow, l, mask=pmask)
            tl.store(
                part_o_ptr + prow[:, None] * D + offs_d[None, :], o, mask=pmask[:, None]
            )
        else:
            M = tl.full([BLOCK_M], float("-inf"), tl.float32)
            L = tl.zeros([BLOCK_M], tl.float32)
            O = tl.zeros([BLOCK_M, D], tl.float32)
            for c in range(0, n_chunks):
                blk_lo = c * CHUNK_BLOCKS
                blk_hi = tl.minimum(blk_lo + CHUNK_BLOCKS, last_blk + 1)
                m, l, o = _attend_chunk(
                    qh, ql, q_inv, q_pos, seq, kvh, kv_len, blk_lo, blk_hi, n_full,
                    kh_d, kl_d, vh_d, vl_d, kinv_ptr, vinv_ptr, bt_ptr, bt_stride,
                    qk_scale, HKV, D, PS, BLOCK_M, BLOCK_N,
                    Q_LO, K_LO, V_LO, P_LO, LO_LO, SEP_ACC, P_SHIFT, KV_FP16,
                )  # fmt: skip
                M, L, O = _fold(M, L, O, m, l, o)
            out, lse = _finalize(M, L, O, P_SHIFT)
            tl.store(out_ptr + q_off, out, mask=qvalid[:, None])
            tl.store(lse_ptr + qrow, lse, mask=qvalid)

    @triton.jit
    def _combine_kernel(
        part_o_ptr,
        part_m_ptr,
        part_l_ptr,
        q_pos_ptr,
        out_ptr,
        lse_ptr,
        nc_max,
        HQ: tl.constexpr,
        D: tl.constexpr,
        QA: tl.constexpr,
        BLOCK_H: tl.constexpr,
        BLOCK_N: tl.constexpr,
        CHUNK_BLOCKS: tl.constexpr,
        P_SHIFT: tl.constexpr,
    ):
        q = tl.program_id(0)
        h = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        pos = tl.load(q_pos_ptr + q)
        tq = pos - pos % QA
        n_chunks = ((tq + QA - 1) // BLOCK_N) // CHUNK_BLOCKS + 1
        offs_d = tl.arange(0, D)
        qrow = q.to(tl.int64) * HQ + h
        M = tl.full([BLOCK_H], float("-inf"), tl.float32)
        L = tl.zeros([BLOCK_H], tl.float32)
        O = tl.zeros([BLOCK_H, D], tl.float32)
        for c in range(0, n_chunks):
            prow = qrow * nc_max + c
            m = tl.load(part_m_ptr + prow)
            l = tl.load(part_l_ptr + prow)
            o = tl.load(part_o_ptr + prow[:, None] * D + offs_d[None, :])
            M, L, O = _fold(M, L, O, m, l, o)
        out, lse = _finalize(M, L, O, P_SHIFT)
        tl.store(out_ptr + qrow[:, None] * D + offs_d[None, :], out)
        tl.store(lse_ptr + qrow, lse)


# ---------------------------------------------------------------------------
# Host API.
# ---------------------------------------------------------------------------


_ALLOCATOR_SET = False


def _ensure_allocator() -> None:
    global _ALLOCATOR_SET
    if not _ALLOCATOR_SET:  # on-device TMA descriptors need scratch memory
        triton.set_allocator(
            lambda size, align, stream: torch.empty(
                size, dtype=torch.int8, device="cuda"
            )
        )
        _ALLOCATOR_SET = True


def hp_attention_cached(
    qh: torch.Tensor,
    ql: torch.Tensor,
    q_inv: torch.Tensor,
    q_seq: torch.Tensor,
    q_pos: torch.Tensor,
    cache: HPKVCache,
    block_table: torch.Tensor,
    kv_lens: torch.Tensor,
    softmax_scale: float,
    split: bool = False,
    max_kv_len: Optional[int] = None,
    tiles=None,
    decode_tiles: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Causal attention of split queries against the paged hp cache.

    qh, ql [N, Hq, D] fp16, q_inv [N, Hq] fp32; q_seq, q_pos [N]: each query's
    sequence (row of ``block_table``) and absolute position; the cache must hold
    keys 0..q_pos of that sequence (``kv_lens[seq] > q_pos``). ``split=True``
    runs one program per KV chunk and a fold kernel (decode parallelism); the
    result is bitwise the same as ``split=False``; so is ``decode_tiles=True``
    (smaller tiles for few new tokens per sequence). Returns (out fp32 [N, Hq, D],
    lse [N, Hq] natural-log units of scale * q.k).
    """
    cfg = cache.cfg
    n, hq, d = qh.shape
    hkv = cache.kh.shape[2]
    g = hq // hkv
    assert hq % hkv == 0 and cfg.block_m % g == 0
    qa = cfg.block_m // g  # position quantum (numerics): the training tile height
    bm = cfg.decode_block_m if decode_tiles else cfg.block_m
    warps = cfg.decode_num_warps if decode_tiles else cfg.num_warps
    tm = bm // g
    assert bm % g == 0 and qa % tm == 0
    assert cfg.block_n % qa == 0 and cfg.chunk % cfg.block_n == 0
    if tiles is None:
        tiles = build_tiles(q_seq, q_pos, tm)
    tile_seq, tile_t0, tile_q = tiles
    nt = tile_seq.numel()
    out = torch.empty((n, hq, d), dtype=torch.float32, device=qh.device)
    lse = torch.empty((n, hq), dtype=torch.float32, device=qh.device)
    chunk_blocks = cfg.chunk // cfg.block_n
    qk_scale = softmax_scale * LOG2E
    if split:
        if max_kv_len is None:
            max_kv_len = int(q_pos.max().item()) + 1
        nc_max = triton.cdiv(triton.cdiv(max_kv_len + qa, cfg.block_n), chunk_blocks)
        part_o = torch.empty((n, hq, nc_max, d), dtype=torch.float32, device=qh.device)
        part_m = torch.empty((n, hq, nc_max), dtype=torch.float32, device=qh.device)
        part_l = torch.empty_like(part_m)
    else:
        nc_max = 1
        part_o = part_m = part_l = out
    kl = cache.kl if cache.kl is not None else cache.kh
    vl = cache.vl if cache.vl is not None else cache.vh
    k_inv = cache.k_inv if cache.k_inv is not None else out  # never read under kv_fp16
    v_inv = cache.v_inv if cache.v_inv is not None else out
    _ensure_allocator()
    assert cache.page_size % cfg.block_n == 0
    _hp_attn_kernel[(nt, hkv, nc_max if split else 1)](
        qh, ql if cfg.q_lo else qh, q_inv,
        cache.kh, kl, cache.vh, vl, k_inv, v_inv,
        cache.kh.shape[0] * cache.page_size, block_table, block_table.stride(0), kv_lens,
        tile_seq, tile_t0, tile_q,
        out, lse, part_o, part_m, part_l, nc_max, qk_scale,
        HQ=hq, HKV=hkv, D=d, G=g, TM=tm, QA=qa, PS=cache.page_size,
        BLOCK_M=bm, BLOCK_N=cfg.block_n, CHUNK_BLOCKS=chunk_blocks,
        SPLIT=split, Q_LO=cfg.q_lo, K_LO=cfg.k_lo, V_LO=cfg.v_lo, P_LO=cfg.p_lo,
        LO_LO=cfg.lo_lo, SEP_ACC=cfg.sep_acc, P_SHIFT=cfg.p_shift, KV_FP16=cfg.kv_fp16,
        num_warps=warps, num_stages=cfg.num_stages, enable_fp_fusion=False,
    )  # fmt: skip
    if split:
        block_h = min(hq, 8)
        _combine_kernel[(n, hq // block_h)](
            part_o, part_m, part_l, q_pos.to(torch.int32).contiguous(), out, lse,
            nc_max, HQ=hq, D=d, QA=qa, BLOCK_H=block_h, BLOCK_N=cfg.block_n,
            CHUNK_BLOCKS=chunk_blocks, P_SHIFT=cfg.p_shift,
            num_warps=4, enable_fp_fusion=False,
        )  # fmt: skip
    return out, lse


def prefill_layout(cu_seqlens: torch.Tensor, page_size: int):
    """Contiguous pages per sequence for a packed (THD) batch.

    Returns (q_seq, q_pos, slots, block_table, kv_lens, num_pages).
    """
    cu = cu_seqlens.long()
    lens = cu[1:] - cu[:-1]
    b = lens.numel()
    t = int(cu[-1].item())
    dev = cu.device
    pages = (lens + page_size - 1) // page_size
    page_start = torch.cumsum(pages, 0) - pages
    max_pages = max(int(pages.max().item()), 1)
    block_table = page_start[:, None] + torch.arange(max_pages, device=dev)[None, :]
    q_seq = torch.repeat_interleave(torch.arange(b, device=dev), lens)
    q_pos = torch.arange(t, device=dev) - cu[:-1][q_seq]
    slots = page_start[q_seq] * page_size + q_pos
    num_pages = int(pages.sum().item())
    return (
        q_seq.to(torch.int32),
        q_pos.to(torch.int32),
        slots,
        block_table.to(torch.int32).contiguous(),
        lens.to(torch.int32),
        num_pages,
    )


def hp_attention_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cfg: HPAttnConfig = HPAttnConfig(),
    softmax_scale: Optional[float] = None,
    page_size: int = 64,
    split: bool = False,
    return_cache: bool = False,
):
    """Training forward / engine prefill: causal attention over packed sequences.

    q [T, Hq, D], k, v [T, Hkv, D] fp32 (post-RoPE); cu_seqlens [B + 1].
    Returns (out fp32 [T, Hq, D], lse [T, Hq]) (+ the cache it built).
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    q_seq, q_pos, slots, bt, kv_lens, num_pages = prefill_layout(cu_seqlens, page_size)
    cache = HPKVCache(num_pages, page_size, k.shape[1], k.shape[2], cfg, q.device)
    cache.write(slots, k, v)
    qh, ql, q_inv = split_q(q, cfg)
    max_len = int(kv_lens.max().item())
    out, lse = hp_attention_cached(
        qh, ql, q_inv, q_seq, q_pos, cache, bt, kv_lens, softmax_scale,
        split=split, max_kv_len=max_len,
    )  # fmt: skip
    if return_cache:
        return out, lse, (cache, bt, kv_lens)
    return out, lse


# ---------------------------------------------------------------------------
# Sync-free layouts (static shapes: CUDA-graph safe, no host reads).
# ---------------------------------------------------------------------------


def static_tiles(
    cu_q: torch.Tensor, kv_lens: torch.Tensor, n_tokens: int, tm: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tiles of TM aligned positions for packed queries, without host syncs.

    ``cu_q`` [B + 1]: cumulative query counts (padded sequences repeat the last value);
    ``kv_lens`` [B]: key count of each sequence *including* its queries, which are its
    last ``cu_q[b + 1] - cu_q[b]`` positions. Query rows ``>= cu_q[-1]`` (padding) are
    in no tile. The tile count is the static bound ``n_tokens // tm + 2 B``; unused
    tiles hold no query (``tile_q == -1``, sequence 0, t0 0).
    Returns (tile_seq, tile_t0, tile_q, q_seq, q_pos, q_valid) (q_* per query row).
    """
    dev = cu_q.device
    cu = cu_q.long()
    b = kv_lens.numel()
    nt = n_tokens // tm + 2 * b
    qlen = cu[1:] - cu[:-1]
    kvl = kv_lens.long()
    first = kvl - qlen
    t = torch.arange(n_tokens, device=dev)
    seq = torch.searchsorted(cu[1:].contiguous(), t, right=True).clamp_(max=max(b - 1, 0))
    valid = t < cu[-1]
    pos = first[seq] + (t - cu[:-1][seq])
    pos = torch.where(valid, pos, torch.zeros_like(pos))
    f_t = torch.div(first, tm, rounding_mode="floor")
    ntile = torch.where(
        qlen > 0, torch.div(kvl - 1, tm, rounding_mode="floor") - f_t + 1, torch.zeros_like(qlen)
    )
    tstart = torch.cumsum(ntile, 0) - ntile
    tile = tstart[seq] + torch.div(pos, tm, rounding_mode="floor") - f_t[seq]
    tile = torch.where(valid, tile, torch.full_like(tile, nt))  # row nt: discarded
    tile_seq = torch.zeros(nt + 1, dtype=torch.int32, device=dev)
    tile_t0 = torch.zeros(nt + 1, dtype=torch.int32, device=dev)
    tile_q = torch.full((nt + 1, tm), -1, dtype=torch.int32, device=dev)
    tile_seq[tile] = seq.to(torch.int32)
    tile_t0[tile] = (pos - pos % tm).to(torch.int32)
    tile_q[tile, pos % tm] = t.to(torch.int32)
    return tile_seq[:nt], tile_t0[:nt], tile_q[:nt].contiguous(), seq, pos, valid


def static_prefill_layout(cu_seqlens: torch.Tensor, n_tokens: int, page_size: int):
    """``prefill_layout`` without host syncs: contiguous pages per packed sequence.

    Page count bound ``n_tokens // page_size + B + 1`` (the last page receives the
    padding rows); block-table width ``ceil(n_tokens / page_size)``.
    Returns (slots [T], block_table [B, W] int32, kv_lens [B] int32, num_pages).
    """
    dev = cu_seqlens.device
    cu = cu_seqlens.long()
    b = cu.numel() - 1
    lens = cu[1:] - cu[:-1]
    pages = torch.div(lens + page_size - 1, page_size, rounding_mode="floor")
    page_start = torch.cumsum(pages, 0) - pages
    num_pages = n_tokens // page_size + b + 1
    width = max((n_tokens + page_size - 1) // page_size, 1)
    bt = (page_start[:, None] + torch.arange(width, device=dev)[None, :]).clamp_(
        max=num_pages - 1
    )
    t = torch.arange(n_tokens, device=dev)
    seq = torch.searchsorted(cu[1:].contiguous(), t, right=True).clamp_(max=max(b - 1, 0))
    valid = t < cu[-1]
    slots = page_start[seq] * page_size + (t - cu[:-1][seq])
    slots = torch.where(valid, slots, torch.full_like(slots, (num_pages - 1) * page_size))
    return slots, bt.to(torch.int32).contiguous(), lens.to(torch.int32), num_pages


# ---------------------------------------------------------------------------
# Autograd: hp forward, FlashAttention fp16 backward at the hi parts.
# ---------------------------------------------------------------------------


class HPAttention(torch.autograd.Function):
    """out = hp_attention_prefill(q, k, v); backward = FA2 fp16 varlen backward.

    The backward recomputes P from fp16(q), fp16(k) with the hp log-sum-exp and
    uses the hp output for D = rowsum(dO * O). dO is scaled by a power of two
    from its amax into fp16 (as in fp16_gemm.FromFp16Scaled) and unscaled after.
    """

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens, max_seqlen, cfg, softmax_scale):  # noqa: D102
        out, lse = hp_attention_prefill(q, k, v, cu_seqlens, cfg, softmax_scale)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens)
        ctx.max_seqlen = max_seqlen
        ctx.scale = softmax_scale
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

        q, k, v, out, lse, cu = ctx.saved_tensors
        g = dout.float()
        amax = g.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.exp2(torch.floor(-torch.log2(amax)).clamp(-60, 60))
        q16, k16, v16 = (t.to(torch.float16) for t in (q, k, v))
        do16 = (g * scale).to(torch.float16)
        o16 = out.to(torch.float16)  # D = rowsum(dO * O) scales with dO
        dq = torch.empty_like(q16)
        dk = torch.empty_like(k16)
        dv = torch.empty_like(v16)
        cu32 = cu.to(torch.int32)
        _flash_attn_varlen_backward(
            do16, q16, k16, v16, o16, lse.t().contiguous(), dq, dk, dv,
            cu32, cu32, ctx.max_seqlen, ctx.max_seqlen, 0.0, ctx.scale, True,
            -1, -1, 0.0, None, False, None,
        )  # fmt: skip
        inv = 1.0 / scale
        return (
            dq.float() * inv,
            dk.float() * inv,
            dv.float() * inv,
            None,
            None,
            None,
            None,
        )


def hp_attention(
    q, k, v, cu_seqlens, max_seqlen, cfg=HPAttnConfig(), softmax_scale=None
):
    """Differentiable hp attention for the training forward (packed THD)."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    return HPAttention.apply(q, k, v, cu_seqlens, max_seqlen, cfg, softmax_scale)


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    softmax_scale: Optional[float] = None,
    dtype=torch.float64,
) -> torch.Tensor:
    """Causal GQA attention in ``dtype`` (fp64 reference), packed sequences."""
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])
    g = q.shape[1] // k.shape[1]
    outs = []
    cu = cu_seqlens.tolist()
    for s, e in zip(cu[:-1], cu[1:]):
        qq = q[s:e].to(dtype).transpose(0, 1)
        kk = k[s:e].to(dtype).repeat_interleave(g, 1).transpose(0, 1)
        vv = v[s:e].to(dtype).repeat_interleave(g, 1).transpose(0, 1)
        sc = qq @ kk.transpose(1, 2) * softmax_scale
        n = e - s
        mask = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        sc = sc.masked_fill(~mask, float("-inf"))
        outs.append((sc.softmax(-1) @ vv).transpose(0, 1))
    return torch.cat(outs)
