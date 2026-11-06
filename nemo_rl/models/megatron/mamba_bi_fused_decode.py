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
"""The batch-invariant Mamba decode step (kernel ask D2, FEEDBACK_1101-decode): MCore's z-free buffered decode
(``batch_invariant_decode.batch_invariant_decode_buffered_scan``) in two or three launches per Mamba layer.

The step's output and the committed SSM state are, bit for bit, the training chunk scan's for the target row
(MCore ``_mamba_chunk_scan_decode_rows`` on the pinned SSD tiles, i.e. the learner's forward through
``mamba_ssd_parity``). Launches:

1. ``_prep_cb_kernel`` (Triton; grid rows x (cumsum head blocks + CB tiles)). Programs of the first kind
   append the current token to the slot's replay rows, write the bookkeeping, and run MCore's
   ``_chunk_cumsum_fwd_kernel`` statements on the pinned [BLOCK_H, chunk] tile and launch (so ``tl.cumsum``
   has the training layout). Programs of the second kind compute row tr of CB = C B^T (columns <= tr) for one
   (group, CB_BLOCK_N-column tile) with the pinned CB tile's per-element chain; they read the current token's
   B / C row from the step inputs, so they run concurrently with the append.
2. ``mamba_bi_scan_cuda`` (CUDA; one program per (row, head)): the target-row chunk scan, D and the gate, and
   the slot's cursor advance. When the step's programs fit in one wave of the GPU (``small_step``), the same
   program also computes the CB row and the chunk boundary, and step 1 launches no CB programs.
3. Otherwise, rows that fill their chunk: the crossing-row chunk state + state passing kernel
   (``mamba_bi_scan_row.chunk_state_pass_rows``; MCore main ships it as
   ``batch_invariant_decode_rows.chunk_state_passing_rows``).

``install()`` (called by ``mamba_bi_decode.install()``) makes this the decode step for z-free buffers; buffers
with z (configurations ``supported_config`` rejects, which ``mamba_bi_decode`` allocates with z) keep MCore's
path. After the merge with Megatron-LM main this module plugs into the same MCore hook (research/cleanup/
MERGE_MAIN_TRIAL.md §7).

Per rank and row-local: programs per (row, head) or (row, group); no communication, no host sync, fixed grids,
no allocation after the first call (CUDA-graph capturable). Cost scales with rows x heads per rank (1/TP).
"""

from __future__ import annotations

from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

# CB-row programs cover CB_BLOCK_N columns each: chunk / 32 = 4 programs per (row, group) at chunk 128, those past
# the target row exit at once (on average half), and 32 columns are a whole number of MMA n-tiles. Independent of
# model scale (chunk is the SSD chunk size).
CB_BLOCK_N = 32

_ORIG: dict[str, Any] = {}

# Lookahead rows (multi-token tail decode, ``tail_mtd``): at most LOOK_MAX rows after a tail's row 0 in one step.
LOOK_MAX = 8
_LOOK: dict = {}


def lookahead_buffers(max_rows: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(offset [max_rows] int32, chain [max_rows, LOOK_MAX] int32) of the decode entries, shared by every Mamba layer.

    offset[e] = 0 for a one-token decode row (the default; the step is then unchanged); offset[e] = i >= 1 for row i
    of a tail, whose row j < i is decode entry chain[e, j]. Written by the tail_mtd layout before each forward (fixed
    storage: CUDA-graph capturable). Grows only before graph capture.
    """
    dev = torch.device(device)
    key = dev.index if dev.index is not None else torch.cuda.current_device()
    cur = _LOOK.get(key)
    if cur is None or cur[0].numel() < max_rows:
        off = torch.zeros(max_rows, dtype=torch.int32, device=dev)
        chain = torch.full((max_rows, LOOK_MAX), -1, dtype=torch.int32, device=dev)
        if cur is not None:
            off[: cur[0].numel()].copy_(cur[0])
            chain[: cur[1].shape[0]].copy_(cur[1])
        cur = _LOOK[key] = (off, chain)
    return cur


def lookahead_rows(max_rows: int, device) -> torch.Tensor:
    """srow [max_rows, LOOK_MAX] int32 of the decode entries, shared by every Mamba layer (tree_spec).

    For a node entry e at offset (depth) d >= 1, srow[e, j] (j < d) is the scratch replay row holding chunk row
    w0 + 1 + j of e's path (its ancestors' rows for j < d - 1, its own for j = d - 1): the node's prep program
    writes its x / B / C / dt there instead of the slot's row, and the scan reads the path's rows from there. -1
    (tail_mtd's chains, every one-token row) means the slot's own row, as before. Written by the layout before each
    forward; fixed storage (CUDA-graph capturable).
    """
    dev = torch.device(device)
    key = ("srow", dev.index if dev.index is not None else torch.cuda.current_device())
    cur = _LOOK.get(key)
    if cur is None or cur.shape[0] < max_rows:
        t = torch.full((max_rows, LOOK_MAX), -1, dtype=torch.int32, device=dev)
        if cur is not None:
            t[: cur.shape[0]].copy_(cur)
        cur = _LOOK[key] = t
    return cur

if HAVE_TRITON:
    from megatron.core.ssm.ops.mamba2.ssd_chunk_state import softplus as _mcore_softplus

    @triton.jit
    def _prep_cb_kernel(
        bi_ptr,
        nbuf_ptr,
        x_ptr,
        dt_ptr,
        b_ptr,
        c_ptr,
        sx_ptr,
        sdt_ptr,
        sb_ptr,
        sc_ptr,
        rows_ptr,
        flags_ptr,
        starts_ptr,
        look_off_ptr,
        look_chain_ptr,
        look_srow_ptr,
        scr_x_ptr,
        scr_dt_ptr,
        scr_b_ptr,
        scr_c_ptr,
        A_ptr,
        dt_bias_ptr,
        dt_out_ptr,
        dA_cs_ptr,
        cb_ptr,
        stride_x,
        stride_dt,
        stride_b,
        stride_c,
        stride_dt_out_head: tl.int64,
        stride_dt_out_chunk: tl.int64,
        stride_dA_cs_head: tl.int64,
        stride_dA_cs_chunk: tl.int64,
        stride_sdt_seqlen: tl.int64,
        stride_sdt_head: tl.constexpr,
        stride_cb_row: tl.int64,
        stride_cb_grp: tl.int64,
        stride_chain: tl.int64,
        stride_srow: tl.int64,
        n_x,
        nheads: tl.constexpr,
        ngroups: tl.constexpr,
        dstate: tl.constexpr,
        n_hblocks: tl.constexpr,
        dt_min: tl.constexpr,
        dt_max: tl.constexpr,
        CHUNK: tl.constexpr,
        HAS_DT_BIAS: tl.constexpr,
        DT_SOFTPLUS: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_CHUNK: tl.constexpr,
        CB_BLOCK_N: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
        BLOCK: tl.constexpr,
        LOOK_MAX: tl.constexpr,
    ):
        """Program (e, j): j < n_hblocks: append + bookkeeping + MCore's chunk cumsum for heads block j;
        j >= n_hblocks: row w of CB (columns <= w) for one (group, column tile), w = num_buffered[slot].

        num_buffered is read here and advanced by the scan kernel.
        """
        e = tl.program_id(axis=0).to(tl.int64)
        hb = tl.program_id(axis=1)
        s = tl.load(bi_ptr + e)
        if hb == 0:
            tl.store(starts_ptr + e, s * CHUNK)
        if s < 0:
            if hb == 0:
                tl.store(rows_ptr + e, -1)
                tl.store(flags_ptr + e, 0)
            return
        # Lookahead rows (multi-token tail decode, tail_mtd): entry e is row `off` of a tail whose row 0 targets the
        # slot's cursor w0; it targets w0 + off. Its predecessors' rows w0 .. w0 + off - 1 are read from their own
        # step inputs (entries look_chain[e, j]), never from the replay buffer their programs write concurrently.
        # off = 0 (every one-token decode row): the step is unchanged.
        off = tl.load(look_off_ptr + e)
        w0 = tl.load(nbuf_ptr + s)
        w = w0 + off
        # Tree node rows (tree_spec): the node's own row goes to its scratch replay row (srow >= 0), never to the
        # slot's row its siblings share; -1 (tail_mtd chains, one-token rows): the slot's row as before.
        own = tl.load(look_srow_ptr + e * stride_srow + tl.maximum(off - 1, 0), mask=off > 0, other=-1)
        n_bc = ngroups * dstate
        if hb >= n_hblocks:
            # ---- CB row w: the pinned CB tile's per-element chain (fp32 / TF32 MMA over dstate, k ascending,
            # zero start) on a 16-row MMA tile holding only row w; one dot over dstate ----
            j = hb - n_hblocks
            pid_g = j % ngroups
            pid_n = j // ngroups
            if pid_n * CB_BLOCK_N > w:
                return
            offs_m = (w // 16) * 16 + tl.arange(0, 16)
            row = offs_m == w
            offs_n = pid_n * CB_BLOCK_N + tl.arange(0, CB_BLOCK_N)
            offs_k = tl.arange(0, BLOCK_SIZE_DSTATE)
            # C[w] and B[w] are this step's token: read from the step inputs (the append runs concurrently)
            a = tl.load(
                c_ptr + e * stride_c + pid_g * dstate + offs_m[:, None] * 0 + offs_k[None, :],
                mask=row[:, None] & (offs_k[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            buf_b = sb_ptr + (s * CHUNK + offs_n[None, :]).to(tl.int64) * n_bc + pid_g * dstate + offs_k[:, None]
            cur_b = b_ptr + e * stride_b + pid_g * dstate + offs_k[:, None] + offs_n[None, :] * 0
            src_b = tl.where(offs_n[None, :] == w, cur_b, buf_b)
            for j in tl.static_range(LOOK_MAX):
                pj = tl.load(look_chain_ptr + e * stride_chain + j)
                pred_b = b_ptr + pj.to(tl.int64) * stride_b + pid_g * dstate + offs_k[:, None] + offs_n[None, :] * 0
                src_b = tl.where((offs_n[None, :] == w0 + j) & (j < off), pred_b, src_b)
            b = tl.load(
                src_b,
                mask=(offs_k[:, None] < dstate) & (offs_n[None, :] <= w),
                other=0.0,
            ).to(tl.float32)
            acc = tl.dot(a, b)
            tl.store(
                cb_ptr + e * stride_cb_row + pid_g * stride_cb_grp + offs_m[:, None] * 0 + offs_n[None, :],
                acc,
                mask=row[:, None] & (offs_n[None, :] <= w),
            )
            return
        row_buf = (s * CHUNK + w).to(tl.int64)
        if own >= 0:
            dx = scr_x_ptr + own.to(tl.int64) * n_x
            db = scr_b_ptr + own.to(tl.int64) * n_bc
            dc = scr_c_ptr + own.to(tl.int64) * n_bc
            ddt = scr_dt_ptr + own.to(tl.int64) * nheads
        else:
            dx = sx_ptr + row_buf * n_x
            db = sb_ptr + row_buf * n_bc
            dc = sc_ptr + row_buf * n_bc
            ddt = sdt_ptr + row_buf * nheads
        offs = tl.arange(0, BLOCK)
        # x and B / C shares of this program
        per_x = tl.cdiv(n_x, n_hblocks)
        lo = hb * per_x
        hi = tl.minimum(lo + per_x, n_x)
        for o in range(lo, hi, BLOCK):
            m = o + offs < hi
            v = tl.load(x_ptr + e * stride_x + o + offs, mask=m)
            tl.store(dx + o + offs, v, mask=m)
        per_bc = tl.cdiv(n_bc, n_hblocks)
        lo = hb * per_bc
        hi = tl.minimum(lo + per_bc, n_bc)
        for o in range(lo, hi, BLOCK):
            m = o + offs < hi
            v = tl.load(b_ptr + e * stride_b + o + offs, mask=m)
            tl.store(db + o + offs, v, mask=m)
            v = tl.load(c_ptr + e * stride_c + o + offs, mask=m)
            tl.store(dc + o + offs, v, mask=m)
        offs_h = hb * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        v = tl.load(dt_ptr + e * stride_dt + offs_h, mask=offs_h < nheads)
        tl.store(ddt + offs_h, v, mask=offs_h < nheads)
        if hb == 0:
            tl.store(rows_ptr + e, w)
            tl.store(flags_ptr + e, ((w + 1 == CHUNK) & (off == 0)).to(tl.int32))
        tl.debug_barrier()  # the dt row just written is read below by other threads of this program

        # ---- MCore _chunk_cumsum_fwd_kernel (HAS_TARGET_ROWS), statement for statement ----
        chunk_seqlen_start = s * CHUNK
        chunk_seqlen_end = chunk_seqlen_start + CHUNK
        dt_win = sdt_ptr + chunk_seqlen_start * stride_sdt_seqlen
        dt_out_ptr += e * stride_dt_out_chunk
        dA_cs_ptr += e * stride_dA_cs_chunk
        offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
        dt_ptrs = dt_win + (offs_h[:, None] * stride_sdt_head + offs_c[None, :] * stride_sdt_seqlen)
        for j in tl.static_range(LOOK_MAX):
            pj = tl.load(look_chain_ptr + e * stride_chain + j)
            pred_dt = dt_ptr + pj.to(tl.int64) * stride_dt + offs_h[:, None] + offs_c[None, :] * 0
            dt_ptrs = tl.where((offs_c[None, :] == w0 + j) & (j < off), pred_dt, dt_ptrs)
        # a node's own dt is not in the slot's row (it went to its scratch row): read it from its step input
        own_dt = dt_ptr + e * stride_dt + offs_h[:, None] + offs_c[None, :] * 0
        dt_ptrs = tl.where((offs_c[None, :] == w) & (own >= 0), own_dt, dt_ptrs)
        A_ptrs = A_ptr + offs_h
        dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :])
        dA_cs_ptrs = dA_cs_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :])
        chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start
        dtv = tl.load(
            dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0
        ).to(tl.float32)
        if HAS_DT_BIAS:
            dt_bias = tl.load(dt_bias_ptr + offs_h, mask=offs_h < nheads, other=0.0).to(tl.float32)
            dtv += dt_bias[:, None]
        if DT_SOFTPLUS:
            dtv = tl.where(dtv <= 20.0, _mcore_softplus(dtv), dtv)
        dtv = tl.clamp(dtv, dt_min, dt_max)
        dtv = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dtv, 0.0)
        tl.store(dt_out_ptrs, dtv, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < CHUNK))
        A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dA = dtv * A[:, None]
        dA_cs = tl.cumsum(dA, axis=1)
        tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < CHUNK))


# ---------------------------------------------------------------------------
# Pinned tiles and support checks
# ---------------------------------------------------------------------------


def _single_config(kernel):
    configs = getattr(kernel, "configs", [])
    return configs[0] if len(configs) == 1 else None


def _warpgroup_tile(cfg) -> bool:
    """A tile Triton compiles to warpgroup / tcgen05 MMAs (4+ warps and a 64+-row M tile) on sm_90 / sm_100."""
    return cfg.num_warps >= 4 and cfg.kwargs.get("BLOCK_SIZE_M", 0) >= 64


def pinned_tiles(dstate: int, chunk: int, capability=None):
    """(cumsum, scan, bmm, state) pinned configs whose per-element chains this step reproduces, else None.

    Every mirrored SSD kernel must be pinned to one config (``mamba_ssd_parity.pin_engine_kernels``): the
    per-element chain of each tile is then fixed. The step issues mma.sync TF32 and reproduces a tile's chains
    exactly when
    - the tile's K blocking adds no masked (zero) K steps beyond the true extent: the chunk scan's BLOCK_K divides
      its M (the decode row's K bound is the end of its M tile), and the CB and chunk-state tiles' BLOCK_K divide
      d_state and chunk;
    - the tile's instruction gives mma.sync's bits: the chunk-scan tile must itself be an mma.sync tile (a
      warpgroup / tcgen05 scan tile differs, MCore main's ``row_kernel_configs`` rule); a warpgroup / tcgen05 CB or
      chunk-state tile is accepted on sm_100, where tcgen05 kind::tf32 equals mma.sync bitwise for these tiles
      (research/mamba_dk2/mma_equiv.py), and not on sm_90 (wgmma, not measured).
    """
    from megatron.core.ssm.ops.mamba2 import ssd_bmm, ssd_chunk_scan, ssd_chunk_state

    cs = _single_config(ssd_chunk_state._chunk_cumsum_fwd_kernel)
    sc = _single_config(ssd_chunk_scan._chunk_scan_fwd_kernel)
    bm = _single_config(ssd_bmm._bmm_chunk_fwd_kernel)
    st = _single_config(ssd_chunk_state._chunk_state_fwd_kernel)
    if any(c is None for c in (cs, sc, bm, st)):
        return None
    m, k = sc.kwargs["BLOCK_SIZE_M"], sc.kwargs["BLOCK_SIZE_K"]
    if m % k or chunk % m or dstate % bm.kwargs["BLOCK_SIZE_K"] or chunk % st.kwargs["BLOCK_SIZE_K"]:
        return None
    cap = tuple(capability) if capability is not None else torch.cuda.get_device_capability()
    if cap >= (9, 0) and _warpgroup_tile(sc):
        return None
    if cap == (9, 0) and (_warpgroup_tile(bm) or _warpgroup_tile(st)):
        return None
    return cs, sc, bm, st


def _ssd_precision_raised() -> bool:
    """The ssd_scan_x3 precision preset (tf32x3 chunk-scan dots) is installed: this step's chain is TF32."""
    from nemo_rl.models.megatron import ssd_x3

    return ssd_x3._STATE["installed"]


def supported_config(nheads: int, headdim: int, ngroups: int, dstate: int, chunk: int, dtype, device) -> bool:
    """Whether this step serves a Mamba layer with these per-rank shapes and fp32 replay buffers on ``device``.

    Decided when the buffers are allocated (``mamba_bi_decode``): a layer this step does not serve gets z
    buffers and runs MCore's path. Requirements: fp32 buffers (the fp16-GEMM recipe's in_proj output), the CUDA
    scan's shape and architecture rules (``mamba_bi_scan_cuda.supported``), every mirrored SSD tile pinned with an
    exact-chain blocking and instruction (``pinned_tiles``), the cumsum tile dividing the heads, and the default
    TF32 SSD numerics.
    """
    if not HAVE_TRITON or dtype != torch.float32 or nheads % ngroups:
        return False
    from nemo_rl.models.megatron import mamba_bi_scan_cuda

    tiles = pinned_tiles(dstate, chunk, torch.cuda.get_device_capability(device))
    if tiles is None or nheads % tiles[0].kwargs["BLOCK_SIZE_H"]:
        return False
    if _ssd_precision_raised():
        return False
    return mamba_bi_scan_cuda.supported(headdim, dstate, chunk, tiles[1].kwargs["BLOCK_SIZE_M"], device)


def small_step(n: int, nheads: int, device, scan_module) -> bool:
    """CB row and boundary inside the scan programs iff the step's programs fit in one wave of the GPU.

    Then the extra in-program work adds latency only (no program waits for a free SM) and the step saves the CB
    launch and the boundary launch; beyond one wave it would add throughput cost to every program. The wave is
    the CUDA occupancy of the small kernel times the SM count (Super-L22 at TP2: 64 heads, 3 programs / SM x 148
    SMs -> up to 6 rows).
    """
    key = (device.index if device.index is not None else torch.cuda.current_device(), id(scan_module))
    wave = _WAVE.get(key)
    if wave is None:
        wave = _WAVE[key] = int(scan_module.resident_programs_per_sm(True)) * torch.cuda.get_device_properties(
            device
        ).multi_processor_count
    return n * nheads <= wave


_WAVE: dict = {}


def _scratch(buffers, name, shape, dtype):
    t = getattr(buffers, name, None)
    if t is None or t.shape != shape:
        t = torch.empty(shape, device=buffers.x.device, dtype=dtype)
        setattr(buffers, name, t)
    return t


def decode_step(buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state):
    """One decode token per entry on z-free fp32 buffers (``batch_invariant_decode_buffered_scan``'s contract)."""
    from nemo_rl.models.megatron import mamba_bi_scan_cuda
    from nemo_rl.models.megatron import mamba_bi_scan_row as sr

    n, tokens_per_entry, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2], B.shape[-1]
    max_req, chunk = buffers.x.shape[0], buffers.x.shape[1]
    assert tokens_per_entry == 1, "batch-invariant Mamba decode: one token per request"
    assert n <= buffers.out.shape[0], "increase max_requests"
    assert batch_indices.dtype == torch.int32 and buffers.num_buffered.dtype == torch.int32
    assert ssm_state.dtype == torch.float32 and z.dtype == torch.float32
    tiles = pinned_tiles(dstate, chunk, torch.cuda.get_device_capability(x.device))
    assert tiles is not None, "batch-invariant Mamba decode: the SSD autotuners must stay pinned"
    cs_cfg, sc_cfg = tiles[0], tiles[1]
    starts_all = getattr(buffers, "chunk_starts", None)
    if starts_all is None:
        starts_all = buffers.chunk_starts = torch.empty_like(buffers.target_rows)
    out = buffers.out[:n]
    target_rows = buffers.target_rows[:n]
    chunk_flags = buffers.chunk_flags[:n]
    chunk_starts = starts_all[:n]
    x2, dt2 = x[:, 0].flatten(1), dt[:, 0]
    b2, c2 = B[:, 0].flatten(1), C[:, 0].flatten(1)
    for t in (x2, dt2, b2, c2):
        assert t.stride(-1) == 1 and t.dtype == torch.float32
    assert out.is_contiguous() and buffers.x.is_contiguous() and buffers.B.is_contiguous()
    assert buffers.C.is_contiguous() and buffers.dt.is_contiguous()
    dt_out_all = _scratch(buffers, "bi_dt_out", (nheads, max_req, chunk), torch.float32)
    dA_all = _scratch(buffers, "bi_dA_cumsum", (nheads, max_req, chunk), torch.float32)
    cb = _scratch(buffers, "bi_cb_row", (max_req, ngroups, chunk), torch.float32)
    if n == 0:
        return out.unsqueeze(1)
    scan = mamba_bi_scan_cuda.module(headdim, dstate, chunk, sc_cfg.kwargs["BLOCK_SIZE_M"], x.device)
    small = small_step(n, nheads, x.device, scan)
    bh = cs_cfg.kwargs["BLOCK_SIZE_H"]
    n_hb = triton.cdiv(nheads, bh)
    n_cb = 0 if small else ngroups * triton.cdiv(chunk, CB_BLOCK_N)
    sdt = buffers.dt.view(-1, nheads)
    look_off, look_chain = lookahead_buffers(buffers.out.shape[0], x.device)
    look_srow = lookahead_rows(buffers.out.shape[0], x.device)
    scr = getattr(buffers, "scratch", None)  # (x, dt, B, C) scratch replay rows (mamba_bi_lookahead), else none
    tree = scr is not None
    if scr is None:
        scr = (buffers.x, buffers.dt, buffers.B, buffers.C)  # never written: every srow is -1 without scratch rows
    _prep_cb_kernel[(n, n_hb + n_cb)](
        batch_indices,
        buffers.num_buffered,
        x2,
        dt2,
        b2,
        c2,
        buffers.x,
        buffers.dt,
        buffers.B,
        buffers.C,
        target_rows,
        chunk_flags,
        chunk_starts,
        look_off,
        look_chain,
        look_srow,
        scr[0],
        scr[1],
        scr[2],
        scr[3],
        A,
        dt_bias,
        dt_out_all,
        dA_all,
        cb,
        x2.stride(0),
        dt2.stride(0),
        b2.stride(0),
        c2.stride(0),
        dt_out_all.stride(0),
        dt_out_all.stride(1),
        dA_all.stride(0),
        dA_all.stride(1),
        sdt.stride(0),
        sdt.stride(1),
        cb.stride(0),
        cb.stride(1),
        look_chain.stride(0),
        look_srow.stride(0),
        nheads * headdim,
        nheads=nheads,
        ngroups=ngroups,
        dstate=dstate,
        n_hblocks=n_hb,
        dt_min=0.0,
        dt_max=float("inf"),
        CHUNK=chunk,
        HAS_DT_BIAS=dt_bias is not None,
        DT_SOFTPLUS=True,
        BLOCK_SIZE_H=bh,
        BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk),
        CB_BLOCK_N=CB_BLOCK_N,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        BLOCK=1024,
        LOOK_MAX=LOOK_MAX,
        num_warps=cs_cfg.num_warps,
        num_stages=cs_cfg.num_stages,
    )
    xs = buffers.x.view(-1, nheads, headdim)
    Bs = buffers.B.view(-1, ngroups, dstate)
    Cs = buffers.C.view(-1, ngroups, dstate)
    dt_out, dA = dt_out_all[:, :n], dA_all[:, :n]
    z2 = z[:, 0]
    assert z2.stride(-1) == 1
    tree_args = (None,) * 5
    if tree:
        tree_args = (look_off, look_srow, scr[0].view(-1, nheads, headdim), scr[3].view(-1, ngroups, dstate),
                     scr[2].view(-1, ngroups, dstate))
    scan.scan_rows(xs, Cs, dt_out, dA, cb, z2, D, ssm_state, out, batch_indices, target_rows, chunk_starts,
                   chunk_flags, buffers.num_buffered, Bs, n, nheads // ngroups, small, *tree_args)
    if not small:
        # rows that filled their chunk (reads the state after the scan read it)
        sr.chunk_state_pass_rows(xs, Bs, dt_out, dA, chunk_starts, chunk_flags, batch_indices, ssm_state,
                                 sr.pinned_state_config())
    return out.unsqueeze(1)


def _scan(buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state):
    """``batch_invariant_decode_buffered_scan``: this step on z-free buffers, MCore's path on buffers with z."""
    if buffers.z is None:
        return decode_step(buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state)
    return _ORIG["scan"](buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state)


def install() -> None:
    """Make this the batch-invariant decode step (idempotent; wraps whatever function MCore's hook holds)."""
    if _ORIG:
        return
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    _ORIG["scan"] = bid.batch_invariant_decode_buffered_scan
    bid.batch_invariant_decode_buffered_scan = _scan
