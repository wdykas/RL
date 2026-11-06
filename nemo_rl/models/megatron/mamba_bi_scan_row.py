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
"""Target-row chunk scan for the batch-invariant Mamba decode replay.

MCore's buffered decode (``batch_invariant_decode._mamba_chunk_scan_decode_rows``) runs the
training chunk-scan kernel (``ssd_chunk_scan._chunk_scan_fwd_kernel``, pinned tile
BLOCK_M 32 x BLOCK_N 64 x BLOCK_K 32, 2 warps) on the whole 32-row M tile that holds the
one row a decode step consumes, for every (request, head).

Per output element the training kernel's arithmetic is

    acc[m, n] = dot(C[m, :], S[:, n])                     (one tl.dot, K = dstate)
    acc[m, n] *= exp(dA_cs[m])
    for k0 in 0, 32, .., K_MAX - 32:                       (K_MAX = end of m's 32-row tile)
        acc[m, n] += dot(cbw[m, k0:k0+32], x[k0:k0+32, n])  (cbw = cb * exp(..) * dt, causal 0)
    acc[m, n] += x[m, n] * D

where each ``dot`` is a chain of tensor-core MMA instructions (fp32 inputs: TF32, the
same instruction for every row) accumulating in fp32 in a fixed k order. A tensor-core
MMA computes every row of its M dimension from that row of A alone, so the bits of row m
depend only on: row m's operands, the B operands, the instruction shape, the k
decomposition and the accumulation chain — not on how many other rows share the tile.

This kernel issues the same instruction chain on a 16-row M tile (the MMA's own M; the
row's 16-row half of the training tile, 1 warp) with only the target row's C and cb rows
loaded, the same BLOCK_N / BLOCK_K / dstate decomposition and the same K_MAX (the
training tile's causal bound, so the same zero-weight products are summed). The equality
is checked bitwise against MCore's tile kernel (tests/unit/models/megatron/
test_mamba_bi_scan_row.py) and the compiled instructions are compared (same
``mma.sync.aligned.m16n8k8 ... tf32`` in both). If the engine's pinned chunk-scan config
is not the one this kernel mirrors, the caller keeps MCore's kernel.

Per rank and row-local: one program per (decode row, head); no communication.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

# The training chunk-scan tile this kernel mirrors (mamba_ssd_parity.PINNED_CONFIGS).
REF_BLOCK_M, REF_BLOCK_N, REF_BLOCK_K, REF_NUM_WARPS = 32, 64, 32, 2
ROW_BLOCK_M = 16  # one MMA M tile
# Launch options of the row kernel (no effect on the per-element instruction chain; swept in
# research/mamba_replay/sweep_row.py).
LAUNCH = {"num_warps": 4, "num_stages": 3}
# hdim columns per program (output columns are independent: each column's k chain is the
# training tile's whatever the N split).
ROW_BLOCK_N = {"n": 64}

if HAVE_TRITON:

    @triton.jit
    def _chunk_scan_row_kernel(
        cb_ptr,
        x_ptr,
        out_ptr,
        dt_ptr,
        dA_cumsum_ptr,
        C_ptr,
        D_ptr,
        initstates_ptr,
        chunk_starts_ptr,
        slots_ptr,
        target_rows_ptr,
        chunk_size: tl.constexpr,
        hdim: tl.constexpr,
        dstate: tl.constexpr,
        nheads_ngroups_ratio: tl.constexpr,
        stride_cb_chunk: tl.int64,
        stride_cb_head: tl.int64,
        stride_cb_csize_m: tl.int64,
        stride_cb_csize_k: tl.constexpr,
        stride_x_seqlen: tl.int64,
        stride_x_head: tl.int64,
        stride_x_hdim: tl.constexpr,
        stride_out_seqlen: tl.int64,
        stride_out_head: tl.int64,
        stride_out_hdim: tl.constexpr,
        stride_dt_chunk: tl.int64,
        stride_dt_head: tl.int64,
        stride_dt_csize: tl.constexpr,
        stride_dA_cs_chunk: tl.int64,
        stride_dA_cs_head: tl.int64,
        stride_dA_cs_csize: tl.constexpr,
        stride_C_seqlen: tl.int64,
        stride_C_head: tl.int64,
        stride_C_dstate: tl.constexpr,
        stride_init_states_batch: tl.int64,
        stride_init_states_head: tl.int64,
        stride_init_states_hdim: tl.int64,
        stride_init_states_dstate: tl.constexpr,
        stride_D_head: tl.constexpr,
        HAS_D: tl.constexpr,
        D_HAS_HDIM: tl.constexpr,
        REF_M: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_DSTATE: tl.constexpr,
    ):
        """``_chunk_scan_fwd_kernel`` (HAS_TARGET_ROWS, causal, init states, no z) for one row.

        Statement for statement the training kernel's, with BLOCK_SIZE_M = 16 rows placed on
        the target row's 16-row half of its REF_M-row training tile, K_MAX from the training
        tile, and only the target row of C / cb loaded (other rows are 0; they never reach
        the stored row).
        """
        pid_c = tl.program_id(axis=0).to(tl.int64)
        pid_h = tl.program_id(axis=1)
        pid_n = tl.program_id(axis=2)
        tr = tl.load(target_rows_ptr + pid_c)
        if tr < 0:
            return
        cb_ptr += pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
        chunk_seqlen_start = tl.load(chunk_starts_ptr + pid_c)
        x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
        dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
        dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
        C_ptr += chunk_seqlen_start * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head

        ref_m = tr // REF_M
        offs_m = (tr // BLOCK_SIZE_M) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        row = offs_m == tr

        seq_idx = tl.load(slots_ptr + pid_c)
        prev_states_ptr = (
            initstates_ptr + seq_idx * stride_init_states_batch + pid_h * stride_init_states_head
        )
        chunk_size_limit = chunk_size

        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        dA_cs_m = tl.load(
            dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0
        ).to(tl.float32)

        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        scale_m = tl.exp(dA_cs_m)
        C = tl.load(
            C_ptrs,
            mask=row[:, None] & (offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate),
            other=0.0,
        )
        prev_states_ptrs = (
            prev_states_ptr
            + offs_n[None, :] * stride_init_states_hdim
            + offs_k_dstate[:, None] * stride_init_states_dstate
        )
        prev_states = tl.load(
            prev_states_ptrs,
            mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim),
            other=0.0,
        )
        prev_states = prev_states.to(C_ptr.dtype.element_ty)
        acc = tl.dot(C, prev_states) * scale_m[:, None]

        offs_k = tl.arange(0, BLOCK_SIZE_K)
        cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
        x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        dt_ptrs = dt_ptr + offs_k * stride_dt_csize
        dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
        K_MAX = min((ref_m + 1) * REF_M, chunk_size_limit)
        for k in range(0, K_MAX, BLOCK_SIZE_K):
            cb = tl.load(
                cb_ptrs,
                mask=row[:, None] & (offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k),
                other=0.0,
            ).to(tl.float32)
            dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
            cb *= tl.exp(tl.minimum(dA_cs_m[:, None] - dA_cs_k[None, :], 0.0))
            dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
            cb *= dt_k
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
            cb = cb.to(x_ptr.dtype.element_ty)
            x = tl.load(
                x_ptrs,
                mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim),
                other=0.0,
            )
            acc += tl.dot(cb, x)
            cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
            x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
            dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
            dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

        offs_out_m = offs_m
        offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(
                    tl.float32
                )
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
            x_residual = tl.load(
                x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                mask=row[:, None] & (offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim),
                other=0.0,
            ).to(tl.float32)
            acc += x_residual * D

        out_ptr += pid_c * stride_out_seqlen + pid_h * stride_out_head
        out_ptrs = out_ptr + (offs_out_m[:, None] * 0 + offs_out_n[None, :] * stride_out_hdim)
        tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] == tr) & (offs_out_n[None, :] < hdim))


def pinned_config_matches() -> bool:
    """True when MCore's chunk-scan autotuner holds exactly the tile this kernel mirrors."""
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_scan

    configs = getattr(ssd_chunk_scan._chunk_scan_fwd_kernel, "configs", [])
    if len(configs) != 1:
        return False
    cfg = configs[0]
    return (
        cfg.kwargs.get("BLOCK_SIZE_M") == REF_BLOCK_M
        and cfg.kwargs.get("BLOCK_SIZE_N") == REF_BLOCK_N
        and cfg.kwargs.get("BLOCK_SIZE_K") == REF_BLOCK_K
        and cfg.num_warps == REF_NUM_WARPS
    )


def supported(x: torch.Tensor, C: torch.Tensor, chunk_size: int) -> bool:
    """Shapes this kernel mirrors: one BLOCK_N tile over hdim, dstate in one dot, fp32."""
    _, _, headdim = x.shape
    dstate = C.shape[-1]
    bds = max(triton.next_power_of_2(dstate), 16)
    return (
        x.dtype == torch.float32
        and C.dtype == torch.float32
        and headdim <= REF_BLOCK_N
        and bds <= 128
        and chunk_size % REF_BLOCK_M == 0
    )


def chunk_scan_rows(cb, x, dt, dA_cumsum, C, out, slots, target_rows, chunk_starts, initial_states, D=None):
    """Drop-in for ``_chunk_scan_fwd(..., target_rows=, chunk_starts=)`` without z."""
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = C.shape
    assert cb.shape == (nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == dA_cumsum.shape == (nheads, nchunks, chunk_size)
    if nchunks == 0:
        return
    block_n = min(ROW_BLOCK_N["n"], REF_BLOCK_N)
    _chunk_scan_row_kernel[(nchunks, nheads, triton.cdiv(headdim, block_n))](
        cb,
        x,
        out,
        dt,
        dA_cumsum,
        C,
        D,
        initial_states,
        chunk_starts,
        slots,
        target_rows,
        chunk_size=chunk_size,
        hdim=headdim,
        dstate=dstate,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_cb_chunk=cb.stride(0),
        stride_cb_head=cb.stride(1),
        stride_cb_csize_m=cb.stride(2),
        stride_cb_csize_k=cb.stride(3),
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_out_seqlen=out.stride(0),
        stride_out_head=out.stride(1),
        stride_out_hdim=out.stride(2),
        stride_dt_chunk=dt.stride(1),
        stride_dt_head=dt.stride(0),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_C_seqlen=C.stride(0),
        stride_C_head=C.stride(1),
        stride_C_dstate=C.stride(2),
        stride_init_states_batch=initial_states.stride(0),
        stride_init_states_head=initial_states.stride(1),
        stride_init_states_hdim=initial_states.stride(2),
        stride_init_states_dstate=initial_states.stride(3),
        stride_D_head=D.stride(0) if D is not None else 0,
        HAS_D=D is not None,
        D_HAS_HDIM=D.dim() == 2 if D is not None else True,
        REF_M=REF_BLOCK_M,
        BLOCK_SIZE_M=ROW_BLOCK_M,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=REF_BLOCK_K,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        **LAUNCH,
    )


# ---------------------------------------------------------------------------
# Chunk boundary: chunk state + state passing for crossing rows, one kernel.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _chunk_state_pass_rows_kernel(
        x_ptr,
        b_ptr,
        dt_ptr,
        dA_cumsum_ptr,
        chunk_starts_ptr,
        chunk_flags_ptr,
        slots_ptr,
        states_ptr,
        nrows,
        hdim: tl.constexpr,
        dstate: tl.constexpr,
        chunk_size: tl.constexpr,
        nheads_ngroups_ratio: tl.constexpr,
        stride_x_seqlen: tl.int64,
        stride_x_head: tl.int64,
        stride_x_hdim: tl.constexpr,
        stride_b_seqlen: tl.int64,
        stride_b_head: tl.int64,
        stride_b_dstate: tl.constexpr,
        stride_dt_head: tl.int64,
        stride_dt_chunk: tl.int64,
        stride_dt_csize: tl.constexpr,
        stride_dA_cs_head: tl.int64,
        stride_dA_cs_chunk: tl.int64,
        stride_dA_cs_csize: tl.constexpr,
        stride_st_batch: tl.int64,
        stride_st_head: tl.int64,
        stride_st_hdim: tl.int64,
        stride_st_dstate: tl.constexpr,
        ROWS_PER_PROGRAM: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """``_chunk_state_fwd_kernel`` (chunk flags) then ``_state_passing_fwd_kernel`` (dst states).

        For each crossing row c (chunk_flags[c] != 0) and (head, hdim x dstate tile): the chunk
        state tile with the chunk-state kernel's statements and tile (its pinned config), then
        the state-passing statement ``exp(dA_cs[last]) * S[slot] + new`` on the same fp32
        values (one fma, as MCore's kernel compiles it), stored to S[slot]. MCore's pair launches one program per (tile, row, head),
        nearly all exiting at the flag (about 1 crossing row in 128 per step), and a serial
        loop over all rows per state-passing program; here a program owns ROWS_PER_PROGRAM
        rows of one (head, tile) and does the tile work only for crossing rows.
        """
        pid_t = tl.program_id(axis=0)
        pid_h = tl.program_id(axis=1)
        pid_r = tl.program_id(axis=2)
        num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
        pid_m = pid_t // num_pid_n
        pid_n = pid_t % num_pid_n
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        for i in range(ROWS_PER_PROGRAM):
            c = (pid_r * ROWS_PER_PROGRAM + i).to(tl.int64)
            flag = tl.load(chunk_flags_ptr + c, mask=c < nrows, other=0)
            if flag != 0:
                chunk_seqlen_start = tl.load(chunk_starts_ptr + c)
                b_ptr_c = b_ptr + chunk_seqlen_start * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
                x_ptr_c = x_ptr + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
                dt_ptr_c = dt_ptr + c * stride_dt_chunk + pid_h * stride_dt_head
                dA_ptr_c = dA_cumsum_ptr + c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
                x_ptrs = x_ptr_c + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
                b_ptrs = b_ptr_c + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
                dt_ptrs = dt_ptr_c + offs_k * stride_dt_csize
                dA_cs_last = tl.load(dA_ptr_c + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
                dA_cumsum_ptrs = dA_ptr_c + offs_k * stride_dA_cs_csize
                chunk_size_limit = chunk_size
                acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
                for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
                    x = tl.load(
                        x_ptrs,
                        mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
                        other=0.0,
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
                        other=0.0,
                    ).to(tl.float32)
                    dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
                        tl.float32
                    )
                    dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
                    scale = tl.exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
                    b *= scale[:, None]
                    b = b.to(x_ptr.dtype.element_ty)
                    acc += tl.dot(x, b)
                    x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
                    b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
                    dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
                    dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
                new_states = acc.to(tl.float32)
                # State passing (dst path): the slot's state is the chunk's initial state.
                slot = tl.load(slots_ptr + c).to(tl.int64)
                st_ptrs = (
                    states_ptr
                    + slot * stride_st_batch
                    + pid_h * stride_st_head
                    + offs_m[:, None] * stride_st_hdim
                    + offs_n[None, :] * stride_st_dstate
                )
                st_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
                states = tl.load(st_ptrs, mask=st_mask, other=0.0).to(tl.float32)
                dA_cs = tl.load(dA_ptr_c + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
                # MCore's state-passing kernel compiles ``exp(dA) * states + new`` to fma.rn.f32 (LLVM
                # contraction); here the add would stay a separate mul / add (packed f32x2 on sm_100),
                # so the fused form is explicit.
                states = tl.fma(tl.exp(dA_cs), states, new_states)
                tl.store(st_ptrs, states, mask=st_mask)


# Rows per boundary program: None = by row count (1 at <= 31 rows, 4 at 64, 16 at >= 256:
# measured device us, research/mamba_replay/sweep_state.py); an int pins it (sweeps).
STATE_ROWS_PER_PROGRAM = {"n": None}


def _rows_per_program(nrows: int) -> int:
    if STATE_ROWS_PER_PROGRAM["n"] is not None:
        return STATE_ROWS_PER_PROGRAM["n"]
    return min(16, max(1, triton.next_power_of_2(nrows // 16 + 1) // 2 if nrows >= 16 else 1))


def pinned_state_config():
    """(kwargs, launch) of MCore's pinned chunk-state tile, or None if not pinned to one config."""
    from megatron.core.ssm.ops.mamba2 import ssd_chunk_state

    configs = getattr(ssd_chunk_state._chunk_state_fwd_kernel, "configs", [])
    if len(configs) != 1:
        return None
    cfg = configs[0]
    kw = {k: cfg.kwargs[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")}
    return kw, {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}


def chunk_state_pass_rows(x, B, dt, dA_cumsum, chunk_starts, chunk_flags, slots, states, config):
    """Crossing rows: boundary state into ``states[slot]`` (MCore's chunk state + state passing)."""
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert states.shape[1:] == (nheads, headdim, dstate) and states.dtype == torch.float32
    if nchunks == 0:
        return
    kw, launch = config
    rpp = _rows_per_program(nchunks)
    grid = (
        triton.cdiv(headdim, kw["BLOCK_SIZE_M"]) * triton.cdiv(dstate, kw["BLOCK_SIZE_N"]),
        nheads,
        triton.cdiv(nchunks, rpp),
    )
    _chunk_state_pass_rows_kernel[grid](
        x,
        B,
        dt,
        dA_cumsum,
        chunk_starts,
        chunk_flags,
        slots,
        states,
        nchunks,
        hdim=headdim,
        dstate=dstate,
        chunk_size=chunk_size,
        nheads_ngroups_ratio=nheads // ngroups,
        stride_x_seqlen=x.stride(0),
        stride_x_head=x.stride(1),
        stride_x_hdim=x.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_b_dstate=B.stride(2),
        stride_dt_head=dt.stride(0),
        stride_dt_chunk=dt.stride(1),
        stride_dt_csize=dt.stride(2),
        stride_dA_cs_head=dA_cumsum.stride(0),
        stride_dA_cs_chunk=dA_cumsum.stride(1),
        stride_dA_cs_csize=dA_cumsum.stride(2),
        stride_st_batch=states.stride(0),
        stride_st_head=states.stride(1),
        stride_st_hdim=states.stride(2),
        stride_st_dstate=states.stride(3),
        ROWS_PER_PROGRAM=rpp,
        **kw,
        **launch,
    )


# ---------------------------------------------------------------------------
# CB = C B^T for the target row only.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _bmm_rows_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        chunk_starts_ptr,
        target_rows_ptr,
        chunk_size: tl.constexpr,
        K: tl.constexpr,
        ngroups: tl.constexpr,
        stride_a_seqlen: tl.int64,
        stride_a_head: tl.int64,
        stride_ak: tl.constexpr,
        stride_b_seqlen: tl.int64,
        stride_b_head: tl.int64,
        stride_bk: tl.constexpr,
        stride_out_chunk: tl.int64,
        stride_out_head: tl.int64,
        stride_outm: tl.int64,
        stride_outn: tl.constexpr,
        dot_dtype: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        """``_bmm_chunk_fwd_kernel`` (HAS_TARGET_ROWS, not causal) loading and storing only row tr.

        Same tile (the pinned BLOCK_M x BLOCK_N, K loop) and instruction chain as MCore's
        target-row launch; the other rows of A are 0 instead of the chunk's C rows, so only
        row tr of the output is meaningful, and only row tr is stored (the row scan reads
        no other row).
        """
        pid_ch = tl.program_id(axis=1).to(tl.int64)
        pid_c = pid_ch // ngroups
        pid_h = pid_ch - pid_c * ngroups
        num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
        pid_m = tl.program_id(axis=0) // num_pid_n
        pid_n = tl.program_id(axis=0) % num_pid_n
        tr = tl.load(target_rows_ptr + pid_c)
        if tr < 0:
            return
        if pid_m != tr // BLOCK_SIZE_M:
            return
        if pid_n * BLOCK_SIZE_N > tr:
            return
        chunk_seqlen_start = tl.load(chunk_starts_ptr + pid_c)
        chunk_seqlen_end = chunk_seqlen_start + chunk_size
        a_ptr += chunk_seqlen_start * stride_a_seqlen + pid_h * stride_a_head
        b_ptr += chunk_seqlen_start * stride_b_seqlen + pid_h * stride_b_head
        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_b_seqlen)
        chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start
        row = offs_m == tr
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(
                a_ptrs,
                mask=row[:, None] & (offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                other=0.0,
            ).to(dot_dtype)
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) & (offs_n[None, :] < chunk_size_limit),
                other=0.0,
            ).to(dot_dtype)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk
        out = acc.to(out_ptr.dtype.element_ty)
        out_ptr += pid_c * stride_out_chunk + pid_h * stride_out_head
        out_ptrs = out_ptr + (stride_outm * offs_m[:, None] + offs_n[None, :] * stride_outn)
        tl.store(out_ptrs, out, mask=row[:, None] & (offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


def pinned_bmm_config():
    """(kwargs, launch) of MCore's pinned CB tile, or None if not pinned to one config."""
    from megatron.core.ssm.ops.mamba2 import ssd_bmm

    configs = getattr(ssd_bmm._bmm_chunk_fwd_kernel, "configs", [])
    if len(configs) != 1:
        return None
    cfg = configs[0]
    kw = {k: cfg.kwargs[k] for k in ("BLOCK_SIZE_M", "BLOCK_SIZE_N", "BLOCK_SIZE_K")}
    return kw, {"num_warps": cfg.num_warps, "num_stages": cfg.num_stages}


def bmm_rows(C, B, chunk_size, target_rows, chunk_starts, config):
    """``_bmm_chunk_fwd(C, B, ..., output_dtype=fp32, target_rows=, chunk_starts=)``: row tr only."""
    seqlen, ngroups, k = C.shape
    assert B.shape == C.shape and C.stride(-1) == 1 and B.stride(-1) == 1
    nchunks = len(chunk_starts)
    out = torch.empty((nchunks, ngroups, chunk_size, chunk_size), device=C.device, dtype=torch.float32)
    if nchunks == 0:
        return out
    kw, launch = config
    dot_dtype = tl.float32 if C.dtype == torch.float32 and B.dtype == torch.float32 else None
    assert dot_dtype is not None, "row CB kernel: fp32 C / B"
    grid = (
        triton.cdiv(chunk_size, kw["BLOCK_SIZE_M"]) * triton.cdiv(chunk_size, kw["BLOCK_SIZE_N"]),
        nchunks * ngroups,
    )
    _bmm_rows_kernel[grid](
        C,
        B,
        out,
        chunk_starts,
        target_rows,
        chunk_size=chunk_size,
        K=k,
        ngroups=ngroups,
        stride_a_seqlen=C.stride(0),
        stride_a_head=C.stride(1),
        stride_ak=C.stride(2),
        stride_b_seqlen=B.stride(0),
        stride_b_head=B.stride(1),
        stride_bk=B.stride(2),
        stride_out_chunk=out.stride(0),
        stride_out_head=out.stride(1),
        stride_outm=out.stride(-2),
        stride_outn=out.stride(-1),
        dot_dtype=dot_dtype,
        **kw,
        **launch,
    )
    return out


# Row CB + crossing-row boundary kernels (off: MCore's CB, chunk-state and state-passing kernels; tests and A/B).
ROW_AUX = {"on": True}


def decode_rows(x, dt, A, B, C, chunk_size, chunk_starts, slots, target_rows, chunk_flags, initial_states, out,
                D=None, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    """``batch_invariant_decode._mamba_chunk_scan_decode_rows`` (z = None) on row kernels.

    cumsum is MCore's kernel, called as MCore calls it; CB is the target-row CB kernel; the
    chunk scan is the target-row kernel; chunk state + state passing are one crossing-row
    kernel. CB / boundary fall back to MCore's kernels when their autotuner is not pinned to
    one config (or ROW_AUX is off).
    """
    from megatron.core.ssm.ops.mamba2.ssd_bmm import _bmm_chunk_fwd
    from megatron.core.ssm.ops.mamba2.ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
    from megatron.core.ssm.ops.mamba2.ssd_state_passing import _state_passing_fwd

    dA_cumsum, dt = _chunk_cumsum_fwd(
        dt, A, chunk_size, None, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit,
        chunk_starts=chunk_starts, target_rows=target_rows,
    )
    state_cfg = pinned_state_config() if ROW_AUX["on"] else None
    if state_cfg is None:
        states = _chunk_state_fwd(
            B, x, dt, dA_cumsum, None, states_in_fp32=True, chunk_flags=chunk_flags, chunk_starts=chunk_starts,
        )
    bmm_cfg = pinned_bmm_config() if ROW_AUX["on"] else None
    if bmm_cfg is not None:
        CB = bmm_rows(C, B, chunk_size, target_rows, chunk_starts, bmm_cfg)
    else:
        CB = _bmm_chunk_fwd(
            C, B, chunk_size, None, output_dtype=torch.float32, target_rows=target_rows, chunk_starts=chunk_starts,
        )
    # The scan reads the slot's state before the boundary update overwrites crossing slots
    # (as MCore orders them).
    chunk_scan_rows(CB, x, dt, dA_cumsum, C, out, slots, target_rows, chunk_starts, initial_states, D=D)
    if state_cfg is not None:
        chunk_state_pass_rows(x, B, dt, dA_cumsum, chunk_starts, chunk_flags, slots, initial_states, state_cfg)
        return
    _state_passing_fwd(
        states.flatten(-2), dA_cumsum, None, initial_states=initial_states.flatten(-2), seq_idx=slots,
        dst_states=initial_states.flatten(-2), dst_indices=slots, dst_flags=chunk_flags,
    )
