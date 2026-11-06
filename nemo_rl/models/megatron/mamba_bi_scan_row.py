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
"""Crossing-row chunk boundary of the batch-invariant Mamba decode step (``mamba_bi_fused_decode``).

For a decode row whose token fills its 128-token chunk: MCore's ``_chunk_state_fwd_kernel`` (chunk flags) on
its pinned tile, then ``_state_passing_fwd_kernel``'s statement for the destination state, in one kernel that
does tile work only for crossing rows. MCore main ships the same kernel as
``batch_invariant_decode_rows.chunk_state_passing_rows`` (the decode step switches to it after the merge).

Per rank and row-local; no communication.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

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
