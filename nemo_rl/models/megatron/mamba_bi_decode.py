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
"""Batch-invariant Mamba decode: no z replay buffer, replay buffers in the memory budget.

Megatron-Core's batch-invariant Mamba decode (``megatron.core.ssm.ops.mamba2.
batch_invariant_decode``) keeps, per request slot and Mamba layer, every token of the
current 128-token chunk (x, z, dt, B, C) and replays the training chunk scan for the one
row it needs. Two things this module changes in the engine:

1. **No z buffer.** The scan applies the gate ``z * sigmoid(z)`` row-locally after the
   accumulation, and decode stores only the target row, so the other buffered rows' z
   never reach an output. With fp32 replay buffers (the fp16 GEMM forward: the Mamba
   in_proj emits fp32) the scan runs without z and one Triton kernel gates the target row
   with the current token's z using the scan kernel's expression on the scan's (exact,
   fp32) pre-gate value — bitwise the same output (tested). With 16-bit buffers the
   pre-gate row would be rounded before the gate, so those keep MCore's path. Saves
   chunk x d_inner x 4 bytes per slot per Mamba layer (Super-120B per rank at TP=1:
   160 MiB of 362 MiB per slot). The same change with the gate inside the chunk-scan
   kernel is the Megatron-LM draft PR ``fix/mamba-bi-decode-drop-z-buffer``.
2. **Accounting.** MCore sizes the Mamba share of the inference buffer from conv + SSM
   state bytes only and allocates the replay buffers lazily outside it, for every
   ``max_requests`` slot. ``install()`` makes ``DynamicInferenceContext`` count them: the
   Mamba region must hold state + replay bytes per request, and the KV region keeps its
   size (``replay_adjusted_budget``).

Per rank, nothing here communicates; under PP each stage counts its own Mamba layers.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

_ORIG: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Replay-buffer bytes and the adjusted inference budget (pure; CPU-testable).
# ---------------------------------------------------------------------------


def replay_bytes_per_request(
    conv_states_shape: tuple[int, ...],
    ssm_states_shape: tuple[int, ...],
    chunk_size: int,
    elem_bytes: int,
    num_mamba_layers: int,
    z_buffered: bool,
) -> int:
    """Bytes of batch-invariant decode buffers one request slot holds over its Mamba layers.

    Per layer: x, B, C rows (conv_dim = d_inner + 2 * ngroups * dstate elements, i.e.
    the conv channel width), dt (nheads) and, if buffered, z (d_inner) for every row of
    a chunk; the target-row output (d_inner); three int32 cursors.
    """
    nheads, headdim = ssm_states_shape[0], ssm_states_shape[1]
    d_inner = nheads * headdim
    conv_dim = max(conv_states_shape)
    row = conv_dim + nheads + (d_inner if z_buffered else 0)
    per_layer = chunk_size * row * elem_bytes + d_inner * elem_bytes + 3 * 4
    return per_layer * num_mamba_layers


def state_bytes_per_request(state_cfg: Any, num_mamba_layers: int) -> int:
    """MCore's ``mamba_states_memory_per_request`` (no speculative tokens)."""
    conv = math.prod(state_cfg.conv_states_shape) * state_cfg.conv_states_dtype.itemsize
    ssm = math.prod(state_cfg.ssm_states_shape) * state_cfg.ssm_states_dtype.itemsize
    return (conv + ssm) * num_mamba_layers


def replay_adjusted_budget(
    buffer_bytes: float,
    paused_bytes: float,
    mamba_memory_ratio: Optional[float],
    max_requests: Optional[int],
    state_bytes: int,
    replay_bytes: int,
) -> tuple[float, float, Optional[float]]:
    """Buffer sizes and Mamba ratio for which MCore's sizing also fits the replay buffers.

    MCore's two hybrid branches:
    - ``mamba_memory_ratio`` r of T = buffer + paused bytes is the Mamba region,
      ``floor(r T / p)`` requests (p = state bytes per request), the rest is KV. With
      replay bytes q the region must give ``r T / (p + q)`` requests and KV keeps
      (1 - r) T: solve for T' and r' with r' T' / p = r T / (p + q),
      (1 - r') T' = (1 - r) T. The replay buffers then take r T q / (p + q).
    - ``max_requests`` N (no ratio): the Mamba region is N p and KV the rest; reserve
      N q more for the replay buffers.
    Returns (buffer_bytes', paused_bytes', ratio') — buffers scaled by one factor.
    """
    total = buffer_bytes + paused_bytes
    if total <= 0 or replay_bytes == 0:
        return buffer_bytes, paused_bytes, mamba_memory_ratio
    if mamba_memory_ratio is not None:
        r, p, q = mamba_memory_ratio, state_bytes, replay_bytes
        mamba_states = r * total * p / (p + q)
        new_total = (1.0 - r) * total + mamba_states
        scale, ratio = new_total / total, mamba_states / new_total
    elif max_requests is not None:
        reserve = max_requests * replay_bytes
        if reserve >= total:
            raise ValueError(
                f"batch-invariant Mamba replay buffers for max_requests={max_requests} need "
                f"{reserve / 2**30:.2f} GiB, more than the {total / 2**30:.2f} GiB inference "
                "buffer."
            )
        scale, ratio = (total - reserve) / total, None
    else:
        raise NotImplementedError(
            "batch-invariant Mamba replay-buffer accounting is built for "
            "mamba_memory_ratio or max_requests (NeMo-RL always sets the ratio)."
        )
    return buffer_bytes * scale, paused_bytes * scale, ratio


# ---------------------------------------------------------------------------
# Decode without the z buffer.
# ---------------------------------------------------------------------------

if HAVE_TRITON:

    @triton.jit
    def _gate_target_rows_kernel(
        out_ptr,
        z_ptr,
        rows_ptr,
        stride_out_b,
        stride_out_h,
        stride_z_b,
        stride_z_h,
        hdim,
        BLOCK_N: tl.constexpr,
    ):
        """out[b, h] *= z[b, h] * sigmoid(z[b, h]) for active entries (target row >= 0).

        The chunk-scan kernel's gate (``acc *= z * tl.sigmoid(z)`` on the fp32 acc),
        applied to the fp32 value it would have gated.
        """
        b = tl.program_id(0)
        h = tl.program_id(1)
        if tl.load(rows_ptr + b) < 0:
            return
        offs = tl.arange(0, BLOCK_N)
        mask = offs < hdim
        o_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + offs
        acc = tl.load(o_ptrs, mask=mask, other=0.0)
        z = tl.load(
            z_ptr + b * stride_z_b + h * stride_z_h + offs, mask=mask, other=0.0
        ).to(tl.float32)
        acc *= z * tl.sigmoid(z)
        tl.store(o_ptrs, acc, mask=mask)


    @triton.jit
    def _decode_prep_kernel(
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
        out_ptr,
        rows_ptr,
        flags_ptr,
        starts_ptr,
        stride_x,
        stride_dt,
        stride_b,
        stride_c,
        n_x,
        n_dt,
        n_bc,
        n_out,
        CHUNK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One decode entry's bookkeeping in one program (all integer moves and copies).

        Equal to MCore's per-step sequence for entry e with slot s = batch_indices[e]:
        write pos w = num_buffered[s]; rows x/dt/B/C of the current token into buffer row
        s * CHUNK + w; target_rows[e] = w (-1 inactive); chunk_flags[e] = (w + 1 == CHUNK);
        chunk_starts[e] = s * CHUNK; out[e] = 0; num_buffered[s] = 0 if crossed else w + 1.
        The scan pipeline reads none of num_buffered, so its update moves here.
        """
        e = tl.program_id(0)
        s = tl.load(bi_ptr + e)
        tl.store(starts_ptr + e, s * CHUNK)
        offs = tl.arange(0, BLOCK)
        zero = tl.zeros([BLOCK], dtype=tl.float32)
        for o in range(0, n_out, BLOCK):
            tl.store(out_ptr + e * n_out + o + offs, zero, mask=o + offs < n_out)
        if s < 0:
            tl.store(rows_ptr + e, -1)
            tl.store(flags_ptr + e, 0)
            return
        w = tl.load(nbuf_ptr + s)
        row = (s * CHUNK + w).to(tl.int64)
        for o in range(0, n_x, BLOCK):
            m = o + offs < n_x
            v = tl.load(x_ptr + e * stride_x + o + offs, mask=m)
            tl.store(sx_ptr + row * n_x + o + offs, v, mask=m)
        for o in range(0, n_dt, BLOCK):
            m = o + offs < n_dt
            v = tl.load(dt_ptr + e * stride_dt + o + offs, mask=m)
            tl.store(sdt_ptr + row * n_dt + o + offs, v, mask=m)
        for o in range(0, n_bc, BLOCK):
            m = o + offs < n_bc
            v = tl.load(b_ptr + e * stride_b + o + offs, mask=m)
            tl.store(sb_ptr + row * n_bc + o + offs, v, mask=m)
            v = tl.load(c_ptr + e * stride_c + o + offs, mask=m)
            tl.store(sc_ptr + row * n_bc + o + offs, v, mask=m)
        crossed = w + 1 == CHUNK
        tl.store(rows_ptr + e, w)
        tl.store(flags_ptr + e, crossed.to(tl.int32))
        tl.store(nbuf_ptr + s, tl.where(crossed, 0, w + 1))


def _gate_target_rows(
    out: torch.Tensor, z: torch.Tensor, target_rows: torch.Tensor
) -> None:
    """Gate out (B, H, P) fp32 in place with z (B, H, P); entries with target row -1 skip."""
    assert out.dtype == torch.float32 and out.stride(2) == 1 and z.stride(2) == 1
    nb, nh, hdim = out.shape
    _gate_target_rows_kernel[(nb, nh)](
        out,
        z,
        target_rows,
        out.stride(0),
        out.stride(1),
        z.stride(0),
        z.stride(1),
        hdim,
        BLOCK_N=triton.next_power_of_2(hdim),
    )


def _z_free(dtype: torch.dtype) -> bool:
    return dtype == torch.float32


def _allocate(
    cls, max_requests, chunk_size, nheads, headdim, ngroups, dstate, device, dtype
):
    """``BatchInvariantDecodeBuffers.allocate`` without z for fp32 buffers."""
    if not _z_free(dtype):
        return _ORIG["allocate"].__func__(
            cls,
            max_requests,
            chunk_size,
            nheads,
            headdim,
            ngroups,
            dstate,
            device,
            dtype,
        )

    def zeros(*shape):
        return torch.zeros(max_requests, chunk_size, *shape, device=device, dtype=dtype)

    buf = cls(
        x=zeros(nheads, headdim),
        z=None,
        dt=zeros(nheads),
        B=zeros(ngroups, dstate),
        C=zeros(ngroups, dstate),
        num_buffered=torch.zeros(max_requests, device=device, dtype=torch.int32),
        out=torch.empty(max_requests, nheads, headdim, device=device, dtype=dtype),
        target_rows=torch.empty(max_requests, device=device, dtype=torch.int32),
        chunk_flags=torch.empty(max_requests, device=device, dtype=torch.int32),
    )
    # Per-entry chunk starts (slot * chunk), written by the fused decode prep kernel.
    buf.chunk_starts = torch.empty(max_requests, device=device, dtype=torch.int32)
    return buf


def _seed(self, x, z, dt, B, C, cu_seqlens, batch_indices):
    """``BatchInvariantDecodeBuffers.seed`` (store each prefill's unfinished chunk).

    MCore's for buffers with z; on z-free buffers the same rows without z.
    """
    if self.z is not None:
        return _ORIG["seed"](self, x, z, dt, B, C, cu_seqlens, batch_indices)
    chunk_size = self.x.shape[1]
    num_seqs = cu_seqlens.numel() - 1
    seq_starts = cu_seqlens[:-1].to(torch.long)
    seq_ends = cu_seqlens[1:].to(torch.long)
    tail_lens = (seq_ends - seq_starts) % chunk_size
    # As MCore: unused rows repeat a valid token of the same sequence (finite padding).
    offsets = torch.arange(chunk_size, device=x.device, dtype=torch.long)
    safe_tail_lens = torch.clamp(tail_lens, min=1)
    safe_tail_offsets = torch.minimum(
        offsets.unsqueeze(0), (safe_tail_lens - 1).unsqueeze(1)
    )
    safe_tail_starts = torch.where(
        tail_lens > 0, seq_ends - tail_lens, torch.clamp(seq_ends - 1, min=0)
    )
    idx = (safe_tail_starts.unsqueeze(1) + safe_tail_offsets).clamp(max=x.shape[0] - 1)
    slots = batch_indices[:num_seqs]
    update = _ORIG["masked_update_rows"]
    update(self.x.flatten(1), slots, x[idx].flatten(1))
    update(self.dt.flatten(1), slots, dt[idx].flatten(1))
    update(self.B.flatten(1), slots, B[idx].flatten(1))
    update(self.C.flatten(1), slots, C[idx].flatten(1))
    update(
        self.num_buffered.unsqueeze(1), slots, tail_lens.to(torch.int32).unsqueeze(1)
    )


def _decode_buffered_scan(
    buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
):
    """``batch_invariant_decode_buffered_scan`` on z-free buffers (MCore's otherwise).

    Same steps as MCore's: write the current token's x / dt / B / C rows, run the chunk
    scan pipeline for the target rows (without z), then gate each target row with the
    current token's z.
    """
    if buffers.z is not None:
        return _ORIG["scan"](
            buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
    if _FUSED_PREP["on"] and x.is_cuda:
        return _decode_buffered_scan_fused(
            buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
        )
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    decode_batch_size, tokens_per_entry, nheads, headdim = x.shape
    dstate = B.shape[-1]
    chunk_size = buffers.x.shape[1]
    assert tokens_per_entry == 1, "batch-invariant Mamba decode: one token per request"
    assert ssm_state.dtype == torch.float32, (
        "batch-invariant Mamba decode: fp32 SSM state"
    )
    assert decode_batch_size <= buffers.out.shape[0], "increase max_requests"

    out = buffers.out[:decode_batch_size]
    target_rows = buffers.target_rows[:decode_batch_size]
    chunk_flags = buffers.chunk_flags[:decode_batch_size]

    active = batch_indices >= 0
    safe_slots = batch_indices.clamp_min(0)
    write_pos = buffers.num_buffered[safe_slots].to(torch.long)
    buffer_rows = torch.where(active, safe_slots * chunk_size + write_pos, -1).to(
        torch.int32
    )

    update = _ORIG["masked_update_rows"]
    update(buffers.x.view(-1, nheads * headdim), buffer_rows, x[:, 0].flatten(1))
    update(buffers.dt.view(-1, nheads), buffer_rows, dt[:, 0])
    update(
        buffers.B.view(-1, buffers.B.shape[-2] * dstate),
        buffer_rows,
        B[:, 0].flatten(1),
    )
    update(
        buffers.C.view(-1, buffers.C.shape[-2] * dstate),
        buffer_rows,
        C[:, 0].flatten(1),
    )

    crossed = active & (write_pos + 1 == chunk_size)
    target_rows.copy_(torch.where(active, write_pos, -1).to(torch.int32))
    chunk_flags.copy_(crossed.to(torch.int32))
    out.zero_()

    bid._mamba_chunk_scan_decode_rows(
        buffers.x.view(-1, nheads, headdim),
        None,
        buffers.dt.view(-1, nheads),
        A,
        buffers.B.view(-1, buffers.B.shape[-2], dstate),
        buffers.C.view(-1, buffers.C.shape[-2], dstate),
        chunk_size,
        chunk_starts=batch_indices * chunk_size,
        slots=batch_indices,
        target_rows=target_rows,
        chunk_flags=chunk_flags,
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )
    _gate_target_rows(out, z[:, 0], target_rows)

    next_write_pos = torch.where(crossed, 0, write_pos + 1).to(torch.int32)
    update(
        buffers.num_buffered.unsqueeze(1), batch_indices, next_write_pos.unsqueeze(1)
    )
    return out.unsqueeze(1)


# Decode-step bookkeeping in one kernel (``_decode_prep_kernel``); THUNDERSYNC_BI_DECODE_FUSED=0
# restores the op-by-op sequence (same bits, ~25 more kernels per Mamba layer per step).
_FUSED_PREP = {"on": os.environ.get("THUNDERSYNC_BI_DECODE_FUSED", "1") != "0"}


def _decode_buffered_scan_fused(
    buffers, x, z, dt, B, C, A, D, dt_bias, batch_indices, ssm_state
):
    """``_decode_buffered_scan`` with its ~26 bookkeeping kernels fused into one.

    The scan pipeline and the gate are unchanged; every value the fused kernel writes is
    an integer or a copied element, so the result is bitwise the op-by-op path's.
    """
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    n, tokens_per_entry, nheads, headdim = x.shape
    ngroups, dstate = B.shape[-2], B.shape[-1]
    chunk_size = buffers.x.shape[1]
    assert tokens_per_entry == 1, "batch-invariant Mamba decode: one token per request"
    assert ssm_state.dtype == torch.float32, (
        "batch-invariant Mamba decode: fp32 SSM state"
    )
    assert n <= buffers.out.shape[0], "increase max_requests"
    assert batch_indices.dtype == torch.int32 and buffers.num_buffered.dtype == torch.int32
    starts_all = getattr(buffers, "chunk_starts", None)
    if starts_all is None:
        starts_all = buffers.chunk_starts = torch.empty_like(buffers.target_rows)
    out = buffers.out[:n]
    target_rows = buffers.target_rows[:n]
    chunk_flags = buffers.chunk_flags[:n]
    chunk_starts = starts_all[:n]
    x2, dt2 = x[:, 0].flatten(1), dt[:, 0]
    b2, c2 = B[:, 0].flatten(1), C[:, 0].flatten(1)
    for t, dtype in ((x2, buffers.x.dtype), (dt2, buffers.dt.dtype), (b2, buffers.B.dtype),
                     (c2, buffers.C.dtype)):
        assert t.stride(-1) == 1 and t.dtype == dtype
    assert out.is_contiguous() and buffers.x.is_contiguous() and buffers.B.is_contiguous()
    if n:
        _decode_prep_kernel[(n,)](
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
            out,
            target_rows,
            chunk_flags,
            chunk_starts,
            x2.stride(0),
            dt2.stride(0),
            b2.stride(0),
            c2.stride(0),
            nheads * headdim,
            nheads,
            ngroups * dstate,
            out[0].numel(),
            CHUNK=chunk_size,
            BLOCK=1024,
        )
    bid._mamba_chunk_scan_decode_rows(
        buffers.x.view(-1, nheads, headdim),
        None,
        buffers.dt.view(-1, nheads),
        A,
        buffers.B.view(-1, ngroups, dstate),
        buffers.C.view(-1, ngroups, dstate),
        chunk_size,
        chunk_starts=chunk_starts,
        slots=batch_indices,
        target_rows=target_rows,
        chunk_flags=chunk_flags,
        initial_states=ssm_state,
        out=out,
        D=D,
        dt_bias=dt_bias,
        dt_softplus=True,
    )
    _gate_target_rows(out, z[:, 0], target_rows)
    return out.unsqueeze(1)


# ---------------------------------------------------------------------------
# DynamicInferenceContext accounting.
# ---------------------------------------------------------------------------


def _replay_dtype(model_config: Any) -> torch.dtype:
    """Dtype of the replay buffers: the Mamba in_proj output dtype."""
    from nemo_rl.models.megatron import fp16_gemm

    if fp16_gemm.enabled():
        return torch.float32  # fp16 GEMM forward: fp32 GEMM outputs
    return model_config.params_dtype


def _context_init(self, model_config, inference_config, *args, **kwargs):
    state_cfg = inference_config.mamba_inference_state_config
    if state_cfg is None or not getattr(model_config, "batch_invariant_mode", False):
        return _ORIG["context_init"](
            self, model_config, inference_config, *args, **kwargs
        )
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    n_mamba = sum(1 for t in state_cfg.layer_type_list if t == Symbols.MAMBA)
    dtype = _replay_dtype(model_config)
    replay = replay_bytes_per_request(
        tuple(state_cfg.conv_states_shape),
        tuple(state_cfg.ssm_states_shape),
        state_cfg.mamba_chunk_size,
        dtype.itemsize,
        n_mamba,
        z_buffered=not _z_free(dtype),
    )
    gib = 1024**3
    paused = inference_config.paused_buffer_size_gb
    buf, paused_b, ratio = replay_adjusted_budget(
        inference_config.buffer_size_gb * gib,
        (paused or 0.0) * gib,
        inference_config.mamba_memory_ratio,
        inference_config.max_requests,
        state_bytes_per_request(state_cfg, n_mamba),
        replay,
    )
    adjusted = dataclasses.replace(
        inference_config,
        buffer_size_gb=buf / gib,
        paused_buffer_size_gb=None if paused is None else paused_b / gib,
        mamba_memory_ratio=ratio,
    )
    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(
            f"[mamba_bi_decode] replay buffers {replay / 2**20:.1f} MiB/request over "
            f"{n_mamba} Mamba layers counted: buffer {inference_config.buffer_size_gb:.2f} "
            f"-> {adjusted.buffer_size_gb:.2f} GiB, mamba_memory_ratio "
            f"{inference_config.mamba_memory_ratio} -> {ratio}",
            flush=True,
        )
    return _ORIG["context_init"](self, model_config, adjusted, *args, **kwargs)


def install() -> None:
    """Patch MCore's batch-invariant Mamba decode and context sizing (idempotent)."""
    if _ORIG:
        return
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    buffers_cls = bid.BatchInvariantDecodeBuffers
    _ORIG["allocate"] = buffers_cls.__dict__["allocate"]
    _ORIG["seed"] = buffers_cls.seed
    _ORIG["scan"] = bid.batch_invariant_decode_buffered_scan
    _ORIG["masked_update_rows"] = bid._masked_update_rows
    _ORIG["context_init"] = DynamicInferenceContext.__init__

    buffers_cls.allocate = classmethod(_allocate)
    buffers_cls.seed = _seed
    bid.batch_invariant_decode_buffered_scan = _decode_buffered_scan
    DynamicInferenceContext.__init__ = _context_init
