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
"""Batch-invariant Mamba decode buffers: no z replay buffer, replay buffers in the memory budget.

Megatron-Core's batch-invariant Mamba decode (``megatron.core.ssm.ops.mamba2.batch_invariant_decode``) keeps,
per request slot and Mamba layer, every token of the current 128-token chunk (x, z, dt, B, C) and replays the
training chunk scan for the one row it needs. This module changes, in the engine:

1. **No z buffer.** The scan applies the gate ``z * sigmoid(z)`` row-locally after the accumulation and decode
   stores only the target row, so the other buffered rows' z never reach an output. For a layer the
   batch-invariant decode step serves (``mamba_bi_fused_decode.supported_config``: fp32 buffers, i.e. the
   fp16-GEMM recipe's in_proj output, and the step's shape / arch / pinned-tile rules) the buffers are allocated
   without z and that step gates the target row with the current token's z. Other layers keep MCore's buffers
   and path. Saves chunk x d_inner x 4 bytes per slot per Mamba layer (Super-120B per rank at TP=1: 160 MiB of
   362 MiB per slot). Upstream: Megatron-LM #8221.
2. **Accounting.** MCore sizes the Mamba share of the inference buffer from conv + SSM state bytes only and
   allocates the replay buffers lazily outside it, for every ``max_requests`` slot. ``install()`` makes
   ``DynamicInferenceContext`` count them: the Mamba region must hold state + replay bytes per request, and the
   KV region keeps its size (``replay_adjusted_budget``).

``install()`` also installs the decode step (``mamba_bi_fused_decode.install``). Per rank, nothing here
communicates; under PP each stage counts its own Mamba layers. After the merge with Megatron-LM main this module
is deleted (cleanup C11): main allocates z-free buffers itself.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Optional

import torch

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
# z-free buffers.
# ---------------------------------------------------------------------------


def _z_free(nheads, headdim, ngroups, dstate, chunk, dtype, device) -> bool:
    """Buffers without z iff the batch-invariant decode step serves the layer (it gates the target row itself)."""
    from nemo_rl.models.megatron import mamba_bi_fused_decode

    return mamba_bi_fused_decode.supported_config(nheads, headdim, ngroups, dstate, chunk, dtype, device)


def _allocate(
    cls, max_requests, chunk_size, nheads, headdim, ngroups, dstate, device, dtype
):
    """``BatchInvariantDecodeBuffers.allocate`` without z for layers the batch-invariant decode step serves."""
    if not _z_free(nheads, headdim, ngroups, dstate, chunk_size, dtype, device):
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
    from nemo_rl.models.megatron import mamba_bi_lookahead

    mamba_bi_lookahead.register_num_buffered(buf.num_buffered)  # cursor update after lookahead rows (tail_mtd)
    cap = mamba_bi_lookahead.tree_capacity()
    if cap > 0:
        # Tree node rows (tree_spec): per-node scratch replay rows (x, dt, B, C), so sibling nodes at one depth never
        # share the slot's chunk row; the accepted path's rows are copied into the slot at commit.
        buf.scratch = (
            torch.zeros(cap, nheads, headdim, device=device, dtype=dtype),
            torch.zeros(cap, nheads, device=device, dtype=dtype),
            torch.zeros(cap, ngroups, dstate, device=device, dtype=dtype),
            torch.zeros(cap, ngroups, dstate, device=device, dtype=dtype),
        )
        mamba_bi_lookahead.register_replay(buf)
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
    nheads, headdim, dstate = tuple(state_cfg.ssm_states_shape)
    ngroups = (max(state_cfg.conv_states_shape) - nheads * headdim) // (2 * dstate)
    device = torch.device("cuda", torch.cuda.current_device())
    z_free = _z_free(nheads, headdim, ngroups, dstate, state_cfg.mamba_chunk_size, dtype, device)
    replay = replay_bytes_per_request(
        tuple(state_cfg.conv_states_shape),
        tuple(state_cfg.ssm_states_shape),
        state_cfg.mamba_chunk_size,
        dtype.itemsize,
        n_mamba,
        z_buffered=not z_free,
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
    """Patch MCore's batch-invariant Mamba buffers and context sizing, install the decode step (idempotent)."""
    if _ORIG:
        return
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.ssm.ops.mamba2 import batch_invariant_decode as bid

    from nemo_rl.models.megatron import mamba_bi_fused_decode

    buffers_cls = bid.BatchInvariantDecodeBuffers
    _ORIG["allocate"] = buffers_cls.__dict__["allocate"]
    _ORIG["seed"] = buffers_cls.seed
    _ORIG["scan"] = bid.batch_invariant_decode_buffered_scan  # MCore's (buffers with z)
    _ORIG["masked_update_rows"] = bid._masked_update_rows
    _ORIG["context_init"] = DynamicInferenceContext.__init__

    buffers_cls.allocate = classmethod(_allocate)
    buffers_cls.seed = _seed
    DynamicInferenceContext.__init__ = _context_init
    mamba_bi_fused_decode.install()
