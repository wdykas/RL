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
"""Mamba-2 training forward on the Megatron engine's SSD kernels (batch-invariant mode).

The learner's Mamba mixer trains through mamba_ssm's fused split-conv1d-scan, while the
Megatron dynamic engine prefills and decodes with Megatron's own copy of the SSD kernels
(``megatron.core.ssm.ops.mamba2``). The two copies select their Triton tile configs
independently, and the dA cumsum head tiling and (for fp32 inputs) the chunk-scan warp
count change the bits, so train == engine held only when the two happened to agree.

``install()`` makes batch-invariant training run the engine's kernels:

- ``MambaMixer._ssm_training`` is replaced, in batch-invariant mode only, by the conv
  followed by ``mamba_chunk_scan_combined_train``: Megatron's varlen SSD forward with the
  gate inside the scan (as prefill and the buffered decode replay apply it), FP32 chunk
  states (as the engine keeps them), and mamba_ssm's backward.
- the engine's SSD forward kernels and varlen conv are pinned to constant configs built
  from a table, whatever candidate list each autotuner kept at import, so the training
  and the rollout processes launch the same tiles.

This is the Megatron-LM change ``fix/mamba-ssd-train-inference-parity`` (draft PR, see
research/thundersync_rl/DESIGN.md), applied to the Megatron-Core this repo pins. The one
difference: that Megatron-Core's chunk-scan kernel cannot store the pre-gate output, so
the backward recomputes it with one extra SSD forward.
"""

from __future__ import annotations

from typing import Optional

import torch

_INSTALLED: list[bool] = []
_ORIGINAL_SSM_TRAINING: list = []

# Tile kwargs and launch options per kernel, as in the upstream change: the minimax choice
# of a per-kernel sweep over Nano / Super / Ultra-scale per-rank shapes, bf16 and fp32,
# 16k and 64k tokens (GB200).
PINNED_CONFIGS = {
    "_bmm_chunk_fwd_kernel": (
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
        {"num_warps": 4, "num_stages": 4},
    ),
    "_chunk_scan_fwd_kernel": (
        {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        {"num_warps": 2, "num_stages": 5},
    ),
    "_chunk_state_fwd_kernel": (
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
        {"num_warps": 2, "num_stages": 4},
    ),
    "_chunk_cumsum_fwd_kernel": (
        {"BLOCK_SIZE_H": 16},
        {"num_warps": 4, "num_stages": 3},
    ),
    "_state_passing_fwd_kernel": (
        {"BLOCK_SIZE": 1024},
        {"num_warps": 4, "num_stages": 3},
    ),
    "_causal_conv1d_varlen_kernel": (
        {"BLOCK_T": 8, "BLOCK_C": 256},
        {"num_warps": 4, "num_stages": 2},
    ),
}


def _pinned_autotuners():
    from megatron.core.ssm.ops.common import causal_conv1d_varlen
    from megatron.core.ssm.ops.mamba2 import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )

    return (
        ssd_bmm._bmm_chunk_fwd_kernel,
        ssd_chunk_scan._chunk_scan_fwd_kernel,
        ssd_chunk_state._chunk_cumsum_fwd_kernel,
        ssd_chunk_state._chunk_state_fwd_kernel,
        ssd_state_passing._state_passing_fwd_kernel,
        causal_conv1d_varlen._causal_conv1d_varlen_kernel,
    )


def pin_engine_kernels() -> None:
    """Give each engine SSD forward kernel and the varlen conv one constant config."""
    import triton

    for kernel in _pinned_autotuners():
        kwargs, launch = PINNED_CONFIGS[kernel.fn.__name__]
        kernel.configs = [triton.Config(kwargs, **launch)]
        kernel.cache.clear()


def _varlen_layout(batch: int, seqlen: int, chunk_size: int, device: torch.device):
    """Chunk metadata for ``batch`` rows of ``seqlen`` tokens, each chunked from its start."""
    chunks_per_seq = -(-seqlen // chunk_size)
    chunk_starts = torch.arange(0, seqlen, chunk_size, dtype=torch.int32, device=device)
    seq_starts = torch.arange(
        0, batch * seqlen, seqlen, dtype=torch.int32, device=device
    )
    cu_chunk_seqlens = torch.cat(
        [
            (seq_starts[:, None] + chunk_starts[None, :]).flatten(),
            torch.full((1,), batch * seqlen, dtype=torch.int32, device=device),
        ]
    )
    seq_idx = torch.arange(batch, dtype=torch.int32, device=device).repeat_interleave(
        chunks_per_seq
    )
    last_chunk_indices = (
        torch.arange(1, batch + 1, dtype=torch.int32, device=device) * chunks_per_seq
        - 1
    )
    return cu_chunk_seqlens, seq_idx, last_chunk_indices


def _engine_scan(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    z,
    dt_bias,
    dt_softplus,
    dt_limit,
    initial_states=None,
    return_chunk_states=False,
):
    """Megatron's varlen SSD forward over (batch, seqlen, ...) rows, FP32 chunk states.

    ``initial_states`` [batch, nheads, headdim, dstate] (fp32): each row's SSM
    state before its first token (the state at a chunk boundary of a longer
    row: the scan then continues that row exactly, ``prefix_state_cache``).
    ``return_chunk_states``: also return the fp32 state at the end of every
    chunk, [batch, chunks per row, nheads, headdim, dstate].
    """
    from megatron.core.ssm.ops.mamba2.ssd_combined import (
        mamba_chunk_scan_combined_varlen,
    )

    batch, seqlen = x.shape[:2]
    cu_chunk_seqlens, seq_idx, last_chunk_indices = _varlen_layout(
        batch, seqlen, chunk_size, x.device
    )

    def flat(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        return None if t is None else t.reshape(batch * seqlen, *t.shape[2:])

    out = torch.empty_like(flat(x))
    states = mamba_chunk_scan_combined_varlen(
        x=flat(x),
        dt=flat(dt),
        A=A,
        B=flat(B),
        C=flat(C),
        chunk_size=chunk_size,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
        D=D,
        z=flat(z),
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=initial_states,
        return_raw_states=return_chunk_states,
        state_dtype=torch.float32,
    )
    if not return_chunk_states:
        return out.view_as(x)
    return out.view_as(x), states[1].view(batch, -1, *states[1].shape[1:])


class _EngineChunkScanFn(torch.autograd.Function):
    """Engine SSD forward with mamba_ssm's chunk-scan backward."""

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D, z, dt_bias, dt_softplus, dt_limit):
        """Gated output of the engine's kernels; saves the inputs for the backward."""
        ctx.save_for_backward(x, dt, A, B, C, D, z, dt_bias)
        ctx.chunk_size = chunk_size
        ctx.dt_softplus = dt_softplus
        ctx.dt_limit = dt_limit
        return _engine_scan(
            x, dt, A, B, C, chunk_size, D, z, dt_bias, dt_softplus, dt_limit
        )

    @staticmethod
    def backward(ctx, dout):
        """mamba_ssm's backward on the pre-gate output, recomputed with the engine kernels."""
        from mamba_ssm.ops.triton.ssd_combined import _mamba_chunk_scan_combined_bwd

        x, dt, A, B, C, D, z, dt_bias = ctx.saved_tensors
        args = (x, dt, A, B, C, ctx.chunk_size, D)
        out_x = _engine_scan(*args, None, dt_bias, ctx.dt_softplus, ctx.dt_limit)
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, _ = _mamba_chunk_scan_combined_bwd(
            dout,
            x,
            dt,
            A,
            B,
            C,
            out_x,
            ctx.chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=ctx.dt_softplus,
            dt_limit=ctx.dt_limit,
        )
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, None, None


def mamba_chunk_scan_combined_train(
    x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, dt_softplus=False
):
    """Differentiable chunk scan, bitwise equal to the engine's prefill of each row."""
    return _EngineChunkScanFn.apply(
        x, dt, A, B, C, chunk_size, D, z, dt_bias, dt_softplus, (0.0, float("inf"))
    )


def _split_proj(self, zxBCdt):
    """(z, xBC, dt) of the in-projection output, (l, b, d) -> (b, l, d) each."""
    from einops import rearrange

    zxBCdt = rearrange(zxBCdt, "l b d -> b l d").contiguous()
    d_inner = self.cp.d_inner_local_tpcp
    groups_state = self.cp.ngroups_local_tpcp * self.d_state
    return torch.split(
        zxBCdt, [d_inner, d_inner + 2 * groups_state, self.cp.nheads_local_tpcp], dim=-1
    )


def _conv(self, xBC_bdl):
    """The non-fused training conv (``MambaMixer._static_prefill``) over (b, d, l) inputs.

    Bitwise equal to the engine's varlen prefill conv and its decode update.
    Returns (b, l, d).
    """
    from causal_conv1d import causal_conv1d_fn
    from einops import rearrange

    out = causal_conv1d_fn(
        x=xBC_bdl,
        weight=rearrange(self.cp.get_conv1d_weight(), "d 1 w -> d w"),
        bias=self.cp.get_conv1d_bias(),
        activation=self.activation,
    )
    return rearrange(out, "b d l -> b l d").contiguous()


def _scan_and_gate(self, xBC, z, dt, scan):
    """SSD scan of the conv output, gate inside the scan, gated RMSNorm: (l, b, d).

    ``scan(*args)``: ``mamba_chunk_scan_combined_train`` (training) or an
    ``_engine_scan`` closure (prefix-state forwards); returns (y, extra).
    """
    from einops import rearrange

    d_inner = self.cp.d_inner_local_tpcp
    groups_state = self.cp.ngroups_local_tpcp * self.d_state
    x, B, C = torch.split(xBC, [d_inner, groups_state, groups_state], dim=-1)
    D = self.cp.get_D()
    if self.D_has_hdim:
        D = rearrange(D.float(), "(h p) -> h p", p=self.headdim)
    y, extra = scan(
        rearrange(x, "b l (h p) -> b l h p", p=self.headdim).contiguous(),
        dt.contiguous(),
        -torch.exp(self.cp.get_A_log().float()),
        rearrange(B, "b l (g n) -> b l g n", n=self.d_state).contiguous(),
        rearrange(C, "b l (g n) -> b l g n", n=self.d_state).contiguous(),
        self.chunk_size,
        D,
        rearrange(z, "b l (h p) -> b l h p", p=self.headdim).contiguous(),
        self.cp.get_dt_bias().float(),
    )
    y = rearrange(y, "b l h p -> l b (h p)").contiguous()
    y = self.cp.post_conv_ssm(y)
    # The gate went through the scan, as the engine's prefill and decode apply it.
    return self.norm(y), extra


def _check_bi_mixer(self, packed_seq_params) -> None:
    assert packed_seq_params is None, (
        "batch-invariant Mamba training does not pack sequences"
    )
    assert self.rmsnorm, "batch-invariant Mamba requires the gated RMSNorm"


def _ssm_training(self, zxBCdt, packed_seq_params=None):
    """``MambaMixer._ssm_training`` on the engine's kernels in batch-invariant mode."""
    from einops import rearrange

    if not self.config.batch_invariant_mode:
        return _ORIGINAL_SSM_TRAINING[0](self, zxBCdt, packed_seq_params)
    _check_bi_mixer(self, packed_seq_params)
    z, xBC, dt = _split_proj(self, zxBCdt)
    xBC = _conv(self, rearrange(xBC, "b l d -> b d l").contiguous())

    def scan(x, dt, A, B, C, chunk_size, D, z, dt_bias):
        y = mamba_chunk_scan_combined_train(
            x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, dt_softplus=True
        )
        return y, None

    return _scan_and_gate(self, xBC, z, dt, scan)[0]


def ssm_prefix_forward(self, zxBCdt, conv_init, ssm_init):
    """``_ssm_training`` of row segments that continue longer rows (no autograd).

    Row b of ``zxBCdt`` (l, b, d) holds the in-projection of positions
    [s_b, s_b + l) of a row, s_b a multiple of ``chunk_size`` (padding past the
    row's end is harmless: everything is causal). ``conv_init`` (b, conv_dim,
    d_conv - 1): the conv inputs (xBC) of positions [s_b - d_conv + 1, s_b)
    (zeros for s_b = 0); ``ssm_init`` (b, nheads, headdim, dstate) fp32: the
    SSM state at s_b (zeros for s_b = 0).

    Bitwise the rows' positions [s_b, s_b + l) of ``_ssm_training`` of the whole
    rows: the conv of every output position sees the same d_conv inputs in
    the same order, and the chunked scan restarted at a chunk boundary from
    that boundary's fp32 state runs the same per-chunk kernels and the same
    state recurrence as the full scan (the engine's batch-invariant prefill
    with prefix caching relies on the same identity).

    Returns (y (l, b, d), xBC_ext (b, conv_dim, d_conv - 1 + l) conv inputs,
    chunk_states (b, chunks, nheads, headdim, dstate) fp32 states at the end
    of each chunk).
    """
    from einops import rearrange

    _check_bi_mixer(self, None)
    assert not torch.is_grad_enabled(), "prefix-state forwards run without autograd"
    z, xBC, dt = _split_proj(self, zxBCdt)
    xBC_ext = torch.cat(
        [conv_init.to(xBC.dtype), rearrange(xBC, "b l d -> b d l")], dim=-1
    ).contiguous()
    width = conv_init.shape[-1]
    xBC_out = _conv(self, xBC_ext)[:, width:].contiguous()

    def scan(x, dt, A, B, C, chunk_size, D, z, dt_bias):
        return _engine_scan(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D,
            z,
            dt_bias,
            True,
            (0.0, float("inf")),
            initial_states=ssm_init,
            return_chunk_states=True,
        )

    y, chunk_states = _scan_and_gate(self, xBC_out, z, dt, scan)
    return y, xBC_ext, chunk_states


def install() -> None:
    """Train batch-invariant Mamba on the engine's pinned SSD kernels (idempotent)."""
    pin_engine_kernels()
    if _INSTALLED:
        return
    from megatron.core.ssm.mamba_mixer import MambaMixer

    _ORIGINAL_SSM_TRAINING.append(MambaMixer._ssm_training)
    MambaMixer._ssm_training = _ssm_training
    _INSTALLED.append(True)
