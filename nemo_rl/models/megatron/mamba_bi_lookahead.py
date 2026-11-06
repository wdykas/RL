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
"""Lookahead decode rows for the batch-invariant Mamba layers (multi-token tail decode, ``tail_mtd``).

A tail row of the engine may carry, in one decode step, extra decode entries ("lookahead rows") for the next
positions of the same request (proposals verified by sample-and-match, ``tail_mtd``). Entry e of such a tail
has offset i >= 1 (``mamba_bi_fused_decode.lookahead_buffers``): it is the token at position P - 1 + i, and its
predecessors' entries are chain[e, 0 .. i-1] (chain[e, 0] = the tail's row 0, which runs the ordinary decode).

Each lookahead row must produce, bit for bit, what one-token decode of its position produces once its
predecessors are committed:

- **SSM** (``mamba_bi_fused_decode``): the row targets chunk row w0 + i of the slot's replay buffer; its prep
  program reads the predecessors' dt / B from their step inputs. The scan's math is the one-token decode's. The
  slot's write cursor is set after acceptance (``commit``), so the scan's cursor writes for these rows are
  overwritten.
- **Conv** (here): the row runs the same ``causal_conv1d_update`` CUDA kernel as one-token decode, on a scratch
  conv state that holds what sequential decode would hold before it: the last ``state_len`` columns of
  [slot state before the step, inputs of rows 0 .. i-1]. The scratch states are extra rows of the context's
  conv-state tensor (one per lookahead entry capacity, ``scratch_rows``), so one kernel call serves every row.
  After acceptance of a >= 2 tokens, the slot's state <- the scratch state of row a - 1 (``commit``).
- Rows never complete a Mamba chunk (the layout keeps i <= chunk - 2 - w0), so no SSM chunk state is ever
  committed for a row that may be rejected, and rejected rows need no rewind: their replay rows sit above the
  cursor and are overwritten by later appends.

Per rank: scratch conv rows = capacity x conv_dim / TP x state_len x 4 B per Mamba layer; one extra Triton launch
per Mamba layer per decode step while installed (the window gather, which also writes the step's conv indices);
no communication; fixed grids and storage (CUDA-graph capturable).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover - CPU-only environments
    HAVE_TRITON = False

_ORIG: dict[str, Any] = {}
_STATE: dict[str, Any] = {"capacity": 0, "nbuf": [], "nbuf_ptrs": None}


def capacity() -> int:
    """Lookahead entries (scratch conv rows) per engine step; 0 when not installed."""
    return int(_STATE["capacity"])


def register_num_buffered(t: torch.Tensor) -> None:
    """Every Mamba layer's replay-buffer write cursors (``num_buffered``), for the post-acceptance cursor update."""
    _STATE["nbuf"].append(t)
    _STATE["nbuf_ptrs"] = None


def num_buffered_ptrs(device) -> torch.Tensor:
    """int64 device table of every registered layer's ``num_buffered`` data pointer (built once, after allocation)."""
    p = _STATE["nbuf_ptrs"]
    if p is None or p.numel() != len(_STATE["nbuf"]):
        p = _STATE["nbuf_ptrs"] = torch.tensor([t.data_ptr() for t in _STATE["nbuf"]], dtype=torch.int64,
                                               device=device)
    return p


if HAVE_TRITON:

    @triton.jit
    def _conv_window_kernel(
        state_ptr,
        x_ptr,
        bi_ptr,
        off_ptr,
        chain_ptr,
        sid_ptr,
        idx_out_ptr,
        stride_sb,
        stride_sd,
        stride_sw,
        stride_x,
        stride_chain,
        conv_dim,
        STATE_LEN: tl.constexpr,
        LOOK_MAX: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Program (e, d-block): conv index of decode entry e (its slot, or its scratch row if it is a lookahead
        row) and, for a lookahead row of offset i, its scratch state = last STATE_LEN columns of
        [slot state, x(chain[e, 0]), ..., x(chain[e, i-1])] (pure copies)."""
        e = tl.program_id(axis=0).to(tl.int64)
        db = tl.program_id(axis=1)
        off = tl.load(off_ptr + e)
        slot = tl.load(bi_ptr + e)
        sid = tl.load(sid_ptr + e)
        if db == 0:
            tl.store(idx_out_ptr + e, tl.where(off > 0, sid, slot))
        if off <= 0:
            return
        offs_d = db * BLOCK_D + tl.arange(0, BLOCK_D)
        md = offs_d < conv_dim
        for c in tl.static_range(STATE_LEN):
            # column c of the window: index q = off + c into [state (STATE_LEN cols), x_0 .. x_{off-1}]
            q = off + c
            from_state = q < STATE_LEN
            col = tl.where(from_state, q, 0)
            v_state = tl.load(state_ptr + slot * stride_sb + offs_d * stride_sd + col * stride_sw,
                              mask=md & from_state, other=0.0)
            j = tl.where(from_state, 0, q - STATE_LEN)
            pj = tl.load(chain_ptr + e * stride_chain + j).to(tl.int64)
            v_x = tl.load(x_ptr + pj * stride_x + offs_d, mask=md & (~from_state), other=0.0)
            tl.store(state_ptr + sid.to(tl.int64) * stride_sb + offs_d * stride_sd + c * stride_sw,
                     tl.where(from_state, v_state, v_x), mask=md)

    @triton.jit
    def _commit_kernel(
        conv_ptr,
        nbuf_ptrs,
        slot_ptr,
        sid_ptr,
        acc_ptr,
        cursor_ptr,
        n_tails,
        n_layers,
        stride_sid,
        stride_cl,
        stride_cb,
        conv_row_elems,
        CHUNK: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Program (tail t, layer l, block): after acceptance of a tokens (a >= 1) of a tail that ran lookahead rows:
        conv state of its slot <- scratch row of its row a - 1 (a >= 2); write cursor <- (w0 + a) mod CHUNK."""
        t = tl.program_id(axis=0)
        lyr = tl.program_id(axis=1).to(tl.int64)
        blk = tl.program_id(axis=2)
        slot = tl.load(slot_ptr + t)
        a = tl.load(acc_ptr + t)
        if (slot < 0) | (a <= 0):
            return
        if blk == 0:
            nb = tl.load(nbuf_ptrs + lyr).to(tl.pointer_type(tl.int32))
            tl.store(nb + slot, (tl.load(cursor_ptr + t) + a) % CHUNK)
        if a < 2:
            return
        sid = tl.load(sid_ptr + t * stride_sid + a - 2).to(tl.int64)  # scratch row of row a - 1 (rows 1.. -> 0..)
        offs = blk * BLOCK + tl.arange(0, BLOCK)
        m = offs < conv_row_elems
        v = tl.load(conv_ptr + lyr * stride_cl + sid * stride_cb + offs, mask=m)
        tl.store(conv_ptr + lyr * stride_cl + slot.to(tl.int64) * stride_cb + offs, v, mask=m)


def full_conv_state(conv_state: torch.Tensor) -> torch.Tensor:
    """The layer's conv-state rows including the scratch rows after the ``max_requests`` slots (see ``install``)."""
    cap = capacity()
    if cap == 0:
        return conv_state
    return torch.as_strided(conv_state, (conv_state.shape[0] + cap,) + tuple(conv_state.shape[1:]),
                            conv_state.stride(), conv_state.storage_offset())


def _step_buffers(n_rows: int, device) -> dict:
    """Per-device step buffers: conv indices per entry and the scratch-row id of each lookahead entry."""
    key = ("bufs", device.index if device.index is not None else torch.cuda.current_device())
    b = _STATE.get(key)
    if b is None or b["conv_idx"].numel() < n_rows:
        b = _STATE[key] = {
            "conv_idx": torch.full((n_rows,), -1, dtype=torch.int32, device=device),
            "scratch_id": torch.full((n_rows,), -1, dtype=torch.int32, device=device),
        }
    return b


def scratch_ids(n_rows: int, device) -> torch.Tensor:
    """[n_rows] int32: scratch conv row of each decode entry that is a lookahead row (written by the layout)."""
    return _step_buffers(n_rows, device)["scratch_id"]


def conv_step(xBC: torch.Tensor, conv_state: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              activation: Optional[str], batch_indices: torch.Tensor, conv_fn) -> torch.Tensor:
    """The batch-invariant decode conv for every decode entry, lookahead rows on their scratch states.

    xBC [b, d] in the conv-state dtype; one-token decode for entries with offset 0 (their own slot, in place).
    """
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    b, d = xBC.shape
    full = full_conv_state(conv_state)
    off, chain = fd.lookahead_buffers(max(b, conv_state.shape[0]), xBC.device)
    bufs = _step_buffers(max(b, conv_state.shape[0]), xBC.device)
    idx = bufs["conv_idx"][:b]
    state_len = full.shape[-1]
    assert xBC.stride(-1) == 1 and full.stride(-2) >= 1
    BLOCK_D = 256
    _conv_window_kernel[(b, triton.cdiv(d, BLOCK_D))](
        full, xBC, batch_indices, off, chain, bufs["scratch_id"], idx,
        full.stride(0), full.stride(1), full.stride(2), xBC.stride(0), chain.stride(0), d,
        STATE_LEN=state_len, LOOK_MAX=fd.LOOK_MAX, BLOCK_D=BLOCK_D,
    )
    return conv_fn(xBC, full, weight, bias, activation, conv_state_indices=idx)


def commit(conv_states_all: torch.Tensor, slots: torch.Tensor, sids: torch.Tensor, accepted: torch.Tensor,
           cursors: torch.Tensor, chunk: int) -> None:
    """After acceptance: per tail t with accepted[t] = a >= 1 tokens: every Mamba layer's write cursor of its slot
    <- (cursors[t] + a) mod chunk, and (a >= 2) its conv state <- the scratch state of its row a - 1.

    conv_states_all: the context's full conv-state tensor [layers, max_requests + scratch rows, ...]
    (``mamba_conv_states_full``); slots / accepted / cursors [T] int32 (slot -1: no tail); sids [T, >= K] int32
    (scratch row of the tail's rows 1, 2, ...).
    """
    T = slots.numel()
    if T == 0 or not _STATE["nbuf"]:
        return
    L = conv_states_all.shape[0]
    assert len(_STATE["nbuf"]) == L, (len(_STATE["nbuf"]), L)
    row = conv_states_all[0, 0].numel()
    BLOCK = 1024
    _commit_kernel[(T, L, triton.cdiv(row, BLOCK))](
        conv_states_all, num_buffered_ptrs(conv_states_all.device), slots, sids, accepted, cursors, T, L,
        sids.stride(0), conv_states_all.stride(0), conv_states_all.stride(1), row, CHUNK=chunk, BLOCK=BLOCK,
    )


# ---------------------------------------------------------------------------
# Installation: scratch conv rows after the context's slots; the decode conv through ``conv_step``.
# ---------------------------------------------------------------------------


def _allocate_mamba_states(self):
    _ORIG["allocate_states"](self)
    cap = capacity()
    if not getattr(self, "is_hybrid_model", False) or cap == 0 or not getattr(self, "batch_invariant_mode", False):
        return
    old = self.mamba_conv_states
    full = torch.empty((old.shape[0], old.shape[1] + cap) + tuple(old.shape[2:]), dtype=old.dtype, device=old.device)
    full[:, : old.shape[1]].copy_(old)
    self.mamba_conv_states = full[:, : old.shape[1]]  # MCore's view: max_requests slots per layer
    self.mamba_conv_states_full = full
    del old


def _ssm_decode(self, zxBCdt, conv_state, ssm_state, batch_indices=None, intermediate_conv_state=None,
                intermediate_ssm_state=None):
    if not self.config.batch_invariant_mode or capacity() == 0:
        return _ORIG["ssm_decode"](self, zxBCdt, conv_state, ssm_state, batch_indices=batch_indices,
                                   intermediate_conv_state=intermediate_conv_state,
                                   intermediate_ssm_state=intermediate_ssm_state)
    from einops import rearrange
    from megatron.core.ssm import mamba_mixer as mm

    assert mm.causal_conv1d_update_cuda is not None, "Batch-invariant Mamba decode requires causal-conv1d"
    batch_size, seq_len, _ = zxBCdt.shape
    assert seq_len == 1 and intermediate_conv_state is None
    assert batch_indices is not None and batch_indices.dtype == torch.int32
    z, xBC, dt = torch.split(
        zxBCdt,
        [self.d_inner_local_tp, self.d_inner_local_tp + 2 * self.ngroups_local_tp * self.d_state,
         self.nheads_local_tp],
        dim=-1,
    )
    xBC_dtype = xBC.dtype
    x2 = xBC.to(conv_state.dtype).squeeze(1)
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    xBC = conv_step(
        x2, conv_state, rearrange(self.conv1d_weight, "d 1 w -> d w").to(conv_state.dtype),
        self.conv1d_bias.to(conv_state.dtype), self.activation, batch_indices, mm.causal_conv1d_update_cuda,
    ).unsqueeze(1).to(xBC_dtype)
    x, B, C = torch.split(
        xBC,
        [self.d_inner_local_tp, self.ngroups_local_tp * self.d_state, self.ngroups_local_tp * self.d_state],
        dim=-1,
    )
    y = self._get_batch_invariant_decoder().step(x, z, dt, B, C, batch_indices, ssm_state)
    if self.rmsnorm:  # as MCore's ssm_decode (batch-invariant: the gate is applied inside the step)
        y = self.norm(y, None)
    return y


def install(lookahead_capacity: int) -> None:
    """Enable lookahead rows (idempotent; ``lookahead_capacity`` > 0 entries per step). Call before the engine's
    context is built (the scratch conv rows are allocated with the Mamba states). Tree speculation owns its node-row
    state (tree_kv) and is exclusive with this."""
    import sys

    cap = int(lookahead_capacity)
    assert cap > 0
    if _ORIG:
        assert cap == _STATE["capacity"], ("lookahead capacity is fixed once installed", cap, _STATE["capacity"])
        return
    tk = sys.modules.get("nemo_rl.models.generation.megatron.tree_kv")
    if tk is not None and tk.capacity() > 0:
        raise RuntimeError("mamba_bi_lookahead: tree node rows (tree_kv) are installed; tail_mtd and tail_tree are "
                           "exclusive")
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.ssm.mamba_mixer import MambaMixer

    _STATE["capacity"] = cap
    _ORIG["allocate_states"] = DynamicInferenceContext._allocate_mamba_states
    _ORIG["ssm_decode"] = MambaMixer.ssm_decode
    DynamicInferenceContext._allocate_mamba_states = _allocate_mamba_states
    MambaMixer.ssm_decode = _ssm_decode
