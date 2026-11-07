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
"""Node-row state of tree speculation (tree_spec), owned here: the attention KV of node rows (private blocks) and
the Mamba node state (scratch conv rows, scratch replay rows, the commit). Nothing here depends on another
speculation module; ``install(node_cap)`` sets up both before the engine's context is built.

**KV.** One private block per node row, filled so that every node attends to exactly the keys and values one-token
decode would see at its prefix.

A tree node at depth d >= 1 of a tail with committed length P is a one-token decode row at position P - 1 + d. Its
attention must read, at logical positions block_start .. P - 1 + d of its last block:
- block_start .. P - 2: the request's committed K / V (the request's real block);
- P - 1: the root row's K / V, appended in the same forward (layer by layer);
- P .. P - 2 + d: its ancestors' K / V, also appended in the same forward;
- P - 1 + d: its own K / V.
Sibling nodes share positions, so a node cannot use the request's block. Instead:
1. ``partial_copy`` (once per forward, before it, outside CUDA graphs): the committed part of the request's last
   block -> the node's private block (every attention layer);
2. the node's own row appends its K / V into its private block (its ``token_to_block_idx``), as any row;
3. ``_layer_gather`` (in every attention layer, right after MCore's KV append, inside the CUDA graph): the root's and
   the ancestors' K / V of that layer -> the node's private block;
4. the node's block-table row is the request's, with the last block replaced by the private block.
The node's logits are then one-token decode's (same kernels, same KV contents). After acceptance, ``commit_copy``
copies the accepted path's K / V (depths 1 .. a - 1) from the private blocks into the request's block.

Depth is capped so that no node leaves the root's block (tree_spec: no node completes a Mamba chunk, and the chunk
divides the block). Private blocks come from the context's KV allocator (``reserve``), re-reserved after every
allocator reset. Per rank: node_cap blocks of the KV pool (node_cap x block bytes), two copy tables; one gather
launch per attention layer per forward (fixed grid, early exit when no node is laid out); no communication.

**Mamba** (batch-invariant decode layers). A node at depth d is a one-token decode row whose Mamba state must be the
one sequential decode would hold after its ancestors:
- conv: the node runs MCore's ``causal_conv1d_update`` on a scratch conv state, gathered in the same launch as the
  step's conv indices: the last ``state_len`` columns of [slot state, x(root), x(ancestors)]. The scratch states are
  ``node_cap`` extra rows of the context's conv-state tensor, after its slots (``_allocate_mamba_states``), so one
  kernel call serves every row;
- SSM: the node's prep program writes its x / dt / B / C to its own scratch replay row (``look_srow``) and the scan
  reads its ancestors' rows from there (mamba_bi_fused_decode / mamba_bi_scan_cuda). The scratch replay rows are
  allocated here, explicitly, on the layer's decode buffers the first time MCore hands them out
  (``MambaBatchInvariantDecode._get_buffers``), whoever allocated those buffers: Megatron-LM main allocates its
  z-free buffers itself (#8221), older pins through nemo_rl's ``mamba_bi_decode``;
- commit (``commit_mamba``): for the accepted path of a tokens, every layer's write cursor <- w0 + a, its replay rows
  w0 + 1 .. w0 + a - 1 <- the path nodes' scratch rows, its conv state <- the depth a - 1 node's scratch state.
A node never completes a Mamba chunk (tree_spec caps the depth by the chunk room), so no SSM state is committed for a
rejected node and rejected nodes need no rewind.
Per rank and Mamba layer: node_cap conv rows (conv_dim / TP x state_len x 4 B each) and node_cap replay rows
((nheads x (headdim + 1) + 2 ngroups x dstate) / TP x 4 B each); one extra launch per layer per decode step (the
window gather); no communication; fixed storage (CUDA-graph capturable).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False

LOOK_MAX = 8  # = mamba_bi_fused_decode.LOOK_MAX (root + up to 7 ancestors)
_STATE: dict[str, Any] = {"cap": 0}
_ORIG: dict[str, Any] = {}

if HAVE_TRITON:

    @triton.jit
    def _layer_gather_kernel(buf, gath, layer_off, s_kv, s_block, s_pos, row_elems,
                             LOOK: tl.constexpr, BLOCK: tl.constexpr):
        """Program (node n, ancestor slot j): K and V of position `off` of the layer at element offset `layer_off`
        (layer x layer stride, computed on the host in Python ints: an in-kernel int32 `layer * s_layer` passed 2^31
        elements on large KV pools -- Nano: 6 attention layers, layer 5 past ~6.5k blocks of 256 tokens -- and
        wrapped to a wrong place; exactness bug): src block -> dst block."""
        n = tl.program_id(0)
        j = tl.program_id(1)
        base = gath + (n * LOOK + j) * 3
        src = tl.load(base + 0)
        if src < 0:
            return
        dst = tl.load(base + 1).to(tl.int64)
        off = tl.load(base + 2).to(tl.int64)
        src = src.to(tl.int64)
        offs = tl.arange(0, BLOCK)
        for kv in tl.static_range(2):
            sp = buf + kv * s_kv + layer_off + src * s_block + off * s_pos
            dp = buf + kv * s_kv + layer_off + dst * s_block + off * s_pos
            for o in range(0, row_elems, BLOCK):
                m = o + offs < row_elems
                tl.store(dp + o + offs, tl.load(sp + o + offs, mask=m), mask=m)

    @triton.jit
    def _partial_copy_kernel(buf, part, s_kv, s_layer, s_block, s_pos, row_elems, BLOCK: tl.constexpr):
        """Program (node n, layer l): positions 0 .. count - 1 of the request's last block -> the node's block."""
        n = tl.program_id(0)
        lyr = tl.program_id(1).to(tl.int64)
        src = tl.load(part + n * 3 + 0)
        if src < 0:
            return
        dst = tl.load(part + n * 3 + 1).to(tl.int64)
        cnt = tl.load(part + n * 3 + 2)
        src = src.to(tl.int64)
        offs = tl.arange(0, BLOCK)
        total = cnt.to(tl.int64) * row_elems
        for kv in tl.static_range(2):
            sp = buf + kv * s_kv + lyr * s_layer + src * s_block
            dp = buf + kv * s_kv + lyr * s_layer + dst * s_block
            for o in range(0, total, BLOCK):
                m = o + offs < total
                tl.store(dp + o + offs, tl.load(sp + o + offs, mask=m), mask=m)

    @triton.jit
    def _commit_copy_kernel(buf, real_blk, acc, p_in, path_blk, s_path, s_kv, s_layer, s_block, s_pos, row_elems,
                            BS: tl.constexpr, BLOCK: tl.constexpr):
        """Program (tail t, layer l): accepted path nodes at depths 1 .. a - 1 (private blocks path_blk[t, d - 1]):
        their K / V at position p - 1 + d -> the request's block."""
        t = tl.program_id(0)
        lyr = tl.program_id(1).to(tl.int64)
        rb = tl.load(real_blk + t)
        a = tl.load(acc + t)
        if (rb < 0) | (a < 2):
            return
        p = tl.load(p_in + t)
        rb = rb.to(tl.int64)
        offs = tl.arange(0, BLOCK)
        for d in range(1, a):
            pb = tl.load(path_blk + t * s_path + d - 1).to(tl.int64)
            off = ((p - 1 + d) % BS).to(tl.int64)
            for kv in tl.static_range(2):
                sp = buf + kv * s_kv + lyr * s_layer + pb * s_block + off * s_pos
                dp = buf + kv * s_kv + lyr * s_layer + rb * s_block + off * s_pos
                for o in range(0, row_elems, BLOCK):
                    m = o + offs < row_elems
                    tl.store(dp + o + offs, tl.load(sp + o + offs, mask=m), mask=m)


def capacity() -> int:
    return int(_STATE["cap"])


def tables(device) -> dict:
    """Device copy tables shared with tree_spec's layout (fixed storage: the gather runs inside CUDA graphs).
    gath [cap, LOOK_MAX, 3] int32 (src block, dst block, in-block offset; src -1 = no copy), part [cap, 3] int32
    (src block, dst block, count; src -1 = none)."""
    dev = torch.device(device)
    key = ("tab", dev.index if dev.index is not None else torch.cuda.current_device())
    t = _STATE.get(key)
    if t is None:
        cap = max(capacity(), 1)
        t = _STATE[key] = {
            "gath": torch.full((cap, LOOK_MAX, 3), -1, dtype=torch.int32, device=dev),
            "part": torch.full((cap, 3), -1, dtype=torch.int32, device=dev),
            "dirty": False,
        }
    return t


def _strides(buf: torch.Tensor):
    assert buf.dim() == 6 and buf.is_contiguous(), "KV memory buffer [2, layers, blocks, block, heads, dim]"
    return buf.stride(0), buf.stride(1), buf.stride(2), buf.stride(3), buf.shape[4] * buf.shape[5]


def _append_key_value_cache(self, layer_number: int, key, value) -> None:
    out = _ORIG["append"](self, layer_number, key, value)
    # Always launched once installed (also while CUDA graphs are captured): an empty table is an early exit.
    if not self.mtp_metadata.forward_active and not self.cache_mla_latent:
        t = tables(self.memory_buffer.device)
        buf = self.memory_buffer
        s_kv, s_layer, s_block, s_pos, row = _strides(buf)
        layer = int(self.layer_map[layer_number - 1])
        layer_gather(buf, t["gath"], layer)
    return out


def layer_gather(buf: torch.Tensor, gath: torch.Tensor, layer: int) -> None:
    """The per-layer node gather on KV buffer ``buf`` (one launch, fixed grid; inside CUDA graphs)."""
    s_kv, s_layer, s_block, s_pos, row = _strides(buf)
    _layer_gather_kernel[(gath.shape[0], LOOK_MAX)](buf, gath, int(layer) * int(s_layer), s_kv, s_block, s_pos, row,
                                                   LOOK=LOOK_MAX, BLOCK=256)


def partial_copy(ctx) -> None:
    t = tables(ctx.memory_buffer.device)
    buf = ctx.memory_buffer
    s_kv, s_layer, s_block, s_pos, row = _strides(buf)
    _partial_copy_kernel[(t["part"].shape[0], buf.shape[1])](buf, t["part"], s_kv, s_layer, s_block, s_pos, row,
                                                             BLOCK=1024)


def clear(ctx) -> None:
    """No node in the next forward: the in-graph gather finds no entry."""
    t = tables(ctx.memory_buffer.device)
    if t["dirty"]:
        t["gath"].fill_(-1)
        t["part"].fill_(-1)
        t["dirty"] = False


def commit_copy(ctx, real_blk: torch.Tensor, acc: torch.Tensor, p_in: torch.Tensor, path_blk: torch.Tensor) -> None:
    T = real_blk.numel()
    if T == 0:
        return
    buf = ctx.memory_buffer
    s_kv, s_layer, s_block, s_pos, row = _strides(buf)
    _commit_copy_kernel[(T, buf.shape[1])](buf, real_blk, acc, p_in, path_blk, path_blk.stride(0), s_kv, s_layer,
                                           s_block, s_pos, row, BS=int(ctx.block_size_tokens), BLOCK=256)


class PrivateBlocks:
    """node_cap private KV blocks reserved from the context's allocator (re-reserved after every allocator reset:
    MCore resets the allocator whenever it resets its metadata, which frees every block)."""

    def __init__(self, ctx, n: int):
        self.ctx, self.n = ctx, int(n)
        alloc = ctx.kv_block_allocator
        self.ids = torch.zeros(self.n, dtype=torch.int32, device=torch.cuda.current_device())
        self._reserve()
        orig_reset = alloc.reset

        def reset(*a, **kw):
            out = orig_reset(*a, **kw)
            self._reserve()
            return out

        alloc.reset = reset
        # live-only KV offload (kv_live_offload, colo engine memory) checks at every suspend that the allocator's
        # used blocks are exactly their owners': these are owned by the tree (scratch: nothing to save)
        try:
            from nemo_rl.models.megatron import kv_live_offload
        except ImportError:  # the branch without live offload: no owner check to satisfy
            kv_live_offload = None
        if kv_live_offload is not None:
            kv_live_offload.register_block_owner("tree_kv.private", lambda: self.host)

    def _reserve(self) -> None:
        ids = self.ctx.kv_block_allocator.allocate_memory_blocks(self.n)
        if ids is None:
            raise RuntimeError(f"[tree_kv] KV pool cannot reserve {self.n} private node blocks")
        self.ids.copy_(ids.to(torch.int32), non_blocking=False)
        self.host = ids.tolist()


# ---------------------------------------------------------------------------------------------------------------------
# Mamba node state (batch-invariant decode layers)
# ---------------------------------------------------------------------------------------------------------------------

# Registered Mamba decode buffers, (layer number, buffers) in any order; tables are built sorted by layer number, the
# order of the context's per-layer conv states.
_MAMBA: dict[str, Any] = {"bufs": [], "nbuf_ptrs": None, "replay_ptrs": None}

if HAVE_TRITON:

    @triton.jit
    def _conv_window_kernel(state_ptr, x_ptr, bi_ptr, off_ptr, chain_ptr, sid_ptr, idx_out_ptr, stride_sb, stride_sd,
                            stride_sw, stride_x, stride_chain, conv_dim, STATE_LEN: tl.constexpr,
                            BLOCK_D: tl.constexpr):
        """Program (e, d-block): conv index of decode entry e (its slot, or its scratch row if it is a node row) and,
        for a node at depth i, its scratch state = last STATE_LEN columns of [slot state, x(chain[e, 0]), ...,
        x(chain[e, i-1])] (pure copies)."""
        e = tl.program_id(axis=0).to(tl.int64)
        db = tl.program_id(axis=1)
        off = tl.load(off_ptr + e)
        slot32 = tl.load(bi_ptr + e)
        slot = slot32.to(tl.int64)  # slot x row stride in int64 (the KV gather's int32-product bug class)
        sid = tl.load(sid_ptr + e)
        if db == 0:
            tl.store(idx_out_ptr + e, tl.where(off > 0, sid, slot32))
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
    def _mamba_commit_kernel(conv_ptr, nbuf_ptrs, rp_ptrs, slot_ptr, path_ptr, acc_ptr, cursor_ptr, n_layers,
                             stride_path, stride_cl, stride_cb, conv_row_elems, scratch_base, n_x, n_dt, n_bc,
                             CHUNK: tl.constexpr, BLOCK: tl.constexpr, SKIP_REPLAY: tl.constexpr):
        """Program (tail t, layer l): commit of a tail's accepted path (a tokens, path nodes at depths 1..a-1 with
        scratch indices path[t, 0 .. a-2]): cursor <- (w0 + a) mod CHUNK; replay rows w0 + 1 + j <- the path node's
        scratch rows (x, dt, B, C); conv state <- the depth a-1 node's scratch conv row (a >= 2)."""
        t = tl.program_id(axis=0)
        lyr = tl.program_id(axis=1).to(tl.int64)
        slot = tl.load(slot_ptr + t)
        a = tl.load(acc_ptr + t)
        if (slot < 0) | (a <= 0):
            return
        w0 = tl.load(cursor_ptr + t)
        nb = tl.load(nbuf_ptrs + lyr).to(tl.pointer_type(tl.int32))
        tl.store(nb + slot, (w0 + a) % CHUNK)
        if a < 2:
            return
        slot64 = slot.to(tl.int64)
        offs = tl.arange(0, BLOCK)
        if not SKIP_REPLAY:
            px = tl.load(rp_ptrs + 0 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            pdt = tl.load(rp_ptrs + 1 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            pb = tl.load(rp_ptrs + 2 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            pc = tl.load(rp_ptrs + 3 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            sx = tl.load(rp_ptrs + 4 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            sdt = tl.load(rp_ptrs + 5 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            sb = tl.load(rp_ptrs + 6 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            sc = tl.load(rp_ptrs + 7 * n_layers + lyr).to(tl.pointer_type(tl.float32))
            for j in range(0, a - 1):
                src = tl.load(path_ptr + t * stride_path + j).to(tl.int64)
                dst = slot64 * CHUNK + w0 + 1 + j
                for o in range(0, n_x, BLOCK):
                    m = o + offs < n_x
                    tl.store(px + dst * n_x + o + offs, tl.load(sx + src * n_x + o + offs, mask=m), mask=m)
                for o in range(0, n_bc, BLOCK):
                    m = o + offs < n_bc
                    tl.store(pb + dst * n_bc + o + offs, tl.load(sb + src * n_bc + o + offs, mask=m), mask=m)
                    tl.store(pc + dst * n_bc + o + offs, tl.load(sc + src * n_bc + o + offs, mask=m), mask=m)
                for o in range(0, n_dt, BLOCK):
                    m = o + offs < n_dt
                    tl.store(pdt + dst * n_dt + o + offs, tl.load(sdt + src * n_dt + o + offs, mask=m), mask=m)
        last = tl.load(path_ptr + t * stride_path + a - 2).to(tl.int64) + scratch_base  # conv scratch row
        for o in range(0, conv_row_elems, BLOCK):
            m = o + offs < conv_row_elems
            v = tl.load(conv_ptr + lyr * stride_cl + last * stride_cb + o + offs, mask=m)
            tl.store(conv_ptr + lyr * stride_cl + slot64 * stride_cb + o + offs, v, mask=m)


def mamba_layers() -> int:
    """Mamba layers whose decode buffers carry the node scratch rows (registered at their first use)."""
    return len(_MAMBA["bufs"])


def _sorted_bufs() -> list:
    return [b for _n, b in sorted(_MAMBA["bufs"], key=lambda nb: nb[0])]


def attach_scratch(buffers, layer_number: int) -> None:
    """Explicit allocation of a Mamba layer's node scratch replay rows (x, dt, B, C) on its batch-invariant decode
    buffers, and their registration for the commit (idempotent per buffers object). Called on the buffers MCore hands
    out (``_get_buffers``), so it serves any allocator of those buffers. The layer must be served by the fused decode
    step: it is the step that writes / reads node rows through ``look_srow``."""
    if getattr(buffers, "scratch", None) is not None:
        return
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    cap = capacity()
    _n, chunk, nheads, headdim = buffers.x.shape
    ngroups, dstate = buffers.B.shape[-2:]
    dev, dt = buffers.x.device, buffers.x.dtype
    has_z = getattr(buffers, "z", None) is not None
    if has_z or not fd.supported_config(nheads, headdim, ngroups, dstate, chunk, dt, dev):
        raise RuntimeError(
            f"[tree_kv] Mamba layer {layer_number}: tree node rows need the batch-invariant fused decode step on every "
            f"Mamba layer (z-free fp32 replay buffers of a supported shape; got z={has_z}, dtype {dt}, nheads {nheads}, "
            f"headdim {headdim}, ngroups {ngroups}, dstate {dstate}, chunk {chunk})")
    buffers.scratch = (
        torch.zeros(cap, nheads, headdim, device=dev, dtype=dt),
        torch.zeros(cap, nheads, device=dev, dtype=dt),
        torch.zeros(cap, ngroups, dstate, device=dev, dtype=dt),
        torch.zeros(cap, ngroups, dstate, device=dev, dtype=dt),
    )
    _MAMBA["bufs"].append((int(layer_number), buffers))
    _MAMBA["nbuf_ptrs"] = _MAMBA["replay_ptrs"] = None


def num_buffered_ptrs(device) -> torch.Tensor:
    """int64 device table of every registered layer's ``num_buffered`` (write cursor) data pointer, layer order."""
    p = _MAMBA["nbuf_ptrs"]
    if p is None or p.numel() != mamba_layers():
        p = _MAMBA["nbuf_ptrs"] = torch.tensor([b.num_buffered.data_ptr() for b in _sorted_bufs()], dtype=torch.int64,
                                               device=device)
    return p


def replay_ptrs(device) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """(int64 [8, L] device table: slot x / dt / B / C then scratch x / dt / B / C data pointers per layer, (n_x,
    nheads, n_bc) row sizes), layer order, built once after allocation."""
    p = _MAMBA["replay_ptrs"]
    if p is None or p[0].shape[1] != mamba_layers():
        bufs = _sorted_bufs()
        rows = [[b.x.data_ptr() for b in bufs], [b.dt.data_ptr() for b in bufs], [b.B.data_ptr() for b in bufs],
                [b.C.data_ptr() for b in bufs]]
        rows += [[b.scratch[i].data_ptr() for b in bufs] for i in range(4)]
        b0 = bufs[0]
        sizes = (b0.x[0, 0].numel(), b0.dt[0, 0].numel(), b0.B[0, 0].numel())
        for b in bufs:
            assert (b.x[0, 0].numel(), b.dt[0, 0].numel(), b.B[0, 0].numel()) == sizes
            assert all(t.is_contiguous() for t in (b.x, b.dt, b.B, b.C) + tuple(b.scratch))
        p = _MAMBA["replay_ptrs"] = (torch.tensor(rows, dtype=torch.int64, device=device), sizes)
    return p


def conv_states_full(ctx) -> Optional[torch.Tensor]:
    """The context's conv states with the node scratch rows after its slots [layers, slots + node_cap, ...]; None
    for a context without Mamba layers (or not batch-invariant)."""
    return getattr(ctx, "tree_conv_states_full", None)


def conv_scratch_base(ctx) -> int:
    """Index of the first node scratch row in ``conv_states_full``: the context's conv-state slot count (max_requests,
    plus MCore's dummy recurrent slot where it reserves one)."""
    return int(ctx.mamba_conv_states.shape[1])


def _step_buffers(n_rows: int, device) -> dict:
    """Per-device step buffers: conv index per decode entry and the scratch conv row of each node entry."""
    key = ("convbufs", device.index if device.index is not None else torch.cuda.current_device())
    b = _STATE.get(key)
    if b is None or b["conv_idx"].numel() < n_rows:
        b = _STATE[key] = {
            "conv_idx": torch.full((n_rows,), -1, dtype=torch.int32, device=device),
            "scratch_id": torch.full((n_rows,), -1, dtype=torch.int32, device=device),
        }
    return b


def scratch_ids(n_rows: int, device) -> torch.Tensor:
    """[n_rows] int32: scratch conv row of each decode entry that is a node row (written by tree_spec's layout)."""
    return _step_buffers(n_rows, torch.device(device))["scratch_id"]


def conv_step(xBC: torch.Tensor, conv_state: torch.Tensor, full: torch.Tensor, weight: torch.Tensor,
              bias: torch.Tensor, activation: Optional[str], batch_indices: torch.Tensor, conv_fn) -> torch.Tensor:
    """The batch-invariant decode conv of every decode entry, node rows on their scratch states. xBC [b, d] in the
    conv-state dtype; one-token decode for entries at offset 0 (their own slot, in place). ``full``: the layer's conv
    states including the scratch rows (``conv_state`` is MCore's view of its slots)."""
    from nemo_rl.models.megatron import mamba_bi_fused_decode as fd

    b, d = xBC.shape
    rows = max(b, conv_state.shape[0])
    off, chain = fd.lookahead_buffers(rows, xBC.device)
    bufs = _step_buffers(rows, xBC.device)
    idx = bufs["conv_idx"][:b]
    assert xBC.stride(-1) == 1 and full.stride(-2) >= 1
    BLOCK_D = 256
    _conv_window_kernel[(b, triton.cdiv(d, BLOCK_D))](
        full, xBC, batch_indices, off, chain, bufs["scratch_id"], idx,
        full.stride(0), full.stride(1), full.stride(2), xBC.stride(0), chain.stride(0), d,
        STATE_LEN=full.shape[-1], BLOCK_D=BLOCK_D,
    )
    return conv_fn(xBC, full, weight, bias, activation, conv_state_indices=idx)


def commit_mamba(ctx, slots: torch.Tensor, path: torch.Tensor, accepted: torch.Tensor, cursors: torch.Tensor,
                 chunk: int, skip_replay: bool = False) -> None:
    """After a tree tail's acceptance: per tail t with a = accepted[t] >= 1 tokens and accepted path nodes at depths
    1..a-1 with scratch indices path[t, :a-1]: every Mamba layer's cursor <- (cursors[t] + a) mod chunk, its replay rows
    cursors[t] + 1 .. + a - 1 <- the path's scratch replay rows, its conv state <- the scratch conv row of the depth a-1
    node. ``skip_replay``: negative control tree_stale_mamba (the slot keeps stale replay rows)."""
    T = slots.numel()
    full = conv_states_full(ctx)
    if T == 0 or full is None or not _MAMBA["bufs"]:
        return
    L = full.shape[0]
    assert mamba_layers() == L, ("tree_kv: registered Mamba layers != the context's", mamba_layers(), L)
    rp, (n_x, n_dt, n_bc) = replay_ptrs(full.device)
    _mamba_commit_kernel[(T, L)](
        full, num_buffered_ptrs(full.device), rp, slots, path, accepted, cursors, L, path.stride(0), full.stride(0),
        full.stride(1), full[0, 0].numel(), conv_scratch_base(ctx), n_x, n_dt, n_bc,
        CHUNK=int(chunk), BLOCK=1024, SKIP_REPLAY=bool(skip_replay),
    )


def _allocate_mamba_states(self):
    """MCore's Mamba state allocation, then node_cap scratch conv rows after the conv-state slots: one tensor
    [layers, slots + node_cap, ...] (``tree_conv_states_full``); MCore keeps its view of the slots."""
    _ORIG["allocate_states"](self)
    if not getattr(self, "is_hybrid_model", False) or not getattr(self, "batch_invariant_mode", False):
        return
    old = self.mamba_conv_states
    full = torch.empty((old.shape[0], old.shape[1] + capacity()) + tuple(old.shape[2:]), dtype=old.dtype,
                       device=old.device)
    full[:, : old.shape[1]].copy_(old)
    self.mamba_conv_states = full[:, : old.shape[1]]
    self.tree_conv_states_full = full
    del old


def _get_buffers(self, max_requests, x, B):
    """``MambaBatchInvariantDecode._get_buffers``: the layer's buffers (whoever allocated them) with node scratch
    rows."""
    buf = _ORIG["get_buffers"](self, max_requests, x, B)
    if getattr(buf, "scratch", None) is None:
        attach_scratch(buf, getattr(self.mixer, "layer_number", None) or mamba_layers())
    return buf


def _ssm_decode(self, zxBCdt, conv_state, ssm_state, batch_indices=None, intermediate_conv_state=None,
                intermediate_ssm_state=None):
    """MCore's batch-invariant ``ssm_decode`` (identical on Megatron-LM main a11c84aec and a68555ff) with the conv
    step through ``conv_step`` (node rows on their scratch conv states)."""
    if not self.config.batch_invariant_mode or capacity() == 0:
        return _ORIG["ssm_decode"](self, zxBCdt, conv_state, ssm_state, batch_indices=batch_indices,
                                   intermediate_conv_state=intermediate_conv_state,
                                   intermediate_ssm_state=intermediate_ssm_state)
    from einops import rearrange
    from megatron.core.ssm import mamba_mixer as mm

    assert mm.causal_conv1d_update_cuda is not None, "Batch-invariant Mamba decode requires causal-conv1d"
    _batch, seq_len, _ = zxBCdt.shape
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
    full = _full_of(conv_state)
    xBC = conv_step(
        x2, conv_state, full, rearrange(self.conv1d_weight, "d 1 w -> d w").to(conv_state.dtype),
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


def _full_of(conv_state: torch.Tensor) -> torch.Tensor:
    """The layer's conv-state rows including the node scratch rows after its slots: ``conv_state`` is a slice of
    ``tree_conv_states_full``, so the same storage viewed with node_cap more rows."""
    return torch.as_strided(conv_state, (conv_state.shape[0] + capacity(),) + tuple(conv_state.shape[1:]),
                            conv_state.stride(), conv_state.storage_offset())


def install(node_cap: int) -> None:
    """Node-row state of tree_spec, before any engine is built (the gather and the conv window are captured in the
    decode graphs; the scratch conv rows are allocated with the context's Mamba states): the KV append hook and the
    Mamba node state (scratch conv rows, the decode conv on them, scratch replay rows on every layer's decode buffers).
    Idempotent; the capacity is fixed once installed."""
    import sys

    cap = int(node_cap)
    assert cap > 0
    if _ORIG:
        assert cap == _STATE["cap"], ("tree_kv capacity is fixed once installed", cap, _STATE["cap"])
        return
    la = sys.modules.get("nemo_rl.models.megatron.mamba_bi_lookahead")
    if la is not None and la.capacity() > 0:
        raise RuntimeError("tree_kv: multi-token tail decode lookahead rows are installed; tail_tree and tail_mtd are "
                           "exclusive")
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.ssm.ops.mamba2.batch_invariant_decode import MambaBatchInvariantDecode

    _STATE["cap"] = cap
    tables(torch.device("cuda", torch.cuda.current_device()))  # fixed storage before any graph capture
    _ORIG["append"] = DynamicInferenceContext.append_key_value_cache
    _ORIG["allocate_states"] = DynamicInferenceContext._allocate_mamba_states
    _ORIG["ssm_decode"] = MambaMixer.ssm_decode
    _ORIG["get_buffers"] = MambaBatchInvariantDecode._get_buffers
    DynamicInferenceContext.append_key_value_cache = _append_key_value_cache
    DynamicInferenceContext._allocate_mamba_states = _allocate_mamba_states
    MambaMixer.ssm_decode = _ssm_decode
    MambaBatchInvariantDecode._get_buffers = _get_buffers
