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
"""KV cache of tree node rows (tree_spec): one private block per node row, filled so that every node attends to
exactly the keys and values one-token decode would see at its prefix.

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
"""

from __future__ import annotations

from typing import Any

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


def install(node_cap: int) -> None:
    """Hook the KV append (class level, before any engine is built: the gather is captured in the decode graphs)."""
    cap = int(node_cap)
    assert cap > 0
    if _ORIG:
        assert cap == _STATE["cap"], ("tree_kv capacity is fixed once installed", cap, _STATE["cap"])
        return
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext

    _STATE["cap"] = cap
    tables(torch.device("cuda", torch.cuda.current_device()))  # fixed storage before any graph capture
    _ORIG["append"] = DynamicInferenceContext.append_key_value_cache
    DynamicInferenceContext.append_key_value_cache = _append_key_value_cache
