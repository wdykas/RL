"""Index-width validator for the engine's pools: the largest element offset every kernel family can be asked to form,
from the allocated context (or from a planned config), against the index width that family computes offsets in.

Background (exactness bug): tree_kv's per-layer gather computed ``layer * layer_stride`` in int32. On full Nano
at buffer_gb >= 25 (>= 6554 blocks) layer 5's offset passed 2^31 elements, the product wrapped, and node rows read
stale K / V (bitwise-wrong log-probs, identical tokens). Any kernel that multiplies a block / slot / row id by a
stride in int32 fails the same way once that product passes 2^31. This module is the backstop. Each pool's maximum
offset is computed per kernel family, and the engine refuses to start when a family's width cannot hold it.

Families and the offset each one forms (elements):
- ``kv_layer``: one layer's K (or V) slice, block id x block stride (Megatron-LM fused_kv_append, the paged attention
  kernels, hp_attention, tree_kv's private-block copies);
- ``kv_pool``: the whole buffer, plane x layer x block (tree_kv gather / partial / commit, kv_live_offload);
- ``mamba_slot``: one Mamba layer's SSM / conv state, slot id x slot stride (Megatron-LM selective_state_update,
  causal_conv1d_update, the BI masked row update, our BI decode / scan / lookahead kernels);
- ``moe_rows``: permuted expert rows, row x max(hidden, F, 2F) (fp16_moe(_dk), device_dispatch, bi_unpermute,
  ep_invariant_combine, Megatron-LM's inference permute / activations).

Width per family: 63 when every kernel of the family forms the offset in int64 (verified by the pool tests
research/colo/pool/test_pool_offsets.py and Megatron-LM tests/unit_tests/inference/test_pool_int64_offsets.py), else
31. Megatron-LM's widths are probed from the loaded source, because the pinned trees before pool-i64 are int32 there.
Per rank: a few integer products at engine setup; no device work.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Optional

LIMIT = {31: 2**31 - 1, 63: 2**63 - 1}


def _src_has(modname: str, needle: str) -> bool:
    import importlib

    try:
        return needle in inspect.getsource(importlib.import_module(modname))
    except (ImportError, OSError, TypeError):
        return False


def widths() -> dict[str, int]:
    """Index width (bits) per kernel family in the loaded code."""
    # fused_kv_append forms block_idx * block_stride in the block ids' dtype: int64 when the kernel widens them
    # (pool-i64) or when the engine's token_to_block_idx is int64 (Megatron-LM's gpu_view; then even the unpatched
    # kernel is int64 -- measured: plain BI decode bitwise equal at 5631 vs 62975 blocks per layer).
    mlm_kv = _src_has("megatron.core.inference.contexts.fused_kv_append_kernel",
                      "tl.load(block_idx_ptr + token_idx).to(tl.int64)") or \
        _src_has("megatron.core.inference.contexts.gpu_view",
                 "self.token_to_block_idx = self._buf[off : off + tok_int64_bytes].view(torch.int64)")
    mlm_slot = _src_has("megatron.core.ssm.ops.mamba2.mamba_ssm", "tl.load(state_batch_indices_ptr).to(tl.int64)") and \
        _src_has("megatron.core.ssm.ops.common.causal_conv1d_triton",
                 "tl.load(conv_state_indices_ptr + batch_id).to(tl.int64)")
    try:
        from nemo_rl.models.generation.megatron import tree_kv

        tree_ok = hasattr(tree_kv, "layer_gather")  # 15b97342: layer offset on the host (int64 when it needs to be)
    except ImportError:
        tree_ok = True  # no tree_kv kernels in this tree
    return {
        "kv_layer": 63 if mlm_kv else 31,
        "kv_pool": 63 if tree_ok else 31,
        "mamba_slot": 63 if mlm_slot else 31,
        "moe_rows": 63,
    }


@dataclass
class Plan:
    """A rank's engine shape (per rank: heads / experts after TP / EP)."""

    kv_blocks: int
    block_tokens: int
    attn_layers: int
    kv_heads_local: int
    head_dim: int
    kv_planes: int = 2
    mamba_slots: int = 0  # max_requests (+ scratch rows)
    ssm_slot_elems: int = 0  # heads_local x head_dim x d_state
    conv_slot_elems: int = 0  # conv_dim_local x (d_conv - 1)
    moe_rows: int = 0  # worst-case permuted rows per rank per forward (tokens x topk, after EP imbalance)
    moe_row_elems: int = 0  # max(hidden, F, 2F) of the widest permuted buffer


def max_offsets(p: Plan) -> dict[str, int]:
    """Largest element offset per family (the last element of the last block / slot / row)."""
    blk = p.block_tokens * p.kv_heads_local * p.head_dim
    layer = p.kv_blocks * blk
    return {
        "kv_layer": layer,
        "kv_pool": p.kv_planes * p.attn_layers * layer,
        "mamba_slot": p.mamba_slots * max(p.ssm_slot_elems, p.conv_slot_elems),
        "moe_rows": p.moe_rows * p.moe_row_elems,
    }


def violations(p: Plan, w: Optional[dict[str, int]] = None) -> list[str]:
    w = widths() if w is None else w
    out = []
    for fam, off in max_offsets(p).items():
        if off > LIMIT[w[fam]]:
            out.append(f"{fam}: max offset {off} elements > int{w[fam] + 1} limit {LIMIT[w[fam]]}")
    return out


def plan_from_context(ctx: Any, moe_rows: int = 0, moe_row_elems: int = 0) -> Plan:
    """The allocated engine context's pools (Megatron-LM DynamicInferenceContext)."""
    buf = ctx.memory_buffer
    # (planes, layers, blocks, block_tokens, heads, dim) or MLA latent (layers, blocks, block_tokens, latent)
    if buf.dim() == 6:
        planes, layers, blocks, bt, heads, dim = buf.shape
    else:
        planes, (layers, blocks, bt, heads), dim = 1, buf.shape, 1
    ssm = getattr(ctx, "mamba_ssm_states", None)
    conv = getattr(ctx, "mamba_conv_states_full", None)
    if conv is None:
        conv = getattr(ctx, "mamba_conv_states", None)
    slots = max(int(ssm.shape[1]) if ssm is not None else 0, int(conv.shape[1]) if conv is not None else 0)
    return Plan(
        kv_blocks=int(blocks), block_tokens=int(bt), attn_layers=int(layers), kv_heads_local=int(heads),
        head_dim=int(dim), kv_planes=int(planes), mamba_slots=slots,
        ssm_slot_elems=int(ssm[0, 0].numel()) if ssm is not None else 0,
        conv_slot_elems=int(conv[0, 0].numel()) if conv is not None else 0,
        moe_rows=int(moe_rows), moe_row_elems=int(moe_row_elems),
    )


def check_engine(engine: Any, moe_rows: int = 0, moe_row_elems: int = 0) -> dict[str, int]:
    """Engine rank, at setup: refuse a context whose pools need offsets wider than the loaded kernels form."""
    p = plan_from_context(engine.context, moe_rows, moe_row_elems)
    w = widths()
    bad = violations(p, w)
    if bad:
        raise RuntimeError("[index width] engine pools exceed the kernels' index width (int32 offsets would wrap to "
                           "a wrong block / slot / row, silently inexact): " + "; ".join(bad))
    return {**{f"max_{k}": v for k, v in max_offsets(p).items()}, **{f"bits_{k}": v for k, v in w.items()}}
