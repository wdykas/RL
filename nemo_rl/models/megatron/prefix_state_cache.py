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
"""Learner prefix-state cache: exact incremental forwards of growing rows (batch-invariant mode).

Verifying a row piece by piece (ThunderSync continuation rounds) re-forwards prompt + prefix +
piece every time: O(L^2 / piece) learner work per row. This cache keeps, per row, the state
of the training model at Mamba chunk boundaries s (multiples of ``chunk_size`` counted from
position 0):

- every attention layer's K and V of positions [0, s) (as fed to core attention);
- every Mamba layer's conv inputs of positions [s - d_conv + 1, s) and fp32 SSM state at s;

and forwards only positions [s, n) of the next row (n its length; at most chunk_size - 1
recomputed positions before the piece). Logits of those positions are bitwise the full
forward's under batch invariance:

- attention: FA4 over the new queries and the cached + new keys, through the entry point
  TE's training path uses (``flash_attn_varlen_func_v4``), called as the engine's
  batch-invariant prefill calls it (paged K/V, ``page_table`` + ``seqused_k``, queries packed
  per row by ``cu_seqlens_q``; num_splits from the layer, anchored split-KV when
  ``bi_split_kv`` is installed). A query row's key blocks are aligned to absolute position 0
  in both, so its reduction order does not change;
- Mamba: ``mamba_ssd_parity.ssm_prefix_forward`` (conv over the cached inputs + new ones;
  chunked scan restarted at the boundary from its fp32 state, the identity the engine's
  batch-invariant prefill relies on);
- everything else is per token (row-invariant GEMMs, norms, dropless MoE with sticky
  anchors applied per position).

Storage (CUDA graphs): preallocated device pools, addressed through device
index tensors, so a cached forward is a fixed sequence of kernels on fixed addresses and is
replayed from a CUDA graph (``learner_graphs``, eligibility through ``graph_key``):

- K / V: per attention layer a page pool [pages, page, kv_heads, head_dim]; a row owns the
  pages of its positions (``page_table`` row); page 0 is a trash page for padding writes;
- Mamba: per layer a slot pool of conv inputs [slots, conv_dim, d_conv - 1] and fp32 SSM
  states [slots, heads, head_dim, d_state]; slot 0 is all-zero (the start of a row), slot 1
  is a trash slot for unused writes;
- per row-count B, static device metadata (start, length, page table, input state slot, kept
  boundary slots / offsets) that the host fills before the forward (``copy_``) and the hooks
  read. The hooks write the new K/V and the kept boundary states into the pools in place.

The shapes of a forward depend only on (B, padded length) and the pool sizes, so one graph
per (B, length bucket) serves every start position (no prefix-boundary classes are needed:
the boundary is data). Pools are allocated once per process at the first cached forward
(``gpu_bytes``, ``NEMO_RL_PREFIX_CACHE_STATE_FRAC`` of it for Mamba slots) and kept; entries
beyond the pools move to pinned host memory (LRU) and come back into free pages / slots
when used, which changes neither shapes nor bits.

Membership is decided by the caller, identically on every rank of a replica (TP x PP): a
row is cached only under the key the caller passes, and dropped only by ``release`` or a
weight change (``set_weights_version``). The cache never drops rows on its own, so all
ranks of a replica pick the same start positions (the forwards' shapes agree).

Validation of a hit is by content: the cached tokens' longest common prefix with the new
row bounds the usable boundary, so a rejected piece (or any change) falls back to an
earlier boundary instead of reusing stale state.

Verify0 capture (``floors``): the first verification of a whole draft can be
forwarded under a key with ``floors[i]`` = the prompt's chunk boundary, so the row's state is
kept at boundaries from the prompt on (any rejection position is >= the prompt length); the
caller then ``trim``s the entry to the one boundary the row's first continuation piece will
start from (or releases it). The forward is the same hooked forward a first piece already
runs (start 0), so verify0's logits do not change.

No new communication: each rank keeps its own layers' (pipeline stage) and heads' (tensor
parallel) state. Not supported: positional embeddings that depend on absolute positions
(RoPE: offsets not built yet), context parallelism, packed sequences, autograd, fp32
attention inputs (precision_raise attn_core).
"""

from __future__ import annotations

import contextlib
import functools
import os
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch

_ACTIVE: list[Optional["_Call"]] = [None]

ZERO_SLOT = 0  # all-zero Mamba state: the input state of a row forwarded from position 0
TRASH_SLOT = 1  # unused kept-boundary writes
TRASH_PAGE = 0  # K/V writes of padding positions; page-table entries past a row's pages


def page_size() -> int:
    """K/V page (tokens). 256 = the engine's block size (the paged FA4 prefill call it runs)."""
    return int(os.environ.get("NEMO_RL_PREFIX_CACHE_PAGE", "256"))


def _state_frac() -> float:
    return float(os.environ.get("NEMO_RL_PREFIX_CACHE_STATE_FRAC", "0.5"))


def _position_embedding_type(chunk: Any) -> Any:
    """The model's position_embedding_type through every wrapper (DDP, Float16Module, ...).

    The guard used to look one ``.module`` level deep, so under
    DDP(Float16Module(model)) a RoPE model read None and passed. Any rotary
    embedding module present also counts as RoPE.
    """
    m, seen = chunk, 0
    while m is not None and seen < 16:
        pos = getattr(m, "position_embedding_type", None)
        if pos is not None:
            return pos
        m, seen = getattr(m, "module", None), seen + 1
    try:
        from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    except ImportError:  # pragma: no cover
        return None
    if any(isinstance(x, RotaryEmbedding) for x in chunk.modules()):
        return "rope"
    return None


@dataclass
class _Entry:
    """One row's cached state on this rank (pages / slots of the pools, or a host copy)."""

    tokens: torch.Tensor  # cpu int64, the row's tokens [0, kv_len)
    kv_len: int  # K/V cached for positions [0, kv_len)
    pages: list[int]  # pool pages of positions [0, kv_len) (page j: [j * page, (j + 1) * page))
    states: dict[int, int]  # boundary position -> Mamba slot
    # "v0": captured by a verify0 forward and not used since ("v0t": trimmed); "piece":
    # committed by a piece forward. Only "v0" / "v0t" entries accept ``trim``.
    origin: str = "piece"
    # Spilled to pinned host memory: {"kv": {layer: (K, V) [n_pages, page, h, d]},
    # "states": {boundary: {layer: (conv, ssm)}}}; pages / slots are then free.
    host: Optional[dict] = None

    @property
    def on_gpu(self) -> bool:
        return self.host is None


@dataclass
class _Row:
    """One batch row of a prefix-state forward (host-side plan)."""

    key: Optional[str]
    tokens: torch.Tensor  # cpu int64, the whole row [0, n)
    start: int  # first forwarded position (a chunk boundary)
    entry: Optional[_Entry]  # cached state used for [0, start) (None: start == 0)
    keep: list[int]  # boundaries to cache after the forward (ascending; [] if key is None)
    # Older cached boundaries (< start) kept as they are (``pins``: carried-walk scans
    # re-forward a block from its floor).
    retain: list[int] = field(default_factory=list)
    capture: bool = False  # a verify0 capture (``floors``): the entry is committed as "v0"
    pages: list[int] = field(default_factory=list)  # pages of positions [0, start + length)
    new_pages: list[int] = field(default_factory=list)  # of those, taken for this forward
    slot_in: int = ZERO_SLOT
    keep_slots: dict[int, int] = field(default_factory=dict)  # new boundary -> slot

    @property
    def length(self) -> int:
        return self.tokens.numel() - self.start

    @property
    def new_kv_len(self) -> int:
        return self.keep[-1] if self.keep else 0


class _Pools:
    """Device pools and per-row-count metadata buffers (fixed addresses for CUDA graphs)."""

    def __init__(self, attn: dict, mamba: dict, page: int, n_pages: int, n_slots: int, maxp: int,
                 max_keep: int, device):
        self.page, self.n_pages, self.n_slots, self.maxp, self.K = page, n_pages, n_slots, maxp, max_keep
        self.device = device
        self.k = {ln: torch.zeros((n_pages, page, h, d), dtype=dt, device=device)
                  for ln, (h, d, dt) in attn.items()}
        self.v = {ln: torch.zeros_like(t) for ln, t in self.k.items()}
        self.conv = {ln: torch.zeros((n_slots, c, w), dtype=dt, device=device)
                     for ln, (c, w, dt, *_rest) in mamba.items()}
        self.ssm = {ln: torch.zeros((n_slots, nh, hd, ds), dtype=torch.float32, device=device)
                    for ln, (_c, _w, _dt, nh, hd, ds) in mamba.items()}
        self.free_pages = list(range(n_pages - 1, TRASH_PAGE, -1))  # pop() -> ascending ids
        self.free_slots = list(range(n_slots - 1, TRASH_SLOT, -1))
        self.meta: dict[int, dict[str, torch.Tensor]] = {}

    def nbytes(self) -> int:
        n = 0
        for d in (self.k, self.v, self.conv, self.ssm):
            n += sum(t.numel() * t.element_size() for t in d.values())
        return n

    def ptr_key(self) -> tuple:
        first = next(iter(self.k.values()), None)
        if first is None:
            first = next(iter(self.ssm.values()), None)
        return (self.page, self.n_pages, self.n_slots, self.maxp, self.K,
                0 if first is None else first.data_ptr())

    def metadata(self, b: int) -> dict[str, torch.Tensor]:
        """Static device metadata of a B-row forward (allocated once per B)."""
        m = self.meta.get(b)
        if m is None:
            dev = self.device
            m = self.meta[b] = dict(
                start=torch.zeros(b, dtype=torch.long, device=dev),
                length=torch.zeros(b, dtype=torch.long, device=dev),
                page_table=torch.zeros((b, self.maxp), dtype=torch.int32, device=dev),
                slot_in=torch.zeros(b, dtype=torch.long, device=dev),
                keep_slot=torch.full((b, self.K), TRASH_SLOT, dtype=torch.long, device=dev),
                keep_off=torch.zeros((b, self.K), dtype=torch.long, device=dev),
            )
        return m


def kv_write_index(meta: dict, s: int, page: int, maxp: int):
    """(valid [B, S], page [B, S], offset [B, S]) of every forwarded position's K/V write.

    Position j of row b is rollout position start_b + j; real positions (j < length_b) go
    to the row's page ``page_table[b, pos // page]`` at ``pos % page``, padding to the trash
    page. Device ops only (graph-capturable).
    """
    start, length = meta["start"], meta["length"]
    t = torch.arange(s, device=start.device)
    valid = t.unsqueeze(0) < length.unsqueeze(1)
    pos = start.unsqueeze(1) + t.unsqueeze(0)
    pidx = torch.clamp(torch.div(pos, page, rounding_mode="floor"), max=maxp - 1)
    pg = meta["page_table"].gather(1, pidx).long()
    pg = torch.where(valid, pg, torch.full_like(pg, TRASH_PAGE))
    return valid, pg, torch.remainder(pos, page)


class _Call:
    """The rows of one forward; consulted by the hooked attention and Mamba layers."""

    def __init__(self, cache: "PrefixStateCache", rows: list[_Row], meta: dict[str, torch.Tensor]):
        self.cache = cache
        self.rows = rows
        self.meta = meta

    def attention(self, mod, layer: int, q, k, v) -> torch.Tensor:
        """Core attention (sbhd in, [s, b, heads * dim] out) over the pooled cached + new keys.

        Device-side only (fixed shapes for a given (B, S)): the new K/V of every real
        position are written into the row's pages, the queries are packed per row
        (``cu_seqlens_q`` from the lengths) and FA4 reads the keys through the page table.
        """
        from transformer_engine.pytorch.attention.dot_product_attention import (
            backends as te_backends,
        )
        from transformer_engine.pytorch.attention.dot_product_attention import (
            utils as dpa_utils,
        )

        fa4 = te_backends.flash_attn_varlen_func_v4
        if fa4 is None:
            raise RuntimeError("learner prefix-state cache needs FlashAttention-4")
        if q.dtype == torch.float32:
            raise NotImplementedError(
                "prefix-state cache: fp32 attention inputs (precision_raise attn_core) not built "
                "for the paged cache"
            )
        pools, meta = self.cache.pools, self.meta
        s, nb = q.shape[:2]
        dev = q.device
        page = pools.page
        start, length = meta["start"], meta["length"]
        valid, pg, off = kv_write_index(meta, s, page, pools.maxp)
        kf = k.transpose(0, 1).reshape(nb * s, *k.shape[2:])
        vf = v.transpose(0, 1).reshape(nb * s, *v.shape[2:])
        if pools.k[layer].dtype != k.dtype:
            # The pools follow the activation dtype the layers feed core attention; set at the
            # first (eager) cached forward, before anything is cached or captured.
            self.cache._retype_kv(k.dtype)
        kp, vp = pools.k[layer], pools.v[layer]
        kp.index_put_((pg.reshape(-1), off.reshape(-1)), kf.to(kp.dtype))
        vp.index_put_((pg.reshape(-1), off.reshape(-1)), vf.to(vp.dtype))
        # Queries packed per row in row order (real positions first; padding at the end).
        flat_valid = valid.reshape(-1)
        rank = (~flat_valid).to(torch.long) * (nb * s) + torch.arange(nb * s, device=dev)
        order = torch.argsort(rank)
        qf = q.transpose(0, 1).reshape(nb * s, *q.shape[2:]).index_select(0, order)
        cu_q = torch.zeros(nb + 1, dtype=torch.int32, device=dev)
        cu_q[1:] = torch.cumsum(length, 0).to(torch.int32)
        out = fa4(
            qf,
            kp,
            vp,
            cu_seqlens_q=cu_q,
            max_seqlen_q=s,
            max_seqlen_k=pools.maxp * page,
            seqused_k=(start + length).to(torch.int32),
            page_table=meta["page_table"],
            # TE keeps the scale on its FlashAttention backend module.
            softmax_scale=mod.flash_attention.softmax_scale,
            causal=True,
            # As TE passes it for a causal training forward.
            window_size=dpa_utils.check_set_window_size("causal", mod.window_size),
            num_splits=1 if mod.num_splits is None else mod.num_splits,
            deterministic=mod.deterministic,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        res = torch.empty_like(out)
        res.index_copy_(0, order, out)
        res = res.reshape(nb, s, -1)
        res = torch.where(valid.unsqueeze(-1), res, torch.zeros((), dtype=res.dtype, device=dev))
        return res.transpose(0, 1).contiguous()

    def mamba(self, mod, layer: int, zxBCdt) -> torch.Tensor:
        """``_ssm_training`` of the rows' segments, continued from their pooled states."""
        from nemo_rl.models.megatron.mamba_ssd_parity import ssm_prefix_forward

        pools, meta = self.cache.pools, self.meta
        if pools.conv[layer].dtype != zxBCdt.dtype:
            # Conv inputs are stored in the activation dtype (ssm_prefix_forward casts its
            # conv_init to it); set at the first (eager) cached forward.
            self.cache._retype_conv(zxBCdt.dtype)
        convp, ssmp = pools.conv[layer], pools.ssm[layer]
        conv_init = convp.index_select(0, meta["slot_in"])
        ssm_init = ssmp.index_select(0, meta["slot_in"])
        y, xbc_ext, chunk_states = ssm_prefix_forward(mod, zxBCdt, conv_init, ssm_init)
        chunk = mod.chunk_size
        nb, kk = meta["keep_off"].shape
        width = convp.shape[-1]
        dev = zxBCdt.device
        off = meta["keep_off"]  # [B, K] boundary - start (>= chunk) or 0 for unused
        bidx = torch.arange(nb, device=dev).unsqueeze(1).expand(nb, kk)
        ci = torch.div(off, chunk, rounding_mode="floor") - 1  # unused: -1 (last chunk, trash)
        ci = torch.remainder(ci, chunk_states.shape[1])
        cs = chunk_states[bidx, ci]  # [B, K, nh, hd, ds]
        xe = xbc_ext.transpose(1, 2)  # [B, W + L, conv_dim]
        idx = off.unsqueeze(-1) + torch.arange(width, device=dev)  # [B, K, W]
        cv = xe[bidx.unsqueeze(-1), idx]  # [B, K, W, conv_dim]
        slots = meta["keep_slot"].reshape(-1)
        ssmp.index_put_((slots,), cs.reshape(nb * kk, *cs.shape[2:]).to(ssmp.dtype))
        convp.index_put_((slots,), cv.transpose(-1, -2).reshape(nb * kk, *convp.shape[1:]).to(convp.dtype))
        return y


def _hooked_attention(mod, layer: int, query, key, value, attention_mask, *args, **kwargs):
    call = _ACTIVE[0]
    if call is None:
        return type(mod).forward(mod, query, key, value, attention_mask, *args, **kwargs)
    if kwargs.get("packed_seq_params") is not None or kwargs.get("attention_bias") is not None:
        raise NotImplementedError("prefix-state forwards: no packed sequences / attention bias")
    return call.attention(mod, layer, query, key, value)


def _hooked_ssm(mod, layer: int, zxBCdt, packed_seq_params=None):
    call = _ACTIVE[0]
    if call is None:
        return type(mod)._ssm_training(mod, zxBCdt, packed_seq_params)
    if packed_seq_params is not None:
        raise NotImplementedError("prefix-state forwards: no packed sequences")
    return call.mamba(mod, layer, zxBCdt)


def _model_chunks(model) -> list:
    return list(model) if isinstance(model, (list, tuple)) else [model]


def graph_key() -> Optional[tuple]:
    """Eligibility of the active prefix-state forward for a learner CUDA graph.

    ``learner_graphs`` calls this when a prefix-state forward is active: None keeps the
    forward eager; a tuple joins the graph's key (pool addresses and sizes: a graph only
    replays on the pools it was captured on). The B-row metadata buffers are static per
    B, which the graph key already holds (the input shape).
    """
    call = _ACTIVE[0]
    if call is None:
        return ()
    cache = call.cache
    if not cache.graphs or cache.pools is None:
        return None
    return ("prefix",) + cache.pools.ptr_key()


class PrefixStateCache:
    """Per-rank store of rows' boundary states and the hooks that use them.

    ``model``: the training model (chunks); hooks are installed on its local
    attention and Mamba layers (they are no-ops outside ``forward``).
    ``gpu_bytes``: device pool budget (K/V pages + Mamba slots); colder entries live in
    pinned host memory. The caller bounds the total (``release``).
    ``max_len``: longest row the pools address (page-table width; default 131072).
    """

    def __init__(self, model, gpu_bytes: int, max_keep: int = 4, max_len: Optional[int] = None):
        from megatron.core.ssm.mamba_mixer import MambaMixer
        from megatron.core.transformer.attention import Attention
        from megatron.core.utils import get_model_config

        config = get_model_config(_model_chunks(model)[0])
        if getattr(config, "context_parallel_size", 1) != 1:
            raise NotImplementedError("prefix-state cache: context parallelism not built yet")
        if not getattr(config, "batch_invariant_mode", False):
            raise ValueError("prefix-state cache requires batch_invariant_mode")
        self._model = model
        self.max_keep = max(2, int(max_keep))
        self.gpu_bytes = int(gpu_bytes)
        self.max_len = int(max_len or 131072)
        self.entries: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU order
        self.weights_version: Any = None
        self.stats = {"hits": 0, "rows": 0, "tokens_skipped": 0, "tokens_forwarded": 0,
                      "v0_captured": 0, "v0_hits": 0, "v0_trimmed": 0, "v0_trim_dropped": 0,
                      "pool_fallback": 0}
        # Per row of the last ``forward``: its start came from a verify0-captured entry.
        self.last_from_v0: list[bool] = []
        # Negative control only (THUNDERSYNC_PREFIX_CACHE_NEGCTL, set by the caller): verify0
        # captures are committed scaled by (1 + negctl), a state that is not the whole-row
        # forward's (as a state kept across a weight change); the cache check must catch it.
        self.negctl = 0.0
        # Cached forwards may be replayed from learner CUDA graphs (``graph_key``).
        self.graphs = os.environ.get("NEMO_RL_PREFIX_CACHE_GRAPHS", "1") == "1"
        self.pools: Optional[_Pools] = None
        self.attention_layers: list[int] = []
        self.mamba_layers: list[int] = []
        self._attn_mods: dict[int, Any] = {}
        self._mamba_mods: dict[int, Any] = {}
        chunks: set[int] = set()
        for chunk in _model_chunks(model):
            pos = _position_embedding_type(chunk)
            if pos not in (None, "none"):
                raise NotImplementedError(
                    f"prefix-state cache: position_embedding_type={pos!r} (absolute "
                    "position offsets for RoPE not built yet)"
                )
            for m in chunk.modules():
                if isinstance(m, Attention):
                    ln = int(m.layer_number)
                    core = m.core_attention
                    core.forward = functools.partial(_hooked_attention, core, ln)
                    self.attention_layers.append(ln)
                    self._attn_mods[ln] = m
                elif isinstance(m, MambaMixer):
                    ln = int(m.layer_number)
                    m._ssm_training = functools.partial(_hooked_ssm, m, ln)
                    self.mamba_layers.append(ln)
                    self._mamba_mods[ln] = m
                    chunks.add(int(m.chunk_size))
        if len(chunks) > 1:
            raise ValueError(f"Mamba layers with different chunk sizes {chunks}")
        # Boundaries: Mamba chunk starts. A stage without Mamba layers still uses the
        # model's chunk size (all stages must agree on start positions).
        self.chunk = chunks.pop() if chunks else int(getattr(config, "mamba_chunk_size", 128) or 128)

    # ---- pools -----------------------------------------------------------------------
    def _layer_shapes(self) -> tuple[dict, dict]:
        from megatron.core.utils import get_model_config

        cfg = get_model_config(_model_chunks(self._model)[0])
        # K / V as fed to core attention: the activation dtype (fp16 recipe: fp16).
        act = torch.float16 if getattr(cfg, "fp16", False) else torch.bfloat16
        try:
            from nemo_rl.models.megatron import fp16_gemm

            if fp16_gemm._STATE.get("attention_fp16"):
                act = torch.float16
        except ImportError:  # pragma: no cover
            pass
        attn = {
            ln: (m.num_query_groups_per_partition, m.hidden_size_per_attention_head, act)
            for ln, m in self._attn_mods.items()
        }
        mamba = {}
        for ln, m in self._mamba_mods.items():
            w = m.cp.get_conv1d_weight()
            mamba[ln] = (w.shape[0], w.shape[-1] - 1, w.dtype, m.cp.nheads_local_tpcp, m.headdim, m.d_state)
        return attn, mamba

    def ensure_pools(self, attn_dtype: Optional[torch.dtype] = None) -> _Pools:
        """Allocate the device pools once (outside any capture)."""
        if self.pools is not None:
            return self.pools
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("prefix-state cache: pools must be allocated before graph capture")
        attn, mamba = self._layer_shapes()
        if attn_dtype is not None:
            attn = {ln: (h, d, attn_dtype) for ln, (h, d, _) in attn.items()}
        page = page_size()
        page_bytes = sum(2 * page * h * d * torch.empty((), dtype=dt).element_size()
                         for h, d, dt in attn.values())
        slot_bytes = sum(c * w * torch.empty((), dtype=dt).element_size() + nh * hd * ds * 4
                         for c, w, dt, nh, hd, ds in mamba.values())
        frac = _state_frac() if mamba and attn else (1.0 if mamba else 0.0)
        n_slots = max(2 + 4 * self.max_keep, int(self.gpu_bytes * frac // max(slot_bytes, 1))) if mamba else 2
        n_pages = max(2 + 2 * -(-self.max_len // page),
                      int(self.gpu_bytes * (1 - frac) // max(page_bytes, 1))) if attn else 2
        maxp = -(-(self.max_len + page) // page)
        dev = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        self.pools = _Pools(attn, mamba, page, n_pages, n_slots, maxp, self.max_keep, dev)
        return self.pools

    def _retype_kv(self, dtype: torch.dtype) -> None:
        p = self.pools
        if any(e.pages or e.host for e in self.entries.values()) or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(f"prefix-state cache: K/V pool dtype changed to {dtype} with state cached")
        p.k = {ln: torch.zeros(t.shape, dtype=dtype, device=t.device) for ln, t in p.k.items()}
        p.v = {ln: torch.zeros_like(t) for ln, t in p.k.items()}

    def row_need(self, key: Optional[str], n: int, start: int, floor: Optional[int] = None) -> tuple[int, int]:
        """(pages, state slots) row ``key`` of n tokens holds during a forward from ``start``:
        the pages of positions [0, n), its entry's slots and the slots of its new boundaries."""
        p = self.ensure_pools()
        e = self.entries.get(key) if key is not None else None
        slots = len(e.states) if (e is not None and start) else 0
        if key is not None:
            lo = int(floor) if floor is not None else 0
            slots += sum(b != start for b in self.keep(start, n, lo=lo))
        return -(-n // p.page), slots

    def fits(self, specs: list[tuple]) -> bool:
        """Whether one call fits the pools with every other entry spilled.

        ``specs``: (key, n, start, floor) per row. One extra page per row is reserved for the
        caller's power-of-two padding rows (one token each).
        """
        p = self.ensure_pools()
        need = [self.row_need(*x) for x in specs]
        pages = sum(x for x, _ in need) + len(specs)
        return pages <= p.n_pages - 1 and sum(y for _, y in need) <= p.n_slots - 2

    def _retype_conv(self, dtype: torch.dtype) -> None:
        p = self.pools
        if any(e.states or e.host for e in self.entries.values()) or (
            torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(f"prefix-state cache: conv pool dtype changed to {dtype} with state cached")
        p.conv = {ln: torch.zeros(t.shape, dtype=dtype, device=t.device) for ln, t in p.conv.items()}

    def drop_pools(self) -> bool:
        """Free the device pools between steps when nothing can address them (eager learner).

        The pools are preallocated at the full ``gpu_bytes`` budget for CUDA graphs (fixed addresses). With
        learner graphs off no graph holds their addresses, and every entry is stale once the weights change at
        the step's optimizer update, so the pools are dropped at the step end and re-allocated on the next use
        (final50, 10-20: the 8 GiB pools stayed resident through every optimizer step and checkpoint save while
        the step used 0.2-5 GiB of them). Returns whether they were dropped.
        """
        from nemo_rl.models.megatron import learner_graphs

        if self.pools is None or learner_graphs.enabled():
            return False
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return False
        self.clear()
        self.pools = None
        return True

    def pool_bytes(self) -> int:
        return 0 if self.pools is None else self.pools.nbytes()

    def _free(self, e: _Entry) -> None:
        """Return an entry's pages / slots (resident) to the pools."""
        if e.host is None and self.pools is not None:
            self.pools.free_pages.extend(e.pages)
            self.pools.free_slots.extend(s for s in e.states.values() if s > TRASH_SLOT)
        e.pages, e.states = [], {b: -1 for b in e.states}

    # ---- membership (caller-driven) -------------------------------------------------
    def set_weights_version(self, version: Any) -> None:
        """Drop everything when the weights changed (cached states are of the old ones)."""
        if version != self.weights_version:
            self.clear()
            self.weights_version = version

    def release(self, keys) -> None:
        for k in keys or ():
            e = self.entries.pop(k, None)
            if e is not None:
                self._free(e)

    def clear(self) -> None:
        for e in self.entries.values():
            self._free(e)
        self.entries.clear()

    # ---- planning --------------------------------------------------------------------
    def start(self, key: Optional[str], row: torch.Tensor, prompt_len: int) -> int:
        """First position to forward for ``row`` (logits are needed from prompt_len - 1).

        The largest cached boundary s <= prompt_len - 1 whose tokens [0, s) equal the
        row's; 0 without a usable entry.
        """
        e = self.entries.get(key) if key is not None else None
        if e is None:
            return 0
        n = min(e.kv_len, row.numel())
        same = e.tokens[:n] == row[:n].cpu()
        lcp = n if bool(same.all()) else int((~same).nonzero()[0])
        limit = min(lcp, prompt_len - 1)
        return max([s for s in e.states if s <= limit], default=0)

    def trim(self, key: str, bound: int) -> None:
        """After verify0's decision: keep only the largest boundary <= ``bound`` of a verify0
        capture (the row's first piece starts there: ``bound`` = its prompt length - 1); drop
        the entry if none. Memory only: ``start`` picks the same boundary without it. An entry
        a piece forward has already replaced is left alone (its boundaries are newer)."""
        e = self.entries.get(key)
        if e is None or e.origin not in ("v0", "v0t"):
            return
        usable = [b for b in e.states if b <= bound]
        if not usable:
            self.release([key])
            self.stats["v0_trim_dropped"] = self.stats.get("v0_trim_dropped", 0) + 1
            return
        b = max(usable)
        if e.host is None:
            if self.pools is not None:
                self.pools.free_slots.extend(s for x, s in e.states.items() if x != b and s > TRASH_SLOT)
            npg = -(-b // self.pools.page) if self.pools is not None else 0
            if self.pools is not None:
                self.pools.free_pages.extend(e.pages[npg:])
            e.pages = e.pages[:npg]
        else:
            e.host["states"] = {b: e.host["states"][b]}
            npg = -(-b // e.host["page"])
            e.host["kv"] = {ln: (k[:npg], v[:npg]) for ln, (k, v) in e.host["kv"].items()}
        e.states = {b: e.states[b]}
        if e.kv_len > b:
            e.kv_len = b
            e.tokens = e.tokens[:b].clone()
        e.origin = "v0t"
        self.stats["v0_trimmed"] = self.stats.get("v0_trimmed", 0) + 1

    def keep(self, start: int, n: int, lo: int = 0) -> list[int]:
        """Boundaries to keep after forwarding [start, n): start and those in (start, n - 1].

        A later row that extends [0, prompt) with prompt <= n (a whole or partial
        acceptance of this piece) uses the largest of them <= prompt - 1, so at most
        chunk - 1 positions before its piece are recomputed. Boundary n itself is
        never usable (position n - 1's logits are always needed again). Long pieces
        keep at most ``max_keep`` boundaries (start, the last one, evenly spaced
        between): a rejection inside such a piece then recomputes up to the spacing.
        ``lo`` > start (verify0 capture): boundaries from ``lo`` (rounded down to a chunk
        boundary) instead of from ``start``; start itself is then not kept.
        """
        first = max(start, lo // self.chunk * self.chunk)
        out = list(range(first, n, self.chunk))
        if not out:
            return []
        if len(out) > self.max_keep:
            k = self.max_keep - 1
            out = sorted({out[round(j * (len(out) - 1) / k)] for j in range(k + 1)})
        return out if out[-1] > 0 else []

    @staticmethod
    def retain(entry: Optional[_Entry], start: int, keep: list[int], pin: Optional[int]) -> list[int]:
        """Older boundary to keep for ``pin``: the largest cached one <= pin below start."""
        if pin is None or entry is None or not keep or not start:
            return []
        old = [b for b in entry.states if b <= pin and b < start]
        return [max(old)] if old else []

    # ---- residency ---------------------------------------------------------------------
    def _spill(self, key: str) -> None:
        """Move an entry's pages / slots to pinned host memory (frees them)."""
        e = self.entries[key]
        if e.host is not None:
            return
        p = self.pools

        def host(t):
            out = torch.empty(t.shape, dtype=t.dtype, pin_memory=t.is_cuda)
            out.copy_(t, non_blocking=True)
            return out

        idx = torch.tensor(e.pages, dtype=torch.long, device=p.device)
        kv = {ln: (host(p.k[ln].index_select(0, idx)), host(p.v[ln].index_select(0, idx))) for ln in p.k}
        st = {}
        for b, s in e.states.items():
            st[b] = {ln: (host(p.conv[ln][s]), host(p.ssm[ln][s])) for ln in p.conv}
        if p.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        self._free(e)
        e.host = {"kv": kv, "states": st, "page": p.page}
        self.stats["host_evictions"] = self.stats.get("host_evictions", 0) + 1

    def _take(self, n_pages: int, n_slots: int, protect: set) -> bool:
        """Make n_pages / n_slots free, spilling least recently used entries not in ``protect``."""
        p = self.pools
        for key in list(self.entries):
            if len(p.free_pages) >= n_pages and len(p.free_slots) >= n_slots:
                break
            if key not in protect and self.entries[key].host is None:
                self._spill(key)
        return len(p.free_pages) >= n_pages and len(p.free_slots) >= n_slots

    def _load(self, key: str) -> None:
        """Bring a spilled entry back into free pages / slots (the caller made room)."""
        e = self.entries[key]
        if e.host is None:
            return
        p = self.pools
        n = len(next(iter(e.host["kv"].values()))[0]) if e.host["kv"] else -(-e.kv_len // p.page)
        pages = [p.free_pages.pop() for _ in range(n)]
        if pages:
            idx = torch.tensor(pages, dtype=torch.long, device=p.device)
            for ln, (k, v) in e.host["kv"].items():
                p.k[ln].index_copy_(0, idx, k.to(p.device, non_blocking=True))
                p.v[ln].index_copy_(0, idx, v.to(p.device, non_blocking=True))
        states = {}
        for b, per in e.host["states"].items():
            s = p.free_slots.pop()
            for ln, (c, x) in per.items():
                p.conv[ln][s].copy_(c, non_blocking=True)
                p.ssm[ln][s].copy_(x, non_blocking=True)
            states[b] = s
        e.pages, e.states, e.host = pages, states, None
        self.stats["host_loads"] = self.stats.get("host_loads", 0) + 1

    # ---- forward ---------------------------------------------------------------------
    @contextlib.contextmanager
    def forward(
        self,
        keys: list[Optional[str]],
        rows: list[torch.Tensor],
        starts: list[int],
        pins: Optional[list[Optional[int]]] = None,
        floors: Optional[list[Optional[int]]] = None,
    ) -> Iterator[None]:
        """Run the model's forward of rows[i][starts[i]:] (left-aligned, padded) inside.

        Hooked layers continue each row from its cached state at starts[i] and record
        the state at the boundaries ``keep(starts[i], n_i)``; on success the rows'
        entries are replaced by the new state. Rows with key None are forwarded from
        their start without caching. ``pins[i]`` (a position or None): also keep the
        largest boundary <= pins[i] that row i's entry already holds below its start
        (all ranks get the same pins, so they keep the same boundaries).
        ``floors[i]`` (verify0 capture, an int or None): keep row i's boundaries from
        floors[i] on (``keep(.., lo=floors[i])``) and commit its entry as a verify0 capture
        (``trim`` applies to it).

        The call must fit the pools with every other entry spilled (``fits``); else it raises
        (the caller's inputs are already built from ``starts``).
        """
        if _ACTIVE[0] is not None:
            raise RuntimeError("nested prefix-state forwards")
        if torch.is_grad_enabled():
            raise RuntimeError("prefix-state forwards run without autograd")
        if len(set(k for k in keys if k is not None)) != sum(k is not None for k in keys):
            raise ValueError("a key appears twice in one prefix-state forward")
        p = self.ensure_pools()
        pins = pins if pins is not None else [None] * len(keys)
        floors = floors if floors is not None else [None] * len(keys)
        call_rows: list[_Row] = []
        protect = {k for k in keys if k is not None}
        for key, row, s, pin, floor in zip(keys, rows, starts, pins, floors):
            entry = self.entries.get(key) if key is not None else None
            if s and (entry is None or s not in entry.states):
                raise KeyError(f"no cached state of {key!r} at {s}")
            tokens = row.detach().to("cpu", torch.long)
            capture = key is not None and floor is not None
            keep = self.keep(s, tokens.numel(), lo=int(floor) if capture else 0) if key is not None else []
            retain = self.retain(entry, s, keep, pin) if key is not None else []
            call_rows.append(_Row(key, tokens, s, entry if s else None, keep, retain, capture=capture))
        # Resources: residency of used entries, pages of every forwarded position, kept slots.
        for r in call_rows:
            if not self._place(r, protect):
                # The caller built its inputs from these starts, so they cannot change here:
                # the caller splits calls to fit (``fits``); a call that does not is a bug.
                self._abort(call_rows)
                raise RuntimeError(
                    f"prefix-state cache: a call of {len(call_rows)} rows "
                    f"({sum(-(-x.tokens.numel() // p.page) for x in call_rows)} pages) does not fit "
                    f"the pools ({p.n_pages - 1} pages of {p.page}, {p.n_slots - 2} state slots); "
                    "split it (PrefixStateCache.fits) or raise learner_prefix_cache_gpu_gb"
                )
        from_v0 = [bool(r.start) and r.entry is not None and r.entry.origin in ("v0", "v0t")
                   for r in call_rows]
        self.last_from_v0 = from_v0
        meta = self._fill_meta(call_rows)
        _ACTIVE[0] = _Call(self, call_rows, meta)
        ok = False
        try:
            yield
            ok = True
        finally:
            _ACTIVE[0] = None
            if not ok:
                self._abort(call_rows)
        self.stats["v0_hits"] += sum(from_v0)
        for r in call_rows:
            self.stats["rows"] += 1
            self.stats["hits"] += int(r.start > 0)
            self.stats["tokens_skipped"] += r.start
            self.stats["tokens_forwarded"] += r.length
            self.stats["v0_captured"] += int(r.capture and bool(r.keep))
            self._commit(r)

    def _place(self, r: _Row, protect: set) -> bool:
        """Pages of positions [0, start + length) and slots of the new boundaries for row r.

        Ownership: a resident entry owns its pages and slots. A row continued from its entry
        (start > 0) reuses the entry's leading pages (positions < start stay; positions >= start
        are rewritten) and its slot at ``start``; new pages / slots come from the free lists.
        A keyed row forwarded from 0 replaces its old entry, which is released first.
        """
        p = self.pools
        e = r.entry
        if r.start == 0 and r.key is not None and r.key in self.entries:
            self.release([r.key])
        if e is not None and e.host is not None:
            need_p = len(next(iter(e.host["kv"].values()))[0]) if e.host["kv"] else 0
            if not self._take(need_p, len(e.host["states"]), protect):
                return False
            self._load(r.key)
        n_pg = -(-r.tokens.numel() // p.page)
        if n_pg > p.maxp:
            raise RuntimeError(f"prefix-state cache: row of {r.tokens.numel()} tokens exceeds max_len {self.max_len}")
        reuse = list(e.pages[:n_pg]) if e is not None else []
        new_b = [b for b in r.keep if b != r.start]
        if not self._take(n_pg - len(reuse), len(new_b), protect):
            return False
        r.new_pages = [p.free_pages.pop() for _ in range(n_pg - len(reuse))]
        r.pages = reuse + r.new_pages
        r.slot_in = e.states[r.start] if r.start else ZERO_SLOT
        r.keep_slots = {b: p.free_slots.pop() for b in new_b}
        return True

    def _abort(self, rows: list[_Row]) -> None:
        """A failed forward: its writes may have overwritten positions >= start; drop the rows."""
        p = self.pools
        for r in rows:
            p.free_pages.extend(r.new_pages)
            p.free_slots.extend(r.keep_slots.values())
            r.new_pages, r.keep_slots = [], {}
            if r.key is not None:
                self.release([r.key])  # frees the entry's own pages / slots (incl. reused ones)

    def _fill_meta(self, rows: list[_Row]) -> dict[str, torch.Tensor]:
        p = self.pools
        b = len(rows)
        meta = p.metadata(b)
        start = torch.tensor([r.start for r in rows], dtype=torch.long)
        length = torch.tensor([r.length for r in rows], dtype=torch.long)
        table = torch.full((b, p.maxp), TRASH_PAGE, dtype=torch.int32)
        slot_in = torch.tensor([r.slot_in for r in rows], dtype=torch.long)
        keep_slot = torch.full((b, p.K), TRASH_SLOT, dtype=torch.long)
        keep_off = torch.zeros((b, p.K), dtype=torch.long)
        for i, r in enumerate(rows):
            if r.pages:
                table[i, : len(r.pages)] = torch.tensor(r.pages, dtype=torch.int32)
            for j, (bd, s) in enumerate(sorted(r.keep_slots.items())):
                keep_slot[i, j] = s
                keep_off[i, j] = bd - r.start
        for name, t in (("start", start), ("length", length), ("page_table", table),
                        ("slot_in", slot_in), ("keep_slot", keep_slot), ("keep_off", keep_off)):
            meta[name].copy_(t, non_blocking=False)
        return meta

    def _commit(self, r: _Row) -> None:
        p = self.pools
        old = self.entries.pop(r.key, None) if r.key is not None else None
        if r.key is None or not r.keep:
            # Not cached: the pages / slots were only for this forward.
            p.free_pages.extend(r.new_pages)
            p.free_slots.extend(r.keep_slots.values())
            if old is not None:
                self.entries[r.key] = old
                self.release([r.key])
            return
        new_len = r.new_kv_len
        n_pg = -(-new_len // p.page)
        pages = r.pages[:n_pg]
        states: dict[int, int] = {}
        for b in r.keep:
            states[b] = r.slot_in if b == r.start else r.keep_slots[b]
        if old is not None:
            for b in r.retain:
                if b in old.states:
                    states[b] = old.states[b]
        states.pop(0, None)  # position 0: the zero state, never stored
        keep_pg, keep_sl = set(pages), set(states.values())
        p.free_pages.extend(pg for pg in r.pages if pg not in keep_pg)
        p.free_slots.extend(s for s in r.keep_slots.values() if s not in keep_sl)
        if old is not None:
            # The old entry's pages beyond the reused ones and its slots no longer kept.
            used = set(r.pages)
            p.free_pages.extend(pg for pg in old.pages if pg not in used)
            p.free_slots.extend(s for s in old.states.values() if s > TRASH_SLOT and s not in keep_sl)
        if r.capture and self.negctl:
            # Negative control: a state that is not this forward's (see ``negctl``).
            f = 1.0 + float(self.negctl)
            idx = torch.tensor(pages, dtype=torch.long, device=p.device)
            for ln in p.k:
                p.k[ln].index_copy_(0, idx, p.k[ln].index_select(0, idx) * f)
            for s in states.values():
                for ln in p.ssm:
                    p.ssm[ln][s].mul_(f)
        self.entries[r.key] = _Entry(
            tokens=r.tokens[:new_len].clone(), kv_len=new_len, pages=pages, states=states,
            origin="v0" if r.capture else "piece",
        )

    # ---- accounting ------------------------------------------------------------------
    def costs(self) -> dict[str, int]:
        """Bytes this rank stores per cached position (K/V) and per kept boundary (Mamba).

        The driver bounds a rank's cache with them (an entry of n positions and k
        boundaries costs n * kv_bytes_per_token + k * state_bytes_per_boundary).
        """
        from megatron.core.ssm.mamba_mixer import MambaMixer
        from megatron.core.transformer.attention import Attention

        kv = state = 0
        for chunk in _model_chunks(self._model):
            for m in chunk.modules():
                if isinstance(m, Attention):
                    elem = next(m.parameters()).element_size()
                    kv += 2 * m.num_query_groups_per_partition * m.hidden_size_per_attention_head * elem
                elif isinstance(m, MambaMixer):
                    w = m.cp.get_conv1d_weight()
                    state += w.shape[0] * (w.shape[-1] - 1) * w.element_size()
                    state += m.cp.nheads_local_tpcp * m.headdim * m.d_state * 4
        return {
            "kv_bytes_per_token": kv,
            "state_bytes_per_boundary": state,
            "chunk": self.chunk,
            "max_keep": self.max_keep,
        }

    def bytes(self) -> dict[str, int]:
        """Bytes of the cached entries: in the device pools (pages / slots in use) and on host."""
        p = self.pools
        gpu = host = 0
        for e in self.entries.values():
            if e.host is None and p is not None:
                gpu += len(e.pages) * sum(2 * t[0].numel() * t.element_size() for t in p.k.values())
                gpu += len(e.states) * sum(
                    p.conv[ln][0].numel() * p.conv[ln].element_size() + p.ssm[ln][0].numel() * 4
                    for ln in p.conv
                )
            elif e.host is not None:
                for k, v in e.host["kv"].values():
                    host += 2 * k.numel() * k.element_size()
                for per in e.host["states"].values():
                    host += sum(c.numel() * c.element_size() + x.numel() * 4 for c, x in per.values())
        return {"gpu": gpu, "host": host, "rows": len(self.entries), "pool": self.pool_bytes()}
