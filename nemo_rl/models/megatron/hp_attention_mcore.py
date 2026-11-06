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
"""``precision_raise`` site ``attn_core_fp32``: near-fp32 attention in Megatron learner and engine.

``install`` (after ``fp16_gemm.install``, before the models are built, in every learner and
engine process) changes the attention of every ``SelfAttention`` layer:

- q / k / v stay fp32 after ``linear_qkv`` (``fp16_gemm._cast_qkv`` is bypassed for
  attention modules), so ``linear_proj`` (tagged hi/lo by ``precision_raise``) receives the
  fp32 attention output;
- learner (``TEDotProductAttention`` / ``DotProductAttention``, THD packed or sbhd): the hp
  kernel (``hp_attention.py``: hi/lo q, K, V, P on fp16 tensor cores, fp32 online softmax,
  fixed 64-key blocks and 1024-key absolute chunks) on a per-micro-batch paged copy of the
  layer's K/V; backward = FlashAttention-2's fp16 varlen backward at the hi parts with the
  hp output / log-sum-exp (``HPAttention``);
- engine (Megatron dynamic inference): the KV cache holds fp16 (kh, vh, kl, vl) planes plus
  fp32 per-row inverse scales; ``append_key_value_cache`` writes them with the fused split
  kernel; ``flash_decode_and_prefill`` runs the same kernel on the paged cache (decode-only
  steps: 16-row decode tiles, bitwise equal to the learner's 64-row tiles; split-KV for few
  rows; both choices bitwise neutral). Tiles are built on the device from the step's
  metadata (``hp_attention.static_tiles``): no host syncs, CUDA-graph safe.

A row's bits depend only on its own q and the keys/values at positions <= its own, so engine
prefill, decode, chunked prefill, speculative verify and the learner's packed forward agree
bitwise whatever the batch, page size or page order.

Not built: RoPE models (Nemotron-H has no position embedding; a bitwise fp32 RoPE shared by
both sides is required first), context parallelism, sliding windows, softcap / sink softmax
offsets, MLA. ``install`` refuses them at the first forward.

Per-rank cost: engine KV bytes 2.016x of the fp16 cache in every attention layer (with
``attn_core_kv`` the block pool is sized for the 4 planes inside ``buffer_size_gb``, so the same
budget holds half the KV tokens); decode attention ~1.6-1.9x of FA at large batch (bandwidth on
2x bytes); learner attention forward ~2.7x the same kernel at fp16 (Triton prototype), backward
unchanged (FA2 fp16). No communication added.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import torch

from nemo_rl.models.megatron import hp_attention as hp

_STATE: dict[str, Any] = {"enabled": False, "cfg": hp.HPAttnConfig(), "orig": {}, "ctx": None,
                          "split_rows": 32, "learner_page": 64}


def enabled() -> bool:
    return bool(_STATE["enabled"])


def _scale(config: Any, head_dim: int) -> float:
    """The one softmax scale both sides use (python float, identical bits)."""
    s = getattr(config, "softmax_scale", None)
    return float(s) if s is not None else 1.0 / math.sqrt(head_dim)


def _check_config(config: Any) -> None:
    bad = []
    pe = getattr(config, "position_embedding_type", "none")
    if pe not in (None, "none"):
        bad.append(f"position_embedding_type={pe}")
    if (getattr(config, "context_parallel_size", 1) or 1) > 1:
        bad.append("context_parallel_size>1")
    if getattr(config, "window_size", None) is not None:
        bad.append("window_size")
    if getattr(config, "attn_logit_softcapping", None) is not None:
        bad.append("attn_logit_softcapping")
    if getattr(config, "softmax_type", "vanilla") not in (None, "vanilla"):
        bad.append(f"softmax_type={config.softmax_type}")
    if getattr(config, "multi_latent_attention", False):
        bad.append("multi_latent_attention")
    if bad:
        raise NotImplementedError("precision_raise attn_core_fp32: not built for " + ", ".join(bad))


# ---------------------------------------------------------------------------
# Learner: packed causal attention with the hp forward.
# ---------------------------------------------------------------------------


def _temp_cache(k, v, cu_k, cfg: hp.HPAttnConfig):
    """A per-call paged hp cache holding packed k/v [Tk, Hkv, D] fp32; (cache, block_table, kv_lens)."""
    nk = k.shape[0]
    ps = _STATE["learner_page"]
    slots, bt, lens, num_pages = hp.static_prefill_layout(cu_k, nk, ps)
    cache = hp.HPKVCache.__new__(hp.HPKVCache)
    shape = (num_pages, ps, k.shape[1], k.shape[2])
    f16 = dict(dtype=torch.float16, device=k.device)
    cache.cfg, cache.page_size = cfg, ps
    cache.kh, cache.vh = torch.empty(shape, **f16), torch.empty(shape, **f16)
    cache.kl = torch.empty(shape, **f16) if cfg.k_lo else None
    cache.vl = torch.empty(shape, **f16) if cfg.v_lo else None
    if cfg.kv_fp16:
        cache.k_inv = cache.v_inv = None
    else:
        cache.k_inv = torch.empty(shape[:3], dtype=torch.float32, device=k.device)
        cache.v_inv = torch.empty(shape[:3], dtype=torch.float32, device=k.device)
    cache.write(slots, k, v)
    return cache, bt, lens


def prefix_attention(q, k, v, cu_q, cu_k, max_kv_len: int, scale: float) -> torch.Tensor:
    """hp attention of packed queries that are the LAST ``cu_q`` rows of each sequence against
    packed keys ``cu_k`` (the learner prefix-state cache: cached + new keys). q [Tq, Hq, D],
    k / v [Tk, Hkv, D]; returns fp32 [Tq, Hq, D]. Same kernel and absolute-position blocks as
    the full forward and the engine, so each row's bits match theirs. Forward only."""
    if torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError("attn_core prefix-state forward: no backward (verification only)")
    cfg = _STATE["cfg"]
    q, k, v = q.float(), k.float(), v.float()
    cache, bt, lens = _temp_cache(k, v, cu_k, cfg)
    n = q.shape[0]
    g = q.shape[1] // k.shape[1]
    tile_seq, tile_t0, tile_q, q_seq, q_pos, valid = hp.static_tiles(cu_q, lens, n, cfg.block_m // g)
    qh, ql, q_inv = hp.split_q(q, cfg)
    out, _ = hp.hp_attention_cached(
        qh, ql, q_inv, q_seq, q_pos.to(torch.int32), cache, bt, lens, scale,
        max_kv_len=int(max_kv_len), tiles=(tile_seq, tile_t0, tile_q),
    )  # fmt: skip
    return out


def _learner_forward(q, k, v, cu, cfg: hp.HPAttnConfig, scale: float):
    """hp attention of packed q [T, Hq, D], k/v [T, Hkv, D] fp32 (no host sync)."""
    n = q.shape[0]
    ps = _STATE["learner_page"]
    slots, bt, lens, num_pages = hp.static_prefill_layout(cu, n, ps)
    cache = hp.HPKVCache.__new__(hp.HPKVCache)
    shape = (num_pages, ps, k.shape[1], k.shape[2])
    f16 = dict(dtype=torch.float16, device=q.device)
    cache.cfg, cache.page_size = cfg, ps
    cache.kh, cache.vh = torch.empty(shape, **f16), torch.empty(shape, **f16)
    cache.kl = torch.empty(shape, **f16) if cfg.k_lo else None
    cache.vl = torch.empty(shape, **f16) if cfg.v_lo else None
    if cfg.kv_fp16:
        cache.k_inv = cache.v_inv = None
    else:
        cache.k_inv = torch.empty(shape[:3], dtype=torch.float32, device=q.device)
        cache.v_inv = torch.empty(shape[:3], dtype=torch.float32, device=q.device)
    cache.write(slots, k, v)
    qh, ql, q_inv = hp.split_q(q, cfg)
    g = q.shape[1] // k.shape[1]
    tiles = hp.static_tiles(cu, lens, n, cfg.block_m // g)
    out, lse = hp.hp_attention_cached(
        qh, ql, q_inv, None, None, cache, bt, lens, scale, tiles=tiles[:3]
    )
    return out, lse


class _HPAttn(torch.autograd.Function):
    """Forward: hp kernel. Backward: FA2 fp16 varlen backward at the hi parts (hp O, lse)."""

    @staticmethod
    def forward(ctx, q, k, v, cu, max_seqlen, scale):  # noqa: D102
        out, lse = _learner_forward(q, k, v, cu, _STATE["cfg"], scale)
        ctx.save_for_backward(q, k, v, out, lse, cu)
        ctx.max_seqlen = max_seqlen
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):  # noqa: D102
        q, k, v, out, lse, cu = ctx.saved_tensors
        dq, dk, dv = _fa2_backward(q, k, v, out, lse, cu, ctx.max_seqlen, ctx.scale, dout)
        return dq, dk, dv, None, None, None


def _fa2_backward(q, k, v, out, lse, cu, max_seqlen, scale, dout):
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward

    g = dout.float()
    amax = g.abs().amax().clamp_min(torch.finfo(torch.float32).tiny)
    sc = torch.exp2(torch.floor(-torch.log2(amax)).clamp(-60, 60))
    q16, k16, v16 = (t.to(torch.float16) for t in (q, k, v))
    do16 = (g * sc).to(torch.float16)
    o16 = out.to(torch.float16)
    dq, dk, dv = torch.empty_like(q16), torch.empty_like(k16), torch.empty_like(v16)
    cu32 = cu.to(torch.int32)
    _flash_attn_varlen_backward(
        do16, q16, k16, v16, o16, lse.t().contiguous(), dq, dk, dv,
        cu32, cu32, max_seqlen, max_seqlen, 0.0, scale, True,
        -1, -1, 0.0, None, False, None,
    )  # fmt: skip
    inv = 1.0 / sc
    return dq.float() * inv, dk.float() * inv, dv.float() * inv


def learner_attention(query, key, value, config, packed_seq_params=None):
    """Core attention of the training model: fp32 q/k/v -> fp32 output (TE's output layout)."""
    _check_config(config)
    d = query.shape[-1]
    scale = _scale(config, d)
    thd = packed_seq_params is not None and getattr(packed_seq_params, "qkv_format", None) == "thd"
    if thd:
        if getattr(packed_seq_params, "cp_group", None) is not None:
            raise NotImplementedError("attn_core_fp32: context parallelism not built")
        cu = packed_seq_params.cu_seqlens_q_padded
        ms = packed_seq_params.max_seqlen_q
        if cu is None:
            cu = packed_seq_params.cu_seqlens_q
        else:  # padded segments may exceed max_seqlen_q: bound FA2's backward grid by T
            ms = None
        ms = int(ms) if isinstance(ms, int) else query.shape[0]
        q, k, v = (t.reshape(t.shape[0], t.shape[-2], d) for t in (query, key, value))
        out = _HPAttn.apply(q.float(), k.float(), v.float(), cu, ms, scale)
        return out.reshape(out.shape[0], -1)
    # sbhd: [s, b, h, d] -> b sequences of s tokens
    s, b = query.shape[0], query.shape[1]

    def pack(t):
        return t.transpose(0, 1).reshape(b * s, t.shape[2], d).float()

    cu = torch.arange(b + 1, device=query.device, dtype=torch.int32) * s
    out = _HPAttn.apply(pack(query), pack(key), pack(value), cu, s, scale)
    return out.view(b, s, -1).transpose(0, 1).contiguous()


# ---------------------------------------------------------------------------
# Engine: hp paged KV cache + attention in Megatron dynamic inference.
# ---------------------------------------------------------------------------


class _LayerCache:
    """hp cache view of one attention layer of the engine buffers (hp_attention_cached reads it)."""

    __slots__ = ("cfg", "page_size", "kh", "vh", "kl", "vl", "k_inv", "v_inv")


def _layer_cache(ctx, layer: int) -> _LayerCache:
    c = _LayerCache()
    buf, inv = ctx.memory_buffer, ctx._hp_kv_inv
    cfg = _STATE["cfg"]
    c.cfg, c.page_size = cfg, ctx.block_size_tokens
    c.kh, c.vh = buf[0, layer], buf[1, layer]
    c.kl = buf[2, layer] if cfg.k_lo else None
    c.vl = buf[3, layer] if cfg.v_lo else None
    c.k_inv, c.v_inv = (None, None) if inv is None else (inv[0, layer], inv[1, layer])
    return c


def _allocate_memory_buffer(self):
    if getattr(self, "cache_mla_latent", False):
        raise NotImplementedError("attn_core_fp32: MLA latent cache not built")
    if self.block_size_tokens % _STATE["cfg"].block_n:
        raise ValueError(
            f"attn_core_fp32 needs block_size_tokens % {_STATE['cfg'].block_n} == 0 "
            f"(got {self.block_size_tokens})"
        )
    dev = torch.cuda.current_device()
    shape = (self.num_attention_layers, self.kv_block_allocator.pool_size, self.block_size_tokens,
             self.num_attention_heads_per_partition, self.hidden_size_per_attention_head)
    cfg = _STATE["cfg"]
    # (kh, vh[, kl, vl]): [0] / [1] keep MCore's key / value meaning (hi parts). kv_fp16
    # (attn_core_qp alone): exactly MCore's fp16 cache, no lo planes, no scales.
    planes = 2 if cfg.kv_fp16 else 4
    self.memory_buffer = torch.zeros((planes, *shape), dtype=torch.float16, device=dev)
    self._hp_kv_inv = (
        None if cfg.kv_fp16 else torch.ones((2, *shape[:4]), dtype=torch.float32, device=dev)
    )
    from megatron.core.inference.contexts.dynamic_context import KVCacheManagementMode

    if (
        self.kv_cache_management_mode == KVCacheManagementMode.OFFLOAD
        and not self._uses_torch_memory_saver
    ):
        for name in ("memory_buffer",) + (() if cfg.kv_fp16 else ("_hp_kv_inv",)):
            self._offloadable_tensor_names.add(name)
            self._offloadable_cpu_backups[name] = torch.empty_like(
                getattr(self, name), device="cpu"
            ).pin_memory()


def _append_key_value_cache(self, layer_number: int, key, value) -> None:
    if key.dtype != torch.float32:
        return _STATE["orig"]["append"](self, layer_number, key, value)
    if self.mtp_metadata.forward_active:
        layer = self.mtp_kv_layer_slot
    else:
        layer = self.layer_map[layer_number - 1]
    _STATE["ctx"] = self
    n = self.padded_active_token_count
    bs = self.block_size_tokens
    slots = (
        self.gpu_view.token_to_block_idx[:n].long() * bs
        + self.gpu_view.token_to_local_position_within_kv_block[:n].long()
    )
    _write_layer(self, layer, slots, key[:n], value[:n])


def _write_layer(ctx, layer: int, slots, key, value) -> None:
    """Write fp32 key / value rows [n, ...] at flat cache slots of one attention layer."""
    cfg = _STATE["cfg"]
    c = _layer_cache(ctx, layer)
    hkv, d = c.kh.shape[2], c.kh.shape[3]
    n = key.shape[0]
    k = key.reshape(n, hkv, d)
    v = value.reshape(n, hkv, d)
    if cfg.kv_fp16:
        hp.write_fp16_kv(c.kh, c.vh, slots, k, v)
        return
    flat = lambda t: t.view(-1, hkv, d)  # noqa: E731
    hp._split_into(k, cfg.qk_emin, cfg.qk_emax, flat(c.kh), flat(c.kl) if cfg.k_lo else None,
                   c.k_inv.view(-1, hkv), slots)
    hp._split_into(v, cfg.v_emin, cfg.v_emax, flat(c.vh), flat(c.vl) if cfg.v_lo else None,
                   c.v_inv.view(-1, hkv), slots)


def engine_attention(module, q, k, cu_seqlens_q, seqlens_k, block_table, is_decode_only,
                     max_seqlen_k) -> torch.Tensor:
    """hp attention for the engine's step (any mix of prefill / decode rows); [T, 1, Hq, D] fp32."""
    _check_config(module.config)
    ctx = _STATE["ctx"]
    buf = ctx.memory_buffer
    layer = (k.data_ptr() - buf.data_ptr()) // (buf.stride(1) * buf.element_size())
    cache = _layer_cache(ctx, int(layer))
    assert cache.kh.data_ptr() == k.data_ptr(), "attn_core_fp32: KV view does not match the hp cache"
    cfg = _STATE["cfg"]
    hq, d = q.shape[-2], q.shape[-1]
    n = q.numel() // (hq * d)
    q3 = q.reshape(n, hq, d).float()
    g = hq // cache.kh.shape[2]
    decode_tiles = bool(is_decode_only)
    bm = cfg.decode_block_m if decode_tiles else cfg.block_m
    tile_seq, tile_t0, tile_q, q_seq, q_pos, valid = hp.static_tiles(
        cu_seqlens_q, seqlens_k, n, bm // g
    )
    split = decode_tiles and n <= _STATE["split_rows"]
    qh, ql, q_inv = hp.split_q(q3, cfg)
    out, _ = hp.hp_attention_cached(
        qh, ql, q_inv, q_seq, q_pos.to(torch.int32), cache, block_table, seqlens_k,
        _scale(module.config, d), split=split, max_kv_len=int(max_seqlen_k),
        tiles=(tile_seq, tile_t0, tile_q), decode_tiles=decode_tiles,
    )  # fmt: skip
    out = torch.where(valid[:, None, None], out, torch.zeros((), dtype=out.dtype, device=out.device))
    return out.unsqueeze(1)


# ---------------------------------------------------------------------------
# install
# ---------------------------------------------------------------------------


def config_for(qp: bool, kv: bool) -> hp.HPAttnConfig:
    """hp numerics of the two attention-core knobs.

    qp (``attn_core_qp``): q and P as hi/lo (O fp32 -> hi/lo linear_proj); without kv, K and V are
    ``fp16(k)``, ``fp16(v)`` exactly as the fp16 recipe rounds them (engine cache = today's fp16
    cache). kv (``attn_core_kv``): K / V as row-prescaled hi/lo planes in the cache (2.016x KV
    bytes); without qp, q and P are single fp16 terms (q row-prescaled, same rounding as fp16(q)
    in fp16's normal range).
    """
    if not (qp or kv):
        raise ValueError("attn core: need qp and/or kv")
    if qp and not kv:
        return hp.HPAttnConfig(k_lo=False, v_lo=False, kv_fp16=True)
    if kv and not qp:
        return hp.HPAttnConfig(q_lo=False, p_lo=False)
    return hp.HPAttnConfig()


def install(megatron_cfg: Optional[dict] = None, qp: bool = True, kv: bool = True,
            cfg: Optional[hp.HPAttnConfig] = None) -> None:
    """Install the attention-core raise in this process (idempotent per config).

    ``qp`` / ``kv``: precision_raise sites ``attn_core_qp`` / ``attn_core_kv`` (``attn_core_fp32`` =
    both); ``cfg`` overrides the numerics (benchmarks / tests).
    """
    cfg = cfg if cfg is not None else config_for(qp, kv)
    if _STATE["enabled"]:
        if cfg != _STATE["cfg"]:
            raise RuntimeError("hp_attention_mcore.install called twice with different numerics")
        return
    _STATE["cfg"] = cfg
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    from megatron.core.inference.contexts.dynamic_context import DynamicInferenceContext
    from megatron.core.transformer.attention import Attention, SelfAttention
    from megatron.core.transformer.dot_product_attention import DotProductAttention

    from nemo_rl.models.megatron import fp16_gemm

    # 1. q / k / v stay fp32 (fp16_gemm's patched get_query_key_value_tensors looks the cast
    # up as a module global at call time).
    orig_cast = fp16_gemm._cast_qkv
    _STATE["orig"]["cast"] = orig_cast

    def _cast_qkv(outputs, box, site=None):
        if isinstance(site, SelfAttention):
            _check_config(site.config)
            return outputs
        return orig_cast(outputs, box, site)

    fp16_gemm._cast_qkv = _cast_qkv

    # 2. learner core attention
    for cls in (TEDotProductAttention, DotProductAttention):
        orig = cls.forward
        _STATE["orig"][cls] = orig

        def forward(self, query, key, value, attention_mask, *args, _orig=orig, **kwargs):
            if query.dtype != torch.float32:
                return _orig(self, query, key, value, attention_mask, *args, **kwargs)
            psp = kwargs.get("packed_seq_params")
            if psp is None and len(args) >= 3:
                psp = args[2]
            mt = kwargs.get("attn_mask_type", args[0] if args else None)
            if mt is not None and getattr(mt, "name", str(mt)) not in ("causal", "padding_causal"):
                raise NotImplementedError(f"attn_core_fp32: attn_mask_type {mt} not built (causal only)")
            return learner_attention(query, key, value, self.config, psp)

        cls.forward = forward

    # 3. engine cache + attention
    _STATE["orig"]["alloc"] = DynamicInferenceContext._allocate_memory_buffer
    _STATE["orig"]["append"] = DynamicInferenceContext.append_key_value_cache
    DynamicInferenceContext._allocate_memory_buffer = _allocate_memory_buffer
    if not cfg.kv_fp16:
        # Budget the hi/lo planes: MCore sizes the block pool from params_dtype.itemsize x (K, V);
        # four fp16 planes = an fp32-sized block, so buffer_size_gb keeps meaning total bytes
        # (the fp32 row scales, 1/64 of a plane, stay outside the budget).
        import copy

        orig_init = DynamicInferenceContext.__init__
        _STATE["orig"]["ctx_init"] = orig_init

        def __init__(self, model_config, inference_config, *a, **kw):
            mc = copy.copy(model_config)
            mc.params_dtype = torch.float32
            orig_init(self, mc, inference_config, *a, **kw)
            self.params_dtype = model_config.params_dtype

        DynamicInferenceContext.__init__ = __init__
    DynamicInferenceContext.append_key_value_cache = _append_key_value_cache
    orig_fdp = Attention.flash_decode_and_prefill
    _STATE["orig"]["fdp"] = orig_fdp

    def flash_decode_and_prefill(self, q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q,
                                 cu_seqlens_k, seqlens_k, block_table, is_decode_only,
                                 softmax_offset=None):
        if q.dtype != torch.float32:
            return orig_fdp(self, q, k, v, max_seqlen_q, max_seqlen_k, cu_seqlens_q,
                            cu_seqlens_k, seqlens_k, block_table, is_decode_only,
                            softmax_offset=softmax_offset)
        if softmax_offset is not None:
            raise NotImplementedError("attn_core_fp32: softmax offset (sink) not built")
        return engine_attention(self, q, k, cu_seqlens_q, seqlens_k, block_table,
                                is_decode_only, max_seqlen_k)

    Attention.flash_decode_and_prefill = flash_decode_and_prefill
    _STATE["enabled"] = True
    print(f"[hp_attention_mcore] attention core raised (qp={qp}, kv={kv}): fp32 q/k/v in, hp attention "
          f"(learner + engine), KV cache {'fp16' if cfg.kv_fp16 else 'hi/lo planes'}", flush=True)
