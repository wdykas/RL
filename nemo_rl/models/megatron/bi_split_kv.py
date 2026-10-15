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
"""Batch-invariant split-KV attention (``policy.megatron_cfg.bi_split_kv``).

Under ``batch_invariant_mode`` Megatron's inference attention and TE's FA4 training path call
FlashAttention-4 with ``num_splits=1``: one CTA per (row, KV head) streams the whole context, so
few-row long-context decode runs at ~60-115 GB/s (295 us/layer at 32k, 1.16 ms at 128k on GB200).

``install`` routes every such call (``num_splits == 1``) to FA4's anchored split-KV instead:

- ``seqlen_k_per_split=S`` (default 2048) with ``split_anchor="absolute"`` when the FA4 build has
  it: split s owns the absolute keys [s*S, (s+1)*S). A row's reduction order
  depends only on its own keys -- not on the batch, the query count, the tiling or the split count.
- ``num_splits=NS``: fixed for every call, so CUDA-graph grids are fixed; empty splits write
  lse=-inf, which the combine turns into an exact zero.
- the combine's per-row reduction order does not depend on the launched split count (each thread
  sums a fixed stride of splits; extra splits add exact zeros), so no further pinning is needed.

Learner forward (TE ``flash_attn_varlen_func_v4``) and engine prefill/decode (Megatron
``flash_attn4_varlen_func``) then compute bitwise-identical rows (verified on GB200 for
decode B=1..64, 1-3 tokens/request, paged/permuted cache, up to 32k; thundersync_work
research/agent_bi_splitkv.md). The backward is FA4's stock backward from (out, lse).

Requires an FA4 build whose ``flash_attn_varlen_func`` takes ``seqlen_k_per_split`` with absolute
split boundaries: upstream flash-attention main (causal; local windows need PR #3000), or the b19
overlay patched by thundersync_work/research/bi_splitkv/patch_fa4_b19.py.

Per-rank cost: fp32 partials NS x rows x heads x head_dim x 4 B per attention call (learner: rows =
packed micro-batch tokens; bound it with NS and the micro-batch size), extra partial write/read
traffic (learner attention fwd+bwd ~1.3-1.5x at 8k-32k, measured), no communication.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable

_STATE: dict[str, Any] = {"cfg": None, "patched": []}


DEFAULT_SEQLEN_K_PER_SPLIT = 2048  # chosen 10-14 from the GB200 decode sweep (bskv-upbench)


def _resolve(cfg: dict, max_seq_len: int | None = None) -> dict:
    s = int(cfg.get("seqlen_k_per_split") or DEFAULT_SEQLEN_K_PER_SPLIT)
    ns = cfg.get("num_splits")
    if ns is None:
        # Per call: ceil(max_seqlen_k / S) (>= 2). Bits do not depend on the split count, and a CUDA
        # graph captures the call's max_seqlen_k as a constant, so the grid stays fixed per graph.
        if max_seq_len and max_seq_len > 256 * s:
            raise ValueError(f"bi_split_kv: max context {max_seq_len} needs > 256 splits of {s} keys")
        return {"seqlen_k_per_split": s, "num_splits": None}
    ns = int(ns)
    if s <= 0 or s % 128:
        raise ValueError(f"bi_split_kv.seqlen_k_per_split={s} must be a positive multiple of 128")
    if not 2 <= ns <= 256:
        raise ValueError(f"bi_split_kv: need 2 <= num_splits={ns} <= 256")
    return {"seqlen_k_per_split": s, "num_splits": ns}


def wrap(fn: Callable, cfg: dict, max_seq_len: int | None = None) -> Callable:
    """Wrap an FA4 ``flash_attn_varlen_func``: BI calls (num_splits=1) become anchored split-KV."""
    if getattr(fn, "_ts_bi_split_kv", False):
        return fn
    params = inspect.signature(fn).parameters
    if "seqlen_k_per_split" not in params:
        raise RuntimeError(
            "bi_split_kv: this flash_attn.cute build lacks seqlen_k_per_split (build the overlay "
            "with research/bi_splitkv/patch_fa4_b19.py or use upstream flash-attention main)"
        )
    c = _resolve(cfg, max_seq_len)
    if "split_anchor" in params:
        # Upstream FA4 (flash-attention PR #3000): absolute split boundaries are opt-in. They equal the
        # relative ones without a sliding window, but only "absolute" keeps decode == prefill with one.
        c["split_anchor"] = "absolute"

    name = f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__name__', '?')}"
    seen = {"logged": False}

    @functools.wraps(fn)
    def anchored(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("num_splits", 1) == 1 and kwargs.get("seqlen_k_per_split") is None:
            kwargs.update(c)
            if kwargs["num_splits"] is None:
                max_k = kwargs.get("max_seqlen_k")
                if max_k is None:  # dense (bshd) path: keys are k.shape[1]
                    max_k = (args[1] if len(args) > 1 else kwargs["k"]).shape[1]
                kwargs["num_splits"] = max(2, -(-int(max_k) // c["seqlen_k_per_split"]))
            if not seen["logged"]:  # non-vacuity evidence: the anchored path actually ran
                seen["logged"] = True
                print(f"[bi_split_kv] first anchored call: {name} {c}", flush=True)
        return fn(*args, **kwargs)

    anchored._ts_bi_split_kv = True  # type: ignore[attr-defined]
    return anchored


def _refuse_bi(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        if kwargs.get("num_splits", 1) == 1:
            raise RuntimeError(
                "bi_split_kv: TE chose the dense FA4 path (bshd, no padding); use packed THD "
                "sequences so attention runs through flash_attn_varlen_func"
            )
        return fn(*args, **kwargs)

    guarded._ts_bi_split_kv = True  # type: ignore[attr-defined]
    return guarded


def install(megatron_cfg: dict, max_seq_len: int | None = None) -> None:
    """Patch the FA4 entry points used by Megatron inference and TE training.

    ``max_seq_len`` (the longest context any call sees) sets num_splits when the config leaves it unset.
    """
    cfg = megatron_cfg["bi_split_kv"]
    if not megatron_cfg.get("batch_invariant_mode"):
        raise ValueError("bi_split_kv requires policy.megatron_cfg.batch_invariant_mode=true")
    _resolve(cfg, max_seq_len)
    _STATE["cfg"] = cfg
    import flash_attn.cute as fa4_pkg
    import flash_attn.cute.interface as fa4_interface

    base = fa4_interface.flash_attn_varlen_func
    wrapped = wrap(base, cfg, max_seq_len)
    targets: list[tuple[Any, str]] = [(fa4_interface, "flash_attn_varlen_func"),
                                      (fa4_pkg, "flash_attn_varlen_func")]
    try:  # Megatron inference attention binds the function at import time
        from megatron.core.transformer import attention as mcore_attention

        if getattr(mcore_attention, "flash_attn4_varlen_func", None) is not None:
            targets.append((mcore_attention, "flash_attn4_varlen_func"))
    except ImportError:
        pass
    try:  # TE's FA4 training path
        from transformer_engine.pytorch.attention.dot_product_attention import backends as te_backends

        if getattr(te_backends, "flash_attn_varlen_func_v4", None) is not None:
            targets.append((te_backends, "flash_attn_varlen_func_v4"))
        dense = getattr(te_backends, "flash_attn_func_v4", None)
        if dense is not None and not getattr(dense, "_ts_bi_split_kv", False):
            # TE takes the dense FA4 path for unpacked (bshd, no padding) batches. Anchor it too when
            # the build supports it; otherwise refuse BI calls there (a silent num_splits=1 would use a
            # different reduction order than the engine).
            if "seqlen_k_per_split" in inspect.signature(dense).parameters:
                te_backends.flash_attn_func_v4 = wrap(dense, cfg, max_seq_len)
                _STATE["patched"].append(f"{te_backends.__name__}.flash_attn_func_v4")
            else:
                te_backends.flash_attn_func_v4 = _refuse_bi(dense)
                _STATE["patched"].append(f"{te_backends.__name__}.flash_attn_func_v4 (guard)")
    except ImportError:
        pass
    for mod, name in targets:
        if not getattr(getattr(mod, name), "_ts_bi_split_kv", False):
            setattr(mod, name, wrapped)
            _STATE["patched"].append(f"{mod.__name__}.{name}")
    print(f"[bi_split_kv] anchored split-KV: {_resolve(cfg, max_seq_len)} -> {_STATE['patched']}", flush=True)
