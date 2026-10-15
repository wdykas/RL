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
of the training model at a Mamba chunk boundary s (a multiple of ``chunk_size`` counted from
position 0):

- every attention layer's K and V of positions [0, s) (as fed to core attention);
- every Mamba layer's conv inputs of positions [s - d_conv + 1, s) and fp32 SSM state at s;

and forwards only positions [s, n) of the next row (n its length; at most chunk_size - 1
recomputed positions before the piece). Logits of those positions are bitwise the full
forward's under batch invariance:

- attention: FA4 over the new queries and the cached + new keys, through the same entry
  point TE's training path uses (``flash_attn_varlen_func_v4``: anchored split-KV when
  ``bi_split_kv`` is installed, num_splits=1 otherwise). A query row's key blocks are
  aligned to absolute position 0 in both, so its reduction order does not change;
- Mamba: ``mamba_ssd_parity.ssm_prefix_forward`` (conv over the cached inputs + new ones;
  chunked scan restarted at the boundary from its fp32 state, the identity the engine's
  batch-invariant prefill relies on);
- everything else is per token (row-invariant GEMMs, norms, dropless MoE with sticky
  anchors applied per position).

Membership is decided by the caller, identically on every rank of a replica (TP x PP): a
row is cached only under the key the caller passes, and dropped only by ``release`` or a
weight change (``set_weights_version``). The cache never drops rows on its own, so all
ranks of a replica pick the same start positions (the forwards' shapes agree). Residency
is local: entries beyond ``gpu_bytes`` move to pinned host memory (LRU) and come back when
used; that changes neither shapes nor bits.

Validation of a hit is by content: the cached tokens' longest common prefix with the new
row bounds the usable boundary, so a rejected piece (or any change) falls back to an
earlier boundary instead of reusing stale state.

No new communication: each rank keeps its own layers' (pipeline stage) and heads' (tensor
parallel) state. Not supported: positional embeddings that depend on absolute positions
(RoPE: offsets not built yet), context parallelism, packed sequences, autograd.
"""

from __future__ import annotations

import contextlib
import functools
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

import torch

_ACTIVE: list[Optional["_Call"]] = [None]


@dataclass
class _Entry:
    """One row's cached state on this rank."""

    tokens: torch.Tensor  # cpu int64, the row's tokens [0, kv_len)
    kv_len: int  # K/V cached for positions [0, kv_len)
    kv: dict[int, tuple[torch.Tensor, torch.Tensor]]  # layer -> (K, V) [cap, heads, dim]
    # boundary position -> layer -> (conv inputs [conv_dim, d_conv - 1], ssm fp32 state)
    states: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]]
    on_gpu: bool = True

    def nbytes(self) -> int:
        n = sum(k.numel() * k.element_size() * 2 for k, _ in self.kv.values())
        for per_layer in self.states.values():
            n += sum(c.numel() * c.element_size() + s.numel() * s.element_size()
                     for c, s in per_layer.values())
        return n

    def to(self, device: str) -> None:
        pin = device == "cpu"

        def move(t: torch.Tensor) -> torch.Tensor:
            if pin:
                out = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                out.copy_(t, non_blocking=True)
                return out
            return t.to(device, non_blocking=True)

        self.kv = {ln: (move(k), move(v)) for ln, (k, v) in self.kv.items()}
        self.states = {
            s: {ln: (move(c), move(x)) for ln, (c, x) in per_layer.items()}
            for s, per_layer in self.states.items()
        }
        self.on_gpu = not pin
        if pin:
            torch.cuda.current_stream().synchronize()


@dataclass
class _Row:
    """One batch row of a prefix-state forward."""

    key: Optional[str]
    tokens: torch.Tensor  # cpu int64, the whole row [0, n)
    start: int  # first forwarded position (a chunk boundary)
    entry: Optional[_Entry]  # cached state used for [0, start) (None: start == 0)
    keep: list[int]  # boundaries to cache after the forward (ascending; [] if key is None)
    pending_kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    pending_states: dict[int, dict[int, tuple[torch.Tensor, torch.Tensor]]] = field(
        default_factory=dict
    )

    @property
    def length(self) -> int:
        return self.tokens.numel() - self.start

    @property
    def new_kv_len(self) -> int:
        return self.keep[-1] if self.keep else 0


class _Call:
    """The rows of one forward; consulted by the hooked attention and Mamba layers."""

    def __init__(self, cache: "PrefixStateCache", rows: list[_Row]):
        self.cache = cache
        self.rows = rows

    def attention(self, mod, layer: int, q, k, v) -> torch.Tensor:
        """Core attention (sbhd in, [s, b, heads * dim] out) over cached + new keys."""
        from transformer_engine.pytorch.attention.dot_product_attention import (
            backends as te_backends,
        )
        from transformer_engine.pytorch.attention.dot_product_attention import (
            utils as dpa_utils,
        )

        fa4 = te_backends.flash_attn_varlen_func_v4
        if fa4 is None:
            raise RuntimeError("learner prefix-state cache needs FlashAttention-4")
        qs, ks, vs = [], [], []
        cu_q, cu_k = [0], [0]
        for b, r in enumerate(self.rows):
            n = r.length
            qs.append(q[:n, b])
            kn, vn = k[:n, b], v[:n, b]
            if r.start:
                kc, vc = r.entry.kv[layer]
                ks += [kc[: r.start], kn]
                vs += [vc[: r.start], vn]
            else:
                ks.append(kn)
                vs.append(vn)
            if r.keep and r.new_kv_len > r.start:
                m = r.new_kv_len - r.start
                r.pending_kv[layer] = (kn[:m].clone(), vn[:m].clone())
            cu_q.append(cu_q[-1] + n)
            cu_k.append(cu_k[-1] + r.start + n)
        dev = q.device
        out = fa4(
            torch.cat(qs),
            torch.cat(ks),
            torch.cat(vs),
            cu_seqlens_q=torch.tensor(cu_q, dtype=torch.int32, device=dev),
            cu_seqlens_k=torch.tensor(cu_k, dtype=torch.int32, device=dev),
            max_seqlen_q=max(r.length for r in self.rows),
            max_seqlen_k=max(r.start + r.length for r in self.rows),
            softmax_scale=mod.softmax_scale,
            causal=True,
            # As TE passes it for a causal training forward.
            window_size=dpa_utils.check_set_window_size("causal", mod.window_size),
            num_splits=1 if mod.num_splits is None else mod.num_splits,
            deterministic=mod.deterministic,
        )
        if isinstance(out, (tuple, list)):
            out = out[0]
        s, nb = q.shape[:2]
        res = q.new_zeros(s, nb, q.shape[2] * v.shape[3])
        for b, r in enumerate(self.rows):
            res[: r.length, b] = out[cu_q[b] : cu_q[b + 1]].reshape(r.length, -1)
        return res

    def mamba(self, mod, layer: int, zxBCdt) -> torch.Tensor:
        """``_ssm_training`` of the rows' segments, continued from their cached states."""
        from nemo_rl.models.megatron.mamba_ssd_parity import ssm_prefix_forward

        conv_w = mod.cp.get_conv1d_weight()  # [conv_dim, 1, d_conv]
        conv_dim, width = conv_w.shape[0], conv_w.shape[-1] - 1
        nb = zxBCdt.shape[1]
        heads, hdim = mod.cp.nheads_local_tpcp, mod.headdim
        conv_init = zxBCdt.new_zeros(nb, conv_dim, width)
        ssm_init = torch.zeros(
            nb, heads, hdim, mod.d_state, dtype=torch.float32, device=zxBCdt.device
        )
        for b, r in enumerate(self.rows):
            if r.start:
                c, s = r.entry.states[r.start][layer]
                conv_init[b] = c
                ssm_init[b] = s
        y, xbc_ext, chunk_states = ssm_prefix_forward(mod, zxBCdt, conv_init, ssm_init)
        chunk = mod.chunk_size
        for b, r in enumerate(self.rows):
            for s in r.keep:
                if s == r.start:
                    continue
                j = s - r.start
                r.pending_states.setdefault(s, {})[layer] = (
                    xbc_ext[b, :, j : j + width].clone(),
                    chunk_states[b, j // chunk - 1].clone(),
                )
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


class PrefixStateCache:
    """Per-rank store of rows' boundary states and the hooks that use them.

    ``model``: the training model (chunks); hooks are installed on its local
    attention and Mamba layers (they are no-ops outside ``forward``).
    ``gpu_bytes``: device residency budget; colder entries live in pinned host
    memory. The caller bounds the total (``release``).
    """

    def __init__(self, model, gpu_bytes: int, max_keep: int = 4):
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
        self.entries: "OrderedDict[str, _Entry]" = OrderedDict()  # LRU order
        self.weights_version: Any = None
        self.stats = {"hits": 0, "rows": 0, "tokens_skipped": 0, "tokens_forwarded": 0}
        self.attention_layers: list[int] = []
        self.mamba_layers: list[int] = []
        chunks: set[int] = set()
        for chunk in _model_chunks(model):
            pos = getattr(chunk, "position_embedding_type", None)
            inner = getattr(chunk, "module", None)
            pos = pos if pos is not None else getattr(inner, "position_embedding_type", None)
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
                elif isinstance(m, MambaMixer):
                    ln = int(m.layer_number)
                    m._ssm_training = functools.partial(_hooked_ssm, m, ln)
                    self.mamba_layers.append(ln)
                    chunks.add(int(m.chunk_size))
        if len(chunks) > 1:
            raise ValueError(f"Mamba layers with different chunk sizes {chunks}")
        # Boundaries: Mamba chunk starts. A stage without Mamba layers still uses the
        # model's chunk size (all stages must agree on start positions).
        self.chunk = chunks.pop() if chunks else int(getattr(config, "mamba_chunk_size", 128) or 128)

    # ---- membership (caller-driven) -------------------------------------------------
    def set_weights_version(self, version: Any) -> None:
        """Drop everything when the weights changed (cached states are of the old ones)."""
        if version != self.weights_version:
            self.entries.clear()
            self.weights_version = version

    def release(self, keys) -> None:
        for k in keys or ():
            self.entries.pop(k, None)

    def clear(self) -> None:
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

    def keep(self, start: int, n: int) -> list[int]:
        """Boundaries to keep after forwarding [start, n): start and those in (start, n - 1].

        A later row that extends [0, prompt) with prompt <= n (a whole or partial
        acceptance of this piece) uses the largest of them <= prompt - 1, so at most
        chunk - 1 positions before its piece are recomputed. Boundary n itself is
        never usable (position n - 1's logits are always needed again). Long pieces
        keep at most ``max_keep`` boundaries (start, the last one, evenly spaced
        between): a rejection inside such a piece then recomputes up to the spacing.
        """
        out = list(range(start, n, self.chunk))
        if len(out) > self.max_keep:
            k = self.max_keep - 1
            out = sorted({out[round(j * (len(out) - 1) / k)] for j in range(k + 1)})
        return out if out[-1] > 0 else []

    # ---- forward ---------------------------------------------------------------------
    @contextlib.contextmanager
    def forward(
        self, keys: list[Optional[str]], rows: list[torch.Tensor], starts: list[int]
    ) -> Iterator[None]:
        """Run the model's forward of rows[i][starts[i]:] (left-aligned, padded) inside.

        Hooked layers continue each row from its cached state at starts[i] and record
        the state at the boundaries ``keep(starts[i], n_i)``; on success the rows'
        entries are replaced by the new state. Rows with key None are forwarded from
        their start without caching.
        """
        if _ACTIVE[0] is not None:
            raise RuntimeError("nested prefix-state forwards")
        if torch.is_grad_enabled():
            raise RuntimeError("prefix-state forwards run without autograd")
        if len(set(k for k in keys if k is not None)) != sum(k is not None for k in keys):
            raise ValueError("a key appears twice in one prefix-state forward")
        call_rows = []
        for key, row, s in zip(keys, rows, starts):
            entry = self.entries.get(key) if key is not None else None
            if s:
                if entry is None or s not in entry.states:
                    raise KeyError(f"no cached state of {key!r} at {s}")
                if not entry.on_gpu:
                    entry.to("cuda")
                    self.stats["host_loads"] = self.stats.get("host_loads", 0) + 1
            tokens = row.detach().to("cpu", torch.long)
            keep = self.keep(s, tokens.numel()) if key is not None else []
            call_rows.append(_Row(key, tokens, s, entry if s else None, keep))
        _ACTIVE[0] = _Call(self, call_rows)
        try:
            yield
        finally:
            _ACTIVE[0] = None
        for r in call_rows:
            self.stats["rows"] += 1
            self.stats["hits"] += int(r.start > 0)
            self.stats["tokens_skipped"] += r.start
            self.stats["tokens_forwarded"] += r.length
            if r.key is not None:
                self._commit(r)
        self._fit_gpu({r.key for r in call_rows})

    def _commit(self, r: _Row) -> None:
        old = self.entries.pop(r.key, None)
        if not r.keep:
            return
        new_len = r.new_kv_len
        kv = {}
        for ln, (kn, vn) in r.pending_kv.items():
            if r.start and old is not None:
                kc, vc = old.kv[ln]
            else:
                kc = vc = None
            kv[ln] = (_grow(kc, kn, r.start, new_len), _grow(vc, vn, r.start, new_len))
        for ln in self.attention_layers:
            if ln not in kv and old is not None and new_len <= r.start:
                kv[ln] = old.kv[ln]  # nothing new to store (new_len == start)
        states = {}
        for s in r.keep:
            if s == r.start:
                states[s] = old.states[s] if s and old is not None else None
            else:
                states[s] = r.pending_states[s]
        states = {s: v for s, v in states.items() if v is not None}
        self.entries[r.key] = _Entry(
            tokens=r.tokens[:new_len].clone(), kv_len=new_len, kv=kv, states=states
        )

    def _fit_gpu(self, in_use: set) -> None:
        total = sum(e.nbytes() for e in self.entries.values() if e.on_gpu)
        for key, e in self.entries.items():  # least recently used first
            if total <= self.gpu_bytes:
                break
            if e.on_gpu and key not in in_use:
                total -= e.nbytes()
                e.to("cpu")
                self.stats["host_evictions"] = self.stats.get("host_evictions", 0) + 1

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
        gpu = sum(e.nbytes() for e in self.entries.values() if e.on_gpu)
        host = sum(e.nbytes() for e in self.entries.values() if not e.on_gpu)
        return {"gpu": gpu, "host": host, "rows": len(self.entries)}


def _grow(buf: Optional[torch.Tensor], new: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """``buf[:start]`` followed by ``new`` (positions [start, end)), capacity doubled as it grows.

    Amortized O(L) copying over a row's pieces (the cached K/V is not re-copied per piece
    except on growth).
    """
    if buf is not None and buf.shape[0] >= end:
        buf[start:end] = new
        return buf
    cap = max(end, 2 * (buf.shape[0] if buf is not None else 0), 256)
    out = new.new_empty((cap, *new.shape[1:]))
    if buf is not None and start:
        out[:start] = buf[:start]
    out[start:end] = new
    return out

