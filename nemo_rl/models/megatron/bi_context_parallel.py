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
"""Batch-invariant context parallelism for the learner: head-parallel sequence mixers.

Every per-token op (embedding, norms, GEMMs, MoE, LM head, log-probs) runs on this rank's contiguous chunk of the
sequence. The two sequence mixers switch from sequence-parallel to head-parallel with all-to-alls (the "Ulysses"
layout), so each runs the whole sequence for a subset of heads with the same kernels and the same per-head
arithmetic as CP 1:

- attention: q all-to-all (sequence -> heads), K / V all-gathered over the sequence (GQA: few KV heads) and narrowed
  to the KV heads of this rank's q heads, the model's own core attention (FA4, num_splits = 1 or anchored split-KV)
  on the whole sequence, output all-to-all back (heads -> sequence). A query row's computation (its keys, its key
  blocks, its reduction order) does not depend on which rank runs it or on how many heads run with it;
- Mamba: Megatron's head-parallel Mamba CP (``MambaContextParallel``: z / x / B / C / dt all-to-all to whole-sequence
  head slices, conv + SSD scan per head on the whole sequence, y back), with this module's CP group and contiguous
  chunks. Our batch-invariant SSD path (mamba_ssd_parity) already reads every per-rank size from ``mixer.cp``.

Megatron's batch-invariant config refuses ``context_parallel_size > 1`` (its CP routes attention through TE's FA2):
the model is built with CP 1 and this module owns the CP group (``install(model, group)``). The learner prefix-state
cache composes when it is created before ``install`` (its hooks then see the head-parallel, whole-sequence tensors:
cached K / V and Mamba states are sharded by heads, not positions).

Per-rank cost (sequence s, CP c, per attention layer with q heads hq / KV heads hkv per TP rank, head dim d; per
Mamba layer with in-projection width w = 2 d_inner + 2 g n + h): activations of per-token layers 1/c; attention moves
2 x s/c x hq d (c-1)/c (q in, out back) + 2 x s hkv d (c-1)/c (K / V gather) fp16 values per layer, and holds the
gathered K / V (2 s hkv d); Mamba moves s/c x w (c-1)/c + s/c x d_inner (c-1)/c values per layer (a2a in / out).
Gradients: parameters are replicated over CP; their gradients are partial sums over this rank's tokens and are summed
over the CP group in rank order (``reduce_grads``). They equal CP 1's up to the order of that sum (a weight
gradient's reduction over tokens is split at the CP boundaries); forward log-probs / logits are bitwise CP-invariant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

_STATE: dict[str, Any] = {"group": None, "active": True, "mixers": [], "layout": "contiguous"}


def group() -> Any:
    return _STATE["group"]


def size() -> int:
    g = _STATE["group"]
    return 1 if g is None or not _STATE["active"] else dist.get_world_size(g)


def set_active(flag: bool) -> None:
    """Switch CP on / off at runtime (probes: the CP 1 reference on the same model)."""
    _STATE["active"] = bool(flag)
    for m in _STATE["mixers"]:
        m.cp = m._bi_cp_c if flag else m._bi_cp_1


def rank() -> int:
    g = _STATE["group"]
    return 0 if g is None else dist.get_rank(g)


class _GatherSeq(torch.autograd.Function):
    """[s/c, b, ...] -> [s, b, ...] (rank order); backward: rank-order sum of the gradient slices."""

    @staticmethod
    def forward(ctx, x, grp):  # noqa: D102
        ctx.grp = grp
        c = dist.get_world_size(grp)
        out = torch.empty((c * x.shape[0], *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=grp)
        return out

    @staticmethod
    def backward(ctx, g):  # noqa: D102
        from nemo_rl.models.megatron.fp16_gemm import ordered_reduce_scatter_rows

        dt = g.dtype
        return ordered_reduce_scatter_rows(g.float().contiguous(), ctx.grp).to(dt), None


def _cp2hp(x: torch.Tensor, grp: Any) -> torch.Tensor:
    from megatron.core.ssm.mamba_context_parallel import _all_to_all_cp2hp

    return _all_to_all_cp2hp(x, grp)


def _hp2cp(x: torch.Tensor, grp: Any) -> torch.Tensor:
    from megatron.core.ssm.mamba_context_parallel import _all_to_all_hp2cp

    return _all_to_all_hp2cp(x, grp)


def _attention(mod: Any, inner: Any, query, key, value, attention_mask, *args, **kwargs):
    grp = _STATE["group"]
    if grp is None or not _STATE["active"] or dist.get_world_size(grp) == 1:
        return inner(query, key, value, attention_mask, *args, **kwargs)
    if kwargs.get("packed_seq_params") is not None:
        raise NotImplementedError("BI context parallel: packed sequences not built")
    c = dist.get_world_size(grp)
    r = dist.get_rank(grp)
    sl, b, hq, d = query.shape
    hkv = key.shape[2]
    if hq % c:
        raise ValueError(f"BI context parallel: q heads per TP rank {hq} not divisible by CP {c}")
    hq_c = hq // c
    zig = _STATE["layout"] == "zigzag"
    q3 = _cp2hp(query.reshape(sl, b, hq * d), grp)
    k = _GatherSeq.apply(key, grp)
    v = _GatherSeq.apply(value, grp)
    if zig:  # rank-order chunks (r, 2c-1-r) -> natural order for the whole-sequence attention
        from megatron.core.ssm.mamba_context_parallel import _undo_attention_load_balancing as _undo

        q3 = _undo(q3, c)
        k = _undo(k.reshape(sl * c, b, -1), c).reshape(sl * c, b, hkv, d)
        v = _undo(v.reshape(sl * c, b, -1), c).reshape(sl * c, b, hkv, d)
    q = q3.reshape(sl * c, b, hq_c, d)
    per = hq // hkv  # q heads per KV head
    lo, hi = r * hq_c // per, ((r + 1) * hq_c - 1) // per + 1  # KV heads of this rank's q heads
    # TE's DotProductAttention checks the KV head count against its per-partition GQA groups.
    te = getattr(mod, "num_gqa_groups_per_partition", None)
    if te is not None:
        mod.num_gqa_groups_per_partition = hi - lo
    try:
        out = inner(q, k[:, :, lo:hi], v[:, :, lo:hi], attention_mask, *args, **kwargs)
    finally:
        if te is not None:
            mod.num_gqa_groups_per_partition = te
    # out: [s, b, hq_c * d] (core attention's flattened heads) -> [s/c, b, hq * d]
    out = out.reshape(sl * c, b, hq_c * d)
    if zig:
        from megatron.core.ssm.mamba_context_parallel import _redo_attention_load_balancing as _redo

        out = _redo(out, c)
    return _hp2cp(out, grp)


def install(model: Any, grp: Any, layout: str = "contiguous") -> dict:
    """Make ``model``'s attention and Mamba layers context-parallel over ``grp``. Idempotent.

    ``layout``: "contiguous" (rank r holds chunk r) or "zigzag" (Megatron / NeMo-RL load-balanced causal layout:
    rank r holds chunks r and 2c-1-r of 2c; ``nemo_rl.distributed.model_utils._get_tokens_on_this_cp_rank``).
    """
    if layout not in ("contiguous", "zigzag"):
        raise ValueError(f"layout must be contiguous|zigzag (got {layout!r})")
    _STATE["layout"] = layout
    import functools

    from megatron.core.ssm.mamba_context_parallel import MambaContextParallel
    from megatron.core.ssm.mamba_mixer import MambaMixer
    from megatron.core.transformer.attention import Attention

    _STATE["group"] = grp
    n = {"attention": 0, "mamba": 0, "router": 0}
    tp_cp = _tp_cp_group(grp)
    chunks = model if isinstance(model, (list, tuple)) else [model]
    for chunk in chunks:
        for m in chunk.modules():
            if isinstance(m, Attention):
                core = m.core_attention
                if getattr(core, "_bi_cp", False):
                    continue
                inner = core.__dict__.get("forward") or core.forward  # a prefix-cache hook if installed
                core.forward = functools.partial(_attention, core, inner)
                core._bi_cp = True
                n["attention"] += 1
            elif hasattr(m, "tp_cp_group") and hasattr(m, "_apply_aux_loss"):
                # MoE router: load-balancing / z-loss statistics and their scaling over TP x CP, as Megatron's
                # own CP does (the model was built at CP 1; without this each rank balanced its own chunk:
                # a different aux-loss gradient into every earlier layer, ~4% median gradient error vs CP 1).
                m.tp_cp_group = tp_cp
                n["router"] += 1
            elif isinstance(m, MambaMixer):
                if getattr(m, "_bi_cp", False):
                    continue
                if getattr(m.cp, "cp_group", None) is grp and m.cp.cp_size == dist.get_world_size(grp):
                    m._bi_cp = True  # Megatron built it on this group (in-loop CP): keep
                    m._bi_cp_1 = m._bi_cp_c = m.cp
                    _STATE["mixers"].append(m)
                    n["mamba"] += 1
                    continue
                m._bi_cp_1 = m.cp
                m.cp = m._bi_cp_c = MambaContextParallel(
                    cp_group=grp,
                    d_inner_local_tp=m.d_inner_local_tp,
                    nheads_local_tp=m.nheads_local_tp,
                    ngroups_local_tp=m.ngroups_local_tp,
                    d_state=m.d_state,
                    conv1d_weight_cp1=m.conv1d_weight,
                    conv1d_bias_cp1=m.conv1d_bias,
                    conv1d_padding=m.d_conv - 1,
                    dt_bias_cp1=m.dt_bias,
                    A_log_cp1=m.A_log,
                    D_cp1=m.D,
                    D_has_hdim=m.D_has_hdim,
                    sequence_is_contiguous=layout == "contiguous",
                )
                m._bi_cp = True
                _STATE["mixers"].append(m)
                n["mamba"] += 1
    return n


def _tp_cp_group(grp: Any) -> Any:
    """The TP x CP group of this rank (TP 1: the CP group). Collective: every rank calls it once."""
    from megatron.core import parallel_state as ps

    tp_ranks = tuple(dist.get_process_group_ranks(ps.get_tensor_model_parallel_group()))
    if len(tp_ranks) == 1:
        return grp
    cp_ranks = tuple(dist.get_process_group_ranks(grp))
    allv = [None] * dist.get_world_size()
    dist.all_gather_object(allv, (tp_ranks, cp_ranks))
    cp_of = {r: set(c) for r, (_, c) in enumerate(allv)}
    sets = sorted({tuple(sorted(set().union(*(cp_of[t] for t in tp)))) for tp, _ in allv})
    mine = tuple(sorted(set().union(*(cp_of[t] for t in tp_ranks))))
    out = None
    for ranks in sets:
        g = dist.new_group(list(ranks))
        if ranks == mine:
            out = g
    return out


def shard(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """This rank's contiguous chunk of ``t`` along ``dim`` (the sequence)."""
    c, r = size(), rank()
    n = t.shape[dim]
    if _STATE["layout"] == "zigzag":
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank

        if n % (2 * c):
            raise ValueError(f"sequence {n} not divisible by 2 x CP {c}")
        return _get_tokens_on_this_cp_rank(t, r, c, seq_dim=dim)
    if n % c:
        raise ValueError(f"sequence {n} not divisible by CP {c}")
    return t.narrow(dim, r * (n // c), n // c)


def gather(t: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Concatenate the CP ranks' chunks along ``dim`` (no autograd)."""
    c = size()
    if c == 1:
        return t
    parts = [torch.empty_like(t) for _ in range(c)]
    dist.all_gather(parts, t.contiguous(), group=_STATE["group"])
    if _STATE["layout"] == "zigzag":
        halves = [p.chunk(2, dim) for p in parts]
        return torch.cat([h[0] for h in halves] + [h[1] for h in reversed(halves)], dim)
    return torch.cat(parts, dim)


def reduce_grads(params, attr: str = "main_grad") -> None:
    """Sum replicated parameters' gradients over the CP group in rank order (fp32, deterministic)."""
    from nemo_rl.models.megatron.fp16_gemm import _fold_rank_order

    grp = _STATE["group"]
    c = size()
    if c == 1:
        return
    for p in params:
        g = getattr(p, attr, None) if attr != "grad" else p.grad
        if g is None:
            continue
        flat = g.detach().float().reshape(-1).contiguous()
        parts = torch.empty((c, flat.numel()), dtype=flat.dtype, device=flat.device)
        dist.all_gather_into_tensor(parts, flat, group=grp)
        g.copy_(_fold_rank_order(parts).view(g.shape).to(g.dtype))


# ---------------------------------------------------------------------------
# Megatron / NeMo-RL integration: real CP process groups, model built as CP 1 numerics.
# ---------------------------------------------------------------------------

_GATHER_OUT = [0]


class gather_outputs:
    """Inside: the CP model wrapper returns whole-sequence outputs (all-gathered over CP) -- for callers that read
    every position (ThunderSync verification). Outside: CP-local outputs (NeMo-RL's CP-aware log-prob / loss path)."""

    def __enter__(self):
        _GATHER_OUT[0] += 1
        return self

    def __exit__(self, *exc):
        _GATHER_OUT[0] -= 1
        return False


def allow_batch_invariant_cp() -> None:
    """Megatron's TransformerConfig refuses batch_invariant_mode with context_parallel_size > 1 (its CP is TE's
    FA2 ring). Under this module's head-parallel CP the model's own attention runs at CP 1 numerics: the check
    sees CP 1 (the field is restored right after). Idempotent; call before the model config is finalized."""
    from megatron.core.transformer.transformer_config import TransformerConfig

    if getattr(TransformerConfig, "_ts_bi_cp", False):
        return
    orig = TransformerConfig.__post_init__

    def __post_init__(self):
        cp = getattr(self, "context_parallel_size", 1) or 1
        if getattr(self, "batch_invariant_mode", False) and cp > 1:
            self.context_parallel_size = 1
            try:
                return orig(self)
            finally:
                self.context_parallel_size = cp
        return orig(self)

    TransformerConfig.__post_init__ = __post_init__
    TransformerConfig._ts_bi_cp = True


def _slice_token_rows(t: torch.Tensor, s: int, b: int) -> torch.Tensor:
    """Rows of a [s * b, ...] s-major per-token tensor (router replay targets) for this rank's sequence chunk."""
    return shard(t.reshape(s, b, *t.shape[1:]), dim=0).reshape(-1, *t.shape[1:]).contiguous()


def _replay_instances(model: Any) -> list:
    try:
        from nemo_rl.models.megatron.router_replay import _router_replay_instances_for_model
    except ImportError:  # pragma: no cover
        return []
    return [inst for inst, _ in _router_replay_instances_for_model(model)]


def install_megatron_cp(model: Any) -> dict:
    """In-loop BI CP over Megatron's context-parallel group (``policy.megatron_cfg.context_parallel_size > 1``).

    - attention: head-parallel wrapper (``install``, zigzag layout = NeMo-RL's ``_get_tokens_on_this_cp_rank``);
      TE's own CP is switched off on every core attention (it would run its FA2 ring on whole-sequence tensors);
    - Mamba: Megatron already built ``MambaContextParallel`` on the CP group (zigzag): kept;
    - routers: Megatron's tp_cp group already spans CP;
    - the top model's forward takes whole rows (NeMo-RL's unpacked micro-batches, ThunderSync verification rows),
      keeps this rank's zigzag chunk of input_ids / position_ids / router-replay targets, and returns the CP-local
      output -- or, inside ``gather_outputs()``, the whole-sequence output.
    """
    from megatron.core import parallel_state as ps
    from megatron.core.transformer.attention import Attention

    grp = ps.get_context_parallel_group()
    chunks = model if isinstance(model, (list, tuple)) else [model]
    n = install(chunks, grp, layout="zigzag")
    for chunk in chunks:
        for m in chunk.modules():
            if isinstance(m, Attention) and hasattr(m.core_attention, "set_context_parallel_group"):
                m.core_attention.set_context_parallel_group(None, None, None)
    for chunk in chunks:
        top = chunk
        while hasattr(top, "module"):
            top = top.module
        if getattr(top, "_bi_cp_forward", False):
            continue
        orig = top.forward

        def forward(input_ids=None, position_ids=None, attention_mask=None, *args, _orig=orig, _top=top, **kw):
            c = size()
            if c == 1 or kw.get("packed_seq_params") is not None or input_ids is None:
                return _orig(input_ids, position_ids, attention_mask, *args, **kw)
            b, s = input_ids.shape[:2]
            ids = shard(input_ids).contiguous()
            pos = shard(position_ids).contiguous() if position_ids is not None else None
            saved = []
            for inst in _replay_instances(_top):
                t = inst.target_topk_idx
                if t is None:
                    continue
                sl = _slice_token_rows(t, s, b)
                saved.append((inst, t))
                inst.target_topk_idx = sl
                if inst.replay_backward_list and inst.replay_backward_list[-1] is t:
                    inst.replay_backward_list[-1] = sl
            try:
                out = _orig(ids, pos, attention_mask, *args, **kw)
            finally:
                for inst, t in saved:
                    inst.target_topk_idx = t
            if _GATHER_OUT[0] <= 0:
                return out
            from nemo_rl.models.megatron.deferred_head import DeferredHeadLogits

            if isinstance(out, DeferredHeadLogits):  # head input x: [s/c, b, h] -> [s, b, h]
                if tuple(out._seq) != (0, int(out._x.shape[0])):
                    raise NotImplementedError("BI CP: a sequence-sliced deferred head before the CP gather")
                # seq None: the whole gathered sequence (the local (0, s/c) slice would keep the first chunk only).
                return DeferredHeadLogits(gather(out._x, dim=0), out._w, out._batch_first, None)
            if isinstance(out, tuple):
                return (gather(out[0], dim=1), *out[1:])
            return gather(out, dim=1)

        top.forward = forward
        top._bi_cp_forward = True
    return n
