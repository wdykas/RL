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
"""Sticky fp16 rounding toward a theta_n anchor forward (ThunderSync, opt-in).

The fp16-GEMM recipe (``fp16_gemm.py``) rounds fp32 activations to fp16 at
fixed sites: linear-layer inputs, the q / k / v casts (also the KV cache) and
the expert GEMM inputs. Sticky rounding replaces round-to-nearest at those
sites by

    r(x; a) = a          if prev16(a) < x < next16(a)
            = RTN16(x)   otherwise

where a is the value the *anchor forward* (the plain forward of the previous
weights theta_n on the same tokens) produced at that site. The policy of step
n+1 is F(theta_{n+1}; anchors); with sticky MoE routing it shares its anchor
schedule: tokens with a route anchor are rounding-anchored, the rest
(rollouts never drafted, padding) anchor to themselves, i.e. RTN, the plain
recipe. Design and exactness argument: research/thundersync_rl/DESIGN.md §14g.

Learner implementation (no engine change: the engine never computes the
anchored policy, see §14g):

- ``ThetaStore`` keeps theta_n on host: before each optimizer step every rank
  copies its own local 16-bit parameters (its TP/PP/EP shard, exactly as the
  model holds them) into pinned host buffers.
- A policy forward (verification, training, log-probs) runs the anchor
  forward in lockstep: for every transformer layer, first the layer at
  theta_n (weights streamed from host, the next unit prefetched on a side
  stream; inference only, no autograd state) on the anchor residual stream,
  recording its fp16 values per site; then the layer at theta_{n+1} on the
  live stream, rounding each site with r(x; a). The anchor residual stream
  starts at the theta_n embedding.
- Expert sites: the fc1 input is a per-token value, rounded in token space
  before dispatch. The fc2 input of (token, expert) is matched by a key
  (source rank, token index) dispatched with the same routing map, so the
  anchor of (t, e) is the anchor forward's value if theta_n routed t to e
  (else self) under any expert-parallel layout.
- Backward: straight-through at the cast, as for round-to-nearest (anchors
  are constants).
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator, Optional

import torch

RECORD = "record"
APPLY = "apply"

# Process state. ``stream``: the anchor stream of the policy forward in flight.
_STATE: dict[str, Any] = {
    "enabled": False,
    "store": None,
    "stream": None,
    "armed": False,
    "stats": None,
    "installed": False,
}

_KEY_COLS = 4  # (source rank, local token index, anchored, unused)
_IDX_BITS = 24  # local token index < 2^24 (exact in fp32)


# ---------------------------------------------------------------------------
# The rule.
# ---------------------------------------------------------------------------


def sticky_round16(
    x: torch.Tensor, a: torch.Tensor, rtn: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """r(x; a) elementwise, as fp16.

    x: any float dtype; a: fp16 anchors broadcastable to x (NaN: no anchor,
    i.e. round to nearest); ``rtn``: ``x.to(fp16)`` if already computed.
    a is used where x lies strictly between a's fp16 neighbours, i.e. where a
    is one of the two fp16 values bracketing x (or x == a); elsewhere RTN.
    """
    if rtn is None:
        rtn = x.to(torch.float16)
    if a.dtype != torch.float16:
        raise TypeError(f"anchors must be fp16 (got {a.dtype})")
    cmp = torch.float64 if x.dtype == torch.float64 else torch.float32
    up = torch.nextafter(a, torch.full_like(a, float("inf"))).to(cmp)
    dn = torch.nextafter(a, torch.full_like(a, float("-inf"))).to(cmp)
    xc = x.to(cmp)
    use = (dn < xc) & (xc < up)
    return torch.where(use, a, rtn)


class StickyCast(torch.autograd.Function):
    """r(x; a) returned in x's dtype (values on the fp16 grid); straight-through backward."""

    @staticmethod
    def forward(ctx, x, a):  # noqa: D102
        out = sticky_round16(x, a)
        _count(out, x)
        return out.to(x.dtype)

    @staticmethod
    def backward(ctx, grad):  # noqa: D102
        return grad, None


def _count(out16: torch.Tensor, x: torch.Tensor) -> None:
    """Stats: elements whose rounding the anchor changed (no host sync)."""
    stats = _STATE["stats"]
    if stats is None:
        return
    stats.append((out16.ne(x.to(torch.float16)).sum(), out16.numel()))


def start_stats() -> None:
    """Count anchored-rounding changes of the APPLY sites until ``pop_stats``."""
    _STATE["stats"] = []


def pop_stats() -> dict[str, float]:
    """(elements, changed fraction) since ``start_stats`` (one host sync)."""
    stats, _STATE["stats"] = _STATE["stats"], None
    if not stats:
        return {"elements": 0.0, "changed": 0.0, "changed_frac": 0.0}
    n = sum(s[1] for s in stats)
    changed = int(torch.stack([s[0] for s in stats]).sum())
    return {
        "elements": float(n),
        "changed": float(changed),
        "changed_frac": changed / max(n, 1),
    }


# ---------------------------------------------------------------------------
# theta_n on host.
# ---------------------------------------------------------------------------


class ThetaStore:
    """theta_n per lockstep unit (embedding, each transformer layer) in pinned host memory.

    Holds exactly this rank's local parameters of each unit (its shard under
    TP / PP / EP, in the model's own dtype): host bytes per rank = the rank's
    16-bit model shard. ``snapshot`` runs before the optimizer step overwrites
    the parameters; ``fetch`` streams one unit to the GPU on a side stream.
    """

    def __init__(self, units: list[torch.nn.Module]):
        self.units = units
        self._index = {id(u): i for i, u in enumerate(units)}
        self._names = [
            [n for n, _ in u.named_parameters(remove_duplicate=False)] for u in units
        ]
        self._host: Optional[list[dict[str, torch.Tensor]]] = None
        self.ready = False
        self._copy_stream: Optional[torch.cuda.Stream] = None

    def index(self, unit: torch.nn.Module) -> int:
        return self._index[id(unit)]

    def host_bytes(self) -> int:
        if self._host is None:
            return 0
        return sum(t.numel() * t.element_size() for d in self._host for t in d.values())

    def _stream(self) -> torch.cuda.Stream:
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream()
        return self._copy_stream

    def snapshot(self) -> None:
        """Copy every unit's current parameters to host (returns when the copy is done)."""
        if self._host is None:
            self._host = [
                {
                    n: torch.empty(
                        p.shape, dtype=p.dtype, device="cpu", pin_memory=p.is_cuda
                    )
                    for n, p in u.named_parameters(remove_duplicate=False)
                }
                for u in self.units
            ]
        params = [dict(u.named_parameters(remove_duplicate=False)) for u in self.units]
        cuda = any(p.is_cuda for d in params for p in d.values())
        ctx = torch.cuda.stream(self._stream()) if cuda else contextlib.nullcontext()
        if cuda:
            self._stream().wait_stream(torch.cuda.current_stream())
        with torch.no_grad(), ctx:
            for host, live in zip(self._host, params):
                for n, h in host.items():
                    h.copy_(live[n].detach(), non_blocking=True)
        if cuda:
            self._stream().synchronize()
        self.ready = True

    def fetch(self, i: int, device: torch.device) -> tuple[dict[str, torch.Tensor], Any]:
        """Start the H2D copy of unit i; returns (tensors, event to wait on)."""
        if not self.ready:
            return {}, None
        if device.type != "cuda":
            return {n: t.to(device) for n, t in self._host[i].items()}, None
        s = self._stream()
        with torch.cuda.stream(s):
            out = {n: t.to(device, non_blocking=True) for n, t in self._host[i].items()}
            ev = torch.cuda.Event()
            ev.record(s)
        return out, ev


# ---------------------------------------------------------------------------
# The anchor stream of one policy forward.
# ---------------------------------------------------------------------------


class AnchorStream:
    """Lockstep state of one policy forward (one micro-batch, one model chunk).

    ``mask``: [S*B] bool in Megatron's sequence-major token order over the full
    (sequence-parallel-gathered) sequence: tokens whose roundings are anchored.
    """

    def __init__(
        self,
        store: Optional[ThetaStore],
        mask: torch.Tensor,
        tp_rank: int = 0,
        tp_size: int = 1,
        rank: int = 0,
    ):
        self.store = store
        self.mask = mask
        self.tp_rank, self.tp_size, self.rank = tp_rank, tp_size, rank
        self.mode: Optional[str] = None
        self.hidden: Optional[torch.Tensor] = None
        self.anchors: dict[Any, list[torch.Tensor]] = {}
        self.pending_keys: Optional[tuple[torch.Tensor, list[int]]] = None
        self._buf: dict[int, tuple[dict[str, torch.Tensor], Any]] = {}

    # -- weights ------------------------------------------------------------
    def params(self, unit: torch.nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
        """theta_n of ``unit`` on GPU; prefetches the next unit, frees older ones.

        At most two units are resident: the one in use and the next in flight.
        """
        if self.store is None or not self.store.ready:
            return {}
        i = self.store.index(unit)
        if i not in self._buf:
            self._buf[i] = self.store.fetch(i, device)
        for j in [j for j in self._buf if j < i]:
            del self._buf[j]
        nxt = i + 1
        if nxt < len(self.store.units) and nxt not in self._buf:
            self._buf[nxt] = self.store.fetch(nxt, device)
        tensors, ev = self._buf[i]
        if ev is not None:
            cur = torch.cuda.current_stream()
            cur.wait_event(ev)
            for t in tensors.values():
                t.record_stream(cur)
        return tensors

    def release(self) -> None:
        self._buf.clear()

    # -- masks --------------------------------------------------------------
    def mask_like(self, t: torch.Tensor) -> torch.Tensor:
        """The token mask broadcastable to a site tensor [S(,/tp), B, ...] or [T, ...]."""
        lead = 2 if t.dim() >= 3 else 1
        n = 1
        for d in t.shape[:lead]:
            n *= d
        m = self.mask
        if n != m.numel():
            if n * self.tp_size != m.numel():
                raise RuntimeError(
                    f"sticky rounding: site with {n} tokens does not match the "
                    f"forward's {m.numel()} tokens (tp {self.tp_size})"
                )
            m = m[self.tp_rank * n : (self.tp_rank + 1) * n]
        return m.view(*t.shape[:lead], *([1] * (t.dim() - lead)))

    def _masked(self, a16: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        nan = torch.full((), float("nan"), dtype=torch.float16, device=a16.device)
        return torch.where(self.mask_like(x), a16, nan)

    # -- sites --------------------------------------------------------------
    def record(self, site: Any, x: torch.Tensor, x16: torch.Tensor) -> None:
        self.anchors.setdefault(site, []).append(self._masked(x16, x))

    def pop(self, site: Any, shape: torch.Size) -> torch.Tensor:
        queue = self.anchors.get(site)
        if not queue:
            raise RuntimeError(f"sticky rounding: no anchor recorded for site {site!r}")
        a = queue.pop(0)
        if a.shape != shape:
            raise RuntimeError(
                f"sticky rounding: anchor shape {tuple(a.shape)} != site shape "
                f"{tuple(shape)} at {site!r}"
            )
        return a

    def end_layer(self) -> None:
        left = [k for k, q in self.anchors.items() if q]
        self.anchors.clear()
        if left:
            raise RuntimeError(f"sticky rounding: anchors never applied at {left!r}")


def active_stream() -> Optional[AnchorStream]:
    return _STATE["stream"]


def site_cast(site: Any, x: torch.Tensor) -> torch.Tensor:
    """fp16 cast of a rounding site (installed into ``fp16_gemm``).

    Outside a policy forward (and for fp16 inputs, where r is the identity):
    ``x.to(fp16)``. Anchor pass: records the value. Policy pass: r(x; a).
    """
    x16 = x.to(torch.float16)
    s = _STATE["stream"]
    if s is None or s.mode is None or x.dtype == torch.float16:
        return x16
    if s.mode == RECORD:
        s.record(site, x, x16)
        return x16
    out = sticky_round16(x, s.pop(site, x16.shape), x16)
    _count(out, x)
    return out


# ---------------------------------------------------------------------------
# MoE: token-space fc1 rounding and keyed fc2 anchors.
# ---------------------------------------------------------------------------


def _dispatch_rows(moe: Any, data: torch.Tensor, probs, routing_map):
    """``data`` [tokens, w] dispatched exactly as the layer's hidden states are."""
    disp = moe.token_dispatcher
    with torch.no_grad():
        h, p = disp.dispatch_preprocess(data, routing_map, probs)
        h, p = disp.token_dispatch(h, p)
        h, tokens_per_expert, _ = disp.dispatch_postprocess(h, p)
    if isinstance(tokens_per_expert, torch.Tensor):
        tokens_per_expert = tokens_per_expert.tolist()
    return h, [int(m) for m in tokens_per_expert]


def _row_keys(rows: torch.Tensor, tokens_per_expert: list[int]) -> torch.Tensor:
    """int64 key (local expert, source rank, token index) per dispatched row."""
    expert = torch.repeat_interleave(
        torch.arange(len(tokens_per_expert), device=rows.device),
        torch.tensor(tokens_per_expert, device=rows.device),
    )
    src = rows[:, 0].to(torch.int64)
    idx = rows[:, 1].to(torch.int64)
    return (expert << 48) | (src << _IDX_BITS) | idx


def moe_preprocess(moe: Any, hidden_states: torch.Tensor, probs, routing_map):
    """Training MoE preprocess hook (before MCore's dispatch).

    Anchor pass: records the token-space fc1 anchors. Policy pass: rounds the
    dispatched copy of the hidden states (not the router's input) with
    r(x; a). Both passes dispatch the token keys with the layer's routing map
    so ``experts_fc2_input`` can match rows of the two passes.
    """
    s = _STATE["stream"]
    if s is None or s.mode is None:
        return hidden_states
    d = hidden_states.shape[-1]
    tokens = hidden_states.numel() // d
    if tokens >= 1 << _IDX_BITS:
        raise RuntimeError("sticky rounding: more than 2^24 tokens per MoE layer call")
    flat_mask = s.mask_like(hidden_states).reshape(-1)
    keys = torch.zeros((tokens, _KEY_COLS), dtype=torch.float32, device=hidden_states.device)
    keys[:, 0] = float(s.rank)
    keys[:, 1] = torch.arange(tokens, device=keys.device, dtype=torch.float32)
    keys[:, 2] = flat_mask.to(torch.float32)
    rows, tpe = _dispatch_rows(moe, keys, probs, routing_map)
    s.pending_keys = (rows, tpe)
    site = (moe, "fc1")
    x16 = hidden_states.to(torch.float16)
    if s.mode == RECORD:
        s.record(site, hidden_states, x16)
        return hidden_states
    return StickyCast.apply(hidden_states, s.pop(site, x16.shape))


def experts_fc2_input(experts: Any, a: torch.Tensor, tokens_per_expert: list[int]):
    """Hook on the expert fc2 input rows (fp32 SwiGLU output x probs)."""
    s = _STATE["stream"]
    if s is None or s.mode is None:
        return a
    if s.pending_keys is None:
        raise RuntimeError("sticky rounding: expert rows without dispatched keys")
    rows, tpe = s.pending_keys
    s.pending_keys = None
    if tpe != list(tokens_per_expert) or rows.shape[0] != a.shape[0]:
        raise RuntimeError("sticky rounding: key dispatch does not match the experts' rows")
    keys = _row_keys(rows, tpe)
    anchored = rows[:, 2] > 0
    site = (experts, "fc2")
    a16 = a.to(torch.float16)
    if s.mode == RECORD:
        nan = torch.full((), float("nan"), dtype=torch.float16, device=a.device)
        vals = torch.where(anchored[:, None], a16, nan)
        srt, order = keys.sort()
        s.anchors.setdefault(site, []).append((srt, vals.index_select(0, order)))
        return a
    queue = s.anchors.get(site)
    if not queue:
        raise RuntimeError(f"sticky rounding: no anchor recorded for site {site!r}")
    srt, vals = queue.pop(0)
    if srt.numel() == 0:
        anchor = torch.full_like(a16, float("nan"))
    else:
        pos = torch.searchsorted(srt, keys).clamp_max(srt.numel() - 1)
        hit = srt.index_select(0, pos).eq(keys) & anchored
        anchor = torch.where(
            hit[:, None],
            vals.index_select(0, pos),
            torch.full((), float("nan"), dtype=torch.float16, device=a.device),
        )
    return StickyCast.apply(a, anchor)


# ---------------------------------------------------------------------------
# Lockstep patches.
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _anchor_pass(s: AnchorStream) -> Iterator[None]:
    from nemo_rl.models.megatron import fp16_moe, sticky_routing

    stats = sticky_routing._STATE["stats"]
    sticky_routing._STATE["stats"] = None  # count the policy pass only
    s.mode = RECORD
    try:
        with torch.no_grad(), fp16_moe.weight_cache_disabled():
            yield
    finally:
        s.mode = None
        sticky_routing._STATE["stats"] = stats


@contextlib.contextmanager
def _policy_pass(s: AnchorStream) -> Iterator[None]:
    s.mode = APPLY
    try:
        yield
    finally:
        s.mode = None


def _device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _layer_forward(orig):
    def forward(self, *args, **kwargs):
        s = _STATE["stream"]
        if s is None or s.mode is not None:
            return orig(self, *args, **kwargs)
        if s.hidden is None:
            raise RuntimeError("sticky rounding: transformer layer before the anchor stream starts")
        if "hidden_states" in kwargs:
            a_kwargs = dict(kwargs, hidden_states=s.hidden)
            a_args = args
        else:
            a_kwargs = kwargs
            a_args = (s.hidden, *args[1:])
        from torch.func import functional_call

        with _anchor_pass(s):
            params = s.params(self, _device(self))
            out = functional_call(self, params, a_args, a_kwargs, strict=False)
        s.hidden = out[0] if isinstance(out, tuple) else out
        with _policy_pass(s):
            result = orig(self, *args, **kwargs)
        s.end_layer()
        return result

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    return forward


def _embedding_forward(orig):
    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        s = _STATE["stream"]
        if s is None or s.mode is not None:
            return out
        from torch.func import functional_call

        with _anchor_pass(s):
            params = s.params(self, _device(self))
            s.hidden = functional_call(self, params, args, kwargs, strict=False)
        return out

    forward.__wrapped__ = orig  # type: ignore[attr-defined]
    return forward


def install() -> None:
    """Patch MCore's embedding / transformer layer and fp16_gemm's sites (idempotent)."""
    if _STATE["installed"]:
        return
    from megatron.core.models.common.embeddings.language_model_embedding import (
        LanguageModelEmbedding,
    )
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from nemo_rl.models.megatron import fp16_gemm, fp16_moe

    LanguageModelEmbedding.forward = _embedding_forward(LanguageModelEmbedding.forward)
    TransformerLayer.forward = _layer_forward(TransformerLayer.forward)
    fp16_gemm.set_sticky_hooks(site_cast, moe_preprocess)
    fp16_moe.set_fc2_input_hook(experts_fc2_input)
    _STATE["installed"] = True


# ---------------------------------------------------------------------------
# Worker-facing API.
# ---------------------------------------------------------------------------


def sticky_rounding_enabled(config: Any) -> bool:
    """``policy.megatron_cfg.sticky_rounding``."""
    return bool(((config or {}).get("megatron_cfg") or {}).get("sticky_rounding"))


def lockstep_units(model: torch.nn.Module) -> list[torch.nn.Module]:
    """This rank's lockstep units in forward order: embedding, then each layer."""
    units: list[torch.nn.Module] = []
    if getattr(model, "embedding", None) is not None and getattr(model, "pre_process", True):
        units.append(model.embedding)
    units.extend(model.decoder.layers)
    return units


def configure(enabled: bool, model: Optional[torch.nn.Module] = None) -> None:
    """Enable sticky rounding for this process and build theta_n's host store."""
    _STATE["enabled"] = bool(enabled)
    if not enabled:
        _STATE["store"] = None
        return
    install()
    if model is not None:
        _STATE["store"] = ThetaStore(lockstep_units(model))


def enabled() -> bool:
    return bool(_STATE["enabled"])


def store() -> Optional[ThetaStore]:
    return _STATE["store"]


def snapshot() -> None:
    """Keep the current weights as theta_n (call right before the optimizer step)."""
    st = _STATE["store"]
    if st is not None:
        st.snapshot()


def token_mask_from_replay(model: torch.nn.Module, tp_group: Any = None) -> Optional[torch.Tensor]:
    """[S*B] anchored-token mask from the armed RouterReplay targets (None: not armed).

    A token is anchored iff it has a valid route anchor at every local MoE
    layer (the sticky routing schedule). Sequence parallel: the targets are
    this rank's sequence shard; the mask is gathered over the TP group.
    """
    from megatron.core.transformer.moe.router_replay import RouterReplayAction

    from nemo_rl.models.megatron.router_replay import _router_replay_instances_for_model
    from nemo_rl.models.megatron.sticky_routing import anchor_valid

    mask = None
    for inst, _ in _router_replay_instances_for_model(model):
        if (
            inst.router_replay_action != RouterReplayAction.REPLAY_FORWARD
            or inst.target_topk_idx is None
        ):
            return None
        v = anchor_valid(inst.target_topk_idx)
        mask = v if mask is None else mask & v
    if mask is None:
        return None
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        parts = [torch.empty_like(mask) for _ in range(torch.distributed.get_world_size(tp_group))]
        torch.distributed.all_gather(parts, mask.contiguous(), group=tp_group)
        mask = torch.cat(parts)
    return mask


@contextlib.contextmanager
def policy_forward(
    model: torch.nn.Module,
    mask: torch.Tensor,
    tp_rank: int = 0,
    tp_size: int = 1,
    rank: int = 0,
) -> Iterator[AnchorStream]:
    """Run the enclosed model forward as F(theta; anchors of the theta_n forward)."""
    if _STATE["stream"] is not None:
        raise RuntimeError("sticky rounding: nested policy forwards")
    st = _STATE["store"]
    if (st is None or not st.ready) and bool(mask.any()):
        raise RuntimeError(
            "sticky rounding: anchored tokens but no theta_n snapshot (the first "
            "drafts must follow an optimizer step)"
        )
    s = AnchorStream(st, mask, tp_rank, tp_size, rank)
    _STATE["stream"] = s
    try:
        yield s
    finally:
        _STATE["stream"] = None
        s.release()
