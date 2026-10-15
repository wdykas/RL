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
"""Sticky MoE routing anchored to a drafter's routes (ThunderSync speculation).

The policy F(theta; A) routes a token at an MoE layer with router scores ``s``
(the scores Megatron's top-k selects on: router logits for softmax routers
without pre-softmax) and anchor set ``A`` (top-k expert ids) by

    keep A   if  min_{e in A} s_e >= max_{e not in A} s_e - delta,
    else plain top-k of s.

An anchor row with a negative id or a repeated id (padding, "no anchor") means
"anchor = self": the plain top-k. A kept anchor set that equals the plain set
is returned in the plain top-k's order (so F(theta; plain_topk(theta)) is
bitwise the plain forward, sorted or unsorted top-k, ties included); a kept set
that differs is ordered by the current scores (descending, stable).

Score functions: ``s`` is whatever MCore's top-k selects on. Softmax routers
without pre-softmax: the router logits. Sigmoid routers with an expert bias
(DeepSeek-V3 / Nemotron-H, ``moe_router_enable_expert_bias``): the biased
selection scores sigmoid(logits) + expert_bias; MCore gathers the unbiased
sigmoid scores at the returned indices itself, so delta is in sigmoid-score
units there. Group-limited routing (``group_topk``): the plain set comes from
MCore's group-limited top-k and the keep test compares against every expert
outside the anchor set (conservative: it can only fall back to plain more often).

Everything here is elementwise or a fixed-shape reduction over the expert
axis: no host synchronization, CUDA-graph capturable, and the same arithmetic
on the engine and the learner (batch invariance carries over: a token's
selection depends only on its own scores and anchors).

Megatron integration (``install``): MCore's ``RouterReplay`` already routes
every top-k through ``get_replay_topk``. With sticky routing installed and a
delta set, a REPLAY_FORWARD forward applies the rule above with the replay
target indices as anchors and stores the effective selection for
REPLAY_BACKWARD (activation recompute replays exactly the routes the forward
used). RECORD records the plain top-k (the drafter's anchors) into the
instance's static buffer when one is set (engine CUDA graphs), only inside
inference forwards. With an engine anchor buffer attached (``APPLY_ATTR``,
anchor horizon), RECORD then applies the sticky rule against the step's
per-token anchors (rows of -1 for tokens without anchors), so one engine
forward both logs plain routes for drafts and samples F(theta; A) for
anchored requests.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

_PATCH_ATTR = "_nrl_sticky_routing_patch"
# The missing-route fallback patch's marker (router_replay.py): set on our
# wrapper too so that patch never wraps on top of it (the sticky wrapper
# handles all-sentinel rows itself).
_MISSING_ROUTE_ATTR = "_nrl_missing_route_fallback_patch"

# Process-wide sticky threshold (router-score units); None: sticky off (plain
# replay semantics unchanged).
_STATE: dict[str, Any] = {
    "delta": None,
    "stats": None,
    "horizon": None,
    "engine_verify": False,
}


def sticky_routing_delta(config: Any) -> Optional[float]:
    """``policy.megatron_cfg.sticky_routing_delta`` (None: sticky routing off)."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    d = megatron_cfg.get("sticky_routing_delta")
    return None if d is None else float(d)


def sticky_anchor_horizon(config: Any) -> Optional[int]:
    """``policy.megatron_cfg.sticky_anchor_horizon`` (None: anchors up to the cap).

    Set: the engine applies per-request anchors (positions below a rollout's
    horizon) in its forwards (ThunderSync anchor horizon).
    """
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    h = megatron_cfg.get("sticky_anchor_horizon")
    return None if h is None else int(h)


def get_anchor_horizon() -> Optional[int]:
    return _STATE.get("horizon")


def engine_verify_enabled(config: Any) -> bool:
    """``policy.megatron_cfg.engine_verify``: the engine verifies drafts itself.

    ThunderSync ``thundersync.verify_on=engine`` (or its prefill check): the
    engine prefills drafted rows at the current weights, applying the drafts'
    anchors in its forwards, and computes p from the captured head inputs.
    """
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    return bool(megatron_cfg.get("engine_verify", False))


def engine_anchor_apply() -> bool:
    """Whether engine forwards apply per-request anchors (horizon or engine verify)."""
    return _STATE.get("horizon") is not None or bool(_STATE.get("engine_verify"))


def sticky_routing_enabled(config: Any) -> bool:
    return sticky_routing_delta(config) is not None


def set_delta(delta: Optional[float]) -> None:
    _STATE["delta"] = None if delta is None else float(delta)


def get_delta() -> Optional[float]:
    return _STATE["delta"]


def anchor_valid(anchors: torch.Tensor) -> torch.Tensor:
    """[T] bool: rows that carry a usable anchor (all ids >= 0, no repeats)."""
    if anchors.shape[-1] <= 1:
        return anchors.ge(0).all(-1)
    srt = anchors.sort(-1).values
    distinct = (srt[..., 1:] != srt[..., :-1]).all(-1)
    return anchors.ge(0).all(-1) & distinct


def sticky_select(
    scores: torch.Tensor,
    anchors: torch.Tensor,
    plain_vals: torch.Tensor,
    plain_idx: torch.Tensor,
    delta: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The sticky rule given the plain top-k of ``scores``.

    scores [T, E]; anchors [T, k] (any int dtype; invalid rows = self);
    plain_vals / plain_idx [T, k] (top-k values and indices of ``scores``).
    Returns (values [T, k], indices [T, k] int64, keep [T] bool), values =
    ``scores.gather(1, indices)`` bitwise.
    """
    valid = anchor_valid(anchors)
    a = anchors.to(torch.long).clamp_min(0)
    s_in = scores.gather(1, a)
    min_in = s_in.min(dim=1).values
    max_out = scores.scatter(1, a, float("-inf")).max(dim=1).values
    keep = valid & (min_in >= max_out - delta)
    s_sorted, order = s_in.sort(dim=1, descending=True, stable=True)
    a_sorted = a.gather(1, order)
    # A kept anchor set equal to the plain set is returned in the plain order, so
    # F(theta; plain_topk(theta)) is bitwise plain even where the plain top-k is
    # unsorted (MCore's no-grad top-k without batch invariance) or has exact ties:
    # the probability normalization sums over k in index order.
    same = (
        a.sort(dim=1).values == plain_idx.to(torch.long).sort(dim=1).values
    ).all(dim=1)
    keep_anchor_order = keep & ~same
    k = keep_anchor_order.unsqueeze(1)
    idx = torch.where(k, a_sorted, plain_idx.to(torch.long))
    vals = torch.where(k, s_sorted, plain_vals)
    return vals, idx, keep


def sticky_topk(
    scores: torch.Tensor,
    anchors: torch.Tensor,
    topk: int,
    delta: float,
    compute_topk: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(values, effective indices, plain indices) of the sticky rule.

    ``compute_topk`` is Megatron's default top-k (``_compute_topk``), so the
    plain selection (and its tie order) is exactly what an unanchored forward
    computes.
    """
    plain_vals, plain_idx = compute_topk(
        scores, topk, num_groups=num_groups, group_topk=group_topk
    )
    vals, idx, keep = sticky_select(scores, anchors, plain_vals, plain_idx, delta)
    stats = _STATE["stats"]
    if stats is not None and not (
        scores.is_cuda and torch.cuda.is_current_stream_capturing()
    ):
        stats.append(
            (
                int(scores.shape[0]),
                keep.sum(),
                (idx.sort(1).values != plain_idx.sort(1).values).any(1).sum(),
            )
        )
    return vals, idx, plain_idx


def start_stats() -> None:
    """Collect per-call (tokens, kept anchors, differs from plain) counters."""
    _STATE["stats"] = []


def pop_stats() -> dict[str, float]:
    """Totals since ``start_stats`` (one host sync); stops collecting."""
    stats, _STATE["stats"] = _STATE["stats"], None
    if not stats:
        return {"token_layers": 0.0, "anchor_kept": 0.0, "differs_from_plain": 0.0}
    n = sum(s[0] for s in stats)
    kept = int(torch.stack([s[1] for s in stats]).sum())
    diff = int(torch.stack([s[2] for s in stats]).sum())
    return {
        "token_layers": float(n),
        "anchor_kept": kept / max(n, 1),
        "differs_from_plain": diff / max(n, 1),
    }


# Engine anchor apply: attribute on an engine router's ``RouterReplay`` holding
# (buffer [max_tokens, K] view of the per-token anchor buffer, row offset
# multiplier) set by the ThunderSync generation worker before CUDA-graph
# capture. Sequence parallelism: TP rank r routes tokens [r T, (r+1) T) of
# the step, so the multiplier is r (0 without sequence parallelism).
APPLY_ATTR = "_nrl_anchor_apply"


def engine_step_anchors(apply: tuple[torch.Tensor, int], n: int) -> torch.Tensor:
    """Rows of the per-token anchor buffer for a router call over ``n`` tokens.

    A fixed slice of a static buffer (its offset depends only on ``n``, i.e.
    on the graph's token count), so it is captured into CUDA graphs and every
    replay reads the anchors the step filled in.
    """
    buf, rank = apply
    lo = rank * n
    return buf[lo : lo + n]


def _engine_select() -> Callable:
    """The rule for engine forwards: one fused kernel on CUDA (bitwise equal to
    ``sticky_select``, see ``sticky_routing_fused``), else the reference."""
    fn = _STATE.get("engine_select")
    if fn is None:
        try:
            from nemo_rl.models.megatron.sticky_routing_fused import sticky_select_fused
        except ImportError:  # pragma: no cover (no Triton)
            sticky_select_fused = None

        def fn(scores, anchors, plain_vals, plain_idx, delta):
            if sticky_select_fused is not None and scores.is_cuda:
                return sticky_select_fused(
                    scores.contiguous(), anchors, plain_vals, plain_idx, delta
                )
            return sticky_select(scores, anchors, plain_vals, plain_idx, delta)

        _STATE["engine_select"] = fn
    return fn


def _inference_active() -> bool:
    try:
        from megatron.core.inference.utils import InferenceMode
    except ImportError:  # pragma: no cover
        return False
    return bool(InferenceMode.is_active())


def install() -> None:
    """Wrap ``RouterReplay.get_replay_topk`` with the sticky rule (idempotent).

    Installs NeMo RL's missing-route fallback first and wraps it, so with
    sticky off (delta None) every action behaves exactly as before.
    """
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    from nemo_rl.models.megatron.router_replay import (
        _install_missing_route_fallback_patch,
    )

    if getattr(RouterReplay.get_replay_topk, _PATCH_ATTR, False):
        return
    _install_missing_route_fallback_patch()
    inner = RouterReplay.get_replay_topk

    def get_replay_topk(
        self,
        scores: torch.Tensor,
        topk: int,
        num_groups: Optional[int] = None,
        group_topk: Optional[int] = None,
        default_compute_topk: Optional[Callable] = None,
    ):
        action = self.router_replay_action
        if action == RouterReplayAction.RECORD:
            # Engine: record the plain top-k (the drafter's anchors) only in
            # inference forwards; a training-shaped forward in the same
            # process neither records nor overruns the static buffer.
            if getattr(self, "_nrl_record_inference_only", False) and not (
                _inference_active()
            ):
                return default_compute_topk(
                    scores, topk, num_groups=num_groups, group_topk=group_topk
                )
            probs, idx = inner(
                self, scores, topk, num_groups, group_topk, default_compute_topk
            )
            apply = getattr(self, APPLY_ATTR, None)
            if apply is None or _STATE["delta"] is None:
                return probs, idx
            # Engine anchor apply (anchor horizon): the plain top-k is
            # recorded above (the drafter's log); the forward uses the sticky
            # rule against this step's per-token anchors (all -1 = self for
            # tokens without anchors: then bitwise the plain selection).
            anchors = engine_step_anchors(apply, int(scores.shape[0]))
            vals, eff, _ = _engine_select()(
                scores, anchors, probs, idx, _STATE["delta"]
            )
            return vals, eff
        delta = _STATE["delta"]
        if (
            delta is None
            or action != RouterReplayAction.REPLAY_FORWARD
            or self.target_topk_idx is None
        ):
            return inner(self, scores, topk, num_groups, group_topk, default_compute_topk)
        anchors = self.target_topk_idx.to(scores.device)
        vals, idx, _ = sticky_topk(
            scores, anchors, topk, delta, default_compute_topk, num_groups, group_topk
        )
        if self.replay_backward_list:
            # set_target_indices appended the anchors for REPLAY_BACKWARD; a
            # recompute must replay the routes this forward actually used.
            self.replay_backward_list[-1] = idx.detach()
        return vals, idx

    setattr(get_replay_topk, _PATCH_ATTR, True)
    setattr(get_replay_topk, _MISSING_ROUTE_ATTR, True)
    get_replay_topk.__wrapped__ = inner  # type: ignore[attr-defined]
    RouterReplay.get_replay_topk = get_replay_topk


def configure(config: Any) -> Optional[float]:
    """Install the patch and set the process delta from a policy config."""
    delta = sticky_routing_delta(config)
    if delta is not None:
        install()
    set_delta(delta)
    _STATE["horizon"] = None if delta is None else sticky_anchor_horizon(config)
    _STATE["engine_verify"] = delta is not None and engine_verify_enabled(config)
    return delta
