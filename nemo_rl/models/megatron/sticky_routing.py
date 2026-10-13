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
inference forwards.
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
_STATE: dict[str, Any] = {"delta": None, "stats": None}


def sticky_routing_delta(config: Any) -> Optional[float]:
    """``policy.megatron_cfg.sticky_routing_delta`` (None: sticky routing off)."""
    megatron_cfg = (config or {}).get("megatron_cfg") or {}
    d = megatron_cfg.get("sticky_routing_delta")
    return None if d is None else float(d)


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
            return inner(self, scores, topk, num_groups, group_topk, default_compute_topk)
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
    return delta
