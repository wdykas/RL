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
"""Trajectory-level gradient streaming for zero-staleness GRPO.

Implements the accumulation scheme of ThunderSyncRL
(https://arxiv.org/abs/2610.05935).

GRPO's per-trajectory advantage ``a_i`` depends on the whole group's rewards,
so a trajectory's gradient cannot be weighted until its group closes. But the
policy-gradient direction is linear in the per-trajectory score gradients
``H_i = d S_i / d theta`` (all evaluated at the same ``theta_k``):

    g_G = sum_i a_i H_i

Any GRPO-style estimator gives every trajectory in a group with the same
reward the same advantage, so

    g_G = sum_{r in R_G} a(r) * B_{G,r},     B_{G,r} = sum_{i: r_i = r} H_i

and ``B_{G,r}`` can be accumulated the moment each trajectory's reward is
known. This is exact for *any* group-wise estimator (mean/std, leave-one-out
baseline, leave-one-out std, advantage clipping, ...), and for binary rewards
it needs exactly the two buffers per open group used in the paper.

When a group produces more than ``max_buckets_per_group`` distinct reward
values, its buckets collapse to the paper's affine form

    G1 = sum_i r_i H_i,   G2 = sum_i H_i,   g_G = alpha * G1 + beta * G2

which is exact whenever ``a(r) = alpha * r + beta`` within the group (e.g.
mean/std normalization with a shared baseline and std). ``close_group``
verifies that assumption and raises if the estimator is not affine.

The accumulator is agnostic to where gradients come from: the caller hands it
the list of flat gradient tensors that one backward pass writes into (for
Megatron, the DDP ``grad_data`` buffers), and calls ``capture`` after each
backward. ``capture`` moves the fresh gradient into storage and zeroes the
gradient tensors, so they act as scratch space between captures.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import AbstractSet, Hashable, Iterator, Literal, Optional, Sequence, TypeVar

import torch

GroupId = Hashable
T = TypeVar("T")


@dataclass
class GradStreamingSpec:
    """Per-step settings for a train step that receives streamed chunks.

    Attributes:
        storage_device: where per-group accumulators live; ``"cpu"`` uses
            pinned host memory.
        max_buckets_per_group: distinct reward values kept per open group
            before collapsing to the affine ``(G1, G2)`` representation.
    """

    storage_device: Literal["cuda", "cpu"]
    max_buckets_per_group: int
    # Upper bound on groups holding buckets at once (enforced by the caller;
    # 0 when the caller never streams open groups); used to check accumulator
    # memory up front.
    max_open_groups: int


@dataclass
class StreamBucket:
    """Identifies the (group, reward) bucket a streamed chunk belongs to.

    Every trajectory in the chunk must belong to ``group``, share ``reward``,
    and carry advantage 1 in the training batch.
    """

    group: GroupId
    reward: float


@dataclass
class _GroupState:
    # reward value -> per-buffer accumulators (bucket mode)
    buckets: dict[float, list[torch.Tensor]] = field(default_factory=dict)
    # (G1, G2) per-buffer accumulators once collapsed to affine mode
    affine: Optional[tuple[list[torch.Tensor], list[torch.Tensor]]] = None
    rewards_seen: set[float] = field(default_factory=set)


class StreamingGroupAccumulator:
    """Accumulates per-group gradient sums and folds them on group close.

    Args:
        grad_tensors: flat gradient tensors written by backward. Must be zero
            when the accumulator is created and between ``capture`` calls.
        storage_device: where accumulators live (``"cuda"`` or ``"cpu"``; CPU
            storage uses pinned memory as in the paper).
        storage_dtype: accumulator dtype; fp32 unless testing in fp64.
        max_buckets_per_group: distinct reward values kept per group before
            collapsing to the affine (G1, G2) representation.
        affine_rtol: relative tolerance for the affine-advantage check.
    """

    def __init__(
        self,
        grad_tensors: list[torch.Tensor],
        *,
        storage_device: str,
        max_buckets_per_group: int,
        storage_dtype: torch.dtype = torch.float32,
        affine_rtol: float = 1e-4,
    ):
        assert max_buckets_per_group >= 2, "need >= 2 buckets to collapse to affine"
        self.grad_tensors = grad_tensors
        self.storage_device = storage_device
        self.storage_dtype = storage_dtype
        self.max_buckets_per_group = max_buckets_per_group
        self.affine_rtol = affine_rtol
        self.groups: dict[GroupId, _GroupState] = {}
        self._staging: Optional[list[torch.Tensor]] = None
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 1
        gpus = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        self._num_cpu_threads = max(1, min(32, cpus // gpus))
        # The batch accumulator always lives next to the gradients: final-chunk
        # captures (the bulk of the work) are then a device-side add, and only
        # the per-group buckets use ``storage_device``.
        self.batch_acc = [
            torch.zeros(g.numel(), dtype=storage_dtype, device=g.device)
            for g in grad_tensors
        ]
        self.peak_open_buffers = 0

    # ── storage helpers ──
    def _new_buffers(self) -> list[torch.Tensor]:
        pin = self.storage_device == "cpu" and torch.cuda.is_available()
        return [
            torch.zeros(
                g.numel(),
                dtype=self.storage_dtype,
                device=self.storage_device,
                pin_memory=pin,
            )
            for g in self.grad_tensors
        ]

    def _copy_grads(self) -> list[torch.Tensor]:
        out = []
        pin = self.storage_device == "cpu" and torch.cuda.is_available()
        for g in self.grad_tensors:
            buf = torch.empty(
                g.numel(),
                dtype=self.storage_dtype,
                device=self.storage_device,
                pin_memory=pin,
            )
            # Non-blocking only on-device (stream-ordered); a device->host
            # copy must complete before the CPU-side reads in close_group.
            buf.copy_(g.view(-1), non_blocking=buf.device == g.device)
            out.append(buf)
        return out

    def _host_staging(self) -> list[torch.Tensor]:
        """Pinned host buffers reused for every device->host gradient copy."""
        if self._staging is None:
            self._staging = [
                torch.empty(g.numel(), dtype=g.dtype, device="cpu", pin_memory=True)
                for g in self.grad_tensors
            ]
        return self._staging

    @contextmanager
    def _cpu_threads(self) -> Iterator[None]:
        """Use several intra-op threads for host-side accumulator math.

        Workers often run with OMP_NUM_THREADS=1, which makes adds over
        multi-GB host buffers single-threaded and slow.
        """
        if self.storage_device != "cpu":
            yield
            return
        prev = torch.get_num_threads()
        torch.set_num_threads(max(prev, self._num_cpu_threads))
        try:
            yield
        finally:
            torch.set_num_threads(prev)

    def _add_grad_into(self, dst: list[torch.Tensor], scale: float = 1.0) -> None:
        on_device = all(g.device == d.device for d, g in zip(dst, self.grad_tensors))
        if on_device:
            srcs = [g.view(-1) for g in self.grad_tensors]
        else:
            # Device -> pinned host at full bandwidth; synchronize before the
            # host-side add reads it.
            srcs = self._host_staging()
            for st, g in zip(srcs, self.grad_tensors):
                st.copy_(g.view(-1), non_blocking=True)
            torch.cuda.current_stream().synchronize()
        with self._cpu_threads():
            for d, src in zip(dst, srcs):
                if scale == 1.0:
                    d.add_(src.to(d.dtype))
                else:
                    d.add_(src.to(d.dtype), alpha=scale)

    _AXPY_PIECE = 1 << 28  # elements per host->device piece (1 GiB of fp32)

    def _axpy(
        self, dst: list[torch.Tensor], src: list[torch.Tensor], alpha: float
    ) -> None:
        if alpha == 0.0:
            return
        with self._cpu_threads():
            for d, s in zip(dst, src):
                if d.device == s.device:
                    d.add_(s, alpha=alpha)
                    continue
                # Host bucket into the device batch accumulator, streamed in
                # bounded pieces so no full-size device temporary is needed.
                for lo in range(0, s.numel(), self._AXPY_PIECE):
                    piece = s[lo : lo + self._AXPY_PIECE].to(
                        d.device, non_blocking=True
                    )
                    d[lo : lo + piece.numel()].add_(piece, alpha=alpha)

    def _zero_grads(self) -> None:
        for g in self.grad_tensors:
            g.zero_()

    def num_open_buffers(self) -> int:
        n = 0
        for st in self.groups.values():
            n += 2 if st.affine is not None else len(st.buckets)
        return n

    # ── public API ──
    def capture(self, group: Optional[GroupId], reward: Optional[float]) -> None:
        """Move the gradient just produced by backward into storage.

        ``group=None`` means the backward already used final advantages, so it
        is added straight into the batch accumulator. Otherwise the backward
        must have used advantage 1 for every trajectory, and all of them must
        belong to ``group`` and share ``reward``.
        """
        if group is None:
            self._add_grad_into(self.batch_acc)
            self._zero_grads()
            return
        r = float(reward)
        st = self.groups.setdefault(group, _GroupState())
        st.rewards_seen.add(r)
        if st.affine is None and (
            r in st.buckets or len(st.buckets) < self.max_buckets_per_group
        ):
            if r not in st.buckets:
                # First chunk of this bucket: copy instead of zero-fill + add.
                st.buckets[r] = self._copy_grads()
            else:
                self._add_grad_into(st.buckets[r])
        else:
            if st.affine is None:
                self._collapse_to_affine(st)
            g1, g2 = st.affine
            self._add_grad_into(g1, scale=r)
            self._add_grad_into(g2)
        self._zero_grads()
        self.peak_open_buffers = max(self.peak_open_buffers, self.num_open_buffers())

    def _collapse_to_affine(self, st: _GroupState) -> None:
        g1 = self._new_buffers()
        g2 = self._new_buffers()
        for r, b in st.buckets.items():
            self._axpy(g1, b, r)
            self._axpy(g2, b, 1.0)
        st.buckets = {}
        st.affine = (g1, g2)

    def close_group(
        self, group: GroupId, advantage_by_reward: dict[float, float]
    ) -> None:
        """Fold a closed group's accumulators into the batch accumulator.

        ``advantage_by_reward`` maps each reward value in the group to its
        (final, post-clipping) advantage. Unknown groups (no streamed
        trajectory on this rank) are ignored.
        """
        st = self.groups.pop(group, None)
        if st is None:
            return
        adv = {float(k): float(v) for k, v in advantage_by_reward.items()}
        missing = st.rewards_seen - set(adv)
        assert not missing, f"group {group}: no advantage for rewards {missing}"
        if st.affine is None:
            for r, b in st.buckets.items():
                self._axpy(self.batch_acc, b, adv[r])
            return
        alpha, beta = self._fit_affine(group, adv)
        g1, g2 = st.affine
        self._axpy(self.batch_acc, g1, alpha)
        self._axpy(self.batch_acc, g2, beta)

    def _fit_affine(
        self, group: GroupId, adv: dict[float, float]
    ) -> tuple[float, float]:
        rs = sorted(adv)
        if len(rs) == 1:
            return 0.0, adv[rs[0]]
        r0, r1 = rs[0], rs[-1]
        alpha = (adv[r1] - adv[r0]) / (r1 - r0)
        beta = adv[r0] - alpha * r0
        scale = max(1.0, max(abs(v) for v in adv.values()))
        for r in rs:
            err = abs(alpha * r + beta - adv[r])
            if err > self.affine_rtol * scale:
                raise ValueError(
                    f"group {group} exceeded max_buckets_per_group="
                    f"{self.max_buckets_per_group} distinct rewards and its "
                    "advantage is not affine in the reward (e.g. leave-one-out "
                    "std). Raise max_buckets_per_group or use an affine "
                    "estimator (normalize with shared mean/std)."
                )
        return alpha, beta

    def discard_group(self, group: GroupId) -> None:
        """Drop an open group's streamed gradients (e.g. its rollout is retried)."""
        self.groups.pop(group, None)

    def finalize(self) -> None:
        """Write the batch accumulator back into the gradient tensors."""
        assert not self.groups, f"groups still open at barrier: {list(self.groups)}"
        for g, a in zip(self.grad_tensors, self.batch_acc):
            g.view(-1).copy_(a)
        self.batch_acc = []


def pick_final_first(
    rows: Sequence[tuple[GroupId, float, T]],
    *,
    closed: AbstractSet[GroupId],
    streaming: AbstractSet[GroupId],
    max_open_groups: int,
    multiple_of: int,
) -> list[T]:
    """Choose which ready trajectories to backpropagate next.

    Final-first scheduling: rows of closed groups (final advantages known)
    always go first. Only when there are none does the learner spend its
    otherwise idle time on bucket work, for a single still-open group: one
    already streaming if possible (it holds buckets anyway), else the one with
    the most ready rows (the largest, most efficient chunk), and only while
    fewer than ``max_open_groups`` groups are streaming. Counts are trimmed to a
    multiple of ``multiple_of`` (per bucket for open groups) so every chunk
    shards evenly over data-parallel ranks; leftovers wait for more arrivals.

    Args:
        rows: ``(group, reward, item)`` for every ready, not yet dispatched row.
        closed: groups whose rewards are all known.
        streaming: open groups that already hold gradient buckets.
        max_open_groups: cap on streaming groups (bounds accumulator memory).
        multiple_of: required divisor of every chunk's row count.

    Returns:
        The items to dispatch now, in input order within each group.
    """

    def trim(items: list[T]) -> list[T]:
        return items[: len(items) - len(items) % multiple_of]

    final = trim([item for group, _, item in rows if group in closed])
    if final:
        return final
    buckets: dict[GroupId, dict[float, list[T]]] = {}
    for group, reward, item in rows:
        if group not in closed:
            buckets.setdefault(group, {}).setdefault(float(reward), []).append(item)
    usable = {
        group: [item for bucket in by_reward.values() for item in trim(bucket)]
        for group, by_reward in buckets.items()
        if group in streaming or len(streaming) < max_open_groups
    }
    usable = {group: items for group, items in usable.items() if items}
    if not usable:
        return []
    best = max(usable, key=lambda g: (g in streaming, len(usable[g])))
    return usable[best]
