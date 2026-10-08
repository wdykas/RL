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
"""Driver-side pieces of gradient streaming: dispatch planning and the learner handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, Optional

import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy

GroupId = Hashable

_SHARD_AXES = dict(
    in_sharded_axes=["data_parallel"],
    replicate_on_axes=["context_parallel", "tensor_parallel", "pipeline_parallel"],
    output_is_replicated=["context_parallel", "tensor_parallel", "pipeline_parallel"],
)


@dataclass
class Trajectory:
    """A finished, rewarded trajectory waiting to be backpropagated."""

    group: GroupId
    index: int  # position in the logical batch
    reward: float
    payload: Any  # whatever the data builder needs (e.g. the message log)
    num_tokens: int


@dataclass
class Chunk:
    """One backward unit sent to one DP rank."""

    group: Optional[GroupId]  # None => final advantages are baked in
    reward: Optional[float]
    trajectories: list[Trajectory]
    advantages: Optional[list[float]]  # final chunks only; None for bucket chunks

    @property
    def num_tokens(self) -> int:
        return sum(t.num_tokens for t in self.trajectories)


@dataclass
class Dispatch:
    per_rank: list[list[Chunk]]
    closes: list[tuple[GroupId, dict[float, float]]]

    @property
    def num_trajectories(self) -> int:
        return sum(len(c.trajectories) for cs in self.per_rank for c in cs)


@dataclass
class _Group:
    size: int
    rewards: dict[int, float] = field(default_factory=dict)  # index -> reward
    streamed_open: bool = False  # some trajectory was dispatched before close
    close_sent: bool = False
    advantages: Optional[dict[int, float]] = None


class StreamPlanner:
    """Turns trajectory arrivals into work-conserving backward dispatches.

    A trajectory is dispatchable as soon as its reward is known. If its group
    is still open, it goes into a (group, reward) bucket chunk with advantage
    1; if its group has closed, it gets its final advantage directly. Groups
    that had streamed (bucketed) trajectories are closed on the workers right
    after their last bucket chunk has been dispatched.

    Args:
        compute_group_advantages: maps the full reward list of a closed group
            (in arrival-independent index order) to per-trajectory advantages.
            Must be the same function the synchronous trainer uses.
        max_chunk_trajectories: split chunks larger than this.
    """

    def __init__(
        self,
        compute_group_advantages: Callable[[list[float]], list[float]],
        *,
        dp_size: int,
        max_chunk_trajectories: int,
        group_only: bool,
        max_open_groups: int,
    ):
        self.compute_group_advantages = compute_group_advantages
        # True: hold each trajectory until its group closes (group-level
        # streaming); False: stream trajectories of open groups into buckets.
        self.group_only = group_only
        # Cap on groups with streamed (bucketed) trajectories at once.
        self.max_open_groups = max_open_groups
        self.dp_size = dp_size
        self.max_chunk_trajectories = max_chunk_trajectories
        self.groups: dict[GroupId, _Group] = {}
        self.ready: list[Trajectory] = []
        self.num_dispatched = 0

    def register_group(self, group: GroupId, size: int) -> None:
        self.groups[group] = _Group(size=size)

    def add(self, traj: Trajectory) -> None:
        g = self.groups[traj.group]
        assert traj.index not in g.rewards, f"duplicate trajectory {traj.index}"
        g.rewards[traj.index] = float(traj.reward)
        if len(g.rewards) == g.size:
            idx = sorted(g.rewards)
            advs = self.compute_group_advantages([g.rewards[i] for i in idx])
            g.advantages = dict(zip(idx, (float(a) for a in advs)))
        self.ready.append(traj)

    def _pending_closes(self) -> list[tuple[GroupId, dict[float, float]]]:
        closes = []
        for gid, g in self.groups.items():
            if g.advantages is not None and g.streamed_open and not g.close_sent:
                adv_by_reward: dict[float, float] = {}
                for i, r in g.rewards.items():
                    a = g.advantages[i]
                    if r in adv_by_reward:
                        assert abs(adv_by_reward[r] - a) <= 1e-6 * max(1.0, abs(a)), (
                            f"group {gid}: equal rewards got different advantages; "
                            "estimator is not group-symmetric"
                        )
                    adv_by_reward[r] = a
                closes.append((gid, adv_by_reward))
                g.close_sent = True
        return closes

    def _dispatchable(self, t: Trajectory) -> bool:
        g = self.groups[t.group]
        if g.advantages is not None or g.streamed_open:
            return True
        if self.group_only:
            return False
        open_groups = sum(
            1 for x in self.groups.values() if x.streamed_open and not x.close_sent
        )
        return open_groups < self.max_open_groups

    def has_work(self) -> bool:
        return any(self._dispatchable(t) for t in self.ready) or any(
            g.advantages is not None and g.streamed_open and not g.close_sent
            for g in self.groups.values()
        )

    def all_done(self) -> bool:
        return (
            not self.has_work()
            and not self.ready
            and all(len(g.rewards) == g.size for g in self.groups.values())
            and self.num_dispatched == sum(g.size for g in self.groups.values())
        )

    def next_dispatch(self) -> Dispatch:
        # Final-first: work of closed groups (final advantages) always goes
        # first; bucket work for open groups only fills otherwise-idle learner
        # time, one open group per dispatch -- preferring one that already
        # holds buckets, then the one with the most ready rows (largest, most
        # efficient chunk). Bucket chunks then never delay closed-group work.
        ready = [t for t in self.ready if self.groups[t.group].advantages is not None]
        if not ready and not self.group_only:
            by_group: dict[GroupId, list[Trajectory]] = {}
            for t in self.ready:
                if self._dispatchable(t):
                    by_group.setdefault(t.group, []).append(t)
            if by_group:
                gid = max(
                    by_group,
                    key=lambda g: (self.groups[g].streamed_open, len(by_group[g])),
                )
                self.groups[gid].streamed_open = True
                ready = by_group[gid]
        taken = {id(t) for t in ready}
        self.ready = [t for t in self.ready if id(t) not in taken]
        final: list[Trajectory] = []
        buckets: dict[tuple[GroupId, float], list[Trajectory]] = {}
        for t in ready:
            g = self.groups[t.group]
            if g.advantages is not None:
                final.append(t)
            else:
                g.streamed_open = True
                buckets.setdefault((t.group, float(t.reward)), []).append(t)

        chunks: list[Chunk] = []
        m = self.max_chunk_trajectories
        for (gid, r), trajs in buckets.items():
            for s in range(0, len(trajs), m):
                chunks.append(
                    Chunk(
                        group=gid,
                        reward=r,
                        trajectories=trajs[s : s + m],
                        advantages=None,
                    )
                )
        # Final trajectories from different groups can share a chunk.
        final.sort(key=lambda t: t.num_tokens)
        for s in range(0, len(final), m):
            trajs = final[s : s + m]
            chunks.append(
                Chunk(
                    group=None,
                    reward=None,
                    trajectories=trajs,
                    advantages=[
                        self.groups[t.group].advantages[t.index] for t in trajs
                    ],
                )
            )

        # Greedy longest-first token balancing over DP ranks.
        per_rank: list[list[Chunk]] = [[] for _ in range(self.dp_size)]
        load = [0] * self.dp_size
        for c in sorted(chunks, key=lambda c: -c.num_tokens):
            r = load.index(min(load))
            per_rank[r].append(c)
            load[r] += c.num_tokens

        self.num_dispatched += sum(len(c.trajectories) for c in chunks)
        # Closes go after this dispatch's chunks (workers apply them last).
        return Dispatch(per_rank=per_rank, closes=self._pending_closes())


def pad_batch_to_multiple(data: BatchedDataDict, multiple: int) -> BatchedDataDict:
    """Append all-masked dummy rows so ``data.size`` is a multiple of ``multiple``.

    Dummy rows have ``sample_mask = 0`` and ``token_mask = 0``, so they add
    neither gradient nor normalization count.
    """
    n = data.size
    pad = (-n) % multiple
    if pad == 0:
        return data
    out = BatchedDataDict()
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            filler = v[:1].expand(pad, *v.shape[1:]).clone()
            if k in ("sample_mask", "token_mask", "advantages"):
                filler.zero_()
            out[k] = torch.cat([v, filler], dim=0)
        elif isinstance(v, list):
            out[k] = v + [v[0]] * pad
        else:
            out[k] = v
    return out


class StreamingLearner:
    """Thin driver-side handle on the ``stream_*`` worker methods."""

    def __init__(self, policy: Policy):
        self.policy = policy
        self.worker_group = policy.worker_group
        self.dp_size = self.worker_group.dp_size

    def begin(
        self,
        loss_fn: LossFunction,
        *,
        gbs: int,
        mbs: int,
        storage_device: str,
        max_buckets_per_group: int,
        max_open_groups: int,
    ) -> None:
        ray.get(
            self.worker_group.run_all_workers_single_data(
                "stream_begin_step",
                loss_fn=loss_fn,
                gbs=gbs,
                mbs=mbs,
                storage_device=storage_device,
                max_buckets_per_group=max_buckets_per_group,
                max_open_groups=max_open_groups,
            )
        )

    def submit(
        self,
        per_rank_items: list[list[dict[str, Any]]],
        closes: list[tuple[GroupId, dict[float, float]]],
    ) -> list[ray.ObjectRef]:
        assert len(per_rank_items) == self.dp_size
        fut = self.worker_group.run_all_workers_sharded_data(
            "stream_accumulate",
            items=per_rank_items,
            closes=[closes] * self.dp_size,
            **_SHARD_AXES,
        )
        return fut.futures

    def finish(self) -> dict[str, Any]:
        results = ray.get(
            self.worker_group.run_all_workers_single_data("stream_finish_step")
        )
        return results[0]

    def abort(self) -> None:
        ray.get(self.worker_group.run_all_workers_single_data("stream_abort_step"))
