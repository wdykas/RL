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

from nemo_rl.algorithms.grad_streaming import pick_final_first
from nemo_rl.algorithms.loss.interfaces import LossFunction
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy
from thundersync_rl.fused_train import FusedRef

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
    # Fused verification + training: the retained verification forward this
    # trajectory trains from (None: regular training forward).
    fused: Optional[FusedRef] = None


@dataclass
class Chunk:
    """One backward unit sent to one DP rank."""

    group: Optional[GroupId]  # None => final advantages are baked in
    reward: Optional[float]
    trajectories: list[Trajectory]
    advantages: Optional[list[float]]  # final chunks only; None for bucket chunks
    # Set for a single-row chunk trained from a retained verification forward;
    # it must run on ``fused.dp_rank``, which holds the graph.
    fused: Optional[FusedRef]

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
    # Set when the group closes: {reward: advantage} (equal rewards share one).
    advantage_by_reward: Optional[dict[float, float]] = None
    streaming: bool = False  # holds gradient buckets on the workers
    close_sent: bool = False


class StreamPlanner:
    """Turns trajectory arrivals into work-conserving backward dispatches.

    What to dispatch is decided by :func:`pick_final_first` (shared with the
    SingleController): rows of closed groups first, with their final
    advantages; otherwise rows of one open group, which go into
    (group, reward) bucket chunks with advantage 1 and are folded in when the
    group closes. ``max_open_groups=0`` never streams an open group, which is
    group-level streaming.

    Args:
        compute_group_advantages: maps the full reward list of a closed group
            to per-trajectory advantages; must be the same function the
            synchronous trainer uses.
        dp_size: number of data-parallel ranks to balance chunks over.
        max_chunk_trajectories: split chunks larger than this.
        max_open_groups: cap on groups holding buckets (0 = group granularity).
    """

    def __init__(
        self,
        compute_group_advantages: Callable[[list[float]], list[float]],
        *,
        dp_size: int,
        max_chunk_trajectories: int,
        max_open_groups: int,
    ):
        self.compute_group_advantages = compute_group_advantages
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
            rewards = [g.rewards[i] for i in sorted(g.rewards)]
            advantages = self.compute_group_advantages(rewards)
            g.advantage_by_reward = dict(zip(rewards, map(float, advantages)))
        self.ready.append(traj)

    def _pick(self) -> list[Trajectory]:
        return pick_final_first(
            [(t.group, t.reward, t) for t in self.ready],
            closed={gid for gid, g in self.groups.items() if g.advantage_by_reward},
            streaming={
                gid
                for gid, g in self.groups.items()
                if g.streaming and not g.close_sent
            },
            max_open_groups=self.max_open_groups,
            multiple_of=1,
        )

    def _pending_closes(self) -> list[tuple[GroupId, dict[float, float]]]:
        closes = []
        for gid, g in self.groups.items():
            if g.advantage_by_reward is not None and g.streaming and not g.close_sent:
                closes.append((gid, g.advantage_by_reward))
                g.close_sent = True
        return closes

    def has_work(self) -> bool:
        return bool(self._pick()) or any(
            g.advantage_by_reward is not None and g.streaming and not g.close_sent
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
        picked = self._pick()
        taken = {id(t) for t in picked}
        self.ready = [t for t in self.ready if id(t) not in taken]
        final: list[Trajectory] = []
        buckets: dict[tuple[GroupId, float], list[Trajectory]] = {}
        chunks: list[Chunk] = []

        def advantage(t: Trajectory) -> float:
            return self.groups[t.group].advantage_by_reward[t.reward]

        for t in picked:
            g = self.groups[t.group]
            if g.advantage_by_reward is not None:
                if t.fused is not None:
                    chunks.append(
                        Chunk(
                            group=None,
                            reward=None,
                            trajectories=[t],
                            advantages=[advantage(t)],
                            fused=t.fused,
                        )
                    )
                else:
                    final.append(t)
            else:
                g.streaming = True
                if t.fused is not None:
                    chunks.append(
                        Chunk(
                            group=t.group,
                            reward=float(t.reward),
                            trajectories=[t],
                            advantages=None,
                            fused=t.fused,
                        )
                    )
                else:
                    buckets.setdefault((t.group, float(t.reward)), []).append(t)

        m = self.max_chunk_trajectories
        for (gid, r), trajs in buckets.items():
            for s in range(0, len(trajs), m):
                chunks.append(
                    Chunk(
                        group=gid,
                        reward=r,
                        trajectories=trajs[s : s + m],
                        advantages=None,
                        fused=None,
                    )
                )

        # Final trajectories from different groups can share a chunk (length-
        # sorted). Zero-advantage rows are chunked separately: with
        # loss_fn.skip_zero_advantage_rows the worker counts and drops an
        # all-zero chunk without a forward, and no chunk mixes the two kinds.
        final.sort(key=lambda t: t.num_tokens)
        for rows in (
            [t for t in final if advantage(t) != 0.0],
            [t for t in final if advantage(t) == 0.0],
        ):
            for s in range(0, len(rows), m):
                trajs = rows[s : s + m]
                chunks.append(
                    Chunk(
                        group=None,
                        reward=None,
                        trajectories=trajs,
                        advantages=[advantage(t) for t in trajs],
                        fused=None,
                    )
                )

        # Fused chunks run where their graph lives; the rest are balanced
        # greedily, longest first, by tokens over the DP ranks.
        per_rank: list[list[Chunk]] = [[] for _ in range(self.dp_size)]
        load = [0] * self.dp_size
        for c in chunks:
            if c.fused is not None:
                if not 0 <= c.fused.dp_rank < self.dp_size:
                    raise ValueError(
                        f"fused trajectory pinned to DP rank {c.fused.dp_rank} "
                        f"of {self.dp_size}"
                    )
                per_rank[c.fused.dp_rank].append(c)
                load[c.fused.dp_rank] += c.num_tokens
        for c in sorted(
            (c for c in chunks if c.fused is None), key=lambda c: -c.num_tokens
        ):
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
