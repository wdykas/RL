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

import math
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
    # Needed a continuation after verification (``tail_excluded_advantages``).
    tail: bool = False


def tail_excluded_advantages(
    reward: float, others: list[float], normalize_std: bool, eps: float = 1e-6
) -> float:
    """Advantage of one row against the baseline set ``others`` (opt-in estimator).

    ``others`` = the group's OTHER rows that did not need a tail (all other
    rows if none): a function of the other rows' (value, tail flag) pairs
    only, never of this row, so E[(r_i - b_i) grad log p(y_i)] is exactly the
    policy gradient (leave-one-out baseline), and with std normalization a
    positive multiple of it (sigma_i is independent of y_i as well). The tail
    flag itself depends on y (P(tail | y) is a function of W*(y)), so the
    baseline must not change with the row's OWN flag: GRPO's self-inclusive
    mean over a flag-selected subset would be biased.
    """
    if not others:
        return 0.0
    b = sum(others) / len(others)
    adv = reward - b
    if normalize_std and len(others) >= 2:
        var = sum((x - b) ** 2 for x in others) / (len(others) - 1)
        adv /= math.sqrt(var) + eps
    return adv


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
    # All-masked filler that only keeps per-rank forward counts equal: built
    # with sample_mask = token_mask = advantages = 0, so it adds no gradient
    # and no normalization count. Its trajectories are templates, not work.
    padding: bool

    @property
    def num_tokens(self) -> int:
        return sum(t.num_tokens for t in self.trajectories)

    @property
    def zero_advantage(self) -> bool:
        """A final chunk whose every row has advantage 0 (no gradient)."""
        return self.advantages is not None and all(a == 0.0 for a in self.advantages)

    def num_microbatches(self, micro_batch_size: int) -> int:
        """Forward/backward calls the worker runs for this chunk.

        A chunk of at most ``micro_batch_size`` rows runs as one micro-batch;
        a larger one is padded to a multiple of ``micro_batch_size``.
        """
        return max(1, math.ceil(len(self.trajectories) / micro_batch_size))


@dataclass
class Dispatch:
    per_rank: list[list[Chunk]]
    closes: list[tuple[GroupId, dict[float, float]]]

    @property
    def num_trajectories(self) -> int:
        return sum(
            len(c.trajectories) for cs in self.per_rank for c in cs if not c.padding
        )


@dataclass
class _Group:
    size: int
    rewards: dict[int, float] = field(default_factory=dict)  # index -> reward
    # Set when the group closes: {reward: advantage} (equal rewards share one).
    advantage_by_reward: Optional[dict[float, float]] = None
    streaming: bool = False  # holds gradient buckets on the workers
    close_sent: bool = False
    # Tail-excluded baseline (opt-in): rows that need a tail, and the rewards
    # of arrived rows that did not (index -> reward).
    tails: int = 0
    nontail: dict[int, float] = field(default_factory=dict)


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
        micro_batch_size: the learner's training micro-batch size.
        equal_rank_microbatches: give every DP rank the same number of
            micro-batch forward/backward calls per dispatch by appending
            all-masked padding chunks to the short ranks. Required with expert
            parallelism, whose token all-to-all spans DP ranks and hangs if
            their forward counts differ.
        skip_zero_advantage_chunks: the learner drops all-zero-advantage
            chunks without a forward (``loss_fn.skip_zero_advantage_rows``
            without expert parallelism). Ranks are then balanced by the tokens
            of live chunks only, since a skipped chunk costs ~nothing; total
            tokens break ties so skipped chunks still spread over ranks.
        min_pipeline_microbatches: under pipeline parallelism, hold rows of
            closed groups back until they make at least this many micro-batches
            per DP rank (``dp_size`` times as many in total), so the rank's one
            pipeline call per dispatch has enough micro-batches to amortize
            the fill/drain bubble. Counted as the worker will run them
            (chunks of ``max_chunk_trajectories`` rows, ``micro_batch_size``
            rows per micro-batch; skipped all-zero-advantage chunks count 0).
            Held rows never stall the step: ``flush=True`` (the default of
            :meth:`has_work` / :meth:`next_dispatch`, used once the step's
            rollouts are done) ignores the threshold. Open-group bucket work
            and pending group closes are not held. 0 disables holding.
    """

    def __init__(
        self,
        compute_group_advantages: Callable[[list[float]], list[float]],
        *,
        dp_size: int,
        max_chunk_trajectories: int,
        max_open_groups: int,
        micro_batch_size: int,
        equal_rank_microbatches: bool,
        skip_zero_advantage_chunks: bool = False,
        min_pipeline_microbatches: int = 0,
        tail_excluded_baseline: Optional[Callable[[float, list[float]], float]] = None,
    ):
        if tail_excluded_baseline is not None and max_open_groups > 0:
            raise ValueError("the tail-excluded baseline closes groups early; no open-group buckets")
        self.tail_excluded_baseline = tail_excluded_baseline
        self.compute_group_advantages = compute_group_advantages
        self.min_pipeline_microbatches = min_pipeline_microbatches
        self.skip_zero_advantage_chunks = skip_zero_advantage_chunks
        self.max_open_groups = max_open_groups
        self.dp_size = dp_size
        self.max_chunk_trajectories = max_chunk_trajectories
        self.micro_batch_size = micro_batch_size
        self.equal_rank_microbatches = equal_rank_microbatches
        self.groups: dict[GroupId, _Group] = {}
        self.ready: list[Trajectory] = []
        self.num_dispatched = 0

    def register_group(self, group: GroupId, size: int, tails: int = 0) -> None:
        """``tails``: rows of the group that need a continuation (tail-excluded baseline)."""
        self.groups[group] = _Group(size=size, tails=tails)

    def add(self, traj: Trajectory) -> None:
        g = self.groups[traj.group]
        assert traj.index not in g.rewards, f"duplicate trajectory {traj.index}"
        g.rewards[traj.index] = float(traj.reward)
        if self.tail_excluded_baseline is not None:
            if not traj.tail:
                g.nontail[traj.index] = float(traj.reward)
                if len(g.nontail) > g.size - g.tails:
                    raise RuntimeError(f"group {traj.group}: more non-tail rows than registered")
            # Every row's baseline set is known once the non-tail rows are in
            # (with fewer than 2 of them, some rows fall back to all others).
            need = g.size if g.size - g.tails < 2 else g.size - g.tails
            done = len(g.rewards) if g.size - g.tails < 2 else len(g.nontail)
            if done == need and g.advantage_by_reward is None:
                g.advantage_by_reward = {}  # closed: advantages per row (_advantage)
            self.ready.append(traj)
            return
        if len(g.rewards) == g.size:
            rewards = [g.rewards[i] for i in sorted(g.rewards)]
            advantages = self.compute_group_advantages(rewards)
            g.advantage_by_reward = dict(zip(rewards, map(float, advantages)))
        self.ready.append(traj)

    def _advantage(self, t: Trajectory) -> float:
        g = self.groups[t.group]
        if self.tail_excluded_baseline is not None:
            others = [r for i, r in g.nontail.items() if i != t.index]
            if not others:
                others = [r for i, r in g.rewards.items() if i != t.index]
            return float(self.tail_excluded_baseline(float(t.reward), others))
        return g.advantage_by_reward[t.reward]

    def final_microbatches(self, trajs: list[Trajectory]) -> int:
        """Micro-batches the learner runs for these closed-group rows.

        Mirrors :meth:`next_dispatch`'s chunking (fused rows: one-row chunks;
        the rest split by zero/nonzero advantage into chunks of at most
        ``max_chunk_trajectories`` rows) and the worker's micro-batching
        (``ceil(rows / micro_batch_size)`` per chunk). With zero-advantage
        skipping an all-zero chunk runs nothing and counts 0.
        """
        n = sum(1 for t in trajs if t.fused is not None)
        rest = [t for t in trajs if t.fused is None]
        m = self.max_chunk_trajectories
        for zero in (False, True):
            if zero and self.skip_zero_advantage_chunks:
                continue
            k = sum(1 for t in rest if (self._advantage(t) == 0.0) == zero)
            for s in range(0, k, m):
                n += math.ceil(min(m, k - s) / self.micro_batch_size)
        return n

    def holding(self) -> bool:
        """Whether closed-group rows are held back for the pipeline threshold."""
        if self.min_pipeline_microbatches <= 0:
            return False
        final = [
            t
            for t in self.ready
            if self.groups[t.group].advantage_by_reward is not None
        ]
        return bool(final) and (
            self.final_microbatches(final)
            < self.min_pipeline_microbatches * self.dp_size
        )

    def _pick(self, flush: bool = True) -> list[Trajectory]:
        rows = self.ready
        if not flush and self.holding():
            # Rows of closed groups wait; idle time still goes to bucket work.
            rows = [t for t in rows if self.groups[t.group].advantage_by_reward is None]
        return pick_final_first(
            [(t.group, t.reward, t) for t in rows],
            closed={gid for gid, g in self.groups.items() if g.advantage_by_reward is not None},
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

    def has_work(self, flush: bool = True) -> bool:
        """Whether :meth:`next_dispatch` has anything to send.

        ``flush=False`` (rollouts still running) applies
        ``min_pipeline_microbatches``; ``flush=True`` sends whatever is ready.
        """
        return bool(self._pick(flush)) or any(
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

    def next_dispatch(self, flush: bool = True) -> Dispatch:
        picked = self._pick(flush)
        taken = {id(t) for t in picked}
        self.ready = [t for t in self.ready if id(t) not in taken]
        final: list[Trajectory] = []
        buckets: dict[tuple[GroupId, float], list[Trajectory]] = {}
        chunks: list[Chunk] = []

        advantage = self._advantage

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
                            padding=False,
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
                            padding=False,
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
                        padding=False,
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
                        padding=False,
                    )
                )

        # Fused chunks run where their graph lives; the rest are balanced
        # greedily, longest first, by (live tokens, total tokens): a chunk the
        # worker skips costs no forward when skipping is on.
        for c in chunks:
            if c.fused is not None and not 0 <= c.fused.dp_rank < self.dp_size:
                raise ValueError(
                    f"fused trajectory pinned to DP rank {c.fused.dp_rank} "
                    f"of {self.dp_size}"
                )
        per_rank = balance_chunks(
            chunks,
            self.dp_size,
            skip_zero_advantage=self.skip_zero_advantage_chunks,
            pinned_rank=lambda c: None if c.fused is None else c.fused.dp_rank,
        )
        if self.equal_rank_microbatches and chunks:
            self._pad_to_equal_microbatches(per_rank, chunks)

        self.num_dispatched += sum(len(c.trajectories) for c in chunks)
        # Closes go after this dispatch's chunks (workers apply them last).
        return Dispatch(per_rank=per_rank, closes=self._pending_closes())

    def _pad_to_equal_microbatches(
        self, per_rank: list[list[Chunk]], chunks: list[Chunk]
    ) -> None:
        """Append one-row padding chunks until every rank runs as many micro-batches.

        Each padding chunk is a single micro-batch built from the dispatch's
        shortest trajectory (cheapest forward) and fully masked out.
        """
        mbs = self.micro_batch_size
        counts = [sum(c.num_microbatches(mbs) for c in cs) for cs in per_rank]
        target = max(counts)
        template = min(
            (t for c in chunks for t in c.trajectories), key=lambda t: t.num_tokens
        )
        for cs, count in zip(per_rank, counts):
            cs.extend(
                Chunk(
                    group=None,
                    reward=None,
                    trajectories=[template],
                    advantages=[0.0],
                    fused=None,
                    padding=True,
                )
                for _ in range(target - count)
            )


def pipeline_bubble_fraction(microbatches_per_call: list[int], pp: int) -> float:
    """Idle fraction of a pipeline stage over a sequence of schedule calls.

    Each 1F1B (non-interleaved) call of ``m`` micro-batches takes
    ``m + pp - 1`` micro-batch slots (forward + backward of one micro-batch
    per slot, uniform cost), of which ``m`` are busy, so one call idles
    ``(pp - 1) / (m + pp - 1)``. ``k`` calls idle ``k (pp - 1)`` slots.
    """
    calls = [m for m in microbatches_per_call if m > 0]
    if not calls:
        return 0.0
    busy = sum(calls)
    return 1.0 - busy / (busy + len(calls) * (pp - 1))


def balance_chunks(
    chunks: list[Chunk],
    dp_size: int,
    *,
    skip_zero_advantage: bool,
    pinned_rank: Optional[Callable[[Chunk], Optional[int]]] = None,
) -> list[list[Chunk]]:
    """Greedy longest-processing-time assignment of chunks to DP ranks.

    A chunk's cost is ``(live tokens, tokens)``, compared lexicographically,
    where live tokens are its tokens unless the worker skips it (an
    all-zero-advantage chunk with ``skip_zero_advantage``). Without skipping
    this is the plain token balance. ``pinned_rank`` places some chunks on a
    fixed rank before the rest are balanced. Only which rank backpropagates a chunk
    changes, so the step's gradient is the same up to fp32 summation order.
    """

    def cost(c: Chunk) -> tuple[int, int]:
        skipped = skip_zero_advantage and c.zero_advantage
        return (0 if skipped else c.num_tokens, c.num_tokens)

    per_rank: list[list[Chunk]] = [[] for _ in range(dp_size)]
    load = [(0, 0)] * dp_size
    free = []
    for c in chunks:
        # Chunks pinned to a rank (fused: the rank holding their graph) first.
        r = None if pinned_rank is None else pinned_rank(c)
        if r is None:
            free.append(c)
            continue
        per_rank[r].append(c)
        live, total = cost(c)
        load[r] = (load[r][0] + live, load[r][1] + total)
    for c in sorted(free, key=cost, reverse=True):
        r = load.index(min(load))
        per_rank[r].append(c)
        live, total = cost(c)
        load[r] = (load[r][0] + live, load[r][1] + total)
    return per_rank


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
        merge_pipeline_calls: bool = True,
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
                merge_pipeline_calls=merge_pipeline_calls,
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
