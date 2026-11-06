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
"""Finalize overlap (part of ``thundersync.train_overlap``; research/nolanes2/DESIGN.md §13).

Under train overlap (R31 Option A, exact variant) every prompt group of batch k trains the moment its G rows are final,
with its own capture into the learner's order-invariant gradient accumulator. Before a group can train, the driver
finalizes it: it rewrites the group's rows into the data plane and runs the group-local advantage stage.

``FinalizePipeline`` finalizes each group as soon as its rows are final, also while the learner is still verifying
later stream chunks or training other groups. The learner trains from a queue of finalized groups: every group that is
ready when the learner becomes free goes into ONE learner call, which trains them one after the other with one
gradient capture per group (no driver round trip between groups).

Exactness: what each group trains is unchanged (the group's rows in slot order, the same advantages, its own capture).
Only the order of the groups, their batching into learner calls and the wall-clock placement of driver work change.
The order-invariant accumulator makes theta_{k+1} independent of that order (R31). ``trained_once_diff`` checks the
invariant this scheduling could break: every output row of batch k trains exactly once in optimizer step k (no
deferral, P5 / R31 (ii)).

Asyncio only (no torch / ray), so the scheduling is unit-testable anywhere.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from typing import Any, Awaitable, Callable, Iterable


class FinalizePipeline:
    """Finalize groups as their rows become final; the learner drains them in readiness order.

    ``prepare(g, result)`` is the driver-side finalize of group g (rewrite + advantage); it returns any object the
    trainer needs (the group's meta). Prepares run one at a time (a lock), in the order their rows become final.
    """

    def __init__(self, prepare: Callable[[str, Any], Awaitable[Any]]):
        self._prepare = prepare
        self._lock = asyncio.Lock()
        self._ready: asyncio.Queue = asyncio.Queue()
        self._futs: dict[str, asyncio.Future] = {}
        self.order: list[str] = []  # finalize-completion order (= the learner's training order)
        self.prepare_s = 0.0  # driver time spent in prepare (all of it overlaps verify / training / tails)
        self._extra = 0  # resubmitted groups not yet counted by ``drain`` (each trains once more)

    def submit(self, g: str, rows_task: Awaitable[Any]) -> None:
        """Group g's rows are being produced by ``rows_task`` (its continuations / scoring); finalize it when done."""
        if g in self._futs:
            raise RuntimeError(f"[finalize overlap] group {g} submitted twice")
        self._futs[g] = asyncio.ensure_future(self._run(g, rows_task))

    def resubmit(self, g: str, rows_task: Awaitable[Any]) -> None:
        """Group g was handed to the learner and must be finalized and trained again (ThunderSync EDV: its training
        call's capture was dropped and its outputs re-walked): one more group for ``drain``."""
        if g not in self._futs:
            raise RuntimeError(f"[finalize overlap] group {g} resubmitted before its first submission")
        self._extra += 1
        self._futs[g] = asyncio.ensure_future(self._run(g, rows_task))

    async def _run(self, g: str, rows_task: Awaitable[Any]) -> None:
        try:
            res = await rows_task
            async with self._lock:
                t = time.perf_counter()
                out = await self._prepare(g, res)
                self.prepare_s += time.perf_counter() - t
            self.order.append(g)
            self._ready.put_nowait((g, out, None))
        except asyncio.CancelledError:
            raise
        except BaseException as e:  # handed to the consumer, which raises it
            self._ready.put_nowait((g, None, e))

    async def drain(
        self, n_groups: int, train: Callable[[list[tuple[str, Any]]], Awaitable[None]]
    ) -> dict[str, float]:
        """Train ``n_groups`` finalized groups, each once, in readiness order.

        ``train(batch)`` receives every group that is finalized when the learner becomes free, as ``[(g, out), ...]`` in
        readiness order: one learner call trains them all (one gradient capture per group inside the worker), so the
        per-call driver round trip is paid once per batch instead of once per group. A batch holds at least one group;
        groups finalized while a batch trains form the next batch. Returns the learner wait probes:
        ``learner_wait`` = total time the learner had no finalized group to train, ``tail_exposed`` = the last such
        wait (the learner was free and only the last-finishing rows were outstanding)."""
        if n_groups != len(self._futs):
            raise RuntimeError(f"[finalize overlap] draining {n_groups} groups, {len(self._futs)} submitted")
        wait = 0.0
        last_wait = 0.0
        self.batch_sizes: list[int] = []
        left = n_groups
        try:
            while left:
                t = time.perf_counter()
                items = [await self._ready.get()]
                dt = time.perf_counter() - t
                wait += dt
                last_wait = dt
                while len(items) < left and not self._ready.empty():
                    items.append(self._ready.get_nowait())
                for _, _, err in items:
                    if err is not None:
                        raise err
                left -= len(items)
                self.batch_sizes.append(len(items))
                await train([(g, out) for g, out, _ in items])
                left, self._extra = left + self._extra, 0  # groups resubmitted during ``train``
        except BaseException:
            for f in self._futs.values():
                f.cancel()
            raise
        return {"learner_wait": wait, "tail_exposed": last_wait}


def trained_once_diff(trained_ids: Iterable[Any], output_ids: Iterable[Any]) -> tuple[list[str], list[str]]:
    """Every output row of batch k trains exactly once in optimizer step k (P5 / R31 (ii): no row deferred, dropped or
    duplicated, whatever the tails' timing). Compares the sample ids handed to the train calls with the step's
    rewritten output rows; returns (missing, extra-or-duplicated) ids, both empty when the invariant holds."""
    got, want = Counter(trained_ids), Counter(output_ids)
    return sorted(map(str, (want - got).elements())), sorted(map(str, (got - want).elements()))


def step_output_ids(gids: list[str], metas: dict[str, Any], has_valid: dict[str, bool]) -> list[Any]:
    """The step's output rows that must train, built independently of the train calls (R35 1(a)): every group of the
    step was finalized (``metas`` / ``has_valid`` are filled where a group is finalized, not where it is trained), and
    the rows of the groups with valid tokens, in the fixed group order. A scheduling bug that never hands a finalized
    group to the learner, or hands one twice, then shows up as missing / extra in ``trained_once_diff``."""
    if set(metas) != set(gids) or set(has_valid) != set(gids):
        raise RuntimeError(
            f"[trained once check] finalized groups != step groups: metas missing {sorted(set(gids) - set(metas))[:3]} "
            f"extra {sorted(set(metas) - set(gids))[:3]}; has_valid missing {sorted(set(gids) - set(has_valid))[:3]}"
        )
    return [sid for g in gids if has_valid[g] for sid in metas[g].sample_ids]
