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
"""Streaming training over the sticky tails (DESIGN.md section 14k): a rollout
with a complete plan is served at once, one whose plan the continuation rounds
complete is served when they do, one never drafted waits for theta_k."""

import asyncio

import pytest
from thundersync_rl.speculative import Plan, SpeculativeGeneration


def _spec():
    spec = SpeculativeGeneration.__new__(SpeculativeGeneration)
    spec._tail_cv = None
    spec._weights_ready = None
    return spec


def _gate(spec, plan):
    """``_one``'s wait, in isolation."""

    async def wait():
        if id(plan) in getattr(spec, "_tail_plans", ()):
            await spec._await_plan(plan)
        elif not plan.complete:
            await spec._weights_ready.wait()
            if spec._tail_error is not None:
                raise RuntimeError("failed") from spec._tail_error

    return wait()


def test_complete_first_then_tails_then_fresh():
    spec = _spec()
    done = Plan(seed=-1, complete=True)
    tail = Plan(seed=-1, prefix=[1, 2])
    fresh = Plan(seed=-1)
    spec._tail_plans = {id(tail)}
    order = []

    async def continue_sticky(pending):
        await asyncio.sleep(0.01)
        order.append("tail rounds")
        for _, p in pending:
            p.complete = True
        await spec._notify_tails()

    spec._continue_sticky = continue_sticky

    async def main():
        spec.begin_streaming_tails()

        async def serve(name, plan):
            await _gate(spec, plan)
            order.append(name)

        tasks = [asyncio.create_task(serve(n, p)) for n, p in
                 (("fresh", fresh), ("tail", tail), ("done", done))]
        await asyncio.sleep(0)
        await spec.complete_tails([((0,), tail)])
        order.append("refit")
        spec.weights_ready()
        await asyncio.gather(*tasks)
        spec.end_streaming_tails()

    asyncio.run(main())
    pos = {name: i for i, name in enumerate(order)}
    assert pos["done"] == 0  # complete plan: served before any tail round
    assert pos["tail rounds"] < pos["tail"]  # served once its rounds complete it
    assert pos["refit"] < pos["fresh"]  # never drafted: waits for theta_k


def test_failed_tails_wake_and_raise():
    spec = _spec()
    tail, fresh = Plan(seed=-1, prefix=[1]), Plan(seed=-1)
    spec._tail_plans = {id(tail)}

    async def boom(pending):
        raise ValueError("engine died")

    spec._continue_sticky = boom

    async def main():
        spec.begin_streaming_tails()
        waiters = [asyncio.create_task(_gate(spec, p)) for p in (tail, fresh)]
        await asyncio.sleep(0)
        with pytest.raises(ValueError):
            await spec.complete_tails([((0,), tail)])
        spec.weights_ready(error=ValueError("engine died"))
        for w in waiters:
            with pytest.raises(RuntimeError):
                await w

    asyncio.run(main())


def _planner(groups, G=2, dp=2):
    from thundersync_rl.streaming import StreamPlanner

    pl = StreamPlanner(
        lambda rewards: [r - sum(rewards) / len(rewards) for r in rewards],
        dp_size=dp,
        max_chunk_trajectories=4,
        max_open_groups=0,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    for g in groups:
        pl.register_group(g, G)
    return pl


def _dispatch_signature(pl):
    out = []
    while pl.has_work():
        d = pl.next_dispatch()
        out.append([[tuple((t.group, t.index) for t in c.trajectories) for c in cs] for cs in d.per_rank])
    return out


def test_canonical_group_order_is_independent_of_arrival():
    from thundersync_rl.streaming import Trajectory

    groups = [(0, p) for p in range(3)]
    rows = [
        Trajectory(group=(0, p), index=2 * p + j, reward=float(j), payload=None, num_tokens=10 + p)
        for p in range(3)
        for j in range(2)
    ]
    sigs = []
    for arrival in (rows, rows[::-1], rows[2:] + rows[:2]):
        pl = _planner(groups)
        pl.set_order([(0, 2), (0, 0), (0, 1)])  # e.g. group 2 ready now, 0 and 1 had tails
        got = []
        for t in arrival:
            pl.add(t)
            while pl.has_work(flush=False):
                got.extend(_dispatch_signature(pl))
        got.extend(_dispatch_signature(pl))
        assert pl.all_done()
        sigs.append(got)
    assert sigs[0] == sigs[1] == sigs[2]
    # One group per dispatch, in the canonical order.
    assert [{r[0][0] for cs in d for r in cs if r} for d in sigs[0]] == [{(0, 2)}, {(0, 0)}, {(0, 1)}]


def test_canonical_order_needs_group_granularity_and_all_groups():
    pl = _planner([(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="every registered group"):
        pl.set_order([(0, 0)])
    pl.max_open_groups = 1
    with pytest.raises(ValueError, match="group granularity"):
        pl.set_order([(0, 0), (0, 1)])
