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
"""Finalize overlap (finalize_pipeline.py): every group is finalized once and trained once, in readiness order;
finalize work runs while the learner trains (overlap is real); groups ready when the learner is free share one learner
call; errors reach the learner; the learner-wait probes; the trained-once invariant (P5) and its negative."""

import asyncio
import random

import pytest

from thundersync_rl.finalize_pipeline import FinalizePipeline, trained_once_diff


def _run(coro):
    return asyncio.run(coro)


def _each(train):
    """Batch callback that trains the groups of each learner call one after the other (as the worker does)."""

    async def batch(items):
        for g, out in items:
            await train(g, out)

    return batch


async def _rows(delay: float, value):
    await asyncio.sleep(delay)
    return value


def test_every_group_finalized_and_trained_once_in_readiness_order():
    async def main():
        prepared, trained = [], []

        async def prepare(g, res):
            prepared.append(g)
            await asyncio.sleep(0.001)
            return ("meta", g, res)

        async def train(g, out):
            assert out == ("meta", g, f"rows-{g}")
            trained.append(g)

        pipe = FinalizePipeline(prepare)
        delays = {"g0": 0.05, "g1": 0.01, "g2": 0.03, "g3": 0.0}
        for g, d in delays.items():
            pipe.submit(g, asyncio.ensure_future(_rows(d, f"rows-{g}")))
        probes = await pipe.drain(len(delays), _each(train))
        return prepared, trained, pipe.order, probes

    prepared, trained, order, probes = _run(main())
    assert sorted(trained) == ["g0", "g1", "g2", "g3"]
    assert trained == order == prepared == ["g3", "g1", "g2", "g0"]  # readiness order, not submit order
    assert probes["learner_wait"] >= probes["tail_exposed"] >= 0.0


def test_finalize_runs_while_the_learner_trains():
    """The point of the flag: group B's finalize happens during group A's train call, not after it."""

    async def main():
        events = []

        async def prepare(g, res):
            events.append(("prep_start", g))
            await asyncio.sleep(0.02)
            events.append(("prep_end", g))
            return g

        async def train(g, out):
            events.append(("train_start", g))
            await asyncio.sleep(0.1)  # the learner is busy (to_thread in the stage)
            events.append(("train_end", g))

        pipe = FinalizePipeline(prepare)
        pipe.submit("A", asyncio.ensure_future(_rows(0.0, None)))
        pipe.submit("B", asyncio.ensure_future(_rows(0.03, None)))
        await pipe.drain(2, _each(train))
        return events

    ev = _run(main())
    i = {e: n for n, e in enumerate(ev)}
    assert i[("prep_end", "B")] < i[("train_end", "A")], ev  # B finalized under A's training
    assert i[("train_start", "B")] > i[("train_end", "A")]  # one train call at a time


def test_groups_ready_while_the_learner_trains_share_one_call():
    """Lever 1 (PERF_RECONCILE_1102): every group finalized while a call runs goes into the next call, in readiness
    order, so the driver round trip is paid per call, not per group. Groups are never split or repeated."""

    async def main():
        calls = []

        async def prepare(g, res):
            return g

        async def train(items):
            calls.append([g for g, _ in items])
            await asyncio.sleep(0.05)

        pipe = FinalizePipeline(prepare)
        pipe.submit("A", asyncio.ensure_future(_rows(0.0, None)))
        for n, d in enumerate((0.01, 0.03, 0.02)):  # ready during A's call: C2 before C1
            pipe.submit(f"C{n}", asyncio.ensure_future(_rows(d, None)))
        pipe.submit("Z", asyncio.ensure_future(_rows(0.2, None)))  # after the second call: its own call
        await pipe.drain(5, train)
        return calls, pipe.batch_sizes, pipe.order

    calls, sizes, order = _run(main())
    assert calls == [["A"], ["C0", "C2", "C1"], ["Z"]]
    assert sizes == [1, 3, 1] and [g for c in calls for g in c] == order


def test_finalize_is_serialized_one_group_at_a_time():
    async def main():
        live, peak = [0], [0]

        async def prepare(g, res):
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            await asyncio.sleep(0.005)
            live[0] -= 1
            return g

        async def train(g, out):
            return None

        pipe = FinalizePipeline(prepare)
        for n in range(8):
            pipe.submit(f"g{n}", asyncio.ensure_future(_rows(0.0, n)))
        await pipe.drain(8, _each(train))
        return peak[0]

    assert _run(main()) == 1


def test_prepare_error_reaches_the_learner_and_cancels_the_rest():
    async def main():
        async def prepare(g, res):
            if g == "bad":
                raise ValueError("rewrite failed")
            return g

        async def train(g, out):
            return None

        pipe = FinalizePipeline(prepare)
        pipe.submit("ok", asyncio.ensure_future(_rows(0.0, None)))
        pipe.submit("bad", asyncio.ensure_future(_rows(0.01, None)))
        pipe.submit("slow", asyncio.ensure_future(_rows(10.0, None)))
        with pytest.raises(ValueError, match="rewrite failed"):
            await pipe.drain(3, _each(train))

    _run(main())


def test_rows_task_error_reaches_the_learner():
    async def boom():
        await asyncio.sleep(0.0)
        raise RuntimeError("continuation failed")

    async def main():
        async def prepare(g, res):
            return g

        async def train(g, out):
            return None

        pipe = FinalizePipeline(prepare)
        pipe.submit("x", asyncio.ensure_future(boom()))
        with pytest.raises(RuntimeError, match="continuation failed"):
            await pipe.drain(1, _each(train))

    _run(main())


def test_double_submit_and_count_mismatch_refused():
    async def main():
        async def prepare(g, res):
            return g

        pipe = FinalizePipeline(prepare)
        pipe.submit("a", asyncio.ensure_future(_rows(0.0, None)))
        with pytest.raises(RuntimeError, match="submitted twice"):
            pipe.submit("a", asyncio.ensure_future(_rows(0.0, None)))
        with pytest.raises(RuntimeError, match="draining 2 groups, 1 submitted"):
            await pipe.drain(2, _each(prepare))
        await pipe.drain(1, lambda items: asyncio.sleep(0))

    _run(main())


def test_tail_exposed_is_the_final_learner_wait():
    async def main():
        async def prepare(g, res):
            return g

        async def train(g, out):
            await asyncio.sleep(0.01)

        pipe = FinalizePipeline(prepare)
        pipe.submit("early", asyncio.ensure_future(_rows(0.0, None)))
        pipe.submit("tail", asyncio.ensure_future(_rows(0.2, None)))  # the long tail
        return await pipe.drain(2, _each(train))

    p = _run(main())
    assert 0.15 < p["tail_exposed"] < 0.3
    assert p["learner_wait"] >= p["tail_exposed"]


def test_random_readiness_trains_each_group_exactly_once():
    for seed in range(5):
        rng = random.Random(seed)

        async def main():
            trained = []

            async def prepare(g, res):
                await asyncio.sleep(rng.random() * 0.002)
                return [f"{g}/{i}" for i in range(4)]

            async def train(g, ids):
                await asyncio.sleep(rng.random() * 0.002)
                trained.extend(ids)

            pipe = FinalizePipeline(prepare)
            gs = [f"g{n}" for n in range(16)]
            for g in gs:
                pipe.submit(g, asyncio.ensure_future(_rows(rng.random() * 0.01, None)))
            await pipe.drain(len(gs), _each(train))
            return trained, [f"{g}/{i}" for g in gs for i in range(4)]

        trained, outputs = _run(main())
        assert trained_once_diff(trained, outputs) == ([], [])


def test_trained_once_diff_detects_deferral_and_duplication():
    outputs = ["a0", "a1", "b0", "b1", "c0", "c1"]
    assert trained_once_diff(outputs[::-1], outputs) == ([], [])
    # drop_late_group: the late group c is not trained, a trains twice in its place (same row count)
    missing, extra = trained_once_diff(["a0", "a1", "b0", "b1", "a0", "a1"], outputs)
    assert missing == ["c0", "c1"] and extra == ["a0", "a1"]
    missing, extra = trained_once_diff(["a0", "a1", "b0", "b1"], outputs)
    assert missing == ["c0", "c1"] and extra == []


def test_trained_once_check_is_independent_of_the_train_calls():
    """R35 1(a): the expected rows come from the finalize side (every step group's meta + has_valid), not from the
    train callback. A scheduler bug in which one finalized group never reaches the learner must fail the check; the
    old construction (expected rows appended inside the train callback) passed it, which is the point of the test."""
    from types import SimpleNamespace

    from thundersync_rl.finalize_pipeline import step_output_ids

    async def main():
        gids = [f"g{n}" for n in range(6)]
        metas, has_valid = {}, {}
        trained, old_expected = [], []

        async def prepare(g, res):  # stage-shaped finalize: records meta + has_valid where the group is finalized
            m = SimpleNamespace(sample_ids=[f"{g}/{i}" for i in range(4)])
            metas[g], has_valid[g] = m, g != "g4"  # g4: no valid tokens (trains nothing, expects nothing)
            return m, has_valid[g]

        async def buggy_train(g, out):  # drops g2 although it was finalized
            m, hv = out
            if not hv or g == "g2":
                return
            old_expected.extend(m.sample_ids)
            trained.extend(m.sample_ids)

        pipe = FinalizePipeline(prepare)
        for n, g in enumerate(gids):
            pipe.submit(g, asyncio.ensure_future(_rows(0.001 * (6 - n), None)))
        await pipe.drain(len(gids), _each(buggy_train))
        return gids, metas, has_valid, trained, old_expected

    gids, metas, has_valid, trained, old_expected = _run(main())
    assert trained_once_diff(trained, old_expected) == ([], [])  # the old check is blind to the dropped group
    missing, extra = trained_once_diff(trained, step_output_ids(gids, metas, has_valid))
    assert missing == [f"g2/{i}" for i in range(4)] and extra == []
    # an all-invalid step expects nothing and trains nothing: no error (R35 1(b))
    none_valid = {g: False for g in gids}
    assert trained_once_diff([], step_output_ids(gids, metas, none_valid)) == ([], [])
    # a step group that was never finalized is refused outright
    with pytest.raises(RuntimeError, match="finalized groups != step groups"):
        step_output_ids(gids + ["g9"], metas, has_valid)
