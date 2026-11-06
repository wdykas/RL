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
"""Coalesced per-trajectory publishing for trajectory-level gradient streaming."""

import asyncio

import pytest

from nemo_rl.experience.rollout_manager import _TrajectoryPublisher


class _Buffer:
    def __init__(self, fail: bool = False):
        self.commits: list[tuple[str, list[int], int]] = []
        self.fail = fail

    async def commit_trajectories(
        self, group_id, indices, record, start_weight_version
    ):
        if self.fail:
            raise RuntimeError("put failed")
        assert len(record.completions) == len(indices)
        self.commits.append((group_id, list(indices), start_weight_version))


_SAMPLE = {"idx": 3, "message_log": [], "extra_env_info": {}, "task_name": "math"}


def _publisher(buf, coalesce_s):
    return _TrajectoryPublisher(
        tq_buffer=buf,
        group_id="g",
        input_sample=_SAMPLE,
        start_weight_version=5,
        coalesce_s=coalesce_s,
    )


def test_completions_within_window_are_one_write_and_drain_flushes_rest():
    buf = _Buffer()

    async def scenario():
        pub = _publisher(buf, coalesce_s=0.05)
        await pub.add(2, object())
        await pub.add(0, object())
        await asyncio.sleep(0.1)  # window fires: one write for both
        await pub.add(1, object())
        await pub.drain()  # flushes the straggler immediately

    asyncio.run(scenario())
    assert buf.commits == [("g", [2, 0], 5), ("g", [1], 5)]


def test_drain_surfaces_write_errors():
    buf = _Buffer(fail=True)

    async def scenario():
        pub = _publisher(buf, coalesce_s=0.0)
        await pub.add(0, object())
        await asyncio.sleep(0.01)
        with pytest.raises(RuntimeError, match="put failed"):
            await pub.drain()

    asyncio.run(scenario())


def test_cancel_drops_pending_rows():
    buf = _Buffer()

    async def scenario():
        pub = _publisher(buf, coalesce_s=10.0)
        await pub.add(0, object())
        pub.cancel()
        await asyncio.sleep(0.01)

    asyncio.run(scenario())
    assert buf.commits == []
