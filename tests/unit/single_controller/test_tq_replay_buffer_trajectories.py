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
"""TQReplayBuffer trajectory-level publishing (gradient streaming)."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.async_utils.replay_buffer as _replay_buffer_module
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord
from tests.unit.single_controller.test_tq_replay_buffer import FakeDataPlaneClient

_N = 3


def _stub_record_to_train_batch(
    record, *, pad_value_dict, include_message_violation_fields
):
    del pad_value_dict, include_message_violation_fields
    n = len(record.completions)
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "input_lengths": torch.full((n,), 4, dtype=torch.long),
            "total_reward": torch.zeros(n, dtype=torch.float32),
        }
    )


@pytest.fixture(autouse=True)
def _patch_converter(monkeypatch):
    monkeypatch.setattr(
        _replay_buffer_module, "record_to_train_batch", _stub_record_to_train_batch
    )


def _record(n: int, metrics: dict | None = None) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=7,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[object() for _ in range(n)],
        rollout_metrics=dict(metrics or {}),
    )


def _claim_all(buf: TQReplayBuffer, target_step: int, max_rows: int = 1 << 30):
    rows = buf.peek_trajectory_rows(target_step=target_step)[:max_rows]
    for gid, meta in rows:
        buf.claim_trajectory_sample_ids(gid, list(meta.sample_ids))
    return rows


def _buffer(dp: FakeDataPlaneClient) -> TQReplayBuffer:
    buf = TQReplayBuffer(
        dp,
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        require_routed_experts=False,
    )
    buf.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    return buf


def test_trajectories_claimable_before_group_completes_then_sealed():
    dp = FakeDataPlaneClient()
    buf = _buffer(dp)
    gid = buf.reserve(weight_version=0, target_step=0)

    async def scenario():
        await buf.commit_trajectories(gid, [2], _record(1), start_weight_version=0)
        rows = _claim_all(buf, 0, 10)
        assert [m.sample_ids for _, m in rows] == [[f"{gid}_g2"]]
        assert buf.ready_list == [False]  # not selectable by group samplers
        assert _claim_all(buf, 0, 10) == []

        await buf.commit_trajectories(gid, [0], _record(1), start_weight_version=0)
        await buf.commit_trajectories(gid, [1], _record(1), start_weight_version=0)
        rows = _claim_all(buf, 0, 1)
        assert len(rows) == 1  # max_rows respected
        assert not buf.trajectory_group_fully_claimed(gid)

        meta = await buf.seal_group(gid, _record(_N, {"m": 1}), end_weight_version=0)
        assert meta.sample_ids == [f"{gid}_g{i}" for i in range(_N)]
        assert meta.extra_info[ROLLOUT_METRICS] == [{"m": 1}]
        assert buf.ready_list == [True]
        assert not buf.trajectory_group_fully_claimed(gid)
        _claim_all(buf, 0, 10)
        assert buf.trajectory_group_fully_claimed(gid)

    asyncio.run(scenario())
    # every row written exactly once, one at a time
    assert [c["sample_ids"] for c in dp.put_calls] == [
        [f"{gid}_g2"],
        [f"{gid}_g0"],
        [f"{gid}_g1"],
    ]


def test_other_steps_rows_are_not_claimed():
    dp = FakeDataPlaneClient()
    buf = _buffer(dp)
    g0 = buf.reserve(weight_version=0, target_step=0)
    g1 = buf.reserve(weight_version=0, target_step=1)

    async def scenario():
        await buf.commit_trajectories(g0, [0], _record(1), start_weight_version=0)
        await buf.commit_trajectories(g1, [0], _record(1), start_weight_version=0)

    asyncio.run(scenario())
    rows = _claim_all(buf, 1, 10)
    assert [g for g, _ in rows] == [g1]


def test_seal_rejects_missing_trajectory():
    dp = FakeDataPlaneClient()
    buf = _buffer(dp)
    gid = buf.reserve(weight_version=0, target_step=0)

    async def scenario():
        await buf.commit_trajectories(gid, [0], _record(1), start_weight_version=0)
        with pytest.raises(ValueError, match="expected"):
            await buf.seal_group(gid, _record(_N), end_weight_version=0)

    asyncio.run(scenario())


def test_duplicate_trajectory_rejected_and_partial_group_removal_clears_rows():
    dp = FakeDataPlaneClient()
    buf = _buffer(dp)
    gid = buf.reserve(weight_version=0, target_step=0)

    async def scenario():
        await buf.commit_trajectories(gid, [0], _record(1), start_weight_version=0)
        with pytest.raises(ValueError, match="duplicate"):
            await buf.commit_trajectories(gid, [0], _record(1), start_weight_version=0)
        # The rejected duplicate must not touch the original row.
        assert f"{gid}_g0" in dp._rows
        assert dp.clear_calls == []
        await buf.commit_trajectories(gid, [1], _record(1), start_weight_version=0)
        removed = await buf.remove_group(gid, remove_in_dp=True)
        assert removed == 1

    asyncio.run(scenario())
    cleared = [sid for call in dp.clear_calls for sid in call]
    assert f"{gid}_g0" in cleared and f"{gid}_g1" in cleared
    assert buf._group_ids == []


def test_multi_row_commit_is_one_put_and_rejects_overlap():
    dp = FakeDataPlaneClient()
    buf = _buffer(dp)
    gid = buf.reserve(weight_version=0, target_step=0)

    async def scenario():
        await buf.commit_trajectories(gid, [2, 0], _record(2), start_weight_version=0)
        with pytest.raises(ValueError, match="duplicate"):
            await buf.commit_trajectories(
                gid, [1, 2], _record(2), start_weight_version=0
            )
        await buf.commit_trajectories(gid, [1], _record(1), start_weight_version=0)
        meta = await buf.seal_group(gid, _record(_N), end_weight_version=0)
        assert meta.sample_ids == [f"{gid}_g{i}" for i in range(_N)]

    asyncio.run(scenario())
    assert [c["sample_ids"] for c in dp.put_calls][0] == [f"{gid}_g2", f"{gid}_g0"]
