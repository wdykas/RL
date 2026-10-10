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
"""Train-pump trajectory streaming reproduces the batch-synchronous gradient.

Drives the real SingleController selection / training / close helpers and a
real TQReplayBuffer. The fake trainer runs the real StreamingGroupAccumulator
on random per-trajectory gradients H_i, so the test checks that the pump's
claims, buckets, advantages, closes and retry discards add up to
sum_i a_i H_i with a_i from the full group (exactly what the synchronous
trainer computes).
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from tensordict import TensorDict

import nemo_rl.algorithms.async_utils.replay_buffer as _replay_buffer_module
from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.algorithms.grad_streaming import StreamingGroupAccumulator
from nemo_rl.algorithms.grpo import GRPOConfig
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    _TrajectoryStepState,
)
from nemo_rl.algorithms.single_controller_utils.config import AdvantageConfig
from nemo_rl.data_plane.schema import STREAM_BUCKET_TAG
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord
from tests.unit.single_controller.test_tq_replay_buffer import FakeDataPlaneClient

_G = 4  # trajectories per group
_P = [9, 4]  # two fake gradient buffers


def _stub_record_to_train_batch(
    record, *, pad_value_dict, include_message_violation_fields
):
    del pad_value_dict, include_message_violation_fields
    n = len(record.completions)
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((n, 4), dtype=torch.long),
            "input_lengths": torch.full((n,), 4, dtype=torch.long),
            "total_reward": torch.tensor([c.reward for c in record.completions]),
        }
    )


@pytest.fixture(autouse=True)
def _patch_converter(monkeypatch):
    monkeypatch.setattr(
        _replay_buffer_module, "record_to_train_batch", _stub_record_to_train_batch
    )


class _DataPlane(FakeDataPlaneClient):
    """Serves advantage inputs for sample ids from a test-owned reward table."""

    def __init__(
        self, rewards: dict[str, float], truncated: frozenset[str] = frozenset()
    ):
        super().__init__()
        self.rewards = rewards
        self.truncated = truncated

    def get_samples(self, sample_ids, partition_id, select_fields=None):
        n = len(sample_ids)
        cols = {
            "total_reward": torch.tensor([self.rewards[s] for s in sample_ids]),
            "prompt_ids_for_adv": torch.zeros(n, 1, dtype=torch.long),
            "token_mask": torch.ones(n, 4),
            "sample_mask": torch.ones(n),
            "mask_sample": torch.zeros(n, dtype=torch.bool),
            "truncated": torch.tensor([s in self.truncated for s in sample_ids]),
        }
        return TensorDict({k: cols[k] for k in select_fields}, batch_size=[n])


class _Trainer:
    """Applies the real accumulator to fake per-trajectory gradients."""

    def __init__(self, H: dict[str, list[torch.Tensor]], dp: int):
        self.H = H
        self.grads = [torch.zeros(p, dtype=torch.float64) for p in _P]
        self.acc = StreamingGroupAccumulator(
            self.grads,
            storage_device="cpu",
            max_buckets_per_group=2,
            storage_dtype=torch.float64,
        )
        self.row_advantages: dict[str, float] = {}
        self.sharding_annotations = SimpleNamespace(get_axis_size=lambda axis: dp)

    def train_microbatches_from_meta(self, meta, train_fields):
        del train_fields
        by_bucket: dict[Any, list[str]] = {}
        for sid, tag in zip(meta.sample_ids, meta.tags):
            key = tag[STREAM_BUCKET_TAG]
            by_bucket.setdefault(None if key is None else tuple(key), []).append(sid)
        for key, sids in by_bucket.items():
            for sid in sids:
                for g, h in zip(self.grads, self.H[sid]):
                    g.add_(h, alpha=self.row_advantages[sid])
            if key is None:
                self.acc.capture(None, None)
            else:
                self.acc.capture(key[0], key[1])

    def close_stream_groups(self, closes):
        for gid, adv in closes:
            self.acc.close_group(gid, adv)

    def discard_stream_groups(self, groups):
        for gid in groups:
            self.acc.discard_group(gid)


def _controller(buffer, dp_client, trainer, max_open_groups=100):
    cls = SingleControllerActor.__ray_metadata__.modified_class
    ctl = object.__new__(cls)
    ctl._buffer = buffer
    ctl._dp_client = dp_client
    ctl._trainer = trainer
    ctl._trainer_version = 0
    ctl._async_cfg = SimpleNamespace(
        trajectory_streaming=SimpleNamespace(max_open_groups=max_open_groups)
    )
    ctl._train_fields = ()
    ctl._advantage_cfg = AdvantageConfig()
    ctl._advantage_estimator = GRPOAdvantageEstimator(
        SimpleNamespace(use_leave_one_out_baseline=True, normalize_rewards=True), None
    )
    ctl._algo_cfg = GRPOConfig.model_construct(
        overlong_filtering=False,
        advantage_clip_low=None,
        advantage_clip_high=None,
        num_generations_per_prompt=_G,
    )
    return ctl


def _record(rewards: list[float]) -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[SimpleNamespace(reward=r) for r in rewards],
        rollout_metrics={},
    )


def _expected(groups: dict[str, list[float]], H) -> list[torch.Tensor]:
    est = GRPOAdvantageEstimator(
        SimpleNamespace(use_leave_one_out_baseline=True, normalize_rewards=True), None
    )
    want = [torch.zeros(p, dtype=torch.float64) for p in _P]
    for gid, rewards in groups.items():
        adv = est.compute_advantage(
            torch.zeros(_G, 1, dtype=torch.long),
            torch.tensor(rewards),
            torch.ones(_G, 1),
        )[:, 0]
        for i, a in enumerate(adv.tolist()):
            for w, h in zip(want, H[f"{gid}_g{i}"]):
                w.add_(h, alpha=a)
    return want


async def _run(
    seed: int,
    dp: int,
    retry: bool,
    max_open_groups: int = 100,
    late_seal: bool = False,
):
    rng = random.Random(seed)
    torch.manual_seed(seed)
    reward_table: dict[str, float] = {}
    H: dict[str, list[torch.Tensor]] = {}
    dp_client = _DataPlane(reward_table)
    buffer = TQReplayBuffer(
        dp_client,
        partition_id="rollout_data",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        require_routed_experts=False,
    )
    buffer.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    trainer = _Trainer(H, dp)
    ctl = _controller(buffer, dp_client, trainer, max_open_groups)
    traj = _TrajectoryStepState()

    final_groups: dict[str, list[float]] = {}
    events: list[tuple[str, int]] = []
    group_rewards: dict[str, list[float]] = {}
    for _ in range(3):
        gid = buffer.reserve(weight_version=0, target_step=0)
        group_rewards[gid] = [float(rng.random() < 0.5) for _ in range(_G)]
        events += [(gid, i) for i in range(_G)]
    rng.shuffle(events)
    if late_seal:
        # Every group's last trajectory arrives at the very end: no group can
        # close early, so all streaming before then is bucket work.
        lasts = {g: max(j for gg, j in events if gg == g) for g in group_rewards}
        events = [e for e in events if e[1] != lasts[e[0]]] + [
            (g, j) for g, j in lasts.items()
        ]

    def publish_meta(gid: str, i: int):
        sid = f"{gid}_g{i}"
        reward_table[sid] = group_rewards[gid][i]
        H[sid] = [torch.randn(p, dtype=torch.float64) for p in _P]

    async def pump_once():
        meta, _, row_adv, buckets = await ctl._select_trajectory_chunk(traj)
        if meta is not None:
            for sid, a in zip(meta.sample_ids, row_adv.tolist()):
                trainer.row_advantages[sid] = a
            await ctl._train_trajectory_chunk(traj, meta, buckets)
        await ctl._close_trajectory_groups(traj)

    published: dict[str, int] = {g: 0 for g in group_rewards}
    retried = False
    removed: set[str] = set()
    for gid, i in events:
        if gid in removed:
            continue
        publish_meta(gid, i)
        await buffer.commit_trajectories(
            gid, [i], _record([group_rewards[gid][i]]), start_weight_version=0
        )
        published[gid] += 1
        if published[gid] == _G:
            await buffer.seal_group(
                gid, _record(group_rewards[gid]), end_weight_version=0
            )
            final_groups[gid] = group_rewards[gid]
        if rng.random() < 0.5:
            await pump_once()
        if retry and not retried and published[gid] == _G - 1:
            # This group's rollout fails after rows may have been trained:
            # remove it and roll it out again under a fresh slot.
            retried = True
            await pump_once()
            await buffer.remove_group(gid, remove_in_dp=True)
            removed.add(gid)
            new_gid = buffer.reserve(weight_version=0, target_step=0)
            group_rewards[new_gid] = [float(rng.random() < 0.5) for _ in range(_G)]
            for j in range(_G):
                publish_meta(new_gid, j)
            await buffer.commit_trajectories(
                new_gid,
                list(range(_G)),
                _record(group_rewards[new_gid]),
                start_weight_version=0,
            )
            await buffer.seal_group(
                new_gid, _record(group_rewards[new_gid]), end_weight_version=0
            )
            final_groups[new_gid] = group_rewards[new_gid]
    for _ in range(20):
        await pump_once()
    assert traj.counted == set(final_groups)
    trainer.acc.finalize()
    return trainer.grads, _expected(final_groups, H)


@pytest.mark.parametrize("dp", [1, 2])
@pytest.mark.parametrize("seed", range(4))
def test_pump_trajectory_streaming_is_exact(dp, seed):
    got, want = asyncio.run(_run(seed, dp, retry=False))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_pump_discards_retried_group(seed):
    got, want = asyncio.run(_run(seed, 1, retry=True))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
def test_pump_open_group_cap_is_exact(seed):
    got, want = asyncio.run(_run(seed, 2, retry=False, max_open_groups=1))
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dp", [1, 2])
def test_pump_final_first_with_all_groups_open_until_end(seed, dp):
    got, want = asyncio.run(
        _run(seed, dp, retry=False, max_open_groups=2, late_seal=True)
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-6, atol=1e-6)


def test_sealed_group_advantage_ignores_masked_rows_with_equal_reward():
    # With overlong filtering the truncated row is left out of the leave-one-out
    # baseline, so it gets a different advantage than the valid row sharing its
    # reward. The bucket for that reward must use the valid row's advantage.
    rewards = {"g_g0": 1.0, "g_g1": 1.0, "g_g2": 0.0, "g_g3": 0.0}
    dp_client = _DataPlane(rewards, truncated=frozenset({"g_g1"}))
    ctl = _controller(None, dp_client, trainer=None)
    ctl._algo_cfg = GRPOConfig.model_construct(
        overlong_filtering=True, advantage_clip_low=None, advantage_clip_high=None
    )
    meta = SimpleNamespace(sample_ids=list(rewards), partition_id="rollout_data")
    by_reward = asyncio.run(ctl._group_advantage_by_reward(meta))

    valid = torch.tensor([1.0, 0.0, 1.0, 1.0])
    want = ctl._advantage_estimator.compute_advantage(
        torch.zeros(4, 1, dtype=torch.long),
        torch.tensor(list(rewards.values())),
        torch.ones(4, 1),
        valid_mask=valid,
    )[:, 0]
    assert want[0] != want[1]  # the masked row really differs
    assert by_reward == {1.0: want[0].item(), 0.0: want[2].item()}


def test_group_size_not_divisible_by_dp_fails_loudly():
    # With G=3 and DP=2 the last row of a step can never be claimed (chunks are
    # trimmed to multiples of DP), so the pump would poll forever.
    from nemo_rl.algorithms.grad_streaming import pick_final_first

    rows = [("g", 1.0, i) for i in range(3)]
    picked = pick_final_first(
        rows, closed={"g"}, streaming=set(), max_open_groups=4, multiple_of=2
    )
    assert picked == [0, 1]
    assert (
        pick_final_first(
            rows[2:], closed={"g"}, streaming=set(), max_open_groups=4, multiple_of=2
        )
        == []
    )

    ctl = _controller(None, _DataPlane({}), _Trainer({}, dp=2))
    ctl._algo_cfg.num_generations_per_prompt = 3
    ctl._buffer = SimpleNamespace(trajectory_step_groups=lambda version: [])
    with pytest.raises(ValueError, match="multiple of the training data-parallel"):
        asyncio.run(ctl._select_trajectory_chunk(_TrajectoryStepState()))
