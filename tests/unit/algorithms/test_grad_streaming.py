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
"""Streamed accumulation must reproduce sum_i a_i H_i for any arrival order."""

import random

import pytest
import torch

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.grad_streaming import StreamingGroupAccumulator


class _EstimatorCfg:
    def __init__(self, leave_one_out: bool):
        self.use_leave_one_out_baseline = leave_one_out
        self.normalize_rewards = True


def _advantages(rewards: list[float], leave_one_out: bool) -> list[float]:
    est = GRPOAdvantageEstimator(_EstimatorCfg(leave_one_out), None)
    m = len(rewards)
    return est.compute_advantage(
        torch.zeros(m, 1, dtype=torch.long),
        torch.tensor(rewards, dtype=torch.float32),
        torch.ones(m, 1),
    )[:, 0].tolist()


def _run(rewards, leave_one_out, max_buckets, seed):
    """Stream each trajectory as its own chunk, close each group after its last."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    sizes = [11, 3]
    grads = [torch.zeros(p, dtype=torch.float64) for p in sizes]
    acc = StreamingGroupAccumulator(
        grads,
        storage_device="cpu",
        max_buckets_per_group=max_buckets,
        storage_dtype=torch.float64,
    )
    trajs = [(g, j) for g, rs in enumerate(rewards) for j in range(len(rs))]
    H = {t: [torch.randn(p, dtype=torch.float64) for p in sizes] for t in trajs}
    rng.shuffle(trajs)
    remaining = {g: len(rs) for g, rs in enumerate(rewards)}
    for g, j in trajs:
        for buf, h in zip(grads, H[(g, j)]):
            buf.add_(h)  # backward with advantage 1
        acc.capture(g, rewards[g][j])
        remaining[g] -= 1
        if remaining[g] == 0:
            advs = _advantages(rewards[g], leave_one_out)
            acc.close_group(g, dict(zip(rewards[g], advs)))
    acc.finalize()

    want = [torch.zeros(p, dtype=torch.float64) for p in sizes]
    for g, rs in enumerate(rewards):
        for j, a in enumerate(_advantages(rs, leave_one_out)):
            for w, h in zip(want, H[(g, j)]):
                w.add_(h, alpha=a)
    return grads, want, acc


@pytest.mark.parametrize("leave_one_out", [True, False])
@pytest.mark.parametrize("seed", range(3))
def test_binary_rewards_bit_exact_with_two_buckets(leave_one_out, seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.5) for _ in range(8)] for _ in range(4)]
    got, want, acc = _run(rewards, leave_one_out, max_buckets=2, seed=seed)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert acc.peak_open_buffers <= 2 * len(rewards)


def test_three_valued_rewards_with_three_buckets_exact():
    rewards = [[0.0, 0.5, 1.0, 0.5, 0.0, 1.0], [1.0, 1.0, 0.0, 0.5, 0.5, 0.5]]
    got, want, _ = _run(rewards, leave_one_out=True, max_buckets=3, seed=0)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)


def test_affine_collapse_exact_for_shared_mean_std():
    rewards = [[0.1, 0.9, 0.3, 0.7, 0.2, 0.5]]
    got, want, _ = _run(rewards, leave_one_out=False, max_buckets=2, seed=0)
    # fp32 advantages => fp32-level agreement of the fitted (alpha, beta)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)


def test_non_affine_estimator_raises_instead_of_wrong_gradient():
    rewards = [[0.1, 0.9, 0.3, 0.7, 0.2, 0.5]]
    with pytest.raises(ValueError, match="not affine"):
        _run(rewards, leave_one_out=True, max_buckets=2, seed=0)


def test_final_chunks_and_unknown_group_close():
    grads = [torch.zeros(4, dtype=torch.float64)]
    acc = StreamingGroupAccumulator(
        grads,
        storage_device="cpu",
        max_buckets_per_group=2,
        storage_dtype=torch.float64,
    )
    grads[0].add_(torch.ones(4, dtype=torch.float64))
    acc.capture(None, None)  # final-advantage chunk goes straight to the batch
    assert torch.count_nonzero(grads[0]) == 0  # grad buffer is scratch again
    acc.close_group("never-streamed-here", {1.0: 2.0})  # no-op on this rank
    acc.finalize()
    torch.testing.assert_close(grads[0], torch.ones(4, dtype=torch.float64))


def test_finalize_with_open_group_fails_loudly():
    grads = [torch.zeros(2, dtype=torch.float64)]
    acc = StreamingGroupAccumulator(
        grads,
        storage_device="cpu",
        max_buckets_per_group=2,
        storage_dtype=torch.float64,
    )
    acc.capture("g", 1.0)
    with pytest.raises(AssertionError, match="still open"):
        acc.finalize()


def test_discard_group_drops_only_that_group():
    grads = [torch.zeros(3, dtype=torch.float64)]
    acc = StreamingGroupAccumulator(
        grads,
        storage_device="cpu",
        max_buckets_per_group=2,
        storage_dtype=torch.float64,
    )
    grads[0].add_(torch.full((3,), 5.0, dtype=torch.float64))
    acc.capture("retried", 1.0)
    grads[0].add_(torch.ones(3, dtype=torch.float64))
    acc.capture("kept", 1.0)
    acc.discard_group("retried")
    acc.close_group("kept", {1.0: 2.0})
    acc.finalize()
    torch.testing.assert_close(grads[0], torch.full((3,), 2.0, dtype=torch.float64))


def _rows(spec):
    """spec: list of (group, reward) -> rows with item = index."""
    return [(g, r, i) for i, (g, r) in enumerate(spec)]


def test_pick_final_first_prefers_closed_groups_and_trims_to_multiple():
    from nemo_rl.algorithms.grad_streaming import pick_final_first

    rows = _rows([("a", 1.0), ("b", 0.0), ("a", 0.0), ("b", 1.0), ("a", 1.0)])
    picked = pick_final_first(
        rows, closed={"a"}, streaming=set(), max_open_groups=4, multiple_of=2
    )
    assert picked == [0, 2]  # a's rows only, trimmed from 3 to 2


def test_pick_final_first_streams_one_open_group_per_buckets_multiple():
    from nemo_rl.algorithms.grad_streaming import pick_final_first

    rows = _rows([("a", 1.0), ("b", 1.0), ("b", 1.0), ("b", 0.0), ("a", 1.0)])
    picked = pick_final_first(
        rows, closed=set(), streaming=set(), max_open_groups=4, multiple_of=2
    )
    # b has a usable reward-1 pair, a has a usable reward-1 pair: tie on size,
    # the first maximal group wins; per-bucket trimming drops b's lone reward 0.
    assert picked in ([0, 4], [1, 2])
    # an already-streaming group is preferred even if smaller
    picked = pick_final_first(
        rows, closed=set(), streaming={"a"}, max_open_groups=4, multiple_of=1
    )
    assert picked == [0, 4]


def test_pick_final_first_respects_open_group_cap():
    from nemo_rl.algorithms.grad_streaming import pick_final_first

    rows = _rows([("c", 1.0), ("c", 1.0)])
    assert (
        pick_final_first(
            rows, closed=set(), streaming={"a", "b"}, max_open_groups=2, multiple_of=1
        )
        == []
    )
