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
"""CPU tests: streamed accumulation reproduces sum_i a_i H_i exactly."""

import random

import pytest
import torch
from thundersync_rl.accumulator import StreamingGroupAccumulator
from thundersync_rl.streaming import StreamPlanner, Trajectory

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator


class _Cfg:
    def __init__(self, loo, norm):
        self.use_leave_one_out_baseline = loo
        self.normalize_rewards = norm


def _adv_fn(loo: bool, norm: bool = True):
    est = GRPOAdvantageEstimator(_Cfg(loo, norm), None)

    def fn(rewards):
        m = len(rewards)
        return est.compute_advantage(
            torch.zeros(m, 1, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.ones(m, 1),
        )[:, 0].tolist()

    return fn


def _simulate(
    rewards_per_group,
    adv_fn,
    dp_size,
    max_buckets,
    seed,
    chunk=3,
    group_only=False,
    max_open_groups=100,
):
    """Drive planner + one accumulator per DP rank with random arrivals."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    G = len(rewards_per_group[0])
    P = [17, 5]  # two "grad buffers"
    n = len(rewards_per_group) * G
    H = [[torch.randn(p, dtype=torch.float64) for p in P] for _ in range(n)]
    grads = [[torch.zeros(p, dtype=torch.float64) for p in P] for _ in range(dp_size)]
    accs = [
        StreamingGroupAccumulator(
            grads[r],
            storage_device="cpu",
            storage_dtype=torch.float64,
            max_buckets_per_group=max_buckets,
        )
        for r in range(dp_size)
    ]
    planner = StreamPlanner(
        adv_fn,
        dp_size=dp_size,
        max_chunk_trajectories=chunk,
        group_only=group_only,
        max_open_groups=max_open_groups,
    )
    for g in range(len(rewards_per_group)):
        planner.register_group(g, G)

    def backward(rank, trajs, advs):
        for t, a in zip(trajs, advs):
            for buf, h in zip(grads[rank], H[t.index]):
                buf.add_(h, alpha=a)

    def run_dispatch():
        d = planner.next_dispatch()
        for rank, chunks in enumerate(d.per_rank):
            for c in chunks:
                advs = c.advantages if c.group is None else [1.0] * len(c.trajectories)
                backward(rank, c.trajectories, advs)
                accs[rank].capture(c.group, c.reward)
            for gid, adv in d.closes:
                accs[rank].close_group(gid, adv)

    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        g = i // G
        planner.add(
            Trajectory(
                group=g,
                index=i,
                reward=rewards_per_group[g][i % G],
                payload=None,
                num_tokens=1,
            )
        )
        if rng.random() < 0.5:
            run_dispatch()
    while planner.has_work():
        run_dispatch()
    assert planner.all_done()
    for a in accs:
        a.finalize()
    # DP reduction = sum over ranks
    got = [sum(grads[r][k] for r in range(dp_size)) for k in range(len(P))]

    want = [torch.zeros(p, dtype=torch.float64) for p in P]
    for g, rs in enumerate(rewards_per_group):
        advs = adv_fn(rs)
        for j, a in enumerate(advs):
            for w, h in zip(want, H[g * G + j]):
                w.add_(h, alpha=a)
    return got, want, accs


@pytest.mark.parametrize("loo", [True, False])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_binary_rewards_exact(loo, dp_size, seed):
    rng = random.Random(100 + seed)
    rewards = [[float(rng.random() < 0.4) for _ in range(8)] for _ in range(6)]
    got, want, accs = _simulate(rewards, _adv_fn(loo), dp_size, 2, seed)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    # binary rewards never need more than 2 buffers per group
    for a in accs:
        assert a.peak_open_buffers <= 2 * 6


@pytest.mark.parametrize("seed", range(3))
def test_multi_valued_rewards_bucket_mode(seed):
    rng = random.Random(seed)
    rewards = [[rng.choice([0.0, 0.5, 1.0]) for _ in range(6)] for _ in range(4)]
    got, want, _ = _simulate(rewards, _adv_fn(True), 2, 3, seed)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_continuous_rewards_affine_mode(seed):
    # shared mean/std (no leave-one-out) => advantage is affine in reward
    rng = random.Random(seed)
    rewards = [[rng.random() for _ in range(8)] for _ in range(3)]
    got, want, _ = _simulate(rewards, _adv_fn(False), 2, 2, seed, chunk=1)
    # The estimator returns fp32 advantages, so the fitted (alpha, beta)
    # carry fp32 rounding (~1e-7 relative); bucket mode above is bit-exact.
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-5)


def test_non_affine_estimator_raises():
    # leave-one-out std is not affine in the reward; with too few buckets the
    # accumulator must refuse rather than silently produce a wrong gradient.
    rewards = [[0.1, 0.9, 0.3, 0.7, 0.2, 0.5]]
    with pytest.raises(ValueError, match="not affine"):
        for seed in range(10):
            _simulate(rewards, _adv_fn(True), 1, 2, seed, chunk=1)


@pytest.mark.parametrize("seed", range(3))
def test_group_granularity_exact_and_never_buckets(seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.5) for _ in range(8)] for _ in range(4)]
    got, want, accs = _simulate(rewards, _adv_fn(True), 2, 2, seed, group_only=True)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert all(a.peak_open_buffers == 0 for a in accs)


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize("seed", range(3))
def test_open_group_cap_exact_and_bounds_buffers(cap, seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.5) for _ in range(8)] for _ in range(6)]
    got, want, accs = _simulate(rewards, _adv_fn(True), 1, 2, seed, max_open_groups=cap)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert accs[0].peak_open_buffers <= 2 * cap
