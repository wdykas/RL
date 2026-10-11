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
from types import SimpleNamespace

import pytest
import torch
from thundersync_rl.grpo_loop import build_chunk_data
from thundersync_rl.streaming import Chunk, StreamPlanner, Trajectory

from nemo_rl.algorithms.advantage_estimator import GRPOAdvantageEstimator
from nemo_rl.algorithms.grad_streaming import StreamingGroupAccumulator


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
    mbs=2,
    equal_rank_microbatches=False,
    skip_zero_advantage_chunks=False,
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
            # Group granularity: final chunks accumulate in place.
            buckets=not group_only and max_open_groups > 0,
        )
        for r in range(dp_size)
    ]
    planner = StreamPlanner(
        adv_fn,
        dp_size=dp_size,
        max_chunk_trajectories=chunk,
        max_open_groups=0 if group_only else max_open_groups,
        micro_batch_size=mbs,
        equal_rank_microbatches=equal_rank_microbatches,
        skip_zero_advantage_chunks=skip_zero_advantage_chunks,
    )
    for g in range(len(rewards_per_group)):
        planner.register_group(g, G)

    def backward(rank, trajs, advs):
        for t, a in zip(trajs, advs):
            for buf, h in zip(grads[rank], H[t.index]):
                buf.add_(h, alpha=a)

    def run_dispatch():
        d = planner.next_dispatch()
        dispatches.append(d)
        for rank, chunks in enumerate(d.per_rank):
            for c in chunks:
                advs = c.advantages if c.group is None else [1.0] * len(c.trajectories)
                # A padding chunk is fully masked: its rows add no gradient.
                if c.padding:
                    advs = [0.0] * len(c.trajectories)
                backward(rank, c.trajectories, advs)
                accs[rank].capture(c.group, c.reward)
            for gid, adv in d.closes:
                accs[rank].close_group(gid, adv)

    dispatches = []
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
                num_tokens=rng.randint(1, 50),
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
    return got, want, accs, dispatches


@pytest.mark.parametrize("loo", [True, False])
@pytest.mark.parametrize("dp_size", [1, 2, 3])
@pytest.mark.parametrize("seed", range(4))
def test_binary_rewards_exact(loo, dp_size, seed):
    rng = random.Random(100 + seed)
    rewards = [[float(rng.random() < 0.4) for _ in range(8)] for _ in range(6)]
    got, want, accs, _ = _simulate(rewards, _adv_fn(loo), dp_size, 2, seed)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    # binary rewards never need more than 2 buffers per group
    for a in accs:
        assert a.peak_open_buffers <= 2 * 6


@pytest.mark.parametrize("seed", range(3))
def test_multi_valued_rewards_bucket_mode(seed):
    rng = random.Random(seed)
    rewards = [[rng.choice([0.0, 0.5, 1.0]) for _ in range(6)] for _ in range(4)]
    got, want, _, _ = _simulate(rewards, _adv_fn(True), 2, 3, seed)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("seed", range(3))
def test_continuous_rewards_affine_mode(seed):
    # shared mean/std (no leave-one-out) => advantage is affine in reward
    rng = random.Random(seed)
    rewards = [[rng.random() for _ in range(8)] for _ in range(3)]
    got, want, _, _ = _simulate(rewards, _adv_fn(False), 2, 2, seed, chunk=1)
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


@pytest.mark.parametrize("skip_zero", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_group_granularity_exact_and_never_buckets(seed, skip_zero):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.5) for _ in range(8)] for _ in range(4)]
    got, want, accs, _ = _simulate(
        rewards,
        _adv_fn(True),
        2,
        2,
        seed,
        group_only=True,
        skip_zero_advantage_chunks=skip_zero,
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert all(a.peak_open_buffers == 0 for a in accs)
    # In-place mode: no batch accumulator was allocated.
    assert all(a.batch_acc == [] and not a.buckets_enabled for a in accs)


@pytest.mark.parametrize("cap", [1, 2])
@pytest.mark.parametrize("seed", range(3))
def test_open_group_cap_exact_and_bounds_buffers(cap, seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.5) for _ in range(8)] for _ in range(6)]
    got, want, accs, _ = _simulate(
        rewards, _adv_fn(True), 1, 2, seed, max_open_groups=cap
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert accs[0].peak_open_buffers <= 2 * cap


def test_zero_advantage_rows_never_share_a_chunk_with_trained_rows():
    planner = StreamPlanner(
        lambda rs: [r - sum(rs) / len(rs) for r in rs],
        dp_size=1,
        max_open_groups=0,
        max_chunk_trajectories=4,
        micro_batch_size=1,
        equal_rank_microbatches=False,
    )
    # Group 0: mixed rewards (nonzero advantages); groups 1-2: all equal (zero).
    members = {0: [1.0, 0.0, 1.0, 0.0, 1.0], 1: [1.0, 1.0], 2: [0.0]}
    idx = 0
    for g, rewards in members.items():
        planner.register_group(g, len(rewards))
        for r in rewards:
            planner.add(
                Trajectory(
                    group=g, index=idx, reward=r, payload=None, num_tokens=idx + 1
                )
            )
            idx += 1
    d = planner.next_dispatch()
    kinds = [{a == 0.0 for a in c.advantages} for c in d.per_rank[0]]
    assert all(len(k) == 1 for k in kinds)  # every chunk is all-zero or all-nonzero
    assert sorted(len(c.trajectories) for c in d.per_rank[0]) == [1, 3, 4]


def _microbatches_per_rank(d, mbs):
    return [sum(c.num_microbatches(mbs) for c in cs) for cs in d.per_rank]


@pytest.mark.parametrize("dp_size", [2, 3, 4])
@pytest.mark.parametrize("mbs", [1, 2, 4])
@pytest.mark.parametrize("seed", range(4))
def test_equal_rank_microbatches_pads_short_ranks_exactly(dp_size, mbs, seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.4) for _ in range(8)] for _ in range(6)]
    got, want, _, dispatches = _simulate(
        rewards,
        _adv_fn(True),
        dp_size,
        2,
        seed,
        chunk=5,
        mbs=mbs,
        equal_rank_microbatches=True,
    )
    # Padding chunks are masked out: the update is unchanged.
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    for d in dispatches:
        counts = _microbatches_per_rank(d, mbs)
        assert len(set(counts)) == 1, counts
        # Only short ranks are padded, one micro-batch per padding chunk, and
        # padding is never counted as dispatched work.
        real = [
            sum(c.num_microbatches(mbs) for c in cs if not c.padding)
            for cs in d.per_rank
        ]
        assert max(real) == counts[0]
        for cs, r in zip(d.per_rank, real):
            pads = [c for c in cs if c.padding]
            assert len(pads) == counts[0] - r
            assert all(
                c.group is None and c.advantages == [0.0] and len(c.trajectories) == 1
                for c in pads
            )
    assert sum(d.num_trajectories for d in dispatches) == 6 * 8


@pytest.mark.parametrize("seed", range(4))
def test_without_equal_rank_microbatches_no_padding(seed):
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.4) for _ in range(8)] for _ in range(6)]
    got, want, _, dispatches = _simulate(
        rewards, _adv_fn(True), 3, 2, seed, chunk=5, mbs=2
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert not any(c.padding for d in dispatches for cs in d.per_rank for c in cs)
    # Token balancing alone leaves ranks with different micro-batch counts.
    assert any(len(set(_microbatches_per_rank(d, 2))) > 1 for d in dispatches)


def test_equal_rank_microbatches_counts_microbatches_not_chunks():
    # Six nonzero-advantage rows in chunks of 3: each chunk is padded to 4 rows
    # and runs as 2 micro-batches of 2. Two ranks get one chunk each; the third
    # gets two one-row padding chunks (2 micro-batches), not one.
    planner = StreamPlanner(
        lambda rs: [r - sum(rs) / len(rs) for r in rs],
        dp_size=3,
        max_open_groups=0,
        max_chunk_trajectories=3,
        micro_batch_size=2,
        equal_rank_microbatches=True,
    )
    rewards = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    planner.register_group(0, len(rewards))
    for i, r in enumerate(rewards):
        planner.add(
            Trajectory(group=0, index=i, reward=r, payload=None, num_tokens=10 + i)
        )
    d = planner.next_dispatch()
    real = sorted(len([c for c in cs if not c.padding]) for cs in d.per_rank)
    assert real == [0, 1, 1]
    assert _microbatches_per_rank(d, 2) == [2, 2, 2]
    (empty,) = [cs for cs in d.per_rank if not any(not c.padding for c in cs)]
    assert len(empty) == 2
    # The padding template is the dispatch's shortest trajectory.
    assert all(c.trajectories[0].num_tokens == 10 for c in empty)
    assert d.num_trajectories == 6
    assert planner.all_done()


def test_padding_chunk_data_is_fully_masked():
    traj = Trajectory(
        group=0,
        index=0,
        reward=1.0,
        payload={
            "message_log": [
                {"role": "user", "token_ids": torch.tensor([1, 2, 3])},
                {
                    "role": "assistant",
                    "token_ids": torch.tensor([4, 5]),
                    "generation_logprobs": torch.tensor([-0.5, -0.25]),
                },
            ],
            "loss_multiplier": 1.0,
        },
        num_tokens=5,
    )
    cfg = SimpleNamespace(policy={"make_sequence_length_divisible_by": 1})
    tok = SimpleNamespace(pad_token_id=0)

    def build(padding):
        chunk = Chunk(
            group=None,
            reward=None,
            trajectories=[traj],
            advantages=[0.0 if padding else 0.7],
            padding=padding,
        )
        return build_chunk_data(chunk, tok, cfg, 1)

    real, pad = build(False), build(True)
    assert real["sample_mask"].sum() > 0 and real["token_mask"].sum() > 0
    # Same tokens (a real forward), but no gradient and no normalization count.
    assert torch.equal(pad["input_ids"], real["input_ids"])
    for k in ("sample_mask", "token_mask", "advantages"):
        assert not pad[k].any(), k


@pytest.mark.parametrize(
    "ep,granularity,open_groups,ok",
    [
        (1, "trajectory", 4, True),
        (2, "group", 4, True),
        (2, "trajectory", 0, True),
        (2, "trajectory", 4, False),
    ],
)
def test_expert_parallel_rejects_open_group_buckets(ep, granularity, open_groups, ok):
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_expert_parallel

    ts = ThunderSyncConfig(granularity=granularity, max_open_groups=open_groups)
    policy = {"megatron_cfg": {"expert_model_parallel_size": ep}}
    if ok:
        validate_expert_parallel(ts, policy)
    else:
        with pytest.raises(ValueError, match="expert parallelism"):
            validate_expert_parallel(ts, policy)


@pytest.mark.parametrize(
    "verify_on,gen_ep,ok",
    [
        ("learner", 2, True),
        ("both", 2, False),
        ("inference", 2, False),
        ("both", 1, True),
    ],
)
def test_expert_parallel_speculation_verifies_on_learner(verify_on, gen_ep, ok):
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_expert_parallel

    ts = ThunderSyncConfig(granularity="group", speculate=True, verify_on=verify_on)
    policy = {
        "megatron_cfg": {"expert_model_parallel_size": 2},
        "generation": {
            "mcore_generation_config": {"expert_model_parallel_size": gen_ep}
        },
    }
    if ok:
        validate_expert_parallel(ts, policy)
    else:
        with pytest.raises(ValueError, match="verify_on='learner'"):
            validate_expert_parallel(ts, policy)


@pytest.mark.parametrize(
    "gen,ok",
    [
        ({"temperature": 1.0, "top_p": 1.0, "top_k": None, "stop_strings": None}, True),
        (
            {"temperature": 0.7, "top_p": 1.0, "top_k": None, "stop_strings": None},
            False,
        ),
        (
            {"temperature": 1.0, "top_p": 0.9, "top_k": None, "stop_strings": None},
            False,
        ),
        ({"temperature": 1.0, "top_p": 1.0, "top_k": 50, "stop_strings": None}, False),
        (
            {"temperature": 1.0, "top_p": 1.0, "top_k": None, "stop_strings": ["</a>"]},
            False,
        ),
    ],
)
def test_speculation_rejects_unreproduced_sampling(gen, ok):
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_speculation

    par = {"tensor_model_parallel_size": 1, "pipeline_model_parallel_size": 1}
    policy = {"generation": gen, "megatron_cfg": dict(par)}
    validate_speculation(ThunderSyncConfig(speculate=False), policy)
    if ok:
        validate_speculation(ThunderSyncConfig(speculate=True), policy)
    else:
        with pytest.raises(ValueError, match="speculate"):
            validate_speculation(ThunderSyncConfig(speculate=True), policy)


@pytest.mark.parametrize(
    "ts_kw,train_tp,gen_tp,ok",
    [
        ({"verify_on": "both"}, 1, 1, True),
        ({"verify_on": "both"}, 1, 2, False),
        ({"verify_on": "learner"}, 2, 2, True),
        ({"verify_on": "learner", "verify_mode": "keyed"}, 2, 2, False),
    ],
)
def test_speculation_parallel_layout_guards(ts_kw, train_tp, gen_tp, ok):
    from thundersync_rl.grpo_loop import ThunderSyncConfig, validate_speculation

    policy = {
        "generation": {
            "temperature": 1.0,
            "top_p": None,
            "top_k": None,
            "stop_strings": None,
            "mcore_generation_config": {"tensor_model_parallel_size": gen_tp},
        },
        "megatron_cfg": {
            "tensor_model_parallel_size": train_tp,
            "pipeline_model_parallel_size": 1,
        },
    }
    ts = ThunderSyncConfig(speculate=True, **ts_kw)
    if ok:
        validate_speculation(ts, policy)
    else:
        with pytest.raises(ValueError, match="speculate"):
            validate_speculation(ts, policy)


def _chunk(tokens, advantages):
    trajs = [
        Trajectory(group=0, index=i, reward=0.0, payload=None, num_tokens=t)
        for i, t in enumerate(tokens)
    ]
    return Chunk(
        group=None,
        reward=None,
        trajectories=trajs,
        advantages=advantages,
        padding=False,
    )


def _old_token_balance(chunks, dp_size):
    """The planner's previous greedy longest-first balance on total tokens."""
    per_rank = [[] for _ in range(dp_size)]
    load = [0] * dp_size
    for c in sorted(chunks, key=lambda c: -c.num_tokens):
        r = load.index(min(load))
        per_rank[r].append(c)
        load[r] += c.num_tokens
    return per_rank


@pytest.mark.parametrize("seed", range(5))
def test_balance_without_skipping_matches_token_balance(seed):
    from thundersync_rl.streaming import balance_chunks

    rng = random.Random(seed)
    chunks = [_chunk([rng.randint(1, 50)], [rng.choice([0.0, 0.5])]) for _ in range(20)]
    got = balance_chunks(chunks, 3, skip_zero_advantage=False)
    want = _old_token_balance(chunks, 3)
    assert [[id(c) for c in cs] for cs in got] == [[id(c) for c in cs] for cs in want]


def test_balance_by_live_tokens_when_zero_advantage_chunks_are_skipped():
    from thundersync_rl.streaming import balance_chunks

    live = [_chunk([100], [1.0]), _chunk([60], [-1.0]), _chunk([40], [0.5])]
    dead = [_chunk([90], [0.0]), _chunk([80], [0.0, 0.0])]
    chunks = live + dead

    def live_load(cs):
        return sum(c.num_tokens for c in cs if not c.zero_advantage)

    # Total-token balance puts both long dead chunks on one rank and all live
    # work (200 tokens) on the other; live-token balance gives 100 / 100.
    per_rank = balance_chunks(chunks, 2, skip_zero_advantage=True)
    assert sorted(live_load(cs) for cs in per_rank) == [100, 100]
    # Skipped chunks still spread over ranks (ties broken by total tokens).
    assert all(any(c.zero_advantage for c in cs) for cs in per_rank)
    old = _old_token_balance(chunks, 2)
    assert sorted(live_load(cs) for cs in old) != [100, 100]
    assert sorted(id(c) for cs in per_rank for c in cs) == sorted(map(id, chunks))


def test_zero_advantage_property():
    assert _chunk([3, 4], [0.0, 0.0]).zero_advantage
    assert not _chunk([3, 4], [0.0, 0.2]).zero_advantage
    bucket = Chunk(
        group=1,
        reward=1.0,
        trajectories=_chunk([3], [0.0]).trajectories,
        advantages=None,
        padding=False,
    )
    assert not bucket.zero_advantage  # bucket chunks train with advantage 1
