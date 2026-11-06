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
"""CPU tests: streaming under pipeline parallelism.

A rank's chunks of one dispatch run as one pipeline schedule call per gradient
destination (worker ``train_microbatches`` + ``group_pipeline_calls``), and the
planner holds rows of closed groups until they fill the pipeline
(``min_pipeline_microbatches``), flushing at the end of the rollouts.

The Megatron schedule is a fake that consumes the call's micro-batches in order
and adds a linear "gradient" sum_rows sample_mask * sum_t adv * token_mask * E[id]
to a buffer, scaled by mcore's per-call loss scaling (LossPostProcessor's
micro-batch count / the call's count) so a wrong normalization shows up.
"""

from __future__ import annotations

import itertools
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("megatron.bridge")

import nemo_rl.models.policy.workers.megatron_policy_worker as mpw  # noqa: E402
from nemo_rl.algorithms.grad_streaming import StreamingGroupAccumulator  # noqa: E402
from nemo_rl.algorithms.loss.loss_functions import (  # noqa: E402
    ClippedPGLossConfig,
    ClippedPGLossFn,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from thundersync_rl.grpo_loop import (  # noqa: E402
    ThunderSyncConfig,
    resolve_min_pipeline_microbatches,
)
from thundersync_rl.streaming import (  # noqa: E402
    StreamPlanner,
    Trajectory,
    pipeline_bubble_fraction,
)
from thundersync_rl.worker import group_pipeline_calls  # noqa: E402

V, D = 50, 6  # vocabulary, "parameter" size of the fake model
EMB = torch.randn(V, D, generator=torch.Generator().manual_seed(0), dtype=torch.float64)


# --------------------------------------------------------------------------
# Fake Megatron schedule
# --------------------------------------------------------------------------


def _contribution(mb: BatchedDataDict) -> torch.Tensor:
    """Gradient of sum_rows sample_mask * sum_{t>=1} adv * token_mask * (w . E[id])."""
    w = (mb["advantages"][:, 1:] * mb["token_mask"][:, 1:]).to(torch.float64)
    w = w * mb["sample_mask"].to(torch.float64).unsqueeze(-1)
    return torch.einsum("rt,rtd->d", w, EMB[mb["input_ids"][:, 1:]])


class _Schedule:
    """Records every forward_backward_func call and accumulates its gradient."""

    def __init__(self, model, grad: torch.Tensor):
        self.model = model
        self.grad = grad
        self.calls: list[dict] = []

    def __call__(self, *, data_iterator, num_microbatches, seq_length, mbs, **kw):
        lpp = kw["post_processing_fn"]
        mbs_seen = [next(data_iterator) for _ in range(num_microbatches)]
        with pytest.raises(StopIteration):
            next(data_iterator)  # the call consumes exactly its micro-batches
        # mcore divides each micro-batch loss by the call's micro-batch count;
        # LossPostProcessor multiplies it back by the count it was built with.
        scale = lpp.num_microbatches / num_microbatches
        for mb in mbs_seen:
            self.grad.add_((_contribution(mb) * scale).to(self.grad.dtype))
        self.calls.append(
            dict(
                num_microbatches=num_microbatches,
                rows=[mb["row_id"].tolist() for mb in mbs_seen],
                shapes=[tuple(mb["input_ids"].shape) for mb in mbs_seen],
                variable=self.model.config.variable_seq_lengths,
                seq_length=seq_length,
                mbs=mbs,
            )
        )
        return [{"loss": torch.tensor(0.0)} for _ in mbs_seen]


def _fake_get_microbatch_iterator(data, cfg, mbs, **kw):
    n = data.size // mbs
    seq = data["input_ids"].shape[1]
    return (
        iter([data.slice(i * mbs, (i + 1) * mbs) for i in range(n)]),
        n,
        mbs,
        seq,
        seq,
    )


class _Rerun:
    def __init__(self):
        self.ran = False

    def should_run_forward_backward(self, it):
        self.ran = not self.ran
        return self.ran


@pytest.fixture
def worker(monkeypatch):
    """A ThunderSync worker with the Megatron schedule replaced by ``_Schedule``."""
    from tests.unit.models.policy.test_megatron_split_state import _make_worker
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    from nemo_rl.algorithms.loss.interfaces import LossType

    base = _make_worker(LossType.TOKEN_LEVEL)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.__dict__.update(base.__dict__)
    w.cfg["megatron_cfg"]["expert_model_parallel_size"] = 1
    w.cfg["megatron_cfg"]["pipeline_model_parallel_size"] = 4
    w.cfg["megatron_cfg"]["moe_router_load_balancing_type"] = "none"
    w.cfg["sequence_packing"] = {"enabled": False}
    w.cfg["dynamic_batching"] = {"enabled": False}
    w.model.config.variable_seq_lengths = False
    w.model.config.moe_token_dispatcher_type = "alltoall"
    grad = torch.zeros(D, dtype=torch.float32)
    sched = _Schedule(w.model, grad)
    w._make_grad_stream_accumulator = lambda spec: StreamingGroupAccumulator(
        [grad],
        storage_device="cpu",
        max_buckets_per_group=spec.max_buckets_per_group,
        buckets=spec.max_open_groups > 0,
    )
    rerun = _Rerun()
    monkeypatch.setattr(mpw, "megatron_forward_backward", sched)
    monkeypatch.setattr(mpw, "get_microbatch_iterator", _fake_get_microbatch_iterator)
    monkeypatch.setattr(mpw, "LossPostProcessor", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(mpw, "broadcast_loss_metrics_from_last_stage", lambda m: m)
    monkeypatch.setattr(mpw, "get_rerun_state_machine", lambda: rerun)
    pstate = MagicMock()
    pstate.is_pipeline_last_stage.return_value = True
    monkeypatch.setattr(mpw, "parallel_state", pstate)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    w._sched = sched
    w._grad = grad
    return w


_row_ids = itertools.count()


def _data(advs: list[float], seq: int, zero_rows: int = 0) -> BatchedDataDict:
    """Rows with per-row advantage ``advs``; ``seq`` tokens; random ids/masks."""
    n = len(advs)
    g = torch.Generator().manual_seed(next(_row_ids))
    token_mask = (torch.rand(n, seq, generator=g) < 0.8).float()
    return BatchedDataDict(
        {
            "input_ids": torch.randint(0, V, (n, seq), generator=g),
            "token_mask": token_mask,
            "sample_mask": torch.ones(n),
            "advantages": torch.tensor(advs).unsqueeze(-1).expand(n, seq).clone(),
            "row_id": torch.tensor([next(_row_ids) for _ in range(n)]),
        }
    )


def _item(data, group=None, reward=None, fused_key=None):
    return {"data": data, "group": group, "reward": reward, "fused_key": fused_key}


def _loss(skip_zero=False):
    return ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            force_on_policy_ratio=True,
            skip_zero_advantage_rows=skip_zero,
        )
    )


def _begin(w, *, merge, open_groups=0, skip_zero=False, mbs=2):
    w.stream_begin_step(
        _loss(skip_zero),
        gbs=64,
        mbs=mbs,
        storage_device="cpu",
        max_buckets_per_group=2,
        max_open_groups=open_groups,
        merge_pipeline_calls=merge,
    )


def _reference(items_with_adv) -> torch.Tensor:
    """sum over (data, per-row advantage multiplier) of the exact gradient (fp64)."""
    out = torch.zeros(D, dtype=torch.float64)
    for data, scale in items_with_adv:
        out += _contribution(data) * scale
    return out


def _final_items():
    # Different row counts (1, 2, 4, 6 rows with mbs=2 -> 1, 1, 2, 3 micro-
    # batches: chunks <= mbs run one micro-batch of exactly their rows) and
    # different sequence lengths, as length-sorted chunks of one dispatch.
    return [
        _item(_data([0.5], 7)),
        _item(_data([-1.0, 0.25], 9)),
        _item(_data([1.5, 0.0, -0.75, 0.3], 9)),
        _item(_data([0.1, 0.2, -0.3, 0.4, -0.5, 0.0], 12)),
    ]


# --------------------------------------------------------------------------
# Worker: one pipeline call per dispatch
# --------------------------------------------------------------------------


def test_group_mode_runs_one_call_with_all_microbatches_in_order(worker):
    items = _final_items()
    _begin(worker, merge=True)
    out = worker.stream_accumulate(items, [])
    (call,) = worker._sched.calls
    # 1 + 1 + 2 + 3 micro-batches.
    assert call["num_microbatches"] == 7
    assert out["num_calls"] == 1 and out["num_items"] == 4
    state = worker._train_step_state
    assert state["num_pipeline_calls"] == 1 and state["num_chunks"] == 4
    assert state["total_num_microbatches"] == 7
    # Micro-batches run in dispatch order, each chunk's rows in order.
    flat = [r for mb in call["rows"] for r in mb]
    assert flat == [r for it in items for r in it["data"]["row_id"].tolist()]
    # Each chunk keeps its own micro-batch size (min(rows, mbs)).
    assert [s[0] for s in call["shapes"]] == [1, 2, 2, 2, 2, 2, 2]
    assert [s[1] for s in call["shapes"]] == [7, 9, 9, 9, 12, 12, 12]
    # Shapes differ and PP > 1: variable_seq_lengths on for the call only.
    assert call["variable"] is True
    assert worker.model.config.variable_seq_lengths is False
    want = _reference([(it["data"], 1.0) for it in items])
    torch.testing.assert_close(worker._grad.double(), want, rtol=1e-6, atol=1e-6)


def test_merged_equals_per_chunk_calls(worker):
    items = _final_items()
    _begin(worker, merge=False)
    worker.stream_accumulate(items, [])
    per_chunk = worker._grad.clone()
    state = worker._train_step_state
    counts = (float(state["local_valid_toks"]), float(state["local_valid_seqs"]))
    assert [c["num_microbatches"] for c in worker._sched.calls] == [1, 1, 2, 3]
    assert state["num_pipeline_calls"] == 4
    worker.abort_train_step()

    worker._grad.zero_()
    worker._sched.calls.clear()
    _begin(worker, merge=True)
    worker.stream_accumulate(items, [])
    state = worker._train_step_state
    assert state["num_pipeline_calls"] == 1
    assert (
        float(state["local_valid_toks"]),
        float(state["local_valid_seqs"]),
    ) == counts
    # Same micro-batches, one call: equal up to fp32 summation order.
    torch.testing.assert_close(worker._grad, per_chunk, rtol=1e-6, atol=1e-6)


def test_uniform_shapes_do_not_switch_variable_seq_lengths(worker):
    items = [_item(_data([1.0, -1.0], 8)) for _ in range(3)]
    _begin(worker, merge=True)
    worker.stream_accumulate(items, [])
    (call,) = worker._sched.calls
    assert call["num_microbatches"] == 3 and call["variable"] is False


def test_pp1_merges_without_variable_seq_lengths(worker):
    worker.cfg["megatron_cfg"]["pipeline_model_parallel_size"] = 1
    _begin(worker, merge=True)
    worker.stream_accumulate(_final_items(), [])
    (call,) = worker._sched.calls
    assert call["num_microbatches"] == 7 and call["variable"] is False


def test_allgather_dispatcher_falls_back_to_per_shape_calls(worker):
    worker.model.config.moe_token_dispatcher_type = "allgather"
    _begin(worker, merge=True)
    with pytest.warns(UserWarning, match="separate pipeline calls"):
        worker.stream_accumulate(_final_items(), [])
    assert [c["num_microbatches"] for c in worker._sched.calls] == [1, 1, 2, 3]
    assert not any(c["variable"] for c in worker._sched.calls)


def test_zero_advantage_skipping_unchanged(worker):
    worker.cfg["megatron_cfg"]["expert_model_parallel_size"] = 1
    items = [
        _item(_data([1.0, -1.0], 6)),
        _item(_data([0.0, 0.0], 6)),  # all-zero chunk: counted, not trained
        _item(_data([0.0, 0.5, 0.0], 6)),  # one live row, topped up to mbs 2
    ]
    results = {}
    for merge in (False, True):
        worker._grad.zero_()
        worker._sched.calls.clear()
        _begin(worker, merge=merge, skip_zero=True)
        worker.stream_accumulate(items, [])
        st = worker._train_step_state
        results[merge] = (
            float(st["local_valid_toks"]),
            float(st["trained_valid_toks"]),
            float(st["trained_valid_seqs"]),
            [r for c in worker._sched.calls for mb in c["rows"] for r in mb],
            worker._grad.clone(),
            st["total_num_microbatches"],
        )
        worker.abort_train_step()
    assert results[False][:4] == results[True][:4]
    assert results[False][5] == results[True][5] == 2  # 1 + 0 + 1 micro-batches
    torch.testing.assert_close(results[False][4], results[True][4])
    # The all-zero chunk's rows never ran.
    assert set(items[1]["data"]["row_id"].tolist()).isdisjoint(results[True][3])


def test_bucket_mode_one_call_per_destination_and_exact_fold(worker):
    g = ("step", 0)
    b1 = [_item(_data([1.0, 1.0], 5), g, 1.0), _item(_data([1.0], 7), g, 1.0)]
    b0 = [_item(_data([1.0], 6), g, 0.0)]
    finals = _final_items()[:2]
    items = [finals[0], b1[0], b0[0], finals[1], b1[1]]
    adv = {1.0: 0.75, 0.0: -0.5}
    _begin(worker, merge=True, open_groups=1)
    out = worker.stream_accumulate(items, [(g, adv)])
    calls = worker._sched.calls
    assert out["num_calls"] == 3
    # Finals first (first item), then bucket r=1, then bucket r=0.
    assert [c["num_microbatches"] for c in calls] == [2, 2, 1]
    assert [r for mb in calls[1]["rows"] for r in mb] == [
        r for it in b1 for r in it["data"]["row_id"].tolist()
    ]
    worker._train_step_state["grad_stream"].finalize()
    want = _reference(
        [(it["data"], 1.0) for it in finals]
        + [(it["data"], adv[1.0]) for it in b1]
        + [(it["data"], adv[0.0]) for it in b0]
    )
    torch.testing.assert_close(worker._grad.double(), want, rtol=1e-6, atol=1e-6)


# --------------------------------------------------------------------------
# Call grouping (pure)
# --------------------------------------------------------------------------


def test_group_pipeline_calls():
    items = [
        {"group": None, "reward": None},
        {"group": "a", "reward": 1.0},
        {"group": None, "reward": None, "fused_key": "k"},
        {"group": "a", "reward": 0.0},
        {"group": None, "reward": None},
        {"group": "a", "reward": 1.0},
        {"group": "b", "reward": 1.0},
    ]
    assert group_pipeline_calls(items, merge=True) == [[0, 4], [1, 5], [2], [3], [6]]
    assert group_pipeline_calls(items, merge=False) == [[i] for i in range(7)]
    assert group_pipeline_calls([], merge=True) == []


def test_bubble_fraction():
    # k chunks of 8 micro-batches: separate calls vs one call.
    assert pipeline_bubble_fraction([8] * 8, 4) == pytest.approx(24 / 88)
    assert pipeline_bubble_fraction([64], 4) == pytest.approx(3 / 67)
    assert pipeline_bubble_fraction([8] * 8, 8) == pytest.approx(56 / 120)
    assert pipeline_bubble_fraction([64], 8) == pytest.approx(7 / 71)
    assert pipeline_bubble_fraction([5], 1) == 0.0
    assert pipeline_bubble_fraction([], 4) == 0.0


# --------------------------------------------------------------------------
# Planner: dispatch structure under PP
# --------------------------------------------------------------------------


def _loo_adv(rewards):
    m = len(rewards)
    return [r - (sum(rewards) - r) / (m - 1) for r in rewards]


def _run(
    seed,
    *,
    dp_size,
    min_mb,
    open_groups=0,
    equal=False,
    skip_zero=False,
    mbs=1,
    chunk=4,
    groups=8,
    G=8,
):
    """Random arrivals; dispatch whenever the planner has (held) work, flush at end.

    Returns (got, want, dispatches) where each dispatch records whether it was
    a flush, its per-rank call micro-batch counts and its final micro-batches.
    """
    rng = random.Random(seed)
    n = groups * G
    rewards = [float(rng.random() < 0.5) for _ in range(n)]
    tokens = [rng.randint(1, 60) for _ in range(n)]
    H = [torch.randn(5, dtype=torch.float64) for _ in range(n)]
    grads = [torch.zeros(5, dtype=torch.float64) for _ in range(dp_size)]
    accs = [
        StreamingGroupAccumulator(
            [grads[r]],
            storage_device="cpu",
            storage_dtype=torch.float64,
            max_buckets_per_group=2,
            buckets=open_groups > 0,
        )
        for r in range(dp_size)
    ]
    planner = StreamPlanner(
        _loo_adv,
        dp_size=dp_size,
        max_chunk_trajectories=chunk,
        max_open_groups=open_groups,
        micro_batch_size=mbs,
        equal_rank_microbatches=equal,
        skip_zero_advantage_chunks=skip_zero,
        min_pipeline_microbatches=min_mb,
    )
    for g in range(groups):
        planner.register_group(g, G)
    log = []

    def dispatch(flush):
        d = planner.next_dispatch(flush=flush)
        per_rank_calls = []
        final_mbs = 0
        for rank, chunks in enumerate(d.per_rank):
            items = [{"group": c.group, "reward": c.reward} for c in chunks]
            calls = group_pipeline_calls(items, merge=True)
            per_rank_calls.append(
                [sum(chunks[i].num_microbatches(mbs) for i in call) for call in calls]
            )
            for c in chunks:
                if c.group is None and not c.padding:
                    if not (skip_zero and c.zero_advantage):
                        final_mbs += c.num_microbatches(mbs)
            # Execute call by call: accumulate, capture once per call.
            for call in calls:
                for i in call:
                    c = chunks[i]
                    advs = (
                        c.advantages if c.group is None else [1.0] * len(c.trajectories)
                    )
                    if c.padding:
                        advs = [0.0]
                    for t, a in zip(c.trajectories, advs):
                        grads[rank].add_(H[t.index], alpha=a)
                accs[rank].capture(chunks[call[0]].group, chunks[call[0]].reward)
            for gid, adv in d.closes:
                accs[rank].close_group(gid, adv)
        log.append(dict(flush=flush, calls=per_rank_calls, final_mbs=final_mbs))

    order = list(range(n))
    rng.shuffle(order)
    for i in order:
        planner.add(
            Trajectory(
                group=i // G,
                index=i,
                reward=rewards[i],
                payload=None,
                num_tokens=tokens[i],
            )
        )
        while planner.has_work(flush=False):
            dispatch(False)
    while planner.has_work(flush=True):
        dispatch(True)
    assert planner.all_done()
    for a in accs:
        a.finalize()
    got = sum(grads)
    want = torch.zeros(5, dtype=torch.float64)
    for g in range(groups):
        idx = range(g * G, (g + 1) * G)
        for i, a in zip(idx, _loo_adv([rewards[i] for i in idx])):
            want += H[i] * a
    return got, want, log


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dp_size", [1, 2, 4])
def test_hold_fills_pipeline_and_flushes_at_end(seed, dp_size):
    min_mb = 8  # 2 * PP at PP=4
    got, want, log = _run(seed, dp_size=dp_size, min_mb=min_mb)
    torch.testing.assert_close(got, want)
    # Every dispatch before the flush carries >= min_mb micro-batches per rank
    # (in total over ranks); whatever is left goes in the end-of-rollout flush.
    early = [d for d in log if not d["flush"]]
    assert all(d["final_mbs"] >= min_mb * dp_size for d in early)
    # Group mode: at most one pipeline call per rank per dispatch.
    assert all(len(calls) <= 1 for d in log for calls in d["calls"])


@pytest.mark.parametrize("dp_size", [1, 2])
def test_threshold_never_reached_flushes_everything_at_end(dp_size):
    got, want, log = _run(0, dp_size=dp_size, min_mb=1000, groups=3, G=4)
    torch.testing.assert_close(got, want)
    assert [d["flush"] for d in log] == [True]
    assert log[0]["final_mbs"] == 12


@pytest.mark.parametrize("seed", range(4))
def test_no_hold_without_threshold(seed):
    _, _, held = _run(seed, dp_size=2, min_mb=8)
    got, want, free = _run(seed, dp_size=2, min_mb=0)
    torch.testing.assert_close(got, want)
    assert len(free) > len(held)  # more, smaller dispatches
    assert all(d["final_mbs"] >= 1 for d in free if d["calls"] != [[], []])


@pytest.mark.parametrize("seed", range(4))
def test_hold_counts_live_microbatches_with_skipping(seed):
    got, want, log = _run(seed, dp_size=2, min_mb=6, skip_zero=True, mbs=2)
    torch.testing.assert_close(got, want)
    assert all(d["final_mbs"] >= 12 for d in log if not d["flush"])


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("dp_size", [2, 3])
def test_expert_parallel_equal_counts_one_call(seed, dp_size):
    # EP (group granularity): every rank runs one call per dispatch with the
    # same number of micro-batches, so EP peers' 1F1B sequences match.
    got, want, log = _run(seed, dp_size=dp_size, min_mb=8, equal=True, mbs=2)
    torch.testing.assert_close(got, want)
    for d in log:
        counts = [sum(calls) for calls in d["calls"]]
        assert len(set(counts)) == 1
        assert all(len(calls) == (1 if counts[0] else 0) for calls in d["calls"])


@pytest.mark.parametrize("seed", range(4))
def test_bucket_work_is_not_held(seed):
    # Trajectory granularity: finals are held, buckets still stream.
    got, want, log = _run(seed, dp_size=2, min_mb=8, open_groups=2)
    torch.testing.assert_close(got, want)
    early = [d for d in log if not d["flush"]]
    assert any(d["final_mbs"] == 0 and any(d["calls"]) for d in early)
    assert all(d["final_mbs"] == 0 or d["final_mbs"] >= 16 for d in early)


def test_resolve_min_pipeline_microbatches():
    cfg = ThunderSyncConfig()
    assert resolve_min_pipeline_microbatches(cfg, 1) == 0
    assert resolve_min_pipeline_microbatches(cfg, 4) == 8
    assert resolve_min_pipeline_microbatches(cfg, 16) == 32
    cfg = ThunderSyncConfig(min_pipeline_microbatches=3)
    assert resolve_min_pipeline_microbatches(cfg, 8) == 3
    with pytest.raises(ValueError):
        resolve_min_pipeline_microbatches(
            ThunderSyncConfig(min_pipeline_microbatches=-1), 4
        )
