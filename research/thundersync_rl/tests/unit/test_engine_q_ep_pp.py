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
"""Engine-sourced q under expert and pipeline parallelism (CPU, fake groups).

Ranks are threads. Process groups are fakes: a rank-order all-gather for the
tensor-parallel vocab gathers (``_ThreadTP``), an object all-gather for the
expert-parallel lockstep, and barriers standing in for the collectives inside
a training forward: the expert all-to-all (one barrier per forward over the
EP group, so unequal forward counts break it) and the pipeline send/receive
(one barrier per forward over the stages of a replica).
"""

import threading
import zlib
from types import SimpleNamespace

import pytest
import torch
from test_engine_q import (
    VERSIONS,
    _engine_drafts,
    _Head,
    _hidden,
    _policy,
    _ThreadTP,
    _ts,
)
from thundersync_rl.grpo_loop import validate_engine_verify_source

V, D = 40, 16


def _cls():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    return ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class


def _weights():
    gen = torch.Generator().manual_seed(0)
    emb = torch.randn(V, D, generator=gen)
    heads = [torch.randn(V, D, generator=gen) * 2]
    for _ in range(VERSIONS - 1):
        heads.append(heads[-1] + torch.randn(V, D, generator=gen) * 0.3)
    return emb, heads


class _Group:
    """Thread ranks of one fake process group."""

    def __init__(self, size):
        self.size = size
        self.slots = [None] * size
        self.barrier = threading.Barrier(size, timeout=20)

    def all_gather_object(self, rank, out, obj):
        self.slots[rank] = obj
        self.barrier.wait()
        out[:] = list(self.slots)
        self.barrier.wait()


class _Rank:
    """Coordinates of one fake rank (read by the patched parallel_state)."""

    def __init__(self, *, dp, dp_size, tp, tp_rank, pp, pp_rank, ep, ep_rank):
        self.dp, self.dp_size = dp, dp_size
        self.tp, self.tp_rank = tp, tp_rank
        self.pp, self.pp_rank = pp, pp_rank
        self.ep, self.ep_rank = ep, ep_rank
        self.tp_handle = SimpleNamespace(size=tp, rank=tp_rank, grp=None)
        self.ep_handle = SimpleNamespace(size=ep, rank=ep_rank, grp=None)


_local = threading.local()


@pytest.fixture
def fake_groups(monkeypatch):
    """Patch parallel_state / torch.distributed to read the thread's _Rank."""
    from megatron.core import parallel_state as ps

    def cur():
        return _local.rank

    for name, fn in {
        "get_data_parallel_world_size": lambda: cur().dp_size,
        "get_data_parallel_rank": lambda: cur().dp,
        "get_tensor_model_parallel_world_size": lambda: cur().tp,
        "get_tensor_model_parallel_rank": lambda: cur().tp_rank,
        "get_tensor_model_parallel_group": lambda: cur().tp_handle,
        "get_pipeline_model_parallel_world_size": lambda: cur().pp,
        "is_pipeline_last_stage": lambda: cur().pp_rank == cur().pp - 1,
        "get_expert_model_parallel_world_size": lambda: cur().ep,
        "get_expert_model_parallel_group": lambda: cur().ep_handle,
    }.items():
        monkeypatch.setattr(ps, name, lambda *a, _f=fn, **k: _f())
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: group.size
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather",
        lambda parts, x, group=None: group.grp.all_gather(group.rank, parts, x),
    )
    monkeypatch.setattr(
        torch.distributed,
        "all_gather_object",
        lambda out, obj, group=None: group.grp.all_gather_object(group.rank, out, obj),
    )


def _worker(emb, heads, rank, *, syncs=()):
    """Fake learner rank: a training forward whose collectives are barriers.

    ``syncs``: barriers every forward passes (expert all-to-all over the EP
    group, pipeline send/receive over the replica's stages). Off the last
    stage the forward returns None, as ``_training_forward`` does there.
    """
    w = object.__new__(_cls())
    w.cfg = {"megatron_cfg": {"batch_invariant_mode": True}}
    w.rank = rank.dp
    w.model = torch.nn.Module()
    w.model.module = torch.nn.Module()
    w.forwards = []
    last = rank.pp_rank == rank.pp - 1
    if last:
        w.model.module.output_layer = _Head(heads[-1].chunk(rank.tp)[rank.tp_rank])
        w._head_stash = {
            v: heads[v].chunk(rank.tp)[rank.tp_rank].clone()
            for v in range(VERSIONS - 1)
        }

    def training_forward(row):
        w.forwards.append(row.clone())
        for b in syncs:
            b.wait()
        if not last:
            return None
        h = _hidden(row, emb)
        return w.model.module.output_layer(h.unsqueeze(1))[0].transpose(0, 1)

    w._training_forward = training_forward
    return w


def _uneven_keys(dp, counts):
    """Draft keys whose owner (crc32 % dp) gets ``counts[r]`` rows, interleaved."""
    pools = [[] for _ in range(dp)]
    i = 0
    while any(len(p) < c for p, c in zip(pools, counts)):
        k = f"key{i}"
        r = zlib.crc32(k.encode()) % dp
        if len(pools[r]) < counts[r]:
            pools[r].append(k)
        i += 1
    out = []
    for j in range(max(counts)):
        out.extend(p[j] for p in pools if j < len(p))
    return out


def _run(ranks, fn):
    """``fn(rank)`` on one thread per rank; returns the results in order."""
    out, errors = [None] * len(ranks), []

    def body(j):
        _local.rank = ranks[j]
        try:
            out[j] = fn(j)
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            errors.append(exc)

    threads = [threading.Thread(target=body, args=(j,)) for j in range(len(ranks))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return out


def _drafts(emb, heads, counts):
    keys = _uneven_keys(len(counts), counts)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(
        emb, heads, n=len(keys), device="cpu", key_names=keys
    )
    return dict(
        rows=rows,
        prompt_lens=plens,
        segments=segs,
        engine_logq=lqs,
        hidden=hids,
        vocab_limit=1 << 30,
        seed=3,
        keys=keys,
    )


def _reference(emb, heads, kw, dp_size):
    """Per data-parallel replica: results at TP = PP = EP = 1 (no lockstep)."""
    ref = []
    for dp in range(dp_size):
        rank = _Rank(
            dp=dp, dp_size=dp_size, tp=1, tp_rank=0, pp=1, pp_rank=0, ep=1, ep_rank=0
        )
        _local.rank = rank
        w = _worker(emb, heads, rank)
        ref.append((w.verify_drafts_block_engine(**kw), len(w.forwards)))
    return ref


def _layout(emb, heads, *, dp_size, tp, pp, ep):
    """Fake ranks of dp_size replicas x pp stages x tp ranks, with their groups.

    EP groups pair the replicas' ranks at the same (stage, TP rank) (EP spans
    data-parallel replicas; here EP = dp_size). TP groups are per (replica,
    stage). Each forward passes the EP barrier (all-to-all) and the replica's
    PP barrier (send/receive).
    """
    assert ep in (1, dp_size)
    tp_grps = {(d, s): _ThreadTP(tp) for d in range(dp_size) for s in range(pp)}
    ep_grps = {(s, t): _Group(ep) for s in range(pp) for t in range(tp)}
    pp_bars = {
        (d, t): threading.Barrier(pp, timeout=20)
        for d in range(dp_size)
        for t in range(tp)
    }
    ranks, workers = [], []
    for d in range(dp_size):
        for s in range(pp):
            for t in range(tp):
                r = _Rank(
                    dp=d,
                    dp_size=dp_size,
                    tp=tp,
                    tp_rank=t,
                    pp=pp,
                    pp_rank=s,
                    ep=ep,
                    ep_rank=d if ep > 1 else 0,
                )
                r.tp_handle.grp = tp_grps[(d, s)]
                r.ep_handle.grp = ep_grps[(s, t)]
                syncs = []
                if ep > 1:
                    syncs.append(ep_grps[(s, t)].barrier)
                if pp > 1:
                    syncs.append(pp_bars[(d, t)])
                ranks.append(r)
                workers.append(_worker(emb, heads, r, syncs=syncs))
    return ranks, workers, tp_grps


@pytest.mark.parametrize(
    "tp, pp, ep",
    [
        (1, 1, 2),  # expert parallel, uneven rows
        (2, 1, 2),  # EP x TP: vocab gathers stay inside each replica
        (1, 2, 1),  # pipeline: last stage verifies
        (2, 2, 1),  # PP x TP
        (1, 2, 2),  # EP x PP
        (2, 2, 2),  # EP x PP x TP
    ],
)
def test_engine_verify_under_ep_pp_matches_single_rank(fake_groups, tp, pp, ep):
    """Equal forward counts across EP; only last stages verify; results == TP=PP=EP=1."""
    emb, heads = _weights()
    counts = [9, 3]  # replica 0 owns 9 rows, replica 1 owns 3
    kw = _drafts(emb, heads, counts)
    ref = _reference(emb, heads, kw, len(counts))
    assert [n for _, n in ref] == counts
    assert (
        sum(r["accepted"] < len(kw["engine_logq"][i]) for res, _ in ref for i, r in res)
        >= 2
    )
    ranks, workers, tp_grps = _layout(emb, heads, dp_size=2, tp=tp, pp=pp, ep=ep)
    res = _run(ranks, lambda j: workers[j].verify_drafts_block_engine(**kw))
    for r, w, out in zip(ranks, workers, res):
        n_real = counts[r.dp]
        # Expert parallelism: every rank runs the group's largest count, the
        # extra ones one-token dummies after the real rows.
        n_fwd = max(counts) if ep > 1 else n_real
        assert len(w.forwards) == n_fwd
        for row in w.forwards[n_real:]:
            assert torch.equal(row, torch.zeros(1, dtype=torch.long))
        if r.pp_rank == pp - 1:
            assert out == ref[r.dp][0]
        else:
            assert out == []  # only the last stage holds logits and verifies
    # Pipeline stages of a replica run the same rows in the same order.
    for d in range(2):
        seqs = [
            [x.tolist() for x in w.forwards]
            for r, w in zip(ranks, workers)
            if r.dp == d
        ]
        assert all(s == seqs[0] for s in seqs)
    # Vocab gathers only inside a replica's last stage, identical on its TP ranks.
    for (d, s), grp in tp_grps.items():
        if tp > 1 and s == pp - 1:
            assert grp.calls[0] and all(c == grp.calls[0] for c in grp.calls)
        else:
            assert not any(grp.calls)


def test_ep_without_lockstep_would_hang(fake_groups, monkeypatch):
    """Control: the fake all-to-all barrier does catch unequal forward counts."""
    emb, heads = _weights()
    kw = _drafts(emb, heads, [4, 1])
    ranks, workers, _ = _layout(emb, heads, dp_size=2, tp=1, pp=1, ep=2)
    for w in workers:
        w._expert_parallel_lockstep = lambda plans: None
    for r in ranks:
        r.ep_handle.grp.barrier._timeout = 1  # fail fast
    with pytest.raises(threading.BrokenBarrierError):
        _run(ranks, lambda j: workers[j].verify_drafts_block_engine(**kw))


def test_stash_head_off_the_last_stage_is_a_no_op():
    w = object.__new__(_cls())
    w.rank = 0
    w.model = torch.nn.Module()
    w.model.module = torch.nn.Module()  # no output_layer: not the last stage
    assert w.stash_head(version=0, keep=[0]) == 0
    assert w.head_stash_bytes() == 0
    w.model.module.post_process = False
    w.model.module.output_layer = _Head(torch.zeros(4, 2))
    assert w.stash_head(version=0, keep=[0]) == 0


def test_hidden_log_is_installed_on_the_engine_last_stage_only(monkeypatch):
    from megatron.core import parallel_state as ps

    head = _Head(torch.zeros(4, 2))
    for last, expected in [(False, None), (True, True)]:
        monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, _l=last, **k: _l)
        monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
        w = object.__new__(_cls())
        kernel = lambda *a, **k: None  # noqa: E731
        w.dynamic_inference_engine = SimpleNamespace(
            requests={},
            controller=SimpleNamespace(_sampling=SimpleNamespace(sample_kernel=kernel)),
        )
        w.inference_context = SimpleNamespace(
            config=SimpleNamespace(materialize_only_last_token_logits=True),
            num_speculative_tokens=0,
            max_requests=8,
        )
        w._gen_model = lambda: SimpleNamespace(
            module=SimpleNamespace(output_layer=head)
        )
        w._install_hidden_copy(w.inference_context)  # every rank, at engine build
        assert w.install_draft_hidden_log() is expected
        # The sampler wrapper (logs rows after CUDA-graph replays too) is
        # installed on the last stage only.
        sampling = w.dynamic_inference_engine.controller._sampling
        assert (sampling.sample_kernel is not kernel) == bool(expected)


# --------------------------------------------------------------------------
# _training_forward through the forward-only pipeline schedule
# --------------------------------------------------------------------------


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Fake microbatch iterator, model_forward and Megatron schedule.

    The schedule mimics forward_backward_pipelining_without_interleaving with
    forward_only=True, collect_non_loss_data=True: it calls the step function
    and, on the last stage only, loss_func(output, non_loss_data=True),
    returning the collected outputs; it enters config.no_sync_func (if any)
    without leaving it, as the real forward-only schedule does.
    """
    import megatron.core.pipeline_parallel as mpp
    import thundersync_rl.worker as worker_mod
    from megatron.core import parallel_state as ps

    state = SimpleNamespace(last=True, pp=2, calls=[], entered=0)
    monkeypatch.setattr(
        ps, "get_pipeline_model_parallel_world_size", lambda *a, **k: state.pp
    )
    monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, **k: state.last)

    def get_microbatch_iterator(data, cfg, mbs, **kw):
        ids = data["input_ids"]
        mb = SimpleNamespace(
            data_dict=data,
            input_ids_cp_sharded=ids,
            position_ids=None,
            attention_mask=None,
            packed_seq_params=None,
            mtp_loss_mask=None,
            padding_mask=None,
            media_token_validity_mask=None,
        )
        return iter([mb]), 1, mbs, ids.shape[1], ids.shape[1]

    def model_forward(model, data_dict, input_ids_cp_sharded, **kw):
        state.calls.append(("model_forward", model, tuple(input_ids_cp_sharded.shape)))
        return input_ids_cp_sharded.float().unsqueeze(-1).expand(-1, -1, 3) * 2

    def schedule(**kw):
        state.calls.append(
            ("schedule", {k: v for k, v in kw.items() if k != "forward_step_func"})
        )
        cfg = kw["model"].config
        if cfg.no_sync_func is not None:
            cfg.no_sync_func()
            state.entered += 1
        out, loss_func = kw["forward_step_func"](kw["data_iterator"], kw["model"])
        if not state.last:
            return []
        return [loss_func(out, non_loss_data=True)]

    monkeypatch.setattr(worker_mod, "get_microbatch_iterator", get_microbatch_iterator)
    monkeypatch.setattr(worker_mod.megatron_train, "model_forward", model_forward)
    monkeypatch.setattr(mpp, "get_forward_backward_func", lambda: schedule)
    return state


def _pp_worker():
    w = object.__new__(_cls())
    w.cfg = {"make_sequence_length_divisible_by": 64}
    w.media_placeholder_token_id = None
    w.mcore_state = SimpleNamespace(straggler_timer=None)
    w.delegate_pack_to_model = False
    w.delegate_mtp_loss_mask_to_model = False
    w.model_slices_context_parallel_inputs = False
    w.mtp_enabled = False
    w.defer_fp32_logits = False
    w.model = torch.nn.Module()
    w.model.config = SimpleNamespace(no_sync_func=None)
    return w


def test_training_forward_pp_runs_the_forward_only_schedule(fake_pipeline):
    w = _pp_worker()
    sync_calls = []
    no_sync = lambda: sync_calls.append(1)  # noqa: E731
    w.model.config.no_sync_func = no_sync
    row = torch.arange(1, 71)
    # PP = 1: model_forward directly on the training model.
    fake_pipeline.pp = 1
    ref = w._training_forward(row)
    assert (
        fake_pipeline.calls[-1][0] == "model_forward"
        and fake_pipeline.calls[-1][1] is w.model
    )
    assert ref.shape == (1, 128, 3)  # padded to the next multiple of 64
    # PP = 2, last stage: the schedule, forward-only, one micro-batch of one row.
    fake_pipeline.pp, fake_pipeline.calls = 2, []
    with torch.no_grad():
        out = w._training_forward(row)
    (sched,) = [c for c in fake_pipeline.calls if c[0] == "schedule"]
    args = sched[1]
    assert args["forward_only"] and args["collect_non_loss_data"]
    assert args["num_microbatches"] == 1 and args["micro_batch_size"] == 1
    assert args["seq_length"] == args["decoder_seq_length"] == 128
    assert args["model"] is w.model
    assert torch.equal(out, ref)  # the same forward, collected on the last stage
    # The DDP no_sync hook is bypassed for the call and restored afterwards.
    assert fake_pipeline.entered == 0 and not sync_calls
    assert w.model.config.no_sync_func is no_sync
    # Other stages: the same schedule, no logits.
    fake_pipeline.last = False
    with torch.no_grad():
        assert w._training_forward(row) is None
    # Autograd across stages needs the backward schedule: not implemented.
    with (
        torch.enable_grad(),
        pytest.raises(NotImplementedError, match="not implemented"),
    ):
        w._training_forward(row)


def test_dummy_training_forward_is_one_padded_token(fake_pipeline):
    w = _pp_worker()
    fake_pipeline.pp = 1
    w._dummy_training_forward()
    (call,) = fake_pipeline.calls
    assert call[2] == (1, 64)  # one token padded to make_sequence_length_divisible_by


def test_engine_verify_rejects_token_dropping_moe():
    w = object.__new__(_cls())
    w.cfg = {"megatron_cfg": {"batch_invariant_mode": True}}
    w.model = torch.nn.Module()
    w.model.config = SimpleNamespace(num_moe_experts=8, moe_expert_capacity_factor=1.0)
    with pytest.raises(RuntimeError, match="dropless MoE"):
        w.verify_drafts_block_engine(
            rows=[],
            prompt_lens=[],
            segments=[],
            engine_logq=[],
            hidden=[],
            vocab_limit=1,
            seed=0,
            keys=[],
        )
    w.model.config.moe_expert_capacity_factor = None
    w._check_dropless_moe()  # dropless: fine


# --------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------


def _layout_policy(pp=1, ep=1, gen=True):
    p = _policy(pipeline_model_parallel_size=pp, expert_model_parallel_size=ep)
    if gen:
        p["generation"]["mcore_generation_config"].update(
            pipeline_model_parallel_size=pp, expert_model_parallel_size=ep
        )
    return p


@pytest.mark.parametrize("pp, ep", [(2, 1), (1, 2), (2, 2), (4, 8)])
def test_engine_source_accepts_pp_and_ep_with_an_untested_warning(pp, ep):
    for gen in (True, False):  # generation layout explicit or inherited
        with pytest.warns(UserWarning, match="untested end to end"):
            validate_engine_verify_source(_ts(), _layout_policy(pp, ep, gen))


def test_engine_source_pp_ep_1_has_no_warning(recwarn):
    validate_engine_verify_source(_ts(), _layout_policy())
    assert not [w for w in recwarn if "untested" in str(w.message)]
