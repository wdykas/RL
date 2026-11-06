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
"""Lane engine per-decode-step host work (C6, agent lanehost 10-26).

The ts engine's host loops read len(prompt_tokens) of every running request several times per decode step (the wide
sampler's key positions and the hidden-state log's row indices). ``_RequestLengths`` caches the lengths while the
request's record and prompt tensors are the same objects; every value must equal the uncached one, through
evictions (new record), prompt re-assignments (``.cpu()``, a stop-word trim) and request-id reuse. MCore's
async-schedule log-prob materialization is replaced by one ``tolist()`` per step with the same output.
"""

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from thundersync_rl.worker import (
    ThunderSyncMegatronPolicyWorker,
    _install_one_tolist_log_probs,
    _request_positions,
    _RequestLengths,
)


class _Prompt(list):
    """A prompt whose len() calls are counted (stands in for the request's prompt tensor)."""

    calls = 0

    def __len__(self):
        _Prompt.calls += 1
        return super().__len__()


def _req(prompt, key, gen=0):
    sp = SimpleNamespace() if key is None else SimpleNamespace(ts_hidden_key=key)
    return SimpleNamespace(prompt_tokens=_Prompt([0] * prompt), generated_tokens=[0] * gen, sampling_params=sp)


class _Engine:
    def __init__(self):
        self.requests = {}

    def get_request(self, rid):
        return self.requests[rid].record[-1]


def _uncached(eng, rid):
    rec = eng.requests[rid].record
    cur = rec[-1]
    pos = len(cur.prompt_tokens) + len(cur.generated_tokens)
    key = getattr(cur.sampling_params, "ts_hidden_key", None)
    return pos, key, pos - len(rec[0].prompt_tokens)


def test_request_lengths_equal_uncached_through_evictions_trims_and_id_reuse():
    rng = random.Random(0)
    eng = _Engine()
    nxt = 0
    for _ in range(12):
        eng.requests[nxt] = SimpleNamespace(record=[_req(rng.randint(5, 50), f"k{nxt}")])
        nxt += 1
    lens = _RequestLengths()
    for step in range(400):
        ids = np.array(sorted(eng.requests), dtype=np.int64)
        got = _request_positions(eng, ids, lens)
        ref = _request_positions(eng, ids)
        assert got.tolist() == ref.tolist(), step
        lens.begin()
        for rid in ids.tolist():
            c = lens.get(eng, rid)
            pos, key, t = _uncached(eng, rid)
            assert (c[2] + len(c[0].generated_tokens), c[5], c[2] + len(c[0].generated_tokens) - c[4]) == (pos, key, t)
        for rid in ids.tolist():
            e = eng.requests[rid]
            cur = e.record[-1]
            u = rng.random()
            if u < 0.03:  # eviction checkpoint: a new record whose prompt holds the generated tokens
                p = list(cur.prompt_tokens) + list(cur.generated_tokens)
                new = _req(0, getattr(cur.sampling_params, "ts_hidden_key", None))
                new.prompt_tokens = _Prompt(p)
                e.record.append(new)
            elif u < 0.05:  # prompt re-assigned to a same-length object (.cpu())
                cur.prompt_tokens = _Prompt(list(cur.prompt_tokens))
            elif u < 0.06 and len(e.record) > 1:  # terminal stop-word trim of the active segment
                cur.prompt_tokens = _Prompt(list(cur.prompt_tokens)[:-1])
            elif u < 0.08:  # finished; its id is reused by a new request with another prompt
                eng.requests[rid] = SimpleNamespace(record=[_req(rng.randint(5, 50), None if u < 0.07 else f"r{step}")])
            else:
                cur.generated_tokens.append(0)
        if rng.random() < 0.1:
            eng.requests[nxt] = SimpleNamespace(record=[_req(rng.randint(5, 50), f"k{nxt}")])
            nxt += 1
    # Bounded: only the requests of the last two calls are kept.
    assert len(lens._old) <= len(eng.requests) + 2 and len(lens._new) <= len(eng.requests) + 2


def test_request_positions_reads_each_prompt_length_once_per_record():
    eng = _Engine()
    for r in range(64):
        eng.requests[r] = SimpleNamespace(record=[_req(100 + r, f"k{r}")])
    ids = np.arange(64)
    lens = _RequestLengths()
    _request_positions(eng, ids, lens)
    _Prompt.calls = 0
    for _ in range(10):
        for e in eng.requests.values():
            e.record[-1].generated_tokens.append(0)
        out = _request_positions(eng, ids, lens)
    assert _Prompt.calls == 0  # was 64 per call (one Tensor.__len__ per row)
    assert out.tolist() == [100 + r + 10 for r in range(64)]


def test_hidden_log_rows_use_cached_lengths(monkeypatch):
    """install_draft_hidden_log's row loop: same (row, key, generated index) entries, no per-step prompt len()."""
    from megatron.core import parallel_state as ps
    from megatron.core.inference import utils as inf_utils

    monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, **k: True)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: 0)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: 1)
    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)

    class Head(torch.nn.Module):
        sequence_parallel = False

        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1, 6))

        def forward(self, input_, weight=None, runtime_gather_output=None):
            return input_.sum(-1, keepdim=True), None

    model = SimpleNamespace(module=SimpleNamespace(output_layer=Head()))
    sampling = SimpleNamespace(sample_kernel=lambda logits, n, context, **kw: None)
    eng = _Engine()
    eng.requests = {1: SimpleNamespace(record=[_req(4, "d|0")]), 2: SimpleNamespace(record=[_req(2, None)]),
                    3: SimpleNamespace(record=[_req(7, "d|1")])}
    eng.controller = SimpleNamespace(_sampling=sampling, _compact_async_sched_logits=lambda idx: None)
    ctx = SimpleNamespace(
        max_requests=8, config=SimpleNamespace(materialize_only_last_token_logits=True), num_speculative_tokens=0,
        paused_request_count=0, total_request_count=3, request_ids=torch.tensor([1, 2, 3]),
        request_in_prefill_status_tensor=torch.tensor([0, 0, 0]),
    )
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = eng
    w.inference_context = ctx
    w._gen_model = lambda: model
    w.inference_model = None
    w._install_hidden_copy(ctx)
    assert w.install_draft_hidden_log() is True
    seen = []
    log = w._hidden_log
    real = log.record
    log.record = lambda entries, hidden: (seen.append(list(entries)), real(entries, hidden))[1]
    head = model.module.output_layer
    for s in range(6):
        head(torch.randn(3, 1, 6))
        if s == 2:  # eviction of request 3: new record, prompt = prompt + 2 generated, generated restarts
            cur = eng.requests[3].record[-1]
            new = _req(0, "d|1")
            new.prompt_tokens = _Prompt([0] * (7 + len(cur.generated_tokens)))
            eng.requests[3].record.append(new)
        if s == 3:
            _Prompt.calls = 0
        sampling.sample_kernel(None, 3, ctx)
        for e in eng.requests.values():
            e.record[-1].generated_tokens.append(0)
    assert [x for e in seen for x in e if x[1] == "d|0"] == [(0, "d|0", t) for t in range(6)]
    assert [x for e in seen for x in e if x[1] == "d|1"] == [(2, "d|1", t) for t in range(6)]
    assert _Prompt.calls == 0  # steps 3-5: every length cached (was 3 per tagged row per step)


def _transfer(rows, row_counts, top_n=None, dims=1):
    view = torch.randn(rows) if dims == 1 else torch.randn(rows, 1)
    n = len(row_counts)
    return SimpleNamespace(
        selected_log_probs_cpu_view=view,
        top_n_log_probs_cpu_view=torch.randn(rows, 2) if top_n else None,
        top_n_token_ids_cpu_view=torch.randint(0, 9, (rows, 2)) if top_n else None,
        row_counts=row_counts,
        top_n_counts=top_n or [0] * n,
        skip_prompt_log_probs=[True] * n,
        num_decode_requests=n,
        cpu_ready_event=None,
        gpu_result=None,
    )


@pytest.mark.parametrize("dims", [1, 2])
def test_one_tolist_log_probs_equal_mcore(dims):
    from megatron.core.inference.text_generation_controllers.text_generation_controller import (
        TextGenerationController,
    )

    orig = TextGenerationController._materialize_async_sched_log_probs
    ctl = SimpleNamespace(_materialize_async_sched_log_probs=orig)
    _install_one_tolist_log_probs(ctl)
    assert ctl._materialize_async_sched_log_probs is not orig
    for counts in ([1] * 384, [1] * 5 + [3, 1, 7], []):
        tr = _transfer(sum(counts), counts, dims=dims)
        assert ctl._materialize_async_sched_log_probs(tr) == orig(tr)
    # Top-n log-probs or MTP accepted counts: MCore's own code.
    tr = _transfer(4, [1, 1, 1, 1], top_n=[0, 2, 0, 0])
    a, b = ctl._materialize_async_sched_log_probs(tr), orig(tr)
    assert a[0] == b[0] and set(a[1]) == set(b[1])
    acc = torch.tensor([0, 1])
    tr = _transfer(4, [2, 2])
    assert ctl._materialize_async_sched_log_probs(tr, acc) == orig(tr, acc)
    assert ctl._materialize_async_sched_log_probs(None) == (None, None)
    _install_one_tolist_log_probs(ctl)  # idempotent
    assert ctl._materialize_async_sched_log_probs._ts_one_tolist


# ---------------------------------------------------------------------------------------------------------------
# P1-P4 (lane engine host work, approved 10-26): the logged bytes, merged records, routes and anchors are those of
# the single-rank logs they replace.
# ---------------------------------------------------------------------------------------------------------------

import asyncio
import pickle
import threading


class _PickleStore:
    """Object store stand-in: pickle protocol 5 with out-of-band buffers (as Ray), copies of the bytes."""

    def __init__(self):
        self.objs, self.n, self.oob = {}, 0, 0
        self.lock = threading.Lock()

    def put(self, obj):
        bufs = []
        data = pickle.dumps(obj, protocol=5, buffer_callback=bufs.append)
        with self.lock:
            self.n += 1
            self.objs[self.n] = (data, [bytes(b.raw()) for b in bufs])
            self.oob += len(bufs)
            return self.n

    def get(self, ref):
        with self.lock:
            data, bufs = self.objs[ref]
        return pickle.loads(data, buffers=[memoryview(b) for b in bufs])

    def free(self, refs):
        with self.lock:
            for r in refs:
                self.objs.pop(r)


def _bits(t):
    return t.contiguous().view(torch.uint8)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_hidden_payload_is_the_same_bytes_out_of_band(dtype):
    """P3: a block travels as an integer view of its bytes (pickle-5 out-of-band, no copy at pickling) and reads
    back bitwise, NaN payloads and signed zeros included."""
    from thundersync_rl.engine_q import hidden_block, hidden_payload

    g = torch.Generator().manual_seed(0)
    if dtype == torch.float32:
        b = torch.randint(-(2**31), 2**31 - 1, (37, 24), generator=g, dtype=torch.int32).view(dtype)
    else:
        b = torch.randint(-(2**15), 2**15, (37, 24), generator=g).to(torch.int16).view(dtype)
    pay = hidden_payload(b)
    bufs = []
    data = pickle.dumps(pay, protocol=5, buffer_callback=bufs.append)
    assert len(bufs) == 1 and bufs[0].raw().nbytes == b.numel() * b.element_size()
    assert np.frombuffer(bufs[0].raw(), np.uint8).ctypes.data == b.data_ptr()  # zero-copy view of the block
    assert len(data) < 512  # only the header is pickled in band
    back = hidden_block(pickle.loads(data, buffers=bufs))
    assert back.dtype == dtype and back.shape == b.shape and torch.equal(_bits(back), _bits(b))
    assert back.untyped_storage().nbytes() == b.numel() * b.element_size()  # owns its bytes
    assert hidden_block(b) is b  # tensors (older blocks, test stores) pass through


def test_sealing_log_ray_path_puts_payloads_read_back_bitwise(monkeypatch):
    """P3 + P4 through the default (ray) put path: sealed blocks and tails are payloads; pop / take / HiddenRows /
    HiddenPrefetch read the logged bytes."""
    import ray

    import thundersync_rl.engine_q as eq

    store = _PickleStore()
    monkeypatch.setattr(ray, "put", store.put)
    monkeypatch.setattr(ray, "get", store.get)
    monkeypatch.setattr(eq, "ray_free", store.free)
    g = torch.Generator().manual_seed(1)
    rows = {k: torch.randn(n, 16, generator=g).to(torch.bfloat16) for k, n in (("a", 70), ("b", 9))}
    log = eq.SealingHiddenLog(chunk_rows=32, block_rows=8)
    for t in range(70):
        ks = [k for k in rows if t < rows[k].shape[0]]
        log.record_rows(list(range(len(ks))), ks, [t] * len(ks), torch.stack([rows[k][t] for k in ks]))
    piece = log.take_refs("a", 0, 20)
    out = log.take_refs("a", 20, 50)  # the rest of a draft taken in pieces
    log.close("a")
    refs = eq.HiddenRefs.from_blocks(piece + out)
    hr = eq.HiddenRows(refs)
    assert torch.equal(_bits(hr(list(range(70)))), _bits(rows["a"]))
    pf = eq.HiddenPrefetch()
    pf.submit([refs])
    hr2 = eq.HiddenRows(refs, prefetched=pf.futures)
    assert torch.equal(_bits(hr2([69, 3, 40])), _bits(rows["a"][[69, 3, 40]]))
    assert torch.equal(_bits(log.pop("b", 9)), _bits(rows["b"]))
    assert store.oob >= len(piece) + len(out)  # every block went out of band
    log.stop()


def test_vectorized_sealing_log_equals_hiddenlog_under_random_streams():
    """P4: the dense-window SealingHiddenLog hands out exactly HiddenLog's rows (bitwise) for random decode orders,
    eviction rewrites (below and above the frontier), holes, pieces, pops and closes, across chunk boundaries;
    every object is handed out or freed and retired chunks are released."""
    from thundersync_rl.engine_q import HiddenLog, SealingHiddenLog

    rng = random.Random(7)
    g = torch.Generator().manual_seed(7)
    for trial in range(4):
        B, d = rng.choice([4, 8]), 6
        store = _PickleStore()
        log = SealingHiddenLog(chunk_rows=rng.choice([16, 50]), block_rows=B, put=store.put, get=store.get,
                               free=store.free)
        ref = HiddenLog(chunk_rows=16)
        keys = [f"k{i}" for i in range(12)]
        lens = {k: rng.randint(1, 60) for k in keys}
        data = {k: torch.randn(lens[k], d, generator=g) for k in keys}
        holes = {(k, rng.randrange(lens[k])) for k in rng.sample(keys, 3)}
        taken = {k: 0 for k in keys}
        handed, closed = [], set()

        def check(got, exp):
            if exp is None:
                assert got is None
            else:
                assert got is not None and torch.equal(torch.cat([store.get(r) for r, _, _ in got]), exp)
                handed.extend(r for r, _, _ in got)

        for t in range(70):
            live = [k for k in keys if t < lens[k] and (k, t) not in holes and k not in closed]
            rng.shuffle(live)
            # An eviction recompute rewrites an earlier position (bitwise the same row; rows already handed out
            # are not rewritten here: the plain reference log has no frontier to drop them).
            tt = [t] * len(live)
            if live and rng.random() < 0.3:
                k = rng.choice(live)
                if taken[k] <= t:
                    live.append(k)
                    tt.append(rng.randint(taken[k], t))
            if live:
                hid = torch.stack([data[k][x] for k, x in zip(live, tt)])
                ents = list(zip(range(len(live)), live, tt))
                log.record(ents, hid)
                ref.record(ents, hid)
            if rng.random() < 0.2:  # a piece of a running draft
                k = rng.choice(keys)
                if k not in closed:
                    n = rng.randint(1, 13)
                    exp = ref.take(k, taken[k], n)
                    check(log.take_refs(k, taken[k], n), exp)
                    taken[k] += 0 if exp is None else int(exp.shape[0])
            if rng.random() < 0.05:
                k = rng.choice(keys)
                if k not in closed:
                    log.close(k), ref.close(k)
                    closed.add(k)
        for k in keys:
            if k in closed:
                assert log.pop_refs(k, lens[k]) is None
                continue
            if taken[k]:  # a draft taken in pieces ends with its last piece, then a close
                check(log.take_refs(k, taken[k], lens[k]), ref.take(k, taken[k], lens[k]))
                log.close(k), ref.close(k)
            else:
                check(log.pop_refs(k, lens[k]), ref.pop(k, lens[k]))
        log.stop()
        assert set(store.objs) == set(handed), trial  # nothing leaked by seals, re-splits or closes
        assert log.live_rows() == 0 and len(log._chunks) <= 1


class _HeadInput(torch.nn.Module):
    sequence_parallel = False

    def __init__(self, d):
        super().__init__()
        # A 4-row vocabulary shard: the engine-q probe reads real logit columns of it.
        self.weight = torch.nn.Parameter(torch.randn(4, d, generator=torch.Generator().manual_seed(0)))

    def forward(self, input_, weight=None, runtime_gather_output=None):
        return input_.sum(-1, keepdim=True), None


def _hidden_rank_worker(monkeypatch, eng, ctx, tp, rank, split):
    from megatron.core import parallel_state as ps

    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: rank)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: tp)
    monkeypatch.setenv("THUNDERSYNC_HIDDEN_TP_SPLIT", "1" if split else "0")
    model = SimpleNamespace(module=SimpleNamespace(output_layer=_HeadInput(8)))
    sampling = SimpleNamespace(sample_kernel=lambda logits, n, context, **kw: None)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = SimpleNamespace(
        requests=eng.requests, controller=SimpleNamespace(_sampling=sampling, _compact_async_sched_logits=lambda i: None)
    )
    w.inference_context = ctx
    w._gen_model = lambda: model
    w.inference_model = None
    w._install_hidden_copy(ctx)
    ok = w.install_draft_hidden_log(seal_blocks=True)
    return w, model.module.output_layer, sampling, ok


@pytest.mark.parametrize("probe", [False, True])
@pytest.mark.parametrize("tp", [2, 4])
def test_hidden_tp_split_merged_equals_tp0_log(monkeypatch, tp, probe):
    """P1: each TP rank logs the drafts it owns (crc32(key) mod TP); the driver's merge of every rank's hand-out
    (dict union, as ``_pop_hidden``) is bitwise TP rank 0's log of every draft, through evictions, id reuse,
    untagged rows and pieces; ranks own disjoint drafts and each rank's row loop visits only its own rows.
    ``probe``: the same with the engine-q source check on (THUNDERSYNC_ENGINE_Q_CHECK=1, CHECKS=1 runs), which
    runs the kernel's probe branch on real logits."""
    import ray
    from megatron.core import parallel_state as ps
    from megatron.core.inference import utils as inf_utils

    import thundersync_rl.engine_q as eq

    import thundersync_rl.worker as worker_mod
    from nemo_rl.models.megatron.fp16_gemm import head_matmul

    store = _PickleStore()
    monkeypatch.setattr(ray, "put", store.put)
    monkeypatch.setattr(ray, "get", store.get)
    monkeypatch.setattr(eq, "ray_free", store.free)
    if probe:
        monkeypatch.setenv("THUNDERSYNC_ENGINE_Q_CHECK", "1")
    else:
        monkeypatch.delenv("THUNDERSYNC_ENGINE_Q_CHECK", raising=False)
    monkeypatch.setattr(ps, "is_pipeline_last_stage", lambda *a, **k: True)
    monkeypatch.setattr(inf_utils.InferenceMode, "is_active", lambda *a, **k: True)
    rng = random.Random(tp)
    eng = _Engine()
    R = 12
    for r in range(R):
        eng.requests[r] = SimpleNamespace(record=[_req(rng.randint(3, 9), None if r % 5 == 4 else f"d|{r}")])
    ctx = SimpleNamespace(
        max_requests=16, config=SimpleNamespace(materialize_only_last_token_logits=True), num_speculative_tokens=0,
        paused_request_count=0, total_request_count=R, request_ids=torch.arange(R),
        request_in_prefill_status_tensor=torch.zeros(R, dtype=torch.int32),
    )
    ranks = [_hidden_rank_worker(monkeypatch, eng, ctx, tp, r, True) for r in range(tp)]
    ref = _hidden_rank_worker(monkeypatch, eng, ctx, tp, 0, False)
    assert all(ok is True for *_, ok in ranks) and ref[3] is True
    visits = [0] * (tp + 1)  # split ranks, then the unsplit reference
    which = {id(w._hidden_lens): i for i, (w, *_rest) in enumerate(ranks + [ref])}
    real_get = worker_mod._RequestLengths.get

    def counted(self, e, rid):
        if id(self) in which:
            visits[which[id(self)]] += 1
        return real_get(self, e, rid)

    monkeypatch.setattr(worker_mod._RequestLengths, "get", counted)
    g = torch.Generator().manual_seed(tp)
    keys_seen = set()
    for step in range(40):
        x = torch.randn(R, 1, 8, generator=g)
        for _w, head, sampling, _ok in ranks + [ref]:
            head(x)
            # The probe compares logged rows with the logits the engine sampled from (this rank's vocab shard).
            logits = head_matmul(x[:, 0], head.weight) if probe else None
            sampling.sample_kernel(logits, R, ctx)
        ctx.request_in_prefill_status_tensor.zero_()  # flags set below mark the next step's new requests
        for rid in range(R):
            e = eng.requests[rid]
            cur = e.record[-1]
            keys_seen.add(getattr(cur.sampling_params, "ts_hidden_key", None))
            u = rng.random()
            if u < 0.04:  # eviction: new record whose prompt holds the generated tokens
                new = _req(0, getattr(cur.sampling_params, "ts_hidden_key", None))
                new.prompt_tokens = _Prompt([0] * (len(cur.prompt_tokens) + len(cur.generated_tokens)))
                e.record.append(new)
            elif u < 0.07:  # finished; the id is reused by a new draft (in prefill this step)
                eng.requests[rid] = SimpleNamespace(record=[_req(rng.randint(3, 9), f"d|{rid}.{step}")])
                ctx.request_in_prefill_status_tensor[rid] = 1
            else:
                cur.generated_tokens.append(0)
    keys = sorted(k for k in keys_seen if k)
    owners = {k: eq.hidden_owner(k, tp) for k in keys}
    assert len(set(owners.values())) > 1  # the drafts really are spread over the ranks
    ka, kb = keys[::2], keys[1::2]
    # Half the drafts popped whole; the other half taken in two pieces (keys stay open), as the driver does.
    calls = (
        ("pop_draft_hidden", ka, {"lengths": [64] * len(ka)}),
        ("take_draft_hidden", kb, {"starts": [0] * len(kb), "lengths": [5] * len(kb)}),
        ("take_draft_hidden", kb, {"starts": [5] * len(kb), "lengths": [59] * len(kb)}),
    )
    for call, ks, kw in calls:
        merged = {}
        for i, (w, *_) in enumerate(ranks):
            r = getattr(w, call)(ks, as_refs=True, **kw)
            r.pop(eq.HIDDEN_STATS_KEY)
            assert not set(r) & set(merged)  # disjoint: one owner per draft
            assert all(owners[k] == i for k in r)
            merged.update(r)
        exp = getattr(ref[0], call)(ks, as_refs=True, **kw)
        assert exp, call
        exp.pop(eq.HIDDEN_STATS_KEY)
        assert set(merged) == set(exp)
        for k in exp:
            a = torch.cat([eq.hidden_block(store.get(b[0])) for b in merged[k]])
            b = torch.cat([eq.hidden_block(store.get(b[0])) for b in exp[k]])
            assert torch.equal(_bits(a), _bits(b)), k
    # Row-loop work: every tagged row is looked up by exactly one rank (TP0 alone used to look up all of them).
    assert sum(visits[:tp]) == visits[tp] and max(visits[:tp]) < visits[tp]
    for w, *_ in ranks + [ref]:
        w._hidden_log.stop()


def _route_rank_worker(monkeypatch, requests, ctx, tp, rank, split, full):
    from megatron.core import parallel_state as ps

    L, K = full[0].shape[1], full[0].shape[2]
    replays = [SimpleNamespace(static_buffer=None, recorded_topk_idx=None) for _ in range(L)]

    def forward_logits(input_ids, position_ids):
        chunk = ctx.padded_active_token_count // tp
        cur = forward_logits.full[0]
        for j, r in enumerate(replays):
            r.recorded_topk_idx = cur[rank * chunk : (rank + 1) * chunk, j].clone()

    forward_logits.full = full
    controller = SimpleNamespace(
        _dynamic_step_forward_logits=forward_logits, _dynamic_step_context_init=lambda **kw: (None, None, None)
    )
    monkeypatch.setattr(ps, "get_tensor_model_parallel_world_size", lambda *a, **k: tp)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda *a, **k: rank)
    monkeypatch.setenv("THUNDERSYNC_ROUTE_TP_SPLIT", "1" if split else "0")
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.dynamic_inference_engine = SimpleNamespace(requests=requests, controller=controller)
    w.inference_context = ctx
    w._route_record = {"replays": replays, "layers": [1, 3], "sequence_parallel": True}
    assert w.install_draft_route_log() == [1, 3]
    return w, controller


@pytest.mark.parametrize("tp", [2, 4])
def test_route_tp_split_merged_equals_tp0_log(monkeypatch, tp):
    """P1: under sequence parallelism each TP rank logs the tagged tokens of its own token shard (no all-gather);
    the driver's merge of the ranks' parts gives TP rank 0's all-gathered log: the same RouteRefs length /
    complete / per-layer prefix, and bitwise the same anchors (whole rows and offset windows), through prefill
    chunks spanning shards, decode steps, a draft taken in pieces and a draft with an unlogged position."""
    import torch.distributed as dist
    from megatron.core import parallel_state as ps

    import thundersync_rl.worker as worker_mod
    from thundersync_rl.routes import fetch_anchors, merge_route_refs, merge_tp_route_parts

    monkeypatch.setattr(worker_mod.ray, "put", lambda t: t)
    monkeypatch.setattr(ps, "get_tensor_model_parallel_group", lambda *a, **k: None)
    L, K, P = 2, 3, 64
    full = [torch.zeros(P, L, K, dtype=torch.int32)]

    def all_gather_into_tensor(out, local, group=None):
        out.copy_(full[0][: out.shape[0]])

    monkeypatch.setattr(dist, "all_gather_into_tensor", all_gather_into_tensor)

    def req(key):
        sp = SimpleNamespace() if key is None else SimpleNamespace(ts_hidden_key=key)
        return SimpleNamespace(record=[SimpleNamespace(sampling_params=sp)])

    requests = {10 + i: req(None if i == 3 else f"r|{i}") for i in range(6)}
    nslots = len(requests)
    ctx = SimpleNamespace(
        active_token_count=0, padded_active_token_count=0,
        token_to_request_idx=torch.zeros(P, dtype=torch.int32), token_to_pos_ids=torch.zeros(P, dtype=torch.long),
        request_ids=torch.tensor(sorted(requests), dtype=torch.int32),
        request_in_prefill_status_tensor=torch.ones(nslots, dtype=torch.int32), total_request_count=nslots,
    )
    ws = [_route_rank_worker(monkeypatch, requests, ctx, tp, r, True, full) for r in range(tp)]
    ref = _route_rank_worker(monkeypatch, requests, ctx, tp, 0, False, full)
    rng = random.Random(tp)
    pos = [0] * nslots
    plen = [rng.randint(3, 9) for _ in range(nslots)]
    g = torch.Generator().manual_seed(tp)

    def step(tokens):  # tokens: [(slot, count)]
        slots, poss = [], []
        for s, c in tokens:
            slots += [s] * c
            poss += list(range(pos[s], pos[s] + c))
            pos[s] += c
        n = len(slots)
        pad = -(-n // (4 * tp)) * 4 * tp
        ctx.active_token_count, ctx.padded_active_token_count = n, pad
        ctx.token_to_request_idx[:n] = torch.tensor(slots, dtype=torch.int32)
        ctx.token_to_pos_ids[:n] = torch.tensor(poss)
        full[0] = torch.randint(0, 64, (P, L, K), generator=g, dtype=torch.int32)
        for _w, c in ws + [ref]:
            c._dynamic_step_context_init(is_dummy_forward=False)
            c._dynamic_step_forward_logits(None, None)

    def pop(keys, close, bases=None):
        parts = {}
        for w, _ in ws:
            for k, v in w.pop_draft_routes(keys, close=close).items():
                parts.setdefault(k, []).extend(v if isinstance(v, list) else [v])
        exp = {k: [v] for k, v in ref[0].pop_draft_routes(keys, close=close).items()}
        assert set(parts) == set(exp)
        bases = bases or {}
        got = {k: merge_route_refs(merge_tp_route_parts(parts[k]), 200, [1, 3], base=bases.get(k, (None, None))[0])
               for k in parts}
        want = {k: merge_route_refs(exp[k], 200, [1, 3], base=bases.get(k, (None, None))[1]) for k in exp}
        for k in got:
            a, b = got[k], want[k]
            assert (a.length, a.complete, a.prefix) == (b.length, b.complete, b.prefix), k
            for lo in (0, 5):
                fa = fetch_anchors(a, [1, 3], K, 60, lambda t: t, lo=lo)
                fb = fetch_anchors(b, [1, 3], K, 60, lambda t: t, lo=lo)
                assert torch.equal(fa, fb), (k, lo)
        return got, want

    step([(s, plen[s]) for s in range(nslots)])  # prefill: every prompt in one step, spanning the shards
    ctx.request_in_prefill_status_tensor.zero_()
    for _ in range(6):
        order = list(range(nslots))
        rng.shuffle(order)
        step([(s, 1) for s in order])
    keys = [f"r|{i}" for i in range(6) if i != 3]
    got0, want0 = pop(keys[:2], close=False)  # a running draft's routes taken in pieces
    pos[0] += 1  # slot 0 skips a position: its draft is cut there (complete stops)
    for _ in range(5):
        step([(s, 1) for s in range(nslots)])
    pop(keys, close=True, bases={k: (got0[k], want0[k]) for k in got0})


def test_merge_tp_route_parts_unions_logged_positions():
    from thundersync_rl.routes import fetch_anchors, logged_mask, merge_route_refs, merge_tp_route_parts, put_route_rows

    rows = torch.arange(10 * 1 * 2, dtype=torch.int16).view(10, 1, 2)
    p0, p1 = np.array([0, 1, 3, 4]), np.array([2, 5, 7])
    parts = [
        ((1,), 0, 2, put_route_rows(p0, rows[p0], lambda t: t), logged_mask(p0, 0)),
        ((1,), 2, 3, put_route_rows(p1, rows[p1], lambda t: t), logged_mask(p1, 2)),
    ]
    (m,) = merge_tp_route_parts(parts)
    assert m[0] == (1,) and (m[1], m[2]) == (0, 6)  # 0..5 logged by one rank or the other; 6 by none
    refs = merge_route_refs([m], 10, [1])
    assert refs.complete == 6
    a = fetch_anchors(refs, [1], 2, 10, lambda t: t)
    logged = sorted(p0.tolist() + p1.tolist())
    assert torch.equal(a[logged], rows[logged]) and a[[6, 8, 9]].eq(-1).all()


def test_cut_reads_come_from_engine_records_not_stream_frames(monkeypatch):
    """P2: peek / collect return the engines' records for running drafts whatever partial frames arrived (frame
    granularity 16 vs 64 tokens gives the same cut), so a coarser stream interval cannot change a cut's contents."""
    import thundersync_rl.worker as worker_mod

    monkeypatch.setattr(worker_mod.ray, "get", lambda refs: [])
    rec = {10: (list(range(100)), [-0.5] * 100), 11: (list(range(7)), [-0.1] * 7)}
    outs = []
    for interval in (16, 64):
        w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        w._inference_loop = loop
        w.inference_client = SimpleNamespace(abort_request=lambda rid: None)
        w.dynamic_inference_engine = None
        # Streamed frames: whole intervals of the 100 / 7 tokens decoded so far.
        w._draft_tokens = [list(range(100 // interval * interval)), list(range(7 // interval * interval))]
        w._draft_logprobs = [[-0.5] * len(w._draft_tokens[0]), [-0.1] * len(w._draft_tokens[1])]
        w._draft_done = [False, False]
        w._draft_streams = {0: SimpleNamespace(request_id=10), 1: SimpleNamespace(request_id=11)}
        peek = w.peek_drafts([0, 1], records=rec)
        coll = w.collect_drafts(records=rec)
        outs.append([(p[0].tolist(), None if p[2] is None else p[2].tolist()) for p in peek + coll])
        loop.call_soon_threadsafe(loop.stop)
    assert outs[0] == outs[1]
    assert outs[0][0][0] == list(range(100)) and outs[0][1][0] == list(range(7))


def test_client_loop_moves_the_listener_and_sends(monkeypatch):
    """P1b: rank 0's client listener runs on its own loop thread; frames, sends and draft coroutines run there;
    the engine loop keeps only the engine. Same client object (socket, request-id counter)."""
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    assert w._own_client_loop_wanted()
    engine_loop = asyncio.new_event_loop()
    threading.Thread(target=engine_loop.run_forever, daemon=True).start()
    w._inference_loop = engine_loop
    seen = {}

    class Client:
        def __init__(self):
            self.next_request_id = 5

        async def _recv_task(self):
            seen["recv"] = threading.current_thread().name
            while True:
                await asyncio.sleep(0.001)

        def pause_engines(self):
            seen["send"] = threading.current_thread().name

    c = Client()
    w.inference_client = c

    async def start():
        c._loop = asyncio.get_running_loop()
        c.listener_task = c._loop.create_task(c._recv_task())
        await asyncio.sleep(0.01)
        await w._move_client_to_own_loop()
        await w._client_send(c.pause_engines)

    asyncio.run_coroutine_threadsafe(start(), engine_loop).result(5)
    import time as _t

    _t.sleep(0.05)
    assert w._client_loop() is not engine_loop and c._loop is w._client_loop()
    assert seen["recv"] == "inference-client" and seen["send"] == "inference-client"
    assert w.inference_client is c and c.next_request_id == 5
    monkeypatch.setenv("THUNDERSYNC_CLIENT_LOOP", "0")
    assert not w._own_client_loop_wanted()
