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
"""Engine hidden states through the object store (hidden_transport="object_store").

The driver handles only references; the verifying learner fetches a draft's
segments only when its lazy scan needs a q row, so fully accepted drafts fetch
nothing; handles are released after verification.
"""

import asyncio
import threading

import pytest
import ray
import thundersync_rl.speculative as spec_mod
import torch
from thundersync_rl.engine_q import (
    HIDDEN_STATS_KEY,
    HiddenLog,
    HiddenRefs,
    HiddenRows,
    cat_hidden,
    put_hidden,
)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _Ref:
    """Stand-in for an ObjectRef (no data)."""

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f"_Ref({self.name})"


class _Store:
    """Fake object store: put/get with a fetch log."""

    def __init__(self):
        self.objs, self.fetches = {}, []
        self._lock = threading.Lock()

    def put(self, t):
        ref = _Ref(len(self.objs))
        self.objs[ref] = t.clone()
        return ref

    def get(self, ref, tag=None):
        with self._lock:
            self.fetches.append((tag, ref))
        return self.objs[ref]

    def free(self, refs):
        for r in refs:
            del self.objs[r]  # raises if freed twice


def _assert_no_tensors(x, path="drafts"):
    """The driver holds handles only: no tensor anywhere in ``x``."""
    assert not isinstance(x, torch.Tensor), path
    if isinstance(x, (list, tuple)):
        for j, y in enumerate(x):
            _assert_no_tensors(y, f"{path}[{j}]")
    elif isinstance(x, dict):
        for k, y in x.items():
            _assert_no_tensors(y, f"{path}[{k!r}]")
    elif isinstance(x, HiddenRefs):
        for s in x.segs:
            assert not isinstance(s.ref, torch.Tensor), path


# --------------------------------------------------------------------------
# Learner-side fetch
# --------------------------------------------------------------------------


def test_hidden_rows_fetch_only_the_segments_asked_for():
    store = _Store()
    h = torch.randn(7, 4, dtype=torch.bfloat16)
    parts = [h[:3], h[3:3], h[3:]]  # an empty middle segment (no tokens drafted)
    refs = cat_hidden([HiddenRefs.from_blocks(put_hidden(p, store.put)) for p in parts])
    assert refs.rows == 7 and len(refs.segs) == 2  # the empty one is skipped
    assert refs.nbytes == h.numel() * 2
    rows = HiddenRows(refs, store.get)
    assert rows.fetched_bytes == 0 and not store.fetches  # nothing until asked
    assert torch.equal(rows([5, 6]), h[[5, 6]])
    assert len(store.fetches) == 1 and rows.fetched_bytes == 4 * 4 * 2
    assert torch.equal(rows([6, 0, 3]), h[[6, 0, 3]])  # bitwise the driver path
    assert len(store.fetches) == 2 and rows.fetched_bytes == h.numel() * 2
    assert torch.equal(rows([1]), HiddenRows(h)([1]))  # cached: no new fetch
    assert len(store.fetches) == 2
    refs.release()
    assert all(s.ref is None for s in refs.segs)
    with pytest.raises(RuntimeError, match="released"):
        HiddenRows(refs, store.get)([0])


def test_blocks_bound_what_a_scan_near_the_end_fetches():
    store = _Store()
    h = torch.randn(600, 8, dtype=torch.bfloat16)
    blocks = put_hidden(h, store.put, block_rows=256)
    assert [n for _, n, _ in blocks] == [256, 256, 88]
    refs = HiddenRefs.from_blocks(blocks)
    rows = HiddenRows(refs, store.get)
    # The scan visits the top candidates and tau, here all in the last block.
    assert torch.equal(rows([599, 590, 540, 520]), h[[599, 590, 540, 520]])
    assert len(store.fetches) == 1 and rows.fetched_bytes == 88 * 8 * 2
    assert torch.equal(rows([300]), h[[300]])
    assert len(store.fetches) == 2


def test_worker_pop_draft_hidden_as_refs(monkeypatch):
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    store = _Store()
    monkeypatch.setattr(ray, "put", store.put)
    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w._hidden_log = HiddenLog(chunk_rows=4)
    h = torch.randn(5, 3)
    w._hidden_log.record([(r, "a", r) for r in range(3)], h[:3])
    w._hidden_log.record([(0, "b", 0)], h[3:4])
    out = w.pop_draft_hidden(["a", "b", "c"], [3, 2, 1], as_refs=True)
    stats = out.pop(HIDDEN_STATS_KEY)
    assert stats["put_bytes"] == (3 + 1) * 3 * 4  # storage bytes, not whole drafts
    # b has only its first row (a 1-row prefix; the driver truncates it), c none.
    assert set(out) == {"a", "b"}
    ((ref_b, n_b, _),) = out["b"]
    assert n_b == 1 and torch.equal(store.objs[ref_b], h[3:4])
    ((ref, n, nbytes),) = out["a"]
    assert isinstance(ref, _Ref) and n == 3 and nbytes == 3 * 3 * 4
    assert torch.equal(store.objs[ref], h[:3])
    assert w._hidden_log.live_rows() == 0  # host rows released at pop


# --------------------------------------------------------------------------
# Worker: object-store rows give exactly the driver path's results
# --------------------------------------------------------------------------


def _as_refs(hids, segs, store):
    """Handles of the test drafts' hidden states (4-row blocks per segment)."""
    out = []
    for h, sg in zip(hids, segs):
        parts = [h[a:b] if h.numel() else h for _, a, b in sg]
        out.append(
            cat_hidden(
                [HiddenRefs.from_blocks(put_hidden(p, store.put, 4)) for p in parts]
            )
        )
    return out


@cuda
def test_engine_worker_object_store_matches_driver_path(monkeypatch):
    from test_engine_q import _engine_drafts, _engine_worker

    w, emb, heads = _engine_worker(monkeypatch)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads, n=40)
    kw = dict(
        rows=rows,
        prompt_lens=plens,
        segments=segs,
        engine_logq=lqs,
        vocab_limit=1 << 30,
        seed=3,
        keys=keys,
    )
    ref = dict(w.verify_drafts_block_engine(hidden=hids, **kw))
    store = _Store()
    refs = _as_refs(hids, segs, store)
    monkeypatch.setattr(ray, "get", store.get)
    got = dict(w.verify_drafts_block_engine(hidden=refs, **kw))
    assert got.keys() == ref.keys()
    whole = rejected = 0
    for i, r in got.items():
        assert {**r, "hidden_bytes_fetched": 0} == ref[i]  # same decisions, log p
        g = len(lqs[i])
        if r["accepted"] == g:
            whole += 1
            assert r["hidden_bytes_fetched"] == 0  # whole drafts fetch nothing
        else:
            rejected += 1
            assert 0 < r["hidden_bytes_fetched"] <= refs[i].nbytes
    assert whole >= 2 and rejected >= 2
    fetched = {id(ref) for _, ref in store.fetches}
    assert len(fetched) == len(store.fetches)  # each segment fetched at most once
    assert sum(r["hidden_bytes_fetched"] for r in got.values()) < sum(
        x.nbytes for x in refs
    )


@cuda
@pytest.mark.parametrize("tp", [2])
def test_tp_ranks_fetch_the_same_segments(monkeypatch, tp):
    from test_engine_q import _engine_drafts, _tp_engine_workers

    w1, rank_worker, emb, heads, grp, local = _tp_engine_workers(monkeypatch, tp)
    rows, plens, segs, lqs, hids, keys = _engine_drafts(emb, heads)
    store = _Store()
    refs = _as_refs(hids, segs, store)
    monkeypatch.setattr(ray, "get", lambda r: store.get(r, local.rank))
    kw = dict(
        rows=rows,
        prompt_lens=plens,
        segments=segs,
        engine_logq=lqs,
        hidden=refs,
        vocab_limit=1 << 30,
        seed=3,
        keys=keys,
    )
    res = grp.run(lambda r: rank_worker(r).verify_drafts_block_engine(**kw))
    assert all(out == res[0] for out in res)
    per_rank = [[ref for t, ref in store.fetches if t == r] for r in range(tp)]
    assert per_rank[0] and all(p == per_rank[0] for p in per_rank)
    local.tp, local.rank = 1, 0
    monkeypatch.setattr(ray, "get", store.get)
    assert [(i, {**r, "hidden_bytes_fetched": 0}) for i, r in res[0]] == [
        (i, {**r, "hidden_bytes_fetched": 0})
        for i, r in w1.verify_drafts_block_engine(**{**kw, "hidden": hids})
    ]


# --------------------------------------------------------------------------
# Driver (fake worker groups)
# --------------------------------------------------------------------------


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _spec(learner, base, **kw):
    from test_speculative_driver import _spec as make

    return make(learner, base_handlers=base, verify_source="engine", **kw)


D = 4


def test_driver_passes_handles_and_frees_them_after_verification(_no_ray):
    from test_speculative_driver import EOD

    store = _Store()
    seen = {}

    def pop(keys, lengths, as_refs=False):
        assert as_refs
        hid = {0: torch.ones(2, D), 1: torch.ones(3, D)}  # draft 2: not logged
        return {
            k: put_hidden(hid[j].bfloat16(), store.put)
            for j, k in enumerate(keys)
            if j in hid
        }

    def verify(rows, keys, engine_logq, hidden, **_):
        seen["hidden"] = hidden
        assert all(isinstance(h, HiddenRefs) for h in hidden)
        assert all(s.ref is not None for h in hidden for s in h.segs)
        # Draft 0 accepted whole (nothing fetched); draft 1 rejected at 1.
        return [
            (
                0,
                {
                    "accepted": 2,
                    "next": 0,
                    "logprobs": [-0.1, -0.2],
                    "draft_logq": [-0.5, -0.25],
                    "hidden_bytes_fetched": 0,
                },
            ),
            (
                1,
                {
                    "accepted": 1,
                    "next": 7,
                    "logprobs": [-0.1, -0.3],
                    "draft_logq": [-0.5],
                    "hidden_bytes_fetched": 24,
                },
            ),
        ]

    s = _spec(
        {"verify_drafts_block_engine": verify, "stash_head": lambda **kw: 1},
        {
            "install_draft_hidden_log": lambda **_: True,
            "start_drafts": lambda prompts, budget, interval, hidden_keys: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": lambda records=None, record_refs=None, request_ids=None: [
                ([4, EOD], True, [-0.5, -0.25]),
                ([4, 5, 6], False, [-0.5, -0.5, -0.5]),
                ([4], False, [-0.5]),
            ],
            "pop_draft_hidden": pop,
        },
    )
    assert s.hidden_transport == "object_store"
    s._free_refs = store.free
    prompts = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
    drafts = asyncio.run(s.draft_streaming(prompts))()
    _assert_no_tensors([d[7] for (d,) in drafts])
    assert isinstance(drafts[0][0][7], HiddenRefs) and drafts[2][0][7] is None
    assert s.stats["spec/hidden_bytes_logged"] == (2 + 3) * D * 2
    s.verify(drafts)
    assert s.stats["spec/engine_q_dropped"] == 1.0
    assert s.stats["spec/hidden_bytes_fetched"] == 24
    # Verified: every handle released on the driver and every object freed.
    assert all(sg.ref is None for h in seen["hidden"] for sg in h.segs)
    assert all(sg.ref is None for (d,) in drafts[:2] for sg in d[7].segs)
    assert store.objs == {}
    assert s.stats["spec/hidden_bytes_freed"] == (2 + 3) * D * 2


def test_driver_cohort_segments_are_handles_and_freed(_no_ray):
    from thundersync_rl.speculative import cohort_draft

    store = _Store()
    rounds = [
        # round 0: two drafts, both logged
        ([([5, 6], False, [-0.1, -0.2]), ([8], False, [-0.4])], {0: 2, 1: 1}),
        # round 1: draft 0 finishes (logged); draft 1's segment is lost
        ([([7], True, [-0.3]), ([9], False, [-0.5])], {0: 1}),
    ]
    cur = {}

    def collect(records=None, record_refs=None, request_ids=None):
        res, cur["hid"] = rounds.pop(0)
        return res

    def pop(keys, lengths, as_refs=False):
        assert as_refs
        return {
            keys[j]: put_hidden(torch.ones(n, D, dtype=torch.bfloat16), store.put)
            for j, n in cur["hid"].items()
        }

    learner = {"verify_drafts_block_engine": None}
    s = _spec(
        learner,
        {
            "install_draft_hidden_log": lambda **_: True,
            "start_drafts": lambda prompts, budgets, interval, hidden_keys: None,
            "draft_request_ids": lambda: {},
            "collect_drafts": collect,
            "pop_draft_hidden": pop,
        },
    )
    s._free_refs = store.free
    prompts = [torch.tensor([1, 1]), torch.tensor([2, 2])]
    asyncio.run(s.draft_cohorts(0, {2: prompts}))()
    assert s.stats["spec/hidden_bytes_pending"] == 3 * D * 2
    first = [d["hidden"][0] for d in s.cohorts[2]]
    asyncio.run(s.draft_cohorts(1, {}))()
    d0, d1 = s.cohorts[2]
    t0 = cohort_draft(d0)
    assert isinstance(t0[7], HiddenRefs) and t0[7].rows == 3
    assert [sg.rows for sg in t0[7].segs] == [2, 1]
    # Draft 1 lost its second segment: unverifiable, its first one is freed.
    assert d1["hidden"] is None and first[1].segs[0].ref is None
    assert len(store.objs) == 2  # draft 0's two segments; draft 1's one freed
    assert s.stats["spec/hidden_bytes_pending"] == 3 * D * 2
    _assert_no_tensors([d["hidden"] for c in s.cohorts.values() for d in c])

    def verify(rows, keys, hidden, **_):
        assert len(hidden) == 1 and hidden[0].rows == 3
        return [
            (
                0,
                {
                    "accepted": 3,
                    "next": 0,
                    "logprobs": [-0.1] * 3,
                    "draft_logq": [-0.1, -0.2, -0.3],
                    "hidden_bytes_fetched": 0,
                },
            )
        ]

    s.learner_policy.worker_group.handlers["verify_drafts_block_engine"] = verify
    assert s.verify_cohort(2)
    assert s.stats["spec/engine_q_dropped"] == 1.0
    assert s.stats["spec/hidden_bytes_fetched"] == 0
    assert all(sg.ref is None for sg in t0[7].segs)  # shared with the cohort dict
    assert first[0].segs[0].ref is None
    assert store.objs == {}  # nothing outlives the step that verified it


def test_driver_transport_is_selectable(_no_ray):
    s = _spec({}, {"install_draft_hidden_log": lambda **_: True}, hidden_transport="driver")
    assert s.hidden_transport == "driver"
    with pytest.raises(ValueError, match="hidden_transport"):
        _spec({}, {"install_draft_hidden_log": lambda **_: True}, hidden_transport="x")


def test_verify_row_batches_respect_the_padded_token_budget():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    w = object.__new__(ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class)
    w.cfg = {"make_sequence_length_divisible_by": 128}
    rows = [torch.zeros(n, dtype=torch.long) for n in (100, 200, 250, 1000, 3000, 50)]
    # Padded lengths 128, 256, 256, 1024, 3072, 128; budget 1024 tokens.
    assert w._row_batches(rows, 1024) == [[0, 1, 2], [3], [4], [5]]
    # No budget: one row per forward (the previous behaviour).
    assert w._row_batches(rows, 0) == [[0], [1], [2], [3], [4], [5]]
    # Every row appears once, in order.
    flat = [j for b in w._row_batches(rows, 4096) for j in b]
    assert flat == list(range(len(rows)))
