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
"""CPU checks of learner_graphs' token buckets and output cutting (GPU bits: research/lgraphs/verify_probe.py)."""

import torch

from nemo_rl.models.megatron import learner_graphs as lg


def test_bucket_length_bounds():
    prev = 0
    for s in range(128, 40960 + 1, 128):
        b = lg.bucket_length(s, div=8)
        assert b >= s and b % 128 == 0
        assert b - s < max(128, s / 8 + 1)  # at most one grid step (<= 12.5% above 1024)
        assert b >= prev  # monotone: a longer row never gets a smaller bucket
        prev = b
        assert lg.bucket_length(b, div=8) == b  # buckets are fixed points


def test_bucket_length_grid_size():
    # ~8 buckets per octave: few distinct shapes between 1k and 10k tokens.
    buckets = {lg.bucket_length(s, div=8) for s in range(1024, 10240 + 1, 128)}
    assert len(buckets) <= 28


def test_bucket_length_disabled():
    for s in (128, 1000, 8320):
        assert lg.bucket_length(s, div=0) == s


def test_cut_plain_output():
    w = lg._Wrapped(torch.nn.Identity(), {})
    out = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)  # [b, s_pad, v]
    cut = lg._cut(out, w, b=2, s=4, s_pad=6, copy=True)
    assert cut.shape == (2, 4, 3)
    assert torch.equal(cut, out[:, :4])
    cut[0, 0, 0] = -1.0
    assert out[0, 0, 0] == 0.0  # copy: the graph's static output is not aliased


class _Group:
    def __init__(self, last):
        self.is_last_microbatch = last


class _DDP(torch.nn.Module):
    def __init__(self, last, fused=False):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))
        if fused:
            self.w.grad_added_to_main_grad = True
        self.bucket_groups = [_Group(last)]
        self.expert_parallel_bucket_groups = [_Group(last)]


def test_grad_sync_off_only_under_no_sync():
    # In-graph main_grad accumulation is what DDP's hook does only under no_sync.
    assert lg._grad_sync_off(_DDP(last=False))
    assert not lg._grad_sync_off(_DDP(last=True))
    assert not lg._grad_sync_off(_DDP(last=False, fused=True))  # TE fused wgrad accumulation
    assert not lg._grad_sync_off(torch.nn.Linear(2, 2))  # not Megatron DDP


def test_wrapped_remap_params_deferred_head():
    w = lg._Wrapped(torch.nn.Identity(), {})
    a, b = torch.zeros(1), torch.ones(1)
    w.deferred = (a, True, None)
    w.remap_params(lambda t: b if t is a else t)
    assert w.deferred[0] is b


class _Buf:
    def __init__(self, n):
        self.param_data = torch.zeros(n)
        self.grad_data = torch.zeros(n)


def test_buffer_ptrs_follow_reallocation():
    # move_model's offload / reload resizes the DDP buffers' storages (new addresses):
    # graphs keyed on the old addresses must not be replayed.
    m = torch.nn.Module()
    m.buffers = [_Buf(4)]
    m.expert_parallel_buffers = [_Buf(2)]
    before = lg._buffer_ptrs(m)
    assert len(before) == 4
    assert lg._buffer_ptrs(m) == before
    m.buffers[0].grad_data = torch.zeros(4)  # reallocated
    assert lg._buffer_ptrs(m) != before
    assert lg._buffer_ptrs(torch.nn.Linear(2, 2)) == ()


class _Inst:
    def __init__(self):
        self.replay_backward_list = []


class _FakeEntry:
    def __init__(self, *a):
        self.graph = lambda: "replayed"
        self.exchanges = True
        self.wrapped = None
        self.stage_in = None  # PP 1
        self.busy = False
        self.slot = 0

    def load(self, *a):
        pass


def test_train_replay_overflow_keeps_backward_anchors(monkeypatch):
    # lg4-x xovf2: a replayed training graph whose exchange overflowed reruns eagerly on
    # Megatron's dispatcher; that rerun's backward recompute pops the worker's anchors
    # (REPLAY_BACKWARD), so the replay path must not have popped them already.
    from nemo_rl.models.megatron import device_dispatch

    inst = _Inst()
    anchors = torch.zeros(4, 2, dtype=torch.long)
    inst.replay_backward_list.append(anchors)
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setattr(lg, "_replay_state", lambda m: (("sticky", 2), None))
    monkeypatch.setattr(lg, "_shared_pool", lambda: False)
    monkeypatch.setattr(lg, "_replay_instances", lambda m: [inst])
    monkeypatch.setattr(lg, "_train_graphs", lambda: True)
    monkeypatch.setattr(lg, "_grad_sync_off", lambda m: True)
    monkeypatch.setattr(lg, "_train_max_tokens", lambda: 1 << 20)
    monkeypatch.setattr(lg, "_ep_size", lambda: 2)
    monkeypatch.setattr(lg, "_snapshot", lambda m: None)
    monkeypatch.setattr(lg, "_FwdEntry", _FakeEntry)
    monkeypatch.setattr(lg, "_cut", lambda out, *a, **k: out)
    monkeypatch.setattr(device_dispatch, "enabled", lambda: True)
    seen = {}

    def rerun(self, snap, ids, pos, mask, kw):
        seen["list"] = list(inst.replay_backward_list)
        return "rerun"

    monkeypatch.setattr(lg.LearnerGraphs, "_megatron_rerun", rerun)
    ids = torch.zeros(2, 2, dtype=torch.long)
    common = ("c",)
    key = (common, True, 2, 2, ("sticky", 2), 0)  # in-flight slot 0
    g.train[key] = _FakeEntry()
    g.train[key].graph = lambda: "replayed"
    monkeypatch.setattr(device_dispatch, "overflowed", lambda force=False: True)
    assert g._train(common, ids, ids, None, {}) == "rerun"
    assert not g.train[key].busy  # a dropped replay frees its slot
    assert len(seen["list"]) == 1 and seen["list"][0] is anchors
    # No overflow: the replayed backward reads the graph's buffer; the anchors are dropped.
    monkeypatch.setattr(device_dispatch, "overflowed", lambda force=False: False)
    assert g._train(common, ids, ids, None, {}) == "replayed"
    assert inst.replay_backward_list == []


class _GraphEntry:
    """A captured no-grad entry stand-in (graph, output, wrapped set)."""

    def __init__(self, captured=True):
        self.graph = object() if captured else None
        self.out = torch.zeros(2) if captured else None
        self.wrapped = object() if captured else None


def test_release_fwd_drops_graphs_keeps_seen():
    # v11oom: a pool kept across training adds to the training peak; release drops every
    # no-grad graph and its output, keeps `seen` (next sighting captures, no eager rerun).
    g = lg.LearnerGraphs(torch.nn.Identity())
    g.pool = "pool-handle"
    g.fwd["a"], g.fwd["b"] = _GraphEntry(), _GraphEntry(captured=False)
    g.seen["a"] = g.seen["b"] = 1
    assert g.release_fwd() == 1
    assert all(e.graph is None and e.out is None and e.wrapped is None for e in g.fwd.values())
    assert g.pool is None and g.seen["a"] == 1 and list(g.fwd) == ["a", "b"]
    assert g.stats["fwd_released"] == 1
    # Nothing captured: no-op (the pool handle is kept).
    g.pool = "p2"
    assert g.release_fwd() == 0 and g.pool == "p2"


def test_grad_call_releases_fwd_graphs(monkeypatch):
    # Separate-pool mode (NEMO_RL_LEARNER_GRAPHS_SHARED_POOL=0, the v11oom layout).
    monkeypatch.setattr(lg, "_shared_pool", lambda: False)
    calls = []
    monkeypatch.setattr(lg, "release_fwd_graphs", lambda: calls.append(1) or 0)
    monkeypatch.setattr(lg, "enabled", lambda: False)
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setattr(lg.LearnerGraphs, "_eager", lambda self, *a: "eager")
    monkeypatch.setitem(lg._CACHES, -1, g)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ids = torch.zeros(1, 2, dtype=torch.long)
    with torch.no_grad():
        assert g(ids, ids, None) == "eager"
    assert calls == []  # a no-grad (verification) call keeps the graphs
    with torch.enable_grad():
        assert g(ids, ids, None) == "eager"
    assert calls == [1]  # a training call releases them first


def test_release_fwd_graphs_empty_cache_only_when_dropped(monkeypatch):
    emptied = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(1))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setitem(lg._CACHES, -2, g)
    g.fwd["x"] = _GraphEntry(captured=False)
    assert lg.release_fwd_graphs() == 0 and emptied == []
    g.fwd["y"] = _GraphEntry()
    assert lg.release_fwd_graphs() == 1 and emptied == [1]
    assert lg.release_fwd_graphs() == 0 and emptied == [1]


def test_grad_call_keeps_graphs_with_shared_pool(monkeypatch):
    # Shared pool: verification and training graphs live in one pool; no release churn.
    calls = []
    monkeypatch.setattr(lg, "_shared_pool", lambda: True)
    monkeypatch.setattr(lg, "release_fwd_graphs", lambda: calls.append(1) or 0)
    monkeypatch.setattr(lg, "enabled", lambda: False)
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setattr(lg.LearnerGraphs, "_eager", lambda self, *a: "eager")
    ids = torch.zeros(1, 2, dtype=torch.long)
    with torch.enable_grad():
        assert g(ids, ids, None) == "eager"
    assert calls == []


def test_why_counters_and_features(monkeypatch):
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setattr(lg, "enabled", lambda: True)
    monkeypatch.setattr(lg.LearnerGraphs, "_common_key", lambda self, *a: (None, "pp"))
    monkeypatch.setattr(lg.LearnerGraphs, "_eager", lambda self, *a: "eager")
    ids = torch.zeros(1, 2, dtype=torch.long)
    with lg.feature("verify"), torch.no_grad():
        assert g(ids, ids, None) == "eager"
    with lg.feature("train"), torch.enable_grad():
        assert g(ids, ids, None) == "eager"
    assert g.stats["eager_ineligible"] == 1 and g.stats["why_fwd_pp"] == 1
    assert g.stats["eager_train"] == 1 and g.stats["why_train_pp"] == 1
    assert g.stats["feat_verify_eager"] == 1 and g.stats["feat_train_eager"] == 1
    assert g.calls == 2 and lg._FEATURE[0] == "other"


def test_open_window_closes_on_backward_or_drop():
    g = lg.LearnerGraphs(torch.nn.Identity())
    assert not g.is_open()
    out = torch.zeros(3)
    box = g._mark_open(out)
    assert g.is_open()
    box[0] = True  # backward replayed
    assert not g.is_open()
    out2 = torch.zeros(3)
    g._mark_open(out2)
    assert g.is_open()
    del out2  # output dropped without a backward (forward-only grad call)
    assert not g.is_open()








def test_train_bucket_default_padding_bound():
    for s in range(1024, 20480 + 1, 64):
        b = lg.bucket_length(s, div=16)
        assert b >= s and b - s <= max(128, s // 16)




def test_release_keeps_in_flight_training_slots(monkeypatch):
    # ts-final semantic check (v11oom pool release x ts-scaleup 1F1B slots): the release at a
    # grad-enabled call drops only no-grad graphs; a PP > 1 training slot whose backward has not
    # run yet (busy) keeps its graph, output, static stage input, own pool and busy flag, and
    # the slot-0 training pool stays.
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 0)
    g = lg.LearnerGraphs(torch.nn.Identity())
    monkeypatch.setitem(lg._CACHES, -3, g)
    g.pool, g.train_pool = "fwd-pool", "train-pool"
    fwd = _GraphEntry()
    fwd.stage_in = torch.zeros(3)  # PP > 1 no-grad static stage input
    g.fwd["v"] = fwd
    slots = []
    for slot, busy in ((0, False), (1, True), (2, True)):
        e = _FakeEntry()
        e.slot, e.busy, e.stage_in = slot, busy, torch.full((3,), float(slot))
        e.out, e.pool = torch.ones(2) * slot, f"slot-pool-{slot}"
        g.train[(("c",), True, 2, 2, ("sticky", 2), slot)] = e
        slots.append((e, e.graph, e.out, e.stage_in))
    assert lg.release_fwd_graphs() == 1
    assert fwd.graph is None and fwd.out is None and g.pool is None
    assert fwd.stage_in is not None  # static inputs stay for the next capture
    assert g.train_pool == "train-pool" and len(g.train) == 3
    for (e, graph, out, stage_in), busy in zip(slots, (False, True, True)):
        assert e.graph is graph and e.out is out and e.stage_in is stage_in
        assert e.busy is busy and e.pool == f"slot-pool-{e.slot}"


def test_no_grad_replay_output_is_a_copy_of_the_pool():
    # A replayed no-grad output handed to the caller (e.g. sent to the next PP stage) must not
    # alias the graph's static output: the release frees that pool at the next training call.
    w = lg._Wrapped(torch.nn.Identity(), {})
    out = torch.ones(2, 4, 3)
    cut = lg._cut(out, w, b=2, s=4, s_pad=4, copy=True)
    assert cut.data_ptr() != out.data_ptr() and torch.equal(cut, out)


def test_shared_pool_release_skips_busy_pp_slots(monkeypatch):
    # ts-final-lg (lgfull core + scaleup PP slots): the shared-pool budget release (release_all)
    # must not drop graphs while a PP > 1 training slot is in flight (its private pool and graph
    # are needed by its backward replay); once no slot is busy it releases as before.
    monkeypatch.setattr(lg, "_shared_pool", lambda: True)
    g = lg.LearnerGraphs(torch.nn.Identity())
    busy = _FakeEntry()
    busy.busy, busy.slot = True, 1
    g.train[("k", 1)] = busy
    g.fwd["v"] = _GraphEntry()
    g.mempool = object()
    assert g.slots_busy()
    assert g.release_all("eager_train") == 0
    assert busy.graph is not None and g.fwd["v"].graph is not None and g.mempool is not None
    g._budget_release(True)  # also a no-op while busy
    assert busy.graph is not None
    busy.busy = False
    assert g.release_all("eager_train") >= 1
    assert g.mempool is None and g.fwd["v"].graph is None


def test_head_out_cap_is_regime_aware(monkeypatch):
    # Default graph output cap for a head-input output: max(1 GiB, T_cap x hidden x 4 B).
    monkeypatch.delenv("NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *a: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(lg, "_row_token_cap", lambda: 20480)  # Nano 8k
    assert lg._head_out_bytes(2688) == 2**30
    monkeypatch.setattr(lg, "_row_token_cap", lambda: 2048 + 131072)  # T_cap of a 128k-row run (MBS 1)
    assert lg._head_out_bytes(2688) == (2048 + 131072) * 2688 * 4 > 2**30  # 1.43 GB: graphed
    monkeypatch.setenv("NEMO_RL_LEARNER_GRAPHS_MAX_OUT_MB", "512")
    assert lg._head_out_bytes(2688) == 512 * 2**20  # an explicit setting wins
