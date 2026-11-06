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
"""THUNDERSYNC_SENTINEL_ASYNC: the learner digest runs off the training path and is compared later, with the same
digests and verdicts as the synchronous check."""

import asyncio
import sys
import threading
import types

import pytest
import torch

import thundersync_rl.sentinel as S
from thundersync_rl import sentinel_async as A


class _W:  # a worker: only its __dict__ is used
    pass


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_digest_chunk_does_not_change_digests():
    g = torch.Generator().manual_seed(0)
    named = [("a", torch.randn(5000, generator=g)), ("b", torch.randn(70000, generator=g).to(torch.bfloat16)),
             ("c", torch.randn(3, generator=g))]
    assert S.digest_named(named, chunk=1000) == S.digest_named(named) == S.digest_named(named, chunk=1 << 22)


def test_begin_result_join_cpu():
    w = _W()
    gate = threading.Event()

    def fn():
        gate.wait(5)
        return "digest"

    A.begin(w, 1, fn)
    with pytest.raises(RuntimeError):
        A.begin(w, 1, fn)  # one digest per token
    gate.set()
    assert A.join_before_write(w, "optimizer step") >= 0.0
    out, secs, waited = _run(A.result(w, 1))
    assert out == "digest" and secs >= 0.0 and waited >= 0.0
    with pytest.raises(RuntimeError):
        _run(A.result(w, 1))  # collected once


def test_digest_exception_reaches_the_driver_not_the_writer():
    w = _W()

    def bad():
        raise S.WeightSentinelError("expert_bias changed")

    A.begin(w, 7, bad)
    A.join_before_write(w, "optimizer step")  # the parameter write is not blocked by the failure
    with pytest.raises(S.WeightSentinelError):
        _run(A.result(w, 7))


def test_join_waits_for_every_digest_in_flight():
    w = _W()
    order = []
    gate = threading.Event()

    def slow():
        gate.wait(5)
        order.append("digest")
        return 0

    A.begin(w, 1, slow)
    A.begin(w, 2, lambda: order.append("digest2") or 0)
    t = threading.Timer(0.2, gate.set)
    t.start()
    A.join_before_write(w, "optimizer step")
    order.append("write")
    assert order == ["digest", "digest2", "write"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_side_stream_digest_reads_the_weights_at_begin():
    """Ordered after the work queued at begin (a pending write is seen), and a write after the join is not."""
    w = _W()
    p = torch.zeros(1 << 22, device="cuda")
    expect = S.digest_named([("p", torch.full_like(p, 3.0))])
    big = torch.randn(4096, 4096, device="cuda")
    for _ in range(200):
        big = torch.tanh(big @ big)  # keep the current stream busy
    p.fill_(3.0)  # queued behind the matmuls when begin records its event
    A.begin(w, 1, lambda: S.digest_named([("p", p)], chunk=A.ASYNC_CHUNK))
    A.join_before_write(w, "optimizer step")
    p.fill_(5.0)  # the write the join protects against
    out, _, _ = _run(A.result(w, 1))
    assert out == expect
    assert out != S.digest_named([("p", p)])


# --- driver side ---------------------------------------------------------------------------------------------


class _Group:
    def __init__(self, workers):
        self.workers = workers

    def run_all_workers_single_data(self, method, **kw):
        out = []
        for wk in self.workers:
            r = getattr(wk, method)(**kw)
            if asyncio.iscoroutine(r):
                r = _run(r)
            out.append(r)
        return out


class _Learner:
    def __init__(self, tensors):
        self.tensors = tensors

    def _digest(self, dtypes):
        return 0, S.digest_named(list(self.tensors.items()), dtypes), {}

    def sentinel_weight_digest(self, dtypes=None, learner=False):
        return self._digest(dtypes)

    def sentinel_digest_begin(self, token, dtypes=None):
        A.begin(self, token, lambda: self._digest(dtypes))
        return True

    async def sentinel_digest_result(self, token):
        return await A.result(self, token)


class _Engine:
    def __init__(self, tensors):
        self.tensors = tensors

    def sentinel_weight_digest(self, **kw):
        return 0, S.digest_named(list(self.tensors.items())), {}

    def sentinel_digest_begin(self, token, dtypes=None, learner=True):
        assert not learner
        snap = dict(self.tensors)  # the weights at begin (a refit after it joins first)
        A.begin(self, token, lambda: (0, S.digest_named(list(snap.items())), {}))
        return True

    async def sentinel_digest_result(self, token):
        return await A.result(self, token)


def _setup(monkeypatch, asynchronous=True):
    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(get=lambda refs: refs))
    monkeypatch.setenv("THUNDERSYNC_SENTINEL_ASYNC", "1" if asynchronous else "0")
    monkeypatch.delenv("THUNDERSYNC_SENTINEL_EVERY", raising=False)
    w = torch.arange(1000, dtype=torch.float32)
    lrn = _Learner({"layer.weight": w.clone()})
    eng = _Engine({"layer.weight": w.clone()})
    policy = types.SimpleNamespace(worker_group=_Group([lrn]))
    gen = types.SimpleNamespace(_policy=types.SimpleNamespace(worker_group=_Group([eng])))
    return S.WeightSentinel(policy), gen, lrn, eng


def test_async_check_compares_at_pop_stats(monkeypatch):
    s, gen, lrn, eng = _setup(monkeypatch)
    s.check(gen, "initial")  # first check of an engine: synchronous (learns the engine dtypes)
    assert s.stats["sentinel/checks"] == 1
    s.check(gen, "theta_1")
    assert s.stats["sentinel/async_begun"] == 1 and s.stats["sentinel/checks"] == 1  # not compared yet
    st = s.pop_stats()
    assert st["sentinel/checks"] == 2 and st["sentinel/tensors"] == 1 and not s._pending


def test_async_mismatch_raises_at_resolution(monkeypatch):
    s, gen, lrn, eng = _setup(monkeypatch)
    s.check(gen, "initial")
    eng.tensors["layer.weight"] = eng.tensors["layer.weight"] + 1  # engine got other weights
    s.check(gen, "theta_1")  # no verdict yet
    with pytest.raises(S.WeightSentinelError, match="theta_1"):
        s.pop_stats()


def test_next_check_of_the_engine_resolves_the_previous(monkeypatch):
    s, gen, lrn, eng = _setup(monkeypatch)
    s.check(gen, "initial")
    eng.tensors["layer.weight"] = eng.tensors["layer.weight"] * 2
    s.check(gen, "theta_1")
    with pytest.raises(S.WeightSentinelError, match="theta_1"):
        s.check(gen, "theta_2")


def test_async_and_sync_give_the_same_stats(monkeypatch):
    out = {}
    for mode in (False, True):
        s, gen, *_ = _setup(monkeypatch, asynchronous=mode)
        for i in range(3):
            s.check(gen, f"theta_{i}")
        st = s.pop_stats()
        out[mode] = {k: v for k, v in st.items() if k in ("sentinel/checks", "sentinel/tensors", "sentinel/elems")}
    assert out[False] == out[True] == {"sentinel/checks": 3.0, "sentinel/tensors": 1.0, "sentinel/elems": 1000.0}


def test_engine_digest_deferred_unless_needed(monkeypatch):
    s, gen, lrn, eng = _setup(monkeypatch)
    s.check(gen, "initial")
    assert s.check(gen, "theta_1") == []  # engine digest deferred too
    assert s.stats.get("sentinel/checks") == 1
    got = s.check(gen, "theta_hat_2", need_engine_digest=True)  # resolves theta_1, engine side synchronous
    assert got and got[0][1] == S.digest_named(list(eng.tensors.items()))
    st = s.pop_stats()
    assert st["sentinel/checks"] == 3 and st["sentinel/async_engine_s"] >= 0 and st["sentinel/async_learner_s"] >= 0


def test_engine_async_mismatch_raises_at_pop_stats(monkeypatch):
    s, gen, lrn, eng = _setup(monkeypatch)
    s.check(gen, "initial")
    lrn.tensors["layer.weight"] = lrn.tensors["layer.weight"] - 1
    s.check(gen, "theta_1")
    with pytest.raises(S.WeightSentinelError, match="theta_1"):
        s.pop_stats()
