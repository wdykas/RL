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
"""Lookahead drafts (X5): theta~ = theta_k + alpha (theta_k - theta_{k-1}), elementwise, the same on every path."""

import sys
import types

import pytest
import torch

from thundersync_rl import lookahead_draft as L
from thundersync_rl.lanes import CohortLanes, Lane


def _w(n, dtype=torch.bfloat16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, generator=g) * 0.02).to(dtype)


def test_alpha_env(monkeypatch):
    monkeypatch.delenv(L.ENV, raising=False)
    assert L.alpha() is None
    monkeypatch.setenv(L.ENV, "0")
    assert L.alpha() == 0.0
    monkeypatch.setenv(L.ENV, "0.75")
    assert L.alpha() == 0.75 and L.alphas() == (0.75,)
    monkeypatch.setenv(L.ENV, "0,0.75,1.5")
    sched = L.alphas()
    assert sched == (0.0, 0.75, 1.5) and [L.alpha_at(sched, v) for v in range(5)] == [0.0, 0.75, 1.5, 0.0, 0.75]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
def test_alpha_zero_is_bitwise_theta_k(dtype):
    cur, prev = _w(4097, dtype, 1), _w(4097, dtype, 2)
    out = L.extrapolate(cur, prev, 0.0)
    assert out.dtype == cur.dtype and torch.equal(out.view(torch.int16 if dtype != torch.float32 else torch.int32),
                                                  cur.view(torch.int16 if dtype != torch.float32 else torch.int32))


def test_elementwise_slices_match_the_whole_tensor():
    cur, prev = _w(64 * 96, seed=3).view(64, 96), _w(64 * 96, seed=4).view(64, 96)
    whole = L.extrapolate(cur, prev, 0.6)
    for sl in [(slice(0, 32), slice(None)), (slice(None), slice(10, 50)), (slice(5, 9), slice(1, 95, 3))]:
        assert torch.equal(L.extrapolate(cur[sl], prev[sl], 0.6), whole[sl])
    # the formula
    ref = (cur.float() + 0.6 * (cur.float() - prev.float())).to(torch.bfloat16)
    assert torch.equal(whole, ref)


def test_snapshot_timeline_and_views():
    s = L.PrevSnapshot()
    w = torch.nn.Parameter(_w(8 * 16, seed=5).view(8, 16), requires_grad=False)
    named = {"w": w}
    s.capture(named)  # lookahead_init: theta_0 pending
    assert s.prev_for(w) is None  # version 0 drafts: D = 0
    theta0 = w.detach().clone()
    s.commit()  # optimizer step 1, before the write
    with torch.no_grad():
        w.add_(0.01)
    s.capture(named)  # after step 1: theta_1 pending
    assert torch.equal(s.prev_for(w), theta0)  # version-1 drafter refits use theta_0
    # a view (one "expert" row block of a grouped weight) finds its slice of the host copy
    assert torch.equal(s.prev_for(w[2:4]), theta0[2:4])
    theta1 = w.detach().clone()
    s.commit()
    assert torch.equal(s.prev_for(w), theta1)


def test_state_roundtrip_and_digest():
    s = L.PrevSnapshot()
    w = _w(1000, seed=6)
    s.capture({"w": w})
    s.commit()
    st, d = s.state(), s.digest()
    s2 = L.PrevSnapshot()
    s2.load_state(st, {"w": w})
    assert s2.digest() == d and torch.equal(s2.prev_for(w), w)


def test_scheduler_refits_a_raw_lane_to_lookahead_weights():
    lanes = [Lane(index=0, generation=None), Lane(index=1, generation=None)]
    sched = CohortLanes(lanes)
    calls = []
    sched.refit(lanes[0], ("theta", 3), lambda: calls.append("raw"))
    sched.refit(lanes[0], ("theta", 3), lambda: calls.append("raw again"))  # same weights: skipped
    sched.refit(lanes[0], ("theta", 3), lambda: calls.append("la"), lookahead=True)  # theta~ != theta
    sched.refit(lanes[0], ("theta", 3), lambda: calls.append("la again"), lookahead=True)
    assert calls == ["raw", "la"] and lanes[0].lookahead


def test_send_transform_sends_theta_tilde():
    pytest.importorskip("megatron.core.resharding.transforms")
    s = L.PrevSnapshot()
    w = _w(6 * 10, seed=7).view(6, 10)
    s.capture({"layer.w": w})
    s.commit()
    w2 = (w.float() + 0.03).to(torch.bfloat16)  # theta_k (live)
    tf = L.make_send_transform(s, 0.5)
    assert tf.should_transform("layer.w") and not tf.should_transform("other")
    sl = (slice(2, 5), slice(None))
    (out,) = tf.prepare_send("layer.w", sl, w2)
    assert torch.equal(out, L.extrapolate(w2, w, 0.5)[sl])


def test_sentinel_passes_lookahead_only_when_set(monkeypatch):
    import thundersync_rl.sentinel as S

    seen = []

    class Lrn:
        def sentinel_weight_digest(self, dtypes=None, learner=False, **kw):
            seen.append(kw)
            return 0, S.digest_named([("w", torch.ones(4))], dtypes), {}

    class Eng:
        def sentinel_weight_digest(self, **kw):
            return 0, S.digest_named([("w", torch.ones(4))]), {}

    class G:
        def __init__(self, w):
            self.w = w

        def run_all_workers_single_data(self, method, **kw):
            return [getattr(self.w, method)(**kw)]

    monkeypatch.setitem(sys.modules, "ray", types.SimpleNamespace(get=lambda r: r))
    monkeypatch.setenv("THUNDERSYNC_SENTINEL_ASYNC", "0")
    s = S.WeightSentinel(types.SimpleNamespace(worker_group=G(Lrn())))
    gen = types.SimpleNamespace(_policy=types.SimpleNamespace(worker_group=G(Eng())))
    s.check(gen, "a")
    s.check(gen, "b", lookahead=True)
    assert seen[-1] == {"lookahead": True} and all(k == {} for k in seen[:-1])
