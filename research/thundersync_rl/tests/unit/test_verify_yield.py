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
"""verify_yield: mailbox decisions, boundary iterator, state snapshots (CPU)."""

import asyncio

import pytest
import torch

from thundersync_rl import verify_yield
from thundersync_rl.verify_yield import Mailbox, boundary_iter


def test_decide_runs_posted_calls_in_order_per_domain():
    mb = Mailbox()
    mb.post(11, ["p11"], 2)
    mb.post(12, ["p12"], 2)
    # dispatch 10 is training; 11 and 12 are posted verifications
    assert mb.decide("A", 10, 0, 0) == 11
    assert mb.decide("A", 10, 0, 1) == 12
    assert mb.decide("A", 10, 0, 2) == -1
    # the other rank of domain A reads the same decisions
    assert [mb.decide("A", 10, 0, j) for j in range(3)] == [11, 12, -1]
    # a later boundary: nothing left
    assert mb.decide("A", 10, 1, 0) == -1
    # domain B decides on its own (no shared collective)
    assert mb.decide("B", 10, 3, 0) == 11


def test_decide_stops_at_a_call_that_is_not_posted():
    mb = Mailbox()
    mb.post(13, ["p13"], 1)  # 11 (say, the next training dispatch) is not posted
    assert mb.decide("A", 10, 0, 0) == -1
    assert mb.decide("A", 12, 0, 0) == 13


def test_decision_is_fixed_once_made():
    mb = Mailbox()
    assert mb.decide("A", 10, 0, 0) == -1
    mb.post(11, ["p11"], 1)  # posted after the first rank asked: same answer for the domain
    assert mb.decide("A", 10, 0, 0) == -1
    assert mb.decide("A", 10, 1, 0) == 11  # the next boundary takes it


def test_decisions_of_earlier_dispatches_are_dropped():
    mb = Mailbox()
    mb.post(11, ["p11"], 1)
    assert mb.decide("A", 10, 0, 0) == 11
    assert mb.decide("A", 20, 0, 0) == -1
    assert not [k for k in mb.decisions if k[1] == 10]
    assert ("A", 10) not in mb.inlined


def test_result_waits_for_every_rank_in_rank_order():
    async def run():
        mb = Mailbox()
        mb.post(11, ["p11"], 2)
        task = asyncio.ensure_future(mb.result(11))
        await asyncio.sleep(0)
        mb.put_result(11, 1, [["r1", None]])
        await asyncio.sleep(0)
        assert not task.done()
        mb.put_result(11, 0, [["r0", None]])
        out = await task
        assert out == [["r0", None], ["r1", None]]
        assert 11 not in mb.posted

    asyncio.run(run())


def test_result_before_post():
    async def run():
        mb = Mailbox()
        mb.put_result(5, 0, [["r0", None]])  # a result recorded before its post
        mb.post(5, ["p5"], 1)
        assert await mb.result(5) == [["r0", None]]

    asyncio.run(run())


def test_boundary_iter_skips_the_dispatch_first_item_only():
    calls = []
    first = [True]
    a = list(boundary_iter(iter([1, 2, 3]), lambda: calls.append("h"), first))
    b = list(boundary_iter(iter([4, 5]), lambda: calls.append("h"), first))
    assert a == [1, 2, 3] and b == [4, 5]
    assert calls == ["h"] * 4  # before 2, 3 (first call) and 4, 5 (second call)


def test_inline_no_grad_forward_leaves_accumulated_gradient_bitwise():
    """A boundary's verification (no-grad forward, mode flip) does not change training."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Dropout(0.1), torch.nn.Linear(32, 4))
    xs = [torch.randn(8, 16) for _ in range(4)]
    v = torch.randn(5, 16)

    def train(inline):
        torch.manual_seed(1)
        m = model
        for p in m.parameters():
            p.grad = None
        m.train()
        for i, x in enumerate(xs):
            if inline and i > 0:
                rng = (torch.get_rng_state(),)
                modes = [(mm, mm.training) for mm in m.modules()]
                m.eval()
                with torch.no_grad():
                    torch.rand(3)  # the verification may consume RNG
                    m(v)
                for mm, t in modes:
                    mm.training = t
                torch.set_rng_state(rng[0])
            m(x).square().sum().backward()
        return [p.grad.clone() for p in m.parameters()]

    plain, inlined = train(False), train(True)
    assert all(torch.equal(a, b) for a, b in zip(plain, inlined))


def test_replay_snapshot_restores_router_replay_fields():
    rr = pytest.importorskip("megatron.core.transformer.moe.router_replay")
    inst = rr.RouterReplay()
    try:
        inst.set_target_indices(torch.tensor([[1, 2]]))
        inst.replay_backward_list = [torch.tensor([3])]
        inst.set_router_replay_action(rr.RouterReplayAction.REPLAY_FORWARD)
        snap = verify_yield.ReplaySnapshot()
        inst.clear_indices()
        inst.clear_router_replay_action()
        snap.restore()
        assert torch.equal(inst.target_topk_idx, torch.tensor([[1, 2]]))
        assert len(inst.replay_backward_list) == 1
        assert inst.router_replay_action == rr.RouterReplayAction.REPLAY_FORWARD
    finally:
        rr.RouterReplay.global_router_replay_instances.remove(inst)


def test_ray_posted_verification_runs_at_the_next_training_boundary():
    """Two ranks of one domain: a verification posted mid-dispatch runs at the next
    micro-batch boundary on both (not after the dispatch), in order, exactly once;
    one posted after the dispatch runs as its ordered call."""
    import time

    ray = pytest.importorskip("ray")
    from thundersync_rl.verify_yield import VerifyMailbox, YieldWorkerMixin

    @ray.remote
    class FakeLearner(YieldWorkerMixin):
        def __init__(self, rank):
            self.rank = rank
            self.model = torch.nn.Linear(2, 2)
            self._next, self._cv, self.log = 0, None, []
            # A learner process has Megatron imported (ReplaySnapshot imports it).
            verify_yield.ReplaySnapshot()

        def _yield_domain_key(self):
            return ("ep", 0)

        async def ordered_call(self, seq, method, **kw):
            if self._cv is None:
                self._cv = asyncio.Condition()
            async with self._cv:
                while self._next != seq:
                    await self._cv.wait()
            y0 = self._yield_begin_call(seq)
            try:
                return getattr(self, method)(**kw)
            finally:
                self._yield_end_call(y0)
                self._next += 1
                async with self._cv:
                    self._cv.notify_all()

        def train(self, n_mb, t_mb):
            for x in boundary_iter(iter(range(n_mb)), self._yield_point, self._yield_first):
                time.sleep(t_mb)
                self.log.append(("mb", x))
            return "trained"

        def verify(self, x):
            self.log.append(("v", x))
            return x * 10

        def get_log(self):
            return self.log

    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=4, include_dashboard=False, log_to_driver=False)
    try:
        mb = VerifyMailbox.remote()
        ranks = [FakeLearner.remote(r) for r in range(2)]
        assert ray.get([w.set_verify_mailbox.remote(mb) for w in ranks]) == [True, True]
        t0 = time.perf_counter()
        train = [w.ordered_call.remote(0, "train", n_mb=6, t_mb=0.4) for w in ranks]
        time.sleep(0.2)
        payload = ray.put({"method": "verify", "kwargs": {"x": 1}})
        ray.get(mb.post.remote(1, [payload], 2))
        calls = [w.ordered_call.remote(1, "posted_call", key=1) for w in ranks]
        boxes = ray.get(mb.result.remote(1))
        latency = time.perf_counter() - t0
        assert [ray.get(b[0]) for b in boxes] == [10, 10] and all(b[1] is None for b in boxes)
        assert latency < 1.5  # boundary after micro-batch 0 (~0.4 s), not after 6 x 0.4 s
        assert ray.get(train) == ["trained", "trained"]
        ray.get(calls)
        for w in ranks:
            log = ray.get(w.get_log.remote())
            assert log.count(("v", 1)) == 1  # ran once, inline
            assert log.index(("v", 1)) < log.index(("mb", 5))
        # After the dispatch: the posted call runs in order.
        payload = ray.put({"method": "verify", "kwargs": {"x": 2}})
        ray.get(mb.post.remote(2, [payload], 2))
        ray.get([w.ordered_call.remote(2, "posted_call", key=2) for w in ranks])
        boxes = ray.get(mb.result.remote(2))
        assert [ray.get(b[0]) for b in boxes] == [20, 20]
        stats = ray.get([w.yield_stats.remote() for w in ranks])
        assert all(s["inlined"] == 1 and s["boundaries"] == 5 for s in stats)
    finally:
        if started:
            ray.shutdown()
