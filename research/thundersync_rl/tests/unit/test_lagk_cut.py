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
"""Lag-K wake_verify (K in 1..4): the stand-down cuts only the drafts of the batch that trains next (target step <=
version + 1); later batches' drafts pause on their engines / wait at the slot gate and resume after the wake."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import thundersync_rl.nolanes_stage as ns
from thundersync_rl.engine_slots import PRIO_CONT, PRIO_DRAFT, PrioritySlots, row_target
from thundersync_rl.nolanes_engine import CUT_ATTR, TARGET_ATTR, cuttable, mark_cut


def test_cuttable_rule():
    sp = SimpleNamespace(**{TARGET_ATTR: 6})
    assert cuttable(sp, None) and cuttable(sp, 6) and cuttable(sp, 7) and not cuttable(sp, 5)
    assert cuttable(SimpleNamespace(), 5)  # unstamped: the lag-1 rule (cut)


def test_row_target():
    assert row_target({"nl_target_step": [7]}) == 7
    assert row_target({"nl_target_step": [None]}) is None
    assert row_target({}) is None


def test_slots_cut_only_the_next_batch():
    async def go():
        s = PrioritySlots(2)
        assert await s.acquire(PRIO_DRAFT, target=5) and await s.acquire(PRIO_DRAFT, target=6)  # full
        w5 = asyncio.ensure_future(s.acquire(PRIO_DRAFT, target=5))
        w6 = asyncio.ensure_future(s.acquire(PRIO_DRAFT, target=6))
        wn = asyncio.ensure_future(s.acquire(PRIO_DRAFT))
        await asyncio.sleep(0)
        s.new_draft_call(5), s.new_draft_call(6)
        assert s.cut_waiting_drafts(5) == 2  # target 5 and the unstamped one end at the gate
        await asyncio.sleep(0)
        assert w5.result() is False and wn.result() is False and not w6.done()
        assert await s.acquire(PRIO_DRAFT, target=4) is False  # a cuttable arrival during the cut ends unsent
        late = asyncio.ensure_future(s.acquire(PRIO_DRAFT, target=7))  # a later batch's arrival waits
        await asyncio.sleep(0)
        assert not late.done()
        assert s.drafts_not_engine_done() == 1  # only the target-5 call counts for the stop rule
        s.release(1, PRIO_DRAFT), s.release(1, PRIO_DRAFT)  # the two granted calls return
        await asyncio.sleep(0)
        assert not w6.done() and not late.done()  # held through the stand-down
        s.hold_drafts()
        assert await s.acquire(PRIO_CONT)  # the wake: continuations first
        s.open_drafts()
        await asyncio.sleep(0)
        assert w6.result() is True and not late.done()  # capacity 2: one cont + one draft
        s.release(1, PRIO_CONT)
        await asyncio.sleep(0)
        assert late.result() is True

    asyncio.run(go())


def test_slots_lag1_unchanged():
    async def go():
        s = PrioritySlots(1)
        assert await s.acquire(PRIO_DRAFT, target=6)
        w = asyncio.ensure_future(s.acquire(PRIO_DRAFT, target=9))
        await asyncio.sleep(0)
        assert s.cut_waiting_drafts() == 1  # no bound: every waiting draft
        await asyncio.sleep(0)
        assert w.result() is False

    asyncio.run(go())


def _sp(**kw):
    sp = SimpleNamespace(**kw)
    sp.add_attributes = lambda d, sp=sp: [setattr(sp, k, v) for k, v in d.items()]
    return sp


def test_mark_cut_keeps_later_batches():
    from thundersync_rl.engine_q import HIDDEN_KEY_ATTR

    class Rec:
        def __init__(self, sp):
            self.requests = [SimpleNamespace(sampling_params=sp)]

        def __getitem__(self, i):
            return self.requests[i]

    sps = {1: _sp(**{HIDDEN_KEY_ATTR: "a", TARGET_ATTR: 5}), 2: _sp(**{HIDDEN_KEY_ATTR: "b", TARGET_ATTR: 6}),
           3: _sp(**{HIDDEN_KEY_ATTR: "c"}), 4: _sp()}
    eng = SimpleNamespace(waiting_request_ids=[], stop_word_finished_request_ids=set(),
                          requests={r: SimpleNamespace(record=Rec(sp)) for r, sp in sps.items()})
    c = mark_cut(eng, 5)
    assert c["cut"] == 2 and c["kept"] == 1  # 1 (target 5) and 3 (unstamped) cut; 2 (target 6) kept; 4 not a draft
    assert eng.stop_word_finished_request_ids == {1, 3}
    assert getattr(sps[2], CUT_ATTR, 0) == 0


class _Remote:
    def __init__(self, fn):
        self.remote = fn


def test_stage_stop_rule_ignores_later_batches(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    seen = []

    class WG:
        def __init__(self):
            self.done = [1]
            self.workers = [SimpleNamespace(nl_done_call_ids=_Remote(lambda since: (self.done[since:], len(self.done))))]

        def run_all_workers_single_data(self, name, **kw):
            seen.append((name, kw.get("max_target")))
            if name == "nl_cut_drafts":
                return [{"cut": 1, "waiting": 0, "kept": 1}]
            return [[0, 0]]

    st = ns.VerifyStage.__new__(ns.VerifyStage)
    st.sc = SimpleNamespace(_inflight_by_group_id={"g": 1})
    st.slots = PrioritySlots(4)
    st.slots.cut_waiting_drafts(5)
    st.slots.new_draft_call(5), st.slots.new_draft_call(6)  # call 2 (target 6) never finishes: a paused later draft
    wg = WG()
    st._gen_wg = lambda: wg
    out = asyncio.run(st.cut_drafts(5))
    assert out["cut_rounds"] == 1 and out["cut_kept_max"] == 1
    assert all(m == 5 for _, m in seen)
