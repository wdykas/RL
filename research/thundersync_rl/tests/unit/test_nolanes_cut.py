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
"""wake_verify cut stop (VerifyStage.cut_drafts), counted: with engine slots it stops once no draft is on an engine and
every granted draft call's engine request is reported finished (results still in transit back drain during training),
whatever rollout groups are still post-processing; it waits for a granted call the engine has not finished; a draft
that reaches an engine after a round is cut by another round."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import thundersync_rl.nolanes_stage as ns
from thundersync_rl.nolanes_stage import VerifyStage


class _Remote:
    def __init__(self, fn):
        self.remote = fn


class _Worker:
    def __init__(self, wg):
        self.nl_done_call_ids = _Remote(lambda since: (wg.done[since:], len(wg.done)))


class _WG:
    def __init__(self, draft_counts, on_poll=None):
        self.draft_counts = list(draft_counts)  # successive nl_draft_requests answers [total, unmarked]
        self.cuts = 0
        self.polls = 0
        self.on_poll = on_poll
        self.done = []  # engine-finished call ids (rank 0's append-only report)
        self.workers = [_Worker(self)]

    def run_all_workers_single_data(self, name, **kw):
        if name == "nl_cut_drafts":
            self.cuts += 1
            return [{"cut": 3, "waiting": 0}]
        if name == "nl_draft_requests":
            self.polls += 1
            if self.on_poll:
                self.on_poll(self.polls)
            return [self.draft_counts.pop(0) if len(self.draft_counts) > 1 else self.draft_counts[0]]
        raise AssertionError(name)


def _stage(wg, inflight, granted=None):
    from thundersync_rl.engine_slots import PrioritySlots

    st = VerifyStage.__new__(VerifyStage)
    st.sc = SimpleNamespace(_inflight_by_group_id=inflight)
    st.slots = None
    if granted is not None:
        st.slots = PrioritySlots(8)
        for _ in range(granted):
            st.slots.new_draft_call()
    st._gen_wg = lambda: wg
    return st


def test_stops_once_the_engine_finished_every_granted_call_with_results_still_in_transit(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    wg = _WG([[0, 0]])
    wg.done = [1, 2, 3]  # three granted calls: their engine requests finished, results not yet back on the driver
    st = _stage(wg, {"g1": 1, "g2": 2}, granted=3)
    out = asyncio.run(st.cut_drafts())
    assert out["cut_rounds"] == 1 and out["cut_inflight_groups"] == 2
    assert len(st.slots.draft_calls) == 3  # still unreturned: they drain during training


def test_waits_for_a_granted_call_the_engine_has_not_finished(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    wg = _WG([[0, 0]])
    wg.done = [1]

    def on_poll(n):  # call 2 (in transit to the engine) finishes on the engine after a few polls
        if n == 5:
            wg.done.append(2)

    wg.on_poll = on_poll
    st = _stage(wg, {"g": 1}, granted=2)
    asyncio.run(st.cut_drafts())
    assert wg.polls >= 5


def test_a_returned_call_needs_no_engine_report(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    wg = _WG([[0, 0]])
    st = _stage(wg, {}, granted=2)
    st.slots.end_draft_call(1)  # returned (or errored / cancelled) before any report
    st.slots.end_draft_call(2)
    out = asyncio.run(st.cut_drafts())
    assert out["cut_rounds"] == 1


def test_a_draft_arriving_on_an_engine_after_a_round_is_cut_by_another_round(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    wg = _WG([[0, 0], [1, 1], [0, 0]])  # a granted call lands unmarked: another round

    def on_poll(n):
        if n == 3:
            wg.done.append(1)

    wg.on_poll = on_poll
    out = asyncio.run(_stage(wg, {}, granted=1).cut_drafts())
    assert wg.cuts == 2 and out["cut_rounds"] == 2


def test_without_slots_the_stop_waits_for_the_rollout_groups(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    inflight = {"g": 1}

    def on_poll(n):
        if n == 4:
            inflight.clear()

    wg = _WG([[0, 0]], on_poll)
    out = asyncio.run(_stage(wg, inflight).cut_drafts())
    assert wg.polls >= 4 and out["cut_inflight_groups"] == 0


def test_engine_reports_done_call_ids_append_only_by_cursor():
    from thundersync_rl.nolanes_engine import NolanesEngineMixin

    w = SimpleNamespace(_nl={"done_call_ids": [4, 5]})
    ids, n = NolanesEngineMixin.nl_done_call_ids(w, 0)
    assert ids == [4, 5] and n == 2
    w._nl["done_call_ids"].append(9)
    assert NolanesEngineMixin.nl_done_call_ids(w, n) == ([9], 3)
