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
no draft call holds a slot, whatever rollout groups are still post-processing; it keeps waiting while a granted draft
call has not returned; a draft that reaches an engine after a round is cut by another round."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import thundersync_rl.nolanes_stage as ns
from thundersync_rl.nolanes_stage import VerifyStage


class _WG:
    def __init__(self, draft_counts, on_poll=None):
        self.draft_counts = list(draft_counts)  # successive nl_draft_requests answers [total, unmarked]
        self.cuts = 0
        self.polls = 0
        self.on_poll = on_poll

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


def _stage(wg, inflight, holding=None):
    st = VerifyStage.__new__(VerifyStage)
    st.sc = SimpleNamespace(_inflight_by_group_id=inflight)
    st.slots = None if holding is None else SimpleNamespace(draft_holding=holding)
    st._gen_wg = lambda: wg
    return st


def test_stops_while_groups_post_process_once_no_draft_holds_a_slot(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    out = asyncio.run(_stage(_WG([[0, 0]]), {"g1": 1, "g2": 2}, holding=0).cut_drafts())
    assert out["cut_rounds"] == 1 and out["cut_inflight_groups"] == 2


def test_waits_for_a_granted_draft_call_to_return(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    st = _stage(None, {"g": 1}, holding=1)

    def on_poll(n):  # the granted call returns after a few polls
        if n == 5:
            st.slots.draft_holding = 0

    wg = _WG([[0, 0]], on_poll)
    st._gen_wg = lambda: wg
    asyncio.run(st.cut_drafts())
    assert wg.polls >= 5


def test_a_draft_arriving_on_an_engine_after_a_round_is_cut_by_another_round(monkeypatch):
    monkeypatch.setattr(ns.ray, "get", lambda x: x)
    st = _stage(None, {}, holding=1)

    def on_poll(n):
        if n == 3:
            st.slots.draft_holding = 0

    wg = _WG([[0, 0], [1, 1], [0, 0]], on_poll)  # a granted call lands unmarked: another round
    st._gen_wg = lambda: wg
    out = asyncio.run(st.cut_drafts())
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
