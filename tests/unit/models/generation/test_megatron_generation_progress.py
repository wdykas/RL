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
"""GenerationProgressMonitor / with_generated_token_counter on a fake engine (CPU)."""

from collections import deque
from enum import Enum, auto
from types import SimpleNamespace

from nemo_rl.models.generation.megatron.progress import (
    GenerationProgressMonitor,
    with_generated_token_counter,
)


class _State(Enum):
    RUNNING = auto()
    PAUSED = auto()


class _Engine:
    def __init__(self):
        self.context = SimpleNamespace(
            total_request_count=0, paused_request_count=0, step_count=0
        )
        self.requests = {}
        self.waiting_request_ids = deque()
        self.finished_request_count = 0
        self.state = _State.RUNNING
        self.seen = []

    def post_process_requests(self, request_ids, finished, *args, **kwargs):
        self.seen.append(list(request_ids))
        self.finished_request_count += len(finished)
        return list(request_ids), []


class _Clock:
    t = 0.0

    def __call__(self):
        return self.t


def _engine_with_counter():
    return with_generated_token_counter(_Engine)()


def test_token_counter_counts_requests_per_step_and_calls_through():
    e = _engine_with_counter()
    e.post_process_requests([1, 2, 3], [], 0.0)
    e.post_process_requests([1, 2], [3], 0.0)
    assert e.generated_token_count == 5
    assert e.seen == [[1, 2, 3], [1, 2]]
    assert e.finished_request_count == 1
    assert type(e).__name__ == "_Engine"


def _monitor(e, clock, stalls, stall_s=600.0, log=True):
    return GenerationProgressMonitor(
        e, 0, interval_s=60, stall_timeout_s=stall_s, log=log, clock=clock,
        on_stall=stalls.append,
    )


def test_progress_line_reports_rates_and_drain(capsys):
    e, clock, stalls = _engine_with_counter(), _Clock(), []
    m = _monitor(e, clock, stalls)
    assert m.tick() is None  # baseline
    e.requests = {i: None for i in range(4)}
    e.context.total_request_count = 4
    e.post_process_requests([0, 1, 2, 3], [], 0.0)
    clock.t = 60.0
    e.generated_token_count = 6000
    line = m.tick()
    assert "active 4 (paused 0)" in line and "tokens 6.00k (+6.00k)" in line
    assert "100 tok/s" in line and "RUNNING" in line
    e.requests = {}
    e.finished_request_count = 4
    clock.t = 120.0
    assert "finished 4 (+4)" in m.tick()  # one line after draining
    clock.t = 180.0
    assert m.tick() is None  # idle: silent
    assert not stalls


def test_stall_fires_only_when_busy_and_frozen():
    e, clock, stalls = _engine_with_counter(), _Clock(), []
    m = _monitor(e, clock, stalls, log=False)
    m.tick()
    e.requests = {0: None}
    for t in range(60, 600, 60):  # progressing: never fires
        clock.t = float(t)
        e.generated_token_count += 10
        m.tick()
    assert not stalls
    last_progress = clock.t
    while clock.t < last_progress + 600:
        clock.t += 60.0
        m.tick()
    assert len(stalls) == 1 and "no request finished" in stalls[0]


def test_no_stall_when_paused_or_idle_or_disabled():
    for setup in ("paused", "idle", "disabled"):
        e, clock, stalls = _engine_with_counter(), _Clock(), []
        m = _monitor(e, clock, stalls, stall_s=0.0 if setup == "disabled" else 600.0)
        m.tick()
        e.requests = {} if setup == "idle" else {0: None}
        if setup == "paused":
            e.state = _State.PAUSED
        for _ in range(30):
            clock.t += 60.0
            m.tick()
        assert not stalls, setup


def test_engine_reset_does_not_go_negative():
    e, clock, stalls = _engine_with_counter(), _Clock(), []
    m = _monitor(e, clock, stalls)
    e.finished_request_count, e.generated_token_count = 100, 1000
    m.tick()
    e.requests = {0: None}
    e.finished_request_count, e.generated_token_count = 2, 1500
    clock.t = 60.0
    line = m.tick()
    assert "(+2)" in line and "(+500)" in line
