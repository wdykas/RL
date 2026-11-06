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
"""StreamingLearner.abort never waits unboundedly on ranks stuck in a collective."""

import ray
from thundersync_rl import streaming


class _Group:
    dp_size = 3

    def run_all_workers_single_data(self, name):
        assert name == "stream_abort_step"
        return ["ok", "failed", "stuck"]


def _learner():
    learner = object.__new__(streaming.StreamingLearner)
    learner.worker_group = _Group()
    learner.dp_size = 3
    return learner


def test_abort_is_bounded(monkeypatch, capsys):
    waits = []

    def fake_wait(refs, num_returns, timeout):
        waits.append(timeout)
        return [r for r in refs if r != "stuck"], ["stuck"]

    def fake_get(ref):
        if ref == "failed":
            raise RuntimeError("actor died")
        return None

    monkeypatch.setattr(ray, "wait", fake_wait)
    monkeypatch.setattr(ray, "get", fake_get)
    assert _learner().abort(timeout_s=5.0) is False
    assert waits == [5.0]
    out = capsys.readouterr().out
    assert "1/3 learner ranks did not abort" in out and "actor died" in out


def test_abort_all_ranks_answer(monkeypatch):
    monkeypatch.setattr(ray, "wait", lambda refs, num_returns, timeout: (refs, []))
    monkeypatch.setattr(ray, "get", lambda ref: None)
    assert _learner().abort() is True
