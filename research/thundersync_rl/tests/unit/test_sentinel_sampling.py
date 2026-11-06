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
"""THUNDERSYNC_SENTINEL_EVERY: the weight sentinel checks the first refit of each engine and then every k-th one."""

from thundersync_rl.sentinel import WeightSentinel


def _bare():
    s = WeightSentinel.__new__(WeightSentinel)
    s.stats, s._sampled = {}, {}
    return s


def test_every_refit_by_default(monkeypatch):
    monkeypatch.delenv("THUNDERSYNC_SENTINEL_EVERY", raising=False)
    s = _bare()
    assert not any(s._sample_skip("lane0") for _ in range(6))


def test_every_fourth_refit_per_engine(monkeypatch):
    monkeypatch.setenv("THUNDERSYNC_SENTINEL_EVERY", "4")
    s = _bare()
    checked = [not s._sample_skip(k) for _ in range(8) for k in ("lane0", "lane1")]
    lane0 = checked[0::2]
    assert lane0 == [True, False, False, False, True, False, False, False]
    assert checked[1::2] == lane0
    assert s.stats["sentinel/skipped"] == 12
