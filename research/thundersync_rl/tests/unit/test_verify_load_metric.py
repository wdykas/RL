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
"""spec/verify_load_max_over_mean covers the step's calls, not only the last one."""

from types import SimpleNamespace

from thundersync_rl.speculative import SpeculativeGeneration


def _count(spec, lengths, owners, ranks):
    SpeculativeGeneration._count_balance(spec, lengths, owners, ranks)


def test_balance_accumulates_over_calls():
    spec = SimpleNamespace(stats={})
    # verify0: 8 equal rows on 8 ranks -> balanced.
    _count(spec, [100] * 8, list(range(8)), 8)
    assert spec.stats["spec/verify_load_first_max_over_mean"] == 1.0
    assert spec.stats["spec/verify_load_max_over_mean"] == 1.0
    # One small piece on one rank: the old metric reported 8.0 for the step.
    _count(spec, [10], [3], 8)
    s = spec.stats
    assert s["spec/verify_load_first_max_over_mean"] == 1.0
    assert s["spec/verify_load_max_over_mean"] == 8 * (100 + 10) / (800 + 10)


def test_imbalanced_first_call():
    spec = SimpleNamespace(stats={})
    _count(spec, [100, 100, 100, 100], [0, 0, 0, 0], 4)
    assert spec.stats["spec/verify_load_first_max_over_mean"] == 4.0
