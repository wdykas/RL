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
"""spec/piece_prefix_reforward_frac is a fraction of the learner's data-parallel replica-seconds."""

from types import SimpleNamespace

import pytest
from thundersync_rl.grpo_loop import _reforward_frac


class _Ann:
    def __init__(self, dp):
        self.dp = dp

    def get_axis_size(self, name):
        assert name == "data_parallel"
        return self.dp


@pytest.mark.parametrize("tp,dp", [(1, 8), (2, 4), (8, 1)])
def test_reforward_frac_divides_by_data_parallel_size(tp, dp):
    # 8 learner workers; the re-forward seconds sum one (owning replica) result per row.
    group = SimpleNamespace(workers=[object()] * (tp * dp), sharding_annotations=_Ann(dp))
    m = {"spec/piece_prefix_reforward_s": 40.0, "time/step_total": 100.0}
    _reforward_frac(m, SimpleNamespace(worker_group=group))
    assert m["spec/piece_prefix_reforward_frac"] == pytest.approx(40.0 / (dp * 100.0))
