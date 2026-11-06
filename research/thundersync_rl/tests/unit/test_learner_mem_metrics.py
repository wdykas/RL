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
"""Per-step learner memory over all ranks (the rank with the peak is not necessarily rank 0)."""

from thundersync_rl.streaming import merge_rank_memory


def test_peaks_max_headroom_min_and_peak_rank():
    r = [
        {"learner_mem_peak_gb": 160.0, "learner_mem_headroom_gb": 6.0, "learner_max_mb_tokens": 16896.0,
         "learner_mem_nontorch_gb": 10.0, "learner_empty_cache_s": 1.0},
        {"learner_mem_peak_gb": 166.5, "learner_mem_headroom_gb": 2.5, "learner_max_mb_tokens": 17408.0,
         "learner_mem_nontorch_gb": 13.5, "learner_empty_cache_s": 9.0},
    ]
    out = merge_rank_memory(r)
    assert out["learner_mem_peak_gb"] == 166.5 and out["learner_mem_peak_rank"] == 1.0
    assert out["learner_mem_headroom_gb"] == 2.5 and out["learner_max_mb_tokens"] == 17408.0
    assert out["learner_mem_nontorch_gb"] == 13.5
    assert "learner_empty_cache_s" not in out  # non-memory keys stay rank 0's


def test_ranks_without_memory_keys_and_ties():
    assert merge_rank_memory([{}, {"x": 1}]) == {}
    r = [{"learner_mem_peak_gb": 5.0}, {}, {"learner_mem_peak_gb": 5.0}]
    out = merge_rank_memory(r)
    assert out["learner_mem_peak_gb"] == 5.0 and out["learner_mem_peak_rank"] == 0.0  # first rank on a tie
