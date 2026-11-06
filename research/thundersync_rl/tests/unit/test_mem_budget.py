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
"""Micro-batch token cap vs learner memory (mem_budget): formula, startup rule, per-step fit."""

import types

import pytest

from thundersync_rl import grpo_loop, mem_budget

MiB = 2**20
# Full Nano 8k EP4 (final50r): peak 165.41 GiB at 16896 tokens, 3.1 MiB/token, 13.56 GiB non-torch, 184.31 GiB device.
NANO = {"bytes_per_token": 3.1 * MiB, "fixed_gb": 165.41 - 16896 * 3.1 / 1024, "nontorch_gb": 13.56, "margin_gb": 5.0}


def test_fit_formula():
    fit = mem_budget.fit_tokens(184.31, 13.56, NANO["fixed_gb"], NANO["bytes_per_token"], 5.0)
    assert 16900 < fit < 17100  # (184.31 - 13.56 - 5 - 114.26) / 3.1 MiB
    assert mem_budget.fit_tokens(10.0, 5.0, 10.0, MiB, 0.0) == 0  # no room


def test_nano_mbs2_cap_does_not_fit_and_names_one_row():
    with pytest.raises(ValueError, match=r"20480 tokens does not fit.*<= 1 rows \(<= 10240 tokens\)"):
        mem_budget.check_cap(20480, 10240, 184.31, NANO)
    assert mem_budget.check_cap(10240, 10240, 184.31, NANO) >= 10240


def test_zero_margin_threshold_matches_observed_oom_region():
    """Without the margin the OOM threshold is ~18.6k tokens: above the largest final50r micro-batch that ran
    (17664) and below the 20480 cap."""
    fit0 = mem_budget.fit_tokens(184.31, 13.56, NANO["fixed_gb"], NANO["bytes_per_token"], 0.0)
    assert 17664 < fit0 < 20480


def test_budget_off_by_default_and_incomplete_rejected():
    assert mem_budget.budget(grpo_loop.ThunderSyncConfig()) is None
    with pytest.raises(ValueError, match="needs train_mem_fixed_gb"):
        mem_budget.budget(types.SimpleNamespace(train_mem_bytes_per_token=1.0, train_mem_fixed_gb=None,
                                                train_mem_nontorch_gb=1.0, train_mem_margin_gb=5.0))


def test_measured_fit_uses_the_step_peak():
    mem_budget.check_cap(10240, 10240, 184.31, NANO)
    f = mem_budget.measured_fit(165.41, 16896, 13.56)
    assert abs(f - mem_budget.fit_tokens(184.31, 13.56, NANO["fixed_gb"], NANO["bytes_per_token"], 5.0)) <= 1
    assert mem_budget.measured_fit(float("nan"), 16896, 13.56) is None
