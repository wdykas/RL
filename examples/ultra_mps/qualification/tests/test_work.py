# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Reject incomplete work and auxiliary-only signals in functional qualification."""

import unittest

from verify_run import validate_work


class WorkTests(unittest.TestCase):
    def work(self):
        rows = {}
        values = {
            "train/global_valid_seqs": 64,
            "train/global_valid_toks": 1000,
            "train/loss": 0.1,
            "train/grad_norm": 0.2,
            "train/reward": 0.5,
            "train/weight_version_lag": 1,
            "train/advantages/min": -1,
            "train/advantages/max": 1,
            "timing/train/total_step_time": 10,
        }
        values.update(
            {
                "train/" + key: 0
                for key in (
                    "dropped_prompt_groups",
                    "evicted_stale_prompt_groups",
                    "aborted_stale_inflight_groups",
                    "replaced_prompt_groups",
                    "promoted_prompt_groups",
                )
            }
        )
        for step in (1, 2, 3):
            for tag, value in values.items():
                rows[step, tag] = (100 + step * 10, value)
        return rows

    def test_mtp_auxiliary_gradient_cannot_stand_in_for_grpo_signal(self):
        self.assertTrue(validate_work(self.work())["GRPO_signal_pass"])
        rows = self.work()
        rows[2, "train/advantages/min"] = (120, 0)
        rows[2, "train/advantages/max"] = (120, 0)
        self.assertFalse(validate_work(rows)["GRPO_signal_pass"])
        for key, value in [
            ("train/global_valid_seqs", 63),
            ("train/loss", float("nan")),
            ("train/weight_version_lag", 2),
            ("train/dropped_prompt_groups", 1),
        ]:
            bad = self.work()
            bad[2, key] = (120, value)
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_work(bad)
