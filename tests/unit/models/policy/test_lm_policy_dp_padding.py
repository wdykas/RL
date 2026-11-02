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
"""Batches that do not divide the data-parallel size are padded, then trimmed."""

from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.lm_policy import Policy


@pytest.mark.parametrize("num_rows,dp_size", [(4720, 64), (5, 4), (64, 64), (3, 8)])
def test_pad_then_trim_round_trips(num_rows, dp_size):
    policy = SimpleNamespace(data_parallel_size=dp_size)
    data = BatchedDataDict({"input_ids": torch.arange(num_rows).reshape(num_rows, 1)})

    padded, original = Policy._pad_for_dp(policy, data)
    assert original == num_rows
    assert padded.size % dp_size == 0
    assert padded.size - num_rows < dp_size

    # Stand in for the worker result: one output row per input row.
    result = BatchedDataDict({"logprobs": padded["input_ids"].float()})
    trimmed = Policy._trim_padding(result, original)

    assert trimmed.size == num_rows
    assert torch.equal(trimmed["logprobs"], data["input_ids"].float())
