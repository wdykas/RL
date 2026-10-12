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

import math

# Mirrors megatron.core.inference.batch_dimensions_utils.TOKEN_ROUNDER; the driver
# venv has no Megatron-Core, so it cannot be imported here.
MCORE_TOKEN_ROUNDER = 64
# Per-rank row multiple of the batch-invariant norm under sequence parallelism
# (its reduction code path depends on rows % 32).
NORM_ROW_CLASS = 32


def batch_invariant_token_multiple(tp_size: int) -> int:
    """Smallest token multiple compatible with batch-invariant MCore inference.

    lcm(64, 32*TP): a multiple of TP whose per-rank share is a multiple of 32.
    Equals MCore's TOKEN_ROUNDER (64) up to TP=2; at TP>=3
    ``ordered_tp_reduce.configure_inference_token_rounder`` raises MCore's to it.
    """
    if tp_size < 1:
        raise ValueError("tp_size must be positive.")
    return math.lcm(MCORE_TOKEN_ROUNDER, NORM_ROW_CLASS * tp_size)
