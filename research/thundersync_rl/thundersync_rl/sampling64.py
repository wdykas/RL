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
"""53-bit categorical draws for the verifier's residual / bonus emission (redteam_1016 §6)."""

from __future__ import annotations

import torch


def sample_categorical64_t(dist: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """One draw from ``dist`` (unnormalized, >= 0) with 53-bit exponentials, in fp64 (device tensor, [1]).

    Replaces ``torch.multinomial(dist.float(), 1)`` in the verifier's residual / bonus
    emission: multinomial(n=1) is argmax(p / E) with fp32 E (floor 2^-24, and the fp32
    cast zeroes entries below 1e-45 and rounds every entry by 2^-24). Here E is an fp64
    exponential (``Tensor.exponential_`` on a float64 tensor draws 53-bit uniforms) and
    the race is log p - log E in fp64. Same generator, one fp64 draw per vocabulary entry.
    """
    d = dist.double()
    e = torch.empty_like(d).exponential_(generator=generator)
    score = torch.where(d > 0, d.log() - e.log(), torch.full_like(d, float("-inf")))
    return torch.argmax(score).view(1)


def sample_categorical64(dist: torch.Tensor, generator: torch.Generator) -> int:
    """``sample_categorical64_t`` as a Python int."""
    return int(sample_categorical64_t(dist, generator))
