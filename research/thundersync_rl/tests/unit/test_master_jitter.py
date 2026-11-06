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
"""master_jitter: bf16 values unchanged, residuals spread (phi ~ 0.98), zeros untouched, seeded."""
import torch

from thundersync_rl import master_jitter as mj


def _grid(n=200_000, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(n, generator=g) * 0.02).to(torch.bfloat16).float()
    x[:100] = 0.0
    x[100:200] = 0.015625  # exact power of two
    x[200:300] = -0.25
    return x


def test_bf16_unchanged_and_phi():
    x = _grid()
    b0 = x.to(torch.bfloat16)
    n, bad, phi = mj.jitter_(x, 7)
    assert torch.equal(x.to(torch.bfloat16), b0)
    assert torch.all(x[:100] == 0)
    assert n + bad == int((b0.float() != 0).sum())
    assert 0.9 < phi / n < 1.05  # uniform on (-0.49, 0.49) ulp -> 4 E|r|/ulp = 0.98
    # only exact powers of two (mantissa 1.0, ~1/128 of bf16 values) can revert, about half of them
    assert bad <= 0.01 * (n + bad)


def test_seeded():
    a, b = _grid(), _grid()
    mj.jitter_(a, 3)
    mj.jitter_(b, 3)
    assert torch.equal(a, b)
    c = _grid()
    mj.jitter_(c, 4)
    assert not torch.equal(a, c)
