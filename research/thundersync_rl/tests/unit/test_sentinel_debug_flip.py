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
"""THUNDERSYNC_DEBUG_SENTINEL_FLIP (negative control for the weight sentinel): one byte, one call, digest moves."""

import pytest
import torch

from thundersync_rl.sentinel import DEBUG_FLIP_ENV, debug_flip_byte, digest_named


def _named():
    g = torch.Generator().manual_seed(0)
    return [("decoder.layers.0.w", torch.randn(64, 32, generator=g).to(torch.bfloat16)),
            ("decoder.layers.1.w", torch.randn(16, generator=g).to(torch.bfloat16))]


def test_flips_one_byte_at_the_named_call_only():
    named = _named()
    before = [t.clone() for _, t in named]
    d0 = digest_named(named, None)
    env = {DEBUG_FLIP_ENV: "2"}
    assert debug_flip_byte(named, 1, env) is None
    assert all(torch.equal(a.view(torch.uint8), t.view(torch.uint8)) for a, (_, t) in zip(before, named))
    assert debug_flip_byte(named, 2, env) == "decoder.layers.0.w"
    diff = (before[0].view(torch.uint8) != named[0][1].view(torch.uint8)).sum().item()
    assert diff == 1
    assert torch.equal(before[1].view(torch.uint8), named[1][1].view(torch.uint8))
    d1 = digest_named(named, None)
    assert d1["decoder.layers.0.w"] != d0["decoder.layers.0.w"]
    assert d1["decoder.layers.1.w"] == d0["decoder.layers.1.w"]
    assert debug_flip_byte(named, 3, env) is None


@pytest.mark.parametrize("env", [{}, {DEBUG_FLIP_ENV: ""}])
def test_off_by_default(env):
    named = _named()
    before = named[0][1].clone()
    for n in range(1, 4):
        assert debug_flip_byte(named, n, env) is None
    assert torch.equal(before.view(torch.uint8), named[0][1].view(torch.uint8))
