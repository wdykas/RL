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
"""Candidate sampler S (keyed_sampling_s.py) draws bitwise the production wide sampler's tokens (key64=True)."""

import pytest
import torch

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
@pytest.mark.parametrize("rows,vocab", [(1, 131072), (7, 131072), (64, 128000), (512, 131072), (33, 1000), (5, 2049)])
@pytest.mark.parametrize("peaked", [False, True])
def test_s_matches_wide_sampler(rows, vocab, peaked):
    from thundersync_rl import keyed_sampling_s as s
    from thundersync_rl.keyed_sampling import wide_sample_logits

    g = torch.Generator(device="cuda").manual_seed(rows * 7 + vocab)
    logits = torch.randn(rows, vocab, device="cuda", generator=g) * (8.0 if peaked else 1.0)
    if peaked:
        logits[:, 3] += 40.0  # near-deterministic rows
    logits[0, : min(vocab, 17)] = float("-inf")
    rids = torch.randint(0, 2**40, (rows,), device="cuda", generator=g)
    pos = torch.randint(0, 2**20, (rows,), device="cuda", generator=g)
    temp = torch.rand(rows, device="cuda", generator=g) * 1.5 + 0.25
    salt = 0x1234_5678_9ABC
    ref = wide_sample_logits(logits, rids, pos, temp, salt, key64=True)
    out = torch.empty(rows, dtype=torch.int64, device="cuda")
    ws = torch.empty(s.workspace_bytes(rows, vocab), dtype=torch.uint8, device="cuda")
    s.wide_sample(logits, rids, pos, temp, salt, out=out, workspace=ws)
    assert torch.equal(out, ref)
