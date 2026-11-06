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
"""Off-grid fp32 master initialization (acceptance-decay causal test of bf16 realization; research only).

``THUNDERSYNC_MASTER_JITTER=1``: before the first optimizer step of a fresh run, every nonzero fp32 master element m
(initialized on the bf16 grid from the bf16 HF weights) gets a seeded uniform offset in (-0.49, 0.49) bf16 ulp of m.
Elements whose bf16 rounding would change (exact powers of two, where the spacing below is half an ulp) are reverted,
so the stored bf16 weights stay bitwise identical: the policy is unchanged, only the masters' sub-ulp residuals start
spread (phase-in state phi ~1 instead of 0). Zeros are left untouched.
"""

from __future__ import annotations

import os
import zlib

import torch


def enabled() -> bool:
    return os.environ.get("THUNDERSYNC_MASTER_JITTER", "0") == "1"


def jitter_(master: torch.Tensor, seed: int) -> tuple[int, int, float]:
    """In place. Returns (elements jittered, elements reverted, sum of 4|r|/ulp over jittered elements)."""
    before = master.to(torch.bfloat16)
    ref = before.float()
    nz = ref != 0
    e = torch.frexp(ref).exponent
    ulp = torch.ldexp(torch.ones_like(ref), e - 8)
    g = torch.Generator(device=master.device)
    g.manual_seed(seed)
    u = (torch.rand(master.shape, generator=g, device=master.device, dtype=torch.float32) * 2 - 1) * 0.49
    new = torch.where(nz, ref + u * ulp, master.float())
    bad = nz & (new.to(torch.bfloat16) != before)
    new = torch.where(bad, master.float(), new)
    master.copy_(new.to(master.dtype))
    ok = nz & ~bad
    phi = float((4 * (master.float() - ref).abs() / ulp)[ok].double().sum())
    return int(ok.sum()), int(bad.sum()), phi


def seed_of(*key) -> int:
    """Seed from an element-identity key (e.g. parameter name, shard start); identical on every replica of the shard."""
    return zlib.crc32(("master_jitter|" + "|".join(str(k) for k in key)).encode())
