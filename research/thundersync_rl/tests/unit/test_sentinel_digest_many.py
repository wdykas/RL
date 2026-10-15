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
"""digest_many (one host sync per check) equals tensor_digest bitwise."""

import pytest
import torch

import thundersync_rl.sentinel as sentinel
from thundersync_rl.sentinel import digest_many, digest_named, tensor_digest


def _tensors():
    g = torch.Generator().manual_seed(0)
    out = [
        ("empty", torch.zeros(0, dtype=torch.bfloat16), None),
        ("scalar", torch.tensor(3.5, dtype=torch.float32), None),
        ("bias", torch.zeros(128, dtype=torch.float32), None),
        ("bool", torch.rand(77, generator=g) > 0.5, None),
        ("bf16", torch.randn(1000, 33, generator=g).to(torch.bfloat16), None),
        ("bf16_as_fp16", torch.randn(513, generator=g).to(torch.bfloat16), torch.float16),
        ("fp32_as_fp16", torch.randn(64, 64, generator=g), torch.float16),
        ("fp16", torch.randn(4097, generator=g).to(torch.float16), None),
        ("int64", torch.randint(-(2**62), 2**62, (300,), generator=g), None),
        ("neg_zero", torch.tensor([0.0, -0.0, 1.0, -1.0]), None),
    ]
    out += [(f"many{i}", torch.randn(i + 1, generator=g), None) for i in range(50)]
    return out


@pytest.mark.parametrize("pack_elems", [1, 7, 64, 5000, 1 << 22])
def test_digest_many_equals_tensor_digest(pack_elems):
    items = _tensors()
    got = digest_many(items, pack_elems=pack_elems)
    assert list(got) == [k for k, _, _ in items]
    for key, t, dtype in items:
        assert got[key] == tensor_digest(t, dtype), key


def test_digest_many_large_tensors_cross_chunks(monkeypatch):
    # Tensors above pack_elems take the chunked path; shrink the chunk so the
    # position weights must continue across chunk boundaries exactly.
    monkeypatch.setattr(sentinel, "DIGEST_CHUNK", 1000)
    g = torch.Generator().manual_seed(1)
    items = [
        ("big", torch.randn(70000, generator=g).to(torch.bfloat16), None),
        ("big_cast", torch.randn(65521 + 3, generator=g), torch.float16),
        ("small", torch.randn(10, generator=g), None),
    ]
    got = digest_many(items, pack_elems=100)
    for key, t, dtype in items:
        assert got[key] == tensor_digest(t, dtype), key


def test_digest_many_detects_one_ulp_and_order():
    t = torch.randn(1000, dtype=torch.bfloat16)
    u = t.clone()
    u.view(torch.int16)[17] ^= 1
    d = digest_many([("a", t, None), ("b", u, None), ("c", t.flip(0), None)], pack_elems=1 << 20)
    assert d["a"] != d["b"] and d["a"] != d["c"]


def test_digest_named_uses_aliases():
    t = torch.randn(32, 8).to(torch.bfloat16)
    named = [("w", t)]
    dtypes = {"w": "float16", "w#buf": "float16"}
    got = digest_named(named, dtypes)
    assert got["w"] == tensor_digest(t, torch.float16)
    assert got["w#buf"] == tensor_digest(t, torch.float16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_digest_many_cuda_matches_cpu():
    items = [(k, t.cuda(), d) for k, t, d in _tensors()]
    got = digest_many(items)
    for key, t, dtype in items:
        assert got[key] == tensor_digest(t.cpu(), dtype), key
