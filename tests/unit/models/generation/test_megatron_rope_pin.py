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
"""RoPE tables read by captured inference CUDA graphs must stay alive (BI path)."""

from functools import lru_cache
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron.core")

from nemo_rl.models.generation.megatron.megatron_worker import (  # noqa: E402
    _pin_rotary_tables,
)

pytestmark = pytest.mark.mcore


class _Rope(torch.nn.Module):
    """Stands in for MCore RotaryEmbedding: forward is a shared lru_cache."""

    calls = 0

    @lru_cache(maxsize=2)
    def forward(self, max_seq_len, offset=0, packed_seq=False, cp_group=None):
        type(self).calls += 1
        return torch.arange(max_seq_len, dtype=torch.float32) + offset


def test_pinned_tables_survive_shared_cache_eviction_and_clear():
    _Rope.forward.cache_clear()
    engine_rope, other = _Rope(), _Rope()
    _pin_rotary_tables(SimpleNamespace(rotary_pos_emb=engine_rope))
    table = engine_rope(16)
    ptr = table.data_ptr()
    # Another model churns the shared cache past its size; a clear also runs.
    for n in range(3, 10):
        other(n)
    _Rope.forward.cache_clear()
    again = engine_rope(16)
    assert again.data_ptr() == ptr and again is table
    # The pinned forward is not an lru_cache, so cache_clear() sites skip it.
    assert not hasattr(engine_rope.forward, "cache_parameters")
    # Unpinned instances keep the original cached behaviour.
    assert hasattr(other.forward, "cache_parameters")


def test_pin_is_a_no_op_without_cached_rope():
    module = SimpleNamespace(rotary_pos_emb=None)
    _pin_rotary_tables(module)  # no error
    plain = SimpleNamespace(rotary_pos_emb=SimpleNamespace(forward=lambda *a: None))
    _pin_rotary_tables(plain)
    assert not hasattr(plain.rotary_pos_emb.forward, "cache_parameters")
