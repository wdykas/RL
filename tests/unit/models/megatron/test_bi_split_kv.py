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
"""CPU tests for the bi_split_kv wrapper (kwarg rewriting, validation). GPU bitwise tests live in
thundersync_work/research/bi_splitkv (bitwise_test.py, test_learner.py)."""

import pytest

from nemo_rl.models.megatron import bi_split_kv

CFG = {"enabled": True, "seqlen_k_per_split": 1024, "num_splits": 128}


def _fa4_stub(q, k, v, num_splits=1, seqlen_k_per_split=None, **kw):
    return dict(num_splits=num_splits, seqlen_k_per_split=seqlen_k_per_split, **kw)


def test_bi_call_becomes_anchored():
    f = bi_split_kv.wrap(_fa4_stub, CFG)
    out = f(1, 2, 3, num_splits=1, causal=True)
    assert out == dict(num_splits=128, seqlen_k_per_split=1024, causal=True)


def test_non_bi_and_explicit_calls_untouched():
    f = bi_split_kv.wrap(_fa4_stub, CFG)
    assert f(1, 2, 3, num_splits=0)["seqlen_k_per_split"] is None
    assert f(1, 2, 3, num_splits=1, seqlen_k_per_split=256)["seqlen_k_per_split"] == 256


def test_wrap_idempotent():
    f = bi_split_kv.wrap(_fa4_stub, CFG)
    assert bi_split_kv.wrap(f, CFG) is f


def test_requires_anchored_fa4():
    def old_fa4(q, k, v, num_splits=1):
        return None

    with pytest.raises(RuntimeError, match="seqlen_k_per_split"):
        bi_split_kv.wrap(old_fa4, CFG)


@pytest.mark.parametrize(
    "cfg",
    [
        {"seqlen_k_per_split": 1000, "num_splits": 8},
        {"seqlen_k_per_split": 1024, "num_splits": 1},
        {"seqlen_k_per_split": 1024, "num_splits": 300},
    ],
)
def test_validation(cfg):
    with pytest.raises(ValueError):
        bi_split_kv._resolve(cfg)


def test_install_requires_bi_mode():
    with pytest.raises(ValueError, match="batch_invariant_mode"):
        bi_split_kv.install({"bi_split_kv": CFG, "batch_invariant_mode": False})
