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
"""``thundersync.verify0_batch_tokens``: verify0 packs more rows per learner forward.

Exactness argument tested here (the GPU half, bitwise logits of a row in any
batch, is batch invariance, checked in-loop by the mismatch check):
* packing is order preserving: the rows a rank verifies, and the order its
  per-call random stream is consumed in, do not depend on the budget;
* only verify0 calls (one-shot waves, streamed calls) get the verify0 budget;
  continuation pieces keep ``verify_batch_tokens``;
* plans are identical with and without the knob on the fake engine.
"""

import asyncio
import random
from types import SimpleNamespace

import pytest
import torch

from thundersync_rl import speculative as spec_mod
from thundersync_rl.block_verification import NULL_TOKEN
from thundersync_rl.verify0_stream import Verify0Risk
from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker


@pytest.fixture(autouse=True)
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _batches(lengths, budget, m=128):
    fake = SimpleNamespace(cfg={"make_sequence_length_divisible_by": m})
    rows = [torch.zeros(n, dtype=torch.long) for n in lengths]
    cls = ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    return cls._row_batches(fake, rows, budget)


@pytest.mark.parametrize("seed", range(20))
def test_row_batches_preserve_order_for_any_budget(seed):
    rng = random.Random(seed)
    lengths = sorted(rng.randint(1, 8448) for _ in range(rng.randint(1, 120)))
    flat = None
    for budget in (0, 16384, 32768, 65536, 131072, 10**9):
        b = _batches(lengths, budget)
        order = [i for g in b for i in g]
        assert order == list(range(len(lengths)))  # same rows, same order: same random stream use
        for g in b:
            pad = -(-max(lengths[i] for i in g) // 128) * 128
            assert len(g) == 1 or len(g) * pad <= budget
        flat = order if flat is None else flat
        assert order == flat


def test_bigger_budget_fewer_forwards_at_8k():
    lengths = [5000] * 20 + [8400] * 80  # whole 8k drafts (padded 8448): one per 16384-token forward
    n16 = len(_batches(lengths, 16384))
    n64 = len(_batches(lengths, 65536))
    assert n16 == 7 + 80  # 3 x 5120 per forward, then one 8448 row per forward
    assert n64 == 14  # 12 x 5120, 8 x 5120, then 7 x 8448 per forward


def _sticky_spec(v0_batch):
    from test_speculative_driver import _spec as make_spec
    from test_sticky_driver import EOD, _Engine
    from test_verify0_stream import _engine_and_spec

    def rounds():
        return [
            [
                ([4, 5, 6], False, [-0.1] * 3, 4, [10, EOD]),
                ([7, EOD], True, [-0.2] * 2, 2),
                ([5], False, [-0.3], 0),
            ],
            [([11, EOD], True, [-0.5] * 2, 3)],
        ]

    outcome = {
        (1, 2, 4, 5, 6): (3, 99),
        (3, 7, EOD): (0, 9),
        (1, 2, 4, 5, 6, 10, EOD): (2, 99),
        (3, 9, 11, EOD): (1, EOD),
    }
    engine, s = _engine_and_spec(make_spec, _Engine, rounds(), outcome)
    s.verify0_batch_tokens = v0_batch
    return engine, s


def _verify_calls(s):
    out = []
    for name, kw in s.learner_policy.worker_group.calls:
        method = kw.get("method", name) if name == "ordered_call" else name
        if method == "verify_drafts_block_engine":
            out.append(kw["batch_tokens"])
    return out


def _plans(s):
    return {k: [(p.prefix, p.complete, p.anchors.length) for p in v] for k, v in s.plans.items()}


