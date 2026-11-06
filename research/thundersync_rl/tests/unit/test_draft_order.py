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
"""next_draft_prompts: ⊥-BV primaries are submitted before the extras, pools keep their order."""

from types import SimpleNamespace

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from thundersync_rl.grpo_loop import next_draft_prompts


def _batch(prompts):
    return BatchedDataDict(
        {"message_log": [[{"token_ids": torch.tensor(p)}] for p in prompts], "idx": list(range(len(prompts)))}
    )


def test_null_bv_primaries_first_and_pool_order_kept():
    G, prompts = 4, [[1, 2], [3], [4, 5, 6]]
    cfg = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=1.5, draft_admission="primaries_first")
    out, slots = next_draft_prompts(_batch(prompts), cfg, G)
    keys = [tuple(p.tolist()) for p in out]
    n = 6  # round(1.5 * 4) drafts per group
    assert len(keys) == n * len(prompts)
    # The first G * groups drafts are G per group, the extras follow.
    assert keys[: G * len(prompts)] == [tuple(p) for p in prompts for _ in range(G)]
    assert keys[G * len(prompts) :] == [tuple(p) for p in prompts for _ in range(n - G)]
    assert slots == {tuple(p): G for p in prompts}


def test_without_null_bv_drafts_are_group_major():
    G, prompts = 2, [[7], [8, 9]]
    out, slots = next_draft_prompts(_batch(prompts), SimpleNamespace(null_bv=False), G)
    assert [tuple(p.tolist()) for p in out] == [(7,), (7,), (8, 9), (8, 9)]
    assert slots is None


def test_group_admission_keeps_whole_pools_together():
    G, prompts = 4, [[1, 2], [3]]
    cfg = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=1.5, draft_admission="group")
    out, slots = next_draft_prompts(_batch(prompts), cfg, G)
    assert [tuple(p.tolist()) for p in out] == [(1, 2)] * 6 + [(3,)] * 6
    assert slots == {(1, 2): G, (3,): G}
