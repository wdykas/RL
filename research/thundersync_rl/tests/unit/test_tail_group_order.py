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
"""Canonical tail-group order (thundersync.tail_group_order, DESIGN 14ae)."""

from collections import Counter

import pytest
from thundersync_rl.grpo_loop import order_tail_groups
from thundersync_rl.speculative import Plan, SpeculativeGeneration
from thundersync_rl.verify0_stream import StreamOrder

G = 2
KEYS = [(i,) for i in range(6)]
ROW_KEYS = [k for k in KEYS for _ in range(G)]


def test_index_is_identity():
    assert order_tail_groups([4, 1, 3], ROW_KEYS, G, lambda k: 5, "index") == [4, 1, 3]


def test_kept_most_kept_first_ties_by_index_none_first():
    kept = {(1,): 100, (2,): 7000, (3,): 100, (4,): None, (5,): 16}
    out = order_tail_groups([5, 3, 1, 2, 4], ROW_KEYS, G, kept.get, "kept")
    assert out == [4, 2, 1, 3, 5]


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        order_tail_groups([1], ROW_KEYS, G, lambda k: 1, "eta")


def test_order_independent_of_input_order():
    kept = {k: (7 * k[0]) % 5 for k in KEYS}
    a = order_tail_groups([0, 1, 2, 3, 4, 5], ROW_KEYS, G, kept.get, "kept")
    b = order_tail_groups([5, 3, 1, 0, 4, 2], ROW_KEYS, G, kept.get, "kept")
    assert a == b


def _spec():
    return SpeculativeGeneration.__new__(SpeculativeGeneration)


def test_note_tails_records_min_kept_of_incomplete_plans():
    s = _spec()
    s._tail_kept = {}
    p1 = Plan(seed=0, verified=(8192, 1249))
    p2 = Plan(seed=1, verified=(8192, 8101))
    done = Plan(seed=2, verified=(8192, 16), complete=True)
    fresh = Plan(seed=3)  # not made from a draft
    s._note_tails([((1,), p1), ((1,), p2), ((1,), done), ((2,), fresh)])
    assert s.tail_kept((1,)) == 1249
    assert s.tail_kept((2,)) is None
    # a later decision of the same prompt only lowers it
    s._note_tails([((1,), Plan(seed=4, verified=(500, 300)))])
    assert s.tail_kept((1,)) == 300


def test_stream_order_close_uses_tail_order():
    need = Counter(ROW_KEYS)
    so = StreamOrder(
        [ROW_KEYS[p * G] for p in range(len(KEYS))],
        need,
        groups_per_block=2,
        tail_order=lambda tails: sorted(tails, reverse=True),
    )
    # groups 0, 2 ready; 1, 3, 4, 5 have tails
    so.decided({KEYS[0]: (G, 0), KEYS[1]: (G, 1), KEYS[2]: (G, 0)})
    so.decided({KEYS[3]: (G, 2), KEYS[4]: (G, 1), KEYS[5]: (G, 1)})
    so.close()
    assert so.order == [0, 2, 5, 4, 3, 1]
    assert so.tail_groups == [5, 4, 3, 1]


def test_stream_order_rejects_non_permutation():
    so = StreamOrder(
        [ROW_KEYS[p * G] for p in range(len(KEYS))],
        Counter(ROW_KEYS),
        groups_per_block=2,
        tail_order=lambda tails: tails[:-1],
    )
    so.decided({k: (G, 1) for k in KEYS})
    with pytest.raises(RuntimeError):
        so.close()
