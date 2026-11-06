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
"""PERF_RECONCILE_1102 lever 1: groups ready together train in ONE learner call. The driver shards each group on its
own and sends every rank its slice of every group in one RPC; the worker trains the groups one after the other, each
exactly as its own call would (own fetch, own train_microbatch, hence own exact-accumulator capture)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nemo_rl.data_plane.worker_mixin import STREAM_BUCKET_TAG, TQWorkerMixin
from nemo_rl.models.policy.tq_policy import TQPolicy


def _policy(dp: int):
    policy = object.__new__(TQPolicy)
    policy.flops_tracker = None
    policy.worker_group = MagicMock()
    # group g's DP slice r: a stand-in meta tagged (g, r)
    policy._shard_train_meta = lambda m, timer, fields: [
        SimpleNamespace(g=m, r=r, sequence_lengths=[]) for r in range(dp)
    ]
    return policy


def test_one_rpc_carries_every_group_per_rank_in_order():
    policy = _policy(dp=3)
    policy.train_groups_from_metas(["gA", "gB"], train_fields=("input_ids",))
    wg = policy.worker_group
    assert wg.run_all_workers_sharded_data.call_count == 1  # one round trip for both groups
    call = wg.run_all_workers_sharded_data.call_args
    assert call.args[0] == "train_groups_presharded"
    assert call.kwargs["in_sharded_axes"] == ["data_parallel"]
    by_rank = call.kwargs["metas"]
    assert [[(m.g, m.r) for m in rank] for rank in by_rank] == [
        [("gA", r), ("gB", r)] for r in range(3)
    ]  # each rank: its own slice of every group, groups in call order
    wg.get_all_worker_results.assert_called_once()


class _Worker:
    """The worker-side entry point on a stand-in worker that records what it is asked to do."""

    train_groups_presharded = TQWorkerMixin.train_groups_presharded

    def __init__(self):
        self.log = []

    def _fetch(self, meta):
        self.log.append(("fetch", meta.name))
        return {"rows": meta.name}

    def _attach_or_repack_pack_metadata(self, data, meta):
        return data

    def train_microbatch(self, data):
        self.log.append(("train", data["rows"]))


def test_worker_trains_each_group_as_its_own_call():
    w = _Worker()
    metas = [SimpleNamespace(name=n, tags=[{}]) for n in ("g0", "g1", "g2")]
    w.train_groups_presharded(metas=metas)
    # one train_microbatch (= one train_microbatches call, = one capture under the exact accumulator) per group
    assert w.log == [("fetch", "g0"), ("train", "g0"), ("fetch", "g1"), ("train", "g1"), ("fetch", "g2"), ("train", "g2")]


def test_worker_refuses_stream_bucket_chunks():
    w = _Worker()
    with pytest.raises(ValueError, match="stream-bucket"):
        w.train_groups_presharded(metas=[SimpleNamespace(name="s", tags=[{STREAM_BUCKET_TAG: ["g", 1.0]}])])
    assert w.log == []
