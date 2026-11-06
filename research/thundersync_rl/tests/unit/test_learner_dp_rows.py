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
"""Every row dealt to the streaming learner reaches exactly one DP replica at TP / PP > 1.

Regression (research/tpfix): StreamingLearner.dp_size was RayWorkerGroup.dp_size, which
counts tied-worker groups (one per worker for a Megatron policy = the world size). Rows
were dealt into world-size lists and the sharded dispatch sent list d only to the ranks
with data-parallel coordinate d, so lists >= model DP were silently dropped (half the
rows at TP2). Uses the real RayWorkerGroup.run_all_workers_sharded_data routing and the
lm_policy sharding layout (pp, dp, cp, tp).
"""

import numpy as np
import pytest
import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.worker_groups import RayWorkerGroup
from thundersync_rl import streaming


class _Method:
    def __init__(self, log, idx):
        self.log, self.idx = log, idx

    def remote(self, **kw):
        self.log.append((self.idx, kw))
        return ("ref", self.idx)


class _Worker:
    def __init__(self, log, idx):
        self.stream_accumulate = _Method(log, idx)
        self.ordered_call = _Method(log, idx)


def _policy(world, tp, pp, cp=1):
    ann = NamedSharding(
        layout=np.arange(world).reshape(pp, -1, cp, tp),
        names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"],
    )
    wg = object.__new__(RayWorkerGroup)
    wg.sharding_annotations = ann
    log = []
    wg._workers = [_Worker(log, i) for i in range(world)]
    # What RayWorkerGroup builds for a Megatron policy: every worker its own tied group.
    wg.dp_leader_worker_indices = list(range(world))

    class _P:
        pass

    p = _P()
    p.worker_group, p.sharding_annotations = wg, ann
    return p, log


def _row(i, length=5):
    ids = torch.full((1, 8), i, dtype=torch.int64)
    tm = torch.zeros(1, 8)
    tm[0, 1:length] = 1
    return {
        "data": BatchedDataDict(
            {
                "input_ids": ids,
                "input_lengths": torch.tensor([length]),
                "token_mask": tm,
                "sample_mask": torch.ones(1),
                "advantages": torch.ones(1, 8),
            }
        ),
        "group": None,
        "reward": None,
        "fused_key": None,
    }


@pytest.mark.parametrize(
    "world,tp,pp", [(8, 1, 1), (8, 2, 1), (8, 4, 1), (8, 2, 2), (8, 8, 1), (16, 2, 2)]
)
@pytest.mark.parametrize("ordered", [False, True])
def test_distinct_trained_rows_equal_dealt_rows(monkeypatch, world, tp, pp, ordered):
    monkeypatch.setattr(ray, "put", lambda x: x)
    monkeypatch.setenv("THUNDERSYNC_ROW_AUDIT", "1")
    policy, log = _policy(world, tp, pp)
    learner = streaming.StreamingLearner(policy)
    dp = world // (tp * pp)
    assert learner.dp_size == dp
    # The old source of the DP size: world size, not DP (the bug).
    assert policy.worker_group.dp_size == world
    n_rows = 64
    per_rank = [[] for _ in range(learner.dp_size)]
    for i in range(n_rows):
        per_rank[i % learner.dp_size].append(_row(i))
    learner.submit(per_rank, [], seq=7 if ordered else None)

    ann = policy.sharding_annotations
    by_replica: dict[int, set] = {}
    for idx, kw in log:
        c = ann.get_worker_coords(idx)
        rows = {int(it["data"]["input_ids"][0, 0]) for it in kw["items"]}
        if c["tensor_parallel"] == 0 and c["pipeline_parallel"] == 0:
            by_replica.setdefault(c["data_parallel"], set()).update(rows)
        else:  # TP / PP ranks see their replica's rows
            assert rows == {int(it["data"]["input_ids"][0, 0]) for it in per_rank[c["data_parallel"]]}
    trained = [r for s in by_replica.values() for r in s]
    assert len(trained) == len(set(trained)) == n_rows  # distinct trained rows == dealt rows
    assert len(log) == world  # every learner rank runs the dispatch

    # The run-time audit sees the same: worker-side counts on one rank per replica.
    results = []
    for idx, kw in log:
        c = ann.get_worker_coords(idx)
        results.append(
            {
                "stream_dp_count_owner": c["tensor_parallel"] == 0 and c["pipeline_parallel"] == 0,
                "stream_rows_received": streaming.row_audit_items(kw["items"], True),
            }
        )
    audit = streaming.row_audit_merge(results, learner._sent)
    assert audit["rows_sent"] == audit["rows_received"] == n_rows
    assert audit["tokens_sent"] == audit["tokens_received"] == n_rows * 4
    assert audit["digest_sent"] == audit["digest_received"] != 0


def test_world_size_lists_are_rejected(monkeypatch):
    """Dealing into world-size lists (the old DP size) now fails loudly instead of dropping rows."""
    monkeypatch.setattr(ray, "put", lambda x: x)
    policy, _ = _policy(8, 2, 1)
    learner = streaming.StreamingLearner(policy)
    with pytest.raises(AssertionError, match="8 row lists for 4 learner DP replicas"):
        learner.submit([[_row(i)] for i in range(8)], [])


def test_audit_detects_dropped_replica():
    sent = streaming.row_audit_items([_row(i) for i in range(4)], True)
    got = [{"stream_dp_count_owner": True, "stream_rows_received": streaming.row_audit_items([_row(0), _row(1)], True)}]
    a = streaming.row_audit_merge(got, sent)
    assert a["rows_sent"] == 4 and a["rows_received"] == 2
    assert a["digest_sent"] != a["digest_received"]


def test_filler_rows_count_zero():
    it = _row(3)
    it["data"]["sample_mask"] = torch.zeros(1)
    assert streaming.row_audit(it["data"], True) == (0, 0, 0)
