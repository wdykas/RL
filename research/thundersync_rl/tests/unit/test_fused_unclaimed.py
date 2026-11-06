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
"""fused/unclaimed (StreamingLearner.finish): retained verification graphs dropped unclaimed, counted once per
learner data-parallel replica at every layout (rank 0's count alone misses DP - 1 replicas; TP / PP ranks of a
replica repeat its count)."""

import numpy as np
import pytest
import ray

from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.worker_groups import RayWorkerGroup
from thundersync_rl import streaming

# (world, tp, pp): TP1 EP4; TP2; TP4; TP8 PP2 one replica; TP8 PP2 four replicas
LAYOUTS = [(4, 1, 1), (16, 2, 1), (16, 4, 1), (16, 8, 2), (64, 8, 2)]


@pytest.fixture(autouse=True)
def _ray_get(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda x: x)


def _finish_policy(world, tp, pp, unclaimed_per_replica):
    ann = NamedSharding(
        layout=np.arange(world).reshape(pp, -1, 1, tp),
        names=["pipeline_parallel", "data_parallel", "context_parallel", "tensor_parallel"],
    )

    def res(c):
        return {
            "grad_norm": 1.0,
            "stream_peak_open_buffers": 0,
            # every rank of a replica holds (and drops) its replica's retained graphs
            "stream_fused_unclaimed": unclaimed_per_replica,
            "learner_mem_peak_gb": 10.0,
            "stream_dp_count_owner": c["tensor_parallel"] == 0 and c["pipeline_parallel"] == 0,
            "stream_rows_received": [0, 0, 0],
        }

    wg = object.__new__(RayWorkerGroup)
    wg.sharding_annotations = ann
    wg.dp_leader_worker_indices = list(range(world))  # Megatron: one tied group per worker
    wg.run_all_workers_single_data = lambda method, **kw: [res(ann.get_worker_coords(i)) for i in range(world)]

    class _P:
        pass

    p = _P()
    p.worker_group, p.sharding_annotations = wg, ann
    return p


@pytest.mark.parametrize("world,tp,pp", LAYOUTS)
def test_fused_unclaimed_counts_every_dp_replica(world, tp, pp):
    out = streaming.StreamingLearner(_finish_policy(world, tp, pp, unclaimed_per_replica=3)).finish()
    assert out["stream_fused_unclaimed"] == 3 * (world // (tp * pp))
