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
"""GPU test: verify_yield.RngSnapshot survives a tracker made graph safe inside the inlined call.

A verification inlined at a training micro-batch boundary (verify_yield._yield_point) can run the learner's first
CUDA-graph capture, which turns Megatron's RNG tracker graph safe (learner_graphs.ensure_graph_safe_rng). The snapshot
taken before the call holds tensor states; restoring them into the graph-safe tracker made the next fork raise
``TypeError: expected a Generator, but got Tensor`` (Super-L8 TP2, LGRAPHS=1, prefix cache, step 2). The restore must
leave a usable tracker that continues from the snapshot's states.
"""

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


@pytest.fixture
def tracker(tmp_path):
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import random as mrandom

    from nemo_rl.models.megatron import learner_graphs

    created = False
    torch.cuda.set_device(0)
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
        created = True
    parallel_state.destroy_model_parallel()
    parallel_state.initialize_model_parallel()
    mrandom.model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    learner_graphs._RNG_DONE[0] = False
    yield mrandom.get_cuda_rng_tracker()
    mrandom.model_parallel_cuda_manual_seed(123, force_reset_rng=True)
    learner_graphs._RNG_DONE[0] = False
    parallel_state.destroy_model_parallel()
    if created:
        dist.destroy_process_group()


def _draw(tracker) -> torch.Tensor:
    with tracker.fork():
        return torch.rand(8, device="cuda")


def test_restore_after_the_call_made_the_tracker_graph_safe(tracker):
    from nemo_rl.models.megatron import learner_graphs
    from thundersync_rl.verify_yield import RngSnapshot

    assert not getattr(tracker, "use_cudagraphable_rng", False)
    snap = RngSnapshot()
    expected = _draw(tracker)  # what the model-parallel stream yields from the snapshot's state
    learner_graphs.ensure_graph_safe_rng()  # the inlined call's first capture
    assert tracker.use_cudagraphable_rng
    snap.restore()
    got = _draw(tracker)  # the old restore: TypeError: expected a Generator, but got Tensor
    assert torch.equal(got, expected)


def test_restore_without_a_mode_change_is_unchanged(tracker):
    from thundersync_rl.verify_yield import RngSnapshot

    snap = RngSnapshot()
    expected = _draw(tracker)
    _draw(tracker)  # the inlined call advances the stream
    snap.restore()
    assert torch.equal(_draw(tracker), expected)
