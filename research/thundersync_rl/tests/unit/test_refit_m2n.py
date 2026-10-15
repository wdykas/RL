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
"""m2n refit (refit_m2n.py, megatron_worker.Fp16HeldWeightReceive), no GPUs, no Ray.

- the bitwise digest the crosscheck compares sees any one-bit change;
- MCore's real generic reshard executor, driven by an in-process copy service,
  writes an fp16-held engine tensor bitwise as the packed broadcast's ``copy_``
  does (bf16 wire, fp16 storage; normal, subnormal and overflowing values), and
  a direct receive of those bytes would not (why the transform exists);
- the learner files and restores every lane's refit state (packed collective and
  native refit group).
"""

import pytest
import torch
from thundersync_rl.refit_m2n import compare_digests, tensor_digest


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32, torch.int64]
)
def test_digest_sees_every_bit(dtype):
    g = torch.Generator().manual_seed(0)
    t = (torch.randn(5000, generator=g) * 100).to(dtype)
    d0 = tensor_digest(t)
    assert tensor_digest(t.clone()) == d0
    bits = t.view({2: torch.int16, 4: torch.int32, 8: torch.int64}[t.element_size()])
    for pos in (0, 1, 2499, 4999):
        for bit in (0, 7, 8 * t.element_size() - 1):
            u = bits.clone()
            u[pos] ^= 1 << bit if bit < 8 * t.element_size() - 1 else -(1 << bit)
            assert tensor_digest(u.view(dtype)) != d0, (pos, bit)
    # A permutation (same multiset of values) is a different tensor.
    assert tensor_digest(t.flip(0)) != d0


def test_compare_digests_names_ranks_and_tensors():
    ref = [{"names": ["a", "b"], "digests": ["1", "2"]}] * 2
    got = [
        {"names": ["a", "b"], "digests": ["1", "2"]},
        {"names": ["a", "b"], "digests": ["1", "3"]},
    ]
    assert compare_digests(ref, got) == ["rank1:b"]
    assert compare_digests(ref, ref) == []
    assert compare_digests(ref, [ref[0], {"names": ["a"], "digests": ["1"]}])[0].startswith(
        "rank1:<tensor set differs"
    )


def _wire_values() -> torch.Tensor:
    """bf16 learner weights: normal values, fp16 subnormals, beyond fp16's range."""
    g = torch.Generator().manual_seed(1)
    v = torch.randn(4096, generator=g) * 0.02
    v[:16] = torch.tensor([1e-6, -3e-7, 6e-8, 2e-5, 7e4, -7e4, 1e5, 65504.0] * 2)
    return v.to(torch.bfloat16)


class _LocalService:
    """In-process CopyService: every send is matched to the receive with its task id."""

    supports_multiple_runs_per_plan = True
    requires_process_group_barrier = False

    def __init__(self):
        self.sends, self.recvs = {}, {}

    def set_plan(self, plan, *, transform=None):
        pass

    def set_model_roles(self, **kwargs):
        pass

    def execute_plan(self, *args, **kwargs):
        return False

    def submit_send(self, tensor, dest_rank, task_id=None):
        self.sends[task_id] = tensor.clone()

    def submit_recv(self, tensor, src_rank, task_id=None):
        self.recvs[task_id] = tensor

    def run(self):
        for task_id, buf in self.recvs.items():
            # Raw bytes, as NCCL moves them (no dtype conversion).
            buf.view(torch.uint8).copy_(self.sends.pop(task_id).view(torch.uint8))
        self.recvs.clear()


def _reshard_once(dst_tensor, wire, transform):
    from megatron.core.resharding.execution import _execute_batch
    from megatron.core.resharding.utils import TransferOp

    full = (slice(None),)
    send = TransferOp(param_name="w", peer_rank=1, is_send=True, my_slice=full, peer_slice=full, task_id=0)
    recv = TransferOp(param_name="w", peer_rank=0, is_send=False, my_slice=full, peer_slice=full, task_id=0)
    service = _LocalService()
    # Learner rank: no transform (bf16 on the wire); engine rank: the transform.
    _execute_batch([send], [], {"w": wire}, {}, service, None, None, None)
    _execute_batch([], [recv], {}, {"w": dst_tensor}, service, None, transform, None)


def test_fp16_receive_matches_broadcast_copy():
    pytest.importorskip("megatron.core.resharding.execution")
    from nemo_rl.models.generation.megatron.megatron_worker import Fp16HeldWeightReceive

    wire = _wire_values()
    # The packed broadcast path: the engine's fp16 parameter, copy_ of the bf16 weight.
    reference = torch.empty(wire.shape, dtype=torch.float16).copy_(wire)

    engine = torch.nn.Module()
    engine.w = torch.nn.Parameter(torch.full(wire.shape, float("nan"), dtype=torch.float16))
    engine.b = torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))
    transform = Fp16HeldWeightReceive(engine)
    assert transform.should_transform("w") and not transform.should_transform("b")
    _reshard_once(engine.w, wire, transform)
    assert torch.equal(engine.w.data.view(torch.int16), reference.view(torch.int16))
    assert torch.isinf(reference[4:7]).all() and (reference[:4] != 0).any()

    # Without the transform the bf16 bytes land in the fp16 storage unconverted.
    raw = torch.nn.Parameter(torch.zeros(wire.shape, dtype=torch.float16))
    _reshard_once(raw, wire, None)
    assert not torch.equal(raw.data.view(torch.int16), reference.view(torch.int16))


def test_lane_refit_state_is_filed_per_lane():
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    W = ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    learner = object.__new__(W)
    for lane in (0, 1):
        # What each lane's init leaves on the learner: native refit group (and,
        # with the crosscheck, the packed collective's group).
        learner.model_update_group = f"bcast{lane}" if lane else None
        learner.refit_pg = f"pg{lane}"
        learner.refit_copy_service = f"svc{lane}"
        learner.refit_dst_rank_offset = 4
        learner.keep_refit_group(lane=lane)
        assert learner.model_update_group is None
    for lane in (1, 0, 1):
        learner.select_refit_group(lane=lane)
        assert learner.refit_pg == f"pg{lane}"
        assert learner.refit_copy_service == f"svc{lane}"
        assert learner.model_update_group == (f"bcast{lane}" if lane else None)
    learner.model_update_group = learner.refit_pg = None
    with pytest.raises(RuntimeError, match="no refit communicator"):
        learner.keep_refit_group(lane=2)


def test_poison_then_rewrite_restores_digest():
    """The crosscheck's poison step: a refit tensor the m2n refit skipped would differ."""
    from thundersync_rl.refit_m2n import model_digest, poison_refit_tensors

    m = torch.nn.Linear(8, 4)
    m.register_buffer("bias_state", torch.arange(4, dtype=torch.int64))  # persistent
    m.register_buffer("cache", torch.ones(3), persistent=False)  # not refit
    names, ref, _ = model_digest(m)
    assert any("buffer/cache" in n for n in names)
    saved = {k: v.detach().clone() for k, v in m.state_dict().items()}
    assert poison_refit_tensors(m) == 3  # weight, bias, persistent buffer
    _, poisoned, _ = model_digest(m)
    changed = {n for n, a, b in zip(names, ref, poisoned) if a != b}
    assert changed == {"0/weight", "0/bias", "0/buffer/bias_state"}
    m.load_state_dict(saved)
    assert model_digest(m)[1] == ref
