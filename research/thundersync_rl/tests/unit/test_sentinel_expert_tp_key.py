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
"""The weight sentinel compares every expert tensor when learner EP != engine EP at TP > 1.

Digests are keyed by (canonical name, tensor-parallel rank). Experts are not TP-sharded
(expert_tensor_parallel_size is 1 on both sides: generation config forces it, refit
requires it), but the worker keyed them by the rank's *dense* TP rank. With ETP = 1
Megatron places EP ranks across TP ranks (order tp-cp-ep-dp: ep = rank % EP), so the
dense TP rank holding expert e depends on the EP size. When the engine's EP differs
from the learner's (EPINV / LANES=2 GEN_EP=EP/2) at TP > 1, the same expert lands under
different keys on the two sides and is silently not compared (only the
`sentinel/engine_only` counter moves; compare() raises only when *nothing* is common).

Simulates every rank of both sides through the real worker
``sentinel_weight_digest`` with a mocked ``parallel_state``.
"""

import pytest
import torch

import megatron.core.parallel_state as ps

from thundersync_rl.sentinel import WeightSentinelError, compare


class _Layer(torch.nn.Module):
    def __init__(self, number, n_local, ep_rank, bump=None):
        super().__init__()
        self.layer_number = number
        self.mlp = torch.nn.Module()
        self.mlp.experts = torch.nn.Module()
        self.mlp.experts.num_local_experts = n_local
        self.mlp.experts.linear_fc1 = torch.nn.Module()
        for j in range(n_local):
            g = ep_rank * n_local + j
            v = torch.full((2,), float(1000 * number + g))
            if bump == (number, g):
                v[0] += 1
            self.mlp.experts.linear_fc1.register_parameter(f"weight{j}", torch.nn.Parameter(v))
        self.norm = torch.nn.LayerNorm(2)


class _Model(torch.nn.Module):
    def __init__(self, n_layers, n_local, ep_rank, bump=None):
        super().__init__()
        self.module = torch.nn.Module()
        self.module.decoder = torch.nn.Module()
        self.module.decoder.layers = torch.nn.ModuleList(
            [_Layer(i + 1, n_local, ep_rank, bump) for i in range(n_layers)]
        )


def _side(monkeypatch, world, tp, ep, n_experts, learner, n_layers=2, bump=None):
    """[(key, digests)] of every rank of one side (one pipeline stage, ETP = 1)."""
    from thundersync_rl.worker import ThunderSyncMegatronPolicyWorker

    cls = ThunderSyncMegatronPolicyWorker.__ray_metadata__.modified_class
    out = []
    for r in range(world):
        tp_r, ep_r = r % tp, r % ep  # Megatron RankGenerator, order tp-cp-ep-dp, etp = 1
        monkeypatch.setattr(ps, "model_parallel_is_initialized", lambda: True)
        monkeypatch.setattr(ps, "get_tensor_model_parallel_rank", lambda v=tp_r: v)
        monkeypatch.setattr(ps, "get_expert_model_parallel_rank", lambda v=ep_r: v)
        monkeypatch.setattr(ps, "get_expert_tensor_parallel_rank", lambda: 0, raising=False)
        w = object.__new__(cls)
        w.model, w.rank = _Model(n_layers, n_experts // ep, ep_r, bump), r
        key, d, _cov = w.sentinel_weight_digest(learner=learner)
        out.append((key, d))
    return out


# (learner world, TP, EP) vs (engine world, TP, GEN_EP), experts
LAYOUTS = [
    ((4, 1, 4), (4, 1, 4), 8),  # Nano TP1 EP4
    ((8, 2, 8), (8, 2, 8), 16),  # Super TP2 EP8, same engine EP
    ((8, 2, 8), (4, 2, 4), 16),  # Super TP2 EP8, LANES=2 GEN_EP=4 (EPINV)
    ((8, 2, 2), (8, 2, 8), 16),  # Super-L8 TP2xEP2 learner, engine EP8
    ((8, 4, 8), (4, 4, 4), 16),  # TP4
    ((64, 8, 64), (64, 8, 64), 128),  # Ultra TP8 EP64 (one PP stage)
    ((64, 8, 64), (32, 8, 32), 128),  # Ultra TP8 EP64, engine EP32
]


@pytest.mark.parametrize("lrn,eng,n_experts", LAYOUTS)
def test_every_expert_compared(monkeypatch, lrn, eng, n_experts):
    n_layers = 2
    e = _side(monkeypatch, *eng, n_experts, learner=False, n_layers=n_layers)
    l_ = _side(monkeypatch, *lrn, n_experts, learner=True, n_layers=n_layers)
    cov = compare(e, l_, "t")
    expert_names = {n for _, d in l_ for n in d if ".experts." in n}
    assert len(expert_names) == n_experts * n_layers
    assert cov["sentinel/engine_only"] == 0 and cov["sentinel/learner_only"] == 0, cov


@pytest.mark.parametrize("lrn,eng,n_experts", LAYOUTS)
def test_every_expert_mismatch_detected(monkeypatch, lrn, eng, n_experts):
    l_ = _side(monkeypatch, *lrn, n_experts, learner=True)
    for g in range(n_experts):
        bad = _side(monkeypatch, *eng, n_experts, learner=False, bump=(2, g))
        with pytest.raises(WeightSentinelError, match=rf"layers\.1\.mlp\.experts\.linear_fc1\.weight{g}\["):
            compare(bad, l_, f"expert {g}")
