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
"""verify_yield agreement domains must contain every collective group.

A rank asks the mailbox whether to inline a posted verification at a micro-batch boundary,
keyed by its ``yield_domain()``. Ranks that share a collective inside a micro-batch (the
tensor/context-parallel group of the dense layers, the expert-parallel (x expert-TP) group of
the MoE layers) must read the SAME decision, i.e. lie in one domain; otherwise a post landing
between two ranks' questions makes them run different calls (collective mismatch / hang).

The groups are Megatron's own (``RankGenerator``, the orders ``initialize_model_parallel``
uses: dense tp-cp-ep-dp-pp with ep=1, expert tp-ep-dp-pp with expert tp = ETP). The domain of
each rank is computed by the real ``verify_yield.yield_domain`` with parallel_state patched to
that rank's coordinates.
"""

import math

import pytest
from megatron.core import parallel_state as ps
from megatron.core.parallel_state import RankGenerator
from thundersync_rl import verify_yield

ORDER = "tp-cp-ep-dp-pp"


def _groups(world, tp, cp, ep, etp):
    dense = RankGenerator(tp=tp, ep=1, dp=world // (tp * cp), pp=1, cp=cp, order=ORDER)
    expert = RankGenerator(tp=etp, ep=ep, dp=world // (etp * ep), pp=1, cp=1, order=ORDER)
    return dense, expert


def _index_in(groups, r):
    for g in groups:
        if r in g:
            return g.index(r)
    raise AssertionError(r)


def _domains(monkeypatch, world, tp, cp, ep, etp):
    dense, expert = _groups(world, tp, cp, ep, etp)
    out = {}
    for r in range(world):
        coords = {
            "get_pipeline_model_parallel_world_size": 1,
            "get_expert_model_parallel_world_size": ep,
            "get_expert_tensor_parallel_world_size": etp,
            "get_tensor_model_parallel_world_size": tp,
            "get_context_parallel_world_size": cp,
            "get_expert_data_parallel_rank": _index_in(expert.get_ranks("dp"), r),
            "get_data_parallel_rank": _index_in(dense.get_ranks("dp"), r),
        }
        for name, v in coords.items():
            monkeypatch.setattr(ps, name, lambda *a, _v=v, **k: _v, raising=False)
        out[r] = verify_yield.yield_domain()
    return out, dense, expert


# (world, tp, cp, ep, etp): Nano TP1 EP4; Super TP2 EP8 / TP2 EP4 / TP4 EP4 / TP4 EP8;
# Ultra-like TP8 EP64 (one PP stage); and TP > EP (TP4 EP2, TP8 EP4, ETP 1).
TARGET = [
    (8, 1, 1, 4, 1),
    (16, 2, 1, 8, 1),
    (16, 2, 1, 4, 1),
    (16, 4, 1, 4, 1),
    (32, 4, 1, 8, 1),
    (64, 8, 1, 64, 1),
    (16, 2, 1, 8, 2),
]
TP_GT_EP = [(8, 4, 1, 2, 1), (16, 8, 1, 4, 1), (8, 2, 2, 2, 1)]


def _check(domains, dense, expert):
    for kind, gens in (("tp", dense), ("cp", dense), ("ep", expert), ("tp", expert)):
        for g in gens.get_ranks(kind):
            ds = {domains[r] for r in g}
            assert len(ds) == 1, f"{kind} group {g} spans domains {ds}"


@pytest.mark.parametrize("world,tp,cp,ep,etp", TARGET)
def test_target_layouts_domains_hold_collective_groups(monkeypatch, world, tp, cp, ep, etp):
    domains, dense, expert = _domains(monkeypatch, world, tp, cp, ep, etp)
    _check(domains, dense, expert)


@pytest.mark.parametrize("world,tp,cp,ep,etp", TP_GT_EP)
def test_tp_larger_than_ep_domains_hold_collective_groups(monkeypatch, world, tp, cp, ep, etp):
    """EP < TP*CP or TP not dividing EP (ETP 1): one domain must still hold each TP group."""
    domains, dense, expert = _domains(monkeypatch, world, tp, cp, ep, etp)
    _check(domains, dense, expert)


# Same property with groups laid out directly (tp = rank % TP, ep = rank % EP, ETP 1).


def _domains_flat(monkeypatch, world, tp, ep, cp=1):
    out = []
    for r in range(world):
        vals = {
            "get_pipeline_model_parallel_world_size": 1,
            "get_tensor_model_parallel_world_size": tp,
            "get_tensor_model_parallel_rank": r % tp,
            "get_context_parallel_world_size": cp,
            "get_data_parallel_rank": r // (tp * cp),
            "get_expert_model_parallel_world_size": ep,
            "get_expert_model_parallel_rank": r % ep,
            "get_expert_data_parallel_rank": r // ep,
            "get_expert_tensor_parallel_world_size": 1,
        }
        for name, v in vals.items():
            monkeypatch.setattr(ps, name, lambda v=v: v, raising=False)
        out.append(verify_yield.yield_domain())
    return out


FLAT_LAYOUTS = [
    (4, 1, 4),  # Nano TP1 EP4
    (8, 1, 1),  # dense / EP1
    (8, 2, 1),  # TP2 EP1
    (8, 2, 2),  # Super-L8 TP2xEP2
    (8, 2, 8),  # Super TP2 EP8
    (8, 4, 8),  # Super TP4 EP8
    (8, 4, 4),  # TP4 EP4
    (32, 4, 32),  # Super full TP4 EP32
    (64, 8, 64),  # Ultra TP8 EP64 (one stage; verify_yield needs PP 1)
    (8, 4, 2),  # EP < TP
    (16, 8, 4),  # EP < TP
    (24, 4, 6),  # TP does not divide EP
]


@pytest.mark.parametrize("world,tp,ep", FLAT_LAYOUTS)
def test_tp_and_ep_groups_inside_one_domain(monkeypatch, world, tp, ep):
    dom = _domains_flat(monkeypatch, world, tp, ep)
    groups = [list(range(s, s + tp)) for s in range(0, world, tp)]
    groups += [list(range(s, s + ep)) for s in range(0, world, ep)] if ep > 1 else []
    for g in groups:
        assert len({dom[r] for r in g}) == 1, (g, [dom[r] for r in g])
