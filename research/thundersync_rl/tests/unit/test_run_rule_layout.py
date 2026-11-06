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
"""Run rule 8 (layout audit): at startup every TP x CP and ETP x EP group lies in one verify-yield domain and
sees one EP lockstep schedule; per step, EP lockstep forward counts are equal within every EP and TP x CP group.

Groups come from Megatron's RankGenerator (the orders initialize_model_parallel uses) and each rank's domain from
the real ``verify_yield.yield_domain`` with parallel_state patched to that rank's coordinates.
"""

import pytest
from megatron.core import parallel_state as ps
from megatron.core.parallel_state import RankGenerator

from thundersync_rl import verify_yield
from thundersync_rl.run_checks import RunCheckError, check_layout, layout_violations, lockstep_violations

ORDER = "tp-cp-ep-dp-pp"


def _find(groups, r):
    for g in groups:
        if r in g:
            return g
    raise AssertionError(r)


def _reports(monkeypatch, world, tp, cp, ep, etp, yield_on=True, domain=None):
    """One ``layout_ids()`` report per rank; ``domain(edp_rank, dp_rank)`` overrides the real yield_domain."""
    dense = RankGenerator(tp=tp, ep=1, dp=world // (tp * cp), pp=1, cp=cp, order=ORDER)
    expert = RankGenerator(tp=etp, ep=ep, dp=world // (etp * ep), pp=1, cp=1, order=ORDER)
    out = []
    for r in range(world):
        dp_g, edp_g = _find(dense.get_ranks("dp"), r), _find(expert.get_ranks("dp"), r)
        coords = {
            "get_pipeline_model_parallel_world_size": 1 if yield_on else 2,
            "get_expert_model_parallel_world_size": ep,
            "get_expert_tensor_parallel_world_size": etp,
            "get_tensor_model_parallel_world_size": tp,
            "get_context_parallel_world_size": cp,
            "get_expert_data_parallel_rank": edp_g.index(r),
            "get_data_parallel_rank": dp_g.index(r),
        }
        for name, v in coords.items():
            monkeypatch.setattr(ps, name, lambda *a, _v=v, **k: _v, raising=False)

        def opt(g):
            return tuple(g) if len(g) > 1 else None

        out.append(
            {
                "rank": r,
                "dp": dp_g.index(r),
                "tp_cp": opt(_find(dense.get_ranks("tp-cp"), r)),
                "etp_ep": opt(_find(expert.get_ranks("tp-ep"), r)),
                "ep": opt(_find(expert.get_ranks("ep"), r)) if ep > 1 else None,
                "yield_domain": (
                    verify_yield.yield_domain() if domain is None else domain(edp_g.index(r), dp_g.index(r))
                ),
            }
        )
    return out


# (world, tp, cp, ep, etp): Nano TP1 EP4 / EP8; Super TP2 EP8 / EP4; TP4 EP4 / EP8; Ultra-like TP8 EP64; ETP2.
TARGET = [
    (8, 1, 1, 4, 1),
    (8, 1, 1, 8, 1),
    (16, 2, 1, 8, 1),
    (16, 2, 1, 4, 1),
    (16, 4, 1, 4, 1),
    (32, 4, 1, 8, 1),
    (64, 8, 1, 64, 1),
    (16, 2, 1, 8, 2),
    (8, 2, 1, 1, 1),  # dense TP2, no EP
]


@pytest.mark.parametrize("world,tp,cp,ep,etp", TARGET)
def test_target_layouts_pass(monkeypatch, world, tp, cp, ep, etp):
    reports = _reports(monkeypatch, world, tp, cp, ep, etp)
    assert layout_violations(reports) == []
    check_layout(reports)


@pytest.mark.parametrize("world,tp,cp,ep,etp", [(8, 4, 1, 2, 1), (16, 8, 1, 4, 1), (8, 2, 2, 2, 1)])
def test_b01_tp_group_across_yield_domains_fires(monkeypatch, world, tp, cp, ep, etp):
    # With EP < TP * CP the pre-fix yield domain (one per EP group) splits a TP x CP group.
    reports = _reports(monkeypatch, world, tp, cp, ep, etp, domain=lambda edp, dp: ("ep", edp))
    v = layout_violations(reports)
    assert any("TP x CP group" in x and "verify-yield domains" in x for x in v), v
    with pytest.raises(RunCheckError, match="learner layout check"):
        check_layout(reports)


def test_tp4_ep6_lockstep_fires_with_yield_off(monkeypatch):
    # The lockstep SUSPECT: TP4 EP6 (24 ranks). TP group 4-7 straddles EP groups 0-5 and 6-11, which cover data
    # replicas {0, 1} and {1, 2}: their lockstep lengths can differ. Yield off (PP > 1) isolates the lockstep rule.
    reports = _reports(monkeypatch, 24, 4, 1, 6, 1, yield_on=False)
    v = layout_violations(reports)
    assert v and all("EP lockstep groups cover different data-parallel replicas" in x for x in v), v
    # TP4 EP2 (8 ranks): both EP groups of a TP group hold one replica's rows -> same schedule, no lockstep
    # violation (the yield-domain violation, absent with yield off).
    assert layout_violations(_reports(monkeypatch, 8, 4, 1, 2, 1, yield_on=False)) == []


def test_injected_domain_split_fires():
    base = [
        {"rank": r, "dp": 0, "tp_cp": (0, 1), "etp_ep": None, "ep": None, "yield_domain": ("dp", 0)} for r in (0, 1)
    ]
    assert layout_violations(base) == []
    base[1]["yield_domain"] = ("dp", 1)
    assert layout_violations(base)


def test_lockstep_counts():
    # 4 ranks, EP groups {0, 1} {2, 3}, TP groups {0, 2} {1, 3}: (count, EP group id, TP group id).
    ok = [(5, 0, 0), (5, 0, 1), (5, 2, 0), (5, 2, 1)]
    assert lockstep_violations([{"stream_lockstep": x} for x in ok]) == []
    bad = list(ok)
    bad[1] = (6, 0, 1)  # rank 1 ran one more forward than its EP and TP peers
    v = lockstep_violations([{"stream_lockstep": x} for x in bad])
    assert any(x.startswith("EP group 0 ") for x in v) and any(x.startswith("TP x CP group 1 ") for x in v)
    assert lockstep_violations([{}, {"stream_lockstep": None}]) == []
