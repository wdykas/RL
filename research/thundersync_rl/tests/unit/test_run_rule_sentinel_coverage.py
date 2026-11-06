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
"""Run rule 1 (layout audit): the weight sentinel fails on tensors it never compares and on a compared-tensor
count that changes between refits."""

import pytest

from thundersync_rl import sentinel
from thundersync_rl.sentinel import WeightSentinel, WeightSentinelError, compare, unmatched

D = ("bfloat16", 4, 1, 2)


def _side(names, tp=0):
    return (tp, {n: D for n in names})


DENSE = ["decoder.layers.0.self_attention.linear_qkv.weight", "decoder.layers.0.mlp.router.weight"]
EXPERTS = [f"decoder.layers.0.mlp.experts.linear_fc1.weight{e}" for e in range(4)]


def test_full_coverage_passes():
    eng = [_side(DENSE + EXPERTS[:2], 0), _side(DENSE + EXPERTS[2:], 1)]
    lrn = [_side(DENSE + EXPERTS[:2], 0), _side(DENSE + EXPERTS[2:], 1)]
    cov = compare(eng, lrn, "ok")
    assert cov["sentinel/engine_only"] == 0 and cov["sentinel/learner_only"] == 0
    assert cov["sentinel/tensors"] == 2 * len(DENSE) + len(EXPERTS)


def test_c01_expert_keyed_by_other_tp_rank_fires():
    # Experts keyed by dense TP rank: at TP2 the learner (EP2) holds experts 0-1 on tp0 and 2-3 on tp1; the engine (EP4) keys expert 1 on tp1.
    eng = [_side(DENSE + EXPERTS[:1], 0), _side(DENSE + EXPERTS[1:], 1)]
    lrn = [_side(DENSE + EXPERTS[:2], 0), _side(DENSE + EXPERTS[2:], 1)]
    with pytest.raises(WeightSentinelError, match=r"1 engine-only and 1 learner-only.*weight1\[tp1\]"):
        compare(eng, lrn, "theta_1")
    e_only, l_only = unmatched(eng, lrn)
    assert e_only == [f"{EXPERTS[1]}[tp1]"] and l_only == [f"{EXPERTS[1]}[tp0]"]


def test_one_sided_tensor_fires_unless_allowlisted(monkeypatch):
    eng = [_side(DENSE + ["mtp.layers.0.weight"])]
    lrn = [_side(DENSE)]
    with pytest.raises(WeightSentinelError, match="engine-only"):
        compare(eng, lrn, "t")
    monkeypatch.setattr(sentinel, "ONE_SIDED_ALLOWLIST", (r"mtp\..*",))
    cov = compare(eng, lrn, "t")
    assert cov["sentinel/one_sided_allowed"] == 1 and cov["sentinel/engine_only"] == 1


def test_tensor_count_change_between_refits_fires(monkeypatch):
    s = WeightSentinel(policy=None)
    names = {"n": DENSE + EXPERTS}
    monkeypatch.setattr(s, "_submit_engine", lambda gen: lambda: [_side(names["n"])])
    monkeypatch.setattr(s, "_submit_learner", lambda dtypes: lambda: [_side(names["n"])])
    gen = object()
    ref = s.check(gen, "theta_0")
    s.check(gen, "theta_1")
    s.check_engine_against(gen, ref, "reload")
    names["n"] = DENSE + EXPERTS[:3]  # both sides drop a tensor: coverage shrank silently
    with pytest.raises(WeightSentinelError, match="tensors compared, 6 at this engine's first check"):
        s.check(gen, "theta_2")
    # Another engine (a drafter lane) has its own count.
    s.check(object(), "lane 1")
