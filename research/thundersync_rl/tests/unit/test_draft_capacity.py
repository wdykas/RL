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
"""draft_admission rounds / fit.

- rounds: primaries, then the extras by index across pools; each pool keeps its order.
- fit: that order cut to the drafting engine's capacity before submission, so no draft queues.
- Exactness: a pool whose set of drafts is fixed before drafting (fit) yields iid p slots at any
  pool size; a draft whose PRESENCE depends on a later draft of its own pool (queued admission
  under wall-clock cuts) biases the slots (enumeration control).
"""

from __future__ import annotations

import asyncio
import itertools
import math
from types import SimpleNamespace

import pytest
import thundersync_rl.speculative as spec_mod
import torch
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from test_null_bv import _draft_outcome_law, _group_law, _model, _per_draft_law, _seq_prob
from thundersync_rl.grpo_loop import DRAFT_ADMISSIONS, next_draft_prompts, validate_null_bv
from thundersync_rl.null_bv import NullPolicy


def _batch(prompts):
    return BatchedDataDict(
        {"message_log": [[{"token_ids": torch.tensor(p)}] for p in prompts], "idx": list(range(len(prompts)))}
    )


@pytest.mark.parametrize("mode", ["rounds", "fit"])
def test_rounds_order_primaries_then_extras_by_index(mode):
    G, prompts = 4, [[1, 2], [3], [4, 5, 6]]
    cfg = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=1.5, draft_admission=mode)
    out, slots = next_draft_prompts(_batch(prompts), cfg, G)
    keys = [tuple(p.tolist()) for p in out]
    n = 6
    assert len(keys) == n * len(prompts)
    assert keys[: G * len(prompts)] == [tuple(p) for p in prompts for _ in range(G)]
    # extra round 1 of every pool, then round 2
    assert keys[G * len(prompts) :] == [tuple(p) for _ in range(n - G) for p in prompts]
    assert slots == {tuple(p): G for p in prompts}
    # any prefix that holds the primaries cuts each pool to an index-order prefix
    for cut in range(G * len(prompts), n * len(prompts) + 1):
        per = {}
        for k in keys[:cut]:
            per[k] = per.get(k, 0) + 1
        assert all(G <= c <= n for c in per.values())
        assert max(per.values()) - min(per.values()) <= 1  # rounds: pools differ by at most one extra


def test_admission_values_validated():
    assert set(DRAFT_ADMISSIONS) == {"group", "primaries_first", "rounds", "fit"}


def _fit_spec(cap, started):
    from test_speculative_driver import _spec as make_spec

    engine = {
        "install_draft_hidden_log": lambda **_: True,
        "engine_request_capacity": lambda: cap,
        "start_drafts": lambda prompts, budget, interval, hidden_keys=None: started.append(prompts),
        "draft_request_ids": lambda: {},
    }
    s = make_spec(
        {"stash_head": lambda **kw: 1},
        base_handlers=engine,
        verify_source="engine",
        hidden_transport="driver",
        null_bv=True,
        null_bv_policy={"fixed_m": 2.0},
        draft_admission="fit",
    )
    s.current_step = 0
    return s


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


@pytest.mark.parametrize("cap,kept", [(7, 7), (100, 9), (4, 4), (3, None)])
def test_fit_submits_what_the_engine_runs_at_once(_no_ray, cap, kept):
    G, prompts = 2, [[1], [2]]
    cfg = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=2.25, draft_admission="fit")
    out, slots = next_draft_prompts(_batch(prompts), cfg, G)  # N = max(G, round(4.5)) = 4 per pool
    assert len(out) == 8
    out = out + [torch.tensor([1])]  # a 9th draft (pool 1, last round) to make the cut visible
    started = []
    s = _fit_spec(cap, started)
    if kept is None:
        # fewer slots than primaries: every primary must run at once -> refuse
        with pytest.raises(RuntimeError, match="primary drafts"):
            asyncio.run(s.draft_streaming(out, null_slots=slots))
        return
    asyncio.run(s.draft_streaming(out, null_slots=slots))
    (sub,) = started
    assert len(sub) == kept
    assert sub == [p.tolist() for p in out[:kept]]  # a prefix of the rounds order
    assert s.stats["draft/submitted"] == kept
    assert s.stats["draft/fit_cut_extras"] == len(out) - kept


def test_fit_rejected_with_prompt_drop():
    from test_null_bv import _master

    with pytest.raises(ValueError, match="draft_admission=fit"):
        validate_null_bv(_master(draft_admission="fit", null_bv_prompt_drop=True, null_bv_fixed_m=2.0))
    validate_null_bv(_master(draft_admission="fit"))
    with pytest.raises(ValueError, match="draft_admission must be auto or one of"):
        validate_null_bv(_master(draft_admission="queue"))


# ---------------------------------------------------------------------------
# Exactness by enumeration
# ---------------------------------------------------------------------------

V, L = 3, 2


def _setup():
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    p1 = {z: _seq_prob(P, z) for z in seqs}
    return P, Q, seqs, p1


class _Rule(NullPolicy):
    def m(self, d, s):
        return 1.0 if d <= s else 2.0


@pytest.mark.parametrize("N", [2, 3, 4, 5])
def test_fixed_pool_of_any_size_is_iid_p(N):
    """fit: pools of different sizes (fixed before drafting) are each exact."""
    P, Q, seqs, p1 = _setup()
    G = 2
    pol = _Rule()
    per = {M: _per_draft_law(P, Q, M, V, L) for M in (1.0, 2.0)}
    joint = _group_law(per, pol, N, G, seqs, p1)
    tv = 0.5 * sum(abs(joint.get(zs, 0.0) - math.prod(p1[z] for z in zs)) for zs in itertools.product(seqs, repeat=G))
    assert tv < 1e-9, tv


def _slot_law_presence(P, Q, seqs, present_given_x1):
    """G = 1 slot, drafts 0 then 1 of one pool. Draft 0 exists iff present_given_x1(x1),
    x1 = draft 1's tokens (queued admission: whether draft 0 started before the cut can depend on
    wall-clock time that draft 1's content changed). M = 2 while d > s, else 1."""
    pd2 = _per_draft_law(P, Q, 2.0, V, L)  # draft 0's outcome law (its own draft marginalised)
    law = {z: 0.0 for z in seqs}
    for x1 in seqs:
        q1 = _seq_prob(Q, x1)
        o1 = _draft_outcome_law(P, Q, x1, 1.0, V, L)  # draft 1 decided at d = s = 1 -> M = 1
        if present_given_x1(x1):
            for z in seqs:
                law[z] += q1 * (pd2.get(z, 0.0) + pd2.get("BOT", 0.0) * o1.get(z, 0.0))
        else:
            for z in seqs:
                law[z] += q1 * o1.get(z, 0.0)
    return law


def test_presence_depending_on_a_later_draft_is_biased_and_fixed_presence_is_exact():
    P, Q, seqs, p1 = _setup()
    fixed = _slot_law_presence(P, Q, seqs, lambda x1: True)
    tv_fixed = 0.5 * sum(abs(fixed[z] - p1[z]) for z in seqs)
    assert tv_fixed < 1e-12, tv_fixed
    absent_fixed = _slot_law_presence(P, Q, seqs, lambda x1: False)
    assert 0.5 * sum(abs(absent_fixed[z] - p1[z]) for z in seqs) < 1e-12
    # Draft 0 present only when draft 1 ends early (e.g. its EOS freed the clock): biased.
    timing = _slot_law_presence(P, Q, seqs, lambda x1: x1[0] == 0)
    tv_timing = 0.5 * sum(abs(timing[z] - p1[z]) for z in seqs)
    assert tv_timing > 1e-3, tv_timing


# ---------------------------------------------------------------------------
# Default (ts-integ5): auto -> fit on cohort lanes with ⊥-BV


@pytest.mark.parametrize(
    "kw,want",
    [
        (dict(), "group"),  # one engine
        (dict(cohort_lanes=2), "fit"),
        (dict(cohort_lanes=2, null_bv=False), "group"),
        (dict(cohort_lanes=2, null_bv_prompt_drop=True, null_bv_fixed_m=2.0), "group"),
        (dict(cohort_lanes=2, draft_admission="group"), "group"),  # still selectable
        (dict(cohort_lanes=2, draft_admission="rounds"), "rounds"),
        (dict(draft_admission="fit"), "fit"),
    ],
)
def test_auto_admission_is_fit_on_lanes_only(kw, want):
    from thundersync_rl.grpo_loop import ThunderSyncConfig, resolve_draft_admission

    base = dict(null_bv=True)
    base.update(kw)
    cfg = ThunderSyncConfig(**base)
    assert ThunderSyncConfig().draft_admission == "auto"
    assert resolve_draft_admission(cfg) == want


def test_auto_on_lanes_orders_like_fit():
    G, prompts = 4, [[1, 2], [3], [4, 5, 6]]
    fit = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=1.5, draft_admission="fit")
    auto = SimpleNamespace(null_bv=True, null_bv_drafts_per_output=1.5, draft_admission="auto", cohort_lanes=2)
    a, sa = next_draft_prompts(_batch(prompts), fit, G)
    b, sb = next_draft_prompts(_batch(prompts), auto, G)
    assert [p.tolist() for p in a] == [p.tolist() for p in b] and sa == sb


def test_auto_validated():
    from test_null_bv import _master

    validate_null_bv(_master())  # auto, one engine
    validate_null_bv(_master(cohort_lanes=2))  # auto -> fit
    validate_null_bv(_master(cohort_lanes=2, null_bv_prompt_drop=True, null_bv_fixed_m=2.0))  # auto -> group
