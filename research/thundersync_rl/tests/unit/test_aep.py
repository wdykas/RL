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
"""Anchored deep pipeline AEP-1E (aep.py, ``thundersync.anchor_epoch``).

- Schedule / rules R1-R4 of ``AnchorEpoch`` (anchor a function of the cohort index, tails only on the anchor
  version, refits refused, per-cohort draft budgets that never overfill the engine).
- Config validation (the draft_lookahead == 1 rule lifted only for anchor_epoch; what AEP-1E refuses).
- Driver (fake engine): lambda cohorts drafted on one engine (start, then add next to running streams), the version
  tag = the anchor, per-cohort budget cut, the anchor head stashed once, unfilled / forced-fresh slots become
  continuations from the empty prefix verified piece by piece, rows never decode unverified on the anchor engine.
- Exactness by enumeration, wired to the real code paths: a toy MoE LM routed by the real sticky rule
  (``sticky_routing.sticky_select``), drafts from plain(theta_0) cut at the cohort's window (from the real
  ``AnchorEpoch`` schedule), verified by the real block verifier (``output_distribution_given_draft``) under the real
  ⊥-BV envelope policy (``NullPolicy``), continuations verified piece by piece against F(theta_k; A_theta_0), unfilled
  slots completed from the empty prefix: every cohort's G slots are iid p_k at ages 1..E. Controls (non-vacuous):
  a tail continued with another version's anchors (toy C1) and a per-row anchor choice by acceptance (toy C2) are
  biased, and ``AnchorEpoch`` refuses both.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import thundersync_rl.speculative as spec_mod
import torch
from test_null_bv import _group_law
from thundersync_rl.aep import AnchorEpoch, AnchorEpochError
from thundersync_rl.block_verification import output_distribution_given_draft
from thundersync_rl.null_bv import NullPolicy
from thundersync_rl.speculative import Plan

from nemo_rl.models.megatron.sticky_routing import sticky_select

# ---------------------------------------------------------------------------
# AnchorEpoch: schedule and rules


def _drive(lam, E, steps):
    """The loop's schedule: at each step, start every target ``targets`` gives, take cohort step+1."""
    aep = AnchorEpoch(lam=lam, epoch=E)
    started_at, taken = {}, []
    for step in range(steps):
        for t in aep.targets(step, draft_limit=steps, batches_ahead=lam):
            aep.started(t, step, aep.anchor(t), f"c{t}", 10)
            started_at[t] = step
        assert len(aep.in_flight()) <= lam
        c = aep.take(step + 1)
        if step + 1 < steps:
            assert c == f"c{step + 1}"
            taken.append(step + 1)
    return aep, started_at, taken


@pytest.mark.parametrize("lam", [1, 3, 4])
def test_schedule_starts_each_cohort_lambda_steps_ahead(lam):
    aep, started_at, taken = _drive(lam, E=8, steps=9)
    assert taken == list(range(1, 9))
    for c, s in started_at.items():
        assert s == max(0, c - lam)  # cohort c drafts from step max(0, c - lambda)
        assert aep.anchor(c) == 0 and aep.age(c) == c
    assert aep.leftover() == []


def test_anchor_is_a_function_of_the_cohort_index_only():
    aep = AnchorEpoch(lam=3, epoch=7)
    assert aep.anchor(0) is None and aep.age(0) == 0  # cohort 0: sampled by the engine at theta_0
    assert [aep.anchor(c) for c in range(1, 8)] == [0] * 7
    # Epoch switches: cohort 8 starts the next epoch (raw: sampled at theta_8), 9.. are anchored at 8.
    assert aep.anchor(8) is None and aep.anchor(9) == 8 and aep.anchor(15) == 8 and aep.anchor(16) is None
    # R1: a draft tagged with another version is refused at submission and at verification
    with pytest.raises(AnchorEpochError, match="R1"):
        aep.started(1, 0, 1, "c", 10)
    aep.check_drafts(3, [0, 0, 0])
    with pytest.raises(AnchorEpochError, match="R1"):
        aep.check_drafts(3, [0, 2, 0])  # per-row choice of anchor epoch (toy C2)


def test_tails_only_on_the_anchor_and_refits_refused():
    aep = AnchorEpoch(lam=3, epoch=7)
    aep.check_tail_version(5, 0)
    with pytest.raises(AnchorEpochError, match="R2"):
        aep.check_tail_version(5, 4)  # toy C1: the tail would use another version's anchors
    with pytest.raises(AnchorEpochError, match="R2"):
        aep.refuse_refit("theta_5")
    assert aep.head_keep() == [0]


def test_draft_budget_never_overfills_the_engine():
    aep = AnchorEpoch(lam=3, epoch=7, reserve_rows=32)
    # full Nano 8k: 64 x 8 = 512 outputs per cohort; max_requests 384 x DP 8 = 3072
    d = aep.draft_budget(3072, 512)
    assert d == (3072 - 512 - 32) // 3 == 842
    assert 3 * d + 512 + 32 <= 3072
    with pytest.raises(AnchorEpochError, match="needs >= 2080"):
        aep.draft_budget(2048, 512)  # 3 x 512 + 512 + 32 > 2048
    with pytest.raises(AnchorEpochError, match="capacity is unknown"):
        aep.draft_budget(0, 512)


# ---------------------------------------------------------------------------
# Config validation


def _master(steps=8, **ts):
    from thundersync_rl.grpo_loop import ThunderSyncConfig

    base = dict(
        speculate=True,
        verify_mode="block",
        verify_source="engine",
        verify_on="learner",
        draft_lookahead=3,
        sticky_routing=True,
        null_bv=True,
        anchor_epoch=8,
        stream_training_over_tails=True,
        canonical_group_order=True,
    )
    base.update(ts)
    return SimpleNamespace(
        thundersync=ThunderSyncConfig(**base),
        grpo=SimpleNamespace(max_rollout_turns=1, max_num_steps=steps, val_period=-1, val_at_start=False),
        policy={
            "megatron_cfg": {"expert_model_parallel_size": 1},
            "generation": {"mcore_generation_config": {"enable_prefix_caching": False}},
        },
    )


def test_config_valid_and_lifts_lookahead_only_for_anchor_epoch():
    from thundersync_rl.grpo_loop import (
        ThunderSyncConfig,
        resolve_draft_admission,
        validate_anchor_epoch,
        validate_null_bv,
        validate_sticky_routing,
    )

    assert ThunderSyncConfig().anchor_epoch is None and ThunderSyncConfig().anchor_weights == "raw"
    m = _master()
    validate_anchor_epoch(m)
    validate_sticky_routing(m)
    validate_null_bv(m)
    assert resolve_draft_admission(m.thundersync) == "fit"  # auto -> fit under anchor epochs
    off = _master(anchor_epoch=None)
    with pytest.raises(ValueError, match="draft_lookahead must be 1"):
        validate_sticky_routing(off)
    with pytest.raises(ValueError, match="draft_lookahead must be 1"):
        validate_null_bv(off)


@pytest.mark.parametrize(
    "ts,steps,match",
    [
        ({"anchor_weights": "momentum"}, 8, "anchor_start >= 1"),
        ({"cohort_lanes": 2}, 8, "one engine"),
        ({"null_bv": False}, 8, "needs null_bv"),
        ({"draft_admission": "group"}, 8, "draft_admission=fit"),
        ({"verify0_stream": True}, 8, "verify0_stream"),
        ({"stream_training_over_tails": False}, 8, "stream_training_over_tails"),
        ({"sticky_routing": False}, 8, "sticky_routing"),
        ({"draft_weights": "momentum"}, 8, "draft_weights must be current"),
        ({"draft_lookahead": 0}, 8, ">= 1"),
    ],
)
def test_config_rejects(ts, steps, match):
    from thundersync_rl.grpo_loop import validate_anchor_epoch

    with pytest.raises(ValueError, match=match):
        validate_anchor_epoch(_master(steps=steps, **ts))


def test_config_rejects_boundaries():
    from thundersync_rl.grpo_loop import validate_anchor_epoch

    # Epoch switches: boundaries only at epoch starts (periods multiple of anchor_epoch + 1 = 9).
    m = _master()
    m.grpo.val_period = 2
    with pytest.raises(ValueError, match="val_period=2 must be a multiple"):
        validate_anchor_epoch(m)
    m.grpo.val_period = 18
    validate_anchor_epoch(m)
    m = _master()
    m.checkpointing = {"enabled": True, "save_period": 4}
    with pytest.raises(ValueError, match="save_period=4 must be a multiple"):
        validate_anchor_epoch(m)
    m.checkpointing = {"enabled": True, "save_period": 9}
    validate_anchor_epoch(m)
    validate_anchor_epoch(_master(steps=500))  # any number of epochs
    # offset boundaries (anchor_start 2) are not "periodic" for NeMo-RL's keep_top_k retention: recency retention needed
    m = _master(anchor_start=2)
    m.checkpointing = {"enabled": True, "save_period": 9, "keep_top_k": 2}
    with pytest.raises(ValueError, match="ft_keep_latest_k"):
        validate_anchor_epoch(m)
    m.checkpointing["ft_keep_latest_k"] = 2
    validate_anchor_epoch(m)


# ---------------------------------------------------------------------------
# Driver (fake engine)


@pytest.fixture
def _no_ray(monkeypatch):
    monkeypatch.setattr(spec_mod.ray, "get", lambda x: x)


def _aep_spec(engine_rounds, verify=None, cap=100):
    from test_speculative_driver import _spec as make_spec
    from test_sticky_driver import _Engine

    engine = _Engine(engine_rounds)
    h = engine.handlers()
    h["engine_request_capacity"] = lambda: cap
    stashed = []
    learner = {"stash_head": lambda **kw: stashed.append(kw) or 1}
    if verify is not None:
        learner["verify_drafts_block_engine"] = verify
    s = make_spec(
        learner,
        base_handlers=h,
        verify_source="engine",
        hidden_transport="driver",
        sticky_delta=0.03,
        null_bv=True,
        null_bv_policy={"fixed_m": 2.0},
        draft_admission="fit",
        sticky_chunk_tokens=2,
    )
    s.aep = AnchorEpoch(lam=2, epoch=4)
    s.cohort_draft_version = 0
    return s, engine, stashed


def test_cohorts_share_the_engine_with_the_anchor_tag(_no_ray):
    from test_speculative_driver import EOD

    s, engine, stashed = _aep_spec(
        [
            [([4, EOD], True, [-0.1] * 2, 3), ([5, EOD], True, [-0.1] * 2, 3), ([6], False, [-0.1], 2)],
            [([7, EOD], True, [-0.1] * 2, 3), ([8, EOD], True, [-0.1] * 2, 3)],
        ]
    )
    s.current_step = 0
    slots = {(1,): 2}
    # cohort 1: three drafts, the per-cohort budget cuts the third (R4, fit order)
    s.aep_fit_cap = 2
    c1 = asyncio.run(
        s.draft_streaming([torch.tensor([1])] * 3, null_slots=slots, target=1, version=0)
    )
    assert s.stats["draft/submitted"] == 2 and s.stats["draft/fit_cut_extras"] == 1
    s.aep_fit_cap = 2
    c2 = asyncio.run(
        s.draft_streaming([torch.tensor([2])] * 2, null_slots={(2,): 2}, target=2, version=0)
    )
    # the first cohort started a round, the second was added next to its running streams
    assert len(engine.streams) == 4 and [p for p, _, _ in engine.started] == [[[1], [1]], [[2], [2]]]
    s.current_step = 1
    d2 = c2()
    assert [d[0][5] for d in d2] == [[(0, 0, 2)], [(0, 0, 2)]]  # anchor version tag, not current_step
    assert [d[0][9]["idx"] for d in d2] == [2, 3]  # its own streams (offset past cohort 1's)
    d1 = c1()
    assert [d[0][9]["idx"] for d in d1] == [0, 1]
    # the anchor's LM head is stashed once (step 0) and kept; never at later steps
    s.current_step = 0
    s.score_prev(d1)
    s.current_step = 1
    s.score_prev(d2)
    assert stashed == [{"version": 0, "keep": [0]}]


def test_unfilled_and_forced_fresh_slots_become_empty_prefix_continuations(_no_ray, monkeypatch):
    from test_speculative_driver import EOD

    monkeypatch.setenv("THUNDERSYNC_DEBUG_FRESH_MOD", "5")  # prompt [5] (sum 5) is forced fresh
    calls = []

    def verify(rows, prompt_lens, segments, keys, engine_logq, hidden, anchors, **_):
        calls.append(([r.tolist() for r in rows], prompt_lens, segments))
        # every piece of the empty-prefix continuations: accept whole
        return [
            (i, {"accepted": len(r) - p, "next": -1, "logprobs": [-1.0] * (len(r) - p + 1),
                 "draft_logq": engine_logq[i]})
            for i, (r, p) in enumerate(zip(rows, prompt_lens))
        ]

    s, engine, _ = _aep_spec(
        [
            # empty-prefix continuation streams (one per filled slot), finished when peeked
            [([20, EOD], True, [-0.2] * 2, 3), ([21, EOD], True, [-0.2] * 2, 3), ([22, EOD], True, [-0.2] * 2, 2)],
        ],
        verify=verify,
    )
    s._aep_streams_open = True
    s.current_step = 3
    a, b = (1, 2), (5,)
    s.plans[a] = [Plan(-1, [9, EOD], [-1.0, -1.0], complete=True, anchor_version=0)]
    s.plans[b] = [Plan(-1, [9, EOD], [-1.0, -1.0], complete=True, anchor_version=0)] * 2
    s.aep_need = collections.Counter({a: 2, b: 2})
    s._aep_fill(s.aep)
    assert s.stats["aep/fill_rows"] == 3 and s.stats["aep/forced_fresh_dropped_plans"] == 2
    fills = [p for q in s.plans.values() for p in q if p.aep_fill]
    assert len(fills) == 3 and all(p.prefix == [] and not p.complete and p.anchor_version == 0 for p in fills)
    assert s.tail_count(torch.tensor(a)) == 1 and s.tail_count(torch.tensor(b)) == 2
    # canonical tail order: a fill kept nothing at verify0 (decodes the whole row) -> ranked as kept 0
    s._tail_kept = {}
    s._note_tails([(k, p) for k, q in s.plans.items() for p in q])
    assert s.tail_kept(b) == 0 and s.tail_kept(a) == 0
    # the empty-prefix plans continue on the engine from the prompt; pieces verified against p_k (version 0 q)
    s.begin_ordered_learner()
    asyncio.run(s._continue_sticky([(k, p) for k, q in s.plans.items() for p in q if not p.complete]))
    s.end_ordered_learner()
    assert [p for p, _, _ in engine.started] == [[[1, 2], [5], [5]]]
    assert all(seg == [(0, 0, 2)] or seg == [(0, 0, 1)] for _, _, segs in calls for seg in segs)
    assert all(p.complete and p.prefix[-1] == EOD and p.anchors is not None for p in fills)


def test_no_unverified_decode_on_the_anchor_engine(_no_ray):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    s, _, _ = _aep_spec([])
    s.current_step = 2
    datum = BatchedDataDict({"input_ids": torch.tensor([[3]]), "input_lengths": torch.tensor([1])})
    with pytest.raises(RuntimeError, match="anchor_epoch"):
        asyncio.run(s._one(0, datum))  # no plan, no tails: would sample theta_0, not p_2
    s.current_step = 0  # the anchor step: the engine holds theta_0 = the target
    asyncio.run(s._one(0, datum))


def test_tail_rounds_refuse_another_version(_no_ray):
    s, _, _ = _aep_spec([])
    s.current_step = 2
    s.cohort_draft_version = 1
    with pytest.raises(AnchorEpochError, match="R2"):
        asyncio.run(s._continue_sticky([]))
    s.cohort_draft_version = 0
    s.aep.engine_version = 1  # an engine that moved off the anchor
    with pytest.raises(AnchorEpochError, match="R2"):
        asyncio.run(s._continue_sticky([]))


# ---------------------------------------------------------------------------
# Exactness by enumeration (real sticky rule, real block verifier, real ⊥-BV policy, real schedule)

V, T, D, DELTA = 3, 3, 4, 0.15
_rng = np.random.default_rng(7)
WR = _rng.normal(size=(2, D))
WE = _rng.normal(size=(2, V, D)) * 1.5
EMB = _rng.normal(size=(V + 1, D))
POS = _rng.normal(size=(T, D))
BASE = _rng.normal(size=(D, D)) * 0.6
DRIFT = _rng.normal(size=(D, D)) * 0.08
NOISE = [_rng.normal(size=(D, D)) * 0.03 for _ in range(20)]


def theta(k):
    return BASE + k * DRIFT + sum(NOISE[:k], np.zeros((D, D)))


def _hidden(th, prefix):
    t = len(prefix)
    h = POS[t] + EMB[V if t == 0 else prefix[-1]] + 0.3 * sum((EMB[y] for y in prefix), np.zeros(D))
    return np.tanh(th @ h)


def _softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def _route(th_k, th_anchor, prefix):
    """Expert of F(theta_k; A_anchor) at this position: the real sticky rule (top-1 of 2 experts)."""
    s = torch.tensor(WR @ _hidden(th_k, prefix), dtype=torch.float64).view(1, -1)
    if th_anchor is None:
        return int(s.argmax())
    a = int(np.argmax(WR @ _hidden(th_anchor, prefix)))
    pv, pi = s.topk(1, dim=1)
    _, idx, _ = sticky_select(s, torch.tensor([[a]]), pv, pi, DELTA)
    return int(idx[0, 0])


def policy(th_k, th_anchor):
    """Token law of F(theta_k; A_anchor) (th_anchor None: plain theta_k)."""

    @functools.lru_cache(maxsize=None)
    def f(prefix):
        return _softmax(WE[_route(th_k, th_anchor, list(prefix))] @ _hidden(th_k, list(prefix)))

    return lambda prefix: f(tuple(prefix))


def seq_law(pol):
    return {y: math.prod(pol(y[:t])[y[t]] for t in range(T)) for y in itertools.product(range(V), repeat=T)}


def _rows(pol, xs, n):
    return torch.tensor(np.log(np.stack([pol(tuple(xs[:t])) for t in range(n)])), dtype=torch.float64)


def _add(out, z, w):
    out[z] = out.get(z, 0.0) + w


def block_outcomes(p, q, start, xs, M):
    """(tau, emission probs) of the real verifier for draft piece xs after prefix ``start``."""
    full = list(start) + list(xs)
    n = len(xs)
    last = len(full) == T
    lp = torch.stack([torch.tensor(np.log(p(tuple(full[: len(start) + t]))), dtype=torch.float64)
                      for t in range(n + (0 if last else 1))])
    if last:
        lp = torch.cat([lp, lp[-1:]])  # bonus row at the cap (never used)
    lq = torch.stack([torch.tensor(np.log(q(tuple(full[: len(start) + t]))), dtype=torch.float64)
                      for t in range(n)])
    return output_distribution_given_draft(lp, lq, torch.tensor(xs), M)


def continuation_law(p, q, piece=2):
    """Law of ``_continue_sticky`` from a prefix: the anchor engine decodes q-pieces of ``piece`` tokens, each
    block-verified against p; whole acceptance continues the stream (no bonus), a rejection appends the residual
    token and restarts. Returns cont(prefix) -> {sequence: prob}."""

    @functools.lru_cache(maxsize=None)
    def cont(prefix):
        prefix = tuple(prefix)
        if len(prefix) == T:
            return {prefix: 1.0}
        n = min(piece, T - len(prefix))
        out: dict = {}
        for xs in itertools.product(range(V), repeat=n):
            qx = math.prod(q(prefix + xs[:t])[xs[t]] for t in range(n))
            for tau, probs in block_outcomes(p, q, prefix, xs, 1.0):
                probs = probs.double().numpy()
                if tau == n:
                    for z, v in cont(prefix + xs).items():
                        _add(out, z, qx * probs.sum() * v)
                    continue
                for y, py in enumerate(probs[:V]):
                    if py > 0:
                        for z, v in cont(prefix + xs[:tau] + (y,)).items():
                            _add(out, z, qx * py * v)
        return out

    return cont


def draft_law(p, q, n, M, cont):
    """Law of one ⊥-BV verification of a q-draft cut to ``n`` tokens at the deadline (+ "BOT"); a whole-accepted
    cut draft goes on as a continuation, a rejection continues from prefix + residual token."""
    out: dict = {}
    for x in itertools.product(range(V), repeat=n):
        qx = math.prod(q(x[:t])[x[t]] for t in range(n))
        for tau, probs in block_outcomes(p, q, (), x, M):
            probs = probs.double().numpy()
            if tau == n:
                for z, v in (cont(x) if n < T else {x: 1.0}).items():
                    _add(out, z, qx * probs.sum() * v)
                continue
            for y, py in enumerate(probs):
                if py == 0:
                    continue
                if y == V:
                    _add(out, "BOT", qx * py)
                    continue
                for z, v in cont(x[:tau] + (y,)).items():
                    _add(out, z, qx * py * v)
    return out


def _tv(a, b):
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b))


SEQS = list(itertools.product(range(V), repeat=T))


@pytest.mark.parametrize("lam", [3, 4])
def test_every_cohort_of_the_epoch_is_iid_p_by_enumeration(lam):
    """Cohorts 1..E of one epoch, scheduled by AnchorEpoch: drafts plain(theta_anchor), cut at their window,
    pooled ⊥-BV (N 3 drafts, G 2 slots, DP envelopes), tails and unfilled slots as continuations on the anchor."""
    E, N, G = 6, 3, 2
    aep, started_at, _ = _drive(lam, E, steps=E + 1)
    pol = NullPolicy(logratio_var=0.2, ref_len=T)
    Ms = {pol.m(d, s) for d in range(1, N + 1) for s in range(1, G + 1)}
    assert len(Ms) > 1, Ms
    worst, ages, cut_lens, accept = 0.0, [], [], []
    for c in range(1, E + 1):
        a = aep.anchor(c)
        th_a = theta(a)
        p, q = policy(theta(c), th_a), policy(th_a, None)
        n = min(T, c - started_at[c])  # tokens drafted by the cut: one per period of window
        cont = continuation_law(p, q)
        fill = cont(())
        P = seq_law(p)
        assert _tv(fill, P) < 1e-12  # an empty-prefix continuation is an exact p_k sample
        per = {M: draft_law(p, q, n, M, cont) for M in Ms}
        joint = _group_law(per, pol, N, G, SEQS, fill)
        tv = 0.5 * sum(abs(joint.get(zs, 0.0) - math.prod(P[z] for z in zs)) for zs in itertools.product(SEQS, repeat=G))
        worst = max(worst, tv)
        ages.append(aep.age(c))
        cut_lens.append(n)
        accept.append(per[1.0].get("BOT", 0.0))
    assert worst < 1e-9, worst
    assert ages == list(range(1, E + 1))
    assert min(cut_lens) < T and max(cut_lens) == T  # cut and complete drafts both occur


def test_sticky_target_differs_from_plain_and_drafts_are_rejected():
    """Non-vacuous: the anchored target is not plain theta_k, and q = plain(theta_0) is far from it at age 6."""
    p6 = policy(theta(6), theta(0))
    assert _tv(seq_law(p6), seq_law(policy(theta(6), None))) > 1e-3
    q = policy(theta(0), None)
    acc = sum(
        math.prod(q(y[:t])[y[t]] * min(1.0, p6(y[:t])[y[t]] / q(y[:t])[y[t]]) for t in range(T)) for y in SEQS
    )
    assert acc < 0.95


def test_control_tail_with_another_versions_anchors_is_biased():
    """Toy C1: after a rejection the tail is verified against F(theta_k; A_new) (anchor evicted): biased."""
    k, a, new = 9, 0, 8
    p = policy(theta(k), theta(a))
    q = policy(theta(a), None)
    p_wrong = policy(theta(k), theta(new))
    cont_wrong = continuation_law(p_wrong, policy(theta(new), None))
    out = draft_law(p, q, T, 1.0, cont_wrong)
    assert _tv(out, seq_law(p)) > 1e-3
    with pytest.raises(AnchorEpochError, match="R2"):
        AnchorEpoch(lam=3, epoch=9).check_tail_version(k, new)


def test_control_anchor_chosen_per_row_by_acceptance_is_biased():
    """Toy C2: a row takes anchor epoch a1 if its a1-draft is accepted whole, else a2's procedure: no mixture of
    the two sticky policies reproduces the output law."""
    k, a1, a2 = 6, 0, 4
    p1, p2 = policy(theta(k), theta(a1)), policy(theta(k), theta(a2))
    q1, q2 = policy(theta(a1), None), policy(theta(a2), None)
    whole = {}
    for x in SEQS:
        qx = math.prod(q1(x[:t])[x[t]] for t in range(T))
        whole[x] = qx * sum(pr.double().sum().item() for tau, pr in block_outcomes(p1, q1, (), x, 1.0) if tau == T)
    w = sum(whole.values())
    out2 = draft_law(p2, q2, T, 1.0, continuation_law(p2, q2))
    out = {z: whole.get(z, 0.0) + (1 - w) * out2.get(z, 0.0) for z in SEQS}
    P1, P2 = seq_law(p1), seq_law(p2)
    best = min(_tv(out, {z: lam * P1[z] + (1 - lam) * P2[z] for z in SEQS}) for lam in np.linspace(0, 1, 101))
    assert best > 1e-3, best
    with pytest.raises(AnchorEpochError, match="R1"):
        AnchorEpoch(lam=3, epoch=7).check_drafts(k, [a1, a2])


# ---------------------------------------------------------------------------
# Fixes after the first L8 run (aep1e-x2, 10-16)


def test_posted_verifications_are_read_from_the_yield_mailbox(_no_ray):
    """verify_yield: a verification run at a training boundary returns None from its ordered call; every
    ordered caller (VIT fallback included) must read the mailbox (aep1e-x1k crashed: 'NoneType' not iterable)."""
    s, _, _ = _aep_spec([])
    refs = [None]  # the ordered call's results: the posted verification already ran at a boundary
    rank_res = [(0, {"accepted": 1})]

    s._yield_mb = SimpleNamespace(result=SimpleNamespace(remote=lambda key: [(rank_res, None)]))
    s._posted_keys = {id(refs): "k1"}
    done = asyncio.run(s._gather_verify([(5, refs)]))
    assert done == [(5, [rank_res])]
    import inspect

    assert "_gather_verify" in inspect.getsource(type(s).vit_fallback)


def test_vit_gate_needs_no_budget_for_one_micro_batch_per_rank(_no_ray):
    """G <= DP x MBS: the group's retained forward is each rank's own training micro-batch -> not gated by
    vit_checkpoint_gb; G > DP x MBS: several micro-batches per rank are retained -> budget applies."""
    from test_speculative_driver import EOD

    def drafts(G):
        return [[(torch.tensor([1]), -1, [4, EOD], True, f"1:{j}", [(0, 0, 2)], [-0.1, -0.1], None, object())]
                for j in range(G)]

    for dp, mbs, gated in ((8, 2, 0), (2, 2, 1)):
        s, _, _ = _aep_spec([])
        s.current_step = 1
        s.verify_in_train, s.vit_group_size, s.vit_dp, s.vit_mbs = True, 8, dp, mbs
        s.vit_checkpoint_bytes, s.vit_bytes_per_token = 1.0, 1e6  # tiny budget
        s.vit_min_pass_prob = 0.0
        names, slots = {(1,): "1:0"}, {"1:0": 8}
        s._vit_split(drafts(8), names, slots)
        assert s.stats["vit/gated_groups"] == gated, (dp, mbs)
        assert s.stats["vit/candidate_groups"] == 1 - gated


# ---------------------------------------------------------------------------
# Momentum-extrapolated anchors (anchor_weights=momentum, anchor_start=w, anchor_momentum_steps=h)


def _harness_psi(w, mm, vv, t, L, lr, b1, b2, eps, wd):
    """``psi(n, st, L)`` of research/sticky_alt/adam_lag_alt.py (the E2 m{a} snapshots), verbatim."""
    for j in range(1, L + 1):
        t1 = t + j
        mm = b1 * mm
        vv = b2 * vv
        w = w - lr * ((mm / (1 - b1**t1)) / ((vv / (1 - b2**t1)).sqrt() + eps) + wd * w)
    return w


@pytest.mark.parametrize("h", [1, 2, 3, 8])
@pytest.mark.parametrize("t", [1, 7, 300])
def test_psi_h_matches_the_e2_harness_and_h_explicit_steps_bitwise(h, t):
    from thundersync_rl.draft_weights import momentum_draft_master

    g = torch.Generator().manual_seed(1000 * h + t)
    w = torch.randn(4097, generator=g)
    m = torch.randn(4097, generator=g) * 1e-3
    v = torch.rand(4097, generator=g) * 1e-6
    hp = dict(lr=1e-5, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    got = momentum_draft_master(w, m, v, step=t + 1, steps=h, **hp)
    ref = _harness_psi(w, m, v, t, h, hp["lr"], hp["beta1"], hp["beta2"], hp["eps"], hp["weight_decay"])
    assert torch.equal(got, ref)
    # h explicit zero-gradient AdamW steps (fp32 throughout, state decayed each step, one final rounding)
    ww, mm, vv = w.clone(), m.clone(), v.clone()
    for j in range(h):
        mm, vv = hp["beta1"] * mm, hp["beta2"] * vv
        c = t + 1 + j
        ww = ww - hp["lr"] * ((mm / (1 - hp["beta1"] ** c)) / ((vv / (1 - hp["beta2"] ** c)).sqrt() + hp["eps"])
                              + hp["weight_decay"] * ww)
    assert torch.equal(got, ww)
    assert torch.equal(got.bfloat16(), ww.bfloat16())
    if h == 1:  # unchanged one-step path (draft_weights=momentum)
        assert torch.equal(got, momentum_draft_master(w, m, v, step=t + 1, **hp))


def test_momentum_schedule_anchors_the_anchor_step_itself():
    aep = AnchorEpoch(lam=3, epoch=6, anchor0=2, kind="momentum", h=2)
    assert aep.anchor(2) == 2 and aep.age(2) == 0 and aep.anchor(8) == 2
    assert not aep.engine_holds_target(2) and not aep.engine_holds_target(3)
    with pytest.raises(AnchorEpochError, match="precedes the first epoch"):
        aep.anchor(1)
    # step 2 (anchor): cohorts 2..4 start at its top; cohort 2 is taken (cut inside step 2); then 5 starts
    assert aep.targets(2, draft_limit=9, batches_ahead=3) == [2, 3, 4]
    for t in (2, 3, 4):
        aep.started(t, 2, 2, f"c{t}", 10)
    assert aep.take(2) == "c2"
    assert aep.targets(2, draft_limit=9, batches_ahead=3) == [5]
    aep.started(5, 2, 2, "c5", 10)
    assert aep.in_flight() == [3, 4, 5] and aep.take(3) == "c3"
    assert aep.targets(3, draft_limit=9, batches_ahead=3) == [6]
    raw = AnchorEpoch(lam=3, epoch=6, anchor0=2)
    assert raw.anchor(2) is None and raw.engine_holds_target(2) and raw.targets(2, 9, 3) == [3, 4, 5]


def test_momentum_config():
    from thundersync_rl.grpo_loop import validate_anchor_epoch

    def m(**kw):
        x = _master(**kw)
        x.policy["megatron_cfg"]["optimizer"] = {"use_distributed_optimizer": True, "optimizer": "adam"}
        return x

    validate_anchor_epoch(m(anchor_weights="momentum", anchor_start=2, anchor_momentum_steps=2, steps=8))
    validate_anchor_epoch(m(anchor_start=2, steps=11))  # raw anchor at a later step
    for kw, steps, match in [
        (dict(anchor_weights="momentum", anchor_start=0), 8, "anchor_start >= 1"),
        (dict(anchor_momentum_steps=2), 8, "momentum only"),
        (dict(anchor_start=-1), 8, "anchor_start"),
    ]:
        with pytest.raises(ValueError, match=match):
            validate_anchor_epoch(m(steps=steps, **kw))
    x = _master(anchor_weights="momentum", anchor_start=2)
    x.policy["megatron_cfg"]["optimizer"] = {"use_distributed_optimizer": False}
    with pytest.raises(ValueError, match="distributed Adam"):
        validate_anchor_epoch(x)


def test_momentum_anchor_head_is_never_overwritten_by_the_live_head(_no_ray):
    s, _, stashed = _aep_spec([])
    s.aep = AnchorEpoch(lam=2, epoch=4, anchor0=2, kind="momentum", h=2)
    for step in (2, 3):
        s.current_step = step
        s.score_prev([[None]])
    assert stashed == []  # theta_hat's head was stashed by the anchor stage, never the learner's live theta_w


def _toy_adam_run(steps, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, wd=0.01, seed=3):
    """Toy parameters trained by real AdamW on a drifting synthetic gradient: (theta_k, m_k, v_k) per step."""
    rng = np.random.default_rng(seed)
    w = torch.tensor(BASE, dtype=torch.float64)
    m = torch.zeros_like(w)
    v = torch.zeros_like(w)
    out = [(w.clone(), m.clone(), v.clone())]
    gdir = torch.tensor(rng.normal(size=(D, D)))
    for k in range(steps):
        gr = gdir + 0.5 * torch.tensor(rng.normal(size=(D, D)))
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr * gr
        t = k + 1
        w = w - lr * ((m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + eps) + wd * w)
        out.append((w.clone(), m.clone(), v.clone()))
    return out, dict(lr=lr, beta1=b1, beta2=b2, eps=eps, weight_decay=wd)


def test_momentum_anchored_epoch_is_iid_p_by_enumeration():
    """Momentum anchor theta_hat = Psi_h(theta_w, m_w, v_w) through the real ``momentum_draft_master``, schedule from
    the real AnchorEpoch (anchor step w = 2, cohort w anchored at age 0, its drafts complete; later cohorts cut at
    their windows), real verifier / ⊥-BV policy / continuation pieces: every cohort's G slots are iid p_c."""
    from thundersync_rl.draft_weights import momentum_draft_master

    w0, E, lam, N, G, h = 2, 5, 3, 3, 2, 2
    run, hp = _toy_adam_run(w0 + E + 1)
    th = [r[0].numpy() for r in run]
    wv, mv, vv = run[w0]
    th_hat = momentum_draft_master(wv, mv, vv, step=w0 + 1, steps=h, **hp).numpy()
    aep = AnchorEpoch(lam=lam, epoch=E, anchor0=w0, kind="momentum", h=h)
    started_at = {}
    for step in range(w0, w0 + E + 1):
        for t in aep.targets(step, draft_limit=w0 + E + 1, batches_ahead=lam):
            aep.started(t, step, aep.anchor(t), f"c{t}", 10)
            started_at[t] = step
        aep.take(step) if step == w0 else None
        aep.take(step + 1)
        if step == w0:
            for t in aep.targets(step, draft_limit=w0 + E + 1, batches_ahead=lam):
                aep.started(t, step, aep.anchor(t), f"c{t}", 10)
                started_at[t] = step
    pol = NullPolicy(logratio_var=0.2, ref_len=T)
    Ms = {pol.m(d, s) for d in range(1, N + 1) for s in range(1, G + 1)}
    worst, acc_mom, acc_raw = 0.0, [], []
    for c in range(w0, w0 + E + 1):
        assert aep.anchor(c) == w0 and aep.age(c) == c - w0
        p, q = policy(th[c], th_hat), policy(th_hat, None)
        n = T if c == w0 else min(T, c - started_at[c])  # the anchor cohort is cut only once complete
        cont = continuation_law(p, q)
        fill = cont(())
        P = seq_law(p)
        assert _tv(fill, P) < 1e-12
        per = {M: draft_law(p, q, n, M, cont) for M in Ms}
        joint = _group_law(per, pol, N, G, SEQS, fill)
        tv = 0.5 * sum(abs(joint.get(zs, 0.0) - math.prod(P[z] for z in zs)) for zs in itertools.product(SEQS, repeat=G))
        worst = max(worst, tv)

        def acc(qq, pp):
            return sum(math.prod(qq(y[:i])[y[i]] * min(1.0, pp(y[:i])[y[i]] / qq(y[:i])[y[i]]) for i in range(T))
                       for y in SEQS)

        acc_mom.append(acc(q, p))
        pr = policy(th[c], th[w0])
        acc_raw.append(acc(policy(th[w0], None), pr))
    assert worst < 1e-9, worst
    assert min(acc_mom) < 0.999  # drafts are rejected: non-vacuous


def test_momentum_anchor_cohort_drafts_at_its_own_step(_no_ray):
    from test_speculative_driver import EOD

    s, engine, _ = _aep_spec(
        [[([4, EOD], True, [-0.1] * 2, 3)] * 2, [([5, EOD], True, [-0.1] * 2, 3)] * 2]
    )
    s.aep = AnchorEpoch(lam=2, epoch=4, anchor0=2, kind="momentum", h=2)
    s.current_step = 2
    s.aep_fit_cap = 2
    c2 = asyncio.run(s.draft_streaming([torch.tensor([1])] * 2, null_slots={(1,): 2}, target=2, version=2))
    asyncio.run(s.draft_streaming([torch.tensor([2])] * 2, null_slots={(2,): 2}, target=3, version=2))
    assert 2 in s._null_slots and 3 in s._null_slots  # the anchor cohort's slots survive the next start
    s._aep_cut_complete = True
    d2 = c2()
    assert [d[0][5] for d in d2] == [[(2, 0, 2)], [(2, 0, 2)]]
    s.current_step = 3
    with pytest.raises(ValueError, match="later step"):
        asyncio.run(s.draft_streaming([torch.tensor([1])], null_slots={(1,): 1}, target=3, version=2))


# ---------------------------------------------------------------------------
# Epoch switches (long runs)


def _drive_epochs(lam, E, steps, kind="raw", start=0):
    """The loop with switches: at every epoch start ``switch``; then start ``targets``, take cohort step+1."""
    aep = AnchorEpoch(lam=lam, epoch=E, anchor0=start, start=start, kind=kind, h=2 if kind == "momentum" else 1)
    started_at, taken, anchors_at_draft = {}, [], {}
    for step in range(start, steps):
        if aep.is_epoch_start(step):
            aep.switch(step)
            if kind == "momentum":  # the anchor step's own cohort, drafted and taken at its step
                for t in aep.targets(step, draft_limit=steps, batches_ahead=lam):
                    aep.started(t, step, aep.anchor(t), f"c{t}", 10)
                    started_at[t] = step
                    anchors_at_draft[t] = aep.engine_version
                assert aep.take(step) == f"c{step}"
        for t in aep.targets(step, draft_limit=steps, batches_ahead=lam):
            aep.started(t, step, aep.anchor(t), f"c{t}", 10)
            started_at[t] = step
            anchors_at_draft[t] = aep.engine_version
        assert len(aep.in_flight()) <= lam
        c = aep.take(step + 1)
        if c is not None:
            taken.append(step + 1)
            aep.check_tail_version(step + 1, aep.engine_version)  # its tails run on the engine's anchor
    return aep, started_at, taken, anchors_at_draft


@pytest.mark.parametrize("lam,E", [(1, 1), (3, 3), (3, 7), (4, 2)])
def test_epoch_switches_schedule(lam, E):
    P = E + 1
    aep, started_at, taken, at_draft = _drive_epochs(lam, E, steps=6 * P)
    assert aep.switches == 5
    for c, s in started_at.items():
        es = (c // P) * P
        assert aep.anchor(c) == es and at_draft[c] == es  # drafted on its own epoch's anchor, index-only
        assert s == max(es, c - lam)  # never drafted before its epoch starts (quiescent switch)
        assert 1 <= aep.age(c) <= E
    # every non-epoch-start cohort was drafted and taken; epoch starts are sampled plainly (raw)
    assert sorted(taken) == [c for c in range(1, 6 * P) if c % P]
    assert aep.leftover() == []


def test_epoch_switches_momentum_anchor_every_cohort():
    aep, started_at, taken, at_draft = _drive_epochs(3, 3, steps=2 + 5 * 4, kind="momentum", start=2)
    for c in range(2, 22):
        assert aep.anchor(c) == 2 + (c - 2) // 4 * 4 and 0 <= aep.age(c) <= 3
    assert all(at_draft[c] == aep.anchor(c) for c in at_draft)
    assert aep.switches == 4


def test_switch_refused_with_cohorts_in_flight_and_old_tails_refused():
    aep = AnchorEpoch(lam=3, epoch=3)
    for t in aep.targets(0, draft_limit=100, batches_ahead=3):
        aep.started(t, 0, aep.anchor(t), "c", 10)
    with pytest.raises(AnchorEpochError, match="not an epoch start"):
        aep.switch(3)
    with pytest.raises(AnchorEpochError, match="R2"):
        aep.switch(4)  # cohorts 1..3 still in flight
    for t in (1, 2, 3):
        aep.take(t)
    aep.switch(4)
    with pytest.raises(AnchorEpochError, match="R2"):
        aep.check_tail_version(3, aep.engine_version)  # a tail of epoch 0 on epoch 1's anchor
    with pytest.raises(AnchorEpochError, match="R1"):
        aep.started(5, 4, 0, "c", 10)  # a draft of epoch 1 tagged with epoch 0's anchor
    assert aep.targets(4, 100, 3) == [5, 6, 7] and aep.targets(7, 100, 3) == []  # 8 is the next epoch start
    assert aep.live_versions(4) == [] and aep.next_epoch_start(5) == 8 and aep.next_epoch_start(8) == 8


def test_resume_only_at_an_epoch_start():
    AnchorEpoch(lam=3, epoch=3, anchor0=10, start=2)  # 2, 6, 10: an epoch start
    with pytest.raises(AnchorEpochError, match="not an epoch start"):
        AnchorEpoch(lam=3, epoch=3, anchor0=9, start=2)


def test_epoch_switch_cohorts_are_iid_p_by_enumeration():
    """Three epochs (lambda 3, E 3) through the real schedule with switches: each anchored cohort's G slots are
    iid p_c under its own epoch's anchor (theta_s) -- and a control: verifying a cohort of epoch 1 with epoch 0's
    anchors but drafts from epoch 1's anchor (what a stale R1 tag would do) is biased."""
    lam, E, N, G = 3, 3, 3, 2
    aep, started_at, _, _ = _drive_epochs(lam, E, steps=3 * (E + 1))
    pol = NullPolicy(logratio_var=0.2, ref_len=T)
    Ms = {pol.m(d, s) for d in range(1, N + 1) for s in range(1, G + 1)}
    worst, anchors = 0.0, set()
    for c in range(1, 3 * (E + 1)):
        a = aep.anchor(c)
        if a is None:
            continue  # epoch start: sampled plainly at theta_c on the engine
        anchors.add(a)
        th_a = theta(a)
        p, q = policy(theta(c), th_a), policy(th_a, None)
        n = min(T, c - started_at[c])
        cont = continuation_law(p, q)
        per = {M: draft_law(p, q, n, M, cont) for M in Ms}
        joint = _group_law(per, pol, N, G, SEQS, cont(()))
        P = seq_law(p)
        tv = 0.5 * sum(abs(joint.get(zs, 0.0) - math.prod(P[z] for z in zs)) for zs in itertools.product(SEQS, repeat=G))
        worst = max(worst, tv)
    assert worst < 1e-9, worst
    assert anchors == {0, 4, 8}
    # Control: cohort 9 (epoch 2) drafted on theta_8 but its tails continued with epoch 0's anchors.
    c = 9
    p = policy(theta(c), theta(8))
    q = policy(theta(8), None)
    out = draft_law(p, q, T, 1.0, continuation_law(policy(theta(c), theta(0)), policy(theta(0), None)))
    assert _tv(out, seq_law(p)) > 1e-3
