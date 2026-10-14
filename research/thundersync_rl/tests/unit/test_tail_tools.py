"""Opt-in tail tools (agent_tail.md): ⊥-BV prompt drop, drop-IS, tail-excluded baseline.

* prompt drop: exact enumeration through ``decide_pools`` — kept groups are
  p^G, P(drop) equals the count-only Binomial law for two different (p, q),
  and a content-based drop (control) is biased;
* drop-IS: the kept rows weighted by max(1, W*/M) are exactly p (up to the
  common constant), per draft and for the pooled process; unweighted (control)
  they are not;
* tail-excluded baseline: exact enumeration of E[sum_i A_i grad log p(y_i)]
  with tail flags that depend on y — equal to G E_p[r grad log p] (mean
  baseline), parallel to it (std-normalized); the GRPO-over-the-tail-free-subset
  variant and dropping the tail rows (controls) are biased;
* planner: tail-free rows dispatch before the group's tail arrives;
* the envelope guard rejects anything but counts.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest
import torch
from thundersync_rl.null_bv import NullGroup, NullPolicy, decide_pools, envelopes
from thundersync_rl.streaming import StreamPlanner, Trajectory, tail_excluded_advantages

from test_null_bv import _draft_outcome_law, _model, _per_draft_law, _seq_prob  # noqa: E402


# --------------------------------------------------------------------------
# Prompt drop (thinning at a constant envelope)
# --------------------------------------------------------------------------


def _pool_law_with_drop(P, Q, V, L, N, G, M, content_drop=None):
    """Exact (P(drop), kept-group law | not dropped) through ``decide_pools``."""
    seqs = list(itertools.product(range(V), repeat=L))
    pol = NullPolicy(fixed_m=M, prompt_drop=True)
    assert envelopes(pol) == (M,)
    per = _per_draft_law(P, Q, M, V, L)
    row = list(per.items())  # outcome -> prob ("BOT" or a sequence)
    kept: dict = {}
    p_drop = 0.0
    for combo in itertools.product(row, repeat=N):
        pr = math.prod(v for _, v in combo)
        outcomes = [{M: o != "BOT"} for o, _ in combo]
        dropped: set = set()
        chosen = decide_pools(["a"] * N, {"a": G}, outcomes, pol, dropped=dropped)
        outs = tuple(o for (o, _), m in zip(combo, chosen) if m is not None and o != "BOT")
        if content_drop is not None and content_drop(outs):
            dropped.add("a")
        if "a" in dropped:
            p_drop += pr
            continue
        assert len(outs) == G
        kept[outs] = kept.get(outs, 0.0) + pr
    return p_drop, kept, seqs


@pytest.mark.parametrize("seeds", [(7, 8), (3, 11)])
def test_prompt_drop_keeps_exact_groups_and_count_only_drop_rate(seeds):
    V, L, N, G, M = 2, 2, 4, 2, 2.0
    P, Q = _model(V, L, seeds[0], conc=0.5), _model(V, L, seeds[1], conc=0.5)
    p_drop, kept, seqs = _pool_law_with_drop(P, Q, V, L, N, G, M)
    # count-only: drop iff fewer than G successes among N Bernoulli(1/M)
    theory = sum(math.comb(N, k) * (1 / M) ** k * (1 - 1 / M) ** (N - k) for k in range(G))
    assert abs(p_drop - theory) < 1e-12, (p_drop, theory)
    p1 = {z: _seq_prob(P, z) for z in seqs}
    z = 1.0 - p_drop
    tv = 0.5 * sum(
        abs(kept.get(zs, 0.0) / z - math.prod(p1[s] for s in zs))
        for zs in itertools.product(seqs, repeat=G)
    )
    assert tv < 1e-9, tv


def test_content_based_drop_is_biased():
    V, L, N, G, M = 2, 2, 4, 2, 2.0
    P, Q = _model(V, L, 7, conc=0.5), _model(V, L, 8, conc=0.5)
    first = (0,) * L
    p_drop, kept, seqs = _pool_law_with_drop(
        P, Q, V, L, N, G, M, content_drop=lambda outs: first in outs
    )
    p1 = {z: _seq_prob(P, z) for z in seqs}
    z = 1.0 - p_drop
    tv = 0.5 * sum(
        abs(kept.get(zs, 0.0) / z - math.prod(p1[s] for s in zs))
        for zs in itertools.product(seqs, repeat=G)
    )
    assert tv > 1e-2, tv


def test_prompt_drop_policy_and_group_state():
    pol = NullPolicy(fixed_m=1.5, prompt_drop=True)
    assert pol.m(5, 5) == 1.5  # never the forced M = 1
    assert pol.drop(2, 3) and not pol.drop(3, 3)
    g = NullGroup(3, 2, pol)
    assert g.next_m() == 1.5
    g.record(False)  # BOT: 2 drafts for 2 slots, still fillable
    assert g.next_m() == 1.5
    g.record(False)  # 1 draft for 2 slots: dropped
    assert g.next_m() is None and g.dropped
    with pytest.raises(ValueError):
        NullPolicy(prompt_drop=True)  # needs a constant envelope


def test_envelope_guard_takes_counts_only():
    pol = NullPolicy(fixed_m=2.0)
    with pytest.raises(TypeError):
        pol.m(3.0, 2)  # e.g. a length or a content statistic passed as d
    with pytest.raises(TypeError):
        pol.m(np.int64(3), 2)


# --------------------------------------------------------------------------
# Drop-IS
# --------------------------------------------------------------------------


def _log_wstar(P, Q, x):
    s, best = 0.0, -math.inf
    for t in range(len(x)):
        s += math.log(P[tuple(x[:t])][x[t]]) - math.log(Q[tuple(x[:t])][x[t]])
        best = max(best, s)
    return best


def _whole_prob(P, Q, x, M, V, L):
    """P(draft x is whole-accepted | x) under ⊥-BV at envelope M (the kept event)."""
    from thundersync_rl.block_verification import output_distribution_given_draft

    lp = torch.tensor(np.log(np.stack([P[tuple(x[:t])] for t in range(L + 1)])), dtype=torch.float64)
    lq = torch.tensor(np.log(np.stack([Q[tuple(x[:t])] for t in range(L)])), dtype=torch.float64)
    return sum(float(pr.sum()) for tau, pr in output_distribution_given_draft(lp, lq, torch.tensor(x), M) if tau == L)


@pytest.mark.parametrize("M", [1.0, 1.5, 2.0])
def test_drop_is_weights_restore_p_per_draft(M):
    V, L = 3, 3
    P, Q = _model(V, L, 7), _model(V, L, 8)
    seqs = list(itertools.product(range(V), repeat=L))
    kept = {x: _seq_prob(Q, x) * _whole_prob(P, Q, x, M, V, L) for x in seqs}
    w = {x: max(1.0, math.exp(_log_wstar(P, Q, x)) / M) for x in seqs}
    for x in seqs:  # q(x) P(whole | x) max(1, W*/M) = p(x) / M exactly
        assert abs(kept[x] * w[x] - _seq_prob(P, x) / M) < 1e-12
    zk = sum(kept.values())
    tv = 0.5 * sum(abs(kept[x] / zk - _seq_prob(P, x)) for x in seqs)
    assert tv > 1e-3  # control: unweighted kept rows are not p


def test_drop_is_pooled_weighted_joint_is_p():
    """Pooled process with tail outputs discarded (NullGroup tail_dropped):
    the weighted joint law of the G kept rows (given the pool filled) is p x p."""
    V, L, N, G, M = 2, 2, 4, 2, 1.5
    P, Q = _model(V, L, 7, conc=0.5), _model(V, L, 8, conc=0.5)
    seqs = list(itertools.product(range(V), repeat=L))
    pol = NullPolicy(fixed_m=M, prompt_drop=True)
    # per draft: BOT, whole x (kept), or a tail output (dropped)
    row: dict = {}
    for x in seqs:
        qx = _seq_prob(Q, x)
        law = _draft_outcome_law(P, Q, x, M, V, L)
        whole = _whole_prob(P, Q, x, M, V, L)
        row[("keep", x)] = row.get(("keep", x), 0.0) + qx * whole
        row["BOT"] = row.get("BOT", 0.0) + qx * law.get("BOT", 0.0)
        row["tail"] = row.get("tail", 0.0) + qx * (1.0 - whole - law.get("BOT", 0.0))
    w = {x: max(1.0, math.exp(_log_wstar(P, Q, x)) / M) for x in seqs}
    joint_w: dict = {}
    joint: dict = {}
    for combo in itertools.product(list(row.items()), repeat=N):
        pr = math.prod(v for _, v in combo)
        grp = NullGroup(N, G, pol)
        outs = []
        for o, _ in combo:
            if grp.next_m() is None:
                break
            out = o != "BOT"
            grp.record(out, tail_dropped=o == "tail")
            if out and o != "tail":
                outs.append(o[1])
        grp.next_m()
        if grp.dropped:
            continue
        key = tuple(outs)
        joint[key] = joint.get(key, 0.0) + pr
        joint_w[key] = joint_w.get(key, 0.0) + pr * math.prod(w[x] for x in key)
    p1 = {z: _seq_prob(P, z) for z in seqs}
    zw, z0 = sum(joint_w.values()), sum(joint.values())
    tv_w = 0.5 * sum(abs(joint_w.get(k, 0) / zw - math.prod(p1[s] for s in k)) for k in itertools.product(seqs, repeat=G))
    tv_0 = 0.5 * sum(abs(joint.get(k, 0) / z0 - math.prod(p1[s] for s in k)) for k in itertools.product(seqs, repeat=G))
    assert tv_w < 1e-9, tv_w
    assert tv_0 > 1e-3, tv_0  # control: without weights


# --------------------------------------------------------------------------
# Tail-excluded baseline
# --------------------------------------------------------------------------


def _expected_gradient(estimator, G=3):
    """Exact E[sum_i A_i grad log p(y_i)] for a 3-way softmax policy with
    rewards r(y) and tail flags whose probability depends on y."""
    p = np.array([0.5, 0.3, 0.2])
    r = np.array([1.0, 0.0, 0.4])
    t = np.array([0.05, 0.6, 0.3])  # P(tail | y): content-dependent
    score = np.eye(3) - p  # grad of log p(y) w.r.t. logits
    total = np.zeros(3)
    for ys in itertools.product(range(3), repeat=G):
        py = math.prod(p[y] for y in ys)
        for fl in itertools.product((False, True), repeat=G):
            pf = math.prod(t[y] if f else 1 - t[y] for y, f in zip(ys, fl))
            adv = estimator([r[y] for y in ys], list(fl))
            for i, y in enumerate(ys):
                total += py * pf * adv[i] * score[y]
    target = G * sum(p[y] * r[y] * score[y] for y in range(3))
    return total, target


def _excluded(normalize):
    def est(rew, fl):
        out = []
        for i in range(len(rew)):
            others = [rew[j] for j in range(len(rew)) if j != i and not fl[j]]
            if not others:
                others = [rew[j] for j in range(len(rew)) if j != i]
            out.append(tail_excluded_advantages(rew[i], others, normalize))
        return out

    return est


def test_tail_excluded_baseline_is_unbiased():
    total, target = _expected_gradient(_excluded(False))
    assert np.abs(total - target).max() < 1e-12, (total, target)
    total, target = _expected_gradient(_excluded(True))
    cos = total @ target / (np.linalg.norm(total) * np.linalg.norm(target))
    assert cos > 1 - 1e-12 and total @ target > 0, cos


def test_tail_excluded_controls_are_biased():
    def grpo_on_subset(rew, fl):
        # GRPO's self-inclusive mean over the tail-free rows for tail-free rows
        # (all rows for tail rows): the formula depends on the row's own flag.
        nt = [x for x, f in zip(rew, fl) if not f]
        return [x - (sum(rew) / len(rew) if f or not nt else sum(nt) / len(nt)) for x, f in zip(rew, fl)]

    def drop_tail_rows(rew, fl):
        out = []
        for i, f in enumerate(fl):
            others = [rew[j] for j in range(len(rew)) if j != i and not fl[j]]
            out.append(0.0 if f or not others else rew[i] - sum(others) / len(others))
        return out

    for est in (grpo_on_subset, drop_tail_rows):
        total, target = _expected_gradient(est)
        cos = total @ target / (np.linalg.norm(total) * np.linalg.norm(target))
        assert cos < 1 - 1e-6, (est.__name__, cos)


def _traj(i, reward, tail):
    return Trajectory(group="g", index=i, reward=reward, payload=None, num_tokens=10, tail=tail)


def test_planner_dispatches_tail_free_rows_before_the_tail():
    fn = lambda r, others: tail_excluded_advantages(r, others, False)  # noqa: E731
    pl = StreamPlanner(
        lambda rs: [0.0] * len(rs), dp_size=1, max_chunk_trajectories=8, max_open_groups=0,
        micro_batch_size=8, equal_rank_microbatches=False, tail_excluded_baseline=fn,
    )
    pl.register_group("g", 4, tails=1)
    for i, r in enumerate([1.0, 0.0, 0.5]):
        pl.add(_traj(i, r, False))
    assert pl.has_work()
    d = pl.next_dispatch()
    adv = {
        t.index: a for cs in d.per_rank for c in cs for t, a in zip(c.trajectories, c.advantages)
    }
    assert sorted(adv) == [0, 1, 2]
    assert adv[0] == pytest.approx(1.0 - 0.25) and adv[1] == pytest.approx(-0.75)
    pl.add(_traj(3, 1.0, True))  # the tail returns: baseline = the 3 tail-free rows
    d = pl.next_dispatch()
    (chunk,) = [c for cs in d.per_rank for c in cs]
    assert chunk.advantages == [pytest.approx(1.0 - 0.5)]
    assert pl.all_done()


def test_prompt_selection_keeps_first_undropped_and_discards_spare_plans():
    from thundersync_rl.speculative import SpeculativeGeneration

    s = object.__new__(SpeculativeGeneration)
    s._null_all_slots = {(1,): 2, (2,): 2, (3,): 2, (4,): 2}  # batch order, 1 spare
    s.null_bv_spare_prompts = 1
    s.null_dropped_keys = {(2,)}
    s.plans = {(1,): ["a"], (3,): ["c"], (4,): ["d"]}
    s._select_prompts()
    assert s.null_selected_keys == [(1,), (3,), (4,)]
    assert set(s.plans) == {(1,), (3,), (4,)}
    s.null_dropped_keys = {(1,), (2,)}  # too few survive: a dropped prompt decodes fresh
    s.plans = {(3,): ["c"], (4,): ["d"]}
    s._select_prompts()
    assert s.null_selected_keys == [(3,), (4,), (1,)]
