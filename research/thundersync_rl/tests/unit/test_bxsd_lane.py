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
"""CPU tests: BX-SD balanced fork lane placement (thundersync_rl.bxsd_lane; FIRSTPRINCIPLES_A §10.1 (c)(iii)).

Layouts: Nano 4 nodes TP1 EP8 (16 DP ranks, 2 EP groups), Super / Ultra-like G 4 (EP 8 x 4, EP 32 x 4).
"""

import itertools
import random

import pytest

from thundersync_rl.bxsd_lane import (
    NANO_EP8_T1_MS,
    LaneLayout,
    LanePlacer,
    balance_share,
    curve,
    is_fork,
    l1_drafts,
    lane_finish_ms,
    window_tokens,
)

T = curve(NANO_EP8_T1_MS)


def nano():
    return LaneLayout.from_group_ids([0] * 8 + [8] * 8)


def test_layout_from_group_ids_nano():
    lay = nano()
    assert lay.groups == (tuple(range(8)), tuple(range(8, 16)))
    assert lay.lane_ranks == tuple(range(8)) and lay.other_ranks == tuple(range(8, 16))
    assert lay.dp_size == 16


def test_layout_interleaved_group_ids():
    # EP groups need not be contiguous in DP rank order; the lane is the group holding DP rank 0.
    lay = LaneLayout.from_group_ids([5, 9] * 8)
    assert lay.lane_ranks == tuple(range(0, 16, 2))
    assert lay.other_ranks == tuple(range(1, 16, 2))


def test_layout_rejects_single_group_and_bad_partitions():
    with pytest.raises(ValueError):
        LaneLayout.from_group_ids([0] * 8)  # G = 1: no lane possible without a layout change
    with pytest.raises(ValueError):
        LaneLayout(groups=((0, 1), (2,)))
    with pytest.raises(ValueError):
        LaneLayout(groups=((0, 1), (3, 4)))
    with pytest.raises(ValueError):
        LaneLayout(groups=((0, 1), (2, 3)), lane_groups=(2,))


def test_fork_rule_is_tau_below_half_at_k1():
    L = 8192
    w = window_tokens(L, 1)
    assert w == 4096
    for tau in (0, 15, 4095, 4096, 4097, 8000):
        assert is_fork(L - tau, w) == (tau < L - w)
    assert window_tokens(L, 2) == 2730 and window_tokens(L, 4) == 1638
    with pytest.raises(ValueError):
        window_tokens(L, 0)


def test_balance_without_forks_is_even():
    assert balance_share(1024, 0, nano(), 2600, 0, T) == 64


def brute(n_bulk, n_fork, lay, it_b, it_f, t):
    nl, no = len(lay.lane_ranks), len(lay.other_ranks)
    best = None
    for b in range(0, -(-n_bulk // nl) + 1):
        rest = max(n_bulk - b * nl, 0)
        g = max(it_b * t(rest / no) if rest else 0.0, lane_finish_ms(b, n_fork / nl, it_b, it_f, t))
        if best is None or g < best[0] - 1e-9:
            best = (g, b)
    return best[1]


@pytest.mark.parametrize("n_fork,it_b,it_f", [(47, 1442, 2388), (47, 2600, 4096), (8, 2600, 3000), (120, 1442, 5000),
                                              (47, 2600, 1200)])
def test_balance_is_the_argmin(n_fork, it_b, it_f):
    lay = nano()
    assert balance_share(1024, n_fork, lay, it_b, it_f, T) == brute(1024, n_fork, lay, it_b, it_f, T)


def test_balance_monotone_in_fork_depth():
    # Longer fork chains move bulk off the lane group (never onto it).
    lay = nano()
    bs = [balance_share(1024, 47, lay, 2600, it_f, T) for it_f in (1000, 2000, 2600, 3200, 4096, 6000)]
    assert all(a >= b for a, b in zip(bs, bs[1:])), bs
    assert bs[0] >= 50 and bs[-1] <= 10, bs


def test_fp_a_10_1_iii_reference_point():
    # FIRSTPRINCIPLES_A §10.1 (iii) (fp_a_s10b.py): with the module-at-target per-iteration curve the balance is
    # ~44 bulk rows per lane rank. Rebuild that curve: t_bulk(r) = (TW4(r) - 2.5 + 3 x 0.07 x T1(r)) x 7.8/7.54.
    tw4 = curve({8: 8.59, 12: 8.82, 16: 9.06, 18: 9.85, 26: 10.55, 32: 11.75, 52: 14.03, 64: 14.73, 85: 18.35,
                 96: 19.59, 128: 23.2})
    t_b = lambda r: (tw4(r) - 2.5 + 0.21 * T(r)) * 7.8 / 7.54
    b = balance_share(1024, 47, nano(), 1442, 2388, t_b)
    assert 40 <= b <= 48, b


def place_phase(lay, n_bulk, n_fork, it_b, it_f, order):
    p = LanePlacer(lay)
    b = p.begin_phase(n_bulk, n_fork, balance_share(n_bulk, n_fork, lay, it_b, it_f, T))
    out = [(k, p.place(k)) for k in order]
    return p, b, out


def test_placer_forks_on_lane_only_and_even():
    lay = nano()
    order = [True] * 47 + [False] * 1024
    random.Random(0).shuffle(order)
    p, b, out = place_phase(lay, 1024, 47, 2600, 2400, order)
    lane = set(lay.lane_ranks)
    assert all(r in lane for k, r in out if k)
    forks = [p.fork[r] for r in lay.lane_ranks]
    assert max(forks) - min(forks) <= 1 and sum(forks) == 47
    assert all(p.fork[r] == 0 for r in lay.other_ranks)


def test_placer_bulk_meets_quotas_exactly():
    lay = nano()
    order = [False] * 1024 + [True] * 47
    p, b, out = place_phase(lay, 1024, 47, 2600, 2400, order)
    assert sum(p.quota.values()) == 1024
    assert all(p.bulk[r] == p.quota[r] for r in range(16))
    assert all(p.quota[r] in (b, b - 1) or b * 8 > 1024 for r in lay.lane_ranks)
    other = [p.bulk[r] for r in lay.other_ranks]
    assert max(other) - min(other) <= 1


def test_placer_deterministic_and_order_only():
    lay = nano()
    order = [i % 23 == 0 for i in range(1071)]
    a = place_phase(lay, 1024, 47, 2600, 2400, order)[2]
    b = place_phase(lay, 1024, 47, 2600, 2400, order)[2]
    assert a == b


def test_placer_overflow_degrades_gracefully():
    lay = nano()
    p = LanePlacer(lay)
    p.begin_phase(160, 0, 0)  # announced 160 bulk rows, 10 per rank
    for _ in range(160 + 32):
        p.place(False)
    tot = [p.bulk[r] for r in range(16)]
    assert max(tot) - min(tot) <= 1 and sum(tot) == 192


def test_placer_before_any_phase_is_least_loaded():
    p = LanePlacer(nano())
    got = [p.place(False) for _ in range(32)]
    assert got[:16] == list(range(16)) and all(v == 2 for v in p.bulk.values())


def test_placer_live_counts_and_release():
    lay = nano()
    p = LanePlacer(lay)
    p.begin_phase(160, 8, balance_share(160, 8, lay, 100, 400, T))
    placed = [(True, p.place(True)) for _ in range(8)] + [(False, p.place(False)) for _ in range(160)]
    assert sum(p.fork.values()) == 8 and sum(p.bulk.values()) == 160
    for k, r in placed[:4] + placed[8:40]:
        p.release(r, k)
    assert sum(p.fork.values()) == 4 and sum(p.bulk.values()) == 128
    with pytest.raises(RuntimeError):
        p.release(15, True)  # no fork row ever lives off the lane
    # the next phase's quotas cover live + new rows
    p.begin_phase(32, 0, 0)
    assert sum(p.quota.values()) == 160 and p.phase == 1


def test_placer_new_phase_fills_toward_quota_given_residents():
    # Residents placed evenly before the wake; the new phase's fork rows push new bulk rows off the lane group.
    lay = nano()
    p = LanePlacer(lay)
    for _ in range(512):
        p.place(False)  # 32 per rank
    b = p.begin_phase(512, 47, balance_share(512, 47, lay, 4096, 8192, T))
    assert b < 64
    for _ in range(47):
        p.place(True)
    for _ in range(512):
        p.place(False)
    lane_bulk = sum(p.bulk[r] for r in lay.lane_ranks)
    other_bulk = sum(p.bulk[r] for r in lay.other_ranks)
    assert lane_bulk == sum(p.quota[r] for r in lay.lane_ranks) or lane_bulk == 8 * 32  # residents can exceed b
    assert other_bulk > lane_bulk


@pytest.mark.parametrize("ranks_per_group,G", [(8, 4), (32, 4)])
def test_placer_at_scale_one_lane_group(ranks_per_group, G):
    gids = [g for g in range(G) for _ in range(ranks_per_group)]
    lay = LaneLayout.from_group_ids(gids)
    n_bulk, n_fork = 2 * 512, 47
    order = [True] * n_fork + [False] * n_bulk
    p, b, out = place_phase(lay, n_bulk, n_fork, 2600, 2400, order)
    lane = set(lay.lane_ranks)
    assert len(lane) == ranks_per_group
    assert all(r in lane for k, r in out if k)
    assert sum(p.bulk.values()) == n_bulk


def test_l1_rule():
    # the critical row always drafts; others only if they would end after max(bulk_left, R_max / s_crit)
    assert l1_drafts([8000, 6000, 3000, 100], 2.0, 2000) == [True, True, False, False]
    assert l1_drafts([8000, 6000, 3000, 100], 2.0, 5000) == [True, True, False, False]
    assert l1_drafts([8000, 6000, 3000], 2.0, 6500) == [True, False, False]
    # unmeasured / underestimated s_crit: only the critical row drafts (never throttled)
    assert l1_drafts([8000, 7999], 1.0, 0) == [True, False]
    assert l1_drafts([5, 5], 1.0, 0) == [True, True]
    assert l1_drafts([], 2.0, 10) == []


def test_l1_never_slows_the_critical_chain():
    # simulate one rank: rows draft at s when on, 1 token/iter when off; the phase ends when every row is done.
    # With L1 the phase length equals always-drafting's (the max row's L / s) for any s_crit estimate <= s.
    import math

    rows0 = [8000, 7000, 5000, 3000, 1000]
    s = 2.0

    def run(rule):
        rem, it, drafted = list(rows0), 0, 0
        while any(r > 0 for r in rem):
            live = [i for i, r in enumerate(rem) if r > 0]
            on = rule([rem[i] for i in live], it)
            for i, d in zip(live, on):
                rem[i] -= s if d else 1
                drafted += int(d)
            it += 1
        return it, drafted

    full, d_full = run(lambda r, it: [True] * len(r))
    assert full == math.ceil(8000 / s)
    for s_est in (1.0, 1.5, 2.0):
        n, d = run(lambda r, it: l1_drafts(r, s_est, 0))
        assert n <= full + 1, (s_est, n, full)
        assert d < d_full



def _infos(world, tp, ep):
    """Megatron-style rank reports: TP groups of consecutive ranks; EP groups over DP ranks of the same TP rank,
    ep consecutive DP ranks each (inference DP = world / tp)."""
    out = []
    for g in range(world):
        dp, t = divmod(g, tp)
        tp_group = [dp * tp + i for i in range(tp)]
        e0 = (dp // ep) * ep
        ep_group = [(e0 + i) * tp + t for i in range(ep)]
        out.append({"global_rank": g, "dp_rank": dp, "lockstep": [ep_group, tp_group]})
    return out


@pytest.mark.parametrize("world,tp,ep,groups", [
    (16, 1, 8, 2),   # Nano 4 nodes TP1 EP8
    (4, 1, 2, 2),    # 1-node gate: Nano-L8 TP1 EP2
    (32, 2, 4, 4),   # Super-like TP2 EP4 on 8 nodes
    (128, 4, 8, 4),  # Ultra-like
])
def test_lockstep_groups(world, tp, ep, groups):
    from thundersync_rl.bxsd_lane import lockstep_groups

    gs = lockstep_groups(_infos(world, tp, ep))
    assert len(gs) == groups
    assert sorted(r for g in gs for r in g) == list(range(world // tp))
    assert all(len(g) == ep for g in gs)
    lay = LaneLayout(groups=gs)
    assert lay.lane_ranks == gs[0]


def test_lockstep_groups_single_group():
    from thundersync_rl.bxsd_lane import lockstep_groups

    assert len(lockstep_groups(_infos(4, 2, 2))) == 1  # TP2 EP2 on 4 GPUs: one group, no lane


def test_phase_placed_audit():
    lay = nano()
    p = LanePlacer(lay)
    p.begin_phase(64, 10, balance_share(64, 10, lay, 100, 400, T))
    for _ in range(10):
        p.place(True)
    for _ in range(64):
        p.place(False)
    assert sum(p.phase_placed[r][1] for r in lay.lane_ranks) == 10
    assert all(p.phase_placed[r][1] == 0 for r in lay.other_ranks)
    assert sum(v[0] for v in p.phase_placed.values()) == 64
    p.begin_phase(0, 0, 0)
    assert all(v == [0, 0] for v in p.phase_placed.values())


def _brute_lane(rem, n_free, lay, t, cap=None, node_cap=0, s_max=1.0, room_full=16.0):
    from thundersync_rl.bxsd_lane import group_ms

    r = sorted(rem, reverse=True)
    nl, no = len(lay.lane_ranks), len(lay.other_ranks)
    n_all = len(r) + n_free
    gm = lambda d, rows: group_ms(d, rows, t, cap, node_cap, s_max, room_full)
    best = (gm(r[0], n_all / lay.dp_size), 0, -(-n_all // lay.dp_size))
    for k in range(1, len(r) + 1):
        rn = r[k] if k < len(r) else 0
        for b in range(0, -(-(n_all - k) // nl) + 1):
            on_o = max(n_all - k - b * nl, 0)
            lane_rows = k / nl + min(b, (n_all - k) / nl)
            if on_o == 0 and rn > 0:
                continue
            if cap is not None and (on_o / no > cap or lane_rows > cap):
                continue
            g = max(gm(rn, on_o / no) if on_o else 0.0, gm(r[0], lane_rows))
            if g < best[0] - 1e-9:
                best = (g, k, b)
    return best


def test_choose_lane_plateau_gives_no_lane():
    # equally deep deadline rows and no free rows: nothing to separate, the even split is optimal
    from thundersync_rl.bxsd_lane import choose_lane

    k, b, g = choose_lane([8000] * 480, 0, nano(), T)
    assert k == 0 and g == pytest.approx(8000 * T(30))
    k, b, g = choose_lane([8000] * 480, 0, nano(), T, row_cap=96, node_cap=48, s_max=2.5)
    assert k == 0


def test_choose_lane_separates_deadline_rows_from_drafts():
    # a plateau of deadline rows plus drafts (no deadline, load only): the split may put deadline rows on the lighter
    # group; it never predicts worse than the even split
    from thundersync_rl.bxsd_lane import choose_lane

    rem = [8100 - (i % 40) * 50 for i in range(468)]
    k, b, g = choose_lane(rem, 512, nano(), T)
    assert g <= max(rem) * T((468 + 512) / 16) + 1e-9


def test_choose_lane_few_deep_forks_over_shallow_bulk():
    # even steps: ~60 rejected rows with 4.5-8k left, ~460 accepted-but-cut rows with <= 2k left, 512 drafts
    from thundersync_rl.bxsd_lane import choose_lane

    rem = [8000 - 60 * i for i in range(60)] + [2000 - 4 * i for i in range(460)]
    k, b, g = choose_lane(rem, 512, nano(), T)
    assert 1 <= k <= 60, k
    even = max(rem) * T((len(rem) + 512) / 16)
    assert g < 0.9 * even


@pytest.mark.parametrize("seed", range(6))
def test_choose_lane_is_the_argmin(seed):
    from thundersync_rl.bxsd_lane import choose_lane

    rng = random.Random(seed)
    rem = [rng.choice([rng.randint(4000, 8192), rng.randint(1, 3000)]) for _ in range(rng.randint(20, 200))]
    nf = rng.randint(0, 300)
    for kw in ({}, dict(cap=96, node_cap=48, s_max=2.5)):
        ck = {"row_cap": kw["cap"], "node_cap": kw["node_cap"], "s_max": kw["s_max"]} if kw else {}
        k, b, g = choose_lane(rem, nf, nano(), T, **ck)
        gb, kb, bb = _brute_lane(rem, nf, nano(), T, **kw)
        assert abs(g - gb) <= 1e-6 * max(gb, 1.0), (kw, k, b, g, kb, bb, gb)


def test_choose_lane_respects_max_requests_and_drafting_room():
    # the measured 4-node pair-2 regime: ~470 continuations 1-8k deep, ~512 drafts, max_requests 96,
    # tail_tree_rows 48, critical tok/iter 2.5 with room: no split may put > 96 rows on a rank, and a split that
    # starves the other group's drafting room must lose to the even split unless the lane chain dominates
    from thundersync_rl.bxsd_lane import choose_lane, group_ms

    rem = [8189 - 30 * i for i in range(100)] + [5300 - 10 * i for i in range(370)]
    k, b, g = choose_lane(rem, 512, nano(), T, row_cap=96, node_cap=48, s_max=2.5)
    n_all = len(rem) + 512
    assert (n_all - k - b * 8) / 8 <= 96 and k / 8 + b <= 96
    even = group_ms(max(rem), n_all / 16, T, 96, 48, 2.5, 16.0)
    assert g <= even + 1e-9
