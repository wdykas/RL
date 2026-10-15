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
"""CPU tests: dealing dispatch rows over every DP rank (StreamPlanner deal_rows, DESIGN 14w).

Shapes: full Nano (learner DP 8, EP on, MBS 1, G 8, canonical blocks of 4 / 2 / 1 groups),
Nano L8 (DP 2), the 1T layout (DP 8, G 16) and a large-DP layout (DP 64).
"""

import math
import random
from types import SimpleNamespace

import pytest
import torch
from test_streaming_equivalence import _adv_fn, _simulate
from thundersync_rl.grpo_loop import canonical_groups_per_block
from thundersync_rl.streaming import (
    StreamPlanner,
    Trajectory,
    cut_units,
    microbatch_counts,
    num_units,
)


def _planner(dp, mbs, G, groups, deal, equal=True, chunk=8, skip_zero=False, adv=None):
    pl = StreamPlanner(
        adv or (lambda rs: [r - sum(rs) / len(rs) for r in rs]),
        dp_size=dp,
        max_chunk_trajectories=chunk,
        max_open_groups=0,
        micro_batch_size=mbs,
        equal_rank_microbatches=equal,
        skip_zero_advantage_chunks=skip_zero,
        deal_rows=deal,
    )
    for g in range(groups):
        pl.register_group(g, G)
    return pl


def _rows(groups, G, seed, max_tokens=8192):
    rng = random.Random(seed)
    out = []
    for g in range(groups):
        for j in range(G):
            out.append(
                Trajectory(
                    group=g,
                    index=g * G + j,
                    reward=float(rng.random() < 0.4),
                    payload=None,
                    num_tokens=rng.randint(50, max_tokens),
                )
            )
    return out


def _run_canonical(dp, mbs, G, groups, bg, deal, seed, arrival_seed=0, **kw):
    """All rows of ``groups`` groups through a canonical order in blocks of ``bg``."""
    pl = _planner(dp, mbs, G, groups, deal, **kw)
    sizes = [min(bg, groups - j) for j in range(0, groups, bg)]
    pl.set_order(list(range(groups)), sizes)
    rows = _rows(groups, G, seed)
    random.Random(arrival_seed).shuffle(rows)
    out = []
    for t in rows:
        pl.add(t)
        while pl.has_work(flush=False):
            out.append(pl.next_dispatch(flush=False))
    while pl.has_work():
        out.append(pl.next_dispatch())
    assert pl.all_done()
    return out


def _per_rank_mbs(d, mbs):
    return [sum(c.num_microbatches(mbs) for c in cs) for cs in d.per_rank]


@pytest.mark.parametrize("bg", [1, 2, 4])
@pytest.mark.parametrize("seed", range(3))
def test_full_nano_dp8_no_fillers(bg, seed):
    """DP 8, EP, MBS 1, G 8: every rank gets rows/8 micro-batches and no filler."""
    dp, mbs, G, groups = 8, 1, 8, 16
    for d in _run_canonical(dp, mbs, G, groups, bg, True, seed):
        mb = microbatch_counts(d, mbs)
        rows = d.num_trajectories
        assert mb["filler"] == 0, (rows, mb)
        assert mb["real"] == rows
        assert _per_rank_mbs(d, mbs) == [math.ceil(rows / dp)] * dp


@pytest.mark.parametrize("bg", [1, 2, 4])
def test_full_nano_dp8_old_layout_pads(bg):
    """The whole-chunk layout this replaces: 1 - bg/DP of each rank's forwards are fillers."""
    dp, mbs, G = 8, 1, 8
    for d in _run_canonical(dp, mbs, G, 8, bg, False, 0):
        mb = microbatch_counts(d, mbs)
        assert mb["filler"] == dp * mb["per_rank"] - d.num_trajectories
        assert mb["filler"] >= dp * mb["per_rank"] // 2  # at least half the forwards


@pytest.mark.parametrize("seed", range(3))
def test_l8_dp2_unchanged_counts(seed):
    """Nano L8 (DP 2, blocks of 4 groups): 16 micro-batches per rank, no filler, as before."""
    for deal in (False, True):
        for d in _run_canonical(2, 1, 8, 8, 4, deal, seed):
            assert _per_rank_mbs(d, 1) == [16, 16]
            assert microbatch_counts(d, 1)["filler"] == 0


@pytest.mark.parametrize("mbs", [1, 2, 4, 8])
@pytest.mark.parametrize("rows", [1, 3, 8, 12, 20, 32, 63, 504])
@pytest.mark.parametrize("dp", [2, 8, 12, 64])
def test_one_dispatch_minimal_microbatches(mbs, rows, dp):
    """Any row count: per-rank micro-batches = ceil(ceil(rows/MBS)/DP), the minimum."""
    pl = _planner(dp, mbs, rows, 1, True)
    for t in _rows(1, rows, rows):
        pl.add(t)
    d = pl.next_dispatch()
    mb = microbatch_counts(d, mbs)
    want = math.ceil(math.ceil(rows / mbs) / dp)
    assert mb["per_rank"] == want
    assert len(set(_per_rank_mbs(d, mbs))) == 1
    # Fillers only where the rows cannot fill whole rounds of DP units (rows < DP * want):
    # always fewer than DP, none when rows is a multiple of DP or rows >= DP * want.
    assert mb["filler"] == dp * want - num_units(rows, dp, mbs)
    assert mb["filler"] < dp
    if rows % dp == 0 or rows >= dp * want:
        assert mb["filler"] == 0
    # Every unit is one micro-batch: no dummy rows and no chunk above the cap.
    for cs in d.per_rank:
        for c in cs:
            assert 1 <= len(c.trajectories) <= mbs
    assert d.num_trajectories == rows


def test_async1_leftover_dispatch():
    """async1's 8-row leftover dispatch at DP 8: one row per rank instead of 7 filler ranks."""
    pl = _planner(8, 1, 8, 1, True)
    for t in _rows(1, 8, 1):
        pl.add(t)
    d = pl.next_dispatch()
    assert microbatch_counts(d, 1) == {
        "real": 8,
        "filler": 0,
        "per_rank": 1,
        "max_rank_tokens": max(
            t.num_tokens for cs in d.per_rank for c in cs for t in c.trajectories
        ),
    }


@pytest.mark.parametrize("seed", range(3))
def test_units_are_length_contiguous_and_tokens_balanced(seed):
    """Units hold consecutive rows in length order (padding to their own longest row);
    within equal micro-batch counts ranks are balanced by tokens (LPT)."""
    dp, mbs, G = 8, 2, 8
    pl = _planner(dp, mbs, G, 8, True)
    rows = _rows(8, G, seed)
    for t in rows:
        pl.add(t)
    d = pl.next_dispatch()
    order = sorted(rows, key=lambda t: t.num_tokens)
    pos = {id(t): i for i, t in enumerate(order)}
    for cs in d.per_rank:
        for c in cs:
            p = sorted(pos[id(t)] for t in c.trajectories)
            assert p == list(range(p[0], p[0] + len(p)))
    loads = [sum(c.num_tokens for c in cs) for cs in d.per_rank]
    longest_unit = max(c.num_tokens for cs in d.per_rank for c in cs)
    assert max(loads) - min(loads) <= longest_unit


@pytest.mark.parametrize("bg", [1, 4])
@pytest.mark.parametrize("dp", [2, 8])
@pytest.mark.parametrize("seed", range(3))
def test_canonical_dispatch_independent_of_arrival_order(bg, dp, seed):
    """Same rows, other arrival orders: identical dispatches (rows, units, ranks, fillers)."""

    def sig(ds):
        return [
            tuple(
                tuple((c.padding, tuple(t.index for t in c.trajectories)) for c in cs)
                for cs in d.per_rank
            )
            for d in ds
        ]

    ref = sig(_run_canonical(dp, 1, 8, 12, bg, True, seed, arrival_seed=0))
    for a in range(1, 5):
        assert sig(_run_canonical(dp, 1, 8, 12, bg, True, seed, arrival_seed=a)) == ref


@pytest.mark.parametrize("dp_size", [2, 3, 8])
@pytest.mark.parametrize("mbs", [1, 2, 4])
@pytest.mark.parametrize("skip_zero", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_dealt_gradient_exact(dp_size, mbs, skip_zero, seed):
    """sum_i a_i H_i over ranks equals the direct sum (fillers add exactly 0)."""
    rng = random.Random(seed)
    rewards = [[float(rng.random() < 0.4) for _ in range(8)] for _ in range(6)]
    got, want, _, dispatches = _simulate(
        rewards,
        _adv_fn(True),
        dp_size,
        2,
        seed,
        chunk=8,
        group_only=True,
        mbs=mbs,
        equal_rank_microbatches=not skip_zero,
        skip_zero_advantage_chunks=skip_zero,
        deal_rows=True,
    )
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-12, atol=1e-12)
    assert sum(d.num_trajectories for d in dispatches) == 6 * 8
    if skip_zero:
        # Zero-advantage rows never share a unit with trained rows.
        for d in dispatches:
            for cs in d.per_rank:
                for c in cs:
                    assert len({a == 0.0 for a in c.advantages}) == 1


@pytest.mark.parametrize("seed", range(3))
def test_final_microbatches_mirrors_dispatch(seed):
    """The pipeline-hold count (final_microbatches) equals what next_dispatch builds."""
    for skip_zero in (False, True):
        for dp, mbs in ((8, 1), (8, 4), (3, 2)):
            pl = _planner(dp, mbs, 8, 4, True, equal=not skip_zero, skip_zero=skip_zero)
            rows = _rows(4, 8, seed)
            for t in rows:
                pl.add(t)
            want = pl.final_microbatches(list(pl.ready))
            d = pl.next_dispatch()
            # Skipped all-zero-advantage units run nothing and count 0.
            got = sum(
                c.num_microbatches(mbs)
                for cs in d.per_rank
                for c in cs
                if not c.padding and not (skip_zero and c.zero_advantage)
            )
            assert got == want


def test_num_units_and_cut_units():
    assert num_units(0, 8, 1) == 0
    assert num_units(32, 8, 1) == 32
    assert num_units(20, 8, 4) == 8  # 5 units of 4 rounded up to 8 units of 2-3 rows
    assert num_units(5, 8, 4) == 5  # never more units than rows
    units = cut_units(list(range(20)), 8, 4)
    assert [len(u) for u in units] == [3, 3, 3, 3, 2, 2, 2, 2]
    assert [x for u in units for x in u] == list(range(20))


@pytest.mark.parametrize(
    "dp,G,bg,want",
    [
        (2, 8, 4, 4),  # Nano L8
        (8, 8, 4, 4),  # full Nano
        (8, 16, 4, 4),  # 1T: TP8 PP8 on 512 GPUs -> DP 8, G 16
        (64, 16, 4, 4),  # 64 rows per block reach 64 ranks
        (128, 8, 4, 16),  # large DP: blocks grow to one row per rank
        (64, 8, 1, 8),
    ],
)
def test_canonical_groups_per_block_scales_with_dp(dp, G, bg, want):
    cfg = SimpleNamespace(canonical_block_groups=bg, deal_rows=True)
    assert canonical_groups_per_block(cfg, dp, G) == want
    assert want * G >= dp
    off = SimpleNamespace(canonical_block_groups=bg, deal_rows=False)
    assert canonical_groups_per_block(off, dp, G) == bg


@pytest.mark.parametrize("dp,G", [(8, 16), (64, 16), (128, 8)])
def test_1t_shapes_no_fillers(dp, G):
    """1T-shaped canonical blocks (block size from canonical_groups_per_block): no fillers."""
    cfg = SimpleNamespace(canonical_block_groups=4, deal_rows=True)
    bg = canonical_groups_per_block(cfg, dp, G)
    for d in _run_canonical(dp, 1, G, 2 * bg, bg, True, 0):
        mb = microbatch_counts(d, 1)
        assert mb["filler"] == 0
        assert mb["per_rank"] == math.ceil(d.num_trajectories / dp)


@pytest.mark.parametrize(
    "dp,G,mbs,pp,want",
    [
        (
            8,
            16,
            1,
            1,
            4,
        ),  # 1T without the pipeline term: 64 rows, 8 micro-batches per rank
        (
            8,
            16,
            1,
            8,
            8,
        ),  # 1T PP 8: 2 * PP = 16 micro-batches per rank -> 128-row blocks
        (8, 8, 1, 2, 4),  # PP 2: 4 per rank is already met by 4 groups of 8 rows
        (2, 8, 8, 4, 16),  # PP 4, MBS 8, DP 2: 8 micro-batches per rank need 128 rows
    ],
)
def test_canonical_groups_per_block_fills_the_pipeline(dp, G, mbs, pp, want):
    """Under PP every rank's one pipeline call per dispatch gets >= 2 * PP micro-batches."""
    cfg = SimpleNamespace(canonical_block_groups=4, deal_rows=True)
    min_pmb = 2 * pp if pp > 1 else 0
    bg = canonical_groups_per_block(cfg, dp, G, mbs, min_pmb)
    assert bg == want
    pl = _planner(dp, mbs, G, bg, True)
    for t in _rows(bg, G, 0):
        pl.add(t)
    d = pl.next_dispatch()
    assert microbatch_counts(d, mbs)["per_rank"] >= min_pmb
    assert microbatch_counts(d, mbs)["filler"] == 0


def test_deal_places_pinned_chunks_and_keeps_counts_level():
    """Fused (pinned) chunks count toward their rank's load before the units are dealt."""
    from thundersync_rl.fused_train import FusedRef
    from thundersync_rl.streaming import Chunk, deal_chunks

    def chunk(n, tok, rank=None):
        ts = [
            Trajectory(group=0, index=i, reward=1.0, payload=None, num_tokens=tok)
            for i in range(n)
        ]
        fused = (
            None
            if rank is None
            else FusedRef(key=f"k{rank}", dp_rank=rank, num_tokens=tok)
        )
        return Chunk(
            group=None,
            reward=None,
            trajectories=ts,
            advantages=[1.0] * n,
            fused=fused,
            padding=False,
        )

    pinned = [chunk(1, 100, 0), chunk(1, 100, 0)]
    units = [chunk(1, 10 + i) for i in range(6)]
    per_rank = deal_chunks(
        pinned + units,
        4,
        skip_zero_advantage=False,
        micro_batch_size=1,
        pinned_rank=lambda c: None if c.fused is None else c.fused.dp_rank,
    )
    counts = [len(cs) for cs in per_rank]
    assert counts == [2, 2, 2, 2]
