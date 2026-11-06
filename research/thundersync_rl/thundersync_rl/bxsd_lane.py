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
"""BX-SD balanced fork lane (FIRSTPRINCIPLES_A §10.1 (c)(iii)): row placement and the L1 drafting rule.

Pure Python (no torch / Megatron): the placement decisions are a function of row counts and row progress only.

Design
- The engine's DP ranks form G lockstep EP groups (one group's ranks step together; groups are independent). F of
  them are LANE groups (F = 1 here). With plain BI the lane must be a group of the bulk layout (FP_B_MEASURE §2.5),
  so its logits are bitwise those of any other group: placement changes which GPU computes a row, never its value.
- At a wake every continuation is a new engine request (it is re-prefilled under the new weights anyway), so
  placement costs nothing extra (no new collective, no KV move).
- A FORK row is a continuation whose remaining serial depth exceeds the bulk window (remaining tokens > window
  tokens; window = L/(K+1), i.e. tau < L/2 at K = 1). Fork rows go to the lane groups.
- Bulk rows fill a per-rank QUOTA: the lane ranks carry b bulk rows each, the other ranks the rest. b balances the
  groups' predicted finish times (the lane group runs min(it_b, it_f) iterations at t(b + f), then fork-only
  iterations at t(f); the other group it_b iterations at t(its bulk share)). Exactness does not depend on b.
- L1: a lane row drafts only while it is behind schedule: remaining tokens > s_ref * (horizon - iterations done).
  It reads the row's own progress only; drafting changes which forwards run, never what is sampled.

Per-rank cost: O(rows) host work per phase on the rank that builds requests; one int per request (the target DP
rank) in the request's sampling params; nothing on the GPU; nothing grows with TP / EP / PP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

# FP_B_MEASURE §2.4 [M]: BI 1-token wall ms per iteration vs rows per rank, Nano TP1 EP8 (random weights, graphs).
# Only used for the balance; any monotone curve works (exactness never depends on it).
NANO_EP8_T1_MS = {8: 6.14, 12: 6.30, 16: 6.39, 18: 6.48, 26: 6.73, 32: 6.85, 52: 7.34, 64: 7.54, 85: 9.15, 96: 9.42,
                  128: 10.5}


def curve(points: dict[int, float]) -> Callable[[float], float]:
    """Piecewise-linear t(rows/rank) through measured points; linear extrapolation past the last two."""
    xs = sorted(points)
    if len(xs) < 2:
        raise ValueError("a timing curve needs >= 2 points")

    def t(x: float) -> float:
        if x <= xs[0]:
            return points[xs[0]] * (0.97 + 0.03 * max(x, 0.0) / xs[0])
        for a, b in zip(xs, xs[1:]):
            if x <= b:
                return points[a] + (points[b] - points[a]) * (x - a) / (b - a)
        a, b = xs[-2], xs[-1]
        return points[b] + (points[b] - points[a]) * (x - b) / (b - a)

    return t


@dataclass(frozen=True)
class LaneLayout:
    """DP ranks of each lockstep EP group, and which groups are lanes."""

    groups: tuple[tuple[int, ...], ...]
    lane_groups: tuple[int, ...] = (0,)

    def __post_init__(self):
        ranks = [r for g in self.groups for r in g]
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError(f"groups must partition DP ranks 0..{len(ranks) - 1}: {self.groups}")
        if len({len(g) for g in self.groups}) != 1:
            raise ValueError(f"EP groups must have equal size: {[len(g) for g in self.groups]}")
        if not self.lane_groups or any(not 0 <= i < len(self.groups) for i in self.lane_groups):
            raise ValueError(f"lane_groups {self.lane_groups} out of range for {len(self.groups)} groups")
        if len(set(self.lane_groups)) != len(self.lane_groups):
            raise ValueError(f"duplicate lane groups {self.lane_groups}")

    @property
    def dp_size(self) -> int:
        return sum(len(g) for g in self.groups)

    @property
    def lane_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(r for i in self.lane_groups for r in self.groups[i]))

    @property
    def other_ranks(self) -> tuple[int, ...]:
        lane = set(self.lane_ranks)
        return tuple(r for r in range(self.dp_size) if r not in lane)

    @classmethod
    def from_group_ids(cls, group_of_rank: Sequence[int], n_lane: int = 1) -> "LaneLayout":
        """Layout from each DP rank's EP-group id (e.g. the min global rank of its expert group, gathered at install).
        Groups are ordered by their smallest DP rank; the first ``n_lane`` are the lanes (a fixed, rank-only rule)."""
        by_id: dict[int, list[int]] = {}
        for dp_rank, gid in enumerate(group_of_rank):
            by_id.setdefault(int(gid), []).append(dp_rank)
        groups = tuple(sorted((tuple(v) for v in by_id.values()), key=lambda g: g[0]))
        if not 1 <= n_lane < len(groups):
            raise ValueError(f"need 1 <= n_lane < G, got n_lane={n_lane}, G={len(groups)}: a lane needs >= 2 EP groups")
        return cls(groups=groups, lane_groups=tuple(range(n_lane)))


def lockstep_groups(infos: Sequence[dict]) -> tuple[tuple[int, ...], ...]:
    """DP ranks of each lockstep engine group from per-rank reports {"global_rank", "dp_rank", "lockstep": [lists of
    global ranks this rank runs collectives with (EP group, model-parallel group)]}: the connected components of
    "steps with", mapped to DP ranks. Groups are ordered by their smallest DP rank."""
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for info in infos:
        g = int(info["global_rank"])
        find(g)
        for ranks in info["lockstep"]:
            for r in ranks:
                a, b = find(g), find(int(r))
                if a != b:
                    parent[max(a, b)] = min(a, b)
    comp_of_dp: dict[int, int] = {}
    for info in infos:
        dp, c = int(info["dp_rank"]), find(int(info["global_rank"]))
        if comp_of_dp.setdefault(dp, c) != c:
            raise ValueError(f"DP rank {dp} spans two lockstep groups")
    by_comp: dict[int, list[int]] = {}
    for dp, c in sorted(comp_of_dp.items()):
        by_comp.setdefault(c, []).append(dp)
    return tuple(sorted((tuple(v) for v in by_comp.values()), key=lambda g: g[0]))


def is_fork(remaining_tokens: int, window_tokens: int) -> bool:
    """A continuation whose remaining serial depth exceeds the bulk window (tau < L/2 at K = 1)."""
    return int(remaining_tokens) > int(window_tokens)


def window_tokens(max_seq_tokens: int, lag: int) -> int:
    """Bulk window of a balanced lag-K schedule: L / (K + 1) tokens per phase."""
    if lag < 1:
        raise ValueError(f"lag must be >= 1, got {lag}")
    return int(max_seq_tokens) // (int(lag) + 1)


def lane_finish_ms(b: float, f: float, it_bulk: float, it_fork: float, t: Callable[[float], float]) -> float:
    """Lane rank: b bulk + f fork rows. Mixed iterations until the shorter class ends, then the longer class alone."""
    if f <= 0:
        return it_bulk * t(b) if b > 0 else 0.0
    if b <= 0:
        return it_fork * t(f)
    mixed = min(it_bulk, it_fork) * t(b + f)
    if it_fork >= it_bulk:
        return mixed + (it_fork - it_bulk) * t(f)
    return mixed + (it_bulk - it_fork) * t(b)


def balance_share(
    n_bulk: int,
    n_fork: int,
    layout: LaneLayout,
    it_bulk: float,
    it_fork: float,
    t: Callable[[float], float],
) -> int:
    """Bulk rows per lane rank (integer b) that minimises the later of the two groups' predicted finish times.

    Ties go to the smaller b (fewer rows sharing the fork rows' iterations). With no fork rows the result is the even
    share (every group alike)."""
    nl, no = len(layout.lane_ranks), len(layout.other_ranks)
    if n_bulk < 0 or n_fork < 0:
        raise ValueError("row counts must be >= 0")
    if n_fork == 0:
        return -(-n_bulk // layout.dp_size)  # ceil: the even share
    f = n_fork / nl
    best_b, best_g = 0, None
    b_max = -(-n_bulk // nl)
    for b in range(0, b_max + 1):
        rest = max(n_bulk - b * nl, 0)
        other = it_bulk * t(rest / no) if rest > 0 else 0.0
        g = max(other, lane_finish_ms(b, f, it_bulk, it_fork, t))
        if best_g is None or g < best_g - 1e-9:
            best_b, best_g = b, g
    return best_b


def group_ms(depth: float, rows: float, t: Callable[[float], float], row_cap: Optional[int], node_cap: int,
             s_max: float, room_full: float) -> float:
    """Predicted ms for a lockstep group whose deepest deadline row has ``depth`` tokens left, at ``rows`` rows per
    rank. tree_spec lays out node rows in the room the rank's rows leave under max_requests (``row_cap``), at most
    ``node_cap``; the critical tail reaches ``s_max`` tokens per iteration once it has ``room_full`` node rows and
    falls linearly to 1 with no room (measured: the 4-node a measured run lane arm, other group at 108 rows / rank over
    max_requests 96 -> 6-8 nodes per forward and critical tok/iter 1.3-1.5 vs 2.5 at an even 64). Iteration time =
    t(rows + nodes)."""
    if depth <= 0:
        return 0.0
    room = float(node_cap) if row_cap is None else max(0.0, min(float(node_cap), row_cap - rows))
    s = 1.0 + (max(s_max, 1.0) - 1.0) * min(1.0, room / max(room_full, 1e-9))
    return depth / s * t(rows + room)


def choose_lane(
    remaining: Sequence[int],
    n_free: int,
    layout: LaneLayout,
    t: Callable[[float], float],
    row_cap: Optional[int] = None,
    node_cap: int = 0,
    s_max: float = 1.0,
    room_full: float = 16.0,
) -> tuple[int, int, float]:
    """Which rows go to the lane, and how much bulk shares it: the makespan-optimal split of one phase.

    ``remaining``: the phase's deadline rows (continuations of the batch that trains after this phase) by remaining
    tokens; ``n_free``: rows without a deadline (the next batch's drafts, cut at the stand-down) that only add load.
    The lane takes the k deepest deadline rows plus b bulk rows per lane rank; the other group the rest.
    - Hard capacity: no rank holds more than ``row_cap`` (the engine's max_requests) rows; a split that would is
      infeasible (rows past max_requests wait in the engine, undecoded).
    - Drafting room: each group's critical tail speed falls with the node room its rows leave (``group_ms``).
    - lane = max(group_ms(r_1, lane rows), group_ms(r_(k+1) bulk on the lane)); other = group_ms(r_(k+1), other rows).
    k = 0 is "no lane" (an even split). Returns (k, b, predicted ms) minimising max(lane, other); ties to the smaller k,
    then the smaller b. Reads row counts and remaining budgets only (placement: exact by construction)."""
    r = sorted((int(x) for x in remaining), reverse=True)
    nl, no = len(layout.lane_ranks), len(layout.other_ranks)
    n_all = len(r) + int(n_free)
    even = n_all / layout.dp_size
    if not r:
        return 0, -(-n_all // layout.dp_size), 0.0

    def gm(depth, rows):
        return group_ms(depth, rows, t, row_cap, node_cap, s_max, room_full)

    best = (gm(r[0], even), 0, -(-n_all // layout.dp_size))  # k = 0: even split, deepest row
    for k in range(1, len(r) + 1):
        r_next = r[k] if k < len(r) else 0
        rest = n_all - k
        for b in range(0, -(-rest // nl) + 1):
            on_other = max(rest - b * nl, 0)
            lane_rows = k / nl + min(b, rest / nl)
            if on_other == 0 and r_next > 0:
                continue  # the deadline rows past k need the other group
            if row_cap is not None and (on_other / no > row_cap or lane_rows > row_cap):
                continue  # over max_requests on a rank: those rows would wait undecoded
            other = gm(r_next, on_other / no) if on_other > 0 else 0.0
            lane = gm(r[0], lane_rows)
            g = max(lane, other)
            if g < best[0] - 1e-9:
                best = (g, k, b)
    g, k, b = best
    return k, b, g


@dataclass
class LanePlacer:
    """Placement on the request-building engine rank, over LIVE rows (placed and not yet finished).

    - Fork rows go to the lane ranks: the fewest live fork rows first (ties: fewest live rows, lowest rank).
    - Bulk rows go to the rank with the largest remaining quota (quota - live bulk rows); past every quota, to the
      rank with the fewest live rows. Before the first phase (no quotas) bulk rows go to the fewest live rows.
    - ``begin_phase`` (at a wake, before the phase's continuations are submitted) sets quotas on the phase's
      expected live bulk total: the lane ranks b each (``balance_share``), the other ranks the even split of the rest.
    Placement reads row counts and the fork bit only. Live counts move with finishes (engine timing), which may change
    WHICH replica runs a row, never its tokens: every EP group of the bulk layout is the same BI forward."""

    layout: LaneLayout
    row_cap: Optional[int] = None  # the engine's max_requests per rank: never place a row past it while a rank has room
    quota: dict[int, int] = field(default_factory=dict)
    bulk: dict[int, int] = field(default_factory=dict)
    fork: dict[int, int] = field(default_factory=dict)
    b: int = 0
    phase: int = -1
    placed: int = 0
    released: int = 0
    # rows placed since the last begin_phase, per DP rank: (bulk, fork) -- the audit of where a phase's rows went
    phase_placed: dict[int, list[int]] = field(default_factory=dict)

    def __post_init__(self):
        n = self.layout.dp_size
        self.bulk = {r: 0 for r in range(n)}
        self.fork = {r: 0 for r in range(n)}
        self.phase_placed = {r: [0, 0] for r in range(n)}

    def begin_phase(self, n_bulk_new: int, n_fork_new: int, b: int) -> int:
        """Quotas for a phase that adds ``n_bulk_new`` bulk and ``n_fork_new`` lane rows to the live ones; ``b`` bulk
        rows per lane rank (``choose_lane``; with no lane rows the split is even whatever b)."""
        n_bulk = sum(self.bulk.values()) + int(n_bulk_new)
        n_fork = sum(self.fork.values()) + int(n_fork_new)
        lane, other = self.layout.lane_ranks, self.layout.other_ranks
        if n_fork == 0:
            # No lane rows: no lane this phase -- an even split over every rank.
            self.b = -(-n_bulk // self.layout.dp_size)
            per, extra = divmod(n_bulk, self.layout.dp_size)
            self.quota = {r: per + (1 if r < extra else 0) for r in range(self.layout.dp_size)}
        else:
            self.b = max(0, int(b))
            lane_total = min(self.b * len(lane), n_bulk)
            per_l, extra_l = divmod(lane_total, len(lane))
            per_o, extra_o = divmod(n_bulk - lane_total, len(other))
            self.quota = {r: per_l + (1 if i < extra_l else 0) for i, r in enumerate(lane)}
            self.quota.update({r: per_o + (1 if i < extra_o else 0) for i, r in enumerate(other)})
        self.phase += 1
        self.phase_placed = {r: [0, 0] for r in range(self.layout.dp_size)}
        return self.b

    def _full(self, r: int) -> bool:
        return self.row_cap is not None and self.bulk[r] + self.fork[r] >= self.row_cap

    def place(self, fork: bool) -> int:
        if fork:
            r = min(self.layout.lane_ranks, key=lambda r: (self._full(r), self.fork[r], self.bulk[r] + self.fork[r], r))
            self.fork[r] += 1
        elif self.quota:
            r = min(range(self.layout.dp_size),
                    key=lambda r: (self._full(r), -(self.quota[r] - self.bulk[r]), self.bulk[r] + self.fork[r], r))
            self.bulk[r] += 1
        else:
            r = min(range(self.layout.dp_size), key=lambda r: (self.bulk[r] + self.fork[r], r))
            self.bulk[r] += 1
        self.placed += 1
        self.phase_placed[r][1 if fork else 0] += 1
        return r

    def release(self, rank: int, fork: bool) -> None:
        """A placed row finished (its result came back)."""
        counts = self.fork if fork else self.bulk
        if counts.get(int(rank), 0) <= 0:
            raise RuntimeError(f"LanePlacer.release: no live {'fork' if fork else 'bulk'} row on DP rank {rank}")
        counts[int(rank)] -= 1
        self.released += 1

    def loads(self) -> dict[int, tuple[int, int]]:
        """Per DP rank (live bulk, live fork) rows."""
        return {r: (self.bulk[r], self.fork[r]) for r in range(self.layout.dp_size)}


def l1_drafts(remaining: Sequence[int], s_crit: float, bulk_left: float) -> list[bool]:
    """L1 (FIRSTPRINCIPLES_A §8.3, "draft only while behind schedule"), per lockstep rank and iteration.

    The phase cannot end before max(bulk_left, R_max / s_crit) more iterations, where R_max is the largest remaining
    budget among the rank's fork rows and s_crit the measured tokens per iteration of a drafting critical row. A fork
    row drafts iff it would finish after that point at plain decode (1 token per iteration), i.e. remaining > that
    deadline; the row holding R_max always drafts (it IS the critical path). An underestimated s_crit only turns
    drafting off for non-critical rows (never for the critical one), so the rule cannot slow the phase's critical
    chain. Reads only the rows' own progress: proposals only, exact by construction."""
    if not remaining:
        return []
    r_max = max(remaining)
    deadline = max(float(bulk_left), r_max / max(float(s_crit), 1.0))
    return [r == r_max or r > deadline for r in remaining]
