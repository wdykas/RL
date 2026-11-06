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
"""Always-on conservation checks of the learner layout and step counts (layout audit run rules).

Pure functions over what every learner rank already reports to the driver: no collectives, no tensors. The driver
raises ``RunCheckError`` on any violation.

* ``layout_violations`` (startup; run rule 8): the ranks that must run identical forward schedules -- every
  tensor x context-parallel group (dense collectives) and every expert-tensor x expert-parallel group (expert
  collectives) -- must (a) lie in one verify-yield domain (they take the same inline-yield decisions) and
  (b) see the same EP lockstep schedule. The lockstep length of a verify call is a function of the plans of the
  ranks in the caller's EP group, and the plans are per data-parallel replica, so (b) holds iff every member's EP
  group spans the same set of data-parallel ranks.
* ``lockstep_violations`` (every step): the forwards each rank scheduled through the EP lockstep (cumulative) are
  equal within every EP group and every TP x CP group.
* ``step_counts_merge`` (every step; run rule 2): the DP-reduced loss denominator equals the valid tokens dealt to
  the learner minus those the learner took back (verify-in-train rejections), every rank holds the same
  denominator, and the TP / PP / CP peers of each data-parallel replica agree on their local counts and
  micro-batch count.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Optional


class RunCheckError(RuntimeError):
    """A layout or conservation run rule failed."""


# ---------------------------------------------------------------------------
# Startup layout (run rule 8)
# ---------------------------------------------------------------------------


def _groups(reports: list[dict[str, Any]], key: str) -> dict[tuple, list[dict[str, Any]]]:
    out: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in reports:
        g = r.get(key)
        if g is not None:
            out[tuple(g)].append(r)
    return out


def layout_violations(reports: list[dict[str, Any]]) -> list[str]:
    """Violations of the learner layout rule; ``reports`` is one ``layout_ids()`` dict per learner rank.

    Each report: ``rank`` (global), ``dp`` (data-parallel rank without CP), ``tp_cp`` / ``etp_ep`` / ``ep``
    (global ranks of the rank's TP x CP, ETP x EP and EP groups; ``ep`` None without expert parallelism) and
    ``yield_domain`` (None: verify-yield off for this layout).
    """
    by_rank = {int(r["rank"]): r for r in reports}
    bad: list[str] = []

    def ep_dp_set(r: dict[str, Any]) -> Optional[frozenset]:
        ep = r.get("ep")
        if ep is None:
            return None
        missing = [s for s in ep if int(s) not in by_rank]
        if missing:
            bad.append(f"rank {r['rank']}: EP group {tuple(ep)} has ranks with no report {missing}")
            return None
        return frozenset(int(by_rank[int(s)]["dp"]) for s in ep)

    for key, what in (("tp_cp", "TP x CP"), ("etp_ep", "ETP x EP")):
        for grp, members in _groups(reports, key).items():
            doms = {repr(m.get("yield_domain")) for m in members}
            if len(doms) > 1:
                bad.append(f"{what} group {grp} spans {len(doms)} verify-yield domains {sorted(doms)}")
            if any(m.get("ep") is not None for m in members):
                sets = {ep_dp_set(m) for m in members}
                if len(sets) > 1:
                    bad.append(
                        f"{what} group {grp}: its ranks' EP lockstep groups cover different data-parallel "
                        f"replicas {sorted(sorted(s) for s in sets if s is not None)} (unequal forward counts)"
                    )
    return bad


def check_layout(reports: list[dict[str, Any]]) -> None:
    bad = layout_violations(reports)
    if bad:
        raise RunCheckError(
            f"learner layout check: {len(bad)} violation(s): " + "; ".join(bad[:8])
        )


# ---------------------------------------------------------------------------
# Per-step EP lockstep forward counts
# ---------------------------------------------------------------------------


def lockstep_violations(results: Iterable[dict[str, Any]]) -> list[str]:
    """Each rank's ``stream_lockstep`` = (cumulative lockstep forwards, EP group id, TP x CP group id; a group id
    is its lowest global rank); counts must be equal within every EP group and every TP x CP group."""
    rows = [(r or {}).get("stream_lockstep") for r in results]
    rows = [x for x in rows if x is not None]
    bad = []
    for idx, what in ((1, "EP"), (2, "TP x CP")):
        groups: dict[Any, set] = defaultdict(set)
        for x in rows:
            if x[idx] is not None:
                groups[x[idx]].add(int(x[0]))
        for g, counts in groups.items():
            if len(counts) > 1:
                bad.append(f"{what} group {g} ran unequal EP lockstep forward counts {sorted(counts)}")
    return bad


# ---------------------------------------------------------------------------
# Tokens in the loss (run rule 2)
# ---------------------------------------------------------------------------

STEP_COUNT_FIELDS = (
    "local_seqs",
    "local_toks",
    "num_microbatches",
    "global_seqs",
    "global_toks",
    "trained_toks",
    "uncounted_toks",
)


def step_counts_merge(results: list[dict[str, Any]], tokens_sent: float) -> Optional[dict[str, Any]]:
    """Merge every learner rank's ``stream_step_counts`` with the tokens the driver dealt.

    Returns None when no rank reported (older worker), else metrics and ``violations`` (list of str):
    tokens_sent == global_valid_toks + uncounted (summed over one rank per DP replica), trained <= global,
    the same global counts on every rank, and equal local counts / micro-batches over each replica's
    TP / PP / CP ranks. Cost: O(ranks) Python ints on the driver.
    """
    rows = [(r or {}).get("stream_step_counts") for r in results]
    owners = [bool((r or {}).get("stream_dp_count_owner", True)) for r in results]
    if not any(x is not None for x in rows):
        return None
    bad: list[str] = []
    if any(x is None for x in rows):
        bad.append(f"{sum(x is None for x in rows)}/{len(rows)} learner ranks sent no step counts")
    rows_o = [(x, o) for x, o in zip(rows, owners) if x is not None]
    glob = {(float(x["global_seqs"]), float(x["global_toks"])) for x, _ in rows_o}
    if len(glob) > 1:
        bad.append(f"learner ranks disagree on (global_valid_seqs, global_valid_toks): {sorted(glob)[:6]}")
    by_dp: dict[Any, set] = defaultdict(set)
    for x, _ in rows_o:
        by_dp[x.get("dp")].add((float(x["local_seqs"]), float(x["local_toks"]), int(x["num_microbatches"])))
    for dp, vals in sorted(by_dp.items(), key=lambda kv: repr(kv[0])):
        if len(vals) > 1:
            bad.append(
                f"DP replica {dp}: TP/PP/CP peers disagree on (local seqs, local toks, micro-batches) {sorted(vals)}"
            )
    g_seqs, g_toks = max(glob)
    own = [x for x, o in rows_o if o]
    local_sum = sum(float(x["local_toks"]) for x in own)
    uncounted = sum(float(x["uncounted_toks"]) for x in own)
    rows_uncounted = sum(float(x.get("uncounted_rows", 0)) for x in own)
    trained = max(float(x["trained_toks"]) for x, _ in rows_o)
    if local_sum != g_toks:
        bad.append(f"global_valid_toks {g_toks:.0f} != sum of the DP replicas' local tokens {local_sum:.0f}")
    if float(tokens_sent) != g_toks + uncounted:
        bad.append(
            f"tokens dealt {float(tokens_sent):.0f} != global_valid_toks {g_toks:.0f} + taken back {uncounted:.0f}"
        )
    if trained > g_toks:
        bad.append(f"trained_valid_toks {trained:.0f} > global_valid_toks {g_toks:.0f}")
    return {
        "global_valid_seqs": g_seqs,
        "global_valid_toks": g_toks,
        "trained_valid_toks": trained,
        "tokens_uncounted": uncounted,
        "rows_uncounted": rows_uncounted,
        "violations": bad,
    }
