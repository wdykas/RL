"""One-line per-step health summary: flags waste that is otherwise silent.

Each step's metrics already carry the evidence of the silent bugs found so far
(filler micro-batches, split dispatches, MBS cut by the token cap, rows decoded
fresh because drafts were dropped, sentinel cost, drain waits), but only as
numbers among ~170 keys. ``step_health`` reads them and returns flags; the loop
prints ``[health step N] OK | WARN <flags> | <info>`` and logs ``health/warn``
(flag count) and ``health/flags`` (the flag names) in the step's history.

Only metrics present in the step are checked, so the function is arm-agnostic
(sync, async1, ThunderSync one-engine or lanes). Thresholds are module
constants; THUNDERSYNC_HEALTH_STRICT=1 makes any flag raise after the step is
logged (for exactness/CI runs where waste means a bug).
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

FILLER_FRAC_MAX = 0.05
SENTINEL_S_MAX = 2.0
DRAIN_WAIT_S_MAX = 2.0
MBS_EFF_MIN_FRAC = 0.75  # live rows per live micro-batch vs the configured MBS
PREFIX_DENIED_FRAC_MAX = 0.10
DISPATCH_GAP_S_INFO = 5.0


def _f(metrics: dict[str, Any], key: str) -> Optional[float]:
    v = metrics.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    return float(v)


def is_speculative_step(metrics: dict[str, Any]) -> bool:
    """ThunderSync-loop arms with drafts (any of their per-step families)."""
    return any(k.startswith(("lanes/", "tail/", "draft/")) for k in metrics) or (
        "spec/draft_tokens" in metrics
    )


def step_health(
    step: int,
    metrics: dict[str, Any],
    prev: Optional[dict[str, Any]] = None,
    mbs: Optional[int] = None,
    expect: Optional[dict[str, Any]] = None,
) -> tuple[list[str], list[str]]:
    """(flags, info) for one step's metrics. ``prev``: the previous step's metrics.

    ``expect``: enabled features whose effect must show ({"verify_yield": True}:
    learner/yield_inlined > 0 once the tails run, from step 1).
    """
    expect = expect or {}
    flags: list[str] = []
    info: list[str] = []
    spec = is_speculative_step(metrics)

    # Learner: fillers, dispatch count/fragments, MBS actually used, idle.
    ff = _f(metrics, "learner/filler_frac")
    if ff is not None and ff > FILLER_FRAC_MAX:
        flags.append(
            f"filler_frac={ff:.3f}(mb={_f(metrics, 'learner/filler_microbatches') or 0:.0f},"
            f"heavy={_f(metrics, 'learner/filler_heavy_dispatches') or 0:.0f})"
        )
    nd = _f(metrics, "learner/dispatches")
    disp = metrics.get("timeline/dispatches")
    if nd is not None:
        if not spec and step >= 1 and nd > 1:
            # sync / async1 have every row in hand: one dispatch (async1 split bug).
            flags.append(f"dispatches={nd:.0f}>1")
        frag = ""
        if isinstance(disp, list) and disp and all(isinstance(d, (list, tuple)) for d in disp):
            rows = [d[2] for d in disp if len(d) > 2]
            if rows:
                big = max(rows)
                nfrag = sum(1 for r in rows if r < big / 2)
                frag = f",frag={nfrag}"
            gaps = [
                disp[i + 1][0] - disp[i][1]
                for i in range(len(disp) - 1)
                if len(disp[i]) > 1 and len(disp[i + 1]) > 0
            ]
            if gaps and max(gaps) > DISPATCH_GAP_S_INFO:
                info.append(f"max_dispatch_gap={max(gaps):.1f}s")
        info.append(f"disp={nd:.0f}{frag}")
    real = _f(metrics, "learner/real_microbatches")
    skipped = _f(metrics, "learner/skipped_microbatches")
    # Rows the live micro-batches carried: the nonzero-advantage rows when
    # zero-advantage rows are skipped (count-only), else every trained row.
    live_rows = _f(metrics, "train/nonzero_adv_rows") if skipped else _f(metrics, "train/rows")
    if real is not None:
        info.append(f"mb_live/skip={real:.0f}/{(skipped or 0):.0f}")
        if mbs and mbs > 1 and live_rows and real > 0:
            eff = live_rows / real
            if eff < MBS_EFF_MIN_FRAC * mbs:
                # The token cap (train_micro_batch_tokens) or packing cut the
                # configured micro-batch rows (the 32768 // 10240 = 3 class).
                flags.append(f"mbs_eff={eff:.2f}<{mbs}")
    dw = _f(metrics, "learner/drain_wait_s")
    if dw is not None and dw > DRAIN_WAIT_S_MAX:
        flags.append(f"drain_wait={dw:.1f}s")
    busy = _f(metrics, "learner/busy_s")
    it = _f(metrics, "time/iteration")
    if busy is not None and it:
        info.append(f"learner_busy={busy / it:.2f}")

    # Drafts: dropped q, fresh rows, coverage, plans left over.
    dropped = _f(metrics, "spec/engine_q_dropped")
    if dropped:
        flags.append(f"q_dropped={dropped:.0f}")
    fresh = _f(metrics, "spec/null_fresh_slots")
    if fresh:
        # Undrafted ⊥-BV slots decode from scratch at theta_k after the tails:
        # a whole generation on the critical path (bh-cc: +80-90 s per step).
        flags.append(f"fresh_rows={fresh:.0f}")
    forced = _f(metrics, "debug/forced_fresh_rows")
    if forced:
        info.append(f"forced_fresh={forced:.0f}")
    if spec and step >= 1:
        fr = _f(metrics, "lanes/fresh_refit")
        if fr and not fresh and not forced:
            flags.append("lane_fresh_refit")
        r, rp = _f(metrics, "lanes/refits"), _f(prev or {}, "lanes/refits")
        if r is not None and rp is not None and r - rp > 1:
            flags.append(f"lane_refits+{r - rp:.0f}")
    trunc = _f(metrics, "spec/route_truncated_drafts")
    if trunc:
        flags.append(f"route_truncated={trunc:.0f}")
    pd = _f(metrics, "spec/plans_dropped")
    if pd:
        flags.append(f"plans_dropped={pd:.0f}")
    fa = _f(metrics, "spec/full_accept")
    if fa is not None:
        info.append(f"full_accept={fa:.2f}")
    acc = _f(metrics, "spec/accepted_frac")
    if acc is not None:
        info.append(f"acc={acc:.3f}")

    # Prefix cache (learner pieces): denials/evictions mean re-prefill.
    elig = _f(metrics, "spec/prefix_cache_eligible")
    denied = _f(metrics, "spec/prefix_cache_denied") or 0.0
    evicted = _f(metrics, "spec/prefix_cache_evicted") or 0.0
    if elig:
        hit = _f(metrics, "spec/prefix_cache_eligible_hit_frac")
        if hit is not None:
            info.append(f"pc_hit={hit:.2f}")
        if denied / elig > PREFIX_DENIED_FRAC_MAX:
            flags.append(f"pc_denied={denied:.0f}/{elig:.0f}")
    if evicted:
        flags.append(f"pc_evicted={evicted:.0f}")
    # verify0 capture: first pieces should start from verify0's state.
    cap = _f(metrics, "spec/prefix_cache_v0_captured")
    cap_denied = _f(metrics, "spec/prefix_cache_v0_denied") or 0.0
    first = _f(metrics, "spec/prefix_cache_first_rows")
    if first:
        fh = _f(metrics, "spec/prefix_cache_first_hit_frac")
        if fh is not None:
            info.append(f"pc_first_hit={fh:.2f}")
        if cap and not _f(metrics, "spec/prefix_cache_first_v0_hits"):
            flags.append("pc_v0_unused")
    if cap is not None and cap + cap_denied > 0 and cap_denied / (cap + cap_denied) > PREFIX_DENIED_FRAC_MAX:
        flags.append(f"pc_v0_denied={cap_denied:.0f}/{cap + cap_denied:.0f}")

    # Enabled features that did nothing.
    if expect.get("verify_yield") and step >= 1 and "learner/yield_inlined" in metrics:
        if not _f(metrics, "learner/yield_inlined"):
            flags.append("yield_inlined=0")
    am = _f(metrics, "engine/anchor_missing")
    if am:
        flags.append(f"anchor_missing={am:.0f}")
    # fp16 range (learner casts; the engine computes the same values).
    ov = _f(metrics, "fp16/act_overflow")
    if ov:
        flags.append(f"fp16_overflow={ov:.0f}")
    wa = _f(metrics, "fp16/weight_amax")
    if wa is not None:
        info.append(f"w_amax={wa:.3g}")
        if wa > 65504.0:
            flags.append(f"fp16_weight_amax={wa:.3g}")
    hg = _f(metrics, "learner/head_materialized_dispatch_grad")
    if hg:
        flags.append(f"head_nograd_materialized={hg:.0f}")

    # Refit-side costs.
    st = _f(metrics, "sentinel/time_s")
    if st is not None and st > SENTINEL_S_MAX and step >= 1:
        flags.append(f"sentinel={st:.1f}s")
    return flags, info


def format_health(step: int, flags: list[str], info: list[str]) -> str:
    head = "OK" if not flags else "WARN " + " ".join(flags)
    return f"[health step {step}] {head} | " + " ".join(info)


def annotate(
    step: int,
    metrics: dict[str, Any],
    prev: Optional[dict[str, Any]] = None,
    mbs: Optional[int] = None,
    expect: Optional[dict[str, Any]] = None,
) -> str:
    """Compute the step's health, store it in ``metrics``, return the line."""
    flags, info = step_health(step, metrics, prev, mbs, expect)
    real = _f(metrics, "learner/real_microbatches")
    if real:
        skipped = _f(metrics, "learner/skipped_microbatches")
        rows = _f(metrics, "train/nonzero_adv_rows") if skipped else _f(metrics, "train/rows")
        if rows is not None:
            # Rows per live micro-batch actually run (vs the configured MBS).
            metrics["learner/rows_per_live_mb"] = rows / real
    metrics["health/warn"] = float(len(flags))
    metrics["health/flags"] = " ".join(f.split("=")[0].split("(")[0] for f in flags)
    return format_health(step, flags, info)


def strict() -> bool:
    return os.environ.get("THUNDERSYNC_HEALTH_STRICT") == "1"


def main(argv: Optional[list[str]] = None) -> int:
    """Replay: ``python -m thundersync_rl.health RUN_DIR|history.json ... [--mbs 2]``."""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--mbs", type=int, default=None)
    a = ap.parse_args(argv)
    for run in a.runs:
        path = run if run.endswith(".json") else os.path.join(run, "history.json")
        try:
            with open(path) as f:
                hist = json.load(f)
        except (OSError, ValueError) as e:
            print(f"== {run}: unreadable history ({e})")
            continue
        print(f"== {run}")
        for i, m in enumerate(hist):
            step = int(m.get("step", i))
            print("  " + annotate(step, dict(m), hist[i - 1] if i else None, a.mbs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
