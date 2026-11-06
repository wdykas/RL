#!/usr/bin/env python3
"""history.json + staleness histogram for an nrl_k run, from the stock SC loop's own per-step prints in run.log.

  python3 scripts/lib/baselines_nrl/parse_nrl.py <run dir> [...]

Per train step the SingleController prints (single_controller.py, train pump):
  "⏱️  Timing:" / "  • Total step time: X s" / "  • <phase>: X s (p%)"  (its Timer, reduction sum)
  "step_metrics={...}"
  "train step N/M  trainer_v=V  lag=L"   (L = trainer version during the step - oldest sample version used)
run.log lines carry a wall-clock prefix "[t=<epoch>]" (stamp.py). Entry i of history.json = train step i + 1:
  time/iteration = the loop's total_step_time (the train pump's whole step: wait for data, logprobs, train, refit),
  wall/iteration = time between consecutive "train step" lines (cross-check), lag, timing/<phase>, train/<metric>.
Writes <run>/history.json and <run>/staleness.json ({lag: count} over all steps; the lag bound check is max <= k).
"""
import ast
import json
import math
import os
import re
import sys

TS = re.compile(r"^\[t=([0-9.]+)\] ?(.*)$")
STEP = re.compile(r"train step (\d+)/(\d+)\s+trainer_v=(-?\d+)\s+lag=(-?\d+)")
TOTAL = re.compile(r"Total step time: ([0-9.]+)s")
PHASE = re.compile(r"• ([A-Za-z0-9_/.\-]+): ([0-9.]+)s \(")
METRICS = re.compile(r"step_metrics=(\{.*\})\s*$")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse(run):
    hist, cur, prev_t = [], {}, None
    with open(os.path.join(run, "run.log"), errors="replace") as f:
        for raw in f:
            m = TS.match(raw.rstrip("\n"))
            t, line = (float(m.group(1)), m.group(2)) if m else (None, raw.rstrip("\n"))
            if (mm := TOTAL.search(line)):
                cur = {"time/iteration": float(mm.group(1))}
                continue
            if (mm := PHASE.search(line)) and "Total step time" not in line:
                cur[f"timing/{mm.group(1)}"] = float(mm.group(2))
                continue
            if (mm := METRICS.search(line)):
                txt = re.sub(r"\b(nan|inf|-inf)\b", "None", mm.group(1))
                txt = re.sub(r"\b(tensor|np\.float\d+|np\.int\d+)\(([^()]*)\)", r"\2", txt)
                try:
                    d = ast.literal_eval(txt)
                    for k, v in d.items():
                        if (x := _num(v)) is not None:
                            cur[f"train/{k}"] = x
                except (ValueError, SyntaxError):
                    cur["train/_unparsed"] = 1.0
                continue
            if (mm := STEP.search(line)):
                n, total, v, lag = map(int, mm.groups())
                e = dict(cur)
                e.update({"step": n - 1, "trainer_v": v, "lag": lag, "planned": total})
                if t is not None:
                    e["t_end"] = t
                    if prev_t is not None:
                        e["wall/iteration"] = t - prev_t
                    prev_t = t
                hist.append(e)
                cur = {}
    return hist


def main():
    for run in sys.argv[1:]:
        h = parse(run)
        with open(os.path.join(run, "history.json"), "w") as f:
            json.dump(h, f, indent=1)
        hist = {}
        for e in h:
            hist[e["lag"]] = hist.get(e["lag"], 0) + 1
        k = None
        try:
            k = int(open(os.path.join(run, "k")).read())
        except (OSError, ValueError):
            pass
        st = {"k": k, "lag_hist": {str(a): b for a, b in sorted(hist.items())}, "steps": len(h),
              "max_lag": max(hist) if hist else None,
              "bound_ok": (max(hist) <= k) if (hist and k is not None) else None,
              "lag_per_step": [e["lag"] for e in h]}
        with open(os.path.join(run, "staleness.json"), "w") as f:
            json.dump(st, f, indent=1)
        ti = [round(e.get("time/iteration", float("nan")), 1) for e in h]
        print(f"[nrl] {os.path.basename(run)}: {len(h)} steps; lag hist {st['lag_hist']} (k={k}, bound "
              f"{'OK' if st['bound_ok'] else st['bound_ok']}); time/iteration {ti}")


if __name__ == "__main__":
    main()
