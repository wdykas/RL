"""Score stock colocated arms (CS) from run.log (agent steady, 11-04): per step the SC timer block (total, exposed
generation, policy training, weight sync, training prep, advantages); steady window = steps >= --from (default 3).
Prints mean / p90 / sd and the router-frozen evidence (frozen_router.py over the run).
  python3 score_cs.py <run dir> [...] [--from=3]"""
import os, re, statistics as S, subprocess, sys

KEYS = ("Total step time", "exposed_generation", "policy_training", "weight_sync", "training_prep", "advantage_calculation")

def steps(d):
    rows, cur = [], {}
    for ln in open(os.path.join(d, "run.log"), errors="replace"):
        for k in KEYS:
            m = re.search(rf"• {k}: ([0-9.]+)s", ln)
            if m:
                cur[k] = float(m.group(1))
        m = re.search(r"train step (\d+)/\d+", ln)
        if m and cur:
            cur["step"] = int(m.group(1)); rows.append(cur); cur = {}
    return rows

def p90(x):
    x = sorted(x); return x[min(len(x) - 1, int(round(0.9 * (len(x) - 1))))]

frm = 3; dirs = []
for a in sys.argv[1:]:
    if a.startswith("--from="): frm = int(a.split("=")[1])
    else: dirs.append(a)
for d in dirs:
    rows = [r for r in steps(d) if r.get("step", 0) > frm]  # "train step N" closes step N-1 (0-based)
    if not rows:
        print(f"{d}: no steps"); continue
    tot = [r["Total step time"] for r in rows if "Total step time" in r]
    parts = {k: S.fmean([r.get(k, 0.0) for r in rows]) for k in KEYS[1:]}
    fr = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib", "baselines_nrl", "frozen_router.py"), d],
                        capture_output=True, text=True).stdout.strip().splitlines()
    print(f"## {os.path.basename(d.rstrip('/'))}: steps {rows[0]['step']-1}..{rows[-1]['step']-1} (n={len(tot)})")
    print(f"  s/step mean {S.fmean(tot):.2f}  p90 {p90(tot):.2f}  sd {S.pstdev(tot):.2f}  | " +
          "  ".join(f"{k} {v:.2f}" for k, v in parts.items()))
    print(f"  {fr[-1] if fr else 'frozen_router: no output'}")
