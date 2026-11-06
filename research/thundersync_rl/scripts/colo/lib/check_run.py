"""Post-run checks every run must pass (colo kit). Prints one verdict line per run dir; exit 1 on any failure.

  python3 check_run.py <run dir> [...]

OURS (run dir has arm = colo):
  - router frozen: >= 1 "[router frozen] step=N ... unchanged ok" line, and no such line without "unchanged ok"
  - sentinel: >= 1 "[nolanes sentinel] step=N train-lp tokens=T inexact=X" line (the learner's train-time logprobs vs the
    engine's sampling logprobs, bitwise), every X == 0
  - with CHECKS=1 also: every "[colo|nolanes mismatch] ... tok=T exact=E" has E == T, every "[law check*] step" line
    (all tokens, first token, first token cumulative) is "ok"
  - NEGATIVE controls (file <run>/negative exists; written for NEG=... / SKIP_RESHARD=1): the run must FAIL the
    sentinel, the mismatch or a law test (a negative that passes means the checks are blind). spare_by_content (XPR-S)
    biases which prompts train, not any token's law given its prompt: it fails a law test, not the sentinel.
STOCK (arm = colo_<variant>_<K>) and SPR-BI (arm = sprbi): frozen_router.py (effective config + per-step [nrl-bias] /
  [nrl-router] probes).
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def check(d):
    arm = open(os.path.join(d, "arm")).read().strip() if os.path.exists(os.path.join(d, "arm")) else "?"
    name = os.path.basename(d.rstrip("/"))
    if arm != "colo":
        r = subprocess.run([sys.executable, os.path.join(HERE, "baselines_nrl", "frozen_router.py"), d],
                           capture_output=True, text=True)
        out = (r.stdout.strip().splitlines() or ["no output"])[-1]
        print(f"[check] {name} (stock {arm}): {'PASS' if r.returncode == 0 else 'FAIL'} router: {out}")
        return r.returncode == 0
    neg = os.path.exists(os.path.join(d, "negative"))
    rf_ok = rf_bad = 0
    sent, sent_bad, mm, mm_bad, law, law_bad = 0, 0, 0, 0, 0, 0
    with open(os.path.join(d, "run.log"), errors="replace") as f:
        for ln in f:
            if "[router frozen] step=" in ln:
                if "unchanged ok" in ln:
                    rf_ok += 1
                else:
                    rf_bad += 1
            m = re.search(r"\[nolanes sentinel\] step=(\d+) train-lp tokens=(\d+) inexact=(\d+)", ln)
            if m:
                sent += 1
                sent_bad += int(m.group(3) != "0")
            m = re.search(r"\[(?:colo|nolanes) mismatch\] step=(\d+) tok=(\d+) exact=(\d+)", ln)
            if m:
                mm += 1
                mm_bad += int(m.group(2) != m.group(3))
            m = re.search(r"\[law check[^\]]*\] step=\d+ .* (ok|FAIL)\s*$", ln)  # all tokens, first token, cumulative
            if m:
                law += 1
                law_bad += int(m.group(1) == "FAIL")
    router = rf_ok > 0 and rf_bad == 0
    stats = (f"router-frozen {rf_ok} ok / {rf_bad} bad; sentinel {sent} steps, {sent_bad} inexact; "
             f"mismatch {mm} steps, {mm_bad} inexact; law {law} steps, {law_bad} FAIL")
    if neg:
        ok = router and (sent_bad > 0 or mm_bad > 0 or law_bad > 0)
        print(f"[check] {name} (NEGATIVE control): {'PASS (failed as it must)' if ok else 'FAIL (checks are blind!)'}: {stats}")
    else:
        ok = router and sent > 0 and sent_bad == 0 and mm_bad == 0 and law_bad == 0
        print(f"[check] {name} (ours): {'PASS' if ok else 'FAIL'}: {stats}")
    return ok


if __name__ == "__main__":
    res = [check(d) for d in sys.argv[1:]]
    sys.exit(0 if res and all(res) else 1)
