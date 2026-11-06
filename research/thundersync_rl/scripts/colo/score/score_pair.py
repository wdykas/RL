"""Score colocated arms (colo lead 11-04): stock SPR / CS and ours XS / XPR, raw and kernel-normalized.

Per arm (run dir = $COLO_OUT/runs/<tag>, or any path to a run dir), steady window steps >= --from (default 3):
  s/step mean, p90, sd; the SC timers (exposed_generation, policy_training, weight_sync, training_prep, nl_verify_stage);
  XPR: the [nolanes verify] parts (verify forward, colo stand-down / wake, tail decode = continue_after_verify),
  whole / continued per step, the min reject position bin; routers: the stock bias/router probe lines or ours'
  [router frozen] + [nolanes sentinel] lines.
Kernel normalization (user 11-04: our BI kernels count as stock speed): with f_gen = CS gen / XS gen (same decode work:
512 rows to the cap at lag 0, same layout) and f_train = CS train / XS train (same tokens), ours' decode seconds
(exposed generation + tail decode) scale by f_gen and its learner seconds (training + verify forward) by f_train;
transitions and driver time stay as measured.
  python3 score_pair.py --cs <cs run> --xs <xs run> <run> [<run> ...] [--from=3]
"""
import os
import re
import statistics as S
import sys

KEYS = ("Total step time", "exposed_generation", "policy_training", "weight_sync", "training_prep", "nl_verify_stage",
        "advantage_calculation")
W = os.environ.get("COLO_OUT", "")


def find(tag):
    if os.path.exists(os.path.join(tag, "run.log")):
        return tag
    d = os.path.join(W, "runs", tag)
    if W and os.path.exists(os.path.join(d, "run.log")):
        return d
    sys.exit(f"no run.log for {tag} (pass a run dir, or set COLO_OUT and pass a tag under $COLO_OUT/runs)")


def parse(d, frm):
    rows, cur, ver = [], {}, {}
    probes = {"bias": 0, "bias_bad": 0, "router_ok": 0, "sentinel": 0, "sentinel_bad": 0}
    for ln in open(os.path.join(d, "run.log"), errors="replace"):
        for k in KEYS:
            m = re.search(rf"• {k}: ([0-9.]+)s", ln)
            if m:
                cur[k] = float(m.group(1))
        m = re.search(r"\[nolanes verify\] step=(\d+) .*?whole=(\d+) continued=(\d+).*?reject_pos_hist=(\{[^}]*\})", ln)
        if m:
            t = {k: float(v) for k, v in re.findall(r" ([a-z_0-9]+)=([0-9.]+)s", ln)}
            ver[int(m.group(1))] = dict(t, whole=int(m.group(2)), cont=int(m.group(3)), hist=m.group(4))
        if "[nrl-bias] call=" in ln:
            probes["bias"] += 1
            probes["bias_bad"] += int(not re.search(r"rate=0\.0 .*max_abs_delta=0\b", ln))
        if re.search(r"\[router frozen\] step=\d+ .*unchanged ok", ln) or re.search(r"\[nrl-router\].*equal_first=True", ln):
            probes["router_ok"] += 1
        m = re.search(r"\[nolanes sentinel\] step=\d+ train-lp tokens=\d+ inexact=(\d+)", ln)
        if m:
            probes["sentinel"] += 1
            probes["sentinel_bad"] += int(m.group(1) != "0")
        m = re.search(r"train step (\d+)/\d+", ln)
        if m and cur:
            cur["step"] = int(m.group(1)) - 1
            rows.append(cur)
            cur = {}
    rows = [r for r in rows if r["step"] >= frm and "Total step time" in r]
    for r in rows:
        v = ver.get(r["step"])
        if v:
            r.update({f"v_{k}": x for k, x in v.items()})
    return rows, probes


def p90(x):
    x = sorted(x)
    return x[min(len(x) - 1, int(round(0.9 * (len(x) - 1))))]


def mean(rows, k):
    return S.fmean([r.get(k, 0.0) for r in rows]) if rows else float("nan")


def main():
    frm, cs, xs, runs = 3, None, None, []
    a = sys.argv[1:]
    i = 0
    while i < len(a):
        if a[i].startswith("--from="):
            frm = int(a[i].split("=")[1])
        elif a[i] == "--cs":
            cs = find(a[i + 1]); i += 1
        elif a[i] == "--xs":
            xs = find(a[i + 1]); i += 1
        else:
            runs.append(find(a[i]))
        i += 1
    f_gen = f_train = 1.0
    if cs and xs:
        rc, _ = parse(cs, frm)
        rx, _ = parse(xs, frm)
        f_gen = mean(rc, "exposed_generation") / mean(rx, "exposed_generation")
        f_train = mean(rc, "policy_training") / mean(rx, "policy_training")
        print(f"normalization: f_gen {f_gen:.3f} (CS {mean(rc, 'exposed_generation'):.1f} / XS "
              f"{mean(rx, 'exposed_generation'):.1f} s), f_train {f_train:.3f} (CS {mean(rc, 'policy_training'):.1f} / XS "
              f"{mean(rx, 'policy_training'):.1f} s)")
    for d in runs:
        rows, pr = parse(d, frm)
        name = os.path.basename(d.rstrip("/"))
        if not rows:
            print(f"## {name}: no steady steps")
            continue
        tot = [r["Total step time"] for r in rows]
        ours = name.startswith(("colo-xs", "colo-xpr"))
        norm = []
        for r in rows:
            gen, train = r.get("exposed_generation", 0.0), r.get("policy_training", 0.0)
            tail = r.get("v_continue_after_verify", 0.0)
            ver = r.get("v_verify", 0.0)
            rest = r["Total step time"] - gen - train - tail - ver
            norm.append(rest + (gen + tail) * f_gen + (train + ver) * f_train if ours else r["Total step time"])
        line = (f"## {name}: steps {rows[0]['step']}..{rows[-1]['step']} (n={len(tot)})  s/step mean {S.fmean(tot):.2f} "
                f"p90 {p90(tot):.2f} sd {S.pstdev(tot):.2f}")
        if ours:
            line += f"  | kernel-normalized mean {S.fmean(norm):.2f} p90 {p90(norm):.2f}"
        print(line)
        print("   " + "  ".join(f"{k} {mean(rows, k):.2f}" for k in KEYS[1:]))
        if any("v_verify" in r for r in rows):
            vk = ("v_verify", "v_colo_standdown", "v_colo_wake", "v_continue_after_verify", "v_colo_standdown2", "v_whole",
                  "v_cont")
            print("   XPR stage: " + "  ".join(f"{k[2:]} {mean(rows, k):.2f}" for k in vk))
            print("   reject_pos_hist per step: " + " | ".join(str(r.get("v_hist")) for r in rows[:4]))
        print(f"   probes: {pr}")


if __name__ == "__main__":
    main()
