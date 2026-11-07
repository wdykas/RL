"""Verdict of nolanes runs (X1 gate and negative controls) from their run.log.

    python nl_summary.py <run dir or run.log> [...]

Per run: EXACT when every step's checks pass and are non-vacuous:
  - [nolanes mismatch]: exact == tok > 0 (learner forward of every output token == its generation log-prob: E5/E7);
  - [engine q check]: lines present, exact == tok, max 0 (logged q == q rebuilt from head input + tagged head: E1);
  - [law check], [law check first-token], [emission law *]: ok (sampling law of every token / row selection / ⊥-BV);
  - [trained once check]: ok (every output row trained exactly once in its step; required under train_overlap);
  - [nolanes verify]: mixed_version_rows > 0 on some step (refits landed mid-row), continued rows > 0;
  - coverage arms (stress and the version-tag / content negatives, COVERAGE_ARMS): mixed_version_rows > 0 AND
    mixed_continued > 0 (a mixed-version draft rejected and continued; R35 2(e)). A negative arm without that
    coverage is NOT a passing control, whatever it failed;
  - [trained once check] non-vacuous under train_overlap: trained > 0 on some step (R35 1(b));
  - keyed sampler S (unconditional): the engine stats line with s_calls > 0 (S actually ran; R35 3(h));
  - THUNDERSYNC_VWALL_CHECK set (effective config): on every step spec/ds2_vw_checked == spec/ds2_rows > 0 and
    spec/ds2_vw_mismatch == 0 (R36 e: the gate check must have run);
  - no exactness exception (SamplerKeyError, ProvenanceError, version-tag errors, validator refusal).
A negative arm (THUNDERSYNC_NL_NEG / tag contains 'neg-') PASSES as a control when the run is not EXACT, names the
failing check, and its neg_tokens > 0 (the negative touched tokens: non-vacuous).
"""

import os
import re
import sys

# Arms whose check needs mixed-version drafts that are rejected and continued (R35 2(e)).
COVERAGE_ARMS = ("stress", "neg_wrong_tag", "neg_tag_off_by_one", "neg_content_m")
# Arms exempt from the mixed-version-row requirement (labelled): sanity (lr 0) and prod (the production shape: lag 1,
# no admission hold, so no draft spans a refit by construction; R37 (iii)).
MIXED_EXEMPT_ARMS = ("sanity", "prod", "prod_stress")
# Failure lines that only say a run was vacuous / not covered. A negative control passes only on a failure that is NOT
# one of these (R37 (i)): a negative that merely ran nothing, or ran without coverage, did not fail a check.
VACUITY_FAILS = (
    "vacuous", "finalize overlap: vacuous", "sampler S never ran", "no verify step", "no engine-q check lines",
    "no mismatch-checked tokens", "no law lines", "train overlap: no grad order", "finalize overlap: no trained-once",
    "no engine stats line", "effective config missing", "router frozen: missing", "edv: vacuous", "edv: no [edv] lines",
)


class ConfigUnreadable(Exception):
    pass


def _read_effective_config(run_dir):
    """The run's effective_config.yaml text, parsed when PyYAML is available; ConfigUnreadable when it is missing,
    empty or does not parse (R38 (b): the run then fails)."""
    f = os.path.join(run_dir, "effective_config.yaml")
    try:
        txt = open(f).read()
    except OSError as e:
        raise ConfigUnreadable(f"{f}: {e}")
    if not txt.strip():
        raise ConfigUnreadable(f"{f}: empty")
    try:
        import yaml
    except ImportError:
        return txt
    try:
        cfg = yaml.safe_load(txt)
    except Exception as e:
        raise ConfigUnreadable(f"{f}: {type(e).__name__}")
    if not isinstance(cfg, dict):
        raise ConfigUnreadable(f"{f}: not a mapping")
    return cfg

ERRORS = (
    "SamplerKeyError", "ProvenanceError", "refit inside an engine step", "thundersync.mode=nolanes:",
    "engine version tags", "Traceback (most recent call last)",
)


def _gate_checks(log):
    """Whether THUNDERSYNC_VWALL_CHECK is set, from the run's effective_config.yaml next to run.log."""
    cfg_path = os.path.join(os.path.dirname(log), "effective_config.yaml")
    if not os.path.exists(cfg_path):
        return False
    try:
        import yaml

        cfg = yaml.safe_load(open(cfg_path)) or {}
    except Exception:  # noqa: BLE001 (summary tool: an unreadable config disables this check only)
        return False
    envs = ((cfg.get("policy") or {}).get("megatron_cfg") or {}).get("env_vars") or {}
    on = lambda v: str(v).strip().lower() not in ("", "0", "false", "no", "off", "none")  # noqa: E731
    return on(envs.get("THUNDERSYNC_VWALL_CHECK", ""))


def parse(path):
    log = path if os.path.isfile(path) else os.path.join(path, "run.log")
    text = open(log, errors="replace").read()
    lines = text.splitlines()
    r = {"log": log, "fails": [], "steps": 0, "mixed": 0, "continued": 0, "rows": 0, "early": 0, "neg_tokens": 0, "eq_lines": 0,
         "mm_tok": 0, "law_lines": 0, "errors": [], "overlap": False, "go_lines": 0,
         "fo": False, "to_lines": 0, "to_trained": 0, "mixed_cont": 0, "s_calls": 0,
         "eng_lines": 0, "router_ok": 0, "router_bad": 0,
         "edv_lines": 0, "edv_deferred": 0, "edv_surprises": 0, "edv_replay": 0, "edv_mismatch": 0, "edv_neg_rows": 0,
         "mtd_lines": 0, "mtd_proposed": 0, "mtd_extra": 0, "mtd_tokens": 0, "mtd_iters": 0}
    vw_on = _gate_checks(log)
    for ln in lines:
        if "[nolanes verify]" in ln:
            r["steps"] += 1
            spec = {k: float(v) for k, v in re.findall(r"'((?:spec|nl)/[a-z0-9_]+)': ([0-9.eE+-]+)", ln)}
            if vw_on:
                rows, chk = spec.get("spec/ds2_rows", 0.0), spec.get("spec/ds2_vw_checked", 0.0)
                if not (rows > 0 and chk == rows) or spec.get("spec/ds2_vw_mismatch", 0.0) != 0:
                    r["fails"].append(f"vwall check: ds2_rows={rows:.0f} checked={chk:.0f} "
                                      f"mismatch={spec.get('spec/ds2_vw_mismatch', 0.0):.0f}")
            m = re.search(r"mixed_version_rows=(\d+)", ln)
            r["mixed"] += int(m.group(1)) if m else 0
            m = re.search(r"mixed_continued=(\d+)", ln)
            r["mixed_cont"] += int(m.group(1)) if m else 0
            m = re.search(r" continued=(\d+)", ln)
            r["continued"] += int(m.group(1)) if m else 0
            m = re.search(r" rows=(\d+)", ln)
            r["rows"] += int(m.group(1)) if m else 0
            m = re.search(r"early_rejects=(\d+)", ln)
            r["early"] += int(m.group(1)) if m else 0
            m = re.search(r"neg_tokens=(\d+)", ln)
            r["neg_tokens"] += int(m.group(1)) if m else 0
        elif "[nolanes mismatch]" in ln:
            m = re.search(r"tok=(\d+) exact=(\d+) max\|d\|=(\S+)", ln)
            if m:
                tok, ex = int(m.group(1)), int(m.group(2))
                r["mm_tok"] += tok
                if tok == 0 or ex != tok:
                    r["fails"].append(f"mismatch tok={tok} exact={ex} max={m.group(3)}")
        elif "[engine q check]" in ln:
            m = re.search(r"tok=(\d+) exact=(\d+) max\|dlogq\|=(\S+)", ln)
            if m:
                r["eq_lines"] += 1
                if int(m.group(1)) != int(m.group(2)) or float(m.group(3)) != 0.0:
                    r["fails"].append(f"engine-q tok={m.group(1)} exact={m.group(2)} max={m.group(3)}")
        elif re.search(r"\[(law check|emission law)[^\]]*\]", ln):
            r["law_lines"] += 1
            if ln.rstrip().endswith("FAIL"):
                r["fails"].append("law: " + ln.strip()[:160])
        elif "[grad order check]" in ln:
            r["go_lines"] += 1
            if ln.rstrip().endswith("FAIL"):
                r["fails"].append("grad order: " + ln.strip()[ln.find("[grad order check]"):][:160])
        elif "[edv] step=" in ln:
            # EDV (edv.py): deferred rows' committed decisions vs their own verification (CHECKS replay).
            r["edv_lines"] += 1
            for k, f in (("deferred", "edv_deferred"), ("surprises", "edv_surprises"), ("replay_rows", "edv_replay"),
                         ("replay_mismatch", "edv_mismatch"), ("neg_edv_rows", "edv_neg_rows")):
                m = re.search(rf" {k}=(\d+)", ln)
                r[f] += int(m.group(1)) if m else 0
            m = re.search(r" replay_mismatch=(\d+)", ln)
            if m and int(m.group(1)) > 0:
                r["fails"].append("edv replay: " + ln.strip()[ln.find("[edv]"):][:160])
        elif "[router frozen]" in ln:
            if "unchanged ok" in ln:
                r["router_ok"] += 1
            else:
                r["router_bad"] += 1
                r["fails"].append("router changed: " + ln.strip()[ln.find("[router frozen]"):][:160])
        elif "[mtd] step=" in ln:
            r["mtd_lines"] += 1
            for k_, f_ in (("mtd_proposed", "proposed"), ("mtd_extra", "extra_rows"), ("mtd_tokens", "tokens"),
                           ("mtd_iters", "tail_iters")):
                m = re.search(rf" {f_}=(\d+)", ln)
                r[k_] += int(m.group(1)) if m else 0
        elif "[trained once check]" in ln:
            r["to_lines"] += 1
            m = re.search(r" trained=(\d+)", ln)
            r["to_trained"] = max(r["to_trained"], int(m.group(1)) if m else 0)
            if ln.rstrip().endswith("FAIL"):
                r["fails"].append("trained once: " + ln.strip()[ln.find("[trained once check]"):][:160])
        elif "[nolanes] mode=nolanes" in ln and "train_overlap=1" in ln:
            r["overlap"] = True
            r["fo"] = True  # train_overlap always finalizes through the pipeline (trained-once check)
        elif "[nolanes engine]" in ln:
            if "step=" in ln:
                r["eng_lines"] += 1
            m = re.search(r"'neg_tokens': (\d+)", ln)
            if m:
                r["neg_tokens"] = max(r["neg_tokens"], int(m.group(1)))
            m = re.search(r"'s_calls': (\d+)", ln)
            if m:
                r["s_calls"] = max(r["s_calls"], int(m.group(1)))
        m = re.search(r"negative tokens (\d+)", ln)
        if m:
            r["neg_tokens"] = max(r["neg_tokens"], int(m.group(1)))
        for e in ERRORS:
            if e in ln:
                r["errors"].append(ln.strip()[:200])
    if r["steps"] == 0:
        r["fails"].append("no verify step")
    if r["eq_lines"] == 0:
        r["fails"].append("no engine-q check lines")
    if r["mm_tok"] == 0:
        r["fails"].append("no mismatch-checked tokens")
    if r["law_lines"] == 0:
        r["fails"].append("no law lines")
    if r["overlap"] and r["go_lines"] == 0:
        r["fails"].append("train overlap: no grad order check lines")
    if r["fo"] and r["to_lines"] == 0:
        r["fails"].append("finalize overlap: no trained-once check lines")
    if r["fo"] and r["to_lines"] and r["to_trained"] == 0:
        r["fails"].append("finalize overlap: vacuous trained-once check (no step trained a row)")
    # R38 (b): the run's effective config must exist and parse. Sampler S is unconditional (keyed_sampling_s), so every
    # run must report its engine stats line with s_calls > 0 (R35 3(h)).
    try:
        _read_effective_config(os.path.dirname(log))
    except ConfigUnreadable as e:
        r["fails"].append(f"effective config missing / unreadable: {str(e)[:120]}")
    if r["eng_lines"] == 0:
        r["fails"].append("no engine stats line")
    elif r["s_calls"] == 0:
        r["fails"].append("sampler S never ran (s_calls == 0)")
    arm_name = os.path.basename(os.path.dirname(log))
    sanity = "sanity" in arm_name
    exempt = any(f"-{a}-" in arm_name or arm_name.startswith(f"x1-{a}") for a in MIXED_EXEMPT_ARMS)
    if r["steps"] and r["mixed"] == 0 and not exempt:
        r["fails"].append("vacuous: no mixed-version row")
    if sanity and (r["continued"] > 0.05 * max(r["rows"], 1) or r["early"] > 0):
        r["fails"].append(f"acceptance sanity: {r['continued']} of {r['rows']} rows rejected, {r['early']} below position 8")
    # User rule 11-02: a router-frozen "unchanged ok" line on every verify step.
    if r["steps"] and r["router_ok"] + r["router_bad"] < r["steps"]:
        r["fails"].append(f"router frozen: missing on {r['steps'] - r['router_ok'] - r['router_bad']} of {r['steps']} steps")
    if r["errors"]:
        r["fails"].append("exception: " + r["errors"][0])
    arm0 = os.path.basename(os.path.dirname(log))
    if "edv" in arm0 or "defer_no_decide" in arm0:
        if r["edv_lines"] == 0:
            r["fails"].append("edv: no [edv] lines (EDV not on)")
        elif r["edv_deferred"] == 0 or r["edv_replay"] == 0:
            r["fails"].append(f"edv: vacuous (deferred={r['edv_deferred']} replayed={r['edv_replay']})")
    r["exact"] = not r["fails"]
    # Coverage (kept apart from fails: a negative arm must fail a CHECK, not merely lack coverage).
    arm = os.path.basename(os.path.dirname(log))
    r["coverage"] = []
    if arm.startswith("x1-neg_content_m_edv-") and r["edv_neg_rows"] < 20:
        r["coverage"].append(f"coverage: content_m_edv moved {r['edv_neg_rows']} EDV rows (need >= 20)")
    if arm.startswith("x1-edv_dp2-") and r["edv_replay"] == 0:
        r["coverage"].append("coverage: edv_dp2 replay_rows=0 (no deferred row was replayed at DP 2)")
    if arm.startswith("x1-edv_stress-") and r["edv_surprises"] == 0:
        r["coverage"].append("coverage: edv_stress surprises=0 (the surprise path did not run)")
    if arm.startswith("x1-prod_stress-") and r["continued"] == 0:
        r["coverage"].append("coverage: prod_stress continued=0 (the lag-1 continuation path did not run)")
    # Multi-token tail decode arms (tail_mtd): the extra rows ran and carried proposals (R: non-vacuous).
    if arm.startswith(("x1-mtd-", "x1-neg_mtd")) and (r["mtd_lines"] == 0 or r["mtd_proposed"] == 0):
        r["coverage"].append(f"coverage: tail_mtd lines={r['mtd_lines']} proposed={r['mtd_proposed']} "
                             f"extra_rows={r['mtd_extra']} (need proposed > 0)")
    if any(arm.startswith(f"x1-{a}-") or f"-{a}-" in arm for a in COVERAGE_ARMS):
        if r["mixed"] == 0 or r["mixed_cont"] == 0:
            r["coverage"].append(f"coverage: mixed_rows={r['mixed']} mixed_continued={r['mixed_cont']} (need both > 0)")
    return r


def judge(r, tag):
    """(ok, verdict) of one parsed run: a negative (tag contains 'neg') must fail a real check with coverage."""
    if "neg" in tag:
        real = [f for f in r["fails"] if not f.startswith(VACUITY_FAILS)]
        ok = bool(real) and not r["coverage"] and (
            r["neg_tokens"] > 0 or any("refus" in f or "thundersync.mode" in f for f in real))
        return ok, ("NEG PASS (failed as required)" if ok else "NEG FAIL (control did not fail or touched nothing)")
    ok = r["exact"] and not r["coverage"]
    return ok, ("EXACT" if ok else "NOT EXACT")


def main(paths):
    ok_all = True
    for p in paths:
        r = parse(p)
        tag = os.path.basename(os.path.dirname(r["log"]))
        ok, verdict = judge(r, tag)
        ok_all &= ok
        print(f"[nl] {tag}: {verdict} steps={r['steps']} mixed_rows={r['mixed']} continued={r['continued']} "
              f"mixed_continued={r['mixed_cont']} mm_tok={r['mm_tok']} eq_lines={r['eq_lines']} law_lines={r['law_lines']} "
              f"neg_tokens={r['neg_tokens']} s_calls={r['s_calls']}"
              + (f" mtd: iters={r['mtd_iters']} tokens={r['mtd_tokens']} extra_rows={r['mtd_extra']} "
                 f"proposed={r['mtd_proposed']}" if r["mtd_lines"] else "")
              + ("" if r["exact"] else f" fails={r['fails'][:4]}") + (f" {r['coverage'][0]}" if r["coverage"] else ""))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
