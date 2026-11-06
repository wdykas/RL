"""Frozen-router gate for the stock nrl arms (hard user rule, 11-02: router gate weights and router expert bias frozen
in every RL run).

  config_frozen(cfg_dict) -> (ok, reason)   effective config: policy.megatron_cfg.freeze_moe_router is true (NeMo-RL
                                            sets router.weight.requires_grad=False before wrapping, setup.py) and
                                            policy.megatron_cfg.moe_router_bias_update_rate == 0
  run_frozen(run_dir) -> (ok, reason)       the run itself: effective_config.yaml passes config_frozen, and run.log has
                                            one `[nrl-bias]` and one `[nrl-router]` probe line (bias_probe/
                                            sitecustomize.py) per optimizer step, every one with rate 0, bias
                                            max_abs_delta 0, n_router > 0, requires_grad 0, equal_first True
  python3 frozen_router.py <run dir>...     prints PASS / FAIL per run (exit 1 on any FAIL)
"""
import json
import os
import re
import sys

BIAS = re.compile(r"\[nrl-bias\] call=(\d+) rate=(\S+) n_bias=(\d+) numel=(\d+) max_abs_delta=(\S+)")
ROUTER = re.compile(r"\[nrl-router\] call=(\d+) n_router=(\d+) requires_grad=(\d+) numel=(\d+) equal_first=(\w+) "
                    r"max_abs_from_first=(\S+)")


def config_frozen(c):
    m = (c.get("policy") or {}).get("megatron_cfg") or {}
    if m.get("freeze_moe_router") is not True:
        return False, f"policy.megatron_cfg.freeze_moe_router={m.get('freeze_moe_router')!r} (must be true)"
    rate = m.get("moe_router_bias_update_rate")
    try:
        if float(rate) != 0.0:
            return False, f"policy.megatron_cfg.moe_router_bias_update_rate={rate!r} (must be 0)"
    except (TypeError, ValueError):
        return False, f"policy.megatron_cfg.moe_router_bias_update_rate={rate!r} (must be 0)"
    return True, "router: frozen (weights+bias)"


def run_frozen(d):
    try:
        import yaml
        c = yaml.safe_load(open(os.path.join(d, "effective_config.yaml")))
    except Exception as e:  # noqa: BLE001
        return False, f"no readable effective_config.yaml ({type(e).__name__})"
    ok, why = config_frozen(c)
    if not ok:
        return False, why
    try:
        steps = len(json.load(open(os.path.join(d, "history.json"))))
    except (OSError, ValueError):
        return False, "no history.json"
    b, r = [], []
    try:
        with open(os.path.join(d, "run.log"), errors="replace") as f:
            for line in f:
                if "[nrl-bias] call=" in line and (m := BIAS.search(line)):
                    b.append(m.groups())
                elif "[nrl-router] call=" in line and (m := ROUTER.search(line)):
                    r.append(m.groups())
    except OSError:
        return False, "no run.log"
    if steps == 0 or not b or not r:  # a run with no steps / no probe lines is no evidence (was an IndexError)
        return False, f"no evidence: {steps} steps, probe lines bias {len(b)} / router {len(r)}"
    if len(b) < steps or len(r) < steps:
        return False, f"probe lines bias {len(b)} / router {len(r)} < {steps} steps (no frozen-router probe)"
    bad_b = [x for x in b if float(x[1]) != 0.0 or float(x[4]) != 0.0 or int(x[2]) == 0]
    bad_r = [x for x in r if int(x[1]) == 0 or int(x[2]) != 0 or x[4] != "True" or float(x[5]) != 0.0]
    if bad_b or bad_r:
        return False, f"probe shows a non-frozen router: {len(bad_b)} bias lines, {len(bad_r)} router lines bad"
    return True, (f"router: frozen (weights+bias); {len(b)} bias + {len(r)} router probe lines, n_bias {b[-1][2]}, "
                  f"n_router {r[-1][1]} ({r[-1][3]} weights), all unchanged")


if __name__ == "__main__":
    bad = 0
    for d in sys.argv[1:]:
        ok, why = run_frozen(d)
        bad += not ok
        print(f"{'PASS' if ok else 'FAIL'} {os.path.basename(os.path.normpath(d))}: {why}")
    sys.exit(1 if bad else 0)
