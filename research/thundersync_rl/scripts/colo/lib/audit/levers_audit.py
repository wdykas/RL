"""Levers audit: every lever (docs/colo/DECISIONS.md) an arm must carry, checked on its effective_config.yaml, PASS / FAIL each.
  python3 levers_audit.py <run> [--ref <baseline run>]      (run dir or tag under $COLO_RUNS; exit 1 on FAIL)
Arm classes: a ThunderSync arm has thundersync.mode; it is a stale-row (speculative) arm when it also has lag >= 1.
Baseline modes (BASELINE_MODES: sprsd = SPR-BI's stock loop + the XS-SD drafter, no verification) are ThunderSync arms
for the bi lever but never stale-row arms: they verify nothing. Like SPR-BI they carry the cross-fade band (F3).
--ref (the SPR-BI arm of the pair) adds the levers that must equal the baseline's: engine shape (PC / CP, KV mode,
graph set, MR, buffer), learner shape (MBS, lr schedule) and the BI recipe.
Levers:
  routers      freeze_moe_router true and moe_router_bias_update_rate exactly 0 (every arm; user rule 11-02)
  bi           batch_invariant_mode + zero_train_gen_mismatch (ThunderSync arms; the baseline when --ref is BI)
  crossfade    the cross-fade band is part of the policy definition (F3, 11-06): EVERY arm (XS, the stock SPR-BI and
               every ThunderSync arm) routes with crossfade_routing_m > 0, sparse + sparse_engine. The band is read
               from policy.megatron_cfg (the stock arm; ThunderSync arms once run_nolanes_sc dumps the applied band),
               else from a ThunderSync arm's thundersync block (older dumps). Off only with
               thundersync.crossfade_off_reason (debug runs).
  offload      policy.offload_optimizer_for_refit false (resident colocated learner; every arm)
  dispatch     NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS >= train MBS x max_total_sequence_length (device-dispatch rule)
  mr           engine max_requests x engine DP >= (lag + 1) x P x G: no row waits or is evicted (D5)
  temperature  generation temperature 1 (stale-row arms: q is the raw log-prob law)
  baseline     a BASELINE_MODES arm carries no thundersync key beyond BASELINE_KEYS (no stale-row stage key)
  ref:*        equal to the baseline: PC (prefix caching), CP (chunked prefill), KV mode, num_cuda_graphs,
               cuda_graph_all_prefills, max_requests, buffer_size_gb, train MBS, lr / min_lr / warmup, BI keys, and
               the cross-fade band (ref:crossfade_band: m, per-layer profile, extra, sparse, sparse_engine, stopgrad)
"""
import os
import sys

import yaml

RUNS = [d for d in (os.environ.get("COLO_RUNS"),) if d]  # the colo kit's run root (env.sh)
BASELINE_MODES = ("sprsd",)  # baseline arms with a thundersync.mode: no stale-row verification (see docstring)
# The only thundersync keys a baseline-mode arm may carry (the engine drafter); any stale-row stage key (wake_verify,
# k_drafts, null_bv_*, spare_prompts, variable_group, alternating, IS-SPR weight / clip, thundersync cross-fade, ...)
# fails "baseline". Its cross-fade band lives in policy.megatron_cfg, as SPR-BI's.
BASELINE_KEYS = {"sprsd": {"mode", "tree_xssd", "tree_xssd_mtp"}}
REF_KEYS = (
    "policy.generation.mcore_generation_config.enable_prefix_caching",
    "policy.generation.mcore_generation_config.enable_chunked_prefill",
    "policy.generation.mcore_generation_config.kv_cache_management_mode",
    "policy.generation.mcore_generation_config.num_cuda_graphs",
    "policy.generation.mcore_generation_config.cuda_graph_all_prefills",
    "policy.generation.mcore_generation_config.max_requests",
    "policy.generation.mcore_generation_config.buffer_size_gb",
    "policy.train_micro_batch_size",
    "policy.megatron_cfg.optimizer.lr",
    "policy.megatron_cfg.optimizer.min_lr",
    "policy.megatron_cfg.scheduler.lr_warmup_init",
    "policy.megatron_cfg.batch_invariant_mode",
    "policy.megatron_cfg.forward_gemm_precision",
    "policy.megatron_cfg.zero_train_gen_mismatch",
)


def find(t):
    for b in RUNS:
        p = os.path.join(b, t)
        if os.path.exists(os.path.join(p, "effective_config.yaml")):
            return p
    return t


def load(run):
    return yaml.safe_load(open(os.path.join(find(run), "effective_config.yaml")))


def get(d, path, default=None):
    cur = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


BAND_KEYS = ("crossfade_routing_m", "crossfade_routing_m_profile", "crossfade_routing_extra",
             "crossfade_routing_sparse", "crossfade_routing_sparse_engine", "crossfade_routing_ramp_stopgrad")


def band(c):
    """The cross-fade band an arm routes with: policy.megatron_cfg when it holds the band (the stock arm; a ThunderSync
    mode's dump after run_nolanes_sc.append_applied_band), else the mode's thundersync block (older dumps, written
    before configure_driver applied it)."""
    mc = get(c, "policy.megatron_cfg") or {}
    ts = get(c, "thundersync") or {}
    src = mc if "crossfade_routing_m" in mc or not (isinstance(ts, dict) and ts.get("mode")) else ts
    return {k: src.get(k) for k in BAND_KEYS}


def audit(c, ref=None):
    """[(lever, ok, detail)] of one effective config (and the baseline's, for the ref:* levers)."""
    out = []
    mc = get(c, "policy.megatron_cfg", {}) or {}
    mcore = get(c, "policy.generation.mcore_generation_config", {}) or {}
    ts = get(c, "thundersync") or {}
    mode = ts.get("mode") if isinstance(ts, dict) else None
    lag = int(get(c, "async_rl.sampler.max_lookahead_versions", 0) or 0)
    stale = mode is not None and mode not in BASELINE_MODES and lag >= 1
    rate = mc.get("moe_router_bias_update_rate")
    out.append(("routers", mc.get("freeze_moe_router") is True and rate is not None and float(rate) == 0.0,
                f"freeze_moe_router={mc.get('freeze_moe_router')} bias_rate={rate}"))
    if mode is not None:
        out.append(("bi", bool(mc.get("batch_invariant_mode")) and bool(mc.get("zero_train_gen_mismatch")),
                    f"batch_invariant_mode={mc.get('batch_invariant_mode')} "
                    f"zero_train_gen_mismatch={mc.get('zero_train_gen_mismatch')}"))
    if mode in BASELINE_MODES:
        extra = sorted(set(ts) - BASELINE_KEYS[mode])
        out.append(("baseline", not extra, f"mode={mode} keys beyond the drafter: {extra}"))
    b = band(c)
    m = b["crossfade_routing_m"]
    off = str((ts.get("crossfade_off_reason") if isinstance(ts, dict) else None) or "").strip()
    ok = (m is not None and float(m) > 0 and b["crossfade_routing_sparse"] is True
          and b["crossfade_routing_sparse_engine"] is True) or (m is None and bool(off))
    out.append(("crossfade", ok, f"m={m} profile={b['crossfade_routing_m_profile']} extra={b['crossfade_routing_extra']} "
                                 f"sparse={b['crossfade_routing_sparse']} sparse_engine={b['crossfade_routing_sparse_engine']}"
                                 + (f" OFF: {off}" if off and m is None else "")))
    if stale:
        t = float(get(c, "policy.generation.temperature", 1.0))
        out.append(("temperature", t == 1.0, f"temperature={t}"))
    out.append(("offload", get(c, "policy.offload_optimizer_for_refit") is False,
                f"offload_optimizer_for_refit={get(c, 'policy.offload_optimizer_for_refit')}"))
    tok = (mc.get("env_vars") or {}).get("NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS")
    need = int(get(c, "policy.train_micro_batch_size", 1)) * int(get(c, "policy.max_total_sequence_length", 0))
    if tok is not None:
        out.append(("dispatch", int(tok) >= need, f"tokens={tok} need={need}"))
    p, g = int(get(c, "grpo.num_prompts_per_step", 0)), int(get(c, "grpo.num_generations_per_prompt", 0))
    gpus = int(get(c, "cluster.gpus_per_node", 0)) * int(get(c, "cluster.num_nodes", 0))
    dp = gpus // max(1, int(mcore.get("tensor_model_parallel_size", 1)) * int(mcore.get("pipeline_model_parallel_size", 1)))
    mr = mcore.get("max_requests")
    if mr is not None and dp:
        out.append(("mr", int(mr) * dp >= (lag + 1) * p * g, f"max_requests={mr} x DP {dp} vs (lag+1) P G = "
                                                             f"{(lag + 1) * p * g}"))
    if ref is not None:
        out.append(("ref:crossfade_band", band(c) == band(ref), f"{band(c)} vs ref {band(ref)}"))
        for k in REF_KEYS:
            a, b = get(c, k, "<absent>"), get(ref, k, "<absent>")
            out.append((f"ref:{k.split('.')[-1]}", a == b, f"{a} vs ref {b}"))
    return out


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    ref = None
    if "--ref" in argv:
        i = argv.index("--ref")
        ref = load(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    res = audit(load(argv[0]), ref)
    for lever, ok, detail in res:
        print(f"{'PASS' if ok else 'FAIL'} {lever:28s} {detail}")
    bad = [lever for lever, ok, _ in res if not ok]
    print(f"[levers] {argv[0]}: " + ("PASS" if not bad else f"FAIL {bad}"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
