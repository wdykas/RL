"""Auditor: classify effective_config differences between two arms.
  python3 pair_check.py <runA> <runB>   (run dir or tag under $COLO_RUNS)
Keys: BI (allowed SPR vs SPR-BI), METHOD (allowed SPR-BI vs ours), else VIOLATION."""
import os, re, sys, yaml
RUNS = [d for d in (os.environ.get("COLO_RUNS"),) if d]  # the colo kit's run root (env.sh)
BI = re.compile(r"batch_invariant|moe_permute_fusion|attention_backend|flash_attention_version|use_fused_linear_logprobs|"
                r"forward_gemm_precision|fp32_residual_connection|zero_train_gen_mismatch|logprobs_mode|CUBLAS|lm_head_precision|"
                r"moe_ep_invariant_combine|mamba_fp32_prenorm|precision_raise|down_proj_prescale|NEMO_RL_VOCAB_LSE|"
                # BI-only (11-06): our BI MoE dispatch path; mamba_recompute only acts inside the BI SSD path
                # (zero_train_gen_mismatch installs mamba_ssd_parity), a no-op on the stock learner
                r"NEMO_RL_MOE_DEVICE_DISPATCH|mamba_recompute")
METHOD = re.compile(r"^thundersync\.|THUNDERSYNC_|num_prompts_per_step|max_lookahead_versions|max_inflight_prompts|max_buffered_rollouts|"
                    r"min_groups_for_streaming_train|rollout_failure|async_sched_mode|stall_timeout|progress_log|"
                    r"train_global_batch_size|"
                    # IS-SPR (configs/colo/colo_isspr.yaml): the constant loss normalizer its group weight needs
                    r"^loss_fn\.token_level_loss$|"
                    # IS-RLOO (configs/colo/colo_isrloo.yaml): the RLOO advantage its row weights are exact for
                    # (objective keys: they change the estimator, not the speed path)
                    r"^grpo\.(adv_estimator\.)?(normalize_rewards|use_leave_one_out_baseline)$|^grpo\.adv_estimator\.name$|"
                    # colo / nolanes keep the learner's grad buffers resident through the engine phase (worker.py)
                    r"^policy\.megatron_cfg\.resident_learner$")
IGN = re.compile(r"log_dir|checkpoint_dir|wandb|expose_http_server")
def find(t):
    for b in RUNS:
        p = os.path.join(b, t)
        if os.path.exists(os.path.join(p, "effective_config.yaml")): return p
    return t
def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items(): o.update(flat(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(d, list) and any(isinstance(x, (dict, list)) for x in d):
        for i, v in enumerate(d): o.update(flat(v, f"{p}[{i}]"))
    else: o[p] = d
    return o
a, b = (flat(yaml.safe_load(open(os.path.join(find(x), "effective_config.yaml")))) for x in sys.argv[1:3])
for k in sorted(set(a) | set(b)):
    va, vb = a.get(k, "<absent>"), b.get(k, "<absent>")
    if va == vb or IGN.search(k): continue
    cls = "BI" if BI.search(k) else "METHOD" if METHOD.search(k) else "VIOLATION"
    print(f"{cls:9s} {k}: {str(va)[:60]} | {str(vb)[:60]}")
print("NOTE: METHOD batch keys (num_prompts/train_global_batch_size) still require equal trained rows: check num_valid_samples in run.log")
