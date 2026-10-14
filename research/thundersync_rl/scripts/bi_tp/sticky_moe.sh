#!/bin/bash
# Sticky MoE routing anchored to the drafter (thundersync.sticky_routing, DESIGN.md section 14c)
# on top of engine-sourced q under batch invariance (engine_q_tp.sh). 2 inference + 2 training
# GPUs of one node, short run (STEPS=4 default; do not use for long perf runs).
#
#   bash research/thundersync_rl/scripts/bi_tp/sticky_moe.sh                      # tiny MoE, EP=2
#   MODEL=$W/models/qwen3-30b-a3b-L4 EP=2 bash .../sticky_moe.sh                  # real 30B-A3B layers
#   STICKY=0 bash .../sticky_moe.sh                                                # A/B: same run, plain routing
#
# THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE (NOISE, default 1) perturbs expert gradients so weights
# (and routes) change every step even when every reward is 0 (truncated or random models).
# Pass (exit 0), as engine_q_tp.sh: every step mismatch/max_abs_logprob_diff == 0 and
# exact_token_frac == 1 (rollout log-probs == the learner's training forward with the same
# anchors), every "[engine q check]" exact, no scorer fallback; plus non-vacuity: the learner's
# parameter checksum changes every step and every step after the first accepts draft tokens;
# with STICKY=1, every step that verified drafts reports spec/sticky_rounds (continuations
# completed at theta_{k-1}), its anchors equal a fresh engine prefill of the same rows
# ("[route check]", spec/route_check_exact == spec/route_check_tokens > 0), and the sticky rule
# changed some route against plain top-k ("[sticky check]" differs_from_plain > 0).
# Compare spec/accepted_frac and spec/full_accept against STICKY=0.
#
# DRAFT_WEIGHTS=momentum (thundersync.draft_weights, DESIGN.md section 14f): the engine drafts
# with the momentum-predicted weights; additionally every step that drafted must report
# a draft-weights refit, and from step 1 on draft/changed_frac > 0 (the draft differs from theta_k).
#
# Env: MODEL, EP (2), TP (1), PP (1), STEPS (4), PROMPTS (16), MAXLEN (1024), DELTA (0.03),
# STICKY (1), NOISE (expert-grad noise, 1), SCHED (engine scheduling: async | legacy; async),
# DRAFT_WEIGHTS (current), ARGS (extra overrides).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=${MODEL:-$W/models/tiny-qwen3-moe-d2048}
export EP=${EP:-2} TP=${TP:-1} PP=${PP:-1} MAXLEN=${MAXLEN:-1024} STEPS=${STEPS:-4}
STICKY=${STICKY:-1} DELTA=${DELTA:-0.03} NOISE=${NOISE:-1} DRAFT_WEIGHTS=${DRAFT_WEIGHTS:-current}
if [ "$NOISE" != 0 ]; then export THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=$NOISE; fi
export RUNS=${RUNS:-runs_sticky_moe}
SCHED=${SCHED:-async}
export TAG=${TAG:-sticky$STICKY-d$DELTA-ep$EP-tp$TP-pp$PP-$SCHED-dw$DRAFT_WEIGHTS-$(basename "$MODEL")}
STICKY_ARGS="++thundersync.draft_lookahead=1 ++policy.generation.mcore_generation_config.async_sched_mode=$SCHED"
STICKY_ARGS+=" ++thundersync.draft_weights=$DRAFT_WEIGHTS"
if [ "$STICKY" = 1 ]; then
  STICKY_ARGS+=" ++thundersync.sticky_routing=true ++thundersync.sticky_routing_delta=$DELTA"
fi
# fp32 router on both sides (inference_optimized requires it; training must match).
export ARGS="policy.megatron_cfg.moe_router_dtype=fp32 policy.generation.mcore_generation_config.enable_prefix_caching=false $STICKY_ARGS ${ARGS:-}"
bash "$HERE/engine_q_tp.sh"
rc=$?
OUT=$W/$RUNS/$TAG
python - "$OUT/history.json" "$OUT/run.log" "$STICKY" "$DRAFT_WEIGHTS" <<'EOF'
import json, re, sys
path, log, sticky = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
momentum = sys.argv[4] == "momentum"
try:
    hist = json.load(open(path))
except (OSError, ValueError) as exc:
    sys.exit(f"[sticky] no history ({exc})")
why = []
for i, h in enumerate(hist):
    keys = ("spec/draft_tokens", "spec/accepted_frac", "spec/full_accept", "spec/sticky_rounds",
            "spec/sticky_cont_draft_tokens", "spec/sticky_cont_accepted_frac",
            "spec/sticky_cont_retries", "spec/route_truncated_drafts", "spec/engine_q_dropped",
            "spec/route_check_tokens", "spec/route_check_exact", "mismatch/param_checksum",
            "draft/changed_frac", "draft/total_s", "draft/peak_extra_gb",
            "time/verify", "time/iteration")
    print(f"[sticky] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if sticky and h.get("spec/draft_tokens") and h.get("spec/sticky_rounds") is None:
        why.append(f"step {i}: no sticky stats")
    if i >= 1 and not (h.get("spec/draft_tokens") or 0) * (h.get("spec/accepted_frac") or 0) > 0:
        why.append(f"step {i}: no accepted draft tokens")
    if i >= 1 and h.get("mismatch/param_checksum") == hist[i - 1].get("mismatch/param_checksum"):
        why.append(f"step {i}: weights unchanged")
    if momentum and i + 1 < len(hist):  # every step but the last drafts the next one
        if h.get("draft/total_s") is None:
            why.append(f"step {i}: no draft-weights refit")
        # Step 0 has no momentum yet (theta_hat = theta_0 - lr wd theta_0, below a bf16 ulp).
        elif i >= 1 and not h.get("draft/changed_frac", 0) > 0:
            why.append(f"step {i}: momentum draft weights equal theta_k")
    tok, exact = h.get("spec/route_check_tokens"), h.get("spec/route_check_exact")
    if sticky and i >= 1 and not (tok and exact == tok):
        why.append(f"step {i}: route check {exact}/{tok}")
text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())  # Ray colors
checks = [(int(n), float(d)) for n, d in re.findall(
    r"\[sticky check\].*token_layers=(\d+) .*differs_from_plain=(\S+)", text)]
if sticky:
    print(f"[sticky] sticky checks: {len(checks)} lines, token_layers={sum(n for n, _ in checks)}, "
          f"max differs_from_plain={max((d for _, d in checks), default=None)}")
    if not any(n > 0 and d > 0 for n, d in checks):
        why.append("the sticky rule never changed a route against plain top-k")
ok = not why
print(f"[sticky] {'OK' if ok else 'FAIL: ' + '; '.join(why)}")
sys.exit(0 if ok else 1)
EOF
src=$?
exit $(( rc != 0 ? rc : src ))
