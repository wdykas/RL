#!/bin/bash
# Sticky MoE routing anchored to the drafter (thundersync.sticky_routing, DESIGN.md section 14c)
# on top of engine-sourced q under batch invariance (engine_q_tp.sh). 2 inference + 2 training
# GPUs of one node, short run (STEPS=4 default; do not use for long perf runs).
#
#   bash research/thundersync_rl/scripts/bi_tp/sticky_moe.sh                      # tiny MoE, EP=2
#   MODEL=$W/models/qwen3-30b-a3b-L4 EP=2 bash .../sticky_moe.sh                  # real 30B-A3B layers
#   STICKY=0 bash .../sticky_moe.sh                                                # A/B: same run, plain routing
#
# The tiny model is random: THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE (default 1 for it) perturbs
# expert gradients so weights (and routes) change every step.
# Pass (exit 0), as engine_q_tp.sh: every step mismatch/max_abs_logprob_diff == 0 and
# exact_token_frac == 1 (rollout log-probs == the learner's training forward with the same
# anchors), every "[engine q check]" exact, no scorer fallback; plus, with STICKY=1, every step
# that verified drafts reports spec/sticky_rounds (continuations completed at theta_{k-1}).
# Compare spec/accepted_frac and spec/full_accept against STICKY=0.
#
# Env: MODEL, EP (2), TP (1), STEPS (4), PROMPTS (16), MAXLEN (1024), DELTA (0.03), STICKY (1),
# NOISE (expert-grad noise; default 1 for the tiny model, 0 otherwise), ARGS (extra overrides).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export MODEL=${MODEL:-$W/models/tiny-qwen3-moe-d2048}
export EP=${EP:-2} TP=${TP:-1} PP=${PP:-1} MAXLEN=${MAXLEN:-1024} STEPS=${STEPS:-4}
STICKY=${STICKY:-1} DELTA=${DELTA:-0.03}
case "$MODEL" in
  *tiny*) NOISE=${NOISE:-1} ;;
  *) NOISE=${NOISE:-0} ;;
esac
if [ "$NOISE" != 0 ]; then export THUNDERSYNC_DEBUG_EXPERT_GRAD_NOISE=$NOISE; fi
export RUNS=${RUNS:-runs_sticky_moe}
export TAG=${TAG:-sticky$STICKY-d$DELTA-ep$EP-tp$TP-$(basename "$MODEL")}
STICKY_ARGS="++thundersync.draft_lookahead=1"
if [ "$STICKY" = 1 ]; then
  STICKY_ARGS+=" ++thundersync.sticky_routing=true ++thundersync.sticky_routing_delta=$DELTA"
fi
# fp32 router on both sides (inference_optimized requires it; training must match).
export ARGS="policy.megatron_cfg.moe_router_dtype=fp32 policy.generation.mcore_generation_config.enable_prefix_caching=false $STICKY_ARGS ${ARGS:-}"
bash "$HERE/engine_q_tp.sh"
rc=$?
OUT=$W/$RUNS/$TAG
python - "$OUT/history.json" "$STICKY" <<'EOF'
import json, sys
path, sticky = sys.argv[1], sys.argv[2] == "1"
try:
    hist = json.load(open(path))
except (OSError, ValueError) as exc:
    sys.exit(f"[sticky] no history ({exc})")
ok = True
for i, h in enumerate(hist):
    keys = ("spec/accepted_frac", "spec/full_accept", "spec/sticky_rounds",
            "spec/sticky_cont_draft_tokens", "spec/sticky_cont_accepted_frac",
            "spec/sticky_cont_retries", "spec/route_truncated_drafts", "spec/engine_q_dropped",
            "time/verify", "time/iteration")
    print(f"[sticky] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if sticky and h.get("spec/draft_tokens") and h.get("spec/sticky_rounds") is None:
        ok = False
print(f"[sticky] {'OK' if ok else 'MISSING sticky stats'}")
sys.exit(0 if ok else 1)
EOF
src=$?
exit $(( rc != 0 ? rc : src ))
