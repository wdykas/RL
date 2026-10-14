#!/bin/bash
# Pooled null-augmented block verification (thundersync.null_bv, DESIGN.md section 14d) on top of
# sticky MoE routing + engine-sourced q under batch invariance (sticky_moe.sh -> engine_q_tp.sh).
# 2 inference + 2 training GPUs of one node, short run (STEPS=4 default; not for perf runs).
# Wrap in the exclusive node lock: flock thundersync_work/gpu.lock bash .../null_bv_moe.sh
#
#   bash research/thundersync_rl/scripts/bi_tp/null_bv_moe.sh                 # tiny MoE, EP=2, K=1.5, M=2
#   FIXED_M=dp bash .../null_bv_moe.sh                                        # DP envelope policy
#   STICKY=0 bash .../null_bv_moe.sh                                          # ⊥-BV without sticky routing
#
# The engine drafts N = round(K G) drafts per GRPO group (G = 8 here); each prompt's drafts are
# verified in drafting order against p/M (+) (1-1/M) BOT until its G slots are filled; leftover
# slots decode fresh. FIXED_M=2 (default) makes BOT frequent (~1/2 of decided drafts) so the
# exactness check exercises it; the DP would pick M ~ 1 on this short-cap toy run.
# Pass (exit 0): engine_q_tp.sh's checks (every step mismatch/max_abs_logprob_diff == 0 and
# exact_token_frac == 1, engine q exact, no scorer fallback), sticky_moe.sh's checks, and every
# step that verified drafts reports the ⊥-BV statistics with outputs <= slots, and every step
# after the first has null_drafts > 0 and null_outputs > 0 (no vacuous all-fresh pass).
#
# Env: K (1.5), FIXED_M (2.0 | dp), STICKY (1), plus everything sticky_moe.sh takes
# (MODEL, EP, TP, STEPS, PROMPTS, MAXLEN, DELTA, NOISE, DRAFT_WEIGHTS, ARGS).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
K=${K:-1.5} FIXED_M=${FIXED_M:-2.0}
export STICKY=${STICKY:-1}
export RUNS=${RUNS:-runs_null_bv}
export DRAFT_WEIGHTS=${DRAFT_WEIGHTS:-current}
export TAG=${TAG:-nullbv-k$K-m$FIXED_M-sticky$STICKY-ep${EP:-2}-dw$DRAFT_WEIGHTS-$(basename "${MODEL:-tiny-qwen3-moe-d2048}")}
NULL_ARGS="++thundersync.null_bv=true ++thundersync.null_bv_drafts_per_output=$K"
if [ "$FIXED_M" != dp ]; then NULL_ARGS+=" ++thundersync.null_bv_fixed_m=$FIXED_M"; fi
export ARGS="$NULL_ARGS ${ARGS:-}"
bash "$HERE/sticky_moe.sh"
rc=$?
OUT=$W/$RUNS/$TAG
python - "$OUT/history.json" <<'PY'
import json, sys
try:
    hist = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    sys.exit(f"[null_bv] no history ({exc})")
ok, seen = True, 0
keys = ("spec/null_drafts", "spec/null_slots", "spec/null_outputs", "spec/null_bot_frac",
        "spec/null_tail_frac", "spec/null_fresh_slots", "spec/null_unused_drafts",
        "spec/null_drafts_per_output", "spec/null_mean_m", "spec/null_longest_min_reject_frac",
        "spec/null_empty_dropped", "spec/accepted_frac", "mismatch/max_abs_logprob_diff",
        "spec/null_dropped_prompts", "spec/drop_is_ess", "spec/drop_is_weighted_frac")
for i, h in enumerate(hist):
    print(f"[null_bv] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if h.get("spec/null_slots") is not None:
        seen += 1
        ok &= h["spec/null_outputs"] <= h["spec/null_slots"]
        # prompt drop (opt-in): a dropped prompt's G = 8 slots are neither filled nor fresh
        dropped = 8 * (h.get("spec/null_dropped_prompts") or 0)
        ok &= h["spec/null_outputs"] + h["spec/null_fresh_slots"] + dropped == h["spec/null_slots"]
    if i >= 1:  # drafts reach the pools and fill slots (no vacuous all-fresh pass)
        ok &= (h.get("spec/null_drafts") or 0) > 0 and (h.get("spec/null_outputs") or 0) > 0
ok &= seen >= max(1, len(hist) - 1)  # every step after the first verifies drafts
print(f"[null_bv] {'OK' if ok else 'FAIL'} ({seen} steps with ⊥-BV pools)")
sys.exit(0 if ok else 1)
PY
src=$?
exit $(( rc != 0 ? rc : src ))
