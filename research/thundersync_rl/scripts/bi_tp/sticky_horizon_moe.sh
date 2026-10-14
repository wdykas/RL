#!/bin/bash
# Sticky routing with an anchor horizon (thundersync.sticky_anchor_horizon, DESIGN.md section 14i):
# sticky_moe.sh (tiny MoE, EP=2, 2+2 GPUs, short run) with HORIZON (default 64) generated positions.
# theta_{k-1} work per drafted rollout ends at the horizon; the refit engine continues every such
# rollout, re-applying its anchors below the horizon in its forwards (engine anchor apply).
# Pass (exit 0): everything sticky_moe.sh checks (mismatch 0 every step: the engine's log-probs of
# the horizon continuations equal the learner's training forward with the same anchors; route check;
# sticky check) plus non-vacuity of the horizon: from step 1 on plans are handed to the refit engine
# (spec/horizon_plans > 0) and by the last step the engine applied anchors to tokens
# (spec/anchor_applied_tokens > 0).
# NOAPPLY=1: control, the engine never applies anchors (THUNDERSYNC_DEBUG_NO_ANCHOR_APPLY=1); the
# mismatch check must then FAIL (expected exit != 0).
# Env: HORIZON (64), NOAPPLY (0), everything sticky_moe.sh takes.
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HORIZON=${HORIZON:-64} NOAPPLY=${NOAPPLY:-0}
export RUNS=${RUNS:-runs_sticky_horizon}
export EP=${EP:-2} TP=${TP:-1} PP=${PP:-1} SCHED=${SCHED:-async}
MODEL_NAME=$(basename "${MODEL:-$W/models/tiny-qwen3-moe-d2048}")
export TAG=${TAG:-horizon$HORIZON-noapply$NOAPPLY-ep$EP-tp$TP-pp$PP-$SCHED-$MODEL_NAME}
if [ "$NOAPPLY" = 1 ]; then export THUNDERSYNC_DEBUG_NO_ANCHOR_APPLY=1; fi
export ARGS="++thundersync.sticky_anchor_horizon=$HORIZON ${ARGS:-}"
bash "$HERE/sticky_moe.sh"
rc=$?
OUT=$W/$RUNS/$TAG
python - "$OUT/history.json" <<'PY'
import json, sys
try:
    hist = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    sys.exit(f"[horizon] no history ({exc})")
why = []
for i, h in enumerate(hist):
    keys = ("spec/horizon_plans", "spec/anchor_applied_tokens", "spec/anchor_loaded_requests",
            "spec/anchor_pool_bytes", "spec/anchor_buffer_bytes", "spec/anchor_fetch_s",
            "spec/sticky_streams", "spec/sticky_cont_draft_tokens",
            "mismatch/max_abs_logprob_diff", "mismatch/exact_token_frac")
    print(f"[horizon] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if i >= 1 and not (h.get("spec/horizon_plans") or 0) > 0:
        why.append(f"step {i}: no plans continued by the refit engine")
if not (hist and (hist[-1].get("spec/anchor_applied_tokens") or 0) > 0):
    why.append("the engine never applied an anchor")
print(f"[horizon] {'OK' if not why else 'FAIL: ' + '; '.join(why)}")
sys.exit(0 if not why else 1)
PY
src=$?
exit $(( rc != 0 ? rc : src ))
