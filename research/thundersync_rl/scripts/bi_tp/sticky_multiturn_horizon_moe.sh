#!/bin/bash
# Multi-turn rollouts with the sticky anchor horizon (DESIGN.md sections 14h + 14i): turns below the
# horizon run at theta_{k-1} (scheme A), the rest of a drafted rollout (later turns included) on the
# refit engine with its anchors re-applied below the horizon (engine anchor apply).
# sticky_multiturn_moe.sh (2+2 GPUs, tiny MoE EP=2, stochastic toy environment) with HORIZON (160:
# above the first turn's 128-token budget, so environment steps happen on both sides of it).
# Pass (exit 0): sticky_multiturn_moe.sh passes (mismatch 0 over every assistant token of every turn,
# route checks exact incl. observation positions, ...) and from step 1 on plans are handed to the
# refit engine (spec/horizon_plans > 0) which finished turns of them (spec/horizon_engine_turns > 0),
# and by the last step the engine applied anchors (spec/anchor_applied_tokens > 0).
# NOAPPLY=1: control (THUNDERSYNC_DEBUG_NO_ANCHOR_APPLY=1), expected to FAIL the mismatch check.
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
HORIZON=${HORIZON:-160} NOAPPLY=${NOAPPLY:-0} TURNS=${TURNS:-3} TURN_TOKENS=${TURN_TOKENS:-128}
export RUNS=${RUNS:-runs_sticky_multiturn_horizon} TURNS TURN_TOKENS
export TAG=${TAG:-mt$TURNS-t$TURN_TOKENS-h$HORIZON-noapply$NOAPPLY-$(basename "${MODEL:-tiny-qwen3-moe-d2048}")}
if [ "$NOAPPLY" = 1 ]; then export THUNDERSYNC_DEBUG_NO_ANCHOR_APPLY=1; fi
export ARGS="++thundersync.sticky_anchor_horizon=$HORIZON ${ARGS:-}"
bash "$HERE/sticky_multiturn_moe.sh"
rc=$?
python - "$W/$RUNS/$TAG/history.json" <<'PY'
import json, sys
try:
    hist = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    sys.exit(f"[mt horizon] no history ({exc})")
why = []
for i, h in enumerate(hist):
    keys = ("spec/horizon_plans", "spec/horizon_engine_turns", "spec/anchor_applied_tokens",
            "spec/anchor_loaded_requests", "spec/sticky_env_turns", "mean_turns", "max_turns")
    print(f"[mt horizon] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if i >= 1 and not (h.get("spec/horizon_plans") or 0) > 0:
        why.append(f"step {i}: no plans continued by the refit engine")
    if i >= 1 and not (h.get("spec/horizon_engine_turns") or 0) > 0:
        why.append(f"step {i}: no turn finished on the refit engine")
if not (hist and (hist[-1].get("spec/anchor_applied_tokens") or 0) > 0):
    why.append("the engine never applied an anchor")
print(f"[mt horizon] {'OK' if not why else 'FAIL: ' + '; '.join(why)}")
sys.exit(0 if not why else 1)
PY
src=$?
exit $(( rc != 0 ? rc : src ))
