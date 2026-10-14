#!/bin/bash
# Multi-turn rollouts under sticky MoE routing, scheme A (DESIGN.md section 14h): every turn of a
# drafted rollout is sampled at theta_{k-1} before the refit, the driver steps the environment at
# verified turn ends, the rollout loop replays the recorded turns. sticky_moe.sh (2+2 GPUs, tiny
# MoE EP=2, STEPS=4) with the stochastic multi-turn toy environment (toy_multi_turn_env.py,
# TURNS turns at most) and a per-turn budget (TURN_TOKENS) small enough for several turns.
#
#   bash research/thundersync_rl/scripts/bi_tp/sticky_multiturn_moe.sh
#
# Pass (exit 0): sticky_moe.sh passes (mismatch 0 / exact_token_frac 1 over every assistant token of
# every turn, engine q checks exact, route check of the drafts exact, weights change, drafts
# accepted, sticky rule active) and, every step after the first: drafted multi-turn rollouts were
# replayed (spec/replayed_multi_turn_rows > 0), the driver stepped the environment
# (spec/sticky_env_turns > 0), some rollout ran more than one turn (max_turns > 1), and the finished
# plans' anchors equal a fresh engine prefill of their full rows ("[route check mt]",
# spec/route_check_mt_exact == spec/route_check_mt_tokens > 0) with observation positions among them
# (spec/route_check_mt_obs_positions > 0).
#
# Env: TURNS (3), TURN_TOKENS (128), MAXLEN (1024) + sticky_moe.sh's.
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TURNS=${TURNS:-3} TURN_TOKENS=${TURN_TOKENS:-128}
export RUNS=${RUNS:-runs_sticky_multiturn}
export TAG=${TAG:-mt$TURNS-t$TURN_TOKENS-$(basename "${MODEL:-tiny-qwen3-moe-d2048}")}
export ARGS="--toy-multi-turn-env $TURNS grpo.max_rollout_turns=$TURNS policy.generation.max_new_tokens=$TURN_TOKENS ${ARGS:-}"
bash "$HERE/sticky_moe.sh"
rc=$?
OUT=$W/$RUNS/$TAG
python - "$OUT/history.json" <<'EOF'
import json, sys
try:
    hist = json.load(open(sys.argv[1]))
except (OSError, ValueError) as exc:
    sys.exit(f"[multiturn] no history ({exc})")
why = []
for i, h in enumerate(hist):
    keys = ("mean_turns", "max_turns", "spec/replayed_multi_turn_rows", "spec/sticky_env_turns",
            "spec/route_check_mt_tokens", "spec/route_check_mt_exact",
            "spec/route_check_mt_obs_positions", "mismatch/max_abs_logprob_diff",
            "mismatch/exact_token_frac")
    print(f"[multiturn] step {i}: " + " ".join(f"{k.split('/')[-1]}={h.get(k)}" for k in keys))
    if i == 0:
        continue
    if not (h.get("spec/replayed_multi_turn_rows") or 0) > 0:
        why.append(f"step {i}: no drafted multi-turn rollout replayed")
    if not (h.get("spec/sticky_env_turns") or 0) > 0:
        why.append(f"step {i}: the driver never stepped the environment")
    if not (h.get("max_turns") or 0) > 1:
        why.append(f"step {i}: no rollout had more than one turn")
    tok, exact = h.get("spec/route_check_mt_tokens"), h.get("spec/route_check_mt_exact")
    if not (tok and exact == tok):
        why.append(f"step {i}: multi-turn route check {exact}/{tok}")
    if not (h.get("spec/route_check_mt_obs_positions") or 0) > 0:
        why.append(f"step {i}: no observation position anchored")
ok = not why
print(f"[multiturn] {'OK' if ok else 'FAIL: ' + '; '.join(why)}")
sys.exit(0 if ok else 1)
EOF
src=$?
exit $(( rc != 0 ? rc : src ))
