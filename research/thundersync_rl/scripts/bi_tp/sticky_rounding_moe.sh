#!/bin/bash
# Sticky fp16 rounding (thundersync.sticky_rounding, DESIGN.md section 14g) on top of sticky MoE
# routing (sticky_moe.sh, which wraps engine_q_tp.sh): 2 inference + 2 training GPUs of one node,
# short run (STEPS=4 default; not for perf runs).
#
#   bash research/thundersync_rl/scripts/bi_tp/sticky_rounding_moe.sh                     # tiny MoE, EP=2
#   MODEL=$W/models/qwen3-30b-a3b-L4 bash .../sticky_rounding_moe.sh                      # real 30B-A3B layers
#   TP=2 EP=1 bash .../sticky_rounding_moe.sh                                             # TP=2 (SP) learner
#   SROUND=0 bash .../sticky_rounding_moe.sh                                              # A/B: sticky routing only
#
# Pass (exit 0): everything sticky_moe.sh checks (mismatch 0 / exact_token_frac 1 every step, every
# "[engine q check]" exact, weights change, drafts accepted, route check exact, sticky rule active),
# plus with SROUND=1: "[sround check]" lines exist and sticky rounding changed some roundings
# (changed > 0) on a step with drafts. Compare spec/accepted_frac and spec/full_accept vs SROUND=0.
#
# Env: as sticky_moe.sh, plus SROUND (1).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SROUND=${SROUND:-1}
export MODEL=${MODEL:-$W/models/tiny-qwen3-moe-d2048}
export EP=${EP:-2} TP=${TP:-1} DELTA=${DELTA:-0.03}
export RUNS=${RUNS:-runs_sticky_rounding_moe}
export TAG=${TAG:-sround$SROUND-d$DELTA-ep$EP-tp$TP-$(basename "$MODEL")}
if [ "$SROUND" = 1 ]; then
  export ARGS="++thundersync.sticky_rounding=true ${ARGS:-}"
fi
STICKY=1 bash "$HERE/sticky_moe.sh"
rc=$?
[ "$SROUND" = 1 ] || exit $rc
OUT=$W/$RUNS/$TAG
python - "$OUT/run.log" <<'PY'
import re, sys
text = re.sub(r"\x1b\[[0-9;]*m", "", open(sys.argv[1], errors="replace").read())  # Ray colors
checks = [(int(n), int(c)) for n, c in re.findall(r"\[sround check\].*elements=(\d+) changed=(\d+)", text)]
n, c = sum(x for x, _ in checks), sum(y for _, y in checks)
print(f"[sround] checks: {len(checks)} lines, elements={n}, changed={c}, frac={c / max(n, 1):.6f}")
ok = any(x > 0 and y > 0 for x, y in checks)
print("[sround] OK" if ok else "[sround] FAIL: sticky rounding never changed a rounding")
sys.exit(0 if ok else 1)
PY
src=$?
exit $(( rc != 0 ? rc : src ))
