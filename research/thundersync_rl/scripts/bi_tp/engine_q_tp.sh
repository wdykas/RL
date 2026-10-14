#!/bin/bash
# Engine-sourced q (thundersync.verify_source=engine) under batch invariance at tensor,
# pipeline and expert parallelism TP / PP / EP (DESIGN.md section 14). Training layout ==
# generation layout (sequence parallel on both when TP>1), BI flags from
# thundersync_work/bi_flags.txt + inference_optimized generation, learner-side block
# verification, THUNDERSYNC_ENGINE_Q_CHECK=1 THUNDERSYNC_MISMATCH_CHECK=1.
# Pass (exit 0): the run exits 0, every step mismatch/max_abs_logprob_diff == 0 and
# exact_token_frac == 1, every "[engine q check]" line exact == tok with max|dlogq| = 0, and
# the run did not fall back to the scorer.
#
#   TP*PP <= 2 : 1 node, 2 generation + 2 training GPUs (interactive node, megatron-rl container):
#              TP=2 MODEL=Qwen/Qwen3-1.7B bash engine_q_tp.sh
#              (EP: engine_q_moe.sh; PP: engine_q_pp.sh, thin wrappers around this script)
#   >=4 GPUs per side (GPUS_PER_SIDE, default max(TP*PP, EP)): 2*GPS/4 nodes; run on the head of a Ray cluster (as
#              tp_parity_submit.sh).
#
# Env: TP (2), PP (1), EP (1), STEPS (4), MODEL (Qwen/Qwen3-1.7B), PROMPTS (16), MAXLEN (2048),
# TAG, RUNS (output directory name under thundersync_work, runs_engine_q_tp), ARGS (extra Hydra
# overrides, appended last), RL_ROOT_OVERRIDE (code checkout; default: the one holding this
# script). Output: thundersync_work/$RUNS/$TAG/{run.log,history.json}.
set -uo pipefail
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TP=${TP:-2} PP=${PP:-1} EP=${EP:-1} STEPS=${STEPS:-4} MODEL=${MODEL:-Qwen/Qwen3-1.7B}
PROMPTS=${PROMPTS:-16} MAXLEN=${MAXLEN:-2048} RUNS=${RUNS:-runs_engine_q_tp}
TAG=${TAG:-engine-q-tp$TP-pp$PP-ep$EP-$(basename "$MODEL")}
export RL_ROOT_OVERRIDE=${RL_ROOT_OVERRIDE:-$(cd "$HERE/../../../.." && pwd)}
source $W/env_sbatch.sh bi
cd "$RL_ROOT" || exit 2
export NVIDIA_TF32_OVERRIDE=1
# PERF=1: timing runs, no debug checks (each adds forwards); the PASS check then only
# confirms the run completed.
if [ "${PERF:-0}" != 1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_ENGINE_Q_CHECK=1
  export THUNDERSYNC_SPEC_PROF=1 THUNDERSYNC_MEM_LOG=1
fi

# GPUs per side (training and generation each): GPUS_PER_SIDE, default max(TP*PP, EP),
# with 4 GPUs per node. 1-2 per side share one node; 4 or more use whole nodes per side.
GPS=${GPUS_PER_SIDE:-$(( TP * PP > EP ? TP * PP : EP ))}
case $GPS in
  1|2) RES=(cluster.num_nodes=1 cluster.gpus_per_node=4
            policy.generation.colocated.resources.num_nodes=1
            policy.generation.colocated.resources.gpus_per_node=2) ;;
  *)   if (( GPS % 4 )); then echo "GPUS_PER_SIDE must be 1, 2 or a multiple of 4 (got $GPS)" >&2; exit 2; fi
       RES=(cluster.num_nodes=$(( 2 * GPS / 4 )) cluster.gpus_per_node=4
            policy.generation.colocated.resources.num_nodes=$(( GPS / 4 ))
            policy.generation.colocated.resources.gpus_per_node=4) ;;
esac
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
PAR=(policy.megatron_cfg.tensor_model_parallel_size=$TP
     policy.megatron_cfg.pipeline_model_parallel_size=$PP
     policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.sequence_parallel=$SP
     policy.megatron_cfg.expert_tensor_parallel_size=1
     ++policy.generation.mcore_generation_config.tensor_model_parallel_size=$TP
     ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=${GEN_PP:-$PP}
     ++policy.generation.mcore_generation_config.expert_model_parallel_size=$EP
     ++policy.generation.mcore_generation_config.materialize_only_last_token_logits=true
     policy.megatron_cfg.activation_checkpointing=false)
read -r -a BIF < $W/bi_flags.txt
BIF+=(++policy.generation.mcore_generation_config.transformer_impl=inference_optimized)
SPEC=(++thundersync.speculate=true ++thundersync.verify_mode=block
      ++thundersync.verify_on=learner ++thundersync.verify_precision=model
      ++thundersync.verify_source=engine ++thundersync.draft_budget=2048
      ++thundersync.draft_lookahead=3)
read -r -a USER_ARGS <<< "${ARGS:-}"

OUT=$W/$RUNS/$TAG
mkdir -p "$OUT"
rm -f "$OUT/history.json"  # written at the end of training: never check a previous run's
echo "[engine_q] TP=$TP PP=$PP EP=$EP STEPS=$STEPS MODEL=$MODEL RL_ROOT=$RL_ROOT -> $OUT"
timeout 2400 python research/thundersync_rl/run_thundersync_grpo.py \
  --config research/thundersync_rl/configs/grpo_math_1.5b_thundersync_megatron.yaml \
  --history-json "$OUT/history.json" thundersync.granularity=group logger.log_dir="$OUT/logs" \
  logger.wandb_enabled=false logger.tensorboard_enabled=false \
  policy.model_name=$MODEL grpo.max_num_steps=$STEPS \
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=8 \
  policy.train_global_batch_size=$((PROMPTS * 8)) \
  policy.max_total_sequence_length=$MAXLEN policy.make_sequence_length_divisible_by=128 \
  loss_fn.skip_zero_advantage_rows=true \
  "${RES[@]}" "${PAR[@]}" "${BIF[@]}" "${SPEC[@]}" "${USER_ARGS[@]}" \
  > "$OUT/run.log" 2>&1
rc=$?
echo "[engine_q] run exit=$rc"
if [ "${PERF:-0}" = 1 ]; then
  echo "[engine_q] SKIPPED (PERF mode, checks off)"
  exit $rc
fi
grep -a "\[engine q check\]\|falling back\|stash_head\|ep_dummy_forwards" "$OUT/run.log" | tail -n 20
python - "$OUT/history.json" "$OUT/run.log" "$STEPS" "$rc" <<'EOF'
import json, re, sys
path, log, steps, rc = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
try:
    hist = json.load(open(path))
except (OSError, ValueError) as exc:
    sys.exit(f"[engine_q] FAIL: no history ({exc})")
ok = rc == 0 and len(hist) >= steps
for i, h in enumerate(hist):
    d, e = h.get("mismatch/max_abs_logprob_diff"), h.get("mismatch/exact_token_frac")
    good = d == 0 and e == 1
    ok &= good
    print(f"[engine_q] step {i}: max_abs_logprob_diff={d} exact_token_frac={e} "
          f"kept={h.get('spec/accepted_frac')} dropped={h.get('spec/engine_q_dropped')} "
          f"{'ok' if good else 'MISMATCH'}")
text = re.sub(r"\x1b\[[0-9;]*m", "", open(log, errors="replace").read())  # Ray colors
checks = re.findall(r"\[engine q check\].*tok=(\d+) exact=(\d+) max\|dlogq\|=(\S+)", text)
bad = [c for c in checks if c[0] != c[1] or float(c[2]) != 0.0]
ok &= bool(checks) and not bad and "falling back to verify_source=scorer" not in text
print(f"[engine_q] engine q checks: {len(checks)} lines, {len(bad)} inexact")
print(f"[engine_q] {'PASS' if ok else 'FAIL'} ({len(hist)}/{steps} steps)")
sys.exit(0 if ok else 1)
EOF
