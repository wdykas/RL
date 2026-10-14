#!/bin/bash
# Run ONE arm of the ThunderSync A/B kit (configs/ab/ab_<ARM>.yaml) on the head node of a Ray cluster
# (sbatch via thundersync_work/sbatch/submit_ab.sh) or on one 4-GPU node (2 engine + 2 learner GPUs).
#
#   ARM=sync|async1|ts_nonull|ts_por|ts_banded MODEL=<hf id or path> TP=1 PP=1 EP=8 \
#     MAXLEN=16384 STEPS=6 PROMPTS=64 RUNS=runs_ab TAG=<tag> bash run_ab_arm.sh
#
# Training and generation each get GPUS_PER_SIDE = max(TP*PP, EP) GPUs (4 GPUs per node; 1-2 per side
# share one node). The engine uses the learner's TP/PP/EP (GEN_PP overrides the engine's PP).
# Exactness checks (every arm but async1, CHECKS=1 default): THUNDERSYNC_MISMATCH_CHECK (one learner
# forward of the step's rollouts vs their rollout log-probs, timed as time/mismatch_check),
# THUNDERSYNC_ENGINE_Q_CHECK (engine q vs the verification forward, per verified token) and a
# route check of ROUTE_CHECK_ROWS (16) sampled drafts per step (timed as time/route_check).
# Output: thundersync_work/$RUNS/$TAG/{run.log,history.json} (history rewritten every step).
# Summarize: python thundersync_work/research/ab/summarize_ab.py thundersync_work/$RUNS/<tags...>
set -uo pipefail
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM=${ARM:?ARM} MODEL=${MODEL:?MODEL}
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} STEPS=${STEPS:-6} PROMPTS=${PROMPTS:-64} G=${G:-8}
MAXLEN=${MAXLEN:-16384} RUNS=${RUNS:-runs_ab} CHECKS=${CHECKS:-1}
TAG=${TAG:-$ARM-$(basename "$MODEL")-tp$TP-pp$PP-ep$EP-L$MAXLEN}
export RL_ROOT_OVERRIDE=${RL_ROOT_OVERRIDE:-$(cd "$HERE/../../../.." && pwd)}
source $W/env_sbatch.sh bi
cd "$RL_ROOT" || exit 2
export NVIDIA_TF32_OVERRIDE=1
CFG=research/thundersync_rl/configs/ab/ab_$ARM.yaml
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 2; }
case $ARM in
  async1) ENTRY=research/thundersync_rl/run_ab_baseline.py ;;
  *)      ENTRY=research/thundersync_rl/run_thundersync_grpo.py ;;
esac
if [ "$CHECKS" = 1 ] && [ "$ARM" != async1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_ENGINE_Q_CHECK=1
  export THUNDERSYNC_ROUTE_CHECK_ROWS=${ROUTE_CHECK_ROWS:-16}
fi
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
     ++policy.generation.mcore_generation_config.tensor_model_parallel_size=$TP
     ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=${GEN_PP:-$PP}
     ++policy.generation.mcore_generation_config.expert_model_parallel_size=$EP)
(( PP > 1 || ${GEN_PP:-$PP} > 1 )) && PAR+=(++policy.megatron_cfg.pipeline_dtype=float32)
read -r -a USER_ARGS <<< "${ARGS:-}"
OUT=$W/$RUNS/$TAG
mkdir -p "$OUT"
rm -f "$OUT/history.json"
echo "$ARM" > "$OUT/arm"
echo "[ab] ARM=$ARM TP=$TP PP=$PP EP=$EP GPS=$GPS STEPS=$STEPS PROMPTS=$PROMPTS G=$G MAXLEN=$MAXLEN MODEL=$MODEL" \
  "RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD) -> $OUT"
timeout ${RUN_TIMEOUT:-14400} python $ENTRY --config $CFG --history-json "$OUT/history.json" \
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS \
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=$G \
  policy.train_global_batch_size=$((PROMPTS * G)) policy.max_total_sequence_length=$MAXLEN \
  "${RES[@]}" "${PAR[@]}" "${USER_ARGS[@]}" > "$OUT/run.log" 2>&1
rc=$?
echo "[ab] run exit=$rc"
python $W/research/ab/summarize_ab.py --check "$OUT"
src=$?
exit $(( rc != 0 ? rc : src ))
