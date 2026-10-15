#!/bin/bash
# Run ONE arm of the ThunderSync A/B kit (configs/ab/ab_<ARM>.yaml) on the head node of a Ray cluster
# (sbatch via thundersync_work/sbatch/submit_ab.sh) or on one 4-GPU node (2 engine + 2 learner GPUs).
#
#   ARM=sync|async1|ts_nonull|ts_por|ts_banded|sync_coloc|async1_nrl [MBS=8 LPBS=4] MODEL=<hf id or path> TP=1 PP=1 EP=8 \
#     MAXLEN=8192 STEPS=6 PROMPTS=64 RUNS=runs_ab TAG=<tag> bash run_ab_arm.sh
#
# GPUS_PER_SIDE (default lcm(TP, EP) * PP) GPUs per side, 4 GPUs per node. sync_coloc (stock NeMo-RL GRPO)
# is colocated: learner and engine share GPUS_PER_SIDE GPUs. Every other arm is split: GPUS_PER_SIDE
# learner + GPUS_PER_SIDE engine GPUs (1-2 per side share one node). The engine uses the learner's
# TP/PP/EP (GEN_PP overrides the engine's PP).
# Exactness checks (ThunderSync-loop arms, CHECKS=1 default): THUNDERSYNC_MISMATCH_CHECK (one learner
# forward of the step's rollouts vs their rollout log-probs, timed as time/mismatch_check),
# THUNDERSYNC_ENGINE_Q_CHECK (engine q vs the verification forward, per verified token) and a
# route check of ROUTE_CHECK_ROWS (16) sampled drafts per step (timed as time/route_check).
# The stock arms (sync_coloc, async1_nrl) report NeMo-RL's token_mult_prob_error / gen_kl_error.
# Generation progress (every 60 s, engine rank 0) and the 10-min stall watchdog: ab_base.yaml.
# Output: thundersync_work/$RUNS/$TAG/{run.log,history.json} (history rewritten every step).
# Summarize: python thundersync_work/research/ab/summarize_ab.py thundersync_work/$RUNS/<tags...>
set -uo pipefail
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM=${ARM:?ARM} MODEL=${MODEL:?MODEL}
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} STEPS=${STEPS:-6} PROMPTS=${PROMPTS:-64} G=${G:-8}
MAXLEN=${MAXLEN:-8192} RUNS=${RUNS:-runs_ab} CHECKS=${CHECKS:-1}
# Lengths, identical on every arm: MAXLEN = generation cap (max_new_tokens); PROMPT_CAP = prompt cap
# (data.max_input_seq_length; longer prompts are masked out by the data processor, loss_multiplier 0);
# policy.max_total_sequence_length = PROMPT_CAP + MAXLEN, so no kept prompt overflows the sequence budget
# (DAPO prompts reach ~1.25k tokens: a 1k total cap overflowed).
PROMPT_CAP=${PROMPT_CAP:-2048}
TAG=${TAG:-$ARM-$(basename "$MODEL")-tp$TP-pp$PP-ep$EP-L$MAXLEN}
export RL_ROOT_OVERRIDE=${RL_ROOT_OVERRIDE:-$(cd "$HERE/../../../.." && pwd)}
source $W/env_sbatch.sh bi
cd "$RL_ROOT" || exit 2
export NVIDIA_TF32_OVERRIDE=1
CFG=research/thundersync_rl/configs/ab/ab_$ARM.yaml
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 2; }
case $ARM in
  sync_coloc|async1_nrl) ENTRY=research/thundersync_rl/run_ab_baseline.py ;;  # stock NeMo-RL GRPO loops
  *)                     ENTRY=research/thundersync_rl/run_thundersync_grpo.py ;;  # ThunderSync learner path
esac
# sync_coloc is colocated (learner and engine share GPUS_PER_SIDE GPUs); every other arm is split
# (GPUS_PER_SIDE learner + GPUS_PER_SIDE engine GPUs).
COLOCATED=$([ "$ARM" = sync_coloc ] && echo 1 || echo 0)
# Exactness checks on the exact ThunderSync-loop arms (async1 is one step stale by design).
if [ "$CHECKS" = 1 ] && [ "$ENTRY" = research/thundersync_rl/run_thundersync_grpo.py ] && [ "$ARM" != async1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_ENGINE_Q_CHECK=1
  export THUNDERSYNC_ROUTE_CHECK_ROWS=${ROUTE_CHECK_ROWS:-16}
fi
gcd() { local a=$1 b=$2; while (( b )); do set -- $b $(( a % b )); a=$1; b=$2; done; echo $a; }
GPS=${GPUS_PER_SIDE:-$(( TP * EP / $(gcd $TP $EP) * PP ))}
if (( GPS > 2 && GPS % 4 )); then echo "GPUS_PER_SIDE must be 1, 2 or a multiple of 4 (got $GPS)" >&2; exit 2; fi
if [ "$COLOCATED" = 1 ]; then
  GPUS=$GPS
  RES=(cluster.num_nodes=$(( GPS > 4 ? GPS / 4 : 1 )) cluster.gpus_per_node=$(( GPS > 4 ? 4 : GPS )))
elif (( GPS <= 2 )); then
  GPUS=$(( 2 * GPS ))
  RES=(cluster.num_nodes=1 cluster.gpus_per_node=$GPUS
       policy.generation.colocated.resources.num_nodes=1
       policy.generation.colocated.resources.gpus_per_node=$GPS)
else
  GPUS=$(( 2 * GPS ))
  RES=(cluster.num_nodes=$(( 2 * GPS / 4 )) cluster.gpus_per_node=4
       policy.generation.colocated.resources.num_nodes=$(( GPS / 4 ))
       policy.generation.colocated.resources.gpus_per_node=4)
fi
# LANES=2 (ThunderSync arms, thundersync.cohort_lanes): a second engine of the
# same shape (one full TP/EP replica per lane) built by run_thundersync_grpo.py
# outside cluster.* (learner + lane 0); the job must hold GPS more GPUs.
LANES=${LANES:-1}
if [ "$LANES" = 2 ]; then
  [ "$COLOCATED" = 0 ] || { echo "LANES=2 needs a split arm" >&2; exit 2; }
  GPUS=$(( 3 * GPS ))
  LANE_ARGS=(++thundersync.cohort_lanes=2)
else
  LANE_ARGS=()
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
PAR=(policy.megatron_cfg.tensor_model_parallel_size=$TP
     policy.megatron_cfg.pipeline_model_parallel_size=$PP
     policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.sequence_parallel=$SP
     ++policy.generation.mcore_generation_config.tensor_model_parallel_size=$TP
     ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=${GEN_PP:-$PP}
     ++policy.generation.mcore_generation_config.expert_model_parallel_size=$EP)
(( PP > 1 || ${GEN_PP:-$PP} > 1 )) && PAR+=(++policy.megatron_cfg.pipeline_dtype=float32)
# Micro-batch sizes (default: ab_base.yaml, 1 / 1 for long caps; the 1k-cap single-node runs use 8 / 4).
[ -n "${MBS:-}" ] && PAR+=(policy.train_micro_batch_size=$MBS)
[ -n "${LPBS:-}" ] && PAR+=(policy.logprob_batch_size=$LPBS)
read -r -a USER_ARGS <<< "${ARGS:-}"
OUT=$W/$RUNS/$TAG
mkdir -p "$OUT"
rm -f "$OUT/history.json"
echo "$ARM" > "$OUT/arm"
echo "$GPUS" > "$OUT/gpus"  # GPUs the arm holds (summarizer: GPU-seconds per step)
echo "[ab] ARM=$ARM TP=$TP PP=$PP EP=$EP GPS=$GPS GPUS=$GPUS COLOCATED=$COLOCATED LANES=$LANES STEPS=$STEPS PROMPTS=$PROMPTS G=$G MAXLEN=$MAXLEN PROMPT_CAP=$PROMPT_CAP MODEL=$MODEL" \
  "RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD) -> $OUT"
timeout ${RUN_TIMEOUT:-14400} python $ENTRY --config $CFG --history-json "$OUT/history.json" \
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS \
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=$G \
  policy.train_global_batch_size=$((PROMPTS * G)) policy.max_total_sequence_length=$((PROMPT_CAP + MAXLEN)) \
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN \
  "${RES[@]}" "${PAR[@]}" "${LANE_ARGS[@]}" "${USER_ARGS[@]}" > "$OUT/run.log" 2>&1
rc=$?
echo "[ab] run exit=$rc"
python $W/research/ab/summarize_ab.py --check "$OUT"
src=$?
exit $(( rc != 0 ? rc : src ))
