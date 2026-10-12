#!/bin/bash
# Submit tp_parity.sh through NeMo-RL's ray.sub (multi-node Ray in the megatron-rl container),
# following thundersync_work/sbatch/submit.sh. TP=4 needs 2 nodes (4 generation + 4 training).
#   ./tp_parity_submit.sh [TP=4] [STEPS=4] [TIME=01:00:00]
# Env forwarded into the job: MODEL, PROMPTS, ARGS, RL_ROOT_OVERRIDE (default: this checkout).
# THUNDERSYNC_MISMATCH_CHECK=1 is set by tp_parity.sh itself.
# Output: thundersync_work/runs_bi_tp/bi-tp$TP-<model>/ (run.log, history.json; Ray logs too).
set -euo pipefail
TP=${1:-4} STEPS=${2:-4} TIME=${3:-01:00:00}
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${RL_ROOT_OVERRIDE:-$(cd "$HERE/../../../.." && pwd)}
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
case $TP in 1|2) NODES=1 ;; 4) NODES=2 ;; *) echo "TP must be 1, 2 or 4" >&2; exit 2 ;; esac
TAG=bi-tp$TP-$(basename "$MODEL")
mkdir -p $W/runs_bi_tp/$TAG
export CONTAINER=/lustre/fsw/portfolios/llmservice/projects/llmservice_fm_text/megatron_rl/adlr+megatron-rl+arm64-260304.sqsh
export MOUNTS="/lustre:/lustre,/home/$USER:/home/$USER"
export GPUS_PER_NODE=4
export BASE_LOG_DIR=$W/runs_bi_tp/$TAG
export COMMAND="THUNDERSYNC_MISMATCH_CHECK=1 RL_ROOT_OVERRIDE=$ROOT ARGS='${ARGS:-}' MODEL=$MODEL PROMPTS=${PROMPTS:-16} TP=$TP STEPS=$STEPS TAG=$TAG bash $ROOT/research/thundersync_rl/scripts/bi_tp/tp_parity.sh"
cd "$ROOT"
sbatch --nodes=$NODES --account=nemotron_sw_pre --partition=batch --time=$TIME \
  --job-name=ts-$TAG --gres=gpu:4 ray.sub
