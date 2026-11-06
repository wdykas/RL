#!/bin/bash
# End-to-end: 2 GRPO steps, Megatron training (1 GPU) + non-colocated Megatron inference (1 GPU).
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR $LOG_DIR

# Base config paths (prompts, datasets) are relative to the repo root.
cd $PROJECT_ROOT/../..
uv run $PROJECT_ROOT/run_thundersync_grpo.py \
    --config $PROJECT_ROOT/configs/grpo_math_1.5b_thundersync_megatron.yaml \
    --history-json $EXP_DIR/history.json \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    grpo.num_prompts_per_step=4 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=2 \
    policy.max_total_sequence_length=512 \
    policy.generation.colocated.resources.gpus_per_node=1 \
    cluster.gpus_per_node=2 \
    logger.log_dir=$LOG_DIR \
    $@ \
    2>&1 | tee $RUN_LOG

grep -q "\[thundersync step 1\]" $RUN_LOG
