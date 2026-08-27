#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

NUM_NODES=1
GPUS_PER_NODE=4
SEGMENT_SIZE=1
STEPS_PER_RUN=1
MAX_STEPS=1
NUM_RUNS=1
NUM_MINUTES=60

exit_if_max_steps_reached

cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=false \
    logger.tensorboard_enabled=true \
    checkpointing.enabled=false \
    "$@" \
    2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    'max(data["train/reward"]) >= 0.0'
