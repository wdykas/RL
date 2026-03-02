#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)
# Mark the current repo as safe, since wandb fetches metadata about the repo
git config --global --add safe.directory $PROJECT_ROOT

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
LOG_DIR=$EXP_DIR/logs
JSON_METRICS=$EXP_DIR/metrics.json
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/uv-cache}

rm -rf $EXP_DIR $LOG_DIR
mkdir -p $EXP_DIR $LOG_DIR

# Non-colocated async GRPO with inference_optimized + MXFP8
# Tests the FlashInfer MXFP8 weight conversion during refit:
#   1. Initial quantize in prepare_for_generation (nn.Parameter -> MXFP8Tensor)
#   2. Restore before swap (MXFP8Tensor -> nn.Parameter placeholder)
#   3. Re-quantize after swap (nn.Parameter -> MXFP8Tensor, copy into persistent buffers)
# Verifies CUDA graphs remain valid across multiple refits via persistent buffer design.

cd $PROJECT_ROOT
uv run python $PROJECT_ROOT/examples/run_grpo.py \
    --config $PROJECT_ROOT/examples/configs/grpo_math_1B_megatron.yaml \
    policy.model_name=Qwen/Qwen3-0.6B \
    policy.generation.backend=megatron \
    policy.generation.mcore_generation_config.async_engine=true \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    policy.train_global_batch_size=4 \
    policy.logprob_batch_size=4 \
    policy.train_micro_batch_size=1 \
    policy.generation.colocated.enabled=false \
    policy.generation.colocated.resources.gpus_per_node=1 \
    policy.generation.colocated.resources.num_nodes=1 \
    +policy.megatron_cfg.refit_backend=gloo \
    +policy.megatron_cfg.transformer_impl=inference_optimized \
    policy.megatron_cfg.fp8_cfg.enabled=true \
    policy.megatron_cfg.fp8_cfg.fp8=e4m3 \
    policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8 \
    policy.megatron_cfg.fp8_cfg.fp8_param=false \
    cluster.gpus_per_node=2 \
    grpo.max_num_steps=20 \
    grpo.async_grpo.enabled=true \
    grpo.async_grpo.max_trajectory_age_steps=1 \
    grpo.async_grpo.in_flight_weight_updates=true \
    loss_fn.use_importance_sampling_correction=true \
    logger.tensorboard_enabled=true \
    logger.log_dir=$LOG_DIR \
    logger.wandb_enabled=false \
    logger.monitor_gpus=true \
    checkpointing.enabled=false \
    $@ \
    2>&1 | tee $RUN_LOG

uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS

# Use a more relaxed tolerance for FP8 quantization noise
uv run tests/check_metrics.py $JSON_METRICS \
    'max(data["train/token_mult_prob_error"]) < 2.0'
