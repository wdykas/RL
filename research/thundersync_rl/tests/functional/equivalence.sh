#!/bin/bash
# Streamed vs batch-synchronous GRPO updates must match (2 GPUs: 1 per policy).
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
RUN_LOG=$EXP_DIR/run.log
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
# True fp32 GEMMs: TF32 rounding (~1e-3) would swamp the comparison.
export NVIDIA_TF32_OVERRIDE=0

rm -rf $EXP_DIR
mkdir -p $EXP_DIR

cd $PROJECT_ROOT
uv run tests/functional/check_equivalence.py \
    --config $PROJECT_ROOT/configs/grpo_math_1.5b_thundersync_megatron.yaml \
    --gpus-per-policy 1 \
    --steps 3 \
    --tol 1e-4 \
    policy.model_name=Qwen/Qwen2.5-0.5B \
    policy.precision=float32 \
    policy.megatron_cfg.optimizer.bf16=false \
    policy.megatron_cfg.optimizer.fp16=false \
    policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false \
    $@ \
    2>&1 | tee $RUN_LOG

grep -q "EQUIVALENCE PASS" $RUN_LOG
