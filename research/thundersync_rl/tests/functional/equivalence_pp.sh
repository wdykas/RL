#!/bin/bash
# Streamed vs batch-synchronous GRPO updates under pipeline parallelism
# (4 GPUs: PP=2 per policy). The streamed policy runs each dispatch's chunks as
# one pipeline call (merge_pipeline_calls) with micro-batches of mixed row
# counts and sequence lengths (--trim-seq, config.variable_seq_lengths), and
# holds closed-group rows until 2 * PP micro-batches are queued. A second run
# with merge_pipeline_calls=false is the per-chunk control.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
PROJECT_ROOT=$(realpath $SCRIPT_DIR/../..)

set -eou pipefail

EXP_NAME=$(basename $0 .sh)
EXP_DIR=$SCRIPT_DIR/$EXP_NAME
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
# True fp32 GEMMs: TF32 rounding (~1e-3) would swamp the comparison.
export NVIDIA_TF32_OVERRIDE=0
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B}

rm -rf $EXP_DIR
mkdir -p $EXP_DIR

cd $PROJECT_ROOT
for MERGE in true false; do
    RUN_LOG=$EXP_DIR/run_merge_${MERGE}.log
    uv run tests/functional/check_equivalence.py \
        --config $PROJECT_ROOT/configs/grpo_math_1.5b_thundersync_megatron.yaml \
        --gpus-per-policy 2 \
        --steps 3 \
        --tol 1e-4 \
        --trim-seq \
        policy.model_name=$MODEL \
        policy.precision=float32 \
        policy.megatron_cfg.optimizer.bf16=false \
        policy.megatron_cfg.optimizer.fp16=false \
        policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false \
        policy.megatron_cfg.pipeline_model_parallel_size=2 \
        policy.megatron_cfg.tensor_model_parallel_size=1 \
        policy.make_sequence_length_divisible_by=16 \
        policy.train_micro_batch_size=4 \
        ++thundersync.max_open_groups=0 \
        ++thundersync.merge_pipeline_calls=$MERGE \
        "$@" \
        2>&1 | tee $RUN_LOG
    grep -q "EQUIVALENCE PASS" $RUN_LOG
done
echo "equivalence_pp: merged and per-chunk pipeline calls both match Policy.train"
