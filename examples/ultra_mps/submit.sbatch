#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
: "${ULTRA_EXAMPLE_ROOT:?launch with submit.sh}"
: "${SLURM_JOB_ID:?Slurm allocation required}"
[[ ${SLURM_JOB_NUM_NODES:?} == 32 && ${SLURM_NNODES:?} == 32 ]]
export ULTRA_DIRECT64_RUN_DIR="$ULTRA_OUTPUT_ROOT/${SLURM_JOB_ID}${SLURM_RESTART_COUNT:+-$SLURM_RESTART_COUNT}-$ULTRA_DIRECT64_ARM"
python3 "$ULTRA_EXAMPLE_ROOT/prepare_config.py" \
    --arm "$ULTRA_DIRECT64_ARM" --model "$ULTRA_MODEL_PATH" \
    --mcore-root "$ULTRA_MCORE_ROOT" --bridge-root "$ULTRA_BRIDGE_ROOT" \
    --run-dir "$ULTRA_DIRECT64_RUN_DIR"
git -C "$ULTRA_REPO_ROOT" rev-parse HEAD > "$ULTRA_DIRECT64_RUN_DIR/source-commit.txt"
export ULTRA_DIRECT64_RUNTIME_CONFIG="$ULTRA_DIRECT64_RUN_DIR/config.json"
export BASE_LOG_DIR="$ULTRA_DIRECT64_RUN_DIR/ray"
export ULTRA_DIRECT64_ENDED="$BASE_LOG_DIR/${SLURM_JOB_ID}${SLURM_RESTART_COUNT:+-$SLURM_RESTART_COUNT}-logs/ENDED"
export COMMAND="bash $ULTRA_EXAMPLE_ROOT/driver.sh"
export SETUP_COMMAND="bash $ULTRA_EXAMPLE_ROOT/setup_node.sh"
export CONTAINER="$ULTRA_CONTAINER"
export MOUNTS="$ULTRA_SHARED_ROOT:$ULTRA_SHARED_ROOT"
for helper in mtp_utils_pytorch.py mtp_utils_triton.py; do
    MOUNTS+=",$ULTRA_EXAMPLE_ROOT/mtp_fix/fixed/$helper:$ULTRA_MCORE_ROOT/megatron/core/inference/text_generation_controllers/$helper"
done
export GPUS_PER_NODE=4 CPUS_PER_WORKER="$ULTRA_CPUS_PER_NODE"
export DEDICATED_RAY_HEAD=0 INTERACTIVE=0
export PERSISTENT_CACHE="$ULTRA_DIRECT64_RUN_DIR/container-cache"
export UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1
unset NEMO_RL_PY_EXECUTABLES_SYSTEM CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY CUDA_MPS_SM_PARTITION
unset SANDBOX_CONTAINER SANDBOX_COMMAND SANDBOX_EXTRA_MOUNTS NEMO_GYM_EXTRA_ROOTS
cd "$ULTRA_REPO_ROOT"
exec bash "$ULTRA_REPO_ROOT/ray.sub"
