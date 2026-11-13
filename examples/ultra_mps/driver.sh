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
export PYTHONPATH="$ULTRA_EXAMPLE_ROOT:$ULTRA_EXAMPLE_ROOT/qualification:$ULTRA_REPO_ROOT:$ULTRA_BRIDGE_ROOT/src:$ULTRA_MCORE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RL_ROOT="$ULTRA_REPO_ROOT" HF_HOME="$ULTRA_HF_HOME"
export UV_NO_SYNC=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export NRL_ULTRA_MTP_FIX_SOURCE="$ULTRA_EXAMPLE_ROOT/mtp_fix/SOURCE.json"
export NRL_ULTRA_MCORE_SOURCE="$ULTRA_MCORE_ROOT"
export NRL_ULTRA_AUDIT_DIR="$ULTRA_DIRECT64_RUN_DIR/observer"
export ULTRA_DIRECT64_TOPOLOGY_OUTPUT="$ULTRA_DIRECT64_RUN_DIR/topology.json"
unset NEMO_RL_PY_EXECUTABLES_SYSTEM CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY
if [[ $ULTRA_DIRECT64_ARM == concurrent ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="/tmp/nrl-ultra-transfer-${SLURM_JOB_ID:?}"
else
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
fi
[[ -x "$ULTRA_DRIVER_PYTHON" && -x "$ULTRA_POLICY_PYTHON" ]]
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10s 300s \
    "$ULTRA_DRIVER_PYTHON" "$ULTRA_EXAMPLE_ROOT/preflight.py" --role driver \
    --config "$ULTRA_DIRECT64_RUNTIME_CONFIG" --output "$ULTRA_DIRECT64_RUN_DIR/driver-preflight.json"
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10s 180s \
    "$ULTRA_POLICY_PYTHON" "$ULTRA_EXAMPLE_ROOT/preflight.py" --role policy \
    --config "$ULTRA_DIRECT64_RUNTIME_CONFIG" --output "$ULTRA_DIRECT64_RUN_DIR/policy-preflight.json"
CUDA_VISIBLE_DEVICES="" timeout --kill-after=10s 180s \
    "$ULTRA_POLICY_PYTHON" "$ULTRA_EXAMPLE_ROOT/qualification/mtp_fix_guard.py" \
    --source-manifest "$NRL_ULTRA_MTP_FIX_SOURCE" --mcore-source "$NRL_ULTRA_MCORE_SOURCE" \
    --output "$ULTRA_DIRECT64_RUN_DIR/actual-mtp-import-guard.json"
set +e
timeout --kill-after=30s 5400s "$ULTRA_DRIVER_PYTHON" "$ULTRA_EXAMPLE_ROOT/entry.py" \
    --config "$ULTRA_DIRECT64_RUNTIME_CONFIG"
entry_status=$?
set -e
printf '%s\n' "$entry_status" > "$ULTRA_DIRECT64_RUN_DIR/entry-exit.txt"
[[ $entry_status == 0 ]] || exit "$entry_status"
"$ULTRA_POLICY_PYTHON" -m ultra_qualification.validate \
    --records "$NRL_ULTRA_AUDIT_DIR" --mode "$ULTRA_DIRECT64_ARM" --updates 3 --world-size 128 \
    --output "$ULTRA_DIRECT64_RUN_DIR/observer-validation.json" \
    > "$ULTRA_DIRECT64_RUN_DIR/observer-validation.log" 2>&1
"$ULTRA_DRIVER_PYTHON" "$ULTRA_EXAMPLE_ROOT/qualification/verify_run.py" \
    --run "$ULTRA_DIRECT64_RUN_DIR" --arm "$ULTRA_DIRECT64_ARM" \
    --source-manifest "$NRL_ULTRA_MTP_FIX_SOURCE"
