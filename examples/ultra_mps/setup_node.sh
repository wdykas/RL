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
: "${ULTRA_DIRECT64_RUN_DIR:?owned result directory required}"
: "${ULTRA_DIRECT64_ENDED:?native launcher completion signal required}"
node_output="$ULTRA_DIRECT64_RUN_DIR/nodes/$(hostname -s)"
mkdir -p "$node_output"
printf '%s\n' UNASSIGNED_BEFORE_NATIVE_PLACEMENT > "$node_output/role.txt"
# MPS starts only after the native placement group selects its actual nodes.
(
    while [[ ! -e "$ULTRA_DIRECT64_ENDED" ]]; do
        date -u +%FT%T.%NZ
        nvidia-smi --query-gpu=index,uuid,memory.used,memory.total --format=csv,noheader
        nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader
        sleep 0.5
    done
) > "$node_output/physical-memory.log" 2>&1 &
"$ULTRA_DRIVER_PYTHON" "$ULTRA_EXAMPLE_ROOT/qualification/host_telemetry.py" \
    --ended "$ULTRA_DIRECT64_ENDED" --output "$node_output/host-memory.jsonl" \
    --interval-seconds 5 > "$node_output/host-memory-monitor.log" 2>&1 &
