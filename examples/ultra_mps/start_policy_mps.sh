#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Called only on actual policy nodes after their native placement group is ready.
set -euo pipefail
: "${ULTRA_DIRECT64_RUN_DIR:?owned outputs required}"
: "${ULTRA_DIRECT64_ENDED:?native launcher cleanup signal required}"
: "${CUDA_MPS_PIPE_DIRECTORY:?private node-local pipe required}"
ultra_node_output="$ULTRA_DIRECT64_RUN_DIR/nodes/$(hostname -s)"
[[ $(cat "$ultra_node_output/role.txt") == policy ]]
ultra_pipe=$CUDA_MPS_PIPE_DIRECTORY
[[ "$ultra_pipe" == /* && "$ultra_pipe" != / ]]
[[ ! -e "$ultra_pipe" ]]
mkdir -m 700 "$ultra_pipe"
export CUDA_MPS_PIPE_DIRECTORY="$ultra_pipe"
export CUDA_MPS_LOG_DIRECTORY="$ultra_node_output/mps-logs"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
unset CUDA_MPS_ACTIVE_THREAD_PERCENTAGE CUDA_MPS_CLIENT_PRIORITY
CUDA_VISIBLE_DEVICES=0,1,2,3 nvidia-cuda-mps-control -d
printf 'get_server_list\n' | nvidia-cuda-mps-control > "$ultra_node_output/mps-start-query.txt"
printf '%s\n' 'PRIVATE_MPS_CONTROL_STARTED_POLICY_ONLY' > "$ultra_node_output/mps-status.txt"
# Actual server/client PID evidence, not an inference from cap environment vars.
(
  while [[ ! -e "$ULTRA_DIRECT64_ENDED" ]]; do
    date -u +%FT%T.%NZ
    ultra_evidence_servers=$(printf 'get_server_list\n' | nvidia-cuda-mps-control)
    for ultra_evidence_server in $ultra_evidence_servers; do
      [[ "$ultra_evidence_server" =~ ^[0-9]+$ ]] || exit 3
      printf 'SERVER_PID %s\n' "$ultra_evidence_server"
      printf 'get_client_list %s\n' "$ultra_evidence_server" | nvidia-cuda-mps-control
      printf 'SERVER_DEFAULT_PERCENTAGE %s\n' "$ultra_evidence_server"
      printf 'get_active_thread_percentage %s\n' "$ultra_evidence_server" | nvidia-cuda-mps-control
    done
    sleep 1
  done
) > "$ultra_node_output/mps-client-registration.log" 2>&1 &
# Keep setup's CUDA_MPS_* exports PRIVATE to this child script. Driver passes the
# private pipe through native actor_env to learner/engine, never job-wide Gym.
(
  while [[ ! -e "$ULTRA_DIRECT64_ENDED" ]]; do sleep 1; done
  # Best effort: upstream ENDED monitor may terminate this process group sooner.
  # Only an explicit cleanup record or native post-run PID query proves cleanup.
  ultra_cleanup_deadline=$((SECONDS+120))
  while [[ $SECONDS -lt $ultra_cleanup_deadline ]]; do
    ultra_clients=0
    ultra_servers=$(printf 'get_server_list\n' | nvidia-cuda-mps-control)
    for ultra_server in $ultra_servers; do
      [[ "$ultra_server" =~ ^[0-9]+$ ]] || exit 3
      ultra_server_clients=$(printf 'get_client_list %s\n' "$ultra_server" | nvidia-cuda-mps-control)
      [[ -z "$ultra_server_clients" ]] || ultra_clients=1
    done
    if [[ "$ultra_clients" == 0 ]]; then
      printf 'quit\n' | nvidia-cuda-mps-control
      printf '%s\n' 'PRIVATE_MPS_QUIT_AFTER_CLIENTS_DRAINED' > "$ultra_node_output/mps-cleanup.txt"
      exit 0
    fi
    sleep 1
  done
  printf '%s\n' 'CLIENTS_REMAIN_AFTER_120S_NO_FORCED_CONTROL_QUIT' > "$ultra_node_output/mps-cleanup.txt"
) > "$ultra_node_output/mps-cleanup.log" 2>&1 &
