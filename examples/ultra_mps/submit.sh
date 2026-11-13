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
export ULTRA_EXAMPLE_ROOT
ULTRA_EXAMPLE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ULTRA_REPO_ROOT
ULTRA_REPO_ROOT=$(cd "$ULTRA_EXAMPLE_ROOT/../.." && pwd)
export ULTRA_DIRECT64_ARM=${1:?Usage: submit.sh alternating|concurrent [sbatch options]}
shift
case "$ULTRA_DIRECT64_ARM" in alternating|concurrent) ;; *) exit 2 ;; esac

: "${ULTRA_ACCOUNT:?destination Slurm account}"
: "${ULTRA_PARTITION:?destination Slurm partition}"
: "${ULTRA_CONTAINER:?destination container image or squashfs path}"
: "${ULTRA_SHARED_ROOT:?shared filesystem root containing repo, model, dependencies and outputs}"
: "${ULTRA_MODEL_PATH:?local Hugging Face Ultra checkpoint directory}"
: "${ULTRA_MCORE_ROOT:?shared checkout of the pinned Megatron-LM revision}"
: "${ULTRA_BRIDGE_ROOT:?shared checkout of the pinned Megatron-Bridge revision}"
: "${ULTRA_OUTPUT_ROOT:?shared output directory}"
: "${ULTRA_HF_HOME:?shared Hugging Face cache directory}"
: "${ULTRA_DRIVER_PYTHON:?absolute Python executable inside the container}"
: "${ULTRA_POLICY_PYTHON:?absolute Megatron policy Python executable inside the container}"
export ULTRA_ACCOUNT ULTRA_PARTITION ULTRA_CONTAINER ULTRA_SHARED_ROOT
export ULTRA_MODEL_PATH ULTRA_MCORE_ROOT ULTRA_BRIDGE_ROOT ULTRA_OUTPUT_ROOT ULTRA_HF_HOME
export ULTRA_DRIVER_PYTHON ULTRA_POLICY_PYTHON

# ray.sub expands paths inside generated container commands; fail before submission
# for unsupported path spelling instead of interpolating shell metacharacters.
for name in ULTRA_SHARED_ROOT ULTRA_REPO_ROOT ULTRA_MODEL_PATH ULTRA_MCORE_ROOT \
    ULTRA_BRIDGE_ROOT ULTRA_OUTPUT_ROOT ULTRA_HF_HOME ULTRA_DRIVER_PYTHON ULTRA_POLICY_PYTHON; do
    if [[ ! ${!name} =~ ^/[a-zA-Z0-9_./-]+$ ]]; then
        echo "$name must be an absolute path without spaces or shell metacharacters." >&2
        exit 2
    fi
done
for name in ULTRA_REPO_ROOT ULTRA_MODEL_PATH ULTRA_MCORE_ROOT ULTRA_BRIDGE_ROOT \
    ULTRA_OUTPUT_ROOT ULTRA_HF_HOME; do
    if [[ ${!name} != "$ULTRA_SHARED_ROOT"/* ]]; then
        echo "$name must be inside ULTRA_SHARED_ROOT (mounted at the same path)." >&2
        exit 2
    fi
done
if [[ ! $ULTRA_CONTAINER =~ ^[a-zA-Z0-9_./:@#=-]+$ ]]; then
    echo "Unsupported container reference spelling." >&2
    exit 2
fi
[[ -f "$ULTRA_MODEL_PATH/config.json" ]]
[[ $(git -C "$ULTRA_MCORE_ROOT" rev-parse HEAD) == 3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a ]]
[[ $(git -C "$ULTRA_BRIDGE_ROOT" rev-parse HEAD) == 7e9dcd609edf2bb9478d3fa4d9b9f8ee3875c16e ]]
git -C "$ULTRA_MCORE_ROOT" diff --quiet HEAD -- megatron
git -C "$ULTRA_BRIDGE_ROOT" diff --quiet HEAD -- src
mkdir -p "$ULTRA_OUTPUT_ROOT" "$ULTRA_HF_HOME"

# Supply the site's placement option here, e.g. --segment=16. Runtime validation
# independently requires each 64-rank EP group to stay inside one NVLink domain.
placement_args=()
if [[ -n ${ULTRA_PLACEMENT_ARGS:-} ]]; then
    read -r -a placement_args <<< "$ULTRA_PLACEMENT_ARGS"
fi
export ULTRA_CPUS_PER_NODE=${ULTRA_CPUS_PER_NODE:-144}
[[ $ULTRA_CPUS_PER_NODE =~ ^[1-9][0-9]*$ ]]
args=(--parsable --nodes=32 --ntasks-per-node=1 --gres=gpu:4
    "--cpus-per-task=$ULTRA_CPUS_PER_NODE" --mem=0 --exclusive
    "--account=$ULTRA_ACCOUNT" "--partition=$ULTRA_PARTITION"
    "--time=${ULTRA_TIME_LIMIT:-02:30:00}" "--job-name=ultra-$ULTRA_DIRECT64_ARM"
    "--chdir=$ULTRA_REPO_ROOT" "--output=$ULTRA_OUTPUT_ROOT/slurm-%j.log"
    --export=ALL "${placement_args[@]}" "$@" "$ULTRA_EXAMPLE_ROOT/submit.sbatch")
cd "$ULTRA_REPO_ROOT"
case "${ULTRA_SUBMIT:-0}" in
    0) printf 'DRY RUN:'; printf ' %q' sbatch "${args[@]}"; printf '\n' ;;
    1) exec sbatch "${args[@]}" ;;
    *) echo "ULTRA_SUBMIT must be 0 or 1." >&2; exit 2 ;;
esac
