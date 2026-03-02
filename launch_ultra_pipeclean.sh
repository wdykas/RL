#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_v3_pipeclean.sh
#
# GRPO Ultra V3 pipe-cleaning on GB200 NVL72 with NeMo Gym
#
# By default, this runs from what's built into the container without overlay mounts applied. 
# Set USE_WORKTREE=1 to overlay your local worktree submodules for development.
# Set INTERACTIVE=1 to get a persistent allocation in slurm for iterative debugging.
#
# Usage:
#   ./launch_ultra_v3_pipeclean.sh                                   # batch, bare container
#   USE_WORKTREE=1 ./launch_ultra_v3_pipeclean.sh                    # batch, overlay local code
#   WALLTIME=4:00:00 ./launch_ultra_v3_pipeclean.sh
#   NUM_ACTOR_NODES=116 INFERENCE_NUM_NODES=52 ./launch_ultra_v3_pipeclean.sh
#
# Interactive debugging (reuse allocation across runs):
#   INTERACTIVE=1 ./launch_ultra_v3_pipeclean.sh                     # submits, auto-runs, waits
#   INTERACTIVE=1 INTERACTIVE_WAIT=0 ./launch_ultra_v3_pipeclean.sh  # submit only (no foreground wait)
#   INTERACTIVE=1 INTERACTIVE_WALLTIME=8:0:0 ./launch_ultra_v3_pipeclean.sh  # longer allocation
#
#   A background watcher auto-runs the training command as soon as Ray is ready,
#   so GPUs are never idle waiting for you to type. After training finishes the
#   allocation stays alive — re-attach and iterate without requeueing.
#
#   Once Ray is up, you can:
#     # Run non-interactively from login node
#     COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
#
#     # Or attach interactively, then run inside the container
#     bash <jobid>-attach.sh
#     source <jobid>-run-cmd.sh
#
#     # Edit and re-run without requeueing
#     vim <jobid>-run-cmd.sh
#     COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
# =============================================================================

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
PROJECT_ROOT=${SCRIPT_DIR}
cd ${PROJECT_ROOT}

USE_WORKTREE="${USE_WORKTREE:-0}"
INTERACTIVE="${INTERACTIVE:-0}"
INTERACTIVE_WAIT="${INTERACTIVE_WAIT:-1}"

# ---------- SLURM configuration ----------
SLURM_ACCOUNT="${SLURM_ACCOUNT:-llmservice_nemotron_ultra}"
PARTITION="${PARTITION:-batch}"
WALLTIME="${WALLTIME:-1:00:00}"


# ---------- Container & mounts ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/ansubramania/containers/nemo-rl-ultra-20260226-428eb84dd-custom-vllm-arm.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: 4 GPUs/node. Must match --gres=gpu:4 passed to sbatch.
export GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# ---------- HuggingFace Configuration ----------
export HF_HOME="${HF_HOME:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# ---------- W&B Configuration ----------
WANDB_PROJ="${WANDB_PROJ:-grpo-ultra-v3-pipeclean}"
WANDB_NAME="${WANDB_NAME:-ultra-v3-grpo-$(date +%m%d-%H%M)}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------- Job Shape ----------
GENERATION_NUM_NODES="${GENERATION_NUM_NODES:-26}"
NUM_ACTOR_NODES="${NUM_ACTOR_NODES:-58}"
COLOCATED_INFERENCE="${COLOCATED_INFERENCE:-False}"

NUM_GENRM_NODES="${NUM_GENRM_NODES:-2}"
NUM_LLMJUDGE_NODES="${NUM_LLMJUDGE_NODES:-2}"
NUM_SAFETY_NODES="${NUM_SAFETY_NODES:-1}"
NUM_GYM_EXTRA_NODES="${NUM_GYM_EXTRA_NODES:-1}"
NUM_JUDGE_NODES=$((NUM_GENRM_NODES + NUM_LLMJUDGE_NODES + NUM_SAFETY_NODES + NUM_GYM_EXTRA_NODES))
NUM_TOTAL_NODES=$((NUM_ACTOR_NODES + NUM_JUDGE_NODES))

# ---------- Model and data paths ----------
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.train.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.val.jsonl}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/nemotron_ultra/sft-runs/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase/hf_converted}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/makeshn/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

EXP_SUFFIX="${EXP_SUFFIX:-ultra-v3-grpo-pipeclean}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${EXP_SUFFIX}}"
mkdir -p "${CHECKPOINT_DIR}"

# ---------- Persistent cache directories ----------
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/users/ansubramania/.cache}"
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}"

VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_aarch64.whl}"

# =============================================================================
# Worktree setup (only when USE_WORKTREE=1)
# =============================================================================
if [[ "${USE_WORKTREE}" == "1" ]]; then
  WORKTREE_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  MAIN_REPO_ROOT="${MAIN_REPO_ROOT:-$(git -C "${WORKTREE_ROOT}" worktree list --porcelain | awk '/^worktree /{print $2}' | grep -v '/.worktrees/' | head -n1)}"

  if [[ -z "${MAIN_REPO_ROOT}" || ! -d "${MAIN_REPO_ROOT}" ]]; then
    echo "Could not resolve MAIN_REPO_ROOT; set MAIN_REPO_ROOT explicitly."
    exit 1
  fi

  if [[ ! -f "${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env" ]]; then
    echo "Missing main vLLM env file: ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env"
    exit 1
  fi

  MISSING=0
  for p in \
    "${WORKTREE_ROOT}/3rdparty/Gym-workspace/Gym/nemo_gym/cli.py" \
    "${WORKTREE_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM" \
    "${WORKTREE_ROOT}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" \
    "${WORKTREE_ROOT}/3rdparty/Automodel-workspace/Automodel"
  do
    if [[ ! -e "${p}" ]]; then
      echo "Missing required worktree path: ${p}"
      MISSING=1
    fi
  done
  if [[ "${MISSING}" -ne 0 ]]; then
    echo "Initialize submodules on login node first:"
    echo "  git -C ${WORKTREE_ROOT} submodule update --init --recursive"
    exit 1
  fi
  echo "Worktree mode: overlaying ${WORKTREE_ROOT}"
  echo "Main repo vLLM: ${MAIN_REPO_ROOT}/3rdparty/vllm"
fi


# =============================================================================
# Code root — container path or worktree
# =============================================================================
# NOTE: In bare container mode we assume /opt/nemo-rl/3rdparty/vllm/nemo-rl.env
# exists inside the container. 
# This can't be verified from the login node.
if [[ "${USE_WORKTREE}" == "1" ]]; then
  CODE_ROOT="${WORKTREE_ROOT}"
  VLLM_ENV_SOURCE="source ${MAIN_REPO_ROOT}/3rdparty/vllm/nemo-rl.env && "
else
  CODE_ROOT="/opt/nemo-rl"
  VLLM_ENV_SOURCE="source /opt/nemo-rl/3rdparty/vllm/nemo-rl.env && "
fi

echo "Nodes: ${NUM_TOTAL_NODES} (actor=${NUM_ACTOR_NODES} [train=$((NUM_ACTOR_NODES - GENERATION_NUM_NODES)), gen=${GENERATION_NUM_NODES}], judge=${NUM_JUDGE_NODES})"
echo "Code root: ${CODE_ROOT}"
echo "Persistent cache root: ${PERSISTENT_CACHE}"

# =============================================================================
# Build the training command
# =============================================================================
# All env vars that need to reach compute nodes are set INSIDE the command
# string. sbatch does not propagate the login node's exports — ray.sub starts
# a fresh shell and executes $COMMAND via enroot exec inside the container.
#
# All static config (parallelism, vLLM kwargs, judge server_args, sequence
# packing, etc.) lives in grpo_ultra_v3.yaml. Only per-run variables are
# overridden here.
TRAIN_CMD="cd ${CODE_ROOT} && date ; \
${VLLM_ENV_SOURCE}\
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
UV_LINK_MODE=symlink uv run nemo_rl/utils/prefetch_venvs.py && \
OMP_NUM_THREADS=16 \
RAY_DEDUP_LOGS=1 \
NRL_VLLM_USE_V1=1 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
UV_CACHE_DIR=${PERSISTENT_CACHE}/uv \
NEMO_GYM_SKIP_VENV_IF_PRESENT=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
UV_HTTP_TIMEOUT=10 \
VLLM_USE_PRECOMPILED=1 \
VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
VLLM_USE_FLASHINFER_MOE_FP8=1 \
VLLM_FLASHINFER_MOE_BACKEND=latency \
FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
NRL_VLLM_ASYNC_TIMEOUT_SECONDS=1800 \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_ultra_v3.yaml \
policy.model_name=${NRL_MODEL_PATH} \
cluster.gpus_per_node=4 \
cluster.num_nodes=${NUM_TOTAL_NODES} \
policy.generation.colocated.enabled=${COLOCATED_INFERENCE} \
policy.generation.colocated.resources.num_nodes=${GENERATION_NUM_NODES} \
policy.generation.colocated.resources.gpus_per_node=4 \
env.nemo_gym.num_gpu_nodes=${NUM_JUDGE_NODES} \
env.nemo_gym.genrm_model.responses_api_models.vllm_model.model=${NRL_GENRM_MODEL_PATH} \
env.nemo_gym.nl2bash_judge_model.responses_api_models.vllm_model.model=${NRL_NL2BASH_JUDGE_MODEL_PATH} \
env.nemo_gym.safety_judge_model.responses_api_models.vllm_model.model=${NRL_SAFETY_MODEL_PATH} \
data.train_jsonl_fpath=${NRL_TRAIN_PATH} \
data.validation_jsonl_fpath=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${CHECKPOINT_DIR}/logs \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ}"


if [[ "${USE_WORKTREE}" == "1" ]]; then
  MOUNTS="${MOUNTS},\
${WORKTREE_ROOT}:${WORKTREE_ROOT},\
${WORKTREE_ROOT}/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym,\
${WORKTREE_ROOT}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM,\
${MAIN_REPO_ROOT}/3rdparty/vllm:/opt/nemo-rl/3rdparty/vllm"
fi

if [[ -n "${EXTRA_MOUNTS:-}" ]]; then
  MOUNTS="${MOUNTS},${EXTRA_MOUNTS}"
fi

export MOUNTS

# Resolve ray.sub
if [[ "${USE_WORKTREE}" == "1" ]]; then
  RAY_SUB="${WORKTREE_ROOT}/ray.sub"
else
  RAY_SUB="${RAY_SUB:-${PROJECT_ROOT}/ray.sub}"
fi

if [[ ! -f "${RAY_SUB}" ]]; then
  echo "ERROR: ray.sub not found at ${RAY_SUB}"
  echo "Set RAY_SUB=/path/to/ray.sub or use USE_WORKTREE=1"
  exit 1
fi

# =============================================================================
# Interactive mode
# =============================================================================
# When COMMAND is empty/unset, ray.sub starts the Ray cluster then idles.
# It creates $SLURM_SUBMIT_DIR/<jobid>-attach.sh which supports:
#   bash <jobid>-attach.sh              # interactive shell on head node
#   bash <jobid>-attach.sh 1            # interactive shell on worker 1
#   COMMAND='...' bash <jobid>-attach.sh # run command non-interactively
#
# We save the training command to <jobid>-run-cmd.sh so the user can:
#   1. Attach interactively and source/paste it
#   2. Run non-interactively: COMMAND="$(cat <jobid>-run-cmd.sh)" bash <jobid>-attach.sh
#   3. Edit and re-run without requeueing
#
# A background watcher auto-runs the training command as soon as Ray is ready,
# so the scheduler never preempts the job for idle GPUs. After training finishes
# the allocation stays alive — re-attach and iterate without requeueing.
# =============================================================================
if [[ "${INTERACTIVE}" == "1" ]]; then
  # Ensure COMMAND is not in the environment. ray.sub does COMMAND=${COMMAND:-}
  # so unset → empty string → idle mode (creates attach script, sleeps forever).
  unset COMMAND 2>/dev/null || true

  # Interactive allocations default to 1h; INTERACTIVE_WALLTIME overrides.
  WALLTIME="${INTERACTIVE_WALLTIME:-1:0:0}"

  echo ""
  echo "================================================================"
  echo "  INTERACTIVE MODE"
  echo "================================================================"
  echo "  Submitting ${NUM_TOTAL_NODES}-node allocation (walltime: ${WALLTIME})"
  echo "  Ray cluster will start; training auto-runs when ready."
  echo ""

  submission_output=$(sbatch \
    --nodes="${NUM_TOTAL_NODES}" \
    --account="${SLURM_ACCOUNT}" \
    --job-name="interactive-${WANDB_NAME}" \
    --partition=batch \
    --time="${WALLTIME}" \
    --gres=gpu:4 \
    --exclusive \
    "${RAY_SUB}")

  echo "${submission_output}"

  if [[ "${submission_output}" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then
    JOB_ID="${BASH_REMATCH[1]}"
  else
    echo "ERROR: Could not parse job ID from sbatch output."
    exit 1
  fi

  # ray.sub writes the attach script to $SLURM_SUBMIT_DIR/<jobid>-attach.sh.
  # SLURM_SUBMIT_DIR is the cwd when sbatch was invoked, which is our $(pwd).
  LAUNCH_DIR="$(pwd)"
  ATTACH_SCRIPT="${LAUNCH_DIR}/${JOB_ID}-attach.sh"
  CMD_FILE="${LAUNCH_DIR}/${JOB_ID}-run-cmd.sh"

  # Save the training command. This file is intended to be:
  #   - Sourced from inside an interactive attach session, OR
  #   - Passed via: COMMAND="$(cat <file>)" bash <jobid>-attach.sh
  cat > "${CMD_FILE}" <<CMDEOF
${TRAIN_CMD}
CMDEOF
  chmod +x "${CMD_FILE}"

  # -----------------------------------------------------------------
  # Background watcher — auto-runs training so GPUs are never idle
  # waiting for a human to type the first command.
  # Polls for the attach script, then fires the training command.
  # After training finishes the allocation stays alive (ray.sub idles)
  # so the user can re-attach and iterate.
  # -----------------------------------------------------------------
  WATCHER_LOG="${LAUNCH_DIR}/${JOB_ID}-watcher.log"

  nohup bash -c '
    set -euo pipefail
    ATTACH_SCRIPT="'"${ATTACH_SCRIPT}"'"
    CMD_FILE="'"${CMD_FILE}"'"
    JOB_ID="'"${JOB_ID}"'"

    echo "[$(date)] Watcher started for job ${JOB_ID}"
    echo "[$(date)] Polling for attach script: ${ATTACH_SCRIPT}"

    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "[$(date)] Job ${JOB_ID} is no longer in the queue. Exiting watcher."
        exit 1
      fi
      echo "[$(date)] Job state: ${state}"
      sleep 15
    done

    echo "[$(date)] Ray cluster ready. Auto-running training command..."
    COMMAND="$(cat "${CMD_FILE}")" bash "${ATTACH_SCRIPT}"
    rc=$?
    echo "[$(date)] Training command finished (exit code: ${rc})."
    echo "[$(date)] Allocation is still alive — re-attach with:"
    echo "  bash ${ATTACH_SCRIPT}"
  ' > "${WATCHER_LOG}" 2>&1 &

  WATCHER_PID=$!
  disown "${WATCHER_PID}"

  echo ""
  echo "  Saved training command to:"
  echo "    ${CMD_FILE}"
  echo ""
  echo "  Background watcher running (PID: ${WATCHER_PID})"
  echo "    Log: ${WATCHER_LOG}"
  echo "    tail -f ${WATCHER_LOG}"
  echo ""
  echo "  Training will auto-start when Ray is ready, even if you're away."
  echo ""
  echo "  After training finishes, the allocation stays alive. Re-attach with:"
  echo "    bash ${ATTACH_SCRIPT}"
  echo "    source ${CMD_FILE}   # edit and re-run"
  echo ""
  echo "  Cancel: scancel ${JOB_ID}"
  echo "  Kill watcher: kill ${WATCHER_PID}"

  if [[ "${INTERACTIVE_WAIT}" == "1" ]]; then
    echo ""
    echo "  Also waiting in foreground (Ctrl+C is safe — watcher continues)..."
    echo ""

    # Foreground poll — purely for UX. The watcher handles the real work.
    prev_state=""
    while [[ ! -f "${ATTACH_SCRIPT}" ]]; do
      state=$(squeue -j "${JOB_ID}" -h -o "%T" 2>/dev/null || true)
      if [[ -z "${state}" ]]; then
        echo "  Job ${JOB_ID} is no longer in the queue. Check: sacct -j ${JOB_ID}"
        echo "  (Watcher may have already handled this — check ${WATCHER_LOG})"
        exit 1
      fi
      if [[ "${state}" != "${prev_state}" ]]; then
        echo "  [$(date +%H:%M:%S)] Job state: ${state}"
        prev_state="${state}"
      fi
      sleep 15
    done

    echo ""
    echo "  Ray cluster is ready! Watcher is auto-running the training command."
    echo "  You can attach to monitor:"
    echo "    bash ${ATTACH_SCRIPT}"
    echo "    tail -f ${WATCHER_LOG}"
    echo ""
  fi

  exit 0
fi

# =============================================================================
# Batch mode — set COMMAND and submit
# =============================================================================
export COMMAND="${TRAIN_CMD}"

sbatch \
  --nodes="${NUM_TOTAL_NODES}" \
  --account="${SLURM_ACCOUNT}" \
  --job-name="${WANDB_NAME}" \
  --partition="${PARTITION}" \
  --time="${WALLTIME}" \
  --gres=gpu:4 \
  --exclusive \
  --dependency=singleton \
  "${RAY_SUB}"