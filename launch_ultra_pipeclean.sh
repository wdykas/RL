#!/bin/bash
set -euo pipefail

# =============================================================================
# launch_ultra_pipeclean.sh
#
# GRPO Ultra V3 pipe-cleaning on GB200 NVL72 with NeMo Gym
#
# By default, this runs from what's built into the container without overlay mounts applied. 
# Set USE_WORKTREE=1 to overlay your local worktree submodules for development.
# Set INTERACTIVE=1 to get a persistent allocation in slurm for iterative debugging.
#
# Usage:
#   ./launch_ultra_pipeclean.sh                                   # batch, bare container (10 steps)
#   NRL_MAX_STEPS=4 ./launch_ultra_pipeclean.sh                   # CI: fewer steps
#   USE_WORKTREE=1 ./launch_ultra_pipeclean.sh                    # batch, overlay local code
#   WALLTIME=4:00:00 ./launch_ultra_pipeclean.sh
#
# Extra positional arguments are forwarded as Hydra overrides:
#   ./launch_ultra_pipeclean.sh grpo.max_num_steps=2 policy.precision=float32
#
# Interactive debugging (reuse allocation across runs):
#   INTERACTIVE=1 ./launch_ultra_pipeclean.sh                     # submits, auto-runs, waits
#   INTERACTIVE=1 INTERACTIVE_WAIT=0 ./launch_ultra_pipeclean.sh  # submit only (no foreground wait)
#   INTERACTIVE=1 INTERACTIVE_WALLTIME=2:0:0 SLURM_QOS=short ./launch_ultra_pipeclean.sh  # submit and wait in foreground
#   INTERACTIVE=1 INTERACTIVE_WALLTIME=8:0:0 ./launch_ultra_pipeclean.sh  # longer allocation
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
SLURM_QOS="${SLURM_QOS:-}"
WALLTIME="${WALLTIME:-4:00:00}"

# ---------- Container & mounts ----------
export CONTAINER="${CONTAINER:-/lustre/fsw/portfolios/llmservice/users/ansubramania/containers/nemo-rl-ultra-20260226-428eb84dd-custom-vllm-arm.sqsh}"
MOUNTS="/lustre:/lustre"

# GB200 NVL72: fixed at 4 GPUs/node. Must match --gres=gpu:4 passed to sbatch.
export GPUS_PER_NODE=4
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-144}"

# ---------- HuggingFace Configuration ----------
export HF_HOME="${HF_HOME:-}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-}"

# ---------- W&B Configuration ----------
WANDB_PROJ="${WANDB_PROJ:-grpo-ultra-v3-pipeclean}"
WANDB_NAME="${WANDB_NAME:-ultra-v3-grpo-$(date +%m%d-%H%M)}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

# ---------- Training ----------
NRL_MAX_STEPS="${NRL_MAX_STEPS:-}"

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

# GB200 NVL72: each rack has 18 nodes sharing an NVLink domain.
# --segment tells SLURM to allocate nodes in groups of this size from
# the same topology block, guaranteeing complete rack-aligned segments for
# training EP. Inference and judges inherit the constraint but don't require it.
# Must stay in sync with cluster.segment_size in the YAML config.
#
# When SEGMENT_SIZE is unset, default to 16 if NUM_TOTAL_NODES >= 16.
# When NUM_TOTAL_NODES < segment size, skip --segment to avoid sbatch failures.
SEGMENT_SIZE="${SEGMENT_SIZE:-}"
if [ -z "${SEGMENT_SIZE}" ] && [ "${NUM_TOTAL_NODES}" -ge 16 ]; then
  SEGMENT_SIZE=16
fi
if [ -n "${SEGMENT_SIZE}" ] && [ "${NUM_TOTAL_NODES}" -lt "${SEGMENT_SIZE}" ]; then
  echo "ERROR: NUM_TOTAL_NODES=${NUM_TOTAL_NODES} < SEGMENT_SIZE=${SEGMENT_SIZE}" >&2
  exit 1
fi

# ---------- Model and data paths ----------
NRL_TRAIN_PATH="${NRL_TRAIN_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.no-swerl.max16k.train.jsonl}"
NRL_VAL_PATH="${NRL_VAL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/data/gym/rl-data-tools/blends/curriculum_v29_warping-muskox.no-swerl.max16k.val.jsonl}"
NRL_MODEL_PATH="${NRL_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/adithyare/nemotron_ultra/sft-runs/ultra-v3-sft-hsg-mainfeb5merge-mxfp8_newbase/hf_converted}"
NRL_GENRM_MODEL_PATH="${NRL_GENRM_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/qwen235b_principle_comparison_genrm_step1230}"
NRL_NL2BASH_JUDGE_MODEL_PATH="${NRL_NL2BASH_JUDGE_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/models/Qwen3-235B-A22B-Instruct-2507-FP8}"
NRL_SAFETY_MODEL_PATH="${NRL_SAFETY_MODEL_PATH:-/lustre/fsw/portfolios/llmservice/users/ansubramania/super_v3/model_checkpoints/Nemotron-Content-Safety-Reasoning-4B}"

# ---------- Lean4 sandbox (for math_formal_lean) ----------
export SANDBOX_CONTAINER="${SANDBOX_CONTAINER:-/lustre/fsw/portfolios/llmservice/users/igitman/images/nemo-skills-sandbox-latest.sqsh}"
export SANDBOX_COMMAND="${SANDBOX_COMMAND:-/start-with-nginx.sh}"
export NEMO_SKILLS_SANDBOX_PORT="${NEMO_SKILLS_SANDBOX_PORT:-6000}"

# ---------- Ray log sync (copy actor logs from /tmp/ray to $LOG_DIR/ray/) ----------
export RAY_LOG_SYNC_FREQUENCY="${RAY_LOG_SYNC_FREQUENCY:-60}"

EXP_SUFFIX="${EXP_SUFFIX:-ultra-v3-grpo-pipeclean}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-results/${EXP_SUFFIX}}"
mkdir -p "${CHECKPOINT_DIR}"

# ---------- Code snapshot ----------
# Batch mode: snapshot by default so code is frozen at submission time.
# Interactive mode: live directory by default for fast iteration.
# Override with USE_SNAPSHOT=0 or USE_SNAPSHOT=1 to force either behavior.
if [[ "${INTERACTIVE}" == "1" ]]; then
  USE_SNAPSHOT="${USE_SNAPSHOT:-0}"
else
  USE_SNAPSHOT="${USE_SNAPSHOT:-1}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  SNAPSHOT_DIR=$(bash "${PROJECT_ROOT}/tools/code_snapshot.sh" "${EXP_SUFFIX}")

  # Symlink 3rdparty/vllm if present (large, not git-tracked in all setups)
  if [[ -d "${PROJECT_ROOT}/3rdparty/vllm" ]] && [[ ! -e "${SNAPSHOT_DIR}/3rdparty/vllm" ]]; then
    mkdir -p "${SNAPSHOT_DIR}/3rdparty"
    ln -s "${PROJECT_ROOT}/3rdparty/vllm" "${SNAPSHOT_DIR}/3rdparty/vllm"
  fi

  echo "Code snapshot: ${SNAPSHOT_DIR}"
  OVERLAY_SOURCE="${SNAPSHOT_DIR}"
else
  OVERLAY_SOURCE="${PROJECT_ROOT}"
fi

# ---------- Persistent cache directories ----------
# Shared project-level cache so all team members reuse compiled artifacts
# (vLLM, FlashInfer cubins, Deep Gemm JIT, uv). Directories use setgid
# (g+rwxs) so new files inherit the llmservice group and stay group-writable.
PERSISTENT_CACHE="${PERSISTENT_CACHE:-/lustre/fsw/portfolios/llmservice/projects/llmservice_nemotron_ultra/nemo_rl/persistent_cache}"
VLLM_CACHE_DIR="${PERSISTENT_CACHE}/vllm_compile_cache"
FLASHINFER_CUBIN_CACHE="${PERSISTENT_CACHE}/flashinfer_cubins"
FLASHINFER_WS_BASE="${PERSISTENT_CACHE}/flashinfer_workspace"
(umask 002 && mkdir -p "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}")
chmod g+rwxs "${PERSISTENT_CACHE}" "${VLLM_CACHE_DIR}" "${FLASHINFER_CUBIN_CACHE}" "${FLASHINFER_WS_BASE}" 2>/dev/null || true

VLLM_PRECOMPILED_WHEEL_LOCATION="${VLLM_PRECOMPILED_WHEEL_LOCATION:-https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_aarch64.whl}"

# =============================================================================
# Validation
# =============================================================================

# Walltime cap: Slurm partitions typically enforce <=4h; fail early.
_walltime_secs() {
  local t="$1" h m s
  IFS=: read -r h m s <<< "${t}"
  echo $(( 10#${h} * 3600 + 10#${m} * 60 + 10#${s} ))
}

if (( $(_walltime_secs "${WALLTIME}") > 4 * 3600 )); then
  echo "ERROR: WALLTIME=${WALLTIME} exceeds the 4-hour maximum."
  exit 1
fi

# QOS=interactive caps walltime at 2 hours.
if [[ "${SLURM_QOS}" == "interactive" ]]; then
  if (( $(_walltime_secs "${INTERACTIVE_WALLTIME:-${WALLTIME}}") > 2 * 3600 )); then
    echo "ERROR: SLURM_QOS=interactive requires walltime <= 2 hours."
    echo "  Set INTERACTIVE_WALLTIME=2:0:0 or use a different QOS (e.g. SLURM_QOS=short)."
    exit 1
  fi
fi

# W&B: warn (but don't fail) if WANDB_API_KEY is unset — runs will log locally only.
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WARNING: WANDB_API_KEY is not set. W&B logging will fail or fall back to offline mode."
  echo "  export WANDB_API_KEY=<your-key> to enable cloud logging."
fi

# HF_TOKEN: required when loading models from the HuggingFace Hub (not local paths).
# Hub IDs look like "org/model-name" (no leading slash). Local paths start with "/".
if [[ "${NRL_MODEL_PATH}" =~ ^[a-zA-Z0-9_-]+/[a-zA-Z0-9_./-]+$ ]]; then
  if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: NRL_MODEL_PATH (${NRL_MODEL_PATH}) looks like a HuggingFace Hub model ID"
    echo "  but HF_TOKEN is not set. Export HF_TOKEN to authenticate with the Hub."
    exit 1
  fi
fi

# =============================================================================
# Worktree setup (only when USE_WORKTREE=1)
# =============================================================================
if [[ "${USE_WORKTREE}" == "1" ]]; then
  WORKTREE_ROOT="$(cd "${SCRIPT_DIR}" && pwd)"
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
HF_HOME=${HF_HOME} \
HF_TOKEN=${HF_TOKEN:-} \
uv run ./examples/nemo_gym/run_grpo_nemo_gym.py \
--config examples/configs/grpo_ultra_64n4g_pipeclean.yaml \
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
data.train.data_path=${NRL_TRAIN_PATH} \
data.validation.data_path=${NRL_VAL_PATH} \
checkpointing.checkpoint_dir=${CHECKPOINT_DIR} \
logger.log_dir=${CHECKPOINT_DIR}/logs \
logger.wandb_enabled=True \
logger.wandb.name=${WANDB_NAME} \
logger.wandb.project=${WANDB_PROJ} \
${NRL_MAX_STEPS:+grpo.max_num_steps=${NRL_MAX_STEPS}} \
${*}"


# =============================================================================
# Overlay mounts
# =============================================================================
# Local source directories are bind-mounted into the container so edits on
# Lustre take effect without rebuilding the container. Each mount can be
# overridden via an env var or disabled by setting it to empty string.
#
#   NRL_NEMO_RL_DIR      → /opt/nemo-rl/nemo_rl          (Python package)
#   NRL_CONFIGS_DIR      → /opt/nemo-rl/examples/configs  (YAML configs)
#   NRL_MEGATRON_LM_DIR  → /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM
#   NRL_MEGATRON_BRIDGE_DIR → /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
#   NRL_GYM_DIR          → /opt/nemo-rl/3rdparty/Gym-workspace/Gym
#   NRL_VLLM_DIR         → /opt/nemo-rl/3rdparty/vllm
#
# Paths that don't exist on disk are silently skipped (container built-ins
# are used instead). Set any var to "" to explicitly skip that mount.
# =============================================================================
NRL_NEMO_RL_DIR="${NRL_NEMO_RL_DIR:-${OVERLAY_SOURCE}/nemo_rl}"
NRL_CONFIGS_DIR="${NRL_CONFIGS_DIR:-${OVERLAY_SOURCE}/examples/configs}"
NRL_MEGATRON_LM_DIR="${NRL_MEGATRON_LM_DIR:-${OVERLAY_SOURCE}/3rdparty/Megatron-LM-workspace/Megatron-LM}"
NRL_MEGATRON_BRIDGE_DIR="${NRL_MEGATRON_BRIDGE_DIR:-${OVERLAY_SOURCE}/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge}"
NRL_GYM_DIR="${NRL_GYM_DIR:-${OVERLAY_SOURCE}/3rdparty/Gym-workspace/Gym}"
NRL_VLLM_DIR="${NRL_VLLM_DIR:-}"  # No default; vLLM from container unless explicitly set

_maybe_mount() {
  local src="$1" dst="$2" label="$3"
  if [[ -z "${src}" ]]; then
    return
  fi
  if [[ -d "${src}" ]]; then
    MOUNTS="${MOUNTS},${src}:${dst}"
    echo "  Mount: ${label} → ${dst}"
  else
    echo "  Skip:  ${label} (${src} not found on disk, using container built-in)"
  fi
}

echo ""
echo "Overlay mounts:"
_maybe_mount "${NRL_NEMO_RL_DIR}" "/opt/nemo-rl/nemo_rl" "nemo_rl"
_maybe_mount "${NRL_CONFIGS_DIR}" "/opt/nemo-rl/examples/configs" "configs"
_maybe_mount "${NRL_MEGATRON_LM_DIR}" "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM" "Megatron-LM"
_maybe_mount "${NRL_MEGATRON_BRIDGE_DIR}" "/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge" "Megatron-Bridge"
_maybe_mount "${NRL_GYM_DIR}" "/opt/nemo-rl/3rdparty/Gym-workspace/Gym" "NeMo-Gym"
_maybe_mount "${NRL_VLLM_DIR}" "/opt/nemo-rl/3rdparty/vllm" "vLLM"

if [[ "${USE_WORKTREE}" == "1" ]]; then
  MOUNTS="${MOUNTS},${WORKTREE_ROOT}:${WORKTREE_ROOT}"
fi

if [[ "${USE_SNAPSHOT}" == "1" ]]; then
  MOUNTS="${MOUNTS},${SNAPSHOT_DIR}:${SNAPSHOT_DIR}"
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
  WALLTIME="${INTERACTIVE_WALLTIME:-4:0:0}"

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
    ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
    ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
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
  echo ""
  echo "  Between runs (clean up GPUs, clear caches, re-run):"
  echo "    python ${PROJECT_ROOT}/reset_ray_cluster.py"
  echo "    source ${CMD_FILE}"
  echo ""
  echo "  Edit the command and re-run without requeueing:"
  echo "    vim ${CMD_FILE}"
  echo "    source ${CMD_FILE}"
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
  ${SEGMENT_SIZE:+--segment="${SEGMENT_SIZE}"} \
  ${SLURM_QOS:+--qos="${SLURM_QOS}"} \
  "${RAY_SUB}"