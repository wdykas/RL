# Colo kit environment. Source it: `source research/thundersync_rl/scripts/colo/env.sh` (login node and inside the
# container). Everything site-specific is a REQUIRED env var (export before sourcing); nothing here names a cluster.
#
# ---- REQUIRED (no defaults) ----
#   COLO_OUT      where ALL outputs go: run dirs, logs, watchdog state, JIT caches, HF home, converted Megatron
#                 checkpoints, overlays. A shared filesystem visible from every node; never under $HOME.
#   COLO_IMAGE    the NeMo-RL container image for srun --container-image (pyxis / enroot), with /opt/nemo_rl_venv.
#                 See docs/colo/HANDOFF.md "Environment".
#   COLO_CONTAINER_MOUNTS   srun --container-mounts value covering the trees, $COLO_OUT and $COLO_MODELS
#                 (e.g. /shared:/shared).
#   COLO_MODELS   directory holding the truncated Nemotron-3 HF checkpoints nemotron3-super-L8 / -L22 and
#                 nemotron3-ultra-L27 (make them with make_trunc_nemotron_h.py). Not needed for Nano-only work: set it
#                 to any directory.
# ---- OPTIONAL ----
#   COLO_ACCOUNT / COLO_PARTITION   sbatch -A / -p (omitted when unset)
#   COLO_SEGMENT=0                  drop sbatch --segment=N (it asks for one NVLink domain; GB200 NVL72 clusters)
#   COLO_RL_STOCK  stock NeMo-RL tree for the SPR / CS baselines (branch handoff/stock-baseline on the fork)
#   COLO_MLM / COLO_MBRIDGE_TREE    Megatron-LM / Megatron-Bridge trees for our arms (default: $COLO_WORKSPACE/...)
#   COLO_MLM_STOCK / COLO_MBRIDGE_STOCK_TREE   the stock baseline's Megatron-LM / Bridge trees
#   COLO_WORKSPACE  directory holding the sibling checkouts named below (default: the parent of this RL checkout)
: "${COLO_OUT:?set COLO_OUT (output root on a shared filesystem; see docs/colo/HANDOFF.md)}"
: "${COLO_IMAGE:?set COLO_IMAGE (NeMo-RL container image; see docs/colo/HANDOFF.md)}"
: "${COLO_CONTAINER_MOUNTS:?set COLO_CONTAINER_MOUNTS (srun --container-mounts value, e.g. /shared:/shared)}"
: "${COLO_MODELS:?set COLO_MODELS (dir with nemotron3-super-L8 / -L22 / nemotron3-ultra-L27 HF checkpoints)}"
export COLO_OUT COLO_IMAGE COLO_CONTAINER_MOUNTS COLO_MODELS
export COLO_ACCOUNT=${COLO_ACCOUNT:-} COLO_PARTITION=${COLO_PARTITION:-}
# Production (speculative exact colocated, fixed G, verify at the wake: arm wvs) config, relative to the RL tree, and its lag K.
export COLO_PROD_CFG=${COLO_PROD_CFG:-research/thundersync_rl/configs/colo/colo_wv.yaml}
export COLO_PROD_K=${COLO_PROD_K:-1}
#
# Python overlays put ahead of the container's site-packages (built once by build_overlays.sh under $COLO_OUT/overlays):
#   overlay_lens nemo-lens (NeMo-RL's pyproject pin; the telemetry imports need span_attributes)
#   overlay_bi   batch-invariant stack: TransformerEngine 2.18, flash-attn-4 4.0.0b19, CuTe DSL 4.6.2 (ours AND stock
#                arms load it)
#   overlay_tms  torch_memory_saver 0.0.10: colocated engines release their scratch and inference weights during
#                training. COLO_TMS=1 (default) puts it on our arms' path (ours and SPR-BI); stock arms run without it
#                (their best config) unless COLO_TMS_STOCK=1.
# COLO_KV: engine KV during training, persist (default: every arm keeps its KV resident, the strongest stock config)
#          or offload (only where memory forces it; it costs stock SPR ~4.7 s/step at Nano 8k and our arms ~0, so it
#          is not a fair default).
export COLO_OVERLAY_UP=${COLO_OVERLAY_UP:-$COLO_OUT/overlays/overlay_lens} COLO_OVERLAY=${COLO_OVERLAY:-$COLO_OUT/overlays/overlay_lens}
export COLO_OVERLAY_BI=${COLO_OVERLAY_BI:-$COLO_OUT/overlays/overlay_bi}
export COLO_OVERLAY_TMS=${COLO_OVERLAY_TMS:-$COLO_OUT/overlays/overlay_tms} COLO_TMS=${COLO_TMS:-1} COLO_KV=${COLO_KV:-persist}
#
# ---- derived (do not edit) ----
export COLO_SCRIPTS=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export COLO_RL=${COLO_RL:-$(cd "$COLO_SCRIPTS/../../../.." && pwd)}   # this checkout (branch colo-handoff)
export COLO_WORKSPACE=${COLO_WORKSPACE:-$(dirname "$COLO_RL")}
export COLO_SHARE=$COLO_WORKSPACE
export COLO_RL_STOCK=${COLO_RL_STOCK:-$COLO_WORKSPACE/RL-stock}  # stock NeMo-RL baseline tree (SPR / CS arms)
# Ours (and SPR-BI, which runs on our tree): Megatron-LM main a11c84aec + Megatron-Bridge main 9981c93 (no patches).
export COLO_MLM=${COLO_MLM:-$COLO_WORKSPACE/Megatron-LM-main}
export COLO_MBRIDGE_TREE=${COLO_MBRIDGE_TREE:-$COLO_WORKSPACE/Megatron-Bridge-main}
# Stock NeMo-RL (arms spr / cs) as is, on its own pins: Megatron-LM 6a3660905 + NVIDIA/Megatron-LM#8358 (cherry-pick),
# Bridge 1f8873bb. (Stock does not run on Megatron-LM main: the API moved under it.)
export COLO_MLM_STOCK=${COLO_MLM_STOCK:-$COLO_WORKSPACE/Megatron-LM-stock}
export COLO_MBRIDGE_STOCK_TREE=${COLO_MBRIDGE_STOCK_TREE:-$COLO_WORKSPACE/Megatron-Bridge-stock}
export COLO_MBRIDGE=$COLO_MBRIDGE_TREE/src
export COLO_PINS=$COLO_OUT/colo.pins COLO_PINS_STOCK=$COLO_OUT/colo.pins.stock   # written by setup.sh
export COLO_RUNS=$COLO_OUT/runs
case "$COLO_OUT" in "$HOME"|"$HOME"/*) echo "[env] COLO_OUT=$COLO_OUT is under $HOME: use a shared filesystem with quota" >&2; return 2 2>/dev/null || exit 2;; esac

# Model checkpoints (presets in lib/presets.sh use these names).
export MODEL_SUPER_L8=$COLO_MODELS/nemotron3-super-L8 MODEL_SUPER_L22=$COLO_MODELS/nemotron3-super-L22
export MODEL_ULTRA_L27=$COLO_MODELS/nemotron3-ultra-L27 MODEL_NANO=nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# Runtime env.
export NRL_IGNORE_VERSION_MISMATCH=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export NEMO_RL_PY_EXECUTABLES_SYSTEM=1 WANDB_MODE=disabled RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1
# Your own HF home (lock files and the dataset arrow cache are written there); setup.sh downloads Nano and DAPO-Math-17k into it.
export HF_HOME=$COLO_OUT/hf_home
# HF -> Megatron conversions of the checkpoints land here on first use (one-time, per model).
export NRL_MEGATRON_CHECKPOINT_DIR=$COLO_OUT/megatron_ckpt
# JIT / compile caches under $COLO_OUT.
export TRITON_CACHE_DIR=$COLO_OUT/cache/triton TORCHINDUCTOR_CACHE_DIR=$COLO_OUT/cache/inductor
export CUDA_CACHE_PATH=$COLO_OUT/cache/cuda MPLCONFIGDIR=$COLO_OUT/cache/mpl TORCH_EXTENSIONS_DIR=$COLO_OUT/cache/torch_extensions
# Shared checkouts may be owned by another user: let git (used by the startup provenance check) read them.
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=safe.directory GIT_CONFIG_VALUE_0='*'
# Pinned Megatron-LM / Megatron-Bridge / TE: every process checks the loaded trees against this file at startup
# (nemo_rl/utils/module_provenance.py: root, git commit, content hash). Written by $COLO_SCRIPTS/setup.sh.
export NEMO_RL_SNAPSHOT_PINS=$COLO_PINS

# colo_pythonpath <rl tree> [bi]: the PYTHONPATH of one arm .
# COLO_STOCK=1 (stock_arm.sh): the stock pins (Megatron-LM / Bridge trees and pins file) instead of ours.
colo_pythonpath() {
  local t=$1 mlm=$COLO_MLM mb=$COLO_MBRIDGE
  local tms=$COLO_TMS
  if [ "${COLO_STOCK:-0}" = 1 ]; then
    mlm=$COLO_MLM_STOCK mb=$COLO_MBRIDGE_STOCK_TREE/src; export NEMO_RL_SNAPSHOT_PINS=$COLO_PINS_STOCK
    # Stock runs its best config: torch_memory_saver costs stock SPR +3.5 s/step at Nano 8k (same-job A/B:
    # 58.93 on vs 55.43 off, mostly weight sync), so stock arms leave it off unless COLO_TMS_STOCK=1.
    tms=${COLO_TMS_STOCK:-0}
  fi
  # HF -> Megatron conversions are Megatron-LM-version specific: one directory per Megatron-LM tree.
  export NRL_MEGATRON_CHECKPOINT_DIR=$COLO_OUT/megatron_ckpt/$(basename "$mlm")
  mkdir -p "$NRL_MEGATRON_CHECKPOINT_DIR"
  export RL_ROOT=$t
  export PYTHONPATH=$mb:$mlm:$t/research/thundersync_rl:$t:$COLO_OVERLAY
  if [ "${2:-}" = bi ]; then
    export PYTHONPATH=$COLO_OVERLAY_BI:$COLO_OVERLAY_BI/nvidia_cutlass_dsl/dsl_packages:$PYTHONPATH
    export CUBLASLT_WORKSPACE_SIZE=0 CUBLAS_WORKSPACE_CONFIG=:0:0
  fi
  [ "$tms" = 1 ] && export PYTHONPATH=$COLO_OVERLAY_TMS:$PYTHONPATH
  export PYTHONPATH=${COLO_PREPEND:+$COLO_PREPEND:}$COLO_OVERLAY_UP:$PYTHONPATH
}
