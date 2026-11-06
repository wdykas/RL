#!/bin/bash
# One-time setup per user (login node, no GPU; writes only under $COLO_OUT): output dirs, your own HF home, the pins file, and read checks on the shared inputs.
#   bash research/thundersync_rl/scripts/colo/setup.sh            (re-run whenever a tree moves or changes commit: the pins name absolute paths + hashes)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
echo "[setup] share $COLO_SHARE -> outputs $COLO_OUT"
mkdir -p "$COLO_RUNS" "$COLO_OUT/runs_wd" "$COLO_OUT/slurm" "$COLO_OUT/cache" "$NRL_MEGATRON_CHECKPOINT_DIR" "$HF_HOME/hub"

# HF home (the hub / datasets caches write lock files, so they must be yours): Nemotron-3 Nano (~60 GB) and
# DAPO-Math-17k, downloaded once here (needs network on this node; jobs run offline). COLO_SKIP_DOWNLOAD=1 skips it.
if [ "${COLO_SKIP_DOWNLOAD:-0}" != 1 ]; then
  HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")
snapshot_download("BytedTsinghua-SIA/DAPO-Math-17k", repo_type="dataset")
import datasets
datasets.load_dataset("BytedTsinghua-SIA/DAPO-Math-17k")   # prepares the arrow shards the offline jobs load
print("[setup] HF downloads OK")
PY
fi
N=models--nvidia--NVIDIA-Nemotron-3-Nano-30B-A3B-BF16

# Read checks on everything the jobs load.
for m in "$MODEL_SUPER_L8" "$MODEL_SUPER_L22" "$MODEL_ULTRA_L27"; do
  [ -r "$m/config.json" ] || echo "[setup] note: no $m (only needed for the Super / Ultra presets)"
done
for p in \
         "$HF_HOME/hub/$N/snapshots" "$COLO_OVERLAY" "$COLO_OVERLAY_UP" "$COLO_OVERLAY_BI/transformer_engine" \
         "$COLO_OVERLAY_TMS/torch_memory_saver" \
         "$COLO_RL/nemo_rl" "$COLO_RL_STOCK/nemo_rl" "$COLO_MLM/megatron/core" "$COLO_MBRIDGE/megatron/bridge" \
         "$COLO_MLM_STOCK/megatron/core" "$COLO_MBRIDGE_STOCK_TREE/src/megatron/bridge"; do
  [ -r "$p" ] || { echo "[setup] FAIL: cannot read $p" >&2; exit 1; }
done
# The code trees must be clean (the provenance check refuses tracked changes in the pinned Megatron trees).
for t in "$COLO_RL" "$COLO_RL_STOCK" "$COLO_MLM" "$COLO_MBRIDGE_TREE" "$COLO_MLM_STOCK" "$COLO_MBRIDGE_STOCK_TREE"; do
  d=$(git -C "$t" status --porcelain --untracked-files=no | wc -l)
  echo "[setup] $(basename $t) @ $(git -C "$t" rev-parse --short=9 HEAD) tracked changes: $d"
done

# Pins: the trees every process must load megatron.core / megatron.bridge / transformer_engine from. Two sets: ours
# (and SPR-BI) on Megatron-LM / Bridge main, stock NeMo-RL on its own pins.
write_pins() {  # <rl tree> <mlm tree> <bridge tree> <out>
  local mls mbs; mls=$(git -C "$2" rev-parse HEAD); mbs=$(git -C "$3" rev-parse HEAD)
  python3 - "$COLO_RL" "$2" "$mls" "$3/src" "$mbs" "$COLO_OVERLAY_BI" > "$4.tmp" <<'PY'
import importlib.util, sys
t, mlm, mls, mb, mbs, te = sys.argv[1:]
spec = importlib.util.spec_from_file_location("mp", f"{t}/nemo_rl/utils/module_provenance.py")
mp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mp)
print("# written by the colo kit setup.sh; checked at startup by nemo_rl/utils/module_provenance.py")
print(f"MEGATRON_LM_ROOT={mlm}\nMEGATRON_LM_SHA={mls}\nMEGATRON_LM_HASH={mp.tree_hash(mlm, 'megatron.core')}")
print(f"MEGATRON_BRIDGE_ROOT={mb}\nMEGATRON_BRIDGE_SHA={mbs}\nMEGATRON_BRIDGE_HASH={mp.tree_hash(mb, 'megatron.bridge')}")
print(f"TE_ROOT={te}\nTE_HASH={mp.tree_hash(te, 'transformer_engine')}")
PY
  mv "$4.tmp" "$4"; echo "[setup] pins $4:"; grep -E "_ROOT|_SHA" "$4"
}
write_pins "$COLO_RL" "$COLO_MLM" "$COLO_MBRIDGE_TREE" "$COLO_PINS"
write_pins "$COLO_RL" "$COLO_MLM_STOCK" "$COLO_MBRIDGE_STOCK_TREE" "$COLO_PINS_STOCK"
echo "[setup] OK. Next: bash research/thundersync_rl/scripts/colo/submit.sh smoke"
