#!/bin/bash
# Example launcher for the Nano multi-depth MTP draft head ("c3"). Everything comes from environment variables;
# run it inside a NeMo-RL (this fork) environment with Megatron-Core / Megatron-Bridge installed, on one node.
#
#   STAGE=<stage> [vars...] bash run_example.sh [extra args passed to the stage's script]
#
# Required for every stage:
#   OUT              output directory for this stage (created)
# Stage-specific:
#   prep         ROWDUMP (row-dump root: <run>/rows_step*.jsonl), DRAFT_HEAD_DATA (output dir for train/heldout.pkl)
#   shim         NANO_HF_DIR (local HF snapshot of NVIDIA-Nemotron-3-Nano-30B-A3B-BF16); optional HEAD_SAFETENSORS
#                (trained HF-named mtp.safetensors; empty = fresh init). Writes $OUT/shim.
#   train1       SHIM, DRAFT_HEAD_DATA                 depth-1 head (train_head.py) -> $OUT/head_mcore.pt, mtp.safetensors
#   chain        SHIM, INIT (depth-1 head_mcore.pt), DRAFT_HEAD_DATA; MODE=smoke|train|eval (default train)
#                multi-depth chained head (train_chain.py; the c3 recipe) -> $OUT/head_mcore.pt
#   v2           SHIM, INIT, DRAFT_HEAD_DATA; MODE=train|eval   round-2 levers (train_v2.py: --ffn/--feats/--ent_gamma)
#   walk         SHIM, HEAD, DRAFT_HEAD_DATA          held-out keyed acceptance + simulated chain walk (eval_walk.py)
#   audit        SHIM, HEAD, DATA_PKL                 acceptance audit vs first-principles bounds (audit_a.py)
#   check        SHIM, HEAD, DRAFT_HEAD_DATA          engine tree_mtp.MTPHead vs training arithmetic (check_engine_head.py)
#   long         NANO_HF_DIR, HEAD, DRAFT_HEAD_DATA   real long rollouts in the engine: plain vs tree+MTP (bitwise)
#   bench        (none)                               draft cost per depth, CUDA-graph replay (mtp_bench2.py)
# Optional: NPROC (GPUs, default 4), TLIM (seconds, default 14000), PYTHON (default python), MASTER_PORT.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE=${STAGE:?set STAGE (prep|shim|train1|chain|v2|walk|audit|check|long|bench)}
OUT=${OUT:?set OUT (output directory)}
PY=${PYTHON:-python}
NPROC=${NPROC:-4}
TLIM=${TLIM:-14000}
PORT=${MASTER_PORT:-$((20000 + RANDOM % 20000))}
mkdir -p "$OUT"
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:0:0}
run_dist() { timeout "$TLIM" "$PY" -m torch.distributed.run --nproc-per-node "$NPROC" --master-port "$PORT" "$@"; }
need() { for v in "$@"; do [ -n "${!v:-}" ] || { echo "run_example.sh: STAGE=$STAGE needs $v" >&2; exit 2; }; done; }

case "$STAGE" in
  prep)   need ROWDUMP DRAFT_HEAD_DATA
          "$PY" "$HERE/prep_data.py" --src "$ROWDUMP" --out "$DRAFT_HEAD_DATA" "$@" ;;
  shim)   need NANO_HF_DIR
          "$PY" "$HERE/mk_shim.py" --nano "$NANO_HF_DIR" --out "$OUT/shim" ${HEAD_SAFETENSORS:+--head "$HEAD_SAFETENSORS"} "$@" ;;
  train1) need SHIM DRAFT_HEAD_DATA
          run_dist "$HERE/train_head.py" --mode train --shim "$SHIM" --out "$OUT" "$@" 2>&1 | tee "$OUT/train1.log" ;;
  chain)  need SHIM INIT DRAFT_HEAD_DATA
          # c3 recipe: depths 1-3 unrolled, lr 5e-4 cosine, 4 rows/rank x 4 ranks per step, ~2.8 epochs
          run_dist "$HERE/train_chain.py" --mode "${MODE:-train}" --shim "$SHIM" --init "$INIT" --out "$OUT" \
            --depth 3 --depth_w 1,1,1 --lr 5e-4 --accum 4 --epochs 2.78 "$@" 2>&1 | tee "$OUT/chain_${MODE:-train}.log" ;;
  v2)     need SHIM INIT DRAFT_HEAD_DATA
          run_dist "$HERE/train_v2.py" --mode "${MODE:-train}" --shim "$SHIM" --init "$INIT" --out "$OUT" "$@" \
            2>&1 | tee "$OUT/v2_${MODE:-train}.log" ;;
  walk)   need SHIM HEAD DRAFT_HEAD_DATA
          run_dist "$HERE/eval_walk.py" --shim "$SHIM" --init "$HEAD" --out "$OUT" --depth 4 --keys 4 "$@" \
            2>&1 | tee "$OUT/walk.log" ;;
  audit)  need SHIM HEAD DATA_PKL
          run_dist "$HERE/audit_a.py" --shim "$SHIM" --init "$HEAD" --data "$DATA_PKL" --tag "${TAG:-heldout}" \
            --out "$OUT" "$@" 2>&1 | tee "$OUT/audit.log" ;;
  check)  need SHIM HEAD DRAFT_HEAD_DATA
          run_dist "$HERE/check_engine_head.py" --shim "$SHIM" --init "$HEAD" --heldout "$DRAFT_HEAD_DATA/heldout.pkl" \
            --out "$OUT" "$@" 2>&1 | tee "$OUT/check.log" ;;
  long)   need NANO_HF_DIR HEAD DRAFT_HEAD_DATA
          E="$HERE/tree_engine_long.py"
          EA=(--model "$NANO_HF_DIR" --prompts real --prompt_keep 0 --n 24 --tails 24 --ntok "${NTOK:-7992}"
              --eos "${EOS:-11}" --max_len 8800 --buffer_gb 30 --repeat 1)
          CUDA_VISIBLE_DEVICES=${GPU:-0} timeout "$TLIM" "$PY" -m torch.distributed.run --nproc-per-node 1 \
            --master-port "$PORT" "$E" --arm plain "${EA[@]}" --out "$OUT/plain" "$@" > "$OUT/plain.log" 2>&1
          # tree + MTP draft source, W16 D4, 128 node rows; must be bitwise equal to the plain arm
          CUDA_VISIBLE_DEVICES=${GPU:-0} timeout "$TLIM" "$PY" -m torch.distributed.run --nproc-per-node 1 \
            --master-port "$((PORT + 1))" "$E" --arm tree "${EA[@]}" --W 16 --D 4 --rows 128 --kinds none \
            --mtp "$HEAD" --ref "$OUT/plain" --out "$OUT/mtp" "$@" > "$OUT/mtp.log" 2>&1
          "$PY" - "$OUT" <<'PYEOF'
import json, sys
o = sys.argv[1]
for f in ("plain", "mtp"):
    d = json.load(open(f"{o}/{f}.rank0.json")); t = d.get("tree", {}) or {}; fin = d.get("fin", {})
    s = sorted(v[2] / max(v[1] - 1, 1) for v in fin.values())
    print(f, "exact", d.get("exact"), "mismatch", len(d.get("mismatch", []) or []), "steps", d["steps"],
          "tok/iter", t.get("tokens_per_tail_iter"), "rates_head", t.get("rates_head"),
          "slowest-row s", round(s[0], 3) if s else None, "median-row s", round(s[len(s) // 2], 3) if s else None)
PYEOF
          ;;
  bench)  for v in "1856 0" "3712 0" "7424 0"; do set -- $v
            echo "FFN=$1 NF=$2"; FFN=$1 NF=$2 TMPDIR="$OUT" timeout 900 "$PY" "$HERE/mtp_bench2.py" | tail -20; done ;;
  *) echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac
