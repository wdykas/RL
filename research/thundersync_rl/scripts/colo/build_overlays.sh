#!/bin/bash
# Build the Python overlays the colo kit puts ahead of the container venv (run ONCE, inside the container, on a node with
# the CUDA toolkit; the TE PyTorch extension compiles here). Outputs under $COLO_OUT/overlays:
#   overlay_lens  nemo-lens at NeMo-RL's pyproject pin (telemetry imports)
#   overlay_bi    batch-invariant stack: flash-attn-4 4.0.0b19 + CuTe DSL 4.6.2 + quack 0.6.4, Transformer Engine 2.18.0
#   overlay_tms   torch_memory_saver 0.0.10 (prebuilt abi3 wheel for this machine's architecture)
#   srun ... --container-image=$COLO_IMAGE --container-mounts=$COLO_CONTAINER_MOUNTS bash build_overlays.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PY=${PY:-/opt/nemo_rl_venv/bin/python}
mkdir -p "$COLO_OUT/overlays"

# nemo-lens (same rev as NeMo-RL's pyproject.toml [tool.uv.sources]).
LENS_REV=$(sed -n 's/^nemo-lens = { git = "\(.*\)", rev = "\(.*\)" }/\1@\2/p' "$COLO_RL/pyproject.toml" | head -1)
$PY -m pip install --no-deps --target "$COLO_OVERLAY_UP" --upgrade "nemo-lens @ git+${LENS_REV%@*}@${LENS_REV##*@}"

# Batch-invariance stack.
O=$COLO_OVERLAY_BI; mkdir -p "$O"
$PY -m pip install --no-deps --target "$O" --upgrade \
  "flash-attn-4==4.0.0b19" "quack-kernels==0.6.4" "torch-c-dlpack-ext" \
  "nvidia-cutlass-dsl==4.6.2" "nvidia-cutlass-dsl-libs-base==4.6.2" "nvidia-cutlass-dsl-libs-core==4.6.2" \
  "nvidia-cutlass-dsl-libs-cu13==4.6.2" "nvidia-cutlass-dsl-libs-cu12==4.6.2" "apache-tvm-ffi==0.1.11"
export NVTE_FRAMEWORK=pytorch MAX_JOBS=${MAX_JOBS:-64} CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
$PY -m pip install --no-deps --no-build-isolation --target "$O" --upgrade \
  "transformer-engine==2.18.0" "transformer-engine-cu13==2.18.0" "transformer-engine-torch==2.18.0"
# flash-attn-4 installs flash_attn/cute only; the container's flash_attn would shadow it, so make the overlay's
# flash_attn a full package (the container's v2 modules + cute).
FA=$($PY -c "import flash_attn, os; print(os.path.dirname(flash_attn.__file__))" 2>/dev/null || true)
[ -n "$FA" ] && [ "$FA" != "$O/flash_attn" ] && cp -rn "$FA/." "$O/flash_attn/"
PYTHONPATH=$O:$O/nvidia_cutlass_dsl/dsl_packages:${PYTHONPATH:-} $PY -c \
  "import transformer_engine, transformer_engine.pytorch, flash_attn.cute; print('[overlays] TE', transformer_engine.__version__, 'FA4 ok')"

# torch_memory_saver 0.0.10 (pinned), unpacked prebuilt wheel.
T=$(mktemp -d); ARCH=$(uname -m); PV=$($PY -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
$PY -m pip download --no-cache-dir --no-deps --only-binary=:all: --platform manylinux2014_$ARCH --python-version $PV \
  --implementation cp --abi abi3 "torch-memory-saver==0.0.10" -d "$T" -q
rm -rf "$COLO_OVERLAY_TMS"; mkdir -p "$COLO_OVERLAY_TMS"
$PY -m zipfile -e "$T"/torch_memory_saver-0.0.10-*.whl "$COLO_OVERLAY_TMS"
sha256sum "$T"/*.whl | tee "$COLO_OVERLAY_TMS/WHEEL_SHA256"; rm -rf "$T"
echo "[overlays] OK: $COLO_OUT/overlays"
