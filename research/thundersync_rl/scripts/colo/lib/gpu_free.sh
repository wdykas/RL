#!/bin/bash
# Make this node's GPUs free before a 1-node run starts. Back-to-back arms in one job (gate pos -> neg, smoke, pair on
# 1 node) start a fresh local Ray per arm; the previous arm's Ray workers can outlive its driver for minutes while they
# still hold their CUDA memory, and the next arm then OOMs at its first allocation. Here: ray stop, TERM the GPU compute
# processes that are still ours, wait until their memory is released, KILL whatever is left after GPU_FREE_WAIT_S.
# Only for local Ray inside a Slurm job (the node is ours); multi-node clusters are torn down by ray_cluster.sh.
set -u
[ -n "${SLURM_JOB_ID:-}" ] || exit 0
ours() { local p; for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '); do
  [ "$(stat -c %u /proc/$p 2>/dev/null)" = "$(id -u)" ] && echo $p; done; }
left=$(ours)
[ -z "$left" ] && exit 0
echo "[gpu-free] GPU processes left from a previous arm: $(echo $left); stopping them"
ray stop > /dev/null 2>&1 || true
for p in $(ours); do kill -TERM "$p" 2>/dev/null; done
for i in $(seq 1 $(( ${GPU_FREE_WAIT_S:-120} / 5 ))); do [ -z "$(ours)" ] && break; sleep 5; done
left=$(ours)
if [ -n "$left" ]; then echo "[gpu-free] KILL $(echo $left)"; for p in $left; do kill -KILL "$p" 2>/dev/null; done; sleep 10; fi
left=$(ours); [ -z "$left" ] || { echo "[gpu-free] GPUs still held by $(echo $left)" >&2; exit 1; }
echo "[gpu-free] GPUs free"
