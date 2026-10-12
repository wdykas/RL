#!/bin/bash
# Engine-sourced q under pipeline parallelism (DESIGN.md section 14, "Engine-sourced q under EP
# and PP"): Qwen3-1.7B, PP=2 in training and inference on 2 + 2 GPUs of one node (TP=1),
# batch invariance (bi_flags.txt + inference_optimized), verify_source=engine,
# THUNDERSYNC_ENGINE_Q_CHECK=1 THUNDERSYNC_MISMATCH_CHECK=1, 4 steps. The learner runs each
# verification forward through the forward-only pipeline schedule and its last stage verifies;
# the engine logs hidden states on its last stage. Same pass criteria as engine_q_tp.sh.
#
# Generation with PP=2 under batch invariance has not been run yet: if the engine side fails
# (build, refit or a nonzero mismatch with speculation off: ARGS=++thundersync.speculate=false),
# that is a BI / inference PP issue, not engine-q's.
#
#   bash research/thundersync_rl/scripts/bi_tp/engine_q_pp.sh
#
# Env: as engine_q_tp.sh (PP 2, TP 1, EP 1, MODEL Qwen/Qwen3-1.7B).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PP=${PP:-2} TP=${TP:-1} EP=${EP:-1}
export MODEL=${MODEL:-Qwen/Qwen3-1.7B} RUNS=${RUNS:-runs_engine_q_pp}
exec bash "$HERE/engine_q_tp.sh"
