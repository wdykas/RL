#!/bin/bash
# Engine-sourced q under expert parallelism (DESIGN.md section 14, "Engine-sourced q under EP
# and PP"): tiny random Qwen3-MoE (4 layers, 8 experts, top-2), EP=2 in training and inference
# on 2 + 2 GPUs of one node (TP=1, so each side is DP=2 x EP=2), batch invariance (bi_flags.txt +
# inference_optimized), verify_source=engine, THUNDERSYNC_ENGINE_Q_CHECK=1
# THUNDERSYNC_MISMATCH_CHECK=1, 4 steps. Same pass criteria as engine_q_tp.sh; also expect
# "ep_dummy_forwards" in the [spec prof] lines (short ranks padding to the EP group's count)
# and no hang.
#
#   bash research/thundersync_rl/scripts/bi_tp/engine_q_moe.sh
#
# Env: as engine_q_tp.sh (EP 2, TP 1, PP 1, MODEL the tiny MoE, MAXLEN 1024).
W=/lustre/fs1/portfolios/llmservice/projects/llmservice_fm_text/users/wdykas/code/on-olicy-rl/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export EP=${EP:-2} TP=${TP:-1} PP=${PP:-1} MAXLEN=${MAXLEN:-1024}
export MODEL=${MODEL:-$W/models/tiny-qwen3-moe} RUNS=${RUNS:-runs_engine_q_moe}
exec bash "$HERE/engine_q_tp.sh"
