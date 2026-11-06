#!/bin/bash
# STOCK baseline: one stock NeMo-RL colocated arm on $COLO_RL_STOCK (SingleController GRPO loop, in_order sampler at lag
# K, engine colocated with the learner). K >= 1 = SPR (stock colocated partial rollout: in-flight requests pause at the
# stand-down, KV persisted, and resume after the wake); K = 0 = CS (stock colocated sync). Runs INSIDE the container.
#
#   K MODEL TP EP GPUS STEPS PROMPTS G MAXLEN PROMPT_CAP PARITY SEED   [GEN_TP GEN_EP KV=persist ENGINE_BUF MAX_REQ]
#   VARIANT=ltuned (default: learner-tuned stock = the ts runs' numerics-neutral learner settings) | tuned | shipped
#   ARGS='<extra hydra>'  TAG=<run>  PREFLIGHT=1 (dry run only)
# Stock numerics: the BI overlay stays on PYTHONPATH (TE 2.18 as every run) but no batch-invariant mode, no cuBLAS
# workspace pins, no TF32 override. Router frozen (freeze_moe_router=true, moe_router_bias_update_rate=0) is forced on
# the CLI and refused otherwise; the bias / router probe (bias_probe/sitecustomize.py) logs both every optimizer step.
# Output: $COLO_RUNS/$TAG/{run.log, preflight.log, effective_config.yaml, history.json}
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
L=$COLO_SCRIPTS/lib; HERE=$L/baselines_nrl
RECIPE=${RECIPE:-examples/configs/recipes/llm/grpo-nanov3-30BA3B-2n8g-megatron_generation-noncolocated-async-gym-single-controller-token_capture.yaml}
K=${K:-1} MODEL=${MODEL:?MODEL} TAG=${TAG:?TAG}
TP=${TP:-1} EP=${EP:-1} GPUS=${GPUS:?GPUS (the one colocated fleet)} GEN_EP=${GEN_EP:-$EP} GEN_TP=${GEN_TP:-$TP}
STEPS=${STEPS:-16} PROMPTS=${PROMPTS:-64} G=${G:-8} MAXLEN=${MAXLEN:-8192} PROMPT_CAP=${PROMPT_CAP:-2048}
DRY=0; [ "${PREFLIGHT:-0}" = 1 ] && DRY=1
(( DRY )) && [[ $TAG != *-dry ]] && TAG=$TAG-dry
export NRL_BIAS_PROBE=1 COLO_PREPEND=$HERE/bias_probe
COLO_STOCK=1 colo_pythonpath "$COLO_RL_STOCK" bi   # stock NeMo-RL on its own (old) pins
unset CUBLASLT_WORKSPACE_SIZE CUBLAS_WORKSPACE_CONFIG NVIDIA_TF32_OVERRIDE   # stock numerics
export PYTHONPATH=$HERE:$PYTHONPATH
cd "$RL_ROOT" || exit 2
[ -f "$RL_ROOT/$RECIPE" ] || { echo "no recipe $RL_ROOT/$RECIPE" >&2; exit 2; }
[ -f "$COLO_PINS_STOCK" ] || { echo "[stock] no pins file $COLO_PINS_STOCK: run $COLO_SCRIPTS/setup.sh first" >&2; exit 2; }
OUT=$COLO_RUNS/$TAG
mkdir -p "$OUT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR"
CFG=$OUT/nrl_config.yaml
printf 'defaults:\n  - %s\n  - %s\n' "$RL_ROOT/$RECIPE" "$HERE/configs/nrl_overrides.yaml" > "$CFG"
if (( GPUS <= 4 )); then
  RES=(cluster.num_nodes=1 cluster.gpus_per_node=$GPUS)
else
  (( GPUS % 4 == 0 )) || { echo "multi-node needs whole 4-GPU nodes" >&2; exit 2; }
  RES=(cluster.num_nodes=$(( GPUS / 4 )) cluster.gpus_per_node=4)
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
PAR=(policy.generation.colocated.enabled=true
     policy.megatron_cfg.tensor_model_parallel_size=$TP policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1
     policy.megatron_cfg.sequence_parallel=$SP
     policy.generation.mcore_generation_config.tensor_model_parallel_size=$GEN_TP
     policy.generation.mcore_generation_config.expert_model_parallel_size=$GEN_EP
     policy.generation.mcore_generation_config.pipeline_model_parallel_size=1
     policy.generation.mcore_generation_config.sequence_parallel=$SP
     policy.megatron_cfg.freeze_moe_router=true policy.megatron_cfg.moe_router_bias_update_rate=0.0)
# Engine KV / state buffer by the same rule as our arms (lib/engine_buf.py at MAX_REQ), unless given.
if [ -z "${ENGINE_BUF:-}" ] && [ -n "${MAX_REQ:-}" ]; then
  MCFG=$MODEL/config.json
  [ -f "$MCFG" ] || MCFG=$(ls "$HF_HOME"/hub/models--${MODEL//\//--}/snapshots/*/config.json 2>/dev/null | head -1)
  ENGINE_BUF=$(python3 $L/engine_buf.py "$MCFG" ${GEN_TP:-$TP} $MAX_REQ $(( PROMPT_CAP + MAXLEN )) 2>/dev/null | tail -1) || ENGINE_BUF=""
fi
[ -n "${ENGINE_BUF:-}" ] && PAR+=(policy.generation.mcore_generation_config.buffer_size_gb=$ENGINE_BUF)
PAR+=(policy.generation.mcore_generation_config.kv_cache_management_mode=${KV:-persist})
[ -n "${MAX_REQ:-}" ] && PAR+=(++policy.generation.mcore_generation_config.max_requests=$MAX_REQ)
[ "${PARITY:-0}" = 1 ] && { export NRL_PARITY_REWARD=1; PAR+=(data.default.env_name=math_parity); }
[ -n "${SEED:-}" ] && PAR+=(grpo.seed=$SEED)
case ${VARIANT:-ltuned} in
  shipped) ;;
  tuned) PAR+=(++loss_fn.force_on_policy_ratio=true grpo.seq_logprob_error_threshold=null) ;;
  # ltuned = the strict bar: tuned + the learner settings of our runs wherever they are numerics-neutral Megatron /
  # NeMo-RL knobs (selective MoE recompute, no packing with LMBS rows, seq multiple LDIV, plain distributed Adam, fp32
  # grad reduce, no param-gather overlap, no empty_cache, no per-layer MoE logging, same optimizer schedule).
  ltuned) PAR+=(++loss_fn.force_on_policy_ratio=true grpo.seq_logprob_error_threshold=null
                ++policy.megatron_cfg.recompute_granularity=selective "++policy.megatron_cfg.recompute_modules=[moe]"
                policy.sequence_packing.enabled=false policy.train_micro_batch_size=${LMBS:-2}
                policy.make_sequence_length_divisible_by=${LDIV:-128}
                policy.megatron_cfg.optimizer.use_precision_aware_optimizer=false
                policy.megatron_cfg.distributed_data_parallel_config.grad_reduce_in_fp32=true
                policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=false
                policy.megatron_cfg.empty_unused_memory_level=0
                policy.megatron_cfg.moe_per_layer_logging=false policy.megatron_cfg.track_moe_metrics=false
                policy.megatron_cfg.optimizer.lr=1.0e-6 policy.megatron_cfg.optimizer.min_lr=1.0e-6
                policy.megatron_cfg.optimizer.weight_decay=0.01
                policy.megatron_cfg.scheduler.lr_warmup_iters=13 policy.megatron_cfg.scheduler.lr_warmup_init=5.0e-7) ;;
  *) echo "VARIANT must be shipped | tuned | ltuned" >&2; exit 2 ;;
esac
# Lag: in_order admits batch s once trainer_version >= s - K; the replay buffer holds (K + 1) batches of groups.
LAG=(async_rl.sampler.name=in_order async_rl.sampler.max_lookahead_versions=$K
     async_rl.max_buffered_rollouts=$(( PROMPTS * (K + 1) )))
read -r -a USER_ARGS <<< "${ARGS:-}"
rm -f "$OUT/history.json" "$OUT/negative"
echo colo_${VARIANT:-ltuned}_$K > "$OUT/arm"; echo $GPUS > "$OUT/gpus"; echo $K > "$OUT/k"
export NRL_EFFECTIVE_CONFIG=$OUT/effective_config.yaml
echo "[stock] K=$K TP=$TP EP=$EP GPUS=$GPUS STEPS=$STEPS PROMPTS=$PROMPTS G=$G MAXLEN=$MAXLEN PARITY=${PARITY:-0}" \
  "ENGINE_BUF=${ENGINE_BUF:-recipe} MAX_REQ=${MAX_REQ:-recipe} MODEL=$MODEL RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD) -> $OUT"
CLI=(--config "$CFG"
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=$G
  policy.train_global_batch_size=$(( PROMPTS * G )) policy.max_total_sequence_length=$(( PROMPT_CAP + MAXLEN ))
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN
  "${RES[@]}" "${PAR[@]}" "${LAG[@]}" "${USER_ARGS[@]}")
NRL_DRY=1 timeout 600 python -u $HERE/run_nrl_sc.py "${CLI[@]}" > "$OUT/preflight.log" 2>&1
grep -q "DRY OK" "$OUT/preflight.log" || { echo "[stock] PREFLIGHT FAIL (see $OUT/preflight.log)"; tail -5 "$OUT/preflight.log"; exit 3; }
grep -q "router: frozen (weights+bias)" "$OUT/preflight.log" || { echo "[stock] REFUSED: router not frozen"; grep REFUSED "$OUT/preflight.log"; exit 4; }
echo "[stock] preflight: $(grep -o 'router: frozen (weights+bias)' "$OUT/preflight.log" | head -1); $(grep 'DRY OK' "$OUT/preflight.log" | head -1)"
(( DRY )) && { echo "[stock] preflight only: OK"; exit 0; }
# 1 node: the previous arm of this job may still hold the GPUs (lib/gpu_free.sh).
(( GPUS <= 4 )) && { bash $L/gpu_free.sh || { echo "[stock] GPUs not free" >&2; exit 4; }; }
WD=${WD:-auto}
if [ "$WD" = auto ]; then
  if [ -z "${SLURM_JOB_ID:-}" ]; then WD=0; elif (( GPUS > 4 )); then WD=enforce; else WD=local; fi
fi
case $WD in
  local) bash $L/gpu_idle_watchdog.sh local-bg "$OUT/run.log" ;;
  enforce) bash $L/gpu_idle_watchdog.sh enforce-bg "$OUT/run.log" ;;
  0) ;;
esac
timeout ${RUN_TIMEOUT:-10800} python -u $HERE/run_nrl_sc.py "${CLI[@]}" 2>&1 | python3 -u $HERE/stamp.py > "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "[stock] run exit=$rc"
python3 $HERE/parse_nrl.py "$OUT" || true
python3 $L/check_run.py "$OUT" || { [ $rc = 0 ] && rc=5; }
exit $rc
