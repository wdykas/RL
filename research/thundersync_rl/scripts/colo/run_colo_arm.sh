#!/bin/bash
# One colo arm (research/colo/MODEL.md): NeMo-RL's stock SingleController loop with a COLOCATED Megatron engine (one
# fleet of GPUS GPUs generates and trains; the engine is a dedicated inference model resharded from the learner at every
# wake), thundersync.mode=colo, BI (V8b) numerics on learner and engine (configs/colo/colo.yaml), lag 0.
# Runs on one 4-GPU node (local Ray) or on the head node of a Ray cluster ($TS_WORK/research/ultra/broker_ray.sh;
# MKCMD=1 writes its one-line command file).
#
#   [TREE=<tree>] [MODEL=<path>] [TP=2 EP=8 GPUS=16] [PROMPTS=64 G=8 MAXLEN=8192 PROMPT_CAP=2048 STEPS=5] [PARITY=1]
#   [CHECKS=1] [NEG=<control>] [SKIP_RESHARD=1] [MAX_REQ=<per engine DP rank> ENGINE_BUF=<GiB> MBS=<rows> SEED=<n>]
#   [TMS=1 (torch_memory_saver overlay: engine scratch released during training) KV=persist|offload]
#   [XA='<extra hydra args>'] [XE='K=V ...'] [TAG=<run>] [RUNS=runs_colo] [WD=auto|local|enforce|0] [PREFLIGHT=1]
#   [MKCMD=1] bash run_colo_arm.sh
#
# CHECKS=1: THUNDERSYNC_MISMATCH_CHECK, THUNDERSYNC_LAW_CHECK, THUNDERSYNC_NL_CHECKS (driver env + learner env_vars):
#   ColoStage runs the mismatch (bitwise) and law tests of every output token each step.
# SKIP_RESHARD=1 (negative control, needs CHECKS=1): NEMO_RL_DEBUG_COLOC_SKIP_RESHARD=1, the engine keeps theta_0 after
#   the first wake; the sentinel and the law / mismatch checks must FAIL from step 1 on (the run continues).
# MAX_REQ default: ceil(PROMPTS x G / engine DP) rounded up to 16 (one batch in flight at lag 0).
# Output: $TS_WORK/$RUNS/$TAG/{run.log, effective_config.yaml, preflight.log, history.json, arm, gpus}.
set -uo pipefail
W=${TS_WORK:?set TS_WORK to the research workspace (models/, overlays, env_sbatch.sh, run dirs)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../../.." && pwd)}
CFG_REL=${CFG_REL:-research/thundersync_rl/configs/colo/colo.yaml}
MODEL=${MODEL:-$W/models/nemotron3-super-L8}
TP=${TP:-2} EP=${EP:-2}
PROMPTS=${PROMPTS:-16} G=${G:-8} MAXLEN=${MAXLEN:-1024} PROMPT_CAP=${PROMPT_CAP:-2048} STEPS=${STEPS:-5}
PARITY=${PARITY:-1} CHECKS=${CHECKS:-1} RUNS=${RUNS:-runs_colo} SKIP_RESHARD=${SKIP_RESHARD:-0}
DRY=0; { [ "${PREFLIGHT:-0}" = 1 ] || [ "${NL_DRY:-0}" = 1 ]; } && DRY=1
gcd() { local a=$1 b=$2; while (( b )); do set -- $b $(( a % b )); a=$1; b=$2; done; echo $a; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
GPUS=${GPUS:-$(lcm $TP $EP)}
TOT=$GPUS
TAG=${TAG:-colo-$(basename "$MODEL")-tp${TP}ep${EP}-n${GPUS}-p${PROMPTS}g${G}-L${MAXLEN}${NEG:+-neg_$NEG}}
(( DRY )) && [[ $TAG != *-dry ]] && TAG=$TAG-dry
OUT=$W/$RUNS/$TAG

# MKCMD=1: write the one-line driver command for broker_ray.sh (multi-node; same form as baselines_nrl/make_cmd.sh)
# and exit. Submit: slurm_submit nodes=N ntasks_per_node=1 segment=N
#   command=["bash","-c","OBJ_STORE_GB=auto bash <W>/research/ultra/broker_ray.sh <cmd file> <TAG>"]
if [ "${MKCMD:-0}" = 1 ]; then
  V=""
  for n in TREE CFG_REL MODEL TP EP GPUS PROMPTS G MAXLEN PROMPT_CAP STEPS PARITY CHECKS NEG SKIP_RESHARD MAX_REQ SPARE \
           ENGINE_BUF KV TMS MBS SEED RUNS WD RUN_TIMEOUT PREFLIGHT NL_DRY; do
    [ -n "${!n:-}" ] && V+="$n=${!n} "
  done
  [ -n "${XA:-}" ] && V+="XA='${XA}' "
  [ -n "${XE:-}" ] && V+="XE='${XE}' "
  mkdir -p $W/research/colo/cmds; F=$W/research/colo/cmds/$TAG.sh
  echo "cd $TREE ; bash $W/sbatch/keep_prev_run.sh $W/$RUNS $TAG && ${V}TAG=$TAG bash $TREE/research/thundersync_rl/scripts/colo/run_colo_arm.sh > $W/runs_ab/$TAG.out 2>&1" > $F
  echo $F; exit 0
fi

export RL_ROOT_OVERRIDE=$TREE
# overlay_up first (nemo.lens span_attributes for the telemetry imports), as every ts / nrl run; env_sbatch.sh bi puts
# the BI overlay (TE 2.18, flash-attn-4, CuTe DSL) on PYTHONPATH and exports the te_native cuBLAS env.
export TS_PYTHONPATH_PREPEND=${TS_PYTHONPATH_PREPEND:-$W/overlay_up}
source $W/env_sbatch.sh bi
export NVIDIA_TF32_OVERRIDE=1 RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1
# Nothing under /home (quota): JIT / compile caches on the shared filesystem unless the caller set them (as envguard.sh).
_cache=$W/cache
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$_cache/triton} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$_cache/inductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-$_cache/cuda} MPLCONFIGDIR=${MPLCONFIGDIR:-$_cache/mpl}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$_cache/torch_extensions}
cd "$RL_ROOT" || exit 2
[ -f "$RL_ROOT/$CFG_REL" ] || { echo "[colo] no config $RL_ROOT/$CFG_REL" >&2; exit 2; }
mkdir -p "$OUT"

if (( TOT <= 4 )); then
  RES=(cluster.num_nodes=1 cluster.gpus_per_node=$TOT)
else
  (( TOT % 4 == 0 )) || { echo "[colo] multi-node needs whole 4-GPU nodes (GPUS=$GPUS)" >&2; exit 2; }
  RES=(cluster.num_nodes=$(( TOT / 4 )) cluster.gpus_per_node=4)
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
# BI token multiple (zero_train_gen_mismatch.validate_batch_invariant_mode): ceil(64 / TP) x TP, lcm'd with 32 x TP at
# TP > 1 (SP RMSNorm rows); then lcm with 128 (the ts arms' value).
BTM=$(( (64 + TP - 1) / TP * TP )); (( TP > 1 )) && BTM=$(lcm $BTM $(( 32 * TP ))); SEQDIV=$(lcm $BTM 128)
PAR=(policy.generation.colocated.enabled=true
     policy.megatron_cfg.tensor_model_parallel_size=$TP policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1
     policy.megatron_cfg.sequence_parallel=$SP
     policy.generation.mcore_generation_config.tensor_model_parallel_size=$TP
     policy.generation.mcore_generation_config.expert_model_parallel_size=$EP
     policy.generation.mcore_generation_config.pipeline_model_parallel_size=1
     policy.generation.mcore_generation_config.sequence_parallel=$SP
     policy.make_sequence_length_divisible_by=$SEQDIV)
# Engine capacity: one batch in flight (lag 0) over the engine DP ranks.
GEN_DP=$(( GPUS / TP )); (( GEN_DP < 1 )) && GEN_DP=1
MAX_REQ=${MAX_REQ:-$(( ( (PROMPTS * G + GEN_DP - 1) / GEN_DP + 15 ) / 16 * 16 ))}
PAR+=(++policy.generation.mcore_generation_config.max_requests=$MAX_REQ)
[ -n "${ENGINE_BUF:-}" ] && PAR+=(policy.generation.mcore_generation_config.buffer_size_gb=$ENGINE_BUF)
# KV=offload (with TMS=1, the torch_memory_saver overlay): the engine's KV / state buffer is released to a CPU backup
# while the learner trains and restored at its addresses (default persist: resident through training).
[ -n "${KV:-}" ] && PAR+=(policy.generation.mcore_generation_config.kv_cache_management_mode=$KV)
[ -n "${MBS:-}" ] && PAR+=(policy.train_micro_batch_size=$MBS)
[ -n "${SEED:-}" ] && PAR+=(grpo.seed=$SEED)
# SPARE=S (XPR-S, configs/colo/colo_xprs.yaml): admit PROMPTS + S prompt groups per step, train PROMPTS of them.
NPS=$PROMPTS
if [ -n "${SPARE:-}" ] && [ "${SPARE}" != 0 ]; then
  NPS=$(( PROMPTS + SPARE )); PAR+=(thundersync.spare_prompts=$SPARE)
fi
if [ "$PARITY" = 1 ]; then export NL_PARITY_REWARD=1; PAR+=(data.default.env_name=math_parity); fi
# Exactness checks / negative control: driver env (1-node Ray workers inherit it) + learner / engine worker env_vars.
unset THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_ENGINE_Q_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS THUNDERSYNC_NL_NEG \
  NEMO_RL_DEBUG_COLOC_SKIP_RESHARD
if [ "$CHECKS" = 1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_LAW_CHECK=1 THUNDERSYNC_NL_CHECKS=1
  for v in THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS; do
    PAR+=("++policy.megatron_cfg.env_vars.$v='1'")
  done
fi
if [ -n "${NEG:-}" ]; then
  export THUNDERSYNC_NL_NEG=$NEG
  PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_NL_NEG='$NEG'")
fi
if [ "$SKIP_RESHARD" = 1 ]; then
  [ "$CHECKS" = 1 ] || { echo "[colo] SKIP_RESHARD=1 is a negative control: needs CHECKS=1" >&2; exit 2; }
  export NEMO_RL_DEBUG_COLOC_SKIP_RESHARD=1
fi
# XE: extra process env ("K=V K2=V2"), exported for the driver (and 1-node Ray workers).
if [ -n "${XE:-}" ]; then
  read -r -a _XE <<< "$XE"
  for kv in "${_XE[@]}"; do [[ $kv == *=* ]] || { echo "[colo] XE entry '$kv' is not K=V" >&2; exit 2; }; export "$kv"; done
fi
read -r -a USER_ARGS <<< "${XA:-}"
echo colo > "$OUT/arm"; echo $TOT > "$OUT/gpus"; echo 0 > "$OUT/lag"
export NL_EFFECTIVE_CONFIG=$OUT/effective_config.yaml
echo "[colo] TAG=$TAG TP=$TP EP=$EP GPUS=$GPUS STEPS=$STEPS PROMPTS=$PROMPTS" \
  "G=$G MAXLEN=$MAXLEN PROMPT_CAP=$PROMPT_CAP MAX_REQ=$MAX_REQ SEQDIV=$SEQDIV PARITY=$PARITY CHECKS=$CHECKS" \
  "NEG=${NEG:-} SKIP_RESHARD=$SKIP_RESHARD MODEL=$MODEL RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD 2>/dev/null)" \
  "dirty=$(git -C "$RL_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) -> $OUT"
CLI=(--config "$RL_ROOT/$CFG_REL"
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS
  grpo.num_prompts_per_step=$NPS grpo.num_generations_per_prompt=$G
  policy.train_global_batch_size=$(( PROMPTS * G )) policy.max_total_sequence_length=$(( PROMPT_CAP + MAXLEN ))
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN
  "${RES[@]}" "${PAR[@]}" "${USER_ARGS[@]}")
ENTRY=$RL_ROOT/research/thundersync_rl/run_nolanes_sc.py
PY=${PY:-python}

# Config dry run (no Ray, no GPUs): the whole run in PREFLIGHT / NL_DRY mode, the gate of the real run otherwise.
NL_DRY=1 timeout ${PREFLIGHT_TIMEOUT:-600} $PY -u $ENTRY "${CLI[@]}" > "$OUT/preflight.log" 2>&1
prc=$?
grep -a "^\[nl\] DRY\|^\[nolanes\]" "$OUT/preflight.log" | cut -c1-600
if ! grep -q "\[nl\] DRY OK" "$OUT/preflight.log"; then
  echo "[colo] PREFLIGHT FAIL rc=$prc (see $OUT/preflight.log)"; tail -15 "$OUT/preflight.log"; exit 3
fi
# User rule 11-02: every run has the MoE router weights and expert bias frozen (from the effective config).
if ! grep -q "\[nl\] DRY router: frozen (weights+bias)" "$OUT/preflight.log"; then
  echo "[colo] PREFLIGHT FAIL: router not frozen (weights+bias) in the effective config"; exit 3
fi
(( DRY )) && { echo "[colo] preflight only: OK"; exit 0; }

# Multi-node under broker_ray.sh: its stall watchdog watches runs_ab/<run>/run.log; point it at this run.
if [ "$RUNS" != runs_ab ] && (( TOT > 4 )) && [ ! -e "$W/runs_ab/$TAG" ]; then ln -s "$OUT" "$W/runs_ab/$TAG"; fi
WD=${WD:-auto}
if [ "$WD" = auto ]; then
  if [ -z "${SLURM_JOB_ID:-}" ]; then WD=0; elif (( TOT > 4 )); then WD=enforce; else WD=local; fi
fi
case $WD in
  local) bash $W/sbatch/gpu_idle_watchdog.sh local-bg "$OUT/run.log" ;;
  enforce) bash $W/sbatch/gpu_idle_watchdog.sh enforce-bg "$OUT/run.log" ;;
  0) ;;
  *) echo "[colo] WD must be auto | local | enforce | 0" >&2; exit 2 ;;
esac
rm -f "$OUT/history.json" "$OUT/staleness.json"
timeout ${RUN_TIMEOUT:-10800} $PY -u $ENTRY "${CLI[@]}" 2>&1 | python3 -u $W/research/baselines_nrl/stamp.py > "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "[colo] run exit=$rc"
[ -f $W/research/baselines_nrl/parse_nrl.py ] && { python3 $W/research/baselines_nrl/parse_nrl.py "$OUT" || true; }
exit $rc
