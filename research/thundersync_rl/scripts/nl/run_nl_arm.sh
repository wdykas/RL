#!/bin/bash
# One nolanes (NL2) arm: NeMo-RL's stock SingleController async GRPO loop (in_order sampler, lag LAG, one non-colocated
# engine fleet, persist refit) with thundersync.mode=nolanes, BI (V8b) numerics on learner and engine
# (configs/nl/nl_base.yaml). Runs on one 4-GPU node (local Ray: RAY_ADDRESS=local CUDA_VISIBLE_DEVICES=0,1,2,3) or on the
# head node of a Ray cluster ($TS_WORK/research/ultra/broker_ray.sh; MKCMD=1 writes its one-line command file).
#
#   [TREE=<tree>] [MODEL=<path>] [TP=2 EP=2 GEN_TP=TP GEN_EP=EP GPS=lcm(TP,EP) GEN_GPUS=lcm(GEN_TP,GEN_EP)] \
#   [PROMPTS=16 G=8 MAXLEN=1024 PROMPT_CAP=2048 STEPS=5 LAG=1 KD=<config's k_drafts>] [PARITY=1] [CHECKS=1] [NEG=<control>] \
#   [MAX_REQ=<per engine DP rank> ENGINE_BUF=<GiB> MBS=<rows> SEED=<n>] [XA='<extra hydra args>'] [XE='K=V ...'] \
#   [TAG=<run>] [RUNS=runs_nl] [WD=auto|local|enforce|0] [PREFLIGHT=1 | NL_DRY=1] [MKCMD=1] bash run_nl_arm.sh
#
# CHECKS=1: THUNDERSYNC_MISMATCH_CHECK, THUNDERSYNC_ENGINE_Q_CHECK, THUNDERSYNC_LAW_CHECK, THUNDERSYNC_NL_CHECKS
#   (exported for the driver / 1-node Ray workers and passed as policy.megatron_cfg.env_vars for multi-node workers).
# NEG=<name>: THUNDERSYNC_NL_NEG negative control (thundersync_rl/nolanes.py NEGATIVES; each must FAIL a check).
# PARITY=1 (default; truncated models): env math_parity (thundersync_rl/nl_parity_env.py), NL_PARITY_REWARD=1.
# Engine EP != learner EP turns on the EP-invariant MoE combine (as run_ab_arm.sh); TP must match (BI validator).
# MAX_REQ default: (PROMPTS x N x (LAG + 1) + PROMPTS x G) / engine DP, rounded up to 64, N = max(G, round(KD x G))
#   drafts per prompt: every in-flight batch's drafts plus one batch of continuations.
# PREFLIGHT=1 (or NL_DRY=1): only the config dry run (compose + MasterConfig + capacity + BI / fp16 / nolanes
#   validators), no Ray, no GPUs; otherwise the same dry run gates the real run (in-job preflight).
# WD (GPU-idle watchdog, $TS_WORK/sbatch/gpu_idle_watchdog.sh): local = local-bg (monitor + enforcer, 1-node
#   job not already covered by envguard.sh), enforce = enforce-bg (head of a broker_ray.sh cluster, whose nodes run the
#   monitors), 0 = none (x1_gate.sh: envguard.sh covers the gate). auto: no slurm job -> 0, > 4 GPUs -> enforce, else local.
# Output: $TS_WORK/$RUNS/$TAG/{run.log (lines prefixed [t=<epoch>]), effective_config.yaml, preflight.log,
#   history.json + staleness.json (baselines_nrl/parse_nrl.py, the nrl_k arm's parser), arm, gpus, lag}.
set -uo pipefail
W=${TS_WORK:?set TS_WORK to the research workspace (models/, overlays, env_sbatch.sh, run dirs)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../../.." && pwd)}
CFG_REL=${CFG_REL:-research/thundersync_rl/configs/nl/nl_base.yaml}  # nl_super.yaml / nl_ultra.yaml: per-model routing
MODEL=${MODEL:-$W/models/nemotron3-super-L8}
TP=${TP:-2} EP=${EP:-2} GEN_TP=${GEN_TP:-$TP} GEN_EP=${GEN_EP:-$EP}
PROMPTS=${PROMPTS:-16} G=${G:-8} MAXLEN=${MAXLEN:-1024} PROMPT_CAP=${PROMPT_CAP:-2048} STEPS=${STEPS:-5} LAG=${LAG:-1}
KD=${KD:-} PARITY=${PARITY:-1} CHECKS=${CHECKS:-1} RUNS=${RUNS:-runs_nl}
DRY=0; { [ "${PREFLIGHT:-0}" = 1 ] || [ "${NL_DRY:-0}" = 1 ]; } && DRY=1
gcd() { local a=$1 b=$2; while (( b )); do set -- $b $(( a % b )); a=$1; b=$2; done; echo $a; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
GPS=${GPS:-$(lcm $TP $EP)} GEN_GPUS=${GEN_GPUS:-$(lcm $GEN_TP $GEN_EP)}
TOT=$(( GPS + GEN_GPUS ))
TAG=${TAG:-nl-$(basename "$MODEL")-tp${TP}ep${EP}-g${GEN_TP}x${GEN_EP}-p${PROMPTS}g${G}-L${MAXLEN}-lag${LAG}${NEG:+-neg_$NEG}}
(( DRY )) && [[ $TAG != *-dry ]] && TAG=$TAG-dry
OUT=$W/$RUNS/$TAG

# MKCMD=1: write the one-line driver command for broker_ray.sh (multi-node; same form as baselines_nrl/make_cmd.sh)
# and exit. Submit: slurm_submit nodes=N ntasks_per_node=1 segment=N
#   command=["bash","-c","OBJ_STORE_GB=auto bash <W>/research/ultra/broker_ray.sh <cmd file> <TAG>"]
if [ "${MKCMD:-0}" = 1 ]; then
  V=""
  for n in TREE CFG_REL MODEL TP EP GEN_TP GEN_EP GPS GEN_GPUS PROMPTS G MAXLEN PROMPT_CAP STEPS LAG KD PARITY CHECKS NEG MAX_REQ \
           ENGINE_BUF MBS SEED RUNS WD RUN_TIMEOUT PREFLIGHT NL_DRY; do
    [ -n "${!n:-}" ] && V+="$n=${!n} "
  done
  [ -n "${XA:-}" ] && V+="XA='${XA}' "
  [ -n "${XE:-}" ] && V+="XE='${XE}' "
  mkdir -p $W/research/nolanes2/cmds; F=$W/research/nolanes2/cmds/$TAG.sh
  echo "cd $TREE ; bash $W/sbatch/keep_prev_run.sh $W/$RUNS $TAG && ${V}TAG=$TAG bash $TREE/research/thundersync_rl/scripts/nl/run_nl_arm.sh > $W/runs_ab/$TAG.out 2>&1" > $F
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
[ -f "$RL_ROOT/$CFG_REL" ] || { echo "[nl] no config $RL_ROOT/$CFG_REL" >&2; exit 2; }
mkdir -p "$OUT"

if (( TOT <= 4 )); then
  RES=(cluster.num_nodes=1 cluster.gpus_per_node=$TOT
       policy.generation.colocated.resources.num_nodes=1 policy.generation.colocated.resources.gpus_per_node=$GEN_GPUS)
else
  (( GPS % 4 == 0 && GEN_GPUS % 4 == 0 )) || { echo "[nl] multi-node needs whole 4-GPU nodes per side (GPS=$GPS GEN_GPUS=$GEN_GPUS)" >&2; exit 2; }
  RES=(cluster.num_nodes=$(( TOT / 4 )) cluster.gpus_per_node=4
       policy.generation.colocated.resources.num_nodes=$(( GEN_GPUS / 4 )) policy.generation.colocated.resources.gpus_per_node=4)
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
GEN_SP=$([ "$GEN_TP" -gt 1 ] && echo true || echo false)
# BI token multiple (zero_train_gen_mismatch.validate_batch_invariant_mode): ceil(64 / TP) x TP, lcm'd with 32 x TP at
# TP > 1 (SP RMSNorm rows); then lcm with 128 (the ts arms' value).
BTM=$(( (64 + TP - 1) / TP * TP )); (( TP > 1 )) && BTM=$(lcm $BTM $(( 32 * TP ))); SEQDIV=$(lcm $BTM 128)
PAR=(policy.generation.colocated.enabled=false
     policy.megatron_cfg.tensor_model_parallel_size=$TP policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.pipeline_model_parallel_size=1 policy.megatron_cfg.context_parallel_size=1
     policy.megatron_cfg.sequence_parallel=$SP
     policy.generation.mcore_generation_config.tensor_model_parallel_size=$GEN_TP
     policy.generation.mcore_generation_config.expert_model_parallel_size=$GEN_EP
     policy.generation.mcore_generation_config.pipeline_model_parallel_size=1
     policy.generation.mcore_generation_config.sequence_parallel=$GEN_SP
     policy.make_sequence_length_divisible_by=$SEQDIV)
# The EP-invariant MoE combine is derived by nolanes.configure_driver (on <=> engine EP != learner EP).
# KD unset: the config's thundersync.k_drafts (CFG_REL and its defaults chain; later wins), else 1.0.
if [ -z "$KD" ]; then
  KD=$(cd "$RL_ROOT" && python3 - "$CFG_REL" <<'PYEOF'
import os, re, sys
def k_of(path, seen=()):
    if not os.path.isfile(path) or path in seen:
        return None
    txt = open(path).read()
    k = None
    m = re.search(r"^defaults:\s*\n((?:\s+-\s*.+\n)+)", txt, re.M)
    for d in (re.findall(r"-\s*(\S+)", m.group(1)) if m else []):
        v = k_of(os.path.normpath(os.path.join(os.path.dirname(path), d)), seen + (path,))
        k = v if v is not None else k
    m = re.search(r"^thundersync:\s*\n(?:[ \t]+.*\n|\s*\n)*?[ \t]+k_drafts:\s*([0-9.]+)", txt, re.M)
    return float(m.group(1)) if m else k
k = k_of(sys.argv[1])
print(k if k is not None else 1.0)
PYEOF
)
  KD_FROM=config
fi
# Engine capacity: every in-flight batch's drafts + one batch of continuations per engine DP rank.
N=$(python3 -c "import sys; k, g = float(sys.argv[1]), int(sys.argv[2]); print(max(g, int(round(k * g))))" "$KD" "$G")
GEN_DP=$(( GEN_GPUS / GEN_TP )); (( GEN_DP < 1 )) && GEN_DP=1
_NEED=$(( (PROMPTS * N * (LAG + 1) + PROMPTS * G + GEN_DP - 1) / GEN_DP ))
MAX_REQ=${MAX_REQ:-$(( (_NEED + 63) / 64 * 64 ))}
PAR+=(++policy.generation.mcore_generation_config.max_requests=$MAX_REQ)
[ -n "${ENGINE_BUF:-}" ] && PAR+=(policy.generation.mcore_generation_config.buffer_size_gb=$ENGINE_BUF)
[ -n "${MBS:-}" ] && PAR+=(policy.train_micro_batch_size=$MBS)
[ -n "${SEED:-}" ] && PAR+=(grpo.seed=$SEED)
[ "${KD_FROM:-}" != config ] && PAR+=(thundersync.k_drafts=$KD)
if [ "$PARITY" = 1 ]; then export NL_PARITY_REWARD=1; PAR+=(data.default.env_name=math_parity); fi
# Exactness checks / negative control: driver env (1-node Ray workers inherit it) + learner / engine worker env_vars.
unset THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_ENGINE_Q_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS THUNDERSYNC_NL_NEG
if [ "$CHECKS" = 1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_ENGINE_Q_CHECK=1 THUNDERSYNC_LAW_CHECK=1 THUNDERSYNC_NL_CHECKS=1
  for v in THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_ENGINE_Q_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS; do
    PAR+=("++policy.megatron_cfg.env_vars.$v='1'")
  done
fi
if [ -n "${NEG:-}" ]; then
  export THUNDERSYNC_NL_NEG=$NEG
  PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_NL_NEG='$NEG'")
fi
# XE: extra process env ("K=V K2=V2"), exported for the driver (and 1-node Ray workers).
if [ -n "${XE:-}" ]; then
  read -r -a _XE <<< "$XE"
  for kv in "${_XE[@]}"; do [[ $kv == *=* ]] || { echo "[nl] XE entry '$kv' is not K=V" >&2; exit 2; }; export "$kv"; done
fi
LAGA=(async_rl.sampler.name=in_order async_rl.sampler.max_lookahead_versions=$LAG)
read -r -a USER_ARGS <<< "${XA:-}"
echo nl > "$OUT/arm"; echo $TOT > "$OUT/gpus"; echo $LAG > "$OUT/lag"
export NL_EFFECTIVE_CONFIG=$OUT/effective_config.yaml
echo "[nl] TAG=$TAG TP=$TP EP=$EP GPS=$GPS GEN_TP=$GEN_TP GEN_EP=$GEN_EP GEN_GPUS=$GEN_GPUS LAG=$LAG STEPS=$STEPS PROMPTS=$PROMPTS" \
  "G=$G KD=$KD N=$N MAXLEN=$MAXLEN PROMPT_CAP=$PROMPT_CAP MAX_REQ=$MAX_REQ SEQDIV=$SEQDIV PARITY=$PARITY CHECKS=$CHECKS" \
  "NEG=${NEG:-} MODEL=$MODEL RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD 2>/dev/null)" \
  "dirty=$(git -C "$RL_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) -> $OUT"
CLI=(--config "$RL_ROOT/$CFG_REL"
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=$G
  policy.train_global_batch_size=$(( PROMPTS * G )) policy.max_total_sequence_length=$(( PROMPT_CAP + MAXLEN ))
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN
  "${RES[@]}" "${PAR[@]}" "${LAGA[@]}" "${USER_ARGS[@]}")
ENTRY=$RL_ROOT/research/thundersync_rl/run_nolanes_sc.py
PY=${PY:-python}

# Config dry run (no Ray, no GPUs): the whole run in PREFLIGHT / NL_DRY mode, the gate of the real run otherwise.
NL_DRY=1 timeout ${PREFLIGHT_TIMEOUT:-600} $PY -u $ENTRY "${CLI[@]}" > "$OUT/preflight.log" 2>&1
prc=$?
grep -a "^\[nl\] DRY\|^\[nolanes\]" "$OUT/preflight.log" | cut -c1-600
if ! grep -q "\[nl\] DRY OK" "$OUT/preflight.log"; then
  echo "[nl] PREFLIGHT FAIL rc=$prc (see $OUT/preflight.log)"; tail -15 "$OUT/preflight.log"; exit 3
fi
# User rule 11-02: every run has the MoE router weights and expert bias frozen (from the effective config).
if ! grep -q "\[nl\] DRY router: frozen (weights+bias)" "$OUT/preflight.log"; then
  echo "[nl] PREFLIGHT FAIL: router not frozen (weights+bias) in the effective config"; exit 3
fi
(( DRY )) && { echo "[nl] preflight only: OK"; exit 0; }

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
  *) echo "[nl] WD must be auto | local | enforce | 0" >&2; exit 2 ;;
esac
rm -f "$OUT/history.json" "$OUT/staleness.json"
timeout ${RUN_TIMEOUT:-10800} $PY -u $ENTRY "${CLI[@]}" 2>&1 | python3 -u $W/research/baselines_nrl/stamp.py > "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "[nl] run exit=$rc"
[ -f $W/research/baselines_nrl/parse_nrl.py ] && { python3 $W/research/baselines_nrl/parse_nrl.py "$OUT" || true; }
exit $rc
