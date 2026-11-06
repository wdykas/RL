#!/bin/bash
# OURS: one exact colocated arm (thundersync.mode=colo) on the RL tree $COLO_RL. Runs INSIDE the container, on one 4-GPU
# node (local Ray) or on the head node of a Ray cluster started by lib/ray_cluster.sh. Outputs under $COLO_OUT.
#
#   CFG_REL   research/thundersync_rl/configs/colo/colo_wv.yaml   production: XPR-S + verify at the wake (fixed G)
#             research/thundersync_rl/configs/colo/colo_xpr.yaml  XPR: exact colocated partial rollout (XPR,
#                                                                   speculative: carried drafts verified exactly), lag K
#             research/thundersync_rl/configs/colo/colo_xprs.yaml XPR-S: XPR + spare prompts (⊥-BV pool at constant
#                                                                   envelope, prompt drop); needs SPARE=S
#             research/thundersync_rl/configs/colo/colo_vg.yaml   VG: variable group size (alternative), lag K
#             research/thundersync_rl/configs/colo/colo.yaml      XS: exact colocated sync (lag 0, nothing to verify)
#             research/thundersync_rl/configs/colo/colo_isspr.yaml  IS-SPR: stock pause / resume + keyed rescore, IS weights
#             research/thundersync_rl/configs/colo/colo_sprbi.yaml  SPR-BI: stock colocated partial rollout loop on this
#                                                                   tree's engine and BI numerics (baseline b), lag K
#   MODEL TP EP GPUS PROMPTS G MAXLEN PROMPT_CAP STEPS SEED   shape (GPUS = the one colocated fleet)
#   K         lag (async_rl.sampler.max_lookahead_versions) for colo_xpr / colo_xprs / colo_vg / colo_sprbi; ignored for colo.yaml
#   PARITY=1  math_parity reward (truncated Super / Ultra models); 0 for full Nano (real math reward)
#   CHECKS=1  per-step bitwise mismatch + law checks of every output token (exactness gate; slower)
#   SPARE=S   XPR-S: admit PROMPTS + S prompt groups per step, train PROMPTS of them (num_prompts_per_step = P + S)
#   NEG=<control>     negative control (needs CHECKS=1): carry_unverified (XPR, VG), spare_by_logp (XPR-S), vg_by_logp (VG);
#                     SKIP_RESHARD=1: XS negative
#   MAX_REQ ENGINE_BUF MBS   engine requests / KV+state buffer GiB / learner micro-batch (defaults: by rule, see below)
#   XA='<extra hydra overrides>'  XE='K=V ...' extra process env   TAG=<run name>   PREFLIGHT=1: config dry run only
# Output: $COLO_RUNS/$TAG/{run.log, preflight.log, effective_config.yaml, history.json, logs/}
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
L=$COLO_SCRIPTS/lib
TREE=${TREE:-$COLO_RL}
CFG_REL=${CFG_REL:-$COLO_PROD_CFG}
MODEL=${MODEL:-$MODEL_SUPER_L8}
TP=${TP:-2} EP=${EP:-4}
PROMPTS=${PROMPTS:-16} G=${G:-8} MAXLEN=${MAXLEN:-1024} PROMPT_CAP=${PROMPT_CAP:-2048} STEPS=${STEPS:-5}
PARITY=${PARITY:-1} CHECKS=${CHECKS:-0} SKIP_RESHARD=${SKIP_RESHARD:-0}
DRY=0; [ "${PREFLIGHT:-0}" = 1 ] && DRY=1
gcd() { local a=$1 b=$2; while (( b )); do set -- $b $(( a % b )); a=$1; b=$2; done; echo $a; }
lcm() { echo $(( $1 / $(gcd $1 $2) * $2 )); }
GPUS=${GPUS:-$(lcm $TP $EP)}
XPR=0; [[ $CFG_REL == *colo_xpr* || $CFG_REL == *colo_wv* ]] && XPR=1
WV=0; [[ $CFG_REL == *colo_wv* ]] && WV=1
VG=0; [[ $CFG_REL == *colo_vg* ]] && VG=1
SPRBI=0; [[ $CFG_REL == *colo_sprbi* ]] && SPRBI=1
ISSPR=0; [[ $CFG_REL == *colo_isspr* ]] && ISSPR=1
LAGGED=$(( XPR || VG || SPRBI || ISSPR ))
SPARE=${SPARE:-0}
(( SPARE > 0 )) && [[ $CFG_REL != *colo_xprs* && $CFG_REL != *colo_wv* ]] && { echo "[colo] SPARE=$SPARE needs colo_xprs.yaml or colo_wv.yaml" >&2; exit 2; }
NPS=$(( PROMPTS + SPARE ))
K=${K:-$COLO_PROD_K}
TAG=${TAG:-colo-$( ((LAGGED)) && echo lag$K || echo xs )-$(basename "$MODEL")-n$GPUS}
(( DRY )) && [[ $TAG != *-dry ]] && TAG=$TAG-dry
OUT=$COLO_RUNS/$TAG

# SPR-BI runs the stock loop (no ThunderSync router fingerprints): the stock arm's per-step router / bias probe instead.
(( SPRBI )) && export NRL_BIAS_PROBE=1 COLO_PREPEND=$L/baselines_nrl/bias_probe
colo_pythonpath "$TREE" bi
export NVIDIA_TF32_OVERRIDE=1
cd "$RL_ROOT" || exit 2
[ -f "$RL_ROOT/$CFG_REL" ] || { echo "[colo] no config $RL_ROOT/$CFG_REL" >&2; exit 2; }
[ -f "$COLO_PINS" ] || { echo "[colo] no pins file $COLO_PINS: run $COLO_SCRIPTS/setup.sh first" >&2; exit 2; }
mkdir -p "$OUT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$CUDA_CACHE_PATH" "$MPLCONFIGDIR" "$TORCH_EXTENSIONS_DIR"

if (( GPUS <= 4 )); then
  RES=(cluster.num_nodes=1 cluster.gpus_per_node=$GPUS)
else
  (( GPUS % 4 == 0 )) || { echo "[colo] multi-node needs whole 4-GPU nodes (GPUS=$GPUS)" >&2; exit 2; }
  RES=(cluster.num_nodes=$(( GPUS / 4 )) cluster.gpus_per_node=4)
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
# BI token multiple: ceil(64 / TP) x TP, lcm'd with 32 x TP at TP > 1 (SP RMSNorm rows); then lcm with 128.
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
# Engine capacity per engine DP rank: sync = one batch in flight; XPR = (K+1) batches of 9 drafts per prompt.
GEN_DP=$(( GPUS / TP )); (( GEN_DP < 1 )) && GEN_DP=1
if (( XPR )) && (( SPARE > 0 )); then
  # XPR-S: (K+1) batches of P + S prompts x N = ceil(k_drafts G) drafts (colo_xprs.yaml: k_drafts 1.625 -> 13 at G 8).
  PAR+=(async_rl.sampler.max_lookahead_versions=$K thundersync.spare_prompts=$SPARE)
  KD=$(awk '$1 == "k_drafts:" {v = $2} END {print v}' "$RL_ROOT/$CFG_REL")
  ND=$(python3 -c "import math; print(math.ceil(float('${KD:?no k_drafts in $CFG_REL}') * $G))")
  if (( WV )); then
    # verify at the wake: 4/3 x (the trained batch's continuations P G + every draft of the lookahead batch (P+S) N) per
    # engine rank. The 4/3 headroom is measured at Super-L22: 256 rows / rank 85.75 s/step vs 93.89 at the bare 192.
    DEF_REQ=$(( (4 * (PROMPTS * G + NPS * ND * K) + 3 * GEN_DP - 1) / (3 * GEN_DP) ))
  else
    DEF_REQ=$(( (NPS * ND * (K + 1) + GEN_DP - 1) / GEN_DP ))
  fi
elif (( VG || SPRBI || ISSPR )); then
  # VG / SPR-BI / IS-SPR: lag K; engine requests set by the preset (SPR's rows in flight), else (K+1) batches of P x G.
  PAR+=(async_rl.sampler.max_lookahead_versions=$K)
  # SPR-BI / IS-SPR: stock partial rollout's buffer and in-flight cap scale with the lag, (K+1) P prompts (stock SPR
  # gets the same from stock_arm.sh's async_rl.max_inflight_prompts).
  (( SPRBI || ISSPR )) && PAR+=(async_rl.max_buffered_rollouts=$(( (K + 1) * PROMPTS )) async_rl.max_inflight_prompts=$(( (K + 1) * PROMPTS )))
  DEF_REQ=$(( (PROMPTS * G * (K + 1) + GEN_DP - 1) / GEN_DP ))
elif (( XPR )); then
  PAR+=(async_rl.sampler.max_lookahead_versions=$K)
  DEF_REQ=$(( (PROMPTS * 9 * (K + 1) + GEN_DP - 1) / GEN_DP ))
else
  DEF_REQ=$(( (PROMPTS * G + GEN_DP - 1) / GEN_DP ))
fi
MAX_REQ=${MAX_REQ:-$(( (DEF_REQ + 15) / 16 * 16 ))}
PAR+=(++policy.generation.mcore_generation_config.max_requests=$MAX_REQ)
# Engine KV / state buffer by rule (lib/engine_buf.py): it stays resident through training, so it is sized to what the
# step holds in flight, never tuned (an oversized buffer, e.g. 24 GiB at Super-L22, makes the learner thrash).
if [ -z "${ENGINE_BUF:-}" ]; then
  MCFG=$MODEL/config.json
  [ -f "$MCFG" ] || MCFG=$(ls "$HF_HOME"/hub/models--${MODEL//\//--}/snapshots/*/config.json 2>/dev/null | head -1)
  ENGINE_BUF=$(python3 $L/engine_buf.py "$MCFG" $TP $MAX_REQ $(( PROMPT_CAP + MAXLEN )) 2>/dev/null) \
    || { echo "[colo] engine_buf.py failed for $MCFG; set ENGINE_BUF" >&2; exit 2; }
fi
PAR+=(policy.generation.mcore_generation_config.buffer_size_gb=$ENGINE_BUF)
PAR+=(policy.generation.mcore_generation_config.kv_cache_management_mode=$COLO_KV)   # offload with torch_memory_saver (COLO_TMS)
[ -n "${MBS:-}" ] && PAR+=(policy.train_micro_batch_size=$MBS)
[ -n "${SEED:-}" ] && PAR+=(grpo.seed=$SEED)
# Cross-fade band (F3): part of the policy definition, a per-model constant, the same in EVERY arm (XS, SPR-BI, every
# ThunderSync arm). Rule: the band with the lowest measured per-token TV between consecutive policies at no step-time
# cost. Nano (23 MoE layers, k 6, E 128): cfS30, m .03 x [.3, 1, .5] over the MoE-depth thirds [0, 6/23) / [6/23, 18/23)
# / [18/23, 1], extra 4. Measured on Nano 8k: TV x0.89-0.95 vs m .01 at K 1 / 2, step time and memory no worse. Other
# models keep the configs' band (m .01, extra 8: the measured Pareto point at Super-L22 / Ultra-L27) until measured the
# same way. Written where the arm reads it: the thundersync block for ThunderSync modes (apply_crossfade copies it to
# policy.megatron_cfg), policy.megatron_cfg for the stock-loop SPR-BI arm (colo_sprbi.yaml). XA still overrides.
CF_BAND=()
case "$(basename "$MODEL")" in
  *[Nn]ano*) CF_BAND=(crossfade_routing_m=0.03 "crossfade_routing_m_profile=[[0.0,0.2608,0.3],[0.2608,0.7826,1.0],[0.7826,1.0,0.5]]"
                      crossfade_routing_extra=4) ;;
esac
CF_PFX=thundersync; [[ $CFG_REL == *colo_sprbi.yaml ]] && CF_PFX=policy.megatron_cfg
for kv in "${CF_BAND[@]}"; do PAR+=("++$CF_PFX.$kv"); done
if [ "$PARITY" = 1 ]; then export NL_PARITY_REWARD=1; PAR+=(data.default.env_name=math_parity); fi
unset THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_ENGINE_Q_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS THUNDERSYNC_NL_NEG \
  NEMO_RL_DEBUG_COLOC_SKIP_RESHARD
if [ "$CHECKS" = 1 ]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_LAW_CHECK=1 THUNDERSYNC_NL_CHECKS=1
  for v in THUNDERSYNC_MISMATCH_CHECK THUNDERSYNC_LAW_CHECK THUNDERSYNC_NL_CHECKS; do
    PAR+=("++policy.megatron_cfg.env_vars.$v='1'")
  done
fi
if [ -n "${NEG:-}" ]; then
  [ "$CHECKS" = 1 ] || { echo "[colo] NEG=$NEG is a negative control: needs CHECKS=1" >&2; exit 2; }
  export THUNDERSYNC_NL_NEG=$NEG
  PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_NL_NEG='$NEG'")
fi
if [ "$SKIP_RESHARD" = 1 ]; then
  [ "$CHECKS" = 1 ] || { echo "[colo] SKIP_RESHARD=1 is a negative control: needs CHECKS=1" >&2; exit 2; }
  export NEMO_RL_DEBUG_COLOC_SKIP_RESHARD=1
fi
if [ -n "${XE:-}" ]; then
  read -r -a _XE <<< "$XE"
  for kv in "${_XE[@]}"; do [[ $kv == *=* ]] || { echo "[colo] XE entry '$kv' is not K=V" >&2; exit 2; }; export "$kv"; done
fi
read -r -a USER_ARGS <<< "${XA:-}"
rm -f "$OUT/negative"; { [ -n "${NEG:-}" ] || [ "$SKIP_RESHARD" = 1 ]; } && echo "${NEG:-skip_reshard}" > "$OUT/negative"
( ((SPRBI)) && echo sprbi || echo colo ) > "$OUT/arm"; echo $GPUS > "$OUT/gpus"; ( ((LAGGED)) && echo $K || echo 0 ) > "$OUT/lag"
export NL_EFFECTIVE_CONFIG=$OUT/effective_config.yaml
echo "[colo] TAG=$TAG CFG=$CFG_REL K=$( ((LAGGED)) && echo $K || echo 0) TP=$TP EP=$EP GPUS=$GPUS STEPS=$STEPS PROMPTS=$PROMPTS" \
  "G=$G MAXLEN=$MAXLEN PROMPT_CAP=$PROMPT_CAP MAX_REQ=$MAX_REQ ENGINE_BUF=$ENGINE_BUF MBS=${MBS:-cfg} SEQDIV=$SEQDIV" \
  "SPARE=$SPARE PARITY=$PARITY CHECKS=$CHECKS NEG=${NEG:-} SKIP_RESHARD=$SKIP_RESHARD MODEL=$MODEL" \
  "RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD 2>/dev/null)" \
  "dirty=$(git -C "$RL_ROOT" status --porcelain --untracked-files=no 2>/dev/null | wc -l) -> $OUT"
CLI=(--config "$RL_ROOT/$CFG_REL"
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS
  grpo.num_prompts_per_step=$NPS grpo.num_generations_per_prompt=$G
  policy.train_global_batch_size=$(( PROMPTS * G )) policy.max_total_sequence_length=$(( PROMPT_CAP + MAXLEN ))
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN
  "${RES[@]}" "${PAR[@]}" "${USER_ARGS[@]}")
ENTRY=$RL_ROOT/research/thundersync_rl/run_nolanes_sc.py
PY=${PY:-python}

# Config dry run (no Ray, no GPU work): compose + validate + capacity; gates the real run.
NL_DRY=1 timeout ${PREFLIGHT_TIMEOUT:-600} $PY -u $ENTRY "${CLI[@]}" > "$OUT/preflight.log" 2>&1
prc=$?
grep -a "^\[nl\] DRY\|^\[nolanes\]" "$OUT/preflight.log" | cut -c1-600
if ! grep -q "\[nl\] DRY OK" "$OUT/preflight.log"; then
  echo "[colo] PREFLIGHT FAIL rc=$prc (see $OUT/preflight.log)"; tail -15 "$OUT/preflight.log"; exit 3
fi
# Hard rule: MoE router weights AND expert bias frozen in every run (effective config).
if ! grep -q "\[nl\] DRY router: frozen (weights+bias)" "$OUT/preflight.log"; then
  echo "[colo] PREFLIGHT FAIL: router not frozen (weights+bias) in the effective config"; exit 3
fi
(( DRY )) && { echo "[colo] preflight only: OK"; exit 0; }
# 1 node: the previous arm of this job may still hold the GPUs (lib/gpu_free.sh).
(( GPUS <= 4 )) && { bash $L/gpu_free.sh || { echo "[colo] GPUs not free" >&2; exit 4; }; }

# GPU-idle watchdog: 1 node = monitor + enforcer here; multi-node = per-node monitors from ray_cluster.sh, enforcer here.
WD=${WD:-auto}
if [ "$WD" = auto ]; then
  if [ -z "${SLURM_JOB_ID:-}" ]; then WD=0; elif (( GPUS > 4 )); then WD=enforce; else WD=local; fi
fi
case $WD in
  local) bash $L/gpu_idle_watchdog.sh local-bg "$OUT/run.log" ;;
  enforce) bash $L/gpu_idle_watchdog.sh enforce-bg "$OUT/run.log" ;;
  0) ;;
  *) echo "[colo] WD must be auto | local | enforce | 0" >&2; exit 2 ;;
esac
rm -f "$OUT/history.json" "$OUT/staleness.json"
timeout ${RUN_TIMEOUT:-10800} $PY -u $ENTRY "${CLI[@]}" 2>&1 | python3 -u $L/baselines_nrl/stamp.py > "$OUT/run.log"
rc=${PIPESTATUS[0]}
echo "[colo] run exit=$rc"
python3 $L/baselines_nrl/parse_nrl.py "$OUT" || true
python3 $L/check_run.py "$OUT" || { [ $rc = 0 ] && rc=5; }
exit $rc
