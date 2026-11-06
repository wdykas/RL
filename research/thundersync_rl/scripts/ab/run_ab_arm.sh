#!/bin/bash
# Run ONE arm of the ThunderSync A/B kit (configs/ab/ab_<ARM>.yaml) on the head node of a Ray cluster
# (sbatch via $TS_WORK/sbatch/submit_ab.sh) or on one 4-GPU node (2 engine + 2 learner GPUs).
#
#   ARM=sync|async1|ts_nonull|ts_por|ts_banded|sync_coloc|async1_nrl [CFG_FILE=research/thundersync_rl/configs/paper/<arm>.yaml] [MBS=8 LPBS=4] MODEL=<hf id or path> TP=1 PP=1 EP=8 \
#     MAXLEN=8192 STEPS=6 PROMPTS=64 RUNS=runs_ab TAG=<tag> bash run_ab_arm.sh
# Lanes-k arms (ts-lanesk; research/lanesk/README.md): ARM=async<k> (k >= 2: fixed-age-k async GRPO on k engine lanes),
# async1l (k = 1 on the lanes-k path: the a1 equivalence gate), ts_d<k> (ThunderSync at depth k on k lanes; ts_d2 =
# LA2). LANES defaults to k for these arms (a different LANES is refused); GEN_GPUS / GEN_EP = one lane's engine.
# Each writes $OUT/depth = k (summarize_ab.py: window max(FIRST, k + 1) .. N - 1 - k, i.e. without the pipeline fill
# and the last k steps, which start no new cohort).
#
# GPUS_PER_SIDE (default lcm(TP, EP) * PP) GPUs per side, 4 GPUs per node. sync_coloc (stock NeMo-RL GRPO)
# is colocated: learner and engine share GPUS_PER_SIDE GPUs. Every other arm is split: GPUS_PER_SIDE
# learner + GPUS_PER_SIDE engine GPUs (1-2 per side share one node). The engine uses the learner's
# TP/PP/EP (GEN_PP overrides the engine's PP).
# Exactness checks (ThunderSync-loop arms, CHECKS=1 default): THUNDERSYNC_MISMATCH_CHECK (one learner
# forward of the step's rollouts vs their rollout log-probs, timed as time/mismatch_check),
# THUNDERSYNC_ENGINE_Q_CHECK (engine q vs the verification forward, per verified token) and a
# route check of ROUTE_CHECK_ROWS (16) sampled drafts per step (timed as time/route_check).
# The stock arms (sync_coloc, async1_nrl) report NeMo-RL's token_mult_prob_error / gen_kl_error.
# Generation progress (every 60 s, engine rank 0) and the 10-min stall watchdog: ab_base.yaml.
# Output: $TS_WORK/$RUNS/$TAG/{run.log,history.json} (history rewritten every step).
# Summarize: python $TS_WORK/research/ab/summarize_ab.py $TS_WORK/$RUNS/<tags...>
set -uo pipefail
W=${TS_WORK:?set TS_WORK to the research workspace (models/, overlays, env_sbatch.sh, run dirs)}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ARM=${ARM:?ARM} MODEL=${MODEL:?MODEL}
TP=${TP:-1} PP=${PP:-1} EP=${EP:-1} STEPS=${STEPS:-6} PROMPTS=${PROMPTS:-64} G=${G:-8}
MAXLEN=${MAXLEN:-8192} RUNS=${RUNS:-runs_ab} CHECKS=${CHECKS:-1}
# Lengths, identical on every arm: MAXLEN = generation cap (max_new_tokens); PROMPT_CAP = prompt cap
# (data.max_input_seq_length; longer prompts are masked out by the data processor, loss_multiplier 0);
# policy.max_total_sequence_length = PROMPT_CAP + MAXLEN, so no kept prompt overflows the sequence budget
# (DAPO prompts reach ~1.25k tokens: a 1k total cap overflowed).
PROMPT_CAP=${PROMPT_CAP:-2048}
TAG=${TAG:-$ARM-$(basename "$MODEL")-tp$TP-pp$PP-ep$EP-L$MAXLEN}
export RL_ROOT_OVERRIDE=${RL_ROOT_OVERRIDE:-$(cd "$HERE/../../../.." && pwd)}
source $W/env_sbatch.sh bi
cd "$RL_ROOT" || exit 2
export NVIDIA_TF32_OVERRIDE=1
# Per-rank lines ([learner prof], [engine prof]) differ only in numbers: Ray's driver-side log
# dedup kept ~1 of 8 ("[repeated 7x across cluster]"), hiding the slowest rank. Off for every arm.
export RAY_DEDUP_LOGS=0
# CFG_FILE=<tree-relative yaml> runs the arm from another config (configs/paper/<arm>.yaml: the paper configuration,
# whose switches come from the file, thundersync.driver_env included); ARM still names the loop (entry) and the arm.
CFG=${CFG_FILE:-research/thundersync_rl/configs/ab/ab_$ARM.yaml}
[ -f "$CFG" ] || { echo "no config $CFG" >&2; exit 2; }
case $ARM in
  sync_coloc|async1_nrl) ENTRY=research/thundersync_rl/run_ab_baseline.py ;;  # stock NeMo-RL GRPO loops
  *)                     ENTRY=research/thundersync_rl/run_thundersync_grpo.py ;;  # ThunderSync learner path
esac
# sync_coloc is colocated (learner and engine share GPUS_PER_SIDE GPUs); every other arm is split
# (GPUS_PER_SIDE learner + GPUS_PER_SIDE engine GPUs).
COLOCATED=$([ "$ARM" = sync_coloc ] && echo 1 || echo 0)
# Exactness checks on the exact ThunderSync-loop arms (async1 is one step stale by design).
# Since ts-exact-harden (research/agent_xhard.md), CHECKS=1 also runs:
#  - THUNDERSYNC_LAW_CHECK: randomized-PIT law test of every trained token under the learner's p, inside the
#    mismatch-check forward (law_check.py; per-step p-value, history law/*; fails at p < THUNDERSYNC_LAW_ALPHA 1e-4);
#  - THUNDERSYNC_ENGINE_PREFILL_CHECK: engine prefill p == learner verification p on <= PREFILL_ROWS (8) rows /
#    PREFILL_TOKENS (32768) tokens per step, rows with a rejection / continuation first (lanes: on the cohort lane
#    once it holds theta_k; timed as time/prefill_check, subtracted from "net");
#  - THUNDERSYNC_TRAIN_LP_CHECK: the training forward's log-probs vs the rollout log-probs (the loss forces the
#    ratio to 1, which hid any difference): trainlp/max_abs_diff must be 0.
# LAW=0 / PREFILL=0 / TRAINLP=0 turn one off. Declared pass rules: EXPECT='rule ...' (research/ab/run_rules.py)
# plus law.present / trainlp.present implied here; summarize_ab.py --check fails the arm on any failed rule.
if [ "$CHECKS" = 1 ] && [ "$ENTRY" = research/thundersync_rl/run_thundersync_grpo.py ] && [[ $ARM != async* ]]; then
  export THUNDERSYNC_MISMATCH_CHECK=1 THUNDERSYNC_ENGINE_Q_CHECK=1
  export THUNDERSYNC_ROUTE_CHECK_ROWS=${ROUTE_CHECK_ROWS:-16}
  [ "${LAW:-1}" = 1 ] && export THUNDERSYNC_LAW_CHECK=1
  if [ "${PREFILL:-1}" = 1 ]; then
    export THUNDERSYNC_ENGINE_PREFILL_CHECK=1 THUNDERSYNC_ENGINE_PREFILL_CHECK_ROWS=${PREFILL_ROWS:-8}
    export THUNDERSYNC_ENGINE_PREFILL_CHECK_TOKENS=${PREFILL_TOKENS:-32768}
  fi
  [ "${TRAINLP:-1}" = 1 ] && export THUNDERSYNC_TRAIN_LP_CHECK=1
fi
# Production subset (ts-nano-scale, DESIGN 14ax): CHECKS=prod = the cheap always-on checks: training-forward log-probs vs
# rollout log-probs on every trained token (3 device scalars per micro-batch), the law test + mismatch on PROD_ROWS rows
# (default 2 x the learner DP size ... set explicitly; 16) chosen by a hash of (seed, step, row index), the verifier
# emission-law PIT on 1 in EMIT_EVERY (4) decisions (hash of the row key). No engine-q rebuild, route or prefill check.
# The row-cap assertion, weight sentinel, non-finite verification and path counters are always on (any CHECKS).
PROD_ENV="THUNDERSYNC_MISMATCH_CHECK=1,THUNDERSYNC_MISMATCH_CHECK_ROWS=${PROD_ROWS:-16},THUNDERSYNC_LAW_CHECK=1,THUNDERSYNC_EMISSION_LAW_EVERY=${EMIT_EVERY:-4},THUNDERSYNC_TRAIN_LP_CHECK=1"
if [ "$CHECKS" = prod ] && [ "$ENTRY" = research/thundersync_rl/run_thundersync_grpo.py ] && [[ $ARM != async* ]]; then
  for kv in ${PROD_ENV//,/ }; do export "$kv"; done
fi
# Sampled full checks for long runs: CHECKS=1 CHECK_EVERY=k = the full set on steps with step % k == CHECK_PHASE
# (default k/2) and CHECK_OFF (prod: the production subset, default; none: nothing) on the others (checks_sched.py).
if [ -n "${CHECK_EVERY:-}" ] && [ "${CHECK_EVERY:-0}" -gt 1 ]; then
  [ "$CHECKS" = 1 ] || { echo "CHECK_EVERY needs CHECKS=1 (the full set it samples)" >&2; exit 2; }
  export THUNDERSYNC_CHECK_EVERY=$CHECK_EVERY
  [ -n "${CHECK_PHASE:-}" ] && export THUNDERSYNC_CHECK_PHASE=$CHECK_PHASE
  case "${CHECK_OFF:-prod}" in
    prod) export THUNDERSYNC_CHECK_OFF_ENV="$PROD_ENV" ;;
    none) export THUNDERSYNC_CHECK_OFF_ENV="" ;;
    *) echo "CHECK_OFF must be prod | none" >&2; exit 2 ;;
  esac
fi
gcd() { local a=$1 b=$2; while (( b )); do set -- $b $(( a % b )); a=$1; b=$2; done; echo $a; }
# CP (BI context parallelism, learner only: bi_context_parallel.py; the engine runs CP 1).
CP=${CP:-1}
GPS=${GPUS_PER_SIDE:-$(( TP * EP / $(gcd $TP $EP) * PP * CP ))}
if (( GPS > 2 && GPS % 4 )); then echo "GPUS_PER_SIDE must be 1, 2 or a multiple of 4 (got $GPS)" >&2; exit 2; fi
# Engine (generation) layout: GEN_EP (default EP) / GEN_PP (default PP) per engine replica of
# GEN_GPS = lcm(TP, GEN_EP) * GEN_PP GPUs. LANES=2 (ThunderSync arms, thundersync.cohort_lanes):
# two such engines, lane 1 built by run_thundersync_grpo.py outside cluster.* (learner + lane 0).
# Equal-GPU A/B at EP8: one engine GEN_EP=8 (8 GPUs) vs LANES=2 GEN_EP=4 (2 x 4 GPUs; EP-invariant
# combine switched on below because the engine EP differs from the learner's).
# Lanes-k arms: k from the arm name (async<k>, async1l = 1, ts_d<k>); LANES must be k (default).
case $ARM in
  async1l) ARM_K=1 ;;
  async[0-9]*) ARM_K=${ARM#async}; ARM_K=${ARM_K%%_*}; [[ $ARM_K =~ ^[0-9]+$ ]] || ARM_K="" ;;  # async<k>[_variant]
  ts_d[0-9]*) ARM_K=${ARM#ts_d}; [[ $ARM_K =~ ^[0-9]+$ ]] || ARM_K="" ;;
  *) ARM_K="" ;;
esac
  [[ $ARM == async1 || $ARM == async1_* ]] && ARM_K=""  # the single-engine async1 arm (rollout_lag=1, no lanes)
if [ -n "$ARM_K" ]; then
  [ -z "${LANES:-}" ] || [ "$LANES" = "$ARM_K" ] || { echo "ARM=$ARM runs on $ARM_K lanes; LANES=$LANES given" >&2; exit 2; }
  LANES=$ARM_K
fi
GEN_EP=${GEN_EP:-$EP} GEN_PP=${GEN_PP:-$PP} LANES=${LANES:-1}
GEN_GPS=${GEN_GPUS:-$(( TP * GEN_EP / $(gcd $TP $GEN_EP) * GEN_PP ))}
ENGINES=$(( LANES > 1 ? LANES : 1 ))
if [ "$COLOCATED" = 1 ]; then
  GPUS=$GPS
  RES=(cluster.num_nodes=$(( GPS > 4 ? GPS / 4 : 1 )) cluster.gpus_per_node=$(( GPS > 4 ? 4 : GPS )))
else
  GPUS=$(( GPS + ENGINES * GEN_GPS ))
  CLUSTER_GPUS=$(( GPS + GEN_GPS ))  # learner + lane 0 (lane 1 outside cluster.*)
  if (( GPUS <= 4 )); then
    RES=(cluster.num_nodes=1 cluster.gpus_per_node=$CLUSTER_GPUS
         policy.generation.colocated.resources.num_nodes=1
         policy.generation.colocated.resources.gpus_per_node=$GEN_GPS)
  elif (( GPS % 4 == 0 && GEN_GPS < 4 && LANES > 1 && (4 % GEN_GPS) == 0 )); then
    # Lanes-k with lanes smaller than a node (e.g. 4 lanes x 2 GPUs under an 8-GPU learner): lane 0 is a partial
    # node (grpo.setup allow_partial_node, ts-lanesk); the other lanes' placement groups fill the remaining GPUs.
    (( (ENGINES * GEN_GPS) % 4 == 0 )) || { echo "LANES x GEN_GPUS = $(( ENGINES * GEN_GPS )) is not whole 4-GPU nodes" >&2; exit 2; }
    RES=(cluster.num_nodes=$(( GPS / 4 + 1 )) cluster.gpus_per_node=4
         policy.generation.colocated.resources.num_nodes=1
         policy.generation.colocated.resources.gpus_per_node=$GEN_GPS
         ++policy.generation.colocated.resources.allow_partial_node=true)
  else
    (( GPS % 4 == 0 && GEN_GPS % 4 == 0 )) || {
      echo "multi-node layouts need GPUS_PER_SIDE and the engine replica in whole 4-GPU nodes (lanes: 1, 2 or 4 GPUs per lane also work)" >&2; exit 2; }
    RES=(cluster.num_nodes=$(( CLUSTER_GPUS / 4 )) cluster.gpus_per_node=4
         policy.generation.colocated.resources.num_nodes=$(( GEN_GPS / 4 ))
         policy.generation.colocated.resources.gpus_per_node=4)
  fi
fi
if (( LANES > 1 )); then
  [ "$COLOCATED" = 0 ] || { echo "LANES=$LANES needs a split arm" >&2; exit 2; }
  LANE_ARGS=(++thundersync.cohort_lanes=$LANES)
else
  LANE_ARGS=()
fi
# Learner prefix-state cache (DESIGN 14p) for the ThunderSync arms at long caps: piece
# verification without it grows with L^2 (0.6-2.0x of training time at 4k). PREFIX_CACHE=0|1 overrides.
if [[ $ARM == ts_* ]] && [ "${PREFIX_CACHE:-$(( MAXLEN >= 4096 ? 1 : 0 ))}" = 1 ]; then
  LANE_ARGS+=(++thundersync.learner_prefix_cache=true)
fi
SP=$([ "$TP" -gt 1 ] && echo true || echo false)
PAR=(policy.megatron_cfg.tensor_model_parallel_size=$TP
     policy.megatron_cfg.pipeline_model_parallel_size=$PP
     policy.megatron_cfg.expert_model_parallel_size=$EP
     policy.megatron_cfg.sequence_parallel=$SP
     ++policy.generation.mcore_generation_config.tensor_model_parallel_size=$TP
     ++policy.generation.mcore_generation_config.pipeline_model_parallel_size=$GEN_PP
     ++policy.generation.mcore_generation_config.expert_model_parallel_size=$GEN_EP)
(( CP > 1 )) && PAR+=(policy.megatron_cfg.context_parallel_size=$CP
                     ++policy.generation.mcore_generation_config.context_parallel_size=1)
(( PP > 1 || GEN_PP > 1 )) && PAR+=(++policy.megatron_cfg.pipeline_dtype=float32)
# Opt-in cuBLAS workspace shim (CUBLAS_WS_SHIM=1; not a default until model-level fingerprint checks pass):
# LD_PRELOAD forces a 0-byte cuBLAS workspace instead of CUBLAS_WORKSPACE_CONFIG=:0:0, which PyTorch 2.11
# re-parses with std::regex per cuBLAS call (58 -> 24 us host per mm, same bits; research/agent_callcost.md).
# The driver env (env_sbatch.sh) reaches Ray workers started from this shell; multi-node workers get it
# through the env_vars overrides below.
if [ "${CUBLAS_WS_SHIM:-0}" = 1 ]; then
  PAR+=("~policy.megatron_cfg.env_vars.CUBLAS_WORKSPACE_CONFIG"
        "++policy.megatron_cfg.env_vars.CUBLAS_WORKSPACE_FIXED_BYTES='0'"
        "++policy.megatron_cfg.env_vars.LD_PRELOAD=$W/research/callcost/cublas_ws_shim/libcublas_ws_shim.so")
fi
# Opt-in learner attribution (multi-node workers get it through env_vars): LEARNER_PROF=1 prints one
# "[learner prof] rank=.. chunks=.. rows=.. shapes=.. secs=.." line per rank per learner call (two device syncs per
# call); LEARNER_TORCHPROF=N runs the torch profiler once, on the call holding the N-th trained item, and prints its
# top-30 CPU-time table (that one dispatch is slower).
[ "${LEARNER_PROF:-0}" = 1 ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_LEARNER_PROF='1'")
[ -n "${LEARNER_TORCHPROF:-}" ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_LEARNER_TORCHPROF='$LEARNER_TORCHPROF'")
# Opt-in engine step timing (engine_prof.py): ENGINE_PROF=1 logs engine/decode_ms_rows* and engine/cycle_ms_rows* per
# step (the driver reads them, so it gets the variable too). DECODE_FUSED=0: the decode fusions of ts-decode-speed
# off (op-by-op paths, same bits) in every worker, the A/B control.
if [ "${ENGINE_PROF:-0}" = 1 ]; then
  export THUNDERSYNC_ENGINE_PROF=1
  PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_ENGINE_PROF='1'")
fi
if [ "${DECODE_FUSED:-1}" = 0 ]; then
  export THUNDERSYNC_BI_DECODE_FUSED=0 THUNDERSYNC_FUSED_DECODE_OPS=0
  PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_BI_DECODE_FUSED='0'"
        "++policy.megatron_cfg.env_vars.THUNDERSYNC_FUSED_DECODE_OPS='0'")
fi
# Opt-in learner host-overhead cuts (research/agent_lgraphs.md): LGRAPHS=dd = host-sync-free MoE dispatch + loss metrics
# resolved once per learner call (NEMO_RL_LEARNER_DEFER_METRICS=1, same values)
# (NEMO_RL_MOE_DEVICE_DISPATCH=1: device counts; EP > 1 symmetric-memory exchange, bitwise Megatron's);
# LGRAPHS=1 = that + CUDA graphs of the learner's no-grad forwards (verification / log-probs, token buckets) and
# LGRAPHS=2 = that + training fwd+bwd graphs (NEMO_RL_LEARNER_GRAPHS_TRAIN=1). The dispatch workspace's token cap T_cap must cover the
# largest learner call per rank: max(micro-batch rows x (PROMPT_CAP + MAXLEN), verify_batch_tokens 16384 x 9/8
# bucket padding) = 18432 at least; receive capacity LGRAPHS_CAP (default 2.0 x T_cap x top-k rows per rank).
# LGRAPHS=full (ts-lgfull, DESIGN 14bl) = every learner call graphed: no-grad + training graphs of every micro-batch up to
# T_cap (token buckets), one shared budgeted pool, direct capture (learner_graphs defaults); LGRAPHS=1 / 2 keep the
# earlier layout (separate pools, no-grad pool released at the first training call, training <= LGRAPHS_TRAIN_MAX).
case "${LGRAPHS:-0}" in
  0) ;;
  dd|1|2|full)
    _TCAP=$(( ${MBS:-2} * (${PROMPT_CAP:-2048} + MAXLEN) )); (( _TCAP < 18432 )) && _TCAP=18432
    PAR+=("++policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH='1'"
          "++policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS='$_TCAP'"
          "++policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH_CAPACITY='${LGRAPHS_CAP:-2.0}'"
          "++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_DEFER_METRICS='1'")
    export NEMO_RL_MOE_DEVICE_DISPATCH=1 NEMO_RL_MOE_DEVICE_DISPATCH_TOKENS=$_TCAP NEMO_RL_MOE_DEVICE_DISPATCH_CAPACITY=${LGRAPHS_CAP:-2.0}
    if [ "$LGRAPHS" != dd ]; then
      PAR+=("++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS='1'")
      export NEMO_RL_LEARNER_GRAPHS=1
    fi
    if [ "$LGRAPHS" = full ]; then
      # + per-call MoE receive rows (expert buffers sized by the call, not T_cap; LGF_CALL_CAP=0: off)
      PAR+=("++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS_TRAIN='1'"
            "++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS_SHARED_POOL='1'"
            "++policy.megatron_cfg.env_vars.NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP='${LGF_CALL_CAP:-1}'")
      export NEMO_RL_LEARNER_GRAPHS_TRAIN=1 NEMO_RL_LEARNER_GRAPHS_SHARED_POOL=1 NEMO_RL_MOE_DEVICE_DISPATCH_CALL_CAP=${LGF_CALL_CAP:-1}
    else
      PAR+=("++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS_SHARED_POOL='0'")
      export NEMO_RL_LEARNER_GRAPHS_SHARED_POOL=0
    fi
    if [ "$LGRAPHS" = 2 ]; then  # + training fwd+bwd graphs (exact micro-batch shapes, rows x length <= LGRAPHS_TRAIN_MAX tokens; MBS x (PROMPT_CAP + MAXLEN) above it runs eagerly)
      PAR+=("++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS_TRAIN='1'"
            "++policy.megatron_cfg.env_vars.NEMO_RL_LEARNER_GRAPHS_TRAIN_MAX_TOKENS='${LGRAPHS_TRAIN_MAX:-8192}'")
      export NEMO_RL_LEARNER_GRAPHS_TRAIN=1 NEMO_RL_LEARNER_GRAPHS_TRAIN_MAX_TOKENS=${LGRAPHS_TRAIN_MAX:-8192}
    fi ;;
  *) echo "LGRAPHS must be 0 | dd | 1 | 2 | full"; exit 1 ;;
esac
# verify0's learner-forward budget (thundersync.verify0_batch_tokens, DESIGN 14ao / 14ar): V0B=<tokens> or V0B=tcap = the
# device-dispatch token cap computed above (the largest verify forward the symmetric exchange takes at EP > 1; with
# LGRAPHS=0 the same formula). Packing only: exact for any value; the validator refuses a budget above the cap.
if [ -n "${V0B:-}" ]; then
  if [ "$V0B" = tcap ]; then
    V0B=$(( ${MBS:-2} * (${PROMPT_CAP:-2048} + MAXLEN) )); (( V0B < 18432 )) && V0B=18432
  fi
  PAR+=(++thundersync.verify0_batch_tokens=$V0B)
fi
# Engine EP != learner EP (e.g. EP4 lanes under an EP8 learner) needs the EP-invariant MoE combine
# (exact limb sums; DESIGN 14u): without it plainly sampled rows differ (np-v2-lanes-ep, epinv-lanes-ep-off).
# EPINV=1 forces it on at equal EP, EPINV=0 off (the validators then refuse an EP mismatch).
if [ "${EPINV:-$([ "$GEN_EP" != "$EP" ] && echo 1 || echo 0)}" = 1 ]; then
  PAR+=(++policy.megatron_cfg.moe_ep_invariant_combine=true)
fi
# Micro-batch sizes (default: ab_base.yaml, train 2 rows bounded by 20480 tokens -> 2 at 8k, 1 at 16k+; logprob 1;
# the 1k-cap single-node runs use 8 / 4). MBS=k sets the rows and, on every ThunderSync-loop arm, the token cap
# k x (PROMPT_CAP + MAXLEN), so the cap never cuts MBS below what was asked (same rule in every arm).
if [ -n "${MBS:-}" ]; then
  PAR+=(policy.train_micro_batch_size=$MBS)
  [ "$ENTRY" = research/thundersync_rl/run_thundersync_grpo.py ] && \
    PAR+=(++thundersync.train_micro_batch_tokens=$(( MBS * (PROMPT_CAP + MAXLEN) )))
fi
[ -n "${LPBS:-}" ] && PAR+=(policy.logprob_batch_size=$LPBS)
# Learner-side switches read in the learner workers (multi-node workers get them through env_vars).
[ "${THUNDERSYNC_TRAIN_LP_CHECK:-0}" = 1 ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_TRAIN_LP_CHECK='1'")
[ "${THUNDERSYNC_LAW_CHECK:-0}" = 1 ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_LAW_CHECK='1'")
[ -n "${THUNDERSYNC_DEBUG_LAW_BUG:-}" ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_DEBUG_LAW_BUG='$THUNDERSYNC_DEBUG_LAW_BUG'")
# ts-lawfix diagnostic: the law test's pre-fix PIT grid (same on every DP replica) on the same tokens, "[law check legacy-grid]".
[ "${THUNDERSYNC_LAW_LEGACY_SHADOW:-0}" = 1 ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_LAW_LEGACY_SHADOW='1'")
[ -n "${THUNDERSYNC_EMISSION_LAW_EVERY:-}" ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_EMISSION_LAW_EVERY='$THUNDERSYNC_EMISSION_LAW_EVERY'")
# Learner prefix-state cache check (each cached forward vs the whole rows; one extra forward per batch) and the
# verify0-capture negative control (captures scaled by 1 + x: the check must FAIL; DESIGN 14bk). V0C=1: verify0 capture on.
[ "${THUNDERSYNC_PREFIX_CACHE_CHECK:-0}" = 1 ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_PREFIX_CACHE_CHECK='1'")
[ -n "${THUNDERSYNC_PREFIX_CACHE_NEGCTL:-}" ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_PREFIX_CACHE_NEGCTL='$THUNDERSYNC_PREFIX_CACHE_NEGCTL'")
[ "${V0C:-0}" = 1 ] && PAR+=(++thundersync.learner_prefix_cache_verify0=true)
# THUNDERSYNC_STICKY_CHECK=0: CHECKS=1 without the sticky-rule counters, so verification forwards stay graph-eligible.
[ -n "${THUNDERSYNC_STICKY_CHECK:-}" ] && PAR+=("++policy.megatron_cfg.env_vars.THUNDERSYNC_STICKY_CHECK='$THUNDERSYNC_STICKY_CHECK'")
# Paired seeds (campaign protocol): SEED sets grpo.seed (prompt order + every keyed stream); default the config's (42).
[ -n "${SEED:-}" ] && PAR+=(grpo.seed=$SEED)
# Engine KV/Mamba buffer per rank (GiB): ENGINE_BUF (default ab_base 48 at 8k; 32k needs ~96 at max_requests 256).
[ -n "${ENGINE_BUF:-}" ] && PAR+=(++policy.generation.mcore_generation_config.buffer_size_gb=$ENGINE_BUF)
# Checkpoints / exact resume (DESIGN 14s): CKPT_DIR=<dir> SAVE_PERIOD=<steps> -> NeMo-RL checkpointing at quiescent
# ThunderSync boundaries; a rerun with the same CKPT_DIR resumes from its latest ThunderSync checkpoint (same config
# fingerprint, same max_num_steps; run_state.resolve_resume). KEEP_CKPT=<n> keeps the newest n.
if [ -n "${CKPT_DIR:-}" ]; then
  PAR+=(checkpointing.enabled=true "checkpointing.checkpoint_dir=$CKPT_DIR" checkpointing.save_period=${SAVE_PERIOD:-10}
        ++checkpointing.save_optimizer=true ++checkpointing.keep_top_k=${KEEP_CKPT:-2}
        # NeMo-RL's retention treats only step % save_period == 0 as periodic; anchor-epoch boundaries sit at
        # anchor_start + k x save_period (DESIGN 14bd) and would all be deleted but the latest: keep the newest n
        # by recency too.
        ++checkpointing.ft_keep_latest_k=${KEEP_CKPT:-2}
        ++checkpointing.metric_name=null ++checkpointing.higher_is_better=true
        ++checkpointing.checkpoint_must_save_by=${CKPT_SAVE_BY:-null}
        # Synchronous distributed save: the async-save worker gets CUDA tensors over CUDA IPC, which fails for
        # the learner's expandable-segment memory (pidfd_getfd; ns-resume 10-16). Every rank still writes its shards.
        ++policy.megatron_cfg.checkpoint.async_save=false)
fi
read -r -a USER_ARGS <<< "${ARGS:-}"
OUT=$W/$RUNS/$TAG
mkdir -p "$OUT"
# Long-run leak probe (leak_probe.py, DESIGN 14bd): the run directory whose size is tracked per step.
export THUNDERSYNC_RUN_DIR=$OUT
rm -f "$OUT/history.json"
echo "$ARM" > "$OUT/arm"
echo "$GPUS" > "$OUT/gpus"  # GPUs the arm holds (summarizer: GPU-seconds per step)
# Machine-checked pass rules (research/ab/run_rules.py): the length cap of a single-turn run, and the declared rules.
rm -f "$OUT/maxlen" "$OUT/expect" "$OUT/depth"
# Lanes-k arms: the pipeline depth k (summarize_ab.py scoring window).
[ -n "$ARM_K" ] && echo "$ARM_K" > "$OUT/depth"
[[ " ${ARGS:-} " == *max_rollout_turns* ]] || echo "$MAXLEN" > "$OUT/maxlen"
_EXP="${EXPECT:-}"
[ "${THUNDERSYNC_LAW_CHECK:-0}" = 1 ] && _EXP+=" law.present"
[ "${THUNDERSYNC_TRAIN_LP_CHECK:-0}" = 1 ] && _EXP+=" trainlp.present"
# Learner graphs: device dispatch must be active (no-grad replays are not implied: under CHECKS=1 the verification
# forwards collect sticky-check statistics -> ineligible, and log-prob forwards return > 1 GiB of logits -> eager).
[ "${LGRAPHS:-0}" = 1 ] || [ "${LGRAPHS:-0}" = 2 ] || [ "${LGRAPHS:-0}" = full ] && _EXP+=" log.device_dispatch>0"
# full: training graphs must replay, and no pool release after warm-up beyond the first-training-call one.
[ "${LGRAPHS:-0}" = full ] && _EXP+=" graphs.replay_train>0 graphs.replay_fwd>0"
[ -n "${_EXP// }" ] && echo $_EXP > "$OUT/expect"
echo "[ab] ARM=$ARM TP=$TP PP=$PP EP=$EP GPS=$GPS GEN_EP=$GEN_EP GEN_GPS=$GEN_GPS GPUS=$GPUS COLOCATED=$COLOCATED LANES=$LANES STEPS=$STEPS PROMPTS=$PROMPTS G=$G MAXLEN=$MAXLEN PROMPT_CAP=$PROMPT_CAP MODEL=$MODEL" \
  "RL_ROOT=$RL_ROOT @ $(git -C "$RL_ROOT" rev-parse --short HEAD) -> $OUT"
timeout ${RUN_TIMEOUT:-14400} python $ENTRY --config $CFG --history-json "$OUT/history.json" \
  logger.log_dir="$OUT/logs" policy.model_name=$MODEL grpo.max_num_steps=$STEPS \
  grpo.num_prompts_per_step=$PROMPTS grpo.num_generations_per_prompt=$G \
  policy.train_global_batch_size=$((PROMPTS * G)) policy.max_total_sequence_length=$((PROMPT_CAP + MAXLEN)) \
  data.max_input_seq_length=$PROMPT_CAP policy.generation.max_new_tokens=$MAXLEN \
  "${RES[@]}" "${PAR[@]}" "${LANE_ARGS[@]}" "${USER_ARGS[@]}" > "$OUT/run.log" 2>&1
rc=$?
echo "[ab] run exit=$rc"
python $W/research/ab/summarize_ab.py --check "$OUT"
src=$?
exit $(( rc != 0 ? rc : src ))
