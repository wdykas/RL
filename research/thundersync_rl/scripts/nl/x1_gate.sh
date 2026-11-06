#!/bin/bash
# X1 gate of the nolanes (NL2) mode: ONE 4-GPU slurm node, arms in sequence (patterns from
# $TS_WORK/research/ultra/uf_1node.sh: per-arm process group, run.log stall detection + py-spy dump, teardown,
# free GPUs required before the next arm).
#   [TREE=<tree>] [MODEL=...] [NEGS="wrong_tag tag_off_by_one content_m key_reuse engine_q_genlp"] [STEPS=4 NEG_STEPS=STEPS]
#   [LAG=2 PROMPTS=16 G=8 MAXLEN=512] [RUNS=runs_nl] [TMO=45 (min per arm)] [STALL_MIN=15] [XA=... XE=...]
#     bash x1_gate.sh [all|pf|pos|neg]                (all = pf, then pos, then every NEGS arm, then the summary)
#   pf : bash -n of the launchers, import check of thundersync_rl.nolanes, NL_DRY preflight (positive + one negative
#        config); the job stops here on any failure (no GPU time spent on a broken config)
#   pos: Super-L8, learner TP2 EP2 (2 GPUs) + engine TP2 EP2 (2 GPUs), BI numerics, CHECKS=1: must pass every check
#   pos_base: the pos arm on nl_base.yaml (IS loss, train_overlap off: the path no other arm runs; R35 item 5)
#   prod: the PRODUCTION shape (R37 (iii)): the config's defaults, LAG=1, no admission hold and no step sleep, CHECKS;
#     must pass every exactness check; exempt (labelled, in nl_summary) only from the mixed-row requirement, which no
#     production lag-1 draft meets by construction. The mixed path stays covered by the lag-2 hold arms.
#   prod_stress: the prod shape under stress lr (R38 (c)): the lag-1 continuation path; must be EXACT with continued > 0.
#   STEPS >= 4 is required: with the hold, mixed rows first appear at step 3.
#   ADMIT_HOLD=1 (default): THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS=1 in every arm (gate tool, refused without CHECKS):
#     new drafts are admitted only at the optimizer step (~1 s before the refit), so they prefill and start decoding
#     at theta_k and decode across the refit: the mixed-version path (incl. rejected / continued mixed rows) is
#     exercised with admit_after_refit on (R35 2(e)), at any MAXLEN and without an engine-step sleep.
#   neg: one arm per NEGS entry (THUNDERSYNC_NL_NEG=<name>, CHECKS=1): each must FAIL a check (nl_summary.py decides)
# Logs: $TS_WORK/research/nolanes2/x1/x1_<arm>_<job>.log (gate), x1_<tag>_<job>.arm.log (one per arm);
# runs: $TS_WORK/$RUNS/x1-<pos|neg_<name>>-<job>/.
set -u
P=${TS_ROOT:?set TS_ROOT to the workspace root holding the RL / Megatron checkouts and TS_WORK}
W=$P/thundersync_work
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
TREE=${TREE:-$(cd "$HERE/../../../.." && pwd)}
ARM_SH=$TREE/research/thundersync_rl/scripts/nl/run_nl_arm.sh
SUMMARY=$TREE/research/thundersync_rl/scripts/nl/nl_summary.py
H=$W/research/nolanes2/x1; mkdir -p $H
RUNS=${RUNS:-runs_nl}; mkdir -p $W/$RUNS
JOB=${SLURM_JOB_ID:-local}
ARMX=${1:-all}
NEGS=${NEGS:-wrong_tag tag_off_by_one content_m content_m_edv defer_no_decide key_reuse engine_q_genlp cf_engine_m grad_order drop_late_group}
STEPS=${STEPS:-4}; NEG_STEPS=${NEG_STEPS:-$STEPS}
LOG=$H/x1_${ARMX}_${JOB}.log; exec >> $LOG 2>&1
echo "[x1] host $(hostname) job $JOB arm $ARMX TREE $TREE @ $(git -C $TREE rev-parse --short HEAD) dirty=$(git -C $TREE status --porcelain --untracked-files=no | wc -l) NEGS=[$NEGS] $(date +%T)"
# Container venv sanity, shared-filesystem caches (never /home), GPU-idle watchdog (local-bg on this script) in slurm jobs.
source $W/research/mvd/slurm/envguard.sh || exit 3
XE_BASE="${XE:-}"
if (( ${STEPS:-4} < 4 )) && [ "${ALLOW_SHORT:-0}" != 1 ]; then
  echo "[x1] REFUSED: STEPS=${STEPS} < 4 (the hold's mixed rows first appear at step 3; ALLOW_SHORT=1 for a smoke run)"; exit 3
fi
ADMIT_HOLD_XE=""
[ "${ADMIT_HOLD:-1}" = 1 ] && ADMIT_HOLD_XE=" THUNDERSYNC_NL_DEBUG_ADMIT_AFTER_TAILS=1"
# The engine-step sleep is no longer needed for coverage (the hold now lasts to the optimizer step); STEP_SLEEP_MS>0
# adds it back for experiments.
(( ${STEP_SLEEP_MS:-0} > 0 )) && ADMIT_HOLD_XE="$ADMIT_HOLD_XE THUNDERSYNC_NL_DEBUG_STEP_SLEEP_MS=$STEP_SLEEP_MS"
# Shared arm settings (every arm: same model, layout, shape; only CHECKS / NEG / STEPS / TAG differ).
COMMON=(TREE=$TREE RUNS=$RUNS MODEL=${MODEL:-$W/models/nemotron3-super-L8} TP=2 EP=2 GEN_TP=2 GEN_EP=2 GPS=2 GEN_GPUS=2
        PROMPTS=${PROMPTS:-16} G=${G:-8} MAXLEN=${MAXLEN:-512} LAG=${LAG:-2} KD=${KD:-1.5} PARITY=1 WD=0
        "XA=${XA:-}" "XE=${XE:-}${ADMIT_HOLD_XE}")
RUN_DIRS=()
ARM_SPANS=()  # "name start end" (epoch s) per arm, for the utilization summary

gpus_free() {  # true when no compute process holds any GPU of this node
  [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' \n')" ]
}
teardown() {  # PGID RC STALLED: end the arm's process group; on failure / stall also its local Ray cluster (plain
  # `ray stop`, never --force, never pattern kills); then require free GPUs before another arm may start.
  local pg=$1 rc=$2 st=$3
  kill -0 -$pg 2>/dev/null && { kill -TERM -$pg 2>/dev/null; sleep 20; kill -KILL -$pg 2>/dev/null; }
  if [ "$rc" != 0 ] || [ "$st" = 1 ]; then
    local rb; rb=$(command -v ray || echo /opt/nemo_rl_venv/bin/ray)
    $rb stop > /dev/null 2>&1 || true
  fi
  for i in $(seq 1 12); do gpus_free && return 0; sleep 10; done
  echo "[x1] FATAL: GPUs still held after the arm ended:"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  return 1
}
run1() {  # NAME CHECKS STEPS NEG [EXTRA_XA] [EXTRA_ENV] -> runs one arm; RC=125 on a stall (STALL_MIN without a new
  # run.log line). EXTRA_ENV: "K=V ..." exported after the shared settings (arm-specific shape).
  local NAME=$1 CK=$2 ST=$3 NG=$4 EXA=${5:-} EXE=${6:-}
  local T=x1-$NAME-$JOB AL=$H/x1_${NAME}_${JOB}.arm.log RL=$W/$RUNS/x1-$NAME-$JOB/run.log
  local APID STALLED=0 now m t0 RC
  gpus_free || { echo "[x1] FATAL: GPUs busy before $T"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 4; }
  echo "=== $T $(date +%T) CHECKS=$CK STEPS=$ST NEG=[$NG] (arm log $AL)"
  set -m   # the arm gets its own process group (PGID = APID): teardown signals exactly this arm
  ( for kv in "${COMMON[@]}"; do export "$kv"; done
    # prod: production shape -- lag 1, no gate-only admission hold / step sleep
    # prod / prod_stress: production shape -- lag 1, no gate-only admission hold / step sleep, the config's K
    if [ "$NAME" = prod ] || [ "$NAME" = prod_stress ]; then export LAG=1 XE="$XE_BASE"; unset KD; fi
    # EDV arms with forced surprises (gate tool, CHECKS only): every finished draft with eta_g <= 0.95 is deferred
    if [ "$NAME" = edv_stress ] || [ "$NAME" = neg_defer_no_decide ] || [ "$NAME" = neg_content_m_edv ]; then export XE="${XE:-} THUNDERSYNC_NL_DEBUG_EDV_E=0.95"; fi
    export TAG=$T CHECKS=$CK STEPS=$ST CUDA_VISIBLE_DEVICES=0,1,2,3 RAY_ADDRESS=local PYTHONDONTWRITEBYTECODE=1
    [ -n "$EXA" ] && export XA="${XA:-} $EXA"
    for kv in $EXE; do export "$kv"; done
    if [ -n "$NG" ]; then export NEG=$NG; else unset NEG; fi
    unset PREFLIGHT NL_DRY
    exec timeout ${TMO:-45}m bash $ARM_SH ) < /dev/null > $AL 2>&1 &
  APID=$!
  set +m
  t0=$(date +%s)
  while kill -0 $APID 2>/dev/null; do
    sleep 30
    now=$(date +%s); m=$(stat -c %Y $RL 2>/dev/null || echo $t0)
    if (( now - m > ${STALL_MIN:-15} * 60 )); then
      echo "[x1] STALL $T: no new run.log line for ${STALL_MIN:-15} min ($(date +%T)); py-spy dump of every process on the node"
      bash $W/research/verify0attr/pyspy_dump_job.sh $H/x1_${NAME}_pyspy_${JOB}.txt
      STALLED=1; break
    fi
  done
  [ $STALLED = 1 ] && { kill -TERM -$APID 2>/dev/null; sleep 20; kill -KILL -$APID 2>/dev/null; }
  wait $APID; RC=$?
  [ $STALLED = 1 ] && RC=125
  grep -a -E "^\[nl\]|Error|error:|Traceback" $AL | cut -c1-300 | tail -30
  grep -a -h -E "\[nolanes\]|engine q check|law check|mismatch check|FAIL|NEGATIVE" $RL 2>/dev/null | cut -c1-240 | tail -12
  ARM_SPANS+=("$NAME $t0 $(date +%s)")
  echo "=== $T rc=$RC stalled=$STALLED $(date +%T)"
  echo "$RC" > $W/$RUNS/$T/rc 2>/dev/null
  RUN_DIRS+=($W/$RUNS/$T)
  teardown $APID $RC $STALLED || exit 4
}
preflight() {  # no GPUs: syntax, import, config dry run (positive + one negative); non-zero on any failure
  local ok=0
  for f in $ARM_SH $TREE/research/thundersync_rl/scripts/nl/x1_gate.sh; do
    bash -n $f && echo "[pf] bash -n OK $f" || { echo "[pf] bash -n FAIL $f"; ok=1; }
  done
  echo "=== import check $(date +%T)"
  ( export RL_ROOT_OVERRIDE=$TREE TS_PYTHONPATH_PREPEND=$W/overlay_up; source $W/env_sbatch.sh bi; cd $TREE && \
    timeout 10m python -c "import thundersync_rl.nolanes as n, thundersync_rl.nl_parity_env as e; assert hasattr(n, 'validate_config') and hasattr(n, 'configure_driver'); print('[pf] import ok', n.MODE, n.NEGATIVES, e.FQN)" 2>&1 | tail -3 ) \
    | tee /dev/stderr | grep -q "\[pf\] import ok" || { echo "[pf] import check FAIL"; ok=1; }
  echo "=== NL_DRY $(date +%T)"
  local ng1; ng1=$(echo $NEGS | awk '{print $1}')
  for spec in "pos::" "neg_$ng1:$ng1"; do
    local nm=${spec%%:*} ng=${spec#*:}; ng=${ng#:}
    ( for kv in "${COMMON[@]}"; do export "$kv"; done
      export TAG=x1-$nm-$JOB CHECKS=1 STEPS=$STEPS PREFLIGHT=1
      if [ -n "$ng" ]; then export NEG=$ng; else unset NEG; fi
      timeout 20m bash $ARM_SH ) 2>&1 | tail -12 | tee $H/.pf_${JOB}_$$.txt
    grep -q "preflight only: OK" $H/.pf_${JOB}_$$.txt && echo "[pf] NL_DRY OK $nm" || { echo "[pf] NL_DRY FAIL $nm"; ok=1; }
    rm -f $H/.pf_${JOB}_$$.txt
  done
  return $ok
}
util_summary() {  # mean GPU utilization (runs_wd/<job>/<host>.util.log, per-GPU per-minute means) per arm + whole job
  local U=$W/runs_wd/$JOB/$(hostname).util.log
  [ -f "$U" ] || { echo "[x1 util] no $U"; return 0; }
  local sp; for sp in "${ARM_SPANS[@]}"; do
    awk -v n="${sp%% *}" -v a="$(echo $sp | awk '{print $2}')" -v b="$(echo $sp | awk '{print $3}')" '
      { t=substr($1,3)+0; if (t<a || t>b) next; for(i=2;i<=NF;i++){split($i,x,"[:%]"); s+=x[2]; c++} }
      END { printf "[x1 util] arm %s: mean GPU util %.0f%% over %.0f min\n", n, (c?s/c:0), (b-a)/60 }' "$U"
  done
  awk '{for(i=2;i<=NF;i++){split($i,x,"[:%]"); s+=x[2]; c++}} END{printf "[x1 util] job %s: mean GPU util %.0f%% over %d samples\n", J, (c?s/c:0), NR}' J=$JOB "$U"
}
summary() {
  util_summary
  (( ${#RUN_DIRS[@]} )) || return 0
  if [ -f $SUMMARY ]; then
    echo "=== summary $(date +%T)"
    ( export RL_ROOT_OVERRIDE=$TREE TS_PYTHONPATH_PREPEND=$W/overlay_up; source $W/env_sbatch.sh bi; cd $TREE && \
      python research/thundersync_rl/scripts/nl/nl_summary.py "${RUN_DIRS[@]}" 2>&1 | tail -60 )
  else
    echo "[x1] no $SUMMARY yet; run dirs: ${RUN_DIRS[*]}"
  fi
}
# content_m (E3) needs power to fail: K 3, 32 prompts, fixed envelope M 4 (BUGS.md #5; at the default shape the
# cumulative first-token law does not reject it within 5 steps; measured at 1fa5121b).
STRESS_XA="policy.megatron_cfg.optimizer.lr=1e-4 policy.megatron_cfg.optimizer.min_lr=1e-4 policy.megatron_cfg.scheduler.lr_warmup_init=1e-4"
# Mild stress (5e-6 from step 0): some rows rejected / continued, most accepted (content_m keeps its selection power).
MILD_XA="policy.megatron_cfg.optimizer.lr=5e-6 policy.megatron_cfg.optimizer.min_lr=5e-6 policy.megatron_cfg.scheduler.lr_warmup_init=5e-6"
# content_m (E3, EDV off): 1e-5 from step 0, between mild and stress: enough mixed continued rows, selection power kept.
CM_XA="policy.megatron_cfg.optimizer.lr=1e-5 policy.megatron_cfg.optimizer.min_lr=1e-5 policy.megatron_cfg.scheduler.lr_warmup_init=1e-5"
BASE_CFG=research/thundersync_rl/configs/nl/nl_base.yaml
# Negatives whose check fails on step 0 or 1 (no mixed-row coverage needed: a real check fails first) run
# FAST_NEG_STEPS (2) steps: a negative only has to fail once (fast gate, 11-02).
FS=${FAST_NEG_STEPS:-2}
negs() {
  for ng in $NEGS; do
    # wrong_tag / tag_off_by_one / content_m need mixed-version drafts that are rejected and continued (R35 2(e)):
    # the admission hold makes the mixed rows, stress lr the continuations.
    # content_m tests E3 on the verifier's walk: EDV off (deferred rows, eta_g <= e, almost never reach the BOT root,
    # so the content-chosen envelope moves few outcomes and the law check loses its power; : 127 rows touched).
    # Coverage (mixed-version drafts rejected AND continued) must not sit at an edge: lr 1e-5 from step 0 and 6 steps
    # (runs at 5e-6 x 4 steps: mixed_continued 0 once in 4 gates; 0-3 per run).
    if [ "$ng" = content_m ]; then run1 neg_$ng 1 6 $ng "++thundersync.null_bv_fixed_m=4.0 ++thundersync.edv_threshold=null $CM_XA" "KD=3 PROMPTS=32"
    elif [ "$ng" = wrong_tag ] || [ "$ng" = tag_off_by_one ]; then run1 neg_$ng 1 $NEG_STEPS $ng "$STRESS_XA"
    # key_reuse (E4) needs continuations: lr stress (1e-4 from step 0) rejects rows (the expert-bias stress of
    # d8812031 is refused under train_overlap, R31 (c)).
    elif [ "$ng" = key_reuse ]; then run1 neg_$ng 1 $FS $ng "$STRESS_XA"
    # grad_order (R31): train overlap with a plain fp32 arrival-order accumulator; the in-loop order check must fail.
    elif [ "$ng" = grad_order ]; then run1 neg_$ng 1 $FS $ng "++thundersync.train_overlap=true $STRESS_XA"
    # drop_late_group (P5, finalize overlap): the last-finishing group is not trained (the first trains twice in its
    # place); the trained-once check must fail. Stress lr so tails finish out of order.
    elif [ "$ng" = drop_late_group ]; then
      run1 neg_$ng 1 $FS $ng "++thundersync.train_overlap=true $STRESS_XA"
    # defer_no_decide (EDV): the loss accepts every deferred row; with e forced to 0.95 and stress lr, deferred rows
    # that are not whole are committed: the replay check (and the law check) must fail.
    # content_m_edv: E3 on the EDV path (EDV on, e forced to 0.95): the content-chosen envelope reaches deferred rows
    # (their loss-side b_g at M_j) and EDV re-walks; nl_summary requires >= 20 such rows (non-vacuous).
    elif [ "$ng" = content_m_edv ]; then run1 neg_$ng 1 $NEG_STEPS content_m "++thundersync.null_bv_fixed_m=4.0 $MILD_XA" "KD=3 PROMPTS=32"
    elif [ "$ng" = defer_no_decide ]; then run1 neg_$ng 1 $FS $ng "++thundersync.edv_threshold=0.5 $STRESS_XA"
    # mtd_accept_all (tail_mtd): proposals accepted without sample-and-match; continuations (stress lr) with a fixed
    # width; the train-lp / mismatch / law checks must fail.
    elif [ "$ng" = mtd_accept_all ]; then
      run1 neg_$ng 1 $FS $ng "++thundersync.tail_mtd=4 $STRESS_XA" "XE=THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K=4${ADMIT_HOLD_XE}"
    else run1 neg_$ng 1 $FS $ng; fi  # engine_q_genlp, cf_engine_m: the mismatch check fails on step 0
  done
}

case $ARMX in
  pf)  preflight || { echo "[x1] PREFLIGHT FAIL"; exit 3; } ;;
  pos) run1 pos 1 $STEPS ""; summary ;;
  pos_base) run1 pos_base 1 $STEPS "" "" "CFG_REL=$BASE_CFG"; summary ;;
  prod) run1 prod 1 $STEPS ""; summary ;;
  prod_stress) run1 prod_stress 1 $STEPS "" "$STRESS_XA"; summary ;;
  # Stress positive: lr 1e-4 from step 0 (many rejections / continuations; tails finish out of order).
  stress) run1 stress 1 $STEPS "" "$STRESS_XA"; summary ;;
  # Multi-token tail decode (tail_mtd K 4): the stress positive with every continuation row at a fixed width 4 (the
  # truncated model's text gives no acceptance, so the width rule alone would choose 0); must be EXACT with proposals.
  mtd) run1 mtd 1 $STEPS "" "++thundersync.tail_mtd=4 $STRESS_XA" "XE=THUNDERSYNC_TAIL_MTD_DEBUG_FORCE_K=4${ADMIT_HOLD_XE}"; summary ;;
  # ACCEPTANCE SANITY (R30): no weight update (lr 0): p == q bitwise at every version, so no draft may be rejected.
  sanity) run1 sanity 1 $STEPS "" "policy.megatron_cfg.optimizer.lr=0.0 policy.megatron_cfg.optimizer.min_lr=0.0 policy.megatron_cfg.scheduler.lr_warmup_init=0.0"; summary ;;
  neg) negs; summary ;;
  # EDV (DESIGN §22.9): deferred rows decided by the training forward, replayed by the verifier (CHECKS).
  edv) run1 edv 1 $STEPS "" "++thundersync.edv_threshold=0.5"; summary ;;
  # EDV surprise path: e forced to 0.95 and stress lr, so deferred rows are not whole: dropped captures, re-walks,
  # retrained groups; must be EXACT with surprises > 0.
  edv_stress) run1 edv_stress 1 $STEPS "" "++thundersync.edv_threshold=0.5 $STRESS_XA"; summary ;;
  all) preflight || { echo "[x1] PREFLIGHT FAIL: no GPU arm started"; exit 3; }
       run1 pos 1 $STEPS ""
       run1 pos_base 1 $STEPS "" "" "CFG_REL=$BASE_CFG"
       run1 prod 1 $STEPS ""
       run1 prod_stress 1 $STEPS "" "$STRESS_XA"
       run1 stress 1 $STEPS "" "$STRESS_XA"
       run1 sanity 1 $STEPS "" "policy.megatron_cfg.optimizer.lr=0.0 policy.megatron_cfg.optimizer.min_lr=0.0 policy.megatron_cfg.scheduler.lr_warmup_init=0.0"
       negs
       summary ;;
  *) echo "arm all|pf|pos|pos_base|prod|prod_stress|stress|sanity|mtd|neg"; exit 1 ;;
esac
echo "[x1] done $(date +%T)"
