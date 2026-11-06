#!/bin/bash
# Exactness gate, 1 node (4 GPUs), Super-L8 TP2 EP4, CHECKS=1 (bitwise mismatch + law test of every output token):
#   bash jobs/gate.sh xpr [K]   PRODUCTION: positive arm at stress lr 1e-4 (must be EXACT every step, with continued rows),
#                               then the carry_unverified negative (must FAIL the sentinel / mismatch / law from step 1)
#   bash jobs/gate.sh wvs [K]   PRODUCTION (verify at the wake): positive arm (16 prompts + 2 spares, MAXLEN 1024 so drafts
#                               are cut at the stand-down, stress lr 1e-4; must be EXACT every step), then carry_unverified
#                               (cut drafts trained unverified; must FAIL)
#   bash jobs/gate.sh xprs [K]  XPR-S: positive arm (SPARE=16: 32 admitted, 16 trained; lr 1e-6, most rows whole drafts;
#                               must be EXACT, spare groups dropped every step), then the spare_by_logp negative (keeps
#                               the pools with the highest draft log-prob: a selection on content; must FAIL the law)
#                               16 steps: the bias of that selection is small per step; at 6 steps its cumulative
#                               first-token law p reached 1e-5 in one run and only 4e-4 in another (alpha 1e-4)
#   bash jobs/gate.sh vg [K]    VG (alternative): positive arm (24 prompts x 8 drafts, M 1.4, stress lr 1e-4; must be
#                               EXACT), then the vg_by_logp negative (drops each group's lowest first-token log-prob
#                               output: a selection on content; must FAIL the first-token law)
#   bash jobs/gate.sh isspr [K] IS-SPR: positive arm (16 prompts, lr 1e-6; every step's rescore audit EXACT, sentinel 0),
#                               then isspr_stale_kv (paused rows resume on stale state; must FAIL the sentinel)
#   bash jobs/gate.sh xs        XS: positive arm, then the skip-reshard negative (engine keeps theta_0; must FAIL)
# Every arm ends with lib/check_run.py (negatives: PASS means "failed as it must").
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source $COLO_SCRIPTS/lib/presets.sh
WHICH=${1:?wvs|xprs|xpr|vg|isspr|xs}; KK=${2:-$COLO_PROD_K}; S=${GATE_ID:-$(date +%m%d-%H%M)}
STRESS="policy.megatron_cfg.optimizer.lr=1e-4 policy.megatron_cfg.optimizer.min_lr=1e-4 policy.megatron_cfg.scheduler.lr_warmup_init=1e-4"
rc_all=0
for pol in pos neg; do
  if [ $WHICH = xprs ]; then
    TAG=gate-xprs$KK-$pol-$S
    ( LR=1e-6; export STEPS=${STEPS:-16} CHECKS=1 RUN_TIMEOUT=3000 SPARE=16 MAX_REQ=512
      export XA_EXTRA="policy.megatron_cfg.optimizer.lr=$LR policy.megatron_cfg.optimizer.min_lr=$LR policy.megatron_cfg.scheduler.lr_warmup_init=$LR"
      [ $pol = neg ] && export NEG=spare_by_logp
      preset xprs l8 $KK || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  elif [ $WHICH = wvs ]; then
    TAG=gate-wvs$KK-$pol-$S
    ( export STEPS=${STEPS:-6} CHECKS=1 RUN_TIMEOUT=2700 MAXLEN=${MAXLEN:-1024} XA_EXTRA="$STRESS"
      [ $pol = neg ] && export NEG=carry_unverified
      preset wvs l8 $KK || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  elif [ $WHICH = isspr ]; then
    TAG=gate-isspr$KK-$pol-$S
    ( LR=1e-6; export STEPS=${STEPS:-6} CHECKS=1 RUN_TIMEOUT=2700
      export XA_EXTRA="policy.megatron_cfg.optimizer.lr=$LR policy.megatron_cfg.optimizer.min_lr=$LR policy.megatron_cfg.scheduler.lr_warmup_init=$LR"
      [ $pol = neg ] && export NEG=isspr_stale_kv
      preset isspr l8 $KK || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  elif [ $WHICH = vg ]; then
    TAG=gate-vg$KK-$pol-$S
    ( export STEPS=${STEPS:-6} CHECKS=1 RUN_TIMEOUT=2700 XA_EXTRA="$STRESS"; [ $pol = neg ] && export NEG=vg_by_logp
      preset vg l8 $KK || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  elif [ $WHICH = xpr ]; then
    TAG=gate-xpr$KK-$pol-$S
    ( export STEPS=${STEPS:-6} CHECKS=1 RUN_TIMEOUT=2700 XA_EXTRA="$STRESS"; [ $pol = neg ] && export NEG=carry_unverified
      preset xpr l8 $KK || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  else
    TAG=gate-xs-$pol-$S
    ( export STEPS=${STEPS:-4} CHECKS=1 RUN_TIMEOUT=2400; [ $pol = neg ] && export SKIP_RESHARD=1
      preset xs l8 || exit 2; export TAG
      bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/colo_arm.sh )
  fi
  rc=$?; echo "[gate] $WHICH $pol exit=$rc"; [ $rc = 0 ] || rc_all=$rc
done
echo "[gate] verdicts:"; G=gate-$WHICH$( [ $WHICH = xs ] || echo $KK ); python3 $COLO_SCRIPTS/lib/check_run.py $COLO_RUNS/$G-pos-$S $COLO_RUNS/$G-neg-$S
exit $(( rc_all || $? ))
