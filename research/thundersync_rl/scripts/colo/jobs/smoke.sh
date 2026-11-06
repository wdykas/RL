#!/bin/bash
# 1-node smoke (4 GPUs, ~30 min): proves the environment, both launchers and the checks work for you.
#   1. config dry runs (no GPU work) of the 4-node Super-L22 arms: production wvs, xprs, xpr, vg, xs, sprbi, spr (K1)
#   2. PRODUCTION arm (wvs, lag K) on Super-L8, 4 steps, CHECKS=1: must be bitwise exact, routers frozen
#   3. stock SPR K1 and SPR-BI K1 on Super-L8, 4 steps: routers frozen (per-step probes)
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source $COLO_SCRIPTS/lib/presets.sh
S=${SMOKE_ID:-$(date +%m%d-%H%M)}; KK=$COLO_PROD_K; rc_all=0
echo "[smoke] share $COLO_SHARE, RL $(git -C $COLO_RL rev-parse --short HEAD), stock $(git -C $COLO_RL_STOCK rev-parse --short HEAD), out $COLO_OUT, job ${SLURM_JOB_ID:-none} on $(hostname)"
/opt/nemo_rl_venv/bin/python -c "import urllib3.exceptions, typing_extensions, requests, ray, torch; print('[smoke] container venv OK, torch', torch.__version__)" \
  || { echo "[smoke] FAIL: container venv imports broken (check COLO_CONTAINER_MOUNTS)"; exit 3; }
for a in wvs xprs xpr vg xs sprbi spr; do
  ( export PREFLIGHT=1; preset $a sl22 $KK || exit 2; export TAG=smoke-pre-$a-sl22-$S; bash $COLO_SCRIPTS/lib/$ARM_LAUNCHER ) > $COLO_RUNS/smoke-pre-$a-sl22-$S.out 2>&1
  rc=$?; echo "[smoke] 4-node $a preflight exit=$rc: $(grep -h 'preflight only\|PREFLIGHT FAIL\|REFUSED' $COLO_RUNS/smoke-pre-$a-sl22-$S.out | head -1)"
  [ $rc = 0 ] || rc_all=$rc
done
for a in wvs spr sprbi; do
  TAG=smoke-$a$KK-l8-$S
  ( [ $a = wvs ] && export CHECKS=1; preset $a l8 $KK || exit 2; export TAG
    bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3; bash $COLO_SCRIPTS/lib/$ARM_LAUNCHER ) > $COLO_RUNS/$TAG.out 2>&1
  rc=$?; echo "[smoke] $a L8 exit=$rc"; tail -4 $COLO_RUNS/$TAG.out | cut -c1-300
  [ $rc = 0 ] || rc_all=$rc
done
echo "[smoke] verdicts:"; python3 $COLO_SCRIPTS/lib/check_run.py $COLO_RUNS/smoke-wvs$KK-l8-$S $COLO_RUNS/smoke-spr$KK-l8-$S $COLO_RUNS/smoke-sprbi$KK-l8-$S || rc_all=5
echo "[smoke] $( [ $rc_all = 0 ] && echo PASS || echo "FAIL rc=$rc_all" )"
exit $rc_all
