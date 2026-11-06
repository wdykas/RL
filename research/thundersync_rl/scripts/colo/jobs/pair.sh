#!/bin/bash
# Paired A/B in ONE allocation: the arms run back to back on the same nodes / Ray cluster, in the given order.
# Submit twice with the order swapped ("both orders") and score all four runs together.
#   bash jobs/pair.sh <model> <K> <order, e.g. spr,xpr> <pair id>
# Arm run tags: colo-<arm><K>-<model>-<pair id>-<position>; progress lines go to stdout ($COLO_RUNS/<PAIR>.out under
# ray_cluster.sh) whenever the running arm's run.log grows, so a hung arm still trips the stall watchdog.
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source $COLO_SCRIPTS/lib/presets.sh
M=${1:?model}; KK=${2:?K}; ORDER=${3:?order}; ID=${4:?pair id}
echo "[pair] $M K=$KK order=$ORDER id=$ID start $(date -Is)"
i=0; rc_all=0
for a in ${ORDER//,/ }; do
  i=$((i + 1))
  case $a in xs|cs) KT="" ;; *) KT=$KK ;; esac
  TAG=colo-$a$KT-$M-$ID-$i
  ( preset $a $M $KK || exit 2; export TAG
    bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3
    exec bash $COLO_SCRIPTS/lib/$ARM_LAUNCHER ) > $COLO_RUNS/$TAG.out 2>&1 &
  P=$!; last=""
  echo "[pair] arm $TAG start $(date -Is)"
  while kill -0 $P 2>/dev/null; do
    sleep 60
    s=$(stat -L -c %s "$COLO_RUNS/$TAG/run.log" 2>/dev/null || echo 0)
    if [ "$s" != "$last" ]; then echo "[pair] $TAG run.log $s B $(date +%T)"; last=$s; fi
  done
  wait $P; rc=$?
  echo "[pair] arm $TAG exit=$rc $(date -Is)"; tail -3 $COLO_RUNS/$TAG.out
  [ $rc = 0 ] || rc_all=$rc
  sleep 20   # Ray actors of the finished driver release their GPUs
done
echo "[pair] done $(date -Is) rc=$rc_all"
exit $rc_all
