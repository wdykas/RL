#!/bin/bash
# One arm (inside the container; under lib/ray_cluster.sh when multi-node).
#   bash jobs/arm.sh <wvs|xprs|xpr|vg|xs|spr|sprbi|cs> <l8|sl22|nano|ul27> [K] [TAG]       (extra knobs: export them, e.g. STEPS=6 CHECKS=1)
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source $COLO_SCRIPTS/lib/presets.sh
ARM=${1:?arm}; M=${2:?model}; KK=${3:-$COLO_PROD_K}
preset $ARM $M $KK || exit 2
case $ARM in xs|cs) KT="" ;; *) KT=$KK ;; esac
export TAG=${4:-${TAG:-colo-$ARM$KT-$M-$(date +%m%d-%H%M)}}
bash $COLO_SCRIPTS/lib/keep_prev_run.sh $COLO_RUNS $TAG || exit 3
bash $COLO_SCRIPTS/lib/$ARM_LAUNCHER
