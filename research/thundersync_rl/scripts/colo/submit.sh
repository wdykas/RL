#!/bin/bash
# Submit a colo-kit job with plain sbatch (login node). DRY=1 prints the sbatch command instead of submitting.
#   bash $COLO_SCRIPTS/submit.sh smoke                                   1 node,  ~30 min  
#   bash $COLO_SCRIPTS/submit.sh gate wvs [K]                            1 node,  ~45 min  exactness gate, PRODUCTION (verify at the wake)
#   bash $COLO_SCRIPTS/submit.sh gate xprs [K]                           1 node,  ~60 min  exactness gate, XPR-S (stand-down verify)
#   bash $COLO_SCRIPTS/submit.sh gate xpr [K]                            1 node,  ~45 min  exactness gate, XPR (no spares)
#   bash $COLO_SCRIPTS/submit.sh gate isspr [K]                          1 node,  ~45 min  exactness gate, IS-SPR
#   bash $COLO_SCRIPTS/submit.sh gate vg [K]                             1 node,  ~45 min  exactness gate, VG (alternative)
#   bash $COLO_SCRIPTS/submit.sh gate xs                                 1 node,  ~30 min  exactness gate, XS
#   bash $COLO_SCRIPTS/submit.sh arm <wvs|xprs|xpr|vg|xs|spr|sprbi|cs> <sl22|nano|ul27|l8> [K]   one arm (4 nodes; ul27: 16; l8: 1)
#   bash $COLO_SCRIPTS/submit.sh pair <sl22|nano> <K> <order> <pair id>  arms back to back in one allocation, e.g. the
#                                                                  3-arm comparison: pair sl22 1 sprbi,wvs,spr a
# Account / partition: COLO_ACCOUNT / COLO_PARTITION (env.sh). TIME=hh:mm:ss overrides the wall time.
# Extra knobs for the arm (STEPS=..., CHECKS=1, XA_EXTRA='hydra overrides', ...) are exported to the job by sbatch.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
[ -f "$COLO_PINS" ] || { echo "no $COLO_PINS: run bash $COLO_SCRIPTS/setup.sh first" >&2; exit 2; }
J=${1:?smoke | gate | arm | pair}; shift
case $J in
  smoke) N=1; T=01:00:00; NAME=colo-smoke ;;
  gate)  N=1; T=01:30:00; NAME=colo-gate-${1:?wvs|xprs|xpr|vg|isspr|xs} ;;
  arm)   M=${2:?model}; case $M in l8) N=1;; ul27) N=16;; *) N=4;; esac; T=01:40:00
         NAME=colo-$1${3:-}-$M; export COLO_RUN=${TAG:-$NAME-$(date +%m%d-%H%M%S)}; export TAG=$COLO_RUN
         set -- "$1" "$M" "${3:-$COLO_PROD_K}" "$TAG" ;;
  pair)  M=${1:?model}; N=4; T=03:00:00; NAME=colo-pair-$M-k${2:?K}-${4:?pair id}; export COLO_RUN=$NAME ;;
  *) echo "job must be smoke | gate | arm | pair" >&2; exit 2 ;;
esac
T=${TIME:-$T}
mkdir -p "$COLO_OUT/slurm"
SEG=(); (( N > 1 )) && [ "${COLO_SEGMENT:-1}" = 1 ] && SEG=(--segment=$N)   # whole NVLink-domain block (GB200 NVL72); COLO_SEGMENT=0 drops it
SITE=(); [ -n "${COLO_ACCOUNT:-}" ] && SITE+=(-A "$COLO_ACCOUNT"); [ -n "${COLO_PARTITION:-}" ] && SITE+=(-p "$COLO_PARTITION")
CMD=(sbatch --parsable "${SITE[@]}" -N $N "${SEG[@]}" -t $T -J $NAME
     --chdir="$COLO_OUT" -o "$COLO_OUT/slurm/%x-%j.out" --export=ALL "$COLO_SCRIPTS/sbatch/colo.sbatch" $J "$@")
if [ "${DRY:-0}" = 1 ]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi
id=$("${CMD[@]}")
echo "submitted $id ($NAME, $N node(s), $T): log $COLO_OUT/slurm/$NAME-$id.out${COLO_RUN:+, runs $COLO_RUNS/$COLO_RUN*}"
