#!/bin/bash
# Generic colo-kit batch script. Prefer submit.sh, which fills in account / partition / nodes / time / output.
# Direct use (account and partition are site-specific: pass them):
#   sbatch -A <account> -p <partition> -N <nodes> -t <time> -o <COLO_OUT>/slurm/%x-%j.out -J <name> \
#     scripts/sbatch/colo.sbatch <job> [args...]
# <job> = smoke | gate | arm | pair  (scripts/jobs/<job>.sh). 1 node: the job runs directly in the container.
# >1 node: lib/ray_cluster.sh brings up Ray on every node and runs the job on the head (RUN name = $COLO_RUN).
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
set -u
JOBNAME=${1:?job: smoke | gate | arm | pair}; shift
: "${COLO_SCRIPTS:?COLO_SCRIPTS unset: submit with submit.sh, or source env.sh first (sbatch exports your env)}"
source $COLO_SCRIPTS/env.sh
NN=${SLURM_JOB_NUM_NODES:-1}
echo "[colo.sbatch] job $SLURM_JOB_ID $JOBNAME $* on $NN node(s) $(hostname); image $COLO_IMAGE; out $COLO_OUT"
# Pyxis / enroot container flags. COLO_CONTAINER_MOUNTS must mount every filesystem holding the trees, $COLO_OUT and models.
SRUN=(srun --ntasks-per-node=1 --container-image=$COLO_IMAGE --container-mounts=$COLO_CONTAINER_MOUNTS --container-name=colo-$SLURM_JOB_ID)
if [ "$NN" = 1 ]; then
  "${SRUN[@]}" bash $COLO_SCRIPTS/jobs/$JOBNAME.sh "$@"
else
  RUN=${COLO_RUN:?COLO_RUN (run name) unset}
  "${SRUN[@]}" bash $COLO_SCRIPTS/lib/ray_cluster.sh $RUN bash $COLO_SCRIPTS/jobs/$JOBNAME.sh "$@"
fi
