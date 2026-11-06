#!/bin/bash
# Multi-node Ray cluster inside a Slurm allocation, then one driver command on the head. Run ONCE PER NODE inside the
# container (srun --ntasks-per-node=1 --container-image=... -- bash ray_cluster.sh <RUN> <driver command...>).
# Same Ray env, ports, per-node resources, NUMA affinity
# file, stall watchdog, py-spy dumps on failure, graceful teardown).
#   - Head = SLURM_NODEID 0: starts the GCS, waits for 4 x nodes worker_units, runs the driver command with stdout/stderr
#     to $COLO_RUNS/<RUN>.out. Workers join and leave when <logdir>/ENDED appears (or the head is unreachable 4x).
#   - Every node runs the GPU-idle monitor (lib/gpu_idle_watchdog.sh); the driver (colo_arm.sh / stock_arm.sh) runs
#     the enforcer.
#   - Stall watchdog: no growth of $COLO_RUNS/<RUN>/run.log and $COLO_RUNS/<RUN>.out for BR_STALL_S (1080 s) -> py-spy
#     dump on all nodes, driver stopped (a NCCL hang spins GPUs at 100% and is invisible to the idle watchdog).
#   - OBJ_STORE_GB=<GiB>|auto (default auto): Ray object store per node = min(0.45 MemTotal, 0.85 /dev/shm, 0.3 cgroup, 300).
set -u
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
L=$COLO_SCRIPTS/lib
RUN=${1:?run name}; shift
[ $# -ge 1 ] || { echo "[ray_cluster] usage: ray_cluster.sh <RUN> <driver command...>"; exit 2; }
JOB=${SLURM_JOB_ID:?not in a slurm job}; NODEID=${SLURM_NODEID:-0}; NN=${SLURM_JOB_NUM_NODES:-1}
LOG_DIR=$COLO_RUNS/$RUN.jobs/$JOB-logs; mkdir -p "$LOG_DIR"
HOST=$(hostname)
exec > >(tee -a "$LOG_DIR/node-$(printf '%03d' $NODEID)-$HOST.log") 2>&1
ph() { echo "[NRL_PHASE][$(date '+%Y-%m-%dT%H:%M:%S%z')] node $NODEID $HOST: $*"; }
ph "start (job $JOB, $NN nodes, run $RUN, driver: $*)"
# Container venv sanity: the venv must import (submit with the container mounts in COLO_CONTAINER_MOUNTS).
/opt/nemo_rl_venv/bin/python -c "import urllib3.exceptions, typing_extensions, requests, ray, torch" \
  || { ph "ENV FAIL: container venv imports broken (COLO_CONTAINER_MOUNTS missing a mount?)"; touch "$LOG_DIR/ENDED"; exit 3; }
[ "${GPU_IDLE_WATCHDOG:-1}" = 0 ] || { export GPU_WD_DIR=$COLO_OUT/runs_wd/$JOB; mkdir -p $GPU_WD_DIR
  nohup setsid bash $L/gpu_idle_watchdog.sh monitor $$ > $GPU_WD_DIR/$HOST.monitor.log 2>&1 < /dev/null & }

export PATH=/opt/nemo_rl_venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/mpi/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/ucx/bin:/opt/amazon/efa/bin
export RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_DEDUP_LOGS=0 OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
unset UV_CACHE_DIR
if [[ $(ulimit -Hn) == unlimited ]] || [[ 65535 -lt $(ulimit -Hn) ]]; then ulimit -Sn 65535; fi
CPUS=$(nproc)
export RAY_gcs_server_rpc_server_thread_num=$CPUS RAY_num_server_call_thread=$(( CPUS / 2 > 0 ? CPUS / 2 : 1 ))
export RAY_gcs_server_rpc_client_thread_num=$CPUS RAY_gcs_max_active_rpcs_per_handler=$(( CPUS * 200 ))
export RAY_worker_register_timeout_seconds=120 RAY_gcs_rpc_server_connect_timeout_s=120
export RAY_gcs_rpc_server_reconnect_timeout_s=120 RAY_gcs_server_request_timeout_seconds=120
export RAY_raylet_report_resources_period_milliseconds=500 RAY_task_events_report_interval_ms=5000
export RAY_num_workers_soft_limit=16 RAY_grpc_keepalive_time_ms=60000 RAY_grpc_keepalive_timeout_ms=60000
export RAY_enable_timeline=false RAY_event_stats=false RAY_ray_syncer_message_refresh_interval_ms=10000
export RAY_core_worker_internal_heartbeat_ms=5000 RAY_health_check_period_ms=10000 RAY_process_group_cleanup_enabled=1
export NRL_JOB_START_EPOCH=$(date +%s.%N) NRL_DEDICATED_RAY_HEAD=0
PORT=1200
# topology probe (as ray.sub): NVLink domain + topology rank resources, so the run pins roles to NVLink domains
CLUSTER_UUID=$(nvidia-smi -q 2>/dev/null | grep 'ClusterUUID' | head -1 | awk -F: '{print $2}' | tr -d ' ')
TOPO_RANK=""
if [[ -n "${SLURM_TOPOLOGY_ADDR:-}" && "${SLURM_TOPOLOGY_ADDR_PATTERN:-}" == block.node ]]; then
  b=$(echo "${SLURM_TOPOLOGY_ADDR%%.*}" | tr -dc '0-9'); n=$(echo "${SLURM_TOPOLOGY_ADDR##*.}" | tr -dc '0-9')
  [[ -n "$b" && -n "$n" ]] && TOPO_RANK=$(( 10#$b * 10000000000 + 10#$n ))
fi
[ -n "$TOPO_RANK" ] || TOPO_RANK=$(( NODEID + 1 ))
export NRL_GPU_CPU_AFFINITY_FILE=/tmp/nrl_gpu_cpu_affinity
nvidia-smi topo -m 2>/dev/null | awk '/^GPU[0-9]/ { g = $1; sub(/GPU/, "", g); u = $(NF-1); sub(/[,-].*/, "", u); if (u ~ /^[0-9]+$/) print g, u }' \
  | while read -r g u; do c=$(cat /sys/devices/system/node/node$u/cpulist 2>/dev/null); [ -n "$c" ] && echo "$g:$c"; done > $NRL_GPU_CPU_AFFINITY_FILE || true
RES='{"slurm_managed_ray_cluster": 1, "worker_units": 4'
[ -n "$CLUSTER_UUID" ] && RES+=', "nvlink_domain_'$CLUSTER_UUID'": 1'
RES+=', "topo_rank": '$TOPO_RANK'}'
ph "ClusterUUID=$CLUSTER_UUID topo_addr=${SLURM_TOPOLOGY_ADDR:-} topo_rank=$TOPO_RANK resources=$RES"
IP=$(hostname -I 2>/dev/null | awk '{print $1}'); [ -n "$IP" ] || IP=$(getent hosts "$HOST" | awk '{print $1}' | head -1)
COMMON=(--disable-usage-stats --min-worker-port=2000 --max-worker-port=2999)
MEM_KB=$(awk '/MemTotal/ {print $2}' /proc/meminfo); SHM_KB=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $2}')
OSG=${OBJ_STORE_GB:-auto}
if [ "$OSG" = auto ]; then
  OSG=$(( MEM_KB * 45 / 100 / 1048576 )); s=$(( ${SHM_KB:-0} * 85 / 100 / 1048576 )); [ "$s" -gt 0 ] && [ "$s" -lt "$OSG" ] && OSG=$s
  CG=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null)
  if [[ "$CG" =~ ^[0-9]+$ ]] && [ "$CG" -lt 1152921504606846976 ]; then c=$(( CG * 30 / 100 / 1073741824 )); [ "$c" -lt "$OSG" ] && OSG=$c; fi
  [ "$OSG" -gt 300 ] && OSG=300
fi
ph "memory: MemTotal $((MEM_KB / 1048576)) GiB, /dev/shm $(( ${SHM_KB:-0} / 1048576 )) GiB, object store ${OSG:-ray-default} GiB"
[ -n "$OSG" ] && COMMON+=(--object-store-memory=$(( OSG * 1073741824 )))
PYSPY=$(command -v py-spy || ls /opt/nemo_rl_venv/bin/py-spy 2>/dev/null | head -1)
dump_node() {  # py-spy every python process of this container (read-only; nothing is killed here)
  mkdir -p "$LOG_DIR/pyspy"; local o="$LOG_DIR/pyspy/$HOST.txt"
  echo "=== $HOST $(date -Is) py-spy=${PYSPY:-none}" >> "$o"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$o" 2>&1
  for p in $(pgrep -u "$(id -u)" -f 'python|ray::' 2>/dev/null); do
    [ "$p" = $$ ] && continue
    echo "--- pid $p $(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-200)" >> "$o"
    [ -n "$PYSPY" ] && timeout 20 "$PYSPY" dump --pid "$p" >> "$o" 2>&1
  done
  touch "$LOG_DIR/pyspy/$HOST.done"
}
leave_node() {  # graceful ray stop, then TERM any GPU compute process still alive, by PID
  ray stop > /dev/null 2>&1 || true; sleep 5
  local left; left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
  if [ -n "$left" ]; then ph "GPU processes left after ray stop: $(echo $left)"; for p in $left; do kill -TERM "$p" 2>/dev/null; done; fi
}

if [ "$NODEID" = 0 ]; then
  rm -f "$LOG_DIR/ENDED" "$LOG_DIR/STARTED_RAY_HEAD"
  echo "$IP" > "$LOG_DIR/head_ip.tmp" && mv "$LOG_DIR/head_ip.tmp" "$LOG_DIR/head_ip"
  ok=0
  for i in 1 2 3; do
    ray stop > /dev/null 2>&1 || true; rm -rf /tmp/ray/session_* 2>/dev/null
    if ray start --head "${COMMON[@]}" --resources="$RES" --node-ip-address="$IP" --port=$PORT --ray-client-server-port=1201 \
         --dashboard-port=8265 --dashboard-host="$IP" --include-dashboard=True --node-manager-port=1302 --object-manager-port=1304 \
         --runtime-env-agent-port=1306 --dashboard-agent-grpc-port=1308 --dashboard-agent-listen-port=1312 --metrics-export-port=1310; then
      ok=1; break; fi
    sleep 5
  done
  [ $ok = 1 ] || { ph "ray head failed"; touch "$LOG_DIR/ENDED"; exit 1; }
  touch "$LOG_DIR/STARTED_RAY_HEAD"; ph "head GCS up at $IP:$PORT"
  want=$(( 4 * NN )); deadline=$(( SECONDS + 1200 ))
  while :; do
    u=$(ray status 2>/dev/null | grep worker_units | awk -F'[/. ]' '{print $4}'); u=${u:-0}
    [ "$u" = "$want" ] && break
    (( SECONDS > deadline )) && { ph "only $u/$want worker_units after 20 min"; touch "$LOG_DIR/ENDED"; exit 1; }
    sleep 5
  done
  ph "cluster ready ($want worker_units); driver starts"
  export NRL_RAY_READY_EPOCH=$(date +%s.%N)
  printf '%q ' "$@" > "$LOG_DIR/driver_command.sh"
  setsid "$@" >> "$COLO_RUNS/$RUN.out" 2>&1 < /dev/null &
  DPID=$!
  STALL=${BR_STALL_S:-1080}; last=$SECONDS; sig=""; stalled=0
  while kill -0 $DPID 2>/dev/null; do
    sleep 30
    s2=$(stat -L -c %s "$COLO_RUNS/$RUN/run.log" "$COLO_RUNS/$RUN.out" 2>/dev/null | tr '\n' ' ')
    if [ "$s2" != "$sig" ]; then sig=$s2; last=$SECONDS
    elif (( SECONDS - last > STALL )); then
      ph "STALL: no run.log / .out growth for $STALL s; py-spy dump on all nodes, then the driver is stopped"
      stalled=1; touch "$LOG_DIR/DUMP"; dump_node
      for i in $(seq 1 18); do [ "$(ls "$LOG_DIR/pyspy/"*.done 2>/dev/null | wc -l)" -ge "$NN" ] && break; sleep 5; done
      kill -TERM -- -$DPID 2>/dev/null; sleep 60; kill -KILL -- -$DPID 2>/dev/null
      break
    fi
  done
  wait $DPID; rc=$?; [ $stalled = 1 ] && rc=124
  ph "driver exited rc=$rc"
  if [ $rc != 0 ] && [ $stalled = 0 ]; then
    touch "$LOG_DIR/DUMP"; dump_node
    for i in $(seq 1 18); do [ "$(ls "$LOG_DIR/pyspy/"*.done 2>/dev/null | wc -l)" -ge "$NN" ] && break; sleep 5; done
  fi
  touch "$LOG_DIR/ENDED"
  leave_node
  exit $rc
else
  for i in $(seq 1 360); do [ -s "$LOG_DIR/head_ip" ] && [ -f "$LOG_DIR/STARTED_RAY_HEAD" ] && break; [ -f "$LOG_DIR/ENDED" ] && exit 1; sleep 5; done
  HIP=$(cat "$LOG_DIR/head_ip" 2>/dev/null); [ -n "$HIP" ] || { ph "no head after 30 min"; exit 1; }
  ok=0
  for i in $(seq 1 20); do
    if ray start --address="$HIP:$PORT" "${COMMON[@]}" --resources="$RES" --node-ip-address="$IP" --node-manager-port=1301 \
         --object-manager-port=1303 --runtime-env-agent-port=1305 --dashboard-agent-grpc-port=1307 \
         --dashboard-agent-listen-port=1311 --metrics-export-port=1309; then ok=1; break; fi
    ray stop > /dev/null 2>&1 || true; sleep 5
  done
  [ $ok = 1 ] || { ph "worker failed to join"; exit 1; }
  ph "joined $HIP:$PORT"
  bad=0; dumped=0
  while [ ! -f "$LOG_DIR/ENDED" ]; do
    sleep 15
    if [ $dumped = 0 ] && [ -f "$LOG_DIR/DUMP" ]; then dumped=1; dump_node; fi
    if timeout 20 ray health-check --address "$HIP:$PORT" > /dev/null 2>&1; then bad=0; else bad=$(( bad + 1 )); fi
    (( bad >= 4 )) && { ph "head GCS unreachable 4x; leaving"; break; }
  done
  ph "ENDED seen; leaving"
  leave_node
  exit 0
fi
