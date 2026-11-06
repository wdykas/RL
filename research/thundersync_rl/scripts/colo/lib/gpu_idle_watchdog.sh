#!/bin/bash
# GPU-idle watchdog (cluster rule 10-27: admins cancel jobs whose GPUs sit idle ~30 min). Fail the job ourselves, loudly,
# before that happens.
#   - Every node samples `nvidia-smi --query-gpu=index,utilization.gpu` every GPU_WD_SAMPLE_MS (2 s); per GPU_WD_POLL_S
#     (60 s) window a GPU counts as used if any sample was non-zero.
#   - The clock starts when the job first shows non-zero utilization on ANY GPU of ANY node; before that only the
#     startup cap below applies. From then on, a GPU's idle time is now - max(its last non-zero sample, arm time).
#   - At GPU_WD_WARN_S (600 s) idle on any GPU: a "[gpu-watchdog] WARNING" line naming host:gpu ids (again whenever the
#     idle set changes).
#   - At GPU_WD_FAIL_S (1200 s) idle on any GPU: "[gpu-watchdog] FATAL" with the reason in the job / run logs and in
#     <state>/FATAL; the job's driver process group gets SIGTERM (SIGKILL 60 s later): the job fails with that reason
#     instead of being cancelled by the admins.
#   - The background processes watch the process they serve and exit as soon as it is gone (a finished job is never
#     held open by them).
# 11-04: what counts as "used" is the reaper's metric. The admins' idle reaper reads DCGM_FI_PROF_SM_ACTIVE (fraction of
#   SMs with a resident warp) from the node's dcgm-exporter and counts a GPU as idle while it is <= 0.01 for 30 min.
#   nvidia-smi utilization.gpu is non-zero whenever any kernel ran in the sample period: a GPU spinning one 1-thread
#   kernel reads 100% util but SM_ACTIVE 0.0065. The monitor now also scrapes the
#   same exporter ($GPU_WD_DCGM_URL, default http://localhost:9400/metrics) every ~10 s and logs the per-GPU window
#   mean SM_ACTIVE; when a GPU has SM_ACTIVE samples in the window it counts as used only if that mean is > 0.01
#   (GPU_WD_SM_IDLE), else (no exporter / no series) it falls back to "any non-zero nvidia-smi util sample".
#   util.log token per GPU: idx:util%/powerW/memMiB[/sm<mean>][/blind][/job<hpc_job>]. The /job field appears only when
#   the exporter's hpc_job label for that GPU is not this job (missing mapping = "none"): the reaper attributes GPUs to
#   jobs through that label.
#   DCGM-blind GPUs: SM_ACTIVE <= 0.01 while the GPU draws >= GPU_WD_BLIND_W (500 W) is not real idleness (an idle
#   GB200 draws ~200-250 W, one spinning 1-thread kernel ~220 W); it is the node's DCGM profiling reporting 0 (seen on
#   one node, 11-04). Such a GPU is tagged /blind, counts as used by util (broken telemetry must not fail our job),
#   and the node gets a WARNING plus <state>/<host>.dcgmblind: the reaper reads the same 0 and will count the node's
#   GPUs idle (the "4/16 GPUs idle for 30m" cancels). The enforcer names blind nodes in its warnings.
#   Fail fast (11-04): a node /blind for GPU_WD_BLIND_FAIL_S (480 s = 8 consecutive windows) makes the enforcer fail the
#   job with "DCGM-BLIND node <host>: resubmit" (excluding that node) instead of burning the GPUs until the reaper's
#   30 min; the node and job id are appended to $COLO_OUT/blind_nodes.txt (evidence for the admins).
#   Frozen exporter (11-04, one node): the exporter can also report a constant non-zero SM_ACTIVE
#   (0.382 on all 4 GPUs for 42 windows, idle or 95% busy); the reaper counted that node 4/16 idle. A non-zero window
#   mean identical (4 decimals) for GPU_WD_FROZEN_N (5) consecutive windows is tagged /frozen (log only: a job
#   survived >30 min with a frozen node, so frozen alone is not a proven reap cause; no fail-fast).
# Startup cap (10-29): if no GPU of the job has any utilization GPU_WD_STARTUP_FAIL_S (1500 s) after the enforcer starts, FATAL
#   too (startup hangs were previously invisible: the idle clock only arms at first GPU use).
# Knobs: GPU_IDLE_WATCHDOG=0 disables; GPU_WD_POLL_S / GPU_WD_WARN_S / GPU_WD_FAIL_S; GPU_WD_GPUS="0,1" restricts the
# watched GPU indices (default all). State: $GPU_WD_DIR or $COLO_OUT/runs_wd/<SLURM_JOB_ID> (per-node status).
# Modes:
#   monitor-bg              (every node; ray.sub SETUP_COMMAND) per-node sampler, lives as long as the node's ray.sub
#                           script (the setup script's parent)
#   enforce-bg [run_log]    (head; in the driver COMMAND) enforcer on the CALLER's process group, lives as long as it
#   local-bg [run_log]      (1-node scripts; sourced envguard.sh) monitor + enforcer on the calling script
#   monitor WATCH_PID / enforce PGID WATCH_PID [run_log]   foreground forms (used by the -bg modes)
set -u
# State lives under $COLO_OUT (your own output root), never under $HOME.
W=${COLO_OUT:?COLO_OUT unset (source $COLO_SCRIPTS/env.sh)}
SELF=$(readlink -f "${BASH_SOURCE[0]}")
POLL=${GPU_WD_POLL_S:-60}; WARN=${GPU_WD_WARN_S:-600}; FAIL=${GPU_WD_FAIL_S:-1200}; SFAIL=${GPU_WD_STARTUP_FAIL_S:-1500}
STALEF=${GPU_WD_STALE_FAIL_S:-1200}  # no fresh GPU status AND no run-log growth this long: FATAL (R15)
JOB=${SLURM_JOB_ID:-local-$$}
D=${GPU_WD_DIR:-$W/runs_wd/$JOB}; export GPU_WD_DIR=$D
HOST=$(hostname 2>/dev/null || echo node)
SMURL=${GPU_WD_DCGM_URL:-http://localhost:9400/metrics}; SMIDLE=${GPU_WD_SM_IDLE:-0.01}; BLINDW=${GPU_WD_BLIND_W:-500}; BLINDF=${GPU_WD_BLIND_FAIL_S:-480}
BLINDLOG=${GPU_WD_BLIND_LOG:-$W/blind_nodes.txt}
# Timing uses bash builtins only (R15: a job's node lost /usr/bin mid-job; `sleep` failed and the old
# `nap() { sleep || exit 0; }` ended monitor and enforcer silently). EPOCHSECONDS / printf %()T / read -t need no binary.
ts() { printf '%(%Y-%m-%dT%H:%M:%S)T' -1; }
exec {_wd_sfd}<> <(:) 2>/dev/null || _wd_sfd=""
alive() { kill -0 "$1" 2>/dev/null; }
_wd_spin=0
nap() {  # sleep POLL s with `read -t` on a pipe nobody writes; a read that returns at once (no fd: container torn
  # down) counts as a spin; after 5 in a row stop, loudly (never spin, never vanish silently)
  # 11-02: in slices of <= 5 s, returning early once the watched pid ($WATCH) is gone, so the watchdog leaves within
  # seconds of its job's script (was up to POLL + the nvidia-smi window, ~2 min, after the script ended)
  local a=$EPOCHSECONDS
  if [ -n "$_wd_sfd" ]; then
    while (( EPOCHSECONDS - a < POLL )); do
      read -r -t 5 -u "$_wd_sfd" _ 2>/dev/null
      [ -n "${WATCH:-}" ] && ! alive "$WATCH" && { _wd_spin=0; return 0; }
      (( EPOCHSECONDS - a < 1 )) && break  # read returned at once: no usable pipe
    done
  fi
  if (( EPOCHSECONDS - a >= POLL - 1 )) || { [ -n "${WATCH:-}" ] && ! alive "$WATCH"; }; then _wd_spin=0; return 0; fi
  sleep "$POLL" 2>/dev/null && { _wd_spin=0; return 0; }
  (( ++_wd_spin >= 5 )) && { m="[gpu-watchdog] $(ts) $$: cannot wait (no read -t pipe, no sleep): exiting"; echo "$m" >&2; echo "$m" >> "$D/enforce.log"; exit 0; }
  return 0; }
[ "${GPU_IDLE_WATCHDOG:-1}" = 0 ] && exit 0
mode=${1:-}; shift || true
case $mode in
monitor)
  # One window per GPU_WD_POLL_S: nvidia-smi streams a sample every GPU_WD_SAMPLE_MS (2 s) for the window; a GPU counts
  # as used in the window if ANY sample is non-zero (a single instantaneous read misses short bursts, e.g. a learner
  # that is busy a few seconds per step, and would report it idle).
  WATCH=$1; mkdir -p "$D" || exit 0; first=0; declare -A last; smjob_seen=""; declare -A smprev smrep; FROZN=${GPU_WD_FROZEN_N:-5}
  # SM_ACTIVE sampler (11-04): scrape the node's dcgm-exporter every ~10 s for one window; one line per GPU and scrape:
  # "<UUID> <SM_ACTIVE> <hpc_job label or none>". The exporter refreshes its values about every 30 s.
  have_curl=1; command -v curl >/dev/null 2>&1 || have_curl=0
  smsample() { local end=$(( EPOCHSECONDS + POLL - 5 ))
    while (( EPOCHSECONDS < end )) && alive "$WATCH"; do
      curl -s --max-time 3 "$SMURL" 2>/dev/null | awk '/^DCGM_FI_PROF_SM_ACTIVE\{/ { u=$0; sub(/.*UUID="/,"",u); sub(/".*/,"",u)
        j="none"; if ($0 ~ /hpc_job="/) { j=$0; sub(/.*hpc_job="/,"",j); sub(/".*/,"",j) }; print u, $NF, j }'
      if [ -n "$_wd_sfd" ]; then read -r -t 10 -u "$_wd_sfd" _ 2>/dev/null; else sleep 10 2>/dev/null || return; fi
    done; }
  while alive "$WATCH"; do
    t0=$EPOCHSECONDS
    smfile="$D/.$HOST.sm.$$"; smpid=""
    [ "$have_curl" = 1 ] && { smsample > "$smfile" 2>/dev/null & smpid=$!; }
    # 11-02: the sampling window runs in the background; the watched pid is checked every ~5 s, so the monitor leaves
    # within seconds of its script instead of after the full window
    smf="$D/.$HOST.smi.$$"
    timeout "$POLL" nvidia-smi --query-gpu=index,utilization.gpu,power.draw,memory.used,uuid --format=csv,noheader,nounits -lms "${GPU_WD_SAMPLE_MS:-2000}" > "$smf" 2>&1 &
    smp=$!
    while alive "$smp"; do
      if [ -n "$_wd_sfd" ]; then read -r -t 5 -u "$_wd_sfd" _ 2>/dev/null; else sleep 5 2>/dev/null; fi
      alive "$WATCH" || { kill "$smp" 2>/dev/null; break; }
    done
    wait "$smp" 2>/dev/null
    out=$(< "$smf"); rm -f "$smf"
    # per-UUID window mean SM_ACTIVE and the hpc_job label(s) seen: "<UUID> <mean> <labels,>"
    smout=""; if [ -n "$smpid" ]; then kill "$smpid" 2>/dev/null; wait "$smpid" 2>/dev/null
      smout=$(awk '$2 ~ /^[0-9.eE+-]+$/ { s[$1]+=$2; n[$1]++; if (!((($1) SUBSEP ($3)) in seen)) { seen[$1,$3]=1; j[$1]=j[$1] $3 "," } }
        END { for (u in n) printf "%s %.4f %s\n", u, s[u]/n[u], j[u] }' "$smfile" 2>/dev/null); rm -f "$smfile"; fi
    declare -A smm=() smj=(); while read -r su sv sj; do [ -n "$su" ] && { smm[$su]=$sv; smj[$su]=${sj%,}; }; done <<< "$smout"
    alive "$WATCH" || break
    now=$EPOCHSECONDS
    if [ -z "$out" ] || [[ "$out" == *[Ff]ailed* ]] || [[ "$out" == *"not found"* ]]; then
      echo "[gpu-watchdog] $(ts) $HOST: nvidia-smi failed: ${out:0:200}" >&2; (( now - t0 < POLL )) && nap; continue; fi
    line="t=$now"; declare -A used=() usum=() ucnt=() psum=() mmax=() uuid=()
    while IFS=', ' read -r idx util pw mem uu; do
      [[ "$idx" =~ ^[0-9]+$ ]] || continue
      if [ -n "${GPU_WD_GPUS:-}" ] && [[ ",$GPU_WD_GPUS," != *",$idx,"* ]]; then continue; fi
      [ -z "${last[$idx]:-}" ] && last[$idx]=0
      [ -n "${uu:-}" ] && uuid[$idx]=$uu
      [ "${util:-0}" != 0 ] && used[$idx]=1
      [[ "${util:-0}" =~ ^[0-9]+$ ]] && { usum[$idx]=$(( ${usum[$idx]:-0} + util )); ucnt[$idx]=$(( ${ucnt[$idx]:-0} + 1 )); }
      [[ "${pw:-}" =~ ^[0-9]+ ]] && psum[$idx]=$(( ${psum[$idx]:-0} + ${BASH_REMATCH[0]} ))
      [[ "${mem:-}" =~ ^[0-9]+$ ]] && (( mem > ${mmax[$idx]:-0} )) && mmax[$idx]=$mem
    done <<< "$out"
    # 11-01: per-window MEAN utilization per GPU (the admins' idle reaper is not "any non-zero sample"; this log is
    # what tells us which GPUs look idle to it)
    # 11-04: + /sm<mean SM_ACTIVE> (the reaper's metric) and, if the exporter maps the GPU to another job or none,
    # /job<label>. A GPU with SM_ACTIVE samples counts as used only if mean > SMIDLE (reaper rule), else util decides.
    uline="t=$now"; badjob=""; blind=""
    for idx in $(printf '%s\n' "${!ucnt[@]}" | sort -n); do
      tok="$idx:$(( usum[$idx] / ucnt[$idx] ))%/$(( ${psum[$idx]:-0} / ucnt[$idx] ))W/${mmax[$idx]:-0}MiB"
      u=${uuid[$idx]:-}; sv=""; [ -n "$u" ] && sv=${smm[$u]:-}
      if [ -n "$sv" ]; then
        tok="$tok/sm$sv"
        if [ "$sv" = "${smprev[$idx]:-}" ]; then smrep[$idx]=$(( ${smrep[$idx]:-1} + 1 )); else smrep[$idx]=1; fi; smprev[$idx]=$sv
        if (( ${smrep[$idx]} >= FROZN )) && awk -v v="$sv" -v t="$SMIDLE" 'BEGIN { exit !(v + 0 > t + 0) }'; then
          tok="$tok/frozen"; fi  # log only: a job ran past 30 min with a frozen node, so not a proven reap cause
        if awk -v v="$sv" -v t="$SMIDLE" 'BEGIN { exit !(v + 0 <= t + 0) }'; then
          if (( ${psum[$idx]:-0} / ucnt[$idx] >= BLINDW )); then tok="$tok/blind"; blind="$blind gpu$idx"  # util decides
          else unset 'used[$idx]'; fi
        else used[$idx]=1; fi
        sj=${smj[$u]:-}
        if [[ "$JOB" =~ ^[0-9]+$ ]] && [[ ",$sj," != *",$JOB,"* ]]; then tok="$tok/job${sj:-none}"; badjob="$badjob $idx:${sj:-none}"; fi
      fi
      uline="$uline $tok"
    done
    echo "$uline" >> "$D/$HOST.util.log" 2>/dev/null || true
    if [ -n "$blind" ]; then
      [ -f "$D/$HOST.dcgmblind" ] || { echo "[gpu-watchdog] $(ts) $HOST: WARNING dcgm-exporter reports SM_ACTIVE <= $SMIDLE at >= ${BLINDW} W on$blind: this node's DCGM profiling metrics are broken; the cluster idle reaper reads them and will count these GPUs idle (cancel at ~30 min). Exclude $HOST and resubmit." >&2
        bsince=$t0; }  # window start: the first blind window counts in full
      # since = start of this consecutive blind run, last = this window (a dead monitor's file goes stale)
      { echo "since=${bsince:-$t0} last=$now gpus=$(echo ${blind} | tr ' ' ',')" > "$D/$HOST.dcgmblind.tmp.$$" && mv -f "$D/$HOST.dcgmblind.tmp.$$" "$D/$HOST.dcgmblind"; } 2>/dev/null
    else rm -f "$D/$HOST.dcgmblind" 2>/dev/null; bsince=""; fi
    if [ "$badjob" != "$smjob_seen" ]; then
      [ -n "$badjob" ] && echo "[gpu-watchdog] $(ts) $HOST: WARNING dcgm-exporter maps GPU(s)$badjob to another job / none (not $JOB): the idle reaper may count them idle" >&2
      smjob_seen=$badjob; fi
    unset usum ucnt psum mmax uuid smm smj
    for idx in $(printf '%s\n' "${!last[@]}" | sort -n); do
      if [ -n "${used[$idx]:-}" ]; then last[$idx]=$now; [ "$first" = 0 ] && first=$now; fi
      line="$line $idx:${last[$idx]}"
    done
    unset used
    # per-process tmp name (two monitors per node raced on one tmp name; 10-29 red-team R7); a failed write is retried next window
    { printf 'first=%s %s\n' "$first" "$line" > "$D/$HOST.status.tmp.$$" && mv -f "$D/$HOST.status.tmp.$$" "$D/$HOST.status"; } || true
  done
  m="[gpu-watchdog] $(ts) $HOST: monitor $$ exits: watched pid $WATCH is gone"; echo "$m" >&2; echo "$m" >> "$D/enforce.log" ;;
monitor-bg)
  mkdir -p "$D"; pidf=${GPU_WD_PIDDIR:-/tmp}/gpu_wd_monitor_$JOB.pid
  if [ -f "$pidf" ] && read -r op < "$pidf" && alive "$op"; then exit 0; fi  # ray.sub retries SETUP_COMMAND
  # caller = the shell running setup_command.sh (exits at once); its parent = this node's ray.sub head / worker script
  WATCH=$(ps -o ppid= -p "$PPID" 2>/dev/null | tr -d ' ')
  { [ -z "$WATCH" ] || [ "$WATCH" -le 1 ]; } && WATCH=$PPID
  BASH_ENV= ENV= nohup setsid bash "$SELF" monitor "$WATCH" > "$D/$HOST.monitor.log" 2>&1 < /dev/null &
  echo $! > "$pidf"
  echo "[gpu-watchdog] $(ts) $HOST: monitor started (job $JOB, poll ${POLL}s, warn ${WARN}s, fail ${FAIL}s, state $D, watching pid $WATCH)" ;;
enforce)
  PG=$1; WATCH=$2; RUNLOG=${3:-}; MONPIDF=${4:-}; mkdir -p "$D" || exit 0; warned=""; stale_seen=""; t0=$EPOCHSECONDS
  last_fresh=$t0; last_grow=$t0
  # Run-log growth with builtins: an fd on RUNLOG read to EOF each poll; lines from any watchdog do not count.
  lfd=""; [ -n "$RUNLOG" ] && [ -r "$RUNLOG" ] && exec {lfd}< "$RUNLOG" 2>/dev/null && while IFS= read -r -u "$lfd" _; do :; done
  grew() { local l g=1; [ -n "$lfd" ] || return 1
    while IFS= read -r -u "$lfd" l || [ -n "$l" ]; do [[ "$l" == "[gpu-watchdog]"* ]] || g=0; l=""; done; return $g; }
  say() {  # loud: the job log (inherited stderr is enforce.out; RUNLOG is the job's / run's own log), the state dir.
    # Watchdog lines are not run-log growth (grew skips them).
    echo "$1" >&2; echo "$1" >> "$D/enforce.log"; [ -n "$RUNLOG" ] && echo "$1" >> "$RUNLOG" 2>/dev/null; true; }
  while nap; alive "$WATCH"; do
    now=$EPOCHSECONDS; armed=0; idle=(); stale=(); worst=0
    # R15 liveness. (1) local mode: a dead monitor is restarted (a broken sampler must not blind the enforcer).
    mp=""; [ -n "$MONPIDF" ] && read -r mp < "$MONPIDF" 2>/dev/null
    if [ -n "$MONPIDF" ] && ! alive "${mp:-x}"; then
      say "[gpu-watchdog] $(ts) WARNING job $JOB: monitor on $HOST is dead; restarting it"
      BASH_ENV= ENV= nohup setsid bash "$SELF" monitor "$WATCH" >> "$D/$HOST.monitor.log" 2>&1 < /dev/null & echo $! > "$MONPIDF"; fi
    # (2) no fresh status for STALEF: fail only if the job's run log has not grown for STALEF either (a healthy job that
    # keeps logging is never killed by a broken monitor; a hung job with a broken monitor is).
    for f in "$D"/*.status; do [ -f "$f" ] || continue; read -r _ t _ < "$f"; t=${t#t=}; (( now - t <= 3 * POLL )) && last_fresh=$now; done
    grew && last_grow=$now
    if (( now - last_fresh >= STALEF )); then
      if [ -n "$RUNLOG" ] && (( now - last_grow >= STALEF )); then
        msg="[gpu-watchdog] $(ts) FATAL job $JOB: no GPU status for $(( (now - last_fresh) / 60 )) min (monitor broken) and no run-log growth for $(( (now - last_grow) / 60 )) min ($RUNLOG): hung; failing the job. Driver process group $PG gets SIGTERM."
        say "$msg"; echo "$msg" > "$D/FATAL"
        kill -TERM -- "-$PG"; POLL=60 nap; alive "$WATCH" && kill -KILL -- "-$PG"; exit 1
      fi
      [ -z "${stalef_warned:-}" ] && say "[gpu-watchdog] $(ts) WARNING job $JOB: no fresh GPU status for $(( (now - last_fresh) / 60 )) min; enforcing on run-log growth only (${RUNLOG:-no run log: not enforcing})"
      stalef_warned=1
    else stalef_warned=""; fi
    for f in "$D"/*.status; do [ -f "$f" ] || continue
      read -r fst rest < "$f"; fst=${fst#first=}
      [ "$fst" != 0 ] && { [ "$armed" = 0 ] || [ "$fst" -lt "$armed" ]; } && armed=$fst
    done
    if [ "$armed" = 0 ]; then  # startup: no GPU has done any work yet; capped at SFAIL (the cluster counts startup idle too)
      fresh=0; for f in "$D"/*.status; do [ -f "$f" ] || continue; read -r _ t _ < "$f"; t=${t#t=}; (( now - t <= 3 * POLL )) && fresh=$((fresh+1)); done
      if (( now - t0 >= SFAIL )) && (( fresh == 0 )); then
        [ -z "${sfail_warned:-}" ] && say "[gpu-watchdog] $(ts) WARNING job $JOB: no fresh GPU status $(( SFAIL / 60 )) min after start (monitors missing?); not enforcing the startup cap"
        sfail_warned=1
      elif (( now - t0 >= SFAIL )); then
        msg="[gpu-watchdog] $(ts) FATAL job $JOB: no GPU used $(( SFAIL / 60 )) min after start (startup hang?); failing the job now instead of an admin cancel. Driver process group $PG gets SIGTERM."
        say "$msg"; echo "$msg" > "$D/FATAL"
        kill -TERM -- "-$PG"; POLL=60 nap; alive "$WATCH" && kill -KILL -- "-$PG"; exit 1
      fi
      continue
    fi
    for f in "$D"/*.status; do [ -f "$f" ] || continue
      h=$(basename "$f" .status); read -r fst t rest < "$f"; t=${t#t=}
      (( now - t > 3 * POLL )) && stale+=("$h")
      for g in $rest; do i=${g%%:*}; l=${g#*:}; s=$(( l > armed ? l : armed )); id=$(( now - s ))
        (( id > worst )) && worst=$id
        (( id >= WARN )) && idle+=("$h:gpu$i=$(( id / 60 ))min"); done
    done
    bl=""; bfail=""
    for f in "$D"/*.dcgmblind; do [ -f "$f" ] || continue; h=$(basename "$f" .dcgmblind); bl="$bl $h"
      bs=""; blst=""; bg=""; read -r bs blst bg < "$f" 2>/dev/null; bs=${bs#since=}; blst=${blst#last=}
      [[ "$bs" =~ ^[0-9]+$ ]] && [[ "$blst" =~ ^[0-9]+$ ]] && (( now - blst <= 3 * POLL && blst - bs >= BLINDF )) && bfail="$bfail $h:${bg#gpus=}:$(( (blst - bs) / 60 ))min"
    done
    if [ -n "$bfail" ]; then
      bh=$(for x in $bfail; do printf '%s,' "${x%%:*}"; done); bh=${bh%,}
      msg="[gpu-watchdog] $(ts) FATAL job $JOB: DCGM-BLIND node $bh: resubmit (exclude $bh). Blind GPUs:$bfail. dcgm-exporter reports SM_ACTIVE <= $SMIDLE on GPUs drawing >= ${BLINDW} W for >= $(( BLINDF / 60 )) min (broken node DCGM profiling); the cluster idle reaper would cancel this job at ~30 min. Driver process group $PG gets SIGTERM."
      say "$msg"; echo "$msg" > "$D/FATAL"
      for x in $bfail; do echo "$(ts) job=$JOB node=${x%%:*} gpus=$(echo "$x" | cut -d: -f2) blind_for=${x##*:}" >> "$BLINDLOG" 2>/dev/null; done
      kill -TERM -- "-$PG"; POLL=60 nap; alive "$WATCH" && kill -KILL -- "-$PG"; exit 1
    fi
    if [ -n "$bl" ] && [ "$bl" != "${blind_seen:-}" ]; then
      say "[gpu-watchdog] $(ts) WARNING job $JOB: DCGM-blind node(s):$bl (dcgm-exporter SM_ACTIVE ~0 on GPUs drawing >= ${BLINDW} W). The cluster idle reaper will likely cancel this job at ~30 min ('N/M GPUs idle'); exclude these nodes on resubmit."; fi
    blind_seen=$bl
    if [ ${#stale[@]} -gt 0 ] && [ "${stale[*]}" != "$stale_seen" ]; then
      say "[gpu-watchdog] $(ts) WARNING job $JOB: no fresh GPU sample from ${stale[*]} (monitor dead or node hung)"; fi
    stale_seen="${stale[*]:-}"
    if (( worst >= FAIL )); then
      msg="[gpu-watchdog] $(ts) FATAL job $JOB: GPU(s) idle >= $(( FAIL / 60 )) min (${idle[*]}); failing the job now instead of an admin cancel (cluster rule: idle GPUs ~30 min). Driver process group $PG gets SIGTERM."
      say "$msg"; echo "$msg" > "$D/FATAL"
      kill -TERM -- "-$PG"; POLL=60 nap; alive "$WATCH" && kill -KILL -- "-$PG"; exit 1
    fi
    if [ ${#idle[@]} -gt 0 ]; then
      cur="${idle[*]%%=*}"
      [ "$cur" != "$warned" ] && say "[gpu-watchdog] $(ts) WARNING job $JOB: GPU(s) idle >= $(( WARN / 60 )) min: ${idle[*]} (job fails at $(( FAIL / 60 )) min)"
      warned=$cur
    else warned=""; fi
  done
  say "[gpu-watchdog] $(ts) enforcer $$ exits: watched pid $WATCH is gone" ;;
enforce-bg|local-bg)
  mkdir -p "$D"; PG=$(ps -o pgid= -p "$PPID" 2>/dev/null | tr -d ' ')
  # one watchdog per job and container: a script sourcing envguard.sh may call another that sources it again
  # Guard: skip only if the recorded enforcer is alive AND the pid it watches is alive AND that pid is this caller or
  # one of its ancestors (nested envguard). A job running phases back to back
  # gets a new watchdog per phase: the previous one is about to exit with its own caller (R15).
  lpid=${GPU_WD_PIDDIR:-/tmp}/gpu_wd_${mode}_${JOB}.pid
  if [ -f "$lpid" ]; then read -r oe ow < "$lpid"
    if alive "${oe:-x}" && alive "${ow:-x}"; then p=$PPID
      while [ -n "$p" ] && [ "$p" -gt 1 ]; do [ "$p" = "$ow" ] && exit 0; p=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' '); done
    fi
    echo "[gpu-watchdog] $(ts) $HOST: previous watchdog (enforcer ${oe:-?}, watching ${ow:-?}) is not this caller's; starting a new one"
  fi
  [ -z "$PG" ] && { echo "[gpu-watchdog] $(ts) cannot read the caller's process group; watchdog NOT enforcing" >&2; exit 0; }
  monpf=""
  if [ "$mode" = local-bg ]; then
    monpf="$D/$HOST.monitor.$PPID.pid"
    BASH_ENV= ENV= nohup setsid bash "$SELF" monitor "$PPID" >> "$D/$HOST.monitor.log" 2>&1 < /dev/null &
    echo $! > "$monpf"
    echo "[gpu-watchdog] $(ts) $HOST: monitor started (job $JOB, poll ${POLL}s, warn ${WARN}s, fail ${FAIL}s, state $D)"
  fi
  BASH_ENV= ENV= nohup setsid bash "$SELF" enforce "$PG" "$PPID" "${1:-}" "$monpf" >> "$D/enforce.out" 2>&1 < /dev/null &
  echo "$! $PPID" > "$lpid"
  echo "[gpu-watchdog] $(ts) $HOST: enforcer started on process group $PG (job $JOB, state $D)" ;;
*) echo "usage: gpu_idle_watchdog.sh monitor-bg | enforce-bg [run_log] | local-bg [run_log] | monitor WATCH_PID | enforce PGID WATCH_PID [run_log]" >&2; exit 2 ;;
esac
