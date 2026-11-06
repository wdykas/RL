#!/bin/bash
# keep_prev_run.sh RUNS_DIR RUN: run at JOB START (submit_ab.sh COMMAND), before the arm writes RUNS_DIR/RUN and RUN.out.
# An earlier attempt of the same tag (finished, crashed or never started) is moved to RUN.prev-<time>{,.out}, so a rerun
# or a second queued job with the same TAG never overwrites it (dc-ab, dc-fit were lost that way, 10-16). If the earlier
# attempt is still being written (run.log / .out modified < 10 min ago: another job of this tag is running), refuse
# (exit 3): two live jobs must never share a run directory. Job logs (RUN.jobs/<jobid>-logs) are per job id: untouched.
set -u
D=${1:?runs dir}; RUN=${2:?run tag}
cd "$D" || exit 2
[ -e "$RUN" ] || [ -s "$RUN.out" ] || exit 0
# Only the files a live run writes count (10-29 red-team R5: the dir's own mtime moved when a tool created a dotdir in it,
# which falsely refused a run).
live=$(find "$RUN/run.log" "$RUN/history.json" "$RUN.out" -maxdepth 0 -mmin -10 2>/dev/null | head -1)
if [ -n "$live" ] && [ "${ALLOW_LIVE_MOVE:-0}" != 1 ]; then
  echo "[keep_prev_run] $D/$RUN is being written ($live modified < 10 min ago): another job of this tag is live; refusing to start" >&2
  exit 3
fi
PREV=$RUN.prev-$(date +%m%d-%H%M%S)
[ -e "$RUN" ] && mv "$RUN" "$PREV"
[ -e "$RUN.out" ] && mv "$RUN.out" "$PREV.out"
echo "[keep_prev_run] earlier attempt of $RUN kept in $D/$PREV"
