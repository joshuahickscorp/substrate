#!/bin/bash
# Detached OS-level watchdog for the MoP general-run chain.
# Every CHECK_INTERVAL seconds: if the orchestrator is not alive, decide what to do from its
# sealed state. It restarts only on an unexpected death or a terminal hold; it stops cleanly
# when the chain has genuinely completed. It NEVER signals the census or any live compute.
set -u
ROOT=/Users/scammermike/Downloads/mop
RUNROOT="$ROOT/runs/generation1/general-run"
STATUS="$RUNROOT/current_status.json"
LOG="$ROOT/runs/generation1/chain_watchdog.log"
PY="$ROOT/.venv/bin/python"
CHECK_INTERVAL=300

cd "$ROOT" || exit 1
echo "$(date '+%F %T') watchdog started (pid $$)" >> "$LOG"

read_state() {
  "$PY" -c "import json,sys
try: print(json.load(open('$STATUS')).get('state','unknown'))
except Exception: print('nostate')" 2>/dev/null
}

while true; do
  TS=$(date '+%F %T')
  # Orchestrator process signature: the detached run --execute parent.
  if pgrep -f "mop_general_run.py run --execute" >/dev/null 2>&1; then
    : # alive, nothing to do
  else
    STATE=$(read_state)
    if [ "$STATE" = "complete" ]; then
      echo "$TS chain COMPLETE (state=complete); watchdog exiting cleanly" >> "$LOG"
      exit 0
    fi
    echo "$TS orchestrator DOWN (state=$STATE); restarting" >> "$LOG"
    case "$STATE" in
      integrity_hold|failure_hold|drained)
        mv "$RUNROOT" "$ROOT/runs/generation1/_HELD-general-run-$(date '+%Y%m%d-%H%M%S')" 2>>"$LOG"
        echo "$TS archived held state root" >> "$LOG"
        ;;
    esac
    PYTHONPATH="$ROOT/src" "$PY" scripts/mop_general_run.py start --execute --root "$RUNROOT" >>"$LOG" 2>&1
    echo "$TS restart command issued" >> "$LOG"
  fi
  sleep "$CHECK_INTERVAL"
done
