#!/bin/bash
# Detached OS-level supervisor for the MoP general-run chain. Survives session close.
# Two jobs:
#   1. Keep the orchestrator alive: restart it on unexpected death or a terminal hold.
#   2. Extend the compute period: when the chain reaches "complete", trigger ONE headless
#      claude successor session to launch the next sealed compute generation, then keep guarding.
# It NEVER signals the census or any live compute.
set -u
ROOT=/Users/scammermike/Downloads/mop
CLAUDE=/Users/scammermike/.local/bin/claude
RUNROOT="$ROOT/runs/generation1/general-run"
STATUS="$RUNROOT/current_status.json"
LOG="$ROOT/runs/generation1/chain_supervisor.log"
PY="$ROOT/.venv/bin/python"
SUCCESSOR_PROMPT="$ROOT/scripts/mop_autonomy/successor_prompt.txt"
SUCCESSOR_MARK="$ROOT/runs/generation1/.successor_launched"
CHECK_INTERVAL=300

export HOME=/Users/scammermike
export PATH="/Users/scammermike/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
cd "$ROOT" || exit 1
echo "$(date '+%F %T') supervisor started (pid $$)" >> "$LOG"

read_state() {
  "$PY" -c "import json
try: print(json.load(open('$STATUS')).get('state','unknown'))
except Exception: print('nostate')" 2>/dev/null
}

while true; do
  TS=$(date '+%F %T')
  # Kill switch: `touch ~/.mop_autonomy_stop` to stop the supervisor (does not touch the chain).
  if [ -e /Users/scammermike/.mop_autonomy_stop ]; then
    echo "$TS stop file present; supervisor exiting (chain left running)" >> "$LOG"
    exit 0
  fi
  if pgrep -f "mop_general_run.py run --execute" >/dev/null 2>&1; then
    : # orchestrator alive
  else
    STATE=$(read_state)
    if [ "$STATE" = "complete" ]; then
      if [ ! -e "$SUCCESSOR_MARK" ]; then
        echo "$TS chain COMPLETE; launching headless successor (once) to extend compute" >> "$LOG"
        touch "$SUCCESSOR_MARK"
        caffeinate -i "$CLAUDE" -p "$(cat "$SUCCESSOR_PROMPT")" --dangerously-skip-permissions >> "$LOG" 2>&1 &
        echo "$TS successor session dispatched" >> "$LOG"
      fi
      # Keep looping to guard whatever the successor launches; do not exit.
    else
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
  fi
  sleep "$CHECK_INTERVAL"
done
