#!/bin/bash
# Detached OS-level supervisor for the MoP general-run chain. Survives session close.
# Jobs:
#   1. Keep the orchestrator alive across TRANSIENT deaths (crash mid-stage): relaunch to resume.
#   2. Extend compute: when the chain reaches "complete", dispatch ONE headless claude successor.
#   3. Do NOT loop on a DETERMINISTIC failure: if the same terminal hold (same stage + same first
#      problem) recurs, stop and flag for a human instead of re-archiving and re-running forever.
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
FP_FILE="$ROOT/runs/generation1/.supervisor_last_fingerprint"
MARKER="$ROOT/runs/generation1/DETERMINISTIC_FAILURE.txt"
STOP=/Users/scammermike/.mop_autonomy_stop
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
read_fingerprint() {
  "$PY" -c "import json
try:
    d=json.load(open('$STATUS')); p=(d.get('problems') or ['?'])
    print(str(d.get('stage'))+'::'+str(p[0] if p else '?'))
except Exception: print('nostate::?')" 2>/dev/null
}

while true; do
  TS=$(date '+%F %T')
  if [ -e "$STOP" ]; then
    echo "$TS stop file present; supervisor exiting (chain left as-is)" >> "$LOG"
    exit 0
  fi
  if pgrep -f "mop_general_run.py run --execute" >/dev/null 2>&1; then
    sleep "$CHECK_INTERVAL"; continue
  fi

  STATE=$(read_state)
  if [ "$STATE" = "complete" ]; then
    if [ ! -e "$SUCCESSOR_MARK" ]; then
      echo "$TS chain COMPLETE; launching headless successor (once) to extend compute" >> "$LOG"
      touch "$SUCCESSOR_MARK"
      caffeinate -i "$CLAUDE" -p "$(cat "$SUCCESSOR_PROMPT")" --dangerously-skip-permissions >> "$LOG" 2>&1 &
      echo "$TS successor session dispatched" >> "$LOG"
    fi
    sleep "$CHECK_INTERVAL"; continue
  fi

  case "$STATE" in
    integrity_hold|failure_hold|drained)
      # Terminal hold. Fingerprint it; if the SAME hold recurs, it is deterministic: stop, do not loop.
      FP=$(read_fingerprint)
      LAST=$(cat "$FP_FILE" 2>/dev/null || echo "")
      if [ "$FP" = "$LAST" ]; then
        echo "$TS DETERMINISTIC failure recurred ($FP); NOT restarting. Flagging + stopping." >> "$LOG"
        printf '%s deterministic chain failure recurred: %s\nSupervisor stopped to avoid a restart loop. Fix required, then remove %s and relaunch.\n' "$TS" "$FP" "$STOP" > "$MARKER"
        PYTHONPATH="$ROOT/src" "$PY" -m mop.studio.telegram_rung_notifier notify --text "MoP chain deterministic failure: $FP. Supervisor stopped." >/dev/null 2>&1 || true
        touch "$STOP"
        exit 1
      fi
      echo "$FP" > "$FP_FILE"
      echo "$TS orchestrator terminal-hold ($FP); archiving + restarting once" >> "$LOG"
      mv "$RUNROOT" "$ROOT/runs/generation1/_HELD-general-run-$(date '+%Y%m%d-%H%M%S')" 2>>"$LOG"
      ;;
    *)
      # Non-terminal death (crashed mid-stage): transient. Relaunch to RESUME (no archive, no fingerprint stop).
      echo "$TS orchestrator died mid-stage (state=$STATE); relaunching to resume" >> "$LOG"
      ;;
  esac
  PYTHONPATH="$ROOT/src" "$PY" scripts/mop_general_run.py start --execute --root "$RUNROOT" >>"$LOG" 2>&1
  echo "$TS restart command issued" >> "$LOG"
  sleep "$CHECK_INTERVAL"
done
