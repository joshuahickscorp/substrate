#!/bin/bash
# Durable substrate-research runner. Installed in crontab (every 12h). Survives session close.
# Invokes a headless, unsupervised claude session to advance the STARSS23 substrate by one bed.
# A lockfile prevents overlapping runs (a bed can take longer than the interval).
set -u
ROOT=/Users/scammermike/Downloads/mop
CLAUDE=/Users/scammermike/.local/bin/claude
PROMPT_FILE="$ROOT/scripts/mop_autonomy/substrate_prompt.txt"
LOCK=/tmp/mop_substrate_research.lock
LOG="$ROOT/runs/generation1/substrate_research.log"
STOP=/Users/scammermike/.mop_autonomy_stop

# Kill switch: `touch ~/.mop_autonomy_stop` to disable all autonomous runs.
if [ -e "$STOP" ]; then
  echo "$(date '+%F %T') stop file present; substrate run disabled" >> "$LOG"
  exit 0
fi

# Skip if a run is already in progress.
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "$(date '+%F %T') substrate run already in progress (pid $(cat "$LOCK")); skipping" >> "$LOG"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd "$ROOT" || exit 1
export HOME=/Users/scammermike
export PATH="/Users/scammermike/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
echo "$(date '+%F %T') === substrate research run starting (pid $$) ===" >> "$LOG"
caffeinate -i "$CLAUDE" -p "$(cat "$PROMPT_FILE")" --dangerously-skip-permissions >> "$LOG" 2>&1
echo "$(date '+%F %T') === substrate research run finished (exit $?) ===" >> "$LOG"
