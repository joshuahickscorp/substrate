# MOP Telegram notifications

The MOP watcher sends event-driven updates through the existing Hawking
`@notification333_bot` private chat. It checks every two minutes and reports:

- each completed Generation 1 supervisor capsule or rung;
- terminal completion, failure holds, integrity holds, and stopped campaigns;
- a stalled canonical supervisor when its exact PID/create-time identity is gone
  and its status snapshot has remained unchanged for at least ten minutes;
- standalone Generation 1 proof results not already represented by a supervisor artifact;
- compact coverage, decision, comparison, attempt, and host-health statistics.

Only self-sealed Generation 1 status documents with an exact program identity
are eligible. Malformed or unsealed status files fail quiet. Rung milestones fire
on progress PERCENTAGE, not a fixed rung count: one notification near each 25%
crossing (25/50/75/100%), on any program size. A fixed count is meaningless
across programs of wildly different totals: on a small stage it either never
divides evenly (silence) or, if it exceeds the total, an escape hatch fires on
every single rung (a flood); a percentage scales correctly to any total by
construction. Compact parent chains and terminal states still emit every stage
completion so prerequisite results are not hidden behind the milestone cadence.

The Progress line always includes the percentage (e.g. "Progress: 19/74 (26%)").
Programs that run a dynamic worker pool (horizon, categorized, full-generations)
do not publish a census-style `adaptive_execution` block, so their Workers/ETA
lines are synthesized: the average rung duration comes from real capsule finish
timestamps, and the worker count is a live sample of the dynamic worker
controller's settled recommendation (never the transient +1-per-tick ramp value
a stateless sample would otherwise report). This is operational telemetry only,
shown for the operator's benefit; it is never a receipt field and never a claim
about what any specific past rung actually ran under.

The watcher reads the existing Hawking token and chat ID from macOS Keychain.
Secrets are never copied into MOP, launchd, logs, state, or command arguments.
Telegram delivery is telemetry only and cannot authorize scientific promotion.
Stall detection uses the latest status update timestamp and fails quiet when the
status has no supervisor identity, has an invalid or missing timestamp, or
process liveness cannot be determined. This
deliberately excludes custom queues that do not publish canonical supervisor
identity and avoids treating an ordinary long-running rung as a dead supervisor.

Commands:

```sh
PYTHONPATH=src:. python3.12 scripts/mop_telegram_notifier.py status
PYTHONPATH=src:. python3.12 scripts/mop_telegram_notifier.py prime
PYTHONPATH=src:. python3.12 scripts/mop_telegram_notifier.py send-test
PYTHONPATH=src:. python3.12 scripts/mop_telegram_notifier.py install
```

State and logs are under `reports/telegram_notifier/`. Historical events are
primed without messages, so installation does not backfill old campaign spam.
