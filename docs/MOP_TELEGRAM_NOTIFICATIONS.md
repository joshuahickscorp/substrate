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
are eligible. Malformed or unsealed status files fail quiet. Long programs emit
every tenth capsule milestone; compact parent chains emit every stage completion
so prerequisite results are not hidden behind the batching threshold.

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
