# P6 null-safe campaign router audit

Status: prepared and validation-only until the final substrate policy/governor migration is complete.

The existing campaign supervisor accepts one exact artifact contract per step and has no conditional
dependency or skipped-step state. A favorable-only P6 verifier contract therefore turns a valid null into
an integrity hold; accepting both outcomes in the same long plan would instead make the next-rung task retry
admission forever. Neither behavior is an honest terminal scientific null.

The null-safe router preserves the existing supervisor and throttle unchanged. It divides the long run into
three separately sealed campaigns:

1. EDCM producer/verifier, X0 producer/verifier, P6 10k resource probe, replication, and verifier;
2. P6 100k replication and verifier; and
3. P6 1m replication and verifier.

Each campaign ends after a verifier with common terminal integrity fields rather than a preselected outcome.
The detached outer router then checks the verifier/source self-hashes, exact source file join, current verifier
implementation receipts, complete check set, recomputation envelope, paired contrast semantics, mutation
suite, and next-rung authority. It also requires the sealed campaign status to bind that verifier file and a
governor completion receipt.

A canonical `null` writes a sealed route receipt and terminates the router successfully as
`complete_null_stop`; no higher-rung campaign is started. A canonical `favorable-rung-pattern` advances only
when the verifier also has `next_rung_allowed=true` for the exact next rung. The one-million-event stage is
always terminal. All route artifacts retain `scientific_promotion=false`.

Prepared validation is safe before migration and never launches work:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py validate
```

After the final migrated live policy and governor validate, perform the live check:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py validate --live
```

Only then may the explicit detached start be used:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py start --execute
```

The router refuses `start` without `--execute`. Status, wait, and non-signaling drain controls are:

```bash
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py status
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py wait
PYTHONPATH=src:. .venv/bin/python scripts/mop_null_safe_campaign.py stop \
  --reason "operator requested drain"
```

The machine-readable router plan is
`configs/campaign/mac_studio_substrate_null_safe_router.json`. Its output root is
`runs/mac_studio_null_safe_router/mac-studio-substrate-null-safe-v1`; route receipts live beneath `routes/`.
No live policy, governor, existing campaign plan, task overlay, or active campaign state is changed by these
files.
