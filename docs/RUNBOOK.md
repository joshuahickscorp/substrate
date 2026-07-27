# Runbook

## Preflight

```bash
cd /Users/scammermike/Downloads/substrate
uv sync --extra dev
substrate verify
substrate test
substrate rehearse
substrate status
```

Confirm the checkout is clean, activation is false, the authority and configuration hashes match, no
worker or supervisor is active, and the status output reports no drift or stale claims.

## Launch

Launch only after explicit operator authorization:

```bash
substrate run
```

## Status

```bash
substrate status
```

Status is read from the same unit receipts and locks used by the executor.

## Stop

```bash
substrate stop
```

The stop command atomically creates the shared stop switch. The executor finishes no new unit after
observing it. Do not kill a writer during its atomic publication window unless the machine is unsafe.

## Resume

```bash
substrate resume
```

Resume removes the stop switch, revalidates the sealed authority, source, configuration, receipts, and
locks, then continues only dependency-ready incomplete units.

## Failure recovery

1. Run `substrate status` and preserve the failing receipt and logs.
2. Run `substrate stop`.
3. Confirm the process tree is quiet and identify the exact failing unit.
4. Do not edit sealed evidence.
5. Reproduce a source defect with a regression test.
6. Apply an append-only repair and regenerate authorities.
7. Rerun verify, test, rehearsal, mutation, and clean-clone gates.
8. Resume. Completed scientific units remain complete.

## Rollback

The immutable rollback tag is `substrate-pre-event-horizon` at `7158451`.

```bash
git fetch origin tag substrate-pre-event-horizon
git worktree add /Users/scammermike/Downloads/substrate-rollback substrate-pre-event-horizon
```

Verify commit `7158451d80cfcacc0763894dad3ee5ee1ca834ec` and tree
`d5f88b918ec74e664f0a3c61aefc592807c2e6da`. Recover with a new branch or explicit revert commits; do not
rewrite published history.
