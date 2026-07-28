# Substrate v4 runbook

## Install and inspect

```bash
cd /Users/scammermike/Downloads/substrate
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"

substrate v4 status
substrate test
```

The terminal checkout should report 2,136/2,136 valid units, stage
`terminal_classification`, classification `functional_proto_nous_candidate`, and activation `false`.

## V4 command family

```bash
substrate v4 preflight
substrate v4 audit
substrate v4 canaries
substrate v4 pilot
substrate v4 rehearse
substrate v4 run
substrate v4 status
substrate v4 stop
substrate v4 resume
substrate v4 verify
```

The principal campaign is already terminal. Do not rerun it to change the replication null. `run` and
`resume` validate existing unit receipts and execute only genuinely incomplete frozen units.

## Verification

```bash
substrate v4 verify
```

Verification consumes raw receipts, independently recomputes all effects, checks controls and checkpoint
identities, injects every declared mutation, performs an isolated install from
`substrate-v4-structural-ready`, runs the full test and lint gates, regenerates a frozen unit twice, and
rebuilds the external review package.

The expected terminal result is `functional_proto_nous_candidate`. The replication effect remains a null
unless a separately preregistered future campaign establishes otherwise.

## Reconstruct raw receipts

The terminal review package contains `artifacts/substrate/v4/review/RAW_RECEIPTS.jsonl.gz`. Each line
records a repository-relative path, SHA-256, and JSON document. Restore those documents under their
recorded paths before independent recomputation in a fresh terminal checkout.

## Failure recovery

1. Preserve the failing receipt, checkpoint, and verifier output.
2. Run `substrate v4 stop` if a worker is live.
3. Confirm the process tree is quiet.
4. Do not change thresholds, generators, splits, seeds, or frozen controls.
5. Reproduce a software or instrument defect with a regression test.
6. Publish an implementation-transition authority naming affected units.
7. Invalidate only affected units; zero-unit verifier repairs must not alter principal receipts.
8. Rerun tests, lint, verification, mutations, and clean clone.

## Immutable points

```text
substrate-v4-pre-structural-understanding
substrate-v4-structural-ready
substrate-v4-terminal
```

Historical v1, v2, and v3 tags must never move.
