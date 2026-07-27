# Substrate Event Horizon precheck

Created at 2026-07-27T16:00:52Z. The machine-readable authority is
`SUBSTRATE_EVENT_HORIZON_PRECHECK.json`.

## Decision

Modification is permitted. No scientific worker or supervisor is running, no process has the selected
source or run roots open, all registered MOP launchd jobs are exited or unloaded, and activation remains
false. This decision does not authorize the scientific long run.

## Source and PR

- Repository: `joshuahickscorp/mop`
- PR: `#35`, open draft, mergeable
- PR head: `agent/mop-temporal-core-mechanism`
- Source commit: `7158451d80cfcacc0763894dad3ee5ee1ca834ec`
- Source tree: `d5f88b918ec74e664f0a3c61aefc592807c2e6da`
- Selected clean source worktree: `/Users/scammermike/Downloads/mop-temporal-core-mechanism`
- Current `/Users/scammermike/Downloads/mop` checkout: clean but 112 commits behind the PR head
- Temporary branch-name holder: `/private/tmp/mop-temporal-hardening-v2.3NVs0a`, clean and 39 commits behind

The remote PR head and selected source commit are object-identical.

## Rollback

The annotated tag `substrate-pre-event-horizon` resolves to `7158451` locally and on `origin`.

1. Stop a future Substrate supervisor with its version-matched stop command.
2. Verify no process has the checkout, evidence, run or artifact roots open.
3. Run `git fetch origin tag substrate-pre-event-horizon`.
4. Run
   `git worktree add /Users/scammermike/Downloads/substrate-rollback substrate-pre-event-horizon`.
5. Verify commit `7158451` and tree `d5f88b9`.
6. Recover on a new branch or with explicit revert commits. Do not reset or rewrite the published branch.

## Baseline

The locked development environment was reconstructed with `uv sync --extra dev --extra ann`. The declared
suite then completed in 125.23 seconds:

```text
693 passed
5 failed
6 skipped
```

The observed failures were the collapse invariant, proof indexing, portability, and two studio-doctor
checks caused by the default development environment omitting `huggingface-hub`. The reported custom
substrate artifact failure is green at the closure commit. Both lists are retained in the JSON authority so
the migration cannot silently redefine the baseline.

The suite updated one generated ledger and `uv.lock`; both precheck side effects were restored byte-for-byte
from `7158451`, and the source worktree was clean before this authority was added.

## Evidence boundary

The tracked `proof/`, `runs/`, `reports/`, and `logs/` roots contain 2,268 files and 88,052,368 bytes. Their
combined Git tree-stream hash at the rollback commit is
`2dfe1c6598ccef7bf3950a34ac3b23d2fced02a9`. Payload bytes are frozen. Migration manifests may map or move
those payloads only with byte identity.

## Frozen scientific result

The verdict remains `certified_cognitive_scaffold`. `grounded_closed_loop`, `unity_under_conflict`, and
`world_self_control_value` remain passes. `endogenous_allocation`, `cross_domain_continuity`, and
`procedural_transfer` remain mechanism nulls. Activation is false.
