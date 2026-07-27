# Development

## Scientific freeze

The verdict, three closure passes, and three mechanism nulls are frozen. A refactor, rename, speedup,
smaller implementation, or cleaner test suite is not scientific evidence. A classification changes only
through a separately preregistered experiment and its full verification path.

## Implementation freedom

Implementation, naming, module boundaries, orchestration, and language boundaries may change when all
contracts remain green. Prefer deletion and ownership collapse. Do not add another cognitive layer.

## Tests

Active tests live only under `tests/substrate/` and are organized around invariants, not historical module
boundaries. Do not hide failures with broad skips, expected failures, or relaxed assertions. A historical
test may move to the archive only when its architecture is gone and an active replacement covers any
surviving invariant.

Run:

```bash
substrate test
substrate verify
substrate rehearse
```

## Artifact rules

- `substrate.evidence` is the only artifact writer.
- Sealed evidence is immutable.
- Mutable receipts belong under `runs/substrate/v1/`.
- Current evidence belongs under `evidence/substrate/v1/`.
- Execution Forge measurements and reports belong under `artifacts/substrate/execution-forge/`.
- Scratch data and caches are never authorities.
- Every artifact has one declared producer and checkout-independent content identity.

## Naming and archive rules

Active product, package, command, path, program, branch, configuration, and environment names use
Substrate. Predecessor identities remain only in sealed historical evidence, citations, and the hash-bound
migration manifest. Active code reads neutral aliases through `substrate.historical`; no compatibility
package or predecessor environment reader exists.

Superseded source, tests, and documents live under `archive/pre-substrate-event-horizon/` and are absent
from active navigation. Exact duplicates are deleted rather than archived repeatedly.

## Rust gate

Rust is admitted only for one narrow deterministic kernel after profiling proves at least 1.5 times
isolated speedup and at least 10 percent projected total-run reduction, or a necessary memory, crash-risk,
or determinism improvement. One crate is the maximum. Scientific policy and classifications remain Python.

## LOC accounting

Report physical and executable production lines separately. Executable lines exclude blanks, comments,
docstrings, and delimiter-only structural lines. Generated artifacts, sealed evidence, archives, third-party code, and tests are separate
categories. A lower candidate is valid only after all load-bearing contracts pass; a rejected candidate
must name its failing invariant and the smallest restoration.
