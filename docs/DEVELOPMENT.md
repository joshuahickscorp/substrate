# Substrate v4 development rules

## Scientific freeze

The v4 principal result is frozen. All ten principal hypotheses are positive; independent replication is
a sub-SESOI null. Refactoring, renaming, performance work, or verifier cleanup cannot promote that null or
change the terminal classification.

Changes to a scientific premise require a separately preregistered future campaign. Do not change frozen
thresholds, generators, latent systems, surfaces, splits, seeds, controls, budgets, or statistics after
observing v4 outcomes.

## Implementation transitions

A post-launch code defect requires:

1. a regression test;
2. an implementation-transition authority;
3. a new source digest;
4. exact affected-unit identification;
5. invalidation of affected units only;
6. safe resume or independent-verification rerun.

The three v4 verification transitions affected zero principal units and are recorded under
`evidence/substrate/v4/`.

## Tests and lint

```bash
substrate test
ruff check src tests
ruff format --check \
  src/substrate/cli.py src/substrate/runtime.py src/substrate/world.py \
  src/substrate/v4*.py tests/substrate/test_v4_mechanisms.py
```

Tests must not publish into frozen evidence in the active checkout. Use temporary roots or monkeypatch the
writer for publishing canaries. Do not hide failures with broad skips, expected failures, or relaxed
assertions.

## Artifact rules

- V4 sealed evidence belongs under `evidence/substrate/v4/`.
- Mutable raw receipts belong under `runs/substrate/v4/`.
- Reviewable terminal reports belong under `artifacts/substrate/v4/`.
- Every authoritative JSON document has activation `false` and a valid self-seal.
- Raw receipt archives must be deterministic and independently hash-indexed.
- Historical v1, v2, and v3 evidence and tags are immutable.
- Scratch data and caches are never authorities.

## Claim boundary

The maximum automatic classification is `nous_ready_for_review`. Unqualified Nous requires external
scientific and philosophical review. No code, evidence, report, tag, PR, or documentation may imply
consciousness, sentience, personhood, life, moral status, or uncontrolled external agency.
