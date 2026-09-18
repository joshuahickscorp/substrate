# Reproduce and verify Substrate

This is the operational entry point for the frozen repository. It describes
how to inspect the implementation and rerun verification without changing a
sealed scientific result. Activation is always `false`.

## Install

From the repository root:

```bash
make install
```

The editable install is the supported checkout path. The repository's sealed
evidence and raw historical receipts remain separate from the installed Python
package; a fresh clone can therefore run deterministic checks without
pretending that unavailable external services are present.

## Validation tiers

```bash
make test-qualification
make test-normal
make test-integration     # declared external services/tools may be required
make test-expensive       # long campaigns or large corpora
make test-full            # complete certification collection
make audit
make accept               # fail-closed verification path
```

Qualification is the seconds-scale gate for deterministic core invariants.
Normal runs the ordinary package suite. Integration and expensive tiers make
external tools, long campaigns, and large corpora explicit. Unavailable
dependencies are reported by their owning test; they are never converted into
a passing result.

Use these commands to inspect the public command surface without launching a
campaign:

```bash
substrate --help
substrate status
substrate genesis --help
substrate sandbox --help
```

## Frozen campaign verification

The v4 structural campaign is terminal. Its available command family is:

```bash
substrate v4 status
substrate v4 preflight
substrate v4 audit
substrate v4 canaries
substrate v4 pilot
substrate v4 rehearse
substrate v4 verify
```

Do not rerun a terminal campaign to alter its classification. `run` and
`resume` validate existing receipts and execute only genuinely incomplete
frozen units. Verification independently recomputes effects, checks controls
and checkpoint identity, injects declared mutations, and performs clean-clone
and isolated-install checks where the required raw run tree is available.

The clean-clone gate intentionally runs the reproducible normal tier. It does
not silently claim certification, integration, or corpus-heavy coverage when
those external or expensive dependencies are unavailable; invoke those tiers
explicitly with `make test-qualification`, `make test-integration`,
`make test-expensive`, or `make test-full`.

The detailed v4 scientific readout is in
[`SCIENTIFIC_STATUS.md`](SCIENTIFIC_STATUS.md). Current cross-campaign
classifications and the Genesis II conclusion are authoritative only through
the sealed files linked from the root README.

## Failure and transition rules

When a check fails:

1. Preserve the receipt, checkpoint, and verifier output.
2. Stop a live worker with the campaign's stop command and confirm the process
   tree is quiet.
3. Do not change thresholds, generators, splits, seeds, frozen controls,
   budgets, or statistics after observing outcomes.
4. If the cause is a software or instrument defect, add a regression test and
   record an implementation-transition authority with a new source digest and
   exact affected-unit set.
5. Invalidate affected units only; zero-unit verifier repairs must not alter
   principal receipts.
6. Rerun tests, lint, verification, mutation checks, and clean-clone checks.

Tests must not publish into frozen evidence from the active checkout. Use a
temporary root or a writer test double for new exploratory receipts. Do not
hide a failure with a broad skip, expected failure, or relaxed assertion.

## Evidence and portability

Sealed classifications live under `evidence/substrate/`; retained reports and
review packages live under `evidence/artifacts/substrate/`. Mutable raw run
state belongs under ignored `runs/` and `artifacts/` namespaces. The historical
reader resolves predecessor aliases through the migration authority and checks
their recorded SHA-256 before use.

For another host, follow the [portability guide](product/PORTABILITY.md) and
restore only the declared external tools, models, and corpora. A clean clone
must pass the structural and evidence-binding gates before any expensive run is
considered meaningful.
