# MOP

MOP is an experimental research system for testing mechanism, substrate, and campaign hypotheses under
explicit controls, budgets, stop rules, and independently checked evidence.

## Current authorities

- ARCHITECTURE.md: maintained system boundaries.
- EXPERIMENTS.md: experiment and verification conventions.
- STATUS.md: where current machine state lives.
- OPERATIONAL_AWARENESS.md: execution and live-run safety.
- DECISIONS.md: maintained architectural decisions.
- docs/ESCS_DEEP_RESEARCH.md: STARSS23 rationale referenced by sealed preregistrations.
- MOP_COLLAPSE_LEDGER.md: active physical-collapse ledger.

Superseded documentation is not maintained beside the implementation. Exact historical bytes, hashes,
Git blobs, and recovery instructions live in collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json.

## Development

Use the project environment and run focused tests for the region being changed. The broad gate is
python -m pytest.

Scientific changes must preserve frozen identities, experimental units, controls, SESOI and multiplicity
rules, stop rules, negative findings, sealed evidence, and producer/verifier independence.
