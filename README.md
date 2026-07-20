# MOP

MOP is an experimental system for testing substrate and learning hypotheses under explicit controls,
budgets, stop rules, and independent evidence checks.

## Read first

- `ARCHITECTURE.md`: maintained code boundaries and scientific contract.
- `STATUS.md`: machine authorities for current state and historical recovery.
- `MOP_COLLAPSE_LEDGER.md`: compact generated reduction status.
- `mop.experiments.REGISTRY`: executable experiment inventory.

## Scientific contract

Every experiment declares its question, null, source and split identities, independent unit, treatments,
controls, metric direction, SESOI, multiplicity policy, budget, stop rule, claim ceiling, and independent
verifier. Null and negative results remain valid outputs. Producer and verifier mathematics stay
structurally separate.

## Operations

Active run source, manifests, state, receipts, controls, seeds, thresholds, and worker policy are
immutable. Work in isolated worktrees with separate outputs and record commits, seals, and rollback tags.
Operational policy may change timing, never scientific identity or promotion requirements.

## Development

Use the project environment. Run focused tests while editing, then `python -m pytest`. Current state is
machine-readable; do not copy counts or run status into prose.
