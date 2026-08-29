# Archived Substrate material

This directory holds historical context that is useful for provenance but is
not part of the active operating surface. Sealed JSON evidence, raw receipts,
and proof ledgers remain in their canonical `evidence/`, `artifacts/`, and
`runs/` locations; this directory contains the prose and review contracts that
explain older campaigns.

## Sections

- `experiments/v4/`, `experiments/v5/`, `experiments/genesis/`,
  `experiments/genesis2/`, and `experiments/final_revision/` — versioned
  campaign plans, reports, and research notes.
- `staging/tangible_sandbox/` — R2/Odyssey launch and custody handoffs. The
  next-launch runbook is retained here because the control-plane code embeds
  it as a frozen adapter contract.
- the archive root — earlier project roadmap, handoff, and reality reports.

The Genesis II source digest includes
`experiments/genesis2/MASTER_PLAN.md`; Final Revision research includes
`experiments/final_revision/RESEARCH_SURVEY.md`. Those are deliberate frozen
inputs, not current roadmap commitments. All activation boundaries remain
false.

`pre-substrate-event-horizon/` is a sealed archive, not disposable staging:
its manifest and evidence authority bind the complete historical snapshot.
Prune individual files only through a future provenance-preserving migration.
