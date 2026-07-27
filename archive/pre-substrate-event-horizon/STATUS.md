# Status

Use machine authorities instead of narrative snapshots:

- Experiments: `mop.experiments.REGISTRY` and `registry/experiments.yaml`.
- Collapse checklist, accounting, audit, reduction history, and tagged code recovery:
  `MOP_COLLAPSE_STATE.json`.
- Historical documents: `collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json`.
- Live work: its immutable run authority.

The protected run checkout and run roots are read-only while active. Do not infer test counts, evidence
status, campaign progress, or hardware requirements from prose; query the authority or run its gate.
