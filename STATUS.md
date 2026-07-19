# Status

Current state is machine-readable:

- Capability and experiment status: runtime registries.
- Collapse progress and accounting: MOP_COLLAPSE_STATE.json.
- Append-only reduction evidence: collapse/MOP_REDUCTION_LOG.json.
- Historical-document recovery: collapse/MOP_HISTORICAL_DOCUMENT_INDEX.json.
- Live campaign state: its immutable run authority.

Do not infer current test counts, campaign progress, evidence status, or hardware requirements from dated
prose. Query the corresponding authority or run the relevant gate.

The General Run checkout and run roots are immutable while active. Collapse work occurs only in its
isolated worktree and must not signal, restart, retune, or rewrite the live campaign.
