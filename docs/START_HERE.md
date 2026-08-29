# Substrate navigation

Start with the root [README](../README.md) for the frozen scientific status
map. The active engineering and product surface is intentionally small:

- `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/RUNBOOK.md`, and
  `docs/SCIENTIFIC_STATUS.md` describe the core v4 implementation.
- `docs/PRODUCT_FOUNDATION.md` and `docs/product/` describe the non-executing
  product contracts.
- `docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md` records the current
  Genesis II conclusion.
- `src/substrate/`, `tests/substrate/`, and `tools/` are the implementation,
  verification, and audit surfaces.
- `evidence/` contains sealed classifications and proof records; `artifacts/`
  and `runs/` contain campaign outputs and mutable runtime state.

The Finder-facing compatibility layout remains available through the lowercase
directories below. They are views into the existing canonical paths, not a
second source tree:

- `run` — transition receipts and runtime state.
- `inputs` — corpora and derived artifacts.
- `evidence` — evidence, proof, and archive material.
- `protocol` — plans and runbook views.
- `code` — source, tests, tools, and configuration.
- `project` — project entry files.
- `operations` — runtime state, logs, and launch templates.

Historical experiment prose, staging handoffs, and reviewer contracts are
under [`docs/archive`](archive/README.md). Nothing in that archive is an
active launch instruction; code references to a few archived documents are
frozen source-input contracts and are listed in the archive index.
