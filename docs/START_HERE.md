# Substrate navigation

Start with the root [README](../README.md) for the frozen scientific status
map. The active engineering and product surface is intentionally small:

- `docs/ARCHITECTURE.md`, `docs/DEVELOPMENT.md`, `docs/RUNBOOK.md`, and
  `docs/SCIENTIFIC_STATUS.md` describe the core v4 implementation.
- `docs/PRODUCT_FOUNDATION.md` and `docs/product/` describe the non-executing
  product contracts.
- `docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md` records the current
  Genesis II conclusion.
- `src/substrate/`, `tests/substrate/`, and `ops/tools/` are the implementation,
  verification, and audit surfaces.
- `ops/configs/`, `ops/tools/`, and `ops/operations/` are the canonical
  operational surfaces.
- `evidence/` contains retained classifications, proof records, and historical
  review artifacts. Local `runs/`, `artifacts/`, `models/`, `data/`, and
  `cache/` directories are ignored runtime state, not tracked authorities.

The canonical tree has five maintained areas: `src/`, `tests/`, `ops/`,
`docs/`, and `evidence/`. Root-level metadata and entry files are intentionally
few. Historical tag manifests may still name predecessor paths; verifiers map
those names to the canonical checkout without changing the historical record.

Historical experiment prose, staging handoffs, and reviewer contracts are
under [`docs/archive`](docs/archive/README.md). Nothing in that archive is an
active launch instruction; code references to a few archived documents are
frozen source-input contracts and are listed in the archive index.
