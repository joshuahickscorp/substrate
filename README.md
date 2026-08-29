# Substrate

Substrate is an experimental Python framework for testing computational
hypotheses about persistent memory, perspective, metacognition, world models,
continuity, adaptive structure, identity-like state, and agency. It does not
claim sentience or consciousness. The current sealed results are null or
unproven, and activation remains disabled.

## Research method

The project uses seeded, deterministic campaigns, explicit controls and
baselines, hash-sealed artifacts, independent recomputation, and mutation tests
that must reject tampered evidence. Historical campaigns cover v1–v5, Genesis,
Genesis II, Nous Closure, the tangible sandbox, and Odyssey gate scaffolding.

The latest Genesis II classification is
`cognitive_material_genesis_ii_complete` with status
`compositional_advantage_unproven`. Four of ten primary claims passed; the
campaign covered 4,245,640 episodes; and all 17 planted defects were detected.
The selected simpler associative monolith did not establish a compositional
advantage over its baseline.

The tangible-sandbox record remains classified as Outcome C,
`terminal_tangible_sandbox_null`, after a protected-resource refusal. Later
acquisition and canary receipts do not overwrite that scientific classification.

These results are scope-specific, not cumulative claims. The v4, Final
Revision, and v5 records remain separately sealed historical campaigns; a later
experiment does not promote an earlier null, and an earlier positive result
does not establish activation or unrestricted capability.

## Frozen status map

| Scope | Supported readout | Authority |
|---|---|---|
| Core v4 runtime | `functional_proto_nous_candidate`; replication remains sub-SESOI null | [v4 classification](evidence/substrate/v4/SUBSTRATE_V4_FINAL_CLASSIFICATION.json) |
| Genesis II | `cognitive_material_genesis_ii_complete`; compositional advantage unproven | [Genesis II classification](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json) |
| Tangible Sandbox | `terminal_tangible_sandbox_null` after protected-resource refusal | [sandbox classification](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json) |

Every authority keeps `activation:false`. No report or fixture authorizes
uncontrolled external action, a live autonomous system, or a claim of
consciousness, sentience, personhood, life, or moral status.

## Current implementation

- `src/substrate/` contains memory, perspective, world-model, ontology,
  metacognitive, plasticity, safety, continuity, runtime, and sandbox modules.
- Campaign harnesses implement the v2–v5, Genesis, Genesis II, Nous Closure,
  Final Revision, tangible-sandbox, and Odyssey workflows.
- `src/substrate/verification.py` independently recomputes sealed records and
  exercises mutation attacks in fresh processes.
- The separate `substrate product` surface records specialist state and plans;
  it does not launch a campaign, container, or tool.

## Run the checks

From the repository root:

```bash
make install
make test-qualification
make test-normal
make test-integration     # requires the declared external services/tools
make test-expensive
make test-full
make audit
make accept
```

The qualification tier is the seconds-scale contract gate for deterministic
core invariants. Normal runs the ordinary package suite; integration and
expensive tiers are explicit about external tools, long campaigns, or large
corpora; full is the complete certification collection. Unavailable external
dependencies are reported as unavailable or skipped by the owning test and
never converted into a passing result.

Use `substrate --help`, `substrate status`, `substrate genesis --help`, and
`substrate sandbox --help` to inspect the available workflows. `make accept`
runs the fail-closed verification path; some historical proofs require their
raw run tree and can therefore reject a fresh or incomplete checkout.

## Canonical repository layout

| Area | Purpose |
|---|---|
| `src/` | The installed `substrate` package and native sandbox implementation. |
| `tests/` | Package, invariant, campaign, and independent-verification tests. |
| `ops/` | Frozen configuration, operational records, tools, and launch/run helpers. |
| `docs/` | Current architecture, development rules, runbooks, product contracts, and historical context. |
| `evidence/` | Retained classifications, proof ledgers, and reviewable historical artifacts. |

The repository root contains only project metadata and entry files. Mutable
execution namespaces such as `runs/`, `artifacts/`, `models/`, `data/`, and
`cache/` are local ignored runtime state; they are not a second source tree or
scientific authority. Historical tag paths remain recognizable to the
independent verifiers, which map them to the canonical checkout only during
current-filesystem comparison.

The [navigation map](docs/START_HERE.md) is for browsing the tree. The
[archive index](docs/archive/README.md) explains which historical documents
remain as evidence context rather than active instructions.

## Evidence

- [Genesis II classification](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json)
- [Genesis II claims](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_CLAIMS.json)
- [Genesis II mutation results](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_MUTATIONS.json)
- [Tangible-sandbox classification](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json)
- [Genesis II report](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md)

All claims are bounded by the cited artifacts. Nothing in this repository is a
claim of unrestricted real-world ability or of a live autonomous system.
