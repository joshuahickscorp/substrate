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

The read-only `substrate status` path keeps receipt state bounded to one
per-invocation scan and leaves expensive synthesis, external tools, and large
corpora behind explicit commands. This improves operator feedback latency
without changing the fail-closed activation, resource, or evidence gates.

## Performance posture

The latest bounded pass removes duplicate canonical-JSON work, caches the
process-scoped source-commit lookup, reuses detached event-state copies, and
avoids sorting mapping keys before the canonical encoder sorts them. Local
parity, seal, activation-false, event-isolation, and checkpoint tests preserve
the existing evidence contracts; scientific classifications and activation
boundaries are unchanged. Measured details are recorded in the implementation
history and are not claims about external hardware or campaign throughput.

Representative local profiles for the pass were:

- A 12.25 KB canonical-JSON payload took about 645 ms for 2,000 old-path
  encodes and about 245 ms through the direct encoder, with byte-parity tests.
- A 64-event permanent-state profile moved from about 78 ms to about 51 ms
  after redundant normalization and copying were removed.
- The next reducer-copy profile moved from about 22 ms to about 8 ms under
  cProfile by cloning only mutable state branches; a regression test checks
  that prior nested branches remain unchanged.
- Internal detachments now copy already-normalized trees directly, removing
  another JSON round-trip from reducer, snapshot, projection, and restore
  paths while keeping external normalization and seal validation unchanged.
- Retaining the payload-boundary activation scan but replacing the redundant
  post-reduction full-state walk with a strict root invariant moved the same
  64-event build profile from about 2.35 ms to about 0.96 ms per build; the
  cProfile view moved from roughly 9 ms to roughly 3 ms.
- Twelve repeated checkpoints over an 80-event context moved from about 209 ms
  to about 57 ms after the process-scoped source-identity lookup was cached.
- In the same local 12-checkpoint/80-event harness, caching the unchanged state
  digest moved the current batch median from about 15.6 ms to about 14.1 ms.

These are bounded local profiles, not claims about campaign-scale throughput;
the source, seal, and state-isolation boundaries remain explicit.

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
| `docs/` | Current architecture, reproduction guide, scientific status, product contracts, and historical context. |
| `evidence/` | Retained classifications, proof ledgers, and reviewable historical artifacts. |

The repository root contains only project metadata and entry files. Mutable
execution namespaces such as `runs/`, `artifacts/`, `models/`, `data/`, and
`cache/` are local ignored runtime state; they are not a second source tree or
scientific authority. Historical tag paths remain recognizable to the
independent verifiers, which map them to the canonical checkout only during
current-filesystem comparison.

For the shortest path through the repository, read the [architecture](docs/ARCHITECTURE.md),
then the [reproduction guide](docs/REPRODUCTION.md) and [scientific status](docs/SCIENTIFIC_STATUS.md).
The non-executing product foundation is documented in [docs/product](docs/product/ARCHITECTURE.md).
The [archive index](docs/archive/README.md) explains which historical documents
remain as evidence context rather than active instructions.

## Evidence

- [Genesis II classification](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_FINAL_CLASSIFICATION.json)
- [Genesis II claims](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_CLAIMS.json)
- [Genesis II mutation results](evidence/substrate/genesis2/SUBSTRATE_GENESIS2_MUTATIONS.json)
- [Tangible-sandbox classification](evidence/substrate/tangible_sandbox/SUBSTRATE_SANDBOX_FINAL_CLASSIFICATION.json)
- [Genesis II report](docs/SUBSTRATE_COGNITIVE_MATERIAL_GENESIS_II_REPORT.md)

All claims are bounded by the cited artifacts. Nothing in this repository is a
claim of unrestricted real-world ability or of a live autonomous system.
