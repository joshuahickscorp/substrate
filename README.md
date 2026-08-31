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
- Copy-on-write reducer setup now uses shared module-level helpers instead of
  creating two closures for every event. In an isolated 10,000-call local
  profile, branch preparation moved from roughly 0.55 to 0.43 microseconds for
  context transitions, with the goal and memory shapes moving from roughly
  0.48 to 0.42 and 0.57 to 0.46 microseconds respectively; state isolation
  tests remain unchanged.
- Internal detachment of already-canonical JSON trees now uses a focused
  dict/list copy instead of generic `copy.deepcopy`; this preserves the public
  snapshot boundary while making representative tree copies about 3.2x faster.
- The bounded v5 kernel candidates now use the same focused tree detacher for
  checkpoint and restore state, with generic copying reserved for non-tree
  leaves. On a representative hybrid checkpoint body, focused detachment is
  roughly 13.1 microseconds versus 25.0 microseconds for generic copying, and
  checkpoint-isolation coverage remains explicit.
- Independent v5 regeneration now reuses its configured stable JSON encoder and
  precomputes phase, shard-modality, and active-requirement projections. A
  warmed one-shard regeneration moved from roughly 20.7 to 19.4 milliseconds
  locally, with exact verifier and digest-parity tests retained.
- Permanent-entity checkpoint cache hits now retain canonical bytes and use the
  JSON loader as the detached-copy boundary. A representative repeated
  checkpoint call moved from roughly 84.6 to 61.5 microseconds locally, while
  caller-mutation isolation and exact restore checks remain covered.
- Sensor-event digesting now reuses its fixed canonical JSON encoder instead of
  rebuilding one per receipt. The representative digest moved from roughly 5.0
  to 4.7 microseconds locally, with an explicit byte-compatibility test.
- Bounded workspace projection uses one full encoding when its first
  `max_items` candidates fit, then an exact branch-size ledger for tight bounds;
  deterministic candidates are generated on demand rather than materialized in
  a second full list. In the same 64-event/32-goal local harness, the wide
  projection moved from roughly 623 to 63 microseconds per call, while the
  2 KB bounded case moved from roughly 350 to 286 microseconds;
  bounded-selection parity matched the prior greedy implementation across
  thousands of byte limits.
- Checkpoints now cache the sealed snapshot for the unchanged event chain while
  returning a detached tree and invalidating on append. With source metadata
  warmed, an 80-event/48 KB profile moved repeated reads from roughly 1.24 ms
  to 0.235 ms each; state/event mutation checks still force a rebuild, and exact
  restore, persistence, and caller-mutation tests remain green.
- Independent v5 regeneration now walks layered sensor keys with one iterative
  accumulator and uses a single validated ingest-and-digest observation rather
  than rebuilding the same public body twice. A representative one-shard
  verifier run measures roughly 28 ms per regeneration after these reductions,
  with the existing exact-reproduction and hidden-target tests unchanged.
- Flat raw and preprocessed sensor layers now materialize directly from their
  typed fields instead of invoking recursive dataclass conversion. The same
  warmed one-shard harness measures roughly 27 ms per regeneration, with a
  parity test pinning the serialized layer shape.
- Sensor-event construction now audits only its mapping-bearing fields for
  hidden-target leakage instead of materializing every typed layer before the
  public serializer performs its complete audit. The same warmed one-shard
  harness measures roughly 24 ms per regeneration, with construction-time and
  post-construction leakage tests retained.
- Events with no optional typed layers now recheck the mutable observation
  mapping directly instead of walking their scalar envelope and empty layer
  lists. The warmed one-shard harness measures roughly 21 ms per regeneration;
  events carrying optional layers still use the complete public-body audit.
- Model-request validation now accumulates only forbidden outcome-authority keys
  while retaining the same recursive payload walk. On representative flat and
  nested public payloads, validation moved from roughly 0.49 to 0.40 and 1.21
  to 1.13 microseconds respectively; hidden-key refusal tests remain in the
  normal suite.
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
