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

The latest bounded pass removes duplicate canonical-JSON work, caches
process-scoped source-commit and source-inventory lookups across versioned
v2–v5 paths, reuses detached event-state copies, and
avoids sorting mapping keys before the canonical encoder sorts them. Frozen
producer and independent-verifier paths also reuse bounded deterministic
request/reference identities. Local
parity, seal, activation-false, event-isolation, and checkpoint tests preserve
the existing evidence contracts; scientific classifications and activation
boundaries are unchanged. Measured details are recorded in the implementation
history and are not claims about external hardware or campaign throughput.

Representative local profiles for the pass were:

- A 12.25 KB canonical-JSON payload took about 645 ms for 2,000 old-path
  encodes and about 245 ms through the direct encoder, with byte-parity tests.
- Structural stale-evidence checking now loads the exact ancestor set of `HEAD`
  once per process instead of spawning one `git merge-base` subprocess per
  sealed artifact. In a fresh audit-plus-independent-verification profile, the
  combined path moved from about 174 to 148 ms; reachable, stale, and malformed
  evidence refusal checks remain unchanged.
- Structural audit source checks now share one immutable in-process source
  snapshot between producer ownership and activation scanning. Across 12 local
  `audit.run()` calls, the repeated audit moved from about 75.0 to 67.5 ms per
  call; the producer and activation reports remain exact.
- Terminal synthesis `status` and dependency readiness now census the receipt
  directory once per invocation instead of statting every declared unit
  separately. In a warmed 1,000-call loop, `status` moved from about 151 to
  73 microseconds per call with the same validated receipt and dependency
  results.
- Versioned v2–v4 provenance now pays for its immutable source snapshot once per
  process. Across eight same-process calls, v2 context construction moved from
  about 148 to 1.5 ms and v3 manifest construction from about 231 to 82 ms;
  commit and source-digest parity remained exact.
- Nous Closure authority construction now applies the same process-scoped cache
  to its exact implementation/configuration snapshot. In two paired no-publish
  pilot trials, 13 digest calls moved from 3.34–3.40 to 3.15–3.21 seconds with
  identical report and admission bytes; the cold snapshot remains about 19 ms.
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
- Public event appends now reuse the normalized detached projection already
  needed for persistence instead of normalizing the return value again. In a
  matched 5,000-context-event loop, wall time moved from about 61.4 to 53.3
  milliseconds locally; the general ``CognitiveEvent.to_dict()`` normalization
  boundary remains unchanged.
- Exact JSON event payloads now use a sorted recursive detacher instead of a
  JSON dump/parse cycle plus a second activation walk; noncanonical values keep
  the prior normalization and refusal path. In a matched 250-entity profile
  with 16 context updates per entity, append construction moved from about 55.9
  to 47.5 milliseconds (15.1%), with key-order, activation, and tuple parity
  checks retained.
- Sensor-event digesting now reuses its fixed canonical JSON encoder instead of
  rebuilding one per receipt. The representative digest moved from roughly 5.0
  to 4.7 microseconds locally, with an explicit byte-compatibility test.
- The frozen `v5experiment` path now shares the same digest encoder, reads the
  binary SHA-256 prefix directly, reuses precomputed capability projections,
  and carries each sensor-event digest forward instead of hashing the same
  event again for the aggregate receipt. Representative full-v5 phase 19
  execution moved from roughly 11.3 to 8.5 milliseconds across this pass, with
  exact output parity across the sampled phases.
- Cached frozen-v5 sensor events now recheck the mutable observation target
  boundary while skipping repeated serialization of their already-validated
  empty typed layers. In a fresh 8-history, 14-arm matched evaluation, wall
  time moved from about 5.99 to 5.74 seconds with the exact table digest
  unchanged; events containing optional typed layers retain the full ingest
  path.
- Cached frozen-v5 event copies now clone the validated frozen shell directly
  while detaching only the public observation mapping. Across five fresh
  8-history, 14-arm phase-19 evaluations, the median moved from 0.287947 to
  0.285946 seconds against the prior copy path (about 0.7%) with identical
  aggregate digest `e188e71e26a1f5308c16bc63d01878ee47c4d3cca50d0e57c2460cab643eed57`;
  the public ingest path remains fully validating.
- Flat scalar model requests now use a shape-proven authority-scan fast path;
  nested, non-string-keyed, and malformed payloads retain the recursive
  defensive walker. Across ten paired fresh-process phase-19 evaluations, the
  median moved from 0.282095 to 0.277948 seconds (about 1.5%) with the exact
  aggregate digest `9f5f3d399ddaa427d4e6810078faa8a518cf0baaa9fcde3fc120a455c16bbea9`.
- Repeated modality requests now reuse bounded immutable request objects across
  arm evaluations, while the public model request type remains unchanged. A
  fresh ten-pair phase-19 comparison moved the median from 0.274914 to
  0.261994 seconds (about 4.7%) with the same exact aggregate digest.
- Repeated v4 structural surface graphs now reuse bounded immutable canonical
  role mappings while returning a fresh mapping to every caller. Across ten
  paired fresh-process 112-row phase-19 evaluations, the median moved from
  1.390540 to 1.280982 seconds (about 7.9%) with the exact aggregate digest
  `0639e304b217a0d32f561edc4327d7366beb0c9f0b4ef92af965e1d426ad617c`.
- The default model fabric now caches its immutable contract and relationship
  definitions while still creating fresh per-registry modules, preserving
  isolated call counts. Registry construction moved from roughly 52 to 3
  microseconds locally, and a representative full-v5 phase 19 run moved from
  about 8.2 to 7.8 milliseconds with identical receipt bytes.
- Independent verification now caches the deterministic v4-retention probe
  behind the same bounded pure-function boundary; repeated verifier phase-19
  work moved from roughly 176 to 8 milliseconds locally with identical
  independently reconstructed receipt bytes.
- The Nous Closure pilot now disables the terminal v4-retention probe when its
  candidate-history reducer discards that field; the `phase_result` default
  remains enabled for callers that consume it. The representative pilot moved
  from about 8.56 to 3.18 seconds with the exact report digest unchanged.
- Frozen producer and independent-verifier request paths now cache bounded
  task identities and generated sensor references, removing 40–60 duplicate
  digest computations from representative phases. A same-process phase-15
  profile moved from roughly 5.2 to 4.8 milliseconds for the producer and
  4.8 to 4.3 milliseconds for the verifier, with exact output parity across
  the sampled phases.
- The frozen producer now uses the sensorium’s combined validate-and-digest
  operation, so each event’s public observation is materialized once instead
  of once for validation and again for the receipt digest. Representative
  phase-19 execution moved from roughly 7.6 to 7.2 milliseconds with exact
  output parity.
- The per-registry model router now caches only its shape-dependent ranking
  inputs, while rebuilding each task-bound decision and invalidating on model
  registration. A bounded 20,000-call route profile moved from roughly 47.2
  to 11.6 milliseconds; representative phase-15 producer and verifier work
  moved from about 4.66 to 4.43 and 4.70 to 4.28 milliseconds respectively,
  with exact decision and receipt parity.
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
- First-time permanent-entity checkpoint sealing now recognizes exact normalized
  JSON trees and skips the redundant normalize/parse round trip, while falling
  back to the general sealer for mutated or non-canonical public values. In a
  matched 250-entity profile with 16 context updates per entity, the checkpoint
  batch median moved from about 106.6 to 88.4 milliseconds (17.1%, or roughly
  426 to 354 microseconds per checkpoint); actual-checkpoint seal parity and
  caller-mutation isolation remain covered.
- Independent v5 regeneration now walks layered sensor keys with one iterative
  accumulator and uses a single validated ingest-and-digest observation rather
  than rebuilding the same public body twice. A representative one-shard
  verifier run measures roughly 28 ms per regeneration after these reductions,
  with the existing exact-reproduction and hidden-target tests unchanged.
- Independent v5 verification now projects already-detached sealed receipt and
  checkpoint trees when removing envelope fields; source rebinding keeps a
  recursive detached-copy path with a defensive fallback for non-canonical
  callers. In a matched 39.9 KB checkpoint projection profile, 1,000 seal
  strips moved from roughly 0.425 to 0.000388 seconds (99.91%), with targeted
  raw-regeneration and checkpoint-chain parity still exact.
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
- Typed optional sensor layers now use explicit field serializers with a
  JSON-shaped detacher for mapping fields instead of generic dataclass walking.
  In a matched local harness, proposal-only public observation moved from about
  18.5 to 17.3 microseconds, while a fully populated optional-layer observation
  moved from about 46.6 to 33.2 microseconds; dataclass-shape and mutation
  isolation parity remain covered.
- The recurrent and hybrid v5 kernels now cache bounded modality/signal digest
  inputs, and normalized state copies use exact built-in dict/list fast paths
  with defensive subclass fallbacks. Kernel checkpoint construction now
  detaches mutable branches before sealing instead of walking the scalar root
  envelope a second time. The matched kernel comparison moved from about 2.44
  to 1.63 milliseconds per 100-candidate benchmark batch; the latent-value and
  checkpoint-isolation contracts remain unchanged.
- Normalized state detachment now avoids redundant ABC dispatch on the common
  built-in tree. A representative 100,000-copy profile moved from about 159
  to 138 milliseconds without changing the defensive copy boundary.
- Hidden-target scans now return after one pass for the common flat sensor
  observation shape, while nested mappings and sequences retain the complete
  recursive walk. A direct 30,000-call flat-observation profile moved from
  roughly 27.4 to 19.1 milliseconds with identical leak detection.
- Clean clones now read retained current experiment receipts when the mutable
  `runs/` tree is absent, and their declared test gate is the reproducible
  normal tier. Certification, integration, and corpus-heavy checks remain
  explicit gates; the one historical supervisor log required by the live
  session authority is tracked under `docs/archive/`, while disposable logs
  stay ignored.
- Model-request validation now accumulates only forbidden outcome-authority keys
  while retaining the same recursive payload walk. On representative flat and
  nested public payloads, validation moved from roughly 0.49 to 0.40 and 1.21
  to 1.13 microseconds respectively; hidden-key refusal tests remain in the
  normal suite.
- The model-request forbidden-key walk now short-circuits exact built-in scalar
  leaves while retaining defensive custom-container fallbacks. In a matched
  100,000-call profile, flat payload scanning moved from about 129 to 95 ms and
  nested scanning from about 397 to 298 ms; subclass traversal is explicitly
  covered by refusal tests.
- Twelve repeated checkpoints over an 80-event context moved from about 209 ms
  to about 57 ms after the process-scoped source-identity lookup was cached.
- In the same local 12-checkpoint/80-event harness, caching the unchanged state
  digest moved the current batch median from about 15.6 ms to about 14.1 ms.
- Legacy v1–v4 evidence hashing now uses CPython's C JSON encoder while
  preserving the historical byte format and cycle-refusal fallback. Across
  five paired fresh-process 14-arm v4 evaluations, median wall time moved from
  1.562379 to 1.374530 seconds (about 12.0%); byte-parity and refusal tests
  remain explicit.
- Deterministic v4 task templates now use a bounded 4,096-entry process cache
  and are detached before returning to callers. Across five fresh-process
  14-arm evaluations, median wall time moved from 1.315786 to 1.261848 seconds
  (about 4.1%); task identity and mutable-output isolation remain explicit.
- v4 task-target generation now computes topology roots, causal endpoints, and
  canonical scope only for workload branches that consume them. In a matched
  five-round mixed-family profile of 1,000 uncached tasks per round, median wall
  time moved from 0.063891 to 0.046388 seconds (about 27.4%); a 2,400-case
  cross-split, cross-family output-parity matrix remains exact.

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
