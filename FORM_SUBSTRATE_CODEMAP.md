# FORM SUBSTRATE CODEMAP

Status note (2026-07-10): this began as a migration-time audit. The live implementation order is
`MOP_MAXIMUM_POTENTIAL_EXECUTION_PLAN.md`; the scored state is `MOP_POTENTIAL_ATLAS_2026_07.md`.
The table now records the shared Wave E0, dense ViT-B, and adaptive-governor seams added by the
maximum-potential run.

The code audit behind the paradigm migration: every existing module mapped to its form-substrate role.
Companion docs: FORM_SUBSTRATE_PROGRAM.md (worldview), PARADIGM_MIGRATION.md (doc migration),
MIGRATION_PHASES.md (phase order), FORM_SUBSTRATE_EXPERIMENTS.md (F-series bank).

House style: no em or en dashes. This document renames concepts, not files. Physical renames are
gated to late phases (MIGRATION_PHASES.md Phase 5+) after tests, imports, and the registry are stable.

---

## 0. The one structural fact that dominates the migration

The repo already contains three parallel "present anything behind one extract() contract" stacks:

| Stack | File | Unit | Status |
|---|---|---|---|
| SubstrateAdapter | src/mop/substrate/adapter.py | one frozen encoder, extract(clips) -> [N,D] | production, wired to caches |
| PerspectiveAdapter | src/mop/perspectives/adapter.py | one view over shared referents | production, wired to DR1/studio lanes |
| FormAdapter | src/mop/substrate/form.py | one observation family over shared referents | new, consumed only by F-series |

FormAdapter is a near-structural clone of PerspectiveAdapter: same referent vocabulary, same
control_for audit pattern, same matrix builder shape, and `_referent_tuple` / `_factor_dict` are
byte-identical copies. What form.py adds that perspectives lacks: closed vocabularies (FORM_KINDS,
OBJECTIVE_FAMILIES), `kinds()`, token_shape and trainable honesty flags, and the affine alignment
primitives (`fit_affine_alignment` / `apply_affine_alignment`). What perspectives has that form.py
lacks: the load-bearing bridges (`LatentStorePerspectiveAdapter`, `SubstratePerspectiveAdapter`,
`PerspectiveRegistry`) and all the real-infrastructure consumers (dr1_perspectives, native_lanes,
dr1_curate_bound_video).

Reconciliation policy (binding):

1. Phase 0-1 (now): form.py is the interface of record for all NEW work. PerspectiveAdapter is
   conceptually renamed "form perspective layer" in docs only. No physical rename.
2. Phase 2-3: one referent-aligned matrix stack survives. Target end state: `FormMeta` becomes a
   superset of `PerspectiveMeta` (kind, objective, token_shape, trainable land there), the bridge
   adapters (`LatentStorePerspectiveAdapter`, `SubstratePerspectiveAdapter`) are ported to the form
   interface, and the duplicated helpers collapse to one definition. Whether the surviving file is
   form.py or adapter.py is decided by import blast radius at that time, not now.
3. A third parallel ABC is never added. Any new observation family enters as a FormAdapter (or a
   PerspectiveAdapter until the merge), never as a new hierarchy.

---

## 1. Module map

Legend. Action: keep (as is), extend (add capability in place), wrap (present behind form interface),
demote (legacy instrument, keep runnable), merge (collapse duplicates), rename-docs (conceptual only).
Priority: P0 (this phase), P1 (next), P2 (later), P3 (gated/optional).

### 1.1 Substrate and form layer

| Module | Current purpose | Form-substrate role | Action | Risk | Tests | Perf note | Pri |
|---|---|---|---|---|---|---|---|
| substrate/form.py | form contract, matrix, audit, affine alignment | Layer 1 Form Intake + Layer 2 Referent Binder (interface of record) | keep, extend with store-backed adapters | clone drift vs perspectives | test_form_substrate | pure torch, cheap | P0 |
| perspectives/adapter.py | referent-aligned multi-arm views + registry + bridges | production backend of the form layer, merge target | merge into form interface (Phase 2-3) | wide import surface (DR1 lanes) | test_perspective_adapter | none | P1 |
| substrate/adapter.py | frozen-encoder substrates + random-init controls | InheritedSubstrateAdapter family: inherited frozen arms behind the form interface | rename-docs, wrap | none | test_substrate_ablation | encode cost dominates | P1 |
| substrate/encoder.py | V-JEPA load, frozen, random-projection fallback | one inherited perceptual arm among many | keep | privilege creep back to V-JEPA | test_substrate | mmap + MPS batch | P2 |
| substrate/latent_store.py | memmap [N,D] and [N,T,D] store | FormFeatureStore (the form store; no new module) | rename-docs, extend with form-kind metadata | second layout back-compat | test_substrate | mmap viability proven | P0 |
| substrate/cache_manifest.py | data-plane receipts, fingerprints, sidecars | form provenance contract (per-form manifests) | extend (form kind + objective fields) | schema version bump | test_cache_manifest | full-hash cost on big caches | P1 |
| substrate/storage.py | cache byte accounting, prune | Layer 9 cost accounting input (bytes) | keep | none | test_storage | none | P0 |
| substrate/cache.py, cache_tools.py | encode-once caching, integrity | form intake for cached arms | keep | none | test_cache_tools | encode-once doctrine | P2 |
| substrate/real_latent.py, video.py, datasets.py, fixtures.py | real video decode, streams, synthetic worlds | vision-form and toy-form sources | keep | none | test_real_latent, test_video | decode throughput | P2 |
| substrate/encoder_registry.py | encoder honesty registry | form-arm honesty registry input | keep | none | test_encoder_registry | none | P2 |
| environments/persistent_grid.py | deterministic persistent local actions, consequences, and cloned-state alternatives | shared causal trajectory instrument for F6, F15, E5, and CM10 mechanics | keep, programmatic-only | synthetic evidence overclaim | test_persistent_grid_environment | bounded CPU, content-addressed | P0 |
| substrate/events.py | immutable event/entity/observation/branch/sensor-clock graph | shared referent, clock, lineage, and intervention identity for Wave E0 and later forms | keep, extend only through schema versioning | identity fork or silent ambiguity | test_expansion_wave0 | canonical JSON and hashes, CPU-only | P0 |
| substrate/lifecycle.py | hash-linked append-only memory lifecycle | shared revision, availability, deletion, conflict, poisoning, and rollback journal | keep, event-bound | ungoverned mutation or false deletion | test_expansion_wave0 | bounded replay, CPU-only | P0 |
| substrate/vjepa21_official.py, vjepa21_dense_tasks.py | official dense ViT-B loader plus cache/control seam | frozen dense measurement instrument, exact random architecture control, and strict same-input cache pair | keep frozen, cache-first | inherited instrument re-centers the platform | test_vjepa21_official, test_vjepa21_dense_tasks | serial CPU path measured | P1 |
| environments/scenario_factory.py | deterministic occlusion, transform, split, merge, delay, and intervention fixtures | shared event/intervention mechanics factory | keep, programmatic-only | fixture result overclaimed as natural transfer | test_expansion_wave0 | tiny CPU fixture | P0 |

### 1.2 Shell, memory, routing, plasticity (the mode ecology)

| Module | Current purpose | Form-substrate role | Action | Risk | Tests | Perf note | Pri |
|---|---|---|---|---|---|---|---|
| shell/workspace.py | WorkspaceShell composition | FormWorkspace (Layer 5 host); composition-only rule stands | rename-docs | parallel-shell temptation | test_shell | none | P1 |
| shell/heads.py | ClassHead, MoE routing, KWTA | mode routing primitive (MoEHead) for Layer 5 | keep | routing wins must beat best-single-mode | test_shell | active-FLOP accounting | P1 |
| shell/buffer.py | PER replay + KVIndex | cross-form memory backend for F5/F19 | extend (referent-keyed retrieval) | faiss fallback on Apple Silicon | test_shell | retrieval latency metric | P1 |
| shell/consolidation.py | EWC, SI | old-form retention machinery (F14) | keep | none | test_shell | none | P2 |
| shell/plasticity.py | PlasticityController, PNN, reopening | Layer 10 plastic rewrite gate substrate-side | keep, gated | overclaim risk | test_shell | none | P3 |
| shell/neuromod.py | DA/ACh/NE gain model | legacy instrument (negative record) | demote | reviving dead lever | test_shell | none | P3 |
| shell/refine.py | iterative refinement + halting | Layer 7 compute-value estimation (adaptive halting) | keep | matched-compute rule | test_refine | active vs total FLOPs | P1 |
| shell/ensemble.py | disagreement uncertainty | OA input (epistemic signal) | keep | none | test_shell | ensemble cost | P1 |
| shell/verifier_exec.py | DSL executable verifier | verifier mode (Layer 5) | keep | none | test_shell | none | P2 |
| shell/capmatch.py | capacity matching | matched-capacity control builder everywhere | keep | none | test_shell | none | P0 |
| shell/predictor.py, modulation.py | latent predictor, gating, WM slots | mode primitives | keep | none | test_shell | none | P2 |
| learning/backprop.py | Learner, continual engine | shell trainer (unchanged) | keep | no dedicated unit test | integration only | none | P2 |
| learning/alternatives.py | 7 local rules (I4) | legacy instrument (negative record) | demote | none | test in integration | none | P3 |
| process_c/dense_tokens.py | gated trainable dense-token pilot | the licensed pattern for any trainable substrate arm (F8 precursor) | keep, gated | license discipline | test_process_c_dense_tokens | dense token memory | P3 |

### 1.3 Diagnostics (the gates)

| Module | Current purpose | Form-substrate role | Action | Pri |
|---|---|---|---|---|
| diagnostics/alignment.py | AlignmentSuite, permutation null | cross-form alignment gate (F1 backend at scale) | extend | P1 |
| diagnostics/cross_substrate.py | CrossSubstrateAgreement | cross-form agreement (rename-docs) | rename-docs | P1 |
| diagnostics/transfer_matrix.py | transfer matrices | form transfer diagnostics (F2 backend) | keep | P1 |
| diagnostics/held_out_combo.py | compositional held-out combos | F9 backend | keep | P1 |
| diagnostics/linear_probe.py, nonlinear_probe.py | decodability, readout DiD | factor preservation gates | keep | P0 |
| diagnostics/calibration.py, riskcov.py | ECE, AUROC, AURC, Pareto area | OA2 confidence calibration + selective routing | keep | P0 |
| diagnostics/compute.py | FLOP and param accounting | performance-density input (matched compute) | extend | P0 |
| diagnostics/noisy_tv.py | epistemic vs aleatoric guard | OA7 rewrite caution + F10 control | keep | P0 |
| diagnostics/seed_consistency.py | cross-seed CKA, code agreement | F12 backend, seed gates | keep | P1 |
| diagnostics/difficulty_calibration.py | ceiling trap guard | mandatory gate on all F-series | keep | P0 |
| diagnostics/bottleneck.py | capability per bit | F3/F13 backend | keep | P1 |
| diagnostics/geometry.py, convergence.py, sysid.py, fisher_trace.py, hardness.py, continual_metrics.py, buffer_compression.py, latent_robustness.py, determinism.py | geometry, dynamics, controllability, criticality, graded tasks, BWT/FWT | unchanged gate library | keep | P2 |
| diagnostics/substrate_ablation.py | substrate specialness controls | non-vacuous form controls (random arms) | keep | P0 |

### 1.4 Experiments

| Module | Current purpose | Form-substrate role | Action | Pri |
|---|---|---|---|---|
| experiments/base.py | Experiment ABC, contract enforced at import | unchanged spine (F-series already conforms) | keep | P0 |
| experiments/f_form_substrate.py | F1, F2, F3, F5 runnable | the active bank; F4/F9/F17 land here next | extend | P0 |
| experiments/e*_.py, ex*.py | conducted bank + bleeding edge | evidence record + reusable harnesses | keep | P2 |
| experiments/a_,b_,c_,d_,i_,n_,p_,s_,y_ files | lens families (perception, biology, cogsci, developmental, infotheory, neuro, philosophy, semiotics, dynamics) | lens banks re-pointed at form matrices where they survive | keep, demote per PARADIGM_MIGRATION.md | P2 |
| experiments/scaffolds.py | registers ~95 scaffolds | unchanged registrar | keep | P0 |
| experiments/expansion_harness.py | shared arm/control/resource/unit/stop contract | Wave E0 and later thin F specifications over one evidence plane | keep, mechanics-first | bespoke experiments fork identity or controls | P0 |
| experiments/e6_relational.py | relational dense-token experiment | strict cache-first learned/random dense readout with no default legacy fallback | keep, natural-input-gated | availability promoted as a result | P1 |

### 1.5 Execution and evidence governance

| Module | Role | Action |
|---|---|---|
| studio/long_run.py, claim_plan.py, spine_plan.py | daemon, gated claim sequences, machine-readable spine | keep |
| studio/scorecard.py, objective_audit.py, artifact_bundle.py | receipt-backed scoring, objective audit, durable index | keep |
| studio/density_receipt.py, disk_recovery.py | repo mass receipts, auditable reclaim | keep (feeds Layer 9) |
| studio/pr9_verdict.py, atlas_verdict.py, dense_atlas_gate.py, process_c_gate.py | verdict ledgers and license gates | keep (Layer 10 pattern) |
| studio/dr1_source_intake.py, dr1_verifier.py, dr1_perspectives.py, dr1_schedule.py | DR1 lane: real bound-attribute forms intake and verification | keep (Phase 3 real-forms source) |
| studio/profiles.py, encode_scheduler.py, memory_envelope.py, native_lanes.py | hardware envelopes and lanes | keep (perf kernel) |
| falsification/null_cards.py, verdict_gate.py | null cards + two-key verdicts | keep (mandatory for F-series promotions) |
| devel/registries.py | preregistration registry + sentience rail | extend (F17-F20 rows) |
| devel/metacognition.py | inspectable self-monitoring report, rail-gated | seed of diagnostics/operational_awareness.py |
| devel/north_star.py | sentience rail + engineering vocabulary | keep (OA doctrine depends on it) |
| devel/curriculum.py | learning-progress lesson selection | F10 backend |
| studio/local_throttle.py | adaptive admission, monitoring, pause/resume, and atomic run receipts | current-host resource governor; signals only its owned process group |
| studies/potential_atlas_validation.py | recompute scores, evidence hashes, queue, dependency, partition, and hardware gates | fail-closed atlas validator and opt-in atomic source-hash refresh |

---

## 2. Shared seams and implemented expansions

Everything else in the master plan's proposed file list already exists under another name. Refusals
first, so the duplicates never get written:

| Proposed file | Verdict | Existing home |
|---|---|---|
| substrate/referent.py | refuse | referent logic lives in build_form_matrix / build_perspective_matrix; a second identity primitive would fork alignment |
| substrate/form_store.py | refuse | latent_store.py + cache_manifest.py are the form store; extend, never duplicate |
| diagnostics/form_alignment.py | refuse | form.fit_affine_alignment + diagnostics/alignment.py |
| diagnostics/referent_binding.py | refuse | form_audit + perspective_audit + shuffled-referent controls in F1/F5 |
| diagnostics/form_transfer.py | refuse | diagnostics/transfer_matrix.py + F2 |
| workspace/form_workspace.py | refuse | shell/workspace.py WorkspaceShell (composition-only rule) |
| routing/mode_router.py | defer (P3) | MoEHead + router scripts; a dedicated module is licensed only by a WS/AL-series positive |
| plasticity/substrate_rewrite.py | defer (P3) | shell/plasticity.py + process_c license-gate pattern; licensed only by F8 gates |

Accepted new modules:

### 2.1 src/mop/diagnostics/performance_density.py (P0)

Job: one call that turns any experiment result into a density block: capability metric, cost metrics
(params, FLOPs, active FLOPs via diagnostics/compute.py, wall-clock, bytes via substrate/storage.py,
peak memory when available), and density ratios (score per FLOP, per byte, per second, per parameter,
per update). Every F-series result dict grows a `density` sub-dict. No new accounting math: compose
compute.py + storage.py + time. Test: tests/unit/test_performance_density.py.

### 2.2 src/mop/diagnostics/operational_awareness.py (P0)

Job: the OA1-OA8 metric suite (OPERATIONAL_AWARENESS.md), composed from existing instruments:
riskcov (AUROC/AURC selective competence), calibration (ECE), ensemble disagreement, noisy_tv guard,
metacognition report fields. Adds the missing composites: missing-form detection score, memory
availability score, mode-selection regret vs oracle and vs random, compute-value estimation. All
free text through north_star.assert_no_sentience_claims. Test: tests/unit/test_operational_awareness.py.

### 2.3 src/mop/experiments/form_rewrite_engine.py (P0)

Job: one fail-closed execution engine shared only by registry-only F8/F16. It consumes a rights-manifested
local tensor package, safe executable encoder weights, reproducible inherited features, bound prerequisite
receipts, and immutable seed/margin/compute plans. It runs the four candidate/control arms under one
explicit estimated end-to-end FLOP convention, preserves partial-attempt and resource receipts, and never
self-promotes natural provenance. This stays in experiments rather than adding the refused parallel
plasticity/substrate_rewrite.py abstraction. Tests: tests/integration/test_f_missing_lanes.py.

### 2.4 src/mop/environments/persistent_grid.py (P0)

Job: one deterministic local observation -> chosen action -> consequence adapter shared by F6, F15,
E5, and CM10. Every event carries stable world, episode, entity, state, event, and branch references,
action costs, affordances, episode boundaries, and all four paired alternatives evaluated from the
same cloned state. Seeded replay and canonical SHA-256 verification fail closed on mutation. It is a
programmatic mechanics instrument, never evidence of natural embodiment or open-endedness. Durable
preflight: proof/LOCAL_ACTION_ENVIRONMENT.json. Tests: test_persistent_grid_environment.py and
test_local_action_environment.py.

### 2.5 Wave E0 event, scenario, lifecycle, and harness seams (P0, implemented)

`substrate/events.py`, `environments/scenario_factory.py`, `substrate/lifecycle.py`, and
`experiments/expansion_harness.py` implement one immutable evidence plane for F23, F29, and F39.
`proof/EXPANSION_WAVE0.json` binds three shared units, nine controls, exact replay, and a separately
implemented mutation verifier. Scope remains deterministic mechanics only.

### 2.6 Official dense ViT-B task and cache seam (P1, implemented)

`substrate/vjepa21_official.py` and `substrate/vjepa21_dense_tasks.py` validate the retained official
bytes, frozen strict load, same-input learned/random plans, atomic row progress, cache manifests, and
E6/DR14 consumers. The no-heavy preflight is green. Natural task inputs and final cache/verifier
receipts remain deliberately absent.

### 2.7 Adaptive local governor (P0, implemented)

`studio/local_throttle.py` and `scripts/local_execution_throttle.py` enforce the 300-minute
operational maximum, one-heavy/two-total lane policy, three-sample admission, memory/swap/thermal/
power/MPS/disk/foreground gates, hysteresis, checkpoint capture, and process-group ownership. It
never signals discovered user processes.

### 2.8 Potential-atlas validator (P0, implemented)

`studies/potential_atlas_validation.py` recomputes all 37 facets, weights, S/I/E/C formulas and caps,
source hashes, evidence paths, queue and dependency references, category-2 partition, and hardware
gate agreement. Source-hash refresh is explicit, candidate-validated, and atomic.

### 2.9 Continual-stream data plane (P1, mechanics implemented)

`substrate/continual_stream.py` and `studies/continual_million_event.py` add fixed-width,
hash-chained disk chunks, atomic publication, exact cursor and learner-state resume, abrupt and
gradual schedules, replay, compute-matched no-replay, and fresh-init controls. The preflight binds
retention, acquisition, future learnability, stale-memory, deletion, and resource metrics to the
shared Wave E0 event and lifecycle primitives. `proof/CONTINUAL_MILLION_EVENT_PREFLIGHT.json`
passes at 384 events per stream and remains mechanics-only. Progressive 10k, 100k, and conditional
one-million-event execution plus independent metric replay remain. The progressive runner and four
governor declarations require an empty lane for the first 10k RSS probe and bind every later rung to
the measured prior receipt with 25 percent memory margin.

### 2.10 Rendered action and world-model plane (P1, mechanics implemented)

`studies/action_world_model.py` binds the persistent grid, Wave E0 event identities, deterministic
RGB rendering, and the shell predictor into one eight-arm P7 surface. Three independent worlds carry
same-parent action branches, disjoint train/intervention/planning splits, prediction, calibration,
horizon, planning, and exact compute ledgers. The compact, object-centered, blind, shuffled, and
matched-depth arms share exact core parameter and counted-MAC budgets. The proof rejects six replay
mutations and records a fixture null because neither learned planner beats the reactive baseline.
`proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json` licenses mechanics reuse only. External trajectories,
an exact-referent action control, and replication remain the scientific gate.

### 2.11 Causal operational-monitoring plane (P1, mechanics implemented)

`studies/p9_causal_monitoring.py` extends `studies/p9_accounting.py` from passive cost measurement to
prospective failure monitoring and bounded intervention. Five independent structural workloads bind
disjoint lineages, same-parent queue/memory/retry/relief branches, a held-out confounder reversal,
seven matched-capacity monitor arms, fixed-threshold and oracle controls, rational calibration, and
paired relief utility. Atomic four-chunk resume, a fifteen-chunk provenance chain, exact replay, and
eight mutation attacks pass. `proof/P9_CAUSAL_MONITORING_PREFLIGHT.json` remains mechanics-only;
natural workload/failure episodes, replicated prospective calibration, and a physical energy meter
remain external gates.

---

## 3. Known gaps this map creates work for

1. F8/F16 lack a rights-clean natural dataset package and an executable real-weight/inherited-feature
   receipt in the supported safe format (environment blocker, not a measured hardware wall).
2. No configured external trust root can attest legal rights or real pretraining provenance, so the
   engine correctly keeps all natural claims nonpromotable.
3. A future full-size encoder backend may need a streamed package format beyond safe MLP NPZ; it must
   preserve the same hashes, referents, split guards, compute ledger, and fail-closed behavior.
4. No F8/F16 M3 OOM, timeout, or disk-limit attempt exists. Resource projections cannot be upgraded to
   Studio boundary evidence without that measurement.
5. E6/DR14 implementation is complete, but the rights-clean annotated natural input manifest,
   untouched test membership, serial learned/random caches, and independent verifier are not.
6. Wave E0 closes shared mechanics for three sentinels, not natural event identity, long-stream
   validity, or external transfer.
7. The continual-stream implementation blocker is closed. Scale, at least five independent seeds,
   full-rung interruption recovery, and independent metric verification remain execution gates.
8. P7 rendering, interventions, controls, and cost accounting are closed locally, but the tiny
   generated fixture supports the null. External action trajectories and exact-referent control
   validity remain before any capability claim.
9. P9 causal monitoring and full workload accounting are closed locally at structural-fixture
   scope. Independent natural failures, real interventions, replication, and metered energy remain.

---

## 4. Performance concerns carried into the density doctrine

- Encode cost is measured per instrument: cache only under a complete checkpoint/input/view/split
  identity and invalidate on any source or transform drift.
- Dense tokens are 8192 tokens per clip (storage.DENSE_TOKENS_PER_CLIP): dense caches are the
  single largest byte cost; F-series must report alignment per cache GB when it touches them.
- MoEHead routing claims need active-FLOP accounting, not just parameter counts (compute.py).
- faiss is optional on Apple Silicon; KVIndex brute fallback changes retrieval-latency numbers,
  so F5-at-scale must report which backend ran.
- Full-file sha256 fingerprints on multi-GB caches are a receipt-cost trade; sampled fingerprints
  are the default for a reason.
