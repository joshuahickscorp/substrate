# MOP Collapse Ledger

Durable progress ledger for MOP_ACCRETION_COLLAPSE.md. Machine authority: `MOP_COLLAPSE_STATE.json`. This file is rendered from it; edit the generator, not this file, for structural changes. Per-checkpoint measurements are appended under History.

## Governing principle

> One scientific kernel. One evidence language. One experiment engine. One controller. One registry. One interface. Full breadth. Minimal mass.

## Boundary

- Branch `agent/mop-accretion-collapse` @ `321cdbb` (base = current origin/main `a19ebe6`).
- Live tree: /Users/scammermike/Downloads/mop (agent/save-mop-stable-work, DO NOT TOUCH).
- Live General Run: {"state": "run_categorized_wave", "stage": "run_categorized_wave", "updated_at": "2026-07-19T05:07:39.390790+00:00", "counts": {"compute_complete": 2, "compute_total": 4, "legacy_complete": 3, "legacy_total": 3, "stage_index": 3, "stage_total": 5}}.
- Only light work while the run occupies the host; heavy validation is queued.

## Baseline (measured, not assumed)

| metric | value |
|---|---|
| tracked_files | 1746 |
| python_files | 1072 |
| global_owned_source_LOC | 401194 |
| global_maintained_source_LOC | 401155 |
| active_src_mop_LOC | 259163 |
| test_LOC | 89983 |
| scripts_LOC | 52009 |
| documentation_LOC | 43932 |
| configuration_LOC | 514051 |
| python_modules | 466 |
| entrypoints | 319 |
| root_md_docs | 35 |
| all_md_docs | 170 |

Lifecycle-boilerplate suffix clusters: `{"_scaffold.py": 37, "_runner.py": 26, "_bed.py": 18, "_gate.py": 15, "_impl.py": 15, "_producer.py": 11, "_verifier.py": 11, "_prereg.py": 10, "_harness.py": 5, "_referee.py": 4}`

## Checklist status

Total items: 215. By status: `{"complete": 12, "active": 47, "pending": 147, "partial": 8, "verified": 1}`.

| id | sec | kind | status | title | next action |
|---|---|---|---|---|---|
| WS-1 | 2 | workspace | complete | Isolated worktree + branch off current origin/main | none |
| WS-2 | 2 | workspace | complete | Separate build/cache/test-temp/report/pack roots | none |
| WS-3 | 2 | invariant | active | Absolute non-interference with the live General Run (2.1) | re-verify live PID 52934 alive and untouched at every checkpoint |
| GIT-1 | 24 | git | pending | Draft PR against current main; not ready/merge until all gates | push precheck commit, open draft PR via gh |
| TAG-mop-collapse-precheck | 24 | rollback_tag | pending | Create rollback tag mop-collapse-precheck | tag mop-collapse-precheck at its green checkpoint |
| TAG-mop-collapse-evidence | 24 | rollback_tag | pending | Create rollback tag mop-collapse-evidence | tag mop-collapse-evidence at its green checkpoint |
| TAG-mop-collapse-experiment-engine | 24 | rollback_tag | pending | Create rollback tag mop-collapse-experiment-engine | tag mop-collapse-experiment-engine at its green checkpoint |
| TAG-mop-collapse-starss23 | 24 | rollback_tag | pending | Create rollback tag mop-collapse-starss23 | tag mop-collapse-starss23 at its green checkpoint |
| TAG-mop-collapse-mechanisms | 24 | rollback_tag | pending | Create rollback tag mop-collapse-mechanisms | tag mop-collapse-mechanisms at its green checkpoint |
| TAG-mop-collapse-controller | 24 | rollback_tag | pending | Create rollback tag mop-collapse-controller | tag mop-collapse-controller at its green checkpoint |
| TAG-mop-collapse-registry-config | 24 | rollback_tag | pending | Create rollback tag mop-collapse-registry-config | tag mop-collapse-registry-config at its green checkpoint |
| TAG-mop-collapse-validation | 24 | rollback_tag | pending | Create rollback tag mop-collapse-validation | tag mop-collapse-validation at its green checkpoint |
| TAG-mop-collapse-docs | 24 | rollback_tag | pending | Create rollback tag mop-collapse-docs | tag mop-collapse-docs at its green checkpoint |
| TAG-mop-collapse-packs | 24 | rollback_tag | pending | Create rollback tag mop-collapse-packs | tag mop-collapse-packs at its green checkpoint |
| TAG-mop-collapse-300k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-300k | tag mop-collapse-300k at its green checkpoint |
| TAG-mop-collapse-250k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-250k | tag mop-collapse-250k at its green checkpoint |
| TAG-mop-collapse-200k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-200k | tag mop-collapse-200k at its green checkpoint |
| TAG-mop-collapse-150k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-150k | tag mop-collapse-150k at its green checkpoint |
| TAG-mop-collapse-125k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-125k | tag mop-collapse-125k at its green checkpoint |
| TAG-mop-collapse-100k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-100k | tag mop-collapse-100k at its green checkpoint |
| TAG-mop-collapse-75k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-75k | tag mop-collapse-75k at its green checkpoint |
| TAG-mop-collapse-50k | 24 | rollback_tag | pending | Create rollback tag mop-collapse-50k | tag mop-collapse-50k at its green checkpoint |
| TAG-mop-collapse-event-horizon | 24 | rollback_tag | pending | Create rollback tag mop-collapse-event-horizon | tag mop-collapse-event-horizon at its green checkpoint |
| PR9-1 | 3 | pr9 | pending | Inspect every PR #9 file | diff origin/agent/mop-extreme-condensation vs main; list files |
| PR9-2 | 3 | pr9 | pending | Test PR #9 controller against current main | run its accounting controller read-only |
| PR9-3 | 3 | pr9 | pending | Port or rewrite only useful mechanisms | port LOC accounting, no-minify/no-pack gates, hydration |
| PR9-4 | 3 | pr9 | pending | Discard assumptions invalidated by Generation-1 era | record discarded assumptions |
| PR9-5 | 3 | pr9 | pending | Replace active-checkout LOC metric with honest global reduction | global accounting is primary |
| PR9-6 | 3 | pr9 | pending | Open a new draft PR from current main | gh pr create --draft |
| PR9-7 | 3 | pr9 | pending | Keep PR #9 open until protections exist on replacement | do not close prematurely |
| PR9-8 | 3 | pr9 | pending | Close PR #9 only after exact retained-vs-retired mapping exists | write mapping artifact |
| ART-CONTEXT-JSON | 6 | artifact | complete | MOP_CONTEXT_SURFACE.json | extend with cold_import/test_collection timings under host headroom |
| ART-CONTEXT-MD | 6 | artifact | pending | MOP_CONTEXT_SURFACE.md (orientation benchmark 10 Qs) | render md from json + run clean-agent orientation benchmark |
| MET-CONTEXT | 6 | metric | partial | Context/orientation metrics measured | add reading tokens + cold_import + collection/docs-validation timings (heavy: queue) |
| ART-MOP_CODEBASE_CENSUS.json | 7 | artifact | complete | MOP_CODEBASE_CENSUS.json | none |
| ART-MOP_CODEBASE_CENSUS.md | 7 | artifact | pending | MOP_CODEBASE_CENSUS.md | generate MOP_CODEBASE_CENSUS.md |
| ART-MOP_IMPORT_GRAPH.json | 7 | artifact | complete | MOP_IMPORT_GRAPH.json | none |
| ART-MOP_CALL_GRAPH.json | 7 | artifact | pending | MOP_CALL_GRAPH.json | generate MOP_CALL_GRAPH.json |
| ART-MOP_COMMAND_GRAPH.json | 7 | artifact | complete | MOP_COMMAND_GRAPH.json | none |
| ART-MOP_SCHEMA_GRAPH.json | 7 | artifact | pending | MOP_SCHEMA_GRAPH.json | generate MOP_SCHEMA_GRAPH.json |
| ART-MOP_CONFIG_GRAPH.json | 7 | artifact | pending | MOP_CONFIG_GRAPH.json | generate MOP_CONFIG_GRAPH.json |
| ART-MOP_TEST_OWNERSHIP.json | 7 | artifact | pending | MOP_TEST_OWNERSHIP.json | generate MOP_TEST_OWNERSHIP.json |
| ART-MOP_DOCUMENTATION_GRAPH.json | 7 | artifact | pending | MOP_DOCUMENTATION_GRAPH.json | generate MOP_DOCUMENTATION_GRAPH.json |
| ART-MOP_DUPLICATION_GRAPH.json | 7 | artifact | complete | MOP_DUPLICATION_GRAPH.json | none |
| ART-MOP_AUTHORITY_GRAPH.json | 7 | artifact | complete | MOP_AUTHORITY_GRAPH.json | none |
| ART-MOP_HISTORICAL_BOUNDARY.json | 7 | artifact | pending | MOP_HISTORICAL_BOUNDARY.json | generate MOP_HISTORICAL_BOUNDARY.json |
| ART-MOP_IRREDUCIBLE_KERNEL_ESTIMATE.json | 7 | artifact | pending | MOP_IRREDUCIBLE_KERNEL_ESTIMATE.json | generate MOP_IRREDUCIBLE_KERNEL_ESTIMATE.json |
| ART-MOP_LIVE_NO_TOUCH.json | 7 | artifact | complete | MOP_LIVE_NO_TOUCH.json | none |
| CENSUS-CLASSIFY | 7 | census | pending | Classify every file into exactly one of 16 categories; unknown->0 | run classification over census records grounded in imports/tests/proofs/git |
| MET-global_owned_source_LOC | 4 | metric | partial | Measure global_owned_source_LOC | refine via classification |
| MET-global_maintained_source_LOC | 4 | metric | partial | Measure global_maintained_source_LOC | refine via classification |
| MET-active_kernel_LOC | 4 | metric | pending | Measure active_kernel_LOC | derive from classified census |
| MET-active_product_LOC | 4 | metric | pending | Measure active_product_LOC | derive from classified census |
| MET-default_validation_LOC | 4 | metric | pending | Measure default_validation_LOC | derive from classified census |
| MET-optional_pack_LOC | 4 | metric | pending | Measure optional_pack_LOC | derive from classified census |
| MET-laboratory_LOC | 4 | metric | pending | Measure laboratory_LOC | derive from classified census |
| MET-compatibility_LOC | 4 | metric | pending | Measure compatibility_LOC | derive from classified census |
| MET-historical_source_LOC | 4 | metric | pending | Measure historical_source_LOC | derive from classified census |
| MET-generated_owned_LOC | 4 | metric | pending | Measure generated_owned_LOC | derive from classified census |
| MET-test_LOC | 4 | metric | partial | Measure test_LOC | refine via classification |
| MET-documentation_LOC | 4 | metric | partial | Measure documentation_LOC | refine via classification |
| MET-configuration_LOC | 4 | metric | partial | Measure configuration_LOC | refine via classification |
| MET-CI_build_LOC | 4 | metric | pending | Measure CI_build_LOC | derive from classified census |
| MET-fixture_LOC | 4 | metric | pending | Measure fixture_LOC | derive from classified census |
| MET-third_party_LOC | 4 | metric | pending | Measure third_party_LOC | derive from classified census |
| RED-eliminated_LOC | 4 | reduction_metric | pending | Track eliminated_LOC (relocation!=elimination) | update per region collapse |
| RED-deduplicated_LOC | 4 | reduction_metric | pending | Track deduplicated_LOC (relocation!=elimination) | update per region collapse |
| RED-relocated_LOC | 4 | reduction_metric | pending | Track relocated_LOC (relocation!=elimination) | update per region collapse |
| RED-archived_LOC | 4 | reduction_metric | pending | Track archived_LOC (relocation!=elimination) | update per region collapse |
| RED-generated_replacement_LOC | 4 | reduction_metric | pending | Track generated_replacement_LOC (relocation!=elimination) | update per region collapse |
| RED-added_LOC | 4 | reduction_metric | pending | Track added_LOC (relocation!=elimination) | update per region collapse |
| RED-net_global_reduction_LOC | 4 | reduction_metric | pending | Track net_global_reduction_LOC (relocation!=elimination) | update per region collapse |
| TGT-KERNEL | 5 | target | pending | active kernel <=25000 (stretch 18000) LOC | drive region collapses toward target; measure |
| TGT-GLOBAL | 5 | target | pending | global maintained <=75000 (extreme 50000) LOC | drive region collapses toward target; measure |
| TGT-TESTS | 5 | target | pending | default test harness <=15000 LOC | drive region collapses toward target; measure |
| TGT-DOCS | 5 | target | pending | current-facing docs <=8 documents and <=8000 lines | drive region collapses toward target; measure |
| TGT-ENTRYPOINTS | 5 | target | pending | normal executable entrypoints <=10 | drive region collapses toward target; measure |
| TGT-CONTROLLER | 5 | target | pending | production controllers exactly 1 | drive region collapses toward target; measure |
| TGT-EVIDENCE | 5 | target | pending | receipt/evidence engines exactly 1 | drive region collapses toward target; measure |
| TGT-EXPERIMENT | 5 | target | pending | experiment execution frameworks exactly 1 | drive region collapses toward target; measure |
| TGT-REGISTRY | 5 | target | pending | capability/mechanism registries exactly 1 | drive region collapses toward target; measure |
| TGT-CONFIG | 5 | target | pending | normal configuration roots exactly 1 typed tree | drive region collapses toward target; measure |
| TGT-CLI | 5 | target | pending | normal user-facing CLI exactly 1 | drive region collapses toward target; measure |
| CKPT-300k | 5 | checkpoint | pending | Reach green global checkpoint 300k | tag mop-collapse-300k when global maintained LOC crosses 300k |
| CKPT-250k | 5 | checkpoint | pending | Reach green global checkpoint 250k | tag mop-collapse-250k when global maintained LOC crosses 250k |
| CKPT-200k | 5 | checkpoint | pending | Reach green global checkpoint 200k | tag mop-collapse-200k when global maintained LOC crosses 200k |
| CKPT-150k | 5 | checkpoint | pending | Reach green global checkpoint 150k | tag mop-collapse-150k when global maintained LOC crosses 150k |
| CKPT-125k | 5 | checkpoint | pending | Reach green global checkpoint 125k | tag mop-collapse-125k when global maintained LOC crosses 125k |
| CKPT-100k | 5 | checkpoint | pending | Reach green global checkpoint 100k | tag mop-collapse-100k when global maintained LOC crosses 100k |
| CKPT-75k | 5 | checkpoint | pending | Reach green global checkpoint 75k | tag mop-collapse-75k when global maintained LOC crosses 75k |
| CKPT-50k | 5 | checkpoint | pending | Reach green global checkpoint 50k | tag mop-collapse-50k when global maintained LOC crosses 50k |
| ESCAPE-RULE | 5 | gate | pending | Two-architecture escape rule before rejecting a lower target | only after 2 architectures implemented+failed for measured reasons + green restore + sealed receipt |
| SEC-8 | 8 | region | pending | Canonical end-state architecture (core/science/mechanisms/substrate/campaign/packs/interface) | converge domains without wrapper dirs |
| SEC-9 | 9 | region | active | One evidence authority (compact evidence core; verifier structurally independent) | deletion map ready (collapse/MOP_EVIDENCE_EQUIVALENCE.json): 64 byte-identical primitive defs collapsible onto one core; implement core, redirect, delete, run parity+mutation+replay (HEAVY: queue behind live run per section 2) |
| SEC-10 | 10 | region | pending | One experiment engine (ExperimentSpec..IndependentVerifier) | build engine; simple<=150 LOC, complex<=400 LOC declarations |
| SEC-11 | 11 | region | pending | STARSS23 first high-pressure region collapse (12-step process) | prove method: parity byte-for-byte, replay, delete superseded, recovery map |
| SEC-12 | 12 | region | pending | Mechanism-family collapse (one provider contract) | replace *_scaffold/_impl/_bed/_runner boilerplate (152 files) |
| SEC-13 | 13 | region | pending | One campaign controller (AFTER live run terminal + PR30 closure) | build vs fixtures only while live; archive historical bytes; replay-equivalence then delete |
| SEC-14 | 14 | region | pending | Entrypoint and script collapse (313 -> ~10 CLI verbs) | classify scripts/; remove wrappers/bootstraps/argparse dup |
| SEC-15 | 15 | region | pending | One registry (typed capability registry) | unify experiment/mechanism/dataset/instrument/verifier registries |
| SEC-16 | 16 | region | pending | One typed configuration authority | separate frozen-identity/runtime-policy/machine-profile/overrides |
| SEC-17 | 17 | region | pending | Validation condensation (properties/matrices/mutation; coverage-equivalence receipts) | reduce handwritten test LOC; keep adversarial rigor + producer/verifier split |
| SEC-18 | 18 | region | pending | Documentation collapse (<=8 front-door docs; sealed history index) | consolidate 34 root md + 169 total; generate current tables from authorities |
| SEC-19 | 19 | region | pending | Proof/evidence compaction (content-addressed index; no claim reduction) | build evidence index; dedupe byte-identical payloads; move to packs after run releases |
| SEC-20 | 20 | region | pending | Packs follow collapse (no pack owns a 2nd controller/engine/registry/CLI) | collapse before packing; report relocation separate from elimination |
| TGT-EXP-SIMPLE | 10 | target | pending | simple experiment <=150 LOC declaration + math | enforce |
| TGT-EXP-COMPLEX | 10 | target | pending | complex experiment <=400 LOC declaration + math | enforce |
| INV-frozen_instruments | 23 | invariant | active | Preserve invariant: frozen_instruments | assert in every region parity+mutation gate |
| INV-owned_substrate_separation | 23 | invariant | active | Preserve invariant: owned_substrate_separation | assert in every region parity+mutation gate |
| INV-nulls | 23 | invariant | active | Preserve invariant: nulls | assert in every region parity+mutation gate |
| INV-controls | 23 | invariant | active | Preserve invariant: controls | assert in every region parity+mutation gate |
| INV-independent_units | 23 | invariant | active | Preserve invariant: independent_units | assert in every region parity+mutation gate |
| INV-SESOI | 23 | invariant | active | Preserve invariant: SESOI | assert in every region parity+mutation gate |
| INV-multiplicity | 23 | invariant | active | Preserve invariant: multiplicity | assert in every region parity+mutation gate |
| INV-stop_rules | 23 | invariant | active | Preserve invariant: stop_rules | assert in every region parity+mutation gate |
| INV-negative_results | 23 | invariant | active | Preserve invariant: negative_results | assert in every region parity+mutation gate |
| INV-exact_evidence_classes | 23 | invariant | active | Preserve invariant: exact_evidence_classes | assert in every region parity+mutation gate |
| INV-independent_scientific_recomputation | 23 | invariant | active | Preserve invariant: independent_scientific_recomputation | assert in every region parity+mutation gate |
| INV-no_activation_or_promotion_without_authority | 23 | invariant | active | Preserve invariant: no_activation_or_promotion_without_authority | assert in every region parity+mutation gate |
| INV-honest_hardware_boundaries | 23 | invariant | active | Preserve invariant: honest_hardware_boundaries | assert in every region parity+mutation gate |
| INV-crash_safe_writes | 23 | invariant | active | Preserve invariant: crash_safe_writes | assert in every region parity+mutation gate |
| INV-deterministic_resume | 23 | invariant | active | Preserve invariant: deterministic_resume | assert in every region parity+mutation gate |
| INV-historical_authority_replay | 23 | invariant | active | Preserve invariant: historical_authority_replay | assert in every region parity+mutation gate |
| INV-producer_verifier_structural_independence | 23 | invariant | active | Preserve invariant: producer_verifier_structural_independence | assert in every region parity+mutation gate |
| GATE-NO-MINIFY | 21 | gate | active | Gate: no minification | apply at each region checkpoint; queue heavy variants until host free |
| GATE-NO-LINE-PACK | 21 | gate | active | Gate: no line packing | apply at each region checkpoint; queue heavy variants until host free |
| GATE-PARITY | 21 | gate | active | Gate: behavior + receipt parity | apply at each region checkpoint; queue heavy variants until host free |
| GATE-MUTATION | 21 | gate | active | Gate: receipt/verifier mutation attacks | apply at each region checkpoint; queue heavy variants until host free |
| GATE-REPLAY | 21 | gate | active | Gate: sealed proof replay | apply at each region checkpoint; queue heavy variants until host free |
| GATE-CRASH-RESUME | 21 | gate | active | Gate: crash and deterministic resume | apply at each region checkpoint; queue heavy variants until host free |
| GATE-REGEN | 21 | gate | active | Gate: deterministic regeneration | apply at each region checkpoint; queue heavy variants until host free |
| GATE-COVERAGE-EQUIV | 21 | gate | active | Gate: coverage-equivalence receipt per replaced cluster | apply at each region checkpoint; queue heavy variants until host free |
| GATE-CLEAN-CLONE | 21 | gate | active | Gate: clean clone builds+validates | apply at each region checkpoint; queue heavy variants until host free |
| GATE-OFFLINE-HYDRATION | 21 | gate | active | Gate: offline pack hydration | apply at each region checkpoint; queue heavy variants until host free |
| GATE-RELOCATION-ACCOUNTING | 21 | gate | active | Gate: relocation/archive/pack counted separately from elimination | apply at each region checkpoint; queue heavy variants until host free |
| GATE-PERF-2PCT | 21 | gate | active | Gate: perf regressions >2% investigated | apply at each region checkpoint; queue heavy variants until host free |
| CC-1 | 26 | completion_condition | partial | current main and live-run identities verified | evidence required per spec; nothing complete from prose |
| CC-2 | 26 | completion_condition | pending | PR #9 useful machinery ported or explicitly retired | evidence required per spec; nothing complete from prose |
| CC-3 | 26 | completion_condition | partial | complete owned-system census exists | evidence required per spec; nothing complete from prose |
| CC-4 | 26 | completion_condition | pending | unknown classifications are zero | evidence required per spec; nothing complete from prose |
| CC-5 | 26 | completion_condition | pending | global accounting is honest | evidence required per spec; nothing complete from prose |
| CC-6 | 26 | completion_condition | pending | one evidence authority remains | evidence required per spec; nothing complete from prose |
| CC-7 | 26 | completion_condition | pending | one experiment engine remains | evidence required per spec; nothing complete from prose |
| CC-8 | 26 | completion_condition | pending | one campaign controller remains active | evidence required per spec; nothing complete from prose |
| CC-9 | 26 | completion_condition | pending | one registry remains | evidence required per spec; nothing complete from prose |
| CC-10 | 26 | completion_condition | pending | one typed configuration authority remains | evidence required per spec; nothing complete from prose |
| CC-11 | 26 | completion_condition | pending | one normal CLI remains | evidence required per spec; nothing complete from prose |
| CC-12 | 26 | completion_condition | pending | STARSS23 framework duplication removed | evidence required per spec; nothing complete from prose |
| CC-13 | 26 | completion_condition | pending | mechanism-family boilerplate removed | evidence required per spec; nothing complete from prose |
| CC-14 | 26 | completion_condition | pending | script wrappers collapsed | evidence required per spec; nothing complete from prose |
| CC-15 | 26 | completion_condition | pending | validation uses shared matrices and properties | evidence required per spec; nothing complete from prose |
| CC-16 | 26 | completion_condition | pending | current-facing docs consolidated | evidence required per spec; nothing complete from prose |
| CC-17 | 26 | completion_condition | pending | historical docs and code sealed and indexed | evidence required per spec; nothing complete from prose |
| CC-18 | 26 | completion_condition | pending | packs contain no duplicate authorities | evidence required per spec; nothing complete from prose |
| CC-19 | 26 | completion_condition | pending | sealed results remain replayable | evidence required per spec; nothing complete from prose |
| CC-20 | 26 | completion_condition | pending | independent verifiers remain structurally independent | evidence required per spec; nothing complete from prose |
| CC-21 | 26 | completion_condition | pending | crash/resume and rollback pass | evidence required per spec; nothing complete from prose |
| CC-22 | 26 | completion_condition | pending | clean clone passes | evidence required per spec; nothing complete from prose |
| CC-23 | 26 | completion_condition | pending | offline hydration passes | evidence required per spec; nothing complete from prose |
| CC-24 | 26 | completion_condition | complete | no live-run source was modified | evidence required per spec; nothing complete from prose |
| CC-25 | 26 | completion_condition | pending | full release validation passes after run releases host | evidence required per spec; nothing complete from prose |
| CC-26 | 26 | completion_condition | pending | global LOC reduction measured | evidence required per spec; nothing complete from prose |
| CC-27 | 26 | completion_condition | pending | orientation-token reduction measured | evidence required per spec; nothing complete from prose |
| CC-28 | 26 | completion_condition | pending | lowest green checkpoint tagged | evidence required per spec; nothing complete from prose |
| CC-29 | 26 | completion_condition | pending | rollback documented | evidence required per spec; nothing complete from prose |
| CC-30 | 26 | completion_condition | pending | draft PR contains the complete measured result | evidence required per spec; nothing complete from prose |
| FO-1 | 27 | forbidden_outcome | active | MUST NOT end with: census only | guard against; do not conclude in this state |
| FO-2 | 27 | forbidden_outcome | active | MUST NOT end with: plan only | guard against; do not conclude in this state |
| FO-3 | 27 | forbidden_outcome | active | MUST NOT end with: new abstraction beside every old abstraction | guard against; do not conclude in this state |
| FO-4 | 27 | forbidden_outcome | active | MUST NOT end with: pack-only reduction | guard against; do not conclude in this state |
| FO-5 | 27 | forbidden_outcome | active | MUST NOT end with: smaller default checkout with unchanged global owned code | guard against; do not conclude in this state |
| FO-6 | 27 | forbidden_outcome | active | MUST NOT end with: duplicated old and new experiment engines | guard against; do not conclude in this state |
| FO-7 | 27 | forbidden_outcome | active | MUST NOT end with: duplicated old and new controllers | guard against; do not conclude in this state |
| FO-8 | 27 | forbidden_outcome | active | MUST NOT end with: deferred documentation consolidation | guard against; do not conclude in this state |
| FO-9 | 27 | forbidden_outcome | active | MUST NOT end with: deletion candidates without deletion | guard against; do not conclude in this state |
| FO-10 | 27 | forbidden_outcome | active | MUST NOT end with: permanent wrappers around legacy implementations | guard against; do not conclude in this state |
| FO-11 | 27 | forbidden_outcome | active | MUST NOT end with: an under-tested generic engine | guard against; do not conclude in this state |
| FO-12 | 27 | forbidden_outcome | active | MUST NOT end with: hidden generated code | guard against; do not conclude in this state |
| FO-13 | 27 | forbidden_outcome | active | MUST NOT end with: deleted scientific evidence | guard against; do not conclude in this state |
| FO-14 | 27 | forbidden_outcome | active | MUST NOT end with: reduced independent-verifier rigor | guard against; do not conclude in this state |
| FO-15 | 27 | forbidden_outcome | active | MUST NOT end with: request to finish next region in another session | guard against; do not conclude in this state |
| FO-16 | 27 | forbidden_outcome | active | MUST NOT end with: claim of irreducible before two architectures attempted | guard against; do not conclude in this state |
| RPT-1 | 28 | report_item | pending | Final report clause 1 | populate from measured artifacts at conclusion |
| RPT-2 | 28 | report_item | pending | Final report clause 2 | populate from measured artifacts at conclusion |
| RPT-3 | 28 | report_item | pending | Final report clause 3 | populate from measured artifacts at conclusion |
| RPT-4 | 28 | report_item | pending | Final report clause 4 | populate from measured artifacts at conclusion |
| RPT-5 | 28 | report_item | pending | Final report clause 5 | populate from measured artifacts at conclusion |
| RPT-6 | 28 | report_item | pending | Final report clause 6 | populate from measured artifacts at conclusion |
| RPT-7 | 28 | report_item | pending | Final report clause 7 | populate from measured artifacts at conclusion |
| RPT-8 | 28 | report_item | pending | Final report clause 8 | populate from measured artifacts at conclusion |
| RPT-9 | 28 | report_item | pending | Final report clause 9 | populate from measured artifacts at conclusion |
| RPT-10 | 28 | report_item | pending | Final report clause 10 | populate from measured artifacts at conclusion |
| RPT-11 | 28 | report_item | pending | Final report clause 11 | populate from measured artifacts at conclusion |
| RPT-12 | 28 | report_item | pending | Final report clause 12 | populate from measured artifacts at conclusion |
| RPT-13 | 28 | report_item | pending | Final report clause 13 | populate from measured artifacts at conclusion |
| RPT-14 | 28 | report_item | pending | Final report clause 14 | populate from measured artifacts at conclusion |
| RPT-15 | 28 | report_item | pending | Final report clause 15 | populate from measured artifacts at conclusion |
| RPT-16 | 28 | report_item | pending | Final report clause 16 | populate from measured artifacts at conclusion |
| RPT-17 | 28 | report_item | pending | Final report clause 17 | populate from measured artifacts at conclusion |
| RPT-18 | 28 | report_item | pending | Final report clause 18 | populate from measured artifacts at conclusion |
| RPT-19 | 28 | report_item | pending | Final report clause 19 | populate from measured artifacts at conclusion |
| RPT-20 | 28 | report_item | pending | Final report clause 20 | populate from measured artifacts at conclusion |
| RPT-21 | 28 | report_item | pending | Final report clause 21 | populate from measured artifacts at conclusion |
| RPT-22 | 28 | report_item | pending | Final report clause 22 | populate from measured artifacts at conclusion |
| RPT-23 | 28 | report_item | pending | Final report clause 23 | populate from measured artifacts at conclusion |
| RPT-24 | 28 | report_item | pending | Final report clause 24 | populate from measured artifacts at conclusion |
| RPT-25 | 28 | report_item | pending | Final report clause 25 | populate from measured artifacts at conclusion |
| RPT-26 | 28 | report_item | pending | Final report clause 26 | populate from measured artifacts at conclusion |
| RPT-27 | 28 | report_item | pending | Final report clause 27 | populate from measured artifacts at conclusion |
| RPT-28 | 28 | report_item | pending | Final report clause 28 | populate from measured artifacts at conclusion |
| RPT-29 | 28 | report_item | pending | Final report clause 29 | populate from measured artifacts at conclusion |
| RPT-30 | 28 | report_item | pending | Final report clause 30 | populate from measured artifacts at conclusion |
| ART-MOP_EVIDENCE_EQUIVALENCE.json | 9 | artifact | complete | MOP_EVIDENCE_EQUIVALENCE.json (evidence-primitive deletion map) | none |
| ART-MOP_EVIDENCE_MIGRATION.json | 9 | artifact | complete | MOP_EVIDENCE_MIGRATION.json (per-duplicate migration table) | execute remaining batches under their gates |
| RED-batch1 | 9 | verified_reduction | verified | Evidence core batch1: 9 studies modules deduplicated onto mop.substrate.events | next batch: sha256_file dominant cluster (9), then _atomic_write (6), then distinct-body inspection |

## History

### precheck (this checkpoint)

- commit: pending; base 321cdbb
- global_owned_source_LOC: 401194
- global_maintained_source_LOC: 401155
- eliminated_LOC: 0; relocated_LOC: 0; archived_LOC: 0; added_LOC: (ledger+census tooling)
- rollback_tag: mop-collapse-precheck (to be created at commit)
- next_exact_edit: generate remaining census graphs (call/command/schema/config/authority/historical-boundary/live-no-touch), classify unknown->0, then port PR #9 protections

