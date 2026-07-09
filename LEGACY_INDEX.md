# LEGACY INDEX

The per-document archive index of the paradigm migration. Legacy documents are archived IN PLACE:
their paths do not move (the markdown ledger, cross-references, and the /goal handoff stack depend
on them), but their ACTIVE status is fixed here. docs/archive/ holds the pointer indexes per legacy
family. Physical relocation, if ever, is a Phase 6 decision (MIGRATION_PHASES.md).

Status vocabulary: preserve (active doctrine or law), demote (era doctrine or instrument, cited not
governing), archive (frozen historical narrative), rewrite (superseded by a named new doc).

House style: no em or en dashes.

---

## Root documents

| Document | Old role | New status | Notes / linked work |
|---|---|---|---|
| FORM_SUBSTRATE_PROGRAM.md | (new) | preserve, ROOT | the active worldview |
| FORM_SUBSTRATE_DOCTRINE.md | (new) | preserve, law | methods constitution |
| FORM_SUBSTRATE_EXPERIMENTS.md | (new) | preserve | F1-F20 + migration map |
| FORM_SUBSTRATE_CODEMAP.md | (new) | preserve | module migration map |
| PERFORMANCE_DENSITY_DOCTRINE.md | (new) | preserve, law | cost law |
| OPERATIONAL_AWARENESS.md | (new) | preserve, law | awareness law |
| PARADIGM_MIGRATION.md | (new) | preserve | consolidation and wipe order |
| MIGRATION_PHASES.md | (new) | preserve | execution phases + first 10 tasks |
| BLACKHOLE.md | code-form doctrine | preserve, law | unchanged; density doctrine extends it |
| DECISIONS.md | append-only decision record | preserve (append-only) | never rewritten, only appended |
| DOCTRINE_SYNTHESIS.md | empirical synthesis of the pre-Studio corpus | preserve (evidence) | the negative-map narrative |
| STATUS.md | session-by-session build record | preserve (append-only) | operational history |
| ISSUES.md | known defects | preserve (operational) | carries the frozen_random refactor item |
| GO.md | cold-start entry | preserve (operational), fix stale MoT path | Phase 0 task 4 |
| README.md | project front door | rewrite in place (Phase 0+1): must present the Form Substrate Program first | keep gate-checked numbers |
| ARCHITECTURE.md | frozen-substrate system sketch | demote (era doctrine) | superseded by FORM_SUBSTRATE_PROGRAM.md section 5 |
| APPLE_SILICON.md, SCALING.md | hardware and scaling notes | preserve (operational) | perf kernel inputs |
| STUDIO_HANDOFF.md | studio cold-start brief | preserve (operational) | spine unchanged by migration |
| CONDENSE_LEDGER.md, CONDENSE_AUDIT.md, CONDENSE_DOCS_REVIEW.md | BLACKHOLE collapse ledgers | preserve (evidence) | numbers-as-proof record |
| EXPERIMENTS.md | rendered bank view | preserve (generated) | re-rendered from registry each change |
| developmental_jepa_corpus.md, vol2, vol3 | V-JEPA-era research corpus | archive (vjepa_legacy) | the origin trilogy; evidence and citations only |

## docs/mixture_of_perspectives/ (the MoP era)

| Document | Old role | New status | Notes / linked work |
|---|---|---|---|
| 16_form_substrate_program.md | the pivot beachhead | preserve (era bridge) | superseded as ROOT by FORM_SUBSTRATE_PROGRAM.md; keeps the original pivot rationale |
| 01_thesis_and_definition.md | MoP thesis, H1-H5 | demote (Layer 5 sub-doctrine) | H1-H5 live inside mode ecology |
| 03_thinking_modes.md | mode taxonomy | demote (Layer 5 catalog) | modes become Layer 5 citizens |
| 04_reasoning_program.md | reasoning line | demote (negative-informed catalog) | carries the 24-null test-time-compute record |
| 05_plasticity_program.md | plasticity line | demote | feeds F7/F8 designs; negatives stand |
| 06_cognitive_currencies_atlas.md | substrates as currencies | demote, conceptually renamed form currency atlas | AT-series continues |
| 07_workspace_layer.md | workspace architectures A1-A14 | demote (Layer 5 design bank) | WS1 feeds OA2 |
| 08_09_custom_model_pathway_and_architectures.md | gates C1-C3, stages, Arch A-F | preserve (gating law referenced by F8/F16) | the escalation ladder |
| 10_compute_tiers.md | tier doctrine | demote (superseded hardware numbers; tiers still used) | M1 Ultra profile governs |
| 11_experiment_registry.md | MoP registry narrative | demote (registry yaml is the source of truth) | dedup ledger kept |
| 12_metrics.md | metrics contract | preserve (instrument definitions) | density doctrine cites it |
| 13_code_scaffolding.md | code audit of the MoP era | demote | superseded by FORM_SUBSTRATE_CODEMAP.md |
| 15_custom_model_skepticism.md | the brake | preserve, LAW | binds F7/F8/F16 verbatim |
| MIXTURE_OF_THINKING.md | MoP master synthesis | demote (era index) | decision tree still cited by CM rows |
| SEMANTIC_POSITIONS.md | 86 semantic positions | preserve (position bank) | form-vs-meaning positions become F-series contrasts |
| RESULTS_LEDGER.md | MoP lane results | preserve (evidence) | primary verdict source |
| STUDIO_RUN_REPORT.md | studio scoreboard | preserve (operational, receipt-backed) | scorecard writes into it |
| STUDIO_TURNKEY_PLAN.md | de-risk checklist | preserve (operational) | tiers 1-4 done |
| M3PRO_RUN_REPORT.md | laptop maximization record | archive (mop_legacy) | corrected-record narrative |
| HANDOFF.md | cold-start brief + verdict ledger | preserve (operational) | MoT-to-MoP rename note lives here |
| EXECUTION_MANIFEST.md | original work-package plan | archive (mop_legacy) | superseded by SCAFFOLD.md; cites old brain/ path |
| STUDIO_GOAL_PROMPT.md | /goal loop prompt | preserve (operational) | update to mention new root docs in Phase 1 |
| POTENTIAL_AUDIT.md | adversarial self-audit | archive (mop_legacy) | its ceiling map fed the pivot |
| STUDIO_POTENTIAL_AUDIT.md | studio ceiling audit | preserve (evidence) | names the readout-vs-formation wall |
| CONDENSATION_PLAN.md | doc-shrink plan | preserve (operational) | must-not-condense list still binds |
| EXPAND_PHASE_PLAN.md | walls-to-experiments map | preserve (plan) | tracks A/B/C feed Phase 3 |
| DEEP_RESEARCH_2026_07.md | literature brief | preserve (evidence) | PR9 certification lives here |
| SCAFFOLD.md | process A/B/C sequencing | preserve (operational) | spine authority with HANDOFF |

## docs/ and other

| Document | Old role | New status | Notes |
|---|---|---|---|
| docs/STUDIO_MAXIMIZATION_2026_06_27.md | M2 Max era studio plan + EX bank | archive (mop_legacy) | hardware superseded by studio-m1ultra; EX bank absorbed by registry |
| proof/* | trust surface | preserve, law | unchanged by migration |
| runs/*, data/cache/* | receipts and assets | preserve, sacred (BLACKHOLE assets clause) | never touched by doc migration |

## Archive families

- docs/archive/mop_legacy/README.md: MoP-era narrative docs frozen above.
- docs/archive/vjepa_legacy/README.md: the corpus trilogy and V-JEPA-shaped framing docs.
- docs/archive/biology_levers_legacy/README.md: the biology-lever program's negative record map
  (experiment banks stay in code and registry; this index points at their verdicts).

Known risks of this archive design: in-place archiving means a reader can still open an era doc and
mistake it for active doctrine. Mitigation: this index is ledgered and linked from
FORM_SUBSTRATE_PROGRAM.md; any doc marked demote or archive here must not be cited as active
doctrine in new work, and check_docs.py keeps the doc set closed so nothing regrows silently.
