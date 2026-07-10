# FORM SUBSTRATE IMPLEMENTATION PLAN

Status note (2026-07-10): this is the original execution snapshot, corrected by the claim-level and
local-ceiling audits. Its implementation list and embedded goal prompt are historical. Use
`MOP_MAXIMUM_POTENTIAL_GOAL.md`, `MOP_MAXIMUM_POTENTIAL_EXECUTION_PLAN.md`, current receipts, and
`FORM_SUBSTRATE_EXPERIMENTS.md` for live order. Unfinished real-data, unification, environment, and
plasticity obligations remain relevant. The old Studio gate does not. No current receipt establishes
a hardware prerequisite for this plan.

The detailed scaffolding and execution plan for building out the Form Substrate Program from the
state the paradigm migration left behind. MIGRATION_PHASES.md is the phase law; this document is
the work order: every task with its files, signatures, configs, registry actions, tests, exit
checks, and risks. A fresh session should be able to open this file and start producing.

House style: no em or en dashes. Environment: install the package with `make install`; `import mop`
must work outside the checkout without `PYTHONPATH`; the interpreter is `.venv/bin/python`. Read FORM_SUBSTRATE_PROGRAM.md and
FORM_SUBSTRATE_DOCTRINE.md before writing any code; read FORM_SUBSTRATE_CODEMAP.md before creating
any file.

---

## 0. State snapshot (what this plan builds on)

Verified green at plan time: full unit suite, acceptance 10/10, docs gate clean, registry 195 rows
valid (20 F-series), ruff and mypy clean.

Exists and runs:

- `src/mop/substrate/form.py`: FormMeta, FormBatch, FormAdapter, TensorFormAdapter, FormMatrix,
  build_form_matrix, form_audit, fit_affine_alignment, apply_affine_alignment.
- `src/mop/experiments/f_form_substrate.py`: F1, F2, F3, F5 (cpu-now toys, controls baked in),
  each result carrying a `density` block.
- `src/mop/diagnostics/performance_density.py`: density_block, timed. Every result needs
  capability, cost, density (PERFORMANCE_DENSITY_DOCTRINE.md).
- `src/mop/diagnostics/operational_awareness.py`: OA1-OA8 composites, rail-gated render.
- Registry rows F1-F20 (F1/F2/F3/F5 implemented, rest registry-only, all R0).
- Root docs: FORM_SUBSTRATE_{PROGRAM,DOCTRINE,CODEMAP,EXPERIMENTS}.md,
  PERFORMANCE_DENSITY_DOCTRINE.md, OPERATIONAL_AWARENESS.md, PARADIGM_MIGRATION.md,
  MIGRATION_PHASES.md, LEGACY_INDEX.md, docs/archive/ pointers.

Does not exist yet (this plan's output): the form/perspective consolidation, store-backed form
adapters, integration tests for the F-series, implementations for F4/F9/F10/F12/F13/F17/F18/F19/F20,
the DR1 real-form matrix, OA scorecard wiring, and the Phase 5 renames.

## 1. Standing constraints (apply to every task below)

1. No new module without checking the refusal table (FORM_SUBSTRATE_CODEMAP.md section 2). No
   referent.py, no form_store.py, no third adapter ABC, no parallel shell class.
2. Every new experiment: registry row first (or flip status), config in configs/experiment/,
   class in f_form_substrate.py with the full contract (id, metric, baseline, ablation,
   null_hypothesis, tier), registration in scaffolds.py, density block in the result, unit or
   integration coverage.
3. Every new markdown file goes into the check_docs.py ledger in the same change.
4. Free text (registry rows, reports, docstrings) passes the sentience rail; engineering
   vocabulary only; no em dashes.
5. Green before and after every task: the verification protocol in section 8. A task that breaks
   a gate is reverted, not patched forward.
6. Natural-data F8/F16 evidence remains license and provenance gated (section 7). The implemented
   local fixture-scientific engine may update a safe fixture encoder to validate mechanics, controls,
   null logic, and receipts, but its taint is irreversible and it cannot promote a natural claim.

## 2. Workstream A: harness parity for the F-series (first, small)

### A1. tests/integration/test_f_series.py

Why: the E/EX series have harness-level integration gates; F1-F5 only have unit tests on form.py.
The F-series must be exercised the way it will actually run.

Scaffold: mirror the pattern of tests/integration/test_scaffolding_experiments.py (compose the
config, run through the runner, assert on the result dict).

For each id in (f1_form_alignment_gate, f2_heldout_form_transfer, f3_form_bottleneck_capacity,
f5_cross_form_memory_binding):

- compose with `experiment=<id>` and reduced hyperparameters for speed (samples <= 120, epochs
  <= 30, seeds [0, 1]) via override strings, run into a tmp_path run dir.
- assert: `null_supported` is a bool; every name in the class `metric` tuple appears in the result;
  `result["density"]["schema"] == "mop-density-block/v1"`; density has at least one ratio.
- assert F1 sanity: `aligned_transfer >= raw_transfer` is NOT required (that would presume the
  result); assert only mechanics (keys, ranges in [0, 1], seeds echoed).
- total wall-clock for the module under ~60 s.

Exit check: `PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_f_series.py -q` green;
full unit suite still green.

## 3. Workstream B: one referent-aligned stack (the consolidation)

Order matters: B1 and B2 are additive and safe; B3 is the merge decision; B4 rides behind it.

### B1. Store-backed form adapters (bridges)

File: extend `src/mop/substrate/form.py` (no new module).

- `class LatentStoreFormAdapter(FormAdapter)`: present a cached LatentStore as one form arm.
  - `__init__(self, meta: FormMeta, store: LatentStore, *, referents: Sequence[str] | None = None,
    factors: Mapping[str, Any] | None = None)`.
  - referents default to the store's keys (clip ids are already referent ids); explicit referents
    must be a subset check, refuse on mismatch.
  - features: pooled `[N, D]` passes through; dense `[N, T, D]` flattens with
    `meta.token_shape == (T, D)` recorded so token geometry is recoverable.
  - factors: accept an explicit mapping, and read the cache's factors.json sidecar when present
    (same contract real_latent.py uses).
- `class SubstrateFormAdapter(FormAdapter)`: wrap a SubstrateAdapter + clips into a form arm
  (encode-once discipline: caller passes cached features or a store, never triggers encode inside
  the adapter; refuse if asked to encode).

Tests (tests/unit/test_form_substrate.py, extend): build a tiny LatentStore via the existing cache
fixtures, wrap it, assert referent alignment against a TensorFormAdapter arm over the same keys,
assert dense flattening records token_shape, assert factor sidecar loads.

Exit check: unit green; form_audit on a mixed (store-backed + tensor) matrix reports all_ok with a
control arm present.

### B2. Helper collapse (delete the byte-identical duplicates)

`_referent_tuple` and `_factor_dict` exist twice (form.py and perspectives/adapter.py, identical).
Canonical home: form.py (the interface of record; perspectives already imports from substrate, so
the dependency direction is fine).

- Delete the copies in perspectives/adapter.py; import from `..substrate.form`.
- Do NOT move the referent-ordering loop yet (that is B3's decision); only the leaf helpers.

Exit check: `grep -rn "_referent_tuple\|_factor_dict" src | grep def` shows exactly one definition
of each; test_perspective_adapter.py and test_form_substrate.py green.

### B3. The merge decision (one matrix stack survives)

Decision procedure, run before any code:

1. Blast radius: `grep -rln "PerspectiveAdapter\|PerspectiveMatrix\|build_perspective_matrix\|
   perspective_audit\|PerspectiveMeta" src scripts tests` and the same for the Form names. Count
   files, not lines.
2. Expected outcome (from the codemap audit): perspectives has ~5-8 consumer files including
   studio lanes (dr1_perspectives.py, native_lanes.py, dr1_curate_bound_video.py); form has 2.
3. Rule: the SMALLER import surface migrates. Expected: FormMeta grows the missing Perspective
   fields and perspectives becomes re-exports, but executed the other way if the grep says
   otherwise. Either way the end state is:
   - one Meta dataclass carrying: tag, kind, feature_dim, source, objective, token_shape,
     time_axis, trainable, control_for, license, notes, plus perspectives' `supervised` and
     `derived` flags (map supervised into OBJECTIVE_FAMILIES where possible; keep `derived` as a
     bool).
   - one Batch, one Matrix, one matrix builder, one audit function.
   - PerspectiveMeta / PerspectiveBatch / PerspectiveMatrix / build_perspective_matrix /
     perspective_audit remain importable as thin aliases for one deprecation phase (Phase 5
     removes them).
   - the PerspectiveRegistry and the two production bridges move to (or are re-exported from) the
     surviving module.
4. Migrate consumers in the same change: studio/dr1_perspectives.py, studio/native_lanes.py,
   scripts/studio/dr1_curate_bound_video.py, plus any test imports.

Exit check: test_perspective_adapter.py, test_form_substrate.py, test_dr1_perspectives.py,
test_native_lanes.py all green; `grep -rn "referents.index" src` shows one ordering implementation;
EXPERIMENTS.md render unchanged; acceptance 10/10.

Risk and rollback: this is the highest-blast-radius task in the plan. Do it in one commit with no
other changes so `git revert` is clean. If dr1 lane tests fail in a way that is not a trivial
import fix, revert and file the failure in ISSUES.md rather than patching forward.

### B4. Form provenance in cache manifests

File: `src/mop/substrate/cache_manifest.py`.

- Schema bump: `mop-cache-data-plane/v2`. New optional payload fields: `form_kind` (one of
  FORM_KINDS), `form_objective` (one of OBJECTIVE_FAMILIES), `referent_scheme` (free text, e.g.
  "clip-id" or "task/episode/scene/object" for F19).
- v1 manifests remain readable: validate_cache_manifest accepts both versions, treats the new
  fields as absent on v1, and never fails a v1 cache for missing form fields.
- write_cache_manifest gains keyword-only `form_kind=None, form_objective=None, referent_scheme=None`.

Tests: extend tests/unit/test_cache_manifest.py: v2 roundtrip with form fields; v1 fixture still
validates; wrong form_kind refused.

Exit check: cache tools tests green; a rebuilt toy cache carries form fields.

## 4. Workstream C: F-series implementations (cpu-now, in this order)

Every task here follows the same checklist: (1) class in f_form_substrate.py with contract,
(2) config in configs/experiment/, (3) scaffolds.py import + SCAFFOLDS append, (4) registry row
status flip to implemented, (5) EXPERIMENTS.md re-render, (6) integration test row added,
(7) density block in the result, (8) smoke run receipt.

### C1. F4 raw_payload_vs_form_tokens (Layer 1/4; the anti-ceremony gate)

Design: heterogeneous payload shapes per form so the tokenizer has real work: vision `[N, 8, 8]`,
audio `[N, 64]`, symbolic `[N, 16]` sign-quantized, timeseries `[N, 4, 16]`. Three arms at matched
final dimension D=64:

- canonical form tokens: per-form learned-free linear map to `[T=4, d=16]` tokens + flatten
  (the "form layer"), referent metadata attached.
- raw flattened payload (zero-pad or project to D=64 with a fixed random map).
- handcrafted per-form features (mean, std, min, max pooled stats, tiled to D).

Same head, same data, same epochs (matched-head control). Metrics per registry row:
cross_form_transfer_per_dim, retention_per_dim, control_delta. Margin 0.05. Null: canonical tokens
tie both controls on every factor.

Config keys: payload dims per form, token_shape, matched_dim, epochs, lr, margin, seeds.

### C2. F17 missing_form_recovery (Layer 7; first OA experiment)

Design: train a head on all four aligned forms (mean-fused features). At eval, drop each form in
turn (cycle so every form is the missing one). Arms:

- recovery arm: mean-fuse the remaining aligned forms.
- impute-by-mean: substitute the training mean of the missing form, fuse all four.
- zero-filled concat: concat features with zeros in the missing slot.
- best-remaining single form (the control to beat).

Confidence: softmax max of the head. Score absence_ece with diagnostics riskcov ece_equal_mass on
the dropped-form episodes. OA wiring: `missing_form_detection` with per-form reconstruction error
(from the affine alignment residual) as the detector score against the true dropped mask;
`confidence_calibration` on the recovery arm. Density: recovery_per_extra_flop (extra imputation
FLOPs via diagnostics compute helpers).

Null (registry): recovery ties best remaining form, or confidence fails to change under absence.

### C3. F9 cross_form_compositional_binding (Layer 2/4)

Design: two ground factors a (4 values) and b (4 values). Form A's projection is dominated by a
(b attenuated to 0.2x), form B's by b (a attenuated). Held-out grid: train on the anti-diagonal
complement of (a, b) combos, test on held-out combos where the answer requires binding a-from-A
with b-from-B over the shared referent. Reuse diagnostics/held_out_combo.py for the report.
Controls: shuffled-label floor, single-form conjunction baseline (train and test on one form only),
shuffled-referent pairs. Seed stability required (this family sign-flipped before, at 3 factors).
Start at 2 factors, add a third only after D3 difficulty calibration certifies the bed.

### C4. F10 intrinsic_form_curriculum (Layer 7)

Design: lessons are (form, item-batch) pairs across the four forms plus a fifth noisy-TV form
(labels re-randomized each epoch, fresh noise each draw so it cannot be memorized). Schedulers:
learning-progress (reuse devel/curriculum.py), uniform, prediction-error, novelty. Metrics:
coverage_per_update (held-out acc across all real forms per update spent), noisy_form_timeshare
(fraction of picks on the noisy form; the e4 lesson says this is the metric that kills), transfer
gain. Null: learning progress ties uniform or chases the noisy form.

### C5. F12 private_form_language_stability (Layer 4)

Design: k-means codebook (k=16) over canonical aligned features, fit per seed. Score cross-seed
Hungarian code agreement (diagnostics/seed_consistency.py) against a random-codebook floor, and
cross-seed probe transfer through the codes. The idiolect record (p5/s5/y3) is the prior: expect a
null, publish it cleanly if it lands. Null: codes are idiolects (agreement at or below floor).

### C6. F19 cross_scale_referent_binding (Layer 6)

Design: nested referent scheme `t{task}/e{episode}/s{scene}/o{object}`; world generates object
latents that compose into scene aggregates, scenes into episodes. Memory arms at MATCHED BYTES
(cap slots so byte totals equal): flat object-vector store, single-scale stores, hierarchical
store (per-scale KVIndex with parent pointers, reuse shell/buffer.py KVIndex, report faiss vs
brute backend in the receipt). Queries: store objects, query with an episode-level aggregate, and
the reverse. Metrics: cross_scale_recall_at_k, flat_memory_recall_at_k, recall_per_byte. Null:
hierarchy ties flat at matched bytes.

### C7. F13 form_energy_budget (Layer 9; the frontier, not a point)

Design: sweep grid over form width {4, 8, 16, 32, 64}, token count {1, 4, 16}, and head size
{small, matched, large}; each cell runs the F2 transfer task; collect (capability, cost) points
with cost from the density blocks (FLOPs and bytes). Report the Pareto frontier and its area
(diagnostics riskcov pareto_frontier / pareto_area) for form-token cells vs raw-feature cells.
Null: same frontier. Output receipt: runs/f13_density_frontier.json plus the frontier table in the
result dict. This experiment is also the template for how density claims are made program-wide.

### C8. F18 counterfactual_form_intervention (Layer 2/8)

Design: referents are before-and-after pairs under a programmatic do-intervention on factor a
(shift by delta in {1, 2, 3}); both forms re-render after the intervention. Train the intervention
predictor on deltas {1, 2}, test on unseen delta {3} (unseen-value generalization). Correlational
baseline: same architecture trained only on observational pairs at matched compute
(diagnostics compute matched_within check in the result). Controls: random intervention direction,
shuffled counterfactual pairs. Leakage metric: seen-value acc minus unseen-value acc. Reuse the
ex11 harness pattern (experiments/ex11_causal_probing.py) rather than re-deriving it.

### C9. F20 substrate_crisis_test (Layer 7/10; feeds the rewrite gate)

Design: fixture bank of matrices with KNOWN verdicts:

- positive-crisis exemplars: an arm whose features carry only nuisance for the target (synthesized
  to the A6 pattern: alignment survives raw, dies under factor partial-out), and a predictor arm
  whose rollout quality sits just above controls (the FACET12 pattern).
- healthy exemplars: arms where probes genuinely reach the target.
- false-alarm bed: noisy-TV streams (diagnostics/noisy_tv.py) that LOOK high-error but are
  aleatoric.

Crisis score candidates (score each separately, no ensemble until one wins): probe-transfer drop
under partial-out, cross-form agreement collapse, confidence collapse at stable accuracy. Score
crisis_auroc against realized probe failure with raw-error and fixed-threshold baselines
(diagnostics/operational_awareness.py crisis_detection, rewrite_caution). Null: ties raw error or
triggers on noise. The verdict wires into the F8/F16 licensing evidence stream; a positive here is
an input to the process_c-style gate, never a bypass of it.

## 5. Workstream D: natural real forms (local, serialized, and input-gated)

Preconditions: a citable natural cache, source intake, caption acceptance where used, merge manifest,
FormMatrix receipt, A6 residual guard, and adversarial verification all pass. Build and verify these
serially on the current host through the adaptive governor. Input authority, identity, and matched
controls are the first gates. A larger machine is not a prerequisite.

### D1. DR1 as a FormMatrix

Build the first real multi-form matrix: vision arm (DR1 V-JEPA store via LatentStoreFormAdapter),
caption-feature arm (the DR1 caption embeddings), programmatic sidecar arm (symbolic), random-init
control arm (the matched random-init cache). Then run the real variants of F1, F5, F9, F12 on it:
same experiment ids, provenance tag natural-video distinguishes them, null cards either way.
Every alignment claim passes the A6 residualization before the word semantic appears anywhere.

### D2. OA block in the evidence scorecard

Extend src/mop/studio/scorecard.py with an operational-awareness section reading a
runs/mot/oa_suite.json receipt (produced by F17/F20 real runs). Missing receipt stays pending;
toy-scale OA runs are non-scoring, same rule as PR9 smoke runs.

### D3. Density receipts into the artifact bundle

F-series run receipts (including the F13 frontier) get rows in the artifact bundle index so
density evidence is durable, hash-pinned, and citable like everything else.

## 6. Workstream E: deprecation and renames (Phase 5, only after B3 has soaked)

Trigger: B3 merged, two weeks of green runs (or the DR1 lane exercised on the merged stack once),
zero grep hits for stale imports.

- Physical renames, each its own commit, each preceded by the grep proof:
  PerspectiveAdapter aliases deleted; SubstrateAdapter conceptually presented as
  InheritedSubstrateAdapter in docs (physical class rename only if the grep shows a small surface);
  LatentStore gains the FormFeatureStore alias only if anything actually needs the name.
- substrate/__init__.py docstring updated to describe the form layer as the package's interface.
- 13_code_scaffolding.md gets a superseded-by-FORM_SUBSTRATE_CODEMAP.md banner.
- Package rename: still deferred (MIGRATION_PHASES.md Phase 6 rule: a rename must move a number).

## 7. Workstream F: plastic branch natural-evidence preconditions

F7 now runs locally. F8/F16 now have a complete fail-closed local execution engine, including fixture
science, four matched-estimator arms, provenance checks, and resource receipts. Do not run or promote
their natural-data claim until EVERY item below exists as a receipt on disk:

1. A PR9 verdict ledger showing the CBP kill-switch fired (moldability dead at frozen), OR a DR1
   representational wall (integrity-clean cache where the target is unreachable by matched shells).
2. A process_c-style license gate JSON with launch_allowed true for the specific F-task.
3. Doc 15 triggers satisfied and written down (which trigger, what evidence, what named property).
4. An F20 crisis verdict consistent with the wall (crisis detected where the wall is claimed).

When natural evidence opens, F8 runs before F16 (rewrite a small encoder on the same curriculum first;
the perfect slate is the escalation, not the entry). Both use a declared estimated end-to-end FLOP
convention and remain nonpromotable until an external rights and weight-provenance authority attests the
package. This is an environment gate, not a Studio claim; no measured M3 failure exists.

## 8. Verification protocol (run at every task boundary)

```text
PYTHONPATH=src .venv/bin/python -m pytest tests/unit -q            (must be green)
PYTHONPATH=src .venv/bin/python -m pytest tests/integration/test_f_series.py -q
PYTHONPATH=src .venv/bin/python scripts/devel.py experiments        (validate + counts)
PYTHONPATH=src .venv/bin/python scripts/devel.py experiments --render   (after registry edits)
PYTHONPATH=src .venv/bin/python scripts/check_docs.py               (after any .md change)
PYTHONPATH=src .venv/bin/python scripts/acceptance.py               (before every commit; 10/10)
PYTHONPATH=src .venv/bin/python -m mop.harness.cli experiment=<id>  (smoke the touched experiment)
```

Commit discipline: one task per commit, imperative one-line message matching the repo style, no
AI attribution of any kind, tests green before and after. Registry edits and their EXPERIMENTS.md
re-render travel in the same commit.

## 9. Risk register

| Risk | Where | Mitigation |
|---|---|---|
| B3 merge breaks the DR1 lane quietly | perspectives consumers | single-commit merge, lane tests in the same run, revert-not-patch rule |
| Toy F-beds ceiling and fake positives | C1-C9 | difficulty calibration gate before trusting any tie or win; the F3 width probe doubles as the bed check |
| OA metrics overclaimed into awareness language | C2, C9, D2 | rail on render paths; no composite OA score exists by design; recorded-null comparisons mandatory |
| Density ratios cited without matched cost | C7, everywhere | acceptance rule: density claims name their matched-cost control or they are bookkeeping |
| Heavy-lane contention corrupts attribution | D | serialize encode/training jobs through the governor; light verification may run in parallel |
| Ledger drift from new docs | all | check_docs in the protocol; new .md and its ledger line in the same commit |
| Sign-flip positives in memory/binding toys | C3, C6 | seeds [0..4] minimum for any candidate positive; riskcov seed_ci + sign_flip_report in the result |

## 10. Sequencing summary

```text
A1                                  (integration parity; half a session)
B1 -> B2 -> B3 -> B4                (consolidation; B3 is the careful one)
C1, C2                              (parallel-safe after B1; each about a session)
C3 -> C4 -> C5                      (compositional, curriculum, codes)
C6 -> C7                            (memory scale, density frontier)
C8 -> C9                            (intervention, crisis)
D1 -> D2 -> D3                      (local, serialized, after citable input and control receipts)
E                                   (after B3 soaks)
F                                   (license-gated; possibly never; a clean never is a result)
```

Definition of done for this plan: F4, F9, F10, F12, F13, F17, F18, F19, F20 flipped to implemented
with green gates and receipts; one matrix stack; DR1 matrix running real F1/F5/F9/F12; OA and
density blocks in the evidence scorecard; zero unledgered docs; acceptance 10/10 throughout.

---

## 11. Current standing goal

The original embedded prompt is retired because it encoded a false Studio gate and stale completion
counts. Use `MOP_MAXIMUM_POTENTIAL_GOAL.md` as the standing execution prompt. It preserves the valid
constraints in this plan, adds the claim-level audit, 37-facet scoring contract, 300-minute adaptive
governor, P4/P5 heavy-lane order, Wave E0 shared substrate, and measured hardware gate.
