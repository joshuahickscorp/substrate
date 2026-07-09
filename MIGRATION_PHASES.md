# MIGRATION PHASES

The execution order of the paradigm migration. Phases are gates, not dates: a phase closes when its
checks pass, and nothing in a later phase may break an earlier phase's green state. BLACKHOLE rules
apply throughout: behavior conserved, tests green before and after, assets untouched, every collapse
logged with a number.

House style: no em or en dashes. Companions: PARADIGM_MIGRATION.md (what moves),
FORM_SUBSTRATE_CODEMAP.md (module actions), FORM_SUBSTRATE_EXPERIMENTS.md (bank).

---

## Phase 0: Archive and root docs (this migration, done in place)

- New root docs land: FORM_SUBSTRATE_PROGRAM.md, FORM_SUBSTRATE_DOCTRINE.md,
  FORM_SUBSTRATE_CODEMAP.md, FORM_SUBSTRATE_EXPERIMENTS.md, PARADIGM_MIGRATION.md,
  PERFORMANCE_DENSITY_DOCTRINE.md, OPERATIONAL_AWARENESS.md, MIGRATION_PHASES.md, LEGACY_INDEX.md.
- docs/archive/ created with pointer indexes (mop_legacy, vjepa_legacy, biology_levers_legacy).
  Legacy docs are archived IN PLACE via LEGACY_INDEX.md status rows; no file moves (the markdown
  ledger, cross-references, and the /goal stack point at current paths).
- scripts/check_docs.py ledger extended with every new doc (nothing unledgered, nothing stale).
- Registry gains F17-F20 rows; EXPERIMENTS.md re-rendered from the registry.
- Fix the known stale reference: GO.md points at docs/mixture_of_thinking/ (does not exist).
- Exit check: scripts/check_docs.py clean, registry validation clean, unit tests green.

## Phase 1: Density and awareness instruments (first new code)

- src/mop/diagnostics/performance_density.py: density block builder (capability, cost, density)
  composing diagnostics/compute.py, substrate/storage.py, wall-clock. Unit test.
- src/mop/diagnostics/operational_awareness.py: OA1-OA8 composites over riskcov, calibration,
  noisy_tv, ensemble, metacognition. All rendered text through the sentience rail. Unit test.
- F1/F2/F3/F5 result dicts gain density blocks (additive change, no behavior change).
- Exit check: new tests green, F1-F5 smoke runs emit density blocks.

## Phase 2: Form interface consolidation (no duplicate stacks)

- Add LatentStoreFormAdapter and SubstrateFormAdapter bridges so cached real latents enter the form
  matrix (mirror of the perspective bridges).
- Collapse the byte-identical helpers: one _referent_tuple, one _factor_dict, one referent-ordering
  implementation shared by form and perspective matrix builders.
- Decide the merge direction (form absorbs perspectives or perspectives grows FormMeta fields) by
  import blast radius; execute with tests green. One referent-aligned matrix stack survives.
- cache_manifest.py grows form kind and objective fields (schema version bump, back-compat read).
- Exit check: dr1_perspectives and native_lanes still green on the merged stack.

## Phase 3: F-series over real forms (Studio spine unchanged)

- DR1 remains first: its cache plus captions is the first real two-form referent corpus.
- Port F1/F5/F9/F12 from toy worlds to the DR1 matrix (vision form + caption form + programmatic
  sidecars + random-init control arm), behind the existing DR1 acceptance gates.
- Implement F4 (raw payload vs form tokens) and F17 (missing-form recovery) as cpu-now.
- Add tests/integration/test_f_series.py (harness-level, like the E/EX series have).
- Exit check: F-series runs on real referents produce null cards either way.

## Phase 4: Awareness and crisis lanes

- Implement F20 (substrate crisis test) using the A6 and FACET12 records as positive-crisis
  exemplars and noisy-TV as the false-alarm bed.
- Implement F18 (counterfactual form intervention) reusing the ex11 harness.
- OA scorecard block added to the studio scorecard (receipt-backed, non-scoring until real runs).
- Exit check: OA metrics reported with baselines; no OA claim without its recorded-null comparison.

## Phase 5: Deprecate old names (conceptual becomes physical)

- Rename in code what Phase 2 merged in concept, only where grep shows the blast radius is closed:
  PerspectiveAdapter to FormPerspectiveAdapter (or the merged single name), SubstrateAdapter to
  InheritedSubstrateAdapter, LatentStore alias FormFeatureStore.
- Update substrate/__init__.py docstring (the "nothing here trains" sentence gains the license-gate
  qualification if and only if F8/F16 ever license a trainable arm).
- Exit check: zero orphan imports, tests green, EXPERIMENTS.md re-rendered, ledger clean.

## Phase 6: Optional package rename (explicitly deferred)

- mop remains the package until imports, docs, and experiments are stable through Phase 5.
- Candidates recorded, none chosen: formsubstrate, substrate, fs, orb, blackhole, brain.
- Rule: a rename is a BLACKHOLE collapse like any other; it must move a number (fewer aliases,
  fewer stale references) or it waits.

## Phase 7: Plastic branch (evidence-gated, may never open)

- F7/F8/F16 implementations begin only when their licenses exist: a PR9 kill-switch receipt or a
  DR1 representational wall (process_c_gate pattern), plus doc 15 triggers.
- The perfect slate (F16) runs on the same curriculum as the inherited arms, matched compute,
  with old-form retention scored. Either verdict is publishable.

## The first ten implementation tasks (concrete, in order)

1. scripts/check_docs.py: add the nine new root docs plus doc 16 and archive indexes to the ledger.
   Expected: docs gate exits 0. (Phase 0)
2. registry/experiments.yaml: add F17-F20 rows (schema-valid, sentience-rail-clean).
   Expected: validate_experiments clean. (Phase 0)
3. Re-render EXPERIMENTS.md via scripts/devel.py experiments --render.
   Expected: 195 catalogued, F-series section shows 20 rows. (Phase 0)
4. Fix GO.md stale docs/mixture_of_thinking/ path to docs/mixture_of_perspectives/.
   Expected: no dead paths in operational docs. (Phase 0)
5. src/mop/diagnostics/performance_density.py + tests/unit/test_performance_density.py.
   Expected: density_block(result, flops=, bytes=, seconds=) returns capability/cost/density dict;
   tests cover zero-cost guard and ratio math. (Phase 1)
6. src/mop/diagnostics/operational_awareness.py + tests/unit/test_operational_awareness.py.
   Expected: oa_suite(...) returns OA1-OA8 sub-scores with baselines; render path rail-gated. (Phase 1)
7. f_form_substrate.py: attach density blocks to F1/F2/F3/F5 results.
   Expected: result dicts gain density sub-dict; existing tests unchanged. (Phase 1)
8. substrate/form.py: LatentStoreFormAdapter bridge + shared helper collapse with perspectives.
   Expected: real cached latents load as a form arm; duplicate helpers deleted. (Phase 2)
9. tests/integration/test_f_series.py: harness-level F1-F5 run on fixture caches.
   Expected: integration parity with E/EX series. (Phase 2-3)
10. F17 missing_form_recovery implementation (cpu-now toy first) + config + registry flip to
    implemented. Expected: OA1/OA2 become experimental, with impute-by-mean and best-remaining-form
    baselines. (Phase 3)
