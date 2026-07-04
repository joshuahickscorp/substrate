# CONDENSE_AUDIT

Branch condense/run-20260703 · baseline tag condense-baseline-20260703 (c084370). Stop reason: FIXPOINT on
the safe classes plus a STOP-AND-REPORT queue of test-coupled / horizon-moving reductions the grader must
rule on. No em/en dashes, middot separators. See CONDENSE_LEDGER.md for the per-iteration record.

## Baseline vs final, per metric

| metric | baseline (c084370) | final (f8bba27) | delta |
|---|---|---|---|
| Python files | 327 | 324 | -3 |
| Python LOC | 65,180 | 65,163 | -17 |
| src/mop files | 139 | 136 | -3 |
| src/mop LOC | 28,352 | 28,330 | -22 |
| Folders (tracked) | 50 | 48 | -2 |
| Tests collected | 703 | 703 | 0 (non-decreasing, invariant held) |
| Assertions | 1,660 | 1,660 | 0 (tests frozen, untouched) |
| Total tracked files | 609 | 609 | 0 net (3 code files removed, 3 protocol bookkeeping docs added) |
| Surface hash (horizon) | d496a189ca12bc2d | d496a189ca12bc2d | EMPTY diff (frozen) |

Net CODE condensation (excluding the three protocol bookkeeping docs CONDENSE_LEDGER/AUDIT/DOCS_REVIEW.md,
grader-facing artifacts of this run, not project code): -3 files, -2 folders, -22 LOC in src/mop. Small,
because the codebase was already dense (BLACKHOLE doctrine previously applied): zero dead private helpers,
zero byte-identical duplicate files, and the single-use private helpers that remain are substantive named
decomposition, not ceremony (inlining them would raise LOC-per-caller and HURT reviewability, the stated
goal, so they were declined).

## Invariants (all held)

- Public contract (horizon) · surface hash d496a189ca12bc2d identical at every pass. EMPTY diff.
- Behavior · TEST_CMD green every pass, same 703-collected pass set, no new skip/xfail.
- Build · green (ruff + format + mypy) every pass.
- Assets · untouched (no model, weight, cache, or fixture moved, rewritten, or deleted).
- Tests · frozen and read-only; not one test edited, renamed, deleted, weakened, skipped, or xfailed. Test
  and assertion counts non-decreasing.
- Coverage · not directly measured (pytest-cov and coverage absent). Held by proxy: no covered code removed,
  tests frozen and all green, so coverage of the tested surface cannot have fallen. Delta >= 0.
- Docs · no content lost (no doc merged or deleted this run; the doc-consolidation opportunities are staged
  in CONDENSE_DOCS_REVIEW.md, not executed).

## Perf regressions

None. The two committed reductions (iter 2 learning/alternatives, iter 3 metrics) are import-structure
collapses that touch no hot path. Baseline bench.py sample recorded in CONDENSE_LEDGER.md; no hot-path sample
approached the 2x red line.

## Flaky list

None. TEST_CMD run 3x at baseline with an identical pass set; nothing quarantined.

## Iterations committed

- iter 1 · dead private helpers · EXHAUSTED, zero reductions (only hit was an implicitly-called dataclass
  __post_init__).
- iter 2 · learning/alternatives package -> module · -1 file, -1 folder, -14 LOC (36beaad).
- iter 3 · metrics/ package -> module · -2 files, -1 folder (f8bba27).

## STOP-AND-REPORT queue (grader decides; each blocked by a named invariant)

Every remaining high-value reduction trips a stop-and-report trigger under this protocol. Ranked by value,
with the exact blocker:

1. 34 mop_* preregistration-mirror configs -> one keyed file (-33 files). BLOCKED: src/mop/harness/
   validate.py globs configs/experiment/*.yaml per file (lines 92-94) and integration tests exercise
   validate; collapsing needs a validate.py change and risks a frozen-test edit.
2. Script families -> subcommands (run_* -> run.py; the 4 dr13 facet-12 scripts -> mop_dr13_predictor.py;
   cache_* -> cache.py; the mt/dr reasoning cluster). BLOCKED: moves the HORIZON (doc-referenced
   scripts/foo.py paths that check_docs enforces) AND frozen-test imports (test_mot_rollout imports
   mop_dr13_horizon_limit).
3. diagnostics/ 24 modules -> ~8 family modules (-16 files). BLOCKED: ~135 by-path importers including frozen
   tests; needs test edits or a shim-per-module hop.
4. Test consolidation (E-series 11 files -> 1 parametrized; studio 7 -> 2). BLOCKED outright: edits/renames
   frozen tests. Forbidden by invariant 5.
5. campaign/legs/trackNN flatten (-13 folders) + track13 16 legs -> 1. BLOCKED: run_queue.yaml path rewrites
   plus test_cost_projection / test_campaign_queue coupling.
6. Studio orphan modules (mop.studio_rehearsal, mop.studio_doctor -> studio/ package). BLOCKED:
   test_studio_rehearsal imports mop.studio_rehearsal (frozen) and DECISIONS.md references it.
7. devel/ single-consumer tools merge. BLOCKED: test_devel_north_star / test_devel_metacognition couple to
   the module paths.
8. Duplicate-function hoist (parse_seeds x17, nmse x2, six synthetic clip generators). BLOCKED under the
   strict footprint rule: the hoist ADDS a shared module file ("nothing up"); a LOC-only win. Grader may
   relax "nothing up" to net-LOC.
9. Doc consolidation (numbered MoP section files, standalone RESULT docs). Docs track, merge-only, staged in
   CONDENSE_DOCS_REVIEW.md; needs the check_docs CANONICAL_MD ledger updated in the same commit. No content
   loss; deferred, not executed.

## One line for the grader

Branch condense/run-20260703 · net -3 code files, -2 folders, -22 src/mop LOC, horizon EMPTY diff, tests 703
unchanged, all invariants held · the big wins (items 1-9) are stop-and-report, each blocked by a frozen test
or the horizon · merge condense/run-20260703 to main? Then rule item-by-item on the queue.
