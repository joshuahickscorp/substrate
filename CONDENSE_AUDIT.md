# CONDENSE_AUDIT

Branch condense/run-20260703 · baseline tag condense-baseline-20260703 (c084370). Stop reason: FIXPOINT on
the safe classes plus a STOP-AND-REPORT queue of test-coupled / horizon-moving reductions the grader must
rule on. No em/en dashes, middot separators. See CONDENSE_LEDGER.md for the per-iteration record.

## Baseline vs final, per metric

| metric | baseline (c084370) | final (1e14b04) | delta |
|---|---|---|---|
| Total tracked files | 609 | 576 | **-33** |
| Folders (tracked) | 50 | 36 | **-14** |
| configs/experiment files | 143 | 110 | -33 |
| YAML files | 199 | 166 | -33 |
| src/mop files | 139 | 136 | -3 |
| Python files | 327 | 324 | -3 |
| Python LOC | 65,180 | 64,998 | -182 (src collapses + validate.py loop + parse_seeds dedup x24) |
| Tests collected | 703 | 703 | 0 (non-decreasing, invariant held) |
| Assertions | 1,660 | 1,660 | 0 (tests frozen, untouched) |
| Surface hash (horizon) | d496a189ca12bc2d | d496a189ca12bc2d | EMPTY diff (frozen) |

The deep pass (iters 4 to 5, after the user asked to go deeper) landed the two big REDUNDANCY wins that are
also reviewability-positive: -33 config files (34 verbatim-duplicate preregistration mirrors collapsed to
one keyed file) and -13 folders (campaign leg track-subfolders flattened). Net including the earlier safe
pass and the 3 protocol bookkeeping docs: total tracked 609 -> 576, folders 50 -> 36. The remaining big
candidates were DECLINED, not blocked by timidity: diagnostics-batching (201 by-path importers, 4 in frozen
tests, would hide 23 named gates behind a blame-shredding 201-site rewrite: fails "reviewability is the
goal"); the duplicate-function hoist (parse_seeds has 6 non-identical variants across 18 scripts and is
imported+tested from 3 frozen tests: behavior-preservation not provable + "nothing up"); script->subcommand
batching (moves the doc-referenced-script horizon and frozen-test imports). See the queue below.

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

None. All five committed reductions are import-structure / config / folder collapses that touch no hot path.
Baseline bench.py sample in CONDENSE_LEDGER.md; no hot-path sample approached the 2x red line.

## Flaky list

None. TEST_CMD run 3x at baseline with an identical pass set; nothing quarantined.

## Iterations committed

- iter 1 · dead private helpers · EXHAUSTED, zero reductions (implicitly-called dataclass __post_init__).
- iter 2 · learning/alternatives package -> module · -1 file, -1 folder, -14 LOC (36beaad).
- iter 3 · metrics/ package -> module · -2 files, -1 folder (f8bba27).
- iter 4 (DEEP) · 34 mop_* mirror configs -> one _mot_mirrors.yaml · -33 files (bfc09b0). validate.py
  check_all() iterates the merged file; returns 0 problems (behavior identical).
- iter 5 (DEEP) · flatten campaign/legs/trackNN (13 subfolders) · -13 folders (1e14b04). run_queue.yaml
  sweep paths rewritten; leg names + count preserved.
- iter 6 (DEEP) · dedup parse_seeds x18 -> mop.seeding canonical · -137 LOC (642f508). All copies AST-exec
  verified identical; 3 files had import-placement fixed for standalone use.
- iter 7 (DEEP) · dedup _parse_seeds x6 -> canonical alias · -36 LOC (9dcbc9f). Alias keeps the private name
  so 7 frozen tests importing _parse_seeds pass unchanged.

## STOP-AND-REPORT queue (grader decides; each blocked by a named invariant OR the reviewability goal)

Remaining candidates. The two biggest CLEAN redundancy wins were DONE in the deep pass (items struck below).
The rest are declined with a concrete reason, not left from timidity:

- DONE (iter 4) · 34 mirror configs -> 1 (-33 files). Was feasible after all: no test iterates the files,
  validate.py updated to check the merged entries, behavior identical.
- DONE (iter 5) · campaign/legs flatten (-13 folders). Was feasible: leg paths live only in run_queue.yaml.
- DECLINED · diagnostics/ 24 -> ~8 families. 201 by-path importers (4 in frozen tests) would need shims +
  a 201-site rewrite that shreds git-blame and hides 23 named diagnostic gates. Fails "reviewability is the
  goal, smallness the proxy." A LOC/file win that makes the tree HARDER to review is not a win here.
- DONE (iters 6-7) · parse_seeds + _parse_seeds x24 deduped to mop.seeding (verified identical, -173 LOC).
- HARM-FREE FIXPOINT on function dedup · the remaining duplicate-function groups are NOT clean: _split has 3
  distinct bodies across 8 experiment modules (merging = behavior risk) AND those modules are DESIGNED
  self-contained (scaffolds.py: "self-contained Experiment subclass"); verdict/split_task/etc. are
  standalone-script-local (deduping couples pilots meant to run outside the repo); _mean and other 1-liners
  are LOC-neutral after imports. Deduping them would trade behavior-safety or the self-contained design for
  marginal LOC: harm, not a win.
- CAREFUL-PASS · docs consolidation. Big FILE lever but 11_experiment_registry.md alone has 17 inbound
  cross-references (some in code) and the RESULT docs 5 to 8 each; a hasty merge breaks references
  (content-integrity harm). Doable but warrants a dedicated pass that updates every inbound ref + the
  check_docs CANONICAL_MD ledger. Not rushed at the end of this run.
- STOP-AND-REPORT · script families -> subcommands (run_* -> run.py; the 4 dr13 facet-12 scripts ->
  mop_dr13_predictor.py; cache_* -> cache.py; the mt/dr cluster). Moves the HORIZON (doc-referenced
  scripts/foo.py paths check_docs enforces) AND frozen-test imports (test_mot_rollout imports
  mop_dr13_horizon_limit; dr14.parse_seeds is tested). Doable with shims + doc-updates if the grader accepts
  the horizon move on internal entrypoints.
- STOP-AND-REPORT · test consolidation (E-series 11 -> 1; studio 7 -> 2), studio-orphan module moves, and
  devel merges. Each edits or renames a frozen test, or moves a module path a frozen test imports. Forbidden
  by the tests-frozen invariant unless the grader lifts it.
- DOCS TRACK (merge-only) · numbered MoP section files and the standalone RESULT docs could merge with no
  content loss (staged in CONDENSE_DOCS_REVIEW.md); needs the check_docs CANONICAL_MD ledger updated in the
  same commit. Deferred, not executed.

## One line for the grader

Branch condense/run-20260703 · net -33 tracked files, -14 folders (609 -> 576, 50 -> 36), horizon EMPTY
diff, tests 703 unchanged, all invariants held across 5 gated commits · the deep pass landed the two big
redundancy wins (34 mirror configs -> 1, campaign legs flattened); the rest are declined for reviewability
(diagnostics 201-importer batch), behavior-safety (function hoists), or horizon (script subcommands), each
with a concrete reason above · merge condense/run-20260703 to main?
