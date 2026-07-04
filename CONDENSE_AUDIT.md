# CONDENSE_AUDIT

Branch condense/run-20260703. Baseline tag condense-baseline-20260703 at c084370. Final committed head for
this audit: 7f25f39. Stop reason: STOP-AND-REPORT because the worktree became dirty with unrelated
doc/script edits and an untracked docs ledger after iter 10; another reduction would violate the clean
snapshot rule and risk mixing ownership. The metrics below are for committed HEAD only, excluding those
uncommitted worktree changes. See CONDENSE_LEDGER.md for the per-iteration record.

## Baseline vs Final

| metric | baseline c084370 | final 7f25f39 | delta |
|---|---:|---:|---:|
| Total tracked files | 609 | 576 | -33 |
| Folders, tracked dirs with files | 49 | 35 | -14 |
| configs/experiment files | 143 | 110 | -33 |
| YAML files | 199 | 166 | -33 |
| Markdown files | 67 | 70 | +3, condense reports |
| Markdown lines | 19,043 | 19,336 | +293, condense reports |
| src/mop Python files | 139 | 136 | -3 |
| Python files | 327 | 324 | -3 |
| Python LOC | 65,182 | 64,829 | -353 |
| Tests collected | 703 | 703 | 0 |
| Test files | 82 | 82 | 0 |
| Assertion-like checks | 1,663 | 1,663 | 0 |
| Surface hash | d496a189ca12bc2d | d496a189ca12bc2d | empty diff |

Net after this continuation: iter 8 to iter 10 removed another 169 Python LOC beyond the prior audit, with
no file or folder increase and no public-surface movement.

## Invariants

- Public contract frozen - surface hash d496a189ca12bc2d held at every committed pass.
- Behavior unchanged - full TEST_CMD passed after each committed reduction, with the same 703 collected tests
  and the same visible 2 skips.
- Build green - ruff check, ruff format check, and mypy passed after each committed reduction.
- Assets untouched - no model, weight, fixture, cache, binary, run artifact, or dataset path was moved,
  rewritten, or deleted by the committed reductions.
- Tests frozen - no test file was edited, renamed, deleted, weakened, skipped, or xfailed. Test and assertion
  counts are non-decreasing.
- Coverage - direct coverage tooling is absent. Coverage proxy held: tests stayed frozen and all green, while
  only duplicate private helper bodies were removed in iters 8 to 10.
- Docs - no committed docs merge or deletion was made in this continuation. A docs consolidation artifact is
  present uncommitted in the dirty worktree and is not part of this audit head.

## Perf

No committed reduction breached the 2x red line on resample. Iter 8 had one noisy `retrieve_brute` reps=3
outlier; the reps=5 resample returned inside the red line. Iter 9 and iter 10 reps=5 samples were inside the
red line for all tracked benches. The final iter 10 sample included `retrieve_brute` 0.000118s vs baseline
0.000106s, `learner_step` 0.002255s vs 0.003097s, and `manifest_write` 0.220672s vs 0.222949s.

## Flaky List

None. Baseline TEST_CMD was run three times with the same pass/skip shape; nothing was quarantined.

## Iterations Committed

- iter 1 - dead private helpers - exhausted, zero reductions.
- iter 2 - learning/alternatives package to module - -1 file, -1 folder, -14 LOC - 36beaad.
- iter 3 - metrics package to module - -2 files, -1 folder - f8bba27.
- iter 4 - redundant preregistration mirror configs 34 to 1 - -33 files - bfc09b0.
- iter 5 - flatten campaign/legs/trackNN folders - -13 folders - 1e14b04.
- iter 6 - dedup parse_seeds x18 to mop.seeding - -137 LOC - 642f508.
- iter 7 - dedup _parse_seeds x6 via canonical alias - -36 LOC - 9dcbc9f.
- iter 8 - dedup private experiment helpers `_split`, `_mean`, `_std` - -84 LOC - b8ce07b.
- iter 9 - dedup `_diag_mean`, `_spread`, tensor split, and `_fit_eval` helpers - -55 LOC - f191327.
- iter 10 - dedup local mean helpers in semiotics and small experiment modules - -30 LOC - 7f25f39.

## Stop Details

After iter 10, the working tree contains unrelated uncommitted changes:

- Modified docs: STUDIO_HANDOFF.md plus multiple docs/mixture_of_perspectives result and plan files.
- Modified proof/script files: proof/NULL_CARDS/FACET12-ROLLOUT-FIDELITY.md,
  scripts/mop_dr13_predictor_fidelity.py, and scripts/mop_dr13_readout_adapter.py.
- Untracked docs/mixture_of_perspectives/RESULTS_LEDGER.md.

I amended iter 10 to remove the accidentally included RESULTS_LEDGER.md from the commit while leaving the
file on disk. Continuing condensation from a dirty tree would violate the required clean snapshot and could
mix these external edits into a reduction commit, so the run stops here for review. In the current dirty
worktree, `scripts/check_docs.py` fails on that untracked markdown file; the committed iter 10 gate was green
before the file was outside the tracked ledger.

## Remaining Queue

- Diagnostics package batching remains declined for now: 201 by-path importers, including frozen tests, make
  a mass rewrite harder to review than the file-count win is worth.
- Script-family subcommands remain STOP-AND-REPORT: they move doc-referenced script paths and frozen-test
  imports, so the grader must explicitly accept the horizon move.
- Test consolidation remains forbidden by the tests-frozen invariant.
- Docs consolidation remains a separate docs-track run. The RESULT docs can merge content-preservingly into
  a RESULTS_LEDGER-style file, but deletions must be reviewed and the check_docs canonical markdown ledger
  must be updated in the same docs-only commit.
- Additional class/function dedup remains possible only after a clean snapshot is restored. The next likely
  candidates are larger duplicate experiment control classes and standalone script-local helpers, each needing
  a behavior or horizon decision.

## One Line For The Grader

Branch condense/run-20260703. Final committed head 7f25f39. Net committed delta from baseline: -33 tracked
files, -14 tracked folders, -353 Python LOC, tests 703 unchanged, assertion checks 1,663 unchanged, surface
hash unchanged. Stop reason: dirty worktree with unrelated doc/script changes after iter 10. Review or shelve
those worktree changes before another condense pass or before merging this branch to main.
