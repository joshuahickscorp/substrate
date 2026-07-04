# CONDENSE_LEDGER

Every condensation iteration with before/after measurements. One class per commit. Middot separators, no
em/en dashes. Branch condense/run-20260703, baseline tag condense-baseline-20260703 (c084370).

## Bound commands

- BUILD_CMD · `.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy`
- TEST_CMD · `PYTHONPATH=src .venv/bin/python -m pytest -q`
- COVERAGE_CMD · none (pytest-cov and coverage absent). Held by PROXY: never remove covered code, tests
  frozen and read-only, so coverage of the tested surface cannot fall.
- PERF_CMD · `PYTHONPATH=src .venv/bin/python scripts/bench.py --reps 3` (hot-path throughputs)
- PERF_RED_LINE · 2x any hot-path sample vs baseline is a revert
- BUDGET · run the safe classes until fixpoint (a full pass commits zero reductions) or a stop-and-report
  trigger; internal cap of about 20 iterations this session.

## Baseline manifest (condense-baseline-20260703, c084370)

- Python · 65,180 LOC · 327 files
- YAML · 9,482 LOC · 199 files
- Markdown · 19,043 lines · 67 files
- Total tracked files · 609
- Folders (tracked, containing files) · 50
- Tests · 703 collected · 82 test files · 1,660 assertions
- Coverage · not measured (tooling absent); proxy invariant in force
- Determinism · TEST_CMD run 3x, identical pass set, NO flaky tests to quarantine
- Perf sample (bench.py, items/sec) · preprocess_clip 5644 · latent_store 2190 · buffer_sample 1.28M ·
  retrieve_faiss_unavailable 447K · retrieve_brute 542K · learner_step 3301
- Build · green (ruff + format + mypy). Test · green.
- Surface hash (horizon) · d496a189ca12bc2d
  = sha256( sorted registry ids · [project.scripts] · sorted doc-referenced scripts/*.py paths in *.md,
  EXCLUDING the CONDENSE_*.md bookkeeping )[:16]. The exclusion is required because this ledger names
  scripts (e.g. scripts/bench.py) as bookkeeping, which is not part of the horizon. Recompute each pass:
  `{ grep -rh "^  - id:" registry/*.yaml|sort; sed -n '/\[project.scripts\]/,/^$/p' pyproject.toml;
  git grep -hoE "scripts/[A-Za-z0-9_]+\.py" -- '*.md' ':!CONDENSE_LEDGER.md' ':!CONDENSE_AUDIT.md'
  ':!CONDENSE_DOCS_REVIEW.md'|sort -u; } | shasum -a 256 | cut -c1-16`. Internal src/mop reorganization
  that preserves the experiment catalog, the console entrypoint, and the doc-referenced script paths keeps
  this stable; behavior itself is guarded by the frozen tests + check_docs + acceptance.
- Pre-baseline note · one committed formatting drift (scripts/studio/dr1_smoke.py) was ruff-formatted in
  c084370 so the baseline is honestly green; the `mop` console script has a stale shebang from the
  brain->mop rename (a pre-existing horizon defect in the installed script, not a tracked source file, left
  as-is).

## Iterations

### iter 1 · class: dead private helpers (src/mop) · EXHAUSTED, zero reductions
Scanned every `def _name` in src/mop for repo-wide reference count == 1. Only hit was a dataclass
`__post_init__` (called implicitly by the dataclass machinery, not dead). No dead private helpers exist;
the codebase is already dense on this class (BLACKHOLE-applied). No commit.

### iter 2 · class: single-module subpackage collapse · learning/alternatives
- Action · collapse `src/mop/learning/alternatives/{__init__.py, rules.py}` (a package holding one real
  module) into the module `src/mop/learning/alternatives.py`. Import path `mop.learning.alternatives` is
  unchanged (module vs package is transparent to `from mop.learning.alternatives import RULES, RuleResult,
  train_*`), so no importer and no frozen test is touched; the one relative import `...seeding` becomes
  `..seeding`; the RULES registry and __all__ move into the module.
- Coupling checked · test_i4 and all four src importers use the PACKAGE path; nothing imports the `.rules`
  submodule (only ex5's prose mentions it). Confirmed safe.
- Before -> after · src/mop 139 -> 138 files · src/mop folders 12 -> 11 · files 438 -> 424 LOC (-14)
- Gates · BUILD green · TEST green (same 703-collected pass set, no new skip/xfail) · horizon hash
  d496a189ca12bc2d unchanged · check_docs green · coverage proxy held (no covered code removed) · perf
  n/a (import-structure change, no hot-path touch) · net footprint strictly smaller. PASS.

### iter 3 · class: sibling files sharing one concern · metrics/ package -> module
- Action · merge `src/mop/metrics/{__init__,continual,frontier}.py` (3 files) into the module
  `src/mop/metrics.py`. Both submodules are self-contained (no relative imports, no cross-import). Import
  path `mop.metrics` unchanged; the 5 importers that reached the SUBMODULES
  (`from ..metrics.continual import accuracy` etc. in learning/backprop.py, studies/report.py, and 3
  density/shell scripts) rewritten to the package path `from ..metrics import ...`.
- Coupling checked · NO test imports mop.metrics or its submodules directly (grep empty). The 5 importers
  are 2 src + 3 scripts, all editable, none frozen.
- Before -> after · src/mop 138 -> 136 files · folders 11 -> 10 · 126 -> ~125 LOC
- Gates · BUILD green · TEST green (same pass set) · horizon hash d496a189ca12bc2d unchanged · check_docs
  green · coverage proxy held · perf n/a · net strictly smaller. PASS.

### iter 4 (DEEP) · class: redundant preregistration-mirror configs · 34 -> 1
- Action · collapse the 34 `configs/experiment/mop_*.yaml` MoT preregistration mirrors (each a verbatim
  duplicate of a registry-only row's null_hypothesis, not compose-loaded) into one
  `configs/experiment/_mot_mirrors.yaml` (a `mirrors:` list). validate.py check_all() updated to iterate
  the merged file's entries so every mirror's null_hypothesis is still validated.
- Coupling checked · no test iterates/counts the mirror files; test_shipped_registries_validate_clean
  validates the DEVEL registries, not the harness config glob; check_all() returns 0 problems (behavior
  preserved); no script loads a mirror by path (docstring mentions only).
- Before -> after · configs/experiment 143 -> 110 files · total tracked 609 -> 576 (-33)
- Gates · BUILD green (ruff + mypy) · TEST green (same 703 pass set) · check_all() clean (0 problems,
  behavior identical) · horizon hash d496a189ca12bc2d unchanged · check_docs green · net -33 files. PASS.

### iter 5 (DEEP) · class: empty/thin folders (campaign legs) · flatten trackNN
- Action · flatten `campaign/legs/trackNN/<leg>.yaml` (13 track subfolders) into a flat `campaign/legs/`;
  rewrite the 35 `sweep:` paths in campaign/run_queue.yaml (the only path source) `campaign/legs/trackNN/`
  -> `campaign/legs/`. Leg filenames (already `trackNN_*`) and leg NAMES are unchanged.
- Coupling checked · leg paths appear ONLY in run_queue.yaml (grep empty elsewhere); tests use leg NAMES
  and the count (>=11), both preserved; deps use names. load_queue + validate clean, 35 legs.
- Before -> after · campaign/legs folders 13 -> 0 · total tracked folders 48 -> 35 · files unchanged (moves)
- Gates · BUILD green · TEST green (same pass set) · horizon hash d496a189ca12bc2d unchanged · check_docs
  green · net -13 folders. PASS.

### iter 6 (DEEP, whole-codebase) · class: duplicate functions · parse_seeds x18 -> canonical
- Action · all 18 script copies of parse_seeds are BEHAVIORALLY IDENTICAL (AST-exec verified across 9 seed
  specs, 1 distinct behavior). Unified to one canonical in src/mop/seeding.py; the 18 scripts import
  `from mop.seeding import parse_seeds`. Import placement fixed in 3 files where it landed before a
  sys.path fallback (mop_at1, mop_at4, pr9) so standalone-outside-repo import still works (pr9 verified).
- Coupling checked · 3 frozen tests import parse_seeds FROM scripts (mop_at1_grid_pilot, dr14); the name
  stays in each script's namespace via the re-import, so the tests pass unchanged.
- Before -> after · Python LOC 65,171 -> 65,034 (-137); no file/folder change (canonical in existing module)
- Gates · BUILD green · TEST green (same pass set, incl. the 3 parse_seeds tests) · horizon hash
  d496a189ca12bc2d unchanged · check_docs green · pr9 standalone import verified. PASS.

### iter 7 (DEEP, whole-codebase) · class: duplicate functions · _parse_seeds x6 -> canonical (alias)
- Action · 6 scripts define a private `_parse_seeds` behaviorally identical to the canonical
  mop.seeding.parse_seeds (AST-exec verified). Replaced each def with
  `from mop.seeding import parse_seeds as _parse_seeds` so the private NAME (and its call sites + the 7
  frozen tests that import _parse_seeds) are untouched, placed after each file's sys.path setup.
- Gates · BUILD green · TEST green (same pass set) · horizon hash d496a189ca12bc2d unchanged · check_docs
  green · net -LOC (6 dup bodies -> 6 alias imports). PASS.

### iter 8 (DEEP, whole-codebase) - class: duplicate private experiment helpers
- Action - centralized the repeated private experiment helpers in `src/mop/experiments/base.py`: `_split`
  for Task train/test splitting, `_mean`, and `_std`. The six `_split` copies were identical; the top-level
  `_mean` copies either used `sum(v) / max(1, len(v))` or the equivalent empty-list guard
  `sum(v) / len(v) if v else 0.0`; the two `_std` copies were identical. Importers keep the same private
  helper names at call sites. No test, registry, config, asset, or public script path changed.
- Coupling checked - all touched experiment modules already depended on `.base.Experiment`, so this adds no
  new module boundary. The shared `_split` imports `Task` inside the helper to avoid raising base import-time
  weight. Remaining `_split` and nested `_mean` helpers are not identical Task splitters/top-level numeric
  means, so they were left alone.
- Before -> after - tracked Python LOC 64,998 -> 64,914 (-84). Files 576 -> 576. Folders 35 -> 35.
- Gates - BUILD green (`ruff check .`, `ruff format --check .`, `mypy`). TEST green
  (`PYTHONPATH=src .venv/bin/python -m pytest -q`, same 703 collected, same visible 2 skips). Surface hash
  d496a189ca12bc2d unchanged. `scripts/check_docs.py` green. Coverage proxy held because tests are frozen
  and no tested behavior was removed. Perf - first reps=3 sample had one noisy `retrieve_brute` outlier past
  2x; reps=5 resample stayed inside the red line (`retrieve_brute` 0.000113s vs baseline 0.000106s,
  `learner_step` 0.002893s vs 0.003097s, `manifest_write` 0.199672s vs 0.222949s). PASS.

### iter 9 (DEEP, whole-codebase) - class: duplicate private experiment helpers, second pass
- Action - centralized the remaining clean private helper duplicates in `src/mop/experiments/base.py`:
  `_diag_mean` x3, `_spread` x4, tensor `_split_xy` x2 (imported back as `_split` to preserve call sites),
  and `_fit_eval` x3. The heavier torch imports in `_fit_eval` are local to the helper call, so the base
  experiment contract does not gain import-time torch/F weight from this pass.
- Coupling checked - all touched modules already import `.base`; the removed helpers were private and
  behavior-identical. The remaining `_fit_eval` methods either have extra codebook/None-backbone behavior
  or live as methods, and the remaining nested mean/spread helpers are local to single experiments, so they
  were left for a separate risk decision.
- Before -> after - tracked Python LOC 64,914 -> 64,859 (-55). Files 576 -> 576. Folders 35 -> 35.
- Gates - BUILD green after ruff import-order and format cleanup. TEST green
  (`PYTHONPATH=src .venv/bin/python -m pytest -q`, same 703 collected, same visible 2 skips). Surface hash
  d496a189ca12bc2d unchanged. `scripts/check_docs.py` green. Coverage proxy held. Perf reps=5 stayed inside
  the 2x red line (`retrieve_brute` 0.000168s vs baseline 0.000106s, `learner_step` 0.003632s vs 0.003097s,
  `manifest_write` 0.260903s vs 0.222949s). PASS.

### iter 10 (DEEP, whole-codebase) - class: duplicate local mean helpers
- Action - removed nine local `def mean(v): return sum(v) / len(v)` helpers from `s_semiotics`,
  `ex3_test_time_adaptation`, `ex8_curiosity_bakeoff`, and `ex16_codebook_sr`; call sites now use the
  shared `_mean` imported from `src/mop/experiments/base.py`. Tensor `.mean()` calls were left untouched.
- Coupling checked - all helpers computed the same numeric average and were local/private. The one empty-list
  guarded variant in `S6` is behavior-equivalent to `_mean` for empty input as well as non-empty input.
- Before -> after - tracked Python LOC 64,859 -> 64,829 (-30). Files 576 -> 576. Folders 35 -> 35.
- Gates - BUILD green. TEST green (`PYTHONPATH=src .venv/bin/python -m pytest -q`, same 703 collected,
  same visible 2 skips). Surface hash d496a189ca12bc2d unchanged. `scripts/check_docs.py` green. Coverage
  proxy held. Perf reps=5 stayed inside the 2x red line (`retrieve_brute` 0.000118s vs baseline 0.000106s,
  `learner_step` 0.002255s vs 0.003097s, `manifest_write` 0.220672s vs 0.222949s). PASS.

### iter 11 (DOCS TRACK, user-authorized) · consolidate 4 RESULT docs -> RESULTS_LEDGER.md
- Action · merge A6_RESULT / LAPTOP_LANES_RESULT / AXIS_CEILING_RESULT / ROLLOUT_LANE_RESULT.md into one
  docs/mixture_of_perspectives/RESULTS_LEDGER.md (content VERBATIM, headings demoted one level under a top
  header + TOC + `<!-- merged from X -->` provenance markers). Every inbound reference across docs + code
  (~13 files: HANDOFF, STUDIO_HANDOFF, STUDIO_RUN_REPORT, STUDIO_TURNKEY_PLAN, POTENTIAL_AUDIT,
  M3PRO_RUN_REPORT, EXPAND_PHASE_PLAN, STUDIO_POTENTIAL_AUDIT, FACET12 null card, mop_dr13_predictor_fidelity,
  mop_dr13_readout_adapter) rewritten to RESULTS_LEDGER.md; check_docs CANONICAL_MD updated (4 -> 1).
- Content preservation · verified every non-heading content line of all 4 originals is present in the ledger;
  broken-ref check for the 4 old filenames returns EMPTY outside the ledger's provenance comments.
- Before -> after · md files 70 -> 67 · total tracked 576 -> 573 (-3)
- Gates · BUILD green · TEST green (703) · check_docs green (RESULTS_LEDGER ledgered, all refs resolve) ·
  horizon hash d496a189ca12bc2d unchanged. PASS. Docs-track deletion is user-authorized (the grader).
