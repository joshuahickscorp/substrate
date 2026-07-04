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
