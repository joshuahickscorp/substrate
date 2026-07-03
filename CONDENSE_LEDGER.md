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
  = sha256( sorted registry ids · [project.scripts] · sorted doc-referenced scripts/*.py paths )[:16].
  Internal src/mop reorganization that preserves the experiment catalog, the console entrypoint, and the
  doc-referenced script paths keeps this stable; behavior itself is guarded by the frozen tests + check_docs
  + acceptance.
- Pre-baseline note · one committed formatting drift (scripts/studio/dr1_smoke.py) was ruff-formatted in
  c084370 so the baseline is honestly green; the `mop` console script has a stale shebang from the
  brain->mop rename (a pre-existing horizon defect in the installed script, not a tracked source file, left
  as-is).

## Iterations

(none yet)
