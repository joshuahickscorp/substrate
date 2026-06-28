VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install test lint types fmt e1 diag i4 queue-dry accept clean doctor bench cache-list storage docs report rehearse local-max studio-plan devel ladder curriculum

install:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev,ann]"

test:
	$(VENV)/pytest

lint:
	$(VENV)/ruff check src tests
	$(VENV)/ruff format --check src tests

fmt:
	$(VENV)/ruff format src tests
	$(VENV)/ruff check --fix src tests

types:
	$(VENV)/mypy

e1:
	$(PY) scripts/run_experiment.py experiment=e1_baseline

diag:
	$(PY) scripts/run_diagnostics.py device=cpu

i4:
	$(PY) scripts/run_experiment.py experiment=i4_backprop_alts

queue-dry:
	$(PY) scripts/run_queue.py --dry-run

accept:
	$(PY) scripts/acceptance.py

rehearse:
	$(PY) scripts/studio_rehearsal.py     # WHOLE Studio workflow on tiny fixtures -> runs/studio_rehearsal/

local-max:
	$(PY) scripts/studio_pipeline.py local-max --download-gb 10 --time-min 90 --cache-clips 64  # current-device max

studio-plan:
	$(PY) scripts/studio_pipeline.py plan --profile studio-1tb --budget-gb 900  # DRY-RUN plan under the 900 GB studio budget

devel:
	$(PY) scripts/devel.py validate       # validate paradigm/capacity/paperwatch registries

ladder:
	$(PY) scripts/devel.py capacities     # the developmental capacity ladder (Frontier 32)

curriculum:
	$(PY) scripts/devel.py curriculum     # next-lesson manifest: REAL probes over controls (Frontier 26/33)

doctor:
	$(PY) scripts/studio_doctor.py        # Studio readiness report (JSON + runs/studio_doctor.md)

bench:
	$(PY) scripts/bench.py                # microbenchmarks (not science; runs/microbench.md)

cache-list:
	$(PY) scripts/cache_tool.py list      # list + integrity of cached latent stores

storage:
	$(PY) scripts/storage_tool.py list    # cache sizes; `prune` (dry-run) / `estimate`

docs:
	$(PY) scripts/check_docs.py           # docs-drift gate (stale counts / dead refs)

report:
	$(PY) scripts/build_report.py         # analysis report scaffold (runs/analysis_report.md)

clean:
	rm -rf runs/* data/cache/* .pytest_cache .mypy_cache .ruff_cache
