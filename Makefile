VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install install-studio verify-install test lint types fmt diag accept clean doctor cache-list storage docs devel ladder curriculum

install:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev,ann]"
	$(MAKE) verify-install

install-studio:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev,ann,encoder,video,apple]"
	$(MAKE) verify-install

verify-install:
	cd /tmp && $(abspath $(PY)) -I -c "import importlib.metadata, mop; print('mop', importlib.metadata.version('mop'), mop.__file__)"

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

diag:
	$(PY) scripts/run_diagnostics.py device=cpu

accept:
	$(PY) scripts/acceptance.py

devel:
	$(PY) scripts/devel.py validate       # validate paradigm/capacity/paperwatch registries

ladder:
	$(PY) scripts/devel.py capacities     # the developmental capacity ladder (Frontier 32)

curriculum:
	$(PY) scripts/devel.py curriculum     # next-lesson manifest: REAL probes over controls (Frontier 26/33)

doctor:
	$(PY) -m scripts.studio doctor        # Studio readiness report (JSON + runs/studio_doctor.md)

cache-list:
	$(PY) scripts/cache_tool.py list      # list + integrity of cached latent stores

storage:
	$(PY) scripts/storage_tool.py list    # cache sizes; `prune` (dry-run) / `estimate`

docs:
	$(PY) scripts/check_docs.py           # docs-drift gate (stale counts / dead refs)

clean:
	rm -rf runs/* data/cache/* .pytest_cache .mypy_cache .ruff_cache
