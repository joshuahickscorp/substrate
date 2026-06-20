VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install test lint types fmt e1 diag i4 queue-dry accept clean

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
	$(PY) scripts/run_experiment.py +run=diagnostics

i4:
	$(PY) scripts/run_experiment.py experiment=i4_backprop_alts

queue-dry:
	$(PY) scripts/run_queue.py --dry-run

accept:
	$(PY) scripts/acceptance.py

clean:
	rm -rf runs/* data/cache/* .pytest_cache .mypy_cache .ruff_cache
