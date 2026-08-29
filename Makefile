VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install verify-install test test-qualification test-normal test-integration test-expensive test-full test-certification certify lint types fmt accept audit clean

install:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	$(MAKE) verify-install

verify-install:
	cd /tmp && $(abspath $(PY)) -I -c "import importlib.metadata, substrate; print('substrate', importlib.metadata.version('substrate'), substrate.__file__)"

test:
	$(VENV)/substrate test

test-qualification:
	$(PY) -m pytest -m qualification

test-normal:
	$(PY) -m pytest -m "normal and not integration and not expensive and not certification"

test-integration:
	$(PY) -m pytest -m integration

test-expensive:
	$(PY) -m pytest -m expensive

test-full:
	$(PY) -m pytest tests/substrate --durations=20

test-certification: test-full

certify: test-certification

lint:
	$(VENV)/ruff check src tests
	$(VENV)/ruff format --check src tests

fmt:
	$(VENV)/ruff format src tests
	$(VENV)/ruff check --fix src tests

types:
	$(VENV)/mypy

accept:
	$(VENV)/substrate verify

audit:
	$(VENV)/substrate audit

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
