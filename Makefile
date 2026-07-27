VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install verify-install test lint types fmt accept audit clean

install:
	uv venv --python 3.12 .venv
	uv pip install -e ".[dev]"
	$(MAKE) verify-install

verify-install:
	cd /tmp && $(abspath $(PY)) -I -c "import importlib.metadata, substrate; print('substrate', importlib.metadata.version('substrate'), substrate.__file__)"

test:
	$(VENV)/substrate test

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
