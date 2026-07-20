VENV=.venv/bin
PY=$(VENV)/python

.PHONY: install install-studio verify-install test lint types fmt accept clean doctor

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

accept:
	$(PY) scripts/acceptance.py

doctor:
	$(VENV)/mop doctor        # Studio readiness report (JSON + runs/studio_doctor.md)

clean:
	rm -rf runs/* data/cache/* .pytest_cache .mypy_cache .ruff_cache
