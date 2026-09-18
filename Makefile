.DEFAULT_GOAL := static
.PHONY: static test

static:
	python3 tools/static_check.py --output refactor/STATIC_INSPECTION.json

# Deliberate operator selection only. This target was NOT run for the refactor.
test:
	python3 -m pytest
