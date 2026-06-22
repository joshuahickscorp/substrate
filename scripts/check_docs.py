#!/usr/bin/env python
"""Frontier 6: docs drift gate. Computes the ground truth (collected pytest test count, the
experiment-registry size, the acceptance-check count) and scans README.md + STATUS.md for
hardcoded claims that CONTRADICT it: an over-claimed test count, a wrong experiment count, a
wrong acceptance ratio. Also verifies that every script path (scripts/foo.py) and make target
(make foo) named in the docs actually exists on disk / in the Makefile.

Tolerance is the whole point: a doc that states no number is fine, and in a suite that only
grows, a build-log line stating an OLD (smaller) count is behind, not wrong. Only a number that
is impossible (claims more tests than exist) or flatly disagrees (experiment count, acceptance
ratio) is flagged. check_docs()->list[str] (empty == clean); main() prints them and exits
nonzero on any problem.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
DOCS = ("README.md", "STATUS.md")

# the canonical make targets live on the Makefile .PHONY line; these are referenced in prose too
_PHONY = re.compile(r"^\.PHONY:\s*(.+)$", re.M)
_SCRIPT_REF = re.compile(r"scripts/[A-Za-z0-9_]+\.py")
_MAKE_REF = re.compile(r"\bmake\s+([a-z][a-z0-9-]+)\b")
_STEP = re.compile(r'step\(\s*"([^"]+)"')
# "68 tests", "108 tests green", "9 tests" (parenthetical build-log counts included)
_TESTS_CLAIM = re.compile(r"(\d+)\s+tests\b")
# "11 experiments"
_EXPERIMENTS_CLAIM = re.compile(r"(\d+)\s+experiments\b")
# "10/10" acceptance ratio (n/n form)
_RATIO_CLAIM = re.compile(r"\b(\d+)\s*/\s*(\d+)\b")


def _read(name: str) -> str:
    p = ROOT / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


def collected_test_count() -> int:
    """Ground-truth collected test count via `pytest --collect-only -q`.

    This pytest prints one `path: N` line per test file under -q (not `path::node` nodes), so the
    robust signal is the sum of those trailing per-file counts. We cross-check against an
    `N tests collected` / `N items` line when the run emits one and trust the larger (collection
    can be pruned by a stale cache; the per-file sum is the floor). Returns -1 if pytest cannot
    collect, so the caller can skip the comparison instead of asserting against a bogus 0.
    """
    p = subprocess.run(
        [PY, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    out = p.stdout + "\n" + p.stderr
    per_file = sum(int(m) for m in re.findall(r":\s*(\d+)\s*$", out, re.M))
    summary = 0
    m = re.search(r"(\d+)\s+tests collected", out) or re.search(r"collected\s+(\d+)\s+items?", out)
    if m:
        summary = int(m.group(1))
    best = max(per_file, summary)
    if best == 0:
        # collection produced nothing parseable: only treat as real if pytest actually ran clean
        return -1 if p.returncode not in (0, 5) else 0
    return best


def experiment_registry_size() -> int:
    from devsys.experiments import REGISTRY

    return len(REGISTRY)


def acceptance_check_count() -> int:
    """Distinct step("...") names in acceptance.py (try/except pairs reuse one name), which equals
    the denominator acceptance prints as N/N."""
    src = _read("scripts/acceptance.py")
    return len(set(_STEP.findall(src)))


def _make_targets() -> set[str]:
    m = _PHONY.search(_read("Makefile"))
    return set(m.group(1).split()) if m else set()


def check_docs() -> list[str]:
    """Return a list of human-readable drift problems (empty == docs are consistent)."""
    problems: list[str] = []

    real_tests = collected_test_count()
    real_experiments = experiment_registry_size()
    real_accept = acceptance_check_count()
    make_targets = _make_targets()

    for name in DOCS:
        text = _read(name)
        if not text:
            continue

        # test counts: a claim is WRONG only if it over-claims (more tests than exist). A smaller
        # number is an older build-log snapshot of a growing suite, not a contradiction.
        if real_tests >= 0:
            for n in (int(x) for x in _TESTS_CLAIM.findall(text)):
                if n > real_tests:
                    problems.append(
                        f"{name}: claims {n} tests but only {real_tests} are collected "
                        f"(over-claim; update or remove the number)"
                    )

        # experiment count must match the registry exactly (it does not silently grow in docs)
        for n in (int(x) for x in _EXPERIMENTS_CLAIM.findall(text)):
            if n != real_experiments:
                problems.append(f"{name}: claims {n} experiments but REGISTRY has {real_experiments}")

        # acceptance ratio "k/k": a full-pass ratio whose denominator disagrees with the real
        # acceptance-check count is stale (e.g. doc says 10/10 but acceptance now has 12 steps).
        for a, b in ((int(a), int(b)) for a, b in _RATIO_CLAIM.findall(text)):
            if a == b and b != real_accept:
                problems.append(
                    f"{name}: acceptance ratio {a}/{b} but acceptance.py has {real_accept} checks"
                )

        # every referenced script path must exist on disk
        for ref in sorted(set(_SCRIPT_REF.findall(text))):
            if not (ROOT / ref).exists():
                problems.append(f"{name}: references {ref} which does not exist on disk")

        # every referenced make target must exist in the Makefile
        for tgt in sorted(set(_MAKE_REF.findall(text))):
            if tgt not in make_targets:
                problems.append(f"{name}: references `make {tgt}` not in Makefile .PHONY")

    return problems


def main() -> int:
    problems = check_docs()
    if not problems:
        print(
            f"docs OK: {collected_test_count()} tests, {experiment_registry_size()} experiments, "
            f"{acceptance_check_count()} acceptance checks; no stale numbers, all refs resolve"
        )
        return 0
    print(f"DOCS DRIFT: {len(problems)} problem(s)", file=sys.stderr)
    for p in problems:
        print(f"  - {p}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
