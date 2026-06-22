"""Frontier 6: docs drift gate. The shipped README/STATUS must not contradict the live ground
truth (collected test count, experiment-registry size, acceptance-check count), and every script
path / make target they name must resolve. Plus fixture tests that the detector actually fires on
stale numbers and dangling references (and stays quiet when a doc omits a number)."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

# scripts/ is not an installed package and there is no conftest path hook, so put it on sys.path,
# then load the gate module by name (import_module keeps isort out of a runtime-path import).
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
check_docs = importlib.import_module("check_docs")


# ---- ground truth is real -------------------------------------------------------------------


def test_real_repo_docs_clean():
    # the shipped docs must be self-consistent with the code: no over-claimed counts, all refs live
    assert check_docs.check_docs() == []


def test_collected_count_matches_independent_collection():
    # cross-check the parser against an independent pytest collection (sum of per-file counts)
    root = Path(check_docs.__file__).resolve().parents[1]
    p = subprocess.run(
        [str(root / ".venv/bin/python"), "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    independent = sum(int(m) for m in re.findall(r":\s*(\d+)\s*$", p.stdout, re.M))
    assert independent > 0, "sanity: collection should find tests"
    assert check_docs.collected_test_count() == independent


def test_registry_and_acceptance_ground_truth():
    from devsys.experiments import REGISTRY

    assert check_docs.experiment_registry_size() == len(REGISTRY)
    # acceptance.py prints N/N where N is the distinct step() name count; re-derive it independently
    # from the source so the parser is checked against truth, not a magic constant.
    root = Path(check_docs.__file__).resolve().parents[1]
    src = (root / "scripts" / "acceptance.py").read_text()
    independent = len(set(re.findall(r'step\(\s*"([^"]+)"', src)))
    assert independent >= 10  # the shipped suite has at least the 10 documented steps
    assert check_docs.acceptance_check_count() == independent


# ---- detector fires on drift (fixtures) -----------------------------------------------------


def _patch_truth(monkeypatch, *, tests=214, experiments=11, accept=10, targets=("test", "accept")):
    monkeypatch.setattr(check_docs, "collected_test_count", lambda: tests)
    monkeypatch.setattr(check_docs, "experiment_registry_size", lambda: experiments)
    monkeypatch.setattr(check_docs, "acceptance_check_count", lambda: accept)
    monkeypatch.setattr(check_docs, "_make_targets", lambda: set(targets))


def _docs(monkeypatch, tmp_path, readme="", status=""):
    (tmp_path / "README.md").write_text(readme)
    (tmp_path / "STATUS.md").write_text(status)
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)


def test_overclaimed_test_count_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, readme="full suite, 999 tests, seconds")
    probs = check_docs.check_docs()
    assert any("999 tests" in p and "214" in p for p in probs)


def test_undercount_is_tolerated(monkeypatch, tmp_path):
    # a growing suite: an old build-log line stating a SMALLER count is behind, not a contradiction
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, status="(9 tests)\n68 tests\n108 tests green")
    assert check_docs.check_docs() == []


def test_exact_match_is_clean(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, readme="full suite, 214 tests, seconds")
    assert check_docs.check_docs() == []


def test_absent_number_is_clean(monkeypatch, tmp_path):
    # tolerance: a doc that states no number at all is fine
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, readme="run the full suite in seconds")
    assert check_docs.check_docs() == []


def test_wrong_experiment_count_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, experiments=11)
    _docs(monkeypatch, tmp_path, status="all 7 experiments registered")
    probs = check_docs.check_docs()
    assert any("7 experiments" in p and "11" in p for p in probs)


def test_correct_experiment_count_is_clean(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, experiments=11)
    _docs(monkeypatch, tmp_path, status="all 11 experiments registered")
    assert check_docs.check_docs() == []


def test_wrong_acceptance_ratio_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, accept=10)
    _docs(monkeypatch, tmp_path, status="acceptance GREEN 12/12")
    probs = check_docs.check_docs()
    assert any("12/12" in p and "10" in p for p in probs)


def test_correct_acceptance_ratio_is_clean(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, accept=10)
    _docs(monkeypatch, tmp_path, status="acceptance GREEN 10/10")
    assert check_docs.check_docs() == []


def test_partial_ratio_is_not_a_count_claim(monkeypatch, tmp_path):
    # "8/10" is a partial-pass report, not a denominator claim about the suite size: don't flag it
    _patch_truth(monkeypatch, accept=10)
    _docs(monkeypatch, tmp_path, status="acceptance 8/10 (two known-skips)")
    assert check_docs.check_docs() == []


def test_missing_script_ref_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch)
    _docs(monkeypatch, tmp_path, readme="run scripts/does_not_exist.py to start")
    probs = check_docs.check_docs()
    assert any("scripts/does_not_exist.py" in p for p in probs)


def test_present_script_ref_is_clean(monkeypatch, tmp_path):
    # ROOT points at tmp_path, so create the referenced script there
    _patch_truth(monkeypatch)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "real.py").write_text("")
    _docs(monkeypatch, tmp_path, readme="run scripts/real.py to start")
    assert check_docs.check_docs() == []


def test_missing_make_target_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, targets=("test", "accept"))
    _docs(monkeypatch, tmp_path, readme="run `make ghost` then `make test`")
    probs = check_docs.check_docs()
    assert any("make ghost" in p for p in probs)
    assert not any("make test" in p for p in probs)  # make test exists -> not flagged


def test_present_make_target_is_clean(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, targets=("test", "accept", "e1"))
    _docs(monkeypatch, tmp_path, readme="run make e1 then make accept")
    assert check_docs.check_docs() == []


def test_main_returns_zero_on_clean_repo():
    # the script entry point exits 0 against the shipped repo
    assert check_docs.main() == 0
