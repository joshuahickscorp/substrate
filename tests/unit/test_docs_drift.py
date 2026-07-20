from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
check_docs = importlib.import_module("check_docs")


def test_real_repo_docs_clean():
    assert check_docs.check_docs() == []


def test_collected_count_matches_independent_collection():
    root = Path(check_docs.__file__).resolve().parents[1]
    p = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    independent = sum(int(m) for m in re.findall(r":\s*(\d+)\s*$", p.stdout, re.M))
    assert independent > 0, "sanity: collection should find tests"
    assert check_docs.collected_test_count() == independent


def test_registry_and_acceptance_ground_truth():
    from mop.experiments import REGISTRY

    assert check_docs.experiment_registry_size() == len(REGISTRY)
    root = Path(check_docs.__file__).resolve().parents[1]
    src = (root / "scripts" / "acceptance.py").read_text()
    independent = len(set(re.findall(r'step\(\s*"([^"]+)"', src)))
    assert independent >= 10  # the shipped suite has at least the 10 documented steps
    assert check_docs.acceptance_check_count() == independent


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
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, status="(9 tests)\n68 tests\n108 tests green")
    assert check_docs.check_docs() == []


def test_exact_match_is_clean(monkeypatch, tmp_path):
    _patch_truth(monkeypatch, tests=214)
    _docs(monkeypatch, tmp_path, readme="full suite, 214 tests, seconds")
    assert check_docs.check_docs() == []


def test_absent_number_is_clean(monkeypatch, tmp_path):
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
    _patch_truth(monkeypatch, accept=10)
    _docs(monkeypatch, tmp_path, status="acceptance 8/10 (two known-skips)")
    assert check_docs.check_docs() == []


def test_missing_script_ref_is_flagged(monkeypatch, tmp_path):
    _patch_truth(monkeypatch)
    _docs(monkeypatch, tmp_path, readme="run scripts/does_not_exist.py to start")
    probs = check_docs.check_docs()
    assert any("scripts/does_not_exist.py" in p for p in probs)


def test_present_script_ref_is_clean(monkeypatch, tmp_path):
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


def test_active_hardware_drift_flags_historical_scale_name(monkeypatch, tmp_path):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    (tmp_path / "GO.md").write_text("Make ViT-H the active Studio-gated path.")
    probs = check_docs._active_hardware_drift_problems()
    assert any("GO.md" in problem and "ViT-H" in problem for problem in probs)
    assert any("GO.md" in problem and "Studio-gated" in problem for problem in probs)


def test_active_hardware_drift_allows_measured_gate_language(monkeypatch, tmp_path):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    (tmp_path / "GO.md").write_text(
        "A Studio is considered only after a measured non-factorizable requirement."
    )
    assert check_docs._active_hardware_drift_problems() == []


@pytest.mark.parametrize(
    "stale",
    (
        "vjepa2_vith",
        "vjepa2_vitg",
        "vit_huge",
        "L/H/g",
        "H/g",
        "proof/ENCODER_SCALE_VITH_CPU_FORWARD.json",
        "proof/REAL_ENCODER_VITG_LOCAL8.json",
        "proof/VJEPA_SCALE_ATLAS_LOCAL.json",
        "proof/FACTORIZED_STIMULUS_IDENTITY.json",
    ),
)
def test_active_hardware_drift_flags_retired_variants(monkeypatch, tmp_path, stale):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    (tmp_path / "GO.md").write_text(f"Make {stale} the current execution path.")
    assert check_docs._active_hardware_drift_problems()


def test_active_hardware_drift_allows_exact_vitb_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(check_docs, "ROOT", tmp_path)
    (tmp_path / "GO.md").write_text(
        "Retain vjepa2_1_vitb_dist_vitG_384.pt as the exact official ViT-B object."
    )
    assert check_docs._active_hardware_drift_problems() == []


def test_main_returns_zero_on_clean_repo():
    assert check_docs.main() == 0
