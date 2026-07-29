"""Terminal publication gate repairs for clean-clone install receipts."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from substrate import final_revision_campaign as campaign
from substrate import final_revision_io as io

REAL_EVIDENCE = Path("/Users/scammermike/Downloads/substrate/evidence")


def _valid_clean_report() -> dict[str, Any]:
    return {
        "schema": "substrate-final-revision-clean-checkout-verification/v1",
        "all_pass": True,
        "commands": {
            "tests": {"command": ["pytest"], "returncode": 0, "passed": True},
            "lint": {"command": ["ruff"], "returncode": 0, "passed": True},
        },
        "checks": {
            "closure": True,
            "canaries": True,
            "pilot": True,
        },
        "recomputations": {
            "principal": {"exact_match": True},
            "replication": {"exact_match": True},
            "hidden_composition": {"exact_match": True},
        },
        "activation": False,
    }


def _valid_regeneration_report() -> dict[str, Any]:
    return {
        "schema": "substrate-final-revision-regeneration-check/v1",
        "exact_agreement": True,
        "activation": False,
    }


def _passing_install_report() -> dict[str, Any]:
    return {
        "command": ["/clone/.venv/bin/python", "-I", "-c", "import substrate"],
        "returncode": 0,
        "stdout_digest": "a" * 64,
        "stderr_digest": "b" * 64,
        "substrate_module_path": "/clone/src/substrate/__init__.py",
        "version": "0.0.0",
        "editable_install_inside_clone": True,
        "dependency_check_passed": True,
        "passed": True,
    }


@pytest.fixture
def isolated_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evidence = tmp_path / "evidence" / "substrate" / "final_revision"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(io, "EVIDENCE", evidence)
    assert evidence == io.EVIDENCE
    assert REAL_EVIDENCE not in io.EVIDENCE.parents and io.EVIDENCE != REAL_EVIDENCE
    return evidence


def test_record_clean_clone_accepts_operator_install_receipt(isolated_evidence: Path) -> None:
    """Pins the install-key mismatch: clean reports without install can still record."""
    clean_report = _valid_clean_report()
    assert "install" not in clean_report
    regeneration_report = _valid_regeneration_report()
    install_report = _passing_install_report()

    result = campaign.record_clean_clone_verification(clean_report, regeneration_report, install_report)

    assert result["all_pass"] is True
    clean_path = isolated_evidence / "SUBSTRATE_FINAL_REVISION_CLEAN_CLONE.json"
    document = io.load_json(clean_path)
    assert document["install_receipt_source"] == "operator_receipt"
    assert document["install"]["passed"] is True
    assert (isolated_evidence / "SUBSTRATE_FINAL_REVISION_REGENERATION.json").is_file()
    independent = io.load_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json")
    assert independent["separate_clean_process"] is True
    assert independent["complete"] is True
    assert not str(isolated_evidence).startswith(str(REAL_EVIDENCE))


def test_record_clean_clone_refuses_each_precondition(isolated_evidence: Path) -> None:
    """One assertion per refusal so a future edit cannot silently drop a gate."""
    clean_report = _valid_clean_report()
    regeneration_report = _valid_regeneration_report()
    install_report = _passing_install_report()

    with pytest.raises(io.Refused, match="installation receipt is absent or failed"):
        campaign.record_clean_clone_verification(clean_report, regeneration_report, None)

    failed_install = {**install_report, "passed": False}
    with pytest.raises(io.Refused, match="installation receipt is absent or failed"):
        campaign.record_clean_clone_verification(clean_report, regeneration_report, failed_install)

    bad_all_pass = {**clean_report, "all_pass": False}
    with pytest.raises(io.Refused, match="clean-clone verification did not pass"):
        campaign.record_clean_clone_verification(bad_all_pass, regeneration_report, install_report)

    bad_agreement = {**regeneration_report, "exact_agreement": False}
    with pytest.raises(io.Refused, match="two terminal regenerations did not agree exactly"):
        campaign.record_clean_clone_verification(clean_report, bad_agreement, install_report)

    bad_recomputations = copy.deepcopy(clean_report)
    bad_recomputations["recomputations"]["principal"] = {"exact_match": False}
    with pytest.raises(io.Refused, match="independent recomputation is incomplete"):
        campaign.record_clean_clone_verification(bad_recomputations, regeneration_report, install_report)

    assert list(isolated_evidence.iterdir()) == []


def test_record_clean_clone_prefers_clean_report_install(isolated_evidence: Path) -> None:
    clean_report = _valid_clean_report()
    clean_report["install"] = _passing_install_report()
    result = campaign.record_clean_clone_verification(
        clean_report,
        _valid_regeneration_report(),
        {**_passing_install_report(), "version": "operator-side"},
    )
    document = result["documents"]["SUBSTRATE_FINAL_REVISION_CLEAN_CLONE.json"]
    assert document["install_receipt_source"] == "clean_report"
    assert document["install"]["version"] == "0.0.0"


def test_verify_preserves_strong_independent_receipt(isolated_evidence: Path) -> None:
    strong = io.authority(
        "substrate-final-revision-independent-verification/v1",
        {
            "complete": True,
            "separate_clean_process": True,
            "marker": "must-not-be-clobbered",
        },
        status="complete",
    )
    path = isolated_evidence / "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json"
    io.write_json(path, strong)
    before = path.read_bytes()

    result = campaign.verify(publish=True)

    assert result["independent_receipt_preserved"] is True
    assert path.read_bytes() == before
    assert io.load_json(path)["marker"] == "must-not-be-clobbered"
    assert "report" in result
    assert result["report"]["schema"] == "substrate-final-revision-verification/v1"


def test_verify_writes_when_no_strong_receipt(isolated_evidence: Path) -> None:
    path = isolated_evidence / "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json"
    assert not path.is_file()

    result = campaign.verify(publish=True)

    assert result["independent_receipt_preserved"] is False
    assert path.is_file()
    document = io.load_json(path)
    assert document.get("separate_clean_process") is not True
    assert "complete" in document
    assert "missing" in document


def test_verify_overwrites_weak_inventory_receipt(isolated_evidence: Path) -> None:
    weak = io.authority(
        "substrate-final-revision-verification/v1",
        {
            "required": 1,
            "existing": 0,
            "missing": ["x"],
            "invalid": [],
            "complete": False,
            "partial_evidence_valid": True,
            "marker": "weak",
        },
        status="incomplete",
    )
    path = isolated_evidence / "SUBSTRATE_FINAL_REVISION_INDEPENDENT_VERIFICATION.json"
    io.write_json(path, weak)

    result = campaign.verify(publish=True)

    assert result["independent_receipt_preserved"] is False
    document = io.load_json(path)
    assert document.get("marker") != "weak"
    assert "missing" in document


def test_clean_clone_install_receipt_fails_when_venv_absent(tmp_path: Path) -> None:
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    receipt = campaign.clean_clone_install_receipt(clone_root)
    assert receipt["passed"] is False
    assert receipt["editable_install_inside_clone"] is False
    assert receipt["dependency_check_passed"] is False
    assert receipt["returncode"] is None
    assert receipt["substrate_module_path"] is None
    assert (clone_root / ".venv" / "bin" / "python").as_posix() in receipt["command"][0]
