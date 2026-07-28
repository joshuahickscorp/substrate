"""Terminal-gate hardening: seal re-execution, content checks, activation keys."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from substrate import final_revision_campaign as campaign
from substrate import final_revision_config as C
from substrate import final_revision_io as io
from substrate import final_revision_verification as verification

REAL_EVIDENCE = Path("/Users/scammermike/Downloads/substrate/evidence")


@pytest.fixture
def isolated_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    evidence = tmp_path / "evidence" / "substrate" / "final_revision"
    evidence.mkdir(parents=True)
    monkeypatch.setattr(io, "EVIDENCE", evidence)
    assert evidence == io.EVIDENCE
    assert REAL_EVIDENCE not in io.EVIDENCE.parents and io.EVIDENCE != REAL_EVIDENCE
    return evidence


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    io.write_json(path, document)


def _minimal_bed_result() -> dict[str, Any]:
    return {
        "schema": "substrate-final-revision-discrimination-bed/v1",
        "effects": {
            "P1_selected_minus_full_transcript_replay": {
                "mean_paired_effect": 0.0,
                "confidence_interval_95": [0.0, 0.0],
                "passes_after_holm": False,
            },
            "P3_selected_minus_strongest_persistent_alternative": {
                "mean_paired_effect": 0.0,
                "confidence_interval_95": [0.0, 0.0],
                "passes_after_holm": False,
            },
        },
        "activation": False,
    }


def test_forged_mutation_report_fails_when_live_disagrees(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Forging zero_survivors:true while live mutation_report disagrees must fail the gate."""
    forged = io.authority(
        "substrate-final-revision-mutation-report/v1",
        {"zero_survivors": True, "survivors": [], "total": 0, "rejected": 0},
        status="complete",
    )
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_MUTATION_REPORT.json", forged)
    monkeypatch.setattr(
        verification,
        "mutation_report",
        lambda: {"zero_survivors": False, "survivors": ["forged_survives"], "total": 1, "rejected": 0},
    )
    monkeypatch.setattr(
        verification,
        "counterfeit_report",
        lambda: {"all_rejected": True, "counterfeits": []},
    )
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    classification = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]
    checks = classification["outcome_b_checks"]

    assert checks["mutation_zero_survivors"] is False
    assert "mutation_zero_survivors" in classification["seal_check_disagreements"]
    disagreement = classification["seal_check_disagreements"]["mutation_zero_survivors"]
    assert disagreement["stored"] is True
    assert disagreement["live"] is False
    assert classification["gate_verification_method"]["mutation_zero_survivors"] == "re_executed_at_seal"
    assert not str(isolated_evidence).startswith(str(REAL_EVIDENCE))


def test_junk_freeze_no_longer_satisfies_candidate_frozen(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    junk = {"note": "not a freeze", "activation": False}
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json", junk)
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    checks = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["outcome_b_checks"]
    assert checks["candidate_frozen"] is False
    assert documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["gate_verification_method"]["candidate_frozen"] == "content_checked"


def test_freeze_with_wrong_sesoi_fails_candidate_frozen(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = io.authority(
        "substrate-final-revision-candidate-freeze/v1",
        {
            "architecture": "I_simplest_sufficient",
            "source_digest": io.source_digest(),
            "sesoi": C.SESOI + 0.01,
            "challenges": list(C.CHALLENGE_FAMILIES),
            "baselines": list(C.BASELINES),
        },
        status="ready_to_tag",
    )
    assert freeze["sesoi"] != C.SESOI
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json", freeze)
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    checks = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["outcome_b_checks"]
    assert checks["candidate_frozen"] is False


def test_freeze_accepts_authorised_predecessor_source_digest(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    predecessor = "83fd33d5f7ec485af941ec494f7361265838752b468138695ce6083fdc21e90b"
    assert predecessor != io.source_digest()
    transition = io.authority(
        "substrate-final-revision-sealed-transition/v1",
        {
            "sequence": 1,
            "ready": {"frozen_source_digest": predecessor, "tag": C.READY_TAG},
            "sesoi": C.SESOI,
        },
        status="complete",
    )
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_TRANSITION_001.json", transition)
    freeze = io.authority(
        "substrate-final-revision-candidate-freeze/v1",
        {
            "architecture": "I_simplest_sufficient",
            "source_digest": predecessor,
            "sesoi": C.SESOI,
            "challenges": list(C.CHALLENGE_FAMILIES),
            "baselines": list(C.BASELINES),
        },
        status="ready_to_tag",
    )
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_CANDIDATE_FREEZE.json", freeze)
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    checks = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["outcome_b_checks"]
    assert checks["candidate_frozen"] is True


def test_junk_principal_no_longer_satisfies_principal_complete(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_json(
        isolated_evidence / "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json",
        {"junk": True, "activation": False},
    )
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    checks = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["outcome_b_checks"]
    assert checks["principal_complete"] is False

    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_PRINCIPAL_RESULT.json", _minimal_bed_result())
    documents = campaign._terminal_documents()
    checks = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]["outcome_b_checks"]
    assert checks["principal_complete"] is True


def test_activation_key_variants_refused_and_false_still_loads(tmp_path: Path) -> None:
    path = tmp_path / "doc.json"
    # Exact False values must still load.
    for payload in (
        {"schema": "test/v1", "activation": False},
        {"schema": "test/v1", "external_activation": False, "activation": False},
        {"schema": "test/v1", "nested": {"activation": False, "external_activation": False}},
    ):
        body = dict(payload)
        body["sha256"] = io.digest({k: v for k, v in body.items() if k != "sha256"})
        path.write_text(io.canonical_bytes(body).decode() + "\n")
        loaded = io.load_json(path)
        assert not io.contains_true_activation(loaded)
        if "activation" in loaded:
            assert loaded["activation"] is False
        if "external_activation" in loaded:
            assert loaded["external_activation"] is False

    # Truthy variant keys must be refused.
    for key in ("Activation", "ACTIVATION", "external_activation", "EXTERNAL_ACTIVATION", "activation"):
        body = {"schema": "test/v1", key: True}
        path.write_text(io.canonical_bytes(body).decode() + "\n")
        with pytest.raises(io.Refused, match="enables activation"):
            io.load_json(path)

    assert io.contains_true_activation({"Activation": True}) is True
    assert io.contains_true_activation({"ACTIVATION": True}) is True
    assert io.contains_true_activation({"external_activation": True}) is True
    assert io.contains_true_activation({"EXTERNAL_ACTIVATION": True}) is True
    assert io.contains_true_activation({"activation": False}) is False
    assert io.contains_true_activation({"external_activation": False}) is False
    assert io.contains_true_activation({"nested": [{"Activation": 1}]}) is True


def test_thresholds_preserved_false_when_live_sesoi_diverges(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    freeze = {
        "schema": "substrate-final-revision-candidate-freeze/v1",
        "sesoi": 0.05,
        "power_target": C.POWER_TARGET,
        "configuration_digest": C.configuration_digest(),
        "challenges": list(C.CHALLENGE_FAMILIES),
        "source_digest": "deadbeef",
    }
    preserved, reason = campaign._thresholds_preserved(freeze)
    assert preserved is True
    assert reason is None

    monkeypatch.setattr(C, "SESOI", 0.99)
    preserved, reason = campaign._thresholds_preserved(freeze)
    assert preserved is False
    assert reason is None


def test_thresholds_preserved_null_when_power_target_unrecorded() -> None:
    freeze = {
        "schema": "substrate-final-revision-candidate-freeze/v1",
        "sesoi": C.SESOI,
        "configuration_digest": C.configuration_digest(),
        "challenges": list(C.CHALLENGE_FAMILIES),
    }
    preserved, reason = campaign._thresholds_preserved(freeze)
    assert preserved is None
    assert reason is not None
    assert "power_target" in reason


def test_history_intact_recomputes_git_diff(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Stored document claims intact, live recomputation reports drift → fail and record disagreement.
    immutability = io.authority(
        "substrate-final-revision-immutability/v1",
        {"historical_evidence_untouched": True, "historical_diff_from_preflight": []},
        status="complete",
    )
    _write_json(isolated_evidence / "SUBSTRATE_FINAL_REVISION_IMMUTABILITY.json", immutability)
    monkeypatch.setattr(
        campaign,
        "_git_diff_names",
        lambda *args, **kwargs: ["evidence/substrate/nous_closure/forged.json"],
    )
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})

    documents = campaign._terminal_documents()
    classification = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]
    assert classification["outcome_b_checks"]["history_intact"] is False
    assert "history_intact" in classification["seal_check_disagreements"]
    assert classification["gate_verification_method"]["history_intact"] == "re_executed_at_seal"


def test_gate_verification_method_covers_all_outcome_b_checks(isolated_evidence: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(verification, "mutation_report", lambda: {"zero_survivors": True})
    monkeypatch.setattr(verification, "counterfeit_report", lambda: {"all_rejected": True})
    monkeypatch.setattr(campaign, "_git_diff_names", lambda *args, **kwargs: [])

    documents = campaign._terminal_documents()
    classification = documents["SUBSTRATE_FINAL_REVISION_FINAL_CLASSIFICATION.json"]
    checks = classification["outcome_b_checks"]
    methods = classification["gate_verification_method"]
    assert set(methods) == set(checks)
    allowed = {"re_executed_at_seal", "content_checked", "field_asserted"}
    assert set(methods.values()) <= allowed
    # Hardened gates must not remain field_asserted.
    assert methods["mutation_zero_survivors"] == "re_executed_at_seal"
    assert methods["counterfeits_rejected"] == "re_executed_at_seal"
    assert methods["history_intact"] == "re_executed_at_seal"
    assert methods["candidate_frozen"] == "content_checked"
    assert methods["principal_complete"] == "content_checked"
    assert methods["long_continuity_complete"] == "field_asserted"
