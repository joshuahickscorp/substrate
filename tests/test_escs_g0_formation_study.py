from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studies.escs_g0_formation import (
    G0FormationLedger,
    G0ParetoArchive,
    G0ParetoDecisionStatus,
    verify_g0_pareto_archive,
)
from mop.studies.escs_g0_formation_study import (
    DEFAULT_CONFIG_PATH,
    build_receipt,
    verify_receipt,
)
from mop.substrate.events import canonical_sha256

SCRIPT = REPO_ROOT / "scripts/run_escs_g0_formation_study.py"


def _write(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_persisted_formation_study_is_inert_replayable_and_low_admission_gate() -> None:
    receipt = build_receipt(DEFAULT_CONFIG_PATH)

    assert receipt["all_ok"] is True
    assert receipt["problems"] == []
    assert receipt["counterfactual_only"] is True
    assert receipt["activation_enabled"] is False
    assert receipt["shadow_execution_authorized"] is False
    assert receipt["factual_effects"] is False
    assert receipt["factual_mutation_authorized"] is False
    assert receipt["scientific_promotion_allowed"] is False
    assert receipt["summary"]["attempt_count"] == 4
    assert receipt["summary"]["structurally_valid_candidate_count"] == 3
    assert receipt["summary"]["evaluated_candidate_count"] == 3
    assert receipt["summary"]["refused_control_count"] == 1
    assert receipt["summary"]["proposal_admission_fraction_microunits"] == 750_000
    assert receipt["summary"]["valid_candidate_admission_fraction_microunits"] == 1_000_000
    assert receipt["summary"]["mutation_admission_fraction_microunits"] == 666_666
    assert receipt["summary"]["trace_count"] == 6

    ledger = G0FormationLedger.from_payload(receipt["ledger"])
    archive = G0ParetoArchive.from_payload(receipt["pareto_archive"])
    assert ledger.verify() == ()
    assert verify_g0_pareto_archive(archive, ledger) is True
    statuses = {row.status for row in archive.decisions}
    assert G0ParetoDecisionStatus.RETAINED in statuses
    assert G0ParetoDecisionStatus.DOMINATED in statuses
    assert G0ParetoDecisionStatus.INELIGIBLE in statuses
    assert receipt["summary"]["pareto_retained_count"] == 2


def test_candidate_bound_episodes_share_candidate_independent_task_cohort() -> None:
    receipt = build_receipt(DEFAULT_CONFIG_PATH)
    ledger = G0FormationLedger.from_payload(receipt["ledger"])
    cohorts = {
        row.attempt.objective.evaluation_cohort_sha256
        for row in ledger.entries
        if row.attempt.objective is not None
    }
    episode_sets = {
        tuple(assessment.episode_sha256 for assessment in row.attempt.assessments)
        for row in ledger.entries
        if row.attempt.objective is not None
    }

    assert len(cohorts) == 1
    assert len(episode_sets) == 3


def test_receipt_is_deterministic_and_resealed_tampering_fails_regeneration(
    tmp_path: Path,
) -> None:
    first = build_receipt(DEFAULT_CONFIG_PATH)
    second = build_receipt(DEFAULT_CONFIG_PATH)
    assert first == second

    receipt_path = tmp_path / "receipt.json"
    _write(receipt_path, first)
    assert verify_receipt(DEFAULT_CONFIG_PATH, receipt_path)["all_ok"] is True

    forged = copy.deepcopy(first)
    forged["summary"]["trace_count"] = 999
    forged_core = dict(forged)
    forged_core.pop("receipt_sha256")
    forged["receipt_sha256"] = canonical_sha256(forged_core)
    _write(receipt_path, forged)
    verification = verify_receipt(DEFAULT_CONFIG_PATH, receipt_path)
    assert verification["all_ok"] is False
    assert "deterministic regeneration" in verification["problems"][0]
    forged_raw = receipt_path.read_bytes()
    assert verification["receipt_authority"]["bytes"] == len(forged_raw)
    assert verification["receipt_authority"]["sha256"] == hashlib.sha256(forged_raw).hexdigest()
    assert verification["declared_receipt_sha256"] == forged["receipt_sha256"]


def test_malformed_receipt_failure_preserves_the_rejected_artifact_authority(
    tmp_path: Path,
) -> None:
    receipt_path = tmp_path / "malformed.json"
    malformed = b'{"not": "finished"'
    receipt_path.write_bytes(malformed)

    verification = verify_receipt(DEFAULT_CONFIG_PATH, receipt_path)

    assert verification["all_ok"] is False
    assert verification["receipt_authority"]["bytes"] == len(malformed)
    assert verification["receipt_authority"]["sha256"] == hashlib.sha256(malformed).hexdigest()


def test_receipt_explicitly_disclaims_proxy_metrics_and_independent_replication() -> None:
    receipt = build_receipt(DEFAULT_CONFIG_PATH)
    nonclaims = " ".join(receipt["nonclaims"])

    assert "synthetic mechanics proxies" in nonclaims
    assert "not an independent scientific replication" in nonclaims


def test_cli_runs_and_verifies_without_activation(tmp_path: Path) -> None:
    receipt_path = tmp_path / "receipt.json"
    verification_path = tmp_path / "verification.json"
    run = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "run",
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--out",
            str(receipt_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    verify = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
            "--config",
            str(DEFAULT_CONFIG_PATH),
            "--receipt",
            str(receipt_path),
            "--out",
            str(verification_path),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify.returncode == 0, verify.stderr
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    assert verification["all_ok"] is True
    assert verification["activation_enabled"] is False
    assert verification["scientific_promotion_allowed"] is False
