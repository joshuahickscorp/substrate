from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from mop.config import REPO_ROOT
from mop.studio import coexistence_cutover as cutover

_REQUIRED_LOCAL_EVIDENCE = (
    cutover.V5_RECEIPT_PATH,
    cutover.V6_ACTIVE_OBSERVER_PATH,
    Path(
        "runs/mac_studio_campaign/"
        "mac-studio-substrate-phase1-coexistence-10k-v2/current_status.json"
    ),
)
pytestmark = pytest.mark.skipif(
    not all((REPO_ROOT / path).is_file() for path in _REQUIRED_LOCAL_EVIDENCE),
    reason="requires curated local coexistence run evidence",
)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _reseal(value: dict[str, object], field: str) -> None:
    core = dict(value)
    core.pop(field, None)
    value[field] = cutover.canonical_sha256(core)


def test_preserved_v1_and_v5_seed_are_exactly_authenticated() -> None:
    supersedes = cutover._supersedes(REPO_ROOT)
    seed = cutover._v5_first_seed_evidence(REPO_ROOT)

    assert supersedes["sha256"] == cutover.V1_FILE_SHA256
    assert supersedes["cutover_sha256"] == cutover.V1_CUTOVER_SHA256
    assert seed["classification"] == "resumable-first-seed-success"
    assert seed["completed_gate_seeds"] == [cutover.V5_GATE_SEED]
    assert seed["gate_row_sha256"] == [cutover.V5_GATE_ROW_SHA256]
    assert seed["governor_receipt"]["status"] == "resumable-invocation-boundary"
    assert seed["governor_receipt"]["final_returncode"] == 2
    assert seed["scientific_promotion"] is False


def test_v5_hold_is_calibration_and_finished_before_governed_child() -> None:
    evidence = cutover._v5_observer_hold_evidence(REPO_ROOT)

    assert evidence["classification"] == "observer-false-positive-calibration"
    assert evidence["campaign_status"]["problem"] == cutover.V5_OBSERVER_PROBLEM
    assert evidence["event_log"]["supervisor_stop_at"] < evidence["governed_child_finished_at"]
    assert evidence["scientific_promotion"] is False


def test_active_observer_snapshot_binds_exact_post_exec_child() -> None:
    evidence = cutover.validate_active_observer_snapshot(REPO_ROOT)
    snapshot = _json(REPO_ROOT / cutover.V6_ACTIVE_OBSERVER_PATH)
    lane = snapshot["campaign_status"]["document"]["active_lanes"][0]
    expected = lane["command"][len(cutover.throttle.TASKPOLICY_COEXISTENCE_PREFIX) :]

    assert evidence["process_observation"]["cmdline"] == expected
    assert evidence["process_observation"]["match_mode"] == "exact-pinned-post-taskpolicy-exec"
    assert evidence["observer_result"]["registered_child_accepted"] is True
    assert evidence["observer_result"]["integrity_hold_before_observation"] is False


def test_seed2_terminal_receipt_retains_v5_prefix() -> None:
    active = cutover.validate_active_observer_snapshot(REPO_ROOT)
    evidence = cutover._v6_terminal_evidence(REPO_ROOT, active, archive=False)

    assert evidence["run_id"] == cutover.V6_SEED2_RUN_ID
    assert evidence["governor_receipt"]["status"] == "resumable-invocation-boundary"
    assert evidence["governor_receipt"]["final_returncode"] == 2
    assert evidence["event_prefix"]["target_event"]["event"] == "resumable-leg"
    assert evidence["event_prefix"]["integrity_hold_before_target"] is False
    assert evidence["completed_gate_seeds"] == [cutover.V5_GATE_SEED, cutover.V6_GATE_SEED]
    assert evidence["gate_row_sha256"][0] == cutover.V5_GATE_ROW_SHA256
    assert evidence["scientific_promotion"] is False


def test_seed2_continuity_rejects_a_resealed_prefix_mutation() -> None:
    run_root = REPO_ROOT / "runs/local_throttle" / cutover.V6_SEED2_RUN_ID / "artifacts"
    artifact = _json(run_root / f"EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-{cutover.V6_GATE_SEED}.json")
    checkpoint = _json(
        run_root / f"EDCM1_EVENT_TRIGGERED_COALITION_V3.seed-{cutover.V6_GATE_SEED}.checkpoint.json"
    )
    mutated = copy.deepcopy(artifact)
    mutated["checkpoint_binding"]["gate_row_sha256"][0] = "0" * 64
    _reseal(mutated, "receipt_sha256")

    with pytest.raises(cutover.EvidenceError, match="lost the archived v5 prefix"):
        cutover._validate_seed2_continuity(mutated, checkpoint)


def test_event_parser_rejects_a_forged_self_seal() -> None:
    raw = (REPO_ROOT / cutover.V6_EVENTS_PATH).read_bytes()
    first, *remaining = raw.splitlines(keepends=True)
    event = json.loads(first)
    event["execute"] = not event["execute"]
    forged = cutover.canonical_bytes(event) + b"\n" + b"".join(remaining)

    with pytest.raises(cutover.EvidenceError, match="self-seal mismatch"):
        cutover._event_rows(forged, "forged")


def test_cutover_build_is_deterministic_nonpromoting_and_ready() -> None:
    readiness = cutover.readiness_report(REPO_ROOT)
    first = cutover.build_document(REPO_ROOT)
    second = cutover.build_document(REPO_ROOT)

    assert readiness["cutover_ready"] is True
    assert first == second
    assert first["runtime_gates"]["minimum_memory_pressure_free_percent"] == 75.0
    assert first["scientific_configuration_changed"] is False
    assert first["scientific_promotion"] is False
    assert first["cutover_sha256"] == cutover.canonical_sha256(
        {key: value for key, value in first.items() if key != "cutover_sha256"}
    )


def test_verify_only_validation_refuses_stale_live_authorities_without_rewrite() -> None:
    path = REPO_ROOT / cutover.DEFAULT_OUTPUT_PATH
    before = path.read_bytes()
    before_stat = path.stat()

    with pytest.raises(
        cutover.EvidenceError,
        match="does not exactly match live authorities and immutable evidence",
    ):
        cutover.validate_cutover(REPO_ROOT)

    after_stat = path.stat()
    assert path.read_bytes() == before
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns
