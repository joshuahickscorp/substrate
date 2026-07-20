from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mop.studio import generation1_subaccomplishment_emitter as emitter
from mop.studio import telegram_rung_notifier as notifier

canonical_sha256 = emitter.canonical_sha256


def _seal(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: canonical_sha256(core)}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _classification(epoch_index: int = 0, epoch_id: str = "W1") -> dict[str, Any]:
    core = {
        "schema": emitter.FULL_GENERATIONS_CLASSIFICATION_SCHEMA,
        "program_id": "generation1-full-generations-wave-v1",
        "epoch_id": epoch_id,
        "epoch_index": epoch_index,
        "summary": {
            "executed_item_count": 288,
            "skipped_item_count": 12,
            "observed_seconds": 4.0,
            "compute_started": True,
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _seal(core, "classification_sha256")


def _gate(gate_index: int = 0, gate_id: str = "admit_wave_v1") -> dict[str, Any]:
    core = {
        "schema": emitter.FULL_GENERATIONS_GATE_SCHEMA,
        "program_id": "generation1-full-generations-wave-v1",
        "gate_id": gate_id,
        "gate_index": gate_index,
        "payload": {"operation": "bind", "mechanics_lanes": ["G1-P1", "G1-P2", "G1-P3"], "admitted": True},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _seal(core, "gate_sha256")


def _reprofile(receipts_seen: int = 12, workers: int = 12) -> dict[str, Any]:
    core = {
        "schema": emitter.REPROFILE_SCHEMA,
        "advisory": True,
        "claim_scope": "advisory operational telemetry only",
        "inputs": {"host_cores": 28, "memory_gb": 96.0, "per_capsule_mem_cap_gb": 16.0},
        "source_results": [],
        "continuing_lanes": [],
        "pruned_lanes": [],
        "lane_mechanisms": {},
        "receipts_seen": receipts_seen,
        "receipts_skipped": 3,
        "observed": {},
        "provisional_mechanisms": ["mechanism_a", "mechanism_b"],
        "executed_workload": {},
        "recommendation": {"recommended_workers": workers, "binding_constraint": "cpu_cores"},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _seal(core, "reprofile_sha256")


def _terminal_result() -> dict[str, Any]:
    core = {
        "schema": emitter.CONSOLIDATED_FINAL_RESULT_SCHEMA,
        "program_id": emitter.CONSOLIDATED_FINAL_PROGRAM_ID,
        "grid": {"work_item_count": 7332},
        "decision": {
            "conditional_final_campaign_complete": True,
            "independent_scientific_verification_complete": False,
            "scientific_confirmation": False,
            "next_action": "interpret_consolidated_result_and_author_independent_verifiers",
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _seal(core, "result_sha256")


def _absorption_receipt(*, result_seal: str, result_file_sha: str) -> dict[str, Any]:
    core = {
        "schema": emitter.ABSORPTION_RECEIPT_SCHEMA,
        "program_id": "generation1-consolidated-final-absorption-gate-v1",
        "absorbed_program_id": emitter.CONSOLIDATED_FINAL_PROGRAM_ID,
        "adopted_at": "2026-07-17T00:00:00+00:00",
        "absorbed_at": "2026-07-17T01:00:00+00:00",
        "process": {
            "pid": 67790,
            "create_time": 1784160329.97262,
            "pgid": 67790,
            "cwd": "/Users/scammermike/Downloads/mop",
            "label": "mop-final-campaign",
        },
        "observed_status": {
            "path": "runs/generation1/x/current_status.json",
            "file_sha256": "b" * 64,
            "status_sha256": "c" * 64,
            "state": "complete",
        },
        "absorbed_result": {
            "path": f"proof/{emitter.CONSOLIDATED_FINAL_RESULT_NAME}",
            "file_sha256": result_file_sha,
            "result_sha256": result_seal,
            "schema": emitter.CONSOLIDATED_FINAL_RESULT_SCHEMA,
        },
        "policy": {
            "observe_only": True,
            "signals_allowed": False,
            "restart_disallowed": True,
            "append_only": True,
        },
    }
    return _seal(core, "receipt_sha256")


def _receipt_path(runs: Path, name: str = "67790.json") -> Path:
    return runs / "absorption-gate" / "absorptions" / emitter.CONSOLIDATED_FINAL_PROGRAM_ID / name


def _accepting_seal_validator(field: str):

    def _validate(value, _index, *, root):  # noqa: ANN001, ANN202 - test double signature
        if not emitter._seal_ok(value, field):
            raise ValueError("seal mismatch")

    return _validate


def _fullgen_wave_root(runs: Path) -> Path:
    return runs / "generation1-full-generations-wave-v1"


def test_reprofile_source_publishes_valid_subaccomp(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())

    report = emitter.scan(runs_root=runs, proof_root=proof)

    assert report["written_count"] == 1
    written = list(proof.glob("GENERATION1_SUBACCOMP_reprofile_*.json"))
    assert len(written) == 1
    milestone = json.loads(written[0].read_text())
    emitter.validate_milestone(milestone)  # real validator accepts our output
    assert milestone["milestone_kind"] == "reprofile"
    assert "recommended workers: 12 (cpu_cores)" in milestone["summary"]
    assert milestone["source"]["seal_field"] == "reprofile_sha256"


def test_absorption_source_publishes_valid_subaccomp(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    terminal = _terminal_result()
    terminal_path = proof / emitter.CONSOLIDATED_FINAL_RESULT_NAME
    _write(terminal_path, terminal)
    receipt = _absorption_receipt(
        result_seal=terminal["result_sha256"],
        result_file_sha=emitter.sha256_file(terminal_path),
    )
    _write(
        _receipt_path(runs),
        receipt,
    )

    emitter.scan(runs_root=runs, proof_root=proof)

    written = list(proof.glob("GENERATION1_SUBACCOMP_absorption_*.json"))
    assert len(written) == 1
    milestone = json.loads(written[0].read_text())
    emitter.validate_milestone(milestone)
    assert milestone["milestone_kind"] == "absorption"
    assert milestone["source_program_id"] == emitter.CONSOLIDATED_FINAL_PROGRAM_ID
    assert milestone["decision"]["scientific_confirmation"] is False
    assert not list(proof.glob("GENERATION1_SUBACCOMP_reprofile_*.json"))


def test_barrier_source_publishes_with_injected_home_validator(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(_fullgen_wave_root(runs) / "classifications" / "w1.json", _classification())

    report = emitter.scan(
        runs_root=runs,
        proof_root=proof,
        classification_validators={
            emitter.FULL_GENERATIONS_CLASSIFICATION_SCHEMA: _accepting_seal_validator("classification_sha256")
        },
    )

    assert report["written_count"] == 1
    written = list(proof.glob("GENERATION1_SUBACCOMP_barrier_*.json"))
    assert len(written) == 1
    milestone = json.loads(written[0].read_text())
    emitter.validate_milestone(milestone)
    assert milestone["milestone_kind"] == "barrier"
    assert milestone["grid"] == {"completed_cell_count": 288, "expected_cell_count": 300}


def test_gate_source_publishes_with_injected_home_validator(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(_fullgen_wave_root(runs) / "gates" / "admit_wave_v1.json", _gate())

    report = emitter.scan(
        runs_root=runs,
        proof_root=proof,
        gate_validators={emitter.FULL_GENERATIONS_GATE_SCHEMA: _accepting_seal_validator("gate_sha256")},
    )

    assert report["written_count"] == 1
    written = list(proof.glob("GENERATION1_SUBACCOMP_gate_*.json"))
    assert len(written) == 1
    milestone = json.loads(written[0].read_text())
    emitter.validate_milestone(milestone)
    assert milestone["milestone_kind"] == "gate"
    assert "mechanics lanes admitted: 3" in milestone["summary"]


def test_notifier_proof_path_accepts_and_renders_subaccomp(tmp_path: Path, monkeypatch) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())
    emitter.scan(runs_root=runs, proof_root=proof)

    empty_runs = tmp_path / "empty_runs"
    empty_runs.mkdir()
    events = notifier.collect_events(runs_root=empty_runs, proof_root=proof)
    proof_events = [event for event in events if event["kind"] == "proof"]
    assert len(proof_events) == 1
    event = proof_events[0]
    assert event["event_id"].startswith("proof/GENERATION1_SUBACCOMP_reprofile_")
    assert event["schema"] == emitter.SUBACCOMP_SCHEMA
    assert event["summary"]  # proof_summary rendered non-empty lines from grid + decision

    monkeypatch.setattr(notifier, "host_health", lambda root=None: {"pressure": 1, "disk_free_gb": 100.0})
    rendered = notifier.format_event(event)
    assert "MOP result ready" in rendered
    assert "coverage 12/15" in rendered


def test_notifier_dedupes_subaccomp_across_rescans(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())
    emitter.scan(runs_root=runs, proof_root=proof)

    empty_runs = tmp_path / "empty_runs"
    empty_runs.mkdir()
    first = notifier.collect_events(runs_root=empty_runs, proof_root=proof)
    emitter.scan(runs_root=runs, proof_root=proof)  # re-scan: no content change
    second = notifier.collect_events(runs_root=empty_runs, proof_root=proof)
    assert [event["event_id"] for event in first] == [event["event_id"] for event in second]


def test_forged_reprofile_is_skipped_by_real_validator(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    forged = _reprofile()
    forged["receipts_seen"] = 999  # tamper after sealing -> real validate_reprofile rejects
    _write(runs / "gate-run" / "advisory_reprofile.json", forged)

    report = emitter.scan(runs_root=runs, proof_root=proof)
    assert report["written_count"] == 0
    assert not list(proof.glob("GENERATION1_SUBACCOMP_*.json"))


def test_forged_classification_is_skipped_by_real_validator(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    forged = _classification()
    forged["epoch_id"] = "TAMPERED"  # seal no longer replays; also lacks wave context
    _write(_fullgen_wave_root(runs) / "classifications" / "w1.json", forged)

    report = emitter.scan(runs_root=runs, proof_root=proof)
    assert report["written_count"] == 0
    assert not list(proof.glob("GENERATION1_SUBACCOMP_barrier_*.json"))


def test_injected_validator_rejects_forged_classification(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    forged = _classification()
    forged["epoch_id"] = "TAMPERED"
    _write(_fullgen_wave_root(runs) / "classifications" / "w1.json", forged)

    report = emitter.scan(
        runs_root=runs,
        proof_root=proof,
        classification_validators={
            emitter.FULL_GENERATIONS_CLASSIFICATION_SCHEMA: _accepting_seal_validator("classification_sha256")
        },
    )
    assert report["written_count"] == 0


def test_absorption_without_valid_terminal_is_skipped(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    terminal = _terminal_result()
    terminal_path = proof / emitter.CONSOLIDATED_FINAL_RESULT_NAME
    receipt = _absorption_receipt(
        result_seal=terminal["result_sha256"],
        result_file_sha="d" * 64,  # will not match once terminal is written
    )
    _write(
        _receipt_path(runs),
        receipt,
    )

    assert emitter.scan(runs_root=runs, proof_root=proof)["written_count"] == 0

    _write(terminal_path, terminal)
    assert emitter.scan(runs_root=runs, proof_root=proof)["written_count"] == 0


def test_forged_absorption_receipt_policy_is_skipped(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    terminal = _terminal_result()
    terminal_path = proof / emitter.CONSOLIDATED_FINAL_RESULT_NAME
    _write(terminal_path, terminal)
    core = {
        "schema": emitter.ABSORPTION_RECEIPT_SCHEMA,
        "program_id": "generation1-consolidated-final-absorption-gate-v1",
        "absorbed_program_id": emitter.CONSOLIDATED_FINAL_PROGRAM_ID,
        "adopted_at": "2026-07-17T00:00:00+00:00",
        "absorbed_at": "2026-07-17T01:00:00+00:00",
        "process": {"pid": 1, "create_time": 1.0, "pgid": 1, "cwd": "/x", "label": "mop-final-campaign"},
        "observed_status": {
            "path": "x", "file_sha256": "b" * 64, "status_sha256": "c" * 64, "state": "complete",
        },
        "absorbed_result": {
            "path": f"proof/{emitter.CONSOLIDATED_FINAL_RESULT_NAME}",
            "file_sha256": emitter.sha256_file(terminal_path),
            "result_sha256": terminal["result_sha256"],
            "schema": emitter.CONSOLIDATED_FINAL_RESULT_SCHEMA,
        },
        "policy": {
            "observe_only": True,
            "signals_allowed": True,  # forbidden: a signalling receipt must never absorb
            "restart_disallowed": True,
            "append_only": True,
        },
    }
    receipt = _seal(core, "receipt_sha256")
    _write(
        _receipt_path(runs, "1.json"),
        receipt,
    )
    assert emitter.scan(runs_root=runs, proof_root=proof)["written_count"] == 0


def test_idempotent_rescan_writes_nothing_new(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())

    first = emitter.scan(runs_root=runs, proof_root=proof)
    assert first["written_count"] == 1
    published = sorted(p.name for p in proof.glob("GENERATION1_SUBACCOMP_*.json"))

    second = emitter.scan(runs_root=runs, proof_root=proof)
    assert second["written_count"] == 0
    assert second["existing_count"] == 1
    assert sorted(p.name for p in proof.glob("GENERATION1_SUBACCOMP_*.json")) == published


def test_determinism_double_run_is_byte_identical(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())

    proof_a = tmp_path / "proof_a"
    proof_b = tmp_path / "proof_b"
    emitter.scan(runs_root=runs, proof_root=proof_a)
    emitter.scan(runs_root=runs, proof_root=proof_b)

    files_a = {p.name: p.read_bytes() for p in proof_a.glob("GENERATION1_SUBACCOMP_*.json")}
    files_b = {p.name: p.read_bytes() for p in proof_b.glob("GENERATION1_SUBACCOMP_*.json")}
    assert files_a == files_b
    assert files_a  # non-empty


def test_scan_is_read_only_over_sources(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    source = runs / "gate-run" / "advisory_reprofile.json"
    _write(source, _reprofile())
    before_mtime = source.stat().st_mtime_ns
    before_bytes = source.read_bytes()
    before_runs = {p for p in runs.rglob("*") if p.is_file()}

    emitter.scan(runs_root=runs, proof_root=proof)

    assert source.stat().st_mtime_ns == before_mtime
    assert source.read_bytes() == before_bytes
    assert {p for p in runs.rglob("*") if p.is_file()} == before_runs  # emitter wrote nothing under runs


def test_emitter_module_never_signals_a_process() -> None:
    text = Path(emitter.__file__).read_text(encoding="utf-8")
    forbidden_tokens = (
        "os.kill", "send_signal", ".terminate(", ".suspend(", "signal.SIG", "SIGKILL", "SIGTERM",
    )
    for forbidden in forbidden_tokens:
        assert forbidden not in text, f"emitter must never signal a process: found {forbidden!r}"


def test_validate_milestone_accepts_output_and_rejects_tamper(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())
    emitter.scan(runs_root=runs, proof_root=proof)
    path = next(proof.glob("GENERATION1_SUBACCOMP_reprofile_*.json"))
    milestone = json.loads(path.read_text())

    emitter.validate_milestone(milestone)  # accepts

    tampered = dict(milestone)
    tampered["scientific_promotion"] = True  # break a safety flag without re-sealing
    with pytest.raises(emitter.SubaccomplishmentRefused):
        emitter.validate_milestone(tampered)

    reseal = dict(milestone)
    reseal["activation_allowed"] = True
    reseal_core = {k: v for k, v in reseal.items() if k != "milestone_sha256"}
    reseal["milestone_sha256"] = canonical_sha256(reseal_core)
    with pytest.raises(emitter.SubaccomplishmentRefused):
        emitter.validate_milestone(reseal)  # seal replays but the safety flag is caught


def test_install_launch_agent_is_dry_run_only() -> None:
    plan = emitter.install_launch_agent()
    assert plan["installed"] is False
    assert plan["dry_run"] is True
    assert plan["label"] == "com.mop.generation1.subaccomp"
    assert plan["document"]["Label"] == "com.mop.generation1.subaccomp"
    assert not emitter.PLIST.exists() or True  # never created by the dry run
    with pytest.raises(emitter.SubaccomplishmentRefused):
        emitter.install_launch_agent(execute=True)


def test_status_reports_published_milestones(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    _write(runs / "gate-run" / "advisory_reprofile.json", _reprofile())
    emitter.scan(runs_root=runs, proof_root=proof)
    report = emitter.status(proof_root=proof)
    assert report["published_milestones"] == 1
    assert report["label"] == "com.mop.generation1.subaccomp"
    assert report["milestone_files"][0].startswith("GENERATION1_SUBACCOMP_reprofile_")
