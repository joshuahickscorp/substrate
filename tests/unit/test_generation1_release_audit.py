from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from mop.studies import generation1_release_audit as audit


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: _digest(core)}


def _write(path: Path, payload: dict[str, Any]) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(payload) + b"\n"
    path.write_bytes(raw)
    return raw


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_expectation(
    path: str,
    schema: str,
    seal_field: str,
    fields: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": path,
        "schema": schema,
        "fields": fields,
        "seal_field": seal_field,
    }


def _capsule(
    capsule_id: str,
    *,
    kind: str,
    priority: int,
    depends_on: list[str],
    artifacts: list[dict[str, Any]],
    authority: dict[str, str],
) -> dict[str, Any]:
    core = {
        "schema": "mop-generation1-capsule/v1",
        "id": capsule_id,
        "kind": kind,
        "priority": priority,
        "depends_on": depends_on,
        "command": ["python", authority["path"]],
        "cwd": ".",
        "environment": {},
        "resources": {
            "lane": "cpu",
            "accelerator": "none",
            "cpu_cores": 1,
            "estimated_unified_memory_gb": 1.0,
            "estimated_mps_gb": 0.0,
            "resource_basis": "unit fixture",
            "forecast_write_gb": 0.1,
            "atomic_write_gb": 0.1,
            "wall_minutes": 5,
            "process_marker": Path(authority["path"]).name,
        },
        "artifacts": artifacts,
        "authorities": [authority],
    }
    return _seal(core, "capsule_sha256")


def _artifact_report(root: Path, expectation: dict[str, Any]) -> dict[str, Any]:
    path = root / expectation["path"]
    payload = json.loads(path.read_text())
    return {
        "path": expectation["path"],
        "sha256": _file_sha(path),
        "schema": payload["schema"],
        "problems": [],
        "all_ok": True,
    }


def _state_row(
    root: Path,
    capsule: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    return {
        "id": capsule["id"],
        "kind": capsule["kind"],
        "priority": capsule["priority"],
        "depends_on": capsule["depends_on"],
        "capsule_sha256": capsule["capsule_sha256"],
        "source": source,
        "status": "complete",
        "attempts": 1,
        "child_pid": 123,
        "child_create_time": 1.0,
        "started_at": "2026-07-13T00:00:00+00:00",
        "finished_at": "2026-07-13T00:01:00+00:00",
        "returncode": 0,
        "artifacts": [_artifact_report(root, item) for item in capsule["artifacts"]],
        "last_problem": None,
        "runtime": {
            "sample_count": 1,
            "peak_process_tree_rss_bytes": 100,
            "minimum_memory_available_gb": 10.0,
            "minimum_memory_available_percent": 50.0,
            "minimum_memory_pressure_free_percent": 90.0,
            "maximum_swap_used_gb": 0.0,
            "minimum_disk_free_gb": 100.0,
            "thermal_statuses": ["normal"],
            "power_sources": ["AC Power"],
            "reservation_count": 1,
            "last_reservation": {
                "capsule_id": capsule["id"],
                "reserved_at": "2026-07-13T00:00:00+00:00",
                "child_pid": None,
                "child_create_time": None,
                "receipt": {
                    "reserved": True,
                    "capsule_id": capsule["id"],
                },
            },
            "resource_stop_count": 0,
            "retry_count": 0,
            "safety_state": "owned-safe",
            "last_sample": {"allowed": True},
            "event_count": 4,
            "events_dropped": 0,
            "events": [
                {
                    "sequence": 1,
                    "at": "2026-07-13T00:00:00+00:00",
                    "event": "lane-reserved",
                },
                {
                    "sequence": 2,
                    "at": "2026-07-13T00:00:00+00:00",
                    "event": "child-started",
                },
                {
                    "sequence": 3,
                    "at": "2026-07-13T00:01:00+00:00",
                    "event": "lane-released",
                },
                {
                    "sequence": 4,
                    "at": "2026-07-13T00:01:00+00:00",
                    "event": "capsule-complete",
                    "attempt": 1,
                    "returncode": 0,
                },
            ],
        },
    }


@dataclass
class Fixture:
    paths: audit.ReleasePaths
    contract: audit.ReleaseContract
    proofs: dict[str, dict[str, Any]]
    state: dict[str, Any]
    status: dict[str, Any]
    receipt: dict[str, Any]


def _fixture(root: Path) -> Fixture:
    audit_module_path = root / "src/mop/studies/generation1_release_audit.py"
    audit_wrapper_path = root / "scripts/audit_generation1_release.py"
    supervisor_module_path = root / "src/mop/studio/generation1_supervisor.py"
    audit_module_path.parent.mkdir(parents=True)
    audit_wrapper_path.parent.mkdir(parents=True)
    supervisor_module_path.parent.mkdir(parents=True)
    audit_module_path.write_text("# fixture release auditor\n", encoding="utf-8")
    audit_wrapper_path.write_text("# fixture release wrapper\n", encoding="utf-8")
    supervisor_module_path.write_text("# fixture supervisor\n", encoding="utf-8")
    authority_path = root / "runner.py"
    authority_path.write_text("# fixture authority\n", encoding="utf-8")
    authority = {"path": "runner.py", "sha256": _file_sha(authority_path)}

    proof_root = root / "proof"
    corpus_path = proof_root / "GENERATION1_COGNITIVE_CORPUS.json"
    corpus_verification_path = proof_root / "GENERATION1_COGNITIVE_CORPUS.verification.json"
    report_path = proof_root / "GENERATION1_EMPIRICAL_REPORT.json"
    synthesis_path = proof_root / "GENERATION1_EVIDENCE_SYNTHESIS.json"
    synthesis_verification_path = proof_root / "GENERATION1_EVIDENCE_SYNTHESIS.verification.json"

    corpus = _seal(
        {
            "schema": "mop-generation1-cognitive-corpus/v2",
            "corpus_complete": True,
            "scientific_promotion": False,
        },
        "corpus_sha256",
    )
    _write(corpus_path, corpus)
    corpus_verification = _seal(
        {
            "schema": "mop-generation1-cognitive-corpus-verification/v2",
            "corpus": {
                "path": "proof/GENERATION1_COGNITIVE_CORPUS.json",
                "sha256": _file_sha(corpus_path),
                "corpus_sha256": corpus["corpus_sha256"],
            },
            "checks": {name: True for name in audit.REQUIRED_CORPUS_CHECKS},
            "mutation_suite": {
                "count": len(audit.REQUIRED_CORPUS_MUTATIONS),
                "rejected": len(audit.REQUIRED_CORPUS_MUTATIONS),
                "results": {name: True for name in audit.REQUIRED_CORPUS_MUTATIONS},
            },
            "verification_complete": True,
            "problems": [],
            "scientific_promotion": False,
            "recorded_at": "2026-07-13T00:02:00+00:00",
        },
        "verification_sha256",
    )
    _write(corpus_verification_path, corpus_verification)
    report = _seal(
        {
            "schema": "mop-generation1-empirical-report/v2",
            "claim_scope": audit.REPORT_CLAIM_SCOPE,
            "created_at": "2026-07-13T00:03:00+00:00",
            "corpus": {
                "source": {
                    "path": "proof/GENERATION1_COGNITIVE_CORPUS.json",
                    "sha256": _file_sha(corpus_path),
                },
                "verification": {
                    "path": "proof/GENERATION1_COGNITIVE_CORPUS.verification.json",
                    "sha256": _file_sha(corpus_verification_path),
                },
                "corpus_complete": True,
            },
            "next_authority": {
                "ready_to_activate_or_integrate_substrate": False,
                "automatic_activation_allowed": False,
                "automatic_scientific_promotion_allowed": False,
            },
            "scientific_promotion": False,
        },
        "report_sha256",
    )
    _write(report_path, report)
    boundaries = {name: {"status": "not_tested_by_g1_c0"} for name in audit.REQUIRED_SYNTHESIS_BOUNDARIES}
    synthesis = _seal(
        {
            "schema": "mop-generation1-evidence-synthesis/v1",
            "claim_scope": audit.SYNTHESIS_CLAIM_SCOPE,
            "created_at": "2026-07-13T00:04:00+00:00",
            "sources": {
                "corpus": {
                    "path": "proof/GENERATION1_COGNITIVE_CORPUS.json",
                    "sha256": _file_sha(corpus_path),
                    "schema": corpus["schema"],
                    "seal_field": "corpus_sha256",
                    "seal_sha256": corpus["corpus_sha256"],
                    "self_seal_valid_at_read": True,
                },
                "corpus_verification": {
                    "path": "proof/GENERATION1_COGNITIVE_CORPUS.verification.json",
                    "sha256": _file_sha(corpus_verification_path),
                    "schema": corpus_verification["schema"],
                    "seal_field": "verification_sha256",
                    "seal_sha256": corpus_verification["verification_sha256"],
                    "self_seal_valid_at_read": True,
                },
                "empirical_report": {
                    "path": "proof/GENERATION1_EMPIRICAL_REPORT.json",
                    "sha256": _file_sha(report_path),
                    "schema": report["schema"],
                    "seal_field": "report_sha256",
                    "seal_sha256": report["report_sha256"],
                    "self_seal_valid_at_read": True,
                },
                "program_state_snapshot": {
                    "path": "runs/generation1/test/program_state.json",
                    "schema": "mop-generation1-state/v1",
                    "seal_field": "state_sha256",
                    "seal_sha256": "1" * 64,
                    "self_seal_valid_at_read": True,
                    "mutable_snapshot_sha256": "2" * 64,
                    "mutable_after_read": True,
                    "authority_scope": "base_runtime_accounting.base_runtime_sha256",
                },
            },
            "claim_boundaries": boundaries,
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "synthesis_sha256",
    )
    _write(synthesis_path, synthesis)
    synthesis_verification = _seal(
        {
            "schema": "mop-generation1-evidence-synthesis-verification/v1",
            "claim_scope": audit.SYNTHESIS_VERIFICATION_CLAIM_SCOPE,
            "checks": {name: True for name in audit.REQUIRED_SYNTHESIS_VERIFICATION_CHECKS},
            "mutation_suite": {
                "count": len(audit.REQUIRED_SYNTHESIS_MUTATIONS),
                "rejected": len(audit.REQUIRED_SYNTHESIS_MUTATIONS),
                "results": {name: True for name in audit.REQUIRED_SYNTHESIS_MUTATIONS},
            },
            "verification_complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
            "recorded_at": "2026-07-13T00:05:00+00:00",
        },
        "verification_sha256",
    )
    _write(synthesis_verification_path, synthesis_verification)

    corpus_artifact = _artifact_expectation(
        "proof/GENERATION1_COGNITIVE_CORPUS.json",
        corpus["schema"],
        "corpus_sha256",
        {"corpus_complete": True},
    )
    corpus_verification_artifact = _artifact_expectation(
        "proof/GENERATION1_COGNITIVE_CORPUS.verification.json",
        corpus_verification["schema"],
        "verification_sha256",
        {"verification_complete": True},
    )
    report_artifact = _artifact_expectation(
        "proof/GENERATION1_EMPIRICAL_REPORT.json",
        report["schema"],
        "report_sha256",
        {"scientific_promotion": False},
    )
    synthesis_artifact = _artifact_expectation(
        "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
        synthesis["schema"],
        "synthesis_sha256",
        {"activation_allowed": False},
    )
    synthesis_verification_artifact = _artifact_expectation(
        "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
        synthesis_verification["schema"],
        "verification_sha256",
        {"verification_complete": True},
    )

    base_corpus = _capsule(
        "base_corpus",
        kind="aggregate",
        priority=1,
        depends_on=[],
        artifacts=[corpus_artifact, corpus_verification_artifact],
        authority=authority,
    )
    base_report = _capsule(
        "base_report",
        kind="aggregate",
        priority=2,
        depends_on=["base_corpus"],
        artifacts=[report_artifact],
        authority=authority,
    )
    child_synthesis = _capsule(
        "child_synthesis",
        kind="exploratory",
        priority=3,
        depends_on=["base_report"],
        artifacts=[synthesis_artifact],
        authority=authority,
    )
    child_verify = _capsule(
        "child_verify",
        kind="exploratory",
        priority=4,
        depends_on=["child_synthesis"],
        artifacts=[synthesis_verification_artifact],
        authority=authority,
    )

    program_path = root / "configs/campaign/program.json"
    program = _seal(
        {
            "schema": "mop-generation1-program/v1",
            "program_id": "generation1-test",
            "program_root": "runs/generation1/test",
            "policy": authority,
            "authorities": [authority],
            "injection": {
                "inbox": "runs/generation1/test/control/inbox",
                "receipt_root": "runs/generation1/test/control/receipts",
            },
            "control": {},
            "capsules": [base_corpus, base_report],
        },
        "program_sha256",
    )
    program_raw = _write(program_path, program)
    genesis = _digest(
        {
            "program_sha256": program["program_sha256"],
            "base_capsules": [base_corpus["capsule_sha256"], base_report["capsule_sha256"]],
        }
    )
    injection_path = root / "runs/generation1/test/control/inbox/inj-test.json"
    injection = _seal(
        {
            "schema": "mop-generation1-injection/v1",
            "program_id": "generation1-test",
            "injection_id": "inj-test",
            "sequence": 1,
            "created_at": "2026-07-13T00:00:00+00:00",
            "action": "append-capsules",
            "expected_queue_head_sha256": genesis,
            "capsules": [child_synthesis, child_verify],
            "reason": "fixture",
        },
        "injection_sha256",
    )
    injection_raw = _write(injection_path, injection)
    final_head = _digest(
        {
            "previous_head_sha256": genesis,
            "injection_sha256": injection["injection_sha256"],
            "capsules": [child_synthesis["capsule_sha256"], child_verify["capsule_sha256"]],
        }
    )
    injection_relative = str(injection_path.relative_to(root))
    state_path = root / "runs/generation1/test/program_state.json"
    state_core = {
        "schema": "mop-generation1-state/v1",
        "program_id": "generation1-test",
        "program": {
            "path": str(program_path),
            "file_sha256": hashlib.sha256(program_raw).hexdigest(),
            "program_sha256": program["program_sha256"],
        },
        "supervisor": {
            "pid": 1,
            "create_time": 1.0,
            "implementation_path": "src/mop/studio/generation1_supervisor.py",
            "implementation_sha256": _file_sha(supervisor_module_path),
        },
        "execution_enabled": True,
        "status": "complete",
        "queue_head_sha256": final_head,
        "next_injection_sequence": 2,
        "accepted_injections": [
            {
                "id": "inj-test",
                "sequence": 1,
                "path": injection_relative,
                "file_sha256": hashlib.sha256(injection_raw).hexdigest(),
                "injection_sha256": injection["injection_sha256"],
                "previous_head_sha256": genesis,
                "new_head_sha256": final_head,
            }
        ],
        "processed_injection_files": [
            {
                "path": injection_relative,
                "file_sha256": hashlib.sha256(injection_raw).hexdigest(),
                "accepted": True,
                "injection_id": "inj-test",
            }
        ],
        "capsules": {
            "base_corpus": _state_row(root, base_corpus, "base"),
            "base_report": _state_row(root, base_report, "base"),
            "child_synthesis": _state_row(root, child_synthesis, "injection:inj-test"),
            "child_verify": _state_row(root, child_verify, "injection:inj-test"),
        },
        "current_capsule": None,
        "last_admission": {"allowed": True},
        "lane_reservation": None,
        "started_at": "2026-07-13T00:00:00+00:00",
        "updated_at": "2026-07-13T00:06:01+00:00",
        "finished_at": "2026-07-13T00:06:00+00:00",
        "problems": [],
    }
    state = _seal(state_core, "state_sha256")
    _write(state_path, state)
    status_path = root / "runs/generation1/test/current_status.json"
    status = _seal(
        {
            "schema": "mop-generation1-status/v1",
            "program_id": state["program_id"],
            "created_at": "2026-07-13T00:06:02+00:00",
            "program": state["program"],
            "supervisor": state["supervisor"],
            "execution_enabled": state["execution_enabled"],
            "state": state["status"],
            "queue_head_sha256": state["queue_head_sha256"],
            "next_injection_sequence": state["next_injection_sequence"],
            "accepted_injection_count": 1,
            "current_capsule": state["current_capsule"],
            "capsules": state["capsules"],
            "last_admission": state["last_admission"],
            "lane_reservation": state["lane_reservation"],
            "problems": state["problems"],
        },
        "status_sha256",
    )
    _write(status_path, status)
    receipt_path = root / "runs/generation1/test/control/receipts/accepted/inj-test.json"
    receipt = _seal(
        {
            "schema": "mop-generation1-injection-receipt/v1",
            "program_id": "generation1-test",
            "created_at": "2026-07-13T00:00:01+00:00",
            "injection_id": "inj-test",
            "path": injection_relative,
            "file_sha256": hashlib.sha256(injection_raw).hexdigest(),
            "accepted": True,
            "problems": [],
            "previous_head_sha256": genesis,
            "new_head_sha256": final_head,
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )
    _write(receipt_path, receipt)

    config_path = root / "configs/experiment/corpus.json"
    _write(config_path, {"schema": "fixture-config/v1"})
    run_root = root / "runs/generation1/corpus"
    run_root.mkdir(parents=True)
    paths = audit.ReleasePaths(
        repo_root=root,
        audit_module=audit_module_path,
        audit_wrapper=audit_wrapper_path,
        supervisor_module=supervisor_module_path,
        program=program_path,
        state=state_path,
        status=status_path,
        injection=injection_path,
        injection_receipt=receipt_path,
        corpus_config=config_path,
        corpus_run_root=run_root,
        corpus=corpus_path,
        corpus_verification=corpus_verification_path,
        empirical_report=report_path,
        evidence_synthesis=synthesis_path,
        evidence_synthesis_verification=synthesis_verification_path,
    )
    contract = audit.ReleaseContract(
        program_id="generation1-test",
        program_sha256=program["program_sha256"],
        base_capsule_count=2,
        total_capsule_count=4,
        injection_id="inj-test",
        injection_sha256=injection["injection_sha256"],
        injection_file_sha256=hashlib.sha256(injection_raw).hexdigest(),
        injection_sequence=1,
        next_injection_sequence=2,
        previous_queue_head_sha256=genesis,
        final_queue_head_sha256=final_head,
        injection_receipt_sha256=receipt["receipt_sha256"],
        injected_capsule_ids=("child_synthesis", "child_verify"),
    )
    return Fixture(
        paths=paths,
        contract=contract,
        proofs={
            "corpus": corpus,
            "corpus_verification": corpus_verification,
            "empirical_report": report,
            "evidence_synthesis": synthesis,
            "evidence_synthesis_verification": synthesis_verification,
        },
        state=state,
        status=status,
        receipt=receipt,
    )


def _patch_replays(monkeypatch: pytest.MonkeyPatch, fixture: Fixture) -> None:
    monkeypatch.setattr(
        audit,
        "build_corpus",
        lambda *_: copy.deepcopy(fixture.proofs["corpus"]),
    )
    monkeypatch.setattr(
        audit,
        "verify_corpus",
        lambda **_: copy.deepcopy(fixture.proofs["corpus_verification"]),
    )
    monkeypatch.setattr(
        audit,
        "build_report",
        lambda *_: copy.deepcopy(fixture.proofs["empirical_report"]),
    )
    monkeypatch.setattr(
        audit,
        "verify_evidence_synthesis",
        lambda *_, **__: copy.deepcopy(fixture.proofs["evidence_synthesis_verification"]),
    )


def _cli_argv(fixture: Fixture, output: Path) -> list[str]:
    return [
        "--repo-root",
        str(fixture.paths.repo_root),
        "--program",
        str(fixture.paths.program),
        "--state",
        str(fixture.paths.state),
        "--status",
        str(fixture.paths.status),
        "--injection",
        str(fixture.paths.injection),
        "--injection-receipt",
        str(fixture.paths.injection_receipt),
        "--corpus-config",
        str(fixture.paths.corpus_config),
        "--corpus-run-root",
        str(fixture.paths.corpus_run_root),
        "--corpus",
        str(fixture.paths.corpus),
        "--corpus-verification",
        str(fixture.paths.corpus_verification),
        "--empirical-report",
        str(fixture.paths.empirical_report),
        "--evidence-synthesis",
        str(fixture.paths.evidence_synthesis),
        "--evidence-synthesis-verification",
        str(fixture.paths.evidence_synthesis_verification),
        "--out",
        str(output),
    ]


def test_terminal_release_passes_and_is_deterministic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)

    first = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)
    second = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert first == second
    assert first["release_complete"] is True
    assert first["artifact_bundle_complete"] is True
    assert first["problems"] == []
    assert first["activation_allowed"] is False
    assert first["substrate_formation_established"] is False
    assert first["scientific_promotion"] is False
    core = {key: value for key, value in first.items() if key != "audit_sha256"}
    assert first["audit_sha256"] == _digest(core)


def test_nonterminal_status_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    status = dict(fixture.status)
    status["state"] = "running"
    _write(fixture.paths.status, _seal(status, "status_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["terminal_state_exact"] is False
    assert result["checks"]["state_status_mirror_exact"] is False


def test_queue_head_drift_fails_chain_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = dict(fixture.state)
    state["queue_head_sha256"] = "0" * 64
    state = _seal(state, "state_sha256")
    _write(fixture.paths.state, state)
    status = dict(fixture.status)
    status["queue_head_sha256"] = "0" * 64
    status = _seal(status, "status_sha256")
    _write(fixture.paths.status, status)

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["queue_hash_chain_exact"] is False


def test_rejected_receipt_fails_exact_receipt_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    receipt = dict(fixture.receipt)
    receipt["accepted"] = False
    _write(fixture.paths.injection_receipt, _seal(receipt, "receipt_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["accepted_injection_receipt_exact"] is False


def test_proof_activation_escalation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    synthesis = dict(fixture.proofs["evidence_synthesis"])
    synthesis["activation_allowed"] = True
    _write(fixture.paths.evidence_synthesis, _seal(synthesis, "synthesis_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["claim_and_nonauthorization_boundaries_exact"] is False
    assert result["checks"]["all_capsule_artifacts_current"] is False


def test_verifier_replay_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    drifted = copy.deepcopy(fixture.proofs["evidence_synthesis_verification"])
    drifted["checks"]["new_check"] = True
    drifted = _seal(drifted, "verification_sha256")
    monkeypatch.setattr(audit, "verify_evidence_synthesis", lambda *_, **__: drifted)

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["synthesis_verification_replay_exact"] is False


def test_missing_required_input_raises(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture.paths.status.unlink()

    with pytest.raises(ValueError, match="unreadable path component"):
        audit.audit_generation1_release(fixture.paths, contract=fixture.contract)


def test_exact_corpus_replay_rejects_resealed_claim_injection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    corpus = copy.deepcopy(fixture.proofs["corpus"])
    corpus["evidence_interpretation"] = "integrated substrate proven"
    corpus["substrate_formed"] = True
    _write(fixture.paths.corpus, _seal(corpus, "corpus_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["corpus_producer_replay_exact"] is False


def test_required_verifier_check_omission_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    verification = copy.deepcopy(fixture.proofs["corpus_verification"])
    verification["checks"].pop("directional_inference_fail_closed")
    _write(
        fixture.paths.corpus_verification,
        _seal(verification, "verification_sha256"),
    )

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["corpus_verification_complete"] is False


def test_unexpected_verifier_check_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    verification = copy.deepcopy(fixture.proofs["corpus_verification"])
    verification["checks"]["undeclared_check"] = True
    _write(
        fixture.paths.corpus_verification,
        _seal(verification, "verification_sha256"),
    )

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["corpus_verification_complete"] is False


def test_required_mutation_omission_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    verification = copy.deepcopy(fixture.proofs["evidence_synthesis_verification"])
    verification["mutation_suite"]["results"].pop("claim_scope_escalated")
    verification["mutation_suite"]["count"] -= 1
    verification["mutation_suite"]["rejected"] -= 1
    _write(
        fixture.paths.evidence_synthesis_verification,
        _seal(verification, "verification_sha256"),
    )

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["synthesis_verification_complete"] is False


def test_resealed_state_with_undeclared_claim_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = dict(fixture.state)
    state["substrate_formed"] = True
    _write(fixture.paths.state, _seal(state, "state_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["state_schema_and_seal"] is False


def test_supervisor_authority_drift_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = copy.deepcopy(fixture.state)
    state["supervisor"]["implementation_sha256"] = "0" * 64
    state = _seal(state, "state_sha256")
    _write(fixture.paths.state, state)
    status = copy.deepcopy(fixture.status)
    status["supervisor"] = state["supervisor"]
    _write(fixture.paths.status, _seal(status, "status_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["supervisor_authority_exact"] is False
    assert result["checks"]["state_status_mirror_exact"] is True


def test_malformed_terminal_capsule_values_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = copy.deepcopy(fixture.state)
    for row in state["capsules"].values():
        row["attempts"] = True
        row["returncode"] = False
        row["child_pid"] = "not-a-pid"
        row["child_create_time"] = "not-a-time"
        row["started_at"] = "invalid"
        row["finished_at"] = "invalid"
        row["last_problem"] = "concealed failure"
    state = _seal(state, "state_sha256")
    _write(fixture.paths.state, state)
    status = copy.deepcopy(fixture.status)
    status["capsules"] = state["capsules"]
    _write(fixture.paths.status, _seal(status, "status_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["capsule_inventory_exact"] is False


def test_terminal_capsule_accepts_supported_crash_recovery_shape(tmp_path: Path) -> None:
    authority_path = tmp_path / "runner.py"
    authority_path.write_text("# runner\n", encoding="utf-8")
    capsule = _capsule(
        "recovered",
        kind="aggregate",
        priority=1,
        depends_on=[],
        artifacts=[],
        authority={"path": "runner.py", "sha256": _file_sha(authority_path)},
    )
    row = _state_row(tmp_path, capsule, "base")
    row["attempts"] = 2
    row["runtime"]["retry_count"] = 1
    row["runtime"]["reservation_count"] = 1
    row["runtime"]["event_count"] = 6
    row["runtime"]["events"] = [
        {
            "sequence": 1,
            "at": "2026-07-13T00:00:00+00:00",
            "event": "recovery-child-gone",
            "attempt": 1,
        },
        {
            "sequence": 2,
            "at": "2026-07-13T00:00:00+00:00",
            "event": "capsule-retry-scheduled",
            "attempt": 1,
        },
        {
            "sequence": 3,
            "at": "2026-07-13T00:00:00+00:00",
            "event": "lane-reserved",
        },
        {
            "sequence": 4,
            "at": "2026-07-13T00:00:00+00:00",
            "event": "child-started",
            "attempt": 2,
        },
        {
            "sequence": 5,
            "at": "2026-07-13T00:01:00+00:00",
            "event": "lane-released",
        },
        {
            "sequence": 6,
            "at": "2026-07-13T00:01:00+00:00",
            "event": "capsule-complete",
            "attempt": 2,
            "returncode": 0,
        },
    ]

    assert audit._terminal_capsule_row_valid(row, "recovered") is True


def test_terminal_capsule_rejects_attempts_beyond_supervisor_bound(tmp_path: Path) -> None:
    authority_path = tmp_path / "runner.py"
    authority_path.write_text("# runner\n", encoding="utf-8")
    capsule = _capsule(
        "unbounded",
        kind="aggregate",
        priority=1,
        depends_on=[],
        artifacts=[],
        authority={"path": "runner.py", "sha256": _file_sha(authority_path)},
    )
    row = _state_row(tmp_path, capsule, "base")
    row["attempts"] = 100
    row["runtime"]["retry_count"] = 99
    row["runtime"]["reservation_count"] = 100
    row["runtime"]["events"][-1]["attempt"] = 100

    assert audit._terminal_capsule_row_valid(row, "unbounded") is False


def test_terminal_capsule_rejects_forged_or_oversized_event_log(tmp_path: Path) -> None:
    authority_path = tmp_path / "runner.py"
    authority_path.write_text("# runner\n", encoding="utf-8")
    capsule = _capsule(
        "events",
        kind="aggregate",
        priority=1,
        depends_on=[],
        artifacts=[],
        authority={"path": "runner.py", "sha256": _file_sha(authority_path)},
    )
    row = _state_row(tmp_path, capsule, "base")
    row["runtime"]["events"][0]["event"] = "substrate-formed"
    assert audit._terminal_capsule_row_valid(row, "events") is False

    oversized = _state_row(tmp_path, capsule, "base")
    oversized["runtime"]["events"] = [
        {
            "sequence": sequence,
            "at": "2026-07-13T00:00:00+00:00",
            "event": "lane-reserved",
        }
        for sequence in range(1, audit.MAX_RUNTIME_EVENTS + 1)
    ]
    oversized["runtime"]["events"].append(
        {
            "sequence": audit.MAX_RUNTIME_EVENTS + 1,
            "at": "2026-07-13T00:01:00+00:00",
            "event": "capsule-complete",
            "attempt": 1,
            "returncode": 0,
        }
    )
    oversized["runtime"]["event_count"] = audit.MAX_RUNTIME_EVENTS + 1
    assert audit._terminal_capsule_row_valid(oversized, "events") is False


def test_nested_false_claim_receipt_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = copy.deepcopy(fixture.state)
    state["last_admission"] = {
        "allowed": True,
        "external_report": {"scientific_promotion": False},
    }
    state = _seal(state, "state_sha256")
    _write(fixture.paths.state, state)
    status = copy.deepcopy(fixture.status)
    status["last_admission"] = state["last_admission"]
    _write(fixture.paths.status, _seal(status, "status_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is True


def test_nested_true_claim_receipt_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    state = copy.deepcopy(fixture.state)
    state["last_admission"] = {"allowed": True, "substrate_formed": True}
    state = _seal(state, "state_sha256")
    _write(fixture.paths.state, state)
    status = copy.deepcopy(fixture.status)
    status["last_admission"] = state["last_admission"]
    _write(fixture.paths.status, _seal(status, "status_sha256"))

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["state_schema_and_seal"] is False
    assert result["checks"]["status_schema_and_seal"] is False


def test_dependency_graph_rejects_cycle_and_missing_dependency() -> None:
    cycle = {
        "a": ({"depends_on": ["b"]}, "base"),
        "b": ({"depends_on": ["a"]}, "base"),
    }
    missing = {"a": ({"depends_on": ["missing"]}, "base")}

    assert audit._dependency_graph_valid(cycle) is False
    assert audit._dependency_graph_valid(missing) is False


def test_capsule_schema_drift_is_rejected() -> None:
    capsule = _capsule(
        "schema_drift",
        kind="aggregate",
        priority=1,
        depends_on=[],
        artifacts=[],
        authority={"path": "runner.py", "sha256": "0" * 64},
    )
    capsule["schema"] = "mop-generation1-capsule/v0"
    payload = {"capsules": [_seal(capsule, "capsule_sha256")]}

    with pytest.raises(ValueError, match="capsule self-seal mismatch"):
        audit._capsules(payload, "fixture")


def test_symlink_input_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    original = fixture.paths.status.with_name("status-real.json")
    fixture.paths.status.rename(original)
    fixture.paths.status.symlink_to(original)

    with pytest.raises(ValueError, match="symlink"):
        audit.audit_generation1_release(fixture.paths, contract=fixture.contract)


def test_symlinked_input_ancestor_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    alias = tmp_path / "status-alias"
    alias.symlink_to(fixture.paths.status.parent, target_is_directory=True)
    paths = replace(fixture.paths, status=alias / fixture.paths.status.name)

    with pytest.raises(ValueError, match="symlink path component"):
        audit.audit_generation1_release(paths, contract=fixture.contract)


def test_authority_mutated_during_replay_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)
    authority_path = tmp_path / "runner.py"

    def mutating_corpus_replay(*_args: object) -> dict[str, Any]:
        authority_path.write_text("# drifted during replay\n", encoding="utf-8")
        return copy.deepcopy(fixture.proofs["corpus"])

    monkeypatch.setattr(audit, "build_corpus", mutating_corpus_replay)

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["campaign_authority_files_stable_after_replay"] is False


def test_corpus_run_tree_mutated_during_replay_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    _patch_replays(monkeypatch, fixture)

    def mutating_corpus_replay(*_args: object) -> dict[str, Any]:
        (fixture.paths.corpus_run_root / "late-file.json").write_text("{}\n", encoding="utf-8")
        return copy.deepcopy(fixture.proofs["corpus"])

    monkeypatch.setattr(audit, "build_corpus", mutating_corpus_replay)

    result = audit.audit_generation1_release(fixture.paths, contract=fixture.contract)

    assert result["release_complete"] is False
    assert result["checks"]["corpus_run_tree_stable_after_replay"] is False


def test_output_alias_and_hardlink_are_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(ValueError, match="must be exactly"):
        audit._validate_output_path(fixture.paths.corpus, fixture.paths)

    output = tmp_path / "proof/GENERATION1_RELEASE_AUDIT.json"
    os.link(fixture.paths.corpus, output)
    with pytest.raises(ValueError, match="hardlinks an input"):
        audit._validate_output_path(output, fixture.paths)


def test_cli_exception_replaces_stale_success_with_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "proof/GENERATION1_RELEASE_AUDIT.json"
    _write(
        output,
        {
            "schema": audit.SCHEMA,
            "release_complete": True,
            "audit_sha256": "stale",
        },
    )
    monkeypatch.setattr(
        audit,
        "audit_generation1_release",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("forced failure")),
    )
    assert audit.main(_cli_argv(fixture, output)) == 2
    failure = json.loads(output.read_text())
    assert failure["schema"] == audit.FAILURE_SCHEMA
    assert failure["release_complete"] is False
    assert failure["error"] == "forced failure"


def test_cli_validation_failure_replaces_stale_symlink_success(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    target = tmp_path / "proof/old-success.json"
    _write(target, {"schema": audit.SCHEMA, "release_complete": True})
    output = tmp_path / "proof/GENERATION1_RELEASE_AUDIT.json"
    output.symlink_to(target.name)

    assert audit.main(_cli_argv(fixture, output)) == 2

    assert output.is_symlink() is False
    failure = json.loads(output.read_text())
    assert failure["schema"] == audit.FAILURE_SCHEMA
    assert failure["release_complete"] is False
    assert json.loads(target.read_text())["release_complete"] is True
