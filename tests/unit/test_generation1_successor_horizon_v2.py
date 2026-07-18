from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mop.config import REPO_ROOT
from mop.studies import generation1_c3_d1_frozen_queue as d1
from mop.studies import generation1_consolidated_final_campaign as consolidated
from mop.studies import generation1_successor_horizon as predecessor
from mop.studies import generation1_successor_horizon_v2 as horizon
from mop.studies import generation1_successor_horizon_verify as predecessor_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics


def _cli() -> ModuleType:
    path = REPO_ROOT / "scripts/generation1_successor_horizon_v2/mop_generation1_successor_horizon.py"
    specification = importlib.util.spec_from_file_location("generation1_successor_horizon_v2_cli", path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _sealed(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: horizon.canonical_sha256(core)}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _fabricated_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    d1_classification: str,
    d1_continue: bool,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    monkeypatch.setattr(horizon, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(predecessor, "validate_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        predecessor,
        "validate_classification",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        predecessor,
        "validate_report_receipt",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        predecessor_verify,
        "validate_verification",
        lambda *_args, **_kwargs: None,
    )

    root = tmp_path / "runs/parent"
    h05 = _sealed(
        {
            "schema": predecessor.CLASSIFICATION_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "claim_scope": predecessor.CLAIM_SCOPE,
            "epoch_id": "H05",
            "epoch_index": 4,
            "cycle_index": 6,
            "d1": {
                "classification": d1_classification,
                "continue_d1": d1_continue,
            },
            "routing": {
                "continue_d1": d1_continue,
                "mechanics_lanes_for_next_epoch": [],
            },
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
            "independent_scientific_confirmation": False,
        },
        "classification_sha256",
    )
    h05_path = root / "classifications/h05.json"
    _write(h05_path, h05)
    classification_rows = [
        {
            "epoch_id": epoch_id,
            "cycle_index": cycle,
            "path": f"runs/parent/classifications/{epoch_id.lower()}.json",
            "file_sha256": str(index + 1) * 64,
            "classification_sha256": str(index + 2) * 64,
        }
        for index, (epoch_id, cycle) in enumerate(
            zip(predecessor.EPOCH_IDS[:-1], predecessor.EPOCH_CYCLES[:-1], strict=True)
        )
    ]
    classification_rows.append(
        {
            "epoch_id": "H05",
            "cycle_index": 6,
            "path": str(h05_path.relative_to(tmp_path)),
            "file_sha256": horizon.sha256_file(h05_path),
            "classification_sha256": h05["classification_sha256"],
        }
    )
    result = _sealed(
        {
            "schema": predecessor.RESULT_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "claim_scope": predecessor.CLAIM_SCOPE,
            "grid": {
                "epoch_count": len(predecessor.EPOCH_IDS),
                "executed_d1_rung_count": 0,
            },
            "classifications": classification_rows,
            "decision": {"independent_scientific_confirmation": False},
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "result_sha256",
    )
    result_path = tmp_path / "proof/parent.json"
    _write(result_path, result)
    verification = _sealed(
        {
            "schema": predecessor_verify.VERIFICATION_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "source": {
                "path": str(result_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(result_path),
                "result_sha256": result["result_sha256"],
            },
            "recomputation": {
                "executed_d1_rung_count": 0,
                "d1_classifications": {"H05": d1_classification},
            },
            "verification_complete": True,
            "independent_scientific_confirmation": False,
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    verification_path = tmp_path / "proof/parent.verification.json"
    _write(verification_path, verification)
    receipt = _sealed(
        {
            "schema": predecessor.REPORT_RECEIPT_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "result": {
                "path": str(result_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(result_path),
            },
            "verification": {
                "path": str(verification_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(verification_path),
            },
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "receipt_sha256",
    )
    receipt_path = root / "report_receipt.json"
    _write(receipt_path, receipt)
    return result_path, verification_path, receipt_path, h05


def test_v2_is_five_fresh_disjoint_cycles_and_more_than_one_parallel_day() -> None:
    assert horizon.PROGRAM_ID == "generation1-successor-horizon-v2"
    assert horizon.EPOCH_IDS == ("H06", "H07", "H08", "H09", "H10")
    assert horizon.EPOCH_CYCLES == (7, 8, 9, 10, 11)
    assert horizon.planned_horizon_compute_seconds() == predecessor.planned_horizon_compute_seconds()
    assert horizon.IDLE_WORKERS == 20
    assert horizon.planned_horizon_compute_seconds() >= 230 * 60 * 60
    # IDLE_WORKERS is the declared 20-worker ceiling, so the ideal parallel time is serial / 20.
    assert horizon.planned_horizon_compute_seconds() / horizon.IDLE_WORKERS >= 11 * 60 * 60

    predecessor_d1 = []
    predecessor_mechanics = []
    extension_d1 = []
    extension_mechanics = []
    for module, d1_destination, mechanics_destination in (
        (predecessor, predecessor_d1, predecessor_mechanics),
        (horizon, extension_d1, extension_mechanics),
    ):
        for epoch_index in range(len(module.EPOCH_IDS)):
            for source_index in range(d1.DEFAULT_RUNG_COUNT):
                config = consolidated.fresh_d1_config(module._d1_work(epoch_index, source_index))
                d1_destination.extend(
                    (
                        int(config[f"{kind}_seed_start"]),
                        int(config[f"{kind}_seed_start"]) + int(config[f"{kind}_seed_count"]),
                    )
                    for kind in ("train", "heldout")
                )
            for source_index in range(len(mechanics.WORK_ITEMS)):
                item = consolidated.fresh_mechanics_item(module._mechanics_work(epoch_index, source_index))
                mechanics_destination.append((item.seed_start, item.seed_start + item.seed_count))
    assert max(end for _, end in predecessor_d1) < min(start for start, _ in extension_d1)
    assert max(end for _, end in predecessor_mechanics) < min(start for start, _ in extension_mechanics)


def test_partitions_and_scoped_work_items_reuse_v1_without_mutating_it() -> None:
    assert horizon.D1_PARTITIONS == predecessor.D1_PARTITIONS
    assert horizon.MECHANICS_PARTITIONS == predecessor.MECHANICS_PARTITIONS
    before = (
        predecessor.PROGRAM_ID,
        predecessor.EPOCH_IDS,
        predecessor.EPOCH_CYCLES,
        predecessor.validate_admission,
    )
    work = horizon.shard_work_items(epoch_index=0, lane="d1", shard_index=0)[0]
    assert work.key.startswith("h06_d1_producer_")
    assert work.cycle == 7
    assert (
        predecessor.PROGRAM_ID,
        predecessor.EPOCH_IDS,
        predecessor.EPOCH_CYCLES,
        predecessor.validate_admission,
    ) == before

    with pytest.raises(RuntimeError, match="forced"), horizon._v1_runtime_scope():
        assert predecessor.PROGRAM_ID == horizon.PROGRAM_ID
        assert predecessor.EPOCH_IDS == horizon.EPOCH_IDS
        assert predecessor.validate_admission is horizon.validate_admission
        raise RuntimeError("forced")
    assert (
        predecessor.PROGRAM_ID,
        predecessor.EPOCH_IDS,
        predecessor.EPOCH_CYCLES,
        predecessor.validate_admission,
    ) == before


def test_dependency_closure_is_transitive_and_ignores_nonmechanics_dependencies() -> None:
    order = [lane.lane_id for lane in mechanics.LANES]
    eligible, pruned = horizon._dependency_closed_lanes([lane for lane in order if lane != "G1-P1"])
    assert "G1-P1" not in eligible
    assert "G1-I1" not in eligible
    assert pruned == ["G1-I1"]
    assert "G1-V1" in eligible  # G1-C1 is not a mechanics-lane dependency.

    without_c0 = [lane for lane in order if lane != "G1-C0"]
    eligible, pruned = horizon._dependency_closed_lanes(without_c0)
    assert "G1-C0" not in eligible
    for downstream in ("G1-E1", "G1-R1", "G1-P1", "G1-A1", "G1-S1", "G1-I1"):
        assert downstream not in eligible
        assert downstream in pruned


def test_admission_records_exact_parent_routes_and_all_safety_boundaries(monkeypatch) -> None:
    order = [lane.lane_id for lane in mechanics.LANES]
    parent = {
        "bindings": {
            "program_manifest": {
                "path": "configs/campaign/parent.json",
                "file_sha256": "3" * 64,
                "program_sha256": "4" * 64,
            },
            "supervisor_status": {
                "path": "runs/parent/current_status.json",
                "file_sha256": "5" * 64,
                "status_sha256": "6" * 64,
            },
            "result": {
                "path": "proof/parent.json",
                "file_sha256": "a" * 64,
                "result_sha256": "b" * 64,
            },
            "verification": {
                "path": "proof/parent.verify.json",
                "file_sha256": "c" * 64,
                "verification_sha256": "d" * 64,
            },
            "report_receipt": {
                "path": "runs/parent/report.json",
                "file_sha256": "e" * 64,
                "receipt_sha256": "f" * 64,
            },
            "final_classification": {
                "path": "runs/parent/classifications/h05.json",
                "file_sha256": "1" * 64,
                "classification_sha256": "2" * 64,
            },
        },
        "d1_classification": "stable_candidate_trace",
        "d1_initially_eligible": True,
        "mechanics_predecessor_survivors": order,
        "mechanics_internal_dependencies": horizon._mechanics_dependency_map(),
        "mechanics_dependency_pruned_lanes": [],
        "mechanics_initially_eligible_lanes": order,
    }
    monkeypatch.setattr(horizon, "_validated_parent_state", lambda **_kwargs: parent)
    value = horizon.build_admission()
    assert value["parent_horizon"] == parent["bindings"]
    assert value["d1_initially_eligible"] is True
    assert value["mechanics_predecessor_survivors"] == order
    assert value["mechanics_dependency_pruned_lanes"] == []
    assert value["mechanics_initially_eligible_lanes"] == order
    assert value["activation_allowed"] is False
    assert value["scientific_promotion"] is False
    assert value["independent_scientific_confirmation"] is False


def test_parent_program_status_binds_manifest_capsules_and_current_artifact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(horizon, "REPO_ROOT", tmp_path)
    parent_root = tmp_path / "runs/parent"
    manifest = _sealed(
        {
            "schema": horizon.supervisor.PROGRAM_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
        },
        "program_sha256",
    )
    manifest_path = tmp_path / "configs/campaign/parent.json"
    _write(manifest_path, manifest)
    artifact_paths = (
        tmp_path / "proof/result.json",
        tmp_path / "proof/verification.json",
        parent_root / "report_receipt.json",
        parent_root / "classifications/h05.json",
    )
    for index, path in enumerate(artifact_paths):
        _write(
            path,
            _sealed(
                {
                    "schema": f"test-schema-{index}",
                    "artifact": index,
                },
                "artifact_sha256",
            ),
        )
    expectations = tuple(
        SimpleNamespace(
            path=str(path.relative_to(tmp_path)),
            schema=f"test-schema-{index}",
            fields=(("artifact", index),),
            seal_field="artifact_sha256",
        )
        for index, path in enumerate(artifact_paths)
    )
    capsule = SimpleNamespace(
        capsule_id="parent-complete",
        kind="aggregate",
        priority=1,
        depends_on=(),
        capsule_sha256="7" * 64,
        artifacts=expectations,
    )
    status_path = parent_root / horizon.supervisor.STATUS_FILE
    program = SimpleNamespace(
        path=manifest_path.resolve(),
        file_sha256=horizon.sha256_file(manifest_path),
        program_sha256=manifest["program_sha256"],
        program_id=predecessor.PROGRAM_ID,
        program_root=parent_root.resolve(),
        status_path=status_path.resolve(),
        capsules=(capsule,),
    )
    status = _sealed(
        {
            "schema": horizon.supervisor.STATUS_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "program": {
                "path": str(program.path),
                "file_sha256": program.file_sha256,
                "program_sha256": program.program_sha256,
            },
            "execution_enabled": True,
            "state": "complete",
            "queue_head_sha256": horizon.canonical_sha256(
                {
                    "program_sha256": program.program_sha256,
                    "base_capsules": [capsule.capsule_sha256],
                }
            ),
            "accepted_injection_count": 0,
            "next_injection_sequence": 1,
            "current_capsule": None,
            "lane_reservation": None,
            "problems": [],
            "capsules": {
                capsule.capsule_id: {
                    "id": capsule.capsule_id,
                    "kind": capsule.kind,
                    "priority": capsule.priority,
                    "depends_on": [],
                    "capsule_sha256": capsule.capsule_sha256,
                    "source": "base",
                    "status": "complete",
                    "attempts": 1,
                    "returncode": 0,
                    "last_problem": None,
                    "artifacts": [
                        {
                            "path": expectation.path,
                            "sha256": horizon.sha256_file(path),
                            "schema": expectation.schema,
                            "problems": [],
                            "all_ok": True,
                        }
                        for expectation, path in zip(
                            expectations,
                            artifact_paths,
                            strict=True,
                        )
                    ],
                }
            },
        },
        "status_sha256",
    )
    _write(status_path, status)
    monkeypatch.setattr(predecessor, "PROGRAM_MANIFEST", manifest_path)
    monkeypatch.setattr(
        horizon.supervisor,
        "load_program",
        lambda *_args, **_kwargs: program,
    )
    monkeypatch.setattr(
        horizon.supervisor,
        "read_status",
        lambda _program: status,
    )

    bindings = horizon._validated_parent_program_state(
        parent_root=parent_root,
        required_artifacts=artifact_paths,
    )
    assert set(bindings) == {"program_manifest", "supervisor_status"}
    assert bindings["program_manifest"]["program_sha256"] == manifest["program_sha256"]
    assert bindings["supervisor_status"]["status_sha256"] == status["status_sha256"]

    _write(
        artifact_paths[0],
        _sealed(
            {
                "schema": "test-schema-0",
                "artifact": 999,
            },
            "artifact_sha256",
        ),
    )
    with pytest.raises(ValueError, match="artifact is not clean"):
        horizon._validated_parent_program_state(
            parent_root=parent_root,
            required_artifacts=artifact_paths,
        )


def test_zero_executed_d1_cannot_authorize_a_stable_candidate_h05(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path, verification_path, receipt_path, _ = _fabricated_parent(
        tmp_path,
        monkeypatch,
        d1_classification="stable_candidate_trace",
        d1_continue=True,
    )

    def replay(path: Path) -> dict[str, Any]:
        result = json.loads(path.read_text(encoding="utf-8"))
        assert result["grid"]["executed_d1_rung_count"] == 0
        assert result["classifications"][-1]["epoch_id"] == "H05"
        return {"independent_replay": "rejects-fabricated-stable-candidate"}

    monkeypatch.setattr(predecessor_verify, "build_verification", replay)
    with pytest.raises(ValueError, match="differs from independent raw-artifact replay"):
        horizon._validated_parent_state(
            result_path=result_path,
            verification_path=verification_path,
            report_receipt_path=receipt_path,
        )


def test_resealed_h05_route_change_is_rejected_by_v1_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path, verification_path, receipt_path, h05 = _fabricated_parent(
        tmp_path,
        monkeypatch,
        d1_classification="stable_null",
        d1_continue=False,
    )

    def replay(_path: Path) -> dict[str, Any]:
        assert h05["routing"]["continue_d1"] is False
        raise ValueError("independent D1 classification differs")

    monkeypatch.setattr(predecessor_verify, "build_verification", replay)
    with pytest.raises(ValueError, match="independent D1 classification differs"):
        horizon._validated_parent_state(
            result_path=result_path,
            verification_path=verification_path,
            report_receipt_path=receipt_path,
        )


def test_admission_requires_exact_fields_and_a_utc_timestamp(monkeypatch) -> None:
    order = [lane.lane_id for lane in mechanics.LANES]
    parent = {
        "bindings": {
            "program_manifest": {
                "path": "configs/campaign/parent.json",
                "file_sha256": "1" * 64,
                "program_sha256": "2" * 64,
            },
            "supervisor_status": {
                "path": "runs/parent/current_status.json",
                "file_sha256": "3" * 64,
                "status_sha256": "4" * 64,
            },
            "result": {
                "path": "proof/parent.json",
                "file_sha256": "5" * 64,
                "result_sha256": "6" * 64,
            },
            "verification": {
                "path": "proof/parent.verification.json",
                "file_sha256": "7" * 64,
                "verification_sha256": "8" * 64,
            },
            "report_receipt": {
                "path": "runs/parent/report_receipt.json",
                "file_sha256": "9" * 64,
                "receipt_sha256": "a" * 64,
            },
            "final_classification": {
                "path": "runs/parent/classifications/h05.json",
                "file_sha256": "b" * 64,
                "classification_sha256": "c" * 64,
            },
        },
        "d1_classification": "stable_candidate_trace",
        "d1_initially_eligible": True,
        "mechanics_predecessor_survivors": order,
        "mechanics_internal_dependencies": horizon._mechanics_dependency_map(),
        "mechanics_dependency_pruned_lanes": [],
        "mechanics_initially_eligible_lanes": order,
    }
    monkeypatch.setattr(horizon, "_validated_parent_state", lambda **_kwargs: parent)
    admission = horizon.build_admission()

    extra_core = {key: value for key, value in admission.items() if key != "admission_sha256"}
    extra_core["unexpected"] = True
    with pytest.raises(ValueError, match="field inventory"):
        horizon.validate_admission(_sealed(extra_core, "admission_sha256"))

    offset_core = {key: value for key, value in admission.items() if key != "admission_sha256"}
    offset_core["created_at"] = "2026-07-16T13:00:00+01:00"
    with pytest.raises(ValueError, match="timestamp is not UTC"):
        horizon.validate_admission(_sealed(offset_core, "admission_sha256"))


def test_run_shard_refuses_any_worker_envelope_other_than_exact_twenty_to_one() -> None:
    with pytest.raises(ValueError, match="exact --idle-workers 20 --hawking-workers 1"):
        horizon.run_shard(
            epoch_index=0,
            lane="d1",
            shard_index=0,
            idle_workers=19,
            hawking_workers=1,
        )
    with pytest.raises(ValueError, match="exact --idle-workers 20 --hawking-workers 1"):
        horizon.run_shard(
            epoch_index=0,
            lane="d1",
            shard_index=0,
            idle_workers=20,
            hawking_workers=2,
        )


def test_zero_work_shard_runs_through_the_scoped_v1_engine(
    monkeypatch,
    tmp_path: Path,
) -> None:
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(
            {
                "d1_initially_eligible": False,
                "mechanics_initially_eligible_lanes": [],
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "v2"
    monkeypatch.setattr(predecessor, "REPO_ROOT", tmp_path)
    monkeypatch.setitem(horizon._V1_PATCH, "validate_admission", lambda _value: None)

    before = (
        predecessor.PROGRAM_ID,
        predecessor.SHARD_SCHEMA,
        predecessor.EPOCH_IDS,
        predecessor.EPOCH_CYCLES,
    )
    shard = horizon.run_shard(
        root=root,
        admission_path=admission_path,
        epoch_index=0,
        lane="d1",
        shard_index=0,
    )
    assert shard["schema"] == horizon.SHARD_SCHEMA
    assert shard["program_id"] == horizon.PROGRAM_ID
    assert shard["epoch_id"] == "H06"
    assert shard["cycle_index"] == 7
    assert shard["planned_item_count"] == len(horizon.D1_PARTITIONS[0])
    assert shard["executed_item_count"] == 0
    assert shard["skipped_item_count"] == len(horizon.D1_PARTITIONS[0])
    assert shard["artifact_index"] == []
    assert before == (
        predecessor.PROGRAM_ID,
        predecessor.SHARD_SCHEMA,
        predecessor.EPOCH_IDS,
        predecessor.EPOCH_CYCLES,
    )


def test_report_gate_rejects_a_resealed_widened_verifier_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(horizon, "REPO_ROOT", tmp_path)
    result = _sealed(
        {
            "schema": horizon.RESULT_SCHEMA,
            "program_id": horizon.PROGRAM_ID,
        },
        "result_sha256",
    )
    result_path = tmp_path / "proof/result.json"
    _write(result_path, result)
    verification = _sealed(
        {
            "schema": horizon.VERIFICATION_SCHEMA,
            "program_id": horizon.PROGRAM_ID,
            "claim_scope": "widened to imply scientific confirmation",
            "source": {
                "path": str(result_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(result_path),
                "result_sha256": result["result_sha256"],
            },
            "verification_complete": True,
            "independent_scientific_confirmation": False,
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    with pytest.raises(ValueError):
        horizon._validate_verification_for_report(
            verification,
            result_path=result_path,
        )


def test_report_gate_runs_the_full_verifier_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(horizon, "REPO_ROOT", tmp_path)
    result = _sealed(
        {
            "schema": horizon.RESULT_SCHEMA,
            "program_id": horizon.PROGRAM_ID,
        },
        "result_sha256",
    )
    result_path = tmp_path / "proof/result.json"
    _write(result_path, result)
    verification = _sealed(
        {
            "schema": horizon.VERIFICATION_SCHEMA,
            "program_id": horizon.PROGRAM_ID,
            "claim_scope": horizon.VERIFICATION_CLAIM_SCOPE,
            "source": {
                "path": str(result_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(result_path),
                "result_sha256": result["result_sha256"],
            },
            "checks": {},
            "recomputation": {"fabricated": True},
            "mutation_suite": {"count": 9, "rejected": 9, "all_rejected": True},
            "verification_complete": True,
            "independent_scientific_confirmation": False,
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    with pytest.raises(ValueError):
        horizon._validate_verification_for_report(
            verification,
            result_path=result_path,
        )


def test_existing_report_receipt_must_bind_the_current_arguments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(horizon, "REPO_ROOT", tmp_path)
    result_path = tmp_path / "proof/result.json"
    verification_path = tmp_path / "proof/verification.json"
    report_path = tmp_path / "runs/report.md"
    receipt_path = tmp_path / "runs/report_receipt.json"
    _write(result_path, {"result_sha256": "1" * 64, "classifications": []})
    _write(verification_path, {"verification_sha256": "2" * 64})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("current report\n", encoding="utf-8")
    stale_report = tmp_path / "runs/stale.md"
    stale_report.write_text("stale report\n", encoding="utf-8")
    _write(
        receipt_path,
        {
            "result": {
                "path": str(result_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(result_path),
            },
            "verification": {
                "path": str(verification_path.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(verification_path),
            },
            "report": {
                "path": str(stale_report.relative_to(tmp_path)),
                "file_sha256": horizon.sha256_file(stale_report),
            },
        },
    )
    monkeypatch.setattr(horizon, "validate_result", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        horizon,
        "_validate_verification_for_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        horizon,
        "validate_report_receipt",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ValueError, match="binds different arguments"):
        horizon.render_report(
            result_path=result_path,
            verification_path=verification_path,
            report_path=report_path,
            receipt_path=receipt_path,
        )


def test_nested_cli_dispatches_exact_parent_and_worker_arguments(monkeypatch) -> None:
    cli = _cli()
    calls: list[tuple[str, dict]] = []

    def record(name: str):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return {"operation": name}

        return invoke

    monkeypatch.setattr(cli.horizon, "admit", record("admit"))
    monkeypatch.setattr(cli.horizon, "run_shard", record("run-shard"))
    commands = (
        [
            "admit",
            "--output",
            "a.json",
            "--parent-result",
            "r.json",
            "--parent-verification",
            "v.json",
            "--parent-report-receipt",
            "p.json",
        ],
        [
            "run-shard",
            "--root",
            "root",
            "--admission",
            "a.json",
            "--epoch-index",
            "4",
            "--lane",
            "mechanics",
            "--shard-index",
            "7",
            "--idle-workers",
            "8",
            "--hawking-workers",
            "1",
        ],
    )
    for arguments in commands:
        result = cli.dispatch(cli.build_parser().parse_args(arguments))
        assert result["operation"] == arguments[0]
    assert calls[0][1] == {
        "output": Path("a.json"),
        "parent_result_path": Path("r.json"),
        "parent_verification_path": Path("v.json"),
        "parent_report_receipt_path": Path("p.json"),
    }
    assert calls[1][1]["idle_workers"] == 8
    assert calls[1][1]["hawking_workers"] == 1
