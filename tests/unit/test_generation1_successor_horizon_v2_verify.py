from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studies import generation1_successor_horizon as predecessor
from mop.studies import generation1_successor_horizon_v2 as horizon
from mop.studies import generation1_successor_horizon_v2_verify as verifier
from mop.studies import generation1_successor_horizon_verify as predecessor_verify
from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio import generation1_supervisor


def _sealed(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: verifier.canonical_sha256(core)}


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _binding(
    root: Path,
    path: Path,
    value: dict[str, Any],
    seal_field: str,
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "file_sha256": verifier.sha256_file(path),
        seal_field: value[seal_field],
    }


def _admission_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)

    lane_order = [lane.lane_id for lane in mechanics.LANES]
    predecessor_survivors = [lane_id for lane_id in lane_order if lane_id != "G1-P1"]
    eligible, dependency_pruned = verifier._dependency_closed_lanes(predecessor_survivors)
    assert dependency_pruned == ["G1-I1"]

    h04_sha256 = "4" * 64
    h05_core = {
        "schema": predecessor.CLASSIFICATION_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "claim_scope": predecessor.CLAIM_SCOPE,
        "epoch_id": "H05",
        "epoch_index": 4,
        "cycle_index": 6,
        "parent_classification_sha256": h04_sha256,
        "d1": {
            "classification": "stable_candidate_trace",
            "continue_d1": True,
        },
        "routing": {
            "continue_d1": True,
            "mechanics_lanes_for_next_epoch": predecessor_survivors,
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    h05 = _sealed(h05_core, "classification_sha256")
    h05_path = tmp_path / "runs/parent/classifications/h05.json"
    _write(h05_path, h05)

    classification_rows = [
        {
            "epoch_id": epoch_id,
            "cycle_index": cycle,
            "path": f"runs/parent/classifications/{epoch_id.lower()}.json",
            "file_sha256": str(index) * 64,
            "classification_sha256": (h04_sha256 if epoch_id == "H04" else str(index + 1) * 64),
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
            "file_sha256": verifier.sha256_file(h05_path),
            "classification_sha256": h05["classification_sha256"],
        }
    )
    parent_result_core = {
        "schema": predecessor.RESULT_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "claim_scope": predecessor.CLAIM_SCOPE,
        "grid": {"epoch_count": len(predecessor.EPOCH_IDS)},
        "decision": {"independent_scientific_confirmation": False},
        "classifications": classification_rows,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    parent_result = _sealed(parent_result_core, "result_sha256")
    parent_result_path = tmp_path / "proof/PARENT_RESULT.json"
    _write(parent_result_path, parent_result)

    parent_verification_core = {
        "schema": verifier._PARENT_VERIFICATION_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "claim_scope": "independent artifact verification",
        "source": {
            "path": str(parent_result_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(parent_result_path),
            "result_sha256": parent_result["result_sha256"],
        },
        "checks": {
            "result_seal_valid": True,
            "admission_and_consolidated_authority_valid": True,
            "all_shards_and_raw_artifacts_valid": True,
            "classifications_independently_reproduced": True,
            "all_seed_intervals_disjoint": True,
            "mutation_suite_passed": True,
            "independent_generator_family_present": False,
        },
        "recomputation": {
            "all_seed_intervals_disjoint": True,
            "bound_shard_count": len(predecessor.EPOCH_IDS)
            * (predecessor.D1_SHARD_COUNT + predecessor.MECHANICS_SHARD_COUNT),
        },
        "mutation_suite": {"count": 9, "rejected": 9, "all_rejected": True},
        "verification_complete": True,
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    parent_verification = _sealed(
        parent_verification_core,
        "verification_sha256",
    )
    parent_verification_path = tmp_path / "proof/PARENT_RESULT.verification.json"
    _write(parent_verification_path, parent_verification)

    report_path = tmp_path / "runs/parent/report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("bounded predecessor report\n", encoding="utf-8")
    report_receipt_core = {
        "schema": predecessor.REPORT_RECEIPT_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "result": {
            "path": str(parent_result_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(parent_result_path),
        },
        "verification": {
            "path": str(parent_verification_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(parent_verification_path),
        },
        "report": {
            "path": str(report_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(report_path),
        },
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    report_receipt = _sealed(report_receipt_core, "receipt_sha256")
    report_receipt_path = tmp_path / "runs/parent/report_receipt.json"
    _write(report_receipt_path, report_receipt)

    program_manifest_core = {
        "schema": generation1_supervisor.PROGRAM_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "program_root": "runs/parent",
    }
    program_manifest = _sealed(program_manifest_core, "program_sha256")
    program_manifest_path = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    _write(program_manifest_path, program_manifest)
    supervisor_status_core = {
        "schema": generation1_supervisor.STATUS_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "execution_enabled": True,
        "state": "complete",
        "problems": [],
    }
    supervisor_status = _sealed(supervisor_status_core, "status_sha256")
    supervisor_status_path = tmp_path / "runs/parent/current_status.json"
    _write(supervisor_status_path, supervisor_status)
    program_binding = _binding(
        tmp_path,
        program_manifest_path,
        program_manifest,
        "program_sha256",
    )
    status_binding = _binding(
        tmp_path,
        supervisor_status_path,
        supervisor_status,
        "status_sha256",
    )
    expected_parent_artifacts = {
        "result": parent_result_path,
        "verification": parent_verification_path,
        "report_receipt": report_receipt_path,
        "final_classification": h05_path,
    }

    def validate_execution_authority(
        observed_program: Any,
        observed_status: Any,
        *,
        expected_parent_root: Path,
    ) -> dict[str, Path]:
        assert expected_parent_root == h05_path.parent.parent
        if dict(observed_program) != program_binding or dict(observed_status) != status_binding:
            raise ValueError("fixture predecessor execution authority drifted")
        return expected_parent_artifacts

    monkeypatch.setattr(
        verifier,
        "_validate_parent_execution_authority",
        validate_execution_authority,
    )
    monkeypatch.setattr(
        predecessor_verify,
        "build_verification",
        lambda _path: copy.deepcopy(parent_verification),
    )
    monkeypatch.setattr(
        predecessor_verify,
        "validate_verification",
        lambda _value: None,
    )

    admission_core = {
        "schema": horizon.ADMISSION_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": horizon.CLAIM_SCOPE,
        "created_at": "2026-07-16T12:00:00+00:00",
        "parent_horizon": {
            "program_manifest": program_binding,
            "supervisor_status": status_binding,
            "result": _binding(
                tmp_path,
                parent_result_path,
                parent_result,
                "result_sha256",
            ),
            "verification": _binding(
                tmp_path,
                parent_verification_path,
                parent_verification,
                "verification_sha256",
            ),
            "report_receipt": _binding(
                tmp_path,
                report_receipt_path,
                report_receipt,
                "receipt_sha256",
            ),
            "final_classification": _binding(
                tmp_path,
                h05_path,
                h05,
                "classification_sha256",
            ),
        },
        "epoch_ids": list(horizon.EPOCH_IDS),
        "fresh_cycle_indices": list(horizon.EPOCH_CYCLES),
        "d1_predecessor_classification": "stable_candidate_trace",
        "d1_initially_eligible": True,
        "mechanics_predecessor_survivors": predecessor_survivors,
        "mechanics_internal_dependencies": verifier._mechanics_dependency_map(),
        "mechanics_dependency_pruned_lanes": dependency_pruned,
        "mechanics_initially_eligible_lanes": eligible,
        "boundary_rules": verifier._boundary_rules(),
        "planned_compute": verifier._planned_compute(),
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }
    admission = _sealed(admission_core, "admission_sha256")
    admission_path = tmp_path / "runs/v2/admission.json"
    _write(admission_path, admission)

    result_core = {
        "schema": horizon.RESULT_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": horizon.CLAIM_SCOPE,
        "admission": {
            "path": str(admission_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(admission_path),
            "admission_sha256": admission["admission_sha256"],
        },
        "grid": {"epoch_count": len(horizon.EPOCH_IDS)},
        "decision": {"independent_scientific_confirmation": False},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = _sealed(result_core, "result_sha256")
    return admission, result, admission_path


def _rewrite_admission_binding(
    admission: dict[str, Any],
    result: dict[str, Any],
    admission_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    admission_core = {key: value for key, value in admission.items() if key != "admission_sha256"}
    admission = _sealed(admission_core, "admission_sha256")
    _write(admission_path, admission)
    result_core = {key: value for key, value in result.items() if key != "result_sha256"}
    result_core["admission"] = {
        **result_core["admission"],
        "file_sha256": verifier.sha256_file(admission_path),
        "admission_sha256": admission["admission_sha256"],
    }
    return admission, _sealed(result_core, "result_sha256")


def test_v2_verifier_uses_and_restores_the_locked_v1_engine_scope() -> None:
    before = {
        "program_id": predecessor.PROGRAM_ID,
        "epochs": predecessor.EPOCH_IDS,
        "cycles": predecessor.EPOCH_CYCLES,
    }
    with verifier._v1_verifier_scope():
        assert predecessor.PROGRAM_ID == horizon.PROGRAM_ID
        assert predecessor.EPOCH_IDS == horizon.EPOCH_IDS
        assert predecessor.EPOCH_CYCLES == horizon.EPOCH_CYCLES
    assert before["program_id"] == predecessor.PROGRAM_ID
    assert before["epochs"] == predecessor.EPOCH_IDS
    assert before["cycles"] == predecessor.EPOCH_CYCLES


def test_parent_program_and_clean_terminal_status_bind_exact_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    manifest_path = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    program_root = tmp_path / "runs/generation1/generation1-successor-horizon-v1"
    manifest = _sealed(
        {
            "schema": generation1_supervisor.PROGRAM_SCHEMA,
            "program_id": predecessor.PROGRAM_ID,
            "program_root": str(program_root.relative_to(tmp_path)),
        },
        "program_sha256",
    )
    _write(manifest_path, manifest)

    capsule_ids = {
        "result": "g1_horizon_aggregate",
        "verification": "g1_horizon_verify",
        "report_receipt": "g1_horizon_report",
        "final_classification": "g1_h05_classify",
    }
    capsules = []
    artifact_paths: dict[str, Path] = {}
    for index, (label, capsule_id) in enumerate(capsule_ids.items()):
        path = tmp_path / f"runs/parent/{label}.json"
        payload = _sealed(
            {
                "schema": f"fixture/{label}",
                "complete": True,
            },
            "payload_sha256",
        )
        _write(path, payload)
        artifact_paths[label] = path
        expectation = SimpleNamespace(
            path=str(path.relative_to(tmp_path)),
            schema=payload["schema"],
            fields=(("complete", True),),
            seal_field="payload_sha256",
        )
        capsules.append(
            SimpleNamespace(
                capsule_id=capsule_id,
                kind="aggregate",
                priority=700 + index,
                depends_on=(),
                capsule_sha256=str(index + 1) * 64,
                artifacts=(expectation,),
            )
        )

    status_path = program_root / generation1_supervisor.STATUS_FILE
    program = SimpleNamespace(
        path=manifest_path,
        file_sha256=verifier.sha256_file(manifest_path),
        program_sha256=manifest["program_sha256"],
        program_id=predecessor.PROGRAM_ID,
        repo_root=tmp_path,
        program_root=program_root,
        status_path=status_path,
        capsules=tuple(capsules),
    )
    status_core = {
        "schema": generation1_supervisor.STATUS_SCHEMA,
        "program_id": predecessor.PROGRAM_ID,
        "program": {
            "path": str(manifest_path),
            "file_sha256": program.file_sha256,
            "program_sha256": program.program_sha256,
        },
        "execution_enabled": True,
        "state": "complete",
        "queue_head_sha256": verifier.canonical_sha256(
            {
                "program_sha256": program.program_sha256,
                "base_capsules": [capsule.capsule_sha256 for capsule in program.capsules],
            }
        ),
        "next_injection_sequence": 1,
        "accepted_injection_count": 0,
        "current_capsule": None,
        "capsules": {
            capsule.capsule_id: {
                "id": capsule.capsule_id,
                "kind": capsule.kind,
                "priority": capsule.priority,
                "depends_on": list(capsule.depends_on),
                "capsule_sha256": capsule.capsule_sha256,
                "source": "base",
                "status": "complete",
                "attempts": 1,
                "returncode": 0,
                "last_problem": None,
                "artifacts": [
                    verifier._artifact_report(program, expectation) for expectation in capsule.artifacts
                ],
            }
            for capsule in program.capsules
        },
        "last_admission": None,
        "lane_reservation": None,
        "problems": [],
    }
    status = _sealed(status_core, "status_sha256")
    _write(status_path, status)

    monkeypatch.setattr(
        generation1_supervisor,
        "load_program",
        lambda path, *, repo_root: (
            program
            if Path(path) == manifest_path and Path(repo_root) == tmp_path
            else (_ for _ in ()).throw(ValueError("unexpected program"))
        ),
    )
    monkeypatch.setattr(
        generation1_supervisor,
        "read_status",
        lambda observed: (
            status if observed is program else (_ for _ in ()).throw(ValueError("unexpected program"))
        ),
    )
    paths = verifier._validate_parent_execution_authority(
        _binding(
            tmp_path,
            manifest_path,
            manifest,
            "program_sha256",
        ),
        _binding(
            tmp_path,
            status_path,
            status,
            "status_sha256",
        ),
        expected_parent_root=program_root,
    )
    assert paths == artifact_paths

    artifact_paths["result"].write_text(
        '{"schema":"fixture/result","complete":false}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="terminal capsule status drifted"):
        verifier._validate_parent_execution_authority(
            _binding(
                tmp_path,
                manifest_path,
                manifest,
                "program_sha256",
            ),
            _binding(
                tmp_path,
                status_path,
                status,
                "status_sha256",
            ),
            expected_parent_root=program_root,
        )


def test_independent_admission_validation_binds_h05_and_dependency_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, result, _ = _admission_fixture(tmp_path, monkeypatch)
    verifier._validate_admission(admission, result)
    assert "G1-P1" not in admission["mechanics_predecessor_survivors"]
    assert "G1-I1" in admission["mechanics_predecessor_survivors"]
    assert admission["mechanics_dependency_pruned_lanes"] == ["G1-I1"]
    assert "G1-I1" not in admission["mechanics_initially_eligible_lanes"]


@pytest.mark.parametrize(
    "mutation",
    (
        "resurrect_dependency_blocked_lane",
        "delete_dependency_edge",
        "erase_dependency_prune",
        "widen_d1_route",
        "change_h05_binding",
    ),
)
def test_resealed_admission_mutations_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    admission, result, admission_path = _admission_fixture(tmp_path, monkeypatch)
    candidate = copy.deepcopy(admission)
    if mutation == "resurrect_dependency_blocked_lane":
        candidate["mechanics_initially_eligible_lanes"].append("G1-I1")
    elif mutation == "delete_dependency_edge":
        candidate["mechanics_internal_dependencies"]["G1-I1"].remove("G1-P1")
    elif mutation == "erase_dependency_prune":
        candidate["mechanics_dependency_pruned_lanes"] = []
    elif mutation == "widen_d1_route":
        candidate["d1_initially_eligible"] = False
    else:
        candidate["parent_horizon"]["final_classification"]["classification_sha256"] = "0" * 64
    candidate, rebound_result = _rewrite_admission_binding(
        candidate,
        result,
        admission_path,
    )
    with pytest.raises(ValueError):
        verifier._validate_admission(candidate, rebound_result)


def test_coherently_resealed_parent_route_fabrication_is_rejected_by_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    admission, result, admission_path = _admission_fixture(tmp_path, monkeypatch)
    h05_path = tmp_path / admission["parent_horizon"]["final_classification"]["path"]
    parent_result_path = tmp_path / admission["parent_horizon"]["result"]["path"]
    parent_verification_path = tmp_path / admission["parent_horizon"]["verification"]["path"]
    report_receipt_path = tmp_path / admission["parent_horizon"]["report_receipt"]["path"]

    h05 = json.loads(h05_path.read_text(encoding="utf-8"))
    h05_core = {key: value for key, value in h05.items() if key != "classification_sha256"}
    h05_core["d1"] = {
        **h05_core["d1"],
        "classification": "stable_null",
        "continue_d1": False,
    }
    h05_core["routing"] = {
        **h05_core["routing"],
        "continue_d1": False,
    }
    h05 = _sealed(h05_core, "classification_sha256")
    _write(h05_path, h05)

    parent_result = json.loads(parent_result_path.read_text(encoding="utf-8"))
    parent_result_core = {key: value for key, value in parent_result.items() if key != "result_sha256"}
    parent_result_core["classifications"][-1] = {
        **parent_result_core["classifications"][-1],
        "file_sha256": verifier.sha256_file(h05_path),
        "classification_sha256": h05["classification_sha256"],
    }
    parent_result = _sealed(parent_result_core, "result_sha256")
    _write(parent_result_path, parent_result)

    parent_verification = json.loads(parent_verification_path.read_text(encoding="utf-8"))
    verification_core = {
        key: value for key, value in parent_verification.items() if key != "verification_sha256"
    }
    verification_core["source"] = {
        **verification_core["source"],
        "file_sha256": verifier.sha256_file(parent_result_path),
        "result_sha256": parent_result["result_sha256"],
    }
    parent_verification = _sealed(verification_core, "verification_sha256")
    _write(parent_verification_path, parent_verification)

    report_receipt = json.loads(report_receipt_path.read_text(encoding="utf-8"))
    report_core = {key: value for key, value in report_receipt.items() if key != "receipt_sha256"}
    report_core["result"] = {
        **report_core["result"],
        "file_sha256": verifier.sha256_file(parent_result_path),
    }
    report_core["verification"] = {
        **report_core["verification"],
        "file_sha256": verifier.sha256_file(parent_verification_path),
    }
    report_receipt = _sealed(report_core, "receipt_sha256")
    _write(report_receipt_path, report_receipt)

    candidate = copy.deepcopy(admission)
    candidate["parent_horizon"]["result"] = _binding(
        tmp_path,
        parent_result_path,
        parent_result,
        "result_sha256",
    )
    candidate["parent_horizon"]["verification"] = _binding(
        tmp_path,
        parent_verification_path,
        parent_verification,
        "verification_sha256",
    )
    candidate["parent_horizon"]["report_receipt"] = _binding(
        tmp_path,
        report_receipt_path,
        report_receipt,
        "receipt_sha256",
    )
    candidate["parent_horizon"]["final_classification"] = _binding(
        tmp_path,
        h05_path,
        h05,
        "classification_sha256",
    )
    candidate["d1_predecessor_classification"] = "stable_null"
    candidate["d1_initially_eligible"] = False
    candidate, rebound_result = _rewrite_admission_binding(
        candidate,
        result,
        admission_path,
    )

    with pytest.raises(
        ValueError,
        match="differs from independent recomputation",
    ):
        verifier._validate_admission(candidate, rebound_result)


def test_v2_result_mutation_suite_matches_the_independent_v1_engine() -> None:
    core = {
        "schema": horizon.RESULT_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": horizon.CLAIM_SCOPE,
        "grid": {"epoch_count": len(horizon.EPOCH_IDS)},
        "decision": {"independent_scientific_confirmation": False},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = _sealed(core, "result_sha256")
    assert verifier._mutation_suite(result) == {
        "count": 9,
        "rejected": 9,
        "all_rejected": True,
    }


def test_seed_formulas_are_disjoint_across_consolidated_v1_and_v2_cycles() -> None:
    assert verifier._all_cycle_seed_spaces_disjoint() is True


@pytest.mark.parametrize("field", ("claim_scope", "cycle_index"))
def test_v2_shard_prescan_rejects_resealed_claim_and_cycle_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    bindings: list[dict[str, Any]] = []
    shard_values: dict[tuple[str, str, int], tuple[Path, dict[str, Any]]] = {}
    for epoch_index, epoch_id in enumerate(horizon.EPOCH_IDS):
        for lane, count in (
            ("d1", horizon.D1_SHARD_COUNT),
            ("mechanics", horizon.MECHANICS_SHARD_COUNT),
        ):
            for shard_index in range(count):
                core = {
                    "schema": horizon.SHARD_SCHEMA,
                    "program_id": horizon.PROGRAM_ID,
                    "claim_scope": horizon.CLAIM_SCOPE,
                    "epoch_id": epoch_id,
                    "cycle_index": horizon.EPOCH_CYCLES[epoch_index],
                    "lane": lane,
                    "shard_index": shard_index,
                    "complete": True,
                    "problems": [],
                    "activation_allowed": False,
                    "scientific_promotion": False,
                    "independent_scientific_confirmation": False,
                }
                shard = _sealed(core, "shard_sha256")
                path = tmp_path / "runs/v2/shards" / epoch_id.lower() / f"{lane}_{shard_index:02d}.json"
                _write(path, shard)
                bindings.append(
                    _binding(tmp_path, path, shard, "shard_sha256")
                    | {
                        "epoch_id": epoch_id,
                        "lane": lane,
                        "shard_index": shard_index,
                    }
                )
                shard_values[(epoch_id, lane, shard_index)] = (path, shard)
    result = {"shard_index": bindings}
    verifier._validate_v2_shard_boundaries(result)

    path, shard = shard_values[(horizon.EPOCH_IDS[0], "d1", 0)]
    mutated_core = {key: value for key, value in shard.items() if key != "shard_sha256"}
    mutated_core[field] = "widened same-code claim" if field == "claim_scope" else horizon.EPOCH_CYCLES[0] + 1
    mutated = _sealed(mutated_core, "shard_sha256")
    _write(path, mutated)
    result["shard_index"][0] = {
        **result["shard_index"][0],
        "file_sha256": verifier.sha256_file(path),
        "shard_sha256": mutated["shard_sha256"],
    }
    with pytest.raises(ValueError, match="claim, cycle"):
        verifier._validate_v2_shard_boundaries(result)


def test_verification_shell_binds_exact_v2_result_and_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(verifier, "REPO_ROOT", tmp_path)
    result_core = {
        "schema": horizon.RESULT_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": horizon.CLAIM_SCOPE,
        "grid": {"epoch_count": len(horizon.EPOCH_IDS)},
        "decision": {"independent_scientific_confirmation": False},
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    result = _sealed(result_core, "result_sha256")
    result_path = tmp_path / "proof/GENERATION1_SUCCESSOR_HORIZON_V2.json"
    _write(result_path, result)
    checks = {
        "result_seal_valid": True,
        "predecessor_authority_chain_valid": True,
        "dependency_closed_admission_valid": True,
        "all_shards_and_raw_artifacts_valid": True,
        "classifications_independently_reproduced": True,
        "all_seed_intervals_disjoint": True,
        "predecessor_and_v2_seed_spaces_disjoint": True,
        "mutation_suite_passed": True,
        "independent_generator_family_present": False,
    }
    verification_core = {
        "schema": verifier.VERIFICATION_SCHEMA,
        "program_id": horizon.PROGRAM_ID,
        "claim_scope": verifier.CLAIM_SCOPE,
        "source": {
            "path": str(result_path.relative_to(tmp_path)),
            "file_sha256": verifier.sha256_file(result_path),
            "result_sha256": result["result_sha256"],
        },
        "checks": checks,
        "recomputation": {
            "d1_classifications": {epoch_id: "not_run_pruned" for epoch_id in horizon.EPOCH_IDS},
            "mechanics_lanes_retained": {epoch_id: [] for epoch_id in horizon.EPOCH_IDS},
            "d1_interval_count": 0,
            "mechanics_interval_count": 0,
            "all_seed_intervals_disjoint": True,
            "bound_shard_count": len(horizon.EPOCH_IDS)
            * (horizon.D1_SHARD_COUNT + horizon.MECHANICS_SHARD_COUNT),
            "executed_d1_rung_count": 0,
            "executed_mechanics_rung_count": 0,
        },
        "mutation_suite": {"count": 9, "rejected": 9, "all_rejected": True},
        "verification_complete": True,
        "independent_scientific_confirmation": False,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    verification = _sealed(verification_core, "verification_sha256")
    monkeypatch.setattr(
        verifier,
        "build_verification",
        lambda _path: copy.deepcopy(verification),
    )
    verifier.validate_verification(verification)

    promoted_core = {key: value for key, value in verification.items() if key != "verification_sha256"}
    promoted_core["scientific_promotion"] = True
    promoted = _sealed(promoted_core, "verification_sha256")
    with pytest.raises(ValueError):
        verifier.validate_verification(promoted)

    widened_core = {key: value for key, value in verification.items() if key != "verification_sha256"}
    widened_core["claim_scope"] = verifier.CLAIM_SCOPE + "; activation candidate"
    widened = _sealed(widened_core, "verification_sha256")
    with pytest.raises(ValueError):
        verifier.validate_verification(widened)

    recomputation_core = {key: value for key, value in verification.items() if key != "verification_sha256"}
    recomputation_core["recomputation"] = copy.deepcopy(recomputation_core["recomputation"])
    recomputation_core["recomputation"]["d1_classifications"]["H06"] = "stable_null"
    recomputation_mutation = _sealed(
        recomputation_core,
        "verification_sha256",
    )
    with pytest.raises(ValueError, match="differs from independent replay"):
        verifier.validate_verification(recomputation_mutation)
