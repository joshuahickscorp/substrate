from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from mop.config import REPO_ROOT
from mop.studies import generation1_full_generations_wave as wave
from mop.studio.generation1_supervisor import load_program, sha256_file
from mop.studio.local_throttle import (
    TASKPOLICY_ADAPTIVE_CAP_GB,
    TASKPOLICY_ADAPTIVE_PREFIX,
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
    load_policy,
)

BUILDER_PATH = "scripts/build_generation1_full_generations_wave_program.py"


def _builder() -> ModuleType:
    path = REPO_ROOT / BUILDER_PATH
    specification = importlib.util.spec_from_file_location(
        "generation1_full_generations_wave_program_builder",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_program_has_exact_123_capsule_serial_gate_and_wave_topology() -> None:
    program = _builder().build_program()
    capsules = program["capsules"]
    by_id = {row["id"]: row for row in capsules}

    assert len(capsules) == len(by_id) == wave.CAPSULE_COUNT == 123
    assert [row["id"] for row in capsules[:5]] == list(wave.GATE_IDS)
    for gate_index, gate_id in enumerate(wave.GATE_IDS):
        expected = [] if gate_index == 0 else [wave.GATE_IDS[gate_index - 1]]
        assert by_id[gate_id]["depends_on"] == expected

    previous = wave.GATE_IDS[-1]
    assert previous == "freeze_routing"
    assert len(wave.EPOCH_IDS) == 14
    for epoch_id in wave.EPOCH_IDS:
        category_capsules = [f"{epoch_id.lower()}_{category_id}" for category_id in wave.CATEGORY_IDS]
        assert len(category_capsules) == 7
        assert all(by_id[capsule_id]["depends_on"] == [previous] for capsule_id in category_capsules)
        classification_id = f"{epoch_id.lower()}_classify"
        assert by_id[classification_id]["depends_on"] == category_capsules
        previous = classification_id

    assert by_id["i1_integrate"]["depends_on"] == ["w21_classify"]
    assert by_id["i1_classify"]["depends_on"] == ["i1_integrate"]
    assert by_id["full_generations_aggregate"]["depends_on"] == ["i1_classify"]
    assert by_id["full_generations_verify"]["depends_on"] == ["full_generations_aggregate"]
    assert by_id["full_generations_report"]["depends_on"] == ["full_generations_verify"]
    assert by_id["full_generations_release_audit"]["depends_on"] == ["full_generations_report"]


def test_compute_capsules_use_dynamic_worker_pool_and_unique_artifacts() -> None:
    builder = _builder()
    capsules = builder.build_program()["capsules"]
    artifact_paths: list[str] = []
    category_ids = {
        f"{epoch_id.lower()}_{category_id}"
        for epoch_id in wave.EPOCH_IDS
        for category_id in wave.CATEGORY_IDS
    }
    compute_ids = {*category_ids, "i1_integrate"}

    for capsule in capsules:
        capsule_id = str(capsule["id"])
        command = tuple(capsule["command"])
        resources = capsule["resources"]
        authorities = {row["path"] for row in capsule["authorities"]}
        assert builder.CLI_PATH in command
        assert builder.CLI_PATH in authorities
        assert resources["process_marker"] == builder.PROCESS_MARKER
        assert f"MOP_REGISTERED_PROCESS_FAMILY={builder.PROCESS_MARKER}" in command
        assert resources["wall_minutes"] < 300
        assert len(capsule["artifacts"]) == 1
        artifact_paths.append(capsule["artifacts"][0]["path"])

        if capsule_id in compute_ids:
            assert command[: len(TASKPOLICY_ADAPTIVE_PREFIX)] == (TASKPOLICY_ADAPTIVE_PREFIX)
            assert resources["cpu_cores"] == wave.IDLE_WORKERS == 16
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_ADAPTIVE_CAP_GB
            if capsule_id == "i1_integrate":
                assert resources["wall_minutes"] == 180
                assert "run-integration" in command
            else:
                assert resources["wall_minutes"] == 285
                assert "run-category" in command
                fields = capsule["artifacts"][0]["fields"]
                assert fields["balanced_planning_shard_count"] == wave.INTERNAL_SHARD_COUNT
                assert "dynamic process pool" in resources["resource_basis"]
                assert "balanced planning shards" in resources["resource_basis"]
        else:
            assert command[: len(TASKPOLICY_COEXISTENCE_PREFIX)] == (TASKPOLICY_COEXISTENCE_PREFIX)
            assert resources["cpu_cores"] == 1
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_COEXISTENCE_CAP_GB

    assert len(artifact_paths) == len(set(artifact_paths)) == wave.CAPSULE_COUNT


def test_commands_bind_taxonomy_new_lane_authority_and_advisory_release_audit() -> None:
    builder = _builder()
    by_id = {row["id"]: row for row in builder.build_program()["capsules"]}

    for epoch_index, epoch_id in enumerate(wave.EPOCH_IDS):
        for category_id in wave.CATEGORY_IDS:
            capsule = by_id[f"{epoch_id.lower()}_{category_id}"]
            command = capsule["command"]
            assert command[command.index("--epoch-index") + 1] == str(epoch_index)
            assert command[command.index("--category") + 1] == category_id
            fields = capsule["artifacts"][0]["fields"]
            assert fields["epoch_id"] == epoch_id
            assert fields["cycle_index"] == wave.EPOCH_CYCLES[epoch_index]
            assert fields["category.id"] == category_id

    admission = by_id["admit_wave_v1"]
    admission_command = admission["command"]
    assert admission_command[admission_command.index("--parent-result") + 1] == (builder.PARENT_RESULT_PATH)
    assert admission_command[admission_command.index("--parent-verification") + 1] == (
        builder.PARENT_VERIFICATION_PATH
    )
    assert admission_command[admission_command.index("--parent-report-receipt") + 1] == (
        builder.PARENT_REPORT_RECEIPT_PATH
    )
    for gate_id in ("carry_d1_freeze", "rescreen_d1_redesign", "admit_new_lanes", "freeze_routing"):
        assert "--parent-result" not in by_id[gate_id]["command"]
        assert "--d1-rung-root" not in by_id[gate_id]["command"]

    authority_paths = {row["path"] for row in admission["authorities"]}
    assert len(builder.NEW_MECHANISM_AUTHORITIES) == 12
    for path in builder.NEW_MECHANISM_AUTHORITIES:
        assert path in authority_paths
    # The construction lane runs on the proven vectorized runner; both vectorized modules are pinned.
    assert len(builder.CONSTRUCTION_VEC_AUTHORITIES) == 2
    for path in builder.CONSTRUCTION_VEC_AUTHORITIES:
        assert path in authority_paths
    assert "src/mop/mechanisms/construction_search_vec_impl.py" in authority_paths
    assert "src/mop/mechanisms/construction_search_vec_runner.py" in authority_paths
    assert "src/mop/studies/generation1_new_mechanisms_queue.py" in authority_paths
    assert "src/mop/studies/generation1_successor_mechanics_queue.py" in authority_paths
    assert "src/mop/ladder/stage3_successor_registry.py" in authority_paths
    assert "src/mop/ladder/stage3_registry.py" in authority_paths
    assert "src/mop/ladder/ladder_contracts.py" in authority_paths
    assert builder.WAVE_V1_MANIFEST_PATH in authority_paths
    assert "scripts/generation1_successor_horizon_v2/mop_generation1_successor_horizon.py" in authority_paths

    release_audit = by_id["full_generations_release_audit"]
    assert release_audit["kind"] == "aggregate"
    assert release_audit["depends_on"] == ["full_generations_report"]
    assert "release-audit" in release_audit["command"]
    audit_command = tuple(release_audit["command"])
    assert audit_command[: len(TASKPOLICY_COEXISTENCE_PREFIX)] == (TASKPOLICY_COEXISTENCE_PREFIX)
    assert release_audit["resources"]["wall_minutes"] == 60
    audit_artifact = release_audit["artifacts"][0]
    assert audit_artifact["schema"] == wave.RELEASE_AUDIT_SCHEMA
    assert audit_artifact["seal_field"] == "release_audit_sha256"
    assert audit_artifact["fields"]["advisory"] is True
    assert audit_artifact["fields"]["activation_allowed"] is False
    assert audit_artifact["fields"]["scientific_promotion"] is False
    assert audit_artifact["fields"]["independent_scientific_confirmation"] is False


def test_program_envelope_and_generated_manifest_are_exact_and_deterministic() -> None:
    builder = _builder()
    expected = builder.build_program()
    generated = json.loads(wave.PROGRAM_MANIFEST.read_text(encoding="utf-8"))

    # Envelope scalars are recomputed from the wave module functions, never hard-coded literals.
    expected_seconds = wave.planned_program_compute_seconds()
    assert wave.planned_serial_hours() == pytest.approx(expected_seconds / 3_600)
    assert wave.planned_ideal_worker_hours() == pytest.approx(
        expected_seconds / wave.IDLE_WORKERS / 3_600
    )
    by_id = {row["id"]: row for row in expected["capsules"]}
    aggregate_fields = by_id["full_generations_aggregate"]["artifacts"][0]["fields"]
    assert aggregate_fields["execution.maximum_planned_serial_seconds"] == expected_seconds
    assert aggregate_fields["grid.maximum_raw_receipt_count"] == wave.MAXIMUM_RAW_RECEIPT_COUNT
    assert aggregate_fields["grid.manifest_capsule_count"] == wave.CAPSULE_COUNT == len(expected["capsules"])
    verify_fields = by_id["full_generations_verify"]["artifacts"][0]["fields"]
    assert verify_fields["mutation_suite.count"] == verify_fields["mutation_suite.rejected"]
    assert verify_fields["recomputation.manifest_capsule_count"] == wave.CAPSULE_COUNT

    assert generated == expected
    assert wave.PROGRAM_MANIFEST.stat().st_mode & 0o777 == 0o644
    loaded = load_program(wave.PROGRAM_MANIFEST)
    assert loaded.program_id == wave.PROGRAM_ID
    assert loaded.program_sha256 == expected["program_sha256"]
    assert len(loaded.capsules) == wave.CAPSULE_COUNT

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(REPO_ROOT / "src"),
                str(REPO_ROOT),
                environment.get("PYTHONPATH"),
            ),
        )
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / BUILDER_PATH),
            "--check",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


def test_manifest_binds_v6_full_generations_policy_with_retained_markers() -> None:
    builder = _builder()
    program = builder.build_program()

    policy_path = "configs/local_execution_throttle_v6_full_generations.yaml"
    assert policy_path == builder.POLICY_PATH
    assert program["policy"] == {
        "path": policy_path,
        "sha256": sha256_file(REPO_ROOT / policy_path),
    }

    policy = load_policy(REPO_ROOT / policy_path)
    known_markers = set(policy.monitor["known_heavy_markers"])
    # The wave's process marker arms the wall/marker admission gate.
    assert builder.PROCESS_MARKER in known_markers
    # The Hawking markers are retained verbatim so backoff-to-zero re-arms when Hawking reappears.
    assert "build/strand-block-parallel/release/quantize-model-block-parallel" in known_markers
    assert any("doctor_v5" in marker for marker in known_markers)
    # Idle-host-only invariants: the loader forces two active lanes and one heavy lane.
    assert int(policy.limits["maximum_active_lanes"]) == 2
    assert int(policy.limits["maximum_heavy_lanes"]) == 1

    wave_tasks = {tid: task for tid, task in policy.tasks.items() if tid.startswith("full_generations_")}
    assert len(wave_tasks) == len(wave.CATEGORY_IDS) + 1  # seven categories plus the I1 integration
    for task in wave_tasks.values():
        assert task.lane == "cpu"
        assert task.cpu_cores == wave.IDLE_WORKERS == 16
        assert task.estimated_unified_memory_gb is not None
        assert task.estimated_unified_memory_gb <= 16.0
