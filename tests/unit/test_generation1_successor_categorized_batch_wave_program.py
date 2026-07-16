from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from types import ModuleType

import pytest

from mop.config import REPO_ROOT
from mop.studies import generation1_successor_categorized_batch_wave as wave
from mop.studio.generation1_supervisor import load_program
from mop.studio.local_throttle import (
    TASKPOLICY_ADAPTIVE_CAP_GB,
    TASKPOLICY_ADAPTIVE_PREFIX,
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
)


def _builder() -> ModuleType:
    path = REPO_ROOT / "scripts/build_generation1_successor_categorized_batch_wave_program.py"
    specification = importlib.util.spec_from_file_location(
        "generation1_successor_categorized_batch_wave_program_builder",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_program_has_exact_59_capsule_serial_gate_and_wave_topology() -> None:
    program = _builder().build_program()
    capsules = program["capsules"]
    by_id = {row["id"]: row for row in capsules}

    assert len(capsules) == len(by_id) == wave.CAPSULE_COUNT == 59
    assert [row["id"] for row in capsules[:5]] == list(wave.GATE_IDS)
    for gate_index, gate_id in enumerate(wave.GATE_IDS):
        expected = [] if gate_index == 0 else [wave.GATE_IDS[gate_index - 1]]
        assert by_id[gate_id]["depends_on"] == expected

    previous = "freeze_d1_redesign_v2"
    for epoch_id in wave.EPOCH_IDS:
        category_capsules = [f"{epoch_id.lower()}_{category_id}" for category_id in wave.CATEGORY_IDS]
        assert all(by_id[capsule_id]["depends_on"] == [previous] for capsule_id in category_capsules)
        classification_id = f"{epoch_id.lower()}_classify"
        assert by_id[classification_id]["depends_on"] == category_capsules
        previous = classification_id

    assert by_id["i1_integrate"]["depends_on"] == ["w07_classify"]
    assert by_id["i1_classify"]["depends_on"] == ["i1_integrate"]
    assert by_id["categorized_wave_aggregate"]["depends_on"] == ["i1_classify"]
    assert by_id["categorized_wave_verify"]["depends_on"] == ["categorized_wave_aggregate"]
    assert by_id["categorized_wave_report"]["depends_on"] == ["categorized_wave_verify"]


def test_compute_capsules_use_dynamic_eight_worker_pool_and_unique_artifacts() -> None:
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
            assert resources["cpu_cores"] == wave.IDLE_WORKERS
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


def test_category_commands_bind_exact_taxonomy_cycles_and_d1_replay_authority() -> None:
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

    d1_gate = by_id["verify_old_d1"]
    assert d1_gate["command"][d1_gate["command"].index("--d1-rung-root") + 1] == (builder.D1_RUNG_ROOT_PATH)
    authority_paths = {row["path"] for row in d1_gate["authorities"]}
    assert "src/mop/studies/generation1_d1_frozen_verify.py" in authority_paths
    assert "src/mop/studies/generation1_d1_redesign_v2.py" in authority_paths
    assert "src/mop/studies/generation1_consolidated_final_campaign.py" in authority_paths
    assert "src/mop/studies/generation1_successor_mechanics_queue.py" in authority_paths


def test_program_envelope_and_generated_manifest_are_exact_and_deterministic() -> None:
    builder = _builder()
    expected = builder.build_program()
    generated = json.loads(wave.PROGRAM_MANIFEST.read_text(encoding="utf-8"))

    assert wave.planned_program_compute_seconds() == pytest.approx(706_256.9000881652)
    assert wave.planned_serial_hours() == pytest.approx(196.18247224671256)
    assert wave.planned_ideal_eight_worker_hours() == pytest.approx(24.52280903083907)
    assert wave.MAXIMUM_RAW_RECEIPT_COUNT == 15_886
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
            str(REPO_ROOT / "scripts/build_generation1_successor_categorized_batch_wave_program.py"),
            "--check",
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
