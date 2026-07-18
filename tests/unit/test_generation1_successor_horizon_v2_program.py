from __future__ import annotations

import importlib.util
import json
from types import ModuleType

from mop.config import REPO_ROOT
from mop.studies import generation1_successor_horizon_v2 as horizon
from mop.studies import generation1_successor_horizon_v2_verify as horizon_verifier
from mop.studio.generation1_supervisor import load_program
from mop.studio.local_throttle import (
    TASKPOLICY_ADAPTIVE_CAP_GB,
    TASKPOLICY_ADAPTIVE_PREFIX,
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
)


def _builder() -> ModuleType:
    path = REPO_ROOT / "scripts/build_generation1_successor_horizon_v2_program.py"
    specification = importlib.util.spec_from_file_location(
        "generation1_successor_horizon_v2_program_builder",
        path,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_program_reuses_scheduler_local_ids_for_five_new_artifact_epochs() -> None:
    program = _builder().build_program()
    capsules = program["capsules"]
    by_id = {row["id"]: row for row in capsules}

    assert len(capsules) == len(by_id) == 74
    assert capsules[0]["id"] == "g1_horizon_admit"
    previous = "g1_horizon_admit"
    for epoch_index, (epoch_id, cycle_index) in enumerate(
        zip(horizon.EPOCH_IDS, horizon.EPOCH_CYCLES, strict=True)
    ):
        scheduler_epoch = f"h{epoch_index + 1:02d}"
        shard_ids = [
            f"g1_{scheduler_epoch}_d1_shard_{index:02d}" for index in range(horizon.D1_SHARD_COUNT)
        ] + [
            f"g1_{scheduler_epoch}_mechanics_shard_{index:02d}"
            for index in range(horizon.MECHANICS_SHARD_COUNT)
        ]
        assert all(by_id[capsule_id]["depends_on"] == [previous] for capsule_id in shard_ids)
        for capsule_id in shard_ids:
            fields = by_id[capsule_id]["artifacts"][0]["fields"]
            assert fields["epoch_id"] == epoch_id
            assert fields["cycle_index"] == cycle_index
        classification_id = f"g1_{scheduler_epoch}_classify"
        assert by_id[classification_id]["depends_on"] == shard_ids
        classification = by_id[classification_id]["artifacts"][0]
        assert classification["path"].endswith(f"/classifications/{epoch_id.lower()}.json")
        assert classification["fields"]["epoch_id"] == epoch_id
        assert classification["fields"]["cycle_index"] == cycle_index
        previous = classification_id

    assert not any(str(row["id"]).startswith("g1_h06_") for row in capsules)
    assert by_id["g1_horizon_aggregate"]["depends_on"] == ["g1_h05_classify"]
    assert by_id["g1_horizon_verify"]["depends_on"] == ["g1_horizon_aggregate"]
    assert by_id["g1_horizon_report"]["depends_on"] == ["g1_horizon_verify"]


def test_capsules_use_nested_cli_exact_worker_envelope_and_unique_artifacts() -> None:
    builder = _builder()
    capsules = builder.build_program()["capsules"]
    artifact_paths: list[str] = []

    assert builder.PROCESS_MARKER == "mop_generation1_successor_horizon.py"
    for capsule in capsules:
        command = tuple(capsule["command"])
        resources = capsule["resources"]
        authorities = {row["path"] for row in capsule["authorities"]}
        assert builder.CLI_PATH in command
        assert builder.CLI_PATH in authorities
        assert resources["process_marker"] == builder.PROCESS_MARKER
        assert resources["wall_minutes"] < 300
        assert len(capsule["artifacts"]) == 1
        artifact_paths.append(capsule["artifacts"][0]["path"])
        if "_shard_" in capsule["id"]:
            assert command[: len(TASKPOLICY_ADAPTIVE_PREFIX)] == (TASKPOLICY_ADAPTIVE_PREFIX)
            assert resources["wall_minutes"] == 285
            assert resources["cpu_cores"] == 20
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_ADAPTIVE_CAP_GB
            assert command[command.index("--idle-workers") + 1] == "20"
            assert command[command.index("--hawking-workers") + 1] == "1"
        else:
            assert command[: len(TASKPOLICY_COEXISTENCE_PREFIX)] == (TASKPOLICY_COEXISTENCE_PREFIX)
            assert resources["cpu_cores"] == 1
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_COEXISTENCE_CAP_GB

    assert len(artifact_paths) == len(set(artifact_paths)) == 74


def test_program_freezes_v2_runtime_and_unchanged_v1_execution_authorities() -> None:
    builder = _builder()
    by_id = {row["id"]: row for row in builder.build_program()["capsules"]}
    compute = {row["path"] for row in by_id["g1_h01_d1_shard_00"]["authorities"]}
    verifier = {row["path"] for row in by_id["g1_horizon_verify"]["authorities"]}

    assert builder.RUNTIME_PATH in compute
    assert builder.VERIFIER_PATH in verifier
    assert "src/mop/studies/generation1_successor_horizon.py" in compute
    assert "src/mop/studies/generation1_successor_horizon_verify.py" in verifier
    assert "configs/campaign/generation1_successor_horizon_v1.json" in verifier
    assert "src/mop/studies/generation1_consolidated_final_campaign.py" in compute
    assert "src/mop/studies/generation1_successor_mechanics_queue.py" in compute
    assert "src/mop/mechanisms/integrated_escs_runner.py" in compute
    assert by_id["g1_horizon_verify"]["artifacts"][0]["schema"] == (horizon.VERIFICATION_SCHEMA)
    assert by_id["g1_horizon_verify"]["artifacts"][0]["fields"]["claim_scope"] == (
        horizon_verifier.CLAIM_SCOPE
    )


def test_generated_manifest_matches_builder_and_generic_loader() -> None:
    builder = _builder()
    expected = builder.build_program()
    generated = json.loads(horizon.PROGRAM_MANIFEST.read_text(encoding="utf-8"))

    assert generated == expected
    loaded = load_program(horizon.PROGRAM_MANIFEST)
    assert loaded.program_id == horizon.PROGRAM_ID
    assert loaded.program_sha256 == expected["program_sha256"]
    assert len(loaded.capsules) == 74
