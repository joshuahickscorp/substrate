from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from mop.config import REPO_ROOT
from mop.studies import generation1_successor_horizon as horizon
from mop.studio.generation1_supervisor import load_program
from mop.studio.local_throttle import (
    TASKPOLICY_ADAPTIVE_CAP_GB,
    TASKPOLICY_ADAPTIVE_PREFIX,
    TASKPOLICY_COEXISTENCE_CAP_GB,
    TASKPOLICY_COEXISTENCE_PREFIX,
)


def _load(name: str, relative: str) -> ModuleType:
    path = REPO_ROOT / relative
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _builder() -> ModuleType:
    return _load(
        "generation1_successor_horizon_program_builder",
        "scripts/build_generation1_successor_horizon_program.py",
    )


def _cli() -> ModuleType:
    return _load(
        "generation1_successor_horizon_cli",
        "scripts/mop_generation1_successor_horizon.py",
    )


def test_program_has_five_parallel_epoch_blocks_and_exact_barriers() -> None:
    program = _builder().build_program()
    capsules = program["capsules"]
    by_id = {row["id"]: row for row in capsules}

    assert len(capsules) == 74
    assert len(by_id) == 74
    assert capsules[0]["id"] == "g1_horizon_admit"
    previous = "g1_horizon_admit"
    for epoch_id in horizon.EPOCH_IDS:
        epoch = epoch_id.lower()
        shard_ids = [f"g1_{epoch}_d1_shard_{index:02d}" for index in range(5)] + [
            f"g1_{epoch}_mechanics_shard_{index:02d}" for index in range(8)
        ]
        assert all(by_id[capsule_id]["depends_on"] == [previous] for capsule_id in shard_ids)
        classification = f"g1_{epoch}_classify"
        assert by_id[classification]["depends_on"] == shard_ids
        previous = classification

    assert by_id["g1_horizon_aggregate"]["depends_on"] == ["g1_h05_classify"]
    assert by_id["g1_horizon_verify"]["depends_on"] == ["g1_horizon_aggregate"]
    assert by_id["g1_horizon_report"]["depends_on"] == ["g1_horizon_verify"]


def test_every_capsule_is_taskpolicy_wrapped_bounded_and_exactly_artifacted() -> None:
    builder = _builder()
    program = builder.build_program()
    artifact_paths: list[str] = []
    for capsule in program["capsules"]:
        command = tuple(capsule["command"])
        resources = capsule["resources"]
        assert resources["wall_minutes"] < 300
        assert resources["process_marker"] == builder.PROCESS_MARKER
        assert builder.CLI_PATH in command
        assert any(row["path"] == builder.CLI_PATH for row in capsule["authorities"])
        assert len(capsule["artifacts"]) == 1
        artifact_paths.append(capsule["artifacts"][0]["path"])
        if "_shard_" in capsule["id"]:
            assert command[: len(TASKPOLICY_ADAPTIVE_PREFIX)] == TASKPOLICY_ADAPTIVE_PREFIX
            assert resources["cpu_cores"] == 8
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_ADAPTIVE_CAP_GB
            assert command[command.index("--idle-workers") + 1] == "8"
            assert command[command.index("--hawking-workers") + 1] == "1"
        else:
            assert command[: len(TASKPOLICY_COEXISTENCE_PREFIX)] == TASKPOLICY_COEXISTENCE_PREFIX
            assert resources["cpu_cores"] == 1
            assert resources["estimated_unified_memory_gb"] == TASKPOLICY_COEXISTENCE_CAP_GB
    assert len(artifact_paths) == len(set(artifact_paths)) == 74


def test_compute_and_verifier_capsules_freeze_their_runtime_sources() -> None:
    builder = _builder()
    by_id = {row["id"]: row for row in builder.build_program()["capsules"]}
    compute_authorities = {row["path"] for row in by_id["g1_h01_d1_shard_00"]["authorities"]}
    verifier_authorities = {row["path"] for row in by_id["g1_horizon_verify"]["authorities"]}

    assert "src/mop/studies/generation1_successor_horizon.py" in compute_authorities
    assert "src/mop/studies/generation1_consolidated_final_campaign.py" in compute_authorities
    assert "src/mop/studies/generation1_c3_router_redesign.py" in compute_authorities
    assert "src/mop/mechanisms/integrated_escs_runner.py" in compute_authorities
    assert "proof/GENERATION1_CONTEXT_ROUTING.verification.json" in compute_authorities
    assert "src/mop/studies/generation1_successor_horizon_verify.py" in verifier_authorities
    assert by_id["g1_horizon_verify"]["artifacts"][0]["schema"] == (
        "mop-generation1-successor-horizon-verification/v1"
    )
    verifier_fields = by_id["g1_horizon_verify"]["artifacts"][0]["fields"]
    assert verifier_fields["mutation_suite.count"] == 9
    assert verifier_fields["mutation_suite.rejected"] == 9


def test_generated_manifest_matches_builder_and_generic_loader() -> None:
    builder = _builder()
    expected = builder.build_program()
    generated = json.loads(horizon.PROGRAM_MANIFEST.read_text(encoding="utf-8"))

    assert generated == expected
    loaded = load_program(horizon.PROGRAM_MANIFEST)
    assert loaded.program_id == horizon.PROGRAM_ID
    assert loaded.program_sha256 == expected["program_sha256"]
    assert len(loaded.capsules) == 74


def test_cli_dispatches_all_six_operations_with_explicit_paths(monkeypatch) -> None:
    cli = _cli()
    calls: list[tuple[str, dict]] = []

    def record(name: str):
        def invoke(**kwargs):
            calls.append((name, kwargs))
            return {"operation": name}

        return invoke

    monkeypatch.setattr(cli.horizon, "admit", record("admit"))
    monkeypatch.setattr(cli.horizon, "run_shard", record("run-shard"))
    monkeypatch.setattr(cli.horizon, "classify_epoch", record("classify"))
    monkeypatch.setattr(cli.horizon, "aggregate", record("aggregate"))
    monkeypatch.setattr(cli.horizon_verify, "verify", record("verify"))
    monkeypatch.setattr(cli.horizon, "render_report", record("report"))

    commands = (
        ["admit", "--output", "a.json", "--final", "f.json"],
        [
            "run-shard",
            "--root",
            "root",
            "--admission",
            "a.json",
            "--epoch-index",
            "2",
            "--lane",
            "d1",
            "--shard-index",
            "3",
            "--idle-workers",
            "8",
            "--hawking-workers",
            "1",
        ],
        ["classify", "--root", "root", "--admission", "a.json", "--epoch-index", "2"],
        ["aggregate", "--root", "root", "--admission", "a.json", "--output", "r.json"],
        ["verify", "--result", "r.json", "--output", "v.json"],
        [
            "report",
            "--result",
            "r.json",
            "--verification",
            "v.json",
            "--report",
            "report.md",
            "--receipt",
            "receipt.json",
        ],
    )
    for arguments in commands:
        result = cli.dispatch(cli.build_parser().parse_args(arguments))
        assert result["operation"] == arguments[0]

    assert [name for name, _ in calls] == [
        "admit",
        "run-shard",
        "classify",
        "aggregate",
        "verify",
        "report",
    ]
    assert calls[1][1]["epoch_index"] == 2
    assert calls[1][1]["idle_workers"] == 8
    assert calls[-1][1]["receipt_path"] == Path("receipt.json")
