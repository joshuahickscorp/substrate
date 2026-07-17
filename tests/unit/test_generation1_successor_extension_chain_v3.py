from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_extension_chain_v3 as chain

NOW = dt.datetime(2026, 7, 17, 4, 0, tzinfo=dt.UTC)
_TEST_VALIDATOR = object()


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(core: dict[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: chain.canonical_sha256(core)}


def _target_manifest(path: Path, *, revision: int = 1) -> str:
    core = {
        "schema": "mop-generation1-program/v1",
        "program_id": chain.TARGET_PROGRAM_ID,
        "revision": revision,
    }
    payload = _sealed(core, "program_sha256")
    _write(path, payload)
    return str(payload["program_sha256"])


def _artifact_report(repo_root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(repo_root)),
        "sha256": chain.sha256_file(path),
        "schema": payload["schema"],
        "all_ok": True,
        "problems": [],
    }


def _predecessor_row(repo_root: Path, capsule_id: str, *, complete: bool) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    if complete:
        artifact_path = (
            repo_root
            / "runs/generation1/generation1-successor-evidence-chain-v6/evidence"
            / f"{capsule_id}.json"
        )
        artifact = {
            "schema": "test-generation1-successor-v6-evidence/v1",
            "capsule_id": capsule_id,
            "complete": True,
        }
        _write(artifact_path, artifact)
        artifacts.append(_artifact_report(repo_root, artifact_path, artifact))
    return {
        "status": "complete" if complete else "pending",
        "attempts": 1 if complete else 0,
        "returncode": 0 if complete else None,
        "started_at": NOW.isoformat() if complete else None,
        "finished_at": NOW.isoformat() if complete else None,
        "artifacts": artifacts,
        "last_problem": None,
        "process": None,
        "adoption_receipts": [],
        "launch_requested_at": None,
        "launched_pid": None,
    }


def _predecessor_status(path: Path, state: str) -> None:
    repo_root = path.parents[3]
    complete = state == "complete"
    capsules = {
        capsule_id: _predecessor_row(repo_root, capsule_id, complete=complete)
        for capsule_id in chain.PREDECESSOR_CAPSULE_IDS
    }
    core = {
        "schema": chain.PREDECESSOR_STATUS_SCHEMA,
        "program_id": chain.PREDECESSOR_PROGRAM_ID,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "supervisor": {"pid": 4001, "create_time": 4001.5},
        "execution_enabled": complete,
        "state": state,
        "horizon_program": {"program_id": "generation1-successor-horizon-v1"},
        "capsules": capsules,
        "counts": {
            "complete": len(capsules) if complete else 0,
            "total": len(capsules),
            "remaining": 0 if complete else len(capsules),
        },
        "finished_at": NOW.isoformat() if complete else None,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        "supersedes": "generation1-successor-evidence-chain-v5",
    }
    _write(path, _sealed(core, "status_sha256"))


def _target_capsule_row(
    capsule: SimpleNamespace,
    *,
    complete: bool,
    artifact_report: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "id": capsule.capsule_id,
        "kind": capsule.kind,
        "priority": capsule.priority,
        "depends_on": list(capsule.depends_on),
        "capsule_sha256": capsule.capsule_sha256,
        "source": "base",
        "status": "complete" if complete else "pending",
        "attempts": 1 if complete else 0,
        "child_pid": 4101 if complete else None,
        "child_create_time": 4101.5 if complete else None,
        "started_at": NOW.isoformat() if complete else None,
        "finished_at": NOW.isoformat() if complete else None,
        "returncode": 0 if complete else None,
        "artifacts": [dict(artifact_report)] if complete else [],
        "last_problem": None,
        "runtime": {},
    }


def _extension(
    tmp_path: Path,
    *,
    predecessor_state: str = "waiting_legacy",
    execute: bool = False,
    target_status: Mapping[str, Any] | None = None,
    target_starter: Any = None,
    process_table: Any = None,
    identity_probe: Any = None,
    predecessor_completion_validator: Any = _TEST_VALIDATOR,
    predecessor_launch_validator: Any = _TEST_VALIDATOR,
) -> tuple[chain.SuccessorExtensionChain, list[Path], SimpleNamespace]:
    manifest_path = tmp_path / "configs/campaign/generation1_successor_horizon_v2.json"
    program_sha256 = _target_manifest(manifest_path)
    predecessor_path = (
        tmp_path / "runs/generation1/generation1-successor-evidence-chain-v6" / "current_status.json"
    )
    _predecessor_status(predecessor_path, predecessor_state)
    status_path = tmp_path / "runs/generation1/generation1-successor-horizon-v2" / "current_status.json"
    target_artifact_path = tmp_path / "proof/test_generation1_successor_horizon_v2.json"
    target_artifact_core = {"schema": "test-generation1-successor-horizon-v2/v1", "result": {"all_ok": True}}
    target_artifact = _sealed(target_artifact_core, "artifact_sha256")
    _write(target_artifact_path, target_artifact)
    expectation = SimpleNamespace(
        path=str(target_artifact_path.relative_to(tmp_path)),
        schema=target_artifact["schema"],
        fields=(("result.all_ok", True),),
        seal_field="artifact_sha256",
    )
    capsule = SimpleNamespace(
        capsule_id="target_capsule",
        kind="verifier",
        priority=1,
        depends_on=(),
        capsule_sha256="c" * 64,
        artifacts=(expectation,),
    )
    program = SimpleNamespace(
        path=manifest_path.resolve(),
        file_sha256=chain.sha256_file(manifest_path),
        program_id=chain.TARGET_PROGRAM_ID,
        program_sha256=program_sha256,
        repo_root=tmp_path.resolve(),
        capsules=(capsule,),
        status_path=status_path,
    )
    if target_status is not None:
        complete = target_status.get("state") == "complete"
        generic_implementation = chain._generic_supervisor_authority(tmp_path)
        supervisor = {
            "pid": 4201,
            "create_time": 4201.5,
            "implementation_path": generic_implementation["path"],
            "implementation_sha256": generic_implementation["sha256"],
        }
        supplied_supervisor = target_status.get("supervisor")
        if isinstance(supplied_supervisor, Mapping):
            supervisor.update(supplied_supervisor)
        target_artifact_report = _artifact_report(tmp_path, target_artifact_path, target_artifact)
        normalized_core: dict[str, Any] = {
            "schema": chain.GENERIC_STATUS_SCHEMA,
            "program_id": chain.TARGET_PROGRAM_ID,
            "created_at": NOW.isoformat(),
            "program": {
                "path": str(program.path),
                "file_sha256": program.file_sha256,
                "program_sha256": program.program_sha256,
            },
            "supervisor": supervisor,
            "execution_enabled": True,
            "state": target_status["state"],
            "queue_head_sha256": chain.canonical_sha256(
                {
                    "program_sha256": program.program_sha256,
                    "base_capsules": [capsule.capsule_sha256],
                }
            ),
            "next_injection_sequence": 1,
            "accepted_injection_count": 0,
            "current_capsule": None,
            "capsules": {
                capsule.capsule_id: _target_capsule_row(
                    capsule,
                    complete=complete,
                    artifact_report=target_artifact_report,
                )
            },
            "last_admission": {"allowed": True} if complete else None,
            "lane_reservation": None,
            "problems": [],
            **{key: value for key, value in target_status.items() if key != "supervisor"},
        }
        normalized_core.pop("status_sha256", None)
        _write(status_path, _sealed(normalized_core, "status_sha256"))
    loads: list[Path] = []

    def load(path: Path, *, repo_root: Path) -> Any:
        assert repo_root == tmp_path.resolve()
        loads.append(path)
        return program

    starter = target_starter if target_starter is not None else lambda *_args, **_kwargs: {}
    validators: dict[str, Any] = {}
    if predecessor_completion_validator is _TEST_VALIDATOR:
        validators["predecessor_completion_validator_fn"] = lambda status: status
    elif predecessor_completion_validator is not None:
        validators["predecessor_completion_validator_fn"] = predecessor_completion_validator
    if predecessor_launch_validator is _TEST_VALIDATOR:
        validators["predecessor_launch_validator_fn"] = lambda: None
    elif predecessor_launch_validator is not None:
        validators["predecessor_launch_validator_fn"] = predecessor_launch_validator
    extension = chain.SuccessorExtensionChain(
        root=tmp_path / "runs/generation1" / chain.PROGRAM_ID,
        repo_root=tmp_path,
        predecessor_status_path=predecessor_path,
        target_program_path=manifest_path,
        execute=execute,
        identity_probe_fn=identity_probe if identity_probe is not None else lambda _identity: "alive",
        process_table_fn=process_table if process_table is not None else lambda: (),
        program_loader_fn=load,
        target_status_reader_fn=lambda _program: json.loads(status_path.read_text(encoding="utf-8")),
        target_starter_fn=starter,
        now_fn=lambda: NOW,
        sleep_fn=lambda _seconds: None,
        **validators,
    )
    return extension, loads, program


# ----------------------------------------------------------------------------
# Predecessor rebind (v6) and preserved target (horizon v2) identity
# ----------------------------------------------------------------------------


def test_extv3_predecessor_binding_points_at_v6_and_target_stays_horizon_v2(tmp_path: Path) -> None:
    assert chain.PROGRAM_ID == "generation1-successor-extension-chain-v3"
    assert chain.STATE_SCHEMA == "mop-generation1-successor-extension-chain-state/v3"
    assert chain.STATUS_SCHEMA == "mop-generation1-successor-extension-chain-status/v3"
    assert chain.PREDECESSOR_PROGRAM_ID == "generation1-successor-evidence-chain-v6"
    assert chain.PREDECESSOR_STATUS_SCHEMA == "mop-generation1-successor-evidence-chain-status/v6"
    assert chain.TARGET_PROGRAM_ID == "generation1-successor-horizon-v2"
    assert str(chain.DEFAULT_ROOT).endswith(
        "runs/generation1/generation1-successor-extension-chain-v3"
    )
    assert str(chain.DEFAULT_PREDECESSOR_STATUS).endswith(
        "runs/generation1/generation1-successor-evidence-chain-v6/current_status.json"
    )
    assert str(chain.DEFAULT_TARGET_PROGRAM).endswith(
        "configs/campaign/generation1_successor_horizon_v2.json"
    )
    assert chain.predecessor_v6.CHAIN_ID == "generation1-successor-evidence-chain-v6"


def test_extv3_initial_state_binds_v6_predecessor_and_horizon_v2_capsule(tmp_path: Path) -> None:
    extension, _loads, _program = _extension(tmp_path)
    predecessor = extension.state["predecessor"]

    assert predecessor["program_id"] == "generation1-successor-evidence-chain-v6"
    assert predecessor["status_schema"] == "mop-generation1-successor-evidence-chain-status/v6"
    assert predecessor["status_path"].endswith(
        "generation1-successor-evidence-chain-v6/current_status.json"
    )
    assert predecessor["signals_allowed"] is False
    assert set(extension.state["capsules"]) == {"predecessor_chain_v6", "successor_horizon_v2"}


# ----------------------------------------------------------------------------
# Observation-only waiting on v6, no signals
# ----------------------------------------------------------------------------


def test_extv3_waits_for_v6_after_validating_target_without_starting_or_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[object] = []
    signals: list[object] = []
    monkeypatch.setattr(chain.os, "kill", lambda *arguments: signals.append(arguments), raising=False)
    extension, loads, _program = _extension(
        tmp_path,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
    )

    status = extension.tick()

    assert status["state"] == "waiting_predecessor"
    assert status["predecessor"]["program_id"] == "generation1-successor-evidence-chain-v6"
    assert status["predecessor"]["signals_allowed"] is False
    assert status["capsules"]["predecessor_chain_v6"]["status"] == "observed"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert loads == [extension.target_program_path]
    assert starts == []
    assert signals == []


def test_extv3_terminal_v6_failure_holds_without_starting_target(tmp_path: Path) -> None:
    starts: list[object] = []
    extension, loads, _program = _extension(
        tmp_path,
        predecessor_state="failure_hold",
        execute=True,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
    )

    status = extension.tick()

    assert status["state"] == "failure_hold"
    assert status["capsules"]["predecessor_chain_v6"]["status"] == "failure_hold"
    assert loads == [extension.target_program_path]
    assert starts == []


def test_extv3_requires_predecessor_supersedes_v5(tmp_path: Path) -> None:
    extension, _loads, _program = _extension(tmp_path, execute=True)
    core = {
        "schema": chain.PREDECESSOR_STATUS_SCHEMA,
        "program_id": chain.PREDECESSOR_PROGRAM_ID,
        "created_at": NOW.isoformat(),
        "updated_at": NOW.isoformat(),
        "supervisor": {"pid": 4001, "create_time": 4001.5},
        "execution_enabled": False,
        "state": "waiting_legacy",
        "horizon_program": {"program_id": "generation1-successor-horizon-v1"},
        "capsules": {
            capsule_id: _predecessor_row(extension.repo_root, capsule_id, complete=False)
            for capsule_id in chain.PREDECESSOR_CAPSULE_IDS
        },
        "counts": {"complete": 0, "total": 4, "remaining": 4},
        "finished_at": None,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
        # v6 must declare it supersedes v5; a v4 claim is the dead v5's own value.
        "supersedes": "generation1-successor-evidence-chain-v4",
    }
    _write(extension.predecessor_status_path, _sealed(core, "status_sha256"))

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert "v6 predecessor status identity, safety, or state drifted" in status["problems"][-1]


# ----------------------------------------------------------------------------
# Launching horizon v2 once v6 is durably complete
# ----------------------------------------------------------------------------


def test_extv3_launches_horizon_v2_when_v6_complete(tmp_path: Path) -> None:
    starts: list[tuple[Any, dict[str, Any]]] = []

    def start(program: Any, **kwargs: Any) -> Mapping[str, Any]:
        starts.append((program, kwargs))
        return {}

    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_starter=start,
    )

    status = extension.tick()

    assert status["state"] == "waiting_target"
    assert status["capsules"]["predecessor_chain_v6"]["status"] == "complete"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "running"
    assert len(starts) == 1
    assert starts[0][0] is program
    assert starts[0][1] == {"execute": True, "use_caffeinate": True}


def test_extv3_prelaunch_authority_replay_runs_before_target_starter(tmp_path: Path) -> None:
    calls: list[str] = []

    def validate_launch() -> None:
        calls.append("validate")

    def start(_program: Any, **_kwargs: Any) -> Mapping[str, Any]:
        calls.append("start")
        return {}

    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_starter=start,
        predecessor_launch_validator=validate_launch,
    )

    status = extension.tick()

    assert status["state"] == "waiting_target"
    assert calls == ["validate", "validate", "start"]


def test_extv3_structurally_complete_fake_v6_status_without_durable_state_is_rejected(
    tmp_path: Path,
) -> None:
    starts: list[object] = []
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
        predecessor_completion_validator=None,
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert starts == []
    assert "v6 predecessor complete snapshot is invalid" in status["problems"][-1]


# ----------------------------------------------------------------------------
# Status round-trip and detached start (v3 entrypoint)
# ----------------------------------------------------------------------------


def test_extv3_validate_extension_status_round_trip(tmp_path: Path) -> None:
    extension, _loads, _program = _extension(tmp_path, execute=True)
    status = extension.tick()

    assert (
        chain.validate_extension_status(
            status,
            repo_root=tmp_path,
            target_program_path=extension.target_program_path,
            predecessor_status_path=extension.predecessor_status_path,
        )
        == "waiting_predecessor"
    )


def test_extv3_detached_start_emits_one_exact_v3_run_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-extension-chain-v3"
    launches: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "validate_extension_status", lambda *_a, **_k: "waiting_predecessor")
    monkeypatch.setattr(chain, "_validate_extension_status_authorities", lambda *_a, **_k: None)

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append((command, kwargs))
        core = {
            "schema": chain.STATUS_SCHEMA,
            "program_id": chain.PROGRAM_ID,
            "state": "waiting_predecessor",
            "activation_allowed": False,
            "scientific_promotion": False,
            "supervisor": {"pid": 820, "create_time": 82.0},
        }
        _write(root / chain.STATUS_FILE, {**core, "status_sha256": chain.canonical_sha256(core)})
        return SimpleNamespace(pid=820, poll=lambda: None)

    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    result = chain.start_extension_detached(root=root, execute=True, use_caffeinate=False)

    expected = [
        str(chain.REPO_ROOT / ".venv/bin/python"),
        str(chain.REPO_ROOT / "scripts/mop_generation1_successor_extension_chain_v3.py"),
        "run",
        "--execute",
        "--root",
        str(root),
    ]
    assert launches[0][0] == expected
    assert launches[0][0].count("run") == 1
    assert result["launched_pid"] == 820
