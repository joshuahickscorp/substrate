from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_extension_chain as chain

NOW = dt.datetime(2026, 7, 16, 8, 0, tzinfo=dt.UTC)


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


def _predecessor_row(
    repo_root: Path,
    capsule_id: str,
    *,
    complete: bool,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    if complete:
        artifact_path = (
            repo_root
            / "runs/generation1/generation1-successor-evidence-chain-v4/evidence"
            / f"{capsule_id}.json"
        )
        artifact = {
            "schema": "test-generation1-successor-v4-evidence/v1",
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
        "supersedes": "generation1-successor-evidence-chain-v3",
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
) -> tuple[chain.SuccessorExtensionChain, list[Path], SimpleNamespace]:
    manifest_path = tmp_path / "configs/campaign/generation1_successor_horizon_v2.json"
    program_sha256 = _target_manifest(manifest_path)
    predecessor_path = (
        tmp_path / "runs/generation1/generation1-successor-evidence-chain-v4" / "current_status.json"
    )
    _predecessor_status(predecessor_path, predecessor_state)
    status_path = tmp_path / "runs/generation1/generation1-successor-horizon-v2" / "current_status.json"
    target_artifact_path = tmp_path / "proof/test_generation1_successor_horizon_v2.json"
    target_artifact_core = {
        "schema": "test-generation1-successor-horizon-v2/v1",
        "result": {"all_ok": True},
    }
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
        target_artifact_report = _artifact_report(
            tmp_path,
            target_artifact_path,
            target_artifact,
        )
        normalized_core: dict[str, Any] = {
            "schema": chain.GENERIC_STATUS_SCHEMA,
            "program_id": chain.TARGET_PROGRAM_ID,
            "created_at": NOW.isoformat(),
            "program": {
                "path": str(program.path),
                "file_sha256": program.file_sha256,
                "program_sha256": program.program_sha256,
            },
            "supervisor": {"pid": 4201, "create_time": 4201.5},
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
            **target_status,
        }
        normalized_core.pop("status_sha256", None)
        _write(status_path, _sealed(normalized_core, "status_sha256"))
    loads: list[Path] = []

    def load(path: Path, *, repo_root: Path) -> Any:
        assert repo_root == tmp_path.resolve()
        loads.append(path)
        return program

    starter = target_starter if target_starter is not None else lambda *_args, **_kwargs: {}
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
    )
    return extension, loads, program


def test_waits_for_v4_after_validating_target_without_starting_or_signalling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    starts: list[object] = []
    signals: list[object] = []
    monkeypatch.setattr(
        chain.os,
        "kill",
        lambda *arguments: signals.append(arguments),
        raising=False,
    )
    extension, loads, _program = _extension(
        tmp_path,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
    )

    status = extension.tick()

    assert status["state"] == "waiting_predecessor"
    assert status["predecessor"]["signals_allowed"] is False
    assert status["capsules"]["predecessor_chain_v4"]["status"] == "observed"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert loads == [extension.target_program_path]
    assert starts == []
    assert signals == []


def test_terminal_v4_failure_holds_without_starting_target(tmp_path: Path) -> None:
    starts: list[object] = []
    extension, loads, _program = _extension(
        tmp_path,
        predecessor_state="failure_hold",
        execute=True,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
    )

    status = extension.tick()

    assert status["state"] == "failure_hold"
    assert status["capsules"]["predecessor_chain_v4"]["status"] == "failure_hold"
    assert loads == [extension.target_program_path]
    assert starts == []


def test_self_sealed_empty_v4_complete_status_is_rejected(tmp_path: Path) -> None:
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
    )
    forged_core = {
        "schema": chain.PREDECESSOR_STATUS_SCHEMA,
        "program_id": chain.PREDECESSOR_PROGRAM_ID,
        "state": "complete",
        "problems": [],
        "capsules": {},
        "counts": {"complete": 0, "total": 0, "remaining": 0},
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(
        extension.predecessor_status_path,
        _sealed(forged_core, "status_sha256"),
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert status["capsules"]["predecessor_chain_v4"]["status"] == "pending"
    assert "v4 predecessor status fields drifted" in status["problems"][-1]


def test_launch_updates_persist_in_same_tick_after_intent_publish(
    tmp_path: Path,
) -> None:
    starts: list[tuple[Any, dict[str, Any]]] = []
    supervisor = {"pid": 9123, "create_time": 9123.5}

    def start(program: Any, **kwargs: Any) -> Mapping[str, Any]:
        starts.append((program, kwargs))
        return {
            "launched_pid": 9123,
            "status": {"supervisor": supervisor},
        }

    extension, loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_starter=start,
    )

    status = extension.tick()
    persisted = json.loads(extension.state_path.read_text(encoding="utf-8"))
    status_row = status["capsules"]["successor_horizon_v2"]
    persisted_row = persisted["capsules"]["successor_horizon_v2"]

    assert status["state"] == "waiting_target"
    assert status_row["status"] == persisted_row["status"] == "running"
    assert status_row["attempts"] == persisted_row["attempts"] == 1
    assert status_row["launched_pid"] == persisted_row["launched_pid"] == 9123
    assert status_row["process"] == persisted_row["process"] == supervisor
    assert status_row["launch_requested_at"] == NOW.isoformat()
    assert status_row["started_at"] == NOW.isoformat()
    assert len(loads) == len(starts) == 1
    assert starts[0][1] == {"execute": True, "use_caffeinate": True}


def test_live_target_is_monitored_without_duplicate_start(tmp_path: Path) -> None:
    starts: list[object] = []
    target_status = {
        "state": "running",
        "supervisor": {"pid": 77, "create_time": 77.5},
    }
    snapshot = chain.ProcessSnapshot(
        pid=77,
        create_time=77.5,
        pgid=77,
        cwd=str(tmp_path.resolve()),
        label=f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",
        command=(f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",),
    )
    extension, loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status=target_status,
        target_starter=lambda *args, **kwargs: starts.append((args, kwargs)),
        process_table=lambda: (snapshot,),
    )

    first = extension.tick()
    second = extension.tick()

    assert first["state"] == second["state"] == "waiting_target"
    assert second["capsules"]["successor_horizon_v2"]["status"] == "running"
    assert second["capsules"]["successor_horizon_v2"]["attempts"] == 0
    assert len(loads) == 2
    assert starts == []


def test_execution_disabled_target_status_is_recoverable_by_execute_start(
    tmp_path: Path,
) -> None:
    starts: list[tuple[Any, dict[str, Any]]] = []
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={
            "state": "execution_disabled",
            "execution_enabled": False,
        },
        target_starter=lambda program, **kwargs: starts.append((program, kwargs)) or {},
        identity_probe=lambda _identity: "gone",
    )

    status = extension.tick()

    assert status["state"] == "waiting_target"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "running"
    assert len(starts) == 1
    assert starts[0][1] == {"execute": True, "use_caffeinate": True}


def test_complete_target_completes_extension_with_bound_status_artifact(
    tmp_path: Path,
) -> None:
    target_status = {"state": "complete"}
    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status=target_status,
    )

    status = extension.tick()
    row = status["capsules"]["successor_horizon_v2"]

    assert status["state"] == "complete"
    assert status["counts"] == {"complete": 2, "remaining": 0, "total": 2}
    assert row["status"] == "complete"
    assert row["artifacts"] == [
        {
            "path": str(program.status_path.relative_to(tmp_path)),
            "sha256": chain.sha256_file(program.status_path),
            "schema": chain.GENERIC_STATUS_SCHEMA,
            "all_ok": True,
            "problems": [],
        }
    ]


def test_self_sealed_injected_empty_target_complete_status_is_rejected(
    tmp_path: Path,
) -> None:
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={
            "state": "complete",
            "accepted_injection_count": 7,
            "next_injection_sequence": 8,
            "capsules": {},
            "execution_enabled": False,
            "current_capsule": "forged",
            "lane_reservation": {"forged": True},
        },
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert "execution authority drifted" in status["problems"][-1]


def test_coherent_complete_target_requires_current_manifest_artifact_reports(
    tmp_path: Path,
) -> None:
    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={"state": "complete"},
    )
    forged = json.loads(program.status_path.read_text(encoding="utf-8"))
    forged.pop("status_sha256")
    forged["capsules"]["target_capsule"]["artifacts"][0]["sha256"] = "0" * 64
    _write(program.status_path, _sealed(forged, "status_sha256"))

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert "artifact report drifted" in status["problems"][-1]


def test_completed_target_status_must_not_disappear(tmp_path: Path) -> None:
    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={"state": "complete"},
    )

    first = extension.tick()
    program.status_path.unlink()
    second = extension.tick()

    assert first["state"] == "complete"
    assert second["state"] == "integrity_hold"
    assert "evidence disappeared or regressed" in second["problems"][-1]


def test_completed_predecessor_evidence_is_monotonic_and_must_remain_exact(
    tmp_path: Path,
) -> None:
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=False,
    )

    first = extension.tick()
    predecessor = first["capsules"]["predecessor_chain_v4"]
    finished_at = predecessor["finished_at"]
    artifacts = predecessor["artifacts"]
    extension.predecessor_status_path.unlink()

    second = extension.tick()

    assert first["state"] == "waiting_target"
    assert second["state"] == "integrity_hold"
    assert second["capsules"]["predecessor_chain_v4"]["finished_at"] == finished_at
    assert second["capsules"]["predecessor_chain_v4"]["artifacts"] == artifacts
    assert "disappeared or regressed" in second["problems"][-1]


@pytest.mark.parametrize("state", ("unknown_state", "", "COMPLETE"))
def test_target_status_state_must_be_from_the_generic_supervisor_vocabulary(
    tmp_path: Path,
    state: str,
) -> None:
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={"state": state},
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert "program binding, or state drifted" in status["problems"][-1]


def test_stale_complete_target_status_cannot_complete_a_new_manifest(
    tmp_path: Path,
) -> None:
    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={
            "state": "complete",
            "program": {
                "path": "stale-manifest.json",
                "file_sha256": "0" * 64,
                "program_sha256": "1" * 64,
            },
        },
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert status["capsules"]["successor_horizon_v2"]["status"] == "pending"
    assert str(program.program_sha256) not in status["problems"][-1]
    assert "program binding, or state drifted" in status["problems"][-1]


def test_conflicting_visible_target_supervisors_fail_closed(tmp_path: Path) -> None:
    target_status = {
        "state": "running",
        "supervisor": {"pid": 77, "create_time": 77.5},
    }
    snapshots = tuple(
        chain.ProcessSnapshot(
            pid=pid,
            create_time=float(pid) + 0.5,
            pgid=pid,
            cwd=str(tmp_path.resolve()),
            label=f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",
            command=(f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",),
        )
        for pid in (77, 88)
    )
    extension, _loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status=target_status,
        process_table=lambda: snapshots,
    )

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert "multiple successor horizon v2 supervisors" in status["problems"][-1]


def test_run_reloads_sealed_state_after_acquiring_lifetime_lock(
    tmp_path: Path,
) -> None:
    extension, _loads, _program = _extension(tmp_path)
    extension._publish()
    persisted = json.loads(extension.state_path.read_text(encoding="utf-8"))
    persisted.pop("state_sha256")
    persisted["status"] = "drained"
    persisted["finished_at"] = NOW.isoformat()
    _write(
        extension.state_path,
        {**persisted, "state_sha256": chain.canonical_sha256(persisted)},
    )
    original_load = extension._load_state
    calls = 0

    def counted_load() -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return original_load()

    extension._load_state = counted_load  # type: ignore[method-assign]

    status = extension.run(max_cycles=1)

    assert status["state"] == "drained"
    assert calls == 1


def test_manifest_drift_enters_integrity_hold(tmp_path: Path) -> None:
    extension, loads, _program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
    )
    _target_manifest(extension.target_program_path, revision=2)

    status = extension.tick()

    assert status["state"] == "integrity_hold"
    assert "manifest authority drifted" in status["problems"][-1]
    assert loads == []


def test_detached_terminal_shortcut_rejects_stale_bound_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs/generation1" / chain.PROGRAM_ID
    target_program_path = tmp_path / "configs/campaign/generation1_successor_horizon_v2.json"
    predecessor_status_path = (
        tmp_path / "runs/generation1/generation1-successor-evidence-chain-v4/current_status.json"
    )
    _target_manifest(target_program_path)
    stale_target = chain._target_program_authority(target_program_path, tmp_path)
    stale_target["program_sha256"] = "f" * 64
    status_core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.PROGRAM_ID,
        "state": "complete",
        "parent_implementation": chain._implementation_authority(tmp_path),
        "predecessor": chain._predecessor_observation_binding(
            predecessor_status_path,
            tmp_path,
        ),
        "target_program": stale_target,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(root / chain.STATUS_FILE, _sealed(status_core, "status_sha256"))
    monkeypatch.setattr(chain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chain, "DEFAULT_TARGET_PROGRAM", target_program_path)
    monkeypatch.setattr(chain, "DEFAULT_PREDECESSOR_STATUS", predecessor_status_path)

    with pytest.raises(
        chain.SuccessorExtensionRefused,
        match="target authority drifted",
    ):
        chain.start_extension_detached(root=root, execute=True)


@pytest.mark.parametrize("drifted_evidence", ("predecessor", "target"))
def test_detached_terminal_shortcut_revalidates_completed_evidence_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted_evidence: str,
) -> None:
    extension, _loads, program = _extension(
        tmp_path,
        predecessor_state="complete",
        execute=True,
        target_status={"state": "complete"},
    )
    terminal = extension.tick()
    assert terminal["state"] == "complete"
    if drifted_evidence == "predecessor":
        extension.predecessor_status_path.unlink()
        expected_problem = "completed predecessor evidence disappeared or regressed"
    else:
        program.status_path.unlink()
        expected_problem = "completed successor horizon v2 evidence disappeared or regressed"

    monkeypatch.setattr(chain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chain, "DEFAULT_TARGET_PROGRAM", extension.target_program_path)
    monkeypatch.setattr(
        chain,
        "DEFAULT_PREDECESSOR_STATUS",
        extension.predecessor_status_path,
    )
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: False)
    monkeypatch.setattr(chain, "SuccessorExtensionChain", lambda **_kwargs: extension)

    first = chain.start_extension_detached(
        root=extension.root,
        execute=True,
        use_caffeinate=False,
    )
    second = chain.start_extension_detached(
        root=extension.root,
        execute=True,
        use_caffeinate=False,
    )

    assert first["already_terminal"] is True
    assert first["status"]["state"] == "integrity_hold"
    assert expected_problem in first["status"]["problems"][-1]
    assert second["status"]["state"] == "integrity_hold"
    assert second["status"]["problems"] == first["status"]["problems"]


def test_detached_live_parent_resolves_before_stale_authority_or_terminal_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs/generation1" / chain.PROGRAM_ID
    target_program_path = tmp_path / "configs/campaign/generation1_successor_horizon_v2.json"
    predecessor_status_path = (
        tmp_path / "runs/generation1/generation1-successor-evidence-chain-v4/current_status.json"
    )
    _target_manifest(target_program_path)
    stale_target = chain._target_program_authority(
        target_program_path,
        tmp_path,
    )
    stale_target["program_sha256"] = "f" * 64
    status_core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.PROGRAM_ID,
        "state": "complete",
        "supervisor": None,
        "parent_implementation": chain._implementation_authority(tmp_path),
        "predecessor": chain._predecessor_observation_binding(
            predecessor_status_path,
            tmp_path,
        ),
        "target_program": stale_target,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(root / chain.STATUS_FILE, _sealed(status_core, "status_sha256"))
    visible = chain.ProcessSnapshot(
        pid=2469,
        create_time=2469.5,
        pgid=2469,
        cwd=str(tmp_path.resolve()),
        label=chain.PARENT_LABEL,
        command=(chain.PARENT_LABEL,),
    )
    revalidations: list[Path] = []
    monkeypatch.setattr(chain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chain, "DEFAULT_TARGET_PROGRAM", target_program_path)
    monkeypatch.setattr(chain, "DEFAULT_PREDECESSOR_STATUS", predecessor_status_path)
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: visible)
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: False)
    monkeypatch.setattr(
        chain,
        "_revalidate_terminal_extension",
        lambda requested_root: revalidations.append(requested_root),
    )

    result = chain.start_extension_detached(
        root=root,
        execute=True,
        use_caffeinate=False,
    )

    assert result["already_running"] is True
    assert result["observed_process"] == visible.identity()
    assert revalidations == []


def test_detached_start_uses_caffeinate_and_requires_sealed_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "runs/generation1" / chain.PROGRAM_ID
    status_path = root / chain.STATUS_FILE
    target_program_path = tmp_path / "configs/campaign/generation1_successor_horizon_v2.json"
    predecessor_status_path = (
        tmp_path / "runs/generation1/generation1-successor-evidence-chain-v4/current_status.json"
    )
    _target_manifest(target_program_path)
    captured: dict[str, Any] = {}
    status_core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.PROGRAM_ID,
        "state": "waiting_predecessor",
        "parent_implementation": chain._implementation_authority(tmp_path),
        "predecessor": chain._predecessor_observation_binding(
            predecessor_status_path,
            tmp_path,
        ),
        "target_program": chain._target_program_authority(
            target_program_path,
            tmp_path,
        ),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    acknowledged = _sealed(status_core, "status_sha256")

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        captured["command"] = command
        captured["kwargs"] = kwargs
        _write(status_path, acknowledged)
        return SimpleNamespace(pid=2468, poll=lambda: None)

    monkeypatch.setattr(chain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chain, "DEFAULT_TARGET_PROGRAM", target_program_path)
    monkeypatch.setattr(chain, "DEFAULT_PREDECESSOR_STATUS", predecessor_status_path)
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: False)
    monkeypatch.setattr(chain.shutil, "which", lambda _name: "/usr/bin/caffeinate")
    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    result = chain.start_extension_detached(
        root=root,
        execute=True,
        use_caffeinate=True,
    )

    command = captured["command"]
    assert command[:2] == ["/usr/bin/caffeinate", "-ims"]
    assert command[-4:] == [
        "run",
        "--execute",
        "--root",
        str(root.resolve()),
    ]
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["env"]["MOP_PROCESS_LABEL"] == chain.PARENT_LABEL
    assert result["launched_pid"] == 2468
    assert result["caffeinate"] is True
    assert result["status"] == acknowledged


def test_cli_exposes_start_run_and_status_controls() -> None:
    parser = chain.build_parser()

    start = parser.parse_args(["start", "--execute", "--no-caffeinate"])
    run = parser.parse_args(["run", "--execute", "--once"])
    status = parser.parse_args(["status"])

    assert (start.command, start.execute, start.no_caffeinate) == (
        "start",
        True,
        True,
    )
    assert (run.command, run.execute, run.once) == ("run", True, True)
    assert status.command == "status"
