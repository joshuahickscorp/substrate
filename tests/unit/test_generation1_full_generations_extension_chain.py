from __future__ import annotations

import copy
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pytest

from mop.studies import generation1_successor_mechanics_queue as mechanics
from mop.studio import generation1_full_generations_extension_chain as chain
from mop.studio import generation1_supervisor as g1

NOW = "2026-07-16T12:00:00+00:00"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    g1.atomic_write_json(path, payload)


def _sealed(core: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {**core, field: g1.canonical_sha256(core)}


def _program_manifest(
    repo_root: Path,
    *,
    program_id: str,
    manifest_name: str,
) -> tuple[Path, g1.Program]:
    runner = repo_root / f"{program_id}.py"
    runner.write_text("# immutable full-generations-chain test runner\n")
    policy = repo_root / f"{program_id}.yaml"
    policy.write_text("schema: test-policy\n")
    artifact = f"proof/{program_id}.json"
    capsule_core: dict[str, Any] = {
        "schema": g1.CAPSULE_SCHEMA,
        "id": f"{program_id}-capsule",
        "kind": "aggregate",
        "priority": 100,
        "depends_on": [],
        "command": [sys.executable, runner.name],
        "cwd": ".",
        "environment": {},
        "resources": {
            "lane": "cpu",
            "accelerator": "none",
            "cpu_cores": 1,
            "estimated_unified_memory_gb": 1.0,
            "estimated_mps_gb": 0.0,
            "resource_basis": "bounded unit-test task",
            "forecast_write_gb": 0.01,
            "atomic_write_gb": 0.01,
            "wall_minutes": 5,
            "process_marker": runner.name,
        },
        "artifacts": [
            {
                "path": artifact,
                "schema": "mop-full-generations-chain-test-artifact/v1",
                "fields": {"ok": True},
                "seal_field": "payload_sha256",
            }
        ],
        "authorities": [
            {
                "path": runner.name,
                "sha256": g1.sha256_file(runner),
            }
        ],
    }
    capsule = _sealed(capsule_core, "capsule_sha256")
    program_root = f"runs/generation1/{program_id}"
    program_core: dict[str, Any] = {
        "schema": g1.PROGRAM_SCHEMA,
        "program_id": program_id,
        "program_root": program_root,
        "policy": {
            "path": policy.name,
            "sha256": g1.sha256_file(policy),
        },
        "authorities": [
            {
                "path": runner.name,
                "sha256": g1.sha256_file(runner),
            }
        ],
        "injection": {
            "inbox": f"{program_root}/control/inbox",
            "receipt_root": f"{program_root}/control/injection_receipts",
        },
        "control": {
            "throttle_state_root": "runs/local_throttle",
            "admission_samples": 1,
            "admission_interval_seconds": 0.01,
            "resource_retry_seconds": 0.01,
            "startup_ack_seconds": 1.0,
        },
        "capsules": [capsule],
    }
    manifest = _sealed(program_core, "program_sha256")
    path = repo_root / "configs/campaign" / manifest_name
    _write_json(path, manifest)
    return path, g1.load_program(path, repo_root=repo_root)


def _generic_status(
    program: g1.Program,
    *,
    state: str,
    pid: int,
) -> dict[str, Any]:
    capsule = program.capsules[0]
    row: dict[str, Any] = {
        "id": capsule.capsule_id,
        "kind": capsule.kind,
        "priority": capsule.priority,
        "depends_on": list(capsule.depends_on),
        "capsule_sha256": capsule.capsule_sha256,
        "source": "base",
        "status": "pending",
        "attempts": 0,
        "child_pid": None,
        "child_create_time": None,
        "started_at": None,
        "finished_at": None,
        "returncode": None,
        "artifacts": [],
        "last_problem": None,
        "runtime": {},
    }
    if state == "complete":
        artifact_path = program.repo_root / capsule.artifacts[0].path
        artifact_core = {
            "schema": capsule.artifacts[0].schema,
            "ok": True,
        }
        artifact = _sealed(artifact_core, "payload_sha256")
        _write_json(artifact_path, artifact)
        row.update(
            {
                "status": "complete",
                "attempts": 1,
                "finished_at": NOW,
                "returncode": 0,
                "artifacts": [
                    {
                        "path": capsule.artifacts[0].path,
                        "sha256": g1.sha256_file(artifact_path),
                        "schema": capsule.artifacts[0].schema,
                        "problems": [],
                        "all_ok": True,
                    }
                ],
            }
        )
    implementation = chain._generic_supervisor_authority(program.repo_root)
    core: dict[str, Any] = {
        "schema": g1.STATUS_SCHEMA,
        "program_id": program.program_id,
        "created_at": NOW,
        "program": chain._program_binding(program),
        "supervisor": {
            "pid": pid,
            "create_time": float(pid) + 0.5,
            "implementation_path": implementation["path"],
            "implementation_sha256": implementation["sha256"],
        },
        "execution_enabled": True,
        "state": state,
        "queue_head_sha256": g1.canonical_sha256(
            {
                "program_sha256": program.program_sha256,
                "base_capsules": [
                    program.capsules[0].capsule_sha256,
                ],
            }
        ),
        "next_injection_sequence": 1,
        "accepted_injection_count": 0,
        "current_capsule": None,
        "capsules": {capsule.capsule_id: row},
        "last_admission": None,
        "lane_reservation": None,
        "problems": [],
    }
    return _sealed(core, "status_sha256")


def _fixture(
    tmp_path: Path,
    *,
    predecessor_state: str,
    target_status: str | None = None,
    host_admission: Callable[[g1.Program, Mapping[str, Any] | None], Mapping[str, Any]],
    target_starter: Callable[..., Mapping[str, Any]],
    identity_probe: chain.IdentityProbe = lambda _identity: "gone",
    process_table: chain.ProcessTable = lambda: (),
) -> tuple[
    chain.FullGenerationsExtensionChain,
    g1.Program,
    g1.Program,
]:
    predecessor_path, predecessor = _program_manifest(
        tmp_path,
        program_id=chain.PREDECESSOR_PROGRAM_ID,
        manifest_name="generation1_successor_categorized_batch_wave_v1.json",
    )
    target_path, target = _program_manifest(
        tmp_path,
        program_id=chain.TARGET_PROGRAM_ID,
        manifest_name="generation1_full_generations_wave_v1.json",
    )
    _write_json(
        predecessor.status_path,
        _generic_status(predecessor, state=predecessor_state, pid=101),
    )
    if target_status is not None:
        _write_json(
            target.status_path,
            _generic_status(target, state=target_status, pid=202),
        )
    owner = chain.FullGenerationsExtensionChain(
        root=tmp_path / "runs/generation1" / chain.PROGRAM_ID,
        repo_root=tmp_path,
        predecessor_program_path=predecessor_path,
        target_program_path=target_path,
        execute=True,
        identity_probe_fn=identity_probe,
        process_table_fn=process_table,
        host_admission_fn=host_admission,
        target_starter_fn=target_starter,
    )
    return owner, predecessor, target


def test_parent_label_uses_uniform_scheme() -> None:
    assert chain.PARENT_LABEL == "mop:fullgen:extension"


def test_incomplete_predecessor_never_reaches_host_or_target(tmp_path: Path) -> None:
    admissions: list[str] = []
    starts: list[str] = []
    owner, _, _ = _fixture(
        tmp_path,
        predecessor_state="running",
        host_admission=lambda *_args: admissions.append("admission") or {"allowed": True},
        target_starter=lambda *_args, **_kwargs: starts.append("start") or {},
    )

    status = owner.tick()

    assert status["state"] == "waiting_predecessor"
    assert status["capsules"]["categorized_batch_wave_v1"]["status"] == "observed"
    assert status["capsules"]["full_generations_wave_v1"] == chain._empty_row()
    assert admissions == []
    assert starts == []
    assert (
        chain.validate_full_generations_extension_status(
            status,
            repo_root=tmp_path,
            predecessor_program_path=owner.predecessor_program_path,
            target_program_path=owner.target_program_path,
        )
        == "waiting_predecessor"
    )


def test_busy_host_never_starts_target(tmp_path: Path) -> None:
    starts: list[str] = []
    owner, _, _ = _fixture(
        tmp_path,
        predecessor_state="complete",
        host_admission=lambda *_args: {
            "allowed": False,
            "reason": "existing long wave owns the host",
        },
        target_starter=lambda *_args, **_kwargs: starts.append("start") or {},
    )

    status = owner.tick()

    assert status["state"] == "waiting_host"
    assert status["last_admission"]["allowed"] is False
    assert status["capsules"]["categorized_batch_wave_v1"]["status"] == "complete"
    assert status["capsules"]["full_generations_wave_v1"]["status"] == "pending"
    assert starts == []
    assert (
        chain.validate_full_generations_extension_status(
            status,
            repo_root=tmp_path,
            predecessor_program_path=owner.predecessor_program_path,
            target_program_path=owner.target_program_path,
        )
        == "waiting_host"
    )


def test_clean_predecessor_and_two_host_gates_start_target_once(tmp_path: Path) -> None:
    admissions: list[str] = []
    starts: list[str] = []

    def admission(*_args: Any) -> Mapping[str, Any]:
        admissions.append("admission")
        return {"allowed": True, "reason": "host is clear"}

    owner, _, target = _fixture(
        tmp_path,
        predecessor_state="complete",
        host_admission=admission,
        target_starter=lambda *_args, **_kwargs: (
            starts.append("start")
            or {
                "launched_pid": 202,
                "status": _generic_status(target, state="running", pid=202),
            }
        ),
    )

    status = owner.tick()

    assert admissions == ["admission", "admission"]
    assert starts == ["start"]
    assert status["state"] == "waiting_target"
    target_row = status["capsules"]["full_generations_wave_v1"]
    assert target_row["status"] == "running"
    assert target_row["attempts"] == 1
    assert target_row["launched_pid"] == 202


def test_predecessor_change_after_launch_intent_fails_closed(tmp_path: Path) -> None:
    starts: list[str] = []
    holder: dict[str, Any] = {}

    def admission(*_args: Any) -> Mapping[str, Any]:
        predecessor = holder["predecessor"]
        changed = _generic_status(predecessor, state="complete", pid=303)
        _write_json(predecessor.status_path, changed)
        return {"allowed": True, "reason": "host is clear"}

    owner, predecessor, _ = _fixture(
        tmp_path,
        predecessor_state="complete",
        host_admission=admission,
        target_starter=lambda *_args, **_kwargs: starts.append("start") or {},
    )
    holder["predecessor"] = predecessor

    status = owner.tick()

    assert status["state"] == "integrity_hold"
    assert starts == []
    assert any("changed across full-generations launch intent" in problem for problem in status["problems"])


def test_live_target_is_observed_without_rechecking_or_restarting(tmp_path: Path) -> None:
    admissions: list[str] = []
    starts: list[str] = []
    snapshot = chain.ProcessSnapshot(
        pid=202,
        create_time=202.5,
        pgid=202,
        cwd=str(tmp_path),
        label=f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",
        command=("mop",),
    )
    owner, _, _ = _fixture(
        tmp_path,
        predecessor_state="complete",
        target_status="running",
        host_admission=lambda *_args: admissions.append("admission") or {"allowed": True},
        target_starter=lambda *_args, **_kwargs: starts.append("start") or {},
        identity_probe=lambda _identity: "alive",
        process_table=lambda: (snapshot,),
    )

    status = owner.tick()

    assert status["state"] == "waiting_target"
    assert status["capsules"]["full_generations_wave_v1"]["status"] == "running"
    assert admissions == []
    assert starts == []


def test_live_target_command_is_observed_without_process_label(tmp_path: Path) -> None:
    admissions: list[str] = []
    starts: list[str] = []
    holder: dict[str, g1.Program] = {}

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        target = holder["target"]
        return (
            chain.ProcessSnapshot(
                pid=202,
                create_time=202.5,
                pgid=202,
                cwd=str(tmp_path),
                label=str(tmp_path / ".venv/bin/python"),
                command=(
                    str(tmp_path / ".venv/bin/python"),
                    str(tmp_path / "scripts/mop_generation1_campaign.py"),
                    "run",
                    "--program",
                    str(target.path),
                    "--execute",
                ),
            ),
        )

    owner, _, target = _fixture(
        tmp_path,
        predecessor_state="complete",
        target_status="running",
        host_admission=lambda *_args: admissions.append("admission") or {"allowed": True},
        target_starter=lambda *_args, **_kwargs: starts.append("start") or {},
        identity_probe=lambda _identity: "alive",
        process_table=process_table,
    )
    holder["target"] = target

    status = owner.tick()

    assert status["state"] == "waiting_target"
    assert status["capsules"]["full_generations_wave_v1"]["status"] == "running"
    assert admissions == []
    assert starts == []


def test_complete_target_rejects_conflicting_visible_supervisor(tmp_path: Path) -> None:
    conflicting = chain.ProcessSnapshot(
        pid=303,
        create_time=303.5,
        pgid=303,
        cwd=str(tmp_path),
        label=f"mop-supervisor:{chain.TARGET_PROGRAM_ID}",
        command=("mop",),
    )
    owner, _, _ = _fixture(
        tmp_path,
        predecessor_state="complete",
        target_status="complete",
        host_admission=lambda *_args: {"allowed": True},
        target_starter=lambda *_args, **_kwargs: {},
        process_table=lambda: (conflicting,),
    )

    status = owner.tick()

    assert status["state"] == "integrity_hold"
    assert any(
        "completed full-generations target conflicts with visible supervisor" in problem
        for problem in status["problems"]
    )


def test_complete_generic_status_replays_artifacts(tmp_path: Path) -> None:
    _, predecessor = _program_manifest(
        tmp_path,
        program_id=chain.PREDECESSOR_PROGRAM_ID,
        manifest_name="generation1_successor_categorized_batch_wave_v1.json",
    )
    status = _generic_status(predecessor, state="complete", pid=101)
    assert (
        chain.validate_generic_status(
            predecessor,
            status,
            require_complete=True,
        )
        == "complete"
    )

    forged = copy.deepcopy(status)
    capsule_id = predecessor.capsules[0].capsule_id
    forged["capsules"][capsule_id]["artifacts"][0]["sha256"] = "0" * 64
    forged_core = dict(forged)
    forged_core.pop("status_sha256")
    forged["status_sha256"] = g1.canonical_sha256(forged_core)

    try:
        chain.validate_generic_status(predecessor, forged, require_complete=True)
    except chain.FullGenerationsExtensionRefused as exc:
        assert "artifact report drifted" in str(exc)
    else:
        raise AssertionError("forged artifact hash was accepted")


def test_repeated_detached_start_returns_same_live_parent_without_spawn(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    root = tmp_path / "waiter"
    _write_json(root / chain.STATUS_FILE, {"placeholder": True})
    status = {
        "state": "waiting_predecessor",
        "supervisor": {"pid": 404, "create_time": 404.5},
    }
    visible = chain.ProcessSnapshot(
        pid=404,
        create_time=404.5,
        pgid=404,
        cwd=str(chain.REPO_ROOT),
        label=chain.PARENT_LABEL,
        command=("mop",),
    )
    monkeypatch.setattr(
        chain,
        "read_full_generations_extension_status",
        lambda _root: status,
    )
    monkeypatch.setattr(
        chain,
        "validate_full_generations_extension_status",
        lambda _status: "waiting_predecessor",
    )
    monkeypatch.setattr(chain, "_visible_parent", lambda _entrypoint: visible)
    monkeypatch.setattr(chain, "_default_identity_probe", lambda _identity: "alive")
    monkeypatch.setattr(
        chain.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repeat start attempted to spawn another parent")
        ),
    )

    first = chain.start_full_generations_extension_detached(
        root=root,
        execute=True,
        use_caffeinate=False,
    )
    second = chain.start_full_generations_extension_detached(
        root=root,
        execute=True,
        use_caffeinate=False,
    )

    assert first == {"already_running": True, "status": status}
    assert second == first


def test_fresh_detached_start_binds_acknowledged_visible_parent_not_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "waiter"
    status = {
        "state": "waiting_predecessor",
        "supervisor": {"pid": 405, "create_time": 405.5},
    }
    visible = chain.ProcessSnapshot(
        pid=405,
        create_time=405.5,
        pgid=404,
        cwd=str(chain.REPO_ROOT),
        label=chain.PARENT_LABEL,
        command=("mop",),
    )
    visible_calls = iter((None, visible))

    class FakeProcess:
        pid = 404

        @staticmethod
        def poll() -> None:
            return None

    def popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        _write_json(root / chain.STATUS_FILE, {"ack": True})
        return FakeProcess()

    monkeypatch.setattr(
        chain,
        "read_full_generations_extension_status",
        lambda _root: status,
    )
    monkeypatch.setattr(
        chain,
        "validate_full_generations_extension_status",
        lambda _status: "waiting_predecessor",
    )
    monkeypatch.setattr(chain, "_visible_parent", lambda _entrypoint: next(visible_calls))
    monkeypatch.setattr(chain, "_default_identity_probe", lambda _identity: "alive")
    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    result = chain.start_full_generations_extension_detached(
        root=root,
        execute=True,
        use_caffeinate=True,
    )

    assert result["launched_pid"] == 404
    assert result["observed_process"] == visible.identity()
    assert result["status"] == status


def test_fresh_detached_start_rejects_unproven_visible_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "waiter"
    status = {
        "state": "waiting_predecessor",
        "supervisor": {"pid": 405, "create_time": 405.5},
    }
    visible = chain.ProcessSnapshot(
        pid=405,
        create_time=405.5,
        pgid=404,
        cwd=str(chain.REPO_ROOT),
        label=chain.PARENT_LABEL,
        command=("mop",),
    )
    visible_calls = iter((None, visible))

    class FakeProcess:
        pid = 404

        @staticmethod
        def poll() -> None:
            return None

    def popen(*_args: Any, **_kwargs: Any) -> FakeProcess:
        _write_json(root / chain.STATUS_FILE, {"ack": True})
        return FakeProcess()

    monkeypatch.setattr(
        chain,
        "read_full_generations_extension_status",
        lambda _root: status,
    )
    monkeypatch.setattr(
        chain,
        "validate_full_generations_extension_status",
        lambda _status: "waiting_predecessor",
    )
    monkeypatch.setattr(chain, "_visible_parent", lambda _entrypoint: next(visible_calls))
    monkeypatch.setattr(chain, "_default_identity_probe", lambda _identity: "gone")
    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    with pytest.raises(
        chain.FullGenerationsExtensionRefused,
        match="not provably alive",
    ):
        chain.start_full_generations_extension_detached(
            root=root,
            execute=True,
            use_caffeinate=False,
        )


def test_partial_category_route_neither_reruns_nor_resurrects_failed_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wave = pytest.importorskip(
        "mop.studies.generation1_full_generations_wave",
        reason="full-generations wave module is delivered by a parallel build",
    )
    required = (
        "CATEGORY_IDS",
        "CATEGORY_LANES",
        "_routing_for_epoch",
        "_load_receipts",
        "_execute_pending",
        "_build_category_result",
        "validate_category",
        "atomic_write_json",
        "run_category",
        "_read_object",
        "_validated_context_for_epoch",
        "_WaveValidationContext",
        "_validate_category_with_context",
        "_binding",
        "build_classification",
    )
    missing = [name for name in required if not hasattr(wave, name)]
    if missing:
        pytest.skip(f"full-generations wave module lacks mirrored surface: {missing}")
    partial_route = {
        "continue": True,
        "eligible_lane_ids": ["G1-E1"],
        "pruned_lane_ids": ["G1-C0"],
        "reason": "all_retained_lanes_completed_mechanics_ok",
    }
    routing = {
        category_id: {
            "continue": False,
            "eligible_lane_ids": [],
            "pruned_lane_ids": list(wave.CATEGORY_LANES[category_id]),
            "reason": "null_safe_pruning_after_route_or_mechanics_warning",
        }
        for category_id in wave.CATEGORY_IDS
    }
    routing["formation_trace"] = partial_route
    captured: list[tuple[Any, ...]] = []
    monkeypatch.setattr(
        wave,
        "_routing_for_epoch",
        lambda _root, _epoch: ({"kind": "prior"}, routing),
    )
    monkeypatch.setattr(
        wave,
        "_load_receipts",
        lambda _root, _works: {},
    )

    def execute(_root: Path, works: tuple[Any, ...]):
        captured.append(works)
        return {}, {}, 0

    monkeypatch.setattr(wave, "_execute_pending", execute)
    monkeypatch.setattr(
        wave,
        "_build_category_result",
        lambda **_kwargs: {"schema": "test-partial-route"},
    )
    monkeypatch.setattr(wave, "validate_category", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(wave, "atomic_write_json", lambda *_args, **_kwargs: None)

    wave.run_category(
        root=tmp_path,
        epoch_index=1,
        category_id="formation_trace",
    )

    assert len(captured) == 1
    assert captured[0]
    assert {mechanics.WORK_ITEMS[work.source_index].lane_id for work in captured[0]} == {"G1-E1"}

    category_payloads = {
        category_id: {
            "lane_results": {
                lane_id: {
                    "continue_lane": (category_id == "formation_trace" and lane_id in {"G1-C0", "G1-E1"})
                }
                for lane_id in wave.CATEGORY_LANES[category_id]
            },
            "execution": {
                "executed_item_count": 0,
                "skipped_item_count": 0,
                "observed_seconds": 0.0,
            },
        }
        for category_id in wave.CATEGORY_IDS
    }

    def read_for_classifier(path: Path) -> dict[str, Any]:
        return category_payloads[path.stem]

    monkeypatch.setattr(wave, "_read_object", read_for_classifier)
    monkeypatch.setattr(
        wave,
        "_validated_context_for_epoch",
        lambda *_args, **_kwargs: wave._WaveValidationContext(
            predecessor_binding={"kind": "prior"},
            routing=routing,
            parent_classification_sha256="1" * 64,
        ),
    )
    monkeypatch.setattr(
        wave,
        "_validate_category_with_context",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        wave,
        "_binding",
        lambda *_args, **_kwargs: {
            "path": "test.json",
            "file_sha256": "2" * 64,
            "category_sha256": "3" * 64,
        },
    )
    with pytest.raises(ValueError, match="attempted lane resurrection"):
        wave.build_classification(root=tmp_path, epoch_index=1)
