
from __future__ import annotations

import datetime as dt
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from mop.studio import general_run_orchestrator as gr
from mop.studio import generation1_categorized_batch_extension_chain as categorized_chain
from mop.studio import generation1_successor_chain_v7 as legacy_chain
from mop.studio import generation1_supervisor as g1

NOW = dt.datetime(2026, 7, 17, 12, 0, tzinfo=dt.UTC)
NOW_ISO = NOW.isoformat()


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
    runner.write_text("# immutable general-run test runner\n")
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
                "schema": "mop-general-run-test-artifact/v1",
                "fields": {"ok": True},
                "seal_field": "payload_sha256",
            }
        ],
        "authorities": [{"path": runner.name, "sha256": g1.sha256_file(runner)}],
    }
    capsule = _sealed(capsule_core, "capsule_sha256")
    program_root = f"runs/generation1/{program_id}"
    program_core: dict[str, Any] = {
        "schema": g1.PROGRAM_SCHEMA,
        "program_id": program_id,
        "program_root": program_root,
        "policy": {"path": policy.name, "sha256": g1.sha256_file(policy)},
        "authorities": [{"path": runner.name, "sha256": g1.sha256_file(runner)}],
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


def _generic_status(program: g1.Program, *, state: str, pid: int) -> dict[str, Any]:
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
        artifact = _sealed({"schema": capsule.artifacts[0].schema, "ok": True}, "payload_sha256")
        _write_json(artifact_path, artifact)
        row.update(
            {
                "status": "complete",
                "attempts": 1,
                "finished_at": NOW_ISO,
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
    implementation = categorized_chain._generic_supervisor_authority(program.repo_root)
    core: dict[str, Any] = {
        "schema": g1.STATUS_SCHEMA,
        "program_id": program.program_id,
        "created_at": NOW_ISO,
        "program": {
            "path": str(program.path),
            "file_sha256": program.file_sha256,
            "program_sha256": program.program_sha256,
        },
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
                "base_capsules": [program.capsules[0].capsule_sha256],
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


COMPUTE_MANIFESTS = {
    "generation1-successor-horizon-v1": "generation1_successor_horizon_v1.json",
    "generation1-successor-horizon-v2": "generation1_successor_horizon_v2.json",
    "generation1-successor-categorized-batch-wave-v1": "generation1_successor_categorized_batch_wave_v1.json",
    "generation1-full-generations-wave-v1": "generation1_full_generations_wave_v1.json",
}


def _build_programs(tmp_path: Path) -> dict[str, g1.Program]:
    programs: dict[str, g1.Program] = {}
    for program_id, manifest_name in COMPUTE_MANIFESTS.items():
        _, program = _program_manifest(tmp_path, program_id=program_id, manifest_name=manifest_name)
        programs[program_id] = program
    return programs


def _legacy_validator(payload: Mapping[str, Any]) -> None:
    if payload.get("complete") is not True:
        raise ValueError("incomplete legacy result")


def _legacy_spec(
    tmp_path: Path,
    stage_id: str = "legacy_final",
    *,
    process_label: str = "mop-final-campaign",
    child_prefix: str = "mop-final-",
) -> legacy_chain.LegacySpec:
    return legacy_chain.LegacySpec(
        stage_id=stage_id,
        program_id=f"program-{stage_id}",
        process_label=process_label,
        child_label_prefixes=(child_prefix,),
        status_path=tmp_path / f"runs/{stage_id}/current_status.json",
        status_schema=f"status-{stage_id}/v1",
        result_path=tmp_path / f"proof/{stage_id}.json",
        result_schema=f"result-{stage_id}/v1",
        restart_command=(str(tmp_path / ".venv/bin/python"), "scripts/legacy.py"),
        result_validator=_legacy_validator,
    )


def _legacy_result(spec: legacy_chain.LegacySpec) -> None:
    _write_json(
        spec.result_path,
        {
            "schema": spec.result_schema,
            "program_id": spec.program_id,
            "complete": True,
            "problems": [],
            "activation_allowed": False,
            "scientific_promotion": False,
        },
    )


def _process(
    tmp_path: Path,
    *,
    pid: int,
    label: str,
    command: tuple[str, ...],
    pgid: int | None = None,
    ppid: int = 1,
    cwd: Path | None = None,
    create_time: float | None = None,
) -> gr.ProcessSnapshot:
    return gr.ProcessSnapshot(
        pid=pid,
        create_time=float(pid) + 0.25 if create_time is None else create_time,
        pgid=pid if pgid is None else pgid,
        cwd=str((cwd or tmp_path).resolve()),
        label=label,
        command=command,
        ppid=ppid,
    )


def _orchestrator(
    tmp_path: Path,
    *,
    legacy_specs: Any,
    execute: bool = True,
    process_table_fn: Any = None,
    identity_probe_fn: Any = None,
    target_starter_fn: Any = None,
    host_admission_fn: Any = None,
    reprofiler_fn: Any = None,
    sleep_fn: Any = None,
) -> gr.GeneralRunOrchestrator:
    horizon = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    if not horizon.exists():
        _program_manifest(
            tmp_path,
            program_id="generation1-successor-horizon-v1",
            manifest_name="generation1_successor_horizon_v1.json",
        )
    return gr.GeneralRunOrchestrator(
        root=tmp_path / "runs/generation1/general-run",
        repo_root=tmp_path,
        horizon_program_path=horizon,
        legacy_specs=legacy_specs,
        execute=execute,
        process_table_fn=process_table_fn or (lambda: ()),
        identity_probe_fn=identity_probe_fn or (lambda _identity: "gone"),
        target_starter_fn=target_starter_fn or (lambda *_a, **_k: {}),
        host_admission_fn=host_admission_fn or (lambda _p, _s: {"allowed": True, "reason": "idle"}),
        reprofiler_fn=reprofiler_fn or (lambda: {"schema": gr.REPROFILE_SCHEMA, "recommended_workers": 12}),
        now_fn=lambda: NOW,
        sleep_fn=sleep_fn or (lambda _seconds: None),
    )


def _seed_stage(orchestrator: gr.GeneralRunOrchestrator, stage_id: str, *, complete_before: bool = True) -> None:

    orchestrator.state["stage"] = stage_id
    orchestrator.state["status"] = stage_id
    if not complete_before:
        return
    for spec in orchestrator._legacy_specs:
        orchestrator.state["legacy_capsules"][spec.stage_id]["status"] = "complete"
    target_index = gr.STAGES.index(stage_id)
    for stage in gr.COMPUTE_STAGES:
        if gr.STAGES.index(stage.stage_id) < target_index:
            row = orchestrator.state["compute_capsules"][stage.stage_id]
            row.update({"status": "complete", "returncode": 0, "finished_at": NOW_ISO})


def _census_parent(
    tmp_path: Path,
    *,
    create_time: float = gr.FINAL_CENSUS_CREATE_TIME,
    label: str = gr.FINAL_CENSUS_LABEL,
) -> gr.ProcessSnapshot:

    return _process(
        tmp_path,
        pid=gr.FINAL_CENSUS_PID,
        label=label,
        command=(label, ""),
        create_time=create_time,
    )


def test_observe_legacy_waits_while_pinned_census_parent_alive(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    census = _census_parent(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (census,),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    status = orchestrator.tick()

    assert status["state"] == "observe_legacy"
    assert status["stage"] == "observe_legacy"
    row = orchestrator.state["legacy_capsules"][spec.stage_id]
    assert row["status"] == "adopted"
    assert row["process"]["pid"] == gr.FINAL_CENSUS_PID
    assert status["problems"] == []
    assert calls == []


def test_observe_legacy_advances_when_census_pid_is_free(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    status = orchestrator.tick()

    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "complete"
    assert status["stage"] == "run_horizon_v1"
    assert status["state"] == "run_horizon_v1"
    assert calls == []


def test_observe_legacy_waits_then_advances_as_census_parent_exits(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    table: dict[str, tuple[gr.ProcessSnapshot, ...]] = {"processes": (_census_parent(tmp_path),)}
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: table["processes"],
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    first = orchestrator.tick()
    assert first["state"] == "observe_legacy"
    assert first["stage"] == "observe_legacy"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "adopted"

    table["processes"] = ()
    second = orchestrator.tick()
    assert second["stage"] == "run_horizon_v1"
    assert second["state"] == "run_horizon_v1"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "complete"
    assert calls == []


def test_observe_legacy_holds_on_reused_pid_imposter(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    imposter = _census_parent(tmp_path, create_time=gr.FINAL_CENSUS_CREATE_TIME + 5000.0)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (imposter,),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    status = orchestrator.tick()

    assert status["state"] == "observe_legacy"
    assert status["stage"] == "observe_legacy"
    assert status["state"] != "integrity_hold"
    assert status["problems"] == []
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "adoption_wait"
    assert calls == []


def test_observe_legacy_ignores_churning_child_pool_and_waits(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    census = _census_parent(tmp_path)
    prefix = spec.child_label_prefixes[0]
    children = tuple(
        _process(
            tmp_path,
            pid=90000 + index,
            label=f"{prefix}{index:04d}",
            command=(f"{prefix}{index:04d}", ""),
            pgid=90000 + index,
            ppid=gr.FINAL_CENSUS_PID,
        )
        for index in range(9)
    )
    sleeps: list[float] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (census, *children),
        sleep_fn=sleeps.append,
    )

    status = orchestrator.tick()

    assert status["state"] == "observe_legacy"
    assert status["problems"] == []
    assert sleeps == []
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "adopted"


def test_horizon_v1_launches_once_when_absent_and_records_reprofile(tmp_path: Path) -> None:
    _build_programs(tmp_path)
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []

    def starter(program: g1.Program, *, execute: bool, use_caffeinate: bool = True) -> Mapping[str, Any]:
        calls.append(program.program_id)
        status = _generic_status(program, state="running", pid=555)
        _write_json(program.status_path, status)
        return {"launched_pid": 555, "status": status}

    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=starter,
        reprofiler_fn=lambda: {"schema": gr.REPROFILE_SCHEMA, "recommended_workers": 11},
    )
    _seed_stage(orchestrator, "run_horizon_v1")

    status = orchestrator.tick()

    assert calls == ["generation1-successor-horizon-v1"]
    assert orchestrator.state["compute_capsules"]["run_horizon_v1"]["status"] == "running"
    assert status["state"] == "run_horizon_v1"
    assert status["reprofile"]["recommended_workers"] == 11
    assert status["reprofile"]["advisory"] is True


def test_compute_never_double_launches_a_visible_supervisor(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    program = programs["generation1-successor-horizon-v1"]
    spec = _legacy_spec(tmp_path)
    supervisor = _process(
        tmp_path,
        pid=911,
        label="mop-supervisor:generation1-successor-horizon-v1",
        command=("mop-supervisor:generation1-successor-horizon-v1",),
        pgid=911,
    )
    assert program.program_id == "generation1-successor-horizon-v1"
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (supervisor,),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )
    _seed_stage(orchestrator, "run_horizon_v1")

    orchestrator.tick()

    assert calls == []
    assert orchestrator.state["compute_capsules"]["run_horizon_v1"]["status"] == "adoption_wait"


def test_compute_running_supervisor_is_observed_not_relaunched(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    program = programs["generation1-successor-horizon-v1"]
    _write_json(program.status_path, _generic_status(program, state="running", pid=555))
    supervisor = _process(
        tmp_path,
        pid=555,
        label="mop-supervisor:generation1-successor-horizon-v1",
        command=("mop-supervisor:generation1-successor-horizon-v1",),
        pgid=555,
        create_time=555.5,
    )
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (supervisor,),
        identity_probe_fn=lambda identity: "alive" if identity.get("pid") == 555 else "gone",
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )
    _seed_stage(orchestrator, "run_horizon_v1")

    orchestrator.tick()

    assert calls == []
    assert orchestrator.state["compute_capsules"]["run_horizon_v1"]["status"] == "running"


def test_compute_advances_on_clean_complete(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    program = programs["generation1-successor-horizon-v1"]
    _write_json(program.status_path, _generic_status(program, state="complete", pid=555))
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )
    _seed_stage(orchestrator, "run_horizon_v1")

    status = orchestrator.tick()

    assert orchestrator.state["compute_capsules"]["run_horizon_v1"]["status"] == "complete"
    assert status["stage"] == "run_horizon_v2"
    assert status["state"] == "run_horizon_v2"
    assert calls == []


def test_categorized_launches_under_predecessor_guard_and_admission(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    predecessor = programs["generation1-successor-horizon-v2"]
    _write_json(predecessor.status_path, _generic_status(predecessor, state="complete", pid=222))
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    admissions: list[str] = []

    def starter(program: g1.Program, *, execute: bool, use_caffeinate: bool = True) -> Mapping[str, Any]:
        calls.append(program.program_id)
        status = _generic_status(program, state="running", pid=777)
        _write_json(program.status_path, status)
        return {"launched_pid": 777, "status": status}

    def admission(program: g1.Program, status: Mapping[str, Any] | None) -> Mapping[str, Any]:
        admissions.append(program.program_id)
        return {"allowed": True, "reason": "host idle"}

    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=starter,
        host_admission_fn=admission,
    )
    _seed_stage(orchestrator, "run_categorized_wave")

    status = orchestrator.tick()

    assert calls == ["generation1-successor-categorized-batch-wave-v1"]
    assert admissions == ["generation1-successor-categorized-batch-wave-v1"]
    assert orchestrator.state["compute_capsules"]["run_categorized_wave"]["status"] == "running"
    assert status["last_admission"]["allowed"] is True


def test_categorized_waits_on_host_when_admission_denied(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    predecessor = programs["generation1-successor-horizon-v2"]
    _write_json(predecessor.status_path, _generic_status(predecessor, state="complete", pid=222))
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
        host_admission_fn=lambda _p, _s: {"allowed": False, "reason": "host busy"},
    )
    _seed_stage(orchestrator, "run_categorized_wave")

    status = orchestrator.tick()

    assert calls == []
    assert status["state"] == "waiting_host"
    assert orchestrator.state["compute_capsules"]["run_categorized_wave"]["status"] == "pending"
    assert status["last_admission"]["allowed"] is False


def test_compute_failure_hold_stops_advancement(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    program = programs["generation1-successor-horizon-v1"]
    _write_json(program.status_path, _generic_status(program, state="failure_hold", pid=555))
    spec = _legacy_spec(tmp_path)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )
    _seed_stage(orchestrator, "run_horizon_v1")

    status = orchestrator.tick()

    assert status["state"] == "failure_hold"
    assert status["stage"] == "run_horizon_v1"  # never advanced
    assert orchestrator.state["compute_capsules"]["run_horizon_v1"]["status"] == "failure_hold"
    assert calls == []


def test_full_run_sequences_every_stage_and_never_double_launches(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    for program in programs.values():
        _write_json(program.status_path, _generic_status(program, state="complete", pid=101))
    spec = _legacy_spec(tmp_path)
    _legacy_result(spec)
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    status = orchestrator.run(max_cycles=12)

    assert status["state"] == "complete"
    assert calls == []  # every stage was already clean-complete; nothing relaunched
    for stage in gr.COMPUTE_STAGES:
        assert orchestrator.state["compute_capsules"][stage.stage_id]["status"] == "complete"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "complete"
    assert orchestrator.tick()["state"] == "complete"


def test_restart_resumes_at_sealed_stage_without_relaunch(tmp_path: Path) -> None:
    programs = _build_programs(tmp_path)
    horizon_v2 = programs["generation1-successor-horizon-v2"]
    _write_json(horizon_v2.status_path, _generic_status(horizon_v2, state="complete", pid=222))
    spec = _legacy_spec(tmp_path)

    first = _orchestrator(tmp_path, legacy_specs=(spec,), process_table_fn=lambda: ())
    _seed_stage(first, "run_horizon_v2")
    first._publish()  # seal the mid-stage state to disk

    calls: list[str] = []
    resumed = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (),
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )
    resumed.state = resumed._load_state()

    assert resumed.state["stage"] == "run_horizon_v2"
    assert resumed.state["compute_capsules"]["run_horizon_v1"]["status"] == "complete"

    status = resumed.tick()

    assert calls == []  # horizon-v2 was already complete on disk; never relaunched
    assert resumed.state["compute_capsules"]["run_horizon_v2"]["status"] == "complete"
    assert status["stage"] == "run_categorized_wave"


def test_drain_control_sets_drained(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    orchestrator = _orchestrator(tmp_path, legacy_specs=(spec,), process_table_fn=lambda: ())
    gr.request_stop(root=orchestrator.root, reason="cutover to general-run")

    status = orchestrator.tick()

    assert status["state"] == "drained"
    assert status["finished_at"] is not None


def test_status_validator_accepts_published_and_rejects_tamper(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    live = _census_parent(tmp_path)
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (live,),
    )

    status = orchestrator.tick()

    assert gr.validate_general_run_status(status, repo_root=tmp_path) == status["state"]
    tampered = {key: value for key, value in status.items() if key != "status_sha256"}
    tampered["signals_allowed"] = True
    resealed = _sealed(tampered, "status_sha256")
    with pytest.raises(gr.GeneralRunRefused, match="identity or safety drifted"):
        gr.validate_general_run_status(resealed, repo_root=tmp_path)
