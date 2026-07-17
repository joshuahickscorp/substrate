"""Unit tests for the single-process general-run orchestrator.

The suite drives the ordered stage machine with injected collaborators (process
table, program loader, status reader, target starter, host admission, and the
advisory reprofiler) exactly like the v7 and categorized-chain tests.  It proves
that observe_legacy reuses the v7 robust hardening (a detached transient worker
that exits is tolerated; a persistent foreign worker still holds), that a stage
advances only on a stable clean-complete, that a running or already-launched
supervisor is never double-launched, that the machine is idempotent and
restartable mid-stage, and that a held or failed stage stops advancement.
"""

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


# ----------------------------------------------------------------------------
# Shared builders: sealed compute manifests and generic statuses (real files so
# load_program, validate_generic_status, and _stable_complete_generic_status all
# run against genuine artifacts, mirroring the categorized-chain test harness).
# ----------------------------------------------------------------------------


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


# ----------------------------------------------------------------------------
# Shared builders: lightweight legacy specs and their status/result files.
# ----------------------------------------------------------------------------


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


def _legacy_status(spec: legacy_chain.LegacySpec, *, state: str = "running") -> None:
    core = {
        "schema": spec.status_schema,
        "program_id": spec.program_id,
        "state": state,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write_json(spec.status_path, {**core, "status_sha256": g1.canonical_sha256(core)})


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
    # The horizon-v1 manifest doubles as the observe-only legacy owner's bound
    # horizon authority, so it must always exist before construction.
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
    """Move the orchestrator to a compute stage with prior stages marked complete."""

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


# ----------------------------------------------------------------------------
# observe_legacy: reuse the v7 legacy adoption path and robust hardening exactly
# ----------------------------------------------------------------------------


def test_observe_legacy_adopts_live_final_and_stays_observing(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    _legacy_status(spec, state="running")
    live = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    calls: list[str] = []
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (live,),
        identity_probe_fn=lambda _identity: "alive",
        target_starter_fn=lambda program, **_k: calls.append(program.program_id) or {},
    )

    status = orchestrator.tick()

    assert status["state"] == "observe_legacy"
    assert status["stage"] == "observe_legacy"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "adopted"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["adoption_receipts"]
    assert calls == []


def test_observe_legacy_completes_and_advances_to_horizon_v1(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    _legacy_result(spec)
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
    # advancing the pointer does not launch anything on the same tick
    assert calls == []


def test_observe_legacy_tolerates_detached_transient_worker_regression(tmp_path: Path) -> None:
    # THE v6 regression, preserved as fixed in v7: a legacy worker mid-exit,
    # already reparented to launchd (ppid 1, its own pgid), still carrying the
    # child label and the repo cwd, is tolerated because it is GONE on the next
    # sample. The orchestrator must adopt cleanly, never fail to integrity_hold.
    spec = _legacy_spec(tmp_path)
    _legacy_status(spec, state="running")
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    detached = _process(
        tmp_path, pid=33846, label=worker_label, command=(worker_label, ""), pgid=33846, ppid=1
    )
    assert detached.ppid != owner.pid and detached.pgid != owner.pid  # genuinely detached
    tables = [(owner, detached), (owner,), (owner,)]
    calls = 0
    sleeps: list[float] = []

    def process_table() -> tuple[gr.ProcessSnapshot, ...]:
        nonlocal calls
        table = tables[min(calls, len(tables) - 1)]
        calls += 1
        return table

    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=process_table,
        sleep_fn=sleeps.append,
        identity_probe_fn=lambda identity: "gone" if identity.get("pid") == detached.pid else "alive",
    )

    status = orchestrator.tick()

    assert status["state"] == "observe_legacy"
    assert status["state"] != "integrity_hold"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert calls == 3
    assert sleeps == [
        legacy_chain.PROCESS_TRANSITION_INTERVAL_SECONDS,
        legacy_chain.PROCESS_TRANSITION_INTERVAL_SECONDS,
    ]


def test_observe_legacy_persistent_foreign_worker_holds(tmp_path: Path) -> None:
    # A detached inexact worker that never exits and never resolves is a genuine
    # stuck/foreign process: it stays inexact across the whole bounded window and
    # fails closed to integrity_hold, exactly as the v7 hardening requires.
    spec = _legacy_spec(tmp_path)
    _legacy_status(spec, state="running")
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    detached = _process(
        tmp_path, pid=33846, label=worker_label, command=(worker_label, ""), pgid=33846, ppid=1
    )

    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (owner, detached),
    )

    status = orchestrator.tick()

    assert status["state"] == "integrity_hold"
    assert "did not stabilize within the bounded window" in status["problems"][-1]
    assert str(detached.pid) in status["problems"][-1]


def test_observe_legacy_failure_holds_on_legacy_failure(tmp_path: Path) -> None:
    spec = _legacy_spec(tmp_path)
    _legacy_status(spec, state="failure_hold")
    orchestrator = _orchestrator(tmp_path, legacy_specs=(spec,), process_table_fn=lambda: ())

    status = orchestrator.tick()

    assert status["state"] == "failure_hold"
    assert orchestrator.state["legacy_capsules"][spec.stage_id]["status"] == "failure_hold"
    assert status["finished_at"] is not None


# ----------------------------------------------------------------------------
# Compute stages: launch, observe, and no double-launch
# ----------------------------------------------------------------------------


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
    # A parked waiter (or a prior incarnation) already started the exact
    # supervisor; observe-before-launch must adopt it and never relaunch.
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


# ----------------------------------------------------------------------------
# Whole-machine sequencing, restart, and drain
# ----------------------------------------------------------------------------


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
    # a terminal tick revalidates the final wave without acting further
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
    _legacy_status(spec, state="running")
    live = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    orchestrator = _orchestrator(
        tmp_path,
        legacy_specs=(spec,),
        process_table_fn=lambda: (live,),
        identity_probe_fn=lambda _identity: "alive",
    )

    status = orchestrator.tick()

    assert gr.validate_general_run_status(status, repo_root=tmp_path) == status["state"]
    tampered = {key: value for key, value in status.items() if key != "status_sha256"}
    tampered["signals_allowed"] = True
    resealed = _sealed(tampered, "status_sha256")
    with pytest.raises(gr.GeneralRunRefused, match="identity or safety drifted"):
        gr.validate_general_run_status(resealed, repo_root=tmp_path)
