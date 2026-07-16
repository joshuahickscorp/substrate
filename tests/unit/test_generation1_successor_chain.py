from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_chain as chain

NOW = dt.datetime(2026, 7, 16, 1, 0, tzinfo=dt.UTC)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _validator(payload: Mapping[str, Any]) -> None:
    core = {key: value for key, value in payload.items() if key != "result_sha256"}
    if payload.get("result_sha256") != chain.canonical_sha256(core):
        raise ValueError("result seal drifted")
    if payload.get("complete") is not True or payload.get("problems") != []:
        raise ValueError("result is incomplete")


def _spec(tmp_path: Path, stage_id: str, label: str) -> chain.LegacySpec:
    program_id = f"program-{stage_id}"
    return chain.LegacySpec(
        stage_id=stage_id,
        program_id=program_id,
        process_label=label,
        child_label_prefixes=(f"{label}-child-",),
        status_path=tmp_path / "runs" / program_id / "current_status.json",
        status_schema=f"status-{stage_id}/v1",
        result_path=tmp_path / "proof" / f"{stage_id}.json",
        result_schema=f"result-{stage_id}/v1",
        restart_command=("python", f"scripts/{stage_id}.py"),
        result_validator=_validator,
    )


def _status(spec: chain.LegacySpec, state: str = "running") -> dict[str, Any]:
    core = {
        "schema": spec.status_schema,
        "program_id": spec.program_id,
        "state": state,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": chain.canonical_sha256(core)}


def _result(spec: chain.LegacySpec) -> dict[str, Any]:
    core = {
        "schema": spec.result_schema,
        "program_id": spec.program_id,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "result_sha256": chain.canonical_sha256(core)}


def _process(
    tmp_path: Path,
    spec: chain.LegacySpec,
    pid: int,
    *,
    label: str | None = None,
    pgid: int | None = None,
    cwd: Path | None = None,
    command: tuple[str, ...] | None = None,
    ppid: int = 1,
) -> chain.ProcessSnapshot:
    actual_label = label or spec.process_label
    return chain.ProcessSnapshot(
        pid=pid,
        create_time=float(pid) + 0.5,
        pgid=pid if pgid is None else pgid,
        cwd=str((cwd or tmp_path).resolve()),
        label=actual_label,
        command=command or (actual_label, ""),
        ppid=ppid,
    )


def _tracker(tmp_path: Path, pid: int, pgid: int) -> chain.ProcessSnapshot:
    python = str(tmp_path / ".venv/bin/python")
    return chain.ProcessSnapshot(
        pid=pid,
        create_time=float(pid) + 0.5,
        pgid=pgid,
        cwd=str(tmp_path.resolve()),
        label=python,
        command=(python, "-c", "from multiprocessing.resource_tracker import main;main(8)"),
        ppid=pgid,
    )


def _ensure_horizon_program(path: Path) -> None:
    if path.exists():
        return
    core = {
        "schema": "mop-generation1-program/v1",
        "program_id": "generation1-successor-horizon-v1",
    }
    _write(path, {**core, "program_sha256": chain.canonical_sha256(core)})


def _parent(
    tmp_path: Path,
    specs: tuple[chain.LegacySpec, ...],
    **kwargs: Any,
) -> chain.SuccessorEvidenceChain:
    horizon_path = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    _ensure_horizon_program(horizon_path)
    return chain.SuccessorEvidenceChain(
        root=tmp_path / "runs/generation1" / chain.CHAIN_ID,
        repo_root=tmp_path,
        horizon_program_path=horizon_path,
        specs=specs,
        now_fn=lambda: NOW,
        sleep_fn=lambda _seconds: None,
        **kwargs,
    )


def test_adopts_all_three_exact_live_parents_and_writes_immutable_receipts(
    tmp_path: Path,
) -> None:
    specs = (
        _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue"),
        _spec(tmp_path, "legacy_successor_mechanics", "mop-successor-mechanics-queue"),
        _spec(tmp_path, "legacy_final", "mop-final-campaign"),
    )
    for spec in specs:
        _write(spec.status_path, _status(spec, "waiting" if spec.stage_id == "legacy_final" else "running"))
    processes = tuple(_process(tmp_path, spec, 100 + index) for index, spec in enumerate(specs))
    launches: list[list[str]] = []

    def popen(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        launches.append(command)
        return SimpleNamespace(pid=999)

    parent = _parent(
        tmp_path,
        specs,
        execute=True,
        process_table_fn=lambda: processes,
        identity_probe_fn=lambda _identity: "alive",
        popen_fn=popen,
    )

    first = parent.tick()
    second = parent.tick()

    assert first["state"] == second["state"] == "waiting_legacy"
    assert launches == []
    assert first["supervisor"]["pid"] > 0
    assert first["counts"] == {"complete": 0, "remaining": 4, "total": 4}
    receipts = sorted((parent.root / "adoptions").glob("*/*.json"))
    assert len(receipts) == 3
    for path in receipts:
        payload = json.loads(path.read_text(encoding="utf-8"))
        declared = payload.pop("receipt_sha256")
        assert declared == chain.canonical_sha256(payload)
        assert payload["process"]["pid"] == payload["process"]["pgid"]
        assert payload["process"]["cwd"] == str(tmp_path)
        assert payload["policy"] == {
            "observe_only": True,
            "restart_only_after_exact_absence": True,
            "signals_allowed": False,
        }


def test_valid_terminal_proof_completes_without_process_or_restart(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_final", "mop-final-campaign")
    payload = _result(spec)
    _write(spec.result_path, payload)
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (),
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, ())

    assert row["status"] == "complete"
    assert row["returncode"] == 0
    assert row["artifacts"][0]["sha256"] == chain.sha256_file(spec.result_path)
    assert launches == []


def test_existing_immutable_receipt_is_recovered_after_state_link_crash(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    _write(spec.status_path, _status(spec))
    process = _process(tmp_path, spec, 321)
    first = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: (process,))
    first_row = first.state["capsules"][spec.stage_id]
    first._write_adoption_receipt(spec, process, _status(spec), first_row)
    receipt_path = next((first.root / "adoptions").glob("*/*.json"))
    original_bytes = receipt_path.read_bytes()

    recovered = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: (process,))
    status = recovered.tick()

    assert status["state"] == "waiting_legacy"
    assert receipt_path.read_bytes() == original_bytes
    assert recovered.state["capsules"][spec.stage_id]["process"]["pid"] == process.pid
    assert recovered.state["capsules"][spec.stage_id]["adoption_receipts"] == [
        str(receipt_path.relative_to(tmp_path))
    ]


def test_absent_queue_persists_intent_then_launches_only_once_inside_grace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    launches: list[tuple[list[str], dict[str, Any]]] = []
    signals: list[object] = []
    monkeypatch.setattr(chain.os, "kill", lambda *args: signals.append(args), raising=False)

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append((command, kwargs))
        return SimpleNamespace(pid=4321)

    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (),
        identity_probe_fn=lambda _identity: "gone",
        popen_fn=popen,
    )
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, ())
    parent._reconcile_legacy(spec, row, ())
    parent._reconcile_legacy(spec, row, ())

    assert len(launches) == 1
    assert launches[0][0] == list(spec.restart_command)
    assert launches[0][1]["start_new_session"] is True
    assert launches[0][1]["env"]["MOP_PROCESS_LABEL"] == spec.process_label
    assert row["status"] == "adoption_wait"
    assert row["attempts"] == 1
    assert signals == []


def test_orphan_residual_worker_waits_and_never_launches(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    residual = _process(
        tmp_path,
        spec,
        777,
        label=f"{spec.process_label}-child-r001",
    )
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (residual,),
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (residual,))

    assert row["status"] == "adoption_wait"
    assert "residual" in row["last_problem"]
    assert launches == []


def test_orphan_resource_tracker_waits_and_never_launches(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    tracker = _tracker(tmp_path, 780, 777)
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (tracker,),
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (tracker,))

    assert row["status"] == "adoption_wait"
    assert str(tracker.pid) in row["last_problem"]
    assert launches == []


def test_exact_resource_tracker_is_allowed_inside_adopted_parent_group(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    process = _process(tmp_path, spec, 790)
    tracker = _tracker(tmp_path, 791, process.pid)
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (process, tracker),
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"


def test_exact_prelabel_spawn_worker_is_allowed_inside_adopted_parent_group(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, "legacy_successor_mechanics", "mop-successor-mechanics-queue")
    process = _process(tmp_path, spec, 800)
    python = str(tmp_path / ".venv/bin/python")
    spawn_worker = _process(
        tmp_path,
        spec,
        801,
        label=python,
        pgid=process.pid,
        ppid=process.pid,
        command=(
            python,
            "-c",
            (
                "from multiprocessing.spawn import spawn_main; "
                "spawn_main(tracker_fd=9, pipe_handle=16)"
            ),
            "--multiprocessing-fork",
        ),
    )
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (process, spawn_worker),
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"


def test_exact_spawn_worker_owned_by_other_parent_does_not_cross_contaminate(
    tmp_path: Path,
) -> None:
    d1 = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    mechanics = _spec(
        tmp_path,
        "legacy_successor_mechanics",
        "mop-successor-mechanics-queue",
    )
    d1_parent = _process(tmp_path, d1, 802)
    mechanics_parent = _process(tmp_path, mechanics, 803)
    python = str(tmp_path / ".venv/bin/python")
    mechanics_spawn = _process(
        tmp_path,
        mechanics,
        804,
        label=python,
        pgid=mechanics_parent.pid,
        ppid=mechanics_parent.pid,
        command=(
            python,
            "-c",
            (
                "from multiprocessing.spawn import spawn_main; "
                "spawn_main(tracker_fd=9, pipe_handle=16)"
            ),
            "--multiprocessing-fork",
        ),
    )
    parent = _parent(
        tmp_path,
        (d1, mechanics),
        execute=True,
        process_table_fn=lambda: (d1_parent, mechanics_parent, mechanics_spawn),
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][d1.stage_id]["status"] == "adopted"
    assert status["capsules"][mechanics.stage_id]["status"] == "adopted"


def test_direct_spawn_worker_outside_parent_group_enters_integrity_hold(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path, "legacy_successor_mechanics", "mop-successor-mechanics-queue")
    process = _process(tmp_path, spec, 805)
    python = str(tmp_path / ".venv/bin/python")
    misgrouped_spawn = _process(
        tmp_path,
        spec,
        806,
        label=python,
        pgid=999,
        ppid=process.pid,
        command=(
            python,
            "-c",
            (
                "from multiprocessing.spawn import spawn_main; "
                "spawn_main(tracker_fd=9, pipe_handle=16)"
            ),
            "--multiprocessing-fork",
        ),
    )
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (process, misgrouped_spawn),
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "escaped the exact parent process group" in status["problems"][-1]


def test_resource_tracker_owned_by_other_exact_parent_does_not_block_restart(
    tmp_path: Path,
) -> None:
    missing = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    other = _spec(tmp_path, "legacy_final", "mop-final-campaign")
    other_parent = _process(tmp_path, other, 810)
    tracker = _tracker(tmp_path, 811, other_parent.pid)
    launches: list[object] = []

    def popen(*args: Any, **kwargs: Any) -> SimpleNamespace:
        launches.append((args, kwargs))
        return SimpleNamespace(pid=900)

    parent = _parent(
        tmp_path,
        (missing, other),
        execute=True,
        process_table_fn=lambda: (other_parent, tracker),
        popen_fn=popen,
    )
    row = parent.state["capsules"][missing.stage_id]

    parent._reconcile_legacy(missing, row, (other_parent, tracker))

    assert row["status"] == "launching"
    assert len(launches) == 1


def test_unexpected_same_group_child_enters_integrity_hold(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    process = _process(tmp_path, spec, 820)
    unexpected = _process(
        tmp_path,
        spec,
        821,
        label="unrecognized-child",
        pgid=process.pid,
        command=("unrecognized-child",),
    )
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (process, unexpected),
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "unexpected members" in status["problems"][-1]


def test_prelabelled_restart_command_is_treated_as_residual(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    residual = _process(
        tmp_path,
        spec,
        778,
        label="python",
        command=("python", str(tmp_path / spec.restart_command[1])),
    )
    parent = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: (residual,))
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (residual,))

    assert row["status"] == "adoption_wait"


@pytest.mark.parametrize(
    "processes, expected",
    [
        ("multiple", "multiple exact parent processes"),
        ("wrong_cwd", "invalid cwd"),
        ("not_leader", "not its process-group leader"),
    ],
)
def test_ambiguous_or_inexact_parent_enters_integrity_hold(
    tmp_path: Path,
    processes: str,
    expected: str,
) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    first = _process(tmp_path, spec, 100)
    table: tuple[chain.ProcessSnapshot, ...]
    if processes == "multiple":
        table = (first, _process(tmp_path, spec, 101))
    elif processes == "wrong_cwd":
        table = (_process(tmp_path, spec, 100, cwd=tmp_path / "other"),)
    else:
        table = (_process(tmp_path, spec, 100, pgid=99),)
    parent = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert expected in status["problems"][-1]


def test_inexact_previously_adopted_identity_holds_without_restart(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (),
        identity_probe_fn=lambda _identity: "unknown",
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    parent.state["capsules"][spec.stage_id]["process"] = {
        "pid": 99,
        "create_time": 10.0,
        "pgid": 99,
        "cwd": str(tmp_path),
        "label": spec.process_label,
    }

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "previously adopted identity is now inexact" in status["problems"][-1]
    assert launches == []


def test_different_visible_parent_cannot_replace_a_live_adopted_identity(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    old_process = _process(tmp_path, spec, 98)
    new_process = _process(tmp_path, spec, 99)
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (new_process,),
        identity_probe_fn=lambda _identity: "alive",
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    parent.state["capsules"][spec.stage_id]["process"] = old_process.identity()

    first = parent.tick()
    parent.process_table_fn = lambda: ()
    second = parent.tick()

    assert first["state"] == second["state"] == "integrity_hold"
    assert "prior adopted identity is alive" in first["problems"][-1]
    assert launches == []


def test_horizon_start_intent_prevents_duplicate_supervisor_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, "legacy_final", "mop-final-campaign")
    status_path = tmp_path / "runs/generation1/generation1-successor-horizon-v1/current_status.json"
    program = SimpleNamespace(
        program_id="generation1-successor-horizon-v1",
        status_path=status_path,
    )
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    starts: list[str] = []

    def start(program_value: Any, **_kwargs: Any) -> dict[str, Any]:
        starts.append(program_value.program_id)
        return {
            "launched_pid": 909,
            "status": {"supervisor": {"pid": 909, "create_time": 90.9}},
        }

    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (),
        identity_probe_fn=lambda identity: "alive" if identity.get("pid") == 909 else "gone",
        supervisor_start_fn=start,
    )
    program.program_sha256 = parent.state["horizon_program"]["program_sha256"]
    row = parent.state["capsules"]["successor_horizon"]

    parent._reconcile_horizon(row, ())
    parent._reconcile_horizon(row, ())

    assert starts == ["generation1-successor-horizon-v1"]
    assert row["status"] == "adoption_wait"
    assert "alive" in row["last_problem"]


def test_horizon_manifest_authority_is_bound_and_valid_change_holds(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_final", "mop-final-campaign")
    launches: list[object] = []
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (),
        popen_fn=lambda *args, **kwargs: launches.append((args, kwargs)),
    )
    bound = dict(parent.state["horizon_program"])
    program_path = tmp_path / bound["path"]
    assert bound["file_sha256"] == chain.sha256_file(program_path)

    changed_core = {
        "schema": "mop-generation1-program/v1",
        "program_id": "generation1-successor-horizon-v1",
        "revision": 2,
    }
    _write(
        program_path,
        {**changed_core, "program_sha256": chain.canonical_sha256(changed_core)},
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "horizon program authority drifted" in status["problems"][-1]
    assert launches == []
    with pytest.raises(chain.SuccessorChainRefused, match="horizon program authority drifted"):
        _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: ())


def test_parent_implementation_authority_drift_holds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, "legacy_final", "mop-final-campaign")
    parent = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: ())
    bound = parent.state["parent_implementation"]
    assert bound["sha256"] == chain.sha256_file(Path(chain.__file__).resolve())
    original_sha256 = chain.sha256_file

    def drifted_sha256(path: Path) -> str:
        if path.resolve() == Path(chain.__file__).resolve():
            return "0" * 64
        return original_sha256(path)

    monkeypatch.setattr(chain, "sha256_file", drifted_sha256)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "parent implementation authority drifted" in status["problems"][-1]


def test_stop_is_a_sealed_drain_request_and_never_signals_adopted_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    process = _process(tmp_path, spec, 123)
    signals: list[object] = []
    monkeypatch.setattr(chain.os, "kill", lambda *args: signals.append(args), raising=False)
    parent = _parent(
        tmp_path,
        (spec,),
        execute=True,
        process_table_fn=lambda: (process,),
    )
    payload = chain.request_stop(parent.root, "test drain")

    status = parent.tick()

    assert payload["adopted_process_action"] == "none"
    assert status["state"] == "drained"
    assert signals == []


def test_global_chain_lock_is_nonblocking(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "legacy_d1", "mop-d1-frozen-queue")
    parent = _parent(tmp_path, (spec,), execute=True, process_table_fn=lambda: ())

    with (
        chain.FileLock(parent.root / chain.LOCK_FILE),
        pytest.raises(chain.Generation1Refused, match="lock is already held"),
    ):
        parent.run(max_cycles=1)


def test_cli_has_single_start_execute_surface() -> None:
    arguments = chain.build_parser().parse_args(["start", "--execute", "--no-caffeinate"])
    assert arguments.command == "start"
    assert arguments.execute is True
    assert arguments.no_caffeinate is True


def test_direct_cli_help_bootstraps_outside_repository(tmp_path: Path) -> None:
    script = chain.REPO_ROOT / "scripts/mop_generation1_successor_chain.py"

    completed = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "successor evidence chain" in completed.stdout


def test_detached_start_observes_visible_parent_and_never_launches_duplicate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    visible = chain.ProcessSnapshot(
        pid=555,
        create_time=55.5,
        pgid=555,
        cwd=str(chain.REPO_ROOT.resolve()),
        label=chain.PARENT_LABEL,
        command=(chain.PARENT_LABEL,),
        ppid=1,
    )
    launches: list[object] = []
    monkeypatch.setattr(chain, "_default_process_table", lambda: (visible,))
    monkeypatch.setattr(
        chain.subprocess,
        "Popen",
        lambda *args, **kwargs: launches.append((args, kwargs)),
    )

    result = chain.start_chain_detached(root=tmp_path / "chain", execute=True)

    assert result["already_running"] is True
    assert result["observed_process"]["pid"] == visible.pid
    assert launches == []
