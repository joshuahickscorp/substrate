from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_chain_v5 as chain

NOW = dt.datetime(2026, 7, 16, 18, 0, tzinfo=dt.UTC)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _validator(payload: Mapping[str, Any]) -> None:
    if payload.get("complete") is not True:
        raise ValueError("incomplete result")


def _spec(tmp_path: Path) -> chain.LegacySpec:
    return chain.LegacySpec(
        stage_id="legacy_d1",
        program_id="program-legacy-d1",
        process_label="mop-d1-frozen-queue",
        child_label_prefixes=("mop-d1-prod-", "mop-d1-chal-"),
        status_path=tmp_path / "runs/program-legacy-d1/current_status.json",
        status_schema="status-legacy-d1/v1",
        result_path=tmp_path / "proof/legacy-d1.json",
        result_schema="result-legacy-d1/v1",
        restart_command=(
            str(tmp_path / ".venv/bin/python"),
            "scripts/generation1_c3/d1_frozen_queue.py",
        ),
        result_validator=_validator,
    )


def _status(spec: chain.LegacySpec) -> dict[str, Any]:
    core = {
        "schema": spec.status_schema,
        "program_id": spec.program_id,
        "state": "running",
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "status_sha256": chain.canonical_sha256(core)}


def _horizon(path: Path) -> None:
    core = {
        "schema": "mop-generation1-program/v1",
        "program_id": "generation1-successor-horizon-v1",
    }
    _write(path, {**core, "program_sha256": chain.canonical_sha256(core)})


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
) -> chain.ProcessSnapshot:
    return chain.ProcessSnapshot(
        pid=pid,
        create_time=float(pid) + 0.25 if create_time is None else create_time,
        pgid=pid if pgid is None else pgid,
        cwd=str((cwd or tmp_path).resolve()),
        label=label,
        command=command,
        ppid=ppid,
    )


def _parent(
    tmp_path: Path,
    spec: chain.LegacySpec,
    *,
    processes: tuple[chain.ProcessSnapshot, ...],
    launches: list[tuple[list[str], dict[str, Any]]],
    process_table_fn: chain.ProcessTable | None = None,
    executable_probe_fn: Any = None,
    identity_probe_fn: Any = None,
    sleep_fn: Any = None,
) -> chain.SuccessorEvidenceChain:
    horizon = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    _horizon(horizon)

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append((command, kwargs))
        return SimpleNamespace(pid=9001)

    return chain.SuccessorEvidenceChain(
        root=tmp_path / "runs/generation1" / chain.CHAIN_ID,
        repo_root=tmp_path,
        horizon_program_path=horizon,
        specs=(spec,),
        execute=True,
        process_table_fn=process_table_fn or (lambda: processes),
        executable_probe_fn=executable_probe_fn or (
            lambda _process: str(tmp_path / ".venv/bin/python")
        ),
        identity_probe_fn=identity_probe_fn or (lambda _identity: "gone"),
        popen_fn=popen,
        now_fn=lambda: NOW,
        sleep_fn=sleep_fn or (lambda _seconds: None),
    )


def _spawn_title_transition(
    tmp_path: Path,
    *,
    pid: int,
    parent_pid: int,
    create_time: float | None = None,
) -> chain.ProcessSnapshot:
    python = str(tmp_path / ".venv/bin/python")
    joined = (
        f"{python} -c from multiprocessing.spawn import spawn_main; "
        "spawn_main(tracker_fd=9, pipe_handle=16) --multiprocessing-fork"
    )
    return _process(
        tmp_path,
        pid=pid,
        label=joined,
        command=(joined, "MOP_TEST_SECRET=must-not-persist", "", ""),
        pgid=parent_pid,
        ppid=parent_pid,
        create_time=create_time,
    )


def _fake_horizon_program(
    parent: chain.SuccessorEvidenceChain,
    tmp_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        program_id="generation1-successor-horizon-v1",
        path=parent.horizon_program_path,
        file_sha256=chain.sha256_file(parent.horizon_program_path),
        program_sha256=parent.state["horizon_program"]["program_sha256"],
        repo_root=tmp_path.resolve(),
        status_path=tmp_path / "runs/generation1/horizon-v1/current_status.json",
        capsules=(),
    )


def _horizon_status(
    program: SimpleNamespace,
    *,
    state: str,
    problems: list[str] | None = None,
    supervisor: Mapping[str, Any] | None = None,
    program_binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    generic_implementation = chain._generic_supervisor_authority(program.repo_root)
    supervisor_identity = {
        "pid": 999,
        "create_time": 99.0,
        "implementation_path": generic_implementation["path"],
        "implementation_sha256": generic_implementation["sha256"],
    }
    if supervisor is not None:
        supervisor_identity.update(supervisor)
    core = {
        "schema": "mop-generation1-status/v1",
        "program_id": program.program_id,
        "created_at": NOW.isoformat(),
        "program": (
            dict(program_binding)
            if program_binding is not None
            else {
                "path": str(program.path),
                "file_sha256": program.file_sha256,
                "program_sha256": program.program_sha256,
            }
        ),
        "supervisor": supervisor_identity,
        "execution_enabled": True,
        "state": state,
        "queue_head_sha256": chain.canonical_sha256(
            {
                "program_sha256": program.program_sha256,
                "base_capsules": [capsule.capsule_sha256 for capsule in program.capsules],
            }
        ),
        "next_injection_sequence": 1,
        "accepted_injection_count": 0,
        "current_capsule": None,
        "capsules": {},
        "last_admission": None,
        "lane_reservation": None,
        "problems": list(problems or []),
    }
    return {**core, "status_sha256": chain.canonical_sha256(core)}


def _complete_parent_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    chain.SuccessorEvidenceChain,
    chain.LegacySpec,
    SimpleNamespace,
    dict[str, Any],
]:
    spec = _spec(tmp_path)
    result = {
        "schema": spec.result_schema,
        "program_id": spec.program_id,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(spec.result_path, result)
    parent = _parent(tmp_path, spec, processes=(), launches=[])
    program = _fake_horizon_program(parent, tmp_path)
    _write(program.status_path, _horizon_status(program, state="complete"))
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    parent.state["capsules"][spec.stage_id].update(
        {
            "status": "complete",
            "returncode": 0,
            "finished_at": NOW.isoformat(),
            "artifacts": [parent._artifact(spec.result_path, result)],
        }
    )
    parent._reconcile_horizon(parent.state["capsules"]["successor_horizon"], ())
    parent.state["status"] = "complete"
    parent.state["finished_at"] = NOW.isoformat()
    return parent, spec, program, parent._publish()


def _adoption_receipt(
    parent: chain.SuccessorEvidenceChain,
    spec: chain.LegacySpec,
    process: Mapping[str, Any],
    *,
    observed_status: Mapping[str, Any] | None = None,
) -> Path:
    identity_sha = chain.canonical_sha256(process)[:16]
    path = parent.root / "adoptions" / spec.stage_id / f"{process.get('pid')}-{identity_sha}.json"
    core = {
        "schema": chain.ADOPTION_SCHEMA,
        "chain_id": chain.CHAIN_ID,
        "stage_id": spec.stage_id,
        "program_id": spec.program_id,
        "adopted_at": NOW.isoformat(),
        "process": dict(process),
        "observed_status": dict(observed_status) if observed_status is not None else None,
        "policy": {
            "observe_only": True,
            "signals_allowed": False,
            "restart_only_after_exact_absence": True,
            "restart_command_match": "exact-executable-and-two-argv-shape",
            "process_title_transition": "exact-spawn-rewrite-bounded-resnapshot",
        },
    }
    _write(
        path,
        {**core, "receipt_sha256": chain.canonical_sha256(core)},
    )
    return path


def _replace_parent_state(
    parent: chain.SuccessorEvidenceChain,
    state: Mapping[str, Any],
) -> None:
    state_core = dict(state)
    state_core.pop("state_sha256", None)
    sealed = {
        **state_core,
        "state_sha256": chain.canonical_sha256(state_core),
    }
    _write(parent.state_path, sealed)
    _write(parent.status_path, chain._status_payload(sealed))


@pytest.mark.parametrize(
    "command",
    [
        (
            "/usr/bin/sed",
            "-n",
            "1,80p",
            "scripts/generation1_c3/d1_frozen_queue.py",
        ),
        (
            "/opt/homebrew/bin/rg",
            "-n",
            "restart",
            "scripts/generation1_c3/d1_frozen_queue.py",
        ),
        (
            "PYTHON",
            "-c",
            "from pathlib import Path; print(Path(__import__('sys').argv[1]).name)",
            "scripts/generation1_c3/d1_frozen_queue.py",
        ),
    ],
)
def test_benign_review_argv_mentions_do_not_count_as_restart_residual(
    tmp_path: Path,
    command: tuple[str, ...],
) -> None:
    spec = _spec(tmp_path)
    actual = tuple(spec.restart_command[0] if value == "PYTHON" else value for value in command)
    review = _process(tmp_path, pid=700, label=actual[0], command=actual)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(review,), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (review,))

    assert len(launches) == 1
    assert launches[0][0] == list(spec.restart_command)
    assert row["status"] == "launching"


def test_only_exact_executable_and_two_argv_restart_shape_blocks_relaunch(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    exact = _process(
        tmp_path,
        pid=701,
        label=spec.restart_command[0],
        command=(
            spec.restart_command[0],
            str(tmp_path / spec.restart_command[1]),
        ),
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(exact,), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (exact,))

    assert launches == []
    assert row["status"] == "adoption_wait"
    assert str(exact.pid) in row["last_problem"]


def test_extra_restart_argv_is_not_the_authorized_restart_shape(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    review = _process(
        tmp_path,
        pid=702,
        label=spec.restart_command[0],
        command=(
            spec.restart_command[0],
            str(tmp_path / spec.restart_command[1]),
            "--help",
        ),
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(review,), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (review,))

    assert len(launches) == 1
    assert row["status"] == "launching"


def test_legacy_launch_intent_publish_has_coherent_parent_state(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, ())

    published = json.loads(parent.status_path.read_text(encoding="utf-8"))
    assert (
        chain.validate_chain_status(
            published,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )
        == "waiting_legacy"
    )
    assert published["capsules"][spec.stage_id]["status"] == "launching"
    assert len(launches) == 1


def test_v5_adopts_exact_live_parent_under_fresh_identity_and_receipt_schema(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    live = _process(
        tmp_path,
        pid=703,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(live,), launches=launches)

    status = parent.tick()

    assert status["schema"] == chain.STATUS_SCHEMA
    assert status["program_id"] == chain.CHAIN_ID
    assert status["supersedes"] == "generation1-successor-evidence-chain-v4"
    assert status["state"] == "waiting_legacy"
    assert status["horizon_program"]["program_id"] == "generation1-successor-horizon-v1"
    assert launches == []
    receipt_path = next((parent.root / "adoptions").glob("*/*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == chain.ADOPTION_SCHEMA
    assert receipt["chain_id"] == chain.CHAIN_ID
    assert receipt["policy"]["signals_allowed"] is False
    assert receipt["policy"]["restart_command_match"] == ("exact-executable-and-two-argv-shape")
    assert receipt["policy"]["process_title_transition"] == (
        "exact-spawn-rewrite-bounded-resnapshot"
    )


def test_v5_resamples_exact_spawn_title_transition_until_worker_label(
    tmp_path: Path,
) -> None:
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=730,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=731,
        parent_pid=owner.pid,
    )
    worker_label = f"{spec.child_label_prefixes[0]}r001"
    resolved = _process(
        tmp_path,
        pid=transition.pid,
        label=worker_label,
        command=(worker_label, ""),
        pgid=owner.pid,
        ppid=owner.pid,
        create_time=transition.create_time,
    )
    tables = [(owner, transition), (owner, resolved), (owner, resolved)]
    calls = 0
    sleeps: list[float] = []

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        table = tables[min(calls, len(tables) - 1)]
        calls += 1
        return table

    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=launches,
        process_table_fn=process_table,
        sleep_fn=sleeps.append,
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert launches == []
    assert calls == 3
    assert sleeps == [
        chain.PROCESS_TRANSITION_INTERVAL_SECONDS,
        chain.PROCESS_TRANSITION_INTERVAL_SECONDS,
    ]
    serialized = parent.status_path.read_text(encoding="utf-8") + next(
        (parent.root / "adoptions").glob("*/*.json")
    ).read_text(encoding="utf-8")
    assert "MOP_TEST_SECRET" not in serialized
    assert "spawn_main" not in serialized


def test_v5_resamples_exact_spawn_title_transition_that_disappears_exactly(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=732,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=733,
        parent_pid=owner.pid,
    )
    tables = [(owner, transition), (owner,), (owner,)]
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        table = tables[min(calls, len(tables) - 1)]
        calls += 1
        return table

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
        identity_probe_fn=lambda identity: (
            "gone" if identity.get("pid") == transition.pid else "alive"
        ),
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert calls == 3


def test_v5_persistent_spawn_title_transition_fails_after_bounded_resamples(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=734,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=735,
        parent_pid=owner.pid,
    )
    calls = 0
    sleeps: list[float] = []

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return owner, transition

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
        sleep_fn=sleeps.append,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "did not stabilize within the bounded window" in status["problems"][-1]
    assert calls == chain.PROCESS_TRANSITION_ATTEMPTS + 1
    assert len(sleeps) == chain.PROCESS_TRANSITION_ATTEMPTS
    assert "MOP_TEST_SECRET" not in parent.status_path.read_text(encoding="utf-8")


def test_v5_valid_transition_cannot_mask_hard_same_group_intruder(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=736,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=737,
        parent_pid=owner.pid,
    )
    intruder = _process(
        tmp_path,
        pid=738,
        label="/bin/sh",
        command=("/bin/sh", "-c", "true"),
        pgid=owner.pid,
        ppid=owner.pid,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return owner, transition, intruder

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert str(intruder.pid) in status["problems"][-1]
    assert calls == 1


def test_v5_parent_identity_cannot_change_during_transition_resample(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=739,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=740,
        parent_pid=owner.pid,
    )
    replaced_owner = _process(
        tmp_path,
        pid=owner.pid,
        label=owner.label,
        command=owner.command,
        create_time=owner.create_time + 10.0,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return (owner, transition) if calls == 1 else (replaced_owner, transition)

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "exact parent changed during process-title stabilization" in status["problems"][-1]
    assert calls == 2


def test_v5_provisional_child_cannot_change_ppid_before_worker_label(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=742,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=743,
        parent_pid=owner.pid,
    )
    worker_label = f"{spec.child_label_prefixes[0]}r004"
    escaped = _process(
        tmp_path,
        pid=transition.pid,
        label=worker_label,
        command=(worker_label, ""),
        pgid=owner.pid,
        ppid=1,
        create_time=transition.create_time,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return (owner, transition) if calls == 1 else (owner, escaped)

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "changed ownership boundary" in status["problems"][-1]
    assert calls == 2


def test_v5_provisional_child_cannot_resolve_as_resource_tracker(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(
        tmp_path,
        pid=744,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    transition = _spawn_title_transition(
        tmp_path,
        pid=745,
        parent_pid=owner.pid,
    )
    python = str(tmp_path / ".venv/bin/python")
    tracker = _process(
        tmp_path,
        pid=transition.pid,
        label=python,
        command=(
            python,
            "-c",
            "from multiprocessing.resource_tracker import main;main(8)",
        ),
        pgid=owner.pid,
        ppid=owner.pid,
        create_time=transition.create_time,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return (owner, transition) if calls == 1 else (owner, tracker)

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "resolved to an unauthorized process class" in status["problems"][-1]
    assert calls == 2


def test_foreign_mop_g1_horizon_child_cannot_contaminate_mechanics_adoption(
    tmp_path: Path,
) -> None:
    base = _spec(tmp_path)
    spec = chain.LegacySpec(
        stage_id="legacy_successor_mechanics",
        program_id=base.program_id,
        process_label="mop-successor-mechanics-queue",
        child_label_prefixes=("mop-g1-",),
        status_path=base.status_path,
        status_schema=base.status_schema,
        result_path=base.result_path,
        result_schema=base.result_schema,
        restart_command=base.restart_command,
        result_validator=base.result_validator,
    )
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path,
        pid=710,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    foreign = _process(
        tmp_path,
        pid=711,
        label="mop-g1-horizon-h01-d1-00",
        command=("mop-g1-horizon-h01-d1-00",),
        pgid=711,
        ppid=999,
        cwd=tmp_path / "other-worktree",
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(
        tmp_path,
        spec,
        processes=(parent_process, foreign),
        launches=launches,
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert launches == []


def test_same_repo_matching_worker_outside_parent_ownership_fails_closed(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path,
        pid=714,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    unowned = _process(
        tmp_path,
        pid=715,
        label=f"{spec.child_label_prefixes[0]}r001",
        command=(f"{spec.child_label_prefixes[0]}r001",),
        pgid=715,
        ppid=999,
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(
        tmp_path,
        spec,
        processes=(parent_process, unowned),
        launches=launches,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "outside the exact repository-cwd" in status["problems"][-1]
    assert launches == []


@pytest.mark.parametrize(
    ("pgid", "ppid"),
    [
        (716, 714),
        (714, 999),
    ],
)
def test_foreign_cwd_worker_claiming_parent_ownership_fails_closed(
    tmp_path: Path,
    pgid: int,
    ppid: int,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path,
        pid=714,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    foreign = _process(
        tmp_path,
        pid=716,
        label=f"{spec.child_label_prefixes[0]}r002",
        command=(f"{spec.child_label_prefixes[0]}r002",),
        pgid=pgid,
        ppid=ppid,
        cwd=tmp_path / "other-worktree",
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(
        tmp_path,
        spec,
        processes=(parent_process, foreign),
        launches=launches,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "outside the exact repository-cwd" in status["problems"][-1]
    assert launches == []


def test_parent_absent_same_repo_exact_worker_blocks_relaunch(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    orphan = _process(
        tmp_path,
        pid=717,
        label=f"{spec.child_label_prefixes[0]}r003",
        command=(f"{spec.child_label_prefixes[0]}r003",),
        pgid=700,
        ppid=1,
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(orphan,), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (orphan,))

    assert row["status"] == "adoption_wait"
    assert str(orphan.pid) in row["last_problem"]
    assert launches == []


def test_parent_absent_foreign_worktree_prelabel_spawn_does_not_contaminate(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    python = spec.restart_command[0]
    foreign_spawn = _process(
        tmp_path,
        pid=718,
        label=python,
        command=(
            python,
            "-c",
            ("from multiprocessing.spawn import spawn_main; spawn_main(tracker_fd=9, pipe_handle=16)"),
            "--multiprocessing-fork",
        ),
        pgid=718,
        ppid=999,
        cwd=tmp_path / "other-worktree",
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(foreign_spawn,), launches=launches)
    row = parent.state["capsules"][spec.stage_id]

    parent._reconcile_legacy(spec, row, (foreign_spawn,))

    assert len(launches) == 1
    assert launches[0][0] == list(spec.restart_command)
    assert row["status"] == "launching"


def test_exact_duplicate_restart_outside_live_parent_group_fails_closed(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path,
        pid=712,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    duplicate = _process(
        tmp_path,
        pid=713,
        label=spec.restart_command[0],
        command=(
            spec.restart_command[0],
            str(tmp_path / spec.restart_command[1]),
        ),
    )
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(
        tmp_path,
        spec,
        processes=(parent_process, duplicate),
        launches=launches,
    )

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "outside the exact parent process group" in status["problems"][-1]
    assert launches == []


def test_v5_state_binds_wrapper_and_inherited_v3_base(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)

    authority = parent.state["parent_implementation"]

    assert authority["path"].endswith("generation1_successor_chain_v5.py")
    assert authority["sha256"] == chain.sha256_file(Path(chain.__file__).resolve())
    assert authority["superseded_predecessor"]["path"].endswith(
        "generation1_successor_chain_v4.py"
    )
    assert authority["inherited_base"]["path"].endswith("generation1_successor_chain.py")
    assert len(authority["inherited_base"]["sha256"]) == 64


def test_v5_tick_preserves_historical_v4_root_byte_for_byte(tmp_path: Path) -> None:
    historical = (
        tmp_path
        / "runs/generation1/generation1-successor-evidence-chain-v4"
    )
    evidence = {
        historical / "chain_state.json": b'{"state":"integrity_hold","version":4}\n',
        historical / "current_status.json": b'{"state":"integrity_hold","version":4}\n',
        historical / "adoptions/legacy_final/receipt.json": b'{"observe_only":true}\n',
    }
    for path, raw in evidence.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    live = _process(
        tmp_path,
        pid=741,
        label=spec.process_label,
        command=(spec.process_label, ""),
    )
    parent = _parent(
        tmp_path,
        spec,
        processes=(live,),
        launches=[],
    )

    status = parent.tick()

    assert status["program_id"] == "generation1-successor-evidence-chain-v5"
    assert parent.root != historical
    assert {path: path.read_bytes() for path in evidence} == evidence


def test_default_constructor_uses_exact_legacy_mechanics_lane_families(
    tmp_path: Path,
) -> None:
    horizon = tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"
    _horizon(horizon)
    parent = chain.SuccessorEvidenceChain(
        root=tmp_path / "runs/generation1" / chain.CHAIN_ID,
        repo_root=tmp_path,
        horizon_program_path=horizon,
        execute=False,
        now_fn=lambda: NOW,
    )
    mechanics = next(spec for spec in parent.specs if spec.stage_id == "legacy_successor_mechanics")

    assert "mop-g1-" not in mechanics.child_label_prefixes
    assert "mop-g1-g1-" in mechanics.child_label_prefixes
    assert "mop-g1-v1-" in mechanics.child_label_prefixes
    assert all(not prefix.startswith("mop-g1-horizon") for prefix in mechanics.child_label_prefixes)


def test_parent_command_detection_requires_exact_argv_positions(tmp_path: Path) -> None:
    entrypoint = chain.REPO_ROOT / "scripts/mop_generation1_successor_chain_v5.py"
    python = str(chain.REPO_ROOT / ".venv/bin/python")
    exact = _process(
        tmp_path,
        pid=704,
        label=python,
        command=(python, str(entrypoint), "run", "--execute"),
    )
    mention = _process(
        tmp_path,
        pid=705,
        label="/usr/bin/sed",
        command=("/usr/bin/sed", "-n", str(entrypoint), "run"),
    )

    assert chain._exact_parent_command(exact, entrypoint) is True
    assert chain._exact_parent_command(mention, entrypoint) is False


def test_run_reloads_post_construction_state_update_under_lifetime_lock(
    tmp_path: Path,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    parent._publish()
    persisted = json.loads(parent.state_path.read_text(encoding="utf-8"))
    persisted.pop("state_sha256")
    persisted["status"] = "drained"
    persisted["finished_at"] = NOW.isoformat()
    _write(
        parent.state_path,
        {**persisted, "state_sha256": chain.canonical_sha256(persisted)},
    )

    status = parent.run(max_cycles=1)

    assert status["state"] == "drained"
    assert launches == []


def test_detached_start_emits_one_exact_run_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-evidence-chain-v5"
    launches: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(
        chain,
        "validate_chain_status",
        lambda *_args, **_kwargs: "waiting_legacy",
    )

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append((command, kwargs))
        core = {
            "schema": chain.STATUS_SCHEMA,
            "program_id": chain.CHAIN_ID,
            "state": "waiting_legacy",
            "supervisor": {"pid": 720, "create_time": 72.0},
        }
        _write(
            root / chain.STATUS_FILE,
            {**core, "status_sha256": chain.canonical_sha256(core)},
        )
        return SimpleNamespace(pid=720, poll=lambda: None)

    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    result = chain.start_chain_detached(
        root=root,
        execute=True,
        use_caffeinate=False,
    )

    expected = [
        str(chain.REPO_ROOT / ".venv/bin/python"),
        str(chain.REPO_ROOT / "scripts/mop_generation1_successor_chain_v5.py"),
        "run",
        "--execute",
        "--root",
        str(root),
    ]
    assert launches[0][0] == expected
    assert launches[0][0].count("run") == 1
    assert launches[0][1]["start_new_session"] is True
    assert result["launched_pid"] == 720


def test_detached_start_waits_past_transient_non_authoritative_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-evidence-chain-v5"
    status_path = root / chain.STATUS_FILE
    seen: list[str] = []

    def write_status(state: str) -> None:
        core = {
            "schema": chain.STATUS_SCHEMA,
            "program_id": chain.CHAIN_ID,
            "state": state,
            "supervisor": {"pid": 721, "create_time": 721.5},
        }
        _write(
            status_path,
            {**core, "status_sha256": chain.canonical_sha256(core)},
        )

    def validate(status: Mapping[str, Any], **_kwargs: Any) -> str:
        state = str(status["state"])
        seen.append(state)
        if state != "waiting_legacy":
            raise chain.SuccessorChainRefused("transient launch intent")
        return state

    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "validate_chain_status", validate)
    monkeypatch.setattr(
        chain.subprocess,
        "Popen",
        lambda *_args, **_kwargs: write_status("starting") or SimpleNamespace(pid=721, poll=lambda: None),
    )
    monkeypatch.setattr(
        chain.time,
        "sleep",
        lambda _seconds: write_status("waiting_legacy"),
    )

    result = chain.start_chain_detached(
        root=root,
        execute=True,
        use_caffeinate=False,
    )

    assert seen == ["starting", "waiting_legacy"]
    assert result["status"]["state"] == "waiting_legacy"


def test_detached_start_rejects_live_sealed_identity_without_exact_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-evidence-chain-v5"
    core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.CHAIN_ID,
        "state": "waiting_legacy",
        "supervisor": {"pid": 722, "create_time": 722.5},
    }
    _write(
        root / chain.STATUS_FILE,
        {**core, "status_sha256": chain.canonical_sha256(core)},
    )
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: True)
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="no exact visible parent",
    ):
        chain.start_chain_detached(
            root=root,
            execute=True,
            use_caffeinate=False,
        )


def test_complete_snapshot_replays_exact_state_status_and_current_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, program, status = _complete_parent_snapshot(tmp_path, monkeypatch)

    validated = chain.read_validated_complete_chain_status(
        root=parent.root,
        repo_root=tmp_path,
        horizon_program_path=parent.horizon_program_path,
        specs=(spec,),
    )

    assert validated == status
    assert validated["state"] == "complete"
    assert validated["capsules"]["legacy_d1"]["attempts"] == 0
    assert validated["capsules"]["legacy_d1"]["adoption_receipts"] == []
    assert program.status_path.is_file()


def test_complete_snapshot_retries_one_torn_state_status_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, status = _complete_parent_snapshot(tmp_path, monkeypatch)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    stale_core = dict(status)
    stale_core.pop("status_sha256")
    stale_core["updated_at"] = (NOW - dt.timedelta(seconds=1)).isoformat()
    stale_status = {
        **stale_core,
        "status_sha256": chain.canonical_sha256(stale_core),
    }
    actual_read = chain._read_json
    injected: list[tuple[Path, dict[str, Any]]] = [
        (parent.state_path, state),
        (parent.status_path, stale_status),
        (parent.state_path, state),
        (parent.status_path, stale_status),
    ]

    def read_with_one_torn_pair(path: Path, label: str) -> dict[str, Any]:
        if injected and path.resolve() == injected[0][0].resolve():
            _expected_path, payload = injected.pop(0)
            return json.loads(json.dumps(payload))
        return actual_read(path, label)

    monkeypatch.setattr(chain, "_read_json", read_with_one_torn_pair)
    monkeypatch.setattr(chain, "SNAPSHOT_INTERVAL_SECONDS", 0.0)

    validated = chain.read_validated_complete_chain_status(
        root=parent.root,
        repo_root=tmp_path,
        horizon_program_path=parent.horizon_program_path,
        specs=(spec,),
    )

    assert injected == []
    assert validated == status


def test_complete_snapshot_retries_transient_lifetime_lock_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, status = _complete_parent_snapshot(tmp_path, monkeypatch)
    actual_lock = chain.FileLock
    attempts = 0

    class RetryOnceLock:
        def __init__(self, path: Path):
            self.path = path
            self.delegate: Any = None

        def __enter__(self) -> Any:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                try:
                    raise BlockingIOError("busy")
                except BlockingIOError as exc:
                    raise chain.Generation1Refused("lock held") from exc
            self.delegate = actual_lock(self.path)
            return self.delegate.__enter__()

        def __exit__(self, *arguments: object) -> None:
            if self.delegate is not None:
                self.delegate.__exit__(*arguments)

    monkeypatch.setattr(chain, "FileLock", RetryOnceLock)
    monkeypatch.setattr(chain, "LOCK_INTERVAL_SECONDS", 0.0)

    validated = chain.read_validated_complete_chain_status(
        root=parent.root,
        repo_root=tmp_path,
        horizon_program_path=parent.horizon_program_path,
        specs=(spec,),
    )

    assert attempts == 2
    assert validated == status


def test_complete_snapshot_rejects_state_change_during_artifact_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    original = chain.SuccessorEvidenceChain._validate_completed_chain_artifacts

    def mutate_after_replay(
        owner: chain.SuccessorEvidenceChain,
        processes: tuple[chain.ProcessSnapshot, ...],
    ) -> None:
        original(owner, processes)
        state = json.loads(owner.state_path.read_text(encoding="utf-8"))
        state_core = dict(state)
        state_core.pop("state_sha256")
        state_core["status"] = "integrity_hold"
        state_core["problems"] = ["predecessor regressed during replay"]
        changed = {
            **state_core,
            "state_sha256": chain.canonical_sha256(state_core),
        }
        _write(owner.state_path, changed)
        _write(owner.status_path, chain._status_payload(changed))

    monkeypatch.setattr(
        chain.SuccessorEvidenceChain,
        "_validate_completed_chain_artifacts",
        mutate_after_replay,
    )

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="changed during terminal replay",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rejects_missing_adoption_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    missing = (parent.root / "adoptions" / spec.stage_id / "missing.json").relative_to(tmp_path)
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(missing)]
    state_core = dict(state)
    state_core.pop("state_sha256")
    changed = {
        **state_core,
        "state_sha256": chain.canonical_sha256(state_core),
    }
    _write(parent.state_path, changed)
    _write(parent.status_path, chain._status_payload(changed))

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="adoption receipt is missing",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_accepts_exact_adoption_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 810,
        "create_time": 810.5,
        "pgid": 810,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    observed_status = {
        "path": str(spec.status_path.relative_to(tmp_path)),
        "file_sha256": "a" * 64,
        "status_sha256": "b" * 64,
        "state": "running",
    }
    receipt_path = _adoption_receipt(
        parent,
        spec,
        process,
        observed_status=observed_status,
    )
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(receipt_path.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    validated = chain.read_validated_complete_chain_status(
        root=parent.root,
        repo_root=tmp_path,
        horizon_program_path=parent.horizon_program_path,
        specs=(spec,),
    )

    assert validated["status_sha256"] != status["status_sha256"]
    assert validated["capsules"][spec.stage_id]["process"] == process


def test_complete_snapshot_rejects_invalid_adopted_process_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": "not-an-int",
        "create_time": "not-a-time",
        "pgid": -7,
        "cwd": "/outside/repository",
        "label": "unrelated-parent",
    }
    receipt_path = _adoption_receipt(parent, spec, process)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(receipt_path.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="adopted process identity drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_applies_public_status_process_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _specification, _program, _status = _complete_parent_snapshot(
        tmp_path,
        monkeypatch,
    )
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"]["successor_horizon"]["process"] = {}
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="successor_horizon process identity drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=parent.specs,
        )


def test_complete_snapshot_binds_horizon_process_to_terminal_supervisor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, _specification, _program, _status = _complete_parent_snapshot(
        tmp_path,
        monkeypatch,
    )
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"]["successor_horizon"]["process"] = {
        "pid": 909,
        "create_time": 909.5,
    }
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="required reconciliation",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=parent.specs,
        )


def test_complete_snapshot_rejects_process_without_adoption_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["process"] = {
        "fabricated": True,
    }
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="process identity without an adoption receipt",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_requires_adopted_parent_process_group_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 823,
        "create_time": 823.5,
        "pgid": 999,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    receipt_path = _adoption_receipt(parent, spec, process)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(receipt_path.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="adopted process identity drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rejects_arbitrary_observed_status_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 811,
        "create_time": 811.5,
        "pgid": 811,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    receipt_path = _adoption_receipt(
        parent,
        spec,
        process,
        observed_status={"fabricated": True},
    )
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(receipt_path.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="observed status authority drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rejects_unknown_observed_status_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 824,
        "create_time": 824.5,
        "pgid": 824,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    receipt_path = _adoption_receipt(
        parent,
        spec,
        process,
        observed_status={
            "path": str(spec.status_path.relative_to(tmp_path)),
            "file_sha256": "a" * 64,
            "status_sha256": "b" * 64,
            "state": "garbage-state",
        },
    )
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(receipt_path.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="observed status authority drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rejects_symlinked_adoption_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 812,
        "create_time": 812.5,
        "pgid": 812,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    receipt_path = _adoption_receipt(parent, spec, process)
    alias = receipt_path.with_name("alias.json")
    alias.symlink_to(receipt_path.name)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(alias.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="regular non-symlink file",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rejects_receipt_through_symlinked_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    process = {
        "pid": 825,
        "create_time": 825.5,
        "pgid": 825,
        "cwd": str(tmp_path.resolve()),
        "label": spec.process_label,
    }
    receipt_path = _adoption_receipt(parent, spec, process)
    alias_directory = receipt_path.parent.with_name(f"{spec.stage_id}_alias")
    alias_directory.symlink_to(receipt_path.parent.name)
    alias = alias_directory / receipt_path.name
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["adoption_receipts"] = [str(alias.relative_to(tmp_path))]
    state["capsules"][spec.stage_id]["process"] = process
    _replace_parent_state(parent, state)

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="canonical regular non-symlink file",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_rechecks_artifacts_after_state_status_reread(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    original = chain.SuccessorEvidenceChain._validate_completed_chain_artifacts
    calls = 0

    def mutate_after_first_replay(
        owner: chain.SuccessorEvidenceChain,
        processes: tuple[chain.ProcessSnapshot, ...],
    ) -> None:
        nonlocal calls
        calls += 1
        original(owner, processes)
        if calls == 1:
            payload = json.loads(spec.result_path.read_text(encoding="utf-8"))
            payload["mutated_after_replay"] = True
            _write(spec.result_path, payload)

    monkeypatch.setattr(
        chain.SuccessorEvidenceChain,
        "_validate_completed_chain_artifacts",
        mutate_after_first_replay,
    )

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="artifact inventory disappeared or drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )

    assert calls == 2


def test_complete_snapshot_performs_independent_final_artifact_hash_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    original = chain.SuccessorEvidenceChain._validate_completed_chain_artifacts
    calls = 0

    def mutate_after_second_replay(
        owner: chain.SuccessorEvidenceChain,
        processes: tuple[chain.ProcessSnapshot, ...],
    ) -> None:
        nonlocal calls
        calls += 1
        original(owner, processes)
        if calls == 2:
            payload = json.loads(spec.result_path.read_text(encoding="utf-8"))
            payload["mutated_after_second_replay"] = True
            _write(spec.result_path, payload)

    monkeypatch.setattr(
        chain.SuccessorEvidenceChain,
        "_validate_completed_chain_artifacts",
        mutate_after_second_replay,
    )

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="final artifact hash drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )

    assert calls == 2


def test_complete_snapshot_rejects_resealed_reused_legacy_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent, spec, _program, _status = _complete_parent_snapshot(tmp_path, monkeypatch)
    state = json.loads(parent.state_path.read_text(encoding="utf-8"))
    state["capsules"][spec.stage_id]["artifacts"] = list(state["capsules"]["successor_horizon"]["artifacts"])
    state_core = dict(state)
    state_core.pop("state_sha256")
    state = {
        **state_core,
        "state_sha256": chain.canonical_sha256(state_core),
    }
    _write(parent.state_path, state)
    _write(parent.status_path, chain._status_payload(state))

    with pytest.raises(
        chain.SuccessorChainRefused,
        match="result or artifact inventory disappeared or drifted",
    ):
        chain.read_validated_complete_chain_status(
            root=parent.root,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_complete_snapshot_requires_sibling_sealed_state(tmp_path: Path) -> None:
    root = tmp_path / "runs/generation1" / chain.CHAIN_ID
    status_core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.CHAIN_ID,
        "state": "complete",
    }
    _write(
        root / chain.STATUS_FILE,
        {**status_core, "status_sha256": chain.canonical_sha256(status_core)},
    )

    with pytest.raises(chain.SuccessorChainRefused, match="state is missing"):
        chain.read_validated_complete_chain_status(
            root=root,
            repo_root=tmp_path,
            horizon_program_path=(tmp_path / "configs/campaign/generation1_successor_horizon_v1.json"),
            specs=(_spec(tmp_path),),
        )


def test_detached_terminal_shortcut_revalidates_complete_evidence_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    result_payload = {
        "schema": spec.result_schema,
        "program_id": spec.program_id,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(spec.result_path, result_payload)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    _write(program.status_path, _horizon_status(program, state="complete"))
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    legacy = parent.state["capsules"][spec.stage_id]
    legacy.update(
        {
            "status": "complete",
            "returncode": 0,
            "finished_at": NOW.isoformat(),
            "artifacts": [parent._artifact(spec.result_path, result_payload)],
        }
    )
    parent._reconcile_horizon(parent.state["capsules"]["successor_horizon"], ())
    parent.state["status"] = "complete"
    parent.state["finished_at"] = NOW.isoformat()
    terminal = parent._publish()
    assert terminal["state"] == "complete"
    spec.result_path.unlink()

    monkeypatch.setattr(chain, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(chain, "DEFAULT_HORIZON_PROGRAM", parent.horizon_program_path)
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: False)
    monkeypatch.setattr(chain, "SuccessorEvidenceChain", lambda **_kwargs: parent)

    first = chain.start_chain_detached(root=parent.root, execute=True, use_caffeinate=False)
    second = chain.start_chain_detached(root=parent.root, execute=True, use_caffeinate=False)

    assert first["already_terminal"] is True
    assert first["status"]["state"] == "integrity_hold"
    assert (
        "completed legacy_d1 result or artifact inventory disappeared or drifted"
        in first["status"]["problems"][-1]
    )
    assert second["status"]["state"] == "integrity_hold"
    assert second["status"]["problems"] == first["status"]["problems"]
    assert launches == []


def test_detached_terminal_status_resolves_live_parent_before_revalidation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-evidence-chain-v5"
    status_core = {
        "schema": chain.STATUS_SCHEMA,
        "program_id": chain.CHAIN_ID,
        "state": "complete",
        "supervisor": None,
    }
    _write(
        root / chain.STATUS_FILE,
        {**status_core, "status_sha256": chain.canonical_sha256(status_core)},
    )
    visible = _process(
        tmp_path,
        pid=721,
        label=chain.PARENT_LABEL,
        command=("mop-successor-evidence-chain-v5",),
    )
    revalidations: list[Path] = []
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: visible)
    monkeypatch.setattr(chain, "_process_identity_alive", lambda _identity: False)
    monkeypatch.setattr(
        chain,
        "_revalidate_terminal_chain",
        lambda requested_root: revalidations.append(requested_root),
    )

    result = chain.start_chain_detached(root=root, execute=True, use_caffeinate=False)

    assert result["already_running"] is True
    assert result["observed_process"] == visible.identity()
    assert revalidations == []


def test_horizon_status_rejects_stale_manifest_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    stale_binding = {
        "path": str(program.path),
        "file_sha256": "0" * 64,
        "program_sha256": program.program_sha256,
    }
    _write(
        program.status_path,
        _horizon_status(program, state="complete", program_binding=stale_binding),
    )
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)

    with pytest.raises(chain.SuccessorChainRefused, match="program authority drifted"):
        parent._reconcile_horizon(parent.state["capsules"]["successor_horizon"], ())

    assert launches == []


def test_horizon_launch_intent_publish_has_coherent_parent_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    result = {
        "schema": spec.result_schema,
        "program_id": spec.program_id,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(spec.result_path, result)
    parent = _parent(tmp_path, spec, processes=(), launches=[])
    program = _fake_horizon_program(parent, tmp_path)
    observed: list[dict[str, Any]] = []

    def start(*_args: Any, **_kwargs: Any) -> Mapping[str, Any]:
        published = json.loads(parent.status_path.read_text(encoding="utf-8"))
        observed.append(published)
        assert (
            chain.validate_chain_status(
                published,
                repo_root=tmp_path,
                horizon_program_path=parent.horizon_program_path,
                specs=(spec,),
            )
            == "waiting_horizon"
        )
        return {}

    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    parent.supervisor_start_fn = start

    status = parent.tick()

    assert status["state"] == "waiting_horizon"
    assert observed[0]["capsules"]["successor_horizon"]["status"] == "launching"


def test_horizon_complete_status_requires_empty_problems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    _write(
        program.status_path,
        _horizon_status(program, state="complete", problems=["dirty completion"]),
    )
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)

    with pytest.raises(chain.SuccessorChainRefused, match="complete status contains problems"):
        parent._reconcile_horizon(parent.state["capsules"]["successor_horizon"], ())

    assert launches == []


def test_completed_horizon_missing_status_never_relaunches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    _write(program.status_path, _horizon_status(program, state="complete"))
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    row = parent.state["capsules"]["successor_horizon"]

    parent._reconcile_horizon(row, ())
    program.status_path.unlink()

    with pytest.raises(chain.SuccessorChainRefused, match="status disappeared"):
        parent._reconcile_horizon(row, ())

    assert row["status"] == "complete"
    assert launches == []


def test_completed_horizon_regressed_status_never_relaunches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    _write(program.status_path, _horizon_status(program, state="complete"))
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    row = parent.state["capsules"]["successor_horizon"]
    parent._reconcile_horizon(row, ())
    _write(program.status_path, _horizon_status(program, state="progressing"))

    with pytest.raises(chain.SuccessorChainRefused, match="status regressed"):
        parent._reconcile_horizon(row, ())

    assert row["status"] == "complete"
    assert launches == []


def test_complete_chain_tick_revalidates_and_holds_on_missing_horizon_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    result = {
        "schema": spec.result_schema,
        "program_id": spec.program_id,
        "complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(spec.result_path, result)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    program = _fake_horizon_program(parent, tmp_path)
    _write(program.status_path, _horizon_status(program, state="complete"))
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    legacy = parent.state["capsules"][spec.stage_id]
    legacy.update(
        {
            "status": "complete",
            "returncode": 0,
            "finished_at": NOW.isoformat(),
            "artifacts": [parent._artifact(spec.result_path, result)],
        }
    )
    horizon = parent.state["capsules"]["successor_horizon"]
    parent._reconcile_horizon(horizon, ())
    parent.state["status"] = "complete"
    parent.state["finished_at"] = NOW.isoformat()

    clean = parent.tick()
    program.status_path.unlink()
    held = parent.tick()

    assert clean["state"] == "complete"
    assert held["state"] == "integrity_hold"
    assert (
        chain.validate_chain_status(
            held,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )
        == "integrity_hold"
    )
    assert "completed successor horizon status disappeared" in held["problems"][-1]
    assert launches == []


def test_live_horizon_status_requires_exact_visible_supervisor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)
    parent.identity_probe_fn = lambda _identity: "alive"
    program = _fake_horizon_program(parent, tmp_path)
    supervisor = {"pid": 800, "create_time": 80.0}
    _write(
        program.status_path,
        _horizon_status(program, state="running", supervisor=supervisor),
    )
    monkeypatch.setattr(chain, "load_program", lambda *_args, **_kwargs: program)
    wrong = _process(
        tmp_path,
        pid=801,
        label="mop-supervisor:generation1-successor-horizon-v1",
        command=("mop-supervisor:generation1-successor-horizon-v1",),
    )
    exact = chain.ProcessSnapshot(
        pid=800,
        create_time=80.0,
        pgid=800,
        cwd=str(tmp_path.resolve()),
        label="mop-supervisor:generation1-successor-horizon-v1",
        command=("mop-supervisor:generation1-successor-horizon-v1",),
        ppid=1,
    )
    row = parent.state["capsules"]["successor_horizon"]

    with pytest.raises(chain.SuccessorChainRefused, match="conflicts with visible identity"):
        parent._reconcile_horizon(row, (wrong,))

    parent._reconcile_horizon(row, (exact,))
    assert row["status"] == "running"
    assert row["process"] == {
        **supervisor,
        "implementation_path": chain._generic_supervisor_authority(tmp_path)["path"],
        "implementation_sha256": chain._generic_supervisor_authority(tmp_path)["sha256"],
    }
    assert launches == []
