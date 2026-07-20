from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_chain_v7 as chain

NOW = dt.datetime(2026, 7, 17, 4, 0, tzinfo=dt.UTC)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _validator(payload: Mapping[str, Any]) -> None:
    if payload.get("complete") is not True:
        raise ValueError("incomplete result")


def _spec(tmp_path: Path) -> chain.LegacySpec:
    return chain.LegacySpec(
        stage_id="legacy_final",
        program_id="program-legacy-final",
        process_label="mop-final-campaign",
        child_label_prefixes=("mop-final-",),
        status_path=tmp_path / "runs/program-legacy-final/current_status.json",
        status_schema="status-legacy-final/v1",
        result_path=tmp_path / "proof/legacy-final.json",
        result_schema="result-legacy-final/v1",
        restart_command=(
            str(tmp_path / ".venv/bin/python"),
            "scripts/generation1_c3/consolidated_final_campaign.py",
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
    supervisor_start_fn: Any = None,
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
        executable_probe_fn=executable_probe_fn or (lambda _process: str(tmp_path / ".venv/bin/python")),
        identity_probe_fn=identity_probe_fn or (lambda _identity: "gone"),
        popen_fn=popen,
        supervisor_start_fn=supervisor_start_fn or (lambda *_a, **_k: {}),
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


def _fake_horizon_program(parent: chain.SuccessorEvidenceChain, tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        program_id="generation1-successor-horizon-v1",
        path=parent.horizon_program_path,
        file_sha256=chain.sha256_file(parent.horizon_program_path),
        program_sha256=parent.state["horizon_program"]["program_sha256"],
        repo_root=tmp_path.resolve(),
        status_path=tmp_path / "runs/generation1/horizon-v1/current_status.json",
        capsules=(),
    )


def test_v7_adopts_exact_live_parent_under_fresh_identity_and_supersedes_v6(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    live = _process(tmp_path, pid=703, label=spec.process_label, command=(spec.process_label, ""))
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(live,), launches=launches)

    status = parent.tick()

    assert status["schema"] == chain.STATUS_SCHEMA == "mop-generation1-successor-evidence-chain-status/v7"
    assert status["program_id"] == chain.CHAIN_ID == "generation1-successor-evidence-chain-v7"
    assert status["supersedes"] == "generation1-successor-evidence-chain-v6"
    assert status["state"] == "waiting_legacy"
    assert status["horizon_program"]["program_id"] == "generation1-successor-horizon-v1"
    assert launches == []


def test_v7_state_binds_wrapper_and_superseded_v6_and_inherited_v3_base(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    parent = _parent(tmp_path, spec, processes=(), launches=[])

    authority = parent.state["parent_implementation"]

    assert authority["path"].endswith("generation1_successor_chain_v7.py")
    assert authority["sha256"] == chain.sha256_file(Path(chain.__file__).resolve())
    assert authority["superseded_predecessor"]["path"].endswith("generation1_successor_chain_v6.py")
    assert authority["inherited_base"]["path"].endswith("generation1_successor_chain.py")
    assert len(authority["inherited_base"]["sha256"]) == 64


def test_v7_status_validator_requires_supersedes_v6(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    live = _process(tmp_path, pid=760, label=spec.process_label, command=(spec.process_label, ""))
    parent = _parent(tmp_path, spec, processes=(live,), launches=[])
    status = parent.tick()

    assert (
        chain.validate_chain_status(
            status,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )
        == "waiting_legacy"
    )
    tampered = dict(status)
    tampered.pop("status_sha256")
    tampered["supersedes"] = "generation1-successor-evidence-chain-v5"
    resealed = {**tampered, "status_sha256": chain.canonical_sha256(tampered)}
    with pytest.raises(chain.SuccessorChainRefused, match="identity or authority drifted"):
        chain.validate_chain_status(
            resealed,
            repo_root=tmp_path,
            horizon_program_path=parent.horizon_program_path,
            specs=(spec,),
        )


def test_v7_resamples_exact_spawn_title_transition_until_worker_label(tmp_path: Path) -> None:
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to("/usr/bin/python3")
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=730, label=spec.process_label, command=(spec.process_label, ""))
    transition = _spawn_title_transition(tmp_path, pid=731, parent_pid=owner.pid)
    worker_label = f"{spec.child_label_prefixes[0]}0490"
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

    parent = _parent(
        tmp_path,
        spec,
        processes=(),
        launches=[],
        process_table_fn=process_table,
        sleep_fn=sleeps.append,
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert calls == 3
    assert sleeps == [chain.PROCESS_TRANSITION_INTERVAL_SECONDS, chain.PROCESS_TRANSITION_INTERVAL_SECONDS]
    serialized = parent.status_path.read_text(encoding="utf-8")
    assert "MOP_TEST_SECRET" not in serialized
    assert "spawn_main" not in serialized


def test_tolerable_transient_worker_gate_truth_table_v7(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    parent = _parent(tmp_path, spec, processes=(), launches=[])
    owner = _process(tmp_path, pid=714, label=spec.process_label, command=(spec.process_label, ""))
    prefix = spec.child_label_prefixes[0]

    detached_launchd = _process(
        tmp_path, pid=33846, label=f"{prefix}0490", command=(f"{prefix}0490",), pgid=33846, ppid=1
    )
    assert parent._tolerable_transient_worker(detached_launchd, owner, spec) is True

    detached_foreign_parent = _process(
        tmp_path, pid=45242, label=f"{prefix}0493", command=(f"{prefix}0493",), pgid=45242, ppid=999
    )
    assert parent._tolerable_transient_worker(detached_foreign_parent, owner, spec) is True

    reparenting = _process(
        tmp_path, pid=45239, label=f"{prefix}0491", command=(f"{prefix}0491",), pgid=45239, ppid=714
    )
    assert parent._tolerable_transient_worker(reparenting, owner, spec) is True

    regrouping = _process(
        tmp_path, pid=45240, label=f"{prefix}0492", command=(f"{prefix}0492",), pgid=714, ppid=1
    )
    assert parent._tolerable_transient_worker(regrouping, owner, spec) is True

    clean_residual = _process(
        tmp_path, pid=45241, label=f"{prefix}0494", command=(f"{prefix}0494",), pgid=714, ppid=714
    )
    assert parent._tolerable_transient_worker(clean_residual, owner, spec) is False

    foreign_cwd = _process(
        tmp_path,
        pid=45243,
        label=f"{prefix}0495",
        command=(f"{prefix}0495",),
        pgid=45243,
        ppid=714,
        cwd=tmp_path / "other-worktree",
    )
    assert parent._tolerable_transient_worker(foreign_cwd, owner, spec) is False

    wrong_label = _process(
        tmp_path, pid=45244, label="mop-other-0490", command=("mop-other-0490",), pgid=45244, ppid=714
    )
    assert parent._tolerable_transient_worker(wrong_label, owner, spec) is False


def test_v7_detached_worker_that_exits_is_tolerated_regression(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    detached = _process(
        tmp_path,
        pid=33846,
        label=worker_label,
        command=(worker_label, ""),
        pgid=33846,  # its own process group
        ppid=1,  # reparented to launchd mid-exit
    )
    assert detached.ppid != owner.pid and detached.pgid != owner.pid  # genuinely detached
    tables = [(owner, detached), (owner,), (owner,)]
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
        identity_probe_fn=lambda identity: "gone" if identity.get("pid") == detached.pid else "alive",
    )

    status = parent.tick()

    assert status["state"] == "waiting_legacy"
    assert status["capsules"][spec.stage_id]["status"] == "adopted"
    assert status["problems"] == []
    assert status["state"] != "integrity_hold"
    assert launches == []
    assert calls == 3
    assert sleeps == [chain.PROCESS_TRANSITION_INTERVAL_SECONDS, chain.PROCESS_TRANSITION_INTERVAL_SECONDS]


def test_v7_detached_worker_that_persists_still_holds(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    detached = _process(
        tmp_path,
        pid=33846,
        label=worker_label,
        command=(worker_label, ""),
        pgid=33846,
        ppid=1,
    )
    calls = 0
    sleeps: list[float] = []

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return owner, detached

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

    assert status["state"] == "integrity_hold"
    assert "did not stabilize within the bounded window" in status["problems"][-1]
    assert str(detached.pid) in status["problems"][-1]
    assert calls == chain.PROCESS_TRANSITION_ATTEMPTS + 1
    assert len(sleeps) == chain.PROCESS_TRANSITION_ATTEMPTS
    assert launches == []


def test_v7_tolerates_attached_mid_fork_worker_until_clean_residual(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    transient = _process(
        tmp_path,
        pid=45239,
        label=worker_label,
        command=(worker_label, ""),
        pgid=45239,
        ppid=owner.pid,  # attached by ppid, mid-regroup
    )
    resolved = _process(
        tmp_path,
        pid=transient.pid,
        label=worker_label,
        command=(worker_label, ""),
        pgid=owner.pid,
        ppid=owner.pid,
        create_time=transient.create_time,
    )
    tables = [(owner, transient), (owner, resolved), (owner, resolved)]
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


def test_v7_clean_worker_set_adopts_without_resample(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker = _process(
        tmp_path,
        pid=98942,
        label=f"{spec.child_label_prefixes[0]}0490",
        command=(f"{spec.child_label_prefixes[0]}0490",),
        pgid=owner.pid,
        ppid=owner.pid,
    )
    calls = 0
    sleeps: list[float] = []

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return owner, worker

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
    assert calls == 1
    assert sleeps == []
    assert launches == []


def test_v7_tracked_worker_that_turns_foreign_cwd_fails_closed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    transient = _process(
        tmp_path,
        pid=45239,
        label=worker_label,
        command=(worker_label, ""),
        pgid=45239,
        ppid=owner.pid,
    )
    escaped = _process(
        tmp_path,
        pid=transient.pid,
        label=worker_label,
        command=(worker_label, ""),
        pgid=transient.pid,
        ppid=1,
        cwd=tmp_path / "other-worktree",
        create_time=transient.create_time,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return (owner, transient) if calls == 1 else (owner, escaped)

    parent = _parent(tmp_path, spec, processes=(), launches=[], process_table_fn=process_table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "resolved to an unauthorized process class" in status["problems"][-1]
    assert calls == 2


def test_v7_boundary_transient_worker_pid_reuse_fails_closed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    transient = _process(
        tmp_path,
        pid=33846,
        label=worker_label,
        command=(worker_label, ""),
        pgid=33846,
        ppid=1,
    )
    reused = _process(
        tmp_path,
        pid=transient.pid,
        label=worker_label,
        command=(worker_label, ""),
        pgid=owner.pid,
        ppid=owner.pid,
        create_time=transient.create_time + 500.0,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return (owner, transient) if calls == 1 else (owner, reused)

    parent = _parent(tmp_path, spec, processes=(), launches=[], process_table_fn=process_table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "changed identity during stabilization" in status["problems"][-1]
    assert calls == 2


def test_v7_same_repo_worker_outside_ownership_that_persists_fails_closed(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path, pid=714, label=spec.process_label, command=(spec.process_label, "")
    )
    unowned = _process(
        tmp_path,
        pid=715,
        label=f"{spec.child_label_prefixes[0]}0490",
        command=(f"{spec.child_label_prefixes[0]}0490",),
        pgid=715,
        ppid=999,
    )
    calls = 0

    def process_table() -> tuple[chain.ProcessSnapshot, ...]:
        nonlocal calls
        calls += 1
        return parent_process, unowned

    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches, process_table_fn=process_table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "did not stabilize within the bounded window" in status["problems"][-1]
    assert launches == []


@pytest.mark.parametrize(("pgid", "ppid"), [(716, 714), (714, 999)])
def test_v7_foreign_cwd_worker_claiming_ownership_still_fails_closed(
    tmp_path: Path,
    pgid: int,
    ppid: int,
) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    parent_process = _process(
        tmp_path, pid=714, label=spec.process_label, command=(spec.process_label, "")
    )
    foreign = _process(
        tmp_path,
        pid=716,
        label=f"{spec.child_label_prefixes[0]}0490",
        command=(f"{spec.child_label_prefixes[0]}0490",),
        pgid=pgid,
        ppid=ppid,
        cwd=tmp_path / "other-worktree",
    )
    parent = _parent(tmp_path, spec, processes=(parent_process, foreign), launches=[])

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "outside the exact repository-cwd" in status["problems"][-1]


def test_v7_hard_same_group_intruder_surfaces_after_boundary_transient_exits(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=67790, label=spec.process_label, command=(spec.process_label, ""))
    worker_label = f"{spec.child_label_prefixes[0]}0490"
    transient = _process(
        tmp_path,
        pid=33846,
        label=worker_label,
        command=(worker_label, ""),
        pgid=33846,
        ppid=1,
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
        return (owner, transient, intruder) if calls == 1 else (owner, intruder)

    parent = _parent(tmp_path, spec, processes=(), launches=[], process_table_fn=process_table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert "process group has unexpected members" in status["problems"][-1]
    assert str(intruder.pid) in status["problems"][-1]
    assert calls == 2


def test_v7_title_transition_cannot_mask_hard_same_group_intruder(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    _write(spec.status_path, _status(spec))
    owner = _process(tmp_path, pid=736, label=spec.process_label, command=(spec.process_label, ""))
    transition = _spawn_title_transition(tmp_path, pid=737, parent_pid=owner.pid)
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

    parent = _parent(tmp_path, spec, processes=(), launches=[], process_table_fn=process_table)

    status = parent.tick()

    assert status["state"] == "integrity_hold"
    assert str(intruder.pid) in status["problems"][-1]
    assert calls == 1


def test_v7_launches_horizon_v1_when_legacy_complete(
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
        return {}

    monkeypatch.setattr(chain, "load_program", lambda *_a, **_k: program)
    parent.supervisor_start_fn = start

    status = parent.tick()

    assert status["state"] == "waiting_horizon"
    assert observed[0]["capsules"]["successor_horizon"]["status"] == "launching"
    assert status["capsules"]["successor_horizon"]["status"] == "running"


def test_v7_detached_start_emits_one_exact_v7_run_subcommand(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "generation1-successor-evidence-chain-v7"
    launches: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)
    monkeypatch.setattr(chain, "validate_chain_status", lambda *_a, **_k: "waiting_legacy")

    def popen(command: list[str], **kwargs: Any) -> SimpleNamespace:
        launches.append((command, kwargs))
        core = {
            "schema": chain.STATUS_SCHEMA,
            "program_id": chain.CHAIN_ID,
            "state": "waiting_legacy",
            "supervisor": {"pid": 720, "create_time": 72.0},
        }
        _write(root / chain.STATUS_FILE, {**core, "status_sha256": chain.canonical_sha256(core)})
        return SimpleNamespace(pid=720, poll=lambda: None)

    monkeypatch.setattr(chain.subprocess, "Popen", popen)

    result = chain.start_chain_detached(root=root, execute=True, use_caffeinate=False)

    expected = [
        str(chain.REPO_ROOT / ".venv/bin/python"),
        str(chain.REPO_ROOT / "scripts/mop_generation1_successor_chain_v7.py"),
        "run",
        "--execute",
        "--root",
        str(root),
    ]
    assert launches[0][0] == expected
    assert launches[0][0].count("run") == 1
    assert result["launched_pid"] == 720
