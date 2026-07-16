from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mop.studio import generation1_successor_chain_v4 as chain

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
) -> chain.ProcessSnapshot:
    return chain.ProcessSnapshot(
        pid=pid,
        create_time=float(pid) + 0.25,
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
        process_table_fn=lambda: processes,
        identity_probe_fn=lambda _identity: "gone",
        popen_fn=popen,
        now_fn=lambda: NOW,
        sleep_fn=lambda _seconds: None,
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
        "supervisor": dict(supervisor or {"pid": 999, "create_time": 99.0}),
        "execution_enabled": True,
        "state": state,
        "queue_head_sha256": "a" * 64,
        "next_injection_sequence": 1,
        "accepted_injection_count": 0,
        "current_capsule": None,
        "capsules": {},
        "last_admission": None,
        "lane_reservation": None,
        "problems": list(problems or []),
    }
    return {**core, "status_sha256": chain.canonical_sha256(core)}


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


def test_v4_adopts_exact_live_parent_under_fresh_identity_and_receipt_schema(
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
    assert status["supersedes"] == "generation1-successor-evidence-chain-v3"
    assert status["state"] == "waiting_legacy"
    assert status["horizon_program"]["program_id"] == "generation1-successor-horizon-v1"
    assert launches == []
    receipt_path = next((parent.root / "adoptions").glob("*/*.json"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == chain.ADOPTION_SCHEMA
    assert receipt["chain_id"] == chain.CHAIN_ID
    assert receipt["policy"]["signals_allowed"] is False
    assert receipt["policy"]["restart_command_match"] == ("exact-executable-and-two-argv-shape")


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


def test_v4_state_binds_wrapper_and_inherited_v3_base(tmp_path: Path) -> None:
    spec = _spec(tmp_path)
    launches: list[tuple[list[str], dict[str, Any]]] = []
    parent = _parent(tmp_path, spec, processes=(), launches=launches)

    authority = parent.state["parent_implementation"]

    assert authority["path"].endswith("generation1_successor_chain_v4.py")
    assert authority["sha256"] == chain.sha256_file(Path(chain.__file__).resolve())
    assert authority["inherited_base"]["path"].endswith("generation1_successor_chain.py")
    assert len(authority["inherited_base"]["sha256"]) == 64


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
    entrypoint = chain.REPO_ROOT / "scripts/mop_generation1_successor_chain_v4.py"
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
    root = tmp_path / "generation1-successor-evidence-chain-v4"
    launches: list[tuple[list[str], dict[str, Any]]] = []
    monkeypatch.setattr(chain, "_visible_parent_process", lambda _entrypoint: None)

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
        str(chain.REPO_ROOT / "scripts/mop_generation1_successor_chain_v4.py"),
        "run",
        "--execute",
        "--root",
        str(root),
    ]
    assert launches[0][0] == expected
    assert launches[0][0].count("run") == 1
    assert launches[0][1]["start_new_session"] is True
    assert result["launched_pid"] == 720


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
    root = tmp_path / "generation1-successor-evidence-chain-v4"
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
        command=("mop-successor-evidence-chain-v4",),
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
    assert row["process"] == supervisor
    assert launches == []
