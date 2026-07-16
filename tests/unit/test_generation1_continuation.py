from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mop.studio import generation1_continuation as continuation


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _plan_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> continuation.ContinuationPlan:
    monkeypatch.setattr(continuation, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(
        continuation,
        "_identity",
        lambda: {
            "pid": 123,
            "create_time": 456.0,
            "implementation_path": "router.py",
            "implementation_sha256": "0" * 64,
        },
    )
    authority = tmp_path / "router.py"
    authority.write_text("# immutable router\n", encoding="utf-8")
    prerequisite = tmp_path / "configs/prerequisite.json"
    target = tmp_path / "configs/target.json"
    _write(prerequisite, {"program": "prerequisite"})
    _write(target, {"program": "target"})
    prerequisite_program_sha = "1" * 64
    target_program_sha = "2" * 64

    def fake_load(reference: continuation.ProgramReference, label: str) -> SimpleNamespace:
        expected = prerequisite_program_sha if label == "prerequisite" else target_program_sha
        return SimpleNamespace(program_id=label, program_sha256=expected)

    monkeypatch.setattr(continuation, "_load_bound_program", fake_load)
    core = {
        "schema": continuation.PLAN_SCHEMA,
        "router_id": "test-continuation",
        "out_dir": "runs/test-continuation",
        "prerequisite": {
            "path": "configs/prerequisite.json",
            "file_sha256": continuation.sha256_file(prerequisite),
            "program_sha256": prerequisite_program_sha,
        },
        "target": {
            "path": "configs/target.json",
            "file_sha256": continuation.sha256_file(target),
            "program_sha256": target_program_sha,
        },
        "authorities": [
            {"path": "router.py", "sha256": continuation.sha256_file(authority)}
        ],
        "control": {"poll_seconds": 1, "startup_ack_seconds": 1},
    }
    plan_path = tmp_path / "configs/continuation.json"
    _write(plan_path, {**core, "plan_sha256": continuation.canonical_sha256(core)})
    return continuation.load_plan(plan_path)


def test_load_plan_rejects_authority_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_fixture(tmp_path, monkeypatch)
    assert plan.router_id == "test-continuation"
    (tmp_path / "router.py").write_text("# drifted\n", encoding="utf-8")
    with pytest.raises(continuation.ContinuationRefused, match="authority drifted"):
        continuation.validate_plan_authority(plan)


def test_continuation_waits_for_prerequisite_without_starting_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_fixture(tmp_path, monkeypatch)
    prerequisite = SimpleNamespace(program_id="prerequisite")
    target = SimpleNamespace(program_id="target")
    monkeypatch.setattr(
        continuation,
        "validate_plan_authority",
        lambda _plan: (prerequisite, target),
    )
    monkeypatch.setattr(
        continuation,
        "_program_status",
        lambda program: {
            "state": "resource_wait",
            "supervisor": {"pid": 1, "create_time": 1.0},
        }
        if program is prerequisite
        else pytest.fail("target status must not be read before prerequisite completion"),
    )
    monkeypatch.setattr(continuation, "_process_alive", lambda _identity: True)
    monkeypatch.setattr(
        continuation,
        "_ensure_running",
        lambda _program: pytest.fail("an alive prerequisite must not be restarted"),
    )
    status = continuation.run_continuation(plan, execute=True, max_cycles=1)
    assert status["state"] == "waiting_prerequisite"
    assert status["prerequisite_state"] == "resource_wait"
    assert status["target_state"] is None


def test_continuation_recovers_dead_prerequisite_then_waits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_fixture(tmp_path, monkeypatch)
    prerequisite = SimpleNamespace(program_id="prerequisite")
    target = SimpleNamespace(program_id="target")
    monkeypatch.setattr(
        continuation,
        "validate_plan_authority",
        lambda _plan: (prerequisite, target),
    )
    statuses = iter(
        [
            {"state": "resource_wait", "supervisor": {"dead": True}},
            {"state": "resource_wait", "supervisor": {"alive": True}},
        ]
    )
    monkeypatch.setattr(continuation, "_program_status", lambda _program: next(statuses))
    monkeypatch.setattr(
        continuation,
        "_process_alive",
        lambda identity: bool(identity.get("alive")),
    )
    starts: list[str] = []
    monkeypatch.setattr(
        continuation,
        "_ensure_running",
        lambda program: starts.append(program.program_id) or {},
    )
    status = continuation.run_continuation(plan, execute=True, max_cycles=1)
    assert starts == ["prerequisite"]
    assert status["state"] == "waiting_prerequisite"
    assert status["prerequisite_start_requests"] == 1


def test_continuation_completes_only_after_target_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan_fixture(tmp_path, monkeypatch)
    prerequisite = SimpleNamespace(program_id="prerequisite")
    target = SimpleNamespace(program_id="target")
    monkeypatch.setattr(
        continuation,
        "validate_plan_authority",
        lambda _plan: (prerequisite, target),
    )
    monkeypatch.setattr(
        continuation,
        "_program_status",
        lambda program: {
            "state": "complete",
            "supervisor": {"pid": 1, "create_time": 1.0},
        },
    )
    monkeypatch.setattr(
        continuation,
        "_ensure_running",
        lambda _program: pytest.fail("complete programs must not be restarted"),
    )
    status = continuation.run_continuation(plan, execute=True)
    assert status["state"] == "complete"
    assert status["prerequisite_state"] == "complete"
    assert status["target_state"] == "complete"
    assert status["finished_at"] is not None
