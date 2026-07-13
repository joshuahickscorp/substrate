from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from mop.studio import generation1_supervisor as g1


def _seal(value: dict[str, object], field: str) -> dict[str, object]:
    return {**value, field: g1.canonical_sha256(value)}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _capsule(
    *,
    capsule_id: str,
    artifact: str,
    authority_path: str,
    authority_sha256: str,
    kind: str = "corpus",
    priority: int = 0,
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": g1.CAPSULE_SCHEMA,
        "id": capsule_id,
        "kind": kind,
        "priority": priority,
        "depends_on": depends_on or [],
        "command": ["python", authority_path, "--out", artifact],
        "cwd": ".",
        "environment": {"OMP_NUM_THREADS": "4"},
        "resources": {
            "lane": "cpu",
            "accelerator": "none",
            "cpu_cores": 4,
            "estimated_unified_memory_gb": 2.0,
            "estimated_mps_gb": 0.0,
            "resource_basis": "bounded test corpus worker pool",
            "forecast_write_gb": 0.1,
            "atomic_write_gb": 0.01,
            "wall_minutes": 5,
            "process_marker": authority_path,
        },
        "artifacts": [
            {
                "path": artifact,
                "schema": "mop-generation1-test-result/v1",
                "fields": {"ok": True},
                "seal_field": "payload_sha256",
            }
        ],
        "authorities": [{"path": authority_path, "sha256": authority_sha256}],
    }
    return _seal(core, "capsule_sha256")


def _program(
    root: Path,
    capsules: list[dict[str, object]],
    *,
    runner_source: str = "# immutable test runner\n",
) -> tuple[Path, Path, str]:
    runner = root / "runner.py"
    runner.write_text(runner_source)
    policy = root / "policy.yaml"
    policy.write_text("schema: fixture\n")
    runner_sha = g1.sha256_file(runner)
    for capsule in capsules:
        authorities = capsule["authorities"]
        assert isinstance(authorities, list)
        authorities[0] = {"path": "runner.py", "sha256": runner_sha}
        command = capsule["command"]
        assert isinstance(command, list)
        command[0] = sys.executable
        command[1] = "runner.py"
        resources = capsule["resources"]
        assert isinstance(resources, dict)
        resources["process_marker"] = "runner.py"
        capsule.pop("capsule_sha256")
        capsule["capsule_sha256"] = g1.canonical_sha256(capsule)
    core: dict[str, object] = {
        "schema": g1.PROGRAM_SCHEMA,
        "program_id": "generation1-test",
        "program_root": "runs/generation1-test",
        "policy": {"path": "policy.yaml", "sha256": g1.sha256_file(policy)},
        "authorities": [{"path": "runner.py", "sha256": runner_sha}],
        "injection": {
            "inbox": "runs/generation1-test/control/inbox",
            "receipt_root": "runs/generation1-test/injection_receipts",
        },
        "control": {
            "throttle_state_root": "runs/local_throttle",
            "admission_samples": 1,
            "admission_interval_seconds": 0.01,
            "resource_retry_seconds": 0.01,
            "startup_ack_seconds": 1.0,
        },
        "capsules": capsules,
    }
    manifest = _seal(core, "program_sha256")
    path = root / "configs/program.json"
    _write_json(path, manifest)
    return path, runner, runner_sha


def _base_capsule(root: Path, capsule_id: str = "fresh-seed") -> dict[str, object]:
    runner = root / "placeholder.py"
    runner.write_text("# replaced by _program\n")
    return _capsule(
        capsule_id=capsule_id,
        artifact=f"proof/{capsule_id}.json",
        authority_path="placeholder.py",
        authority_sha256=g1.sha256_file(runner),
    )


def _allowed(_capsule: g1.Capsule) -> g1.Admission:
    return g1.Admission(True, {"allowed": True, "denied_reasons": []})


def _denied(_capsule: g1.Capsule) -> g1.Admission:
    return g1.Admission(False, {"allowed": False, "denied_reasons": ["memory pressure"]})


def _reserve_lane(
    capsule: g1.Capsule,
    child_pid: int | None,
    child_create_time: float | None,
) -> Mapping[str, object]:
    return {
        "reserved": True,
        "capsule_id": capsule.capsule_id,
        "child_pid": child_pid,
        "child_create_time": child_create_time,
    }


def _release_lane(_capsule: g1.Capsule) -> None:
    return None


def _runtime_policy(interval: float = 0.05) -> SimpleNamespace:
    return SimpleNamespace(
        monitor={
            "sample_interval_seconds": interval,
            "graceful_stop_seconds": 0.2,
            "pause_bad_samples": 1,
        }
    )


def _runtime_sample(
    *,
    rss: int = 100_000_000,
    memory_gb: float = 70.0,
    memory_percent: float = 72.0,
    pressure_percent: float = 76.0,
    swap_gb: float = 0.0,
    disk_gb: float = 200.0,
    allowed: bool = True,
    critical: bool = False,
) -> dict[str, object]:
    return {
        "created_at": "2026-07-13T00:00:00+00:00",
        "child": {"pid": 123, "create_time": 10.0, "identity_state": "alive"},
        "host": {
            "memory_available_gb": memory_gb,
            "memory_available_percent": memory_percent,
            "memory_pressure_available": True,
            "memory_pressure_free_percent": pressure_percent,
            "swap_used_gb": swap_gb,
            "disk_free_gb": disk_gb,
            "thermal_available": True,
            "thermal_status": "normal",
            "power_available": True,
            "power_source": "AC Power",
            "power_on_ac": True,
            "missing_required_telemetry": [],
        },
        "process_tree_rss_bytes": rss,
        "declared_process_tree_rss_bytes": 2_000_000_000,
        "allowed": allowed,
        "critical": critical,
        "denied_reasons": [] if allowed else ["runtime resource threshold crossed"],
        "disk_forecast": {
            "free_gb": disk_gb,
            "projected_free_gb": disk_gb - 1.0,
            "floor_gb": 40.0,
        },
    }


SUCCESS_RUNNER = """
import argparse
import hashlib
import json
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
time.sleep(0.18)
core = {"schema": "mop-generation1-test-result/v1", "ok": True}
encoded = json.dumps(
    core, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
).encode("utf-8")
payload = {**core, "payload_sha256": hashlib.sha256(encoded).hexdigest()}
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(payload, sort_keys=True) + "\\n")
"""


SLEEP_RUNNER = """
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--out", required=True)
parser.parse_args()
time.sleep(30.0)
"""


def _runner(calls: list[str], program: g1.Program) -> g1.Runner:
    def run(
        capsule: g1.Capsule,
        stdout_path: Path,
        stderr_path: Path,
        _environment: Mapping[str, str],
        on_start: g1.StartCallback,
    ) -> g1.CommandResult:
        calls.append(capsule.capsule_id)
        on_start(123, 10.0)
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("ok\n")
        stderr_path.write_text("")
        for expectation in capsule.artifacts:
            core = {"schema": expectation.schema, "ok": True}
            payload = _seal(core, expectation.seal_field) if expectation.seal_field is not None else core
            g1.atomic_write_json(program.repo_root / expectation.path, payload)
        return g1.CommandResult(
            returncode=0,
            child_pid=123,
            child_create_time=10.0,
            started_at="2026-07-13T00:00:00+00:00",
            finished_at="2026-07-13T00:00:01+00:00",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    return run


def _supervisor(
    program: g1.Program,
    *,
    admission: g1.AdmissionProbe = _allowed,
    calls: list[str] | None = None,
) -> g1.Generation1Supervisor:
    return g1.Generation1Supervisor(
        program,
        execute=True,
        admission_probe=admission,
        runner=_runner(calls if calls is not None else [], program),
        lane_reserver=_reserve_lane,
        lane_releaser=_release_lane,
        now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
        sleep_fn=lambda _seconds: None,
    )


def _injection(
    *,
    program: g1.Program,
    capsule: dict[str, object],
    injection_id: str = "inj-000001-extra",
    sequence: int = 1,
    head: str | None = None,
) -> dict[str, object]:
    core: dict[str, object] = {
        "schema": g1.INJECTION_SCHEMA,
        "program_id": program.program_id,
        "injection_id": injection_id,
        "sequence": sequence,
        "created_at": "2026-07-13T00:00:00+00:00",
        "action": "append-capsules",
        "expected_queue_head_sha256": head or g1.canonical_sha256({"stale": True}),
        "capsules": [capsule],
        "reason": "additional exploratory cognitive angle",
    }
    return _seal(core, "injection_sha256")


def test_manifest_is_self_hashed_and_rejects_drift(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    assert program.program_id == "generation1-test"

    payload = json.loads(path.read_text())
    payload["program_id"] = "changed"
    _write_json(path, payload)
    with pytest.raises(g1.Generation1Refused, match="self-seal mismatch"):
        g1.load_program(path, repo_root=tmp_path)


def test_deterministic_priority_then_id_schedule_and_dependency_chain(tmp_path: Path) -> None:
    high = _base_capsule(tmp_path, "z-corpus")
    high["priority"] = 10
    high.pop("capsule_sha256")
    high["capsule_sha256"] = g1.canonical_sha256(high)
    low = _base_capsule(tmp_path, "a-corpus")
    aggregate = _base_capsule(tmp_path, "aggregate")
    aggregate["kind"] = "aggregate"
    aggregate["depends_on"] = ["a-corpus", "z-corpus"]
    aggregate.pop("capsule_sha256")
    aggregate["capsule_sha256"] = g1.canonical_sha256(aggregate)
    path, _runner_path, _sha = _program(tmp_path, [high, aggregate, low])
    program = g1.load_program(path, repo_root=tmp_path)
    calls: list[str] = []

    status = _supervisor(program, calls=calls).run()

    assert status["state"] == "complete"
    assert calls == ["a-corpus", "z-corpus", "aggregate"]
    assert g1.read_status(program)["status_sha256"] == status["status_sha256"]


def test_resource_refusal_waits_without_running_or_changing_capsule(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    calls: list[str] = []

    status = _supervisor(program, admission=_denied, calls=calls).run(max_cycles=1)

    assert status["state"] == "resource_wait"
    assert calls == []
    assert status["capsules"]["fresh-seed"]["status"] == "pending"
    assert status["last_admission"]["denied_reasons"] == ["memory pressure"]


def test_exploratory_injection_is_accepted_at_boundary_and_hash_chained(tmp_path: Path) -> None:
    path, _runner_path, runner_sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    calls: list[str] = []
    supervisor = _supervisor(program, calls=calls)
    extra = _capsule(
        capsule_id="extra-angle",
        artifact="proof/extra-angle.json",
        authority_path="runner.py",
        authority_sha256=runner_sha,
        kind="exploratory",
        priority=50,
        depends_on=["fresh-seed"],
    )
    request = _injection(
        program=program,
        capsule=extra,
        head=supervisor.state["queue_head_sha256"],
    )
    injection_path = program.inbox / "inj-000001-extra.json"
    _write_json(injection_path, request)

    status = supervisor.run()

    assert status["state"] == "complete"
    assert calls == ["fresh-seed", "extra-angle"]
    assert status["accepted_injection_count"] == 1
    assert status["next_injection_sequence"] == 2
    receipts = list((program.receipt_root / "accepted").glob("*.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["accepted"] is True


def test_stale_injection_is_rejected_without_changing_queue(tmp_path: Path) -> None:
    path, _runner_path, runner_sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = _supervisor(program)
    original_head = supervisor.state["queue_head_sha256"]
    extra = _capsule(
        capsule_id="stale-extra",
        artifact="proof/stale-extra.json",
        authority_path="runner.py",
        authority_sha256=runner_sha,
        kind="exploratory",
    )
    _write_json(program.inbox / "stale.json", _injection(program=program, capsule=extra))

    status = supervisor.run()

    assert status["state"] == "complete"
    assert status["accepted_injection_count"] == 0
    assert status["queue_head_sha256"] == original_head
    receipt = json.loads(next((program.receipt_root / "rejected").glob("*.json")).read_text())
    assert receipt["accepted"] is False
    assert "stale" in " ".join(receipt["problems"])


def test_accepted_injection_mutation_causes_integrity_hold(tmp_path: Path) -> None:
    path, _runner_path, runner_sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = _supervisor(program)
    extra = _capsule(
        capsule_id="extra",
        artifact="proof/extra.json",
        authority_path="runner.py",
        authority_sha256=runner_sha,
        kind="exploratory",
        depends_on=["fresh-seed"],
    )
    injection_path = program.inbox / "inj.json"
    request = _injection(
        program=program,
        capsule=extra,
        head=supervisor.state["queue_head_sha256"],
    )
    _write_json(injection_path, request)
    assert supervisor.run()["state"] == "complete"
    request["reason"] = "mutated after acceptance"
    request.pop("injection_sha256")
    request["injection_sha256"] = g1.canonical_sha256(request)
    _write_json(injection_path, request)

    status = _supervisor(program).run(max_cycles=1)

    assert status["state"] == "integrity_hold"
    assert "accepted injection file changed" in " ".join(status["problems"])


def test_injection_cannot_add_claim_bearing_capsule(tmp_path: Path) -> None:
    path, _runner_path, runner_sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = _supervisor(program)
    aggregate = _capsule(
        capsule_id="injected-aggregate",
        artifact="proof/injected-aggregate.json",
        authority_path="runner.py",
        authority_sha256=runner_sha,
        kind="aggregate",
    )
    _write_json(
        program.inbox / "bad-kind.json",
        _injection(
            program=program,
            capsule=aggregate,
            head=supervisor.state["queue_head_sha256"],
        ),
    )

    status = supervisor.run()

    assert status["state"] == "complete"
    assert status["accepted_injection_count"] == 0
    receipt = json.loads(next((program.receipt_root / "rejected").glob("*.json")).read_text())
    assert "kind is not permitted" in " ".join(receipt["problems"])


def test_detached_start_requires_explicit_execution(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)

    with pytest.raises(g1.Generation1Refused, match="explicit --execute"):
        g1.start_detached(program, execute=False, use_caffeinate=False)


def test_resumable_command_failure_retries_then_completes(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    calls: list[str] = []
    success = _runner(calls, program)

    def flaky(
        capsule: g1.Capsule,
        stdout_path: Path,
        stderr_path: Path,
        environment: Mapping[str, str],
        on_start: g1.StartCallback,
    ) -> g1.CommandResult:
        if len(calls) >= 2:
            return success(capsule, stdout_path, stderr_path, environment, on_start)
        calls.append(capsule.capsule_id)
        on_start(123, 10.0)
        return g1.CommandResult(
            returncode=2,
            child_pid=123,
            child_create_time=10.0,
            started_at="2026-07-13T00:00:00+00:00",
            finished_at="2026-07-13T00:00:01+00:00",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    supervisor = g1.Generation1Supervisor(
        program,
        execute=True,
        admission_probe=_allowed,
        runner=flaky,
        lane_reserver=_reserve_lane,
        lane_releaser=_release_lane,
        now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
        sleep_fn=lambda _seconds: None,
    )

    status = supervisor.run()

    assert status["state"] == "complete"
    assert calls == ["fresh-seed", "fresh-seed", "fresh-seed"]
    assert status["capsules"]["fresh-seed"]["attempts"] == g1.MAX_CAPSULE_ATTEMPTS


def test_command_failure_holds_after_bounded_retries(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    calls: list[str] = []

    def always_fails(
        capsule: g1.Capsule,
        stdout_path: Path,
        stderr_path: Path,
        _environment: Mapping[str, str],
        on_start: g1.StartCallback,
    ) -> g1.CommandResult:
        calls.append(capsule.capsule_id)
        on_start(123, 10.0)
        return g1.CommandResult(
            returncode=2,
            child_pid=123,
            child_create_time=10.0,
            started_at="2026-07-13T00:00:00+00:00",
            finished_at="2026-07-13T00:00:01+00:00",
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
        )

    supervisor = g1.Generation1Supervisor(
        program,
        execute=True,
        admission_probe=_allowed,
        runner=always_fails,
        lane_reserver=_reserve_lane,
        lane_releaser=_release_lane,
        now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
        sleep_fn=lambda _seconds: None,
    )

    status = supervisor.run()

    assert status["state"] == "failure_hold"
    assert len(calls) == g1.MAX_CAPSULE_ATTEMPTS
    assert status["capsules"]["fresh-seed"]["status"] == "failed"


def test_atomic_generation1_lane_uses_shared_throttle_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = g1.Generation1Supervisor(program, execute=True, admission_probe=_allowed)
    capsule = program.capsules[0]
    reservations: list[tuple[Path, str, g1.TaskDeclaration]] = []
    updates: list[tuple[Path, str, object]] = []

    monkeypatch.setattr(supervisor, "_load_live_policy", lambda: object())
    monkeypatch.setattr(g1, "collect_host_telemetry", lambda *_args, **_kwargs: {"ok": True})

    def reserve(
        root: Path,
        run_id: str,
        task: g1.TaskDeclaration,
        _telemetry: object,
        _policy: object,
    ) -> dict[str, object]:
        reservations.append((root, run_id, task))
        return {"allowed": True, "atomic": True}

    def update(root: Path, run_id: str, row: object) -> None:
        updates.append((root, run_id, row))

    monkeypatch.setattr(g1.local_throttle_runtime, "_reserve_registry", reserve)
    monkeypatch.setattr(g1.local_throttle_runtime, "_update_registry", update)

    supervisor._reserve_lane(capsule)
    supervisor._release_lane(capsule)

    assert len(reservations) == 1
    assert reservations[0][0] == program.throttle_state_root
    assert reservations[0][1] == "generation1-generation1-test"
    assert reservations[0][2].requires_empty_lanes is True
    assert updates == [(program.throttle_state_root, "generation1-generation1-test", None)]
    runtime = supervisor.state["capsules"]["fresh-seed"]["runtime"]
    assert runtime["reservation_count"] == 1
    assert [event["event"] for event in runtime["events"]] == [
        "lane-reserved",
        "lane-released",
    ]


def test_live_runtime_monitor_publishes_compact_durable_resource_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _runner_path, _sha = _program(
        tmp_path,
        [_base_capsule(tmp_path)],
        runner_source=SUCCESS_RUNNER,
    )
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = g1.Generation1Supervisor(
        program,
        execute=True,
        admission_probe=_allowed,
        lane_reserver=_reserve_lane,
        lane_releaser=_release_lane,
        now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
        sleep_fn=lambda _seconds: None,
    )
    monkeypatch.setattr(supervisor, "_load_live_policy", lambda: _runtime_policy())
    sample_count = 0

    def sample(
        _capsule: g1.Capsule,
        child_pid: int,
        child_create_time: float,
    ) -> dict[str, object]:
        nonlocal sample_count
        sample_count += 1
        value = _runtime_sample(
            rss=100_000_000 + sample_count * 25_000_000,
            memory_gb=80.0 - sample_count,
            memory_percent=75.0 - sample_count,
            pressure_percent=78.0 - sample_count,
            swap_gb=sample_count / 10,
            disk_gb=210.0 - sample_count,
        )
        value["child"] = {
            "pid": child_pid,
            "create_time": child_create_time,
            "identity_state": "alive",
        }
        return value

    monkeypatch.setattr(supervisor, "_sample_runtime", sample)

    status = supervisor.run()

    assert status["state"] == "complete"
    runtime = status["capsules"]["fresh-seed"]["runtime"]
    assert runtime["sample_count"] >= 2
    assert runtime["peak_process_tree_rss_bytes"] == 100_000_000 + sample_count * 25_000_000
    assert runtime["minimum_memory_available_gb"] == 80.0 - sample_count
    assert runtime["minimum_memory_available_percent"] == 75.0 - sample_count
    assert runtime["minimum_memory_pressure_free_percent"] == 78.0 - sample_count
    assert runtime["maximum_swap_used_gb"] == sample_count / 10
    assert runtime["minimum_disk_free_gb"] == 210.0 - sample_count
    assert runtime["thermal_statuses"] == ["normal"]
    assert runtime["power_sources"] == ["AC Power"]
    assert runtime["reservation_count"] == 1
    assert runtime["last_reservation"]["receipt"]["reserved"] is True
    assert runtime["resource_stop_count"] == 0
    assert runtime["retry_count"] == 0
    assert g1.read_status(program)["capsules"]["fresh-seed"]["runtime"] == runtime


def test_live_resource_stop_signals_only_owned_groups_and_retries_are_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _runner_path, _sha = _program(
        tmp_path,
        [_base_capsule(tmp_path)],
        runner_source=SLEEP_RUNNER,
    )
    program = g1.load_program(path, repo_root=tmp_path)
    supervisor = g1.Generation1Supervisor(
        program,
        execute=True,
        admission_probe=_allowed,
        lane_reserver=_reserve_lane,
        lane_releaser=_release_lane,
        now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
        sleep_fn=lambda _seconds: None,
    )
    monkeypatch.setattr(supervisor, "_load_live_policy", lambda: _runtime_policy())
    monkeypatch.setattr(
        supervisor,
        "_sample_runtime",
        lambda _capsule, _pid, _create_time: _runtime_sample(
            allowed=False,
            critical=True,
            memory_gb=1.0,
            memory_percent=2.0,
            pressure_percent=2.0,
            swap_gb=8.0,
            disk_gb=39.0,
        ),
    )
    original_killpg = g1.os.killpg
    signaled_groups: list[int] = []

    def signal_group(pid: int, requested_signal: int) -> None:
        signaled_groups.append(pid)
        original_killpg(pid, requested_signal)

    monkeypatch.setattr(g1.os, "killpg", signal_group)

    status = supervisor.run()

    assert status["state"] == "failure_hold"
    row = status["capsules"]["fresh-seed"]
    runtime = row["runtime"]
    child_pids = {event["pid"] for event in runtime["events"] if event["event"] == "child-started"}
    assert len(child_pids) == g1.MAX_CAPSULE_ATTEMPTS
    assert set(signaled_groups) <= child_pids
    assert runtime["resource_stop_count"] == g1.MAX_CAPSULE_ATTEMPTS
    assert runtime["retry_count"] == g1.MAX_CAPSULE_ATTEMPTS - 1
    assert runtime["maximum_swap_used_gb"] == 8.0
    assert runtime["minimum_disk_free_gb"] == 39.0
    assert row["attempts"] == g1.MAX_CAPSULE_ATTEMPTS


def test_crash_recovery_retries_when_recorded_child_is_gone(tmp_path: Path) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    original = _supervisor(program)
    row = original.state["capsules"]["fresh-seed"]
    row["status"] = "running"
    row["attempts"] = 1
    row["child_pid"] = 99_999_999
    row["child_create_time"] = 1.0
    original.state["current_capsule"] = "fresh-seed"
    original.state["status"] = "running"
    original._publish()
    calls: list[str] = []

    status = _supervisor(program, calls=calls).run()

    assert status["state"] == "complete"
    assert calls == ["fresh-seed"]
    recovered = status["capsules"]["fresh-seed"]
    assert recovered["attempts"] == 2
    assert recovered["runtime"]["retry_count"] == 1
    assert "recovery-child-gone" in [event["event"] for event in recovered["runtime"]["events"]]


def test_crash_recovery_observes_exact_live_child_then_retries_after_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _runner_path, _sha = _program(tmp_path, [_base_capsule(tmp_path)])
    program = g1.load_program(path, repo_root=tmp_path)
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    try:
        child_create_time = g1.psutil.Process(child.pid).create_time()
        original = _supervisor(program)
        row = original.state["capsules"]["fresh-seed"]
        row["status"] = "running"
        row["attempts"] = 1
        row["child_pid"] = child.pid
        row["child_create_time"] = child_create_time
        original.state["current_capsule"] = "fresh-seed"
        original.state["status"] = "running"
        original._publish()
        calls: list[str] = []
        reservations: list[int | None] = []
        releases: list[str] = []

        def reserve(
            capsule: g1.Capsule,
            child_pid: int | None,
            _child_create_time: float | None,
        ) -> Mapping[str, object]:
            reservations.append(child_pid)
            return {"reserved": True, "capsule_id": capsule.capsule_id}

        def release(capsule: g1.Capsule) -> None:
            releases.append(capsule.capsule_id)

        supervisor = g1.Generation1Supervisor(
            program,
            execute=True,
            admission_probe=_allowed,
            runner=_runner(calls, program),
            lane_reserver=reserve,
            lane_releaser=release,
            now_fn=lambda: datetime(2026, 7, 13, tzinfo=UTC),
            sleep_fn=lambda _seconds: None,
        )
        monkeypatch.setattr(
            supervisor,
            "_sample_runtime",
            lambda _capsule, _pid, _create_time: _runtime_sample(),
        )
        monkeypatch.setattr(
            g1.os,
            "killpg",
            lambda *_args: pytest.fail("recovery observation must not signal a process"),
        )

        observed = supervisor.run(max_cycles=1)

        assert observed["state"] == "recovery_observe"
        assert observed["capsules"]["fresh-seed"]["status"] == "running"
        assert calls == []
        assert reservations == [child.pid]
        child.terminate()
        child.wait(timeout=5)

        completed = supervisor.run()

        assert completed["state"] == "complete"
        assert calls == ["fresh-seed"]
        assert reservations == [child.pid, None]
        assert releases == ["fresh-seed", "fresh-seed"]
        assert completed["capsules"]["fresh-seed"]["attempts"] == 2
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)


def test_inexact_child_identity_never_receives_a_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    signaled: list[int] = []
    monkeypatch.setattr(g1.os, "killpg", lambda pid, _signal: signaled.append(pid))
    try:
        with pytest.raises(g1.Generation1Refused, match="no process signal"):
            g1.Generation1Supervisor._signal_exact_owned_group(
                child,
                1.0,
                g1.signal.SIGTERM,
            )
        assert signaled == []
    finally:
        child.terminate()
        child.wait(timeout=5)
