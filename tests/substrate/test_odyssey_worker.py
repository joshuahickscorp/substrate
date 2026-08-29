"""Focused execution-boundary tests for the deterministic Odyssey worker."""
from __future__ import annotations

import json
import shlex
import sys
import time
from pathlib import Path

import pytest

from substrate import odyssey_task_bank as task_bank
from substrate import odyssey_worker as worker


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _adapter(path: Path, *, fail_once_cycle: int | None = None) -> None:
    path.write_text(
        """import json
import sys
from pathlib import Path

request = json.loads(Path(sys.argv[1]).read_text())
failure_cycle = __FAILURE_CYCLE__
failure_marker = Path('.adapter-failed-once')
if failure_cycle is not None and request['cycle'] == failure_cycle and not failure_marker.exists():
    failure_marker.write_text('interrupted')
    raise SystemExit(31)
receipt = {
    'schema': 'SUBSTRATE_ODYSSEY_ADAPTER_RECEIPT/v1',
    'activation': False,
    'authority_sha256': request['authority_sha256'],
    'run_id': request['run_id'],
    'frontier': request['frontier'],
    'role': request['role'],
    'cycle': request['cycle'],
    'phase': request['phase'],
    'task_id': request['task']['task_id'],
    'candidate_manifest_sha256': request['candidate_manifest_sha256'],
    'request_sha256': request['request_sha256'],
    'elapsed_seconds': 0.001,
}
output = Path(request['receipt_path'])
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(receipt, sort_keys=True))
""".replace("__FAILURE_CYCLE__", repr(fail_once_cycle)),
        encoding="utf-8",
    )


def _authority(
    root: Path,
    *,
    program: str = worker.TEST_PROGRAM,
    microcycles: int = 1,
    fail_once_cycle: int | None = None,
) -> Path:
    candidate, _ = task_bank.materialize(
        worker._digest({"seed": "worker-fixture-seed"}),
        "worker-fixture-seed",
        "A",
        microcycles,
    )
    candidate_path = root / "candidate-visible/A.json"
    _write(candidate_path, candidate)
    adapter = root / "adapter.py"
    _adapter(adapter, fail_once_cycle=fail_once_cycle)
    command = shlex.join([sys.executable, str(adapter)])
    body = {
        "schema": "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "run_id": "worker-fixture-001",
        "program": {
            "id": program,
            "duration_seconds": 0 if program == worker.TEST_PROGRAM else 7 * 24 * 3600,
            "launch_allowed": True,
        },
        "seal": {"status": "sealed"},
        "launch_gates": [{"id": "G01", "status": "pass"}],
        "worker_source_sha256": worker.file_digest(Path(worker.__file__)),
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
        "worker": {
            "test_mode": program == worker.TEST_PROGRAM,
            "run_root": "runs/worker-fixture",
            "publication_root": "evidence/worker-fixture",
            "frontiers": [
                {
                    "id": "A",
                    "candidate_manifest": str(candidate_path.relative_to(root)),
                    "candidate_manifest_sha256": worker.file_digest(candidate_path),
                    "candidate_command": command,
                    "control_command": command,
                }
            ],
            "phase_names": ["retrieval"],
            "phase_seconds": 0,
            "microcycles_per_frontier": microcycles,
            "max_parallel_frontiers": 1,
        },
    }
    body["sha256"] = worker._digest(body)
    name = "ODYSSEY_7D.test.authority.json" if program == worker.TEST_PROGRAM else "ODYSSEY_7D.authority.json"
    authority_path = root / "docs/plans/substrate/tangible_next_launch" / name
    _write(authority_path, body)
    return authority_path


class _ScriptedTelemetry:
    """Fast deterministic telemetry fixture for broker-boundary tests."""

    actions: list[str] = []

    def __init__(self, *_args: object) -> None:
        self._current: dict[str, object] = {}
        self._context: dict[str, object] = {}

    def update_context(self, **context: object) -> None:
        self._context = context

    def sample(self) -> None:
        action = self.actions.pop(0) if self.actions else "admit_or_resume"
        payload: dict[str, object] = {
            "schema": "SUBSTRATE_ODYSSEY_LIVE_TELEMETRY/v1",
            "sample_status": "observed",
            "broker_action": action,
            "host_rss_bytes": {
                "admit_or_resume": 74 * worker.GIB,
                "deny_new_work": 75 * worker.GIB,
                "checkpoint_reduce_p2": 80 * worker.GIB,
                "pause_p1_checkpoint_p2": 82 * worker.GIB,
            }.get(action, 85 * worker.GIB),
            "resident_cap_bytes": 85 * worker.GIB,
            "active_cores_equivalent": 0.0,
            "worker_tree_rss_bytes": 1,
            **self._context,
        }
        payload["sha256"] = worker._digest(payload)
        self._current = payload

    def start(self) -> None:
        self.sample()

    def stop(self) -> None:
        self.sample()

    def latest(self) -> dict[str, object]:
        return dict(self._current)

    def assert_admissible(self) -> None:
        if self._current.get("broker_action") == "safe_hold_non_p0":
            raise worker.Refused("worker telemetry observed the 85 GiB resident-memory cap")


def test_test_identity_runs_one_paired_event_then_requests_evaluator_release(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path)
    result = worker.run(tmp_path, authority_file=authority_path)
    assert result["status"] == "trace_locked_waiting_for_independent_evaluation"
    trace = tmp_path / "runs/worker-fixture/EVENTS.jsonl"
    assert len(trace.read_text(encoding="utf-8").splitlines()) == 1
    release = json.loads((tmp_path / "evidence/worker-fixture/EVALUATOR_RELEASE_REQUEST.json").read_text())
    assert release["worker_accessed_evaluator_answers"] is False
    assert "answer_manifest" not in json.dumps(release, sort_keys=True).casefold()
    state = json.loads((tmp_path / "runs/worker-fixture/STATE.json").read_text())
    assert state["complete"] is True
    assert state["completed_paired_events"] == 1
    assert state["sha256"] == worker._digest({key: value for key, value in state.items() if key != "sha256"})
    live = json.loads((tmp_path / "runs/worker-fixture/LIVE_TELEMETRY.json").read_text())
    assert live["sample_status"] == "observed"
    assert isinstance(live["host_rss_bytes"], int) and live["host_rss_bytes"] > 0
    assert live["resident_cap_bytes"] == 85 * worker.GIB
    assert live["sha256"] == worker._digest({key: value for key, value in live.items() if key != "sha256"})
    assert live["memory_broker_certification"] == "observational_telemetry_only_not_G08_certification"
    assert "G09" not in json.dumps(live, sort_keys=True)


def test_phase_budget_refuses_to_stretch_a_sealed_active_phase() -> None:
    clock = [100.0]

    worker._assert_phase_within_budget(started=0.0, phase_seconds=100, monotonic=lambda: clock[0])
    clock[0] = 100.001
    with pytest.raises(worker.Refused, match="exceeded the sealed phase budget"):
        worker._assert_phase_within_budget(started=0.0, phase_seconds=100, monotonic=lambda: clock[0])


def test_adapter_deadline_terminates_a_hung_arm_before_a_receipt_is_accepted(tmp_path: Path) -> None:
    adapter = tmp_path / "sleeping-adapter.py"
    adapter.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    deadline = time.monotonic() + 0.05

    with pytest.raises(worker.Refused, match="phase dispatch deadline"):
        worker._adapter(
            tmp_path,
            authority_sha256="a" * 64,
            run_id="deadline-fixture",
            worker_root=tmp_path / "worker",
            frontier="A",
            role="candidate",
            command=[sys.executable, str(adapter)],
            manifest_sha256="b" * 64,
            task={"task_id": "deadline-task"},
            cycle=0,
            phase="retrieval",
            deadline_monotonic=deadline,
        )
    assert not (tmp_path / "worker/arms/A/candidate/receipts/000-retrieval.json").exists()


def test_shared_paired_dispatcher_revalidates_before_each_arm_in_candidate_control_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task = {"task_id": "paired-dispatch-task"}
    calls: list[str] = []

    def manifest(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"tasks": [task]}

    def adapter(*_args: object, role: str, pre_dispatch_check: object, **_kwargs: object) -> dict[str, object]:
        assert callable(pre_dispatch_check)
        pre_dispatch_check()
        calls.append(role)
        return {"receipt": {"elapsed_seconds": 0.1}, "receipt_sha256": worker._digest({"role": role})}

    monkeypatch.setattr(worker, "_manifest_for_frontier", manifest)
    monkeypatch.setattr(worker, "_adapter", adapter)
    frontier, event = worker._dispatch_paired_frontier(
        tmp_path,
        authority_sha256="a" * 64,
        run_id="paired-fixture",
        worker_root=tmp_path / "worker",
        frontier_entry={
            "id": "A",
            "candidate_manifest_sha256": "b" * 64,
            "candidate_command": ["candidate"],
            "control_command": ["control"],
        },
        task=task,
        cycle=0,
        phase="retrieval",
        full_source_guard=True,
        task_count=1,
    )

    assert frontier == "A"
    assert calls == ["candidate", "control"]
    assert event["source_bundle_guard_calls"] == 2
    assert event["candidate_receipt_sha256"] != event["control_receipt_sha256"]


def test_worker_rejects_source_drift_before_any_adapter_runs(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path)
    authority = json.loads(authority_path.read_text())
    authority["worker_source_sha256"] = "0" * 64
    authority.pop("sha256")
    authority["sha256"] = worker._digest(authority)
    _write(authority_path, authority)
    with pytest.raises(worker.Refused, match="source drift"):
        worker.run(tmp_path, authority_file=authority_path)
    assert not (tmp_path / "runs/worker-fixture/EVENTS.jsonl").exists()


def test_worker_refuses_authority_without_an_exact_self_digest(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path)
    authority = json.loads(authority_path.read_text())
    authority.pop("sha256")
    _write(authority_path, authority)
    with pytest.raises(worker.Refused, match="exact self-digest"):
        worker.run(tmp_path, authority_file=authority_path)
    assert not (tmp_path / "runs/worker-fixture/EVENTS.jsonl").exists()


def test_full_program_rejects_reduced_test_schedule(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path, program=worker.PROGRAM)
    with pytest.raises(worker.Refused, match="all eight frontiers"):
        worker.run(tmp_path, authority_file=authority_path)


def test_resume_reconstructs_state_and_preserves_checkpoint_parent(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path, microcycles=2, fail_once_cycle=1)
    with pytest.raises(worker.Refused, match="adapter failed"):
        worker.run(tmp_path, authority_file=authority_path)

    run_root = tmp_path / "runs/worker-fixture"
    first = json.loads((run_root / "checkpoints/delta-001.json").read_text())
    assert first["parent_checkpoint_sha256"] == ""
    state_path = run_root / "STATE.json"
    interrupted_state = json.loads(state_path.read_text())
    assert interrupted_state["completed_phase_count"] == 1
    assert interrupted_state["checkpoint_sha256"] == first["sha256"]

    # Simulate the narrow post-checkpoint/pre-state-replace crash window.  The
    # worker must recover strictly from the verified trace and checkpoint, not
    # replay cycle zero or start a new chain at an empty parent.
    state_path.unlink()
    result = worker.run(tmp_path, authority_file=authority_path)
    assert result["status"] == "trace_locked_waiting_for_independent_evaluation"
    second = json.loads((run_root / "checkpoints/delta-002.json").read_text())
    assert second["parent_checkpoint_sha256"] == first["sha256"]
    assert second["completed_phase_count"] == 2
    final_state = json.loads(state_path.read_text())
    assert final_state["complete"] is True
    assert final_state["checkpoint_count"] == 2
    assert final_state["sha256"] == worker._digest({key: value for key, value in final_state.items() if key != "sha256"})


def test_resume_refuses_tampered_durable_state(tmp_path: Path) -> None:
    authority_path = _authority(tmp_path, microcycles=2, fail_once_cycle=1)
    with pytest.raises(worker.Refused, match="adapter failed"):
        worker.run(tmp_path, authority_file=authority_path)
    state_path = tmp_path / "runs/worker-fixture/STATE.json"
    state = json.loads(state_path.read_text())
    state["checkpoint_count"] = 999
    _write(state_path, state)
    with pytest.raises(worker.Refused, match="integrity digest"):
        worker.run(tmp_path, authority_file=authority_path)


def test_macos_total_minute_cpu_time_is_measured_not_discarded() -> None:
    assert worker._parse_cpu_seconds("560:31.84") == pytest.approx(33631.84)


@pytest.mark.parametrize(
    ("hold_action", "expected_status"),
    [
        ("deny_new_work", "holding_before_adapter_dispatch"),
        ("checkpoint_reduce_p2", "holding_before_adapter_dispatch_with_durable_boundary"),
        ("pause_p1_checkpoint_p2", "holding_before_adapter_dispatch_with_durable_boundary"),
    ],
)
def test_broker_holds_adapter_dispatch_until_safe_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hold_action: str,
    expected_status: str,
) -> None:
    authority_path = _authority(tmp_path)
    # start() sees safe, the phase boundary sees the hold, and the 30-second
    # non-invasive poll then sees safe admission.  No adapter may begin before
    # the latter observation.
    _ScriptedTelemetry.actions = ["admit_or_resume", hold_action, "admit_or_resume"]
    monkeypatch.setattr(worker, "_TelemetryRecorder", _ScriptedTelemetry)
    result = worker.run(tmp_path, authority_file=authority_path, sleep=lambda _seconds: None)
    assert result["status"] == "trace_locked_waiting_for_independent_evaluation"
    status = json.loads((tmp_path / "runs/worker-fixture/BROKER_STATUS.json").read_text())
    observations = status["observations"]
    assert observations[0]["broker_action"] == hold_action
    assert observations[0]["phase_boundary_status"] == expected_status
    assert observations[0]["no_new_adapter_work"] is True
    assert observations[1]["broker_action"] == "admit_or_resume"
    assert observations[1]["phase_boundary_status"] == "safe_admission_restored"
    assert observations[1]["no_new_adapter_work"] is False
    assert all(
        observation["memory_broker_certification"] == "observational_telemetry_only_not_G08_certification"
        for observation in observations
    )


def test_broker_hold_extends_wall_schedule_without_consuming_active_phase_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_path = _authority(tmp_path)
    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    _ScriptedTelemetry.actions = ["admit_or_resume", "deny_new_work", "admit_or_resume"]
    monkeypatch.setattr(worker, "_TelemetryRecorder", _ScriptedTelemetry)
    worker.run(tmp_path, authority_file=authority_path, monotonic=monotonic, sleep=sleep)

    run_root = tmp_path / "runs/worker-fixture"
    status = json.loads((run_root / "BROKER_STATUS.json").read_text())
    restored = status["observations"][-1]
    state = json.loads((run_root / "STATE.json").read_text())
    assert restored["broker_hold_seconds"] == 30.0
    assert state["broker_hold_seconds"] == 30.0
    assert state["elapsed_seconds"] == 0.0
    assert state["timing_policy"] == "scheduled_active_time_preserved_broker_holds_extend_wall_schedule"


def _frozen_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    input_path = tmp_path / "inputs/frozen-input.json"
    implementation_path = tmp_path / "src/substrate/frozen-implementation.py"
    input_path.parent.mkdir(parents=True)
    implementation_path.parent.mkdir(parents=True)
    input_path.write_text("input", encoding="utf-8")
    implementation_path.write_text("implementation", encoding="utf-8")
    inputs = {"fixture_input": input_path}
    implementations = {"fixture_implementation": implementation_path}
    monkeypatch.setattr(worker.odyssey_transition, "build_inputs", lambda _root: inputs)
    monkeypatch.setattr(worker.odyssey_transition, "implementation_inputs", lambda _root: implementations)
    frozen = {
        "schema": "SUBSTRATE_ODYSSEY_FROZEN_BUILD/v1",
        "program": worker.odyssey_transition.PROGRAM,
        "activation": False,
        "scientific_status": "frozen_waiting_for_verified_r2",
        "input_sha256": {name: worker.file_digest(path) for name, path in inputs.items()},
        "implementation_sha256": {name: worker.file_digest(path) for name, path in implementations.items()},
    }
    frozen["sha256"] = worker._digest(frozen)
    frozen_path = tmp_path / "docs/plans/substrate/tangible_next_launch/ODYSSEY_FROZEN_BUILD.json"
    _write(frozen_path, frozen)
    return frozen


def test_full_frozen_binding_requires_self_digest_and_exact_current_maps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    frozen = _frozen_fixture(tmp_path, monkeypatch)
    authority = {
        "frozen_build_sha256": frozen["sha256"],
        "seal": {"frozen_build_sha256": frozen["sha256"]},
    }
    worker._validate_full_frozen_build(tmp_path, authority)
    implementation = tmp_path / "src/substrate/frozen-implementation.py"
    implementation.write_text("drift", encoding="utf-8")
    with pytest.raises(worker.Refused, match="implementation source drift"):
        worker._validate_full_frozen_build(tmp_path, authority)


def test_full_manifest_revalidates_candidate_source_assets_at_dispatch(tmp_path: Path) -> None:
    manifest, _ = task_bank.materialize(worker._digest({"seed": "source-asset"}), "source-asset", "A", 1)
    source = tmp_path / "inputs/candidate-stimulus.txt"
    source.parent.mkdir(parents=True)
    source.write_text("original", encoding="utf-8")
    assets = [
        {
            "path": str(source.relative_to(tmp_path)),
            "sha256": worker.file_digest(source),
            "role": "candidate_stimulus",
            "read_only": True,
        }
    ]
    manifest["source_bundle"] = {
        "selection_sha256": worker._digest({"frontier": "A", "assets": assets}),
        "assets": assets,
    }
    manifest.pop("sha256")
    manifest["sha256"] = worker._digest(manifest)
    manifest_path = tmp_path / "candidate-visible/A.json"
    _write(manifest_path, manifest)
    frontier = {
        "id": "A",
        "candidate_manifest": str(manifest_path.relative_to(tmp_path)),
        "candidate_manifest_sha256": worker.file_digest(manifest_path),
        "candidate_command": [sys.executable],
        "control_command": [sys.executable],
    }
    worker._manifest_for_frontier(tmp_path, frontier, full=True, task_count=1)
    source.write_text("drift", encoding="utf-8")
    with pytest.raises(worker.Refused, match="source asset.*drifted"):
        worker._manifest_for_frontier(tmp_path, frontier, full=True, task_count=1)


def _runtime_lease(
    root: Path,
    *,
    authority_sha256: str,
    run_id: str,
    worker_config: dict,
    supervisor_pid: int,
    attempt: int = 1,
) -> Path:
    path = root / worker_config["run_root"] / "leases" / f"attempt-{attempt:03d}.json"
    lease = {
        "schema": "SUBSTRATE_ODYSSEY_SUPERVISOR_RUNTIME_LEASE/v1",
        "activation": False,
        "authority_sha256": authority_sha256,
        "run_id": run_id,
        "supervisor_pid": supervisor_pid,
        "attempt": attempt,
        "worker_argv_sha256": worker._digest({"argv": worker_config["argv"]}),
        "issued_at_epoch": 1.0,
    }
    lease["sha256"] = worker._digest(lease)
    _write(path, lease)
    path.chmod(0o600)
    return path


def test_full_worker_refuses_missing_runtime_lease_before_manifest_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    full_authority = {"program_config": {"id": worker.PROGRAM}, "program": {"id": worker.PROGRAM}, "run_id": "full-fixture"}
    worker_config = {"run_root": "runs/full-fixture", "argv": [sys.executable, "-m", "substrate.odyssey_worker"]}
    monkeypatch.setattr(worker, "validate_authority", lambda *_args: (full_authority, worker_config, "a" * 64))
    monkeypatch.delenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH", raising=False)
    monkeypatch.delenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_SHA256", raising=False)
    with pytest.raises(worker.Refused, match="runtime lease environment"):
        worker.run(tmp_path, authority_file=tmp_path / "sealed.json")


def test_full_worker_refuses_wrong_parent_runtime_lease_before_manifest_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    full_authority = {"program_config": {"id": worker.PROGRAM}, "program": {"id": worker.PROGRAM}, "run_id": "full-fixture"}
    worker_config = {"run_root": "runs/full-fixture", "argv": [sys.executable, "-m", "substrate.odyssey_worker"]}
    lease_path = _runtime_lease(
        tmp_path,
        authority_sha256="a" * 64,
        run_id="full-fixture",
        worker_config=worker_config,
        supervisor_pid=4242,
    )
    monkeypatch.setattr(worker, "validate_authority", lambda *_args: (full_authority, worker_config, "a" * 64))
    monkeypatch.setattr(worker.os, "getppid", lambda: 9999)
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH", str(lease_path))
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_SHA256", json.loads(lease_path.read_text())["sha256"])
    with pytest.raises(worker.Refused, match="runtime lease is invalid"):
        worker.run(tmp_path, authority_file=tmp_path / "sealed.json")


@pytest.mark.parametrize("failure", ("tampered", "wrong_parent", "symlink"))
def test_runtime_lease_refuses_tampering_and_wrong_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    authority_sha256 = "a" * 64
    run_id = "lease-fixture"
    worker_config = {"run_root": "runs/lease-fixture", "argv": [sys.executable, "-m", "substrate.odyssey_worker"]}
    lease_path = _runtime_lease(
        tmp_path,
        authority_sha256=authority_sha256,
        run_id=run_id,
        worker_config=worker_config,
        supervisor_pid=4242,
    )
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH", str(lease_path))
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_SHA256", json.loads(lease_path.read_text())["sha256"])
    if failure == "tampered":
        lease = json.loads(lease_path.read_text())
        lease["run_id"] = "changed-after-issue"
        _write(lease_path, lease)
        lease_path.chmod(0o600)
        expected = "integrity digest"
    elif failure == "symlink":
        target = tmp_path / "outside-runtime-lease.json"
        lease_path.rename(target)
        lease_path.symlink_to(target)
        expected = "direct non-symlink"
    else:
        expected = "runtime lease is invalid"
    monkeypatch.setattr(worker.os, "getppid", lambda: 9999 if failure == "wrong_parent" else 4242)
    with pytest.raises(worker.Refused, match=expected):
        worker._validate_runtime_lease(
            tmp_path,
            worker=worker_config,
            authority_sha256=authority_sha256,
            run_id=run_id,
        )
