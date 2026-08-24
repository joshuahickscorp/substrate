import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from substrate import odyssey7d as odyssey
from substrate import odyssey_task_bank as task_bank
from substrate import odyssey_transition as transition
from substrate import odyssey_worker


def make_root(tmp_path: Path) -> Path:
    source_root = Path(__file__).parents[2] / "plans/substrate/tangible_next_launch"
    target_root = tmp_path / "plans/substrate/tangible_next_launch"
    target_root.mkdir(parents=True)
    for filename in ("ODYSSEY_7D.hardened.draft.json", "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json"):
        (target_root / filename).write_bytes((source_root / filename).read_bytes())
    return tmp_path


def test_hardened_authority_validates() -> None:
    root = make_root(Path(__import__("tempfile").mkdtemp()))
    assert all(odyssey.validate(json.loads(odyssey.authority_path(root).read_text())).values())


def test_render_creates_eight_templates_and_schedule(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    result = odyssey.render(root)
    assert len(result["frontier_templates"]) == 8
    assert len(result["custodian_skeletons"]) == 8
    assert result["schedule_entries"] == 8 * 7 * 12
    philosophy = json.loads((root / "plans/substrate/tangible_next_launch/frontiers/E_philosophy_self_model.manifest.template.json").read_text())
    logic = json.loads((root / "plans/substrate/tangible_next_launch/frontiers/C_formal_logic.manifest.template.json").read_text())
    assert "epistemic_commitment_revision" in philosophy["task_contract"]["task_families"]
    assert "satisfiability_and_countermodel" in logic["task_contract"]["task_families"]


def test_broker_and_checkpoint_reject_bad_inputs() -> None:
    assert odyssey.broker_action(74.9) == "admit_or_resume"
    assert odyssey.broker_action(80) == "checkpoint_reduce_p2"
    assert odyssey.broker_action(85) == "safe_hold_non_p0"
    assert odyssey.checkpoint_chain_valid("a", [{"parent_digest": "a", "digest": "b"}])
    assert not odyssey.checkpoint_chain_valid("a", [{"parent_digest": "x", "digest": "b"}])
    assert odyssey.wedge_detected(
        {"cpu_time_seconds": 1, "event_count": 1, "checkpoint_count": 1}, {"cpu_time_seconds": 1, "event_count": 1, "checkpoint_count": 1}, True
    )
    assert odyssey.storage_required(5, 3, 2, 1) == 32


def test_canaries_and_readiness_fail_closed(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    report = odyssey.canaries(root)
    assert report["launch_allowed"] is False
    assert report["checkpoint_chain"]["forged_parent_rejected"] is True
    readiness = odyssey.readiness(root)
    assert readiness["launch_allowed"] is False
    assert len(readiness["remaining_blockers"]) == 15


def test_readiness_surfaces_valid_machine_gates_without_launching(tmp_path: Path, monkeypatch) -> None:
    root = make_root(tmp_path)
    monkeypatch.setattr(odyssey.odyssey_authority, "machine_gate_ids", lambda _root: frozenset({"G13", "G14", "G15"}))

    readiness = odyssey.readiness(root)

    assert readiness["gates"]["G13"] == readiness["gates"]["G14"] == readiness["gates"]["G15"] == "pass"
    assert readiness["clean_clone"] == readiness["ci"] == "verified"
    assert readiness["telegram"] == "delivery_verified"
    assert readiness["protocol_digests"] == "verified"
    assert readiness["launch_allowed"] is False


def test_detached_template_is_staged_not_activated(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    template = odyssey.detached_supervisor_template(root)
    assert template["activation"] is False
    assert template["launchd"]["KeepAlive"] is False
    assert template["launchd"]["Umask"] == 0o077
    assert template["launchd"]["ProgramArguments"][:3] == ["/usr/bin/caffeinate", "-i", "-s"]
    assert template["launchd"]["EnvironmentVariables"][odyssey.odyssey_detachment.POWER_ASSERTION_ENV] == (
        odyssey.odyssey_detachment.POWER_ASSERTION_VALUE
    )
    assert "current_user_caffeinate_assertion" in template["must_require"]
    assert template["power_resilience"] == odyssey.odyssey_detachment.power_resilience_contract()


def test_supervisor_refuses_non_launchd_and_unsealed_authority(tmp_path: Path, monkeypatch) -> None:
    root = make_root(tmp_path)
    monkeypatch.delenv("SUBSTRATE_ODYSSEY_SUPERVISOR", raising=False)
    with __import__("pytest").raises(odyssey.Refused):
        odyssey.supervise(root, root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json")


def test_supervisor_requires_a_verified_detachment_receipt_before_spawn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    authority = {"run_id": "receipt-fixture", "worker": {"run_root": "runs/substrate/receipt-fixture"}, "storage": {"required_free_bytes": 1}}
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_SUPERVISOR", "launchd")
    monkeypatch.setattr(odyssey, "_assert_current_user_caffeinate_contract", lambda: None)
    monkeypatch.setattr(
        odyssey,
        "_validated_launch_authority",
        lambda _root, _authority: (authority, authority["worker"], ["/opt/synthetic-worker"], "a" * 64),
    )
    monkeypatch.setattr(
        odyssey.odyssey_detachment,
        "verify_receipt",
        lambda _root: (_ for _ in ()).throw(odyssey.odyssey_detachment.Refused("missing receipt")),
    )
    monkeypatch.setattr(odyssey.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("receipt failure must prevent spawn"))

    with pytest.raises(odyssey.Refused, match="detachment configuration receipt"):
        odyssey.supervise(root, root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json")


def test_supervisor_refuses_missing_current_user_caffeinate_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(odyssey.odyssey_detachment.POWER_ASSERTION_ENV, raising=False)
    with pytest.raises(odyssey.Refused, match="must inherit"):
        odyssey._assert_current_user_caffeinate_contract()

    monkeypatch.setenv(
        odyssey.odyssey_detachment.POWER_ASSERTION_ENV,
        odyssey.odyssey_detachment.POWER_ASSERTION_VALUE,
    )
    odyssey._assert_current_user_caffeinate_contract()


def test_supervisor_health_uses_observed_telemetry_without_certifying_gates(tmp_path: Path) -> None:
    authority = {
        "run_id": "telemetry-fixture",
        "worker": {"run_root": "runs/substrate/odyssey7d/v1", "resident_cap_bytes": 85 * odyssey.GIB},
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
    }
    authority_sha256 = odyssey._digest(authority)
    first, sample = odyssey._supervisor_health(
        tmp_path,
        authority=authority,
        authority_sha256=authority_sha256,
        worker_pid=os.getpid(),
        supervisor_started=time.monotonic(),
        previous_sample=None,
    )
    second, _ = odyssey._supervisor_health(
        tmp_path,
        authority=authority,
        authority_sha256=authority_sha256,
        worker_pid=os.getpid(),
        supervisor_started=time.monotonic(),
        previous_sample=sample,
    )
    assert first["resident_memory"] > 0
    assert first["broker_action"] in {"admit_or_resume", "deny_new_work", "checkpoint_reduce_p2", "pause_p1_checkpoint_p2", "safe_hold_non_p0"}
    assert first["cpu_time_deltas"]["logical_cores_available"] > 0
    assert second["cpu_time_deltas"]["sampling_interval_seconds"] is not None
    assert first["sampling_interval_target_seconds"] == 30.0
    assert first["memory_broker_certification"] == "observational_telemetry_only_not_G08_certification"
    assert first["durability_certification"] == "not_G09_certification"
    assert "pending" not in json.dumps(first, sort_keys=True)


def test_supervisor_consumes_a_self_digested_live_worker_sample(tmp_path: Path) -> None:
    authority = {
        "run_id": "live-telemetry-fixture",
        "worker": {"run_root": "runs/substrate/odyssey7d/v1", "resident_cap_bytes": 85 * odyssey.GIB},
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
    }
    authority_sha256 = odyssey._digest(authority)
    run_root = tmp_path / "runs/substrate/odyssey7d/v1"
    run_root.mkdir(parents=True)
    live = {
        "schema": "SUBSTRATE_ODYSSEY_LIVE_TELEMETRY/v1",
        "authority_sha256": authority_sha256,
        "run_id": authority["run_id"],
        "worker_pid": os.getpid(),
        "completed_phase_count": 5,
        "total_phase_count": 336,
        "completion_percent": round(100 * 5 / 336, 6),
        "cycle": 2,
        "phase": "exposure",
        "phase_status": "running",
        "active_frontiers": ["A", "B"],
    }
    live["sha256"] = odyssey_worker._digest(live)
    (run_root / "LIVE_TELEMETRY.json").write_text(json.dumps(live), encoding="utf-8")

    health, _ = odyssey._supervisor_health(
        tmp_path,
        authority=authority,
        authority_sha256=authority_sha256,
        worker_pid=os.getpid(),
        supervisor_started=time.monotonic(),
        previous_sample=None,
    )

    assert health["completed_phase_count"] == 5
    assert health["cycle"] == 2 and health["phase"] == "exposure"
    assert health["frontier_health"]["source"] == "worker_live_telemetry"


def _completed_test_worker(root: Path) -> tuple[Path, dict[str, object]]:
    """Produce one real, tiny worker trace for postflight integrity checks."""
    candidate, _ = task_bank.materialize(odyssey_worker._digest({"seed": "postflight-fixture"}), "postflight-fixture", "A", 1)
    candidate_path = root / "candidate-visible/A.json"
    candidate_path.parent.mkdir(parents=True)
    candidate_path.write_text(json.dumps(candidate, sort_keys=True), encoding="utf-8")
    adapter = root / "adapter.py"
    adapter.write_text(
        """import json
import sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
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
path = Path(request['receipt_path'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(receipt, sort_keys=True))
""",
        encoding="utf-8",
    )
    command = shlex.join([sys.executable, str(adapter)])
    worker_config: dict[str, object] = {
        "test_mode": True,
        "run_root": "runs/postflight-fixture",
        "publication_root": "evidence/postflight-fixture",
        "frontiers": [
            {
                "id": "A",
                "candidate_manifest": str(candidate_path.relative_to(root)),
                "candidate_manifest_sha256": odyssey_worker.file_digest(candidate_path),
                "candidate_command": command,
                "control_command": command,
            }
        ],
        "phase_names": ["retrieval"],
        "phase_seconds": 0,
        "microcycles_per_frontier": 1,
        "max_parallel_frontiers": 1,
    }
    authority: dict[str, object] = {
        "schema": "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "run_id": "postflight-fixture",
        "program": {"id": odyssey_worker.TEST_PROGRAM, "duration_seconds": 0, "launch_allowed": True},
        "seal": {"status": "sealed"},
        "launch_gates": [{"id": "G01", "status": "pass"}],
        "worker_source_sha256": odyssey_worker.file_digest(Path(odyssey_worker.__file__)),
        "storage": {"required_free_bytes": 1, "launch_required_free_bytes": 1},
        "worker": worker_config,
    }
    authority["sha256"] = odyssey_worker._digest(authority)
    authority_path = root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.test.authority.json"
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text(json.dumps(authority, sort_keys=True), encoding="utf-8")
    result = odyssey_worker.run(root, authority_file=authority_path)
    assert result["status"] == "trace_locked_waiting_for_independent_evaluation"
    return authority_path, authority


def test_postflight_receipt_revalidates_complete_trace_checkpoint_and_custody_artifacts(tmp_path: Path) -> None:
    _authority_path, authority = _completed_test_worker(tmp_path)
    worker_config = authority["worker"]
    assert isinstance(worker_config, dict)
    authority_sha256 = authority["sha256"]
    assert isinstance(authority_sha256, str)

    receipt = odyssey._write_or_verify_postflight_receipt(
        tmp_path,
        run_root=tmp_path / "runs/postflight-fixture",
        authority_sha256=authority_sha256,
        run_id="postflight-fixture",
        worker_config=worker_config,
    )

    assert receipt["schema"] == odyssey.POSTFLIGHT_RECEIPT_SCHEMA
    assert receipt["outcome"] == "worker_trace_locked_waiting_for_independent_evaluation"
    assert receipt["scientific_results_included"] is False
    assert receipt["evaluator_release_request"]["worker_accessed_evaluator_answers"] is False
    path = tmp_path / "runs/postflight-fixture" / odyssey.POSTFLIGHT_RECEIPT_NAME
    assert path.stat().st_mode & 0o777 == 0o600
    assert (
        odyssey._write_or_verify_postflight_receipt(
            tmp_path,
            run_root=tmp_path / "runs/postflight-fixture",
            authority_sha256=authority_sha256,
            run_id="postflight-fixture",
            worker_config=worker_config,
        )
        == receipt
    )


def test_recovered_postflight_marks_complete_before_interrupted_attempt_recovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lineage = odyssey._lineage_document("a" * 64, "postflight-recovery")
    lineage["attempts"].append({"attempt": 1, "outcome": "running", "worker_pid": 999999})
    receipt = {"schema": odyssey.POSTFLIGHT_RECEIPT_SCHEMA, "sha256": "b" * 64}
    monkeypatch.setattr(odyssey, "_write_or_verify_postflight_receipt", lambda *_args, **_kwargs: receipt)

    recovered, observed = odyssey._recover_completed_postflight(
        tmp_path,
        run_root=tmp_path / "runs/postflight-recovery",
        authority_sha256="a" * 64,
        run_id="postflight-recovery",
        worker_config={},
        lineage=lineage,
        epoch=123.0,
    )
    preserved, terminal = odyssey._recover_interrupted_attempt(recovered, epoch=124.0)

    assert observed == receipt
    assert recovered["attempts"][-1]["outcome"] == "worker_complete_recovered_from_durable_postflight"
    assert preserved["terminal_status"] == "worker_complete"
    assert terminal is None


def test_validated_launch_authority_requires_a_self_digest_before_any_source_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    authority_path = tmp_path / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json"
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text(json.dumps({"run_id": "missing-digest"}), encoding="utf-8")

    def unexpected_frozen_validation(*_args, **_kwargs) -> None:
        raise AssertionError("source-map validation must not run for an unsigned authority")

    monkeypatch.setattr(odyssey.odyssey_authority, "_validate_frozen_build", unexpected_frozen_validation)

    with pytest.raises(odyssey.Refused, match="self-digest is missing or invalid"):
        odyssey._validated_launch_authority(tmp_path, authority_path)


def test_validated_launch_authority_requires_current_frozen_map_and_returns_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    worker_path = tmp_path / "src/substrate/odyssey_worker.py"
    worker_path.parent.mkdir(parents=True)
    worker_path.write_text("# synthetic worker fixture\n", encoding="utf-8")
    authority_path = tmp_path / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json"
    authority_path.parent.mkdir(parents=True)
    worker_argv = ["/usr/bin/true", "--synthetic-worker"]
    frozen_sha256 = "f" * 64
    authority = {
        "schema": "SUBSTRATE_ODYSSEY_7D_AUTHORITY/v1",
        "run_id": "validated-launch-fixture",
        "frozen_build_sha256": frozen_sha256,
        "program": {"id": odyssey.PROGRAM, "launch_allowed": True},
        "seal": {
            "status": "sealed",
            "frozen_build_sha256": frozen_sha256,
            "protocol_digest": "frozen-protocol-digest",
            "authority_source_sha256": odyssey.file_digest(Path(odyssey.odyssey_authority.__file__)),
        },
        "worker": {"argv": worker_argv, "run_root": "runs/substrate/validated-launch-fixture"},
        "detached_worker_command": shlex.join(worker_argv),
        "supervisor_source_sha256": odyssey.file_digest(Path(odyssey.__file__)),
        "worker_source_sha256": odyssey.file_digest(worker_path),
    }
    authority["sha256"] = odyssey.odyssey_authority.digest(authority)
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    seen: dict[str, object] = {}

    def frozen_validation(root: Path, digest: str) -> dict[str, object]:
        seen["frozen_root"] = root
        seen["frozen_digest"] = digest
        return {"fixture": "frozen"}

    monkeypatch.setattr(odyssey.odyssey_authority, "_validate_frozen_build", frozen_validation)
    monkeypatch.setattr(odyssey.odyssey_authority, "protocol_digest_for_frozen", lambda _frozen: "frozen-protocol-digest")
    monkeypatch.setattr(odyssey.odyssey_authority, "verify", lambda _root, _path: {"all_pass": True})
    monkeypatch.setattr(
        odyssey.odyssey_worker,
        "validate_authority",
        lambda _root, _path: (authority, authority["worker"], authority["sha256"]),
    )

    observed_authority, _worker, observed_argv, observed_digest = odyssey._validated_launch_authority(tmp_path, authority_path)

    assert observed_authority == authority
    assert observed_argv == worker_argv
    assert observed_digest == authority["sha256"]
    assert seen == {"frozen_root": tmp_path.resolve(), "frozen_digest": frozen_sha256}


def test_supervisor_lease_is_write_once_self_digested_and_bound_to_exact_argv(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/substrate/lease-fixture"
    argv = ["/usr/bin/true", "--synthetic-worker"]

    lease_path, lease = odyssey._runtime_lease(
        run_root,
        authority_sha256="a" * 64,
        run_id="lease-fixture-雪",
        attempt=1,
        worker_argv=argv,
        epoch=123.0,
    )

    assert lease["schema"] == odyssey.SUPERVISOR_LEASE_SCHEMA
    assert lease["activation"] is False
    assert lease["worker_argv_sha256"] == odyssey_worker._digest({"argv": argv})
    assert lease["sha256"] == odyssey._digest({key: value for key, value in lease.items() if key != "sha256"})
    assert odyssey._read_self_digested(lease_path, schema=odyssey.SUPERVISOR_LEASE_SCHEMA, label="lease") == lease
    assert lease_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(odyssey.Refused, match="refusing to overwrite"):
        odyssey._runtime_lease(
            run_root,
            authority_sha256="a" * 64,
            run_id="lease-fixture-雪",
            attempt=1,
            worker_argv=argv,
            epoch=124.0,
        )


def test_supervisor_flock_releases_after_owner_exits_and_refuses_only_live_duplicate(tmp_path: Path) -> None:
    run_root = tmp_path / "runs/substrate/lock-fixture"
    run_root.mkdir(parents=True)
    # An old O_EXCL-era lock file must not be a permanent restart blocker.
    (run_root / "supervisor.lock").write_text("legacy marker", encoding="utf-8")
    first = odyssey._acquire_supervisor_lock(run_root)
    try:
        with pytest.raises(odyssey.Refused, match="duplicate"):
            odyssey._acquire_supervisor_lock(run_root)
    finally:
        odyssey.fcntl.flock(first.fileno(), odyssey.fcntl.LOCK_UN)
        first.close()
    second = odyssey._acquire_supervisor_lock(run_root)
    second.close()


def test_supervisor_retries_only_bounded_abnormal_exits_and_writes_terminal_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path.resolve()
    run_root = root / "runs/substrate/restart-fixture"
    argv = ["/opt/synthetic-odyssey-worker", "--fixture"]
    authority = {"run_id": "restart-fixture", "worker": {"run_root": str(run_root.relative_to(root))}, "storage": {"required_free_bytes": 1}}
    authority_sha256 = "b" * 64
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_SUPERVISOR", "launchd")
    monkeypatch.setattr(odyssey, "_assert_current_user_caffeinate_contract", lambda: None)
    monkeypatch.setattr(odyssey, "_validated_launch_authority", lambda _root, _authority: (authority, authority["worker"], argv, authority_sha256))
    monkeypatch.setattr(odyssey.odyssey_detachment, "verify_receipt", lambda _root: {"authority_sha256": authority_sha256})
    sleeps: list[float] = []
    monkeypatch.setattr(odyssey.time, "sleep", lambda seconds: sleeps.append(seconds))
    calls: list[tuple[list[str], dict[str, object]]] = []

    class FailingWorker:
        next_pid = 7100

        def __init__(self, command: list[str], **kwargs: object) -> None:
            type(self).next_pid += 1
            self.pid = type(self).next_pid
            self.returncode = 23
            calls.append((command, kwargs))

        def poll(self) -> int:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

    monkeypatch.setattr(odyssey.subprocess, "Popen", FailingWorker)

    result = odyssey.supervise(root, root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json")

    assert result == {"status": "terminal_safe_hold", "reason": "restart_budget_exhausted", "run_id": "restart-fixture"}
    assert [command for command, _kwargs in calls] == [argv] * (odyssey.MAX_ABNORMAL_RESTARTS + 1)
    assert all(kwargs["start_new_session"] is True for _command, kwargs in calls)
    assert all("SUBSTRATE_ODYSSEY_RUNTIME_LEASE_PATH" in kwargs["env"] for _command, kwargs in calls)
    assert sleeps == list(odyssey.RESTART_BACKOFF_SECONDS)
    lineage = json.loads((run_root / "SUPERVISOR_LINEAGE.json").read_text(encoding="utf-8"))
    assert lineage["abnormal_restart_count"] == odyssey.MAX_ABNORMAL_RESTARTS + 1
    assert lineage["terminal_status"] == "restart_budget_exhausted"
    assert lineage["sha256"] == odyssey._digest({key: value for key, value in lineage.items() if key != "sha256"})
    state = json.loads((run_root / "SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
    assert state["run_active"] is False
    assert state["status"] == "terminal_safe_hold"
    assert state["terminal_reason"] == "restart_budget_exhausted"
    assert state["sha256"] == odyssey._digest({key: value for key, value in state.items() if key != "sha256"})
    # The lock file remains as a harmless lineage marker, but no stale O_EXCL
    # ownership blocks the next bounded supervisor attempt.
    handle = odyssey._acquire_supervisor_lock(run_root)
    handle.close()


def test_supervisor_treats_a_zero_exit_without_verified_postflight_as_a_safe_hold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    run_root = root / "runs/substrate/postflight-refusal"
    authority = {
        "run_id": "postflight-refusal",
        "worker": {"run_root": str(run_root.relative_to(root))},
        "storage": {"required_free_bytes": 1},
    }
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_SUPERVISOR", "launchd")
    monkeypatch.setattr(odyssey, "_assert_current_user_caffeinate_contract", lambda: None)
    monkeypatch.setattr(
        odyssey,
        "_validated_launch_authority",
        lambda _root, _authority: (authority, authority["worker"], ["/opt/synthetic-worker"], "d" * 64),
    )
    monkeypatch.setattr(odyssey.odyssey_detachment, "verify_receipt", lambda _root: {"authority_sha256": "d" * 64})

    class SuccessfulButUnverifiedWorker:
        pid = 9234
        returncode = 0

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def poll(self) -> int:
            return self.returncode

    monkeypatch.setattr(odyssey.subprocess, "Popen", SuccessfulButUnverifiedWorker)

    result = odyssey.supervise(root, root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json")

    assert result == {
        "status": "terminal_safe_hold",
        "reason": "postflight_verification_failed_safe_hold",
        "run_id": "postflight-refusal",
    }
    state = json.loads((run_root / "SUPERVISOR_STATE.json").read_text(encoding="utf-8"))
    assert state["status"] == "terminal_safe_hold"
    assert state["terminal_reason"] == "postflight_verification_failed_safe_hold"
    assert state["run_active"] is False


def test_supervisor_kills_an_owned_process_group_with_bounded_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    signals: list[int] = []

    class HangingWorker:
        pid = 8123
        returncode: int | None = None
        waits = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waits += 1
            if self.waits == 1:
                raise odyssey.subprocess.TimeoutExpired(["fixture"], 1)
            self.returncode = -9
            return self.returncode

    monkeypatch.setattr(odyssey.os, "killpg", lambda _pid, signal: signals.append(signal))

    worker = HangingWorker()
    assert odyssey._terminate_worker_group(worker) == -9
    assert signals == [odyssey.signal.SIGTERM, odyssey.signal.SIGKILL]


def test_supervisor_cleans_the_active_worker_before_releasing_its_flock_on_interrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path.resolve()
    run_root = root / "runs/substrate/interrupt-fixture"
    authority = {"run_id": "interrupt-fixture", "worker": {"run_root": str(run_root.relative_to(root))}, "storage": {"required_free_bytes": 1}}
    monkeypatch.setenv("SUBSTRATE_ODYSSEY_SUPERVISOR", "launchd")
    monkeypatch.setattr(odyssey, "_assert_current_user_caffeinate_contract", lambda: None)
    monkeypatch.setattr(
        odyssey,
        "_validated_launch_authority",
        lambda _root, _authority: (authority, authority["worker"], ["/opt/synthetic-worker"], "c" * 64),
    )
    monkeypatch.setattr(odyssey.odyssey_detachment, "verify_receipt", lambda _root: {"authority_sha256": "c" * 64})

    class RunningWorker:
        pid = 9191
        returncode: int | None = None

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def poll(self) -> None:
            return None

    cleaned: list[int] = []
    monkeypatch.setattr(odyssey.subprocess, "Popen", RunningWorker)
    monkeypatch.setattr(odyssey, "_supervisor_health", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(odyssey, "_terminate_worker_group", lambda worker: cleaned.append(worker.pid) or 0)

    with pytest.raises(KeyboardInterrupt):
        odyssey.supervise(root, root / "plans/substrate/tangible_next_launch/ODYSSEY_7D.authority.json")

    assert cleaned == [RunningWorker.pid]
    handle = odyssey._acquire_supervisor_lock(run_root)
    handle.close()


def _copy_transition_inputs(root: Path) -> None:
    source_root = Path(__file__).parents[2]
    for filename in (
        "ODYSSEY_7D.hardened.draft.json",
        "R2_TO_ODYSSEY_AUTOPIVOT_POLICY.sealed.json",
        "ODYSSEY_TASK_BANK_MANIFEST.draft.json",
        "RESOURCE_CALIBRATION_SPEC.draft.json",
        "ODYSSEY_SHARED_STORAGE_RESERVE.draft.json",
        "ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json",
        "ODYSSEY_SOURCE_SELECTION.template.json",
        "ODYSSEY_PUBLIC_MODEL_CANARY.template.json",
        "ODYSSEY_HUMAN_EVIDENCE_PACK.template.json",
    ):
        source = source_root / "plans/substrate/tangible_next_launch" / filename
        target = root / "plans/substrate/tangible_next_launch" / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    decision_source = source_root / transition.OPERATOR_DECISION
    decision_target = root / transition.OPERATOR_DECISION
    decision_target.parent.mkdir(parents=True, exist_ok=True)
    decision_target.write_bytes(decision_source.read_bytes())
    # Keep the clone fixture aligned with the transition's real frozen source
    # map.  A hand-maintained subset makes a clean-clone test pass only until a
    # newly bound implementation file is introduced.
    for source in transition.implementation_inputs(source_root).values():
        target = root / source.relative_to(source_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    odyssey.render(root)


def _r2_evidence(root: Path, *, complete: bool) -> None:
    evidence = root / "evidence/substrate/tangible_sandbox"
    evidence.mkdir(parents=True, exist_ok=True)
    result = {"scientific_status": "complete" if complete else "running", "actual_wall_hours": 24 if complete else 1, "continuity_passing": complete}
    result["sha256"] = transition.digest(result)
    continuity = {
        "schema": "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION/v1",
        "scientific_status": "pass" if complete else "fail",
        "independently_verified": complete,
    }
    continuity["sha256"] = transition.digest(continuity)
    provenance = {
        "schema": "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION/v1",
        "scientific_status": "pass" if complete else "fail",
        "independently_verified": complete,
    }
    provenance["sha256"] = transition.digest(provenance)
    (evidence / "SUBSTRATE_SANDBOX_LONGITUDINAL_RESULT.json").write_text(json.dumps(result))
    (evidence / "SUBSTRATE_SANDBOX_R2_CONTINUITY_VERIFICATION.json").write_text(json.dumps(continuity))
    (evidence / "SUBSTRATE_SANDBOX_R2_PROVENANCE_VERIFICATION.json").write_text(json.dumps(provenance))


def test_frozen_r2_transition_waits_then_authorizes_preflight_only(tmp_path: Path) -> None:
    _copy_transition_inputs(tmp_path)
    frozen = transition.freeze(tmp_path)
    assert frozen["scientific_status"] == "frozen_waiting_for_verified_r2"
    missing = transition.transition(tmp_path)
    assert missing["state"] == "waiting_for_verified_r2"
    assert missing["details"]["reason"] == "r2_longitudinal_result_missing_or_invalid"
    _r2_evidence(tmp_path, complete=False)
    waiting = transition.transition(tmp_path)
    assert waiting["state"] == "waiting_for_verified_r2"
    _r2_evidence(tmp_path, complete=True)
    authorized = transition.transition(tmp_path)
    assert authorized["state"] == "odyssey_preflight_authorized"
    assert authorized["preflight_authorized"] is True
    assert authorized["sha256"] == transition.digest({key: value for key, value in authorized.items() if key != "sha256"})
    assert "scientific_worker_launch" in authorized["forbidden_actions_not_taken"]


def test_freeze_binds_the_exact_g06_calibration_ladder_and_source_selection_template(tmp_path: Path) -> None:
    _copy_transition_inputs(tmp_path)
    plan = tmp_path / "plans/substrate/tangible_next_launch"
    design = json.loads((plan / "ODYSSEY_7D.hardened.draft.json").read_text(encoding="utf-8"))
    calibration = json.loads((plan / "RESOURCE_CALIBRATION_SPEC.draft.json").read_text(encoding="utf-8"))

    assert calibration["widths"] == design["resources"]["widths_to_calibrate"] == [1, 2, 4, 6, 8]
    assert calibration["repetitions"] == design["resources"]["calibration_repetitions"] == 3
    assert calibration["full_program_requires_width"] == design["resources"]["full_program_requires_width"] == 8
    frozen = transition.freeze(tmp_path)
    assert {"resource_calibration", "source_selection_template", "operator_decision"} <= set(frozen["input_sha256"])

    calibration["widths"] = [1, 2, 4]
    (plan / "RESOURCE_CALIBRATION_SPEC.draft.json").write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(transition.Refused, match="1/2/4/6/8"):
        transition.freeze(tmp_path)


def test_freeze_rejects_operator_decision_that_no_longer_matches_the_design(tmp_path: Path) -> None:
    _copy_transition_inputs(tmp_path)
    decision_path = tmp_path / transition.OPERATOR_DECISION
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["statistics"]["score_weights"]["task_utility"] = 0.4
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    with pytest.raises(transition.Refused, match="score weights"):
        transition.freeze(tmp_path)


def test_frozen_transition_rejects_task_contract_drift(tmp_path: Path) -> None:
    _copy_transition_inputs(tmp_path)
    transition.freeze(tmp_path)
    contract = tmp_path / "plans/substrate/tangible_next_launch/ODYSSEY_FRONTIER_TASK_CONTRACTS.frozen.json"
    value = json.loads(contract.read_text())
    value["frontiers"][4]["task_families"].append("result_dependent_rewrite")
    contract.write_text(json.dumps(value))
    _r2_evidence(tmp_path, complete=True)
    with __import__("pytest").raises(transition.Refused, match="frozen input drift"):
        transition.transition(tmp_path)


def test_logic_and_philosophy_generators_are_deterministic_and_blind() -> None:
    seed = "custodian-seed"
    commitment = hashlib.sha256(json.dumps({"seed": seed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    logic_a, _ = task_bank.materialize(commitment, seed, "C", 3)
    logic_b, _ = task_bank.materialize(commitment, seed, "C", 3)
    philosophy, _ = task_bank.materialize(commitment, seed, "E", 3)
    assert logic_a["sha256"] == logic_b["sha256"]
    assert task_bank.candidate_is_structurally_safe(logic_a)
    assert task_bank.candidate_is_structurally_safe(philosophy)
    assert all("commitment_ledger" in task["required_receipt"] for task in philosophy["tasks"])


def test_every_frontier_has_a_distinct_blind_materializer() -> None:
    seed = "all-frontier-custodian-seed"
    commitment = hashlib.sha256(json.dumps({"seed": seed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidate_schemas = set()
    for frontier in "ABCDEFGH":
        candidate, evaluator = task_bank.materialize(commitment, seed, frontier, 2)
        assert candidate["frontier"] == evaluator["frontier"] == frontier
        assert task_bank.candidate_is_structurally_safe(candidate)
        candidate_schemas.add(candidate["tasks"][0]["schema"])
    assert len(candidate_schemas) == 8


def test_logic_generator_truth_matches_its_visible_rules() -> None:
    seed = "logic-semantics-seed"
    commitment = hashlib.sha256(json.dumps({"seed": seed}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    candidate, evaluator = task_bank.materialize(commitment, seed, "C", 32)
    for task, answer in zip(candidate["tasks"], evaluator["answers"], strict=True):
        rules = task["rules"]
        if answer["satisfiable"]:
            assignment = answer["witness_assignment"]
            assert assignment is not None
            for rule in rules:
                if "->" in rule:
                    left, right = [part.strip() for part in rule.split("->")]
                    assert not assignment[left] or assignment[right]
                elif rule.startswith("not ("):
                    parts = rule.removeprefix("not (").removesuffix(")").split(" and ")
                    assert not (assignment[parts[0]] and assignment[parts[1]])
        else:
            implication = rules[1]
            b = implication.split("->")[1].strip()
            assert rules[2] == f"not {b}"
            assert answer["minimal_unsat_core"] == rules
