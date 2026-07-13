from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mop.config import REPO_ROOT
from mop.studio import campaign_supervisor as campaign

PLAN_PATH = REPO_ROOT / "configs/campaign/mac_studio_local.json"


class FixedClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 12, 8, 15, tzinfo=UTC)

    def now(self) -> datetime:
        return self.value


def _supervisor(
    tmp_path: Path,
    *,
    active_probe=lambda: [],
    completion_probe=lambda _step: campaign.CompletionResult(False),
    outcome_probe=lambda _run_id: None,
    admission_probe=lambda _task: campaign.AdmissionResult(True),
    launcher=lambda _task, _run, _summary: campaign.LaunchResult(9, (), "out", "err"),
    execute: bool = False,
) -> campaign.CampaignSupervisor:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    clock = FixedClock()
    policy_hash = plan.policy.sha256
    governor_hash = campaign.sha256_file(campaign.THROTTLE_IMPLEMENTATION_PATH)
    return campaign.CampaignSupervisor(
        plan,
        out_dir=tmp_path,
        execute=execute,
        active_probe=active_probe,
        completion_probe=completion_probe,
        outcome_probe=outcome_probe,
        admission_probe=admission_probe,
        launcher=launcher,
        identity_probe=lambda: (policy_hash, governor_hash),
        telemetry_probe=lambda: {"memory": {"pressure": "normal"}},
        now_fn=clock.now,
        sleep_fn=lambda _seconds: None,
    )


def test_plan_is_valid_dag_and_rejects_arbitrary_commands(tmp_path: Path) -> None:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    assert [step.task_id for step in plan.steps] == [
        "p5fresh_challenge_cpu",
        "p5verify_cpu",
        None,
    ]

    raw = json.loads(PLAN_PATH.read_text())
    raw["steps"][0]["command"] = ["echo", "not-authorized"]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="unknown keys"):
        campaign.load_campaign_plan(invalid)


def test_plan_artifact_must_equal_the_task_declared_output(tmp_path: Path) -> None:
    raw = json.loads(PLAN_PATH.read_text())
    raw["steps"][0]["artifact"]["path"] = "proof/WRONG_OUTPUT.json"
    invalid = tmp_path / "wrong-output.json"
    invalid.write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="artifact path must equal.*declared output"):
        campaign.load_campaign_plan(invalid)


@pytest.mark.parametrize("flag", ("--out", "--output", "--verification-out"))
def test_task_output_authority_accepts_each_supported_flag(flag: str) -> None:
    task = campaign.load_campaign_plan(PLAN_PATH).policy.task("p5verify_cpu")
    output = "proof/DECLARED_OUTPUT.json"
    command = ("python", "runner.py", flag, output)

    assert campaign._declared_task_output(replace(task, command=command), "test task") == output


def test_task_output_authority_rejects_ambiguous_flags() -> None:
    task = campaign.load_campaign_plan(PLAN_PATH).policy.task("p5verify_cpu")
    command = (
        "python",
        "runner.py",
        "--out",
        "proof/ONE.json",
        "--verification-out",
        "proof/TWO.json",
    )

    with pytest.raises(ValueError, match="exactly one output authority"):
        campaign._declared_task_output(replace(task, command=command), "test task")


def test_task_output_authority_rejects_an_option_as_the_target() -> None:
    task = campaign.load_campaign_plan(PLAN_PATH).policy.task("p5verify_cpu")

    with pytest.raises(ValueError, match="missing its output target"):
        campaign._declared_task_output(
            replace(task, command=("python", "runner.py", "--out", "--device", "cpu")),
            "test task",
        )


@pytest.mark.parametrize(
    ("schema", "seal_field"),
    tuple(campaign.NATIVE_ARTIFACT_SEAL_FIELDS.items()),
)
def test_native_substrate_artifact_seals_are_validated(schema: str, seal_field: str) -> None:
    core = {"schema": schema, "scientific_promotion": False}
    payload = {**core, seal_field: campaign.canonical_sha256(core)}

    assert campaign._native_artifact_seal_problems(payload, schema) == []

    payload["scientific_promotion"] = True
    assert "self-seal mismatch" in campaign._native_artifact_seal_problems(payload, schema)[0]


def test_plan_rejects_dependency_cycles(tmp_path: Path) -> None:
    raw = json.loads(PLAN_PATH.read_text())
    raw["steps"][0]["depends_on"] = ["p5_independent_verification"]
    invalid = tmp_path / "cycle.json"
    invalid.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="dependency cycle"):
        campaign.load_campaign_plan(invalid)


def test_campaign_lock_is_nonblocking(tmp_path: Path) -> None:
    lock_path = tmp_path / "campaign.lock"
    with (
        campaign.CampaignLock(lock_path),
        pytest.raises(RuntimeError, match="already held"),
        campaign.CampaignLock(lock_path),
    ):
        pass


def test_campaign_lock_is_global_across_output_overrides(tmp_path: Path) -> None:
    plan = replace(campaign.load_campaign_plan(PLAN_PATH), campaign_root=tmp_path / "campaign-root")
    first = campaign.CampaignSupervisor(
        plan,
        out_dir=tmp_path / "one",
        telemetry_probe=lambda: {},
    )
    second = campaign.CampaignSupervisor(
        plan,
        out_dir=tmp_path / "two",
        telemetry_probe=lambda: {},
    )

    assert first.lock_path == second.lock_path
    with (
        campaign.CampaignLock(first.lock_path),
        pytest.raises(RuntimeError, match="already held"),
        campaign.CampaignLock(second.lock_path),
    ):
        pass


def test_registered_child_command_accepts_exact_declared_or_pinned_post_exec() -> None:
    payload = (
        ".venv/bin/python",
        "scripts/run_edcm1_event_triggered_coalition.py",
        "--device",
        "cpu",
    )
    declared = (*campaign.TASKPOLICY_COEXISTENCE_PREFIX, *payload)

    assert campaign._registered_child_command_matches(declared, declared)
    assert campaign._registered_child_command_matches(declared, payload)


@pytest.mark.parametrize(
    "observed",
    (
        ("scripts/run_edcm1_event_triggered_coalition.py", "--device", "cpu"),
        (
            ".venv/bin/python",
            "scripts/run_edcm1_event_triggered_coalition.py",
            "--device",
            "mps",
        ),
        (
            ".venv/bin/python",
            "scripts/run_edcm1_event_triggered_coalition.py",
            "--device",
            "cpu",
            "--spoof",
        ),
    ),
)
def test_registered_child_command_rejects_partial_or_spoofed_post_exec(
    observed: tuple[str, ...],
) -> None:
    payload = (
        ".venv/bin/python",
        "scripts/run_edcm1_event_triggered_coalition.py",
        "--device",
        "cpu",
    )
    declared = (*campaign.TASKPOLICY_COEXISTENCE_PREFIX, *payload)

    assert not campaign._registered_child_command_matches(declared, observed)


def test_registered_child_command_rejects_arbitrary_unpinned_suffix() -> None:
    payload = (".venv/bin/python", "scripts/run.py")
    declared = ("/usr/bin/env", "OMP_NUM_THREADS=1", *payload)

    assert not campaign._registered_child_command_matches(declared, payload)


def test_existing_governor_is_adopted_observationally_and_state_is_sealed(
    tmp_path: Path,
) -> None:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    task = plan.policy.task("p5fresh_challenge_cpu")
    observed = campaign.ObservedRun(
        task_id=task.task_id,
        run_id="existing-p5-leg",
        scheduler_pid=101,
        child_pid=102,
        status="running",
        command=task.command,
        receipt_path="runs/local_throttle/existing-p5-leg/run_receipt.json",
    )
    supervisor = _supervisor(tmp_path, active_probe=lambda: [observed])

    status = supervisor.tick()

    assert status["state"] == "observing_existing"
    row = status["steps"]["p5_fresh_challenge"]
    assert row["adopted"] is True
    assert row["run_id"] == "existing-p5-leg"
    assert status["steps"]["p5_fresh_challenge"]["leg"] == 0
    assert campaign.read_campaign_status(tmp_path)["status_sha256"] == status["status_sha256"]
    state = campaign.read_json(tmp_path / campaign.STATE_FILE)
    campaign._validate_seal(state, "state_sha256", "test state")
    assert len(list((tmp_path / campaign.HOURLY_DIR).glob("*.json"))) == 1


def test_adopted_governor_birth_identity_cannot_change_between_polls(tmp_path: Path) -> None:
    observations = [
        campaign.ObservedRun(
            task_id="p5fresh_challenge_cpu",
            run_id="same-run-id",
            scheduler_pid=10,
            child_pid=11,
            status="running",
            command=(),
            receipt_path="receipt.json",
            scheduler_create_time=100.0,
        )
    ]
    supervisor = _supervisor(tmp_path, active_probe=lambda: list(observations))
    supervisor.tick()
    observations[0] = replace(observations[0], scheduler_create_time=101.0)

    status = supervisor.tick()

    assert status["state"] == "integrity_hold"
    assert "changed scheduler birth identity" in status["problems"][0]


def test_ready_step_launches_only_named_throttle_task(tmp_path: Path) -> None:
    launched: list[str] = []

    def completion(step: campaign.CampaignStep) -> campaign.CompletionResult:
        return campaign.CompletionResult(
            step.step_id == "p5_fresh_challenge",
            artifact_sha256="a" * 64,
        )

    def launcher(task_id: str, _run_id: str, _summary: Path) -> campaign.LaunchResult:
        launched.append(task_id)
        return campaign.LaunchResult(42, ("local_execution_throttle.py",), "out", "err")

    supervisor = _supervisor(
        tmp_path,
        completion_probe=completion,
        launcher=launcher,
        execute=True,
    )
    status = supervisor.tick()

    assert status["state"] == "running"
    assert launched == ["p5verify_cpu"]
    assert status["steps"]["p5_independent_verification"]["leg"] == 1


def test_default_launcher_invokes_throttle_wrapper_with_explicit_pythonpath(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class Process:
        pid = 777

    def fake_popen(command: tuple[str, ...], **kwargs: object) -> Process:
        captured["command"] = command
        captured["environment"] = kwargs["env"]
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", fake_popen)
    supervisor = _supervisor(tmp_path)
    result = supervisor._default_launcher("p5verify_cpu", "test-run", tmp_path / "summary.json")

    command = list(result.command)
    assert command[1].endswith("scripts/local_execution_throttle.py")
    assert command[2:5] == ["--policy", str(supervisor.plan.policy_path), "run"]
    assert command[command.index("--task") + 1] == "p5verify_cpu"
    environment = captured["environment"]
    assert isinstance(environment, dict)
    assert str(REPO_ROOT / "src") in environment["PYTHONPATH"]
    assert str(REPO_ROOT) in environment["PYTHONPATH"]


def test_foreground_run_publishes_start_ack_before_slow_admission(tmp_path: Path) -> None:
    base = campaign.load_campaign_plan(PLAN_PATH)
    plan = replace(base, campaign_root=tmp_path / "campaign-root")
    out_dir = tmp_path / "out"
    ack_visible: list[bool] = []

    def admission(_task: str) -> campaign.AdmissionResult:
        ack_visible.append((out_dir / campaign.STATUS_FILE).is_file())
        return campaign.AdmissionResult(False, ("test refusal",))

    supervisor = campaign.CampaignSupervisor(
        plan,
        out_dir=out_dir,
        execute=True,
        active_probe=lambda: [],
        completion_probe=lambda _step: campaign.CompletionResult(False),
        admission_probe=admission,
        telemetry_probe=lambda: {},
        sleep_fn=lambda _seconds: None,
    )

    supervisor.run(max_cycles=1)

    assert ack_visible == [True]
    assert campaign.read_campaign_status(out_dir)["state"] == "backoff"


def test_default_admission_probe_propagates_distinct_decision_denials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor = _supervisor(tmp_path)
    monkeypatch.setattr(
        campaign,
        "dry_run_decision",
        lambda *_args, **_kwargs: {
            "admission": {"allowed": False, "reason": "aggregate refusal"},
            "decisions": [
                {
                    "allowed": False,
                    "denied_reasons": ["CPU pressure", "memory pressure"],
                },
                {
                    "allowed": False,
                    "denied_reasons": ["CPU pressure", "thermal pressure"],
                },
            ],
        },
    )

    result = supervisor._default_admission_probe("p5verify_cpu")

    assert result.allowed is False
    assert result.reasons == ("CPU pressure", "memory pressure", "thermal pressure")


def test_launched_admission_refusal_preserves_aggregate_denials(tmp_path: Path) -> None:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    task = plan.policy.task("p5fresh_challenge_cpu")
    active = [
        campaign.ObservedRun(
            task_id=task.task_id,
            run_id="refused-leg",
            scheduler_pid=401,
            child_pid=None,
            status="launching",
            command=task.command,
            receipt_path="receipt.json",
        )
    ]
    supervisor = _supervisor(
        tmp_path,
        active_probe=lambda: list(active),
        outcome_probe=lambda _run_id: campaign.RunOutcome(
            "admission-refused",
            None,
            None,
            "receipt.json",
            ("CPU pressure", "memory pressure"),
        ),
    )
    supervisor.tick()
    active.clear()

    status = supervisor.tick()

    row = status["steps"]["p5_fresh_challenge"]
    assert row["status"] == "backoff"
    assert row["last_problem"] == "CPU pressure; memory pressure"


def test_completion_receipt_without_current_prerequisite_join_is_rejected() -> None:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    problems = campaign._governor_receipt_valid(
        {},
        task_id="p5fresh_challenge_cpu",
        artifact_path="proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json",
        artifact_sha256="a" * 64,
        policy=plan.policy,
    )

    assert any("did not bind current prerequisite" in problem for problem in problems)


def test_drift_holds_before_admission_or_launch(tmp_path: Path) -> None:
    launched: list[str] = []
    supervisor = _supervisor(
        tmp_path,
        execute=True,
        launcher=lambda task, _run, _summary: (
            launched.append(task) or campaign.LaunchResult(1, (), "out", "err")
        ),
    )
    supervisor.identity_probe = lambda: ("0" * 64, supervisor.pinned_throttle_sha256)

    status = supervisor.tick()

    assert status["state"] == "policy_drift_hold"
    assert launched == []


def test_sealed_marker_is_the_only_policy_baseline_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = campaign.load_campaign_plan(PLAN_PATH)
    marker_dir = REPO_ROOT / "runs" / f"campaign_supervisor_test_{tmp_path.name}"
    marker_path = marker_dir / "MIGRATION_READY.json"
    marker_relative = str(marker_path.relative_to(REPO_ROOT))
    marker_step = replace(
        base.steps[-1],
        artifact=replace(base.steps[-1].artifact, path=marker_relative),
    )
    plan = replace(base, steps=(*base.steps[:-1], marker_step))
    clock = FixedClock()
    supervisor = campaign.CampaignSupervisor(
        plan,
        out_dir=tmp_path,
        active_probe=lambda: [],
        completion_probe=lambda _step: campaign.CompletionResult(False),
        telemetry_probe=lambda: {},
        now_fn=clock.now,
        sleep_fn=lambda _seconds: None,
    )
    for step in plan.steps[:-1]:
        supervisor.state["steps"][step.step_id]["status"] = "complete"
    new_policy_hash = "1" * 64
    replacement_policy = replace(base.policy, sha256=new_policy_hash)
    monkeypatch.setattr(campaign, "load_policy", lambda _path: replacement_policy)
    supervisor.identity_probe = lambda: (new_policy_hash, supervisor.pinned_throttle_sha256)
    core = {
        "schema": campaign.MIGRATION_MARKER_SCHEMA,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.sha256,
        "ready": True,
        "expected_old": {
            "policy_sha256": supervisor.pinned_policy_sha256,
            "governor_sha256": supervisor.pinned_throttle_sha256,
        },
        "expected_new": {
            "policy_sha256": new_policy_hash,
            "governor_sha256": supervisor.pinned_throttle_sha256,
        },
        "created_at": clock.now().isoformat(),
        "reason": "unit-test controlled transition",
    }
    marker = campaign._sealed(core, "marker_sha256")
    campaign.atomic_write_json(marker_path, marker)
    try:
        status = supervisor.tick()
    finally:
        marker_path.unlink(missing_ok=True)
        marker_dir.rmdir()

    assert status["state"] == "complete"
    assert supervisor.pinned_policy_sha256 == new_policy_hash
    assert status["steps"][marker_step.step_id]["artifact_sha256"] is not None
    assert supervisor.state["baseline_transitions"][0]["marker_sha256"] == marker["marker_sha256"]


def test_governor_code_transition_requires_fresh_supervisor_process(tmp_path: Path) -> None:
    base = campaign.load_campaign_plan(PLAN_PATH)
    marker_dir = REPO_ROOT / "runs" / f"campaign_supervisor_restart_test_{tmp_path.name}"
    marker_path = marker_dir / "MIGRATION_READY.json"
    marker_step = replace(
        base.steps[-1],
        artifact=replace(base.steps[-1].artifact, path=str(marker_path.relative_to(REPO_ROOT))),
    )
    plan = replace(base, steps=(*base.steps[:-1], marker_step))
    supervisor = campaign.CampaignSupervisor(
        plan,
        out_dir=tmp_path,
        active_probe=lambda: [],
        completion_probe=lambda _step: campaign.CompletionResult(False),
        telemetry_probe=lambda: {},
        now_fn=FixedClock().now,
        sleep_fn=lambda _seconds: None,
    )
    for step in plan.steps[:-1]:
        supervisor.state["steps"][step.step_id]["status"] = "complete"
    new_governor_hash = "2" * 64
    supervisor.identity_probe = lambda: (supervisor.pinned_policy_sha256, new_governor_hash)
    core = {
        "schema": campaign.MIGRATION_MARKER_SCHEMA,
        "campaign_id": plan.campaign_id,
        "plan_sha256": plan.sha256,
        "ready": True,
        "expected_old": {
            "policy_sha256": supervisor.pinned_policy_sha256,
            "governor_sha256": supervisor.pinned_throttle_sha256,
        },
        "expected_new": {
            "policy_sha256": supervisor.pinned_policy_sha256,
            "governor_sha256": new_governor_hash,
        },
        "created_at": "2026-07-12T08:15:00+00:00",
        "reason": "unit-test governor transition",
    }
    marker = campaign._sealed(core, "marker_sha256")
    campaign.atomic_write_json(marker_path, marker)
    try:
        status = supervisor.tick()
    finally:
        marker_path.unlink(missing_ok=True)
        marker_dir.rmdir()

    assert status["state"] == "migration_restart_required"
    assert supervisor.state["pending_baseline_transition"]["marker_sha256"] == marker["marker_sha256"]


def test_resumable_boundary_records_progress_without_failure(tmp_path: Path) -> None:
    plan = campaign.load_campaign_plan(PLAN_PATH)
    task = plan.policy.task("p5fresh_challenge_cpu")
    active = [
        campaign.ObservedRun(
            task_id=task.task_id,
            run_id="resumable-leg",
            scheduler_pid=201,
            child_pid=202,
            status="running",
            command=task.command,
            receipt_path="receipt.json",
        )
    ]
    supervisor = _supervisor(
        tmp_path,
        active_probe=lambda: list(active),
        outcome_probe=lambda _run_id: campaign.RunOutcome(
            "resumable-wall-boundary",
            2,
            "b" * 64,
            "receipt.json",
        ),
    )
    supervisor.tick()
    active.clear()

    status = supervisor.tick()

    row = status["steps"]["p5_fresh_challenge"]
    assert row["run_id"] is None
    assert row["last_checkpoint_sha256"] == "b" * 64
    assert row["failures"] == 0
    assert row["no_progress_legs"] == 0


def test_probe_run_outcome_requires_sealed_invocation_progress(tmp_path: Path) -> None:
    run_id = "sealed-one-seed"
    receipt_path = tmp_path / run_id / "run_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    checkpoint = "c" * 64
    task = {"task_id": "edcm1_official_cpu", "command": ["python", "study.py"]}
    policy = {"path": "policy.yaml", "sha256": "p" * 64}
    implementation = {"path": "governor.py", "sha256": "g" * 64}
    task_policy_authority = {"authority_sha256": "a" * 64}
    child_resource = {"peak_rss_bytes": 1234}
    progress = {
        "schema": campaign.PROGRESS_AUTHORITY_SCHEMA,
        "task_id": task["task_id"],
        "task": task,
        "command": task["command"],
        "policy": policy,
        "implementation": implementation,
        "task_policy_authority": task_policy_authority,
        "returncode": 2,
        "final_checkpoint_aggregate_sha256": checkpoint,
        "owned_child_active": False,
        "child_resource": child_resource,
    }
    core = {
        "status": "resumable-invocation-boundary",
        "final_returncode": 2,
        "task": task,
        "policy": policy,
        "implementation": implementation,
        "task_policy_authority": task_policy_authority,
        "child_resource": child_resource,
        "final_checkpoint": {"aggregate_sha256": checkpoint},
        "progress_authority": progress,
    }
    receipt = {**core, "payload_sha256": campaign.canonical_sha256(core)}
    campaign.atomic_write_json(receipt_path, receipt)

    outcome = campaign.probe_run_outcome(run_id, tmp_path)
    assert outcome is not None
    assert outcome.status == "resumable-invocation-boundary"
    assert outcome.checkpoint_sha256 == checkpoint

    receipt["progress_authority"]["owned_child_active"] = True
    campaign.atomic_write_json(receipt_path, receipt)
    tampered = campaign.probe_run_outcome(run_id, tmp_path)
    assert tampered is not None
    assert tampered.status == "receipt-invalid"


def test_drain_request_never_launches_or_signals_active_work(tmp_path: Path) -> None:
    launched: list[str] = []
    supervisor = _supervisor(
        tmp_path,
        active_probe=lambda: [
            campaign.ObservedRun(
                task_id="p5fresh_challenge_cpu",
                run_id="leave-me-running",
                scheduler_pid=301,
                child_pid=302,
                status="running",
                command=(),
                receipt_path="receipt.json",
            )
        ],
        launcher=lambda task, _run, _summary: (
            launched.append(task) or campaign.LaunchResult(1, (), "out", "err")
        ),
        execute=True,
    )
    campaign.request_drain(tmp_path, "test drain")

    status = supervisor.tick()

    assert status["state"] == "draining"
    assert launched == []
