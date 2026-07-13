from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mop.config import REPO_ROOT
from mop.studio import campaign_supervisor
from mop.studio.local_throttle import TaskDeclaration, load_policy
from mop.studio.policy_overlay import load_task_overlay

PLAN_PATH = REPO_ROOT / "configs/campaign/mac_studio_substrate_phase1.json"
POLICY_PATH = REPO_ROOT / "configs/local_execution_throttle.yaml"
OVERLAY_PATH = REPO_ROOT / "configs/campaign/substrate_task_overlay.yaml"


def _future_policy():
    policy = load_policy(POLICY_PATH)
    overlay = load_task_overlay(OVERLAY_PATH, repository_root=REPO_ROOT)
    tasks = dict(policy.tasks)
    tasks.update(
        {
            task_id: TaskDeclaration.from_mapping(task_id, dict(declaration))
            for task_id, declaration in overlay.tasks.items()
        }
    )
    return replace(policy, tasks=tasks)


def _declared_output(command: tuple[str, ...]) -> str:
    matches = [
        command[index + 1]
        for index, value in enumerate(command[:-1])
        if value in {"--out", "--output", "--verification-out"}
    ]
    assert len(matches) == 1
    return matches[0]


def _option_value(command: tuple[str, ...], option: str) -> str:
    matches = [command[index + 1] for index, value in enumerate(command[:-1]) if value == option]
    assert len(matches) == 1
    return matches[0]


def test_post_migration_campaign_is_one_dependency_ordered_long_run(
    monkeypatch,
) -> None:
    future_policy = _future_policy()
    monkeypatch.setattr(campaign_supervisor, "load_policy", lambda _path: future_policy)

    plan = campaign_supervisor.load_campaign_plan(PLAN_PATH)

    assert len(plan.steps) == 11
    assert plan.steps[0].step_id == "edcm1_official"
    assert plan.steps[-1].step_id == "p6_1m_independent_verification"
    assert all(step.kind == "throttle-task" for step in plan.steps)
    assert all(step.task_id is not None for step in plan.steps)
    assert len({step.task_id for step in plan.steps}) == len(plan.steps)
    for step in plan.steps:
        assert step.task_id is not None
        task = future_policy.tasks[step.task_id]
        assert step.artifact.path == _declared_output(task.command)

    expected_dependencies = {
        "edcm1_official": (),
        "edcm1_independent_verification": ("edcm1_official",),
        "escs_x0_official": ("edcm1_independent_verification",),
        "escs_x0_fresh_verification": ("escs_x0_official",),
        "p6_10k_resource_probe": ("escs_x0_fresh_verification",),
        "p6_10k_replication": ("p6_10k_resource_probe",),
        "p6_10k_independent_verification": ("p6_10k_replication",),
        "p6_100k_replication": ("p6_10k_independent_verification",),
        "p6_100k_independent_verification": ("p6_100k_replication",),
        "p6_1m_replication": ("p6_100k_independent_verification",),
        "p6_1m_independent_verification": ("p6_1m_replication",),
    }
    assert {step.step_id: step.depends_on for step in plan.steps} == expected_dependencies


def test_verifier_source_and_output_flags_match_each_runner(monkeypatch) -> None:
    future_policy = _future_policy()
    monkeypatch.setattr(campaign_supervisor, "load_policy", lambda _path: future_policy)
    plan = campaign_supervisor.load_campaign_plan(PLAN_PATH)
    by_id = {step.step_id: step for step in plan.steps}

    pairs = (
        (
            "edcm1_independent_verification",
            "--verify",
            "edcm1_official",
            "--verification-out",
        ),
        (
            "escs_x0_fresh_verification",
            "--verify",
            "escs_x0_official",
            "--verification-out",
        ),
        (
            "p6_10k_independent_verification",
            "--source",
            "p6_10k_replication",
            "--out",
        ),
        (
            "p6_100k_independent_verification",
            "--source",
            "p6_100k_replication",
            "--out",
        ),
        (
            "p6_1m_independent_verification",
            "--source",
            "p6_1m_replication",
            "--out",
        ),
    )
    for verifier_id, source_flag, producer_id, output_flag in pairs:
        verifier = by_id[verifier_id]
        assert verifier.task_id is not None
        command = future_policy.tasks[verifier.task_id].command
        assert _option_value(command, source_flag) == by_id[producer_id].artifact.path
        assert _option_value(command, output_flag) == verifier.artifact.path


def test_p6_scale_edges_require_favorable_independent_verdicts(monkeypatch) -> None:
    future_policy = _future_policy()
    monkeypatch.setattr(campaign_supervisor, "load_policy", lambda _path: future_policy)
    plan = campaign_supervisor.load_campaign_plan(PLAN_PATH)
    by_id = {step.step_id: dict(step.artifact.fields) for step in plan.steps}

    for rung, next_rung in (("10k", 100_000), ("100k", 1_000_000)):
        fields = by_id[f"p6_{rung}_independent_verification"]
        assert fields["verification_complete"] is True
        assert fields["errors"] == []
        assert fields["independent_recompute.decision.verdict"] == "favorable-rung-pattern"
        assert fields["independent_recompute.decision.null_supported"] is False
        assert fields["prerequisite.next_rung"] == next_rung
        assert fields["prerequisite.next_rung_allowed"] is True

    final_fields = by_id["p6_1m_independent_verification"]
    assert final_fields["verification_complete"] is True
    assert final_fields["errors"] == []
    assert final_fields["independent_recompute.decision.verdict"] == "favorable-rung-pattern"
    assert final_fields["independent_recompute.decision.null_supported"] is False


def test_phase1_requires_the_reviewed_overlay_before_it_can_load() -> None:
    # The live policy intentionally remains the P5 authority while P5 is active.
    # This refusal is the migration barrier, not a malformed campaign plan.
    try:
        campaign_supervisor.load_campaign_plan(PLAN_PATH)
    except ValueError as exc:
        assert "not a configured throttle task" in str(exc)
    else:  # pragma: no cover - becomes reachable only after the controlled migration
        assert {"edcm1_official_cpu", "escs_x0_official_cpu"} <= set(load_policy(POLICY_PATH).tasks)


def test_campaign_paths_are_repository_relative() -> None:
    payload = PLAN_PATH.read_text(encoding="utf-8")
    assert str(Path(PLAN_PATH).resolve()).startswith(str(REPO_ROOT.resolve()))
    assert '"campaign_root": "runs/mac_studio_campaign"' in payload
