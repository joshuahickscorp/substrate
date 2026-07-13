from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mop.config import REPO_ROOT
from mop.studies.continual_million_event_verify import IMPLEMENTATION_PATHS, TIE_RULE
from mop.studio import campaign_supervisor
from mop.studio.local_throttle import TaskDeclaration, load_policy
from mop.studio.null_safe_campaign_router import (
    EXPECTED_CHECK_NAMES,
    EXPECTED_CONTROLS,
    EXPECTED_METRIC_FAMILIES,
    EXPECTED_SCHEDULES,
    FAVORABLE_REASON,
    NULL_REASON,
    P6_RUNG_SCHEMA,
    P6_VERIFIER_SCHEMA,
    RouterRefused,
    Stage,
    canonical_sha256,
    load_router_plan,
    start_router_detached,
    validate_prepared_router_plan,
    validate_terminal_verifier,
)
from mop.studio.policy_overlay import load_task_overlay


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage(rung: int) -> Stage:
    labels = {
        10_000: (
            "fixture-10k",
            "configs/campaign/mac_studio_substrate_phase1_null_safe_10k.json",
            "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
            "proof/P6_CONTINUAL_10K.json",
            100_000,
        ),
        100_000: (
            "fixture-100k",
            "configs/campaign/mac_studio_substrate_p6_100k_null_safe.json",
            "proof/P6_CONTINUAL_100K_INDEPENDENT_VERIFICATION.json",
            "proof/P6_CONTINUAL_100K.json",
            1_000_000,
        ),
    }
    stage_id, plan, verifier, source, next_rung = labels[rung]
    return Stage(
        stage_id=stage_id,
        plan_path=REPO_ROOT / plan,
        verifier_path=REPO_ROOT / verifier,
        source_path=REPO_ROOT / source,
        rung=rung,
        next_rung=next_rung,
    )


def _paired_row(seed: int, *, retention: float, future: float) -> dict[str, object]:
    return {
        "seed": seed,
        "retention_delta": retention,
        "future_first_window_delta": future,
        "tie_is_null": retention == 0.0 or future == 0.0,
        "nonpositive_is_null": retention <= 0.0 or future <= 0.0,
    }


def _fixture(tmp_path: Path, *, favorable: bool) -> Stage:
    stage = _stage(10_000)
    seeds = [101, 102, 103, 104, 105]
    source_core: dict[str, object] = {
        "schema": P6_RUNG_SCHEMA,
        "mode": "replication",
        "rung": stage.rung,
        "all_mechanics_ok": True,
        "replication_execution_complete": True,
        "scientific_promotion": False,
        "identity_sha256": "b" * 64,
        "plan": {"seeds": seeds},
    }
    source = {**source_core, "payload_sha256": canonical_sha256(source_core)}
    source_path = tmp_path / stage.source_path.relative_to(REPO_ROOT)
    _write_json(source_path, source)

    contrasts: list[dict[str, object]] = []
    null_count = 0
    for schedule in EXPECTED_SCHEDULES:
        for control in EXPECTED_CONTROLS:
            null_row = not favorable and schedule == "abrupt" and control == "no-replay"
            pairs = [
                _paired_row(
                    seed,
                    retention=0.0 if null_row and index == 0 else 1.0,
                    future=1.0,
                )
                for index, seed in enumerate(seeds)
            ]
            nonpositive = any(bool(row["nonpositive_is_null"]) for row in pairs)
            strict = not nonpositive
            null_count += int(not strict)
            contrasts.append(
                {
                    "schedule": schedule,
                    "control": control,
                    "independent_units": len(seeds),
                    "paired_seed_deltas": pairs,
                    "retention_mean_delta": 0.8 if null_row else 1.0,
                    "future_first_window_mean_delta": 1.0,
                    "aggregate_tie_is_null": False,
                    "any_seed_nonpositive_is_null": nonpositive,
                    "null_contrast": not strict,
                    "strict_joint_gain": strict,
                }
            )
    decision = {
        "primary_endpoints": [
            "retention.domain_zero_final_accuracy",
            "future_learnability.first_window_accuracy",
        ],
        "independent_unit": "seed within transition schedule",
        "controls": list(EXPECTED_CONTROLS),
        "tie_rule": TIE_RULE,
        "contrasts": contrasts,
        "aggregate_tie_count": null_count,
        "strict_joint_gain_all_schedules_and_controls": favorable,
        "verdict": "favorable-rung-pattern" if favorable else "null",
        "null_supported": not favorable,
        "scientific_promotion": False,
        "claim_boundary": "fixture",
    }
    implementation: list[dict[str, str]] = []
    for index, relative in enumerate(IMPLEMENTATION_PATHS):
        implementation_path = tmp_path / relative
        implementation_path.parent.mkdir(parents=True, exist_ok=True)
        implementation_path.write_text(f"# frozen fixture {index}\n", encoding="utf-8")
        implementation.append(
            {
                "path": relative,
                "sha256": _sha256_file(implementation_path),
            }
        )
    next_allowed = favorable
    core: dict[str, object] = {
        "schema": P6_VERIFIER_SCHEMA,
        "claim_scope": "fixture",
        "source_rung": {
            "path": str(stage.source_path.relative_to(REPO_ROOT)),
            "file_sha256": _sha256_file(source_path),
            "payload_sha256": source["payload_sha256"],
            "identity_sha256": source["identity_sha256"],
            "rung": stage.rung,
            "mode": "replication",
        },
        "live_dependencies": [],
        "progress_authority": {},
        "independent_recompute": {
            "cell_count": 30,
            "metric_families": list(EXPECTED_METRIC_FAMILIES),
            "checkpoint_state_recomputed": True,
            "controls_recomputed": True,
            "paired_metrics_recomputed": True,
            "decision": decision,
        },
        "mutation_suite": {
            "count": 12,
            "rejected": 12,
            "all_rejected": True,
            "mutations": [
                {"mutation": f"fixture-{index}", "rejected": True, "problems": ["rejected"]}
                for index in range(12)
            ],
        },
        "checks": {name: True for name in EXPECTED_CHECK_NAMES},
        "verification_complete": True,
        "errors": [],
        "prerequisite": {
            "source_rung": stage.rung,
            "source_rung_file_sha256": _sha256_file(source_path),
            "source_identity_sha256": source["identity_sha256"],
            "verification_complete": True,
            "valid_controls": True,
            "tie_is_null": True,
            "mutation_suite_all_rejected": True,
            "next_rung": stage.next_rung,
            "next_rung_allowed": next_allowed,
            "next_rung_reason": FAVORABLE_REASON if next_allowed else NULL_REASON,
        },
        "scientific_promotion": False,
        "implementation": implementation,
    }
    verifier = {**core, "payload_sha256": canonical_sha256(core)}
    _write_json(tmp_path / stage.verifier_path.relative_to(REPO_ROOT), verifier)
    return stage


def test_prepared_router_plans_are_null_terminal_and_nonlaunching() -> None:
    plan = load_router_plan()
    report = validate_prepared_router_plan(plan)

    assert report["valid"] is True
    assert report["problems"] == []
    assert report["execution_authorized"] is False
    assert report["live_missing_tasks"] == []


def test_each_rung_plan_loads_under_the_prepared_additive_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = load_policy()
    overlay = load_task_overlay(
        REPO_ROOT / "configs/campaign/substrate_task_overlay.yaml",
        repository_root=REPO_ROOT,
    )
    tasks = dict(policy.tasks)
    tasks.update(
        {
            task_id: TaskDeclaration.from_mapping(task_id, dict(declaration))
            for task_id, declaration in overlay.tasks.items()
        }
    )
    future_policy = replace(policy, tasks=tasks)
    monkeypatch.setattr(campaign_supervisor, "load_policy", lambda _path: future_policy)

    plans = [campaign_supervisor.load_campaign_plan(stage.plan_path) for stage in load_router_plan().stages]

    assert [len(plan.steps) for plan in plans] == [7, 2, 2]
    assert [plan.steps[-1].task_id for plan in plans] == [
        "p6_10k_verify_cpu",
        "p6_100k_verify_cpu",
        "p6_1m_verify_cpu",
    ]


@pytest.mark.parametrize(("favorable", "outcome"), [(True, "favorable"), (False, "null")])
def test_terminal_verifier_routes_exact_favorable_or_valid_null(
    tmp_path: Path,
    favorable: bool,
    outcome: str,
) -> None:
    stage = _fixture(tmp_path, favorable=favorable)

    result = validate_terminal_verifier(stage, root=tmp_path)

    assert result["outcome"] == outcome
    assert result["next_rung_allowed"] is favorable
    assert result["scientific_promotion"] is False


def test_null_cannot_authorize_next_rung(tmp_path: Path) -> None:
    stage = _fixture(tmp_path, favorable=False)
    verifier_path = tmp_path / stage.verifier_path.relative_to(REPO_ROOT)
    payload = json.loads(verifier_path.read_text(encoding="utf-8"))
    payload["prerequisite"]["next_rung_allowed"] = True
    core = dict(payload)
    core.pop("payload_sha256")
    payload["payload_sha256"] = canonical_sha256(core)
    _write_json(verifier_path, payload)

    with pytest.raises(RouterRefused, match="next-rung authority drifted"):
        validate_terminal_verifier(stage, root=tmp_path)


def test_detached_start_requires_explicit_execute() -> None:
    with pytest.raises(RouterRefused, match="explicit --execute"):
        start_router_detached(load_router_plan(), execute=False, use_caffeinate=False)
