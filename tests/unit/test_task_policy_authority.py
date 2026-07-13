from __future__ import annotations

import copy

import pytest

from mop.studio.task_policy_authority import (
    POLICY_BASELINE_MANIFEST_SCHEMA,
    TASK_POLICY_AUTHORITY_SCHEMA,
    build_policy_baseline_manifest,
    build_policy_safety_contract,
    build_task_policy_authority,
    canonical_sha256,
    policy_baseline_manifest_problems,
    receipt_task_policy_authority_problems,
    task_policy_authority_problems,
)


def fixture() -> dict:
    return {
        "policy_schema": "mop-local-execution-throttle-policy/v1",
        "policy_path": "/repo/configs/local_execution_throttle.yaml",
        "full_policy_sha256": "a" * 64,
        "profile_name": "m3pro-local-max",
        "safety_contract": {
            "profile": {"min_free_disk_gb": 40.0},
            "limits": {"max_heavy": 1},
            "monitor": {"admission_good_samples": 3},
            "thresholds": {"first_lane": {"maximum_cpu": 0.85}},
        },
        "foreground_markers": ("Blender",),
        "known_heavy_markers": ("p5_context_capability.py",),
        "task_id": "p5pilot_cpu",
        "task_payload": {"task_id": "p5pilot_cpu", "cpu_cores": 10},
    }


def reseal(authority: dict) -> None:
    authority.pop("authority_sha256", None)
    authority["authority_sha256"] = canonical_sha256(authority)


def test_task_policy_authority_accepts_unrelated_policy_addition_and_marker_superset() -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    current = {
        **values,
        "full_policy_sha256": "b" * 64,
        "known_heavy_markers": (*values["known_heavy_markers"], "edcm1.py"),
    }

    assert authority["schema"] == TASK_POLICY_AUTHORITY_SCHEMA
    assert authority["scientific_promotion"] is False
    assert task_policy_authority_problems(authority, **current) == []


@pytest.mark.parametrize(
    ("field", "replacement", "problem"),
    [
        ("safety_contract", {"limits": {"max_heavy": 2}}, "safety contract"),
        ("task_payload", {"task_id": "p5pilot_cpu", "cpu_cores": 11}, "task declaration"),
        ("task_id", "p5fresh_challenge_cpu", "task id"),
    ],
)
def test_task_policy_authority_rejects_scope_drift(
    field: str,
    replacement: object,
    problem: str,
) -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    current = {**values, field: replacement}

    assert any(problem in row for row in task_policy_authority_problems(authority, **current))


def test_task_policy_authority_rejects_marker_removal_and_splice() -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    no_marker = {**values, "known_heavy_markers": ()}
    assert "task-policy known_heavy_markers were removed" in task_policy_authority_problems(
        authority,
        **no_marker,
    )

    spliced = copy.deepcopy(authority)
    spliced["task_sha256"] = "0" * 64
    assert "task-policy authority self-seal mismatch" in task_policy_authority_problems(
        spliced,
        **values,
    )


def test_task_policy_authority_rejects_noncanonical_or_promoting_payload() -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    authority["known_heavy_markers"] = ["z", "a"]
    authority["scientific_promotion"] = True
    reseal(authority)
    problems = task_policy_authority_problems(authority, **values)

    assert "task-policy known_heavy_markers are not canonical" in problems
    assert "task-policy authority cannot grant scientific promotion" in problems


def test_builder_rejects_duplicate_markers_and_invalid_digest() -> None:
    values = fixture()
    with pytest.raises(ValueError, match="duplicates"):
        build_task_policy_authority(
            **{**values, "known_heavy_markers": ("same", "same")},
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_task_policy_authority(
            **{**values, "full_policy_sha256": "invalid"},
        )


def test_safety_contract_separates_only_monotone_marker_sets() -> None:
    contract = build_policy_safety_contract(
        profile={"maximum_cpu_fraction": 0.85},
        limits={"maximum_active_heavy": 1},
        monitor={
            "admission_good_samples": 3,
            "foreground_markers": ["Blender"],
            "known_heavy_markers": ["p5.py"],
        },
        thresholds={"first_lane": {"maximum_cpu": 0.85}},
    )

    assert contract == {
        "profile": {"maximum_cpu_fraction": 0.85},
        "limits": {"maximum_active_heavy": 1},
        "monitor": {"admission_good_samples": 3},
        "thresholds": {"first_lane": {"maximum_cpu": 0.85}},
    }


def test_policy_baseline_manifest_seals_sorted_task_authorities() -> None:
    values = fixture()
    manifest = build_policy_baseline_manifest(
        policy_schema=values["policy_schema"],
        policy_path=values["policy_path"],
        full_policy_sha256=values["full_policy_sha256"],
        governor_implementation_path="src/mop/studio/local_throttle.py",
        governor_implementation_sha256="b" * 64,
        profile_name=values["profile_name"],
        safety_contract=values["safety_contract"],
        foreground_markers=values["foreground_markers"],
        known_heavy_markers=values["known_heavy_markers"],
        task_payloads={
            "p5verify_cpu": {"task_id": "p5verify_cpu"},
            values["task_id"]: values["task_payload"],
        },
    )

    assert manifest["schema"] == POLICY_BASELINE_MANIFEST_SCHEMA
    assert [row["task_id"] for row in manifest["task_authorities"]] == [
        "p5pilot_cpu",
        "p5verify_cpu",
    ]
    assert policy_baseline_manifest_problems(manifest) == []

    spliced = copy.deepcopy(manifest)
    spliced["task_authorities"][0]["task_sha256"] = "0" * 64
    assert {
        "policy baseline self-seal mismatch",
        "policy baseline task authority self-seal mismatch",
    } <= set(policy_baseline_manifest_problems(spliced))


def test_receipt_policy_join_accepts_embedded_and_reviewed_legacy_authority() -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    declared_policy = {
        "path": values["policy_path"],
        "sha256": values["full_policy_sha256"],
    }
    declared_implementation = {
        "path": "src/mop/studio/local_throttle.py",
        "sha256": "b" * 64,
    }
    current = {
        "declared_policy": declared_policy,
        "declared_implementation": declared_implementation,
        "declared_task_id": values["task_id"],
        "declared_task_payload": values["task_payload"],
        "legacy_manifests": (),
        "current_policy_schema": values["policy_schema"],
        "current_policy_path": values["policy_path"],
        "current_full_policy_sha256": "c" * 64,
        "current_profile_name": values["profile_name"],
        "current_safety_contract": values["safety_contract"],
        "current_foreground_markers": values["foreground_markers"],
        "current_known_heavy_markers": (*values["known_heavy_markers"], "new-task.py"),
        "current_task_payload": values["task_payload"],
    }
    assert (
        receipt_task_policy_authority_problems(
            embedded_authority=authority,
            **current,
        )
        == []
    )

    baseline = build_policy_baseline_manifest(
        policy_schema=values["policy_schema"],
        policy_path=values["policy_path"],
        full_policy_sha256=values["full_policy_sha256"],
        governor_implementation_path=declared_implementation["path"],
        governor_implementation_sha256=declared_implementation["sha256"],
        profile_name=values["profile_name"],
        safety_contract=values["safety_contract"],
        foreground_markers=values["foreground_markers"],
        known_heavy_markers=values["known_heavy_markers"],
        task_payloads={values["task_id"]: values["task_payload"]},
    )
    assert (
        receipt_task_policy_authority_problems(
            embedded_authority=None,
            **{**current, "legacy_manifests": (baseline,)},
        )
        == []
    )


def test_receipt_policy_join_rejects_unknown_legacy_and_task_splice() -> None:
    values = fixture()
    authority = build_task_policy_authority(**values)
    problems = receipt_task_policy_authority_problems(
        declared_policy={"path": values["policy_path"], "sha256": values["full_policy_sha256"]},
        declared_implementation={
            "path": "src/mop/studio/local_throttle.py",
            "sha256": "b" * 64,
        },
        declared_task_id=values["task_id"],
        declared_task_payload={"task_id": values["task_id"], "cpu_cores": 999},
        embedded_authority=authority,
        legacy_manifests=(),
        current_policy_schema=values["policy_schema"],
        current_policy_path=values["policy_path"],
        current_full_policy_sha256=values["full_policy_sha256"],
        current_profile_name=values["profile_name"],
        current_safety_contract=values["safety_contract"],
        current_foreground_markers=values["foreground_markers"],
        current_known_heavy_markers=values["known_heavy_markers"],
        current_task_payload=values["task_payload"],
    )
    assert "receipt task-policy declared task binding drifted" in problems

    unknown = receipt_task_policy_authority_problems(
        declared_policy={"path": values["policy_path"], "sha256": values["full_policy_sha256"]},
        declared_implementation={
            "path": "src/mop/studio/local_throttle.py",
            "sha256": "b" * 64,
        },
        declared_task_id=values["task_id"],
        declared_task_payload=values["task_payload"],
        embedded_authority=None,
        legacy_manifests=(),
        current_policy_schema=values["policy_schema"],
        current_policy_path=values["policy_path"],
        current_full_policy_sha256=values["full_policy_sha256"],
        current_profile_name=values["profile_name"],
        current_safety_contract=values["safety_contract"],
        current_foreground_markers=values["foreground_markers"],
        current_known_heavy_markers=values["known_heavy_markers"],
        current_task_payload=values["task_payload"],
    )
    assert unknown == ["receipt maps to 0 reviewed legacy policy baselines"]
