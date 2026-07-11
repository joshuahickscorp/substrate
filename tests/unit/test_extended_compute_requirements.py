from __future__ import annotations

import copy
import json

import pytest
from scripts.build_extended_compute_requirements import (
    P5_SMOKE_RECEIPT_PATH,
    P5_VERIFICATION_FIELDS,
    P5_VERIFICATION_PATH,
    P5_VERIFICATION_SCHEMA,
    _p5_smoke_refusal_summary,
    _p6_dry_prerequisite_state,
    registry_rows,
)

from mop.config import REPO_ROOT


def test_registry_only_rows_are_preregistration_only_requirements() -> None:
    rows, registry_ids = registry_rows()
    by_id = {row["id"]: row for row in rows}

    assert len(registry_ids) == 227
    assert by_id["f21_asynchronous_temporal_binding"]["primary_category"] == 2
    assert by_id["f21_asynchronous_temporal_binding"]["measured"]["present"] is False
    assert by_id["f21_asynchronous_temporal_binding"]["classification_basis"] == (
        "registry-preregistration-only"
    )

    assert by_id["f65_specimen_to_specimen_transfer"]["primary_category"] == 6
    assert by_id["f65_specimen_to_specimen_transfer"]["required_rung"] == "L6"
    assert by_id["f66_cross_substrate_form_portability"]["primary_category"] == 2
    assert by_id["f66_cross_substrate_form_portability"]["post_blocker_local_rung"] == "L0"

    p5 = by_id["mop_p5_context_capability"]
    assert p5["primary_category"] == 1
    assert p5["classification_basis"] == "implemented-governor-execution-pending"
    assert p5["measured"]["present"] is False
    assert p5["primary_blocker"] == (
        "complete the implemented conditional P5 sequence through the local governor after three "
        "healthy admission samples"
    )
    assert "local:proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json" in p5["evidence_refs"]
    admission = p5["measured"]["governor_admission"]
    assert admission["state"] == "memory-only-admission-refusal"
    assert admission["available_memory_gb"] == [8.16013312, 8.30414848, 8.381218816]
    assert admission["required_memory_gb"] == 10.0
    assert admission["command_executed"] is False

    e5 = by_id["e5_curiosity"]
    assert e5["primary_category"] == 1
    assert e5["classification_basis"] == "implemented-local-evidence-source-rerun-pending"
    assert "local:proof/LOCAL_ACTION_ENVIRONMENT.json" in e5["evidence_refs"]

    category2_current = sum(
        row["scope"] == "current_registry" and row["primary_category"] == 2 for row in rows
    )
    assert category2_current == 39


def test_p5_smoke_refusal_parser_fails_closed_on_semantic_drift() -> None:
    receipt = json.loads((REPO_ROOT / P5_SMOKE_RECEIPT_PATH).read_text())
    summary = _p5_smoke_refusal_summary(receipt)
    assert summary["decision_count"] == 3
    assert summary["power_source"] == "AC Power"
    assert summary["minimum_projected_disk_gb"] > 40.0

    mutations = []
    changed_status = copy.deepcopy(receipt)
    changed_status["status"] = "complete"
    mutations.append(changed_status)
    executed = copy.deepcopy(receipt)
    executed["command_executed"] = True
    mutations.append(executed)
    admitted = copy.deepcopy(receipt)
    admitted["admission"]["allowed"] = True
    mutations.append(admitted)
    mixed_failure = copy.deepcopy(receipt)
    next(gate for gate in mixed_failure["decisions"][0]["gates"] if gate["name"] == "power")["ok"] = False
    mutations.append(mixed_failure)
    stale_policy = copy.deepcopy(receipt)
    stale_policy["policy"]["sha256"] = "0" * 64
    mutations.append(stale_policy)
    stale_run = copy.deepcopy(receipt)
    stale_run["run_id"] = "p5smoke_20000101_leg99"
    mutations.append(stale_run)
    fabricated_limit = copy.deepcopy(receipt)
    for decision in fabricated_limit["decisions"]:
        next(gate for gate in decision["gates"] if gate["name"] == "candidate_memory_headroom")["limit"] = (
            11.0
        )
    mutations.append(fabricated_limit)
    fabricated_observation = copy.deepcopy(receipt)
    for decision in fabricated_observation["decisions"]:
        next(gate for gate in decision["gates"] if gate["name"] == "candidate_memory_headroom")[
            "observed"
        ] = 1.0
    mutations.append(fabricated_observation)

    for mutation in mutations:
        with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
            _p5_smoke_refusal_summary(mutation)


def test_p6_dry_receipt_accepts_only_exact_missing_p5_refusal(tmp_path) -> None:
    task = {
        "prerequisites": [
            {
                "path": P5_VERIFICATION_PATH,
                "schema": P5_VERIFICATION_SCHEMA,
                "fields": list(P5_VERIFICATION_FIELDS.items()),
            }
        ]
    }
    reason = "P6 and other dependent tasks fail closed until immutable prior receipts pass"
    decisions = [
        {
            "denied_reasons": [reason],
            "gates": [
                {
                    "name": "receipt_prerequisites",
                    "ok": False,
                    "reason": reason,
                    "observed": [
                        {
                            "path": P5_VERIFICATION_PATH,
                            "all_ok": False,
                            "schema": None,
                            "sha256": None,
                            "governor_provenance": None,
                            "problems": ["receipt is missing"],
                        }
                    ],
                }
            ],
        }
        for _ in range(3)
    ]

    assert _p6_dry_prerequisite_state(task, decisions, evidence_root=tmp_path) == (
        True,
        "missing-sealed-p5-verifier",
    )
    for decision in decisions:
        decision["denied_reasons"].append("battery power blocks compute-heavy admission")
        decision["gates"].append(
            {
                "name": "power",
                "ok": False,
                "reason": "battery power blocks compute-heavy admission",
            }
        )
    assert _p6_dry_prerequisite_state(task, decisions, evidence_root=tmp_path) == (
        True,
        "missing-sealed-p5-verifier",
    )
    decisions[0]["denied_reasons"].append("unbacked denial")
    assert _p6_dry_prerequisite_state(task, decisions, evidence_root=tmp_path) == (
        False,
        "missing-prerequisite-refusal-drift",
    )
    decisions[0]["denied_reasons"].pop()
    task["prerequisites"][0]["fields"] = [
        pair for pair in task["prerequisites"][0]["fields"] if pair[0] != "problems"
    ]
    assert _p6_dry_prerequisite_state(task, decisions, evidence_root=tmp_path) == (
        False,
        "task-prerequisite-contract-drift",
    )
