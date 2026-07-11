from __future__ import annotations

from scripts.build_extended_compute_requirements import (
    P5_VERIFICATION_FIELDS,
    P5_VERIFICATION_PATH,
    P5_VERIFICATION_SCHEMA,
    _p6_dry_prerequisite_state,
    registry_rows,
)


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
    assert "local:proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json" in p5["evidence_refs"]

    e5 = by_id["e5_curiosity"]
    assert e5["primary_category"] == 1
    assert e5["classification_basis"] == "implemented-local-evidence-source-rerun-pending"
    assert "local:proof/LOCAL_ACTION_ENVIRONMENT.json" in e5["evidence_refs"]

    category2_current = sum(
        row["scope"] == "current_registry" and row["primary_category"] == 2 for row in rows
    )
    assert category2_current == 39


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
