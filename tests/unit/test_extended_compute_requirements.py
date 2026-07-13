from __future__ import annotations

import copy
import json

import pytest
from scripts.build_extended_compute_requirements import (
    P5_SMOKE_RECEIPT_PATH,
    P5_TERMINAL_EVIDENCE_PATHS,
    P5_VERIFICATION_FIELDS,
    P5_VERIFICATION_PATH,
    P5_VERIFICATION_SCHEMA,
    _p5_terminal_evidence_summary,
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
    assert p5["classification_basis"] == "governed-programmatic-terminal-null"
    assert p5["measured"]["present"] is True
    assert p5["measured"]["execution_verified_at_generation"] is True
    assert p5["measured"]["ledger_snapshot_superseded_by_terminal_evidence"] is True
    assert p5["primary_blocker"] == (
        "the registered deterministic-programmatic P5 sequence is closed by an independent null; "
        "natural-video and general-capability claims remain outside its validated scope"
    )
    assert {f"local:{path}" for path in P5_TERMINAL_EVIDENCE_PATHS}.issubset(p5["evidence_refs"])
    terminal = p5["measured"]["terminal_evidence"]
    assert terminal["state"] == "governed-terminal-null"
    assert terminal["outcome"] == "null"
    assert terminal["primary_outcome"] == "favorable-programmatic-only"
    assert terminal["primary_seed_count"] == 5
    assert terminal["fresh_training_seeds"] == [5101, 5102, 5103]
    assert terminal["verified_pattern_count"] == 0
    assert terminal["mutation_count"] == 23
    assert terminal["scientific_promotion"] is False

    e5 = by_id["e5_curiosity"]
    assert e5["primary_category"] == 1
    assert e5["classification_basis"] == "implemented-local-evidence-source-rerun-pending"
    assert "local:proof/LOCAL_ACTION_ENVIRONMENT.json" in e5["evidence_refs"]

    category2_current = sum(
        row["scope"] == "current_registry" and row["primary_category"] == 2 for row in rows
    )
    assert category2_current == 39


def test_p5_terminal_parser_fails_closed_on_semantic_drift() -> None:
    receipt = json.loads((REPO_ROOT / P5_SMOKE_RECEIPT_PATH).read_text())
    summary = _p5_terminal_evidence_summary(receipt)
    assert summary["state"] == "governed-terminal-null"
    assert summary["governor_receipts"]["smoke"]["run_id"] == "p5smoke_20260711_leg4"
    assert summary["artifacts"]["verifier"]["sha256"] == (
        "743ce07180f0728f3074b2ac9c78a9aa12ff23f33aeebeffd45bebecacb5f077"
    )

    mutations = []
    changed_status = copy.deepcopy(receipt)
    changed_status["status"] = "admission-refused"
    mutations.append(changed_status)
    executed = copy.deepcopy(receipt)
    executed["command_executed"] = False
    mutations.append(executed)
    admitted = copy.deepcopy(receipt)
    admitted["admission"]["allowed"] = False
    mutations.append(admitted)
    output_splice = copy.deepcopy(receipt)
    output_splice["completion_authority"]["output"]["sha256"] = "0" * 64
    mutations.append(output_splice)
    stale_policy = copy.deepcopy(receipt)
    stale_policy["policy"]["sha256"] = "0" * 64
    mutations.append(stale_policy)
    stale_run = copy.deepcopy(receipt)
    stale_run["run_id"] = "p5smoke_20260711_leg3"
    mutations.append(stale_run)

    for mutation in mutations:
        with pytest.raises(ValueError, match="invalid terminal P5 evidence"):
            _p5_terminal_evidence_summary(mutation)


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
