"""Contract and mutation tests for the durable MOP potential-atlas validator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from mop.studies.p5_terminal_evidence import (
    P5_SMOKE_RECEIPT_PATH,
    P5_TERMINAL_EVIDENCE_PATHS,
)
from mop.studies.potential_atlas_validation import (
    RECEIPT_SCHEMA,
    _parse_p5_terminal_evidence,
    main,
    refresh_source_hashes,
    validate_potential_atlas,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED_ATLAS = ROOT / "proof" / "MOP_POTENTIAL_ATLAS.json"
MARKDOWN = ROOT / "MOP_POTENTIAL_ATLAS_2026_07.md"
REQUIREMENTS = ROOT / "proof" / "EXTENDED_COMPUTE_REQUIREMENTS.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reconcile_source_hashes(payload: dict[str, Any]) -> None:
    for row in payload["source_snapshot"]:
        row["sha256"] = _sha256(ROOT / row["path"])


@pytest.fixture
def canonical_atlas(tmp_path: Path) -> Path:
    """Copy the atlas with current source hashes so concurrent doc integration is isolated."""
    payload = _load(COMMITTED_ATLAS)
    _reconcile_source_hashes(payload)
    path = tmp_path / "atlas.json"
    _write(path, payload)
    return path


def _validate(path: Path) -> dict[str, Any]:
    return validate_potential_atlas(
        path,
        repo_root=ROOT,
        requirements_path=REQUIREMENTS,
        markdown_path=MARKDOWN,
    )


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(check for check in report["checks"] if check["name"] == name)


def test_canonical_contract_recomputes_every_fixed_total(canonical_atlas: Path) -> None:
    report = _validate(canonical_atlas)
    assert report["all_ok"], report["problems"]
    assert report["schema"] == RECEIPT_SCHEMA
    assert report["summary"] == {
        "facet_count": 41,
        "facet_weight_total": 100.0,
        "weighted_score": pytest.approx(5.736),
        "domain_count": 7,
        "source_count": 66,
        "evidence_reference_count": 204,
        "requirements_row_count": 321,
        "primary_category_counts": {1: 168, 2: 119, 3: 29, 6: 5},
        "category2_count": 119,
        "category2_cluster_count": 8,
        "category2_current_registry_rows": 39,
        "registry_non_f_rows": 177,
        "scientific_claim_ready_rows": 0,
        "frontier_historical_rows": 24,
        "queue_count": 16,
        "facet_dependency_count": 114,
        "dense_integration_complete": True,
        "wave_e0_mechanics_pass": True,
        "p6_mechanics_pass": True,
        "p6_current_events_per_stream": 384,
        "p7_mechanics_pass": True,
        "p7_independent_units": 3,
        "p7_arm_count": 8,
        "p9_mechanics_pass": True,
        "p9_independent_units": 5,
        "p9_total_lineages": 260,
        "p9_total_branches": 1300,
        "p9_arm_count": 9,
        "p5_terminal_state": "governed-terminal-null",
        "p5_outcome": "null",
        "p5_smoke_run_id": "p5smoke_20260711_leg4",
        "p5_pilot_run_id": "p5pilot_20260712_leg6",
        "p5_fresh_challenge_run_id": "p5fresh_challenge_20260712_leg2",
        "p5_verifier_run_id": (
            "mac-studio-substrate-policy-transition-v1-p5verify_cpu-20260712T135308Z-leg01"
        ),
        "p5_primary_seed_count": 5,
        "p5_fresh_training_seed_count": 3,
        "p5_verified_pattern_count": 0,
        "p5_scientific_promotion": False,
        "studio_scale_required_now": False,
    }
    unsealed = dict(report)
    stored_digest = unsealed.pop("payload_sha256")
    canonical = json.dumps(
        unsealed,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    assert stored_digest == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "check_name", "problem_fragment"),
    [
        (lambda atlas: atlas["facets"].pop(), "facet_contract", "expected 41 facets"),
        (
            lambda atlas: atlas["facets"][0].__setitem__("weight", 4),
            "facet_contract",
            "weights sum to 101",
        ),
        (
            lambda atlas: atlas["scoring"].__setitem__("raw_formula", "changed"),
            "score_contract",
            "formula has drifted",
        ),
        (
            lambda atlas: atlas["scoring"].__setitem__("bottleneck_caps", []),
            "score_contract",
            "bottleneck caps have drifted",
        ),
        (
            lambda atlas: atlas["facets"][0]["scores"].__setitem__("raw", 0),
            "score_contract",
            "raw score",
        ),
        (
            lambda atlas: atlas["facets"][0]["scores"].__setitem__("overall", 10),
            "score_contract",
            "capped score",
        ),
    ],
)
def test_facet_weight_formula_and_cap_drift_fail_closed(
    canonical_atlas: Path,
    mutation: Any,
    check_name: str,
    problem_fragment: str,
) -> None:
    payload = _load(canonical_atlas)
    mutation(payload)
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert not report["all_ok"]
    assert any(problem_fragment in problem for problem in _check(report, check_name)["problems"])


def test_source_hash_and_evidence_path_drift_are_independent_failures(
    canonical_atlas: Path,
) -> None:
    payload = _load(canonical_atlas)
    payload["source_snapshot"][0]["sha256"] = "0" * 64
    payload["facets"][0]["evidence"][0] = "proof/DOES_NOT_EXIST.json"
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert not _check(report, "source_snapshot")["ok"]
    assert "source hash drift" in " ".join(_check(report, "source_snapshot")["problems"])
    assert not _check(report, "evidence_paths")["ok"]
    assert "does not exist" in " ".join(_check(report, "evidence_paths")["problems"])


def test_retired_scale_instruments_cannot_reenter_current_evidence(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    retired = "proof/VJEPA_SCALE_ATLAS_LOCAL.json"
    payload["source_snapshot"].append({"path": retired, "sha256": _sha256(ROOT / retired)})
    payload["facets"][0]["evidence"].append(retired)
    payload["retired_historical_evidence"]["current_dependency"] = True
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert not report["all_ok"]
    assert "retired sources" in " ".join(_check(report, "source_snapshot")["problems"])
    assert "retired current-path evidence" in " ".join(_check(report, "evidence_paths")["problems"])
    assert "current_dependency" in " ".join(_check(report, "facet_contract")["problems"])


def test_retired_p5_preflight_cannot_reenter_current_evidence(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    retired = "proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json"
    payload["source_snapshot"].append({"path": retired, "sha256": _sha256(ROOT / retired)})
    op3 = next(facet for facet in payload["facets"] if facet["id"] == "OP3")
    op3["evidence"].append(retired)
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert not report["all_ok"]
    assert "retired sources" in " ".join(_check(report, "source_snapshot")["problems"])
    assert "retired current-path evidence" in " ".join(_check(report, "evidence_paths")["problems"])


def test_p5_terminal_summary_and_op3_narrative_are_evidence_derived(
    canonical_atlas: Path,
) -> None:
    payload = _load(canonical_atlas)
    payload["p5_terminal_evidence"]["state"] = "governed-favorable"
    payload["p5_terminal_evidence"]["outcome"] = "favorable-programmatic-only"
    op3 = next(facet for facet in payload["facets"] if facet["id"] == "OP3")
    index = next(
        index
        for index, value in enumerate(op3["readiness_not_capability"])
        if value.startswith("P5 final verifier returned null")
    )
    op3["readiness_not_capability"][index] = "P5 produced a favorable capability result"
    op3["local_to_10"][0] = "rerun P5 before any P6 work"
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "p5_terminal_evidence")["problems"])
    assert not report["all_ok"]
    assert "summary disagrees" in problems
    assert "narrative disagrees" in problems
    assert "local path" in problems


def test_p5_terminal_parser_joins_every_governed_stage_and_rejects_splices() -> None:
    documents = {path: _load(ROOT / path) for path in P5_TERMINAL_EVIDENCE_PATHS}
    problems, summary = _parse_p5_terminal_evidence(documents, ROOT)
    assert not problems
    assert summary["state"] == "governed-terminal-null"
    assert summary["fresh_training_seeds"] == [5101, 5102, 5103]
    assert summary["verified_pattern_count"] == 0

    stale_run = json.loads(json.dumps(documents))
    stale_run[P5_SMOKE_RECEIPT_PATH]["run_id"] = "p5smoke_20260711_leg3"
    assert _parse_p5_terminal_evidence(stale_run, ROOT)[0]

    verifier_promotion = json.loads(json.dumps(documents))
    verifier_path = "proof/P5_CONTEXT_CAPABILITY_VERIFICATION.json"
    verifier_promotion[verifier_path]["classification"] = "favorable-programmatic-only"
    assert _parse_p5_terminal_evidence(verifier_promotion, ROOT)[0]

    cross_hash_splice = json.loads(json.dumps(documents))
    fresh_path = "proof/P5_CONTEXT_CAPABILITY_FRESH_CHALLENGE.json"
    cross_hash_splice[fresh_path]["primary_receipt"]["sha256"] = "0" * 64
    assert _parse_p5_terminal_evidence(cross_hash_splice, ROOT)[0]


def test_p5_requirements_cannot_restore_retired_preflight(canonical_atlas: Path, tmp_path: Path) -> None:
    requirements = _load(REQUIREMENTS)
    p5 = next(row for row in requirements["rows"] if row["id"] == "mop_p5_context_capability")
    p5["evidence_refs"].append("local:proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json")
    mutated_requirements = tmp_path / "requirements.json"
    _write(mutated_requirements, requirements)
    report = validate_potential_atlas(
        canonical_atlas,
        repo_root=ROOT,
        requirements_path=mutated_requirements,
        markdown_path=MARKDOWN,
    )
    problems = " ".join(_check(report, "p5_terminal_evidence")["problems"])
    assert not report["all_ok"]
    assert "bind the exact terminal chain" in problems


def test_category2_partition_requires_the_exact_member_set(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    cluster = payload["category2_harness_clusters"]["clusters"][0]
    removed = cluster["members"].pop()
    cluster["count"] -= 1
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "category2_partition")["problems"])
    assert not report["all_ok"]
    assert removed in problems
    assert "member set drift" in problems


def test_e6_dr14_reclassification_is_bound_to_category_three(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["category2_harness_clusters"]["reclassified_after_local_integration"]["to_category"] = 2
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "category2_partition")["problems"])
    assert not report["all_ok"]
    assert "2 to 3" in problems


def test_p7_reclassification_is_bound_to_category_three(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["category2_harness_clusters"]["p7_reclassified_after_local_integration"]["to_category"] = 2
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "category2_partition")["problems"])
    assert not report["all_ok"]
    assert "P7 action-world-model category transition is not declared as 2 to 3" in problems


def test_p9_reclassification_is_bound_to_category_three(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["category2_harness_clusters"]["p9_reclassified_after_local_integration"]["to_category"] = 2
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "category2_partition")["problems"])
    assert not report["all_ok"]
    assert "P9 monitoring/accounting category transition is not declared as 2 to 3" in problems


def test_registry_and_requirements_accounting_summaries_are_bound(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["portfolio"]["current_registry_summary"]["rights_data_blocked"] = 9
    payload["portfolio"]["requirements_summary"]["category2_current_registry_rows"] = 12
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "snapshot_accounting")["problems"])
    assert not report["all_ok"]
    assert "rights_data_blocked" in problems
    assert "requirements_summary" in problems


def test_mechanics_progress_summaries_fail_closed_on_promotion_drift(
    canonical_atlas: Path,
) -> None:
    payload = _load(canonical_atlas)
    payload["dense_task_integration"]["scientific_promotion"] = True
    payload["smallest_reusable_local_wave"]["acceptance_satisfied"] = False
    payload["continual_million_event_preflight"]["progressive_rungs"] = [1000000]
    payload["continual_million_event_preflight"]["scheduler_preflight"]["command_executed"] = True
    payload["action_world_model_preflight"]["scientific_promotion"] = True
    payload["causal_monitoring_accounting_preflight"]["scientific_promotion"] = True
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "mechanics_progress")["problems"])
    assert not report["all_ok"]
    assert "scientific_promotion" in problems
    assert "Wave E0 acceptance" in problems
    assert "P6 field progressive_rungs" in problems
    assert "P6 scheduler field command_executed" in problems
    assert "P7 field scientific_promotion" in problems
    assert "P9 field scientific_promotion" in problems


def test_queue_and_dependency_references_must_resolve(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["highest_leverage_local_queue"][0]["facets"].append("UNKNOWN_FACET")
    payload["facets"][0]["dependencies"].append("UNKNOWN_DEPENDENCY")
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert "UNKNOWN_FACET" in " ".join(_check(report, "queue_contract")["problems"])
    assert "UNKNOWN_DEPENDENCY" in " ".join(_check(report, "dependency_contract")["problems"])


def test_studio_gate_cannot_be_earned_without_bound_source_agreement(
    canonical_atlas: Path,
) -> None:
    payload = _load(canonical_atlas)
    payload["studio_escalation"]["earned_now"] = True
    payload["portfolio"]["hardware_boundary"]["studio_scale_required_now"] = True
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    problems = " ".join(_check(report, "studio_gate_consistency")["problems"])
    assert not report["all_ok"]
    assert "required now" in problems
    assert "fail-closed false" in problems


def test_malformed_nested_data_returns_a_failure_receipt(canonical_atlas: Path) -> None:
    payload = _load(canonical_atlas)
    payload["dependency_graph"]["edges"] = [[[]]]
    _write(canonical_atlas, payload)
    report = _validate(canonical_atlas)
    assert not report["all_ok"]
    assert report["payload_sha256"]
    assert _check(report, "dependency_contract")["problems"]


def test_default_cli_writes_a_sealed_receipt_without_touching_atlas(
    canonical_atlas: Path, tmp_path: Path
) -> None:
    before = canonical_atlas.read_bytes()
    out = tmp_path / "receipt.json"
    status = main(
        [
            "--repo-root",
            str(ROOT),
            "--atlas",
            str(canonical_atlas),
            "--markdown",
            str(MARKDOWN),
            "--requirements",
            str(REQUIREMENTS),
            "--out",
            str(out),
        ]
    )
    assert status == 0
    assert canonical_atlas.read_bytes() == before
    receipt = _load(out)
    assert receipt["all_ok"] is True
    assert receipt["schema"] == RECEIPT_SCHEMA


def test_refresh_cli_changes_only_stale_source_hash_fields(canonical_atlas: Path, tmp_path: Path) -> None:
    payload = _load(canonical_atlas)
    source = payload["source_snapshot"][0]
    source["sha256"] = "0" * 64
    _write(canonical_atlas, payload)
    before = _load(canonical_atlas)
    before_text = canonical_atlas.read_text(encoding="utf-8")
    out = tmp_path / "refresh-receipt.json"
    status = main(
        [
            "--repo-root",
            str(ROOT),
            "--atlas",
            str(canonical_atlas),
            "--markdown",
            str(MARKDOWN),
            "--requirements",
            str(REQUIREMENTS),
            "--refresh-source-hashes",
            "--out",
            str(out),
        ]
    )
    assert status == 0
    after = _load(canonical_atlas)
    actual = _sha256(ROOT / source["path"])
    before["source_snapshot"][0]["sha256"] = actual
    assert after == before
    after_text = canonical_atlas.read_text(encoding="utf-8")
    assert after_text.replace(actual, "0" * 64, 1) == before_text
    receipt = _load(out)
    assert receipt["source_hash_refresh"]["published"] is True
    assert receipt["source_hash_refresh"]["changed_count"] == 1


def test_refresh_refuses_semantically_invalid_candidate_without_writing(
    canonical_atlas: Path,
) -> None:
    payload = _load(canonical_atlas)
    payload["source_snapshot"][0]["sha256"] = "0" * 64
    payload["highest_leverage_local_queue"][0]["facets"].append("UNKNOWN_FACET")
    _write(canonical_atlas, payload)
    before = canonical_atlas.read_bytes()
    receipt = refresh_source_hashes(
        canonical_atlas,
        repo_root=ROOT,
        requirements_path=REQUIREMENTS,
        markdown_path=MARKDOWN,
    )
    assert not receipt["all_ok"]
    assert receipt["source_hash_refresh"]["published"] is False
    assert "candidate failed full validation" in " ".join(receipt["problems"])
    assert canonical_atlas.read_bytes() == before
