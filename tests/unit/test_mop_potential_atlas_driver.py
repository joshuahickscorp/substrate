from __future__ import annotations

import copy
import json

import pytest

from mop.config import REPO_ROOT
from mop.studies.potential_atlas_driver import (
    P5_SMOKE_RECEIPT_PATH,
    SERVED_SCAFFOLD_FACETS,
    _p5_smoke_refusal_summary,
    build_atlas,
    render_markdown,
)


def _inputs() -> tuple[dict, dict]:
    atlas = json.loads((REPO_ROOT / "proof/MOP_POTENTIAL_ATLAS.json").read_text())
    requirements = json.loads((REPO_ROOT / "proof/EXTENDED_COMPUTE_REQUIREMENTS.json").read_text())
    return atlas, requirements


def test_semantic_driver_adds_low_scored_gaps_and_rescores_served_scaffolds() -> None:
    base, requirements = _inputs()
    atlas = build_atlas(base, requirements)
    facets = {row["id"]: row for row in atlas["facets"]}

    assert len(facets) == 41
    assert sum(row["weight"] for row in facets.values()) == 100
    assert all(facets[facet_id]["scores"]["scaffolding"] == 8 for facet_id in SERVED_SCAFFOLD_FACETS)
    for facet_id in ("EV6", "OP5", "SG4", "SG5"):
        assert facets[facet_id]["scores"] == {
            "scaffolding": 2,
            "implementation": 0,
            "experiment": 0,
            "confirmation": 0,
            "raw": 0.4,
            "overall": 0.4,
        }

    assert "broadcast" in facets["PA6"]["title"].lower()
    assert "recurrent-processing" in facets["PA6"]["title"].lower()
    assert "evaluator integrity" in facets["SG1"]["title"].lower()
    assert "evaluator" not in facets["SG2"]["title"].lower()
    exhaustion = json.loads((REPO_ROOT / "proof/PROJECT_EXPERIMENT_EXHAUSTION.json").read_text())
    counts = exhaustion["coverage"]["classification_counts"]
    summary = atlas["portfolio"]["current_registry_summary"]
    assert summary["freshly_executed_verified"] == counts["freshly-executed-verified"]
    assert summary["runnable_not_yet_run"] == counts["runnable-not-yet-run"]


def test_category2_partition_and_driver_are_exactly_idempotent() -> None:
    base, requirements = _inputs()
    first = build_atlas(base, requirements)
    second = build_atlas(first, requirements)
    assert second == first

    block = first["category2_harness_clusters"]
    assert block["category2_row_count"] == 119
    assert block["scope_counts"]["current_registry"] == 39
    assert {row["id"]: row["count"] for row in block["clusters"]} == {
        "H1_temporal_binding_acquisition": 16,
        "H2_action_boundary_world_model": 16,
        "H3_memory_workspace_self_model": 18,
        "H4_lifetime_plasticity_openended": 23,
        "H5_social_reference_culture": 16,
        "H6_transactional_safety_material": 19,
        "H7_dense_substrate_controls_search": 10,
        "H8_execution_density": 1,
    }


def test_operational_revision_removes_stale_p4_and_post_p4_claims() -> None:
    base, requirements = _inputs()
    atlas = build_atlas(base, requirements)
    facets = {row["id"]: row for row in atlas["facets"]}
    op2 = facets["OP2"]
    op3 = facets["OP3"]

    assert "partial five-seed P4 execution" not in op2["demonstrated_components"]
    assert "full five-seed P4 response surface is incomplete" not in op2["readiness_not_capability"]
    assert not any("partial P4" in value for value in op2["local_to_10"])
    assert any("completed 12-cell" in value for value in op2["demonstrated_components"])
    assert not any("refused concurrent admission" in value for value in op3["demonstrated_components"])
    assert not any("post-P4 P6 admission" in value for value in op3["readiness_not_capability"])
    assert any(
        "normalized one-minute load 3.862 to 3.987 against 0.85" in value
        and "6.830 to 8.909 GB" in value
        and "AC power passed" in value
        for value in op3["readiness_not_capability"]
    )
    admission = atlas["p5_local_admission"]
    assert admission["state"] == "cpu-load-and-memory-admission-refusal"
    assert admission["failed_gates"] == ["cpu_load", "candidate_memory_headroom"]
    assert admission["cpu_load_per_logical_cpu"] == [
        3.8621012369791665,
        3.8621012369791665,
        3.986572265625,
    ]
    assert admission["maximum_cpu_load_per_logical_cpu"] == 0.85
    assert admission["available_memory_gb"] == [6.830424064, 8.909111296, 8.705048576]
    assert admission["required_memory_gb"] == 10.0
    assert admission["command_executed"] is False
    assert sum(value.startswith("P5 smoke is fail-closed") for value in op3["readiness_not_capability"]) == 1
    assert "proof/LOCAL_THROTTLE_P5_SMOKE_RUN.json" in op3["evidence"]
    assert "proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json" not in op3["evidence"]
    assert "proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json" not in {
        row["path"] for row in atlas["source_snapshot"]
    }
    p6 = atlas["continual_million_event_preflight"]
    assert p6["scheduler_preflight"]["admission_allowed"] is False
    assert "post-P4" in p6["scheduler_preflight"]["interpretation"]
    assert "independent null or favorable verification" in p6["scheduler_preflight"]["interpretation"]
    assert not any("release the exclusive lane after P4" in value for value in p6["remaining"])
    assert any("verify the P5 sequence" in value for value in p6["remaining"])


def test_atlas_p5_refusal_parser_rejects_stale_narrative_inputs() -> None:
    receipt = json.loads((REPO_ROOT / P5_SMOKE_RECEIPT_PATH).read_text())
    summary = _p5_smoke_refusal_summary(receipt)
    assert min(summary["cpu_load_per_logical_cpu"]) == 3.8621012369791665
    assert max(summary["cpu_load_per_logical_cpu"]) == 3.986572265625
    assert min(summary["available_memory_gb"]) == 6.830424064
    assert max(summary["available_memory_gb"]) == 8.909111296

    for key, replacement in (
        ("mode", "execute"),
        ("status", "complete"),
        ("command_executed", True),
    ):
        mutation = copy.deepcopy(receipt)
        mutation[key] = replacement
        with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
            _p5_smoke_refusal_summary(mutation)

    mutation = copy.deepcopy(receipt)
    memory_gate = next(
        gate for gate in mutation["decisions"][0]["gates"] if gate["name"] == "candidate_memory_headroom"
    )
    memory_gate["observed"] = memory_gate["limit"]
    with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
        _p5_smoke_refusal_summary(mutation)

    stale_run = copy.deepcopy(receipt)
    stale_run["run_id"] = "p5smoke_20260710_leg2"
    with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
        _p5_smoke_refusal_summary(stale_run)

    fabricated_limit = copy.deepcopy(receipt)
    for decision in fabricated_limit["decisions"]:
        next(gate for gate in decision["gates"] if gate["name"] == "candidate_memory_headroom")["limit"] = (
            11.0
        )
    with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
        _p5_smoke_refusal_summary(fabricated_limit)

    fabricated_observation = copy.deepcopy(receipt)
    for decision in fabricated_observation["decisions"]:
        next(gate for gate in decision["gates"] if gate["name"] == "candidate_memory_headroom")[
            "observed"
        ] = 1.0
    with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
        _p5_smoke_refusal_summary(fabricated_observation)

    fabricated_cpu_observation = copy.deepcopy(receipt)
    for decision in fabricated_cpu_observation["decisions"]:
        next(gate for gate in decision["gates"] if gate["name"] == "cpu_load")["observed"] = 1.0
    with pytest.raises(ValueError, match="invalid P5 smoke admission refusal"):
        _p5_smoke_refusal_summary(fabricated_cpu_observation)


def test_executed_toy_receipts_are_bound_without_physical_or_capability_score_inflation() -> None:
    base, requirements = _inputs()
    atlas = build_atlas(base, requirements)
    facets = {row["id"]: row for row in atlas["facets"]}

    assert facets["PA6"]["scores"] == {
        "scaffolding": 8,
        "implementation": 6,
        "experiment": 3,
        "confirmation": 3,
        "raw": 4.75,
        "overall": 4.5,
    }
    assert "proof/INTEGRATION_BROADCAST_VERIFICATION.json" in facets["PA6"]["evidence"]
    assert "proof/SENSING_SCAFFOLD_VERIFICATION.json" in facets["SR3"]["evidence"]
    assert "proof/F59_F60_INTEGRITY_VERIFICATION.json" in facets["SG2"]["evidence"]
    assert "proof/F61_F64_MATERIAL_TWIN_VERIFICATION.json" in facets["BM4"]["evidence"]
    assert "proof/F22_F28_F50_F58_ECOLOGY_VERIFICATION.json" in facets["PA8"]["evidence"]
    assert facets["BM4"]["scores"]["overall"] == 2.0
    assert any("digital twins only" in value for value in facets["BM4"]["readiness_not_capability"])


def test_markdown_renderer_is_ascii_and_has_one_score_row_per_facet() -> None:
    base, requirements = _inputs()
    atlas = build_atlas(base, requirements)
    rendered = render_markdown(atlas)
    rendered.encode("ascii")
    assert "proof/MOP_POTENTIAL_ATLAS.json" in rendered
    for facet in atlas["facets"]:
        assert sum(line.startswith(f"| {facet['id']} |") for line in rendered.splitlines()) == 1
