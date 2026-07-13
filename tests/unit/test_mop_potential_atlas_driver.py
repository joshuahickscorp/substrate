from __future__ import annotations

import copy
import json

import pytest

from mop.config import REPO_ROOT
from mop.studies.potential_atlas_driver import (
    P5_SMOKE_RECEIPT_PATH,
    P5_TERMINAL_EVIDENCE_PATHS,
    SERVED_SCAFFOLD_FACETS,
    _p5_terminal_evidence_summary,
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
        "P5 final verifier returned null" in value
        and "verified 0 of 4 pilot patterns" in value
        and "grants no scientific or confirmatory promotion" in value
        for value in op3["readiness_not_capability"]
    )
    terminal = atlas["p5_terminal_evidence"]
    assert "p5_local_admission" not in atlas
    assert terminal["state"] == "governed-terminal-null"
    assert terminal["outcome"] == "null"
    assert terminal["primary_outcome"] == "favorable-programmatic-only"
    assert terminal["fresh_training_seeds"] == [5101, 5102, 5103]
    assert terminal["verified_pattern_count"] == 0
    assert terminal["scientific_promotion"] is False
    assert (
        sum(value.startswith("P5 final verifier returned null") for value in op3["readiness_not_capability"])
        == 1
    )
    assert set(P5_TERMINAL_EVIDENCE_PATHS).issubset(op3["evidence"])
    assert "proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json" not in op3["evidence"]
    assert "proof/LOCAL_THROTTLE_P5_SMOKE_PREFLIGHT.json" not in {
        row["path"] for row in atlas["source_snapshot"]
    }
    p6 = atlas["continual_million_event_preflight"]
    assert p6["scheduler_preflight"]["admission_allowed"] is False
    assert "P5 now supplies the required independent null" in p6["scheduler_preflight"]["interpretation"]
    assert "live resource gates" in p6["scheduler_preflight"]["interpretation"]
    assert not any("release the exclusive lane after P4" in value for value in p6["remaining"])
    assert not any("verify the P5 sequence" in value for value in p6["remaining"])
    assert any("admit the P6 10k resource probe" in value for value in p6["remaining"])


def test_atlas_p5_terminal_parser_rejects_stale_inputs() -> None:
    receipt = json.loads((REPO_ROOT / P5_SMOKE_RECEIPT_PATH).read_text())
    summary = _p5_terminal_evidence_summary(receipt)
    assert summary["state"] == "governed-terminal-null"
    assert summary["governor_receipts"]["smoke"]["run_id"] == "p5smoke_20260711_leg4"
    assert summary["governor_receipts"]["pilot"]["run_id"] == "p5pilot_20260712_leg6"

    for key, replacement in (
        ("mode", "execute-refused"),
        ("status", "admission-refused"),
        ("command_executed", False),
    ):
        mutation = copy.deepcopy(receipt)
        mutation[key] = replacement
        with pytest.raises(ValueError, match="invalid terminal P5 evidence"):
            _p5_terminal_evidence_summary(mutation)

    mutation = copy.deepcopy(receipt)
    mutation["completion_authority"]["output"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="invalid terminal P5 evidence"):
        _p5_terminal_evidence_summary(mutation)

    stale_run = copy.deepcopy(receipt)
    stale_run["run_id"] = "p5smoke_20260711_leg3"
    with pytest.raises(ValueError, match="invalid terminal P5 evidence"):
        _p5_terminal_evidence_summary(stale_run)


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
