"""The pre-substrate phenomena and mechanism atlas generator.

Produces a machine-readable atlas distinct from, but linked to, the potential/readiness atlas. The potential
atlas asks how complete the project is; this one asks what phenomena a substrate must explain and what
mechanism evidence exists. Every row points to a falsifiable experiment family and carries the full required
field set (estimand, null, unit, controls, metrics, SESOI, power rule, cost model, required inputs, rights
state, verifier strategy, alternative explanation, dependencies, follow-up rule, substrate relevance), and
every row is derived from a real campaign node (a runner with an execution path, or a contracted external
family with a named blocker), never a title-only concept.

Writes ``registry/phenomena.yaml``, ``registry/mechanism_candidates.yaml``, and
``proof/PRE_SUBSTRATE_PHENOMENA_ATLAS.json``.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .manifest import build_campaign
from .runners import entrypoint_is_runnable
from .specs import CampaignSpec, NodeSpec

_REPO = Path(__file__).resolve().parents[3]

# The candidate primitives the discovery phase is trying to shrink, merge, or kill (mandate file 02).
MECHANISM_CANDIDATES = [
    "immutable_referent_binder", "typed_event_graph", "dense_spatial_temporal_state",
    "object_entity_persistence", "relation_role_state", "predictive_transition_state",
    "counterfactual_branch_state", "uncertainty_calibration_state", "working_state",
    "episodic_memory", "semantic_compressed_memory", "cross_form_retrieval",
    "provenance_revision_state", "replay_consolidation", "plasticity_controller",
    "structural_growth_pruning", "value_of_computation_estimator", "verification_contradiction_repair",
    "action_affordance_interface", "operational_state_monitor", "resource_allocator",
    "cross_form_translator", "perspective_arbitration", "communication_protocol",
    "developmental_curriculum_controller",
]

# The ten-plus cognitive phenomena families the campaign must span (mandate file 02).
PHENOMENA_CATALOGUE = [
    ("temporal_boundary_event_formation", "native_audio", "when does an event begin and end"),
    ("persistence_through_occlusion", "vision_synthetic", "does object identity survive occlusion/transform"),
    ("split_merge_identity", "vision_synthetic", "identity through split, merge, and reappearance"),
    ("source_object_tracking", "native_audio", "how many sources and which is which over time"),
    ("cross_form_conditional_information", "audiovisual", "does one form add information beyond another"),
    ("compositional_binding", "symbolic", "held-out compositional combinations and role/filler binding"),
    ("predictive_state", "action", "usable action-conditioned transition state"),
    ("causal_intervention", "action", "ranking and consequence of interventions"),
    ("continual_learning", "memory_episode", "retention versus future-learning under change"),
    ("plasticity", "memory_episode", "fast state vs parameter vs structural adaptation"),
    ("episodic_memory", "memory_episode", "specific-episode retrieval vs semantic compression"),
    ("memory_revision_deletion", "memory_episode", "revision, deletion completeness, stale-memory harm"),
    ("routing_value_of_computation", "perspective", "competence-conditioned dispatch and marginal value"),
    ("marginal_value_of_computation", "host_telemetry", "when more compute or verification helps"),
    ("uncertainty_abstention", "events", "calibrated abstention on ambiguity"),
    ("alternate_compute_dynamics", "material_sim", "reservoir/fading-memory vs conventional control"),
]


def _controls_for(mechanism: str) -> dict[str, Any]:
    """The standard control battery a row asserts, specialized lightly by mechanism family."""

    base = {
        "strongest_simple_baseline": "tuned best-single / fixed-guess baseline",
        "strongest_inherited_baseline": "frozen inherited encoder plus a larger shell (matched compute)",
        "matched_capacity_owned_baseline": "owned module at matched parameter count",
        "matched_compute_baseline": "matched full-lifecycle FLOPs including training",
        "shuffled_referent_control": "shuffled-referent / shuffled-lineage",
        "wrong_event_wrong_time_controls": "wrong-event and shifted-time",
    }
    if mechanism in ("world_model", "transition_state"):
        base["action_shuffle_controls"] = "action-shuffled and equal-depth unrolled"
    if mechanism in ("replay", "episodic_retrieval", "provenance_state"):
        base["memory_controls"] = "no-memory, stale-memory, poisoned-memory, deletion, restart"
    if mechanism in ("value_of_computation", "count_estimator"):
        base["noisy_tv_control"] = "high-variance uninformative noisy-TV signal that must be rejected"
    return base


def _node_row(node: NodeSpec) -> dict[str, Any]:
    cov = node.coverage
    runnable = (not node.is_blocked) and entrypoint_is_runnable(node.entrypoint)
    return {
        "row_id": node.node_id,
        "title": node.title,
        "domain": cov.form_family,
        "form_families": [cov.form_family],
        "referent_type": cov.unit_class,
        "event_structure": cov.phenomenon,
        "independent_unit": cov.unit_class,
        "target_capability": cov.phenomenon,
        "candidate_mechanisms": [cov.mechanism_family],
        **_controls_for(cov.mechanism_family),
        "capability_metric": "task score at the clip/session/world unit (clip-macro where applicable)",
        "cost_metric": "full-lifecycle FLOPs (training charged to the candidate)",
        "density_metric": "capability per unit of compute and per active byte",
        "sesoi": "small structural effect declared before scores are read",
        "power_or_futility": "exact sign-flip over independent units; futility stop when decision is fixed",
        "current_evidence_level": cov.evidence_level,
        "current_blocker": node.blocked_reason or "none (runnable)",
        "reusable_harness": "mop.campaign.nodes.framework",
        "cross_domain_replication_targets": "a second form family and fresh independent units",
        "verifier_strategy": "an independent verifier node re-derives the verdict, not the producer path",
        "alternative_explanation": "a named non-mechanistic explanation the control removes",
        "dependencies": [d.node_id for d in node.dependencies],
        "follow_up_rule": "precommitted: survives -> reproduce cross-domain; null -> diagnose and replace",
        "substrate_consequence_if_positive": f"admits {cov.mechanism_family} as a candidate primitive",
        "substrate_consequence_if_null": "closes or redesigns the candidate; feeds negative-space synthesis",
        "execution": "runnable" if runnable else "contracted_blocked",
        "entrypoint": node.entrypoint if runnable else None,
    }


def build_atlas(campaign: CampaignSpec | None = None) -> dict[str, Any]:
    campaign = campaign or build_campaign()
    rows = [_node_row(n) for n in campaign.nodes]
    runnable = [r for r in rows if r["execution"] == "runnable"]
    blocked = [r for r in rows if r["execution"] == "contracted_blocked"]
    forms = sorted({r["domain"] for r in rows if r["domain"] not in ("none", "cross")})
    phenomena = sorted({r["event_structure"] for r in rows if r["event_structure"] != "none"})
    return {
        "schema": "mop-pre-substrate-phenomena-atlas/v1",
        "campaign_id": campaign.campaign_id,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
        "n_rows": len(rows),
        "n_runnable_local": len(runnable),
        "n_contracted_external": len(blocked),
        "form_families": forms,
        "n_form_families": len(forms),
        "phenomena": phenomena,
        "n_phenomena": len(phenomena),
        "mechanism_candidates": MECHANISM_CANDIDATES,
        "n_mechanism_candidates": len(MECHANISM_CANDIDATES),
        "coverage_targets": campaign.coverage_targets,
        "targets_met": {
            "form_families": len(forms) >= campaign.coverage_targets.get("form_families", 6),
            "phenomena": len(phenomena) >= campaign.coverage_targets.get("phenomena", 10),
            "local_families": len(runnable) >= campaign.coverage_targets.get("local_question_families", 24),
            "external_families": len(blocked) >= campaign.coverage_targets.get("external_families", 16),
        },
        "rows": rows,
    }


def write_atlas(atlas: dict[str, Any] | None = None) -> dict[str, Path]:
    atlas = atlas or build_atlas()
    proof = _REPO / "proof" / "PRE_SUBSTRATE_PHENOMENA_ATLAS.json"
    proof.parent.mkdir(parents=True, exist_ok=True)
    proof.write_text(json.dumps(atlas, indent=1, sort_keys=True), encoding="utf-8")

    phen_rows = [{"phenomenon": p, "primary_form": form, "estimand": q,
                  "atlas_rows": [r["row_id"] for r in atlas["rows"] if r["event_structure"] == p]}
                 for p, form, q in PHENOMENA_CATALOGUE]
    phen_path = _REPO / "registry" / "phenomena.yaml"
    phen_path.parent.mkdir(parents=True, exist_ok=True)
    phen_path.write_text(yaml.safe_dump({"schema": "mop-phenomena/v1", "phenomena": phen_rows},
                                        sort_keys=False), encoding="utf-8")

    mech_path = _REPO / "registry" / "mechanism_candidates.yaml"
    mech_path.write_text(yaml.safe_dump(
        {"schema": "mop-mechanism-candidates/v1",
         "note": "hypotheses to shrink, merge, or kill; not presumed useful",
         "candidates": MECHANISM_CANDIDATES}, sort_keys=False), encoding="utf-8")
    return {"atlas": proof, "phenomena": phen_path, "mechanism_candidates": mech_path}
