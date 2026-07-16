"""Append-only G1-D1 redesign scaffold after the frozen centroid prune.

The redesign keeps the completed centroid router as a permanently noneligible
control and opens three disjoint program categories: utility residuals,
pairwise rankings, and calibrated abstention.  This module contains no model
training.  It defines deterministic candidate evidence, screen, and freeze
artifacts that later execution programs can fill without changing the frozen
thresholds or heldout-input boundary.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from mop.studies import generation1_d1_frozen_verify as frozen

SCREEN_SCHEMA = "mop-generation1-d1-append-only-redesign-screen/v2"
EVIDENCE_SCHEMA = "mop-generation1-d1-redesign-candidate-evidence/v2"
FREEZE_SCHEMA = "mop-generation1-d1-redesign-freeze/v2"
PROGRAM_ID = "generation1-d1-append-only-redesign-v2"
CLAIM_SCOPE = (
    "append-only generated D1 mechanism redesign screening and categorized freeze; "
    "no confirmatory, activation, or scientific-promotion claim"
)

FAMILIES = ("utility-residual", "pairwise-ranking", "calibrated-abstaining")
PHASES = ("producer", "challenge")
STATIC_CONTROLS = ("global_static", "difficulty_static")
VISIBLE_INPUTS = (
    "latent_vector",
    "difficulty_index",
    "labeled_training_support_geometry",
)
FORBIDDEN_HELDOUT_INPUTS = (
    "context_id",
    "truth",
    "actor_predictions",
    "oracle_actor_id",
    "heldout_route_labels",
)
CRITERIA = dict(frozen.CRITERIA)

WINNER = "winner"
NO_CANDIDATE = "no_candidate"

LEGACY_CONTROL = {
    "candidate_id": frozen.FROZEN_VARIANT_ID,
    "family": "legacy-centroid",
    "role": "retired_noneligible_control_only",
    "eligible_for_screen": False,
    "eligible_for_freeze": False,
    "activation_allowed": False,
    "scientific_promotion": False,
}
APPEND_ONLY_POLICY = {
    "predecessor_evidence_classification_required": frozen.NULL_SAFE_PRUNE,
    "prior_centroid_result_mutated": False,
    "legacy_centroid_role": "retired_noneligible_control_only",
    "candidate_ids_are_new": True,
    "future_revisions_must_add_new_candidate_ids": True,
}

_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "utility-residual-linear-v2",
        "family": "utility-residual",
        "program_category": "d1_utility_residual",
        "objective": "predict fully charged utility residual over the best static control",
        "model_class": "regularized_linear_residual",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
    {
        "candidate_id": "utility-residual-monotone-mlp-v2",
        "family": "utility-residual",
        "program_category": "d1_utility_residual",
        "objective": "predict fully charged utility residual over the best static control",
        "model_class": "monotone_bounded_residual_mlp",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
    {
        "candidate_id": "pairwise-ranking-bradley-terry-v2",
        "family": "pairwise-ranking",
        "program_category": "d1_pairwise_ranking",
        "objective": "rank actor utility pairwise before choosing one charged actor",
        "model_class": "regularized_bradley_terry",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
    {
        "candidate_id": "pairwise-ranking-margin-mlp-v2",
        "family": "pairwise-ranking",
        "program_category": "d1_pairwise_ranking",
        "objective": "rank actor utility pairwise before choosing one charged actor",
        "model_class": "antisymmetric_margin_mlp",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
    {
        "candidate_id": "calibrated-abstaining-temperature-v2",
        "family": "calibrated-abstaining",
        "program_category": "d1_calibrated_abstaining",
        "objective": "dispatch only when calibrated utility exceeds the static fallback",
        "model_class": "temperature_scaled_abstaining_router",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
    {
        "candidate_id": "calibrated-abstaining-isotonic-v2",
        "family": "calibrated-abstaining",
        "program_category": "d1_calibrated_abstaining",
        "objective": "dispatch only when calibrated utility exceeds the static fallback",
        "model_class": "isotonic_abstaining_router",
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "legacy_centroid_control": False,
        "eligible_for_screen": True,
        "execution_authorized": False,
    },
)

_EVIDENCE_FIELDS = frozenset(
    {
        "schema",
        "candidate_id",
        "family",
        "program_category",
        "candidate_definition_sha256",
        "phases",
        "heldout_contract",
        "evidence_status",
        "summary_authority",
        "complete",
        "problems",
        "independent_scientific_generation",
        "ready_for_confirmatory_claim",
        "activation_allowed",
        "scientific_promotion",
        "evidence_sha256",
    }
)
_PHASE_METRIC_FIELDS = frozenset(
    {
        "completed_cell_count",
        "learned_dispatch_differences",
        "favorable_seed_fraction",
        "mean_gap_below_context_route",
        "work_saving_vs_all_five_actors",
    }
)
_SCREEN_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "claim_scope",
        "append_only_revision",
        "append_only_policy",
        "predecessor_byte_bound_verification",
        "legacy_control",
        "criteria",
        "visible_inputs",
        "forbidden_inputs",
        "families",
        "candidate_evidence",
        "candidate_assessments",
        "ranking",
        "eligible_candidate_ids",
        "decision",
        "complete",
        "problems",
        "independent_scientific_generation",
        "ready_for_confirmatory_claim",
        "activation_allowed",
        "scientific_promotion",
        "screen_sha256",
    }
)
_FREEZE_FIELDS = frozenset(
    {
        "schema",
        "program_id",
        "claim_scope",
        "screen_binding",
        "outcome",
        "selected_candidate",
        "candidate_ranking",
        "categorized_program_admission",
        "legacy_control",
        "ready_for_untouched_replication_design",
        "ready_for_confirmatory_claim",
        "execution_authorized",
        "activation_allowed",
        "scientific_promotion",
        "freeze_sha256",
    }
)


def candidate_catalog() -> list[dict[str, Any]]:
    """Return fresh copies of the sealed append-only candidate inventory."""

    return copy.deepcopy(list(_CATALOG))


def _catalog_index() -> dict[str, dict[str, Any]]:
    return {row["candidate_id"]: row for row in candidate_catalog()}


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_fields(value: Mapping[str, Any], fields: frozenset[str], label: str) -> None:
    if set(value) != fields:
        raise ValueError(f"{label} field inventory drifted")


def _is_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sealed(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = {key: item for key, item in value.items() if key != field}
    return {**core, field: frozen.canonical_sha256(core)}


def _validate_ci(value: Any, label: str, expected_n: int) -> None:
    row = _require_mapping(value, label)
    _require_exact_fields(row, frozenset({"mean", "lo", "hi", "n"}), label)
    if not _is_integer(row["n"]) or row["n"] != expected_n:
        raise ValueError(f"{label} observation count drifted")
    if not all(_is_number(row[field]) for field in ("mean", "lo", "hi")):
        raise ValueError(f"{label} contains a non-finite estimate")
    if not float(row["lo"]) <= float(row["mean"]) <= float(row["hi"]):
        raise ValueError(f"{label} interval ordering drifted")


def _validate_phase_metrics(value: Any, label: str) -> None:
    metrics = _require_mapping(value, label)
    _require_exact_fields(metrics, _PHASE_METRIC_FIELDS, label)
    completed = metrics["completed_cell_count"]
    if not _is_integer(completed) or completed <= 0:
        raise ValueError(f"{label} completed-cell count is invalid")
    differences = _require_mapping(
        metrics["learned_dispatch_differences"],
        f"{label} differences",
    )
    _require_exact_fields(
        differences,
        frozenset(STATIC_CONTROLS),
        f"{label} differences",
    )
    for control in STATIC_CONTROLS:
        _validate_ci(differences[control], f"{label} {control} difference", completed)
    favorable = _require_mapping(
        metrics["favorable_seed_fraction"],
        f"{label} favorable fractions",
    )
    _require_exact_fields(
        favorable,
        frozenset(STATIC_CONTROLS),
        f"{label} favorable fractions",
    )
    if any(
        not _is_number(favorable[control]) or not 0.0 <= float(favorable[control]) <= 1.0
        for control in STATIC_CONTROLS
    ):
        raise ValueError(f"{label} favorable fraction is invalid")
    if (
        not _is_number(metrics["mean_gap_below_context_route"])
        or float(metrics["mean_gap_below_context_route"]) < 0.0
    ):
        raise ValueError(f"{label} context-route gap is invalid")
    if (
        not _is_number(metrics["work_saving_vs_all_five_actors"])
        or not 0.0 <= float(metrics["work_saving_vs_all_five_actors"]) <= 1.0
    ):
        raise ValueError(f"{label} work saving is invalid")


def _phase_gates(metrics: Mapping[str, Any]) -> dict[str, bool]:
    differences = _require_mapping(
        metrics["learned_dispatch_differences"],
        "candidate differences",
    )
    favorable = _require_mapping(
        metrics["favorable_seed_fraction"],
        "candidate favorable fractions",
    )
    return {
        "static_margin_gate": all(
            float(_require_mapping(differences[control], f"{control} difference")["mean"])
            >= float(CRITERIA["minimum_mean_advantage_over_each_static_control"])
            and float(_require_mapping(differences[control], f"{control} difference")["lo"])
            > float(CRITERIA["comparison_interval_lower_bound_must_exceed"])
            for control in STATIC_CONTROLS
        ),
        "favorable_seed_fraction_gate": all(
            float(favorable[control]) >= float(CRITERIA["minimum_favorable_seed_fraction"])
            for control in STATIC_CONTROLS
        ),
        "context_route_gap_gate": float(metrics["mean_gap_below_context_route"])
        <= float(CRITERIA["maximum_mean_gap_below_fixed_c2_context_route"]),
        "work_saving_gate": float(metrics["work_saving_vs_all_five_actors"])
        >= float(CRITERIA["minimum_work_saving_vs_all_five_actors"]),
    }


def build_candidate_evidence(
    candidate_id: str,
    *,
    producer: Mapping[str, Any],
    challenge: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal nonadmissible planning summaries for one unexecuted candidate."""

    catalog = _catalog_index()
    if candidate_id == frozen.FROZEN_VARIANT_ID:
        raise ValueError("the legacy centroid control is never a redesign candidate")
    if candidate_id not in catalog:
        raise ValueError("D1 redesign candidate is outside the append-only catalog")
    definition = catalog[candidate_id]
    phases = {
        "producer": copy.deepcopy(dict(producer)),
        "challenge": copy.deepcopy(dict(challenge)),
    }
    for phase in PHASES:
        _validate_phase_metrics(phases[phase], f"D1 redesign {candidate_id} {phase}")
    core = {
        "schema": EVIDENCE_SCHEMA,
        "candidate_id": candidate_id,
        "family": definition["family"],
        "program_category": definition["program_category"],
        "candidate_definition_sha256": frozen.canonical_sha256(definition),
        "phases": phases,
        "heldout_contract": {
            "visible_inputs": list(VISIBLE_INPUTS),
            "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
            "support_geometry_source": "labeled_training_partition_only",
            "contract_honored": True,
            "heldout_sensitive_payloads_emitted": False,
        },
        "evidence_status": "unexecuted_preregistration_summary_only",
        "summary_authority": "caller_supplied_nonadmissible",
        "complete": False,
        "problems": ["raw candidate execution receipts are not bound"],
        "independent_scientific_generation": False,
        "ready_for_confirmatory_claim": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return _sealed(core, "evidence_sha256")


def validate_candidate_evidence(value: Mapping[str, Any]) -> None:
    """Validate one candidate receipt and its heldout leakage boundary."""

    evidence = _require_mapping(value, "D1 redesign candidate evidence")
    _require_exact_fields(evidence, _EVIDENCE_FIELDS, "D1 redesign candidate evidence")
    core = {key: item for key, item in evidence.items() if key != "evidence_sha256"}
    if not _is_sha256(evidence["evidence_sha256"]) or evidence["evidence_sha256"] != frozen.canonical_sha256(
        core
    ):
        raise ValueError("D1 redesign candidate evidence self-seal is invalid")
    catalog = _catalog_index()
    candidate_id = evidence["candidate_id"]
    if not isinstance(candidate_id, str) or candidate_id not in catalog:
        raise ValueError("D1 redesign candidate evidence is outside the append-only catalog")
    if candidate_id == frozen.FROZEN_VARIANT_ID:
        raise ValueError("the legacy centroid control is never eligible")
    definition = catalog[candidate_id]
    if (
        evidence["schema"] != EVIDENCE_SCHEMA
        or evidence["family"] != definition["family"]
        or evidence["program_category"] != definition["program_category"]
        or evidence["candidate_definition_sha256"] != frozen.canonical_sha256(definition)
        or evidence["evidence_status"] != "unexecuted_preregistration_summary_only"
        or evidence["summary_authority"] != "caller_supplied_nonadmissible"
        or evidence["complete"] is not False
        or evidence["problems"] != ["raw candidate execution receipts are not bound"]
        or evidence["independent_scientific_generation"] is not False
        or evidence["ready_for_confirmatory_claim"] is not False
        or evidence["activation_allowed"] is not False
        or evidence["scientific_promotion"] is not False
    ):
        raise ValueError("D1 redesign candidate identity or safety boundary drifted")

    contract = _require_mapping(
        evidence["heldout_contract"],
        "D1 redesign heldout contract",
    )
    _require_exact_fields(
        contract,
        frozenset(
            {
                "visible_inputs",
                "forbidden_inputs",
                "support_geometry_source",
                "contract_honored",
                "heldout_sensitive_payloads_emitted",
            }
        ),
        "D1 redesign heldout contract",
    )
    if contract != {
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "support_geometry_source": "labeled_training_partition_only",
        "contract_honored": True,
        "heldout_sensitive_payloads_emitted": False,
    }:
        raise ValueError("D1 redesign heldout-input boundary drifted")

    phases = _require_mapping(evidence["phases"], "D1 redesign evidence phases")
    _require_exact_fields(phases, frozenset(PHASES), "D1 redesign evidence phases")
    for phase in PHASES:
        _validate_phase_metrics(
            phases[phase],
            f"D1 redesign {candidate_id} {phase}",
        )


def assess_candidate_evidence(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the two-phase frozen gates and deterministic ranking score."""

    validate_candidate_evidence(evidence)
    phase_gates: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        conditions = _phase_gates(
            _require_mapping(
                _require_mapping(evidence["phases"], "candidate phases")[phase],
                f"candidate {phase} metrics",
            )
        )
        phase_gates[phase] = {
            "conditions": conditions,
            "advisory_thresholds_satisfied": all(conditions.values()),
        }
    advisory_passed = all(bool(phase_gates[phase]["advisory_thresholds_satisfied"]) for phase in PHASES)
    static_los: list[float] = []
    static_means: list[float] = []
    favorable: list[float] = []
    context_gaps: list[float] = []
    work_savings: list[float] = []
    phases = _require_mapping(evidence["phases"], "candidate phases")
    for phase in PHASES:
        metrics = _require_mapping(phases[phase], f"candidate {phase} metrics")
        differences = _require_mapping(
            metrics["learned_dispatch_differences"],
            f"candidate {phase} differences",
        )
        fractions = _require_mapping(
            metrics["favorable_seed_fraction"],
            f"candidate {phase} favorable fractions",
        )
        for control in STATIC_CONTROLS:
            difference = _require_mapping(
                differences[control],
                f"candidate {phase} {control} difference",
            )
            static_los.append(float(difference["lo"]))
            static_means.append(float(difference["mean"]))
            favorable.append(float(fractions[control]))
        context_gaps.append(float(metrics["mean_gap_below_context_route"]))
        work_savings.append(float(metrics["work_saving_vs_all_five_actors"]))
    return {
        "candidate_id": evidence["candidate_id"],
        "family": evidence["family"],
        "program_category": evidence["program_category"],
        "phase_gates": phase_gates,
        "caller_summary_thresholds_satisfied": advisory_passed,
        "eligible_for_freeze": False,
        "ranking_score": {
            "minimum_static_interval_lower_bound": min(static_los),
            "minimum_static_mean_advantage": min(static_means),
            "minimum_favorable_seed_fraction": min(favorable),
            "maximum_context_route_gap": max(context_gaps),
            "minimum_work_saving": min(work_savings),
        },
    }


def _ranking_key(assessment: Mapping[str, Any]) -> tuple[Any, ...]:
    score = _require_mapping(assessment["ranking_score"], "candidate ranking score")
    return (
        not bool(assessment["eligible_for_freeze"]),
        -float(score["minimum_static_interval_lower_bound"]),
        -float(score["minimum_static_mean_advantage"]),
        -float(score["minimum_favorable_seed_fraction"]),
        float(score["maximum_context_route_gap"]),
        -float(score["minimum_work_saving"]),
        str(assessment["candidate_id"]),
    )


def _family_rows() -> list[dict[str, Any]]:
    catalog = candidate_catalog()
    return [
        {
            "family": family,
            "program_category": next(row["program_category"] for row in catalog if row["family"] == family),
            "candidate_ids": [row["candidate_id"] for row in catalog if row["family"] == family],
            "parallel_screen_lane": True,
        }
        for family in FAMILIES
    ]


def _screen_core(
    predecessor_byte_bound_verification: Mapping[str, Any],
    candidate_evidence: Sequence[Mapping[str, Any]],
    *,
    predecessor_source_path: Path,
    predecessor_rung_root: Path,
) -> dict[str, Any]:
    frozen.validate_byte_bound_verification(
        predecessor_byte_bound_verification,
        source_path=predecessor_source_path,
        rung_root=predecessor_rung_root,
    )
    predecessor_classification = _require_mapping(
        predecessor_byte_bound_verification["classification"],
        "D1 predecessor classification",
    )
    if (
        predecessor_classification.get("evidence_classification") != frozen.NULL_SAFE_PRUNE
        or predecessor_classification.get("scientific_interpretation") != frozen.NONCONFIRMATORY_GATE_FAILURE
        or predecessor_classification.get("exact_centroid_design_eligible_for_future_freeze") is not False
    ):
        raise ValueError("D1 redesign requires the exact null-safe centroid prune")

    evidence_by_id: dict[str, dict[str, Any]] = {}
    for item in candidate_evidence:
        validate_candidate_evidence(item)
        candidate_id = str(item["candidate_id"])
        if candidate_id in evidence_by_id:
            raise ValueError("D1 redesign candidate evidence is duplicated")
        evidence_by_id[candidate_id] = copy.deepcopy(dict(item))
    expected_ids = [row["candidate_id"] for row in candidate_catalog()]
    if set(evidence_by_id) != set(expected_ids):
        raise ValueError("D1 redesign screen requires the exact append-only candidate inventory")
    ordered_evidence = [evidence_by_id[candidate_id] for candidate_id in expected_ids]
    assessments = [assess_candidate_evidence(evidence) for evidence in ordered_evidence]
    assessments.sort(key=_ranking_key)
    ranking = [str(row["candidate_id"]) for row in assessments]
    if any(row["eligible_for_freeze"] is not False for row in assessments):
        raise ValueError("D1 preregistration summaries cannot become freeze evidence")
    eligible: list[str] = []
    return {
        "schema": SCREEN_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "append_only_revision": 2,
        "append_only_policy": dict(APPEND_ONLY_POLICY),
        "predecessor_byte_bound_verification": copy.deepcopy(dict(predecessor_byte_bound_verification)),
        "legacy_control": dict(LEGACY_CONTROL),
        "criteria": dict(CRITERIA),
        "visible_inputs": list(VISIBLE_INPUTS),
        "forbidden_inputs": list(FORBIDDEN_HELDOUT_INPUTS),
        "families": _family_rows(),
        "candidate_evidence": ordered_evidence,
        "candidate_assessments": assessments,
        "ranking": ranking,
        "eligible_candidate_ids": eligible,
        "decision": {
            "screen_complete": False,
            "provisional_freeze_outcome": NO_CANDIDATE,
            "freeze_required": False,
            "candidate_execution_receipts_required": True,
            "legacy_centroid_eligible": False,
            "independent_verification_required_before_any_scientific_claim": True,
        },
        "complete": False,
        "problems": ["candidate execution receipts are absent; redesign v2 is preregistration only"],
        "independent_scientific_generation": False,
        "ready_for_confirmatory_claim": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def build_screen(
    predecessor_byte_bound_verification: Mapping[str, Any],
    candidate_evidence: Sequence[Mapping[str, Any]],
    *,
    predecessor_source_path: Path = frozen.DEFAULT_RESULT,
    predecessor_rung_root: Path = frozen.DEFAULT_RUNG_ROOT,
) -> dict[str, Any]:
    """Build a live-authority-bound preregistration across all three categories."""

    return _sealed(
        _screen_core(
            predecessor_byte_bound_verification,
            candidate_evidence,
            predecessor_source_path=predecessor_source_path,
            predecessor_rung_root=predecessor_rung_root,
        ),
        "screen_sha256",
    )


def validate_screen(
    value: Mapping[str, Any],
    *,
    predecessor_source_path: Path = frozen.DEFAULT_RESULT,
    predecessor_rung_root: Path = frozen.DEFAULT_RUNG_ROOT,
) -> None:
    """Validate a screen by regenerating its complete semantic projection."""

    screen = _require_mapping(value, "D1 redesign screen")
    _require_exact_fields(screen, _SCREEN_FIELDS, "D1 redesign screen")
    core = {key: item for key, item in screen.items() if key != "screen_sha256"}
    if not _is_sha256(screen["screen_sha256"]) or screen["screen_sha256"] != frozen.canonical_sha256(core):
        raise ValueError("D1 redesign screen self-seal is invalid")
    predecessor = _require_mapping(
        screen["predecessor_byte_bound_verification"],
        "D1 redesign predecessor byte-bound verification",
    )
    evidence = screen["candidate_evidence"]
    if not isinstance(evidence, list):
        raise ValueError("D1 redesign candidate evidence must be a list")
    expected = _screen_core(
        predecessor,
        evidence,
        predecessor_source_path=predecessor_source_path,
        predecessor_rung_root=predecessor_rung_root,
    )
    if frozen.canonical_bytes(core) != frozen.canonical_bytes(expected):
        raise ValueError("D1 redesign screen semantic projection drifted")


def _freeze_core(
    screen: Mapping[str, Any],
    *,
    predecessor_source_path: Path,
    predecessor_rung_root: Path,
) -> dict[str, Any]:
    validate_screen(
        screen,
        predecessor_source_path=predecessor_source_path,
        predecessor_rung_root=predecessor_rung_root,
    )
    ranking = list(screen["ranking"])
    eligible = list(screen["eligible_candidate_ids"])
    if eligible:
        raise ValueError("D1 preregistration screen cannot authorize a winner")
    categorized_admission = []
    for family_row in _family_rows():
        categorized_admission.append(
            {
                "family": family_row["family"],
                "program_category": family_row["program_category"],
                "status": "preregistered_unexecuted",
                "eligible_candidate_ids": [],
                "selected_candidate_id": None,
                "execution_authorized": False,
            }
        )
    return {
        "schema": FREEZE_SCHEMA,
        "program_id": PROGRAM_ID,
        "claim_scope": CLAIM_SCOPE,
        "screen_binding": {
            "screen_sha256": screen["screen_sha256"],
            "predecessor_artifact_sha256": screen["predecessor_byte_bound_verification"]["artifact_sha256"],
        },
        "outcome": NO_CANDIDATE,
        "selected_candidate": None,
        "candidate_ranking": ranking,
        "categorized_program_admission": categorized_admission,
        "legacy_control": dict(LEGACY_CONTROL),
        "ready_for_untouched_replication_design": False,
        "ready_for_confirmatory_claim": False,
        "execution_authorized": False,
        "activation_allowed": False,
        "scientific_promotion": False,
    }


def freeze_screen(
    screen: Mapping[str, Any],
    *,
    predecessor_source_path: Path = frozen.DEFAULT_RESULT,
    predecessor_rung_root: Path = frozen.DEFAULT_RUNG_ROOT,
) -> dict[str, Any]:
    """Seal the preregistration as no-candidate until raw execution exists."""

    return _sealed(
        _freeze_core(
            screen,
            predecessor_source_path=predecessor_source_path,
            predecessor_rung_root=predecessor_rung_root,
        ),
        "freeze_sha256",
    )


def validate_freeze(
    value: Mapping[str, Any],
    screen: Mapping[str, Any],
    *,
    predecessor_source_path: Path = frozen.DEFAULT_RESULT,
    predecessor_rung_root: Path = frozen.DEFAULT_RUNG_ROOT,
) -> None:
    """Validate a freeze against the exact screen that authorized it."""

    freeze = _require_mapping(value, "D1 redesign freeze")
    _require_exact_fields(freeze, _FREEZE_FIELDS, "D1 redesign freeze")
    core = {key: item for key, item in freeze.items() if key != "freeze_sha256"}
    if not _is_sha256(freeze["freeze_sha256"]) or freeze["freeze_sha256"] != frozen.canonical_sha256(core):
        raise ValueError("D1 redesign freeze self-seal is invalid")
    expected = _freeze_core(
        screen,
        predecessor_source_path=predecessor_source_path,
        predecessor_rung_root=predecessor_rung_root,
    )
    if frozen.canonical_bytes(core) != frozen.canonical_bytes(expected):
        raise ValueError("D1 redesign freeze semantic projection drifted")


__all__ = [
    "APPEND_ONLY_POLICY",
    "CLAIM_SCOPE",
    "CRITERIA",
    "EVIDENCE_SCHEMA",
    "FAMILIES",
    "FORBIDDEN_HELDOUT_INPUTS",
    "FREEZE_SCHEMA",
    "LEGACY_CONTROL",
    "NO_CANDIDATE",
    "PROGRAM_ID",
    "SCREEN_SCHEMA",
    "VISIBLE_INPUTS",
    "WINNER",
    "assess_candidate_evidence",
    "build_candidate_evidence",
    "build_screen",
    "candidate_catalog",
    "freeze_screen",
    "validate_candidate_evidence",
    "validate_freeze",
    "validate_screen",
]
