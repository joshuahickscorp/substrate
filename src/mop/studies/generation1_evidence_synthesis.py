"""Conservative post-run evidence synthesis for the sealed Generation-1 census."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from mop.substrate.events import canonical_bytes, canonical_sha256

SCHEMA = "mop-generation1-evidence-synthesis/v1"
CORPUS_SCHEMA = "mop-generation1-cognitive-corpus/v2"
VERIFICATION_SCHEMA = "mop-generation1-cognitive-corpus-verification/v2"
REPORT_SCHEMA = "mop-generation1-empirical-report/v2"
STATE_SCHEMA = "mop-generation1-state/v1"
CLAIM_SCOPE = (
    "read-only Generation-1 C0 evidence synthesis and successor-hypothesis selection; "
    "no context-disjoint niche, cooperation, integrated-substrate, activation, or "
    "scientific-promotion claim"
)

DEFAULT_CORPUS = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
DEFAULT_REPORT = REPO_ROOT / "proof/GENERATION1_EMPIRICAL_REPORT.json"
DEFAULT_PROGRAM_STATE = (
    REPO_ROOT / "runs/generation1/generation1-empirical-cognitive-corpus-v2/program_state.json"
)
DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_EVIDENCE_SYNTHESIS.json"
DEFAULT_TEXT = REPO_ROOT / "runs/generation1/GENERATION1_EVIDENCE_SYNTHESIS.txt"

DIMENSIONS = (
    "biological_plausibility",
    "computational_plausibility",
    "engineering_feasibility",
    "scaling_behavior",
    "sample_efficiency",
    "reasoning_quality",
    "robustness",
    "interpretability",
    "emergent_behavior",
    "computational_efficiency",
)
STATUS_VALUES = frozenset({"measured", "partial", "unmeasured"})

MECHANISMS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("G1-C0", "Which existing cognitive traces are stable?", (), ()),
    ("G1-C1", "Do actors have reproducible, disjoint niches?", (), ("G1-C0",)),
    ("G1-C2", "Is there real complementarity before routing?", (), ("G1-C1",)),
    (
        "G1-E1",
        "Can relational and temporal state form useful events?",
        ("spatial_temporal_motor", "uncertainty_calibration_selective_compute"),
        ("G1-C0",),
    ),
    (
        "G1-D1",
        "Can value-of-computation dispatch exploit validated niches?",
        ("uncertainty_calibration_selective_compute",),
        ("G1-C1", "G1-C2"),
    ),
    (
        "G1-M1",
        "Can bounded messages create causal cooperation?",
        ("communication_social_distributed_cognition",),
        ("G1-C2",),
    ),
    (
        "G1-V1",
        "When is verification worth its cost?",
        ("metacognition_verification_reflection",),
        ("G1-C1",),
    ),
    (
        "G1-K1",
        "Can contradiction be detected, localized, and repaired?",
        ("robustness_dynamics_contradiction", "metacognition_verification_reflection"),
        ("G1-C1",),
    ),
    (
        "G1-R1",
        "What memory organization improves future decisions?",
        ("episodic_semantic_working_memory",),
        ("G1-C0",),
    ),
    (
        "G1-P1",
        "Can a stable core coexist with rapid adaptation?",
        ("continual_learning_plasticity",),
        ("G1-C0",),
    ),
    (
        "G1-A1",
        "Does causal reasoning improve interventions?",
        ("causal_counterfactual_reasoning",),
        ("G1-C0",),
    ),
    (
        "G1-S1",
        "Does imagination or simulation improve action?",
        ("planning_imagination_simulation",),
        ("G1-C0",),
    ),
    (
        "G1-U1",
        "Can uncertainty improve decisions rather than merely describe them?",
        ("uncertainty_calibration_selective_compute",),
        ("G1-C0",),
    ),
    (
        "G1-N1",
        "Can curiosity seek reducible, useful novelty?",
        ("curiosity_novelty_curriculum",),
        ("G1-C0",),
    ),
    (
        "G1-G1",
        "Can finite construction search form better shadow coalitions?",
        ("topology_ecology_reorganization",),
        ("G1-C1", "G1-C2"),
    ),
    (
        "G1-I1",
        "Does an integrated ESCS coalition beat simpler organizations?",
        (),
        ("G1-E1", "G1-D1", "G1-M1", "G1-V1", "G1-R1", "G1-P1"),
    ),
)

NEGATIVE_ACTIONS = {
    "stable_null": "retain_as_control_and_retire_or_narrow_the_candidate",
    "mixed_or_seed_sensitive": "identify_a_moderating_context_before_retest",
    "descriptive_only": "instrument_an_exhaustive_directional_null_before_inference",
    "descriptive_seed_invariant": "raise_difficulty_or_refine_the_measurement",
    "descriptive_seed_adapter_unverified": "repair_seed_authority_before_inference",
    "descriptive_fixed_case": "retain_as_noninferential_single_case",
    "mechanics_noninferential": "retain_as_mechanics_only_not_efficacy_evidence",
}

_SHARED_IMPLEMENTATION_ROLES = frozenset({"experiment_harness", "generation1_driver"})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{label} is not a regular file: {resolved}")
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object: {resolved}")
    return payload, raw


def _repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == canonical_sha256(core)


def _binding(
    path: Path,
    payload: dict[str, Any],
    raw: bytes,
    *,
    seal_field: str,
    mutable: bool = False,
) -> dict[str, Any]:
    row = {
        "path": _repository_path(path),
        "schema": payload.get("schema"),
        "seal_field": seal_field,
        "seal_sha256": payload.get(seal_field),
        "self_seal_valid_at_read": _valid_seal(payload, seal_field),
    }
    if mutable:
        row.update(
            {
                "mutable_snapshot_sha256": _sha256_bytes(raw),
                "mutable_after_read": True,
                "authority_scope": "base_runtime_accounting.base_runtime_sha256",
            }
        )
    else:
        row["sha256"] = _sha256_bytes(raw)
    return row


def _wilson_95(successes: int, observations: int) -> dict[str, Any] | None:
    if observations <= 0:
        return None
    z = 1.959963984540054
    fraction = successes / observations
    denominator = 1.0 + z * z / observations
    center = (fraction + z * z / (2.0 * observations)) / denominator
    radius = (
        z
        * math.sqrt(fraction * (1.0 - fraction) / observations + z * z / (4.0 * observations * observations))
        / denominator
    )
    return {
        "method": "wilson_score",
        "confidence": 0.95,
        "low": round(max(0.0, center - radius), 8),
        "high": round(min(1.0, center + radius), 8),
    }


def _pack_memberships(corpus: dict[str, Any]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = defaultdict(list)
    packs = corpus.get("capability_pack_summaries")
    if not isinstance(packs, dict):
        raise ValueError("corpus capability-pack summaries are missing")
    for pack, raw in packs.items():
        if not isinstance(pack, str) or not isinstance(raw, dict):
            raise ValueError("corpus capability-pack summary is malformed")
        ids = raw.get("experiment_ids")
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise ValueError(f"capability pack {pack} has malformed experiment IDs")
        for experiment_id in ids:
            memberships[experiment_id].append(pack)
    return {key: sorted(values) for key, values in memberships.items()}


def _implementation_family(experiment_id: str, cell_index: dict[str, Any]) -> dict[str, Any]:
    cells = cell_index.get(experiment_id)
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"cell authority index is missing {experiment_id}")
    families: dict[str, list[dict[str, str]]] = {}
    for cell in cells:
        authorities = cell.get("implementation_authorities") if isinstance(cell, dict) else None
        if not isinstance(authorities, list) or not authorities:
            raise ValueError(f"implementation authorities are missing for {experiment_id}")
        if any(
            not isinstance(row, dict)
            or not isinstance(row.get("path"), str)
            or not isinstance(row.get("role"), str)
            or not isinstance(row.get("sha256"), str)
            for row in authorities
        ):
            raise ValueError(f"implementation authority is malformed for {experiment_id}")
        normalized = sorted(
            (
                {
                    "path": row["path"],
                    "role": row["role"],
                    "sha256": row["sha256"],
                }
                for row in authorities
                if row["role"] not in _SHARED_IMPLEMENTATION_ROLES
            ),
            key=lambda row: (row["path"], row["role"], row["sha256"]),
        )
        if not normalized:
            raise ValueError(f"experiment source authority is missing for {experiment_id}")
        families[canonical_sha256(normalized)] = normalized
    if len(families) != 1:
        raise ValueError(f"{experiment_id} has multiple implementation families")
    digest, authorities = next(iter(families.items()))
    return {"sha256": digest, "authorities": authorities}


def _direction(classification: str) -> str:
    if classification == "stable_candidate_trace":
        return "candidate"
    if classification == "stable_null":
        return "null"
    if classification == "mixed_or_seed_sensitive":
        return "mixed"
    return "non_directional"


def _trace_index(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = corpus.get("experiment_summaries")
    cell_index = corpus.get("cell_authority_index")
    if not isinstance(summaries, dict) or not isinstance(cell_index, dict):
        raise ValueError("corpus experiment summaries or cell authority index are missing")
    memberships = _pack_memberships(corpus)
    rows: list[dict[str, Any]] = []
    for experiment_id, raw in sorted(summaries.items()):
        if not isinstance(experiment_id, str) or not isinstance(raw, dict):
            raise ValueError("corpus experiment summary is malformed")
        classification = raw.get("classification")
        nulls = raw.get("null_supported")
        if not isinstance(classification, str) or not isinstance(nulls, dict):
            raise ValueError(f"summary direction is malformed for {experiment_id}")
        observations = nulls.get("observations")
        null_count = nulls.get("true")
        candidate_count = nulls.get("false")
        if (
            isinstance(observations, bool)
            or not isinstance(observations, int)
            or isinstance(null_count, bool)
            or not isinstance(null_count, int)
            or isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or observations < 0
            or null_count < 0
            or candidate_count < 0
            or null_count + candidate_count != observations
        ):
            raise ValueError(f"summary counts are malformed for {experiment_id}")
        null_fraction = round(null_count / observations, 8) if observations else None
        candidate_fraction = round(candidate_count / observations, 8) if observations else None
        expected_null_interval = _wilson_95(null_count, observations)
        if nulls.get("fraction") != null_fraction or nulls.get("wilson_95") != expected_null_interval:
            raise ValueError(f"summary direction statistics differ for {experiment_id}")
        packs = memberships.get(experiment_id, [])
        rows.append(
            {
                "experiment_id": experiment_id,
                "classification": classification,
                "direction": _direction(classification),
                "evidence_class": raw.get("evidence_class"),
                "seed_mode": raw.get("seed_mode"),
                "effective_observation_count": raw.get("effective_observation_count"),
                "null_count": null_count,
                "null_fraction": null_fraction,
                "null_wilson_95": expected_null_interval,
                "candidate_count": candidate_count,
                "candidate_fraction": candidate_fraction,
                "candidate_wilson_95": _wilson_95(candidate_count, observations),
                "capability_packs": packs,
                "contract": raw.get("contract", {}),
                "implementation_family": _implementation_family(experiment_id, cell_index),
                "tag_overlap_only": len(packs) > 1,
            }
        )
    return rows


def _recurrence_analysis(traces: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in traces if row["classification"] == "stable_candidate_trace"]
    tag_overlap = [
        {
            "experiment_id": row["experiment_id"],
            "capability_packs": row["capability_packs"],
            "implementation_family_sha256": row["implementation_family"]["sha256"],
        }
        for row in candidates
        if row["tag_overlap_only"]
    ]
    by_pack: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        for pack in row["capability_packs"]:
            by_pack[pack].append(row)
    source_distinct = []
    for pack, rows in sorted(by_pack.items()):
        families = sorted({row["implementation_family"]["sha256"] for row in rows})
        if len(rows) >= 2 and len(families) >= 2:
            source_distinct.append(
                {
                    "capability_pack": pack,
                    "experiment_ids": sorted(row["experiment_id"] for row in rows),
                    "implementation_family_sha256s": families,
                    "status": "descriptive_source_distinct_recurrence_only",
                }
            )
    return {
        "tag_overlap_candidates": tag_overlap,
        "source_distinct_candidate_recurrence": source_distinct,
        "tag_overlap_is_independent_recurrence": False,
        "source_distinct_recurrence_establishes_context_disjoint_niches": False,
        "limits": [
            "Capability packs are analyst-defined overlapping tags.",
            "Distinct implementation families do not establish independent generators, "
            "context-disjoint niches, causal reuse, or nonlinear cooperation.",
        ],
    }


def _negative_register(traces: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in traces:
        classification = trace["classification"]
        action = NEGATIVE_ACTIONS.get(classification)
        if action is None:
            continue
        rows.append(
            {
                "evidence_id": f"g1_c0:{trace['experiment_id']}",
                "evidence_type": "corpus_classification",
                "classification": classification,
                "experiment_ids": [trace["experiment_id"]],
                "evidence_refs": [f"per_experiment_trace_index.{trace['experiment_id']}"],
                "action": action,
                "limits": ["This action is bounded to the experiment's frozen contract."],
            }
        )
    prior_actions = {
        "event_formation": "redesign_relational_and_temporal_event_state_before_retest",
        "complementarity": "repair_the_bed_before_training_dispatch",
        "continual_learning": "stop_scale_up_until_a_new_plasticity_mechanism_is_preregistered",
        "planning": "require_realized_action_or_intervention_benefit",
    }
    constraints = report.get("generation0_constraints")
    if not isinstance(constraints, dict):
        raise ValueError("empirical report generation-0 constraints are missing")
    for name, action in prior_actions.items():
        raw = constraints.get(name)
        if not isinstance(raw, dict):
            raise ValueError(f"empirical report constraint is missing: {name}")
        rows.append(
            {
                "evidence_id": f"generation0:{name}",
                "evidence_type": "prior_constraint",
                "classification": raw.get("verdict", raw.get("status")),
                "experiment_ids": [],
                "evidence_refs": [f"report.generation0_constraints.{name}"],
                "action": action,
                "limits": [str(raw.get("successor_constraint", "bounded prior constraint"))],
            }
        )
    return sorted(rows, key=lambda row: row["evidence_id"])


def _claim_boundaries() -> dict[str, dict[str, Any]]:
    common = {
        "status": "not_tested_by_g1_c0",
        "evidence_refs": ["corpus.claim_scope", "report.claim_scope"],
    }
    return {
        "context_disjoint_actor_niches": {
            **common,
            "limits": ["Pack overlap and stable per-experiment direction do not test actor-context niches."],
        },
        "nonlinear_perspective_cooperation": {
            **common,
            "limits": ["The census contains no integrated coalition interaction or causal message lesion."],
        },
        "integrated_substrate_advantage": {
            **common,
            "limits": ["No integrated ESCS organization was compared with the strongest simpler system."],
        },
        "natural_world_generality": {
            **common,
            "limits": ["The CPU-now census uses programmatic and synthetic evidence, not natural tasks."],
        },
    }


def _dimension_row(
    mechanism_id: str,
    dimension: str,
    pack_refs: list[str],
) -> dict[str, Any]:
    boundary_ref = "report.next_authority.ready_to_activate_or_integrate_substrate"
    if dimension == "biological_plausibility":
        return {
            "status": "unmeasured",
            "evidence_refs": ["report.claim_scope"],
            "limits": ["Synthetic programmatic evidence is not biological evidence."],
        }
    if dimension == "emergent_behavior":
        return {
            "status": "unmeasured",
            "evidence_refs": [boundary_ref],
            "limits": ["No active integrated population was observed."],
        }
    if dimension == "scaling_behavior":
        return {
            "status": "partial" if mechanism_id == "G1-C0" else "unmeasured",
            "evidence_refs": ["corpus.seed_count", *pack_refs],
            "limits": ["Fresh-seed stability is not architecture, task-size, or horizon scaling."],
        }
    if dimension == "sample_efficiency":
        status = "partial" if mechanism_id in {"G1-P1", "G1-R1"} else "unmeasured"
        return {
            "status": status,
            "evidence_refs": pack_refs or [boundary_ref],
            "limits": ["No harmonized examples-to-performance curve exists for this mechanism."],
        }
    if dimension == "computational_plausibility":
        return {
            "status": "measured" if mechanism_id == "G1-C0" else "partial",
            "evidence_refs": pack_refs or ["corpus.experiment_summaries"],
            "limits": ["Runnable components do not establish an integrated mechanism."],
        }
    if dimension == "engineering_feasibility":
        return {
            "status": "measured" if mechanism_id == "G1-C0" else "partial",
            "evidence_refs": ["report.fresh_mechanics_evidence", *pack_refs],
            "limits": ["Mechanics and interfaces do not establish useful runtime activation."],
        }
    if dimension == "reasoning_quality":
        return {
            "status": "partial",
            "evidence_refs": pack_refs or ["corpus.experiment_summaries"],
            "limits": ["Experiment-specific metrics are not a common reasoning-quality scale."],
        }
    if dimension == "robustness":
        status = "partial" if mechanism_id in {"G1-C0", "G1-E1", "G1-K1", "G1-P1", "G1-G1"} else "unmeasured"
        return {
            "status": status,
            "evidence_refs": [
                "corpus.capability_pack_summaries.robustness_dynamics_contradiction",
                *pack_refs,
            ],
            "limits": ["No unified mechanism-level attack and restoration suite was run."],
        }
    if dimension == "interpretability":
        return {
            "status": "partial",
            "evidence_refs": ["corpus.cell_authority_index", "corpus.experiment_summaries"],
            "limits": ["Exact lineage is not a causal explanation of internal computation."],
        }
    if dimension == "computational_efficiency":
        status = "partial" if mechanism_id in {"G1-C0", "G1-E1", "G1-V1", "G1-P1"} else "unmeasured"
        return {
            "status": status,
            "evidence_refs": [
                "corpus.operational_summary",
                "report.resource_authority",
                *pack_refs,
            ],
            "limits": ["Campaign cost and predecessor costs do not establish intelligence per compute."],
        }
    raise AssertionError(dimension)


def _dimension_matrix(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    packs = corpus["capability_pack_summaries"]
    rows = []
    for mechanism_id, question, mechanism_packs, _ in MECHANISMS:
        missing = sorted(set(mechanism_packs) - set(packs))
        if missing:
            raise ValueError(f"mechanism {mechanism_id} references missing packs: {missing}")
        pack_refs = [f"corpus.capability_pack_summaries.{pack}" for pack in mechanism_packs]
        dimensions = {
            dimension: _dimension_row(mechanism_id, dimension, pack_refs) for dimension in DIMENSIONS
        }
        if set(dimensions) != set(DIMENSIONS) or any(
            row["status"] not in STATUS_VALUES for row in dimensions.values()
        ):
            raise AssertionError("dimension matrix construction violated its closed schema")
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "question": question,
                "capability_packs": list(mechanism_packs),
                "dimensions": dimensions,
            }
        )
    return rows


def _priority_queue(traces: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    candidate_ids = sorted(
        row["experiment_id"] for row in traces if row["classification"] == "stable_candidate_trace"
    )
    items = []
    for mechanism_id, question, packs, dependencies in MECHANISMS:
        refs = [f"corpus.capability_pack_summaries.{pack}" for pack in packs]
        if mechanism_id == "G1-C0":
            readiness = "complete_verified_discovery_epoch"
            action = "preserve_the_verified_atlas_and_use_it_only_for_hypothesis_selection"
        elif not dependencies or dependencies == ("G1-C0",):
            readiness = "eligible_for_preregistration_not_activation"
            action = "write_a_separate_mechanism_epoch_with_strongest_controls"
        else:
            readiness = "dependency_gated"
            action = "wait_for_positive_prerequisite_evidence_without_weakening_the_gate"
        items.append(
            {
                "mechanism_id": mechanism_id,
                "question": question,
                "readiness": readiness,
                "prerequisite_ids": list(dependencies),
                "evidence_refs": refs or ["corpus.experiment_summaries", "report.generation0_constraints"],
                "condition": "new_sealed_epoch_bound_to_this_synthesis",
                "next_action": action,
                "positive_route": "advance_only_to_the_declared_bounded_successor",
                "null_route": "retain_the_control_and_retire_or_narrow_the_candidate",
                "invalid_bed_route": "repair_or_replace_the_bed_without_calling_a_mechanism_null",
                "cross_metric_rank": None,
            }
        )
    return {
        "policy": "dependency_order_only_no_cross_metric_ranking",
        "cross_metric_ranking_performed": False,
        "candidate_trace_count": len(candidate_ids),
        "candidate_experiment_ids": candidate_ids,
        "items": items,
    }


def _finite_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(float(value)) else None


def _base_runtime_accounting(state: dict[str, Any]) -> dict[str, Any]:
    capsules = state.get("capsules")
    if not isinstance(capsules, dict):
        raise ValueError("program state capsules are missing")
    rows = []
    for capsule_id, raw in sorted(capsules.items()):
        if not isinstance(raw, dict) or raw.get("source") != "base":
            continue
        runtime = raw.get("runtime")
        if not isinstance(runtime, dict):
            raise ValueError(f"base capsule runtime is malformed: {capsule_id}")
        events = runtime.get("events")
        if not isinstance(events, list):
            raise ValueError(f"base capsule events are malformed: {capsule_id}")
        event_counts = Counter(
            str(event.get("event"))
            for event in events
            if isinstance(event, dict) and isinstance(event.get("event"), str)
        )
        artifacts = raw.get("artifacts")
        if not isinstance(artifacts, list):
            raise ValueError(f"base capsule artifacts are malformed: {capsule_id}")
        rows.append(
            {
                "id": capsule_id,
                "source": "base",
                "kind": raw.get("kind"),
                "status": raw.get("status"),
                "attempts": raw.get("attempts"),
                "returncode": raw.get("returncode"),
                "started_at": raw.get("started_at"),
                "finished_at": raw.get("finished_at"),
                "artifacts": sorted(
                    (
                        {
                            "path": artifact.get("path"),
                            "schema": artifact.get("schema"),
                            "sha256": artifact.get("sha256"),
                        }
                        for artifact in artifacts
                        if isinstance(artifact, dict)
                    ),
                    key=lambda item: str(item["path"]),
                ),
                "runtime": {
                    "event_count": runtime.get("event_count"),
                    "events_dropped": runtime.get("events_dropped"),
                    "event_type_counts": dict(sorted(event_counts.items())),
                    "reservation_count": runtime.get("reservation_count"),
                    "retry_count": runtime.get("retry_count"),
                    "resource_stop_count": runtime.get("resource_stop_count"),
                    "sample_count": runtime.get("sample_count"),
                    "peak_process_tree_rss_bytes": runtime.get("peak_process_tree_rss_bytes"),
                    "minimum_disk_free_gb": runtime.get("minimum_disk_free_gb"),
                    "minimum_memory_available_gb": runtime.get("minimum_memory_available_gb"),
                    "minimum_memory_pressure_free_percent": runtime.get(
                        "minimum_memory_pressure_free_percent"
                    ),
                    "maximum_swap_used_gb": runtime.get("maximum_swap_used_gb"),
                    "thermal_statuses": sorted(set(runtime.get("thermal_statuses") or [])),
                    "power_sources": sorted(set(runtime.get("power_sources") or [])),
                    "safety_state": runtime.get("safety_state"),
                },
            }
        )
    if not rows or any(row["status"] != "complete" for row in rows):
        raise ValueError("all base capsules must be complete before evidence synthesis")
    if not any(row["id"] == "g1_empirical_report" for row in rows):
        raise ValueError("base empirical-report capsule is missing")

    def total(field: str, *, runtime: bool = False) -> int:
        values = [(row["runtime"].get(field) if runtime else row.get(field)) or 0 for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise ValueError(f"base runtime integer field is malformed: {field}")
        return sum(values)

    def extrema(field: str, operation: str) -> float | int | None:
        values = [value for row in rows if (value := _finite_number(row["runtime"].get(field))) is not None]
        return (min(values) if operation == "min" else max(values)) if values else None

    core = {
        "program_id": state.get("program_id"),
        "base_capsule_count": len(rows),
        "status_counts": dict(sorted(Counter(row["status"] for row in rows).items())),
        "kind_counts": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "total_attempts": total("attempts"),
        "total_retries": total("retry_count", runtime=True),
        "total_resource_stops": total("resource_stop_count", runtime=True),
        "total_reservations": total("reservation_count", runtime=True),
        "total_samples": total("sample_count", runtime=True),
        "total_runtime_events": total("event_count", runtime=True),
        "total_runtime_events_dropped": total("events_dropped", runtime=True),
        "peak_process_tree_rss_bytes": extrema("peak_process_tree_rss_bytes", "max"),
        "minimum_disk_free_gb": extrema("minimum_disk_free_gb", "min"),
        "minimum_memory_available_gb": extrema("minimum_memory_available_gb", "min"),
        "minimum_memory_pressure_free_percent": extrema("minimum_memory_pressure_free_percent", "min"),
        "maximum_swap_used_gb": extrema("maximum_swap_used_gb", "max"),
        "capsules": rows,
    }
    return {**core, "base_runtime_sha256": canonical_sha256(core)}


def _validate_inputs(
    corpus_path: Path,
    verification_path: Path,
    report_path: Path,
    program_state_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    corpus, corpus_raw = _read(corpus_path, "cognitive corpus")
    verification, verification_raw = _read(verification_path, "corpus verification")
    report, report_raw = _read(report_path, "empirical report")
    state, state_raw = _read(program_state_path, "program state")
    requirements = (
        (corpus, CORPUS_SCHEMA, "corpus_sha256", "cognitive corpus"),
        (verification, VERIFICATION_SCHEMA, "verification_sha256", "corpus verification"),
        (report, REPORT_SCHEMA, "report_sha256", "empirical report"),
        (state, STATE_SCHEMA, "state_sha256", "program state"),
    )
    for payload, schema, seal, label in requirements:
        if payload.get("schema") != schema or not _valid_seal(payload, seal):
            raise ValueError(f"{label} schema or self-seal is invalid")
    checks = verification.get("checks")
    if (
        verification.get("verification_complete") is not True
        or verification.get("problems") != []
        or not isinstance(checks, dict)
        or not checks
        or any(value is not True for value in checks.values())
    ):
        raise ValueError("corpus verifier did not complete every declared check")
    corpus_file_sha = _sha256_bytes(corpus_raw)
    verification_file_sha = _sha256_bytes(verification_raw)
    verification_corpus = verification.get("corpus")
    report_corpus = report.get("corpus")
    if (
        not isinstance(verification_corpus, dict)
        or verification_corpus.get("sha256") != corpus_file_sha
        or not isinstance(report_corpus, dict)
        or report_corpus.get("source", {}).get("sha256") != corpus_file_sha
        or report_corpus.get("verification", {}).get("sha256") != verification_file_sha
    ):
        raise ValueError("verification/report inputs are not bound to the supplied corpus bytes")
    if (
        corpus.get("corpus_complete") is not True
        or report_corpus.get("corpus_complete") is not True
        or corpus.get("scientific_promotion") is not False
        or report.get("scientific_promotion") is not False
    ):
        raise ValueError("Generation-1 inputs are incomplete or promotion-unsafe")
    sources = {
        "corpus": _binding(corpus_path, corpus, corpus_raw, seal_field="corpus_sha256"),
        "corpus_verification": _binding(
            verification_path,
            verification,
            verification_raw,
            seal_field="verification_sha256",
        ),
        "empirical_report": _binding(report_path, report, report_raw, seal_field="report_sha256"),
        "program_state_snapshot": _binding(
            program_state_path,
            state,
            state_raw,
            seal_field="state_sha256",
            mutable=True,
        ),
    }
    return corpus, verification, report, state, sources


def build_synthesis(
    corpus_path: Path,
    verification_path: Path,
    report_path: Path,
    program_state_path: Path,
    *,
    created_at: str | None = None,
) -> dict[str, Any]:

    corpus, _, report, state, sources = _validate_inputs(
        corpus_path.resolve(),
        verification_path.resolve(),
        report_path.resolve(),
        program_state_path.resolve(),
    )
    traces = _trace_index(corpus)
    core = {
        "schema": SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "created_at": created_at or datetime.now(UTC).isoformat(),
        "sources": sources,
        "per_experiment_trace_index": traces,
        "recurrence_analysis": _recurrence_analysis(traces),
        "negative_evidence_register": _negative_register(traces, report),
        "claim_boundaries": _claim_boundaries(),
        "mechanism_dimension_matrix": _dimension_matrix(corpus),
        "conditional_priority_queue": _priority_queue(traces, report),
        "base_runtime_accounting": _base_runtime_accounting(state),
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    return {**core, "synthesis_sha256": canonical_sha256(core)}


def render_text(synthesis: dict[str, Any]) -> str:

    traces = synthesis["per_experiment_trace_index"]
    counts = Counter(row["classification"] for row in traces)
    runtime = synthesis["base_runtime_accounting"]
    lines = [
        "MOP GENERATION 1 EVIDENCE SYNTHESIS",
        "",
        f"Experiments: {len(traces)}",
        "Classifications:",
        *(f"  {key}: {value}" for key, value in sorted(counts.items())),
        "",
        "Negative and null-safe actions:",
    ]
    for row in synthesis["negative_evidence_register"]:
        lines.append(f"  {row['evidence_id']}: {row['action']}")
    lines.extend(["", "Claim boundaries:"])
    for name, row in synthesis["claim_boundaries"].items():
        lines.append(f"  {name}: {row['status']}")
    lines.extend(["", "Mechanism evidence dimensions:"])
    for row in synthesis["mechanism_dimension_matrix"]:
        summary = Counter(item["status"] for item in row["dimensions"].values())
        rendered = ", ".join(f"{key}={value}" for key, value in sorted(summary.items()))
        lines.append(f"  {row['mechanism_id']}: {rendered}")
    lines.extend(
        [
            "",
            "Base runtime accounting:",
            f"  capsules: {runtime['base_capsule_count']}",
            f"  attempts: {runtime['total_attempts']}",
            f"  retries: {runtime['total_retries']}",
            f"  resource stops: {runtime['total_resource_stops']}",
            f"  peak process-tree RSS bytes: {runtime['peak_process_tree_rss_bytes']}",
            "",
            "Cross-metric ranking performed: false",
            "Activation allowed: false",
            "Scientific promotion: false",
            f"Base runtime SHA256: {runtime['base_runtime_sha256']}",
            f"Synthesis SHA256: {synthesis['synthesis_sha256']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--program-state", type=Path, default=DEFAULT_PROGRAM_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-out", type=Path, default=DEFAULT_TEXT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    synthesis = build_synthesis(
        arguments.corpus,
        arguments.verification,
        arguments.report,
        arguments.program_state,
    )
    _atomic(arguments.out.resolve(), canonical_bytes(synthesis) + b"\n")
    _atomic(arguments.text_out.resolve(), render_text(synthesis).encode("utf-8"))
    print(
        json.dumps(
            {
                "output": str(arguments.out),
                "text_output": str(arguments.text_out),
                "experiment_count": len(synthesis["per_experiment_trace_index"]),
                "activation_allowed": False,
                "scientific_promotion": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
