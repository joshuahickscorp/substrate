"""Independent verifier for the derived Generation-1 evidence synthesis.

The verifier deliberately does not import the synthesis producer.  It rebuilds
the producer's evidence-facing views from the sealed corpus, corpus verifier,
empirical report, and detached-program state, then compares those views with
the supplied synthesis.  This keeps a producer bug from becoming its own
verification oracle.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from mop.substrate.events import canonical_bytes

SYNTHESIS_SCHEMA = "mop-generation1-evidence-synthesis/v1"
VERIFICATION_SCHEMA = "mop-generation1-evidence-synthesis-verification/v1"
CORPUS_SCHEMA = "mop-generation1-cognitive-corpus/v2"
CORPUS_VERIFICATION_SCHEMA = "mop-generation1-cognitive-corpus-verification/v2"
REPORT_SCHEMA = "mop-generation1-empirical-report/v2"
PROGRAM_STATE_SCHEMA = "mop-generation1-state/v1"
SYNTHESIS_CLAIM_SCOPE = (
    "read-only Generation-1 C0 evidence synthesis and successor-hypothesis selection; "
    "no context-disjoint niche, cooperation, integrated-substrate, activation, or "
    "scientific-promotion claim"
)

SYNTHESIS_KEYS = frozenset(
    {
        "schema",
        "claim_scope",
        "created_at",
        "sources",
        "per_experiment_trace_index",
        "recurrence_analysis",
        "negative_evidence_register",
        "claim_boundaries",
        "mechanism_dimension_matrix",
        "conditional_priority_queue",
        "base_runtime_accounting",
        "activation_allowed",
        "scientific_promotion",
        "synthesis_sha256",
    }
)

DEFAULT_CORPUS = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.json"
DEFAULT_CORPUS_VERIFICATION = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
DEFAULT_REPORT = REPO_ROOT / "proof/GENERATION1_EMPIRICAL_REPORT.json"
DEFAULT_PROGRAM_STATE = (
    REPO_ROOT / "runs/generation1/generation1-empirical-cognitive-corpus-v2/program_state.json"
)
DEFAULT_SYNTHESIS = REPO_ROOT / "proof/GENERATION1_EVIDENCE_SYNTHESIS.json"
DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json"

CANONICAL_MECHANISM_IDS = (
    "G1-C0",
    "G1-C1",
    "G1-C2",
    "G1-E1",
    "G1-D1",
    "G1-M1",
    "G1-V1",
    "G1-K1",
    "G1-R1",
    "G1-P1",
    "G1-A1",
    "G1-S1",
    "G1-U1",
    "G1-N1",
    "G1-G1",
    "G1-I1",
)

MECHANISM_CONTRACTS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
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

REQUESTED_DIMENSIONS = (
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

ALLOWED_DIMENSION_STATUSES = frozenset({"measured", "partial", "unmeasured"})

REQUIRED_UNTESTED_CLAIMS = (
    "context_disjoint_actor_niches",
    "nonlinear_perspective_cooperation",
    "integrated_substrate_advantage",
    "natural_world_generality",
)

UNTESTED_CLAIM_STATUS = "not_tested_by_g1_c0"

_SHARED_IMPLEMENTATION_ROLES = frozenset({"experiment_harness", "generation1_driver"})


class EvidenceSynthesisVerificationError(ValueError):
    pass


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceSynthesisVerificationError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceSynthesisVerificationError(f"{label} is not a JSON object: {path}")
    return payload


def _valid_seal(payload: Mapping[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    try:
        expected = _canonical_sha256(core)
    except (TypeError, ValueError):
        return False
    return isinstance(declared, str) and declared == expected


def _repo_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _source_receipt(path: Path, payload: Mapping[str, Any], seal_field: str) -> dict[str, Any]:
    return {
        "path": _repo_path(path),
        "sha256": _sha256_file(path),
        "schema": payload.get("schema"),
        "seal_field": seal_field,
        "seal_sha256": payload.get(seal_field),
        "self_seal_valid_at_read": _valid_seal(payload, seal_field),
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def _pack_memberships(corpus: Mapping[str, Any]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    packs = corpus.get("capability_pack_summaries")
    if not isinstance(packs, dict):
        return memberships
    for pack, raw in packs.items():
        if not isinstance(pack, str) or not isinstance(raw, dict):
            continue
        members = raw.get("experiment_ids")
        if not isinstance(members, list):
            continue
        for experiment_id in members:
            if isinstance(experiment_id, str):
                memberships.setdefault(experiment_id, []).append(pack)
    for values in memberships.values():
        values.sort()
    return memberships


def _implementation_family(
    experiment_id: str,
    corpus: Mapping[str, Any],
) -> dict[str, Any]:
    index = corpus.get("cell_authority_index")
    rows = index.get(experiment_id) if isinstance(index, dict) else None
    authorities: dict[tuple[str, str, str], dict[str, str]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            raw_authorities = row.get("implementation_authorities")
            if not isinstance(raw_authorities, list):
                continue
            for raw in raw_authorities:
                if not isinstance(raw, dict):
                    continue
                path = raw.get("path")
                role = raw.get("role")
                sha256 = raw.get("sha256")
                if (
                    isinstance(path, str)
                    and isinstance(role, str)
                    and isinstance(sha256, str)
                    and role not in _SHARED_IMPLEMENTATION_ROLES
                ):
                    authorities[(path, role, sha256)] = {
                        "path": path,
                        "role": role,
                        "sha256": sha256,
                    }
    ordered = [authorities[key] for key in sorted(authorities)]
    return {"authorities": ordered, "sha256": _canonical_sha256(ordered)}


def _direction(classification: str) -> str:
    if classification == "stable_candidate_trace":
        return "candidate"
    if classification == "stable_null":
        return "null"
    if classification == "mixed_or_seed_sensitive":
        return "mixed"
    return "non_directional"


def _recompute_trace_index(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    summaries = corpus.get("experiment_summaries")
    if not isinstance(summaries, dict):
        return []
    memberships = _pack_memberships(corpus)
    result: list[dict[str, Any]] = []
    for experiment_id, raw in sorted(summaries.items()):
        if not isinstance(experiment_id, str) or not isinstance(raw, dict):
            continue
        null = raw.get("null_supported")
        null = null if isinstance(null, dict) else {}
        observations = null.get("observations")
        null_count = null.get("true")
        candidate_count = null.get("false")
        observations = observations if isinstance(observations, int) else 0
        null_count = null_count if isinstance(null_count, int) else 0
        candidate_count = candidate_count if isinstance(candidate_count, int) else 0
        null_wilson = _wilson_95(null_count, observations)
        candidate_wilson = _wilson_95(candidate_count, observations)
        classification = str(raw.get("classification", "unclassified"))
        result.append(
            {
                "experiment_id": experiment_id,
                "classification": classification,
                "direction": _direction(classification),
                "evidence_class": raw.get("evidence_class"),
                "seed_mode": raw.get("seed_mode"),
                "effective_observation_count": raw.get("effective_observation_count"),
                "null_count": null_count,
                "candidate_count": candidate_count,
                "null_fraction": (round(null_count / observations, 8) if observations > 0 else None),
                "candidate_fraction": (
                    round(candidate_count / observations, 8) if observations > 0 else None
                ),
                "null_wilson_95": null_wilson,
                "candidate_wilson_95": candidate_wilson,
                "capability_packs": memberships.get(experiment_id, []),
                "contract": raw.get("contract", {}),
                "implementation_family": _implementation_family(experiment_id, corpus),
                "tag_overlap_only": len(memberships.get(experiment_id, [])) > 1,
            }
        )
    return result


def _recompute_negative_register(
    trace_index: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trace in trace_index:
        classification = str(trace.get("classification"))
        action = NEGATIVE_ACTIONS.get(classification)
        if action is None:
            continue
        experiment_id = str(trace.get("experiment_id"))
        rows.append(
            {
                "evidence_id": f"g1_c0:{experiment_id}",
                "evidence_type": "corpus_classification",
                "classification": classification,
                "experiment_ids": [experiment_id],
                "evidence_refs": [f"per_experiment_trace_index.{experiment_id}"],
                "action": action,
                "limits": ["This action is bounded to the experiment's frozen contract."],
            }
        )
    prior_actions = {
        "event_formation": "redesign_relational_and_temporal_event_state_before_retest",
        "complementarity": "repair_the_bed_before_training_dispatch",
        "continual_learning": ("stop_scale_up_until_a_new_plasticity_mechanism_is_preregistered"),
        "planning": "require_realized_action_or_intervention_benefit",
    }
    constraints = report.get("generation0_constraints")
    constraints = constraints if isinstance(constraints, dict) else {}
    for name, action in prior_actions.items():
        raw = constraints.get(name)
        raw = raw if isinstance(raw, dict) else {}
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
    return sorted(rows, key=lambda row: str(row["evidence_id"]))


def _finite_value(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return value if math.isfinite(float(value)) else None


def _recompute_current_base_runtime(program_state: Mapping[str, Any]) -> dict[str, Any]:
    capsules = program_state.get("capsules")
    raw_capsules = capsules if isinstance(capsules, dict) else {}
    rows: list[dict[str, Any]] = []
    for capsule_id, raw in sorted(raw_capsules.items()):
        if not isinstance(capsule_id, str) or not isinstance(raw, dict):
            continue
        if raw.get("source") != "base":
            continue
        runtime = raw.get("runtime")
        runtime = runtime if isinstance(runtime, dict) else {}
        events = runtime.get("events")
        events = events if isinstance(events, list) else []
        event_counts = Counter(
            str(event.get("event"))
            for event in events
            if isinstance(event, dict) and isinstance(event.get("event"), str)
        )
        artifacts = raw.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, list) else []
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

    def total(field: str, *, runtime: bool = False) -> int:
        values = [(row["runtime"].get(field) if runtime else row.get(field)) or 0 for row in rows]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            return -1
        return sum(values)

    def extrema(field: str, operation: str) -> float | int | None:
        values = [value for row in rows if (value := _finite_value(row["runtime"].get(field))) is not None]
        return (min(values) if operation == "min" else max(values)) if values else None

    core = {
        "program_id": program_state.get("program_id"),
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
    return {**core, "base_runtime_sha256": _canonical_sha256(core)}


def _recompute_recurrence(trace_index: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidates = [row for row in trace_index if row.get("classification") == "stable_candidate_trace"]
    tag_overlap = [
        {
            "experiment_id": row["experiment_id"],
            "capability_packs": row["capability_packs"],
            "implementation_family_sha256": row["implementation_family"]["sha256"],
        }
        for row in candidates
        if row.get("tag_overlap_only") is True
    ]
    by_pack: dict[str, list[Mapping[str, Any]]] = {}
    for row in candidates:
        packs = row.get("capability_packs")
        if not isinstance(packs, list):
            continue
        for pack in packs:
            if isinstance(pack, str):
                by_pack.setdefault(pack, []).append(row)
    source_distinct: list[dict[str, Any]] = []
    for pack, rows in sorted(by_pack.items()):
        families = sorted(
            {
                str(family.get("sha256"))
                for row in rows
                if isinstance((family := row.get("implementation_family")), dict)
            }
        )
        if len(rows) >= 2 and len(families) >= 2:
            source_distinct.append(
                {
                    "capability_pack": pack,
                    "experiment_ids": sorted(str(row["experiment_id"]) for row in rows),
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


def _expected_claim_boundaries() -> dict[str, Any]:
    common = {
        "status": UNTESTED_CLAIM_STATUS,
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


def _expected_dimension_row(
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
        return {
            "status": "partial" if mechanism_id in {"G1-P1", "G1-R1"} else "unmeasured",
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
    raise EvidenceSynthesisVerificationError(f"unknown requested dimension: {dimension}")


def _expected_dimension_matrix(corpus: Mapping[str, Any]) -> list[dict[str, Any]]:
    packs = corpus.get("capability_pack_summaries")
    pack_names = set(packs) if isinstance(packs, dict) else set()
    rows: list[dict[str, Any]] = []
    for mechanism_id, question, mechanism_packs, _ in MECHANISM_CONTRACTS:
        if not set(mechanism_packs).issubset(pack_names):
            continue
        pack_refs = [f"corpus.capability_pack_summaries.{pack}" for pack in mechanism_packs]
        rows.append(
            {
                "mechanism_id": mechanism_id,
                "question": question,
                "capability_packs": list(mechanism_packs),
                "dimensions": {
                    dimension: _expected_dimension_row(mechanism_id, dimension, pack_refs)
                    for dimension in REQUESTED_DIMENSIONS
                },
            }
        )
    return rows


def _expected_priority_queue(
    trace_index: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_ids = sorted(
        str(row.get("experiment_id"))
        for row in trace_index
        if row.get("classification") == "stable_candidate_trace"
    )
    items: list[dict[str, Any]] = []
    for mechanism_id, question, packs, dependencies in MECHANISM_CONTRACTS:
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
                "invalid_bed_route": ("repair_or_replace_the_bed_without_calling_a_mechanism_null"),
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


def _static_source_matches(raw: Any, expected: Mapping[str, Any]) -> bool:

    if not isinstance(raw, dict):
        return False
    return all(raw.get(field) == expected.get(field) for field in expected)


def _program_source_matches(raw: Any, expected_path: Path) -> bool:

    if not isinstance(raw, dict):
        return False
    return (
        raw.get("path") == _repo_path(expected_path)
        and raw.get("schema") == PROGRAM_STATE_SCHEMA
        and raw.get("seal_field") == "state_sha256"
        and isinstance(raw.get("seal_sha256"), str)
        and raw.get("self_seal_valid_at_read") is True
        and isinstance(raw.get("mutable_snapshot_sha256"), str)
        and raw.get("mutable_after_read") is True
        and raw.get("authority_scope") == "base_runtime_accounting.base_runtime_sha256"
    )


def _nonempty_string_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _claim_boundaries_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_UNTESTED_CLAIMS):
        return False
    for claim in REQUIRED_UNTESTED_CLAIMS:
        row = value.get(claim)
        if not isinstance(row, dict) or row.get("status") != UNTESTED_CLAIM_STATUS:
            return False
        if not _nonempty_string_list(row.get("evidence_refs")):
            return False
        if not _nonempty_string_list(row.get("limits")):
            return False
    return True


def _matrix_rows(value: Any) -> list[dict[str, Any]] | None:
    if isinstance(value, list) and all(isinstance(row, dict) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, dict):
        rows: list[dict[str, Any]] = []
        for mechanism_id, raw in value.items():
            if not isinstance(mechanism_id, str) or not isinstance(raw, dict):
                return None
            if "mechanism_id" in raw and raw.get("mechanism_id") != mechanism_id:
                return None
            rows.append({"mechanism_id": mechanism_id, **raw})
        return rows
    return None


def _mechanism_matrix_valid(value: Any) -> bool:
    rows = _matrix_rows(value)
    if rows is None:
        return False
    ids = [row.get("mechanism_id") for row in rows]
    if ids != list(CANONICAL_MECHANISM_IDS):
        return False
    for row in rows:
        mechanism_id = str(row["mechanism_id"])
        dimensions = row.get("dimensions")
        if not isinstance(dimensions, dict) or set(dimensions) != set(REQUESTED_DIMENSIONS):
            return False
        for dimension in REQUESTED_DIMENSIONS:
            cell = dimensions.get(dimension)
            if not isinstance(cell, dict) or set(cell) != {
                "status",
                "evidence_refs",
                "limits",
            }:
                return False
            status = cell.get("status")
            if status not in ALLOWED_DIMENSION_STATUSES:
                return False
            if mechanism_id != "G1-C0" and status == "measured":
                return False
            if not _nonempty_string_list(cell.get("evidence_refs")):
                return False
            if not _nonempty_string_list(cell.get("limits")):
                return False
    return True


def _priority_queue_valid(value: Any) -> bool:
    if not isinstance(value, dict) or set(value) != {
        "policy",
        "cross_metric_ranking_performed",
        "candidate_trace_count",
        "candidate_experiment_ids",
        "items",
    }:
        return False
    if (
        value.get("policy") != "dependency_order_only_no_cross_metric_ranking"
        or value.get("cross_metric_ranking_performed") is not False
    ):
        return False
    items = value.get("items")
    if not isinstance(items, list) or not items:
        return False
    ids: list[str] = []
    for row in items:
        if not isinstance(row, dict):
            return False
        mechanism_id = row.get("mechanism_id")
        if mechanism_id not in CANONICAL_MECHANISM_IDS:
            return False
        ids.append(str(mechanism_id))
        if not _nonempty_string_list(row.get("evidence_refs")):
            return False
        if not isinstance(row.get("condition"), str) or not row["condition"]:
            return False
        if row.get("cross_metric_rank") is not None:
            return False
    return ids == list(CANONICAL_MECHANISM_IDS)


def _source_checks(
    *,
    corpus_path: Path,
    corpus: Mapping[str, Any],
    corpus_verification_path: Path,
    corpus_verification: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
    program_state_path: Path,
    program_state: Mapping[str, Any],
) -> dict[str, bool]:
    corpus_receipt = _source_receipt(corpus_path, corpus, "corpus_sha256")
    verification_receipt = _source_receipt(
        corpus_verification_path,
        corpus_verification,
        "verification_sha256",
    )
    report_receipt = _source_receipt(report_path, report, "report_sha256")
    verification_checks = corpus_verification.get("checks")
    report_corpus = report.get("corpus")
    report_corpus = report_corpus if isinstance(report_corpus, dict) else {}
    corpus_source = report_corpus.get("source")
    verification_source = report_corpus.get("verification")
    verifier_corpus = corpus_verification.get("corpus")
    return {
        "corpus_schema": corpus.get("schema") == CORPUS_SCHEMA,
        "corpus_self_seal": _valid_seal(corpus, "corpus_sha256"),
        "corpus_verification_schema": (corpus_verification.get("schema") == CORPUS_VERIFICATION_SCHEMA),
        "corpus_verification_self_seal": _valid_seal(corpus_verification, "verification_sha256"),
        "corpus_verification_complete": (
            corpus_verification.get("verification_complete") is True
            and corpus_verification.get("problems") == []
        ),
        "corpus_verification_all_checks": (
            isinstance(verification_checks, dict)
            and bool(verification_checks)
            and all(value is True for value in verification_checks.values())
        ),
        "corpus_verification_file_binding": (
            isinstance(verifier_corpus, dict)
            and verifier_corpus.get("path") == corpus_receipt["path"]
            and verifier_corpus.get("sha256") == corpus_receipt["sha256"]
            and verifier_corpus.get("corpus_sha256") == corpus.get("corpus_sha256")
        ),
        "report_schema": report.get("schema") == REPORT_SCHEMA,
        "report_self_seal": _valid_seal(report, "report_sha256"),
        "report_corpus_file_binding": (
            isinstance(corpus_source, dict)
            and corpus_source.get("path") == corpus_receipt["path"]
            and corpus_source.get("sha256") == corpus_receipt["sha256"]
        ),
        "report_verification_file_binding": (
            isinstance(verification_source, dict)
            and verification_source.get("path") == verification_receipt["path"]
            and verification_source.get("sha256") == verification_receipt["sha256"]
        ),
        "program_state_schema": program_state.get("schema") == PROGRAM_STATE_SCHEMA,
        "program_state_current_self_seal": _valid_seal(program_state, "state_sha256"),
        "base_capsules_complete": (
            isinstance(program_state.get("capsules"), dict)
            and bool(program_state["capsules"])
            and all(
                not isinstance(row, dict) or row.get("source") != "base" or row.get("status") == "complete"
                for row in program_state["capsules"].values()
            )
        ),
        "source_receipts_constructible": all(
            isinstance(receipt.get("sha256"), str)
            for receipt in (corpus_receipt, verification_receipt, report_receipt)
        ),
        "program_state_path_present": program_state_path.is_file(),
    }


REQUIRED_CORPUS_VERIFICATION_CHECKS = frozenset(
    {
        "corpus_schema",
        "corpus_self_hash",
        "config_schema",
        "config_hash_bound",
        "experiment_set_exact",
        "seed_set_exact",
        "all_attempt_receipts_valid",
        "all_cell_authorities_valid",
        "all_mutations_rejected",
        "all_seed_receipts_valid",
        "corpus_complete",
        "directional_inference_fail_closed",
        "full_regeneration_match",
        "independent_summary_match",
        "no_pseudoreplication",
        "promotion_blocked",
        "seed_authority_exact",
    }
)


def _synthesis_checks(
    *,
    corpus_path: Path,
    corpus: Mapping[str, Any],
    corpus_verification_path: Path,
    corpus_verification: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
    program_state_path: Path,
    program_state: Mapping[str, Any],
    synthesis: Mapping[str, Any],
) -> dict[str, bool]:
    source_checks = _source_checks(
        corpus_path=corpus_path,
        corpus=corpus,
        corpus_verification_path=corpus_verification_path,
        corpus_verification=corpus_verification,
        report_path=report_path,
        report=report,
        program_state_path=program_state_path,
        program_state=program_state,
    )
    expected_trace_index = _recompute_trace_index(corpus)
    expected_recurrence = _recompute_recurrence(expected_trace_index)
    expected_negative = _recompute_negative_register(expected_trace_index, report)
    expected_runtime = _recompute_current_base_runtime(program_state)
    expected_matrix = _expected_dimension_matrix(corpus)
    expected_priority = _expected_priority_queue(expected_trace_index)
    sources = synthesis.get("sources")
    sources = sources if isinstance(sources, dict) else {}
    static_receipts = {
        "corpus": _source_receipt(corpus_path, corpus, "corpus_sha256"),
        "corpus_verification": _source_receipt(
            corpus_verification_path,
            corpus_verification,
            "verification_sha256",
        ),
        "empirical_report": _source_receipt(report_path, report, "report_sha256"),
    }
    verification_checks = corpus_verification.get("checks")
    verification_check_names = set(verification_checks) if isinstance(verification_checks, dict) else set()
    base_runtime = synthesis.get("base_runtime_accounting")
    runtime_core = dict(base_runtime) if isinstance(base_runtime, dict) else {}
    declared_runtime_sha = runtime_core.pop("base_runtime_sha256", None)
    checks = {
        **{f"source_{name}": passed for name, passed in source_checks.items()},
        "synthesis_schema": synthesis.get("schema") == SYNTHESIS_SCHEMA,
        "synthesis_self_seal": _valid_seal(synthesis, "synthesis_sha256"),
        "top_level_contract_exact": set(synthesis) == SYNTHESIS_KEYS,
        "claim_scope_exact": synthesis.get("claim_scope") == SYNTHESIS_CLAIM_SCOPE,
        "source_receipts_exact": (
            set(sources) == {"corpus", "corpus_verification", "empirical_report", "program_state_snapshot"}
            and all(
                _static_source_matches(sources.get(name), receipt)
                for name, receipt in static_receipts.items()
            )
            and _program_source_matches(sources.get("program_state_snapshot"), program_state_path)
        ),
        "required_corpus_verification_checks_present": (
            verification_check_names >= REQUIRED_CORPUS_VERIFICATION_CHECKS
        ),
        "trace_index_exact": synthesis.get("per_experiment_trace_index") == expected_trace_index,
        "recurrence_analysis_exact": synthesis.get("recurrence_analysis") == expected_recurrence,
        "negative_evidence_register_exact": synthesis.get("negative_evidence_register") == expected_negative,
        "claim_boundaries_exact": synthesis.get("claim_boundaries") == _expected_claim_boundaries(),
        "claim_boundaries_valid": _claim_boundaries_valid(synthesis.get("claim_boundaries")),
        "mechanism_dimension_matrix_exact": synthesis.get("mechanism_dimension_matrix") == expected_matrix,
        "mechanism_dimension_matrix_valid": _mechanism_matrix_valid(
            synthesis.get("mechanism_dimension_matrix")
        ),
        "conditional_priority_queue_exact": synthesis.get("conditional_priority_queue") == expected_priority,
        "conditional_priority_queue_valid": _priority_queue_valid(
            synthesis.get("conditional_priority_queue")
        ),
        "base_runtime_accounting_exact": base_runtime == expected_runtime,
        "base_runtime_subset_self_hash": (
            isinstance(declared_runtime_sha, str) and declared_runtime_sha == _canonical_sha256(runtime_core)
        ),
        "promotion_and_activation_blocked": (
            synthesis.get("activation_allowed") is False and synthesis.get("scientific_promotion") is False
        ),
    }
    return checks


def _reseal_synthesis(payload: Mapping[str, Any]) -> dict[str, Any]:
    core = dict(payload)
    core.pop("synthesis_sha256", None)
    return {**core, "synthesis_sha256": _canonical_sha256(core)}


def _mutation_suite(
    *,
    corpus_path: Path,
    corpus: Mapping[str, Any],
    corpus_verification_path: Path,
    corpus_verification: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
    program_state_path: Path,
    program_state: Mapping[str, Any],
    synthesis: Mapping[str, Any],
) -> dict[str, bool]:
    def rejected(mutated: dict[str, Any]) -> bool:
        sealed = _reseal_synthesis(mutated)
        checks = _synthesis_checks(
            corpus_path=corpus_path,
            corpus=corpus,
            corpus_verification_path=corpus_verification_path,
            corpus_verification=corpus_verification,
            report_path=report_path,
            report=report,
            program_state_path=program_state_path,
            program_state=program_state,
            synthesis=sealed,
        )
        return not all(checks.values())

    mutations: dict[str, dict[str, Any]] = {}

    trace_removed = copy.deepcopy(dict(synthesis))
    traces = trace_removed.get("per_experiment_trace_index")
    if isinstance(traces, list) and traces:
        traces.pop()
    mutations["trace_membership_removed"] = trace_removed

    dimension_removed = copy.deepcopy(dict(synthesis))
    matrix = dimension_removed.get("mechanism_dimension_matrix")
    if isinstance(matrix, list) and matrix and isinstance(matrix[0], dict):
        dimensions = matrix[0].get("dimensions")
        if isinstance(dimensions, dict):
            dimensions.pop(REQUESTED_DIMENSIONS[0], None)
    mutations["dimension_removed"] = dimension_removed

    status_escalated = copy.deepcopy(dict(synthesis))
    matrix = status_escalated.get("mechanism_dimension_matrix")
    if isinstance(matrix, list):
        successor = next(
            (row for row in matrix if isinstance(row, dict) and row.get("mechanism_id") == "G1-C1"),
            None,
        )
        dimensions = successor.get("dimensions") if isinstance(successor, dict) else None
        cell = dimensions.get("biological_plausibility") if isinstance(dimensions, dict) else None
        if isinstance(cell, dict):
            cell["status"] = "measured"
    mutations["successor_dimension_status_escalated"] = status_escalated

    source_drifted = copy.deepcopy(dict(synthesis))
    sources = source_drifted.get("sources")
    corpus_source = sources.get("corpus") if isinstance(sources, dict) else None
    if isinstance(corpus_source, dict):
        corpus_source["sha256"] = "0" * 64
    mutations["source_receipt_drifted"] = source_drifted

    activation_enabled = copy.deepcopy(dict(synthesis))
    activation_enabled["activation_allowed"] = True
    mutations["activation_enabled"] = activation_enabled

    promotion_enabled = copy.deepcopy(dict(synthesis))
    promotion_enabled["scientific_promotion"] = True
    mutations["scientific_promotion_enabled"] = promotion_enabled

    claim_scope_escalated = copy.deepcopy(dict(synthesis))
    claim_scope_escalated["claim_scope"] = "integrated substrate advantage proven"
    mutations["claim_scope_escalated"] = claim_scope_escalated

    undeclared_claim_added = copy.deepcopy(dict(synthesis))
    undeclared_claim_added["substrate_formed"] = True
    mutations["undeclared_top_level_claim_added"] = undeclared_claim_added

    return {name: rejected(payload) for name, payload in sorted(mutations.items())}


def verify_evidence_synthesis(
    corpus_path: Path,
    corpus_verification_path: Path,
    report_path: Path,
    program_state_path: Path,
    synthesis_path: Path,
    *,
    recorded_at: str | None = None,
) -> dict[str, Any]:

    corpus_path = corpus_path.resolve()
    corpus_verification_path = corpus_verification_path.resolve()
    report_path = report_path.resolve()
    program_state_path = program_state_path.resolve()
    synthesis_path = synthesis_path.resolve()
    corpus = _load_object(corpus_path, "Generation-1 corpus")
    corpus_verification = _load_object(corpus_verification_path, "Generation-1 corpus verification")
    report = _load_object(report_path, "Generation-1 empirical report")
    program_state = _load_object(program_state_path, "Generation-1 program state")
    synthesis = _load_object(synthesis_path, "Generation-1 evidence synthesis")
    checks = _synthesis_checks(
        corpus_path=corpus_path,
        corpus=corpus,
        corpus_verification_path=corpus_verification_path,
        corpus_verification=corpus_verification,
        report_path=report_path,
        report=report,
        program_state_path=program_state_path,
        program_state=program_state,
        synthesis=synthesis,
    )
    mutation_results = _mutation_suite(
        corpus_path=corpus_path,
        corpus=corpus,
        corpus_verification_path=corpus_verification_path,
        corpus_verification=corpus_verification,
        report_path=report_path,
        report=report,
        program_state_path=program_state_path,
        program_state=program_state,
        synthesis=synthesis,
    )
    checks["all_synthesis_mutations_rejected"] = all(mutation_results.values())
    problems = sorted(name for name, passed in checks.items() if not passed)
    core = {
        "schema": VERIFICATION_SCHEMA,
        "claim_scope": (
            "independent verification of the derived G1-C0 evidence synthesis only; "
            "no mechanism or substrate promotion"
        ),
        "sources": {
            "corpus": _source_receipt(corpus_path, corpus, "corpus_sha256"),
            "corpus_verification": _source_receipt(
                corpus_verification_path,
                corpus_verification,
                "verification_sha256",
            ),
            "empirical_report": _source_receipt(report_path, report, "report_sha256"),
            "synthesis": _source_receipt(synthesis_path, synthesis, "synthesis_sha256"),
            "program_state": {
                "path": _repo_path(program_state_path),
                "schema": program_state.get("schema"),
                "base_runtime_sha256": _recompute_current_base_runtime(program_state).get(
                    "base_runtime_sha256"
                ),
            },
        },
        "checks": checks,
        "mutation_suite": {
            "count": len(mutation_results),
            "rejected": sum(mutation_results.values()),
            "results": mutation_results,
        },
        "verification_complete": not problems,
        "problems": problems,
        "activation_allowed": False,
        "scientific_promotion": False,
        "recorded_at": recorded_at or datetime.now(UTC).isoformat(),
    }
    return {**core, "verification_sha256": _canonical_sha256(core)}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--verification", type=Path, default=DEFAULT_CORPUS_VERIFICATION)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--program-state", type=Path, default=DEFAULT_PROGRAM_STATE)
    parser.add_argument("--synthesis", type=Path, default=DEFAULT_SYNTHESIS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = verify_evidence_synthesis(
        arguments.corpus,
        arguments.verification,
        arguments.report,
        arguments.program_state,
        arguments.synthesis,
    )
    _atomic_json(arguments.out.resolve(), result)
    print(
        json.dumps(
            {
                "verification": str(arguments.out),
                "verification_complete": result["verification_complete"],
                "problems": result["problems"],
                "activation_allowed": False,
                "scientific_promotion": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["verification_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
