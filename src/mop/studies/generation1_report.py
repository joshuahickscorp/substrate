"""Synthesize the Generation-1 corpus with the evidence that motivated it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from .generation1_cognitive_corpus import (
    CORPUS_SCHEMA,
    canonical_bytes,
    canonical_sha256,
)
from .generation1_cognitive_corpus_verify import VERIFICATION_SCHEMA

SCHEMA = "mop-generation1-empirical-report/v2"
DEFAULT_CORPUS = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.json"
DEFAULT_VERIFICATION = REPO_ROOT / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
DEFAULT_OUTPUT = REPO_ROOT / "proof/GENERATION1_EMPIRICAL_REPORT.json"
DEFAULT_TEXT = REPO_ROOT / "runs/generation1/GENERATION1_EMPIRICAL_REPORT.txt"
RESOURCE_CANARY = "proof/GENERATION1_RESOURCE_CANARY.json"
RESOURCE_CANARY_SCHEMA = "mop-generation1-resource-canary/v1"
MECHANICS_ARTIFACTS = (
    ("x1_invalid_bed", "proof/ESCS_X1_DISPATCH.json"),
    ("x1_invalid_bed_verification", "proof/ESCS_X1_DISPATCH.verification.json"),
    ("ecology", "runs/generation1/mechanics/ecology.json"),
    ("ecology_verification", "runs/generation1/mechanics/ecology.verification.json"),
    ("integrity", "runs/generation1/mechanics/integrity.json"),
    ("integrity_verification", "runs/generation1/mechanics/integrity.verification.json"),
    ("material_twin", "runs/generation1/mechanics/material_twin.json"),
    (
        "material_twin_verification",
        "runs/generation1/mechanics/material_twin.verification.json",
    ),
    ("broadcast", "runs/generation1/mechanics/broadcast.json"),
    ("broadcast_verification", "runs/generation1/mechanics/broadcast.verification.json"),
    ("sensing", "runs/generation1/mechanics/sensing.json"),
    ("sensing_verification", "runs/generation1/mechanics/sensing.verification.json"),
    ("g0_formation", "runs/generation1/mechanics/g0_formation.json"),
    (
        "g0_formation_verification",
        "runs/generation1/mechanics/g0_formation.verification.json",
    ),
    ("p7_action_world", "runs/generation1/mechanics/p7_action_world.json"),
    ("p9_causal_monitor", "runs/generation1/mechanics/p9_causal_monitor.json"),
)


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "sha256": _sha_file(path),
        "bytes": path.stat().st_size,
    }


def _valid_seal(payload: dict[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == canonical_sha256(core)


def _pack_memberships(corpus: dict[str, Any]) -> dict[str, list[str]]:
    memberships: dict[str, list[str]] = {}
    packs = corpus.get("capability_pack_summaries", {})
    if not isinstance(packs, dict):
        return memberships
    for pack, raw in packs.items():
        if not isinstance(raw, dict):
            continue
        for experiment_id in raw.get("experiment_ids", []):
            if isinstance(experiment_id, str):
                memberships.setdefault(experiment_id, []).append(str(pack))
    for values in memberships.values():
        values.sort()
    return memberships


def _atlas_row(
    experiment_id: str,
    row: dict[str, Any],
    memberships: dict[str, list[str]],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "classification": row.get("classification"),
        "evidence_class": row.get("evidence_class", "inferential"),
        "seed_mode": row.get("seed_mode", "legacy_unspecified"),
        "completed_seed_count": row.get("completed_seed_count"),
        "effective_observation_count": row.get(
            "effective_observation_count", row.get("completed_seed_count")
        ),
        "distinct_seed_authority_count": row.get("distinct_seed_authority_count"),
        "variation_canary": row.get("variation_canary"),
        "null_supported": row.get("null_supported"),
        "boolean_rates": row.get("boolean_rates", {}),
        "numeric_summaries": row.get("numeric_summaries", {}),
        "capability_packs": memberships.get(experiment_id, []),
        "contract": row.get("contract", {}),
    }


def _build_atlases(
    summaries: dict[str, Any],
    memberships: dict[str, list[str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    atlases: dict[str, list[dict[str, Any]]] = {}
    for experiment_id, raw in sorted(summaries.items()):
        if not isinstance(raw, dict):
            continue
        classification = str(raw.get("classification", "unclassified"))
        atlases.setdefault(classification, []).append(
            _atlas_row(experiment_id, raw, memberships)
        )
    counts = {key: len(rows) for key, rows in sorted(atlases.items())}
    return dict(sorted(atlases.items())), counts


def _pack_profiles(
    corpus: dict[str, Any],
    summaries: dict[str, Any],
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    packs = corpus.get("capability_pack_summaries", {})
    if not isinstance(packs, dict):
        return profiles
    for pack, raw in sorted(packs.items()):
        if not isinstance(raw, dict):
            continue
        ids = [item for item in raw.get("experiment_ids", []) if isinstance(item, str)]
        by_classification: dict[str, list[str]] = {}
        by_evidence_class: dict[str, list[str]] = {}
        for experiment_id in ids:
            row = summaries.get(experiment_id, {})
            if not isinstance(row, dict):
                continue
            by_classification.setdefault(
                str(row.get("classification", "unclassified")), []
            ).append(experiment_id)
            by_evidence_class.setdefault(
                str(row.get("evidence_class", "inferential")), []
            ).append(experiment_id)
        profiles.append(
            {
                "capability_pack": pack,
                "experiment_count": len(ids),
                "classification_members": dict(sorted(by_classification.items())),
                "evidence_class_members": dict(sorted(by_evidence_class.items())),
                "source_summary": raw,
            }
        )
    return profiles


def _innovation_traces(
    atlases: dict[str, list[dict[str, Any]]],
    pack_profiles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    candidates = atlases.get("stable_candidate_trace", [])
    cross_pack = [
        row["experiment_id"] for row in candidates if len(row.get("capability_packs", [])) > 1
    ]
    if cross_pack:
        traces.append(
            {
                "trace": "cross_pack_candidate_stability",
                "experiment_ids": cross_pack,
                "interpretation": (
                    "candidate traces recur in more than one analyst-defined capability pack; "
                    "causal reuse and complementarity remain untested"
                ),
            }
        )
    dense_packs = []
    for profile in pack_profiles:
        ids = profile["classification_members"].get("stable_candidate_trace", [])
        if len(ids) >= 2:
            dense_packs.append(
                {"capability_pack": profile["capability_pack"], "experiment_ids": ids}
            )
    if dense_packs:
        traces.append(
            {
                "trace": "candidate_dense_capability_packs",
                "packs": dense_packs,
                "interpretation": (
                    "several independently registered experiments expose candidate traces in the "
                    "same broad pack; explicit context-disjoint niche tests are still required"
                ),
            }
        )
    stable_nulls = atlases.get("stable_null", [])
    if stable_nulls:
        traces.append(
            {
                "trace": "repeated_control_or_null_dominance",
                "experiment_ids": [row["experiment_id"] for row in stable_nulls],
                "interpretation": (
                    "these valid inferential rows repeatedly retained their registered null; "
                    "larger scaling is not licensed without a changed mechanism"
                ),
            }
        )
    mixed = atlases.get("mixed_or_seed_sensitive", [])
    if mixed:
        traces.append(
            {
                "trace": "seed_sensitive_context_candidates",
                "experiment_ids": [row["experiment_id"] for row in mixed],
                "interpretation": (
                    "direction varies across effective seed authorities; the next test should "
                    "identify a moderating task or context variable"
                ),
            }
        )
    invariant = atlases.get("descriptive_seed_invariant", [])
    if invariant:
        traces.append(
            {
                "trace": "scientific_output_invariance",
                "experiment_ids": [row["experiment_id"] for row in invariant],
                "interpretation": (
                    "effective seed authorities changed while the registered scientific payload "
                    "did not; this may indicate saturation, a coarse metric, or a genuinely "
                    "deterministic boundary and licenses a targeted difficulty or measurement epoch"
                ),
            }
        )
    adapter_failures = atlases.get("descriptive_seed_adapter_unverified", [])
    if adapter_failures:
        traces.append(
            {
                "trace": "seed_authority_instrumentation_gap",
                "experiment_ids": [row["experiment_id"] for row in adapter_failures],
                "interpretation": (
                    "these rows cannot vote scientifically until every consumed stochastic path "
                    "and receipt binding survives the independent authority audit"
                ),
            }
        )
    return traces


def _mechanics_evidence() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evidence_id, relative in MECHANICS_ARTIFACTS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ValueError(f"required Generation-1 mechanics artifact is missing: {relative}")
        payload = _load(path)
        rows.append(
            {
                "evidence_id": evidence_id,
                "source": _source(path),
                "schema": payload.get("schema"),
                "status": payload.get("status", payload.get("execution_status")),
                "terminal_route": payload.get("aggregate", {}).get("terminal_route")
                if isinstance(payload.get("aggregate"), dict)
                else payload.get("terminal_route"),
                "verified": payload.get("verified"),
                "all_ok": payload.get("all_ok"),
                "all_mechanics_ok": payload.get("all_mechanics_ok"),
                "all_mutations_rejected": payload.get("all_mutations_rejected"),
                "scientific_capability_claim": payload.get("scientific_capability_claim"),
                "scientific_promotion": payload.get(
                    "scientific_promotion", payload.get("scientific_promotion_allowed")
                ),
            }
        )
    return rows


def _resource_evidence() -> dict[str, Any]:
    path = REPO_ROOT / RESOURCE_CANARY
    if not path.is_file():
        raise ValueError(f"required Generation-1 resource canary is missing: {RESOURCE_CANARY}")
    payload = _load(path)
    if (
        payload.get("schema") != RESOURCE_CANARY_SCHEMA
        or not _valid_seal(payload, "receipt_sha256")
        or payload.get("complete") is not True
        or payload.get("scientific_promotion") is not False
        or payload.get("orchestration_problems") != []
    ):
        raise ValueError("Generation-1 resource canary is incomplete or invalid")
    recommendation = payload.get("recommendation")
    measurements = payload.get("measurements")
    source_authority = payload.get("source_authority")
    if (
        not isinstance(recommendation, dict)
        or recommendation.get("eligible") is not True
        or not isinstance(measurements, dict)
        or measurements.get("runtime_safety_problems") != []
        or not isinstance(source_authority, dict)
        or source_authority.get("stable") is not True
    ):
        raise ValueError("Generation-1 resource canary lacks safe stable authority")
    before = source_authority.get("before")
    if not isinstance(before, dict) or not isinstance(
        before.get("aggregate_sha256"), str
    ):
        raise ValueError("Generation-1 resource canary source digest is missing")
    return {
        "source": _source(path),
        "run_id": payload.get("run_id"),
        "wall_seconds": payload.get("wall_seconds"),
        "outcome_counts": payload.get("outcome_counts"),
        "measurements": measurements,
        "recommendation": recommendation,
        "source_authority_sha256": before["aggregate_sha256"],
        "scientific_promotion": False,
    }


def build_report(corpus_path: Path, verification_path: Path) -> dict[str, Any]:
    corpus = _load(corpus_path)
    verification = _load(verification_path)
    if corpus.get("schema") != CORPUS_SCHEMA or not _valid_seal(corpus, "corpus_sha256"):
        raise ValueError("Generation-1 cognitive corpus schema or self-seal is invalid")
    if verification.get("schema") != VERIFICATION_SCHEMA or not _valid_seal(
        verification, "verification_sha256"
    ):
        raise ValueError("Generation-1 cognitive verification schema or self-seal is invalid")
    if verification.get("verification_complete") is not True or verification.get("problems") != []:
        raise ValueError("Generation-1 cognitive corpus is not independently verified")
    verification_corpus = verification.get("corpus", {})
    if not isinstance(verification_corpus, dict) or verification_corpus.get("sha256") != _sha_file(
        corpus_path
    ):
        raise ValueError("Generation-1 verification is not bound to the supplied corpus bytes")
    verification_checks = verification.get("checks", {})
    if not isinstance(verification_checks, dict):
        raise ValueError("Generation-1 verification checks are missing")
    required_checks = {
        "corpus_complete",
        "full_regeneration_match",
        "all_mutations_rejected",
        "all_seed_receipts_valid",
        "promotion_blocked",
        "directional_inference_fail_closed",
    }
    failed_required = sorted(
        key for key in required_checks if verification_checks.get(key) is not True
    )
    if failed_required:
        raise ValueError(f"Generation-1 verification lacks critical checks: {failed_required}")
    summaries = corpus.get("experiment_summaries")
    if not isinstance(summaries, dict):
        raise ValueError("Generation-1 cognitive corpus summaries are missing")
    memberships = _pack_memberships(corpus)
    atlases, classification_counts = _build_atlases(summaries, memberships)
    pack_profiles = _pack_profiles(corpus, summaries)
    innovation_traces = _innovation_traces(atlases, pack_profiles)
    advanced_check_names = (
        "all_attempt_receipts_valid",
        "all_cell_authorities_valid",
        "seed_authority_exact",
        "no_pseudoreplication",
        "independent_summary_match",
        "directional_inference_fail_closed",
    )
    advanced_checks = {
        name: verification_checks.get(name)
        for name in advanced_check_names
        if name in verification_checks
    }
    seed_authority_summary = corpus.get("seed_authority_summary")
    if not isinstance(seed_authority_summary, dict):
        seed_authority_summary = {}
    readiness_checks = {
        "corpus_complete": corpus.get("corpus_complete") is True,
        "verification_complete": verification.get("verification_complete") is True,
        "base_verifier_checks": not failed_required,
        "advanced_verifier_checks": bool(advanced_checks)
        and all(value is True for value in advanced_checks.values()),
        "seed_authority_summary_present": bool(seed_authority_summary),
        "seed_authority_structurally_independent": seed_authority_summary.get(
            "no_pseudoreplication"
        )
        is True,
    }
    ready_to_preregister = all(readiness_checks.values())

    x0_path = REPO_ROOT / "proof/ESCS_X0_EVENT_FORMATION.verification.json"
    edcm_path = REPO_ROOT / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json"
    p6_path = REPO_ROOT / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json"
    p7_path = REPO_ROOT / "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json"
    x0 = _load(x0_path)
    edcm = _load(edcm_path)
    p6 = _load(p6_path)
    p7 = _load(p7_path)
    x0_intervals = x0["fresh_aggregate"]["paired_intervals_95"]
    edcm_gate = edcm["aggregate"]["gate"]
    p6_decision = p6["independent_recompute"]["decision"]
    p6_contrasts = {
        f"{row['schedule']}:{row['control']}": {
            "retention_mean_delta": row["retention_mean_delta"],
            "future_first_window_mean_delta": row["future_first_window_mean_delta"],
            "strict_joint_gain": row["strict_joint_gain"],
        }
        for row in p6_decision["contrasts"]
    }

    core = {
        "schema": SCHEMA,
        "claim_scope": (
            "programmatic Generation-1 capability corpus and hypothesis-selection evidence only; "
            "no integrated architecture or natural-world promotion"
        ),
        "created_at": datetime.now(UTC).isoformat(),
        "corpus": {
            "source": _source(corpus_path),
            "verification": _source(verification_path),
            "seed_count": corpus["seed_count"],
            "eligible_experiment_count": corpus["eligible_experiment_count"],
            "complete_experiment_count": corpus["complete_experiment_count"],
            "corpus_complete": corpus["corpus_complete"],
            "classification_counts": classification_counts,
            "atlases": atlases,
            "capability_packs": corpus["capability_pack_summaries"],
            "capability_pack_profiles": pack_profiles,
            "seed_authority_summary": corpus.get("seed_authority_summary", {}),
            "operational_summary": corpus.get(
                "operational_summary",
                {
                    "total_manifest_bytes": corpus.get("total_manifest_bytes"),
                    "note": "legacy corpus did not expose full attempt/resource accounting",
                },
            ),
            "verification_checks": verification_checks,
        },
        "generation0_constraints": {
            "event_formation": {
                "verdict": x0["fresh_aggregate"]["verdict"],
                "work_saving_mean": x0_intervals["work_saving_vs_always_on"]["mean"],
                "utility_loss_mean": x0_intervals["utility_loss_vs_always_on"]["mean"],
                "required_direction_every_seed": x0["fresh_aggregate"]["checks"][
                    "required_direction_every_seed"
                ],
                "successor_constraint": (
                    "cheap high-recall reflex followed by temporal/cross-sensor relational "
                    "adjudication and a value-of-compute veto"
                ),
            },
            "complementarity": {
                "verdict": edcm["aggregate"]["verdict"],
                "niche_advantage_95": edcm_gate["niche_advantage_95"],
                "oracle_headroom_95": edcm_gate["oracle_headroom_95"],
                "verifier_disagreement_gain_95": edcm_gate["verifier_disagreement_gain_95"],
                "verifier_agreement_effect_95": edcm_gate["verifier_agreement_effect_95"],
                "successor_constraint": (
                    "competence maps, contextual abstention, dormant actors, and "
                    "disagreement-triggered verification"
                ),
            },
            "continual_learning": {
                "verdict": p6_decision["verdict"],
                "next_rung_allowed": p6["prerequisite"]["next_rung_allowed"],
                "contrasts": p6_contrasts,
                "successor_constraint": (
                    "stable core plus transient fresh actors and selective consolidation; "
                    "replay alone remains retired"
                ),
            },
            "planning": {
                "status": p7["status"],
                "all_mechanics_ok": p7["all_mechanics_ok"],
                "claim_boundary": p7["claim_boundary"],
                "successor_constraint": (
                    "prediction quality cannot satisfy a planning gate without realized "
                    "intervention or action benefit"
                ),
            },
        },
        "fresh_mechanics_evidence": _mechanics_evidence(),
        "resource_authority": _resource_evidence(),
        "empirical_innovation_traces": innovation_traces,
        "successor_hypotheses": [
            {
                "id": "G1-E1",
                "name": "reflex-relational event cascade",
                "evidence_status": "licensed_hypothesis_not_positive",
            },
            {
                "id": "G1-D1",
                "name": "competence-mapped sparse coalition with abstention",
                "evidence_status": "licensed_hypothesis_not_positive",
            },
            {
                "id": "G1-V1",
                "name": "disagreement-triggered verification",
                "evidence_status": "licensed_hypothesis_not_positive",
            },
            {
                "id": "G1-PL1",
                "name": "stable-transient plastic ecology",
                "evidence_status": "licensed_hypothesis_not_positive",
            },
            {
                "id": "G1-M1",
                "name": "exact episodic ledger plus rebuildable semantic projections",
                "evidence_status": "licensed_hypothesis_not_positive",
            },
        ],
        "competence_tensor_proposal": {
            "status": "schema_proposal_not_measured_tensor",
            "axes": [
                "task_family",
                "context",
                "difficulty",
                "distribution_shift",
                "actor",
                "effective_seed_authority",
            ],
            "measures": [
                "quality",
                "abstention",
                "calibration",
                "latency",
                "charged_compute",
                "memory",
                "verification_value",
            ],
            "construction_rule": (
                "populate only from explicit actor-context interventions; corpus pack membership "
                "alone does not establish actor competence"
            ),
        },
        "complementarity_shortlist": [
            {
                "capability_pack": profile["capability_pack"],
                "candidate_experiment_ids": profile["classification_members"].get(
                    "stable_candidate_trace", []
                ),
                "status": "requires_context_disjoint_niche_and_oracle_headroom_test",
            }
            for profile in pack_profiles
            if profile["classification_members"].get("stable_candidate_trace")
        ],
        "next_authority": {
            "ready_to_preregister_mechanism_epoch": ready_to_preregister,
            "readiness_checks": readiness_checks,
            "advanced_verifier_checks": advanced_checks,
            "ready_to_activate_or_integrate_substrate": False,
            "automatic_activation_allowed": False,
            "automatic_scientific_promotion_allowed": False,
            "required_order": [
                "difficulty_and_complementarity_atlas",
                "G1-E1_event_cascade",
                "G1-D1_competence_dispatch",
                "G1-PL1_plastic_ecology",
                "G1-M1_dual_memory",
                "planning_causal_social_workspace_security_lanes",
                "topology_only_after_upstream_positives",
                "integration_only_if_earned",
            ],
        },
        "sources": [_source(path) for path in (x0_path, edcm_path, p6_path, p7_path)],
        "scientific_promotion": False,
    }
    return {**core, "report_sha256": canonical_sha256(core)}


def render_text(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    lines = [
        "MOP GENERATION 1 EMPIRICAL CORPUS REPORT",
        "",
        f"Corpus complete: {corpus['corpus_complete']}",
        f"Outer seeds: {corpus['seed_count']}",
        f"Locally runnable experiment classes: {corpus['eligible_experiment_count']}",
        f"Experiments meeting frozen coverage: {corpus['complete_experiment_count']}",
        "",
        "Classification counts:",
    ]
    for name, count in corpus["classification_counts"].items():
        lines.append(f"  {name}: {count}")
    lines.extend(["", "Seed authority:"])
    seed_authority = corpus.get("seed_authority_summary", {})
    if seed_authority:
        for name, value in sorted(seed_authority.items()):
            if isinstance(value, (str, int, float, bool)) or value is None:
                lines.append(f"  {name}: {value}")
    else:
        lines.append("  unavailable")
    lines.extend(["", "Generation-0 constraints carried forward:"])
    for name, row in report["generation0_constraints"].items():
        lines.append(f"  {name}: {row['successor_constraint']}")
    lines.extend(["", "Fresh mechanics evidence:"])
    for row in report["fresh_mechanics_evidence"]:
        outcome = row.get("terminal_route") or row.get("status")
        if outcome is None and row.get("verified") is not None:
            outcome = f"verified={row['verified']}"
        lines.append(f"  {row['evidence_id']}: {outcome}")
    resource = report["resource_authority"]
    recommendation = resource["recommendation"]
    measurements = resource["measurements"]
    lines.extend(
        [
            "",
            "Measured resource authority:",
            f"  run_id: {resource['run_id']}",
            "  recommended_max_workers: "
            f"{recommendation['recommended_max_workers']}",
            "  recommended_unified_memory_gb: "
            f"{recommendation['recommended_estimated_unified_memory_gb']}",
            "  observed_aggregate_process_tree_peak_rss_bytes: "
            f"{measurements['aggregate_process_tree_peak_rss_bytes']}",
        ]
    )
    lines.extend(["", "Empirical innovation traces (not causal positives):"])
    for row in report["empirical_innovation_traces"]:
        lines.append(f"  {row['trace']}: {row['interpretation']}")
    if not report["empirical_innovation_traces"]:
        lines.append("  none")
    lines.extend(["", "Successor hypotheses (not positives):"])
    for row in report["successor_hypotheses"]:
        lines.append(f"  {row['id']}: {row['name']}")
    lines.extend(
        [
            "",
            "Next-authority readiness:",
            "  ready_to_preregister_mechanism_epoch: "
            f"{report['next_authority']['ready_to_preregister_mechanism_epoch']}",
            "  ready_to_activate_or_integrate_substrate: false",
        ]
    )
    lines.extend(
        [
            "",
            "Interpretation:",
            "  This corpus selects and falsifies successor hypotheses. It does not establish an",
            "  integrated substrate, natural-world generality, biological equivalence, or a",
            "  scientific intelligence-per-compute advantage.",
            "",
            f"Report SHA256: {report['report_sha256']}",
        ]
    )
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--verification", type=Path, default=DEFAULT_VERIFICATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-out", type=Path, default=DEFAULT_TEXT)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    report = build_report(arguments.corpus.resolve(), arguments.verification.resolve())
    _atomic(arguments.out.resolve(), canonical_bytes(report) + b"\n")
    _atomic(arguments.text_out.resolve(), render_text(report).encode("utf-8"))
    print(
        json.dumps(
            {
                "corpus_complete": report["corpus"]["corpus_complete"],
                "report": str(arguments.out),
                "text_report": str(arguments.text_out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
