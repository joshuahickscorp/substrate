from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mop.studies import generation1_evidence_synthesis as synthesis
from mop.studies import generation1_evidence_synthesis_verify as synthesis_verify


@dataclass(frozen=True)
class EvidencePaths:
    corpus: Path
    verification: Path
    report: Path
    program_state: Path
    synthesis: Path


def _seal(payload: dict[str, Any], field: str) -> dict[str, Any]:
    core = {key: value for key, value in payload.items() if key != field}
    return {**core, field: synthesis.canonical_sha256(core)}


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(synthesis.canonical_bytes(payload) + b"\n")


def _source(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _runtime(*, peak_rss: int, retries: int = 0) -> dict[str, Any]:
    return {
        "event_count": 3,
        "events": [
            {"event": "admitted"},
            {"event": "sampled"},
            {"event": "completed"},
        ],
        "events_dropped": 0,
        "maximum_swap_used_gb": 0.0,
        "minimum_disk_free_gb": 200.0,
        "minimum_memory_available_gb": 70.0,
        "minimum_memory_available_percent": 70.0,
        "minimum_memory_pressure_free_percent": 90.0,
        "peak_process_tree_rss_bytes": peak_rss,
        "reservation_count": 1,
        "resource_stop_count": 0,
        "retry_count": retries,
        "sample_count": 4,
        "safety_state": "complete",
        "thermal_statuses": ["normal"],
        "power_sources": ["AC Power"],
    }


def _null_summary(*, null_count: int, candidate_count: int) -> dict[str, Any]:
    observations = null_count + candidate_count
    return {
        "observations": observations,
        "true": null_count,
        "false": candidate_count,
        "fraction": round(null_count / observations, 8),
        "wilson_95": synthesis._wilson_95(null_count, observations),
    }


def _write_valid_sources(root: Path) -> EvidencePaths:
    corpus_path = root / "proof/GENERATION1_COGNITIVE_CORPUS.json"
    verification_path = root / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
    report_path = root / "proof/GENERATION1_EMPIRICAL_REPORT.json"
    state_path = root / "runs/generation1/program_state.json"
    synthesis_path = root / "proof/GENERATION1_EVIDENCE_SYNTHESIS.json"

    corpus = _seal(
        {
            "schema": synthesis.CORPUS_SCHEMA,
            "campaign_id": "generation1-tiny-corpus",
            "claim_scope": "tiny synthetic screening fixture only",
            "seed_count": 3,
            "eligible_experiment_count": 3,
            "complete_experiment_count": 3,
            "corpus_complete": True,
            "experiment_summaries": {
                "candidate_alpha": {
                    "classification": "stable_candidate_trace",
                    "evidence_class": "inferential",
                    "seed_mode": "varied",
                    "effective_observation_count": 3,
                    "null_supported": _null_summary(null_count=0, candidate_count=3),
                    "contract": {"claim": "bounded candidate alpha"},
                },
                "candidate_beta": {
                    "classification": "stable_candidate_trace",
                    "evidence_class": "inferential",
                    "seed_mode": "varied",
                    "effective_observation_count": 3,
                    "null_supported": _null_summary(null_count=0, candidate_count=3),
                    "contract": {"claim": "bounded candidate beta"},
                },
                "stable_null": {
                    "classification": "stable_null",
                    "evidence_class": "inferential",
                    "seed_mode": "varied",
                    "effective_observation_count": 3,
                    "null_supported": _null_summary(null_count=3, candidate_count=0),
                    "contract": {"claim": "bounded stable null"},
                },
            },
            "capability_pack_summaries": {
                "episodic_semantic_working_memory": {
                    "experiment_ids": ["candidate_alpha", "candidate_beta", "stable_null"]
                },
                "planning_imagination_simulation": {"experiment_ids": ["candidate_alpha"]},
                "spatial_temporal_motor": {"experiment_ids": []},
                "uncertainty_calibration_selective_compute": {"experiment_ids": []},
                "communication_social_distributed_cognition": {"experiment_ids": []},
                "metacognition_verification_reflection": {"experiment_ids": []},
                "robustness_dynamics_contradiction": {"experiment_ids": []},
                "continual_learning_plasticity": {"experiment_ids": []},
                "causal_counterfactual_reasoning": {"experiment_ids": []},
                "curiosity_novelty_curriculum": {"experiment_ids": []},
                "topology_ecology_reorganization": {"experiment_ids": []},
            },
            "cell_authority_index": {
                "candidate_alpha": [
                    {
                        "implementation_authorities": [
                            {
                                "path": "src/alpha.py",
                                "role": "experiment_implementation",
                                "sha256": "a" * 64,
                            },
                            {
                                "path": "src/harness.py",
                                "role": "experiment_harness",
                                "sha256": "f" * 64,
                            },
                        ]
                    }
                ],
                "candidate_beta": [
                    {
                        "implementation_authorities": [
                            {
                                "path": "src/beta.py",
                                "role": "experiment_implementation",
                                "sha256": "b" * 64,
                            }
                        ]
                    }
                ],
                "stable_null": [
                    {
                        "implementation_authorities": [
                            {
                                "path": "src/null.py",
                                "role": "experiment_implementation",
                                "sha256": "c" * 64,
                            }
                        ]
                    }
                ],
            },
            "operational_summary": {"attempt_count": 9, "retry_count": 1},
            "scientific_promotion": False,
        },
        "corpus_sha256",
    )
    _write(corpus_path, corpus)

    checks = {
        "corpus_schema": True,
        "corpus_self_hash": True,
        "config_schema": True,
        "config_hash_bound": True,
        "experiment_set_exact": True,
        "seed_set_exact": True,
        "all_attempt_receipts_valid": True,
        "all_cell_authorities_valid": True,
        "all_mutations_rejected": True,
        "all_seed_receipts_valid": True,
        "corpus_complete": True,
        "directional_inference_fail_closed": True,
        "full_regeneration_match": True,
        "independent_summary_match": True,
        "no_pseudoreplication": True,
        "promotion_blocked": True,
        "seed_authority_exact": True,
    }
    verification = _seal(
        {
            "schema": synthesis.VERIFICATION_SCHEMA,
            "verification_complete": True,
            "problems": [],
            "checks": checks,
            "authority_audit": {
                "expected_effective_cell_count": 9,
                "selected_effective_cell_count": 9,
            },
            "corpus": {
                **_source(corpus_path),
                "corpus_sha256": corpus["corpus_sha256"],
            },
            "scientific_promotion": False,
        },
        "verification_sha256",
    )
    _write(verification_path, verification)

    report = _seal(
        {
            "schema": synthesis.REPORT_SCHEMA,
            "claim_scope": "tiny synthetic report fixture only",
            "corpus": {
                "source": _source(corpus_path),
                "verification": _source(verification_path),
                "corpus_complete": True,
            },
            "generation0_constraints": {
                "event_formation": {
                    "verdict": "null",
                    "successor_constraint": "relational-temporal redesign required",
                },
                "complementarity": {
                    "verdict": "invalid_bed",
                    "successor_constraint": "repair the actor niche bed",
                },
                "continual_learning": {
                    "verdict": "null",
                    "successor_constraint": "new plasticity mechanism required",
                },
                "planning": {
                    "status": "mechanics-pass",
                    "successor_constraint": "realized action value required",
                },
            },
            "next_authority": {
                "ready_to_preregister_mechanism_epoch": True,
                "ready_to_activate_or_integrate_substrate": False,
            },
            "scientific_promotion": False,
        },
        "report_sha256",
    )
    _write(report_path, report)

    program_state = _seal(
        {
            "schema": synthesis.STATE_SCHEMA,
            "program_id": "generation1-tiny-corpus",
            "status": "complete",
            "capsules": {
                "g1_cognitive_seed_fixture": {
                    "source": "base",
                    "kind": "corpus",
                    "status": "complete",
                    "attempts": 1,
                    "returncode": 0,
                    "runtime": _runtime(peak_rss=1_000_000_000),
                    "artifacts": [
                        {
                            "path": "proof/base-a.json",
                            "schema": "fixture-result/v1",
                            "sha256": "d" * 64,
                        }
                    ],
                },
                "g1_empirical_report": {
                    "source": "base",
                    "kind": "verifier",
                    "status": "complete",
                    "attempts": 2,
                    "returncode": 0,
                    "runtime": _runtime(peak_rss=2_000_000_000, retries=1),
                    "artifacts": [
                        {
                            "path": "proof/base-b.json",
                            "schema": "fixture-report/v1",
                            "sha256": "e" * 64,
                        }
                    ],
                },
                "injected-observer": {
                    "source": "injection:test",
                    "kind": "exploratory",
                    "status": "running",
                    "attempts": 1,
                    "returncode": None,
                    "runtime": _runtime(peak_rss=9_000_000_000),
                    "artifacts": [],
                },
            },
        },
        "state_sha256",
    )
    _write(state_path, program_state)
    return EvidencePaths(
        corpus=corpus_path,
        verification=verification_path,
        report=report_path,
        program_state=state_path,
        synthesis=synthesis_path,
    )


def _build(paths: EvidencePaths) -> dict[str, Any]:
    payload = synthesis.build_synthesis(
        paths.corpus,
        paths.verification,
        paths.report,
        paths.program_state,
    )
    _write(paths.synthesis, payload)
    return payload


def _verify(paths: EvidencePaths) -> dict[str, Any]:
    return synthesis_verify.verify_evidence_synthesis(
        paths.corpus,
        paths.verification,
        paths.report,
        paths.program_state,
        paths.synthesis,
    )


def _rewrite_synthesis(paths: EvidencePaths, payload: dict[str, Any]) -> None:
    _write(paths.synthesis, _seal(payload, "synthesis_sha256"))


def test_valid_tiny_corpus_report_and_state_build_and_verify_end_to_end(
    tmp_path: Path,
) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)

    assert payload["schema"] == synthesis.SCHEMA
    assert [row["experiment_id"] for row in payload["per_experiment_trace_index"]] == [
        "candidate_alpha",
        "candidate_beta",
        "stable_null",
    ]
    assert [
        row["experiment_ids"]
        for row in payload["negative_evidence_register"]
        if row["evidence_type"] == "corpus_classification"
    ] == [["stable_null"]]
    assert payload["base_runtime_accounting"]["base_capsule_count"] == 2
    assert payload["base_runtime_accounting"]["status_counts"] == {"complete": 2}
    assert payload["base_runtime_accounting"]["total_attempts"] == 3
    assert payload["base_runtime_accounting"]["total_retries"] == 1
    assert payload["base_runtime_accounting"]["peak_process_tree_rss_bytes"] == 2_000_000_000
    assert payload["activation_allowed"] is False
    assert payload["scientific_promotion"] is False
    assert len(payload["mechanism_dimension_matrix"]) == 16

    verification = _verify(paths)
    assert verification["verification_complete"] is True
    assert verification["problems"] == []
    assert all(verification["checks"].values())


def test_tampered_corpus_seal_is_rejected_before_synthesis(tmp_path: Path) -> None:
    paths = _write_valid_sources(tmp_path)
    corpus = json.loads(paths.corpus.read_text(encoding="utf-8"))
    corpus["seed_count"] = 999
    _write(paths.corpus, corpus)

    with pytest.raises(ValueError, match="cognitive corpus schema or self-seal is invalid"):
        synthesis.build_synthesis(
            paths.corpus,
            paths.verification,
            paths.report,
            paths.program_state,
        )


def test_trace_membership_and_source_distinct_recurrence_are_exact(tmp_path: Path) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)
    traces = {row["experiment_id"]: row for row in payload["per_experiment_trace_index"]}

    assert traces["candidate_alpha"]["capability_packs"] == [
        "episodic_semantic_working_memory",
        "planning_imagination_simulation",
    ]
    assert traces["candidate_alpha"]["tag_overlap_only"] is True
    assert traces["candidate_beta"]["capability_packs"] == ["episodic_semantic_working_memory"]
    assert traces["candidate_beta"]["tag_overlap_only"] is False
    assert traces["candidate_alpha"]["implementation_family"]["authorities"] == [
        {
            "path": "src/alpha.py",
            "role": "experiment_implementation",
            "sha256": "a" * 64,
        }
    ]
    recurrence = payload["recurrence_analysis"]["source_distinct_candidate_recurrence"]
    assert recurrence == [
        {
            "capability_pack": "episodic_semantic_working_memory",
            "experiment_ids": ["candidate_alpha", "candidate_beta"],
            "implementation_family_sha256s": sorted(
                [
                    traces["candidate_alpha"]["implementation_family"]["sha256"],
                    traces["candidate_beta"]["implementation_family"]["sha256"],
                ]
            ),
            "status": "descriptive_source_distinct_recurrence_only",
        }
    ]

    tampered = copy.deepcopy(payload)
    tampered["per_experiment_trace_index"][0]["capability_packs"] = ["invented-pack"]
    _rewrite_synthesis(paths, tampered)
    verification = _verify(paths)
    assert verification["verification_complete"] is False
    assert verification["checks"]["trace_index_exact"] is False


@pytest.mark.parametrize("mutation", ["dimension_omitted", "successor_status_escalated"])
def test_dimension_omission_and_status_escalation_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)
    changed = copy.deepcopy(payload)
    successor = next(row for row in changed["mechanism_dimension_matrix"] if row["mechanism_id"] == "G1-C1")
    if mutation == "dimension_omitted":
        del successor["dimensions"]["biological_plausibility"]
    else:
        successor["dimensions"]["biological_plausibility"]["status"] = "measured"
    _rewrite_synthesis(paths, changed)

    verification = _verify(paths)
    assert verification["verification_complete"] is False
    assert verification["checks"]["mechanism_dimension_matrix_valid"] is False


@pytest.mark.parametrize("field", ["activation_allowed", "scientific_promotion"])
def test_activation_or_promotion_leakage_is_rejected(tmp_path: Path, field: str) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)
    changed = copy.deepcopy(payload)
    changed[field] = True
    _rewrite_synthesis(paths, changed)

    verification = _verify(paths)
    assert verification["verification_complete"] is False
    assert verification["checks"]["promotion_and_activation_blocked"] is False


def test_synthesis_source_receipt_drift_is_rejected(tmp_path: Path) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)
    changed = copy.deepcopy(payload)
    changed["sources"]["corpus"]["sha256"] = "0" * 64
    _rewrite_synthesis(paths, changed)

    verification = _verify(paths)
    assert verification["verification_complete"] is False
    assert verification["checks"]["source_receipts_exact"] is False


@pytest.mark.parametrize(
    "missing_check",
    sorted(synthesis_verify.REQUIRED_CORPUS_VERIFICATION_CHECKS),
)
def test_required_corpus_verification_check_cannot_be_omitted(
    tmp_path: Path,
    missing_check: str,
) -> None:
    paths = _write_valid_sources(tmp_path)
    verification = json.loads(paths.verification.read_text(encoding="utf-8"))
    del verification["checks"][missing_check]
    _write(paths.verification, _seal(verification, "verification_sha256"))

    report = json.loads(paths.report.read_text(encoding="utf-8"))
    report["corpus"]["verification"] = _source(paths.verification)
    _write(paths.report, _seal(report, "report_sha256"))

    payload = _build(paths)
    result = _verify(paths)
    assert (
        payload["sources"]["corpus_verification"]["sha256"]
        == hashlib.sha256(paths.verification.read_bytes()).hexdigest()
    )
    assert result["verification_complete"] is False
    assert result["checks"]["required_corpus_verification_checks_present"] is False


@pytest.mark.parametrize("mutation", ["claim_scope", "unknown_top_level_claim"])
def test_claim_scope_and_top_level_contract_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    paths = _write_valid_sources(tmp_path)
    payload = _build(paths)
    changed = copy.deepcopy(payload)
    if mutation == "claim_scope":
        changed["claim_scope"] = "integrated substrate advantage proven"
        expected_check = "claim_scope_exact"
    else:
        changed["substrate_formed"] = True
        expected_check = "top_level_contract_exact"
    _rewrite_synthesis(paths, changed)

    result = _verify(paths)
    assert result["verification_complete"] is False
    assert result["checks"][expected_check] is False
