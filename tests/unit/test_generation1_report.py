from __future__ import annotations

import json
from pathlib import Path

import pytest

from mop.studies import generation1_report as report
from mop.studies.generation1_cognitive_corpus import CORPUS_SCHEMA, canonical_sha256
from mop.studies.generation1_cognitive_corpus_verify import VERIFICATION_SCHEMA


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sealed(payload: dict[str, object], field: str) -> dict[str, object]:
    return {**payload, field: canonical_sha256(payload)}


def _evidence_fixture(root: Path) -> None:
    _write(
        root / "proof/ESCS_X0_EVENT_FORMATION.verification.json",
        {
            "fresh_aggregate": {
                "verdict": "strong_null_not_rejected",
                "paired_intervals_95": {
                    "work_saving_vs_always_on": {"mean": 0.38},
                    "utility_loss_vs_always_on": {"mean": 0.97},
                },
                "checks": {"required_direction_every_seed": False},
            }
        },
    )
    _write(
        root / "proof/EDCM1_EVENT_TRIGGERED_COALITION_V3.json",
        {
            "aggregate": {
                "verdict": "complementarity_gate_failed",
                "gate": {
                    "niche_advantage_95": {},
                    "oracle_headroom_95": {},
                    "verifier_disagreement_gain_95": {},
                    "verifier_agreement_effect_95": {},
                },
            }
        },
    )
    _write(
        root / "proof/P6_CONTINUAL_10K_INDEPENDENT_VERIFICATION.json",
        {
            "prerequisite": {"next_rung_allowed": False},
            "independent_recompute": {
                "decision": {"verdict": "null", "contrasts": []}
            },
        },
    )
    _write(
        root / "proof/P7_ACTION_WORLD_MODEL_PREFLIGHT.json",
        {"status": "mechanics-pass", "all_mechanics_ok": True, "claim_boundary": {}},
    )
    canary_core: dict[str, object] = {
        "schema": report.RESOURCE_CANARY_SCHEMA,
        "run_id": "fixture-canary",
        "wall_seconds": 12.0,
        "outcome_counts": {"ok": 16},
        "measurements": {
            "runtime_safety_problems": [],
            "aggregate_process_tree_peak_rss_bytes": 5_500_000_000,
        },
        "recommendation": {
            "eligible": True,
            "recommended_max_workers": 16,
            "recommended_estimated_unified_memory_gb": 10.0,
        },
        "source_authority": {
            "stable": True,
            "before": {"aggregate_sha256": "a" * 64},
        },
        "orchestration_problems": [],
        "complete": True,
        "scientific_promotion": False,
    }
    _write(
        root / report.RESOURCE_CANARY,
        _sealed(canary_core, "receipt_sha256"),
    )
    for evidence_id, relative in report.MECHANICS_ARTIFACTS:
        _write(
            root / relative,
            {
                "schema": f"fixture-{evidence_id}/v1",
                "status": "mechanics-pass",
                "all_ok": True,
                "scientific_promotion": False,
            },
        )


def _corpus_fixture() -> dict[str, object]:
    core: dict[str, object] = {
        "schema": CORPUS_SCHEMA,
        "seed_count": 2,
        "eligible_experiment_count": 2,
        "complete_experiment_count": 2,
        "corpus_complete": True,
        "experiment_summaries": {
            "candidate": {
                "classification": "stable_candidate_trace",
                "evidence_class": "inferential",
                "seed_mode": "varied",
                "completed_seed_count": 2,
                "effective_observation_count": 2,
                "distinct_seed_authority_count": 2,
                "variation_canary": {"passed": True},
                "null_supported": {"observations": 2, "true": 0, "false": 2},
                "boolean_rates": {"null_supported": {"n": 2, "true_fraction": 0.0}},
                "numeric_summaries": {"gain": {"n": 2, "mean": 0.2}},
                "contract": {"claim": "fixture"},
            },
            "mechanics": {
                "classification": "mechanics_noninferential",
                "evidence_class": "mechanics_noninferential",
                "seed_mode": "mechanics",
                "completed_seed_count": 1,
                "effective_observation_count": 1,
                "null_supported": {"observations": 0, "true": 0, "false": 0},
                "contract": {},
            },
        },
        "capability_pack_summaries": {
            "planning": {
                "experiment_ids": ["candidate", "mechanics"],
                "classification_counts": {
                    "stable_candidate_trace": 1,
                    "mechanics_noninferential": 1,
                },
            },
            "causal": {
                "experiment_ids": ["candidate"],
                "classification_counts": {"stable_candidate_trace": 1},
            },
        },
        "seed_authority_summary": {
            "varied_experiment_count": 1,
            "variation_canary_failures": 0,
            "no_pseudoreplication": True,
        },
        "operational_summary": {"attempt_count": 3, "retry_count": 1},
        "total_manifest_bytes": 100,
    }
    return _sealed(core, "corpus_sha256")


def test_report_contains_detailed_atlases_and_conservative_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    _evidence_fixture(tmp_path)
    corpus_path = tmp_path / "proof/GENERATION1_COGNITIVE_CORPUS.json"
    corpus = _corpus_fixture()
    _write(corpus_path, corpus)
    checks = {
        "corpus_complete": True,
        "full_regeneration_match": True,
        "all_mutations_rejected": True,
        "all_seed_receipts_valid": True,
        "promotion_blocked": True,
        "all_attempt_receipts_valid": True,
        "all_cell_authorities_valid": True,
        "seed_authority_exact": True,
        "no_pseudoreplication": True,
        "independent_summary_match": True,
        "directional_inference_fail_closed": True,
    }
    verification_path = tmp_path / "proof/GENERATION1_COGNITIVE_CORPUS.verification.json"
    verification_core = {
        "schema": VERIFICATION_SCHEMA,
        "verification_complete": True,
        "problems": [],
        "corpus": {"sha256": report._sha_file(corpus_path)},
        "checks": checks,
    }
    _write(
        verification_path,
        _sealed(verification_core, "verification_sha256"),
    )

    result = report.build_report(corpus_path, verification_path)

    assert result["corpus"]["classification_counts"] == {
        "mechanics_noninferential": 1,
        "stable_candidate_trace": 1,
    }
    assert result["corpus"]["atlases"]["stable_candidate_trace"][0][
        "distinct_seed_authority_count"
    ] == 2
    assert result["empirical_innovation_traces"][0]["trace"] == (
        "cross_pack_candidate_stability"
    )
    assert result["next_authority"]["ready_to_preregister_mechanism_epoch"] is True
    assert result["next_authority"]["ready_to_activate_or_integrate_substrate"] is False
    assert len(result["fresh_mechanics_evidence"]) == len(report.MECHANICS_ARTIFACTS)
    assert result["resource_authority"]["recommendation"]["recommended_max_workers"] == 16
    assert "ready_to_activate_or_integrate_substrate: false" in report.render_text(result)


def test_report_rejects_tampered_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(report, "REPO_ROOT", tmp_path)
    corpus = _corpus_fixture()
    corpus["corpus_complete"] = False
    corpus_path = tmp_path / "corpus.json"
    verification_path = tmp_path / "verification.json"
    _write(corpus_path, corpus)
    _write(verification_path, {"verification_complete": True, "problems": []})

    with pytest.raises(ValueError, match="self-seal"):
        report.build_report(corpus_path, verification_path)
