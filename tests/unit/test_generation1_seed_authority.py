from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

import mop.studies.generation1_cognitive_corpus as corpus


def _production_config() -> dict[str, Any]:
    return corpus.load_config(corpus.DEFAULT_CONFIG)


def _strict_summary_config() -> dict[str, Any]:
    seeds = list(range(101, 125))
    return {
        "schema": corpus.CONFIG_SCHEMA,
        "campaign_id": "generation1-seed-authority-unit",
        "claim_scope": "synthetic unit evidence only",
        "result_tag": "generation1-seed-authority-unit",
        "seeds": seeds,
        "experiment_scope": {
            "tiers": ["cpu-now"],
            "include_f_series": True,
            "include_wrapper_smokes": True,
            "excluded_ids": [],
        },
        "classification": {
            "minimum_complete_seeds": 20,
            "stable_fraction": 0.8,
            "minimum_boolean_observations": 20,
            "tie_label": "mixed_or_seed_sensitive",
            "missing_null_label": "descriptive_only",
        },
        "seed_authority": {
            "schema": corpus.SEED_POLICY_SCHEMA,
            "algorithm": corpus.SEED_ALGORITHM,
            "variation_canary_outer_seeds": seeds[:5],
            "outer_seed_experiment_ids": ["alpha"],
            "fixed_case_experiment_ids": [],
            "mechanics_only_experiment_ids": [],
            "experiment_seed_paths": {},
        },
        "capability_packs": {"unit": ["alpha"]},
    }


def _authority_manifest(
    *,
    seed: int,
    null_supported: bool | None = True,
    score: float | None = None,
    mode: str = corpus.SEED_MODE_VARIED,
    evidence_class: str = corpus.EVIDENCE_INFERENTIAL,
    authority_suffix: str | None = None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"score": float(seed if score is None else score)}
    if null_supported is not None:
        metrics["null_supported"] = null_supported
    suffix = str(seed) if authority_suffix is None else authority_suffix
    overrides = [{"role": "campaign_outer_seed", "path": "seed", "effective": seed}]
    return {
        "name": "alpha",
        "seed": seed,
        "status": "ok",
        "result_tag": "generation1-seed-authority-unit",
        "metrics": metrics,
        "extra": {
            "contract": {"experiment_id": "alpha", "promotion": False},
            "generation1_cell_authority": {
                "schema": corpus.CELL_AUTHORITY_SCHEMA,
                "evidence_class": evidence_class,
                "seed_mode": mode,
                "seed_authority": {
                    "authority_sha256": f"authority-{suffix}",
                    "effective_overrides": overrides,
                },
                "scientific_metrics_sha256": corpus._scientific_fingerprint(metrics),
            },
        },
    }


def test_production_policy_covers_all_cpu_experiments_and_five_seed_canaries() -> None:
    config = _production_config()
    experiment_ids = corpus.eligible_experiment_ids(config)

    assert len(experiment_ids) == 128
    assert config["seed_authority"]["variation_canary_outer_seeds"] == config["seeds"][:5]
    modes = Counter(corpus._seed_mode(config, experiment_id)[0] for experiment_id in experiment_ids)
    assert modes == {corpus.SEED_MODE_VARIED: 125, corpus.SEED_MODE_MECHANICS: 2, corpus.SEED_MODE_FIXED: 1}

    for experiment_id in experiment_ids:
        mode, _ = corpus._seed_mode(config, experiment_id)
        outer_seeds = config["seeds"][:5] if mode == corpus.SEED_MODE_VARIED else config["seeds"][:1]
        authorities = [
            corpus._compose_worker_config(
                experiment_id=experiment_id,
                outer_seed=outer_seed,
                result_tag=config["result_tag"],
                config=config,
            )[1]
            for outer_seed in outer_seeds
        ]
        assert all(authority["mode"] == mode for authority in authorities)
        if mode == corpus.SEED_MODE_VARIED:
            assert len({authority["authority_sha256"] for authority in authorities}) == 5
            assert len(
                {
                    corpus.canonical_sha256(authority["effective_overrides"])
                    for authority in authorities
                }
            ) == 5
        else:
            assert authorities[0]["execute_once"] is True
            assert authorities[0]["reference_outer_seed"] == config["seeds"][0]


@pytest.mark.parametrize(
    ("experiment_id", "expected_paths"),
    [
        ("a1_affordance_decode", ["seed", "experiment.seeds"]),
        ("e1_baseline", ["seed"]),
        ("ex13_long_stream", ["seed"]),
        ("ex2_latent_planning", ["seed", "experiment.seed"]),
        (
            "mop_cm7_min_objective_probe",
            ["seed", "experiment.data.seed", "experiment.training.seeds"],
        ),
        ("p5_private_language", ["seed", "experiment.data_seed"]),
        ("y3_seed_consistent_fixed_points", ["seed", "experiment.data_seed"]),
        (
            "f12_private_form_language_stability",
            ["seed", "experiment.seeds", "experiment.base_seed"],
        ),
    ],
)
def test_effective_seed_adapter_records_only_consumed_seed_controls(
    experiment_id: str,
    expected_paths: list[str],
) -> None:
    config = _production_config()
    first_cfg, first = corpus._compose_worker_config(
        experiment_id=experiment_id,
        outer_seed=config["seeds"][0],
        result_tag=config["result_tag"],
        config=config,
    )
    repeated_cfg, repeated = corpus._compose_worker_config(
        experiment_id=experiment_id,
        outer_seed=config["seeds"][0],
        result_tag=config["result_tag"],
        config=config,
    )
    _, second = corpus._compose_worker_config(
        experiment_id=experiment_id,
        outer_seed=config["seeds"][1],
        result_tag=config["result_tag"],
        config=config,
    )

    assert first == repeated
    assert OmegaConf.to_container(first_cfg, resolve=True) == OmegaConf.to_container(
        repeated_cfg, resolve=True
    )
    assert [row["path"] for row in first["effective_overrides"]] == expected_paths
    assert first["authority_sha256"] != second["authority_sha256"]
    for row in first["effective_overrides"]:
        effective = row["effective"]
        values = effective if isinstance(effective, list) else [effective]
        assert values
        assert len(values) == len(set(values))
        assert all(isinstance(value, int) and 0 <= value < 2**31 - 1 for value in values)

    plain = OmegaConf.to_container(first_cfg.experiment, resolve=True)
    assert isinstance(plain, dict)
    if experiment_id == "ex13_long_stream":
        assert plain["seeds"] == [0]
    if experiment_id == "mop_cm7_min_objective_probe":
        assert plain["promotion"]["min_seeds"] == 5
    if experiment_id == "y3_seed_consistent_fixed_points":
        assert plain["k_seeds"] == 4


def test_fixed_and_mechanics_cells_execute_once_and_are_never_inferential() -> None:
    config = _production_config()
    first_seed, later_seed = config["seeds"][:2]
    expected = {
        "s5_code_stability": (corpus.SEED_MODE_FIXED, corpus.EVIDENCE_FIXED),
        "mop_p4_capability_density_screen": (
            corpus.SEED_MODE_MECHANICS,
            corpus.EVIDENCE_MECHANICS,
        ),
        "mop_p5_context_capability": (
            corpus.SEED_MODE_MECHANICS,
            corpus.EVIDENCE_MECHANICS,
        ),
    }
    for experiment_id, (mode, evidence) in expected.items():
        _, authority = corpus._compose_worker_config(
            experiment_id=experiment_id,
            outer_seed=first_seed,
            result_tag=config["result_tag"],
            config=config,
        )
        assert authority["mode"] == mode
        assert authority["evidence_class"] == evidence
        assert authority["execute_once"] is True
        assert authority["reference_outer_seed"] == first_seed
        assert corpus._execution_seed(config, experiment_id, later_seed) == first_seed

    without_wrappers = copy.deepcopy(config)
    without_wrappers["experiment_scope"]["include_wrapper_smokes"] = False
    eligible = corpus.eligible_experiment_ids(without_wrappers)
    assert "mop_p4_capability_density_screen" not in eligible
    assert "mop_p5_context_capability" not in eligible
    assert "s5_code_stability" in eligible


def test_directional_classification_requires_five_seed_canary_and_exhaustive_nulls() -> None:
    config = _strict_summary_config()
    manifests = [_authority_manifest(seed=seed) for seed in config["seeds"][:20]]
    summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=manifests,
        config=config,
    )

    assert summary["classification"] == "stable_null"
    assert summary["directional_evidence_eligible"] is True
    assert summary["variation_canary"]["status"] == "passed"
    assert summary["variation_canary"]["expected_count"] == 5
    assert summary["effective_observation_count"] == 20
    assert summary["coverage_complete"] is False
    assert summary["null_supported"]["wilson_95"] == {
        "method": "wilson_score",
        "confidence": 0.95,
        "low": 0.83887484,
        "high": 1.0,
    }

    selective = [
        _authority_manifest(seed=seed, null_supported=None if index == 19 else True)
        for index, seed in enumerate(config["seeds"][:20])
    ]
    selective_summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=selective,
        config=config,
    )
    assert selective_summary["classification"] == "descriptive_only"
    assert selective_summary["directional_evidence_eligible"] is False
    assert selective_summary["null_supported"]["observations"] == 19

    underpowered = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=manifests[:19],
        config=config,
    )
    assert underpowered["classification"] == "descriptive_only"
    assert underpowered["directional_evidence_eligible"] is False


def test_invariant_and_structurally_invalid_canaries_are_quarantined_separately() -> None:
    config = _strict_summary_config()
    invariant = [
        _authority_manifest(seed=seed, score=1.0) for seed in config["seeds"][:20]
    ]
    invariant_summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=invariant,
        config=config,
    )
    assert invariant_summary["variation_canary"]["status"] == "scientific_output_invariant"
    assert invariant_summary["variation_canary"]["structurally_varied"] is True
    assert invariant_summary["classification"] == "descriptive_seed_invariant"
    assert invariant_summary["directional_evidence_eligible"] is False

    duplicate = [_authority_manifest(seed=seed) for seed in config["seeds"][:20]]
    duplicate[1]["extra"]["generation1_cell_authority"]["seed_authority"][
        "authority_sha256"
    ] = "authority-101"
    duplicate_summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=duplicate,
        config=config,
    )
    assert duplicate_summary["variation_canary"]["status"] == "seed_authority_failed"
    assert duplicate_summary["classification"] == "descriptive_seed_adapter_unverified"

    incomplete_summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=duplicate[:4],
        config=config,
    )
    assert incomplete_summary["variation_canary"]["status"] == "incomplete"
    assert incomplete_summary["classification"] == "descriptive_seed_adapter_unverified"


@pytest.mark.parametrize(
    ("mode", "evidence_class", "expected_classification"),
    [
        (
            corpus.SEED_MODE_MECHANICS,
            corpus.EVIDENCE_MECHANICS,
            "mechanics_noninferential",
        ),
        (corpus.SEED_MODE_FIXED, corpus.EVIDENCE_FIXED, "descriptive_fixed_case"),
    ],
)
def test_execute_once_modes_cannot_become_stable_null(
    mode: str,
    evidence_class: str,
    expected_classification: str,
) -> None:
    config = _strict_summary_config()
    config["seed_authority"]["outer_seed_experiment_ids"] = []
    field = (
        "mechanics_only_experiment_ids"
        if mode == corpus.SEED_MODE_MECHANICS
        else "fixed_case_experiment_ids"
    )
    config["seed_authority"][field] = ["alpha"]
    manifest = _authority_manifest(
        seed=config["seeds"][0],
        mode=mode,
        evidence_class=evidence_class,
    )

    summary = corpus._experiment_summary(
        experiment_id="alpha",
        manifests=[manifest],
        config=config,
    )

    assert summary["classification"] == expected_classification
    assert summary["effective_observation_count"] == 1
    assert summary["expected_execution_count"] == 1
    assert summary["variation_canary"]["status"] == "not_applicable"
    assert summary["directional_evidence_eligible"] is False


def test_attempt_receipt_binds_manifest_config_implementation_and_seed_authority(
    tmp_path: Path,
) -> None:
    config = _production_config()
    experiment_id = "a1_affordance_decode"
    seed = config["seeds"][0]
    run_dir = tmp_path / "attempt_001"
    mpl_config_dir = tmp_path / "mpl"
    mpl_config_dir.mkdir()
    expected = corpus._expected_cell_authority(config, experiment_id, seed)

    result = corpus._run_subprocess(
        script=corpus.REPO_ROOT / "scripts/generation1_cognitive_corpus.py",
        experiment_id=experiment_id,
        seed=seed,
        run_dir=run_dir,
        result_tag=config["result_tag"],
        timeout_seconds=120.0,
        mpl_config_dir=mpl_config_dir,
        config_path=corpus.DEFAULT_CONFIG,
        cell_authority=expected,
    )

    assert result["returncode"] == 0, result["stderr_tail"]
    assert result["timed_out"] is False
    assert corpus._manifest_ok(
        run_dir / "manifest.json",
        experiment_id=experiment_id,
        seed=seed,
        result_tag=config["result_tag"],
        expected_cell_authority=expected,
    )
    attempt = json.loads((run_dir / "attempt_receipt.json").read_text(encoding="utf-8"))
    assert attempt["schema"] == corpus.ATTEMPT_SCHEMA
    assert corpus._valid_seal(attempt, "attempt_sha256")
    assert attempt["manifest"]["sha256"] == corpus.sha256_file(run_dir / "manifest.json")
    assert attempt["resolved_config"]["sha256"] == corpus.sha256_file(run_dir / "config.yaml")
    assert attempt["seed_authority"] == expected["seed_authority"]
    assert attempt["experiment_config"] == expected["experiment_config"]
    assert attempt["implementation_authorities"] == expected["implementation_authorities"]
    cell_receipt = corpus._cell_receipt(
        attempt=run_dir,
        experiment_id=experiment_id,
        requested_outer_seed=seed,
        reference_outer_seed=seed,
    )
    assert cell_receipt["attempt_receipt"]["self_seal_valid"] is True
    assert cell_receipt["attempt_receipt"]["attempt_sha256"] == attempt["attempt_sha256"]
