from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

import mop.studies.generation1_cognitive_corpus as corpus_module
import mop.studies.generation1_cognitive_corpus_verify as verify_module


class _AlphaExperiment:
    tier = "cpu-now"


class _MechanicsExperiment:
    tier = "cpu-now"


_EXPERIMENTS: dict[str, dict[str, Any]] = {
    "alpha": {"id": "alpha", "seeds": [7, 11], "tier": "cpu-now"},
    "mechanics": {"id": "mechanics", "seed": 3, "tier": "cpu-now"},
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(corpus_module.canonical_bytes(payload) + b"\n")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload)), encoding="utf-8")


def _set_path(root: dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    if parts[0] == "experiment":
        parts = parts[1:]
    parent = root
    for part in parts[:-1]:
        parent = parent[part]
    parent[parts[-1]] = value


def _campaign_config() -> dict[str, Any]:
    seeds = [101, 102, 103, 104, 105]
    return {
        "schema": corpus_module.CONFIG_SCHEMA,
        "campaign_id": "generation1-independent-verifier-unit",
        "claim_scope": "synthetic verifier unit evidence only",
        "result_tag": "generation1-independent-verifier-unit-result",
        "seeds": seeds,
        "experiment_scope": {
            "tiers": ["cpu-now"],
            "include_f_series": True,
            "include_wrapper_smokes": True,
            "excluded_ids": [],
        },
        "classification": {
            "minimum_complete_seeds": 5,
            "stable_fraction": 0.8,
            "minimum_boolean_observations": 5,
            "tie_label": "mixed_or_seed_sensitive",
            "missing_null_label": "descriptive_only",
        },
        "seed_authority": {
            "schema": corpus_module.SEED_POLICY_SCHEMA,
            "algorithm": corpus_module.SEED_ALGORITHM,
            "variation_canary_outer_seeds": seeds[:5],
            "outer_seed_experiment_ids": [],
            "fixed_case_experiment_ids": [],
            "mechanics_only_experiment_ids": ["mechanics"],
            "experiment_seed_paths": {"alpha": ["experiment.seeds"]},
        },
        "capability_packs": {"reasoning": ["alpha", "mechanics"]},
    }


def _fake_compose(overrides: list[str]) -> DictConfig:
    experiment_id = next(row.split("=", 1)[1] for row in overrides if row.startswith("experiment="))
    result_tag = next(row.split("=", 1)[1] for row in overrides if row.startswith("result_tag="))
    return OmegaConf.create(
        {
            "seed": 0,
            "result_tag": result_tag,
            "experiment": copy.deepcopy(_EXPERIMENTS[experiment_id]),
        }
    )


def _make_successful_attempt(
    *,
    run_root: Path,
    config: dict[str, Any],
    experiment_id: str,
    seed: int,
    metrics: dict[str, Any],
) -> Path:
    attempt_dir = run_root / f"seed_{seed}" / "classes" / experiment_id / "attempt_001"
    attempt_dir.mkdir(parents=True)
    expected = corpus_module._expected_cell_authority(config, experiment_id, seed)
    experiment = copy.deepcopy(_EXPERIMENTS[experiment_id])
    for override in expected["seed_authority"]["effective_overrides"]:
        if override["path"].startswith("experiment."):
            _set_path(experiment, override["path"], override["effective"])
    resolved_payload = {
        "seed": seed,
        "result_tag": config["result_tag"],
        "experiment": experiment,
        "generation1_seed_authority": expected["seed_authority"],
        "generation1_evidence_class": expected["evidence_class"],
    }
    resolved_path = attempt_dir / "config.yaml"
    _write_yaml(resolved_path, resolved_payload)
    resolved_binding = {
        "path": corpus_module._repository_path(resolved_path),
        "sha256": corpus_module.sha256_file(resolved_path),
    }
    fingerprint = corpus_module._scientific_fingerprint(metrics)
    cell = {
        **expected,
        "resolved_config": resolved_binding,
        "scientific_metrics_sha256": fingerprint,
    }
    manifest = {
        "name": experiment_id,
        "seed": seed,
        "status": "ok",
        "result_tag": config["result_tag"],
        "metrics": metrics,
        "extra": {
            "contract": {"experiment_id": experiment_id, "promotion": False},
            "generation1_cell_authority": cell,
        },
    }
    manifest_path = attempt_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_binding = {
        "path": corpus_module._repository_path(manifest_path),
        "sha256": corpus_module.sha256_file(manifest_path),
    }
    worker_report = {
        "experiment_id": experiment_id,
        "seed": seed,
        "seconds": 0.25,
        "maximum_rss_bytes": 4096,
        "metric_keys": sorted(metrics),
        "evidence_class": cell["evidence_class"],
        "seed_mode": cell["seed_mode"],
        "seed_authority": cell["seed_authority"],
        "experiment_config": cell["experiment_config"],
        "implementation_authorities": cell["implementation_authorities"],
        "resolved_config": resolved_binding,
        "manifest": manifest_binding,
        "scientific_metrics_sha256": fingerprint,
    }
    attempt = corpus_module._sealed(
        {
            "schema": corpus_module.ATTEMPT_SCHEMA,
            "experiment_id": experiment_id,
            "seed": seed,
            "run_dir": corpus_module._repository_path(attempt_dir),
            "returncode": 0,
            "timed_out": False,
            "seconds": 0.5,
            "stdout_tail": "GENERATION1_WORKER={...}",
            "stderr_tail": "",
            "evidence_class": cell["evidence_class"],
            "seed_mode": cell["seed_mode"],
            "seed_authority": cell["seed_authority"],
            "experiment_config": cell["experiment_config"],
            "implementation_authorities": cell["implementation_authorities"],
            "resolved_config": resolved_binding,
            "manifest": manifest_binding,
            "worker_report": worker_report,
            "recorded_at": "2026-07-13T00:00:00+00:00",
        },
        "attempt_sha256",
    )
    _write_json(attempt_dir / "attempt_receipt.json", attempt)
    return attempt_dir


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    invariant: bool = False,
    missing_null: bool = False,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    registry = {"alpha": _AlphaExperiment, "mechanics": _MechanicsExperiment}
    monkeypatch.setattr(corpus_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verify_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(corpus_module, "REGISTRY", registry)
    monkeypatch.setattr(verify_module, "REGISTRY", registry)
    monkeypatch.setattr(corpus_module, "compose", _fake_compose)
    monkeypatch.setattr(verify_module, "compose", _fake_compose)

    (tmp_path / "src/mop/harness").mkdir(parents=True)
    (tmp_path / "src/mop/harness/runner.py").write_text("# verifier fixture\n", encoding="utf-8")
    for experiment_id, payload in _EXPERIMENTS.items():
        _write_yaml(tmp_path / "configs/experiment" / f"{experiment_id}.yaml", payload)

    config = _campaign_config()
    config_path = tmp_path / "configs/experiment/generation1_cognitive_corpus.json"
    _write_json(config_path, config)
    run_root = tmp_path / "runs/generation1/cognitive_corpus"
    for index, seed in enumerate(config["seeds"]):
        metrics: dict[str, Any] = {
            "null_supported": False,
            "score": 1.0 if invariant else float(index + 1),
        }
        if missing_null and index == len(config["seeds"]) - 1:
            metrics.pop("null_supported")
        _make_successful_attempt(
            run_root=run_root,
            config=config,
            experiment_id="alpha",
            seed=seed,
            metrics=metrics,
        )
    _make_successful_attempt(
        run_root=run_root,
        config=config,
        experiment_id="mechanics",
        seed=config["seeds"][0],
        metrics={"null_supported": True, "score": 1.0},
    )
    for seed in config["seeds"]:
        corpus_module.run_seed(
            config_path=config_path,
            run_root=run_root,
            seed=seed,
            output=run_root / f"seed_{seed}" / "seed_receipt.json",
            max_workers=2,
            timeout_seconds=1.0,
            wall_seconds=10.0,
            script=tmp_path / "unused.py",
        )
    built = corpus_module.build_corpus(config_path, run_root)
    corpus_path = tmp_path / "proof/GENERATION1_COGNITIVE_CORPUS.json"
    _write_json(corpus_path, built)
    return corpus_path, config_path, run_root, built


def test_strict_v2_verifier_independently_accepts_complete_authority_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, config_path, run_root, _ = _fixture(tmp_path, monkeypatch)

    result = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )

    assert result["schema"] == verify_module.VERIFICATION_SCHEMA
    assert result["verification_complete"] is True
    assert result["problems"] == []
    assert all(result["checks"].values())
    for required in (
        "all_attempt_receipts_valid",
        "all_cell_authorities_valid",
        "seed_authority_exact",
        "no_pseudoreplication",
        "independent_summary_match",
    ):
        assert result["checks"][required] is True
    assert result["mutation_suite"]["results"] == {
        "plan_tamper": True,
        "seed_authority_tamper": True,
        "evidence_class_tamper": True,
        "attempt_tamper": True,
        "summary_tamper": True,
    }
    assert verify_module._valid_seal(result, "verification_sha256") is True


@pytest.mark.parametrize(
    ("case", "failed_check"),
    [
        ("plan", "seed_set_exact"),
        ("seed_authority", "seed_authority_exact"),
        ("evidence_class", "independent_summary_match"),
        ("summary", "independent_summary_match"),
    ],
)
def test_resealed_corpus_mutations_are_rejected_semantically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    failed_check: str,
) -> None:
    corpus_path, config_path, run_root, built = _fixture(tmp_path, monkeypatch)
    mutated = copy.deepcopy(built)
    if case == "plan":
        mutated["seeds"] = list(reversed(mutated["seeds"]))
    elif case == "seed_authority":
        mutated["cell_authority_index"]["alpha"][0]["seed_authority"][
            "authority_sha256"
        ] = "0" * 64
    elif case == "evidence_class":
        mutated["experiment_summaries"]["alpha"]["evidence_class"] = (
            "mechanics_noninferential"
        )
    else:
        mutated["experiment_summaries"]["alpha"]["classification"] = "stable_null"
    mutated = corpus_module._sealed(mutated, "corpus_sha256")
    _write_json(corpus_path, mutated)

    result = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )

    assert result["checks"]["corpus_self_hash"] is True
    assert result["checks"][failed_check] is False
    assert result["verification_complete"] is False


def test_resealed_attempt_authority_mutation_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, config_path, run_root, _ = _fixture(tmp_path, monkeypatch)
    path = run_root / "seed_101/classes/alpha/attempt_001/attempt_receipt.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt["seed_authority"]["outer_seed"] = 999
    receipt = corpus_module._sealed(receipt, "attempt_sha256")
    _write_json(path, receipt)

    result = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )

    assert result["checks"]["all_attempt_receipts_valid"] is False
    assert result["checks"]["all_cell_authorities_valid"] is False
    assert result["verification_complete"] is False


def test_scientific_canary_invariance_is_valid_but_directional_inference_is_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, config_path, run_root, built = _fixture(
        tmp_path, monkeypatch, invariant=True
    )
    summary = built["experiment_summaries"]["alpha"]
    assert summary["variation_canary"]["status"] == "scientific_output_invariant"
    assert summary["classification"] == "descriptive_seed_invariant"
    assert built["seed_authority_summary"]["no_pseudoreplication"] is True

    result = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )

    assert result["verification_complete"] is True
    assert result["checks"]["no_pseudoreplication"] is True
    assert result["checks"]["directional_inference_fail_closed"] is True


def test_missing_boolean_null_observation_cannot_receive_directional_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_path, config_path, run_root, built = _fixture(
        tmp_path, monkeypatch, missing_null=True
    )
    summary = built["experiment_summaries"]["alpha"]
    assert summary["null_supported"]["observations"] == 4
    assert summary["effective_observation_count"] == 5
    assert summary["classification"] == "descriptive_only"

    result = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )

    assert result["verification_complete"] is True
    assert result["checks"]["directional_inference_fail_closed"] is True
