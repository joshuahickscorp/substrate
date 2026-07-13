from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import mop.studies.generation1_cognitive_corpus as corpus_module
import mop.studies.generation1_cognitive_corpus_verify as verify_module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(corpus_module.canonical_bytes(payload) + b"\n")


def _config(*, include_f_series: bool = True) -> dict[str, Any]:
    return {
        "schema": corpus_module.CONFIG_SCHEMA,
        "campaign_id": "generation1-unit-corpus",
        "claim_scope": "synthetic unit-test evidence only",
        "result_tag": "generation1-unit-result",
        "seeds": [101, 102, 103, 104, 105],
        "experiment_scope": {
            "tiers": ["cpu-now"],
            "include_f_series": include_f_series,
            "excluded_ids": ["excluded"],
        },
        "classification": {
            "minimum_complete_seeds": 5,
            "stable_fraction": 0.8,
            "minimum_boolean_observations": 5,
            "tie_label": "mixed_or_seed_sensitive",
            "missing_null_label": "descriptive_only",
        },
        "capability_packs": {"reasoning": ["alpha", "f_beta"]},
    }


def _fake_registry() -> dict[str, SimpleNamespace]:
    return {
        "alpha": SimpleNamespace(tier="cpu-now"),
        "excluded": SimpleNamespace(tier="cpu-now"),
        "f_beta": SimpleNamespace(tier="cpu-now"),
        "later": SimpleNamespace(tier="deferred"),
    }


def _manifest(
    experiment_id: str,
    seed: int,
    *,
    null_supported: bool | None,
    score: float,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "score": score,
        "nested": {"confidence": score / 10.0},
        "ignored_sequence": [score],
    }
    if null_supported is not None:
        metrics["null_supported"] = null_supported
    return {
        "name": experiment_id,
        "seed": seed,
        "status": "ok",
        "result_tag": "generation1-unit-result",
        "metrics": metrics,
        "extra": {"contract": {"experiment_id": experiment_id, "promotion": False}},
    }


def _make_synthetic_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    monkeypatch.setattr(corpus_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(verify_module, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(corpus_module, "REGISTRY", _fake_registry())

    config_path = tmp_path / "configs" / "generation1.json"
    run_root = tmp_path / "runs" / "generation1"
    corpus_path = tmp_path / "proof" / "corpus.json"
    config = _config()
    _write_json(config_path, config)

    null_directions = {
        "alpha": [True, True, True, True, False],
        "f_beta": [False, False, False, False, True],
    }
    eligible_ids = ["alpha", "f_beta"]
    for seed_index, seed in enumerate(config["seeds"]):
        seed_root = run_root / f"seed_{seed}"
        for experiment_index, experiment_id in enumerate(eligible_ids):
            manifest = _manifest(
                experiment_id,
                seed,
                null_supported=null_directions[experiment_id][seed_index],
                score=float(seed_index + experiment_index + 1),
            )
            _write_json(
                seed_root / "classes" / experiment_id / "attempt_001" / "manifest.json",
                manifest,
            )

        receipt = corpus_module._sealed(
            {
                "schema": corpus_module.SEED_SCHEMA,
                "seed": seed,
                "all_complete": True,
                "eligible_ids": eligible_ids,
            },
            "receipt_sha256",
        )
        _write_json(seed_root / "seed_receipt.json", receipt)

    built = corpus_module.build_corpus(config_path, run_root)
    _write_json(corpus_path, built)
    return corpus_path, config_path, run_root, built


def test_config_validation_and_eligible_set_respect_tier_f_series_and_exclusions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(corpus_module, "REGISTRY", _fake_registry())
    config_path = tmp_path / "config.json"
    config = _config(include_f_series=False)
    _write_json(config_path, config)

    loaded = corpus_module.load_config(config_path)
    assert loaded == config
    assert corpus_module.eligible_experiment_ids(loaded) == ["alpha"]

    loaded["experiment_scope"]["include_f_series"] = True
    assert corpus_module.eligible_experiment_ids(loaded) == ["alpha", "f_beta"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value["seeds"].__setitem__(4, 101), "five distinct"),
        (lambda value: value["seeds"].__setitem__(0, True), "nonnegative integers"),
        (lambda value: value["classification"].__setitem__("stable_fraction", 0.5), "stable_fraction"),
        (lambda value: value["capability_packs"]["reasoning"].append("unknown"), "unknown experiments"),
    ],
)
def test_config_validation_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    monkeypatch.setattr(corpus_module, "REGISTRY", _fake_registry())
    config = _config()
    mutate(config)
    config_path = tmp_path / "invalid.json"
    _write_json(config_path, config)

    with pytest.raises(ValueError, match=message):
        corpus_module.load_config(config_path)


def test_canonical_seal_is_order_independent_deterministic_and_tamper_evident() -> None:
    first = {"z": [3, 2, 1], "a": {"unicode": "caf\u00e9", "flag": False}}
    second = {"a": {"flag": False, "unicode": "caf\u00e9"}, "z": [3, 2, 1]}

    assert corpus_module.canonical_bytes(first) == corpus_module.canonical_bytes(second)
    assert corpus_module.canonical_sha256(first) == corpus_module.canonical_sha256(second)

    sealed = corpus_module._sealed(first, "payload_sha256")
    resealed = corpus_module._sealed({**second, "payload_sha256": "stale"}, "payload_sha256")
    assert sealed == resealed
    assert verify_module._valid_seal(sealed, "payload_sha256") is True

    tampered = copy.deepcopy(sealed)
    tampered["z"][0] = 99
    assert verify_module._valid_seal(tampered, "payload_sha256") is False


@pytest.mark.parametrize(
    ("observations", "expected"),
    [
        ([True, True, True, True, False], "stable_null"),
        ([False, False, False, False, True], "stable_candidate_trace"),
        ([True, True, True, False, False], "mixed_or_seed_sensitive"),
        ([None, None, None, None, None], "descriptive_only"),
    ],
)
def test_experiment_classification_uses_registered_outer_seed_policy(
    observations: list[bool | None],
    expected: str,
) -> None:
    manifests = [
        _manifest("alpha", seed, null_supported=value, score=float(index + 1))
        for index, (seed, value) in enumerate(zip([101, 102, 103, 104, 105], observations, strict=True))
    ]

    summary = corpus_module._experiment_summary(
        experiment_id="alpha",
        manifests=manifests,
        config=_config(),
    )

    assert summary["classification"] == expected
    assert summary["completed_seed_count"] == 5
    assert summary["numeric_summaries"]["score"] == {
        "n": 5,
        "mean": 3.0,
        "std": pytest.approx(2**0.5),
        "min": 1.0,
        "max": 5.0,
    }
    assert "ignored_sequence" not in summary["numeric_summaries"]


def test_build_corpus_aggregates_synthetic_seed_manifests_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, config_path, run_root, built = _make_synthetic_corpus(tmp_path, monkeypatch)

    assert built["eligible_experiment_ids"] == ["alpha", "f_beta"]
    assert built["eligible_experiment_count"] == 2
    assert built["complete_experiment_ids"] == ["alpha", "f_beta"]
    assert built["corpus_complete"] is True
    assert all(row["complete"] is True for row in built["seed_coverage"].values())
    assert built["experiment_summaries"]["alpha"]["classification"] == "stable_null"
    assert built["experiment_summaries"]["f_beta"]["classification"] == "stable_candidate_trace"
    assert built["capability_pack_summaries"]["reasoning"]["classification_counts"] == {
        "stable_candidate_trace": 1,
        "stable_null": 1,
    }
    assert built["scientific_promotion"] is False
    assert verify_module._valid_seal(built, "corpus_sha256") is True
    assert corpus_module.build_corpus(config_path, run_root) == built


def test_verifier_regenerates_corpus_rejects_mutations_and_fails_on_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_path, config_path, run_root, built = _make_synthetic_corpus(tmp_path, monkeypatch)

    verification = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )
    assert verification["verification_complete"] is True
    assert verification["problems"] == []
    assert all(verification["checks"].values())
    assert verification["mutation_suite"] == {
        "count": 3,
        "rejected": 3,
        "results": {
            "promotion_flip": True,
            "seed_order_flip": True,
            "classification_tamper": True,
        },
    }
    assert verify_module._valid_seal(verification, "verification_sha256") is True

    tampered = copy.deepcopy(built)
    tampered["experiment_summaries"]["alpha"]["classification"] = "positive"
    _write_json(corpus_path, tampered)
    rejected = verify_module.verify_corpus(
        corpus_path=corpus_path,
        config_path=config_path,
        run_root=run_root,
    )
    assert rejected["verification_complete"] is False
    assert rejected["checks"]["corpus_self_hash"] is False
    assert rejected["checks"]["full_regeneration_match"] is False
    assert {"corpus_self_hash", "full_regeneration_match"}.issubset(rejected["problems"])
