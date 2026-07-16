from __future__ import annotations

import json
from pathlib import Path

from mop.studies import generation1_competence_atlas as atlas
from mop.studies import generation1_competence_atlas_verify as verify


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture_config(root: Path) -> Path:
    synthesis_path = root / "proof/GENERATION1_EVIDENCE_SYNTHESIS.json"
    verification_path = root / "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json"
    synthesis = {
        "schema": "mop-generation1-evidence-synthesis/v1",
        "synthesis_sha256": "1" * 64,
        "claim_boundaries": {"context_disjoint_actor_niches": {"status": "not_tested_by_g1_c0"}},
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    verification = {
        "schema": "mop-generation1-evidence-synthesis-verification/v1",
        "verification_sha256": "2" * 64,
        "verification_complete": True,
        "problems": [],
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    _write(synthesis_path, synthesis)
    _write(verification_path, verification)
    config: dict[str, object] = {
        "schema": atlas.CONFIG_SCHEMA,
        "campaign_id": "g1-c1-test",
        "claim_scope": atlas.CLAIM_SCOPE,
        "prerequisite": {
            "synthesis_path": "proof/GENERATION1_EVIDENCE_SYNTHESIS.json",
            "synthesis_file_sha256": atlas.sha256_file(synthesis_path),
            "synthesis_sha256": synthesis["synthesis_sha256"],
            "verification_path": "proof/GENERATION1_EVIDENCE_SYNTHESIS.verification.json",
            "verification_file_sha256": atlas.sha256_file(verification_path),
            "verification_sha256": verification["verification_sha256"],
        },
        "seeds": list(range(101, 109)),
        "difficulty_separations": [0.06, 0.1, 0.16],
        "dataset": {"n_train": 120, "n_test": 60, "n_classes": 4, "dim": 16},
        "training": {
            "epochs": 1,
            "homogeneous_actor": "mlp",
            "homogeneous_copies": 2,
            "torch_threads": 1,
        },
        "criteria": {
            "min_niche_advantage": 0.0,
            "min_oracle_headroom": 0.0,
            "min_reproducible_fraction": 0.0,
            "off_ceiling_max_accuracy": 1.0,
            "above_chance_margin": 0.0,
        },
        "controls": {
            "best_single": True,
            "random": True,
            "homogeneous": True,
            "oracle_actor": True,
            "abstention": True,
        },
        "activation_allowed": False,
        "scientific_promotion": False,
    }
    path = root / "configs/atlas.json"
    _write(path, config)
    return path


def test_small_atlas_resumes_and_independently_verifies(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    work_root = tmp_path / "runs/atlas"
    output = tmp_path / "proof/ATLAS.json"

    first = atlas.run_atlas(config, work_root, output, repo_root=tmp_path, seed_workers=3)

    assert first["complete"] is True
    assert first["grid"] == {
        "expected_seed_count": 8,
        "completed_seed_count": 8,
        "expected_seed_difficulty_cells": 24,
        "completed_seed_difficulty_cells": 24,
    }
    assert len(first["competence_tensor"]) == 3 * len(atlas.CONTEXTS) * len(atlas.ACTORS)
    assert first["activation_allowed"] is False
    assert first["scientific_promotion"] is False
    before = {path.name: path.stat().st_mtime_ns for path in (work_root / "seeds").glob("*.json")}

    second = atlas.run_atlas(config, work_root, output, repo_root=tmp_path)

    after = {path.name: path.stat().st_mtime_ns for path in (work_root / "seeds").glob("*.json")}
    assert second == first
    assert after == before
    assert [row["seed"] for row in second["seed_receipts"]] == list(range(101, 109))

    verification_path = tmp_path / "proof/ATLAS.verification.json"
    result = verify.verify_atlas(
        config,
        output,
        verification_path,
        repo_root=tmp_path,
    )

    assert result["verification_complete"] is True
    assert result["problems"] == []
    assert result["fresh_canary"]["matched"] is True
    assert result["mutation_suite"]["count"] == 8
    assert result["mutation_suite"]["rejected"] == 8
    assert result["verified_decision"]["ready_to_train_dispatcher"] is False


def test_seed_receipt_rejects_prediction_removal(tmp_path: Path) -> None:
    config_path = _fixture_config(tmp_path)
    config, config_sha256 = atlas.load_config(config_path)
    receipt = atlas.run_seed(config, config_sha256, int(config["seeds"][0]))
    receipt["difficulties"][0]["predictions"][atlas.ACTORS[0]].pop()
    receipt["seed_sha256"] = atlas.canonical_sha256(
        {key: value for key, value in receipt.items() if key != "seed_sha256"}
    )

    try:
        atlas.validate_seed_receipt(
            receipt,
            config,
            config_sha256,
            int(config["seeds"][0]),
        )
    except ValueError as exc:
        assert "prediction length" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("prediction removal was accepted")


def test_seed_worker_bounds_fail_closed(tmp_path: Path) -> None:
    config = _fixture_config(tmp_path)
    for workers in (0, atlas.MAX_SEED_WORKERS + 1):
        try:
            atlas.run_atlas(
                config,
                tmp_path / "runs/atlas",
                tmp_path / "proof/ATLAS.json",
                repo_root=tmp_path,
                seed_workers=workers,
            )
        except ValueError as exc:
            assert "seed workers" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("out-of-envelope seed worker count was accepted")
