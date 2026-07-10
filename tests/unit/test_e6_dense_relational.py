import json
from pathlib import Path

import numpy as np
import pytest

from mop.experiments.e6_dense_relational import (
    TokenReadoutSpec,
    build_mechanics_fixture,
    build_pair_gate,
    run_dense_relational,
)
from mop.substrate.cache_manifest import (
    RANDOM_INIT_RECEIPT_SCHEMA,
    validate_cache_manifest,
    write_cache_manifest,
)
from mop.substrate.latent_store import LatentStore


@pytest.fixture
def dense_pair(tmp_path):
    fixture = build_mechanics_fixture(tmp_path)
    return fixture, fixture["stores"]["learned"], fixture["stores"]["random"]


def test_fixture_pair_is_citable_but_never_promotion_ready(dense_pair):
    fixture, learned, random = dense_pair
    assert fixture["all_ok"] is True
    gate = build_pair_gate(learned, random)
    assert gate["mechanics_ok"] is True
    assert gate["promotion_ready"] is False
    joined = " ".join(gate["promotion_problems"])
    assert "natural-video" in joined
    assert "inherited-frozen" in joined
    assert "random-control" in joined


def test_token_readout_is_bounded_parameter_matched_and_nonpromotable(dense_pair):
    _, learned, random = dense_pair
    result = run_dense_relational(
        learned,
        random,
        spec=TokenReadoutSpec(bins=4, feature_rank=8, summary_dim=16),
    )
    assert result["all_ok"] is True
    assert result["scientific_promotion"] is False
    assert result["bounded_interface"]["legacy_flattened_dim"] == 16 * 8
    assert result["bounded_interface"]["learned_head_input_dim"] == 16
    assert result["bounded_interface"]["full_cache_flattened_matrix_materialized"] is False
    assert all(row["parameter_match"] for row in result["per_seed"])
    assert result["claim_boundary"]["larger_vjepa21_variants_unlocked"] is False


def test_pair_gate_refuses_referent_or_sidecar_drift(dense_pair):
    _, learned, random = dense_pair
    referent_path = Path(random) / "referents.json"
    referents = json.loads(referent_path.read_text())
    referents[0] = "drifted-referent"
    referent_path.write_text(json.dumps(referents))
    gate = build_pair_gate(learned, random)
    assert gate["mechanics_ok"] is False
    assert "referent" in " ".join(gate["mechanics_problems"])


def test_pair_gate_requires_random_arm_to_declare_random_initialization(dense_pair):
    _, learned, random = dense_pair
    manifest_path = Path(random) / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["encoder_config"]["random_init"] = False
    # Keep the generic manifest self-hash coherent so this test isolates the E6 control-design gate.
    from mop.substrate.cache_manifest import json_sha256

    manifest["encoder_config_hash"] = json_sha256(manifest["encoder_config"])
    manifest_path.write_text(json.dumps(manifest))
    gate = build_pair_gate(learned, random)
    assert gate["mechanics_ok"] is False
    assert "random_init=true" in " ".join(gate["mechanics_problems"])


def test_citable_manifest_accepts_pinned_official_random_init_backend(tmp_path):
    store = LatentStore.create(tmp_path, "official-random", (2, 3), 4, 3, has_labels=True)
    features = np.arange(24, dtype="float32").reshape(4, 2, 3)
    store.write_batch(0, features, features.mean(axis=1), np.arange(4, dtype="int64"))
    store.finalize()
    receipt = {
        "schema": RANDOM_INIT_RECEIPT_SCHEMA,
        "weights_real": False,
        "backend": "vjepa_official_random_init",
        "model_id": "official-pytorch-only-vjepa21-vitb",
        "revision": "fixture-version",
        "seed": 7,
        "parameter_count": 16,
        "state_dict_tensors": 2,
        "state_dict_sha256": "c" * 64,
        "model_class": "app.vjepa_2_1.models.vision_transformer.VisionTransformer",
        "repository_commit": "a" * 40,
        "architecture_files": [
            {
                "path": "app/vjepa_2_1/models/vision_transformer.py",
                "bytes": 18_195,
                "sha256": "b" * 64,
            }
        ],
    }
    (store.root / "initialization_receipt.json").write_text(json.dumps(receipt))
    write_cache_manifest(
        store.root,
        encoder_config={
            "name": "official-random",
            "hf_id": "official-pytorch-only-vjepa21-vitb",
            "revision": "fixture-version",
            "actual_backend": "vjepa_official_random_init",
            "random_init": True,
            "random_init_seed": 7,
            "prefer_real": False,
            "require_real": False,
        },
        factors={"factor_a": [0, 0, 1, 1], "factor_b": [0, 1, 0, 1]},
        splits={"train": [0, 3], "val": [1], "test": [2]},
        referents=["r0", "r1", "r2", "r3"],
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="fixture-id",
        full_hash_arrays=True,
    )
    assert validate_cache_manifest(store.root, citable=True) == []
