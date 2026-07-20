import json

import numpy as np
import pytest

from mop.substrate.cache_manifest import (
    ENCODER_RECEIPT_SCHEMA,
    RANDOM_INIT_RECEIPT_SCHEMA,
    SCHEMA,
    json_sha256,
    validate_cache_manifest,
    write_cache_manifest,
)
from mop.substrate.cache_tools import cache_info, validate_cache
from mop.substrate.latent_store import LatentStore


def _store(tmp_path, n=6):
    store = LatentStore.create(tmp_path, "cache", (4,), n, 4, dtype="float32", has_labels=True)
    latents = np.arange(n * 4, dtype="float32").reshape(n, 4)
    keys = latents / 10.0
    labels = np.arange(n, dtype="int64") % 2
    store.write_batch(0, latents, keys, labels)
    store.finalize()
    return store.root


def test_write_manifest_records_arrays_sidecars_splits_and_encoder_hash(tmp_path):
    root = _store(tmp_path)
    encoder_config = {"name": "toy_encoder", "embed_dim": 4, "dense": False}
    manifest = write_cache_manifest(
        root,
        encoder_config=encoder_config,
        factors={
            "object": ["cup", "cup", "ball", "ball", "book", "book"],
            "action": ["lift", "drop", "lift", "drop", "lift", "drop"],
        },
        splits={"train": [0, 1, 2, 3], "val": [4], "test": [5]},
        full_hash_arrays=True,
    )

    assert manifest["schema"] == SCHEMA
    assert manifest["encoder_config_hash"] == json_sha256(encoder_config)
    assert {a["role"] for a in manifest["arrays"]} == {"latents", "keys", "labels"}
    assert {s["role"] for s in manifest["sidecars"]} == {"factors", "splits"}
    columns = {(c["kind"], c["name"]) for c in manifest["index"]["columns"]}
    assert ("factor", "object") in columns
    assert ("split", "train") in columns
    assert validate_cache_manifest(root) == []
    assert validate_cache(root) == []
    assert cache_info(root)["facts"]["cache_manifest_clean"] is True


def test_manifest_validation_catches_sidecar_tampering(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(
        root,
        factors={"object": ["cup", "cup", "ball", "ball", "book", "book"]},
        splits={"train": [0, 1, 2, 3], "val": [4, 5]},
    )
    factors = json.loads((root / "factors.json").read_text())
    factors["object"][0] = "changed"
    (root / "factors.json").write_text(json.dumps(factors))

    problems = validate_cache_manifest(root)
    assert any("factors.json sha256 changed" in p for p in problems)
    all_cache_problems = validate_cache(root)
    assert any("cache_manifest.json" in p for p in all_cache_problems)


def test_manifest_rejects_factor_length_mismatch(tmp_path):
    root = _store(tmp_path, n=4)
    with pytest.raises(ValueError, match="length 2 != cache count 4"):
        write_cache_manifest(root, factors={"object": ["cup", "ball"]})


def test_manifest_rejects_bad_split_membership(tmp_path):
    root = _store(tmp_path, n=4)
    with pytest.raises(ValueError, match="duplicate"):
        write_cache_manifest(root, splits={"train": [0, 0]})
    with pytest.raises(ValueError, match="out-of-range"):
        write_cache_manifest(root, splits={"train": [0, 5]})
    with pytest.raises(ValueError, match="overlaps"):
        write_cache_manifest(root, splits={"train": [0, 1], "val": [1, 2]})


def test_manifest_missing_is_a_manifest_problem_not_a_cache_problem(tmp_path):
    root = _store(tmp_path)
    assert validate_cache_manifest(root) == ["cache_manifest.json missing"]
    assert validate_cache(root) == []


def test_manifest_records_form_declaration(tmp_path):
    root = _store(tmp_path)
    manifest = write_cache_manifest(
        root,
        form_kind="vision",
        form_objective="inherited-frozen",
        referent_scheme="clip-id",
    )
    assert manifest["schema"] == SCHEMA
    assert manifest["form"] == {
        "kind": "vision",
        "objective": "inherited-frozen",
        "referent_scheme": "clip-id",
    }
    assert validate_cache_manifest(root) == []


def test_manifest_without_form_block_is_valid(tmp_path):
    root = _store(tmp_path)
    manifest = write_cache_manifest(root)
    assert manifest["form"] is None
    assert validate_cache_manifest(root) == []


def test_manifest_rejects_bad_form_fields(tmp_path):
    root = _store(tmp_path)
    with pytest.raises(ValueError, match="form_kind"):
        write_cache_manifest(root, form_kind="not_a_kind")
    with pytest.raises(ValueError, match="form_objective"):
        write_cache_manifest(root, form_objective="not_an_objective")


def test_v1_manifest_still_validates(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(root)
    manifest = json.loads((root / "cache_manifest.json").read_text())
    manifest["schema"] = "mop-cache-data-plane/v1"
    manifest.pop("form")  # a genuine v1 manifest has no form block
    (root / "cache_manifest.json").write_text(json.dumps(manifest, indent=2))
    assert validate_cache_manifest(root) == []


def test_citable_manifest_requires_form_encoder_and_referents(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(root)
    problems = validate_cache_manifest(root, citable=True)
    assert any("form declaration" in p for p in problems)
    assert any("encoder_config" in p for p in problems)
    assert any("referents.json" in p for p in problems)
    assert validate_cache(root, citable=True)


def test_citable_manifest_roundtrip_with_v2_factor_metadata(tmp_path):
    root = _store(tmp_path)
    refs = [f"clip-{i}" for i in range(6)]
    manifest = write_cache_manifest(
        root,
        encoder_config={
            "name": "toy_encoder",
            "hf_id": "fixture/toy-encoder",
            "revision": "fixture-revision",
        },
        encoder_receipt={
            "schema": ENCODER_RECEIPT_SCHEMA,
            "weights_real": True,
            "model_id": "fixture/toy-encoder",
            "revision": "fixture-revision",
            "files": [{"path": "model.safetensors", "bytes": 4, "sha256": "a" * 64}],
        },
        factors={"shape": [0, 0, 1, 1, 2, 2]},
        factor_metadata={"clipset": "fixture", "seed": 0},
        referents=refs,
        form_kind="vision",
        form_objective="inherited-frozen",
        referent_scheme="clip-id",
    )
    factors = json.loads((root / "factors.json").read_text())
    assert factors["metadata"] == {"clipset": "fixture", "seed": 0}
    assert factors["columns"]["shape"] == [0, 0, 1, 1, 2, 2]
    assert any(s["role"] == "referents" for s in manifest["sidecars"])
    assert any(s["role"] == "encoder_receipt" for s in manifest["sidecars"])
    assert validate_cache_manifest(root, citable=True) == []
    assert validate_cache(root, citable=True) == []


def test_citable_learned_cache_requires_immutable_weight_receipt(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(
        root,
        encoder_config={"name": "toy_encoder", "revision": "fixture"},
        referents=[f"clip-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="inherited-frozen",
        referent_scheme="clip-id",
    )
    problems = validate_cache_manifest(root, citable=True)
    assert any("encoder_receipt.json" in problem for problem in problems)


def test_programmatic_citable_cache_does_not_forge_weight_receipt(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(
        root,
        encoder_config={"name": "known-programmatic-generator", "revision": "fixture"},
        referents=[f"referent-{i}" for i in range(6)],
        form_kind="symbolic",
        form_objective="programmatic",
        referent_scheme="generated-id",
    )
    assert validate_cache_manifest(root, citable=True) == []


def test_random_control_requires_hashed_architecture_and_seed(tmp_path):
    root = _store(tmp_path)
    encoder_config = {
        "name": "random-vit",
        "hf_id": "fixture/random-vit",
        "revision": "fixture",
        "actual_backend": "vjepa_hf_random_init",
        "random_init": True,
        "random_init_seed": 7,
        "prefer_real": False,
        "require_real": False,
    }
    write_cache_manifest(
        root,
        encoder_config=encoder_config,
        referents=[f"referent-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="generated-id",
    )
    assert any(
        "initialization_receipt.json" in problem for problem in validate_cache_manifest(root, citable=True)
    )
    receipt = {
        "schema": RANDOM_INIT_RECEIPT_SCHEMA,
        "weights_real": False,
        "backend": "vjepa_hf_random_init",
        "model_id": "fixture/random-vit",
        "revision": "fixture",
        "seed": 7,
        "parameter_count": 16,
        "state_dict_tensors": 2,
        "state_dict_sha256": "c" * 64,
        "model_class": "fixture.RandomVit",
        "architecture_files": [{"path": "config.json", "bytes": 4, "sha256": "b" * 64}],
    }
    (root / "initialization_receipt.json").write_text(json.dumps(receipt))
    write_cache_manifest(
        root,
        encoder_config=encoder_config,
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="generated-id",
    )
    assert validate_cache_manifest(root, citable=True) == []


def test_random_control_receipt_must_match_hashed_encoder_config(tmp_path):
    root = _store(tmp_path)
    encoder_config = {
        "name": "random-vit",
        "hf_id": "fixture/random-vit",
        "revision": "revision-a",
        "actual_backend": "vjepa_hf_random_init",
        "random_init": True,
        "random_init_seed": 7,
        "prefer_real": False,
        "require_real": False,
    }
    receipt = {
        "schema": RANDOM_INIT_RECEIPT_SCHEMA,
        "weights_real": False,
        "backend": "vjepa_hf_random_init",
        "model_id": "fixture/random-vit",
        "revision": "revision-b",
        "seed": 8,
        "parameter_count": 16,
        "state_dict_tensors": 2,
        "state_dict_sha256": "c" * 64,
        "model_class": "fixture.RandomVit",
        "architecture_files": [{"path": "config.json", "bytes": 4, "sha256": "b" * 64}],
    }
    (root / "initialization_receipt.json").write_text(json.dumps(receipt))
    write_cache_manifest(
        root,
        encoder_config=encoder_config,
        referents=[f"referent-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="generated-id",
    )
    problems = " ".join(validate_cache_manifest(root, citable=True))
    assert "revision mismatch" in problems
    assert "random seed mismatch" in problems


def test_random_control_requires_realized_state_hash_not_only_claimed_seed(tmp_path):
    root = _store(tmp_path)
    encoder_config = {
        "name": "random-vit",
        "hf_id": "fixture/random-vit",
        "revision": "fixture",
        "actual_backend": "vjepa_hf_random_init",
        "random_init": True,
        "random_init_seed": 7,
        "prefer_real": False,
        "require_real": False,
    }
    receipt = {
        "schema": RANDOM_INIT_RECEIPT_SCHEMA,
        "weights_real": False,
        "backend": "vjepa_hf_random_init",
        "model_id": "fixture/random-vit",
        "revision": "fixture",
        "seed": 7,
        "parameter_count": 16,
        "state_dict_tensors": 2,
        "model_class": "fixture.RandomVit",
        "architecture_files": [{"path": "config.json", "bytes": 4, "sha256": "b" * 64}],
    }
    (root / "initialization_receipt.json").write_text(json.dumps(receipt))
    write_cache_manifest(
        root,
        encoder_config=encoder_config,
        referents=[f"referent-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="generated-id",
    )
    assert "state_dict_sha256" in " ".join(validate_cache_manifest(root, citable=True))


def test_weight_receipt_identity_must_match_encoder_config(tmp_path):
    root = _store(tmp_path)
    write_cache_manifest(
        root,
        encoder_config={
            "name": "toy",
            "hf_id": "fixture/model-a",
            "revision": "revision-a",
            "actual_backend": "vjepa_hf",
        },
        encoder_receipt={
            "schema": ENCODER_RECEIPT_SCHEMA,
            "weights_real": True,
            "model_id": "fixture/model-b",
            "revision": "revision-b",
            "backend": "vjepa_hf",
            "files": [{"path": "model.safetensors", "bytes": 4, "sha256": "a" * 64}],
        },
        referents=[f"clip-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="inherited-frozen",
        referent_scheme="clip-id",
    )
    problems = " ".join(validate_cache_manifest(root, citable=True))
    assert "model_id mismatch" in problems
    assert "revision mismatch" in problems


def test_malformed_random_receipt_fails_closed_instead_of_crashing(tmp_path):
    root = _store(tmp_path)
    encoder_config = {
        "name": "random-vit",
        "hf_id": "fixture/random-vit",
        "revision": "fixture",
        "actual_backend": "vjepa_hf_random_init",
        "random_init": True,
        "random_init_seed": 7,
        "prefer_real": False,
        "require_real": False,
    }
    (root / "initialization_receipt.json").write_text("[]")
    write_cache_manifest(
        root,
        encoder_config=encoder_config,
        referents=[f"referent-{i}" for i in range(6)],
        form_kind="vision",
        form_objective="random-control",
        referent_scheme="generated-id",
    )
    assert "JSON mapping" in " ".join(validate_cache_manifest(root, citable=True))
