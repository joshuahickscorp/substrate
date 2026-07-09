"""Cache data-plane receipts: sidecars, split membership, hashes, and cache-tool integration."""

import json

import numpy as np
import pytest

from mop.substrate.cache_manifest import (
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
