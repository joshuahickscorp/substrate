import hashlib
import json

import pytest

from mop.evidence import canonical_sha256
from mop.substrate.cache_manifest import (
    ENCODER_RECEIPT_SCHEMA,
    RANDOM_INIT_RECEIPT_SCHEMA,
    SCHEMA,
    validate_cache_manifest,
)


def _json(path, value):
    path.write_text(json.dumps(value))


def _fingerprint(path, role):
    data = path.read_bytes()
    digest = hashlib.sha256(f"{path.name}:{len(data)}:".encode() + data).hexdigest()
    return {
        "role": role,
        "path": path.name,
        "bytes": len(data),
        "sha256": digest,
        "hash_kind": "full",
        "sample_bytes": 1024 * 1024,
    }


def _random_config(**changes):
    return {
        "hf_id": "fixture/random-vit",
        "revision": "fixture",
        "actual_backend": "vjepa_hf_random_init",
        "random_init": True,
        "random_init_seed": 7,
        "prefer_real": False,
        "require_real": False,
        **changes,
    }


def _random_receipt(**changes):
    return {
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
        **changes,
    }


def _weighted_receipt(**changes):
    return {
        "schema": ENCODER_RECEIPT_SCHEMA,
        "weights_real": True,
        "backend": "vjepa_hf",
        "model_id": "fixture/model",
        "revision": "fixture",
        "files": [{"path": "model.safetensors", "bytes": 4, "sha256": "a" * 64}],
        **changes,
    }


def _store(tmp_path, objective="programmatic", *, config=None, receipt="default"):
    root = tmp_path / "cache"
    root.mkdir(parents=True)
    meta = {"count": 2}
    _json(root / "meta.json", meta)
    _json(root / "referents.json", ["a", "b"])
    if config is None:
        config = (
            _random_config()
            if objective == "random-control"
            else {
                "hf_id": "fixture/model",
                "revision": "fixture",
                "actual_backend": "vjepa_hf",
            }
        )
    sidecars = [_fingerprint(root / "referents.json", "referents")]
    if objective in {
        "inherited-frozen",
        "self-supervised",
        "supervised",
        "learned-shell",
        "custom-substrate",
    }:
        filename, role = "encoder_receipt.json", "encoder_receipt"
        receipt = _weighted_receipt() if receipt == "default" else receipt
    elif objective == "random-control":
        filename, role = "initialization_receipt.json", "initialization_receipt"
        receipt = _random_receipt() if receipt == "default" else receipt
    else:
        receipt = None if receipt == "default" else receipt
    if receipt is not None:
        _json(root / filename, receipt)
        sidecars.append(_fingerprint(root / filename, role))
    manifest = {
        "schema": SCHEMA,
        "store": {"path": root.name, "meta": meta},
        "encoder_config": config,
        "encoder_config_hash": canonical_sha256(config),
        "form": {"kind": "vision", "objective": objective, "referent_scheme": "fixture-id"},
        "arrays": [],
        "sidecars": sidecars,
    }
    _json(root / "cache_manifest.json", manifest)
    return root, manifest


def test_missing_or_nonmapping_manifest_fails_closed(tmp_path):
    assert validate_cache_manifest(tmp_path) == ["cache_manifest.json missing"]
    _json(tmp_path / "cache_manifest.json", [])
    assert validate_cache_manifest(tmp_path) == ["cache_manifest.json must contain a JSON mapping"]


def test_programmatic_manifest_is_citable_and_detects_tampering(tmp_path):
    root, _ = _store(tmp_path)
    assert validate_cache_manifest(root) == []
    _json(root / "referents.json", ["changed", "b"])
    assert "sidecar referents.json sha256 changed" in " ".join(validate_cache_manifest(root))


def test_citable_manifest_requires_schema_form_encoder_and_referents(tmp_path):
    root, manifest = _store(tmp_path)
    manifest.update(schema="v1", form=None, encoder_config=None, encoder_config_hash=None, sidecars=[])
    (root / "referents.json").unlink()
    _json(root / "cache_manifest.json", manifest)
    problems = " ".join(validate_cache_manifest(root))
    for expected in (
        "schema",
        "form declaration",
        "encoder_config",
        "referent sidecar",
        "requires referents",
    ):
        assert expected in problems


def test_weighted_manifest_requires_and_validates_immutable_receipt(tmp_path):
    root, manifest = _store(tmp_path, "inherited-frozen", receipt=None)
    assert "requires encoder_receipt.json" in " ".join(validate_cache_manifest(root))
    root, _ = _store(tmp_path / "valid", "inherited-frozen")
    assert validate_cache_manifest(root) == []
    receipt = _weighted_receipt(model_id="other/model", revision="other")
    _json(root / "encoder_receipt.json", receipt)
    problems = " ".join(validate_cache_manifest(root))
    assert "model_id mismatch" in problems and "revision mismatch" in problems


def test_random_control_requires_and_validates_realized_initialization(tmp_path):
    root, _ = _store(tmp_path, "random-control", receipt=None)
    assert "requires initialization_receipt.json" in " ".join(validate_cache_manifest(root))
    root, _ = _store(tmp_path / "valid", "random-control")
    assert validate_cache_manifest(root) == []
    receipt = _random_receipt(revision="other", seed=8)
    receipt.pop("state_dict_sha256")
    _json(root / "initialization_receipt.json", receipt)
    problems = " ".join(validate_cache_manifest(root))
    for expected in ("state_dict_sha256", "revision mismatch", "random seed mismatch"):
        assert expected in problems


@pytest.mark.parametrize(
    "objective,filename",
    [("inherited-frozen", "encoder_receipt.json"), ("random-control", "initialization_receipt.json")],
)
def test_malformed_receipt_fails_closed_instead_of_crashing(tmp_path, objective, filename):
    root, _ = _store(tmp_path, objective, receipt=[])
    assert f"{filename} must contain a JSON mapping" in validate_cache_manifest(root)


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (lambda root, manifest: manifest["form"].update(objective="invalid"), "form.objective"),
        (lambda root, manifest: manifest["store"].update(meta={"count": 3}), "meta.json differs"),
        (lambda root, manifest: manifest.update(encoder_config_hash="0" * 64), "encoder_config_hash"),
        (lambda root, manifest: (root / "referents.json").unlink(), "sidecar referents.json missing"),
    ],
)
def test_core_manifest_mutations_fail_closed(tmp_path, mutation, expected):
    root, manifest = _store(tmp_path)
    mutation(root, manifest)
    _json(root / "cache_manifest.json", manifest)
    assert expected in " ".join(validate_cache_manifest(root))


def test_factor_referent_and_split_shape_contracts(tmp_path):
    root, _ = _store(tmp_path)
    _json(root / "factors.json", {"shape": [0]})
    _json(root / "referents.json", ["duplicate", "duplicate"])
    _json(root / "splits.json", {"train": [0, 0]})
    problems = " ".join(validate_cache_manifest(root))
    assert "factor 'shape' length 1 != cache count 2" in problems
    assert "referents contain duplicate ids" in problems
    assert "split 'train' contains duplicate indices" in problems
