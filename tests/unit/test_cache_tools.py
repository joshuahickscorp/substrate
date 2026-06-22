"""Cache integrity tools: build a real LatentStore via the caching pipeline (frozen-random
encoder over synthetic clips), then exercise list_caches / cache_info / validate_cache on the
clean store, and prove validate_cache flags a deliberately corrupted store."""

import json

import numpy as np

from devsys import devices
from devsys.substrate import EncoderSpec, FrozenEncoder, cache_latents, synthetic_clips
from devsys.substrate.cache_tools import cache_info, list_caches, validate_cache


def _build(tmp_path, name="cache", n=24, dim=16, n_classes=4):
    """Cache n synthetic clips through a frozen-random encoder into tmp_path/<name>."""
    enc = FrozenEncoder(EncoderSpec(name, dim, dense=False, pool="mean"))
    dev = devices.resolve("cpu")
    store = cache_latents(
        enc,
        synthetic_clips(n=n, batch=8, n_classes=n_classes),
        tmp_path,
        name,
        total=n,
        device=dev,
    )
    return store.root


def test_validate_clean_store_returns_empty(tmp_path):
    root = _build(tmp_path)
    assert validate_cache(root) == []


def test_clean_store_writes_provenance_and_meta(tmp_path):
    root = _build(tmp_path)
    assert (root / "meta.json").exists()
    assert (root / "provenance.json").exists()


def test_cache_info_has_meta_and_provenance(tmp_path):
    root = _build(tmp_path)
    info = cache_info(root)
    assert info["meta"] is not None
    assert info["meta"]["count"] == 24
    assert info["provenance"] is not None
    # recomputed content id matches the provenance the writer stamped
    assert info["facts"]["cache_id_matches"] is True
    assert info["facts"]["recomputed_cache_id"] == info["provenance"]["cache_id"]


def test_cache_info_reports_label_range(tmp_path):
    root = _build(tmp_path, n_classes=4)
    facts = cache_info(root)["facts"]
    assert facts["label_min"] >= 0
    assert facts["label_max"] < 4


def test_list_caches_finds_the_store(tmp_path):
    _build(tmp_path, name="alpha")
    _build(tmp_path, name="beta")
    listed = list_caches(tmp_path)
    names = {c["name"] for c in listed}
    assert {"alpha", "beta"} <= names
    one = next(c for c in listed if c["name"] == "alpha")
    assert one["count"] == 24
    assert one["dim"] == 16
    # frozen-random encoder -> backend tag recorded in provenance
    assert one["backend"] == "frozen_random"
    assert one["result_tag"] in ("provisional", "structured-synthetic")


def test_list_caches_empty_root_is_empty(tmp_path):
    assert list_caches(tmp_path / "does_not_exist") == []


def test_corrupt_labels_and_count_is_flagged(tmp_path):
    """Truncate labels AND inflate the meta count: validate must complain."""
    root = _build(tmp_path, n=24)
    # truncate labels to fewer rows than count claims
    np.save(root / "labels.npy", np.zeros(5, dtype="int64"))
    meta = json.loads((root / "meta.json").read_text())
    meta["count"] = 24  # latents still 24 rows, labels now 5 -> mismatch
    (root / "meta.json").write_text(json.dumps(meta))
    problems = validate_cache(root)
    assert problems
    assert any("labels length" in p for p in problems)


def test_wrong_count_in_meta_flagged(tmp_path):
    """Inflate meta count beyond the latents on disk -> shape mismatch reported."""
    root = _build(tmp_path, n=24)
    meta = json.loads((root / "meta.json").read_text())
    meta["count"] = 999
    (root / "meta.json").write_text(json.dumps(meta))
    problems = validate_cache(root)
    assert any("shape" in p for p in problems)


def test_missing_meta_is_flagged(tmp_path):
    root = _build(tmp_path)
    (root / "meta.json").unlink()
    assert validate_cache(root) == ["meta.json missing"]


def test_dtype_mismatch_flagged(tmp_path):
    root = _build(tmp_path)
    meta = json.loads((root / "meta.json").read_text())
    meta["dtype"] = "float64"  # latents on disk are float32
    (root / "meta.json").write_text(json.dumps(meta))
    problems = validate_cache(root)
    assert any("dtype" in p for p in problems)


def test_bad_result_tag_in_provenance_flagged(tmp_path):
    root = _build(tmp_path)
    prov = json.loads((root / "provenance.json").read_text())
    prov["result_tag"] = "totally-made-up"
    (root / "provenance.json").write_text(json.dumps(prov))
    problems = validate_cache(root)
    assert any("result_tag" in p for p in problems)


def test_cache_id_mismatch_flagged(tmp_path):
    root = _build(tmp_path)
    prov = json.loads((root / "provenance.json").read_text())
    prov["cache_id"] = "deadbeefdeadbeef"
    (root / "provenance.json").write_text(json.dumps(prov))
    problems = validate_cache(root)
    assert any("cache_id mismatch" in p for p in problems)
