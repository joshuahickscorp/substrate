
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mop.beds.starss23.feature_cache import (
    BuildReport,
    FeatureCacheRefusal,
    _config_payload,
    _feature_bytes,
    cache_dir_for,
    cache_key,
    cache_policy,
    load_cached_corpus,
)
from mop.substrate.events import atomic_write_bytes

HISTORICAL_KEYS = {
    "base": (
        "afe183956a01c573433057f615d3283c9727ad88ea65c2e8c09722d8f7a4faf2",
        "3c44a5778f904c47ba53a4fc985dc25c9fbfcac1cc11811748984c0a94bda875",
    ),
    "superflux": (
        "903b28b961377c27916d094f78825895ce00e6fd3020a5e79a7a2b8d4d09b2a7",
        "69b1845da8c77034753deb28d43e69a1fc1ab37ca0986a5d3099aa675f9193c3",
    ),
    "spatial_doa": (
        "d586b7ddcd88c18a8468ce9cd633c1e0beef6e84ec8aeb3afb2edcdad4bb01c7",
        "dad87344b09369dff2c35f17b2d4166103b2cf50544a1351980ee6dca224b69a",
    ),
}


@pytest.mark.parametrize("front_end", HISTORICAL_KEYS)
def test_all_policies_preserve_historical_cache_keys(front_end):
    default, custom = HISTORICAL_KEYS[front_end]
    assert cache_key(front_end=front_end) == default
    assert cache_key(
        front_end=front_end, foa_root=Path("/tmp/foa"), metadata_root=Path("/tmp/meta"),
        max_frames=123, n_val_rooms=2,
    ) == custom


@pytest.mark.parametrize(
    ("front_end", "config_identity", "manifest_identity"),
    [
        ("base", None, None),
        ("superflux", ("featurizer_id", "superflux_spectral"),
         ("featurizer_id", "superflux_spectral")),
        ("spatial_doa", ("front_end", "spatial_doa_active_intensity"), None),
    ],
)
def test_policy_projection_preserves_variant_identity(front_end, config_identity, manifest_identity):
    policy = cache_policy(front_end)
    assert policy.config_identity == config_identity
    assert policy.manifest_identity == manifest_identity
    payload = _config_payload(policy, policy.factory(), Path("/f"), Path("/m"), 7, 2)
    if config_identity:
        assert payload[config_identity[0]] == config_identity[1]
    assert payload["schema"] == policy.schema
    assert payload["max_frames"] == 7 and payload["n_val_rooms"] == 2


def test_partial_and_malformed_manifests_are_refused(tmp_path):
    kwargs = {"front_end": "base", "cache_root": tmp_path, "foa_root": "/f", "metadata_root": "/m"}
    directory = cache_dir_for(**kwargs)
    directory.mkdir(parents=True)
    (directory / "manifest.json.tmp").write_text("partial", encoding="utf-8")
    with pytest.raises(FeatureCacheRefusal, match="build the cache first"):
        load_cached_corpus(**kwargs)

    (directory / "manifest.json").write_text(json.dumps({"schema": "wrong/v9"}), encoding="utf-8")
    with pytest.raises(FeatureCacheRefusal, match="unexpected base cache schema"):
        load_cached_corpus(**kwargs)


def test_atomic_bytes_replace_and_feature_byte_projection(tmp_path):
    path = tmp_path / "features" / "clip.f8"
    expected = _feature_bytes(np.array([[1.0, 2.0]], dtype=np.float64))
    atomic_write_bytes(path, b"old")
    atomic_write_bytes(path, expected)
    assert path.read_bytes() == expected
    assert not path.with_suffix(".f8.tmp").exists()


def test_report_policies_keep_historical_public_shapes(tmp_path):
    base = BuildReport("base", "k", tmp_path, tmp_path / "m", 2, 4, 32, 10, 1.0, 0.5).numbers()
    superflux = BuildReport(
        "superflux", "k", tmp_path, tmp_path / "m", 2, 4, 32, 10, 1.0, 0.5
    ).numbers()
    spatial = BuildReport("spatial_doa", "k", tmp_path, tmp_path / "m", 2, 4, 32, 10, 1.0, 0.5).numbers()
    assert "manifest_bytes" in base and "seconds_per_clip" in base
    assert "total_superflux_featurize_flops" in superflux
    assert "manifest_bytes" not in spatial and "total_superflux_featurize_flops" not in spatial


def test_unknown_policy_is_refused():
    with pytest.raises(FeatureCacheRefusal, match="unknown STARSS23 cache front end"):
        cache_policy("invented")
