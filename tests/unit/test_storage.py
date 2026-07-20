
import json

import numpy as np
import pytest

from mop.substrate.storage import (
    DENSE_TOKENS_PER_CLIP,
    dir_size,
    estimate_cache_bytes,
    estimate_for_encoder,
    human_bytes,
    list_caches_with_size,
    prune_caches,
)


def test_estimate_cache_bytes_math():
    latents = 1000 * 1024 * 4
    keys = 1000 * 1024 * 4
    labels = 1000 * 8
    assert estimate_cache_bytes(1000, [1024], dtype="float32", has_labels=True) == latents + keys + labels


def test_estimate_cache_bytes_no_labels_drops_label_term():
    with_labels = estimate_cache_bytes(500, [256], has_labels=True)
    without = estimate_cache_bytes(500, [256], has_labels=False)
    assert with_labels - without == 500 * 8


def test_estimate_cache_bytes_fp16_halves_latents():
    f32 = estimate_cache_bytes(100, [512], dtype="float32", has_labels=False)
    f16 = estimate_cache_bytes(100, [512], dtype="float16", has_labels=False)
    assert (f32 - f16) == 100 * 512 * (4 - 2)


def test_estimate_cache_bytes_multi_dim_feat():
    b = estimate_cache_bytes(10, [8, 1024], dtype="float32", has_labels=False)
    assert b == 10 * 8 * 1024 * 4 + 10 * 1024 * 4


def test_estimate_cache_bytes_matches_real_npy_array_bytes(tmp_path):
    arr = np.zeros((96, 1024), dtype="float32")
    np.save(tmp_path / "latents.npy", arr)
    on_disk = (tmp_path / "latents.npy").stat().st_size
    raw = 96 * 1024 * 4
    assert on_disk - raw == 128  # npy header, the term estimate_cache_bytes intentionally omits


def test_human_bytes_formats_units():
    assert human_bytes(0) == "0 B"
    assert human_bytes(512) == "512 B"
    assert human_bytes(1024) == "1.00 KB"
    assert human_bytes(1536) == "1.50 KB"
    assert human_bytes(1024**2) == "1.00 MB"
    assert human_bytes(int(2.5 * 1024**3)) == "2.50 GB"
    assert human_bytes(1024**4).endswith("TB")


def test_estimate_for_encoder_pooled_vs_dense_differ_by_token_factor():
    cfg = {"name": "vjepa2_vitl_fpc64_256", "embed_dim": 1024}
    pooled = estimate_for_encoder(cfg, n_clips=100, dense=False)
    dense = estimate_for_encoder(cfg, n_clips=100, dense=True)
    assert pooled["tokens_per_clip"] == 1
    assert dense["tokens_per_clip"] == DENSE_TOKENS_PER_CLIP
    pooled_lat = 100 * 1024 * 4
    dense_lat = 100 * DENSE_TOKENS_PER_CLIP * 1024 * 4
    assert (dense["bytes"] - dense_lat) == (pooled["bytes"] - pooled_lat)  # identical keys+labels
    assert dense_lat // pooled_lat == DENSE_TOKENS_PER_CLIP
    assert dense["bytes"] > pooled["bytes"] * 1000


def test_estimate_for_encoder_honors_config_dense_flag():
    cfg = {"name": "dense_fixture", "embed_dim": 1024, "dense": True}
    out = estimate_for_encoder(cfg, n_clips=10)  # dense=False default, but config says dense
    assert out["dense"] is True
    assert out["tokens_per_clip"] == DENSE_TOKENS_PER_CLIP


def test_estimate_for_encoder_dense_tokens_override():
    cfg = {"name": "wide_fixture", "embed_dim": 1408}
    out = estimate_for_encoder(cfg, n_clips=5, dense=True, dense_tokens=10976)
    assert out["tokens_per_clip"] == 10976
    assert out["embed_dim"] == 1408


def _fake_cache(root, name, count=8, feat=4):
    d = root / name
    d.mkdir(parents=True)
    np.save(d / "latents.npy", np.zeros((count, feat), dtype="float32"))
    np.save(d / "keys.npy", np.zeros((count, feat), dtype="float32"))
    np.save(d / "labels.npy", np.zeros((count,), dtype="int64"))
    (d / "meta.json").write_text(
        json.dumps(
            {
                "name": name,
                "feat_shape": [feat],
                "dtype": "float32",
                "key_dim": feat,
                "count": count,
                "has_labels": True,
            }
        )
    )
    (d / "provenance.json").write_text(
        json.dumps({"encoder_backend": "frozen_random", "result_tag": "provisional"})
    )
    return d


def test_dir_size_sums_files(tmp_path):
    d = _fake_cache(tmp_path, "c")
    expected = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    assert dir_size(d) == expected
    assert dir_size(tmp_path / "missing") == 0


def test_list_caches_with_size_reports_meta_and_sorts(tmp_path):
    _fake_cache(tmp_path, "small", count=4)
    _fake_cache(tmp_path, "big", count=64)
    (tmp_path / "not_a_cache").mkdir()  # no meta.json -> skipped
    caches = list_caches_with_size(tmp_path)
    names = [c["name"] for c in caches]
    assert names == ["big", "small"]  # largest first
    assert caches[0]["count"] == 64
    assert caches[0]["feat_shape"] == [4]
    assert caches[0]["backend"] == "frozen_random"
    assert caches[0]["bytes"] > 0


def test_list_caches_missing_root_is_empty(tmp_path):
    assert list_caches_with_size(tmp_path / "nope") == []


def test_prune_dry_run_deletes_nothing(tmp_path):
    a = _fake_cache(tmp_path, "keep_me")
    b = _fake_cache(tmp_path, "drop_me")
    plan = prune_caches(tmp_path, keep=["keep_me"], dry_run=True)
    assert a.exists() and b.exists()
    assert (a / "latents.npy").exists() and (b / "latents.npy").exists()
    by_name = {p["name"]: p for p in plan}
    assert by_name["keep_me"]["kept"] is True
    assert by_name["keep_me"]["would_delete"] is False
    assert by_name["drop_me"]["would_delete"] is True
    assert all(p["deleted"] is False for p in plan)


def test_prune_default_is_dry_run(tmp_path):
    b = _fake_cache(tmp_path, "drop_me")
    plan = prune_caches(tmp_path)
    assert b.exists() and (b / "latents.npy").exists()
    assert plan[0]["would_delete"] is True
    assert plan[0]["deleted"] is False


def test_prune_apply_deletes_only_unkept(tmp_path):
    a = _fake_cache(tmp_path, "keep_me")
    b = _fake_cache(tmp_path, "drop_me")
    plan = prune_caches(tmp_path, keep=["keep_me"], dry_run=False)
    assert a.exists()  # kept survives
    assert not b.exists()  # unkept is gone
    by_name = {p["name"]: p for p in plan}
    assert by_name["keep_me"]["deleted"] is False
    assert by_name["drop_me"]["deleted"] is True


def test_estimate_cache_bytes_rejects_bad_input():
    with pytest.raises(ValueError):
        estimate_cache_bytes(-1, [16])
    with pytest.raises(ValueError):
        estimate_cache_bytes(10, [])
    with pytest.raises(ValueError):
        estimate_cache_bytes(10, [16], dtype="float8")
