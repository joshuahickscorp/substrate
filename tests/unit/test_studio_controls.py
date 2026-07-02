"""Synthetic control expansion: families generate deterministically, flow through the real
validate_source contract, honor the byte budget (truncate, never overrun), and stay reproducible."""

import numpy as np
import pytest

from mop.studio import controls
from mop.substrate.video import validate_source


def test_generate_default_families(tmp_path):
    out = controls.generate_controls(tmp_path, clips_per_class=2, frames=4, h=16, w=16, seed=0)
    names = {f["name"] for f in out["families"]}
    assert names == set(controls.DEFAULT_FAMILIES)
    assert out["mocked"] is True and out["kind"] == "structured-synthetic"
    assert all(f["n_clips"] > 0 for f in out["families"])


def test_each_family_is_a_valid_class_folder_source(tmp_path):
    out = controls.generate_controls(tmp_path, clips_per_class=2, frames=4, h=16, w=16, seed=1)
    for fam in out["families"]:
        v = validate_source(fam["root"])  # the REAL contract the cache pipeline uses
        assert len(v["classes"]) >= 2
        assert v["n_clips"] == fam["n_clips"]


def test_clips_have_expected_shape(tmp_path):
    controls.generate_controls(
        tmp_path, families=["moving_object"], clips_per_class=1, frames=6, h=12, w=12, seed=0
    )
    arrs = list((tmp_path / "moving_object").rglob("*.npy"))
    assert arrs
    a = np.load(arrs[0])
    assert a.dtype == np.uint8
    assert a.shape == (6, 12, 12, 3)


def test_deterministic_by_seed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    controls.generate_controls(a, families=["moving_object"], clips_per_class=2, frames=4, h=8, w=8, seed=7)
    controls.generate_controls(b, families=["moving_object"], clips_per_class=2, frames=4, h=8, w=8, seed=7)
    fa = sorted((a / "moving_object").rglob("*.npy"))
    fb = sorted((b / "moving_object").rglob("*.npy"))
    assert len(fa) == len(fb) and fa
    for pa, pb in zip(fa, fb, strict=True):
        assert np.array_equal(np.load(pa), np.load(pb))


def test_noisy_tv_classes_differ(tmp_path):
    # the noisy-TV control: class1 is pure noise, class0 a structured signal; they must differ a lot
    controls.generate_controls(
        tmp_path, families=["noisy_tv"], clips_per_class=1, frames=6, h=16, w=16, seed=0
    )
    c0 = np.load(next((tmp_path / "noisy_tv" / "class0").glob("*.npy"))).astype(float)
    c1 = np.load(next((tmp_path / "noisy_tv" / "class1").glob("*.npy"))).astype(float)
    assert abs(c0.std() - c1.std()) > 1.0  # noise has much higher variance


def test_byte_budget_truncates_not_overruns(tmp_path):
    # a budget that fits only a couple of clips: generation stops and reports truncation
    out = controls.generate_controls(
        tmp_path,
        families=["moving_object", "navigation", "occlusion_reveal"],
        clips_per_class=10,
        frames=8,
        h=32,
        w=32,
        seed=0,
        budget_gb=8 * 32 * 32 * 3 * 3 / 1e9,  # room for ~3 clips
    )
    assert out["truncated"] is True
    assert out["total_bytes"] <= out["budget_bytes"]


def test_unknown_family_raises(tmp_path):
    with pytest.raises(ValueError):
        controls.generate_controls(tmp_path, families=["not_a_family"])


def test_class_incremental_has_more_classes(tmp_path):
    out = controls.generate_controls(
        tmp_path, families=["class_incremental"], clips_per_class=1, frames=4, h=8, w=8
    )
    fam = out["families"][0]
    assert len(fam["classes"]) >= 4  # growing label space
