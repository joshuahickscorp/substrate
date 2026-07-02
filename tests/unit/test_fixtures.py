"""Tiny video-corpus generator: deterministic, class-foldered, with duplicate + short clips."""

import numpy as np

from mop.substrate import generate_video_corpus


def test_corpus_layout_and_counts(tmp_path):
    man = generate_video_corpus(tmp_path, n_classes=2, clips_per_class=3, frames=6, h=16, w=16, seed=0)
    assert man["classes"] == ["class0", "class1"]
    assert man["mocked"] is True and man["ext"] == ".npy"
    # class0 has 3 + duplicate + short = 5; class1 has 3
    assert man["per_class"]["class0"] == 5 and man["per_class"]["class1"] == 3
    assert man["n_clips"] == 8
    assert (tmp_path / "class0" / "clip0.npy").exists()


def test_deterministic_by_seed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_video_corpus(a, seed=7, frames=4, h=8, w=8)
    generate_video_corpus(b, seed=7, frames=4, h=8, w=8)
    x = np.load(a / "class0" / "clip0.npy")
    y = np.load(b / "class0" / "clip0.npy")
    assert np.array_equal(x, y)  # same seed -> identical bytes


def test_duplicate_clip_matches_clip0(tmp_path):
    man = generate_video_corpus(tmp_path, frames=4, h=8, w=8, seed=0, inject_duplicate=True)
    orig = np.load(tmp_path / "class0" / "clip0.npy")
    dup = np.load(man["duplicate"])
    assert np.array_equal(orig, dup)  # identical content -> detectable duplicate


def test_short_clip_has_fewer_frames(tmp_path):
    man = generate_video_corpus(tmp_path, frames=8, h=8, w=8, seed=0, inject_short=True)
    short = np.load(man["short"])
    full = np.load(tmp_path / "class0" / "clip0.npy")
    assert short.shape[0] < full.shape[0]
