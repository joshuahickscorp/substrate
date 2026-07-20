import json

import pytest
import torch

from mop.devices import resolve
from mop.substrate import cache_latents
from mop.substrate import video as V
from mop.substrate.encoder import EncoderSpec, FrozenEncoder


def _make_tree(root, layout):
    for c, n in layout.items():
        (root / c).mkdir(parents=True)
        for i in range(n):
            (root / c / f"clip{i}.mp4").write_bytes(b"\x00")
    return root


def _frames(t=6, h=20, w=20, fill=None):
    if fill is not None:
        return torch.full((t, h, w, 3), fill, dtype=torch.uint8)
    return torch.randint(0, 256, (t, h, w, 3), dtype=torch.uint8)


@pytest.fixture
def mock_decode(monkeypatch):
    by_name: dict[str, torch.Tensor] = {}

    def fake(path):
        from pathlib import Path

        key = Path(path).name
        if key in by_name:
            return by_name[key]
        g = torch.Generator().manual_seed(abs(hash(key)) % (2**31))
        return torch.randint(0, 256, (6, 20, 20, 3), generator=g, dtype=torch.uint8)

    monkeypatch.setattr(V, "read_video", fake)
    return by_name


def _cpu_encoder(dim=8):
    return FrozenEncoder(EncoderSpec(name="t", embed_dim=dim, dense=False, pool="mean"))


def test_validate_source_manifest(tmp_path):
    _make_tree(tmp_path, {"cat": 2, "dog": 3, "ant": 1})  # unsorted on purpose
    m = V.validate_source(tmp_path)
    assert m["classes"] == ["ant", "cat", "dog"]  # sorted
    assert m["label_map"] == {"ant": 0, "cat": 1, "dog": 2}
    assert m["per_class"] == {"ant": 1, "cat": 2, "dog": 3}
    assert m["n_clips"] == 6


def test_validate_source_missing_dir(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        V.validate_source(tmp_path / "nope")


def test_validate_source_not_a_dir(tmp_path):
    p = tmp_path / "f.mp4"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="not a directory"):
        V.validate_source(p)


def test_validate_source_no_class_subfolders(tmp_path):
    (tmp_path / "loose.mp4").write_bytes(b"")  # files at top level, no class dirs
    with pytest.raises(ValueError, match="no class subfolders"):
        V.validate_source(tmp_path)


def test_validate_source_empty_class(tmp_path):
    _make_tree(tmp_path, {"cat": 2})
    (tmp_path / "empty").mkdir()  # a class folder with zero video files
    with pytest.raises(ValueError, match="empty"):
        V.validate_source(tmp_path)


def test_validate_source_ignores_non_video(tmp_path):
    _make_tree(tmp_path, {"cat": 1})
    (tmp_path / "cat" / "notes.txt").write_text("x")  # non-video ignored, not counted
    m = V.validate_source(tmp_path)
    assert m["per_class"]["cat"] == 1


def test_cache_end_to_end_label_map_and_hashes(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "src", {"a": 2, "b": 2})
    manifest = V.validate_source(src)
    hashes: list[str] = []
    clips = V.iter_video_clips(src, frames_per_clip=4, res=16, batch=2, hashes_out=hashes)
    store = cache_latents(_cpu_encoder(), clips, tmp_path / "cache", "vid", total=4, device=resolve("cpu"))
    assert len(store) == 4
    assert store.meta.feat_shape == [8]
    p = V.write_label_map(store.root, manifest["label_map"])
    assert json.loads(p.read_text()) == {"a": 0, "b": 1}
    assert V.read_label_map(store.root) == {"a": 0, "b": 1}
    assert len(hashes) == 4
    assert all(len(h) == 64 for h in hashes)


def test_duplicate_content_detected(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "src", {"a": 1, "b": 1})
    mock_decode["clip0.mp4"] = _frames(fill=7)
    records = list(V.iter_clip_records(src, frames_per_clip=4, res=16))
    hs = [h for _, _, h in records]
    assert len(hs) == 2
    assert hs[0] == hs[1]  # same content -> same hash
    assert len(set(hs)) == 1  # duplicate count derivable as len-len(set)


def test_duplicate_warns(tmp_path, mock_decode, caplog):
    import logging

    src = _make_tree(tmp_path / "src", {"a": 1, "b": 1})
    mock_decode["clip0.mp4"] = _frames(fill=3)
    with caplog.at_level(logging.WARNING, logger="substrate.video"):
        list(V.iter_clip_records(src, frames_per_clip=4, res=16))
    assert any("duplicate clip content" in r.message for r in caplog.records)


def test_short_clip_padded(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "src", {"a": 1})
    mock_decode["clip0.mp4"] = _frames(t=2)  # only 2 frames, want 8
    (clip, label, h) = next(iter(V.iter_clip_records(src, frames_per_clip=8, res=16)))
    assert clip.shape == (8, 3, 16, 16)  # padded up to frames_per_clip
    assert label == 0 and len(h) == 64


def test_preprocess_empty_clip_raises():
    with pytest.raises(ValueError, match="empty clip"):
        V.preprocess_clip(torch.zeros(0, 16, 16, 3), frames_per_clip=4, res=8)


def test_corrupt_clip_skipped_not_crash(tmp_path, monkeypatch, caplog):
    import logging

    src = _make_tree(tmp_path / "src", {"a": 3})

    calls = {"n": 0}

    def flaky(path):
        calls["n"] += 1
        if calls["n"] == 2:  # second file is "corrupt"
            raise ValueError("moov atom not found")
        return _frames()

    monkeypatch.setattr(V, "read_video", flaky)
    with caplog.at_level(logging.WARNING, logger="substrate.video"):
        records = list(V.iter_clip_records(src, frames_per_clip=4, res=16))
    assert len(records) == 2  # 3 files, 1 skipped, did not crash
    assert any("skip undecodable clip" in r.message for r in caplog.records)


def test_no_backend_runtimeerror_surfaces(tmp_path, monkeypatch):
    src = _make_tree(tmp_path / "src", {"a": 1})

    def no_backend(path):
        raise RuntimeError("no video backend installed; ...")

    monkeypatch.setattr(V, "read_video", no_backend)
    with pytest.raises(RuntimeError, match="video backend"):
        list(V.iter_clip_records(src, frames_per_clip=4, res=16))


def test_partial_final_batch(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "src", {"a": 2, "b": 1})  # 3 clips, batch=2 -> [2, 1]
    batches = list(V.iter_video_clips(src, frames_per_clip=4, res=16, batch=2))
    sizes = [b[0].shape[0] for b in batches]
    assert sizes == [2, 1]
    labels = torch.cat([b[1] for b in batches])
    assert set(labels.tolist()) == {0, 1}


def test_limit_caps_clips(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "src", {"a": 5})
    hashes: list[str] = []
    list(V.iter_video_clips(src, frames_per_clip=4, res=16, batch=2, limit=3, hashes_out=hashes))
    assert len(hashes) == 3


def test_detect_partial_cache_absent(tmp_path):
    info = V.detect_partial_cache(tmp_path / "cache", "vid")
    assert info["exists"] is False
    assert info["count"] == 0
    assert info["has_label_map"] is False


def test_detect_partial_cache_present(tmp_path, mock_decode, caplog):
    import logging

    src = _make_tree(tmp_path / "src", {"a": 2, "b": 2})
    clips = V.iter_video_clips(src, frames_per_clip=4, res=16, batch=2)
    store = cache_latents(_cpu_encoder(), clips, tmp_path / "cache", "vid", total=4, device=resolve("cpu"))
    V.write_label_map(store.root, {"a": 0, "b": 1})
    with caplog.at_level(logging.WARNING, logger="substrate.video"):
        info = V.detect_partial_cache(tmp_path / "cache", "vid")
    assert info["exists"] is True
    assert info["has_meta"] is True
    assert info["count"] == 4
    assert info["has_label_map"] is True
    assert any("existing cache" in r.message for r in caplog.records)


def test_stratified_cap_covers_all_classes(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "vid", {"a": 5, "b": 5, "c": 5, "d": 5})
    labels = []
    for _x, y in V.iter_video_clips(src, frames_per_clip=4, res=8, batch=2, limit=4, stratified=True):
        labels.extend(int(v) for v in y.tolist())
    assert sorted(labels) == [0, 1, 2, 3]  # every class represented under the cap


def test_unstratified_cap_is_a_prefix(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "vid", {"a": 5, "b": 5, "c": 5, "d": 5})
    labels = []
    for _x, y in V.iter_video_clips(src, frames_per_clip=4, res=8, batch=2, limit=4):
        labels.extend(int(v) for v in y.tolist())
    assert set(labels) == {0}  # all 4 came from the first class


def test_limit_zero_yields_no_clips(tmp_path, mock_decode):
    src = _make_tree(tmp_path / "vid", {"a": 3, "b": 3})
    batches = list(V.iter_video_clips(src, frames_per_clip=4, res=8, batch=2, limit=0))
    assert batches == []
