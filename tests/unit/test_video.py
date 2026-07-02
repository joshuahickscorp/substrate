"""Real-video preprocessing + ingestion. The decode backend is lazy (no codec needed here):
read_video is monkeypatched so the clip-batching/labeling logic is tested without a real file."""

import torch

from mop.substrate import video


def test_preprocess_clip_thwc_shape_and_norm():
    frames = torch.randint(0, 256, (10, 48, 48, 3), dtype=torch.uint8)
    clip = video.preprocess_clip(frames, frames_per_clip=8, res=32)
    assert clip.shape == (8, 3, 32, 32)
    assert clip.dtype == torch.float32 and torch.isfinite(clip).all()
    # ImageNet-normalized: not in [0,1] anymore, roughly zero-centered
    assert clip.min() < 0.0 and abs(float(clip.mean())) < 2.0


def test_preprocess_clip_accepts_chw():
    frames = torch.rand(5, 3, 40, 40)  # [T,3,H,W] float already in [0,1]
    clip = video.preprocess_clip(frames, frames_per_clip=8, res=24)
    assert clip.shape == (8, 3, 24, 24)


def test_preprocess_temporal_pad_when_short():
    frames = torch.rand(2, 16, 16, 3)  # only 2 frames -> uniformly sampled up to 8
    clip = video.preprocess_clip(frames, frames_per_clip=8, res=16)
    assert clip.shape == (8, 3, 16, 16)


def test_read_video_no_backend_raises_clear(monkeypatch):
    # force the no-backend path and assert the unblock message
    import builtins

    real_import = builtins.__import__

    def block(name, *a, **k):
        if name in ("torchvision.io", "torchvision", "decord"):
            raise ImportError(name)
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", block)
    try:
        video.read_video("/nope/clip.mp4")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "video backend" in str(e)


def test_iter_video_clips_batches_and_labels(tmp_path, monkeypatch):
    (tmp_path / "catA").mkdir()
    (tmp_path / "catB").mkdir()
    for c in ("catA", "catB"):
        for i in range(2):
            (tmp_path / c / f"clip{i}.mp4").write_bytes(b"")  # placeholder; read_video is patched
    monkeypatch.setattr(
        video, "read_video", lambda p: torch.randint(0, 256, (6, 20, 20, 3), dtype=torch.uint8)
    )
    batches = list(video.iter_video_clips(tmp_path, frames_per_clip=4, res=16, batch=2))
    clips = torch.cat([b[0] for b in batches])
    labels = torch.cat([b[1] for b in batches])
    assert clips.shape == (4, 4, 3, 16, 16)  # 4 clips, 4 frames, 3, 16, 16
    assert set(labels.tolist()) == {0, 1}  # both classes present, sorted-folder index
