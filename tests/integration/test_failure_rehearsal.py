
from __future__ import annotations

import builtins
import json

import numpy as np
import pytest

from mop import devices
from mop.config import compose
from mop.harness import cpu_pool, queue
from mop.harness.validate import (
    ConfigError,
    validate_device,
    validate_encoder,
    validate_experiment,
)
from mop.substrate import video


def test_missing_video_backend_raises_with_install_hint(monkeypatch):
    real_import = builtins.__import__
    blocked_names = []

    def blocked(name, *args, **kwargs):
        if name == "decord" or name.split(".")[0] == "torchvision":
            blocked_names.append(name)
            raise ImportError(f"blocked {name} for the test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError) as ei:
        video.read_video("/tmp/does-not-matter.mp4")
    msg = str(ei.value).lower()
    assert "video backend" in msg
    assert "pip install" in msg and (".[video]" in msg or "decord" in msg)
    assert any(n.split(".")[0] == "torchvision" for n in blocked_names)
    assert "decord" in blocked_names


def test_placeholder_vjepa21_under_prefer_real_raises():
    cfg = compose(["encoder=vjepa21_vitl", "encoder.prefer_real=true"])
    assert cfg.encoder.available is False
    assert cfg.encoder.prefer_real is True
    with pytest.raises(ConfigError) as ei:
        validate_encoder(cfg)
    m = str(ei.value)
    assert "prefer_real" in m and "available=false" in m  # explains the contradiction


def test_bad_device_kind_raises():
    from omegaconf import OmegaConf

    with pytest.raises(ConfigError):
        validate_device(OmegaConf.create({"device": {"kind": "quantum"}}))
    validate_device(OmegaConf.create({"device": {"kind": "cpu"}}))


def test_missing_null_hypothesis_raises():
    from omegaconf import OmegaConf

    with pytest.raises(ConfigError) as ei:
        validate_experiment(OmegaConf.create({"experiment": {"id": "demo"}}))
    assert "null_hypothesis" in str(ei.value)
    with pytest.raises(ConfigError):
        validate_experiment(OmegaConf.create({"experiment": {"id": "demo", "null_hypothesis": "  "}}))
    validate_experiment(OmegaConf.create({"experiment": {"id": "demo", "null_hypothesis": "no gain"}}))


def test_device_resolve_route_annotation():
    cpu = devices.resolve("cpu")
    assert cpu.kind == "cpu"
    assert cpu.supports_amp is False
    assert cpu.amp_dtype == "none"  # no fp16 autocast claim on cpu
    auto = devices.resolve("auto")
    assert auto.kind in {"cpu", "mps", "cuda"}  # a real, present device, never "auto"
    with devices.autocast(cpu):
        pass


def test_bad_cache_source_path_raises(tmp_path):
    missing = tmp_path / "no-such-clips-dir"
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        list(video.iter_video_clips(missing, frames_per_clip=4, res=8, batch=1))
    not_a_dir = tmp_path / "a_file.mp4"
    not_a_dir.write_bytes(b"not a video")
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        list(video.iter_video_clips(not_a_dir, frames_per_clip=4, res=8, batch=1))


def test_empty_queue_plan_is_safe_noop():
    out = queue.run_queue(dry_run=True, enabled_tiers=set())
    assert out["dry_run"] is True
    assert "totals" in out and "planned" in out
    assert out["totals"]["planned_legs"] == len(out["planned"])
    for tiers in ({"E"}, {"R"}, {"E", "R"}):
        empty = queue.run_queue(dry_run=True, enabled_tiers=tiers)
        assert empty["totals"]["planned_legs"] == 0
        assert empty["planned"] == []
        assert empty["totals"]["run_units_this_scale"] == 0


def test_interrupted_run_resumes_from_checkpoint(tmp_path):
    key = "track99_demo_seed0"
    payload = {
        "key": key,
        "overrides": ["experiment=e8_dendritic", "device=cpu"],
        "metrics": {"avg_accuracy": 0.81},
        "seconds": 4.2,
        "cached": False,
    }
    (tmp_path / f"{key}.json").write_text(json.dumps(payload))
    rec = cpu_pool._run_one(key, ["experiment=e8_dendritic", "device=cpu"], str(tmp_path))
    assert rec["cached"] is True  # the loud signal that it was reused, not recomputed
    assert rec["metrics"] == {"avg_accuracy": 0.81}  # the cached result came back intact
    assert rec["key"] == key


def test_empty_class_folder_raises(tmp_path):
    (tmp_path / "full").mkdir()
    np.save(tmp_path / "full" / "clip0.npy", np.zeros((4, 8, 8, 3), dtype=np.uint8))
    (tmp_path / "empty").mkdir()  # a class with zero clips
    with pytest.raises(ValueError) as ei:
        video.validate_source(tmp_path)
    assert "empty" in str(ei.value)  # the offending class is named


def test_bad_cache_metadata_flagged(tmp_path):
    from mop import devices
    from mop.substrate import EncoderSpec, FrozenEncoder, cache_latents, synthetic_clips
    from mop.substrate.cache_tools import validate_cache

    enc = FrozenEncoder(EncoderSpec("vjepa2_vitl_fpc64_256", 16, dense=False, pool="mean"))
    store = cache_latents(
        enc, synthetic_clips(n=8, batch=4, n_classes=2), tmp_path, "c", total=8, device=devices.resolve("cpu")
    )
    assert validate_cache(store.root) == []  # honest cache is clean
    meta = json.loads((store.root / "meta.json").read_text())
    meta["count"] = 999  # corrupt the declared length
    (store.root / "meta.json").write_text(json.dumps(meta))
    assert validate_cache(store.root), "corrupted meta.json must be flagged, not trusted"
