"""Frontier 8: failure rehearsal. A system that fails SILENTLY (wrong answer, no signal) is
worse than one that crashes loudly. These tests rehearse the unhappy paths and assert each one
fails LOUDLY and clearly: a missing video codec, a placeholder encoder asked for real weights, a
bad device kind, an experiment with no null, a wrong cache path, an empty plan, and a resume that
must reuse a checkpoint instead of recomputing. Every assertion is on the failure (or the safe
no-op), never on a value that papers over the mistake. Small, deterministic, no downloads.
"""

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


# (1) missing video backend -> read_video raises RuntimeError with an install hint.
def test_missing_video_backend_raises_with_install_hint(monkeypatch):
    real_import = builtins.__import__
    blocked_names = []

    def blocked(name, *args, **kwargs):
        # block exactly the two lazy decode backends, leave everything else importable
        if name == "decord" or name.split(".")[0] == "torchvision":
            blocked_names.append(name)
            raise ImportError(f"blocked {name} for the test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError) as ei:
        video.read_video("/tmp/does-not-matter.mp4")
    msg = str(ei.value).lower()
    # loud AND actionable: it names the unblock, not just "error"
    assert "video backend" in msg
    assert "pip install" in msg and (".[video]" in msg or "decord" in msg)
    # prove the RuntimeError is the blocked-backend path (it actually tried both backends),
    # not some incidental failure that happens to raise RuntimeError
    assert any(n.split(".")[0] == "torchvision" for n in blocked_names)
    assert "decord" in blocked_names


# (2) placeholder V-JEPA 2.1 dense config under prefer_real -> validate_encoder raises ConfigError.
def test_placeholder_vjepa21_under_prefer_real_raises():
    cfg = compose(["encoder=vjepa21_vitl", "encoder.prefer_real=true"])
    # the config is genuinely the unavailable placeholder, not a typo on our side
    assert cfg.encoder.available is False
    assert cfg.encoder.prefer_real is True
    with pytest.raises(ConfigError) as ei:
        validate_encoder(cfg)
    m = str(ei.value)
    assert "prefer_real" in m and "available=false" in m  # explains the contradiction


# (3) bad device kind -> validate_device raises.
def test_bad_device_kind_raises():
    from omegaconf import OmegaConf

    with pytest.raises(ConfigError):
        validate_device(OmegaConf.create({"device": {"kind": "quantum"}}))
    # a real kind passes (guards against the check rejecting everything)
    validate_device(OmegaConf.create({"device": {"kind": "cpu"}}))


# (4) missing null_hypothesis -> validate_experiment raises (doctrine contract).
def test_missing_null_hypothesis_raises():
    from omegaconf import OmegaConf

    with pytest.raises(ConfigError) as ei:
        validate_experiment(OmegaConf.create({"experiment": {"id": "demo"}}))
    assert "null_hypothesis" in str(ei.value)
    # blank string is treated the same as missing
    with pytest.raises(ConfigError):
        validate_experiment(OmegaConf.create({"experiment": {"id": "demo", "null_hypothesis": "  "}}))
    # a declared null passes
    validate_experiment(OmegaConf.create({"experiment": {"id": "demo", "null_hypothesis": "no gain"}}))


# (5) MPS-fallback route annotation: cpu has no amp, auto resolves to a real present device.
def test_device_resolve_route_annotation():
    cpu = devices.resolve("cpu")
    assert cpu.kind == "cpu"
    assert cpu.supports_amp is False
    assert cpu.amp_dtype == "none"  # no fp16 autocast claim on cpu
    auto = devices.resolve("auto")
    assert auto.kind in {"cpu", "mps", "cuda"}  # a real, present device, never "auto"
    # autocast is a safe no-op on cpu (the MPS-fallback contract: never crash off-Metal)
    with devices.autocast(cpu):
        pass


# (6) bad cache source path -> iter_video_clips on a nonexistent dir raises a clear error.
def test_bad_cache_source_path_raises(tmp_path):
    missing = tmp_path / "no-such-clips-dir"
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        # iterator is lazy; drain it to force the walk
        list(video.iter_video_clips(missing, frames_per_clip=4, res=8, batch=1))
    # a path that is a FILE, not a dir, is also rejected (not silently treated as empty)
    not_a_dir = tmp_path / "a_file.mp4"
    not_a_dir.write_bytes(b"not a video")
    with pytest.raises((FileNotFoundError, NotADirectoryError)):
        list(video.iter_video_clips(not_a_dir, frames_per_clip=4, res=8, batch=1))


# (7) empty queue plan -> run_queue with no enabled tiers plans 0 legs and does not crash.
def test_empty_queue_plan_is_safe_noop():
    # the literal spec call: must not crash and must return a well-formed summary
    out = queue.run_queue(dry_run=True, enabled_tiers=set())
    assert out["dry_run"] is True
    assert "totals" in out and "planned" in out
    assert out["totals"]["planned_legs"] == len(out["planned"])
    # a tier set disjoint from every ENABLED leg yields a genuinely empty plan (0 run-units),
    # which is the real "nothing to run" assertion; E and R are gated out at toy scale.
    for tiers in ({"E"}, {"R"}, {"E", "R"}):
        empty = queue.run_queue(dry_run=True, enabled_tiers=tiers)
        assert empty["totals"]["planned_legs"] == 0
        assert empty["planned"] == []
        assert empty["totals"]["run_units_this_scale"] == 0


# (8) interrupted-run resume: an existing checkpoint is reused without recomputing.
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
    # _run_one must short-circuit on the existing ckpt: no compose, no run_experiment.
    rec = cpu_pool._run_one(key, ["experiment=e8_dendritic", "device=cpu"], str(tmp_path))
    assert rec["cached"] is True  # the loud signal that it was reused, not recomputed
    assert rec["metrics"] == {"avg_accuracy": 0.81}  # the cached result came back intact
    assert rec["key"] == key


# (9) empty class folder -> validate_source raises with a clear, listing message (not silent skip).
def test_empty_class_folder_raises(tmp_path):
    (tmp_path / "full").mkdir()
    np.save(tmp_path / "full" / "clip0.npy", np.zeros((4, 8, 8, 3), dtype=np.uint8))
    (tmp_path / "empty").mkdir()  # a class with zero clips
    with pytest.raises(ValueError) as ei:
        video.validate_source(tmp_path)
    assert "empty" in str(ei.value)  # the offending class is named


# (10) bad cache metadata -> validate_cache flags it (does not pretend the cache is fine).
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
