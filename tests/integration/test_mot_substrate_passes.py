
import json
import math
import types

import pytest
import scripts.cache_randominit_vitl_features as crf
import scripts.compositional_under_nuisance as cun
import torch
from scripts.cache_randominit_vitl_features import (
    CANONICAL,
    NUISANCE_COLS,
    assert_encoder_lane_free,
    extract_nuisance_draws,
    iter_nuisance_clips,
    load_feature_cache,
    write_feature_cache,
)
from scripts.cache_vjepa_single_frame import single_frame_clip
from scripts.mop_at3_time_axis import run as at3_run
from scripts.mop_at3_time_axis import temporal_labels
from scripts.mop_pr2_plasticity_substrates import _parse_seeds, _tasks_from_classes
from scripts.mop_pr2_plasticity_substrates import run as pr2_run


def _tiny(monkeypatch):
    monkeypatch.setattr(cun, "RES", 32)
    monkeypatch.setattr(cun, "FRAMES", 8)


def test_extract_nuisance_draws_consumes_nothing_and_matches_source_transforms():
    g = torch.Generator().manual_seed(3)
    state = g.get_state()
    draws = extract_nuisance_draws(g)
    assert torch.equal(g.get_state(), state), "extraction must clone, never consume the stream"
    u = [float(torch.rand(1, generator=g)) for _ in range(6)]
    assert draws["r"] == pytest.approx(0.15 + 0.2 * u[0])
    assert draws["rot"] == pytest.approx(u[1] * 2 * math.pi)
    assert draws["x0"] == pytest.approx(-0.5 + u[2])
    assert draws["y0"] == pytest.approx(-0.5 + u[3])
    assert draws["vx"] == pytest.approx(0.4 * (u[4] - 0.5))
    assert draws["vy"] == pytest.approx(0.4 * (u[5] - 0.5))
    assert list(draws) == list(NUISANCE_COLS)


def test_iter_nuisance_clips_is_deterministic_cell_major(monkeypatch):
    _tiny(monkeypatch)
    runs = []
    for _ in range(2):
        out = list(iter_nuisance_clips(2, 2, 1, seed=7))
        runs.append(out)
    assert [(i, s, c) for i, s, c, _, _ in runs[0]] == [(0, 0, 0), (1, 0, 1), (2, 1, 0), (3, 1, 1)]
    for (_, _, _, d0, clip0), (_, _, _, d1, clip1) in zip(runs[0], runs[1], strict=True):
        assert d0 == d1
        assert torch.equal(clip0, clip1), "same seed must regenerate byte-identical clips"
    assert runs[0][0][4].shape == (8, 3, 32, 32)
    other = next(iter_nuisance_clips(2, 2, 1, seed=8))
    assert not torch.equal(other[4], runs[0][0][4])


def test_canonical_population_is_the_inflight_grid():
    assert CANONICAL == {"n_shape": 5, "n_color": 5, "per": 8, "seed": 0}


def test_feature_cache_roundtrip(tmp_path):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(6, 4, generator=g)
    nuis = torch.randn(6, 6, generator=g)
    write_feature_cache(
        tmp_path / "c",
        features=x,
        labels_shape=[0, 1, 2, 0, 1, 2],
        labels_color=[0, 0, 1, 1, 0, 0],
        nuisance=nuis,
        meta={"tag": "t", "pretrained": False},
    )
    got = load_feature_cache(tmp_path / "c")
    assert torch.allclose(got["features"], x)
    assert got["features"].dtype == torch.float32
    assert got["labels_shape"].tolist() == [0, 1, 2, 0, 1, 2]
    assert got["labels_shape"].dtype == torch.int64
    assert got["labels_color"].tolist() == [0, 0, 1, 1, 0, 0]
    assert torch.allclose(got["nuisance"], nuis)
    assert got["meta"]["tag"] == "t" and got["meta"]["pretrained"] is False


def test_feature_cache_reads_latentstore_and_pt_layouts(tmp_path):
    import numpy as np

    x = torch.randn(5, 3)
    d1 = tmp_path / "npy"
    d1.mkdir()
    np.save(d1 / "latents.npy", x.numpy())
    np.save(d1 / "labels.npy", np.array([0, 1, 0, 1, 0]))
    got = load_feature_cache(d1)
    assert torch.allclose(got["features"], x)
    assert got["labels_shape"].tolist() == [0, 1, 0, 1, 0]

    d2 = tmp_path / "pt"
    d2.mkdir()
    torch.save({"features": x, "labels": torch.tensor([1, 1, 0, 0, 1])}, d2 / "features.pt")
    got2 = load_feature_cache(d2)
    assert torch.allclose(got2["features"], x)
    assert got2["labels_shape"].tolist() == [1, 1, 0, 0, 1]

    with pytest.raises(FileNotFoundError):
        load_feature_cache(tmp_path / "missing")


def test_encoder_lane_guard_passes_when_lane_free():
    assert_encoder_lane_free(patterns=("definitely_not_a_running_process_xyzzy",))


def test_encoder_lane_guard_refuses_when_busy(monkeypatch):
    def fake_run(cmd, capture_output=True, text=True):
        return types.SimpleNamespace(stdout="424242\n", returncode=0)

    monkeypatch.setattr(crf.subprocess, "run", fake_run)
    with pytest.raises(SystemExit, match="encoder lane BUSY"):
        assert_encoder_lane_free(patterns=("compositional_under_nuisance",))


def test_encoder_lane_guard_ignores_own_and_parent_pid(monkeypatch):
    import os

    def fake_run(cmd, capture_output=True, text=True):
        return types.SimpleNamespace(stdout=f"{os.getpid()}\n{os.getppid()}\n", returncode=0)

    monkeypatch.setattr(crf.subprocess, "run", fake_run)
    assert_encoder_lane_free(patterns=("compositional_under_nuisance",))


def test_single_frame_clip_is_token_matched_and_time_constant():
    g = torch.Generator().manual_seed(1)
    clip = torch.rand(8, 3, 16, 16, generator=g)
    sf = single_frame_clip(clip)
    assert sf.shape == clip.shape, "token/frame count must be matched"
    assert torch.equal(sf[0], clip[4]), "middle frame is the static content"
    assert float((sf - sf[0]).abs().max()) == 0.0, "zero temporal variation"


def _write_cache(root, x, y, yc=None, nuis=None):
    write_feature_cache(root, features=x, labels_shape=y, labels_color=yc, nuisance=nuis, meta={})
    return str(root)


def _class_features(y, n_classes, d, scale, noise, seed):
    g = torch.Generator().manual_seed(seed)
    proto = torch.randn(n_classes, d, generator=g)
    return scale * proto[y] + noise * torch.randn(len(y), d, generator=g)


def test_pr2_tasks_and_seed_parsing():
    assert _tasks_from_classes(4) == [[0, 1], [2, 3]]
    assert _tasks_from_classes(5) == [[0, 1], [2, 3], [4]]
    assert _parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert _parse_seeds("0,2,7") == [0, 2, 7]


def test_pr2_end_to_end_tiny(tmp_path):
    n, d, n_classes = 64, 10, 4
    y = torch.arange(n) % n_classes
    real = _class_features(y, n_classes, d, scale=3.0, noise=0.2, seed=0)
    rand = torch.randn(n, d, generator=torch.Generator().manual_seed(9))
    cfg = {
        "real_cache": _write_cache(tmp_path / "real", real, y.tolist()),
        "randinit_cache": _write_cache(tmp_path / "rand", rand, y.tolist()),
        "seeds": [0, 1, 2],
        "steps_per_task": 25,
        "hidden": 8,
    }
    out = pr2_run(cfg, "cpu", tmp_path / "run")
    assert isinstance(out["null_supported"], bool)
    assert len(out["per_seed"]) == 3
    for r in out["per_seed"]:
        assert set(r) >= {"seed", "real", "randinit", "bwt_delta", "speed_delta_steps", "fa_delta"}
    for key in ("bwt_delta_ci", "speed_delta_ci", "fa_delta_ci_secondary"):
        assert set(out[key]) >= {"n", "mean", "lo", "hi", "unstable"}
    assert out["null_supported"] == (not (out["bwt_win"] or out["speed_win"]))
    assert "null_hypothesis" in out["contract"] and out["contract"]["tier"] == "cpu-now"
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["null_supported"] == out["null_supported"]


def test_pr2_label_mismatch_violates_clip_identity(tmp_path):
    y = [0, 1, 0, 1]
    x = torch.randn(4, 3)
    cfg = {
        "real_cache": _write_cache(tmp_path / "a", x, y),
        "randinit_cache": _write_cache(tmp_path / "b", x, [1, 0, 1, 0]),
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="clip identity"):
        pr2_run(cfg, "cpu", None)


def test_temporal_labels_quadrants_and_speed_split():
    nuis = torch.zeros(4, 6)
    vx_i, vy_i = NUISANCE_COLS.index("vx"), NUISANCE_COLS.index("vy")
    nuis[:, vx_i] = torch.tensor([1.0, -1.0, -1.0, 1.0])
    nuis[:, vy_i] = torch.tensor([1.0, 1.0, -1.0, -1.0])
    lab = temporal_labels(nuis)
    assert sorted(lab["motion_dir4"].tolist()) == [0, 1, 2, 3], "four quadrants, four labels"
    nuis[:, vx_i] = torch.tensor([0.1, 0.1, 2.0, 2.0])
    nuis[:, vy_i] = 0.0
    assert temporal_labels(nuis)["speed2"].tolist() == [0, 0, 1, 1]


def _at3_caches(tmp_path, temporal_in_full):
    g = torch.Generator().manual_seed(0)
    n, n_shape = 96, 4
    y = torch.arange(n) % n_shape
    yc = (torch.arange(n) // n_shape) % 2
    nuis = torch.zeros(n, 6)
    dirs = torch.randint(0, 4, (n,), generator=g)  # decorrelated from shape/color by construction
    ang = -math.pi + (dirs.float() + 0.5) * (math.pi / 2)
    speed = torch.where((torch.arange(n) // 2) % 2 == 0, 0.1, 0.3)  # two speed classes, unencoded
    nuis[:, NUISANCE_COLS.index("vx")] = speed * torch.cos(ang)
    nuis[:, NUISANCE_COLS.index("vy")] = speed * torch.sin(ang)
    shape_part = 3.0 * torch.nn.functional.one_hot(y, n_shape).float()
    color_part = 3.0 * torch.nn.functional.one_hot(yc, 2).float()
    dir_part = 3.0 * torch.nn.functional.one_hot(dirs, 4).float()
    noise = 0.1 * torch.randn(n, n_shape + 2 + 4, generator=g)
    single = torch.cat([shape_part, color_part, torch.zeros(n, 4)], dim=1) + noise
    full = single + torch.cat([torch.zeros(n, n_shape + 2), dir_part], dim=1) if temporal_in_full else single
    return {
        "full_cache": _write_cache(tmp_path / "full", full, y.tolist(), yc.tolist(), nuis),
        "single_cache": _write_cache(tmp_path / "single", single, y.tolist(), yc.tolist(), nuis),
        "seeds": [0, 1, 2],
        "probe_epochs": 60,
    }


def test_at3_detects_a_temporal_currency(tmp_path):
    out = at3_run(_at3_caches(tmp_path, temporal_in_full=True), "cpu", tmp_path / "run")
    assert set(out["factors"]) == {"shape", "color", "motion_dir4", "speed2"}
    assert "motion_dir4" in out["needs_time_factors"]
    assert out["null_supported"] is False
    md = out["factors"]["motion_dir4"]
    assert md["single_acc_mean"] <= md["chance"] + 0.1 and md["delta_win"] is True
    assert (tmp_path / "run" / "result.json").exists()


def test_at3_identical_arms_support_the_null(tmp_path):
    out = at3_run(_at3_caches(tmp_path, temporal_in_full=False), "cpu", None)
    assert out["null_supported"] is True
    assert out["needs_time_factors"] == []
    assert "NULL SUPPORTED" in out["verdict"]


def test_at3_missing_nuisance_raises(tmp_path):
    y = [0, 1, 0, 1]
    x = torch.randn(4, 3)
    cfg = {
        "full_cache": _write_cache(tmp_path / "f", x, y),
        "single_cache": _write_cache(tmp_path / "s", x, y),
        "seeds": [0],
    }
    with pytest.raises(ValueError, match="nuisance"):
        at3_run(cfg, "cpu", None)
