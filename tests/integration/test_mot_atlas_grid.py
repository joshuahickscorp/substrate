"""WP-14 atlas grid pilot tests (AT1/AL2) plus the WP-01 featurizer pure helpers they consume. Tiny
synthetic tensors only, no network, no weights, no encoder loads; the nuisance generator is exercised at
a PATCHED tiny resolution (RES=32, FRAMES=8) so clip-identity determinism is tested in milliseconds.
Asserts MECHANICS and preregistered verdict logic, never a particular scientific outcome."""

import json

import scripts.compositional_under_nuisance as cun
import torch
from scripts.cache_qwen_textified import textify_clip
from scripts.cache_wav2vec2_sonified import SEG_SECONDS, SR, sonify_clip
from scripts.featurize_handcrafted import DESCRIPTOR_DIM, HUE_BINS, handcrafted_descriptor, hue_histogram
from scripts.featurize_programmatic import (
    N_SCALAR_DRAWS,
    clip_checksum,
    iter_bound_nuisance_clips,
    scene_program_vector,
    to_uint8,
    write_factors_sidecar,
)
from scripts.mot_al2_alignment_pilot import evaluate_pairs, learned_vs_floors, pair_report
from scripts.mot_at1_grid_pilot import (
    classify_column,
    clip_identity_check,
    evaluate_grid,
    grid_scope,
    parse_seeds,
)

from devsys.substrate import LatentStore

# ---------------------------------------------------------------- WP-01 clip identity helpers


def _tiny(monkeypatch):
    monkeypatch.setattr(cun, "RES", 32)
    monkeypatch.setattr(cun, "FRAMES", 8)


def test_clip_regeneration_is_deterministic_and_records_draws(monkeypatch):
    _tiny(monkeypatch)
    runs = []
    for _ in range(2):
        out = list(iter_bound_nuisance_clips(2, 2, 1, seed=7, record=True))
        runs.append([(i, s, c, clip_checksum(clip), tuple(d)) for i, s, c, clip, d in out])
    assert runs[0] == runs[1], "same seed must regenerate byte-identical clips (clip identity rule)"
    assert len(runs[0]) == 4
    for _, _, _, _, draws in runs[0]:
        assert len(draws) == N_SCALAR_DRAWS
        assert all(0.0 <= v <= 1.0 for v in draws)
    assert torch.rand is not None and not hasattr(torch.rand, "__wrapped__")  # recorder restored


def test_different_seeds_give_different_clips(monkeypatch):
    _tiny(monkeypatch)
    a = next(iter_bound_nuisance_clips(1, 1, 1, seed=0))[3]
    b = next(iter_bound_nuisance_clips(1, 1, 1, seed=1))[3]
    assert clip_checksum(a) != clip_checksum(b)


def test_scene_program_vector_layout():
    v = scene_program_vector(2, 1, 5, 5, [0.5] * N_SCALAR_DRAWS)
    assert v.shape == (5 + 5 + N_SCALAR_DRAWS,)
    assert v[2] == 1.0 and v[:5].sum() == 1.0
    assert v[5 + 1] == 1.0 and v[5:10].sum() == 1.0
    assert torch.all(v[10:] == 0.5)


# ---------------------------------------------------------------- WP-01 featurizers (pure, tiny)


def _tiny_clip(seed=0, t=6, res=32):
    g = torch.Generator().manual_seed(seed)
    return to_uint8(torch.rand(t, 3, res, res, generator=g))


def test_handcrafted_descriptor_shape_and_determinism():
    frames = _tiny_clip()
    z1, z2 = handcrafted_descriptor(frames), handcrafted_descriptor(frames)
    assert z1.shape == (DESCRIPTOR_DIM,)
    assert torch.equal(z1, z2)
    assert torch.isfinite(z1).all()
    assert clip_checksum(frames.float() / 255) != clip_checksum(_tiny_clip(seed=1).float() / 255)


def test_hue_histogram_normalized():
    h = hue_histogram(_tiny_clip())
    assert h.shape == (HUE_BINS,)
    assert 0.9 < float(h.sum()) < 1.1


def test_textify_is_deterministic_and_label_free():
    frames = _tiny_clip()
    t1, t2 = textify_clip(frames, tsub=3, grid=2), textify_clip(frames, tsub=3, grid=2)
    assert t1 == t2
    assert t1 != textify_clip(_tiny_clip(seed=3), tsub=3, grid=2)
    assert t1.count("frame") == 3  # only pixels in, prose out: the signature takes no labels at all


def test_sonify_is_deterministic_normalized_and_content_sensitive():
    frames = _tiny_clip()
    w1, w2 = sonify_clip(frames, tsub=4), sonify_clip(frames, tsub=4)
    assert torch.equal(w1, w2)
    assert w1.shape == (int(4 * SEG_SECONDS * SR),)
    assert abs(float(w1.mean())) < 1e-4 and abs(float(w1.std()) - 1.0) < 1e-3
    assert not torch.equal(w1, sonify_clip(_tiny_clip(seed=5), tsub=4))


# ---------------------------------------------------------------- AT1 verdict logic (preregistered)


def test_classify_column_decision_order():
    strong, none = [0.30, 0.28, 0.32, 0.29, 0.31], [0.01, -0.02, 0.02, -0.01, 0.0]
    assert classify_column(none, none)["verdict"] == "random-control-artifact"
    assert classify_column(strong, strong)["verdict"] == "genuine-substrate-signal"
    assert classify_column(strong, none)["verdict"] == "probe-specific"
    flippy = [0.40, 0.45, 0.42, 0.44, -0.05]
    assert classify_column(flippy, flippy)["verdict"] == "non-replicating"


def test_grid_scope_typing():
    assert grid_scope([])["scope"] == "no-columns-available"
    art = {"tag": "a", "modality": "video", "verdict": "random-control-artifact"}
    assert "grid-wide" in grid_scope([art])["scope"]
    v = {"tag": "v", "modality": "video", "verdict": "genuine-substrate-signal"}
    i = {"tag": "i", "modality": "image", "verdict": "genuine-substrate-signal"}
    assert "modality-specific (video)" in grid_scope([v, art])["scope"]
    assert "pilot-universal" in grid_scope([v, i])["scope"]


def test_parse_seeds():
    assert parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert parse_seeds("0,2,7") == [0, 2, 7]


# ---------------------------------------------------------------- AT1/AL2 end to end on tiny stores


def _make_store(root, name, x, y, checks=("aaa", "zzz")):
    store = LatentStore.create(
        root, name, feat_shape=(x.shape[1],), capacity=x.shape[0], key_dim=x.shape[1], has_labels=True
    )
    store.write_batch(0, x.numpy(), x.numpy(), y.numpy())
    store.finalize()
    params = {"seed": 0, "n_shape": 3, "n_color": 1, "per": x.shape[0] // 3}
    write_factors_sidecar(
        store.root, params, y.tolist(), [0] * len(y), {"first": checks[0], "last": checks[1]}, source="test"
    )
    return store.root


def test_at1_grid_pilot_end_to_end(tmp_path):
    g = torch.Generator().manual_seed(0)
    n, d = 60, 12
    y = torch.arange(n) % 3
    signal = torch.nn.functional.one_hot(y, 3).float() @ torch.randn(3, d, generator=g)
    x_real = 3.0 * signal + 0.1 * torch.randn(n, d, generator=g)
    x_rand = torch.randn(n, d, generator=g)
    _make_store(tmp_path, "real_a", x_real, y)
    _make_store(tmp_path, "rand_a", x_rand, y)
    _make_store(tmp_path, "real_b", x_rand, y)  # a column where real IS noise: expected artifact
    _make_store(tmp_path, "rand_b", x_rand + 0.01 * torch.randn(n, d, generator=g), y)
    cols = [
        {
            "tag": "hot",
            "modality": "video",
            "family": "f",
            "real": "real_a",
            "randinit": "rand_a",
            "note": "",
        },
        {
            "tag": "cold",
            "modality": "image",
            "family": "f",
            "real": "real_b",
            "randinit": "rand_b",
            "note": "",
        },
        {
            "tag": "absent",
            "modality": "audio",
            "family": "f",
            "real": "nope",
            "randinit": "rand_a",
            "note": "",
        },
    ]
    rep = evaluate_grid(
        tmp_path, seeds=[0, 1, 2], columns=cols, reference_columns=["real_a"], probe_epochs=60, mlp_epochs=40
    )
    by_tag = {c["tag"]: c for c in rep["grid"]}
    assert by_tag["hot"]["verdict"] == "genuine-substrate-signal"
    assert by_tag["cold"]["verdict"] == "random-control-artifact"
    assert rep["skipped_columns"][0]["tag"] == "absent"
    assert rep["null_supported"] is False
    assert rep["clip_identity"]["identical"] is True
    assert "real_a" in rep["reference_columns"]
    assert isinstance(rep["null_supported"], bool) and "null_hypothesis" in rep["experiment"]


def test_clip_identity_check_flags_mismatch():
    rep = clip_identity_check(
        {
            "a": {"checksum_first": "x", "checksum_last": "y"},
            "b": {"checksum_first": "x", "checksum_last": "DIFFERENT"},
            "c": None,
        }
    )
    assert rep["identical"] is False and rep["unverified_columns"] == ["c"]


def test_al2_learned_map_beats_floors_only_with_shared_structure():
    g = torch.Generator().manual_seed(0)
    n, da, db = 100, 10, 12
    xa = torch.randn(n, da, generator=g)
    xb = xa @ torch.randn(da, db, generator=g) + 0.05 * torch.randn(n, db, generator=g)
    shared = pair_report(xa, xb, seeds=[0, 1, 2], ranks=[4, 8])
    assert shared["primary_rank"] == 8  # PRIMARY_RANK absent from tiny sweep, falls back to largest
    assert shared["verdict"] == "genuine-shared-structure"
    xc = torch.randn(n, db, generator=g)
    indep = pair_report(xa, xc, seeds=[0, 1, 2], ranks=[4, 8])
    assert indep["verdict"] != "genuine-shared-structure"
    one = learned_vs_floors(xa, xb, seed=0, rank=4)
    assert set(one) == {"learned_r2", "random_map_r2", "shuffled_fit_r2", "delta"}


def test_al2_pilot_end_to_end(tmp_path, capsys):
    g = torch.Generator().manual_seed(1)
    n = 90
    y = torch.arange(n) % 3
    xa = torch.randn(n, 8, generator=g)
    xb = xa @ torch.randn(8, 10, generator=g) + 0.05 * torch.randn(n, 10, generator=g)
    _make_store(tmp_path, "arm_a", xa, y)
    _make_store(tmp_path, "arm_b", xb, y)
    rep = evaluate_pairs(tmp_path, seeds=[0, 1, 2], arms=["arm_a", "arm_b", "arm_missing"], ranks=[4, 8])
    assert rep["arms_missing"] == ["arm_missing"]
    assert rep["pairs"]["arm_a -> arm_b"]["verdict"] == "genuine-shared-structure"
    assert rep["null_supported"] is False
    assert json.dumps(rep)  # the whole report must be json-serializable for runs/mot
