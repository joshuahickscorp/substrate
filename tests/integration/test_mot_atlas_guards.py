"""WP-09 atlas guard tests (AT4 programmatic ceiling, AT5 probe-class sweep): tiny synthetic
LatentStores in tmp_path, seconds per test, no encoder loads, no network, no real-cache reads.
Asserts MECHANICS and preregistered verdict logic (ceiling certification, headroom vs test-too-easy,
probe-specific detection, missing stores skipped honestly), never a particular scientific outcome."""

import json

import torch
from scripts.mop_at4_programmatic_ceiling import (
    CEILING_COLUMN,
    PERCEPTUAL_STORES,
    clip_identity_check,
    factor_reading,
    load_column,
    parse_seeds,
)
from scripts.mop_at4_programmatic_ceiling import (
    CONTRACT as AT4_CONTRACT,
)
from scripts.mop_at4_programmatic_ceiling import run as at4_run
from scripts.mop_at5_probe_class_sweep import (
    CONTRACT as AT5_CONTRACT,
)
from scripts.mop_at5_probe_class_sweep import (
    PROBE_CLASSES,
    cell_verdict,
    nonlinear_gain_probe,
)
from scripts.mop_at5_probe_class_sweep import run as at5_run

from mop.substrate import LatentStore

SEEDS = [0, 1]
N = 360
N_CLASSES = 3


def _labels(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return (
        torch.randint(0, N_CLASSES, (N,), generator=g),
        torch.randint(0, N_CLASSES, (N,), generator=g),
    )


def _onehot_features(shape: torch.Tensor, color: torch.Tensor) -> torch.Tensor:
    """The programmatic-style ceiling column: both factors one-hot, decodable at 1.0 by construction."""
    x = torch.zeros(N, 2 * N_CLASSES)
    x[torch.arange(N), shape] = 1.0
    x[torch.arange(N), N_CLASSES + color] = 1.0
    return x


def _blob_features(shape: torch.Tensor, color: torch.Tensor, seed: int = 5) -> torch.Tensor:
    """Linearly separable in both factors (a strong perceptual stand-in)."""
    g = torch.Generator().manual_seed(seed)
    cs = torch.randn(N_CLASSES, 12, generator=g)
    cc = torch.randn(N_CLASSES, 12, generator=g)
    return torch.cat([cs[shape], cc[color]], dim=1) * 3.0 + 0.2 * torch.randn(N, 24, generator=g)


def _noise_features(seed: int = 9, dim: int = 12) -> torch.Tensor:
    return torch.randn(N, dim, generator=torch.Generator().manual_seed(seed))


def _antipodal_features(shape: torch.Tensor, seed: int = 11) -> torch.Tensor:
    """Each shape class is a +mu/-mu pair: at chance for ANY linear readout (class-conditional
    symmetry about 0), solvable after an elementwise nonlinearity or by the MLP. Signs alternate
    per class occurrence so every class is EXACTLY balanced (a sampled imbalance would hand the
    linear probe a spurious direction)."""
    g = torch.Generator().manual_seed(seed)
    mu = torch.randn(N_CLASSES, 12, generator=g)
    seen = [0] * N_CLASSES
    signs = torch.empty(N, 1)
    for i, c in enumerate(shape.tolist()):
        signs[i, 0] = 1.0 if seen[c] % 2 == 0 else -1.0
        seen[c] += 1
    return mu[shape] * signs * 2.0 + 0.2 * torch.randn(N, 12, generator=g)


def _write_store(cache_root, name, x, shape, color, checksums=("csA", "csB"), sidecar=True):
    store = LatentStore.create(
        cache_root, name, feat_shape=(x.shape[1],), capacity=N, key_dim=x.shape[1], has_labels=True
    )
    store.write_batch(0, x.numpy(), x.numpy(), shape.numpy())
    store.finalize()
    if sidecar:
        (store.root / "factors.json").write_text(
            json.dumps(
                {
                    "clipset": "bound_nuisance_v1",
                    "checksum_first": checksums[0],
                    "checksum_last": checksums[1],
                    "shape": shape.tolist(),
                    "color": color.tolist(),
                }
            )
        )
    return store.root


# ---------------------------------------------------------------- shared loaders and guards


def test_parse_seeds_forms():
    assert parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert parse_seeds("0,2,5") == [0, 2, 5]


def test_load_column_honest_about_missing_or_mismatched(tmp_path):
    shape, color = _labels(0)
    assert load_column(tmp_path / "absent") is None
    root = _write_store(tmp_path, "no_sidecar", _noise_features(), shape, color, sidecar=False)
    assert load_column(root) is None, "a store without a factors sidecar is not an atlas column"
    root2 = _write_store(tmp_path, "short_factors", _noise_features(), shape, color)
    sc = json.loads((root2 / "factors.json").read_text())
    sc["shape"], sc["color"] = sc["shape"][:-5], sc["color"][:-5]
    (root2 / "factors.json").write_text(json.dumps(sc))
    assert load_column(root2) is None, "factor length mismatch must not be silently truncated"
    root3 = _write_store(tmp_path, "good", _noise_features(), shape, color)
    col = load_column(root3)
    assert col is not None and set(col["factors"]) == {"shape", "color"}
    assert col["x"].shape == (N, 12)
    assert torch.equal(col["factors"]["shape"], shape)


def test_clip_identity_check_flags_mismatched_clip_sets():
    same = {
        "a": {"checksum_first": "x", "checksum_last": "y"},
        "b": {"checksum_first": "x", "checksum_last": "y"},
    }
    assert clip_identity_check(same)["identical"]
    diff = {
        "a": {"checksum_first": "x", "checksum_last": "y"},
        "b": {"checksum_first": "z", "checksum_last": "y"},
    }
    out = clip_identity_check(diff)
    assert not out["identical"]
    unverified = {"a": {"checksum_first": "x", "checksum_last": "y"}, "b": {}}
    assert clip_identity_check(unverified)["unverified_columns"] == ["b"]


# ---------------------------------------------------------------- AT4 verdict logic


def test_factor_reading_degenerate_when_ceiling_broken():
    r = factor_reading([0.8, 0.82], {"p": [0.5, 0.5]}, chance=1 / 3)
    assert r["reading"] == "DEGENERATE"


def test_factor_reading_headroom_and_tie():
    prog = [1.0, 1.0, 1.0]
    weak = {"p": [0.4, 0.42, 0.41]}
    r = factor_reading(prog, weak, chance=1 / 3)
    assert r["reading"] == "HEADROOM" and r["best_perceptual"] == "p"
    assert r["delta_ci"]["lo"] > 0
    tied = {"p": [1.0, 1.0, 1.0]}
    assert factor_reading(prog, tied, chance=1 / 3)["reading"] == "TEST-TOO-EASY"
    assert factor_reading(prog, {}, chance=1 / 3)["reading"] == "NO-PERCEPTUAL-COLUMNS"


def test_at4_end_to_end_test_too_easy_supports_null(tmp_path):
    shape, color = _labels(0)
    _write_store(tmp_path, CEILING_COLUMN, _onehot_features(shape, color), shape, color)
    _write_store(tmp_path, PERCEPTUAL_STORES[0], _blob_features(shape, color), shape, color)
    r = at4_run(tmp_path, SEEDS, probe_epochs=80)
    assert r["contract"] == AT4_CONTRACT
    for field in ("metric", "baseline", "ablation", "null_hypothesis", "tier"):
        assert AT4_CONTRACT[field]
    assert r["verdict"].startswith("NULL") and r["null_supported"] is True
    for factor in ("shape", "color"):
        assert r["per_factor_reading"][factor]["reading"] == "TEST-TOO-EASY"
    assert r["clip_identity"]["identical"]
    # the shuffled-label floor must sit near chance, far below the real probe
    floor = r["shuffled_label_floor"][CEILING_COLUMN]["shape"]
    assert floor < 1 / N_CLASSES + 0.2
    assert len(r["skipped_columns"]) > 0, "absent perceptual stores must be reported, not invented"


def test_at4_end_to_end_headroom_arms_the_trigger(tmp_path):
    shape, color = _labels(1)
    _write_store(tmp_path, CEILING_COLUMN, _onehot_features(shape, color), shape, color)
    _write_store(tmp_path, PERCEPTUAL_STORES[0], _noise_features(), shape, color)
    r = at4_run(tmp_path, SEEDS, probe_epochs=80)
    assert r["verdict"].startswith("HEADROOM") and r["null_supported"] is False
    assert r["per_factor_reading"]["shape"]["reading"] == "HEADROOM"


def test_at4_no_ceiling_store_is_not_a_null(tmp_path):
    shape, color = _labels(2)
    _write_store(tmp_path, PERCEPTUAL_STORES[0], _blob_features(shape, color), shape, color)
    r = at4_run(tmp_path, SEEDS, probe_epochs=40)
    # not evaluable: nothing was tested, so neither null branch may fire (None, not False)
    assert r["verdict"].startswith("NO CEILING") and r["null_supported"] is None


# ---------------------------------------------------------------- AT5 probe classes


def test_nonlinear_gain_probe_contract_and_capacity():
    shape, color = _labels(3)
    x = _blob_features(shape, color)
    out = nonlinear_gain_probe(x, shape, epochs=120, seed=0)
    assert set(out) == {"metric", "score", "chance", "decodable"}
    assert out["decodable"], "linearly separable blobs must stay decodable under the gain probe"
    noise = nonlinear_gain_probe(_noise_features(), shape, epochs=120, seed=0)
    assert not noise["decodable"], "pure noise must not become decodable under the gain probe"


def test_cell_verdict_rule():
    dec = {c: [0.9, 0.92] for c in PROBE_CLASSES}
    v = cell_verdict(dec, chance=1 / 3)
    assert v["verdict"] == "probe-invariant-decodable" and not v["probe_specific"]
    und = {c: [0.34, 0.33] for c in PROBE_CLASSES}
    v = cell_verdict(und, chance=1 / 3)
    assert v["verdict"] == "probe-invariant-undecodable" and not v["probe_specific"]
    mixed = {"linear": [0.34, 0.35], "nonlinear_gain": [0.35, 0.33], "mlp": [0.9, 0.88]}
    v = cell_verdict(mixed, chance=1 / 3)
    assert v["probe_specific"] and v["verdict"].startswith("probe-specific")
    assert "mlp" in v["verdict"]


def test_at5_probe_invariant_cells_support_null(tmp_path):
    shape, color = _labels(4)
    _write_store(tmp_path, "easy_col", _blob_features(shape, color), shape, color)
    _write_store(tmp_path, "noise_col", _noise_features(), shape, color)
    r = at5_run(
        tmp_path,
        SEEDS,
        columns=["easy_col", "noise_col", "absent_col"],
        probe_epochs={"linear": 80, "nonlinear_gain": 80, "mlp": 80},
    )
    assert r["contract"] == AT5_CONTRACT
    assert r["null_supported"] is True and r["verdict"].startswith("NULL")
    assert r["skipped_columns"] == ["absent_col"]
    assert len(r["cells"]) == 4  # 2 columns x 2 factors
    for cell in r["cells"]:
        assert set(cell["per_seed"]) == set(PROBE_CLASSES)
        assert all(len(v) == len(SEEDS) for v in cell["per_seed"].values())


def test_at5_catches_a_probe_specific_cell(tmp_path):
    shape, color = _labels(5)
    _write_store(tmp_path, "antipodal_col", _antipodal_features(shape), shape, color)
    r = at5_run(
        tmp_path,
        SEEDS,
        columns=["antipodal_col"],
        probe_epochs={"linear": 80, "nonlinear_gain": 150, "mlp": 150},
    )
    cell = next(c for c in r["cells"] if c["factor"] == "shape")
    assert not cell["decodable_under"]["linear"], "antipodal classes are at chance for any linear probe"
    assert cell["decodable_under"]["mlp"], "the capacity-capped MLP must solve the antipodal pairing"
    assert cell["probe_specific"]
    assert r["null_supported"] is False and "antipodal_col/shape" in r["probe_specific_cells"]


def test_at5_no_cells_is_not_a_null(tmp_path):
    r = at5_run(tmp_path, SEEDS, columns=["nothing_here"])
    # not evaluable: nothing was tested, so neither null branch may fire (None, not False)
    assert r["null_supported"] is None and r["verdict"].startswith("NO CELLS")
