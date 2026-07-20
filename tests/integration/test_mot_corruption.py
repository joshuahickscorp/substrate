
import json
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import mop_dr14_corruption as dr14  # noqa: E402

from mop.devices import resolve  # noqa: E402
from mop.experiments.e6_dense_relational import build_mechanics_fixture  # noqa: E402
from mop.substrate.cache_manifest import write_cache_manifest  # noqa: E402
from mop.substrate.cache_tools import validate_cache  # noqa: E402

DEV = resolve("cpu")


def _tiny_cfg():
    over = {
        "seeds": [0],
        "dim": 12,
        "n_classes": 3,
        "n_train": 120,
        "n_test": 90,
        "sep_grid": [0.5, 1.0, 2.0],
        "refiner_steps": 2,
        "hidden": 12,
        "epochs": 40,
        "rank_fracs": [1.0, 0.25],
        "bits": [32, 3],
        "noise_levels": [0.0, 1.0],
        "use_real_cache": False,
        "use_dense_cache": False,
    }
    return OmegaConf.merge(dr14.default_cfg(), OmegaConf.create(over))


def test_corruption_operators_identity_at_zero_severity():
    x = torch.randn(20, 6)
    assert torch.allclose(dr14.corrupt_quantize(x, 32), x)
    assert torch.allclose(dr14.corrupt_noise(x, 0.0, seed=0), x)
    train = torch.randn(40, 6)
    assert torch.allclose(dr14.corrupt_lowrank(x, train, 1.0), x)
    lr = dr14.corrupt_lowrank(x, train, 0.99)  # forces the SVD path at full-ish rank
    assert torch.allclose(lr, x, atol=1e-4)


def test_corruption_operators_actually_corrupt():
    x = torch.randn(30, 8)
    train = torch.randn(60, 8)
    assert not torch.allclose(dr14.corrupt_lowrank(x, train, 0.25), x)
    assert not torch.allclose(dr14.corrupt_quantize(x, 2), x)
    assert not torch.allclose(dr14.corrupt_noise(x, 1.0, seed=0), x)


def test_corrupt_noise_deterministic_given_seed():
    x = torch.randn(10, 4)
    a = dr14.corrupt_noise(x, 0.5, seed=3)
    b = dr14.corrupt_noise(x, 0.5, seed=3)
    assert torch.allclose(a, b)  # both arms would see the identical corrupted tensor


def test_pure_noise_destroys_signal_but_matches_scale():
    x = torch.randn(200, 5) * 3.0 + 1.0
    n = dr14.pure_noise_like(x, seed=0)
    assert abs(float(n.std()) - float(x.std())) / float(x.std()) < 0.2
    assert n.shape == x.shape


def test_fit_slope_recovers_linear_trend():
    assert abs(dr14.fit_slope([0.0, 0.5, 1.0], [0.0, 0.1, 0.2]) - 0.2) < 1e-9
    assert dr14.fit_slope([0.0, 1.0], [0.3, 0.3]) == 0.0


def test_contract_declared():
    c = dr14.DR14Corruption().contract()
    assert c["null_hypothesis"] and c["baseline"] and c["ablation"]
    assert c["tier"] == "env-later"


def test_dr14_runs_end_to_end(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert out["survives"] == (not out["null_supported"])
    assert (tmp_path / "dr14_corruption.json").exists()
    assert out["preregistered"]["slope_margin"] == dr14.SLOPE_MARGIN
    r = out["per_seed"][0]["synthetic"]
    assert set(r["families"].keys()) == {"lowrank_vq", "quantize", "noise"}
    for fam in r["families"].values():
        assert len(fam["acc_reasoning"]) == len(fam["acc_single_pass"]) == len(fam["severity"])
        assert isinstance(fam["reasoning_flatter"], bool)
    assert set(r["noisy_tv_guard"]["pure_noise_acc"].keys()) == {"reasoning", "single_pass"}
    assert set(r["shuffled_feature_floor"].keys()) == {"reasoning", "single_pass"}
    assert "skipped" in out["per_seed"][0]["real_cache_pilot"]


def test_dr14_arms_are_flop_matched(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    comp = out["per_seed"][0]["synthetic"]["compute"]
    assert comp["refiner_blocks"] == comp["single_pass_blocks"]
    assert comp["matched"]["matched"] is True
    assert comp["params"]["single_pass"] > comp["params"]["reasoning"]


def test_dr14_guard_accuracy_near_chance_on_pure_noise(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    g = out["per_seed"][0]["synthetic"]["noisy_tv_guard"]["pure_noise_acc"]
    chance = out["per_seed"][0]["synthetic"]["chance"]
    assert abs(g["reasoning"] - chance) < 0.25
    assert abs(g["single_pass"] - chance) < 0.25


def test_parse_seeds_forms():
    assert dr14.parse_seeds("0-2") == [0, 1, 2]
    assert dr14.parse_seeds("0,2") == [0, 2]
    assert dr14.parse_seeds("1") == [1]


def test_dense_drop_runner_uses_one_nested_shared_view_for_both_arms(tmp_path):
    fixture = build_mechanics_fixture(tmp_path / "fixture")
    cache = Path(fixture["stores"]["learned"])
    factors = json.loads((cache / "factors.json").read_text())
    columns = factors.get("columns", factors)
    labels = np.load(cache / "labels.npy", mmap_mode="r+")
    labels[:] = np.asarray(columns["factor_a"], dtype="int64")
    labels.flush()
    manifest = json.loads((cache / "cache_manifest.json").read_text())
    form = manifest["form"]
    write_cache_manifest(
        cache,
        encoder_config=manifest["encoder_config"],
        form_kind=form["kind"],
        form_objective=form["objective"],
        referent_scheme=form["referent_scheme"],
        full_hash_arrays=True,
    )
    assert validate_cache(cache, citable=True) == []

    cfg = _tiny_cfg()
    cfg.allow_dense_fixture = True
    cfg.dense_drop_fractions = [0.0, 0.25, 0.5, 0.75]
    cfg.dense_channel_group_width = 2
    cfg.epochs = 15
    result = dr14.run_dense_drop_sweep(cfg, cache, seed=4)
    assert "skipped" not in result
    assert result["scientific_promotion"] is False
    assert result["verdict_setting"] is False
    assert result["shared_view_receipt"]["shared_corrupted_tensor_for_both_arms"] is True
    masks = result["shared_view_receipt"]["masks"]
    assert set(masks["0.250000"]["dropped_groups"]) < set(masks["0.500000"]["dropped_groups"])
    assert len(result["acc_reasoning"]) == len(result["acc_single_pass"]) == 4
    assert result["compute"]["matched"]["matched"] is True
