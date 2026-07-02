"""WP-13 DR14 corruption tests: tiny synthetic tensors only, seconds per test, no encoder loads,
no network, no real-cache reads (use_real_cache=False in the tiny cfg). Asserts the honesty
machinery structurally: identical corruption for both arms, FLOP matching, preregistered nulls,
the noisy-TV guard and the destroyed-structure floor, and the corruption operators' identities."""

import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import mop_dr14_corruption as dr14  # noqa: E402

from mop.devices import resolve  # noqa: E402

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
    }
    return OmegaConf.merge(dr14.default_cfg(), OmegaConf.create(over))


# ---------------------------------------------------------------- corruption operators


def test_corruption_operators_identity_at_zero_severity():
    x = torch.randn(20, 6)
    assert torch.allclose(dr14.corrupt_quantize(x, 32), x)
    assert torch.allclose(dr14.corrupt_noise(x, 0.0, seed=0), x)
    # full-rank projection reconstructs (up to numerical noise)
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


# ---------------------------------------------------------------- contract + end to end


def test_contract_declared():
    c = dr14.DR14Corruption().contract()
    assert c["null_hypothesis"] and c["baseline"] and c["ablation"]
    assert c["tier"] == "cpu-now"


def test_dr14_runs_end_to_end(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert out["survives"] == (not out["null_supported"])
    assert (tmp_path / "dr14_corruption.json").exists()
    assert out["preregistered"]["slope_margin"] == dr14.SLOPE_MARGIN
    r = out["per_seed"][0]["synthetic"]
    # all three laptop families present with both arms on the SAME severity axis
    assert set(r["families"].keys()) == {"lowrank_vq", "quantize", "noise"}
    for fam in r["families"].values():
        assert len(fam["acc_reasoning"]) == len(fam["acc_single_pass"]) == len(fam["severity"])
        assert isinstance(fam["reasoning_flatter"], bool)
    # noisy-TV guard and destroyed-structure floor are reported for BOTH arms
    assert set(r["noisy_tv_guard"]["pure_noise_acc"].keys()) == {"reasoning", "single_pass"}
    assert set(r["shuffled_feature_floor"].keys()) == {"reasoning", "single_pass"}
    # real cache disabled in the tiny cfg: reported as skipped, never crashes
    assert "skipped" in out["per_seed"][0]["real_cache_pilot"]


def test_dr14_arms_are_flop_matched(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    comp = out["per_seed"][0]["synthetic"]["compute"]
    assert comp["refiner_blocks"] == comp["single_pass_blocks"]
    assert comp["matched"]["matched"] is True
    # untied control honestly carries more params, and says so
    assert comp["params"]["single_pass"] > comp["params"]["reasoning"]


def test_dr14_guard_accuracy_near_chance_on_pure_noise(tmp_path):
    out = dr14.DR14Corruption().run(_tiny_cfg(), DEV, tmp_path)
    g = out["per_seed"][0]["synthetic"]["noisy_tv_guard"]["pure_noise_acc"]
    chance = out["per_seed"][0]["synthetic"]["chance"]
    # loose structural bound at toy scale (the preregistered GUARD_TOL governs full scale)
    assert abs(g["reasoning"] - chance) < 0.25
    assert abs(g["single_pass"] - chance) < 0.25


def test_parse_seeds_forms():
    assert dr14.parse_seeds("0-2") == [0, 1, 2]
    assert dr14.parse_seeds("0,2") == [0, 2]
    assert dr14.parse_seeds("1") == [1]
