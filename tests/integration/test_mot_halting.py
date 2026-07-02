"""WP-03 halting and verifier harness (MP5, MP6, DR9, DR8). Tiny synthetic tensors only, seconds,
no encoder loads, no caches touched (DR8 runs its synthetic arm; the real-cache path is asserted to
fail loudly, not exercised). Asserts MECHANICS and contract honesty, never a scientific outcome (the
preregistered nulls may hold)."""

import sys
from pathlib import Path

import pytest

from mop.devices import resolve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import mop_dr8_fixed_point as dr8  # noqa: E402
import mop_dr9_verify_revise as dr9  # noqa: E402
import mop_mt5_adaptive_halting as mt5  # noqa: E402
import mop_mt6_confidence_stop as mt6  # noqa: E402

DEV = resolve("cpu")

TINY_MT = dict(dim=16, hidden=32, n_classes=4, samples=240, max_steps=4, epochs=30, lr=1e-2)


def test_parse_seeds_and_rerun_naming():
    assert mt5.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert mt5.parse_seeds("0,2,3") == [0, 2, 3]
    assert mt5.parse_seeds("7") == [7]
    out = mt5.resolve_out("runs/mot/mt5_adaptive_halting.json", rerun=True)
    assert out.name == "mt5_adaptive_halting_seeds10.json"
    assert mt5.resolve_out("runs/mot/x.json", rerun=False).name == "x.json"


def test_mt5_runs_and_matches_mean_flops(tmp_path):
    cfg = mt5.default_cfg([0, 1], **TINY_MT)
    out = mt5.MT5AdaptiveHalting().run(cfg, DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert len(out["per_seed"]) == 2
    for rec in out["per_seed"]:
        assert 1.0 <= rec["mean_halt_steps"] <= TINY_MT["max_steps"]
        # the random-allocation control realizes the adaptive arm's mean step budget
        assert abs(rec["control_mean_steps"] - rec["mean_halt_steps"]) < 0.6
        assert set(rec["compute"]) >= {"matched", "ratio", "tol"}
        assert "regime_calibrated" in rec["d3"]
        assert "gradient_present" in rec["hardness_gradient"]
    assert out["halt_entropy_bits_mean"] >= 0.0
    assert out["preregistered"]["entropy_floor_bits"] == mt5.ENTROPY_FLOOR_BITS
    assert isinstance(out["verdict"], str) and out["verdict"]


def test_mt5_entropy_collapse_is_automatic_null():
    import torch

    # a constant halt-step distribution has zero entropy: the automatic-null trigger
    assert mt5.step_entropy_bits(torch.full((32,), 3.0)) == 0.0
    two_level = mt5.step_entropy_bits(torch.tensor([1.0] * 16 + [2.0] * 16))
    assert abs(two_level - 1.0) < 1e-6


def test_mt6_runs_and_free_rule_is_tuned(tmp_path):
    cfg = mt6.default_cfg([0], **TINY_MT)
    out = mt6.MT6ConfidenceStop().run(cfg, DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    rec = out["per_seed"][0]
    # the free rule's threshold was tuned toward the trained arm's mean steps on train
    assert abs(rec["free_mean_steps"] - rec["trained_mean_steps"]) < 1.5
    for key in ("auroc_trained", "auroc_free", "auroc_shuffled_labels"):
        assert 0.0 <= rec[key] <= 1.0
    # shuffled-labels control must sit near chance, else the AUROC yardstick is broken
    assert 0.2 <= out["auroc_shuffled_labels_mean"] <= 0.8
    assert isinstance(out["verdict"], str)


def test_dr9_runs_all_arms_at_tuned_budgets(tmp_path):
    cfg = dr9.default_cfg(
        [0],
        dim=16,
        hidden=32,
        verifier_hidden=16,
        n_classes=4,
        samples=240,
        n_single=2,
        n_start=1,
        n_max=4,
        epochs=30,
        lr=1e-2,
    )
    out = dr9.DR9VerifyRevise().run(cfg, DEV, tmp_path)
    assert isinstance(out["null_supported"], bool)
    rec = out["per_seed"][0]
    assert set(rec["arms"]) == {"verify_revise", "shuffled_verifier", "free_norm"}
    for arm in rec["arms"].values():
        assert 1.0 <= arm["mean_steps"] <= 4.0
        assert arm["mean_score_evals"] >= 0.0
        assert set(arm["compute"]) >= {"matched", "ratio"}
    assert "gain_vs_single_ci" in out and "delta_vs_shuffled_ci" in out
    assert out["preregistered"]["flop_tol"] == dr9.FLOP_TOL


def test_dr8_synthetic_arm_runs(tmp_path):
    cfg = dr8.default_cfg(
        [0],
        cache="synthetic",
        dim=16,
        hidden=32,
        n_classes=4,
        samples=200,
        n_train_steps=3,
        epochs=40,
        lr=1e-2,
    )
    out = dr8.DR8FixedPoint().run(cfg, DEV, tmp_path)
    assert out["cache"] == "synthetic"
    assert isinstance(out["null_supported"], bool)
    rec = out["per_seed"][0]
    assert rec["convergence"]["classification"] in {"converges", "drifts", "limit_cycle"}
    # the curve carries every preregistered checkpoint plus the trained horizon (3 here)
    assert set(rec["ce_curve"]) == {str(k) for k in dr8.CHECKPOINTS} | {"3"}
    assert isinstance(rec["past_horizon_loss_rises"], bool)
    assert isinstance(rec["fixed_point_collapsed"], bool)
    # weight-tied refiner has strictly fewer params than the untied matched-depth control
    assert out["params"]["refiner"] < out["params"]["untied_control"]


def test_dr8_cache_arms_fail_loudly(tmp_path):
    with pytest.raises(KeyError):
        dr8.load_cache_xy(dr8.default_cfg([0], cache="bogus").experiment)
    with pytest.raises(FileNotFoundError):
        dr8.load_cache_xy(dr8.default_cfg([0], cache="randominit_vitl", data_dir=str(tmp_path)).experiment)


def test_wp03_experiments_declare_contract():
    for cls in (mt5.MT5AdaptiveHalting, mt6.MT6ConfidenceStop, dr9.DR9VerifyRevise, dr8.DR8FixedPoint):
        exp = cls()
        assert exp.id and exp.metric and exp.baseline and exp.ablation
        assert exp.null_hypothesis and exp.tier == "cpu-now"
