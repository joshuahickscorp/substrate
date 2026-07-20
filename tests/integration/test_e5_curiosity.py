
from mop import config, devices
from mop.experiments.e5_curiosity import E5  # direct import, NOT via the registry


def test_e5_curiosity_data_selection(tmp_path):
    cfg = config.compose(
        [
            "experiment=e5_curiosity",
            "device=cpu",
            "experiment.n_each=64",
            "experiment.rounds=6",
            "experiment.train_steps=8",
        ]
    )
    out = E5().run(cfg, devices.resolve("cpu"), tmp_path / "e5_curiosity")

    policies = ("random", "prediction_error", "learning_progress")
    for p in policies:
        assert p in out["noise_fraction"] and p in out["learnable_improvement"]
        assert 0.0 <= out["noise_fraction"][p] <= 1.0

    assert "pe_chases_noise" in out and "lp_resists_noise" in out
    assert isinstance(out["pe_chases_noise"], bool)
    assert isinstance(out["lp_resists_noise"], bool)
    assert out["null_supported"] == (out["pe_chases_noise"] and out["lp_resists_noise"])

    pe_nf = out["noise_fraction"]["prediction_error"]
    lp_nf = out["noise_fraction"]["learning_progress"]
    assert pe_nf > lp_nf, (pe_nf, lp_nf)

    assert out["env_smoke_ok"] is True
    assert out["env_rollout_ok"] is True
    assert out["env_tier"] == "cpu-now"
    assert out["environment_contract"]["counterfactuals"] > 0
    assert out["environment_contract"]["natural_embodiment_claim"] is False
    assert E5.tier == "cpu-now"

    assert (tmp_path / "e5_curiosity" / "e5_curiosity.png").exists()
