
from mop import config, devices
from mop.experiments.e4_neuromod import E4


def test_e4_neuromod_runs_and_checks_null(tmp_path):
    cfg = config.compose(
        [
            "experiment=e4_neuromod",
            "device=cpu",
            "experiment.n=128",
            "experiment.steps=80",
            "experiment.batch=48",
            "experiment.ensemble_size=4",
            "experiment.cal_epochs=60",
        ]
    )
    out = E4().run(cfg, devices.resolve("cpu"), tmp_path / "e4_neuromod")

    for k in ("adapt_speed_learnable", "gate_on_noise", "calibration_ece"):
        assert k in out, k
    gon = out["gate_on_noise"]
    assert set(gon) == {"ungated", "point_error", "disagreement"}
    for v in gon.values():
        assert isinstance(v, float)
    assert gon["ungated"] == 1.0  # ungated is the normalized reference

    assert isinstance(out["point_error_chases_noise"], bool)
    assert isinstance(out["disagreement_ignores_noise"], bool)
    assert isinstance(out["null_supported"], bool)
    assert out["null_supported"] == (out["point_error_chases_noise"] and out["disagreement_ignores_noise"])

    assert 0.0 <= out["calibration_ece"] <= 1.0
    assert isinstance(out["adapt_speed_learnable"], float)

    diag = out["diagnostic"]
    assert "disagreement" in diag and "raw_error" in diag

    assert (tmp_path / "e4_neuromod" / "e4_neuromod.png").exists()
