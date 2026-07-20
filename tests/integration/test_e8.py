
from mop import config, devices
from mop.experiments.e8_dendritic import E8


def test_e8_runs_and_checks_null(tmp_path):
    cfg = config.compose(
        [
            "experiment=e8_dendritic",
            "device=cpu",
            "experiment.n_tasks=3",
            "experiment.samples_per_task=160",
            "experiment.seeds=[0,1]",
        ]
    )
    out = E8().run(cfg, devices.resolve("cpu"), tmp_path / "e8")

    for arm in ("mlp", "dendritic"):
        row = out["arms"][arm]
        for key in ("capacity_per_param", "adaptation_speed", "forgetting"):
            assert key in row, (arm, key)
        assert row["params"] > 0

    pm = out["matched_params"]
    assert abs(pm["mlp"] - pm["dendritic"]) <= 0.1 * pm["dendritic"], pm

    assert "dendritic_beats_mlp" in out
    assert isinstance(out["dendritic_beats_mlp"], bool)
    assert out["null_supported"] == (not out["dendritic_beats_mlp"])
    assert "capacity_per_param_gap" in out and "adaptation_speed_gap" in out

    chance = 1.0 / int(cfg.experiment.classes_per_task)
    for arm in ("mlp", "dendritic"):
        assert out["arms"][arm]["final_acc_mean"] > chance, arm

    assert (tmp_path / "e8" / "e8_dendritic.png").exists()
    assert out["plot"].endswith("e8_dendritic.png")
