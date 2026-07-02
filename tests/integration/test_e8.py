"""E8 runs at toy scale: matched-param MLP vs dendritic head on an online domain stream,
reporting capacity-per-param, adaptation speed, forgetting, and the explicit null check."""

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

    # the metric keys the doctrine declares are present per arm
    for arm in ("mlp", "dendritic"):
        row = out["arms"][arm]
        for key in ("capacity_per_param", "adaptation_speed", "forgetting"):
            assert key in row, (arm, key)
        assert row["params"] > 0

    # comparison is at MATCHED params, not added ones (within a small tolerance)
    pm = out["matched_params"]
    assert abs(pm["mlp"] - pm["dendritic"]) <= 0.1 * pm["dendritic"], pm

    # the null check ran and is a real boolean answering dendritic_beats_mlp
    assert "dendritic_beats_mlp" in out
    assert isinstance(out["dendritic_beats_mlp"], bool)
    assert out["null_supported"] == (not out["dendritic_beats_mlp"])
    # the comparison gaps were actually computed
    assert "capacity_per_param_gap" in out and "adaptation_speed_gap" in out

    # both arms actually learned above chance (chance = 1/classes_per_task)
    chance = 1.0 / int(cfg.experiment.classes_per_task)
    for arm in ("mlp", "dendritic"):
        assert out["arms"][arm]["final_acc_mean"] > chance, arm

    # saved plot exists
    assert (tmp_path / "e8" / "e8_dendritic.png").exists()
    assert out["plot"].endswith("e8_dendritic.png")
