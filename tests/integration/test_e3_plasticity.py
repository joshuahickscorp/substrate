
from mop import config, devices
from mop.experiments.e3_plasticity import ARMS, E3


def _toy_cfg():
    return config.compose(
        [
            "experiment=e3_plasticity",
            "device=cpu",
            "experiment.stream.dim=48",
            "experiment.stream.n_tasks=4",
            "experiment.stream.samples_per_task=160",
            "experiment.train.epochs_per_task=3",
            "experiment.fisher.checkpoints=6",
            "experiment.fisher.steps_per_ckpt=6",
        ]
    )


def test_e3_runs_arms_null_and_fisher(tmp_path):
    cfg = _toy_cfg()
    out = E3().run(cfg, devices.resolve("cpu"), tmp_path / "e3_plasticity")

    assert set(out["arms"]) == set(ARMS)
    assert set(out["bwt"]) == set(ARMS)
    for arm in ARMS:
        assert "backward_transfer" in out["arms"][arm]
    assert "frontier_auc" in out

    for key in ("staged_beats_constant", "staged_beats_decay", "staged_beats_both"):
        assert key in out and isinstance(out[key], bool)
    assert out["staged_beats_both"] == (out["staged_beats_constant"] and out["staged_beats_decay"])
    margin = out["tie_margin"]
    assert out["staged_beats_decay"] == (out["bwt"]["staged"] > out["bwt"]["decay"] + margin)

    assert isinstance(out["fisher_trace"], list) and len(out["fisher_trace"]) >= 3
    assert isinstance(out["fisher_peak_index"], int)
    assert isinstance(out["fisher_rise_then_fall"], bool)

    assert (tmp_path / "e3_plasticity" / "e3_plasticity.png").exists()
