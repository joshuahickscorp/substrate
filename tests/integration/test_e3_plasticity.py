"""E3 critical-period schedule (staged plasticity). Toy-scale check that the four plasticity
arms run, the frontier is computed, the staged-vs-(constant,decay) null comparison actually
ran, the Fisher critical-period signature is reported, and the plot is saved. The class is
imported directly, not via the registry."""

from mop import config, devices
from mop.experiments.e3_plasticity import ARMS, E3


def _toy_cfg():
    # toy-scale overrides: small dim, fewer tasks, few epochs => seconds
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

    # all four plasticity arms ran and report BWT (the metric)
    assert set(out["arms"]) == set(ARMS)
    assert set(out["bwt"]) == set(ARMS)
    for arm in ARMS:
        assert "backward_transfer" in out["arms"][arm]
    assert "frontier_auc" in out

    # the null check keys exist and the comparison actually ran (booleans)
    for key in ("staged_beats_constant", "staged_beats_decay", "staged_beats_both"):
        assert key in out and isinstance(out[key], bool)
    # staged_beats_both is exactly the conjunction of the two arm comparisons
    assert out["staged_beats_both"] == (out["staged_beats_constant"] and out["staged_beats_decay"])
    # the comparison is grounded in real numbers: margin applied to the BWT gap
    margin = out["tie_margin"]
    assert out["staged_beats_decay"] == (out["bwt"]["staged"] > out["bwt"]["decay"] + margin)

    # Fisher critical-period signature reported
    assert isinstance(out["fisher_trace"], list) and len(out["fisher_trace"]) >= 3
    assert isinstance(out["fisher_peak_index"], int)
    assert isinstance(out["fisher_rise_then_fall"], bool)

    # plot saved
    assert (tmp_path / "e3_plasticity" / "e3_plasticity.png").exists()
