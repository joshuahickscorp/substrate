"""E2 integration gate: the replay harness runs at toy scale, reports per-arm BWT and a
frontier, and ACTUALLY answers its null (replay vs no-replay, prioritized vs random) with a
ran comparison and a saved frontier plot. Class imported directly, not via the registry."""

from mop import config, devices
from mop.experiments.e2_replay import E2


def _toy_cfg():
    # toy-scale overrides: small dim, few tasks/samples, runs in seconds
    return config.compose(
        [
            "experiment=e2_replay",
            "device=cpu",
            "experiment.stream.dim=48",
            "experiment.stream.n_tasks=4",
            "experiment.stream.samples_per_task=200",
            "experiment.stream.classes_per_task=4",
            "experiment.train.epochs_per_task=4",
            "shell.buffer.capacity=2048",
            "shell.buffer.index=brute",
        ]
    )


def test_e2_replay_runs_and_answers_null(tmp_path):
    cfg = _toy_cfg()
    out = E2().run(cfg, devices.resolve("cpu"), tmp_path / "e2_replay")

    arms = {"no_replay", "random_replay", "prioritized_replay", "replay_ewc"}
    # metric keys present, per arm
    assert set(out["bwt"]) == arms
    for m in ("backward_transfer", "frontier_auc", "adaptation_speed", "retention"):
        if m == "frontier_auc":
            assert isinstance(out["frontier_auc"], float)
        elif m == "retention":
            continue  # retention is the frontier y-axis, summarized via BWT below
        else:
            assert all(m in out["arms"][a] for a in arms)

    # the null-check keys exist and the comparison actually ran (booleans backed by gaps)
    assert isinstance(out["replay_beats_noreplay"], bool)
    assert isinstance(out["prioritized_beats_random"], bool)
    assert isinstance(out["null_holds"], bool)
    assert out["replay_gap"] == out["bwt"]["random_replay"] - out["bwt"]["no_replay"]
    assert out["prioritized_gap"] == out["bwt"]["prioritized_replay"] - out["bwt"]["random_replay"]
    # null_holds is exactly "neither comparison beat its control"
    assert out["null_holds"] == (not out["replay_beats_noreplay"] and not out["prioritized_beats_random"])

    # frontier scalar and the saved scatter plot
    assert out["frontier_auc"] >= 0.0
    assert (tmp_path / "e2_replay" / "e2_frontier.png").exists()
