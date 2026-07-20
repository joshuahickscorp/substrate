
from mop import config, devices
from mop.experiments import get_experiment


def _toy_cfg():
    return config.compose(
        [
            "experiment=e1_baseline",
            "device=cpu",
            "experiment.stream.dim=48",  # toy dim for speed; regime is unchanged
            "experiment.stream.samples_per_task=200",
            "shell.buffer.capacity=2048",
            "shell.buffer.index=brute",
            "shell.consolidation.method=ewc",
            "shell.consolidation.ewc_lambda=1000.0",
        ]
    )


def test_e1_forget_then_retain(tmp_path):
    cfg = _toy_cfg()
    dev = devices.resolve("cpu")
    out = get_experiment("e1_baseline").run(cfg, dev, tmp_path / "e1")

    gate = out["gate"]
    assert gate["naive_forgets"], f"naive did not forget: {gate}"
    assert gate["protected_retains"], f"protected did not retain over naive: {gate}"
    assert gate["both_learn_last_task"], f"an arm failed to learn last task: {gate}"
    assert gate["passed"], f"E1 gate failed: {gate}"
    assert (tmp_path / "e1" / "e1_frontier.png").exists()


def test_e1_protected_frontier_dominates_or_matches():
    cfg = _toy_cfg()
    dev = devices.resolve("cpu")
    out = get_experiment("e1_baseline").run(cfg, dev, None or __import__("pathlib").Path("runs/_e1_test"))
    arms = out["arms"]
    assert arms["protected"]["backward_transfer"] > arms["naive"]["backward_transfer"]
