"""Phase 4 gate: E1 must demonstrate forgetting (naive) then retention (replay+EWC) at toy
scale under fixed seeds, with a saved plot. This is the contract that licenses every later
experiment."""

from mop import config, devices
from mop.experiments import get_experiment


def _toy_cfg():
    # toy-scale overrides: small dim, few tasks, runs in seconds
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
    # 1) the naive arm visibly forgets earlier tasks
    assert gate["naive_forgets"], f"naive did not forget: {gate}"
    # 2) replay+EWC retains better than naive by a margin
    assert gate["protected_retains"], f"protected did not retain over naive: {gate}"
    # 3) both arms actually learn the final task (sensitivity, not just rigidity)
    assert gate["both_learn_last_task"], f"an arm failed to learn last task: {gate}"
    # 4) the whole contract
    assert gate["passed"], f"E1 gate failed: {gate}"
    # plot saved
    assert (tmp_path / "e1" / "e1_frontier.png").exists()


def test_e1_protected_frontier_dominates_or_matches():
    cfg = _toy_cfg()
    dev = devices.resolve("cpu")
    out = get_experiment("e1_baseline").run(cfg, dev, None or __import__("pathlib").Path("runs/_e1_test"))
    arms = out["arms"]
    # protected has higher (less negative) backward transfer than naive
    assert arms["protected"]["backward_transfer"] > arms["naive"]["backward_transfer"]
