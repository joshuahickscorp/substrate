
from pathlib import Path

from mop import config, devices
from mop.experiments import get_experiment


def _run(eid, overrides, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu", *overrides])
    dev = devices.resolve("cpu")
    return get_experiment(eid).run(cfg, dev, Path(tmp_path) / eid)


def test_ex13_long_stream_runs(tmp_path):
    out = _run(
        "ex13_long_stream",
        [
            "experiment.n_tasks=12",
            "experiment.n_tasks_control=6",
            "experiment.anchor_tasks=2",
            "experiment.eval_every=2",
        ],
        tmp_path,
    )
    assert isinstance(out, dict)
    assert "null_supported" in out and isinstance(out["null_supported"], bool)
    assert "forgetting_curve" in out and "effective_rank" in out
    assert "survives_frozen_random_control" in out


def test_ex5_local_rules_scale_runs(tmp_path):
    out = _run(
        "ex5_local_rules_scale",
        [
            "experiment.n_tasks=4",
            "experiment.hidden_widths=[8,16]",
            "experiment.epochs_per_task=3",
            "experiment.n_anchors=3",
        ],
        tmp_path,
    )
    assert isinstance(out, dict)
    assert "null_supported" in out and isinstance(out["null_supported"], bool)
    assert "table" in out and "depth_sweep" in out


def test_studio_gated_implementable_declare_contract():
    for eid in ("ex13_long_stream", "ex5_local_rules_scale"):
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
