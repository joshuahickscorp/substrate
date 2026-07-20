
from pathlib import Path

from mop import config, devices
from mop.experiments import get_experiment


def _run(eid, overrides, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu", *overrides])
    dev = devices.resolve("cpu")
    return get_experiment(eid).run(cfg, dev, Path(tmp_path) / eid)


def test_ex2_latent_planning_runs(tmp_path):
    out = _run(
        "ex2_latent_planning",
        [
            "experiment.n_trials=20",
            "experiment.n_shooting_samples=64",
            "experiment.dim=16",
            "experiment.horizon=4",
            "experiment.epochs=40",
        ],
        tmp_path,
    )
    assert isinstance(out, dict)
    assert "null_supported" in out and isinstance(out["null_supported"], bool)
    assert "planner_beats_flat_head" in out
    assert "planner_beats_action_shuffle" in out
    assert "planning_licensed" in out


def test_ex9_slot_attention_runs(tmp_path):
    out = _run(
        "ex9_slot_attention",
        ["experiment.n_windows=200", "experiment.epochs=40", "experiment.window_len=4"],
        tmp_path,
    )
    assert isinstance(out, dict)
    assert "null_supported" in out and isinstance(out["null_supported"], bool)
    assert "relation_decoding" in out and "pooled_ceiling_gap" in out


def test_ex15_rejuvenation_runs(tmp_path):
    out = _run(
        "ex15_rejuvenation",
        [
            "experiment.stream.n_tasks=12",
            "experiment.n_tasks_control=6",
            "experiment.stream.dim=16",
            "experiment.head.hidden=16",
            "experiment.rejuvenation_interval=4",
            "experiment.eval_every=3",
        ],
        tmp_path,
    )
    assert isinstance(out, dict)
    assert "null_supported" in out and isinstance(out["null_supported"], bool)
    assert "effective_rank" in out and "dead_unit_count" in out and "retained_accuracy" in out


def test_migrated_experiments_declare_contract():
    for eid in ("ex2_latent_planning", "ex9_slot_attention", "ex15_rejuvenation"):
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
