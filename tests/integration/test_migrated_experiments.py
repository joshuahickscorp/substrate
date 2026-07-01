"""ex2_latent_planning (synthetic arm), ex9_slot_attention, and ex15_rejuvenation: three rows migrated
from the Studio-gated list to cpu-now this session, once their real blocker was found to be unwritten code
(ex15, ex9) or a live-environment arm that is separable from a synthetic precursor (ex2), not a hardware
ceiling. Each ships a SCALED (not toy) default; these tests use small dotlist overrides to stay fast.
Asserts MECHANICS only, never a particular scientific outcome."""

from pathlib import Path

from devsys import config, devices
from devsys.experiments import get_experiment


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
    # the two standing controls and the rollout-predictability gate must all be reported
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
