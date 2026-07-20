
from mop import config, devices
from mop.experiments import get_experiment
from mop.learning.alternatives import RULES


def test_i4_runs_and_tabulates(tmp_path):
    cfg = config.compose(
        [
            "experiment=i4_backprop_alts",
            "device=cpu",
            "experiment.samples=240",
            "experiment.epochs=120",
            "experiment.seeds=[0,1]",
        ]
    )
    out = get_experiment("i4_backprop_alts").run(cfg, devices.resolve("cpu"), tmp_path / "i4")

    assert set(out["table"]) == set(RULES)
    ceiling = out["ceiling_backprop_acc"]
    for name, row in out["table"].items():
        assert row["acc_mean"] <= ceiling + 0.05, (name, row["acc_mean"], ceiling)
    for name, row in out["table"].items():
        assert row["acc_mean"] > row["chance"] + 0.1, (name, row)
    assert out["table"]["backprop"]["weight_transport"] is True
    assert out["table"]["forward_forward"]["local"] is True
    assert out["table"]["forward_forward"]["separate_backward"] is False
    assert (tmp_path / "i4" / "i4_comparison.png").exists()
    assert (tmp_path / "i4" / "i4_table.md").exists()
    assert out["best_local_rule"] in RULES


def test_i4_rules_locality_contract():
    from mop.substrate.datasets import make_task_stream

    t = make_task_stream(n_tasks=1, dim=24, classes_per_task=3, samples_per_task=240, separation=2.0, seed=0)[
        0
    ]
    bp = RULES["backprop"](t.x, t.y, seed=0)
    ff = RULES["forward_forward"](t.x, t.y, seed=0)
    assert bp.weight_transport and bp.separate_backward  # backprop needs both
    assert ff.local and ff.activation_memory == 0.0  # FF is local, no stored activations
