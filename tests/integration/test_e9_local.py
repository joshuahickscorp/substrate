"""E9 runs the streaming local-learning comparison at toy scale: backprop ceiling vs the
local rules, reporting accuracy, activation memory, and online stability, and answering the
null (any_local_within_margin, best_local_memory_win) explicitly. Imported directly, not via
the registry."""

from mop import config, devices
from mop.experiments.e9_local import E9, LOCAL_RULES


def test_e9_streaming_local_learning(tmp_path):
    cfg = config.compose(
        [
            "experiment=e9_local",
            "device=cpu",
            "experiment.samples=200",
            "experiment.passes=3",
            "experiment.seeds=[0,1]",
        ]
    )
    out = E9().run(cfg, devices.resolve("cpu"), tmp_path / "e9_local")

    # the metric keys are present
    for k in ("ceiling_backprop_acc", "backprop_activation_memory", "backprop_online_stability_std"):
        assert k in out, k
    # every rule (backprop + the local subset) is tabulated with the streaming metrics
    assert set(out["table"]) == {"backprop", *LOCAL_RULES}
    for name, row in out["table"].items():
        assert "acc_mean" in row and "activation_memory" in row and "online_stability_std" in row, (name, row)

    # the local subset is genuinely local and never out-transports backprop
    for name in LOCAL_RULES:
        assert out["table"][name]["local"] is True, name
    assert out["table"]["backprop"]["weight_transport"] is True

    # the null-check keys exist and are real booleans (the comparison ran)
    assert isinstance(out["any_local_within_margin"], bool)
    assert isinstance(out["best_local_memory_win"], bool)
    assert isinstance(out["null_supported"], bool)
    # the comparison actually executed: every local rule got a gap and a memory verdict vs backprop
    ceiling = out["ceiling_backprop_acc"]
    bp_mem = out["backprop_activation_memory"]
    for name in LOCAL_RULES:
        row = out["table"][name]
        assert abs(row["gap_to_backprop"] - (ceiling - row["acc_mean"])) < 1e-9, name
        assert row["memory_win"] == (row["activation_memory"] < bp_mem), name
    # null_supported is consistent with its two component answers
    assert out["null_supported"] == (
        (not out["any_local_within_margin"]) and (not out["best_local_memory_win"])
    )
    # MEASURED, falsifiable: backprop retains a real autograd activation footprint and the
    # leanest local rule (manual, no autograd graph) retains strictly less. Would fail if the
    # memory measurement broke or every local rule used autograd as heavily as backprop.
    bp_mem = out["backprop_activation_memory"]
    min_local_mem = min(out["table"][n]["activation_memory"] for n in LOCAL_RULES)
    assert bp_mem > 0 and min_local_mem < bp_mem, (min_local_mem, bp_mem)

    # the plot was saved
    assert (tmp_path / "e9_local" / "e9_streaming.png").exists()
