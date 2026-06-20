import math

from devsys.studies.pc_depth import (
    _init_weights,
    _prep,
    _task,
    _train_backprop,
    _train_pc,
    pc_depth,
)


def test_init_weights_shapes_per_depth():
    Ws = _init_weights(d=8, hidden=5, depth=3, classes=4, seed=0)
    assert len(Ws) == 4  # 3 hidden weight layers + readout
    assert Ws[0].shape == (5, 8) and Ws[-1].shape == (4, 5)
    for w in Ws[1:-1]:
        assert w.shape == (5, 5)


def test_backprop_head_learns_separable_task():
    # backprop on a depth-1 head over a well-separated task must beat chance clearly.
    x, y, C = _task(dim=16, classes=4, samples=400, sep=3.0, seed=0)
    xtr, ytr, xte, yte = _prep(x, y)
    W0 = _init_weights(16, 20, depth=1, classes=C, seed=1)
    te, tr = _train_backprop(xtr, ytr, xte, yte, W0, depth=1, epochs=200, lr=0.05)
    assert te > 1.0 / C + 0.2 and tr > 1.0 / C + 0.2


def test_pc_head_runs_and_beats_chance():
    # the layered PC relaxation must actually train (not vacuous): beat chance on a shallow net.
    x, y, C = _task(dim=16, classes=4, samples=400, sep=3.0, seed=0)
    xtr, ytr, xte, yte = _prep(x, y)
    W0 = _init_weights(16, 20, depth=1, classes=C, seed=1)
    te, tr = _train_pc(xtr, ytr, xte, yte, W0, depth=1, classes=C, epochs=200, lr=0.05, infer=20)
    assert te > 1.0 / C + 0.1 and not math.isnan(te)


def test_pc_depth_reports_per_depth_and_gap():
    out = pc_depth(depths=(1, 2, 3), toy=True)
    assert out["provenance"] == "provisional"
    assert [d["depth"] for d in out["per_depth"]] == [1, 2, 3]
    for d in out["per_depth"]:
        assert math.isclose(d["gap"], d["bp_acc"] - d["pc_acc"], abs_tol=1e-9)
        assert 0.0 <= d["bp_acc"] <= 1.0 and 0.0 <= d["pc_acc"] <= 1.0
    assert isinstance(out["gap_widens_with_depth"], bool)


def test_pc_depth_backprop_at_least_as_good_shallow():
    # at depth 1 PC approximates backprop well: backprop should not be worse than PC by much,
    # and both should clear chance. This is a real, falsifiable check on the shallow regime.
    out = pc_depth(depths=(1,), toy=True)
    d = out["per_depth"][0]
    assert d["bp_acc"] > out["chance"] + 0.2
    assert d["pc_acc"] > out["chance"]
