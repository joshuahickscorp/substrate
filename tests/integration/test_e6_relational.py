"""E6 runs the relational 2-vs-2.1 (pooled-vs-dense) comparison at toy scale: two encoders,
a recombination-generalization task, a structured head vs a parameter-matched flat head per
substrate, and the explicit null check (structured_beats_flat AND dense_gain_larger)."""

from mop import config, devices
from mop.experiments.e6_relational import E6


def test_e6_relational_comparison(tmp_path):
    cfg = config.compose(
        [
            "experiment=e6_relational",
            "device=cpu",
            "experiment.samples=96",  # toy: keep it fast
            "experiment.epochs=40",
            "experiment.n_attr=4",
        ]
    )
    out = E6().run(cfg, devices.resolve("cpu"), tmp_path / "e6_relational")

    # headline metric keys present
    for k in ("pooled_delta", "dense_delta", "headline_2_vs_21_delta"):
        assert k in out, (k, list(out))

    # the comparison actually ran on both substrates with real (above-chance-capable) heads
    for sub in ("pooled", "dense"):
        s = out[sub]
        for k in ("structured_acc", "flat_acc", "chance", "n_test_heldout_pairs"):
            assert k in s, (sub, k, list(s))
        assert 0.0 <= s["structured_acc"] <= 1.0 and 0.0 <= s["flat_acc"] <= 1.0
        assert s["n_test_heldout_pairs"] > 0  # held-out recombinations exist

    # the structured head is parameter-matched to the flat baseline (within a small factor)
    for sub in ("pooled", "dense"):
        fp, sp = out[sub]["flat_params"], out[sub]["structured_params"]
        assert 0.5 <= sp / fp <= 1.5, (sub, fp, sp)

    # the headline delta is the dense-minus-pooled delta, computed (not a placeholder)
    assert out["headline_2_vs_21_delta"] == out["dense_delta"] - out["pooled_delta"]

    # the explicit null check ran and is boolean
    assert isinstance(out["structured_beats_flat"], bool)
    assert isinstance(out["dense_gain_larger"], bool)
    assert out["null_rejected"] == (out["structured_beats_flat"] and out["dense_gain_larger"])

    # plot saved
    assert (tmp_path / "e6_relational" / "e6_relational.png").exists()
