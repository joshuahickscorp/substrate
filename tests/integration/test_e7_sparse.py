"""E7 runs the three matched-parameter head variants on a domain-incremental stream and
answers its null (sparse_beats_dense) with the BWT comparison, at toy scale. The class is
imported directly, not via the registry."""

from mop import config, devices
from mop.experiments.e7_sparse import E7


def test_e7_runs_and_answers_null(tmp_path):
    cfg = config.compose(
        [
            "experiment=e7_sparse",
            "device=cpu",
            "experiment.stream.n_tasks=4",
            "experiment.stream.dim=48",
            "experiment.stream.samples_per_task=128",
            "experiment.epochs_per_task=25",
        ]
    )
    out = E7().run(cfg, devices.resolve("cpu"), tmp_path / "e7_sparse")

    # the three arms ran and BWT (the interference metric) is reported per arm
    arms = {"dense", "kwta", "moe"}
    assert set(out["bwt"]) == arms
    assert set(out["forgetting"]) == arms
    assert set(out["n_params"]) == arms

    # the explicit null check exists, is boolean, and the dense-vs-sparse comparison ran
    assert "sparse_beats_dense" in out
    assert isinstance(out["sparse_beats_dense"], bool)
    assert out["null_supported"] == (not out["sparse_beats_dense"])
    assert set(out["bwt_gain_over_dense"]) == {"kwta", "moe"}
    assert out["best_sparse"] in {"kwta", "moe"}
    # the comparison is consistent: best_sparse is the largest BWT gain over dense
    gains = out["bwt_gain_over_dense"]
    assert out["best_sparse"] == max(gains, key=lambda n: gains[n])
    assert out["dense_bwt"] == out["bwt"]["dense"]

    # parameter-matching control: the variants are matched to dense (the corpus control)
    assert out["param_matched"] is True

    # MoE routing entropy is reported (a corpus null signature lives here)
    assert "moe_first" in out["routing_entropy"]
    assert "moe_last" in out["routing_entropy"]
    assert out["routing_entropy"]["moe_first"] >= 0.0

    # speedup is a SEPARATE, gpu-later, not-required claim
    assert out["speedup"]["claim_tier"] == "gpu-later"
    assert out["speedup"]["required_for_success"] is False
    assert set(out["speedup"]["wall_clock_s"]) == arms

    # the saved plot exists
    assert (tmp_path / "e7_sparse" / "e7_interference.png").exists()
