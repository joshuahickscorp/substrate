
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

    arms = {"dense", "kwta", "moe"}
    assert set(out["bwt"]) == arms
    assert set(out["forgetting"]) == arms
    assert set(out["n_params"]) == arms

    assert "sparse_beats_dense" in out
    assert isinstance(out["sparse_beats_dense"], bool)
    assert out["null_supported"] == (not out["sparse_beats_dense"])
    assert set(out["bwt_gain_over_dense"]) == {"kwta", "moe"}
    assert out["best_sparse"] in {"kwta", "moe"}
    gains = out["bwt_gain_over_dense"]
    assert out["best_sparse"] == max(gains, key=lambda n: gains[n])
    assert out["dense_bwt"] == out["bwt"]["dense"]

    assert out["param_matched"] is True

    assert "moe_first" in out["routing_entropy"]
    assert "moe_last" in out["routing_entropy"]
    assert out["routing_entropy"]["moe_first"] >= 0.0

    assert out["speedup"]["claim_tier"] == "gpu-later"
    assert out["speedup"]["required_for_success"] is False
    assert set(out["speedup"]["wall_clock_s"]) == arms

    assert (tmp_path / "e7_sparse" / "e7_interference.png").exists()
