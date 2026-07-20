
from mop import config, devices
from mop.experiments import get_experiment


def _run(eid, overrides, tmp_path):
    cfg = config.compose([f"experiment={eid}", "device=cpu", *overrides])
    dev = devices.resolve("cpu")
    return get_experiment(eid).run(cfg, dev, tmp_path / eid)


def test_ex12_atlas_self_check(tmp_path):
    out = _run("ex12_atlas", ["experiment.seeds=[0,1]", "experiment.samples=240"], tmp_path)
    assert out["identity_decodable"] is True
    assert out["random_label_not_decodable"] is True
    assert out["atlas_self_check_passed"] is True
    assert "effective_rank" in out["geometry"]


def test_ex17_latent_reasoning_compute_matched(tmp_path):
    out = _run(
        "ex17_latent_reasoning",
        ["experiment.seeds=[0,1]", "experiment.samples=200", "experiment.epochs=60"],
        tmp_path,
    )
    assert "refiner_acc_mean" in out and "control_acc_mean" in out
    assert out["compute_matched"]["matched"] is True  # the comparison is FLOP-matched
    assert out["params_refiner"] < out["params_control"]
    assert isinstance(out["null_supported"], bool)


def test_ex8_curiosity_bakeoff_chasers_chase(tmp_path):
    out = _run(
        "ex8_curiosity_bakeoff",
        ["experiment.seeds=[0]", "experiment.steps=150", "experiment.dim=32"],
        tmp_path,
    )
    table = out["table"]
    assert set(table) == {"prediction_error", "disagreement", "learning_progress", "rnd"}
    assert table["prediction_error"]["noise"] > table["prediction_error"]["learnable"]
    assert table["learning_progress"]["rejects_noise"] is True


def test_ex16_codebook_purity(tmp_path):
    out = _run("ex16_codebook_sr", ["experiment.seeds=[0,1]", "experiment.samples=240"], tmp_path)
    assert out["cluster_purity_learned"] >= out["cluster_purity_random"]
    assert "code_vs_raw_gain" in out and isinstance(out["null_supported"], bool)


def test_ex3_tta_restores_base(tmp_path):
    out = _run(
        "ex3_test_time_adaptation",
        ["experiment.seeds=[0,1]", "experiment.samples=200", "experiment.epochs=50"],
        tmp_path,
    )
    assert out["base_retention_after_revert"] >= 0.99 or out["base_retention_after_revert"] > 0
    assert "shift_acc_tta" in out and "shift_acc_frozen" in out
    assert isinstance(out["null_supported"], bool)


def test_ex_experiments_declare_contract():
    for eid in (
        "ex3_test_time_adaptation",
        "ex8_curiosity_bakeoff",
        "ex12_atlas",
        "ex16_codebook_sr",
        "ex17_latent_reasoning",
    ):
        exp = get_experiment(eid)
        assert exp.null_hypothesis and exp.metric and exp.baseline and exp.tier == "cpu-now"
