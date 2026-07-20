
from mop import config, devices
from mop.experiments.e6_relational import E6


def test_e6_relational_comparison(tmp_path):
    cfg = config.compose(
        [
            "experiment=e6_relational",
            "device=cpu",
            "experiment.execution_path=legacy-fixture",
            "experiment.allow_legacy_fixture=true",
            "experiment.samples=96",  # toy: keep it fast
            "experiment.epochs=40",
            "experiment.n_attr=4",
        ]
    )
    out = E6().run(cfg, devices.resolve("cpu"), tmp_path / "e6_relational")

    for k in ("pooled_delta", "dense_delta", "headline_2_vs_21_delta"):
        assert k in out, (k, list(out))

    for sub in ("pooled", "dense"):
        s = out[sub]
        for k in ("structured_acc", "flat_acc", "chance", "n_test_heldout_pairs"):
            assert k in s, (sub, k, list(s))
        assert 0.0 <= s["structured_acc"] <= 1.0 and 0.0 <= s["flat_acc"] <= 1.0
        assert s["n_test_heldout_pairs"] > 0  # held-out recombinations exist

    for sub in ("pooled", "dense"):
        fp, sp = out[sub]["flat_params"], out[sub]["structured_params"]
        assert 0.5 <= sp / fp <= 1.5, (sub, fp, sp)

    assert out["headline_2_vs_21_delta"] == out["dense_delta"] - out["pooled_delta"]

    assert isinstance(out["structured_beats_flat"], bool)
    assert isinstance(out["dense_gain_larger"], bool)
    assert out["null_rejected"] == (out["structured_beats_flat"] and out["dense_gain_larger"])

    assert (tmp_path / "e6_relational" / "e6_relational.png").exists()
    assert out["scientific_promotion"] is False
    assert out["claim_boundary"]["legacy_fixture"] is True


def test_e6_default_fails_closed_without_cache_pair(monkeypatch, tmp_path):
    cfg = config.compose(
        [
            "experiment=e6_relational",
            f"experiment.learned_cache={tmp_path / 'missing-learned'}",
            f"experiment.random_cache={tmp_path / 'missing-random'}",
        ]
    )

    def forbid_encoder_load(*args, **kwargs):
        raise AssertionError("cache-first E6 must not enter the generic encoder loader")

    monkeypatch.setattr("mop.experiments.e6_relational.load_encoder", forbid_encoder_load)
    run_dir = tmp_path / "cache-first"
    out = E6().run(cfg, devices.resolve("cpu"), run_dir)
    assert out["execution_status"] == "blocked-missing-or-invalid-cache-pair"
    assert out["scientific_promotion"] is False
    assert out["claim_boundary"]["legacy_fallback_used"] is False
    assert out["claim_boundary"]["model_loaded"] is False
    assert (run_dir / "e6_relational.json").is_file()
