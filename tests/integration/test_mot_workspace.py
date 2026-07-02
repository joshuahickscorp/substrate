"""WP-12 workspace-layer tests (WS1-WS4). Tiny synthetic dual-source caches only, no network, no
weights, no encoder loads; noisy-TV runs at reduced steps/dim via cfg overrides. Asserts MECHANICS,
guard wiring, the WS1 -> WS3 gate, and capmatch discipline, never a particular scientific outcome.
The capmatch_tol used at toy dims is looser than the preregistered 2 percent DEFAULT because parameter
granularity at tiny widths is coarse; the DEFAULTS entry (the preregistered value) is asserted intact."""

import json

import pytest
import torch
from scripts.cache_randominit_vitl_features import write_feature_cache
from scripts.mop_ws1_agreement_vs_confidence import (
    DEFAULTS as WS1_DEFAULTS,
)
from scripts.mop_ws1_agreement_vs_confidence import (
    _signals,
    load_dual_source,
    split_train_test,
)
from scripts.mop_ws1_agreement_vs_confidence import (
    run as ws1_run,
)
from scripts.mop_ws2_fusion_tournament import (
    ARM_MAKERS,
    CrossAttention,
    build_matched_arms,
)
from scripts.mop_ws2_fusion_tournament import (
    DEFAULTS as WS2_DEFAULTS,
)
from scripts.mop_ws2_fusion_tournament import (
    run as ws2_run,
)
from scripts.mop_ws3_arbitration import inverse_variance_fuse, routed_accuracy
from scripts.mop_ws3_arbitration import run as ws3_run
from scripts.mop_ws4_bandwidth_sweep import BottleneckFusion
from scripts.mop_ws4_bandwidth_sweep import run as ws4_run

from mop.diagnostics.compute import param_count

NTV_FAST = {"noisy_tv_steps": 60, "noisy_tv_dim": 16}


def _write_cache(root, x, y):
    write_feature_cache(root, features=x, labels_shape=y, meta={})
    return str(root)


def _dual_caches(tmp_path, n=96, da=24, db=20, n_classes=4, noise=0.6, seed=0):
    """Two genuinely different sources over the SAME clip population: each source sees the class
    prototypes through its own random projection plus independent noise, so neither is a linear remap
    of the other and their errors are partly complementary."""
    g = torch.Generator().manual_seed(seed)
    y = torch.arange(n) % n_classes
    proto_a = torch.randn(n_classes, da, generator=g)
    proto_b = torch.randn(n_classes, db, generator=g)
    za = proto_a[y] + noise * torch.randn(n, da, generator=g)
    zb = proto_b[y] + noise * torch.randn(n, db, generator=g)
    return {
        "cache_a": _write_cache(tmp_path / "a", za, y.tolist()),
        "cache_b": _write_cache(tmp_path / "b", zb, y.tolist()),
    }


# ---------------------------------------------------------------- dual-source loading contract


def test_load_dual_source_roundtrip_and_identity_checks(tmp_path):
    caches = _dual_caches(tmp_path)
    za, zb, y = load_dual_source(caches["cache_a"], caches["cache_b"])
    assert za.shape == (96, 24) and zb.shape == (96, 20) and len(y) == 96

    short = _write_cache(tmp_path / "short", torch.randn(4, 3), [0, 1, 0, 1])
    with pytest.raises(ValueError, match="clip counts"):
        load_dual_source(caches["cache_a"], short)

    relabeled = _write_cache(tmp_path / "relabel", torch.randn(96, 5), ((torch.arange(96) + 1) % 4).tolist())
    with pytest.raises(ValueError, match="clip identity"):
        load_dual_source(caches["cache_a"], relabeled)


def test_split_train_test_is_seeded_and_disjoint():
    tr, te = split_train_test(50, 0.3, seed=3)
    tr2, te2 = split_train_test(50, 0.3, seed=3)
    assert torch.equal(tr, tr2) and torch.equal(te, te2)
    assert len(tr) + len(te) == 50 and set(tr.tolist()).isdisjoint(te.tolist())


# ---------------------------------------------------------------- WS1 signals + end to end


def test_agreement_signal_math():
    pa = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y = torch.tensor([0, 1])
    same = _signals(pa, pa.clone(), y)
    assert torch.allclose(same["agreement"], torch.ones(2)), "identical one-hots agree with prob 1"
    disjoint = _signals(pa, pa.flip(-1), y)
    assert torch.allclose(disjoint["agreement"], torch.zeros(2)), "disjoint one-hots never coincide"
    assert same["best"] in ("a", "b") and same["correct"].tolist() == [1.0, 1.0]


def test_ws1_end_to_end_tiny(tmp_path):
    cfg = {**_dual_caches(tmp_path), "seeds": [0, 1], "epochs": 60, **NTV_FAST}
    out = ws1_run(cfg, "cpu", tmp_path / "run")
    assert isinstance(out["ws1_positive"], bool)
    assert out["null_supported"] == (not out["ws1_positive"])
    assert set(out["guards"]) == {"remap_pass", "shuffle_pass", "noisy_tv_pass"}
    assert len(out["per_seed"]) == 2
    for r in out["per_seed"]:
        assert set(r) >= {
            "auroc_confidence",
            "auroc_agreement",
            "delta",
            "aurc_confidence",
            "aurc_agreement",
            "err_corr",
            "shuffle_margin",
            "remap_margin",
            "noisy_tv_pass",
        }
        assert 0.0 <= r["auroc_agreement"] <= 1.0 and 0.0 <= r["auroc_confidence"] <= 1.0
    assert isinstance(out["non_ceiling"], bool)
    if out["ws1_positive"]:
        assert all(out["guards"].values()) and out["non_ceiling"]
    saved = json.loads((tmp_path / "run" / "result.json").read_text())
    assert saved["ws1_positive"] == out["ws1_positive"]
    assert "null_hypothesis" in out["contract"] and out["contract"]["tier"] == "cpu-now"


def test_ws1_preregistered_defaults_intact():
    assert WS1_DEFAULTS["ceiling"] == 0.97 and WS1_DEFAULTS["seeds"] == (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- WS2 capmatch discipline


def test_ws2_arms_are_param_matched_and_odd_widths_constructible():
    CrossAttention(6, 5, 3, 7)  # odd request rounds up internally, must never crash the search
    tol = 0.25  # toy-dims granularity; the preregistered tol lives in DEFAULTS
    arms = build_matched_arms(24, 20, 4, 16, tol)
    assert set(arms) == {"concat_mlp"} | set(ARM_MAKERS)
    ref = arms["concat_mlp"]["params"]
    for name, spec in arms.items():
        assert abs(spec["params"] - ref) <= tol * ref, name
        assert param_count(spec["make"]()) == spec["params"]
    with pytest.raises(ValueError, match="capmatch failed"):
        build_matched_arms(6, 5, 3, 16, 1e-6)  # a capmatch miss must abort, never silently compare
    assert WS2_DEFAULTS["capmatch_tol"] == 0.02


def test_ws2_end_to_end_tiny(tmp_path):
    cfg = {
        **_dual_caches(tmp_path),
        "seeds": [0, 1],
        "epochs": 40,
        "width": 16,
        "capmatch_tol": 0.25,
    }
    out = ws2_run(cfg, "cpu", tmp_path / "run")
    assert isinstance(out["null_supported"], bool)
    assert set(out["arms"]) == set(ARM_MAKERS)
    for rep in out["arms"].values():
        assert set(rep) >= {"width", "params", "acc_delta_ci", "nll_delta_ci", "acc_win", "nll_win"}
    for r in out["per_seed"]:
        assert set(r["scores"]) == {"concat_mlp"} | set(ARM_MAKERS)
        for sc in r["scores"].values():
            assert 0.0 <= sc["acc"] <= 1.0 and sc["nll"] >= 0.0
    assert (tmp_path / "run" / "result.json").exists()


# ---------------------------------------------------------------- WS3 gate + arbitration mechanics


def test_inverse_variance_fuse_prefers_the_confident_source():
    pa = torch.tensor([[0.98, 0.02]])  # near-zero entropy
    pb = torch.tensor([[0.5, 0.5]])  # max entropy
    fused = inverse_variance_fuse(pa, pb, eps=0.05)
    assert torch.allclose(fused.sum(-1), torch.ones(1), atol=1e-5)
    assert float(fused[0, 0]) > 0.75, "the low-entropy source must dominate the weights"


def test_routed_accuracy_mechanics():
    y = torch.tensor([0, 0, 1, 1])
    fused = torch.nn.functional.one_hot(1 - y, 2).float()  # always wrong
    expensive = torch.nn.functional.one_hot(y, 2).float()  # always right
    all_on = torch.ones(4, dtype=torch.bool)
    assert routed_accuracy(fused, expensive, all_on, y) == 1.0
    assert routed_accuracy(fused, expensive, ~all_on, y) == 0.0
    half = torch.tensor([True, True, False, False])
    assert routed_accuracy(fused, expensive, half, y) == 0.5


def test_ws3_skips_without_a_positive_ws1(tmp_path):
    cfg = {"ws1_verdict": str(tmp_path / "missing.json")}
    out = ws3_run(cfg, "cpu", tmp_path / "run")
    assert out["skipped"] is True and out["null_supported"] is True
    assert "SKIPPED" in out["verdict"]

    neg = tmp_path / "ws1_neg.json"
    neg.write_text(json.dumps({"ws1_positive": False, "verdict": "NULL SUPPORTED: test"}))
    out2 = ws3_run({"ws1_verdict": str(neg)}, "cpu", None)
    assert out2["skipped"] is True and out2["ws1_verdict"] == "NULL SUPPORTED: test"


def test_ws3_runs_when_the_gate_is_open(tmp_path):
    gate = tmp_path / "ws1.json"
    gate.write_text(json.dumps({"ws1_positive": True}))
    cfg = {
        **_dual_caches(tmp_path),
        "ws1_verdict": str(gate),
        "seeds": [0, 1],
        "epochs": 40,
        **NTV_FAST,
    }
    out = ws3_run(cfg, "cpu", tmp_path / "run")
    assert out["skipped"] is False
    assert isinstance(out["null_supported"], bool)
    assert out["null_supported"] == (not (out["invvar_win"] and out["routing_win"] and out["noisy_tv_pass"]))
    for r in out["per_seed"]:
        assert set(r["acc"]) == {
            "single_best",
            "equal_weight",
            "inverse_variance",
            "concat_mlp",
            "disagreement_routed",
            "random_routed",
        }
    assert (tmp_path / "run" / "result.json").exists()


# ---------------------------------------------------------------- WS4 bottleneck sweep


def test_bottleneck_fusion_module_mechanics():
    widths = [4, 8, 16]
    counts = [param_count(BottleneckFusion(10, 8, 3, w, 4)) for w in widths]
    assert counts == sorted(counts), "param count must be monotone in width for the capmatch solver"
    za, zb = torch.randn(5, 10), torch.randn(5, 8)
    assert BottleneckFusion(10, 8, 3, 8, 4)(za, zb).shape == (5, 3)
    wo = BottleneckFusion(10, 8, 3, 8, 4, write_only=True)
    assert wo(za, zb).shape == (5, 3)
    assert param_count(wo) < param_count(BottleneckFusion(10, 8, 3, 8, 4)), "write-only drops broadcast"


def test_ws4_end_to_end_tiny(tmp_path):
    cfg = {
        **_dual_caches(tmp_path),
        "seeds": [0, 1],
        "epochs": 30,
        "width": 8,
        "slots": (2, 8),
        "dropout_grid": (0.2,),
        "weight_decay_grid": (1e-3,),
        "capmatch_tol": 0.25,
    }
    out = ws4_run(cfg, "cpu", tmp_path / "run")
    assert isinstance(out["null_supported"], bool)
    assert set(out["sweep"]) == {2, 8}
    for spec in out["sweep"].values():
        assert abs(spec["params"] - out["total_params"]) <= 0.25 * out["total_params"]
    for r in out["per_seed"]:
        assert set(r) >= {
            "best_slot",
            "best_slot_acc",
            "write_only_acc",
            "unbottlenecked_acc",
            "best_reg_acc",
            "bandwidth_benefit",
            "reg_delta",
            "broadcast_minus_write_only",
            "interior_optimum",
        }
        assert r["best_slot"] in (2, 8)
    assert isinstance(out["interior_majority"], bool)
    for key in ("bandwidth_benefit_ci", "reg_delta_ci", "broadcast_minus_write_only_ci"):
        assert set(out[key]) >= {"n", "mean", "lo", "hi", "unstable"}
    assert (tmp_path / "run" / "result.json").exists()
