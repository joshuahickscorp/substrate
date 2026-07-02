"""WP-06 memory and retrieval scripts (DR10 retrieve-then-reason, PR8 retrieval head, PR7 fast/slow)
run end to end on tiny synthetic tensors and return the Experiment-contract dict. Asserts MECHANICS
only (shapes, keys, matched capacity, verdict plumbing), never a scientific outcome: the
preregistered nulls holding is an honest finding. No encoder, no weights, seconds per test."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from devsys.devices import resolve

torch.set_num_threads(2)  # live-state rule: an encode may own the CPU

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
TINY_UNIVERSE = dict(
    seeds=[0, 1],
    d_signal=8,
    d_nuisance=8,
    n_base=4,
    n_new=2,
    mem_per_class=10,
    k_shot=3,
    train_per_class=15,
    test_per_class=10,
    k=5,
    hidden=16,
    epochs=2,
    batch=32,
)


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dev():
    return resolve("cpu")


def test_dr10_retrieve_reason_runs(dev, tmp_path):
    mod = load_script("mot_dr10_retrieve_reason")
    cfg = mod.default_cfg(**TINY_UNIVERSE, refine_steps=1)
    out = mod.MotDR10RetrieveReason().run(cfg, dev, tmp_path)
    assert isinstance(out, dict) and isinstance(out["null_supported"], bool)
    for arm in mod.ARMS:
        assert len(out["new_class_acc"][arm]) == 2
        for acc in out["new_class_acc"][arm]:
            assert 0.0 <= acc <= 1.0
    assert out["delta_retrieval_vs_best_control"]["ci"]["n"] == 2
    assert out["knn_flops_per_query"] > 0
    # the shots curve spans 1..k_shot and adaptation_speed reads it
    assert len(out["adaptation_shots_curve_mean"]) == TINY_UNIVERSE["k_shot"]
    assert "steps" in out["adaptation_speed_shots"]
    # verdict plumbing: win in the delta report is the negation of null_supported
    assert out["null_supported"] == (not out["delta_retrieval_vs_best_control"]["win"])


def test_dr10_declares_contract():
    mod = load_script("mot_dr10_retrieve_reason")
    exp = mod.MotDR10RetrieveReason()
    assert exp.null_hypothesis and exp.metric and exp.baseline and exp.ablation
    assert exp.tier == "cpu-now"


def test_pr8_retrieval_head_runs_and_matches_params(dev, tmp_path):
    mod = load_script("mot_pr8_retrieval_head")
    cfg = mod.default_cfg(**TINY_UNIVERSE, attn_dim=8, capmatch_tol=0.10)
    out = mod.MotPR8RetrievalHead().run(cfg, dev, tmp_path)
    assert isinstance(out["null_supported"], bool)
    pc = out["param_counts"]
    # the doctrine capacity match: parametric arm within tol of the retrieval head
    assert pc["match_rel_err"] <= 0.10
    for arm in mod.ARMS:
        assert len(out["new_class_acc"][arm]) == 2
    # a win requires BOTH deltas to pass; null_supported is the negation
    both = out["delta_vs_knn"]["win"] and out["delta_vs_parametric"]["win"]
    assert out["null_supported"] == (not both)


def test_pr8_declares_contract():
    mod = load_script("mot_pr8_retrieval_head")
    exp = mod.MotPR8RetrievalHead()
    assert "kNN" in exp.null_hypothesis and exp.tier == "cpu-now"


def test_pr7_fast_slow_runs(dev, tmp_path):
    mod = load_script("mot_pr7_fast_slow")
    cfg = mod.default_cfg(seeds=[0, 1], dim=16, classes_per_task=4, n_tasks=2, samples_per_task=80, batch=8)
    out = mod.MotPR7FastSlow().run(cfg, dev, tmp_path)
    assert isinstance(out["null_supported"], bool)
    for arm in mod.ARMS:
        assert len(out["online_acc"][arm]["per_seed"]) == 2
        for acc in out["online_acc"][arm]["per_seed"]:
            assert 0.0 <= acc <= 1.0
        assert len(out["post_decay_retention"][arm]["per_seed"]) == 2
    assert out["retention_delta_vs_slow_only"]["margin"] == mod.RETENTION_MARGIN
    win = out["delta_online_vs_best_control"]["win"] and out["retention_delta_vs_slow_only"]["ok"]
    assert out["null_supported"] == (not win)


def test_pr7_fast_store_mechanics():
    mod = load_script("mot_pr7_fast_slow")
    store = mod.HebbianFastStore(dim=8, n_classes=3, eta=0.5, decay=0.9)
    z = torch.randn(4, 8)
    y = torch.tensor([0, 1, 2, 0])
    assert store.read(z).shape == (4, 3)
    assert float(store.read(z).abs().sum()) == 0.0  # empty store reads zero
    store.write(z, y)
    assert float(store.a.abs().sum()) > 0.0
    store.reset()
    assert float(store.a.abs().sum()) == 0.0
    cache = mod.EpisodicCache(dim=8, n_classes=3, k=2)
    # matched float budget: capacity = floor(C*D / (D+1))
    assert cache.capacity == (3 * 8) // 9
    assert cache.read(z).shape == (4, 3)
    cache.write(z, y)
    assert len(cache.x) <= cache.capacity
