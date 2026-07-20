
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from mop.devices import resolve

torch.set_num_threads(2)  # live-state rule: an encode may own the CPU

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
TINY_STREAM = dict(
    seeds=[0, 1],
    dim=16,
    classes_per_task=4,
    n_tasks=3,
    samples_per_task=80,
    batch=8,
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


def test_pr4_epistemic_gate_runs(dev, tmp_path):
    mod = load_script("mop_pr4_epistemic_gate")
    cfg = mod.default_cfg(
        seeds=[0, 1], dim=8, hidden=16, ensemble_size=3, batch=16, n_steps=20, guard_dim=16, guard_steps=40
    )
    out = mod.MotPR4EpistemicGate().run(cfg, dev, tmp_path)
    assert isinstance(out, dict) and isinstance(out["null_supported"], bool)
    for arm in mod.ARMS:
        fr = out["reducible_lr_fraction"][arm]["per_seed"]
        assert len(fr) == 2
        for f in fr:
            assert 0.0 <= f <= 1.0
    assert len(out["lr_integrals_per_seed"]) == 2
    for row in out["lr_integrals_per_seed"]:
        assert set(row) == {"gated", "ungated"}
        for parts in row.values():
            assert set(parts) == {"reducible", "noise"}
    guard = out["noisy_tv_guard"]
    for key in ("noise_error_stays_high", "epistemic_collapses_on_noise", "learning_progress_separates"):
        assert isinstance(guard[key], bool)
    assert guard["pass"] == (
        guard["noise_error_stays_high"]
        and guard["epistemic_collapses_on_noise"]
        and guard["learning_progress_separates"]
    )
    win = (
        out["delta_vs_ungated"]["win"]
        and out["delta_vs_shuffled"]["win"]
        and out["curriculum_permutation_sign_ok"]
        and guard["pass"]
    )
    assert out["null_supported"] == (not win)


def test_pr4_declares_contract():
    mod = load_script("mop_pr4_epistemic_gate")
    exp = mod.MotPR4EpistemicGate()
    assert exp.null_hypothesis and exp.metric and exp.baseline and exp.ablation
    assert "e4" in exp.null_hypothesis and exp.tier == "cpu-now"


def test_pr5_content_gated_runs(dev, tmp_path):
    mod = load_script("mop_pr5_content_gated_cp")
    cfg = mod.default_cfg(**TINY_STREAM, steps_per_task=20, eval_every=5)
    out = mod.MotPR5ContentGatedCP().run(cfg, dev, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert isinstance(out["lr_integral_matched"], bool)
    assert len(out["lr_integral_matched_per_seed"]) == 2
    for arm in mod.ARMS:
        assert len(out["bwt"][arm]["per_seed"]) == 2
        assert len(out["fwt"][arm]["per_seed"]) == 2
        assert len(out["adaptation_steps_later_tasks"][arm]["per_seed"]) == 2
    win = out["lr_integral_matched"] and (
        out["retention_delta"]["win"] or out["reopening_delta_steps"]["win"]
    )
    assert out["null_supported"] == (not win)


def test_pr5_matched_integral_solver(dev):
    mod = load_script("mop_pr5_content_gated_cp")
    e = mod.default_cfg(**TINY_STREAM, steps_per_task=10, eval_every=5).experiment
    order = list(range(int(e.n_tasks)))
    content = mod.run_arm(e, 0, "content", order, budget=None)
    budget = content["lrint"].total()
    assert budget > 0.0
    for arm in ("cosine", "constant"):
        res = mod.run_arm(e, 0, arm, order, budget=budget)
        assert content["lrint"].matched(res["lrint"])


def test_pr5_declares_contract():
    mod = load_script("mop_pr5_content_gated_cp")
    exp = mod.MotPR5ContentGatedCP()
    assert "matched LR-integral" in exp.null_hypothesis and exp.tier == "cpu-now"


def test_pr6_sleep_consolidation_runs(dev, tmp_path):
    mod = load_script("mop_pr6_sleep_consolidation")
    cfg = mod.default_cfg(**TINY_STREAM, wake_steps=10, sleep_steps=10, buffer_capacity=100, ewc_samples=2)
    out = mod.MotPR6SleepConsolidation().run(cfg, dev, tmp_path)
    assert isinstance(out["null_supported"], bool)
    assert out["steps_matched"] is True
    expected = TINY_STREAM["n_tasks"] * (10 + 10)
    for arm in mod.ARMS:
        assert out["steps_per_arm"][arm] == [expected, expected]
        for acc in out["final_mean_acc"][arm]["per_seed"]:
            assert 0.0 <= acc <= 1.0
        assert len(out["bwt"][arm]["per_seed"]) == 2
        assert len(out["forgetting_area_task0"][arm]["per_seed"]) == 2
    win = out["steps_matched"] and (out["delta_final_acc"]["win"] or out["delta_bwt"]["win"])
    assert out["null_supported"] == (not win)


def test_pr6_declares_contract():
    mod = load_script("mop_pr6_sleep_consolidation")
    exp = mod.MotPR6SleepConsolidation()
    assert "matched total gradient steps" in exp.null_hypothesis and exp.tier == "cpu-now"
