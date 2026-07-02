import torch

from mop import devices
from mop.diagnostics import (
    assert_reproducible,
    critical_period_signature,
    determinism_loop,
    fisher_trace_over_training,
    linear_probe,
    noisy_tv_diagnostic,
    reliability,
)
from mop.substrate.datasets import make_task_stream


# ---- linear probe ----
def test_linear_probe_decodes_separable_classes():
    tasks = make_task_stream(
        n_tasks=1, dim=32, classes_per_task=4, samples_per_task=400, separation=3.0, seed=0
    )
    r = linear_probe(tasks[0].x, tasks[0].y, classification=True, epochs=150)
    assert r["decodable"] and r["score"] > r["chance"] + 0.2


def test_linear_probe_flags_undecodable_noise():
    x = torch.randn(400, 32)
    y = torch.randint(0, 4, (400,))  # labels independent of x -> not decodable
    r = linear_probe(x, y, classification=True, epochs=150)
    assert not r["decodable"]


def test_linear_probe_regression_r2():
    x = torch.randn(300, 16)
    w = torch.randn(16, 1)
    y = x @ w  # perfectly linear
    r = linear_probe(x, y, classification=False, epochs=300)
    assert r["metric"] == "r2" and r["score"] > 0.9


# ---- noisy-TV ----
def test_noisy_tv_epistemic_collapses_but_error_stays_high():
    out = noisy_tv_diagnostic(dim=48, device=devices.resolve("cpu"), steps=250, seed=0)
    assert out["noise_error_stays_high"], out
    assert out["epistemic_collapses_on_noise"], out
    assert out["learning_progress_separates"], out


# ---- calibration ----
def test_reliability_perfectly_calibrated_low_ece():
    # confidence c == P(correct): predicted class is 1 with confidence c in [0.5,1], and the
    # sample is correct with probability c, so per-bin accuracy ~ confidence -> ECE ~ 0.
    torch.manual_seed(0)
    n = 6000
    c = 0.5 + 0.5 * torch.rand(n)
    probs = torch.stack([1 - c, c], dim=1)  # argmax is class 1, confidence c
    correct = torch.rand(n) < c
    labels = torch.where(correct, torch.ones(n, dtype=torch.long), torch.zeros(n, dtype=torch.long))
    rel = reliability(probs, labels, bins=10)
    assert rel["ece"] < 0.1


def test_reliability_overconfident_high_ece():
    n = 1000
    probs = torch.stack([torch.zeros(n) + 0.01, torch.zeros(n) + 0.99], dim=1)  # always 99% conf
    labels = torch.randint(0, 2, (n,))  # but 50% accuracy
    rel = reliability(probs, labels, bins=10)
    assert rel["ece"] > 0.3


# ---- fisher trace ----
def test_fisher_trace_curve_nonneg_and_signature():
    data = [(torch.randn(16, 8), torch.randint(0, 2, (16,))) for _ in range(6)]

    def mk():
        return torch.nn.Linear(8, 2)

    def lf_factory(model):
        return lambda b: torch.nn.functional.cross_entropy(model(b[0]), b[1])

    trace = fisher_trace_over_training(mk, data, lf_factory, checkpoints=6, steps_per_ckpt=8)
    assert len(trace) == 6 and all(t >= 0 for t in trace)
    sig = critical_period_signature(trace)
    assert "peak_index" in sig


# ---- determinism ----
def test_determinism_loop_cpu_bit_identical():
    def fn():
        return torch.randn(64) @ torch.randn(64)

    rep = determinism_loop(fn, runs=5, seed=3)
    assert rep.bit_identical and rep.tol() > 0


def test_assert_reproducible_passes_on_cpu():
    rep = assert_reproducible(lambda: torch.randn(32).sum(), runs=4, seed=1)
    assert rep.runs == 4


def test_assert_reproducible_can_fail():
    # a fn whose output drifts despite reseeding must trip the absolute-tolerance assertion
    import pytest

    state = {"n": 0.0}

    def drifting():
        state["n"] += 1.0
        return torch.tensor(state["n"])

    with pytest.raises(AssertionError):
        assert_reproducible(drifting, runs=4, seed=0, tol=1e-4)
