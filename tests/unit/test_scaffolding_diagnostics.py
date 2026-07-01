"""Known-answer tests for the four remaining cpu-now scaffolding rows completed after the roots-
expansion build: difficulty calibration (D3), transfer matrix (D4), replay-buffer compression (A3),
and latent robustness (A4). These extract/generalize patterns already proven inline (I5's quantize-on-
store loop, the linear_probe-based comparisons), so the tests check the reusable module directly."""

import torch

from devsys.diagnostics import (
    calibrated_tie,
    degradation_curve,
    reference_separation,
    retention_per_byte,
    transfer_matrix,
)
from devsys.substrate.datasets import make_task_stream


def _g(s=0):
    return torch.Generator().manual_seed(s)


def _separable(n=120, dim=16, k=3, sep=4.0, seed=0):
    g = _g(seed)
    y = torch.randint(0, k, (n,), generator=g)
    centers = torch.randn(k, dim, generator=g) * sep
    x = centers[y] + 0.3 * torch.randn(n, dim, generator=g)
    return x, y


def test_reference_separation_certifies_easy_regime():
    x, y = _separable(sep=6.0)
    r = reference_separation(x, y, seed=0)
    assert r["regime_calibrated"] is True
    assert r["reference_score"] > r["chance"]


def test_reference_separation_flags_uncalibrated_regime():
    x, y = _separable(dim=16, sep=0.0)
    x = torch.randn_like(x)  # pure noise, no class structure at all
    r = reference_separation(x, y, seed=0)
    assert r["regime_calibrated"] is False


def test_calibrated_tie_only_meaningful_when_calibrated():
    x, y = _separable(sep=6.0)
    ct = calibrated_tie(x, y, score_a=0.5, score_b=0.51, seed=0)
    assert ct["is_tie"] is True
    assert ct["tie_is_meaningful"] is True


def test_transfer_matrix_shape_and_diagonal_high():
    tasks = make_task_stream(
        n_tasks=3, dim=16, classes_per_task=3, samples_per_task=60, separation=5.0, seed=0
    )
    pairs = [(t.x, t.y) for t in tasks]
    rep = transfer_matrix(pairs, epochs=120, seed=0)
    grid = rep["transfer_matrix"]
    assert len(grid) == 3 and all(len(row) == 3 for row in grid)
    # a head trained and evaluated on its own separable task should decode far above chance
    assert rep["diag_mean"] > rep["chance"] + 0.2


def test_retention_per_byte_returns_frontier_shape():
    tasks = make_task_stream(
        n_tasks=2,
        dim=16,
        classes_per_task=3,
        samples_per_task=40,
        separation=4.0,
        incremental="domain",
        seed=0,
    )
    rep = retention_per_byte(tasks, dim=16, n_classes=3, bits=(32, 4), epochs=15, seed=0)
    assert set(rep["backward_transfer"]) == {32, 4}
    assert rep["bytes_per_exemplar"][32] > rep["bytes_per_exemplar"][4]
    assert isinstance(rep["frontier_present"], bool)


def test_degradation_curve_monotone_under_heavy_noise():
    x, y = _separable(sep=6.0)
    rep = degradation_curve(x, y, noise_levels=(0.0, 3.0), dropout_levels=(0.0,), bits=(32,), seed=0)
    assert rep["noise_curve"][3.0] <= rep["base_accuracy"] + 1e-6
    assert "shuffled_feature_floor" in rep
