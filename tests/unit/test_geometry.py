import torch

from mop.diagnostics import geometry as G


def _rng(seed=0):
    return torch.Generator().manual_seed(seed)


def test_linear_cka_self_is_one():
    x = torch.randn(100, 16, generator=_rng())
    assert abs(G.linear_cka(x, x) - 1.0) < 1e-5


def test_linear_cka_rotation_invariant():
    x = torch.randn(100, 16, generator=_rng())
    q, _ = torch.linalg.qr(torch.randn(16, 16, generator=_rng(1)))
    assert G.linear_cka(x, x @ q) > 0.99  # CKA is invariant to orthogonal transforms


def test_linear_cka_independent_is_low():
    x = torch.randn(200, 16, generator=_rng(1))
    y = torch.randn(200, 16, generator=_rng(2))
    assert G.linear_cka(x, y) < 0.5


def test_kernel_cka_self_is_one():
    x = torch.randn(80, 8, generator=_rng())
    assert G.kernel_cka(x, x) > 0.99


def test_rsa_self_is_one():
    x = torch.randn(60, 8, generator=_rng())
    assert G.rsa(x, x) > 0.99


def test_effective_rank_isotropic_near_dim():
    x = torch.randn(4000, 16, generator=_rng())
    assert G.effective_rank(x) > 14  # isotropic 16d -> effective rank near 16


def test_effective_rank_rank_one_is_one():
    x = torch.randn(120, 1, generator=_rng()) @ torch.randn(1, 16, generator=_rng(1))
    assert G.effective_rank(x) < 1.5


def test_anisotropy_rank_one_is_high():
    x = torch.randn(120, 1, generator=_rng()) @ torch.randn(1, 16, generator=_rng(1))
    assert G.anisotropy(x) > 0.9  # one direction dominates


def test_neighborhood_overlap_self_is_one():
    x = torch.randn(50, 8, generator=_rng())
    assert G.neighborhood_overlap(x, x, k=5) > 0.99


def test_geometry_report_keys():
    x = torch.randn(80, 12, generator=_rng())
    rep = G.geometry_report(x, reference=x)
    for k in (
        "effective_rank",
        "anisotropy",
        "intrinsic_dim",
        "linear_cka",
        "kernel_cka",
        "rsa",
        "neighborhood_overlap",
    ):
        assert k in rep
    assert rep["linear_cka"] > 0.99  # self-reference
