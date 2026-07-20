
import torch

from mop.diagnostics import linear_probe
from mop.diagnostics.substrate_ablation import (
    frozen_random_projection,
    quantize_dequantize,
    rank_reduced_projection,
    shuffled_pairing,
    substrate_ablation,
)


def _rng(seed=0):
    return torch.Generator().manual_seed(seed)


def test_frozen_random_projection_is_vacuous_for_a_linear_metric():
    x = torch.randn(200, 16, generator=_rng())
    y = (x[:, 0] > 0).long()
    xp = frozen_random_projection(x, seed=0)
    assert xp.shape == x.shape
    assert linear_probe(xp, y)["score"] > 0.8


def test_rank_reduced_projection_can_lose_linear_information():
    x = torch.randn(300, 32, generator=_rng())
    w = torch.randn(32, generator=_rng(1))
    y = ((x @ w) > 0).long()
    full = linear_probe(frozen_random_projection(x, seed=0), y)["score"]
    reduced = linear_probe(rank_reduced_projection(x, rank=1, seed=0), y)["score"]
    assert reduced < full  # the rank-1 bottleneck loses information the full-rank map preserves


def test_shuffled_pairing_destroys_decodability():
    x = torch.randn(200, 16, generator=_rng())
    y = (x[:, 0] > 0).long()
    sx, sy = shuffled_pairing(x, y, seed=0)
    assert linear_probe(sx, sy)["score"] < 0.7  # broken correspondence -> chance-ish


def test_quantize_dequantize_is_bounded_roundtrip():
    x = torch.randn(100, 8, generator=_rng())
    q = quantize_dequantize(x, bits=4)
    assert q.shape == x.shape
    assert (q.max() <= x.max() + 1e-4) and (q.min() >= x.min() - 1e-4)


def test_substrate_ablation_report_keys():
    x = torch.randn(200, 16, generator=_rng())
    y = (x[:, 0] > 0).long()
    out = substrate_ablation(x, y)
    for k in (
        "real",
        "frozen_random",
        "rank_reduced",
        "shuffled",
        "compressed",
        "delta_shuffled",
        "delta_frozen_random",
        "needs_real",
        "beats_frozen_random",
    ):
        assert k in out
    assert out["delta_shuffled"] > 0.1


def test_needs_real_gates_on_the_shuffled_floor_not_frozen_random():
    g = _rng()
    centers = torch.randn(2, 16, generator=g) * 4.0  # well separated so decodability saturates at 1.0
    y = torch.randint(0, 2, (240,), generator=g)
    x = centers[y] + 0.3 * torch.randn(240, 16, generator=g)
    out = substrate_ablation(x, y)
    assert out["needs_real"] is True  # decodable above the shuffled floor
    assert abs(out["delta_frozen_random"]) < 0.05  # ties frozen_random (invertible map, both saturate)
    assert out["beats_frozen_random"] is False  # so beating frozen_random is NOT required for needs_real


def test_needs_real_is_false_when_target_is_pure_noise():
    x = torch.randn(240, 16, generator=_rng())
    y = torch.randint(0, 2, (240,), generator=_rng(7))  # independent of x
    out = substrate_ablation(x, y)
    assert out["needs_real"] is False
