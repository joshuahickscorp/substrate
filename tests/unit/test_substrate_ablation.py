"""Substrate-ablation control (D2): frozen-random projection preserves linear decodability (the
'any projection' truth), shuffled pairing collapses it (the chance floor), quantize-dequantize is a
bounded round-trip, and the report exposes the deltas vs real."""

import torch

from devsys.diagnostics import linear_probe
from devsys.diagnostics.substrate_ablation import (
    frozen_random_projection,
    quantize_dequantize,
    shuffled_pairing,
    substrate_ablation,
)


def _rng(seed=0):
    return torch.Generator().manual_seed(seed)


def test_frozen_random_projection_preserves_dim_and_linear_info():
    x = torch.randn(200, 16, generator=_rng())
    y = (x[:, 0] > 0).long()
    xp = frozen_random_projection(x, seed=0)
    assert xp.shape == x.shape
    # a square random linear map preserves linear separability (decodability is projection-invariant)
    assert linear_probe(xp, y)["score"] > 0.8


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
    for k in ("real", "frozen_random", "shuffled", "compressed", "delta_shuffled", "needs_real"):
        assert k in out
    # real decodability must beat the shuffled chance floor (the target is genuinely there)
    assert out["delta_shuffled"] > 0.1
