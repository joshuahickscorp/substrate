"""IterativeRefiner (EX17 primitive): shape preservation, determinism, fixed-N step accounting,
adaptive halting never exceeds the budget and stays weight-tied (fewer params than untied depth)."""

import torch

from mop.diagnostics.compute import param_count
from mop.seeding import seed_everything
from mop.shell.refine import IterativeRefiner


def test_refiner_preserves_shape():
    r = IterativeRefiner(dim=16, hidden=32, steps=4)
    z = torch.randn(8, 16)
    out, used = r(z)
    assert out.shape == z.shape
    assert float(used.float().mean()) == 4.0  # fixed-N uses every step


def test_refiner_deterministic():
    seed_everything(0)
    r1 = IterativeRefiner(16, 32, 3)
    seed_everything(0)
    r2 = IterativeRefiner(16, 32, 3)
    z = torch.randn(4, 16, generator=torch.Generator().manual_seed(1))
    o1, _ = r1(z)
    o2, _ = r2(z)
    assert torch.allclose(o1, o2)


def test_weight_tying_is_cheaper_than_untied_depth():
    from mop.experiments.ex17_latent_reasoning import _UntiedDepth

    tied = IterativeRefiner(32, 64, steps=4)
    untied = _UntiedDepth(32, 64, steps=4)
    # the tied refiner reuses one block across steps -> far fewer params at matched FLOPs
    assert param_count(tied) < param_count(untied)


def test_adaptive_halting_never_exceeds_budget():
    r = IterativeRefiner(16, 32, steps=6, halt=True, halt_threshold=0.5)
    z = torch.randn(10, 16, generator=torch.Generator().manual_seed(2))
    _, used = r(z)
    assert used.max().item() <= 6
    assert used.min().item() >= 1


def test_block_count():
    r = IterativeRefiner(16, 32, steps=5)
    assert r.block_count() == 5
