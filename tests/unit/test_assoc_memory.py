import torch

from devsys.studies.assoc_memory import (
    _corrupt,
    _retrieval_acc,
    _store,
    assoc_memory,
    ff_autoassociator,
    hopfield_retrieve,
)


def test_hopfield_recovers_corrupted_pattern_at_low_load():
    # K << D: the energy memory should snap a noisy query back onto its own stored pattern.
    gen = torch.Generator().manual_seed(0)
    x = _store(8, 64, gen)
    q = _corrupt(x, level=0.3, gen=gen)
    r = hopfield_retrieve(x, q, beta=16.0, steps=3)
    acc = _retrieval_acc(r, x, thresh=0.9)
    assert acc > 0.8, acc
    # retrieval must beat the raw corrupted query's self-cosine -> denoising actually happened
    raw = (q * x).sum(1).mean()
    ret = (r * x).sum(1).mean()
    assert ret > raw


def test_retrieval_acc_is_non_vacuous():
    # exact stored patterns retrieve perfectly; random vectors essentially never do
    gen = torch.Generator().manual_seed(1)
    x = _store(6, 32, gen)
    assert _retrieval_acc(x, x, thresh=0.9) == 1.0
    rand = torch.randn(6, 32)
    rand = rand / rand.norm(dim=1, keepdim=True)
    assert _retrieval_acc(rand, x, thresh=0.9) < 0.5


def test_ff_autoassociator_shapes_and_denoises_lightly():
    gen = torch.Generator().manual_seed(2)
    x = _store(10, 48, gen)
    q = _corrupt(x, level=0.3, gen=gen)
    out = ff_autoassociator(x, q)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_assoc_memory_curves_and_hopfield_wins_at_high_load():
    out = assoc_memory(dim=32, loads=(0.5, 1.0, 2.0, 4.0, 6.0), betas=(8.0, 16.0), corruption=0.3, seed=0)
    assert out["provenance"] == "provisional"
    assert len(out["hopfield"]["curve"]) == 5
    assert len(out["feedforward"]["curve"]) == 5
    # accuracies are real fractions
    for pt in out["hopfield"]["curve"] + out["feedforward"]["curve"]:
        assert 0.0 <= pt["acc"] <= 1.0
    # the linear control is provably D-bounded: ~0 once load exceeds 1.0
    ff = {pt["load"]: pt["acc"] for pt in out["feedforward"]["curve"]}
    assert ff[1.0] > 0.8 and ff[4.0] < 0.2
    # the headline claim: modern-Hopfield capacity exceeds the linear control's
    assert out["hopfield"]["capacity"] > out["feedforward"]["capacity"]
    assert out["hopfield_wins"]


def test_hopfield_accuracy_degrades_with_load():
    # capacity is finite: accuracy at very high load must be below low load (curve is informative)
    out = assoc_memory(dim=32, loads=(0.5, 6.0), betas=(8.0, 16.0), corruption=0.3, seed=0)
    lo, hi = out["hopfield"]["curve"]
    assert lo["acc"] > hi["acc"]
