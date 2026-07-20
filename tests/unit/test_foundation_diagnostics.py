
import numpy as np
import torch

from mop.diagnostics import (
    basin_stability,
    capability_per_bit,
    code_stability,
    compositionality_report,
    convergence_report,
    cross_seed_cka,
    readout_contribution,
    sysid_report,
)
from mop.diagnostics.seed_consistency import _hungarian
from mop.shell.refine import IterativeRefiner, Verifier


def _g(s=0):
    return torch.Generator().manual_seed(s)


def test_convergence_report_classifies():
    r = IterativeRefiner(16, 32, steps=4)
    z = torch.randn(8, 16, generator=_g())
    rep = convergence_report(r, z, steps=24)
    assert rep["classification"] in ("converges", "drifts", "limit_cycle")
    assert "contraction_factor" in rep and len(rep["update_norms"]) == 24


def test_basin_stability_keys():
    r = IterativeRefiner(16, 32, steps=4)
    z = torch.randn(8, 16, generator=_g())
    bs = basin_stability(r, z, eps=0.1, steps=24)
    assert "contraction_ratio" in bs and isinstance(bs["stable"], bool)


def test_refiner_unroll_returns_trajectory():
    r = IterativeRefiner(16, 32, steps=4)
    z = torch.randn(4, 16, generator=_g())
    zf, norms = r.unroll(z, 10)
    assert zf.shape == z.shape and len(norms) == 10


def test_refiner_predictive_coding_mode_runs():
    r = IterativeRefiner(16, 32, steps=3, mode="predictive_coding")
    z = torch.randn(4, 16, generator=_g())
    out, used = r(z)
    assert out.shape == z.shape and float(used.float().mean()) == 3.0


def test_verifier_scores():
    v = Verifier(16, 32)
    z = torch.randn(5, 16, generator=_g())
    assert v.score(z).shape == (5,)


def test_readout_contribution_on_nonlinear_target():
    g = _g()
    x = torch.randn(300, 16, generator=g)
    y = ((x[:, 0] > 0) ^ (x[:, 1] > 0)).long()  # XOR: nonlinearly decodable
    rc = readout_contribution(x, y, seed=0)
    assert rc["nonlinear_real"] >= rc["linear_real"]  # nonlinear probe reads the XOR the linear cannot
    assert "readout_contribution_index" in rc


def test_compositionality_additive_vs_entangled():
    add = compositionality_report(interaction=0.0, seed=0)
    ent = compositionality_report(interaction=4.0, seed=0)
    assert add["real"]["compositional"] is True  # additive factors decode in held-out combinations
    assert add["real"]["heldout_acc"] > ent["real"]["heldout_acc"]  # entanglement breaks held-out decoding


def test_cross_seed_cka_discriminates():
    same = [torch.randn(40, 8, generator=_g(1))] * 2
    indep = [torch.randn(40, 8, generator=_g(s)) for s in (1, 2)]
    assert cross_seed_cka(same)["mean_cka"] > 0.99
    assert cross_seed_cka(indep)["mean_cka"] < 0.6


def test_code_stability_random_near_chance():
    codes = [torch.randint(0, 4, (60,), generator=_g(s)) for s in range(3)]
    cs = code_stability(codes, 4)
    assert cs["stable"] is False and abs(cs["mean_agreement"] - cs["chance"]) < 0.2


def test_hungarian_assignment_is_exact_without_optional_scipy():
    cost = np.array([[1.0, 2.0, 100.0], [1.0, 100.0, 100.0], [100.0, 1.0, 1.0]])
    assignment = _hungarian(cost)
    assert assignment == [1, 0, 2]
    assert sum(cost[r, c] for r, c in enumerate(assignment)) == 4.0


def test_capability_per_bit_curve():
    g = _g()
    x = torch.randn(300, 32, generator=g)
    y = (x[:, 0] > 0).long()
    cpb = capability_per_bit(x, y, widths=(1, 2, 4, 8), seed=0)
    assert set(cpb["acc_real"]) == {1, 2, 4, 8} and "knee_width_95pct" in cpb


def test_sysid_licenses_controllable_system():
    g = _g()
    d, a, n = 6, 3, 400
    A = torch.randn(d, d, generator=g) * 0.3
    B = torch.randn(d, a, generator=g)
    Z = torch.randn(n, d, generator=g)
    Ac = torch.randn(n, a, generator=g)
    Zn = Z @ A.T + Ac @ B.T + 0.01 * torch.randn(n, d, generator=g)
    sr = sysid_report(Z, Ac, Zn, seed=0)
    assert sr["one_step_r2"] > 0.9 and sr["actions_move_state"] and sr["planning_licensed"]


def test_sysid_rejects_uncontrollable():
    # actions are pure noise unrelated to the transition: not controllable, planning not licensed
    g = _g()
    d, a, n = 5, 2, 300
    A = torch.randn(d, d, generator=g) * 0.3
    Z = torch.randn(n, d, generator=g)
    Ac = torch.randn(n, a, generator=g)
    Zn = Z @ A.T + 0.01 * torch.randn(n, d, generator=g)  # Znext does not depend on actions
    sr = sysid_report(Z, Ac, Zn, seed=0)
    assert sr["actions_move_state"] is False and sr["planning_licensed"] is False
