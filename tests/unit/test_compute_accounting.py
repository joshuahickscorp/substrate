
from torch import nn

from mop.diagnostics import compute as C


def test_param_count():
    m = nn.Linear(10, 4)
    assert C.param_count(m) == 10 * 4 + 4  # weights + bias


def test_linear_flops_is_two_mac():
    assert C.linear_flops(8, 4, batch=2) == 2 * 2 * 8 * 4


def test_mlp_flops_sums_layers():
    assert C.mlp_flops([16, 32, 8], batch=1) == C.linear_flops(16, 32) + C.linear_flops(32, 8)


def test_refiner_flops_scales_with_steps():
    one = C.refiner_flops(16, 32, steps=1)
    four = C.refiner_flops(16, 32, steps=4)
    assert four == 4 * one


def test_matched_within_true_for_equal():
    out = C.matched_within(1000, 1000)
    assert out["matched"] is True and out["ratio"] == 1.0


def test_matched_within_false_for_double():
    out = C.matched_within(1000, 2000, tol=0.1)
    assert out["matched"] is False and out["ratio"] == 2.0


def test_depth_for_matched_flops():
    assert C.depth_for_matched_flops(16, 32, refiner_steps=5) == 5


def test_accounting_record():
    m = nn.Linear(8, 8)
    rec = C.accounting(m, [8, 16, 8], steps=3)
    assert rec["params"] == C.param_count(m)
    assert rec["flops_total"] == 3 * rec["flops_per_pass"]
