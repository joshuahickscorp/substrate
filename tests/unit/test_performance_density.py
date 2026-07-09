import time

import pytest

from mop.diagnostics.performance_density import DENSITY_SCHEMA, density_block, timed


def test_density_block_computes_primary_ratios():
    block = density_block(
        {"transfer": 0.8, "retention": 0.6},
        flops=2.0e6,
        bytes=4.0e3,
        seconds=2.0,
    )
    assert block["schema"] == DENSITY_SCHEMA
    assert block["primary"] == "transfer"
    assert block["capability"] == {"transfer": 0.8, "retention": 0.6}
    assert block["density"]["transfer_per_flops"] == pytest.approx(0.8 / 2.0e6)
    assert block["density"]["transfer_per_bytes"] == pytest.approx(0.8 / 4.0e3)
    assert block["density"]["transfer_per_seconds"] == pytest.approx(0.4)


def test_density_block_named_primary_and_zero_cost_gets_no_ratio():
    block = density_block(
        {"acc": 0.5, "recall": 0.9},
        primary="recall",
        params=100.0,
        seconds=0.0,
    )
    assert block["primary"] == "recall"
    assert block["density"] == {"recall_per_params": pytest.approx(0.009)}
    assert block["cost"]["seconds"] == 0.0


def test_density_block_refuses_unpriced_and_bad_inputs():
    with pytest.raises(ValueError, match="at least one cost"):
        density_block({"acc": 0.5})
    with pytest.raises(ValueError, match="negative"):
        density_block({"acc": 0.5}, flops=-1.0)
    with pytest.raises(ValueError, match="capability"):
        density_block({}, flops=1.0)
    with pytest.raises(ValueError, match="primary"):
        density_block({"acc": 0.5}, primary="missing", flops=1.0)


def test_timed_measures_wallclock():
    with timed() as t:
        time.sleep(0.01)
    assert t["seconds"] >= 0.005
