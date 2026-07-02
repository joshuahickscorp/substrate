"""Frontier 1: full-grid accounting. The run_queue manifest, what `run_queue --full` actually
expands (sweep), and what cost_projection projects must agree EXACTLY, all from the leg YAMLs.
Also guards that every toy axis is a subset of the full grid (no toy knob the full grid lacks)."""

from mop.harness.queue import load_queue
from mop.harness.sweep import expand, full_run_units, load_leg, toy_run_units
from mop.studies.cost_projection import TIER_FULL, cost_projection


def _legs():
    return [(leg, load_leg(leg.sweep)) for leg in load_queue()]


def test_manifest_run_units_equal_full_grid():
    for leg, d in _legs():
        assert leg.run_units == full_run_units(d), (leg.name, leg.run_units, full_run_units(d))
        assert leg.toy_run_units == toy_run_units(d), leg.name


def test_sweep_full_expansion_matches_count():
    # the number of override-lists `run_sweep(toy=False)` would run == full_run_units, exactly
    for _leg, d in _legs():
        axes = dict(d.get("full_axes", d.get("axes", {})))
        seeds = list(d.get("full_seeds", d.get("seeds", [0])))
        assert len(expand(axes, seeds, [])) == full_run_units(d)


def test_toy_axes_are_subset_of_full_axes():
    for leg, d in _legs():
        toy_keys = set(dict(d.get("axes", {})))
        full_keys = set(dict(d.get("full_axes", d.get("axes", {}))))
        assert toy_keys <= full_keys, (leg.name, toy_keys - full_keys)


def test_cost_projection_units_match_manifest():
    cp = cost_projection({}, workers=4, full_seed=5)
    by_name = {row["name"]: row for row in cp["per_leg"]}
    for leg, d in _legs():
        row = by_name[leg.name]
        mult = TIER_FULL.get(leg.tier, TIER_FULL["C"])["unit_mult"]
        expected = full_run_units(d) * (mult if leg.tier != "C" else 1)
        assert row["run_units_full"] == expected, (leg.name, row["run_units_full"], expected)


def test_every_leg_declares_a_full_grid():
    for leg, d in _legs():
        assert "full_axes" in d and "full_seeds" in d, leg.name
        assert full_run_units(d) >= len(list(d.get("full_seeds", [])))
