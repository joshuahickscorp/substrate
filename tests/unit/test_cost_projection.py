
from __future__ import annotations

import json

from mop.harness.queue import load_queue
from mop.studies.cost_projection import (
    cost_projection,
    load_timings_from_checkpoint,
    render_md,
)


def test_covers_every_queued_leg_including_disabled_ER():
    timings = {"e1_baseline": 4.0, "e2_replay": 7.5}
    proj = cost_projection(timings, workers=4, full_seed=5)
    names = {r["name"] for r in proj["per_leg"]}
    queued = {l.name for l in load_queue()}
    assert names == queued, "projection must cover exactly the queued legs"
    tiers = {r["tier"] for r in proj["per_leg"]}
    assert {"C", "E", "R"} <= tiers, "disabled E and R legs must be projected too"
    assert any(not r["enabled"] for r in proj["per_leg"]), "disabled legs included"


def test_measured_vs_assumed_and_full_scale_units():
    from mop.harness.sweep import full_run_units, load_leg

    timings = {"e1_baseline": 4.0}
    proj = cost_projection(timings, workers=8, full_seed=5)
    by = {r["name"]: r for r in proj["per_leg"]}
    legs = {leg.name: leg for leg in load_queue()}
    e1 = by["track01_e1_gate"]
    assert e1["basis"] == "measured" and e1["per_unit_s_measured_or_assumed"] == 4.0
    assert e1["run_units_full"] == full_run_units(load_leg(legs["track01_e1_gate"].sweep))  # 3x2x5 = 30
    e2 = by["track02_e2_replay"]
    assert e2["basis"] == "assumed", "C leg with no timing falls back to assumed"
    assert e2["run_units_full"] == full_run_units(load_leg(legs["track02_e2_replay"].sweep))  # 2x2x5 = 20
    er = [r for r in proj["per_leg"] if r["tier"] in ("E", "R")]
    assert er and all(r["basis"] == "assumed" for r in er), "E/R are assumption-based"


def test_C_units_come_from_leg_full_grid_not_param():
    t = {"e1_baseline": 1.0}
    p5 = cost_projection(t, workers=4, full_seed=5)
    p3 = cost_projection(t, workers=4, full_seed=3)
    u5 = {r["name"]: r["run_units_full"] for r in p5["per_leg"] if r["tier"] == "C"}
    u3 = {r["name"]: r["run_units_full"] for r in p3["per_leg"] if r["tier"] == "C"}
    assert u5 == u3, "C units are canonical (leg full grid), independent of the full_seed param"


def test_parallelism_reduces_wall_and_totals_are_consistent():
    t = {"e1_baseline": 4.0, "e2_replay": 7.5, "e3_plasticity": 5.0}
    p1 = cost_projection(t, workers=1)
    pN = cost_projection(t, workers=10)
    assert pN["grand_total_parallel_wall_s"] < p1["grand_total_parallel_wall_s"]
    assert abs(p1["grand_total_parallel_wall_s"] - p1["grand_total_serial_s"]) < 1e-6
    assert abs(pN["grand_total_parallel_wall_s"] - pN["grand_total_serial_s"] / 10) < 1e-6
    s = sum(v["serial_s"] for v in pN["totals_by_tier"].values())
    assert abs(s - pN["grand_total_serial_s"]) < 1e-3, "tier totals sum to grand serial"


def test_provenance_notes_and_md():
    proj = cost_projection({"e1_baseline": 4.0}, workers=6)
    assert proj["provenance"] == "provisional"
    assert any("laptop-throttled" in n for n in proj["notes"])
    md = render_md(proj)
    assert "Studio cost projection" in md and "track01_e1_gate" in md
    assert "tier" in md and "parallel_wall_s" in md


def test_load_timings_from_checkpoint(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "manifest.json").write_text(
        json.dumps({"name": "e1_baseline", "started": 100.0, "finished": 106.0})
    )
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "manifest.json").write_text(
        json.dumps({"name": "e1_baseline", "started": 0.0, "finished": 4.0})
    )
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "manifest.json").write_text(json.dumps({"name": "x", "started": 1.0}))  # skip
    t = load_timings_from_checkpoint(tmp_path)
    assert t["e1_baseline"] == 5.0, "mean of (6,4) per-run durations"
    assert "x" not in t, "manifest missing finished is skipped"
