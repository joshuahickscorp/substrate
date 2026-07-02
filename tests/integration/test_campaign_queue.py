"""The Volume IV campaign queue (synthesized from the corpus DAG) validates, resolves tiers
and dependencies, and runs one Tier C leg end to end at toy scale."""

from mop.harness.queue import load_queue, plan, run_queue, topo_order, validate


def test_manifest_validates():
    legs = load_queue()
    validate(legs)  # raises on bad tier / unknown dep / cycle
    assert len(legs) >= 11  # >= 11 tracks worth of legs


def test_dependencies_resolve_in_topo_order():
    legs = load_queue()
    order = [l.name for l in topo_order(legs)]
    by = {l.name: l for l in legs}
    for name in order:
        for dep in by[name].depends_on:
            assert order.index(dep) < order.index(name)


def test_tier_gating_defaults_C_only():
    legs = load_queue()
    c_plan = plan(legs, {"C"})
    assert all(l.tier == "C" and l.enabled for l in c_plan)
    # E and R legs are gated out by default
    names = {l.name for l in c_plan}
    assert not any("e10" in n or "poet" in n or "cultural" in n for n in names)
    # all of E1's children that are Tier C appear, E1 gate first
    assert c_plan[0].depends_on == []


def test_level5_after_its_dependencies():
    legs = load_queue()
    c_plan = [l.name for l in plan(legs, {"C"})]
    i5 = next(i for i, n in enumerate(c_plan) if "level5" in n)
    for dep in ("track02_e2_replay", "track03_e3_plasticity", "track04_e4_neuromod"):
        assert c_plan.index(dep) < i5


def test_dry_run_plans_C_and_gates_ER():
    out = run_queue(dry_run=True, enabled_tiers={"C"})
    assert len(out["planned"]) >= 8
    assert len(out["gated_out"]) >= 3  # the E/R legs


def test_one_toy_tier_c_leg_runs():
    out = run_queue(dry_run=False, enabled_tiers={"C"}, toy=True, max_runs_per_leg=1, max_legs=1)
    assert len(out["results"]) == 1
    only = next(iter(out["results"].values()))
    assert only == 1  # one combo ran
