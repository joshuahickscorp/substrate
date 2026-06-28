"""Automated ablation/hypothesis engine: ranks by expected info gain per compute hour, gates on what
is runnable in a scope, respects an hour budget, detects redundant candidates, and names a runnable
next-best experiment."""

import pytest

from devsys.devel import ablation, metacognition


def test_local_plan_ranks_and_names_next_best():
    plan = ablation.plan_ablations(scope="local")
    assert plan["ranked"]
    nb = plan["next_best"]
    assert nb is not None and nb["runnable_now"] is True
    # next best is the highest eig/hour among runnable candidates
    runnable = [r for r in plan["ranked"] if r["runnable_now"]]
    assert nb["eig_per_hour"] == max(r["eig_per_hour"] for r in runnable)


def test_local_scope_gates_out_studio_and_blocked():
    plan = ablation.plan_ablations(scope="local")
    for r in plan["ranked"]:
        if r["status"] in ("studio-later", "paper-watch", "blocked"):
            assert r["runnable_now"] is False
            assert r["gate"]  # a reason is recorded


def test_studio_scope_enables_studio_later():
    local = {r["slug"]: r for r in ablation.plan_ablations(scope="local")["ranked"]}
    studio = {r["slug"]: r for r in ablation.plan_ablations(scope="studio")["ranked"]}
    studio_later = [s for s, r in local.items() if r["status"] == "studio-later"]
    assert studio_later  # there is at least one
    for s in studio_later:
        assert local[s]["runnable_now"] is False
        assert studio[s]["runnable_now"] is True


def test_budget_prefix_respected():
    plan = ablation.plan_ablations(scope="local", budget_hours=1.0)
    # the within-budget set must not exceed the hour budget by summed est_hours
    by = {r["slug"]: r for r in plan["ranked"]}
    total = sum(by[s]["est_hours"] for s in plan["within_budget"])
    assert total <= 1.0


def test_competing_groups_detected():
    plan = ablation.plan_ablations(scope="local")
    # at least one family+changes pair has more than one candidate (same subsystem; run one first)
    assert plan["competing_groups"]
    for g in plan["competing_groups"]:
        assert len(g["candidates"]) > 1
    # every ranked row carries basis=assumption (the cost/eig are levers, not measurements)
    assert all(r["basis"] == "assumption" for r in plan["ranked"])


def test_bad_scope_raises():
    with pytest.raises(ValueError):
        ablation.plan_ablations(scope="moon")


def test_metacognition_wires_ablation_next_best():
    plan = ablation.plan_ablations(scope="local")
    rep = metacognition.report(ablation_plan=plan)
    assert rep["which_mechanism_next"]["slug"] == plan["next_best"]["slug"]
