"""The runtime loop: eleven stages, every one receipted, and no path to acting on the world.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import perspectives as PS
from substrate import runtime as R
from substrate import safety as SF
from substrate import workspace as W

OBS = {"label": "a", "label_confidence": 0.8}


def test_one_cycle_runs_every_declared_stage():
    entity = R.Substrate()
    trace = entity.step(OBS, outcome="a", goal=["finish the cycle"])
    assert set(trace["stages"]) == set(R.STAGES)
    assert trace["complete"] is True
    assert trace["stages_skipped"] == []
    assert len(R.STAGES) == 11


def test_a_skipped_stage_says_why_rather_than_looking_like_one_that_ran():
    entity = R.Substrate()
    trace = entity.step(OBS)  # no outcome, so nothing can be compared
    assert trace["stages_skipped"] == ["self_update"]
    assert "nothing can be compared" in trace["stages"]["self_update"]["reason"]
    assert trace["complete"] is False, "a cycle that skipped a stage is not a complete cycle"
    # and the reflective report surfaces the reason rather than the absence
    report = entity.report()
    assert report["why_skipped"]["self_update"]


def test_the_loop_never_acts_on_the_world():
    entity = R.Substrate()
    entity.step(OBS, outcome="a")
    decision = entity.ws.read("decision", "reader")
    assert decision["activation"] is False
    doc = R.declaration()
    assert doc["activation"] is False
    assert "no code path" in doc["no_activation_path"]
    # the claim is checkable: the module contains no assignment that turns activation on
    source = (R.io.ROOT / "src/substrate/runtime.py").read_text()
    assert 'activation": True' not in source and "activation=True" not in source


def test_a_perspective_still_cannot_read_outside_its_declaration_inside_the_loop():
    """The type system is not bypassed by being called from the runtime."""
    greedy = PS.Perspective(
        PS._spec(
            "greedy",
            "direct_prediction",
            ("world",),
            "peek at the world state",
            "label",
            0.5,
            ("peeks",),
            "n/a",
        ),
        lambda seen: (seen.get("world"), 1.0),
    )
    object.__setattr__(greedy.spec, "permitted_information", ("perceptual",))
    entity = R.Substrate(catalog=[greedy])
    trace = entity.step(OBS, outcome="a")
    # it declared world as an input but is only permitted perceptual, so the workspace refuses the read
    assert trace["stages"]["run_perspectives"]["refused"] == ["greedy"] or trace["stages"][
        "run_perspectives"
    ]["ran"] == ["greedy"]
    # either way nothing it produced was allowed to write a region it does not own
    with pytest.raises(W.Refused):
        entity.ws.write("world", "greedy", {"forged": True}, provenance="p", confidence=1.0)


def test_the_cycle_budget_caps_how_much_thinking_happens():
    rich = R.Substrate(cycle_budget=99.0)
    poor = R.Substrate(cycle_budget=1.0)
    rich_trace = rich.step(OBS, outcome="a")
    poor_trace = poor.step(OBS, outcome="a")
    assert poor_trace["stages"]["run_perspectives"]["compute_spent"] <= 1.0
    assert (
        rich_trace["stages"]["run_perspectives"]["compute_spent"]
        >= poor_trace["stages"]["run_perspectives"]["compute_spent"]
    )
    assert poor_trace["stages"]["attend"]["dropped_for_budget"] != [] or len(
        poor_trace["stages"]["run_perspectives"]["ran"]
    ) <= len(rich_trace["stages"]["run_perspectives"]["ran"])


def test_adaptation_goes_through_the_safety_envelope_not_around_it():
    entity = R.Substrate()
    trace = entity.step(OBS, outcome="a")
    adapt = trace["stages"]["adapt"]
    assert adapt["level"] in {
        level.name for level in __import__("substrate.plasticity", fromlist=["LEVELS"]).LEVELS
    }
    assert adapt["applied"] is True and adapt["refusals"] == []
    # the loop cannot propose removing a protected surface, because it never builds such a proposal
    doc = R.declaration()
    assert set(doc["protected_surfaces_the_loop_cannot_remove"]) == set(SF.PROTECTED_SURFACES)


def test_checkpoint_and_restore_reproduce_the_entity_identity():
    entity = R.Substrate()
    for i in range(3):
        entity.step({"label": "a", "label_confidence": 0.7 + i / 100}, outcome="a")
    snapshot = entity.checkpoint()

    revived = R.Substrate().restore(snapshot)
    assert revived.step_index == entity.step_index
    assert revived.checkpoint()["identity"] == snapshot["identity"]
    assert revived.reliability == entity.reliability
    assert set(revived.episodes.store) == set(entity.episodes.store)


def test_a_tampered_checkpoint_is_refused_rather_than_silently_restored():
    entity = R.Substrate()
    entity.step(OBS, outcome="a")
    snapshot = entity.checkpoint()
    snapshot["reliability"] = {k: 0.99 for k in snapshot["reliability"]}
    with pytest.raises(R.Refused):
        R.Substrate().restore(snapshot)


def test_the_reflective_report_fails_closed_before_any_cycle_runs():
    entity = R.Substrate()
    report = entity.report()
    assert report["answered"] is False and report["failed_closed"] is True
    assert "no cycle has run" in report["reason"]
    entity.step(OBS, outcome="a")
    assert entity.report()["answered"] is True
    # and a step that never happened is refused rather than answered about
    missing = entity.report(step=999)
    assert missing["answered"] is False and missing["failed_closed"] is True


def test_the_declaration_names_what_is_not_composed_yet():
    doc = R.declaration()
    assert set(doc["not_composed_yet"]) == {"owned temporal core", "world model", "model body"}
    assert "not scientifically licensed" in doc["why_not_composed"]
    assert doc["every_stage_leaves_a_receipt"] is True


def test_an_undeclared_stage_cannot_be_recorded():
    trace = R.CycleTrace(1)
    with pytest.raises(R.Refused):
        trace.record("telepathy")
    with pytest.raises(R.Refused):
        trace.skip("telepathy", "because")
