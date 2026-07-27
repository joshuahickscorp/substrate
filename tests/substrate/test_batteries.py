"""The four entity batteries, each built to be failable.

House style: no dashes.
"""

from __future__ import annotations

from substrate import batteries as B
from substrate import memory as M
from substrate import perspectives as PS


def test_thinking_requires_a_declared_alternative_to_beat():
    # latency alone is refused outright, not scored badly
    refused = B.thinking_battery(
        {"routes": ["maintaining_state"], "beats": {"latency": {"substrate": 10, "alternative": 1}}}
    )
    assert refused["scored"] is False and refused["supported"] is False
    assert "excludes" in refused["reason"]
    assert refused["non_evidence_named"] == ["latency"]

    # so are hidden activations
    assert (
        B.thinking_battery({"beats": {"hidden_activations": {"substrate": 1, "alternative": 0}}})["scored"]
        is False
    )

    # a partial comparison is scored but not supported, and it names what is missing
    partial = B.thinking_battery(
        {
            "routes": ["maintaining_state"],
            "beats": {"larger_static_model": {"substrate": 0.9, "alternative": 0.8}},
        }
    )
    assert partial["scored"] is True and partial["supported"] is False
    assert len(partial["alternatives_missing"]) == 5
    assert "never compared against" in partial["reason"]

    # all six, each clearing the SESOI, is supported
    full = B.thinking_battery(
        {
            "routes": list(B.THINKING_ROUTES),
            "beats": {a: {"substrate": 0.9, "alternative": 0.7} for a in B.ALTERNATIVES},
        }
    )
    assert full["supported"] is True and full["alternatives_missing"] == []

    # one alternative inside the SESOI is enough to withhold support
    narrow = B.thinking_battery(
        {
            "routes": list(B.THINKING_ROUTES),
            "beats": {
                **{a: {"substrate": 0.9, "alternative": 0.7} for a in B.ALTERNATIVES},
                "more_tokens": {"substrate": 0.9, "alternative": 0.88},
            },
        }
    )
    assert narrow["supported"] is False

    # an undeclared thinking route is named
    odd = B.thinking_battery(
        {
            "routes": ["telepathy"],
            "beats": {a: {"substrate": 0.9, "alternative": 0.7} for a in B.ALTERNATIVES},
        }
    )
    assert odd["undeclared_routes"] == ["telepathy"] and odd["supported"] is False


def test_continuity_survives_context_removal_from_owned_state():
    entity, probes = B._demo_entity()
    report = B.continuity_battery(entity, probes)
    assert report["owned_state_score"] == 1.0
    assert report["stressors_applied"] == ["context_removal", "checkpoint_restore"]
    assert report["stressors_not_applied"], "the stressors not run are listed rather than implied"
    # the margin that supports the claim is the contested one
    assert report["supported"] is True
    assert report["contested_margin"] > B.SESOI
    assert report["contested_surfaces"], "there has to be something the replay arm could have won"


def test_the_replay_control_is_capable_of_winning():
    """A control that cannot win at any budget is not a control, and this asserts it can."""
    entity, probes = B._demo_entity()
    report = B.continuity_battery(entity, probes)
    assert report["control_is_capable"] is True
    assert report["unbounded_replay_score"] > report["transcript_replay_score"]
    # and the surfaces no transcript ever held are excluded from the head to head, not counted as wins
    assert report["surfaces_no_replay_can_reach"]
    for surface in report["surfaces_no_replay_can_reach"]:
        assert surface not in report["contested_surfaces"]


def test_owned_state_actually_carries_the_answers_after_a_restore():
    entity, probes = B._demo_entity()
    snapshot = entity.checkpoint()
    revived = B.Entity().restore(snapshot)
    assert revived.transcript == [], "the restored entity has no transcript at all"
    for surface, probe in probes.items():
        assert probe["read"](revived) == probe["expected"], surface


def test_unity_measures_global_availability_not_shared_mutability():
    entity, _ = B._demo_entity()
    outputs = [
        PS.Output("a", "left", 0.9, ["a:perceptual"], 1.0),
        PS.Output("b", "left", 0.7, ["b:temporal"], 1.0),
        PS.Output("c", "right", 0.8, ["c:episodic_context"], 1.0),
    ]
    report = B.unity_battery(entity, outputs)
    ga = report["measures"]["global_availability"]
    assert ga["regions_readable_by_any_component"] > 0
    assert ga["regions_writable_by_any_component"] == 0, "shared write access would be the corruption"
    assert report["measures"]["preservation_of_alternatives"]["minority_preserved"] == 1
    assert set(report["measures_declared"]) == set(B.UNITY_MEASURES)
    assert set(report["measures"]) == set(B.UNITY_MEASURES)


def test_reflective_report_fails_closed_without_provenance():
    entity, _ = B._demo_entity()
    grounded = B.reflective_report(entity, "f1")
    assert grounded["answered"] is True and grounded["bound_to_receipts"] is True
    assert set(grounded["answers"]) == set(B.REFLECTIVE_QUESTIONS)
    assert grounded["answers"]["where_a_belief_came_from"].startswith("proof/")

    unsourced = B.reflective_report(entity, "f2")
    assert unsourced["answered"] is False and unsourced["failed_closed"] is True
    assert all(v is None for v in unsourced["answers"].values())
    assert "no provenance" in unsourced["reason"]

    absent = B.reflective_report(entity, "nothing_like_this")
    assert absent["answered"] is False and absent["failed_closed"] is True
    assert "not known" in absent["reason"]

    battery = B.reflective_battery(entity, ["f1", "f2", "nothing_like_this"])
    assert battery["answered"] == 1 and battery["failed_closed"] == 2
    assert battery["every_answer_bound_to_a_receipt"] is True


def test_a_superseded_belief_reports_its_own_failure():
    entity, _ = B._demo_entity()
    entity.semantic.supersede(
        "f1", M.Fact("f3", "only one bed is temporal", 0.9, provenance="proof/later.json")
    )
    report = B.reflective_report(entity, "f1")
    assert report["answers"]["failure"] == "superseded by f3"
    assert B.reflective_report(entity, "f3")["answers"]["what_evidence_supports_it"] == ["f3", "f1"]


def test_the_declaration_does_not_claim_thinking():
    doc = B.declaration()
    assert doc["thinking"]["supported"] is False
    assert "No thinking claim has been supported" in doc["honest_state"]
