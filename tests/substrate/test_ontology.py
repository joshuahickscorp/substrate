"""The section 6.2 ontology battery, all ten required tests.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import ontology as O


def _o():
    return O.Ontology()


def test_every_declared_type_and_distinction_exists():
    assert len(O.TYPES) == 28  # twenty seven declared plus unknown
    assert "unknown" in O.TYPES
    assert len(O.DISTINCTIONS) == 9
    assert set(O.ITEM_FIELDS) == {"identity", "type", "attributes", "relations", "temporal_extent",
                                  "source", "confidence", "persistence", "supersession"}


def test_identity_preservation_over_time():
    o = _o()
    o.add(O.Item("cup", "object", {"colour": "red"}, temporal_extent=(0, 10)))
    o.add(O.Item("cup_later", "object", {"colour": "red"}, temporal_extent=(5, 20)))
    merged = o.merge("cup", "cup_later", evidence="same position across the boundary")
    assert merged.temporal_extent == (0, 20), "identity over time spans both observations"
    assert merged.merged_from == ("cup", "cup_later")


def test_referent_merging_needs_evidence_and_stays_reversible():
    o = _o()
    o.add(O.Item("a", "object", {"mass": 1}, temporal_extent=(0, 5)))
    o.add(O.Item("b", "object", {"colour": "red"}, temporal_extent=(0, 5)))
    with pytest.raises(O.Refused):
        o.merge("a", "b", evidence="")
    merged = o.merge("a", "b", evidence="one referent seen under two names")
    assert merged.attributes == {"mass": 1, "colour": "red"}
    assert o.get("a").supersession == merged.identity
    restored = o.split(merged.identity, reason="later observation separated them")
    assert {i.identity for i in restored} == {"a", "b"}
    assert o.get("a").supersession is None
    assert o.get(merged.identity).supersession == "split", "the mistake stays in the record"


def test_mistaken_merge_detection():
    o = _o()
    o.add(O.Item("x", "object", {"colour": "red"}, temporal_extent=(0, 5)))
    o.add(O.Item("y", "object", {"colour": "blue"}, temporal_extent=(0, 5)))
    merged = o.merge("x", "y", evidence="assumed the same object")
    verdict = o.detect_mistaken_merge(merged.identity)
    assert verdict["mistaken"] is True
    assert verdict["clashing_attributes"] == ["colour"]
    assert o.contradictions, "the contradiction is kept, not discarded"


def test_mistaken_split_detection():
    o = _o()
    o.add(O.Item("p", "object", {"colour": "red", "mass": 1}, temporal_extent=(0, 5)))
    o.add(O.Item("q", "object", {"colour": "red", "mass": 1}, temporal_extent=(0, 5)))
    assert o.detect_mistaken_split("p", "q")["mistaken_split"] is True
    o.add(O.Item("r", "object", {"colour": "blue"}, temporal_extent=(0, 5)))
    assert o.detect_mistaken_split("p", "r")["mistaken_split"] is False


def test_type_revision_leaves_a_chain():
    o = _o()
    o.add(O.Item("thing", "unknown", unknown_reason="no matching prototype"))
    with pytest.raises(O.Refused):
        o.revise_type("thing", "object", evidence="")
    with pytest.raises(O.Refused):
        o.revise_type("thing", "sandwich", evidence="looks like one")
    o.revise_type("thing", "object", evidence="persisted across occlusion")
    assert o.get("thing").type == "object" and o.get("thing").unknown_reason == ""
    assert o.revisions[-1]["from"] == "unknown" and o.revisions[-1]["to"] == "object"


def test_relation_consistency():
    o = _o()
    o.add(O.Item("whole", "system"))
    o.add(O.Item("part", "component", relations=[("part_of", "whole")]))
    assert o.relation_consistency()["consistent"] is True
    o.add(O.Item("orphan", "component", relations=[("part_of", "nothing_here")]))
    report = o.relation_consistency()
    assert report["consistent"] is False and report["dangling"][0]["to"] == "nothing_here"
    # a composition cycle is refused as well
    o2 = _o()
    o2.add(O.Item("a", "system", relations=[("part_of", "b")]))
    o2.add(O.Item("b", "system", relations=[("part_of", "a")]))
    assert o2.relation_consistency()["cyclic_composition"]


def test_temporal_consistency():
    o = _o()
    o.add(O.Item("whole", "process", temporal_extent=(0, 10)))
    o.add(O.Item("part", "event", temporal_extent=(2, 3), relations=[("part_of", "whole")]))
    assert o.temporal_consistency()["consistent"] is True
    o.add(O.Item("late", "event", temporal_extent=(50, 60), relations=[("part_of", "whole")]))
    assert o.temporal_consistency()["consistent"] is False


def test_unknown_object_handling():
    o = _o()
    with pytest.raises(O.Refused):
        o.add(O.Item("mystery", "unknown"))  # unknown must say why
    o.add(O.Item("mystery", "unknown", unknown_reason="no prototype matched and no rule fired"))
    assert o.get("mystery").type == "unknown"
    # and the typing ladder returns unknown rather than guessing
    assert O.type_of({"id": "z"}, "rule_based", rules={"wheels": "object"}) == "unknown"
    assert O.type_of({"id": "z"}, "prototype", prototypes={"object": ("mass",)}) == "unknown"


def test_counterfactual_objects_never_merge_with_actual_ones():
    o = _o()
    o.add(O.Item("real", "object", {"colour": "red"}, temporal_extent=(0, 5), modality="actual"))
    o.add(O.Item("imagined", "object", {"colour": "red"}, temporal_extent=(0, 5),
                 modality="counterfactual"))
    assert o.distinguishable("real", "imagined")["separated_by"] == ["possible_versus_actual"]
    with pytest.raises(O.Refused):
        o.merge("real", "imagined", evidence="they look identical")


def test_the_self_environment_boundary_holds():
    o = _o()
    o.add(O.Item("me", "self", temporal_extent=(0, None)))
    o.add(O.Item("them", "other", temporal_extent=(0, None)))
    assert o.self_environment_boundary()["singleton"] is True
    assert o.distinguishable("me", "them")["any"] is True
    with pytest.raises(O.Refused):
        o.merge("me", "them", evidence="both are agents")


def test_an_observation_is_never_merged_with_an_inference():
    o = _o()
    o.add(O.Item("seen", "observation", {"v": 1}, source="observed", temporal_extent=(0, 1)))
    o.add(O.Item("guessed", "belief", {"v": 1}, source="inferred", temporal_extent=(0, 1)))
    separated = o.distinguishable("seen", "guessed")["separated_by"]
    assert "observation_versus_inferred" in separated
    assert "evidence_versus_belief" in separated
    with pytest.raises(O.Refused):
        o.merge("seen", "guessed", evidence="same value")


def test_the_learned_typer_stays_closed_without_headroom():
    for simple in O.SIMPLE_TYPING:
        O.type_of({"id": "a", "mass": 1}, simple, rules={"mass": "object"},
                  prototypes={"object": ("mass",)}, retrieved={"a": "object"})
    with pytest.raises(O.Refused):
        O.type_of({"id": "a"}, "learned")
    with pytest.raises(O.Refused):
        O.type_of({"id": "a"}, "learned", headroom={"residual_lower_95_cb": 0.01})
    assert O.type_of({"id": "a"}, "learned", headroom={"residual_lower_95_cb": 0.2},
                     retrieved={"a": "object"}) == "object"
    with pytest.raises(O.Refused):
        O.type_of({"id": "a"}, "oracle")
