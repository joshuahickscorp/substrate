"""The model body contract is explicit enough that a body can fail it.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import body as B


def test_the_contract_declares_nine_kinds_and_six_modes():
    assert len(B.MESSAGE_KINDS) == 9
    assert len(B.INTEGRATION_MODES) == 6
    assert set(B.REQUIRED_FIELDS) == set(B.MESSAGE_KINDS)


def test_a_message_missing_a_required_field_is_named_not_accepted():
    assert B.validate_message("checkpoint", {"identity": "core_v1", "sha256": "a" * 64}) == []
    gaps = B.validate_message("checkpoint", {"identity": "core_v1"})
    assert gaps == ["checkpoint: sha256 not supplied"]
    # an adaptation proposal carries the same seven fields the safety envelope requires
    from mop.cognition import safety
    assert set(B.REQUIRED_FIELDS["adaptation_proposal"]) == set(safety.ADAPTATION_FIELDS)
    with pytest.raises(B.Refused):
        B.validate_message("telepathy", {})


def test_a_partial_body_is_reported_as_partial_not_as_conforming():
    partial = B.BodyContract("compact_specialist", "sidecar_temporal_core",
                             ("inference", "hidden_state", "checkpoint"))
    report = B.conformance(partial)
    assert report["conforms"] is False and report["partial"] is True
    assert set(report["missing"]) == set(B.MESSAGE_KINDS) - set(partial.implements)

    full = B.BodyContract("full", "adapter_layer", B.MESSAGE_KINDS)
    assert B.conformance(full)["conforms"] is True

    broken = B.BodyContract("bad", "telepathy", ("inference", "reading_minds"))
    report = B.conformance(broken)
    assert report["conforms"] is False and report["partial"] is False
    assert len(report["declaration_violations"]) == 2


def test_the_declaration_says_no_body_is_attached():
    doc = B.declaration()
    assert doc["any_body_attached"] is False
    assert doc["attached_bodies"] == []
    assert "no model body is attached" in doc["honest_state"]
    assert set(doc["bodies_to_test"]) == set(B.TEST_BODIES)
