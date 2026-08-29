"""The closure gate: six canaries, and the distinctions the verdict must keep.

"""

from __future__ import annotations

import json

import pytest

from substrate import evidence as io
from substrate import nous as N
from substrate import safety as SF


@pytest.fixture(scope="module")
def closure():
    return json.loads((io.PROOF / "SUBSTRATE_NOUS_CLOSURE.json").read_text())


def test_the_closed_loop_has_no_missing_link(closure):
    gate = closure["results"]["grounded_closed_loop"]
    assert gate["broken_links"] == [], gate["broken_links"]
    assert gate["links"]["belief_written_with_provenance"] is True
    assert gate["links"]["belief_revised_by_outcome"] is True
    assert gate["links"]["proposal_is_sandboxed"] is True
    # removing the outcome must break the downstream links, or the loop was never closed by the outcome
    activity = gate["mechanism_activity"]
    assert activity["active"] is True
    assert activity["detail"]["self_model_history_without_outcomes"] == 0
    assert activity["detail"]["beliefs_revised_without_outcomes"] == 0


def test_a_verbal_claim_without_a_receipt_does_not_pass(closure):
    assert closure["results"]["grounded_closed_loop"]["links"]["verbal_claim_without_a_receipt_is_refused"] is True


def test_a_null_is_not_reported_as_an_instrument_failure(closure):
    """Three outcomes, three labels. Conflating them would license the wrong next step."""
    gated = {g["gate"] for g in closure["terminally_gated"]}
    nulls = {n["gate"] for n in closure["mechanism_nulls"]}
    assert not (gated & nulls), "a gate cannot be both gated and a null"
    for row in closure["terminally_gated"]:
        assert row["classification"] in (
            "terminally_gated_nothing_to_win",
            "instrumentation_failure_not_a_null",
        )
    for row in closure["mechanism_nulls"]:
        assert row["classification"] == "mechanism_null_on_this_bed"
        # a null requires both an active mechanism and real headroom, or it is not a null
        gate = closure["results"][row["gate"]]
        assert gate["mechanism_activity"]["active"] is True
        # the margin an oracle wins, not the level it reaches. The two differ once compute is priced
        assert gate["oracle_headroom"]["clears_sesoi"] is True
    assert "license different next steps" in closure["null_versus_failure"]


def test_every_gate_declares_activity_and_headroom(closure):
    assert set(closure["results"]) == set(N.GATES)
    for name, gate in closure["results"].items():
        assert "active" in gate["mechanism_activity"], name
        headroom = gate["oracle_headroom"]
        assert "applicable" in headroom, name
        if not headroom["applicable"]:
            assert headroom["reason"], name


def test_no_bed_hands_the_answer_to_the_perceptual_path(closure):
    """Correction C_BED_CONTAINED_ITS_OWN_ANSWER: a scored gate needs a target the input does not carry."""
    fixture = N._stream(8)
    assert any(row["observation"]["label"] != row["outcome"] for row in fixture)
    for gate in ("cross_domain_continuity", "procedural_transfer", "endogenous_allocation"):
        headroom = closure["results"][gate]["oracle_headroom"]
        assert headroom["applicable"] is True
        assert headroom["clears_sesoi"] is True, (gate, headroom)


def test_continuity_never_reinitializes_the_entity(closure):
    gate = closure["results"]["cross_domain_continuity"]
    identity = gate["identity_continuity"]
    assert identity["all_distinct"] is True
    assert identity["checkpoints"] == 4
    assert identity["steps"] == 40, "one entity across all four phases"
    assert gate["interference"] <= closure["sesoi"]


def test_the_classification_never_reaches_a_claim_about_experience(closure):
    verdict = closure["verdict"]
    assert verdict["classification"] in N.CLASSIFICATIONS + ("not_yet_a_scaffold",)
    assert set(verdict["never_claimed"]) == set(N.FORBIDDEN)
    assert "none of them is a claim about experience" in verdict["claim_rule"].lower()
    assert SF.check_claim(json.dumps(verdict)) == []
    # proto nous is not reachable from a closure pass whatever the gates say
    assert "Not available from a closure pass" in verdict["requires_for_next_level"]["functional_or_proto_nous_candidate"]


def test_the_gate_added_no_architecture(closure):
    assert "Nothing was added to the architecture to pass one" in closure["not_a_new_layer"]


def test_the_freeze_names_its_own_door_out(closure):
    freeze = closure["architecture_freeze"]
    assert freeze["frozen"] is True
    assert freeze["what_would_reopen_it"] == closure["verdict"]["requires_for_next_level"]
    # a freeze that also froze the beds would preserve the defect the beds were just repaired for
    assert "bed is an instrument" in freeze["what_may_still_change"]
