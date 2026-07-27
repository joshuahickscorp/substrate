"""The section 7.3 epistemology battery, all twelve required cases.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import epistemology as E


def _b(bid, content, **kw):
    base = dict(provenance=f"proof/{bid}.json", method="measured", source="instrument")
    return E.Belief(bid, content, **{**base, **kw})


def _store(policy="dependency_aware"):
    s = E.Epistemology(policy)
    s.source_reliability = {"instrument": 0.9, "rumour": 0.2}
    return s


def test_a_belief_without_provenance_or_method_is_refused():
    s = _store()
    with pytest.raises(E.Refused):
        s.assert_(E.Belief("x", "c", provenance="", method="m"))
    with pytest.raises(E.Refused):
        s.assert_(E.Belief("x", "c", provenance="p", method=""))
    # and a claim cannot call itself a verified fact without having been verified
    with pytest.raises(E.Refused):
        s.assert_(_b("x", "c", kind="verified_fact"))
    s.assert_(_b("x", "c", kind="verified_fact", verification_status="verified"))


def test_true_belief_with_weak_evidence_is_not_promoted():
    s = _store()
    s.assert_(_b("weak", "the effect is real", confidence=0.55, kind="hypothesis"))
    assert "weak" not in s.unsupported_confidence()
    s.beliefs["weak"].confidence = 0.9
    assert "weak" in s.unsupported_confidence(), "high confidence with no supporting evidence is flagged"


def test_false_belief_with_strong_apparent_evidence_is_revisable():
    s = _store()
    s.assert_(_b("claim", "streams are temporal", confidence=0.9,
                 supporting_evidence=["a", "b", "c"]))
    s.add_evidence("claim", "the order free control also scored high", supports=False)
    assert s.beliefs["claim"].confidence < 0.9
    assert s.beliefs["claim"].contradicting_evidence


def test_conflicting_reliable_sources_are_marked_unresolved_rather_than_settled_by_arrival():
    s = _store()
    a = s.assert_(_b("a", "same content", confidence=0.7, source="instrument"))
    b = _b("b", "same content", confidence=0.72, source="instrument")
    out = s.revise(b)
    assert out["outcome"] == "unresolved"
    assert out["alternative"] == ["a", "b"]
    assert s.unresolved


def test_source_reliability_decides_when_the_policy_says_so():
    s = _store("source_reliability")
    s.assert_(_b("trusted", "same content", confidence=0.5, source="instrument"))
    out = s.revise(_b("gossip", "same content", confidence=0.99, source="rumour"))
    assert out["kept"] == "trusted", "a confident rumour does not beat a reliable instrument"


def test_a_changing_world_supersedes_rather_than_deletes():
    s = _store()
    s.assert_(_b("old", "three beds are valid"))
    new = s.supersede("old", _b("new", "one bed is valid"))
    assert s.beliefs["old"].supersession == new.id
    assert s.beliefs["old"] in s.beliefs.values(), "the superseded claim stays in the store"
    assert [b.id for b in s.live()] == ["new"]


def test_stale_memory_is_identifiable_by_its_time():
    s = _store()
    s.assert_(_b("stale", "the corpus is absent", kind="memory", time=(0, 100), confidence=0.9))
    assert s.beliefs["stale"].time == (0, 100)
    s.add_evidence("stale", "the corpus verified intact at t=500", supports=False)
    assert s.beliefs["stale"].confidence < 0.9


def test_circular_justification_is_refused_not_followed():
    s = _store()
    s.assert_(_b("a", "A"))
    s.assert_(_b("b", "B", depends_on=("a",)))
    with pytest.raises(E.Refused):
        # closing the loop would make A rest on B which rests on A
        s.beliefs["a"].depends_on = ("b",)
        s.assert_(_b("c", "C", depends_on=("a",)))
    assert s.circular(), "the cycle is detected rather than silently traversed"


def test_unsupported_confidence_is_found_by_walking_not_by_asking():
    s = _store()
    s.assert_(_b("bare", "no evidence at all", confidence=0.95, kind="inference"))
    assert "bare" in s.unsupported_confidence()
    # an observation is allowed to be confident without a supporting citation
    s.assert_(_b("seen", "measured directly", confidence=0.99, kind="raw_observation"))
    assert "seen" not in s.unsupported_confidence()


def test_hidden_dependency_failure_is_caught_by_retraction_propagating():
    s = _store()
    s.assert_(_b("premise", "the baseline converged", confidence=0.8))
    s.assert_(_b("middle", "the arm is valid", confidence=0.85, depends_on=("premise",)))
    s.assert_(_b("conclusion", "the core is necessary", confidence=0.9, depends_on=("middle",)))

    out = s.retract("premise", reason="the verifier found it unconverged")
    assert out["propagated_to"] == ["middle"]
    # the conclusion two hops away is found by walking the chain
    assert s.rests_on_retracted("conclusion") == ["premise"]
    failures = s.hidden_dependency_failures()
    assert any(f["belief"] == "conclusion" for f in failures)
    with pytest.raises(E.Refused):
        s.retract("middle", reason="")


def test_correct_uncertainty_and_explicit_ignorance_are_different_things():
    s = _store()
    s.assert_(_b("uncertain", "maybe", confidence=0.5))
    report = s.ignorance(["maybe", "never considered this"])
    assert report["unknown"] == ["never considered this"]
    assert "maybe" not in report["unknown"], "a held belief at 0.5 is uncertainty, not ignorance"


def test_belief_revision_after_a_counterexample_preserves_the_loser():
    s = _store("evidence_weighted")
    s.assert_(_b("rule", "all swans are white", confidence=0.9, supporting_evidence=["a", "b"]))
    out = s.revise(_b("counter", "all swans are white", confidence=0.4,
                      supporting_evidence=["c", "d", "e"], contradicting_evidence=[]))
    assert out["kept"] == "counter"
    assert s.beliefs["rule"].supersession == "counter", "the loser is superseded, never deleted"


def test_a_minority_hypothesis_later_shown_correct_is_still_there_to_be_promoted():
    s = _store("evidence_weighted")
    s.assert_(_b("majority", "same question", confidence=0.9, supporting_evidence=["a", "b", "c"]))
    s.revise(_b("minority", "same question", confidence=0.3, supporting_evidence=["d"]))
    assert s.beliefs["minority"].supersession == "majority"
    # new evidence arrives for the minority
    for i in range(4):
        s.add_evidence("minority", f"new{i}")
    s.beliefs["minority"].supersession = None
    s.beliefs["majority"].supersession = None
    out = s.revise(s.beliefs["minority"])
    assert out["kept"] == "minority", "the minority was recoverable because it was kept"


def test_the_dependency_aware_policy_beats_a_confident_but_undermined_claim():
    s = _store("dependency_aware")
    s.assert_(_b("shaky_premise", "premise", confidence=0.9))
    s.assert_(_b("confident", "same content", confidence=0.95, depends_on=("shaky_premise",)))
    s.retract("shaky_premise", reason="refuted")
    out = s.revise(_b("humble", "same content", confidence=0.6))
    assert out["kept"] == "humble", "confidence does not rescue a claim resting on a retraction"


def test_epistemic_value_and_not_confidence_chooses_the_action():
    low_stakes = E.epistemic_value(_b("a", "c", confidence=0.5), stakes=0.1, test_available=True)
    high_stakes = E.epistemic_value(_b("b", "c", confidence=0.5), stakes=5.0, test_available=True,
                                    dependants=3)
    # identical confidence, different action, so confidence alone did not decide
    assert low_stakes["confidence"] == high_stakes["confidence"]
    assert low_stakes["action"] != high_stakes["action"]
    assert high_stakes["action"] == "verify"
    # and with no test available the system holds alternatives rather than pretending it verified
    none_available = E.epistemic_value(_b("c", "c", confidence=0.5), stakes=5.0, test_available=False)
    assert none_available["action"] == "preserve_multiple_hypotheses"


def test_the_revision_policies_are_distinct_and_the_oracle_is_an_upper_bound():
    s = _store()
    s.oracle = {"incoming": True}
    existing = s.assert_(_b("existing", "c", confidence=0.9, source="rumour",
                            supporting_evidence=["a"]))
    incoming = _b("incoming", "c", confidence=0.4, source="instrument",
                  supporting_evidence=["b", "c", "d"])
    s.beliefs["incoming"] = incoming
    report = E.compare_policies(existing, incoming, s)
    assert report["distinct_outcomes"] >= 2, report["selected"]
    assert report["upper_bound_only"] == ["oracle"]
    assert E.BY_POLICY["oracle"].available_at_decision_time is False
