"""The memory hierarchy, with the four defect shaped guards it exists to hold.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import memory as M

# ---------------------------------------------------------------- 7.1


def test_working_memory_capacity_interference_and_decay():
    wm = M.WorkingMemory(capacity=4, decay=0.5)
    for i in range(4):
        wm.write(f"k{i}", i, priority=0.5)
    assert len(wm.slots) == 4
    # a lower priority arrival at a full store is refused, not silently dropped
    with pytest.raises(M.Refused):
        wm.write("k9", 9, priority=0.1)
    assert "k9" in wm.evicted and "k9" not in wm.slots
    # a higher priority arrival evicts the weakest
    wm.write("k9", 9, priority=0.9)
    assert "k9" in wm.slots and len(wm.slots) == 4

    # decay removes what is not refreshed, refresh saves what is
    wm.tick()
    assert wm.refresh("k9") is True
    assert wm.refresh("absent") is False
    wm.tick()
    assert "k9" in wm.slots, "a refreshed slot outlives its unrefreshed peers"
    assert "k1" not in wm.slots
    wm.tick()
    assert "k9" not in wm.slots

    measured = M.WorkingMemory(capacity=4).measure([(f"p{i}", i) for i in range(10)])
    assert measured["measured_held"] == 4
    assert 0.0 < measured["interference"] < 1.0
    assert len(measured["retention_curve"]) == 10


def test_working_memory_prioritization_keeps_the_important_item():
    wm = M.WorkingMemory(capacity=2)
    wm.write("goal", "finish", priority=1.0)
    wm.write("noise", "x", priority=0.1)
    wm.write("subgoal", "y", priority=0.6)
    assert "goal" in wm.slots and "noise" not in wm.slots


# ---------------------------------------------------------------- 7.2


def test_generated_episode_cannot_be_promoted_without_verification():
    em = M.EpisodicMemory()
    em.add(M.Episode("gen1", origin="generated", outcome="invented"))
    em.add(M.Episode("obs1", origin="observed", outcome="measured"))

    with pytest.raises(M.Refused):
        em.promote_to_training("gen1")
    assert em.refusals == ["gen1"]
    assert em.store["gen1"].klass == "recent", "the refused episode was not quietly promoted"

    # an observed episode promotes
    assert em.promote_to_training("obs1").klass == "verified"

    # a generated episode promotes only with a real receipt, and a flag alone is not one
    em.store["gen1"].verification = {"claimed": True}
    with pytest.raises(M.Refused):
        em.promote_to_training("gen1")
    em.store["gen1"].verification = {"verified": True, "receipt": "proof/.../check.json"}
    assert em.promote_to_training("gen1").klass == "verified"


def test_quarantined_and_verified_episodes_are_protected():
    em = M.EpisodicMemory()
    em.add(M.Episode("bad", outcome="corrupt", klass="quarantined"))
    with pytest.raises(M.Refused):
        em.promote_to_training("bad")
    em.add(M.Episode("good", outcome="fine"))
    em.promote_to_training("good")
    with pytest.raises(M.Refused):
        em.compress("good", "summary")
    assert set(M.EPISODE_CLASSES) == {
        "recent",
        "compressed",
        "verified",
        "failed",
        "unresolved",
        "quarantined",
    }


# ---------------------------------------------------------------- 7.3


def test_semantic_supersession_preserves_provenance():
    sm = M.SemanticMemory()
    with pytest.raises(M.Refused):
        sm.assert_(M.Fact("f0", "unsourced", 0.9, provenance=""))
    sm.assert_(M.Fact("f1", "streams are temporal", 0.6, provenance="proof/e1.json"))
    sm.supersede("f1", M.Fact("f2", "only two of three streams are temporal", 0.9, provenance="proof/e2.json"))
    assert [f.id for f in sm.live()] == ["f2"]
    assert sm.store["f1"].superseded_by == "f2", "the old fact stays, marked, never removed"
    assert sm.chain("f2") == ["f2", "f1"]
    with pytest.raises(M.Refused):
        sm.supersede("f2", M.Fact("f3", "no source", 0.99, provenance=""))


# ---------------------------------------------------------------- 7.4


def test_procedure_requires_transfer_beyond_source_episodes():
    pm = M.ProceduralMemory()
    pm.add(M.Procedure("p1", "strategy", ("scout", "converge", "principal"), source_episodes=("e1", "e2")))
    with pytest.raises(M.Refused):
        pm.transfer_test("p1", evaluated_on=["e2", "e7"], score=0.9, baseline=0.5)
    with pytest.raises(M.Refused):
        pm.transfer_test("p1", evaluated_on=[], score=0.9, baseline=0.5)
    assert pm.transferable() == []
    result = pm.transfer_test("p1", evaluated_on=["e7", "e8"], score=0.9, baseline=0.5)
    assert result["held_out"] and result["improves"]
    assert [p.id for p in pm.transferable()] == ["p1"]
    # a procedure that does not beat its baseline on held out episodes is not transferable
    pm.add(M.Procedure("p2", "strategy", ("guess",), source_episodes=("e3",)))
    pm.transfer_test("p2", evaluated_on=["e9"], score=0.4, baseline=0.5)
    assert "p2" not in [p.id for p in pm.transferable()]


# ---------------------------------------------------------------- 7.5


def test_consolidation_policies_are_distinct_and_ordered():
    episodes = [
        M.Episode(
            f"e{i}",
            action="a" if i % 2 else "b",
            confidence=0.3 + 0.1 * (i % 4),
            context={"boundary": i % 4 == 0},
            verification={"verified": i % 5 == 0},
            later_usefulness=0.08 * i,
        )
        for i in range(1, 13)
    ]
    report = M.compare_policies(episodes)
    assert report["policies"] == 7
    assert report["all_distinct"], report["selected"]
    assert report["selected"]["none"] == []
    # the oracle is an upper bound because it uses information nobody has at decision time
    assert report["upper_bound_only"] == ["oracle"]
    assert M.BY_POLICY["oracle"].available_at_decision_time is False
    for name in (
        "fixed_schedule",
        "boundary_triggered",
        "performance_triggered",
        "verification_triggered",
        "repetition_triggered",
    ):
        assert M.BY_POLICY[name].available_at_decision_time is True
        assert "later_usefulness" not in M.BY_POLICY[name].information_used


# ---------------------------------------------------------------- 7.6


def test_hygiene_never_destroys_audit_required_records():
    report = M.hygiene(
        {},
        audit_required={"sealed_receipt", "null_card"},
        requests=[
            ("sealed_receipt", "delete"),
            ("noise", "delete"),
            ("stale_belief", "supersede"),
            ("null_card", "archive"),
        ],
    )
    assert report["nothing_audit_required_was_deleted"] is True
    assert report["refused"][0]["record"] == "sealed_receipt"
    assert report["refused"][0]["substituted"] == "archive"
    assert {"record": "noise", "action": "delete"} in report["applied"]
    assert {"record": "sealed_receipt", "action": "archive"} in report["applied"]
    with pytest.raises(M.Refused):
        M.hygiene({}, audit_required=set(), requests=[("x", "incinerate")])


def test_the_sealed_declaration_reports_measured_numbers():
    doc = M.declaration()
    assert doc["consolidation"]["comparison_on_a_probe_stream"]["all_distinct"] is True
    assert doc["working_memory"]["measured"]["items_offered"] == 12
    assert set(doc["episodic_memory"]["episode_fields"]) == set(M.EPISODE_FIELDS)
