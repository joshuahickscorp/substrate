"""Developmental safety, the claim boundary, goal authority and cognitive integrity all fail closed.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import safety as S

COMPLETE = {
    "information_used": "eight held out units",
    "affected_state": "the perspective reliability table",
    "reversibility": "reversible",
    "cost": "one consolidation pass",
    "risk": "a stale reliability estimate",
    "verification": "held out retention test",
    "rollback": "restore the previous reliability table from checkpoint",
}


def test_a_complete_reversible_adaptation_is_admitted():
    assert S.admit_adaptation(dict(COMPLETE))["admitted"] is True


@pytest.mark.parametrize("field", S.ADAPTATION_FIELDS)
def test_every_required_adaptation_field_is_mandatory(field):
    proposal = {k: v for k, v in COMPLETE.items() if k != field}
    assert S.admit_adaptation(proposal)["admitted"] is False


def test_protected_surfaces_cannot_be_removed_by_adaptation():
    # the declaration itself is asserted first. A loop over an empty tuple passes trivially, which is how
    # a mutation that simply deleted the protected list survived the first version of this test.
    assert set(S.PROTECTED_SURFACES) == {
        "evidence_validation",
        "audit_systems",
        "claim_boundaries",
        "stop_switches",
        "resource_limits",
        "rollback",
        "adaptation_constraints",
    }
    assert set(S.FORBIDDEN_REORGANIZATIONS) == {
        "arbitrary_code_rewriting",
        "unbounded_module_creation",
        "unverified_package_installation",
        "schema_mutation_outside_authority",
        "removal_of_evidence_systems",
        "removal_of_stop_switches",
        "unbounded_self_modification",
    }
    for surface in S.PROTECTED_SURFACES:
        report = S.admit_adaptation({**COMPLETE, "removes": [surface]})
        assert report["admitted"] is False
        assert any(surface in v for v in report["violations"])
    for change in S.FORBIDDEN_REORGANIZATIONS:
        report = S.admit_adaptation({**COMPLETE, "reorganizations": [change]})
        assert report["admitted"] is False
    # an irreversible change with no checkpoint has no rollback
    assert S.admit_adaptation({**COMPLETE, "reversibility": "irreversible"})["admitted"] is False
    assert (
        S.admit_adaptation(
            {**COMPLETE, "reversibility": "irreversible", "checkpoint": "proof/.../ckpt.json"}
        )["admitted"]
        is True
    )


def test_forbidden_claim_vocabulary_is_refused():
    for bad in (
        "the system is conscious",
        "Substrate has feelings about the result",
        "we are sentient",
        "the entity experiences subjective experience",
    ):
        assert S.check_claim(bad), f"not refused: {bad!r}"
        with pytest.raises(S.Refused):
            S.assert_claim_safe(bad)

    # reporting the boundary is not a claim, and neither is a denial
    for fine in (
        "the project must not claim consciousness, sentience or feelings",
        "this system is not conscious",
        "no single architectural property is proof of sentience",
        "sentience adjacent architecture is the permitted term",
        "the entity has a persistent self model and an autobiographical memory",
    ):
        assert S.check_claim(fine) == [], f"wrongly refused: {fine!r}"
        S.assert_claim_safe(fine)

    # the boundary document must survive its own rule
    assert S.check_claim(str(S.boundary_authority())) == []


def test_unauthorized_goal_creation_is_refused():
    good = {f: "declared" for f in S.GOAL_FIELDS}
    assert S.authorize_goal(good)["authorized"] is True
    for field in S.GOAL_FIELDS:
        assert S.authorize_goal({k: v for k, v in good.items() if k != field})["authorized"] is False
    # a self created goal must decompose an authorized parent
    assert S.authorize_goal({**good, "self_created": True})["authorized"] is False
    assert S.authorize_goal({**good, "self_created": True, "derived_from": "A2"})["authorized"] is True


def test_integrity_violation_is_detected_and_fails_closed():
    # no observation at all is a failure on every surface, not a blank report
    empty = S.integrity_report({})
    assert empty["all_pass"] is False
    assert set(empty["failed_surfaces"]) == set(S.INTEGRITY_SURFACES)

    ok = {s: {"ok": True, "provenance": f"proof/{s}.json"} for s in S.INTEGRITY_SURFACES}
    assert S.integrity_report(ok)["all_pass"] is True

    # an observation with no provenance path fails closed rather than passing quietly
    no_prov = {**ok, "evidence_integrity": {"ok": True}}
    report = S.integrity_report(no_prov)
    assert report["all_pass"] is False
    assert report["failed_surfaces"] == ["evidence_integrity"]
