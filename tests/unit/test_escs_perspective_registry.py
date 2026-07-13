from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from mop.escs import (
    CandidateInterface,
    EffectBoundary,
    EvidenceReference,
    EvidenceStanding,
    IntegrationDisposition,
    PerspectiveCandidateRegistry,
    PerspectiveFacet,
    PerspectiveGuard,
    TriggerAuthority,
    load_perspective_candidate_registry,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPOSITORY_ROOT / "configs/experiment/escs_perspective_candidates.json"


def _payload() -> dict[str, object]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _candidate_row(payload: dict[str, object], facet: PerspectiveFacet) -> dict[str, object]:
    rows = payload["candidates"]
    assert isinstance(rows, list)
    return next(row for row in rows if isinstance(row, dict) and row["facet"] == facet.value)


def test_catalog_is_complete_disabled_nonpromotable_and_evidence_resolved() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)

    assert len(registry.candidates) == len(PerspectiveFacet) == 31
    assert {candidate.facet for candidate in registry.candidates} == set(PerspectiveFacet)
    assert not registry.default_activation_enabled
    assert not registry.scientific_promotion_allowed
    assert all(not candidate.activation_enabled for candidate in registry.candidates)
    assert all(not candidate.scientific_promotion_allowed for candidate in registry.candidates)
    assert registry.missing_evidence_paths(REPOSITORY_ROOT) == ()
    assert PerspectiveCandidateRegistry.from_payload(registry.payload()) == registry
    assert len(registry.sha256) == 64


def test_evidence_standing_strictly_determines_integration_disposition() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    expected = {
        EvidenceStanding.MECHANICS: IntegrationDisposition.INFRASTRUCTURE,
        EvidenceStanding.TOY_POSITIVE: IntegrationDisposition.FEATURE_FLAGGED,
        EvidenceStanding.NULL: IntegrationDisposition.CONTROL_ONLY,
        EvidenceStanding.FAILED: IntegrationDisposition.EXCLUDED,
        EvidenceStanding.PENDING: IntegrationDisposition.SANDBOX_STUB,
        EvidenceStanding.BLOCKED: IntegrationDisposition.SANDBOX_STUB,
        EvidenceStanding.UNTESTED: IntegrationDisposition.SANDBOX_STUB,
    }

    for candidate in registry.candidates:
        assert candidate.integration_disposition is expected[candidate.evidence_standing]
        if candidate.integration_disposition is IntegrationDisposition.EXCLUDED:
            assert candidate.effect_boundary is EffectBoundary.NONE
            assert candidate.trigger_authority is TriggerAuthority.NONE
        if candidate.integration_disposition in {
            IntegrationDisposition.CONTROL_ONLY,
            IntegrationDisposition.SANDBOX_STUB,
        }:
            assert candidate.effect_boundary is not EffectBoundary.CHASSIS_COMMITMENT


def test_uncertainty_novelty_and_curiosity_cannot_authorize_work() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    required = {
        PerspectiveGuard.NOISY_TV,
        PerspectiveGuard.REDUCIBLE_ERROR,
        PerspectiveGuard.DECISION_VALUE,
    }

    for facet in (
        PerspectiveFacet.UNCERTAINTY_ESTIMATION,
        PerspectiveFacet.NOVELTY_DETECTION,
        PerspectiveFacet.CURIOSITY,
    ):
        candidate = registry.candidate_for(facet)
        assert candidate.trigger_authority is TriggerAuthority.NONE
        assert candidate.effect_boundary is not EffectBoundary.CHASSIS_COMMITMENT
        assert required <= set(candidate.required_guards)

    payload = _payload()
    row = _candidate_row(payload, PerspectiveFacet.UNCERTAINTY_ESTIMATION)
    row["trigger_authority"] = "event-authorized"
    with pytest.raises(ValueError, match="cannot independently trigger"):
        PerspectiveCandidateRegistry.from_payload(payload)

    payload = _payload()
    row = _candidate_row(payload, PerspectiveFacet.NOVELTY_DETECTION)
    guards = row["required_guards"]
    assert isinstance(guards, list)
    guards.remove("noisy-tv")
    with pytest.raises(ValueError, match="require noisy-TV"):
        PerspectiveCandidateRegistry.from_payload(payload)

    payload = _payload()
    row = _candidate_row(payload, PerspectiveFacet.CURIOSITY)
    row["trigger_authority"] = "decision-value-gated"
    with pytest.raises(ValueError, match="cannot trigger work"):
        PerspectiveCandidateRegistry.from_payload(payload)


def test_simulation_and_imagination_are_counterfactual_only() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    required = {
        PerspectiveGuard.BOUNDED_ACTIVATION,
        PerspectiveGuard.COUNTERFACTUAL_BRANCH,
        PerspectiveGuard.NO_FACTUAL_EFFECT,
        PerspectiveGuard.QUIESCENCE,
    }

    for facet in (PerspectiveFacet.SIMULATION, PerspectiveFacet.IMAGINATION):
        candidate = registry.candidate_for(facet)
        assert candidate.interface is CandidateInterface.ENDOGENOUS_HYPOTHESIS
        assert candidate.effect_boundary is EffectBoundary.COUNTERFACTUAL_ONLY
        assert required <= set(candidate.required_guards)

    payload = _payload()
    row = _candidate_row(payload, PerspectiveFacet.SIMULATION)
    row["effect_boundary"] = "chassis-commitment"
    row["interface"] = "chassis-action"
    with pytest.raises(ValueError, match="counterfactual-only"):
        PerspectiveCandidateRegistry.from_payload(payload)


def test_only_feature_flagged_chassis_candidates_may_describe_factual_effects() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    factual = tuple(
        candidate
        for candidate in registry.candidates
        if candidate.effect_boundary is EffectBoundary.CHASSIS_COMMITMENT
    )

    assert {candidate.facet for candidate in factual} == {
        PerspectiveFacet.MOTOR_REASONING,
        PerspectiveFacet.PLANNING,
    }
    for candidate in factual:
        assert candidate.integration_disposition is IntegrationDisposition.FEATURE_FLAGGED
        assert candidate.interface is CandidateInterface.CHASSIS_ACTION
        assert {
            PerspectiveGuard.ATOMIC_ROLLBACK,
            PerspectiveGuard.CHASSIS_COMMITMENT,
            PerspectiveGuard.EXTERNAL_CONSEQUENCE,
        } <= set(candidate.required_guards)
        assert not candidate.activation_enabled


def test_invalid_standing_activation_promotion_and_evidence_paths_fail_closed() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    null_candidate = registry.candidate_for(PerspectiveFacet.VERIFICATION)
    with pytest.raises(ValueError, match="null evidence requires control-only"):
        replace(null_candidate, integration_disposition=IntegrationDisposition.FEATURE_FLAGGED)
    with pytest.raises(ValueError, match="must be the boolean false"):
        replace(null_candidate, activation_enabled=True)
    with pytest.raises(ValueError, match="must be the boolean false"):
        replace(null_candidate, scientific_promotion_allowed=True)

    reference = null_candidate.evidence_refs[0]
    with pytest.raises(ValueError, match="repository-relative"):
        replace(reference, artifact_path="../forged.json")
    with pytest.raises(ValueError, match="must live under"):
        EvidenceReference("configs/experiment/forged.json", "$.verdict", "forged")


def test_registry_contracts_are_immutable() -> None:
    registry = load_perspective_candidate_registry(REGISTRY_PATH)
    candidate = registry.candidate_for(PerspectiveFacet.VISUAL_REASONING)
    with pytest.raises(FrozenInstanceError):
        candidate.activation_enabled = True  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        registry.scientific_promotion_allowed = True  # type: ignore[misc]


def test_registry_rejects_missing_facets_and_extra_fields() -> None:
    payload = _payload()
    rows = payload["candidates"]
    assert isinstance(rows, list)
    rows.pop()
    with pytest.raises(ValueError, match="omits requested facets"):
        PerspectiveCandidateRegistry.from_payload(payload)

    payload = copy.deepcopy(_payload())
    payload["activation_policy"] = "on"
    with pytest.raises(ValueError, match="fields mismatch"):
        PerspectiveCandidateRegistry.from_payload(payload)
