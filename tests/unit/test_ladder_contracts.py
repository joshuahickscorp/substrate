"""Tests for the ladder run-receipt contract, especially the mechanics vs scientific boundary.

The boundary is the honesty guarantee for the whole "scaffold without results" effort: a toy demonstration
can never be cleared and can never open a stage gate; only a matched, control-cleared, null-overturning
scientific confirmation can. These tests pin that down. No capability is claimed.
"""

from __future__ import annotations

import pytest

from mop.ladder.ladder_contracts import (
    KIND_CONFIRMATION,
    KIND_DEMONSTRATION,
    VERDICT_CLEARED,
    VERDICT_MECHANICS_OK,
    Bed,
    LadderContractRefusal,
    MechanismRunner,
    RunReceipt,
    ladder_readiness,
    mint_confirmation,
    mint_demonstration,
    scientific_confirmations,
    to_confirmation,
)
from mop.ladder.stage_ladder import ConfirmationReceipt, MatchedBudget

DIGEST = "a" * 64


def a_budget() -> MatchedBudget:
    return MatchedBudget(params=1, flops=1, wall_ns=1, seeds=8)


def a_demo(**overrides: object) -> RunReceipt:
    kwargs: dict[str, object] = dict(
        mechanism_id="event_formation",
        stage=3,
        requirement_id="s3.event_formation",
        controls_cleared=("wrong-time", "wrong-event"),
        evidence_digest=DIGEST,
    )
    kwargs.update(overrides)
    return mint_demonstration(**kwargs)  # type: ignore[arg-type]


def a_confirmation(**overrides: object) -> RunReceipt:
    kwargs: dict[str, object] = dict(
        mechanism_id="event_formation",
        stage=3,
        requirement_id="s3.event_formation",
        controls_cleared=("wrong-time", "wrong-event"),
        overturns_null="x0 strong null",
        matched=a_budget(),
        evidence_digest=DIGEST,
    )
    kwargs.update(overrides)
    return mint_confirmation(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The honesty boundary.
# ---------------------------------------------------------------------------


def test_demonstration_defaults_to_null_and_is_not_a_confirmation() -> None:
    receipt = a_demo()
    assert receipt.kind == KIND_DEMONSTRATION
    assert receipt.verdict == "null"
    assert receipt.is_confirmation is False


def test_demonstration_cannot_be_cleared() -> None:
    with pytest.raises(LadderContractRefusal):
        RunReceipt(
            kind=KIND_DEMONSTRATION,
            mechanism_id="event_formation",
            stage=3,
            requirement_id="s3.event_formation",
            verdict=VERDICT_CLEARED,
            controls_cleared=("wrong-time",),
            evidence_digest=DIGEST,
        )


def test_demonstration_mechanics_ok_is_allowed_but_not_a_confirmation() -> None:
    receipt = a_demo(verdict=VERDICT_MECHANICS_OK)
    assert receipt.verdict == VERDICT_MECHANICS_OK
    assert receipt.is_confirmation is False


def test_confirmation_is_cleared_and_bridges_to_ladder() -> None:
    receipt = a_confirmation()
    assert receipt.kind == KIND_CONFIRMATION
    assert receipt.is_confirmation is True
    bridged = to_confirmation(receipt)
    assert isinstance(bridged, ConfirmationReceipt)
    assert bridged.requirement_id == "s3.event_formation"
    assert bridged.overturns_null == "x0 strong null"


def test_demonstration_cannot_bridge_to_a_gate() -> None:
    with pytest.raises(LadderContractRefusal):
        to_confirmation(a_demo())


def test_cleared_confirmation_requires_matched_controls_and_null() -> None:
    with pytest.raises(LadderContractRefusal):
        RunReceipt(
            kind=KIND_CONFIRMATION,
            mechanism_id="event_formation",
            stage=3,
            requirement_id="s3.event_formation",
            verdict=VERDICT_CLEARED,
            controls_cleared=(),  # no controls
            evidence_digest=DIGEST,
            overturns_null="x0 strong null",
            matched=a_budget(),
        )


# ---------------------------------------------------------------------------
# Validation fails closed.
# ---------------------------------------------------------------------------


def test_rejects_widened_claim_scope() -> None:
    with pytest.raises(LadderContractRefusal):
        RunReceipt(
            kind=KIND_DEMONSTRATION,
            mechanism_id="m",
            stage=3,
            requirement_id="s3.m",
            verdict="null",
            controls_cleared=(),
            evidence_digest=DIGEST,
            claim_scope="a capability was shown",
        )


def test_rejects_bad_digest() -> None:
    with pytest.raises(LadderContractRefusal):
        a_demo(evidence_digest="short")


def test_rejects_duplicate_controls() -> None:
    with pytest.raises(LadderContractRefusal):
        a_demo(controls_cleared=("wrong-time", "wrong-time"))


def test_rejects_negative_stage() -> None:
    with pytest.raises(LadderContractRefusal):
        a_demo(stage=-1)


def test_digest_is_stable() -> None:
    assert a_demo().digest() == a_demo().digest()
    assert len(a_demo().digest()) == 64


# ---------------------------------------------------------------------------
# Aggregation: demonstrations never count toward a gate.
# ---------------------------------------------------------------------------


def test_ladder_readiness_counts_only_confirmations() -> None:
    receipts = [a_demo(), a_demo(mechanism_id="niche_dispatch"), a_confirmation()]
    summary = ladder_readiness(receipts)
    assert summary["total_receipts"] == 3
    assert summary["demonstrations"] == 2
    assert summary["scientific_confirmations"] == 1
    assert summary["confirmations_by_stage"] == {"3": 1}


def test_scientific_confirmations_filters_demonstrations() -> None:
    assert scientific_confirmations([a_demo(), a_confirmation()]) == [a_confirmation()]


# ---------------------------------------------------------------------------
# Protocols are structural and runtime-checkable.
# ---------------------------------------------------------------------------


def test_protocols_are_runtime_checkable() -> None:
    class ToyBed:
        mechanism_id = "toy"

        def controls(self) -> tuple[str, ...]:
            return ("c",)

        def matched_cost(self) -> MatchedBudget:
            return a_budget()

        def null_regime(self, seed: int) -> object:
            return {"seed": seed}

        def favorable_regime(self, seed: int) -> object:
            return {"seed": seed}

    class ToyRunner:
        mechanism_id = "toy"

        def run(self, bed: object, seed: int) -> object:
            return {"seed": seed}

        def mint(self, results: object) -> RunReceipt:
            return a_demo()

    assert isinstance(ToyBed(), Bed)
    assert isinstance(ToyRunner(), MechanismRunner)
