
from __future__ import annotations

import pytest

from mop.ladder.stage_ladder import (
    CLAIM_SCOPE,
    CURRENT_STAGE_INDEX,
    REQUIRED_MECHANISM_CONTROLS,
    SCIENTIFIC_CAPABILITY_CLAIM,
    SESSION_DISJOINT_AXES,
    STAGE3_FORCING_NULL,
    STAGE4_FORCING_NULL,
    STAGE5_FORCING_NULL,
    STAGE_NAMES,
    STAGE_STATUSES,
    ConfirmationReceipt,
    ControlManifest,
    LadderRefusal,
    MatchedBudget,
    StageActivationGate,
    StageDefinition,
    StageLadder,
    assert_control_manifest,
    build_default_ladder,
    build_stage_definitions,
    coverage,
)

STAGE_LADDER_SCHEMA = "mop-stage-ladder/v1"
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def _budget() -> MatchedBudget:
    return MatchedBudget(params=1_000_000, flops=2_000_000, wall_ns=5_000_000, seeds=5)


def _stage3_receipt(digest: str = _DIGEST_A) -> ConfirmationReceipt:
    return ConfirmationReceipt(
        requirement_id="stage3.confirmed_useful_mechanism",
        digest=digest,
        controls_cleared=REQUIRED_MECHANISM_CONTROLS,
        overturns_null=STAGE3_FORCING_NULL,
        matched=_budget(),
    )


def _stage4_receipt(digest: str) -> ConfirmationReceipt:
    return ConfirmationReceipt(
        requirement_id="stage4.integrated_architecture_advantage",
        digest=digest,
        controls_cleared=REQUIRED_MECHANISM_CONTROLS,
        overturns_null=STAGE4_FORCING_NULL,
        matched=_budget(),
    )


def _stage5_receipts() -> tuple[ConfirmationReceipt, ConfirmationReceipt]:
    validity = ConfirmationReceipt(
        requirement_id="stage5.session_disjoint_validity",
        digest="c" * 64,
        controls_cleared=SESSION_DISJOINT_AXES,
        overturns_null=STAGE5_FORCING_NULL,
        matched=_budget(),
    )
    efficiency = ConfirmationReceipt(
        requirement_id="stage5.measured_efficiency",
        digest="d" * 64,
        controls_cleared=("matched-params", "matched-flops", "measured-wall-clock"),
        overturns_null=STAGE5_FORCING_NULL,
        matched=_budget(),
    )
    return validity, efficiency


def test_ladder_digest_is_stable() -> None:
    ladder = build_default_ladder()
    assert ladder.digest() == build_default_ladder().digest()
    assert len(ladder.digest()) == 64


def test_ladder_starts_at_stage_two() -> None:
    ladder = build_default_ladder()
    assert ladder.current_stage_index == CURRENT_STAGE_INDEX == 2
    assert ladder.claim_scope == CLAIM_SCOPE
    assert SCIENTIFIC_CAPABILITY_CLAIM is False


def test_stage_definitions_are_byte_faithful() -> None:
    stages = build_stage_definitions()
    assert tuple(stage.name for stage in stages) == STAGE_NAMES
    assert tuple(stage.documented_status for stage in stages) == STAGE_STATUSES
    assert stages[3].documented_status == "NOT achieved; first mechanisms failed or are blocked"
    assert stages[4].documented_status == "not entered"
    assert stages[5].documented_status == "not entered"


def test_stage_definition_rejects_out_of_range_index() -> None:
    with pytest.raises(LadderRefusal, match="range 0 to 5"):
        StageDefinition(
            stage_index=6,
            name="rogue",
            documented_status="not entered",
            entry_requirements=(),
            min_mechanism_receipts=0,
            requires_matched_cost=True,
            forcing_null="x",
        )


def test_stage_definition_rejects_widened_claim_scope() -> None:
    with pytest.raises(LadderRefusal, match="claim scope"):
        StageDefinition(
            stage_index=0,
            name=STAGE_NAMES[0],
            documented_status="complete",
            entry_requirements=(),
            min_mechanism_receipts=0,
            requires_matched_cost=False,
            forcing_null="",
            claim_scope="a capability was demonstrated",
        )


def test_complete_stage_cannot_carry_an_unearned_bar() -> None:
    with pytest.raises(LadderRefusal, match="unearned-stage bar"):
        StageDefinition(
            stage_index=1,
            name=STAGE_NAMES[1],
            documented_status="complete",
            entry_requirements=(),
            min_mechanism_receipts=0,
            requires_matched_cost=True,
            forcing_null="some null",
        )


def test_unearned_stage_must_name_a_forcing_null() -> None:
    with pytest.raises(LadderRefusal, match="prior null"):
        StageDefinition(
            stage_index=3,
            name=STAGE_NAMES[3],
            documented_status="NOT achieved; first mechanisms failed or are blocked",
            entry_requirements=("stage3.confirmed_useful_mechanism",),
            min_mechanism_receipts=1,
            requires_matched_cost=True,
            forcing_null="   ",
        )


def test_advance_forbids_skipping() -> None:
    ladder = build_default_ladder()
    with pytest.raises(LadderRefusal, match="skipping is forbidden"):
        ladder.advance(4, (_stage3_receipt(),))


def test_advance_forbids_backward_or_in_place() -> None:
    ladder = build_default_ladder()
    with pytest.raises(LadderRefusal, match="at or below"):
        ladder.advance(2, ())


def test_advance_to_stage3_refuses_without_receipts() -> None:
    ladder = build_default_ladder()
    with pytest.raises(LadderRefusal, match="activation gate is closed"):
        ladder.advance(3, ())


def test_advance_to_stage3_refuses_unmatched_receipt() -> None:
    ladder = build_default_ladder()
    unmatched = ConfirmationReceipt(
        requirement_id="stage3.confirmed_useful_mechanism",
        digest=_DIGEST_A,
        controls_cleared=REQUIRED_MECHANISM_CONTROLS,
        overturns_null=STAGE3_FORCING_NULL,
        matched=None,
    )
    with pytest.raises(LadderRefusal, match="unmet requirements"):
        ladder.advance(3, (unmatched,))


def test_advance_to_stage3_refuses_wrong_null() -> None:
    ladder = build_default_ladder()
    wrong = ConfirmationReceipt(
        requirement_id="stage3.confirmed_useful_mechanism",
        digest=_DIGEST_A,
        controls_cleared=REQUIRED_MECHANISM_CONTROLS,
        overturns_null="an unrelated null",
        matched=_budget(),
    )
    with pytest.raises(LadderRefusal, match="unmet requirements"):
        ladder.advance(3, (wrong,))


def test_advance_to_stage3_refuses_incomplete_controls() -> None:
    ladder = build_default_ladder()
    thin = ConfirmationReceipt(
        requirement_id="stage3.confirmed_useful_mechanism",
        digest=_DIGEST_A,
        controls_cleared=("untrained-control", "matched-sparse"),
        overturns_null=STAGE3_FORCING_NULL,
        matched=_budget(),
    )
    with pytest.raises(LadderRefusal, match="unmet requirements"):
        ladder.advance(3, (thin,))


def test_advance_to_stage3_succeeds_with_valid_receipt() -> None:
    ladder = build_default_ladder()
    advanced = ladder.advance(3, (_stage3_receipt(),))
    assert advanced.current_stage_index == 3
    assert ladder.current_stage_index == 2


def test_advance_is_deterministic_under_same_receipts() -> None:
    def run() -> str:
        return build_default_ladder().advance(3, (_stage3_receipt(),)).digest()

    assert run() == run()


def test_advance_to_stage4_needs_two_confirmed_mechanisms() -> None:
    ladder = build_default_ladder().advance(3, (_stage3_receipt(),))
    with pytest.raises(LadderRefusal, match="confirmed mechanisms"):
        ladder.advance(4, (_stage4_receipt(_DIGEST_A),))
    two = ladder.advance(4, (_stage4_receipt(_DIGEST_A), _stage4_receipt(_DIGEST_B)))
    assert two.current_stage_index == 4


def test_advance_to_stage5_needs_validity_and_efficiency() -> None:
    ladder = (
        build_default_ladder()
        .advance(3, (_stage3_receipt(),))
        .advance(4, (_stage4_receipt(_DIGEST_A), _stage4_receipt(_DIGEST_B)))
    )
    validity, efficiency = _stage5_receipts()
    with pytest.raises(LadderRefusal, match="measured_efficiency"):
        ladder.advance(5, (validity,))
    reached = ladder.advance(5, (validity, efficiency))
    assert reached.current_stage_index == 5


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(LadderRefusal, match="non-vacuous"):
        MatchedBudget(params=0, flops=1, wall_ns=1, seeds=1)


def test_control_manifest_detects_membership_or_order_drift() -> None:
    assert assert_control_manifest().digest() == ControlManifest().digest()
    with pytest.raises(LadderRefusal, match="mechanism control set"):
        ControlManifest(mechanism_controls=("matched-sparse", "untrained-control"))
    with pytest.raises(LadderRefusal, match="session-disjoint axis"):
        ControlManifest(session_axes=("fresh-seed", "fresh-session"))


def test_receipt_digest_is_stable_and_widening_rejected() -> None:
    receipt = _stage3_receipt()
    assert receipt.sha256 == _stage3_receipt().sha256
    assert len(receipt.sha256) == 64
    with pytest.raises(LadderRefusal, match="claim scope"):
        ConfirmationReceipt(
            requirement_id="stage3.confirmed_useful_mechanism",
            digest=_DIGEST_A,
            controls_cleared=REQUIRED_MECHANISM_CONTROLS,
            claim_scope="capability shown",
        )


def test_receipt_rejects_bad_digest() -> None:
    with pytest.raises(LadderRefusal, match="SHA-256"):
        ConfirmationReceipt(
            requirement_id="stage3.confirmed_useful_mechanism",
            digest="not-a-digest",
            controls_cleared=REQUIRED_MECHANISM_CONTROLS,
        )


def test_activation_gate_refuses_by_default() -> None:
    gate = StageActivationGate()
    with pytest.raises(LadderRefusal, match="not earned"):
        gate.authorize_local()


def test_activation_gate_rejects_permissive_construction() -> None:
    with pytest.raises(LadderRefusal, match="never permitted"):
        StageActivationGate(local_activation_permitted=True)


def test_ladder_state_lists_all_stages_and_unmet() -> None:
    state = build_default_ladder().ladder_state()
    assert state["current_stage_index"] == 2
    assert len(state["stages"]) == 6
    reached = [row for row in state["stages"] if row["reached"]]
    assert {row["stage_index"] for row in reached} == {0, 1, 2}
    stage3 = state["stages"][3]
    assert stage3["reached"] is False
    assert stage3["unmet_requirements"]
    assert state["next_stage_index"] == 3
    assert state["next_unmet_requirements"]


def test_ladder_rejects_wrong_stage_count() -> None:
    with pytest.raises(LadderRefusal, match="exactly six stages"):
        StageLadder(stages=build_stage_definitions()[:5])


def test_ladder_rejects_position_below_floor() -> None:
    with pytest.raises(LadderRefusal, match="Stage 2 floor"):
        StageLadder(stages=build_stage_definitions(), current_stage_index=1)


def test_coverage_lists_every_stage() -> None:
    cov = coverage()
    assert set(cov) == {f"stage_{index}" for index in range(6)}
    for bullets in cov.values():
        assert len(bullets) >= 2
