from __future__ import annotations

import pytest

from mop.mechanisms.intervention_simulation_scaffold import (
    CLAIM_SCOPE,
    INTERVENTION_CONTROLS,
    INTERVENTION_NULL,
    INTERVENTION_SIM_SCHEMA,
    NOVELTY_CONTROLS,
    NOVELTY_NULL,
    PLANNING_NULL,
    SCIENTIFIC_CAPABILITY_CLAIM,
    SIMULATION_CONTROLS,
    UNCERTAINTY_CONTROLS,
    UNCERTAINTY_NULL,
    CalibratedUncertaintyContract,
    ControlWinOutcome,
    DeploymentActivationGate,
    EpochScaffold,
    InterventionContract,
    InterventionSimulationRefusal,
    MatchedBudget,
    ReducibleNoveltyContract,
    SimulationForActionContract,
    assert_control_registry_intact,
    build_epoch_scaffold,
    control_registry_digest,
    default_activation_gate,
    default_intervention_contract,
    default_novelty_contract,
    default_simulation_contract,
    default_uncertainty_contract,
    deterministic_unit_score,
    evaluate_control_win,
    require_control_win,
)


def test_capability_claim_is_false() -> None:
    assert SCIENTIFIC_CAPABILITY_CLAIM is False
    assert CLAIM_SCOPE == "deterministic programmatic mechanics only; no capability or natural-data claim"


def test_epoch_scaffold_digest_is_stable() -> None:
    first = build_epoch_scaffold().sha256
    second = build_epoch_scaffold().sha256
    assert first == second
    assert len(first) == 64


def test_contract_digests_are_stable() -> None:
    assert default_intervention_contract().digest() == default_intervention_contract().digest()
    assert default_simulation_contract().digest() == default_simulation_contract().digest()
    assert default_uncertainty_contract().digest() == default_uncertainty_contract().digest()
    assert default_novelty_contract().digest() == default_novelty_contract().digest()


def test_matched_budget_must_be_non_vacuous() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="non-vacuous"):
        MatchedBudget(params=0, flops=1, memory_bytes=1, rollout_steps=1)


@pytest.mark.parametrize(
    "factory",
    [
        default_intervention_contract,
        default_simulation_contract,
        default_uncertainty_contract,
        default_novelty_contract,
    ],
)
def test_contracts_refuse_when_matched_cost_not_required(factory: object) -> None:
    payload = factory().payload()  # type: ignore[operator]
    assert payload["matched_cost_required"] is True


def test_intervention_refuses_without_matched_cost() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="matched full-system cost"):
        InterventionContract(
            do_operator_arm="do-operator-arm",
            observational_control="observational-only",
            controls=INTERVENTION_CONTROLS,
            prior_null=INTERVENTION_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=False,
            margin_required=0.01,
        )


def test_simulation_refuses_without_matched_cost() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="matched full-system cost"):
        SimulationForActionContract(
            policy_id="rollout-value-policy",
            controls=SIMULATION_CONTROLS,
            rollout_horizon=4,
            value_margin_required=0.05,
            prior_null=PLANNING_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=False,
        )


def test_control_registry_digest_is_stable() -> None:
    assert control_registry_digest() == control_registry_digest()
    assert_control_registry_intact()


def test_intervention_rejects_control_order_drift() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="controls or order drift"):
        InterventionContract(
            do_operator_arm="do-operator-arm",
            observational_control="observational-only",
            controls=("backdoor-adjusted", "observational-only"),
            prior_null=INTERVENTION_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
            margin_required=0.01,
        )


def test_uncertainty_rejects_control_membership_drift() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="controls or order drift"):
        CalibratedUncertaintyContract(
            reliability_metric="expected-calibration-error",
            controls=("overconfident",),
            calibration_bins=10,
            max_calibration_error=0.1,
            reliability_margin_required=0.02,
            prior_null=UNCERTAINTY_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
        )


def test_simulation_bound_to_planning_null() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="P7 planning null"):
        SimulationForActionContract(
            policy_id="rollout-value-policy",
            controls=SIMULATION_CONTROLS,
            rollout_horizon=4,
            value_margin_required=0.05,
            prior_null="a-weaker-null",
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
        )


def test_novelty_bound_to_noise_seeking_null() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="noise-seeking null"):
        ReducibleNoveltyContract(
            curiosity_signal="information-gain-signal",
            novelty_target="reducible",
            reducibility_metric="learning-progress",
            controls=NOVELTY_CONTROLS,
            curiosity_margin_required=0.03,
            prior_null="something-else",
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
        )


def test_novelty_refuses_irreducible_noise_seeking() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="irreducible-noise seeking refused"):
        ReducibleNoveltyContract(
            curiosity_signal="information-gain-signal",
            novelty_target="aleatoric-noise",
            reducibility_metric="learning-progress",
            controls=NOVELTY_CONTROLS,
            curiosity_margin_required=0.03,
            prior_null=NOVELTY_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
        )


def test_novelty_refuses_unknown_target() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="unsupported novelty target"):
        ReducibleNoveltyContract(
            curiosity_signal="information-gain-signal",
            novelty_target="epistemic",
            reducibility_metric="learning-progress",
            controls=NOVELTY_CONTROLS,
            curiosity_margin_required=0.03,
            prior_null=NOVELTY_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
        )


def test_intervention_rejects_widened_claim_scope() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="claim scope cannot be widened"):
        InterventionContract(
            do_operator_arm="do-operator-arm",
            observational_control="observational-only",
            controls=INTERVENTION_CONTROLS,
            prior_null=INTERVENTION_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
            margin_required=0.01,
            claim_scope="a capability was demonstrated",
        )


def test_uncertainty_rejects_bad_schema() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="unsupported intervention-simulation schema"):
        CalibratedUncertaintyContract(
            reliability_metric="expected-calibration-error",
            controls=UNCERTAINTY_CONTROLS,
            calibration_bins=10,
            max_calibration_error=0.1,
            reliability_margin_required=0.02,
            prior_null=UNCERTAINTY_NULL,
            matched=MatchedBudget(params=1, flops=1, memory_bytes=1, rollout_steps=1),
            matched_cost_required=True,
            schema="mop-wrong/v9",
        )


def test_deterministic_unit_score_is_reproducible() -> None:
    a = deterministic_unit_score(seed=7, label="do-operator-arm")
    b = deterministic_unit_score(seed=7, label="do-operator-arm")
    assert a == b
    assert 0.0 <= a < 1.0


def test_deterministic_unit_score_rejects_negative_seed() -> None:
    with pytest.raises(InterventionSimulationRefusal):
        deterministic_unit_score(seed=-1, label="do-operator-arm")


def test_evaluate_control_win_is_deterministic() -> None:
    first = evaluate_control_win(
        arm_id="do-operator-arm", control_id="observational-only", seed=3, margin_required=0.01
    )
    second = evaluate_control_win(
        arm_id="do-operator-arm", control_id="observational-only", seed=3, margin_required=0.01
    )
    assert isinstance(first, ControlWinOutcome)
    assert first.payload() == second.payload()


def test_control_win_outcome_refuses_self_comparison() -> None:
    with pytest.raises(InterventionSimulationRefusal, match="cannot be compared against itself"):
        ControlWinOutcome(
            arm_id="same-arm",
            control_id="same-arm",
            arm_score=0.9,
            control_score=0.1,
            margin_required=0.1,
        )


def test_require_control_win_refuses_when_arm_loses() -> None:
    outcome = ControlWinOutcome(
        arm_id="do-operator-arm",
        control_id="observational-only",
        arm_score=0.10,
        control_score=0.50,
        margin_required=0.05,
    )
    assert outcome.beats_control is False
    with pytest.raises(InterventionSimulationRefusal, match="not rejected"):
        require_control_win(outcome, null_name=INTERVENTION_NULL)


def test_require_control_win_accepts_a_clear_win() -> None:
    outcome = ControlWinOutcome(
        arm_id="do-operator-arm",
        control_id="observational-only",
        arm_score=0.90,
        control_score=0.10,
        margin_required=0.05,
    )
    assert outcome.beats_control is True
    require_control_win(outcome, null_name=INTERVENTION_NULL)


def test_require_control_win_rejects_unknown_null() -> None:
    outcome = default_intervention_contract().evaluate(seed=1)
    with pytest.raises(InterventionSimulationRefusal, match="unknown prior null"):
        require_control_win(outcome, null_name="not-a-real-null")


def test_activation_gate_refuses_by_default() -> None:
    gate = default_activation_gate()
    with pytest.raises(InterventionSimulationRefusal, match="off by default"):
        gate.authorize()


def test_activation_gate_refuses_without_receipt() -> None:
    gate = DeploymentActivationGate(preregistration_digest="a" * 64, activation_requested=True)
    with pytest.raises(InterventionSimulationRefusal, match="confirmation receipt"):
        gate.authorize()


def test_activation_gate_refuses_mismatched_receipt() -> None:
    gate = DeploymentActivationGate(
        preregistration_digest="a" * 64,
        activation_requested=True,
        confirmation_receipt="b" * 64,
    )
    with pytest.raises(InterventionSimulationRefusal, match="does not match"):
        gate.authorize()


def test_activation_gate_accepts_matching_receipt() -> None:
    digest = "c" * 64
    gate = DeploymentActivationGate(
        preregistration_digest=digest,
        activation_requested=True,
        confirmation_receipt=digest,
    )
    gate.authorize()


def test_default_contracts_carry_bound_nulls() -> None:
    assert default_intervention_contract().prior_null == INTERVENTION_NULL
    assert default_simulation_contract().prior_null == PLANNING_NULL
    assert default_uncertainty_contract().prior_null == UNCERTAINTY_NULL
    assert default_novelty_contract().prior_null == NOVELTY_NULL


def test_epoch_scaffold_schema_pin() -> None:
    scaffold = build_epoch_scaffold()
    assert isinstance(scaffold, EpochScaffold)
    assert scaffold.payload()["schema"] == INTERVENTION_SIM_SCHEMA
