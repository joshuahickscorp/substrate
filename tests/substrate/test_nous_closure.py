from __future__ import annotations

import inspect

from substrate import nous_closure_campaign as campaign
from substrate import nous_closure_config as C
from substrate import nous_closure_experiment as experiment
from substrate import nous_closure_io as io


def _no_true_activation(value: object) -> bool:
    if isinstance(value, dict):
        return all(key != "activation" or child is False for key, child in value.items()) and all(_no_true_activation(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_true_activation(child) for child in value)
    return True


def test_closure_configuration_freezes_exact_constitution_and_claim_boundary() -> None:
    assert tuple(C.FACETS) == tuple(range(1, 21))
    assert tuple(C.HYPOTHESES) == tuple(f"H_NC{index}" for index in range(1, 21))
    assert len(C.SANDBOX_FAMILIES) == 12
    assert set(C.BASELINES) == {f"S{index}" for index in range(8)}
    assert len(C.CANARY_REQUIREMENTS) == 32
    assert len(C.MUTATIONS) == 25
    assert len(C.PRIMARY_DELIVERABLES) == len(set(C.PRIMARY_DELIVERABLES))
    assert C.OUTCOME_A == ("functional_nous_candidate", "nous_external_adjudication_ready")
    assert C.OUTCOME_B == "terminal_closed_null"
    assert C.CLAIM_BOUNDARY["unqualified_nous"] is False
    assert _no_true_activation(C.configuration())


def test_direct_policy_cannot_receive_oracle_target() -> None:
    signature = inspect.signature(experiment.direct_policy_score)
    assert tuple(signature.parameters) == ("observation", "rule")
    _identity, observation, _target = experiment.v5experiment._public_task(
        "test",
        1,
        0,
        0,
    )
    assert isinstance(experiment.direct_policy_score(observation, "median_all"), float)


def test_construction_selection_is_frozen_before_pilot_and_outcome_blind() -> None:
    selection = experiment.construction_selection()
    assert selection["selection_frozen_before_pilot"]
    assert selection["target_available_to_policy"] is False
    assert set(selection["phase_rules"]) == {str(index) for index in range(20)}
    assert set(selection["phase_rules"].values()) <= set(C.DIRECT_POLICY_RULES)


def test_stateful_sandbox_has_active_mechanism_but_monolithic_tie() -> None:
    report = experiment.sandbox_pilot(seeds=range(15_100, 15_104))
    assert report["candidate_mean_accuracy"] == 1.0
    assert report["monolith_mean_accuracy"] == 1.0
    assert report["stateless_mean_accuracy"] == 0.0
    assert report["candidate_minus_stateless"]["passes"]
    assert report["candidate_minus_monolith"]["mean_paired_effect"] == 0.0
    assert report["candidate_minus_monolith"]["confidence_interval_95"] == [0.0, 0.0]
    assert not report["candidate_minus_monolith"]["passes"]
    assert report["classification"] == "mechanism_null"


def test_counterfeit_audit_rejects_every_positive_and_accepts_clean_fixture() -> None:
    audit, fixtures, rejection = campaign.counterfeit_documents()
    assert audit["all_pass"]
    assert audit["passed"] == len(C.COUNTERFEIT_EXPLANATIONS)
    assert rejection["rejected"]
    for name, rows in fixtures["fixtures"].items():
        assert campaign.detect_counterfeit(name, rows["positive"])
        assert not campaign.detect_counterfeit(name, rows["clean_negative"])


def test_paired_zero_is_a_null_not_a_positive() -> None:
    result = experiment.paired_effect(
        {index: 1.0 for index in range(8)},
        {index: 1.0 for index in range(8)},
        identity="exact-tie",
    )
    assert result["mean_paired_effect"] == 0.0
    assert result["confidence_interval_95"] == [0.0, 0.0]
    assert not result["clears_sesoi"]
    assert not result["lower_bound_above_zero"]
    assert not result["passes"]


def test_authority_hash_and_activation_fail_closed() -> None:
    document = io.authority("test/v1", {"value": 1})
    unsigned = dict(document)
    claimed = unsigned.pop("sha256")
    assert io.digest(unsigned) == claimed
    assert not io._contains_true_activation(document)
    assert io._contains_true_activation({"activation": True})


def test_terminal_scorecard_cannot_average_away_critical_null() -> None:
    pilot = experiment.pilot()
    scorecard = campaign._scorecard(pilot)
    assert scorecard["score"] == 10.0
    assert scorecard["critical_failed_facets"] == [20]
    assert not scorecard["all_facets_one"]
    assert not scorecard["zeros_averaged_away"]
    assert scorecard["rows"][19]["status"] == "mechanism_null"
