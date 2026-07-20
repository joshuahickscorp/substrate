
from __future__ import annotations

import copy

import pytest

from mop.mechanisms.messaging_repair_impl import CONTROL_POLICIES, MECHANISM_POLICY
from mop.mechanisms.messaging_repair_runner import MessagingRepairRunner
from mop.studies import generation1_c3_communication as communication


def _config(seed_count: int = 8) -> dict:
    return communication.build_config(seed_count=seed_count)


def _reseal_config(config: dict) -> None:
    config["config_sha256"] = communication.canonical_sha256(
        {key: value for key, value in config.items() if key != "config_sha256"}
    )


def _reseal_result(result: dict) -> None:
    result["result_sha256"] = communication.canonical_sha256(
        {key: value for key, value in result.items() if key != "result_sha256"}
    )


def test_default_config_is_c2_bound_fresh_disjoint_and_sealed() -> None:
    config = _config()
    communication.validate_config(config)

    assert config["prerequisite"]["c2_complete"] is True
    assert config["prerequisite"]["independent_verification_complete"] is True
    assert config["seed_ranges"]["G1-V1"]["start"] >= communication.FIRST_FRESH_SEED
    assert config["seed_ranges"]["G1-M1"]["start"] >= communication.FIRST_FRESH_SEED
    assert config["scientific_execution_authorized"] is False
    assert config["activation_allowed"] is False
    assert config["scientific_promotion"] is False


def test_exact_lane_controls_and_metrics_are_load_bearing() -> None:
    config = _config()
    assert config["lanes"]["G1-V1"]["controls"] == list(communication.V1_CONTROLS)
    assert config["lanes"]["G1-M1"]["controls"] == list(communication.M1_CONTROLS)
    for lane in config["lanes"].values():
        assert lane["metrics"] == list(communication.AGGREGATE_METRICS)

    mutated = copy.deepcopy(config)
    mutated["lanes"]["G1-V1"]["controls"].reverse()
    _reseal_config(mutated)
    with pytest.raises(communication.CommunicationPilotError, match="authority drifted"):
        communication.validate_config(mutated)


def test_multi_seed_pilot_discriminates_null_from_favorable_without_promotion() -> None:
    config = _config(12)
    result = communication.run_pilot(config)
    communication.validate_result(result)

    for lane_id in ("G1-V1", "G1-M1"):
        lane = result["lanes"][lane_id]
        assert lane["null"]["strict_win_seed_fraction"] == 0.0
        assert lane["favorable"]["strict_win_seed_fraction"] == 1.0
        assert lane["discrimination"]["favorable_minimum_margin"] > 0
        assert lane["discrimination"]["pilot_mechanics_discrimination_passed"] is True
    assert result["decision"]["all_lanes_discriminate_as_constructed"] is True
    assert result["decision"]["scientific_confirmation"] is False
    assert result["independent_scientific_verification_complete"] is False
    assert result["activation_allowed"] is False
    assert result["scientific_promotion"] is False


def test_result_and_digest_are_deterministic() -> None:
    config = _config(5)
    first = communication.run_pilot(config)
    second = communication.run_pilot(config)
    assert first == second
    assert first["result_sha256"] == second["result_sha256"]
    assert len(first["result_sha256"]) == 64


def test_seed_overlap_is_rejected_even_after_resealing() -> None:
    config = _config()
    config["seed_ranges"]["G1-M1"]["start"] = config["seed_ranges"]["G1-V1"]["start"] + 7
    _reseal_config(config)
    with pytest.raises(communication.CommunicationPilotError, match="seed ranges overlap"):
        communication.validate_config(config)


def test_c2_proof_binding_drift_is_rejected_even_after_resealing() -> None:
    config = _config()
    config["prerequisite"]["result_file_sha256"] = "0" * 64
    _reseal_config(config)
    with pytest.raises(communication.CommunicationPilotError, match="proof binding drifted"):
        communication.validate_config(config)


def test_a_leaking_v1_control_fails_the_v1_pilot_closed() -> None:
    leaking = dict(CONTROL_POLICIES)
    leaking["no-verify"] = MECHANISM_POLICY
    runner = MessagingRepairRunner(control_policies=leaking)
    result = communication.run_pilot(_config(4), runner=runner)

    assert result["lanes"]["G1-V1"]["discrimination"][
        "pilot_mechanics_discrimination_passed"
    ] is False
    assert result["decision"]["all_lanes_discriminate_as_constructed"] is False
    assert result["decision"]["scientific_confirmation"] is False


def test_resealed_attempt_to_enable_activation_is_still_rejected() -> None:
    result = communication.run_pilot(_config(3))
    result["activation_allowed"] = True
    _reseal_result(result)
    with pytest.raises(communication.CommunicationPilotError, match="activation flag drifted"):
        communication.validate_result(result, replay=False)


def test_unsealed_result_mutation_is_rejected_before_replay() -> None:
    result = communication.run_pilot(_config(3))
    result["decision"]["scientific_confirmation"] = True
    with pytest.raises(communication.CommunicationPilotError, match="self-seal drifted"):
        communication.validate_result(result, replay=False)
