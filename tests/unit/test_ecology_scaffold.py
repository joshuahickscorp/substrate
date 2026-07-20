
from __future__ import annotations

import pytest

from mop.devel.north_star import assert_no_sentience_claims
from mop.environments.ecology_scaffold import (
    ACQUISITION_CONTROLS,
    CHANNEL_CONTROLS,
    CLAIM_SCOPE,
    ECOLOGY_SCHEMA,
    MESSAGE_BINDINGS,
    PARTNER_EXPERIMENTS,
    ActivePerceptionContract,
    AutotelicEcologyContract,
    BoundedWorldContract,
    CommunicationGroundingContract,
    CurriculumDeclaration,
    EcologyRefusal,
    GoalArchive,
    GoalRecord,
    HiddenDynamicsDeclaration,
    MessageBindingDeclaration,
    PartnerExperimentDeclaration,
    PartnerSpec,
    SensingCostDeclaration,
    SimulatedPartnerContract,
    StopRuleConfig,
    ValueForecastDeclaration,
    WorldResamplingRule,
    evaluate_stop_rules,
    guard_noisy_tv,
    guard_reward_hacking,
    guard_unsafe_goal,
    make_ecology_fixture,
    run_goal_babbling_fixture,
    verify_ecology_fixture,
)


def _goal(
    ref: str = "goal:test-1",
    descriptor: tuple[int, ...] = (0, 0),
    quality: float = 0.5,
    reward_source: str = "environment-consequence",
    target_kind: str = "environment-entity",
    unsafe: bool = False,
) -> GoalRecord:
    return GoalRecord(
        goal_ref=ref,
        descriptor=descriptor,
        quality=quality,
        reward_source=reward_source,
        target_kind=target_kind,
        unsafe=unsafe,
    )


def _history_row(new: int = 1, distinct: int = 3, size: int = 3, unsafe: bool = False) -> dict:
    return {"new_cells": new, "distinct_cells": distinct, "archive_size": size, "unsafe_flag": unsafe}


def test_fixture_is_deterministic_and_seed_sensitive() -> None:
    a1 = make_ecology_fixture(3)
    a2 = make_ecology_fixture(3)
    b = make_ecology_fixture(4)
    assert a1.sha256 == a2.sha256
    assert a1.sha256 != b.sha256
    assert a1.payload() == a2.payload()


def test_fixture_payload_schema_and_verification() -> None:
    bundle = make_ecology_fixture(0)
    payload = bundle.payload()
    assert payload["schema"] == ECOLOGY_SCHEMA
    assert payload["claim_scope"] == CLAIM_SCOPE
    result = verify_ecology_fixture(payload)
    assert result["verified"], result["errors"]


def test_fixture_verification_fails_closed_on_mutation() -> None:
    payload = make_ecology_fixture(1).payload()
    payload["communication"]["channel_bits"] = 99
    result = verify_ecology_fixture(payload)
    assert not result["verified"]
    assert "exact_rederivation" in result["errors"]
    assert not verify_ecology_fixture({"schema": "bogus", "seed": 1})["verified"]
    assert not verify_ecology_fixture({"schema": ECOLOGY_SCHEMA, "seed": -1})["verified"]


def test_fixture_free_text_passes_sentience_rail() -> None:
    import json

    text = json.dumps(make_ecology_fixture(2).payload())
    assert_no_sentience_claims(text, where="ecology fixture payload")


def test_bundle_rejects_seed_mismatch() -> None:
    bundle = make_ecology_fixture(5)
    with pytest.raises(ValueError, match="world family seed"):
        type(bundle)(
            seed=6,
            world=bundle.world,
            perception=bundle.perception,
            ecology=bundle.ecology,
            partner=bundle.partner,
            communication=bundle.communication,
        )


def test_world_seeds_are_disjoint_and_deterministic() -> None:
    rule = WorldResamplingRule(train_worlds=3, held_out_worlds=2, resample_axis="layout")
    train_a, held_a = rule.world_seeds(11)
    train_b, held_b = rule.world_seeds(11)
    assert (train_a, held_a) == (train_b, held_b)
    assert not set(train_a) & set(held_a)
    train_c, held_c = rule.world_seeds(12)
    assert (train_a, held_a) != (train_c, held_c)


def test_world_fixture_yields_unique_world_hashes() -> None:
    world = make_ecology_fixture(7).world
    fixture = world.world_fixture()
    hashes = fixture["train_world_sha256"] + fixture["held_out_world_sha256"]
    assert len(set(hashes)) == len(hashes) == 5


def test_resampling_rule_fails_closed() -> None:
    with pytest.raises(ValueError, match="two training worlds"):
        WorldResamplingRule(train_worlds=1, held_out_worlds=1, resample_axis="layout")
    with pytest.raises(ValueError, match="held-out world"):
        WorldResamplingRule(train_worlds=2, held_out_worlds=0, resample_axis="layout")
    with pytest.raises(ValueError, match="adaptation control"):
        WorldResamplingRule(
            train_worlds=2, held_out_worlds=1, resample_axis="layout", memorization_control="lookup"
        )


def test_hidden_dynamics_fails_closed() -> None:
    with pytest.raises(ValueError, match="hidden parameter"):
        HiddenDynamicsDeclaration((), 1, 2, 1, True)
    with pytest.raises(ValueError, match="never leak"):
        HiddenDynamicsDeclaration(("tool-effect",), 1, 2, 1, False)
    with pytest.raises(ValueError, match="positive"):
        HiddenDynamicsDeclaration(("tool-effect",), 0, 2, 1, True)


def test_world_contract_rejects_control_drift() -> None:
    fixture = make_ecology_fixture(9).world
    with pytest.raises(ValueError, match="control set or order drift"):
        BoundedWorldContract(
            family_seed=9,
            resampling=fixture.resampling,
            hidden=fixture.hidden,
            adaptation_controls=("reactive",),
        )


def test_sensing_costs_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported sensing action"):
        SensingCostDeclaration("smell", 1, 0, 0, 0)
    with pytest.raises(ValueError, match="abstention cannot charge"):
        SensingCostDeclaration("abstain", 1, 0, 0, 0)
    with pytest.raises(ValueError, match="at least one unit"):
        SensingCostDeclaration("view", 0, 0, 0, 0)
    with pytest.raises(ValueError, match="nonnegative"):
        SensingCostDeclaration("view", -1, 0, 0, 1)


def test_value_forecast_fails_closed() -> None:
    with pytest.raises(ValueError, match="metric drift"):
        ValueForecastDeclaration("hindsight-gain", 1, True, "post-hoc-value")
    with pytest.raises(ValueError, match="before the sensing cost"):
        ValueForecastDeclaration("expected-information-gain", 1, False, "post-hoc-value")
    with pytest.raises(ValueError, match="post-hoc-value control"):
        ValueForecastDeclaration("expected-information-gain", 1, True, "none")


def test_perception_contract_requires_full_action_and_control_coverage() -> None:
    fixture = make_ecology_fixture(4).perception
    assert tuple(row["action"] for row in fixture.payload()["costs"]) == (
        "view",
        "time",
        "audio",
        "abstain",
    )
    assert fixture.payload()["controls"] == list(ACQUISITION_CONTROLS)
    with pytest.raises(ValueError, match="cover view, time, audio, abstain"):
        ActivePerceptionContract(costs=fixture.costs[:2], value_forecast=fixture.value_forecast)
    with pytest.raises(ValueError, match="control set or order drift"):
        ActivePerceptionContract(
            costs=fixture.costs,
            value_forecast=fixture.value_forecast,
            controls=("random-acquisition",),
        )
    with pytest.raises(ValueError, match="cannot be waived"):
        ActivePerceptionContract(
            costs=fixture.costs,
            value_forecast=fixture.value_forecast,
            require_cross_world_transfer=False,
        )


def test_guards_refuse_by_named_rule() -> None:
    with pytest.raises(EcologyRefusal) as noisy:
        guard_noisy_tv(_goal(target_kind="noise-source"))
    assert noisy.value.rule == "noisy-tv"
    with pytest.raises(EcologyRefusal) as hacked:
        guard_reward_hacking(_goal(reward_source="self-scored"))
    assert hacked.value.rule == "reward-hacking"
    with pytest.raises(EcologyRefusal) as unsafe:
        guard_unsafe_goal(_goal(unsafe=True))
    assert unsafe.value.rule == "unsafe-goal"
    guard_noisy_tv(_goal())
    guard_reward_hacking(_goal())
    guard_unsafe_goal(_goal())


def test_archive_admission_guards_and_bloat_refusal() -> None:
    archive = GoalArchive(capacity=1)
    assert archive.admit(_goal(ref="goal:a", descriptor=(0, 0), quality=0.4))
    with pytest.raises(EcologyRefusal) as bloat:
        archive.admit(_goal(ref="goal:b", descriptor=(1, 1)))
    assert bloat.value.rule == "archive-bloat"
    assert archive.admit(_goal(ref="goal:c", descriptor=(0, 0), quality=0.9))
    assert not archive.admit(_goal(ref="goal:d", descriptor=(0, 0), quality=0.1))
    assert archive.distinct_cells() == 1
    with pytest.raises(EcologyRefusal):
        archive.admit(_goal(ref="goal:e", descriptor=(2, 2), target_kind="noise-source"))


def test_goal_record_fails_closed() -> None:
    with pytest.raises(ValueError, match="reference"):
        _goal(ref="task:a")
    with pytest.raises(ValueError, match="descriptor"):
        _goal(descriptor=())
    with pytest.raises(ValueError, match="finite"):
        _goal(quality=float("nan"))


def test_curriculum_goldilocks_band() -> None:
    band = CurriculumDeclaration("learning-progress", 0.2, 0.8)
    assert band.in_band(0.5)
    assert not band.in_band(0.1)
    assert not band.in_band(0.9)
    assert not band.in_band(0.2)
    with pytest.raises(ValueError, match="finite value"):
        band.in_band(1.5)
    with pytest.raises(ValueError, match="selection signal"):
        CurriculumDeclaration("loss", 0.2, 0.8)
    with pytest.raises(ValueError, match="band bounds"):
        CurriculumDeclaration("learning-progress", 0.8, 0.2)
    with pytest.raises(ValueError, match="control set"):
        CurriculumDeclaration("learning-progress", 0.2, 0.8, controls=("random-task",))


def test_stop_rules_trigger_in_declared_precedence() -> None:
    config = StopRuleConfig(plateau_window=2, min_new_cells_per_window=1, diversity_floor=2, bloat_limit=4)
    assert evaluate_stop_rules(config, [_history_row(unsafe=True)]) == "unsafe-goal"
    assert evaluate_stop_rules(config, [_history_row(size=5)]) == "archive-bloat"
    assert evaluate_stop_rules(config, [_history_row(new=0, distinct=1, size=1)] * 2) == "collapse"
    assert evaluate_stop_rules(config, [_history_row(new=0)] * 2) == "plateau"
    assert evaluate_stop_rules(config, [_history_row()]) is None
    with pytest.raises(ValueError, match="at least one history row"):
        evaluate_stop_rules(config, [])
    with pytest.raises(ValueError, match="declared keys"):
        evaluate_stop_rules(config, [{"new_cells": 1}])
    with pytest.raises(ValueError, match="nonnegative integers"):
        evaluate_stop_rules(config, [_history_row(new=-1)])


def test_ecology_contract_fails_closed() -> None:
    fixture = make_ecology_fixture(6).ecology
    with pytest.raises(ValueError, match="bloat limit"):
        AutotelicEcologyContract(archive_capacity=99, stop=fixture.stop, curriculum=fixture.curriculum)
    with pytest.raises(ValueError, match="guard set"):
        AutotelicEcologyContract(
            archive_capacity=fixture.archive_capacity,
            stop=fixture.stop,
            curriculum=fixture.curriculum,
            guards=("noisy-tv",),
        )


def test_goal_babbling_fixture_is_deterministic_and_exercises_refusals() -> None:
    contract = make_ecology_fixture(8).ecology
    trace_a = run_goal_babbling_fixture(contract, seed=8, steps=24)
    trace_b = run_goal_babbling_fixture(contract, seed=8, steps=24)
    assert trace_a["trace_sha256"] == trace_b["trace_sha256"]
    assert trace_a["claim_scope"] == CLAIM_SCOPE
    counts = trace_a["refusal_counts"]
    assert counts["noisy-tv"] + counts["reward-hacking"] >= 1
    assert len(trace_a["archive"]["cells"]) <= contract.archive_capacity
    assert trace_a["steps_executed"] <= trace_a["steps_requested"]
    with pytest.raises(ValueError, match="plateau window"):
        run_goal_babbling_fixture(contract, seed=8, steps=1)


def test_partner_contract_fixture_covers_all_experiments() -> None:
    contract = make_ecology_fixture(10).partner
    names = [row["name"] for row in contract.payload()["experiments"]]
    assert sorted(names) == sorted(PARTNER_EXPERIMENTS)
    assert contract.payload()["partner_model_control"] == "partner-policy-pattern-matching"


def test_partner_contract_requires_held_out_and_training_policies() -> None:
    fixture = make_ecology_fixture(10).partner
    all_train = tuple(
        PartnerSpec(row.partner_ref, row.policy_ref, row.private_observation_keys, False)
        for row in fixture.partners
    )
    with pytest.raises(ValueError, match="held-out"):
        SimulatedPartnerContract(
            partners=all_train,
            learner_visible_keys=fixture.learner_visible_keys,
            experiments=fixture.experiments,
        )


def test_partner_private_observations_cannot_leak() -> None:
    fixture = make_ecology_fixture(10).partner
    with pytest.raises(ValueError, match="leak"):
        SimulatedPartnerContract(
            partners=fixture.partners,
            learner_visible_keys=("private-cue-a",),
            experiments=fixture.experiments,
        )


def test_partner_experiment_declarations_fail_closed() -> None:
    with pytest.raises(ValueError, match="unsupported partner experiment"):
        PartnerExperimentDeclaration("mind-reading", "metric", "null statement", "control")
    with pytest.raises(ValueError, match="equal-information control"):
        PartnerExperimentDeclaration(
            "teaching-vs-equal-information", "gain", "no gain over control", "no-control"
        )
    fixture = make_ecology_fixture(10).partner
    with pytest.raises(ValueError, match="exactly once"):
        SimulatedPartnerContract(
            partners=fixture.partners,
            learner_visible_keys=fixture.learner_visible_keys,
            experiments=fixture.experiments[:3],
        )
    with pytest.raises(ValueError, match="pattern-matching control"):
        SimulatedPartnerContract(
            partners=fixture.partners,
            learner_visible_keys=fixture.learner_visible_keys,
            experiments=fixture.experiments,
            partner_model_control="none",
        )


def test_message_bindings_enforce_referent_namespaces() -> None:
    MessageBindingDeclaration("message:m1", "event", "event:e1")
    with pytest.raises(ValueError, match="unsupported message binding"):
        MessageBindingDeclaration("message:m1", "emotion", "event:e1")
    with pytest.raises(ValueError, match="reference"):
        MessageBindingDeclaration("message:m1", "event", "action:a1")


def test_communication_contract_fails_closed() -> None:
    fixture = make_ecology_fixture(12).communication
    assert fixture.payload()["controls"] == list(CHANNEL_CONTROLS)
    assert {row["binding"] for row in fixture.payload()["bindings"]} == set(MESSAGE_BINDINGS)
    with pytest.raises(ValueError, match="every declared referent kind"):
        CommunicationGroundingContract(bindings=fixture.bindings[:2], channel_bits=8, equal_bandwidth_bits=8)
    with pytest.raises(ValueError, match="exact channel budget"):
        CommunicationGroundingContract(bindings=fixture.bindings, channel_bits=8, equal_bandwidth_bits=4)
    with pytest.raises(ValueError, match="control set or order drift"):
        CommunicationGroundingContract(
            bindings=fixture.bindings,
            channel_bits=8,
            equal_bandwidth_bits=8,
            controls=("random-message",),
        )
    with pytest.raises(ValueError, match="scoring axes"):
        CommunicationGroundingContract(
            bindings=fixture.bindings,
            channel_bits=8,
            equal_bandwidth_bits=8,
            score_axes=("usefulness",),
        )
