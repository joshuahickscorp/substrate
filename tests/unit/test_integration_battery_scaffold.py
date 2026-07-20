
from __future__ import annotations

import dataclasses

import pytest

from mop.devel.north_star import assert_no_sentience_claims
from mop.diagnostics.operational_awareness import OA_COMPONENTS
from mop.studies.integration_battery_scaffold import (
    BROADCAST_MODES,
    CLAIM_SCOPE,
    OPERATION_LEVELS,
    REQUIRED_BROADCAST_CONTROLS,
    REQUIRED_METACOG_CONTROLS,
    REQUIRED_REPORT_GROUNDING_CONTROLS,
    REQUIRED_SELF_MODEL_CONTROLS,
    SELF_MODEL_KINDS,
    TELEMETRY_FIELDS,
    TELEMETRY_NUMERIC_FIELDS,
    BoundedPrediction,
    DisconfirmingPattern,
    HomeostaticSetpoint,
    MetacognitiveEfficiencyContract,
    NeighboringDissociation,
    Observation,
    PredictionTarget,
    SelfModelContract,
    SelfReportGroundingContract,
    TheoryBatteryContract,
    TheoryEntry,
    make_broadcast_contract,
    make_hardware_body_contract,
    make_homeostatic_control_contract,
    make_metacognitive_efficiency_contract,
    make_self_report_grounding_contract,
    make_telemetry_prediction_contract,
    make_telemetry_trace_fixture,
    make_theory_battery_fixture,
    make_tool_incorporation_contract,
    refuse_composite_metric,
    refuse_interpretation_tokens,
    scaffold_manifest,
    score_functional_outcomes,
)


def test_theory_battery_fixture_constructs_and_is_deterministic():
    contract_a, obs_a = make_theory_battery_fixture(seed=0)
    contract_b, obs_b = make_theory_battery_fixture(seed=0)
    assert contract_a.sha256 == contract_b.sha256
    assert [o.payload() for o in obs_a] == [o.payload() for o in obs_b]
    contract_c, obs_c = make_theory_battery_fixture(seed=1)
    assert contract_a.sha256 == contract_c.sha256  # contract is seed-free
    assert [o.payload() for o in obs_a] != [o.payload() for o in obs_c]


def test_theory_battery_requires_all_five_levels():
    contract, _ = make_theory_battery_fixture()
    theory = contract.theories[0]
    truncated = tuple(row for row in theory.predictions if row.level != "restoration")
    with pytest.raises(ValueError, match="misses prediction levels"):
        TheoryEntry(
            id=theory.id,
            name=theory.name,
            predictions=truncated,
            disconfirmers=theory.disconfirmers,
            dissociations=theory.dissociations,
        )


def test_theory_battery_requires_disconfirmers_and_dissociations():
    contract, _ = make_theory_battery_fixture()
    theory = contract.theories[0]
    with pytest.raises(ValueError, match="disconfirming pattern"):
        dataclasses.replace(theory, disconfirmers=())
    with pytest.raises(ValueError, match="dissociation"):
        dataclasses.replace(theory, dissociations=())


def test_theory_battery_refuses_convergent_theories():
    contract, _ = make_theory_battery_fixture()
    base = contract.theories[0]
    clone = TheoryEntry(
        id="theory:clone",
        name=base.name,
        predictions=tuple(dataclasses.replace(row, theory_id="theory:clone") for row in base.predictions),
        disconfirmers=tuple(dataclasses.replace(row, theory_id="theory:clone") for row in base.disconfirmers),
        dissociations=tuple(dataclasses.replace(row, theory_id="theory:clone") for row in base.dissociations),
    )
    with pytest.raises(ValueError, match="no divergent bounded prediction"):
        TheoryBatteryContract(
            theories=(contract.theories[0], clone),
            operations=contract.operations,
        )


def test_theory_battery_needs_two_theories_and_declared_operations():
    contract, _ = make_theory_battery_fixture()
    with pytest.raises(ValueError, match="at least two theories"):
        TheoryBatteryContract(theories=(contract.theories[0],), operations=contract.operations)
    with pytest.raises(ValueError, match="undeclared operation"):
        TheoryBatteryContract(theories=contract.theories, operations=("op:workspace-report",))


def test_prediction_rail_refusal_is_construction_time():
    with pytest.raises(ValueError, match="sentience rail"):
        BoundedPrediction(
            theory_id="theory:x",
            operation_id="op:workspace-report",
            level="behavior",
            metric="transfer_accuracy",
            direction="increase",
            lower_bound=0.0,
            upper_bound=1.0,
            rationale="the system is conscious under this operation",
        )


def test_prediction_rejects_bad_bounds_level_direction():
    good = dict(
        theory_id="theory:x",
        operation_id="op:workspace-report",
        level="behavior",
        metric="transfer_accuracy",
        direction="increase",
        lower_bound=0.0,
        upper_bound=1.0,
        rationale="bounded behavioral prediction",
    )
    with pytest.raises(ValueError, match="inverted"):
        BoundedPrediction(**{**good, "lower_bound": 2.0})
    with pytest.raises(ValueError, match="finite"):
        BoundedPrediction(**{**good, "upper_bound": float("nan")})
    with pytest.raises(ValueError, match="level"):
        BoundedPrediction(**{**good, "level": "vibes"})
    with pytest.raises(ValueError, match="direction"):
        BoundedPrediction(**{**good, "direction": "sideways"})
    with pytest.raises(ValueError, match="interpretation vocabulary"):
        BoundedPrediction(**{**good, "metric": "moral_weight"})


def test_disconfirmer_and_dissociation_validation():
    with pytest.raises(ValueError, match="direction"):
        DisconfirmingPattern(
            theory_id="theory:x",
            operation_id="op:a",
            level="lesion",
            metric="transfer_accuracy",
            disconfirming_direction="sideways",
            note="bad direction",
        )
    with pytest.raises(ValueError, match="distinct neighboring operations"):
        NeighboringDissociation(
            theory_id="theory:x",
            moving_operation_id="op:a",
            unaffected_operation_id="op:a",
            level="lesion",
            metric="transfer_accuracy",
            note="same operation twice",
        )


def test_scorer_scores_fixture_and_stays_functional():
    contract, observations = make_theory_battery_fixture(seed=3)
    result = score_functional_outcomes(contract, observations)
    assert result["claim_scope"] == CLAIM_SCOPE
    assert result["battery_sha256"] == contract.sha256
    broadcast = result["per_theory"]["theory:capacity-broadcast"]
    dense = result["per_theory"]["theory:dense-integration"]
    assert broadcast["functional_match_fraction"] == 1.0
    assert broadcast["disconfirmed"] is False
    assert broadcast["dissociations_checked"] == 1
    assert broadcast["dissociations_passed"] == 1
    assert dense["functional_match_fraction"] < 1.0
    assert dense["disconfirmed"] is True


def test_scorer_refuses_interpretation_annotations():
    contract, observations = make_theory_battery_fixture()
    with pytest.raises(ValueError, match="interpretation vocabulary"):
        Observation(
            operation_id="op:workspace-report",
            level="behavior",
            metric="transfer_accuracy",
            direction="increase",
            value=0.5,
            annotations=(("moral_status", "high"),),
        )
    extra = Observation(
        operation_id="op:workspace-report",
        level="behavior",
        metric="transfer_accuracy",
        direction="increase",
        value=0.5,
        annotations=(("fixture", "unit"),),
    )
    result = score_functional_outcomes(contract, tuple(observations) + (extra,))
    assert "per_theory" in result


def test_scorer_fails_closed_on_empty_or_duplicate_observations():
    contract, observations = make_theory_battery_fixture()
    with pytest.raises(ValueError, match="at least one observation"):
        score_functional_outcomes(contract, ())
    with pytest.raises(ValueError, match="duplicate observation"):
        score_functional_outcomes(contract, tuple(observations) + (observations[0],))


def test_refuse_interpretation_tokens_rule():
    refuse_interpretation_tokens(("transfer_accuracy", "load_1m"), "test")
    for bad in ("welfare_score", "phenomenal_depth", "suffering_index"):
        with pytest.raises(ValueError, match="interpretation vocabulary"):
            refuse_interpretation_tokens((bad,), "test")


@pytest.mark.parametrize(
    "builder",
    [
        make_hardware_body_contract,
        make_tool_incorporation_contract,
        make_telemetry_prediction_contract,
        make_homeostatic_control_contract,
    ],
)
def test_self_model_builders_construct_and_hash_deterministically(builder):
    a, b = builder(seed=0), builder(seed=0)
    assert a.sha256 == b.sha256
    assert a.kind in SELF_MODEL_KINDS
    assert a.claim_scope == CLAIM_SCOPE
    assert builder(seed=1).sha256 != a.sha256


def test_prediction_target_vocabulary_fails_closed():
    PredictionTarget("cpu", "utilization_fraction", 1, 0.1)
    with pytest.raises(ValueError, match="unknown telemetry channel"):
        PredictionTarget("gpu", "utilization_fraction", 1, 0.1)
    with pytest.raises(ValueError, match="unknown telemetry field"):
        PredictionTarget("cpu", "vram_used", 1, 0.1)
    with pytest.raises(ValueError, match="horizon"):
        PredictionTarget("cpu", "load_1m", 0, 0.1)
    with pytest.raises(ValueError, match="tolerance"):
        PredictionTarget("cpu", "load_1m", 1, 0.0)
    with pytest.raises(ValueError, match="phase"):
        PredictionTarget("cpu", "load_1m", 1, 0.1, phase="mid-tool")


def test_telemetry_vocabulary_mirrors_throttle_names():
    assert set(TELEMETRY_FIELDS) == {
        "cpu",
        "memory",
        "swap",
        "disk",
        "processes",
        "mps",
        "thermal",
        "power",
    }
    for channel, fields in TELEMETRY_NUMERIC_FIELDS.items():
        assert set(fields) <= set(TELEMETRY_FIELDS[channel])
    assert "utilization_fraction" in TELEMETRY_FIELDS["cpu"]
    assert "available_percent" in TELEMETRY_FIELDS["memory"]
    assert "status" not in TELEMETRY_NUMERIC_FIELDS.get("thermal", ())


def test_self_model_control_drift_fails_closed():
    base = make_hardware_body_contract()
    with pytest.raises(ValueError, match="control drift"):
        dataclasses.replace(base, controls=("boundary-shuffled",))


def test_hardware_body_needs_two_channels_and_no_tool():
    with pytest.raises(ValueError, match="two telemetry channels"):
        SelfModelContract(
            kind="hardware-body",
            targets=(PredictionTarget("cpu", "load_1m", 1, 0.1),),
            controls=REQUIRED_SELF_MODEL_CONTROLS["hardware-body"],
            seed=0,
        )
    with pytest.raises(ValueError, match="no tool"):
        dataclasses.replace(make_hardware_body_contract(), tool_id="tool:x")


def test_tool_incorporation_requires_matched_pre_post_pairs():
    good = make_tool_incorporation_contract()
    unmatched = tuple(row for row in good.targets if row.phase == "pre-tool")
    with pytest.raises(ValueError, match="matched pre-tool and post-tool"):
        dataclasses.replace(good, targets=unmatched)
    standing = (PredictionTarget("disk", "free_gb", 1, 1.0),)
    with pytest.raises(ValueError, match="pre-tool or post-tool"):
        dataclasses.replace(good, targets=standing)


def test_telemetry_prediction_requires_numeric_multi_step_targets():
    good = make_telemetry_prediction_contract()
    with pytest.raises(ValueError, match="numeric telemetry fields"):
        dataclasses.replace(good, targets=(PredictionTarget("thermal", "status", 2, 1.0),))
    nowcast = (PredictionTarget("cpu", "load_1m", 1, 0.5),)
    with pytest.raises(ValueError, match="two or more steps"):
        dataclasses.replace(good, targets=nowcast)


def test_homeostatic_control_requires_setpoints_and_known_actuators():
    good = make_homeostatic_control_contract()
    with pytest.raises(ValueError, match="at least one declared setpoint"):
        dataclasses.replace(good, setpoints=())
    with pytest.raises(ValueError, match="unknown actuator"):
        HomeostaticSetpoint("memory", "available_percent", 10.0, 90.0, "overclock")
    with pytest.raises(ValueError, match="numeric telemetry field"):
        HomeostaticSetpoint("thermal", "status", 0.0, 1.0, "defer-admission")
    with pytest.raises(ValueError, match="lower < upper"):
        HomeostaticSetpoint("memory", "available_percent", 90.0, 10.0, "defer-admission")


def test_report_grounding_contract_validates_fields_and_controls():
    good = make_self_report_grounding_contract()
    assert good.metric_names == ("grounded_fraction", "shared_fields")
    assert good.controls == REQUIRED_REPORT_GROUNDING_CONTROLS
    with pytest.raises(ValueError, match="not in the telemetry vocabulary"):
        dataclasses.replace(good, report_fields=("cpu.vibes",))
    with pytest.raises(ValueError, match="controls must be"):
        dataclasses.replace(good, controls=("shuffled-report",))
    with pytest.raises(ValueError, match="metric names must be"):
        dataclasses.replace(good, metric_names=("grounded_fraction",))
    with pytest.raises(ValueError, match="no report fields"):
        SelfReportGroundingContract(report_fields=(), controls=REQUIRED_REPORT_GROUNDING_CONTROLS, seed=0)


@pytest.mark.parametrize("mode", BROADCAST_MODES)
def test_broadcast_contract_constructs_per_mode(mode):
    contract = make_broadcast_contract(mode)
    assert contract.controls == REQUIRED_BROADCAST_CONTROLS[mode]
    assert "unrestricted-bus" in contract.controls
    assert contract.capacity_slots < contract.bus_slots
    assert contract.n_consumers >= 2
    assert make_broadcast_contract(mode).sha256 == contract.sha256


def test_broadcast_contract_fails_closed():
    good = make_broadcast_contract("necessity")
    with pytest.raises(ValueError, match="vacuous"):
        dataclasses.replace(good, bus_slots=1)
    with pytest.raises(ValueError, match="two separated consumers"):
        dataclasses.replace(good, n_consumers=1)
    with pytest.raises(ValueError, match="control drift"):
        dataclasses.replace(good, controls=REQUIRED_BROADCAST_CONTROLS["sufficiency"])
    with pytest.raises(ValueError, match="FLOP budget"):
        dataclasses.replace(good, matched_flop_budget=0)
    with pytest.raises(ValueError, match="mode"):
        make_broadcast_contract("vibes")


def test_metacog_contract_reuses_oa_names_and_fails_closed():
    good = make_metacognitive_efficiency_contract()
    assert set(good.components) <= set(OA_COMPONENTS)
    assert good.controls == REQUIRED_METACOG_CONTROLS
    with pytest.raises(ValueError, match="unknown OA components"):
        dataclasses.replace(good, components=("oa9_vibes",))
    with pytest.raises(ValueError, match="without a named baseline"):
        dataclasses.replace(good, baselines=good.baselines[:1])
    with pytest.raises(ValueError, match="real baseline"):
        MetacognitiveEfficiencyContract(
            components=("oa2_calibration",),
            baselines=(("oa2_calibration", "none"),),
            monitor_flop_budget=1,
            monitor_seconds_budget=1.0,
            controls=REQUIRED_METACOG_CONTROLS,
            seed=0,
        )
    with pytest.raises(ValueError, match="FLOP budget"):
        dataclasses.replace(good, monitor_flop_budget=0)
    with pytest.raises(ValueError, match="controls must be"):
        dataclasses.replace(good, controls=("no-monitor",))


def test_refuse_composite_metric_rule():
    refuse_composite_metric("benefit_per_monitor_flop", ("oa2_calibration",))
    with pytest.raises(ValueError, match="composite"):
        refuse_composite_metric("overall_score", ("oa2_calibration", "oa6_crisis_detection"))
    with pytest.raises(ValueError, match="unknown OA components"):
        refuse_composite_metric("benefit_per_monitor_flop", ("oa9_vibes",))


def test_telemetry_trace_fixture_is_deterministic_and_vocabulary_bound():
    a = make_telemetry_trace_fixture(seed=0, steps=4)
    b = make_telemetry_trace_fixture(seed=0, steps=4)
    assert a == b
    assert make_telemetry_trace_fixture(seed=1, steps=4) != a
    assert len(a) == 4
    for snapshot in a:
        for channel, values in snapshot.items():
            if channel == "step":
                continue
            assert set(values) <= set(TELEMETRY_NUMERIC_FIELDS[channel])
    with pytest.raises(ValueError, match="at least one step"):
        make_telemetry_trace_fixture(seed=0, steps=0)


def test_manifest_and_all_contract_text_pass_sentience_rail():
    manifest = scaffold_manifest()
    assert manifest["claim_scope"] == CLAIM_SCOPE
    assert manifest["operation_levels"] == list(OPERATION_LEVELS)
    contract, _ = make_theory_battery_fixture()
    texts = [str(manifest)]
    for theory in contract.theories:
        texts.extend(theory.free_text())
    for builder in (
        make_hardware_body_contract,
        make_tool_incorporation_contract,
        make_telemetry_prediction_contract,
        make_homeostatic_control_contract,
    ):
        texts.append(str(builder().payload()))
    texts.append(str(make_self_report_grounding_contract().payload()))
    texts.append(str(make_broadcast_contract("necessity").payload()))
    texts.append(str(make_broadcast_contract("sufficiency").payload()))
    texts.append(str(make_metacognitive_efficiency_contract().payload()))
    for text in texts:
        assert_no_sentience_claims(text, where="integration battery scaffold")
