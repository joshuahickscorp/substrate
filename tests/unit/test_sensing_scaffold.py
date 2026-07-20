
from __future__ import annotations

import dataclasses
import hashlib

import pytest

from mop.devel import north_star
from mop.environments.scenario_factory import make_scenario
from mop.experiments.expansion_harness import CLAIM_SCOPE as HARNESS_CLAIM_SCOPE
from mop.substrate import sensing_scaffold as ss
from mop.substrate.sensing_scaffold import (
    AudioProbeContract,
    ContradictionTriangulationContract,
    PerspectiveDeclaration,
    PerspectivePluralityContract,
    RightsFieldDeclaration,
    SampleClockBinding,
    ScaffoldRefusal,
    SourceCard,
    SourceReport,
    UniqueInformationMetricContract,
    apply_slot_removal,
    assert_natural_experiment_admissible,
    build_causal_binding_contract,
    build_contradiction_contract,
    build_temporal_binding_contract,
    make_sensing_fixture,
    make_synthetic_waveform,
    validate_audio_alignment,
)

SEED = 7


@pytest.fixture(scope="module")
def fixture() -> ss.SensingScaffoldFixture:
    return make_sensing_fixture(seed=SEED)


def test_module_prose_passes_the_sentience_rail() -> None:
    assert north_star.scan_text(ss.__doc__ or "") == []
    assert north_star.scan_text(ss.CLAIM_SCOPE) == []


def test_claim_scope_is_pinned_to_the_wave_e0_harness() -> None:
    assert ss.CLAIM_SCOPE == HARNESS_CLAIM_SCOPE


def test_waveforms_are_deterministic_and_kind_distinct() -> None:
    raw_a, spec_a = make_synthetic_waveform(seed=SEED, kind="primary")
    raw_b, spec_b = make_synthetic_waveform(seed=SEED, kind="primary")
    assert raw_a == raw_b and spec_a == spec_b
    assert spec_a.waveform_sha256 == hashlib.sha256(raw_a).hexdigest()
    digests = {
        kind: make_synthetic_waveform(seed=SEED, kind=kind)[1].waveform_sha256 for kind in ss.WAVEFORM_KINDS
    }
    assert len(set(digests.values())) == len(digests)


def test_muted_waveform_is_all_zero_bytes() -> None:
    raw, spec = make_synthetic_waveform(seed=SEED, kind="muted")
    assert raw == b"\x00" * spec.byte_count


def test_waveform_refusals() -> None:
    with pytest.raises(ScaffoldRefusal):
        make_synthetic_waveform(seed=SEED, kind="not-a-kind")
    with pytest.raises(ScaffoldRefusal):
        make_synthetic_waveform(seed=-1, kind="primary")
    with pytest.raises(ScaffoldRefusal):
        make_synthetic_waveform(seed=SEED, kind="primary", num_samples=1)


def test_clock_binding_consistency_pass_and_fail(fixture: ss.SensingScaffoldFixture) -> None:
    audio = fixture.audio
    observations = {str(row.ref): row for row in fixture.scenario.graph.observations}
    clock = observations[str(audio.alignment.observation_ref)].clock
    audio.binding.assert_consistent(clock)
    wrong = SampleClockBinding(
        waveform_sha256=audio.binding.waveform_sha256,
        clock_ref=audio.binding.clock_ref,
        sample_rate_hz=audio.binding.sample_rate_hz,
        num_samples=audio.binding.num_samples + 1,
    )
    with pytest.raises(ScaffoldRefusal):
        wrong.assert_consistent(clock)


def test_audio_alignment_validates_and_fails_closed(fixture: ss.SensingScaffoldFixture) -> None:
    audio = fixture.audio
    validate_audio_alignment(
        graph=fixture.scenario.graph,
        alignment=audio.alignment,
        spec=audio.spec,
        binding=audio.binding,
    )
    tampered = dataclasses.replace(audio.alignment, graph_sha256="0" * 64)
    with pytest.raises(ScaffoldRefusal):
        validate_audio_alignment(
            graph=fixture.scenario.graph, alignment=tampered, spec=audio.spec, binding=audio.binding
        )
    other_graph = make_scenario(seed=SEED + 1).graph
    with pytest.raises(ScaffoldRefusal):
        validate_audio_alignment(
            graph=other_graph, alignment=audio.alignment, spec=audio.spec, binding=audio.binding
        )


def test_native_audio_contract_is_complete(fixture: ss.SensingScaffoldFixture) -> None:
    audio = fixture.audio
    assert tuple(row.arm for row in audio.controls) == ss.SPECTRAL_CONTROL_ARMS
    assert tuple(row.control for row in audio.probes) == ss.AUDIO_PROBE_CONTROLS
    assert audio.decode_performed is False
    assert all(row.expected_control_accept is False for row in audio.probes)
    assert len(audio.sha256) == 64


def test_native_audio_contract_refuses_decode_claims(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.audio, decode_performed=True)


def test_native_audio_contract_refuses_widened_claim_scope(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.audio, claim_scope="native audio capability demonstrated")


def test_native_audio_contract_refuses_missing_probe(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.audio, probes=fixture.audio.probes[:2])


def test_native_audio_contract_refuses_unmatched_control(fixture: ss.SensingScaffoldFixture) -> None:
    audio = fixture.audio
    mismatched_spec = make_synthetic_waveform(
        seed=SEED, kind="random-spectral", num_samples=audio.spec.num_samples * 2
    )[1]
    bad_control = dataclasses.replace(audio.controls[0], spec=mismatched_spec)
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(audio, controls=(bad_control, audio.controls[1]))


def test_probe_contract_refusal_rules() -> None:
    digest = "a" * 64
    with pytest.raises(ScaffoldRefusal):
        AudioProbeContract(
            control="muted",
            probe_waveform_sha256=digest,
            is_silent=False,
            vision_withheld=False,
            expected_control_accept=False,
            ground_truth_relation="x",
        )
    with pytest.raises(ScaffoldRefusal):
        AudioProbeContract(
            control="audio-only",
            probe_waveform_sha256=digest,
            is_silent=False,
            vision_withheld=True,
            expected_control_accept=True,
            ground_truth_relation="x",
        )
    with pytest.raises(ScaffoldRefusal):
        AudioProbeContract(
            control="shuffled-time",
            probe_waveform_sha256=digest,
            is_silent=False,
            vision_withheld=True,
            expected_control_accept=False,
            ground_truth_relation="x",
        )


def _metric(name: str = "value_metric") -> UniqueInformationMetricContract:
    return UniqueInformationMetricContract(
        metric_name=name,
        evaluation_split=ss.HELD_OUT_SPLIT,
        baselines=ss.REQUIRED_PERSPECTIVE_BASELINES,
        conditional_value_definition="held-out delta at matched capacity",
    )


def test_perspective_contract_requires_distinct_source_or_computation() -> None:
    twin = PerspectiveDeclaration(
        perspective_id="a",
        information_source="vision-cache",
        computation="linear probe",
        metric=_metric(),
    )
    other = dataclasses.replace(twin, perspective_id="b")
    with pytest.raises(ScaffoldRefusal):
        PerspectivePluralityContract(declarations=(twin, other))
    distinct = dataclasses.replace(other, computation="nonlinear probe")
    contract = PerspectivePluralityContract(declarations=(twin, distinct))
    assert len(contract.sha256) == 64


def test_metric_contract_fails_closed_on_split_and_baselines() -> None:
    with pytest.raises(ScaffoldRefusal):
        UniqueInformationMetricContract(
            metric_name="m",
            evaluation_split="train",
            baselines=ss.REQUIRED_PERSPECTIVE_BASELINES,
            conditional_value_definition="x",
        )
    with pytest.raises(ScaffoldRefusal):
        UniqueInformationMetricContract(
            metric_name="m",
            evaluation_split=ss.HELD_OUT_SPLIT,
            baselines=("all-other-perspectives",),
            conditional_value_definition="x",
        )


def test_slot_removal_rule(fixture: ss.SensingScaffoldFixture) -> None:
    contract = fixture.perspectives
    ids = [row.perspective_id for row in contract.declarations]
    values = {ids[0]: 0.5, ids[1]: 0.0, ids[2]: -0.1}
    verdict = apply_slot_removal(contract, values, min_conditional_value=0.05)
    assert verdict["retained"] == [ids[0]]
    assert sorted(verdict["removed"]) == sorted(ids[1:])
    with pytest.raises(ScaffoldRefusal):
        apply_slot_removal(contract, {ids[0]: 0.5}, min_conditional_value=0.05)
    with pytest.raises(ScaffoldRefusal):
        apply_slot_removal(contract, {**values, ids[1]: float("nan")}, min_conditional_value=0.05)
    with pytest.raises(ScaffoldRefusal):
        apply_slot_removal(contract, values, min_conditional_value=0.05, evaluation_split="train")


def _card(**overrides: object) -> SourceCard:
    cleared = RightsFieldDeclaration(status="cleared", authority="steward statement")
    kwargs: dict = {
        "source_id": "unit-source",
        "steward": "unit steward",
        "fields": dict.fromkeys(ss.RIGHTS_FIELDS, cleared),
        "allowed_uses": ("research-local",),
    }
    kwargs.update(overrides)
    return SourceCard(**kwargs)


def test_source_card_admits_declared_use() -> None:
    assert_natural_experiment_admissible(_card(), "research-local")


def test_source_card_refuses_unknown_authority() -> None:
    fields = dict(_card().fields)
    fields["privacy"] = RightsFieldDeclaration(status="unknown", authority="")
    with pytest.raises(ScaffoldRefusal, match="unknown authority"):
        assert_natural_experiment_admissible(_card(fields=fields), "research-local")


def test_source_card_refuses_dataset_level_inference() -> None:
    fields = dict(_card().fields)
    fields["publicity"] = RightsFieldDeclaration(
        status="cleared", authority="dataset license", inferred_from_dataset_license=True
    )
    with pytest.raises(ScaffoldRefusal, match="inferred authority"):
        assert_natural_experiment_admissible(_card(fields=fields), "research-local")


def test_source_card_refuses_split_reuse() -> None:
    with pytest.raises(ScaffoldRefusal, match="split reuse"):
        assert_natural_experiment_admissible(_card(), "redistribution")


def test_source_card_refuses_restricted_required_field() -> None:
    fields = dict(_card().fields)
    fields["redistribution"] = RightsFieldDeclaration(status="restricted", authority="steward statement")
    card = _card(fields=fields, allowed_uses=("research-local", "redistribution"))
    with pytest.raises(ScaffoldRefusal, match="restricted authority"):
        assert_natural_experiment_admissible(card, "redistribution")


def test_source_card_requires_full_field_coverage() -> None:
    partial = {name: RightsFieldDeclaration(status="cleared", authority="s") for name in ss.RIGHTS_FIELDS[:5]}
    with pytest.raises(ScaffoldRefusal):
        _card(fields=partial)


def test_temporal_contract_builds_from_the_wave_e0_fixture(fixture: ss.SensingScaffoldFixture) -> None:
    temporal = fixture.temporal
    assert temporal.arrival_skew_ticks > 0
    assert temporal.binding_window_ticks >= temporal.arrival_skew_ticks
    assert temporal.shuffled_timing_gap_ticks > temporal.binding_window_ticks
    assert len(temporal.sha256) == 64


def test_temporal_contract_refuses_window_below_skew(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.temporal, binding_window_ticks=fixture.temporal.arrival_skew_ticks - 1)


def test_temporal_contract_refuses_window_covering_the_control(
    fixture: ss.SensingScaffoldFixture,
) -> None:
    with pytest.raises(ScaffoldRefusal):
        build_temporal_binding_contract(
            fixture.scenario,
            binding_window_ticks=fixture.temporal.shuffled_timing_gap_ticks + 1,
        )


def test_contradiction_contract_has_exactly_one_dissent(fixture: ss.SensingScaffoldFixture) -> None:
    contract = fixture.contradiction
    dissent = [row for row in contract.reports if row.source_id == contract.contradicted_source_id]
    consensus = [row for row in contract.reports if row.source_id != contract.contradicted_source_id]
    assert len(dissent) == 1 and len(consensus) >= 2
    assert len({row.claim for row in consensus}) == 1
    assert dissent[0].claim != consensus[0].claim
    assert contract.baselines == ss.CONTRADICTION_BASELINES


def test_contradicted_source_is_seed_deterministic() -> None:
    first = build_contradiction_contract(make_scenario(seed=SEED))
    second = build_contradiction_contract(make_scenario(seed=SEED))
    assert first.sha256 == second.sha256
    other = build_contradiction_contract(make_scenario(seed=SEED + 1))
    assert other.contradicted_source_id != first.contradicted_source_id or other.sha256 != first.sha256


def test_contradiction_contract_refusals(fixture: ss.SensingScaffoldFixture) -> None:
    contract = fixture.contradiction
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(contract, reports=contract.reports[:2])
    honest = next(row for row in contract.reports if row.source_id != contract.contradicted_source_id)
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(contract, contradicted_source_id=honest.source_id)
    second_dissent = SourceReport(
        source_id="extra",
        event_ref=contract.event_ref,
        claim=honest.claim + 99,
        content=contract.reports[0].content,
    )
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(contract, reports=(*contract.reports, second_dissent))


def test_causal_contract_builds_from_the_wave_e0_fixture(fixture: ss.SensingScaffoldFixture) -> None:
    causal = fixture.causal
    assert len(causal.branch_refs) >= 2
    assert causal.chosen_branch_ref in causal.branch_refs
    assert causal.baseline_arm == ss.CAUSAL_BASELINE_ARM
    assert len(causal.sha256) == 64


def test_causal_contract_refuses_single_branch(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.causal, branch_refs=fixture.causal.branch_refs[:1])


def test_causal_contract_refuses_foreign_chosen_branch(fixture: ss.SensingScaffoldFixture) -> None:
    with pytest.raises(ScaffoldRefusal):
        dataclasses.replace(fixture.causal, chosen_branch_ref="branch:not/in-the-fork-0000000000")


def test_causal_control_is_synchronous_but_unrelated(fixture: ss.SensingScaffoldFixture) -> None:
    observations = {str(row.ref): row for row in fixture.scenario.graph.observations}
    left = observations[str(fixture.causal.synchronous_unrelated_left_ref)]
    right = observations[str(fixture.causal.synchronous_unrelated_right_ref)]
    assert left.clock.overlaps(right.clock)
    assert left.event_ref != right.event_ref


def test_direct_contradiction_contract_construction(fixture: ss.SensingScaffoldFixture) -> None:
    rebuilt = ContradictionTriangulationContract(
        event_ref=fixture.contradiction.event_ref,
        reports=fixture.contradiction.reports,
        contradicted_source_id=fixture.contradiction.contradicted_source_id,
    )
    assert rebuilt.sha256 == fixture.contradiction.sha256


def test_fixture_is_deterministic_and_seed_sensitive(fixture: ss.SensingScaffoldFixture) -> None:
    replay = make_sensing_fixture(seed=SEED)
    assert replay.sha256 == fixture.sha256
    other = make_sensing_fixture(seed=SEED + 1)
    assert other.sha256 != fixture.sha256


def test_fixture_and_causal_builders_agree(fixture: ss.SensingScaffoldFixture) -> None:
    assert build_causal_binding_contract(fixture.scenario).sha256 == fixture.causal.sha256
    assert build_temporal_binding_contract(fixture.scenario).sha256 == fixture.temporal.sha256


def test_fixture_free_text_passes_the_sentience_rail(fixture: ss.SensingScaffoldFixture) -> None:
    import json

    north_star.assert_no_sentience_claims(json.dumps(fixture.payload()), where="sensing fixture")
