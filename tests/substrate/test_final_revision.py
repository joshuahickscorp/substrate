from __future__ import annotations

import copy

import pytest

from substrate import final_revision_campaign as campaign
from substrate import final_revision_config as C
from substrate import final_revision_experiment as experiment
from substrate import final_revision_io as io
from substrate import final_revision_verification as verification
from substrate.final_revision_continuity import run_segment
from substrate.final_revision_kernel import ArchitecturePrototype, EventSourcedKernel, developmental_fixture
from substrate.final_revision_readiness import ActionProposal, bounded_smoke
from substrate.final_revision_sensorium import Sensorium, controlled_media, structural_sensorium_report


def _no_true_activation(value: object) -> bool:
    if isinstance(value, dict):
        return all(key != "activation" or child is False for key, child in value.items()) and all(_no_true_activation(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_no_true_activation(child) for child in value)
    return True


def test_final_revision_constitution_is_complete_and_activation_false() -> None:
    assert len(C.CONTRACTS) == 16
    assert len(C.FACETS) == 20
    assert len(C.CANDIDATES) == 9
    assert len(C.REVIEW_CELLS) == 32
    assert len(C.REVIEW_ROUNDS) == 8
    assert len(C.CHALLENGE_FAMILIES) == 12
    assert len(C.MUTATIONS) == 21
    assert len(C.REQUIRED_DELIVERABLES) == len(set(C.REQUIRED_DELIVERABLES))
    assert C.STARTING_CLOSURE_RESULT == "terminal_closed_null"
    assert C.SESOI == 0.05
    assert _no_true_activation(C.configuration())


def test_preflight_preserves_historical_closure_null() -> None:
    report = campaign.preflight(publish=False)
    assert report["all_pass"], report["preflight"]["failed"]
    assert report["immutability"]["historical_evidence_untouched"]
    assert report["immutability"]["immutable_null"]["effect"] == 0.0
    assert report["immutability"]["immutable_null"]["confidence_interval_95"] == [0.0, 0.0]
    assert report["grok"]["completed_distinct_reviewer_count"] == 0
    assert not report["grok"]["minimum_complete"]


def test_preflight_accepts_detached_ci_without_local_main(monkeypatch: pytest.MonkeyPatch) -> None:
    original = campaign.io.ref_or_none

    def detached(ref: str, *, peel: bool = False) -> str | None:
        return None if ref == "main" else original(ref, peel=peel)

    monkeypatch.setattr(campaign.io, "ref_or_none", detached)
    report = campaign.preflight(publish=False)
    assert report["all_pass"], report["preflight"]["failed"]
    assert report["preflight"]["checks"]["local_main_absent_or_matches_orientation"]
    assert report["preflight"]["checks"]["remote_main_matches_orientation"]


def test_event_kernel_covers_contracts_and_restores_exactly() -> None:
    prototype = ArchitecturePrototype("I_simplest_sufficient", "entity-test")
    fixture = developmental_fixture(prototype)
    checkpoint = prototype.kernel.checkpoint()
    restored = EventSourcedKernel.restore(checkpoint)
    assert tuple(fixture["interfaces"]) == C.CONTRACTS
    assert restored.state == prototype.kernel.state
    assert restored.identity_digest() == prototype.kernel.identity_digest()
    assert restored.query("goals") == prototype.query("goals")
    assert "model-b" in restored.query("model_fabric")["models"]
    assert "lesson-update" in restored.query("learning")["admitted"]


def test_checkpoint_tampering_and_unsupported_knowledge_fail_closed() -> None:
    prototype = ArchitecturePrototype("I_simplest_sufficient", "entity-tamper")
    prototype.append(
        "belief",
        {"key": "weak", "value": True, "confidence": 0.2, "warrants": []},
        provenance="test://weak-belief",
    )
    with pytest.raises(io.Refused):
        prototype.append("knowledge", {"key": "weak", "value": True}, provenance="test://knowledge")
    checkpoint = prototype.kernel.checkpoint()
    corrupted = copy.deepcopy(checkpoint)
    corrupted["state"]["identity"]["id"] = "counterfeit"
    with pytest.raises(io.Refused):
        EventSourcedKernel.restore(corrupted)


def test_learning_rejects_negative_transfer_and_preserves_retention() -> None:
    prototype = ArchitecturePrototype("I_simplest_sufficient", "entity-learning")
    developmental_fixture(prototype)
    prior = prototype.query("memory")["developmental"]
    prototype.append(
        "learning_propose",
        {
            "update_id": "bad",
            "namespace": "semantic",
            "key": "poison",
            "value": "bad",
            "data_split": "construction",
            "source": "quarantined-teacher",
        },
        provenance="test://learning/propose",
    )
    prototype.append(
        "learning_admit",
        {
            "update_id": "bad",
            "held_out_before": 0.8,
            "held_out_after": 0.7,
            "retention_before": 0.9,
            "retention_after": 0.4,
        },
        provenance="test://learning/admit",
    )
    assert any(row["update_id"] == "bad" for row in prototype.query("learning")["rejected"])
    assert prototype.query("memory")["developmental"] == prior
    assert "poison" not in prototype.query("memory")["semantic"]


def test_sensorium_processes_arrays_waveforms_and_geometry_not_labels() -> None:
    report = structural_sensorium_report()
    assert len(report["modalities"]) == 9
    assert all(report["real_structures"].values())
    assert report["modality_content_digests_distinct"]
    assert report["corruption_changes_image_features"]
    assert report["cross_modal_timing"]["within_tolerance"]
    assert not report["hidden_labels_used"]
    sensorium = Sensorium()
    receipts = {name: sensorium.process(packet) for name, packet in controlled_media().items()}
    assert all(not receipt["hidden_label_used"] for receipt in receipts.values())
    assert receipts["speech"]["features"]["transcript"] is None
    assert receipts["depth"]["features"]["point_count"] > 0


def test_architecture_tournament_allows_s2_family_to_win_by_simplicity() -> None:
    report = experiment.architecture_tournament()
    assert report["selected_candidate"] == "I_simplest_sufficient"
    assert report["selected_architecture"].startswith("S2-derived")
    assert not report["architectural_advantage_claimed"]
    rows = {row["candidate_id"]: row for row in report["candidates"]}
    assert not rows["H_causal_temporal_ledger"]["eligible_after_stage_3"]
    assert "Grok-original" in rows["H_causal_temporal_ledger"]["loss_reason"]
    assert all(row["interface_conformance"] for row in rows.values())
    assert all(row["mechanism_ablation_detected"] for row in rows.values())
    assert len({row["mechanism_decision"]["mechanism_field"] for row in rows.values()}) == len(rows)


def test_new_bed_has_headroom_but_preserves_architecture_null() -> None:
    bed = experiment.run_discrimination_bed(
        split="test_final_revision",
        seeds=range(8),
        episodes_per_family=16,
    )
    assert bed["oracle_headroom_preferred_0_10"]
    assert bed["oracle_headroom"] > C.SESOI
    effect = bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]
    assert effect["mean_paired_effect"] == 0.0
    assert effect["confidence_interval_95"] == [0.0, 0.0]
    assert not effect["passes"]
    transcript = bed["effects"]["P1_selected_minus_full_transcript_replay"]
    assert transcript["mean_paired_effect"] == 0.0
    assert bed["classification"] == "mechanism_null"


def test_moderate_pilot_executes_required_scale_without_claim_inflation() -> None:
    report = experiment.moderate_pilot()
    assert report["scale"]["independent_histories"] == 32
    assert report["scale"]["architecture_candidates"] == 9
    assert report["scale"]["task_families"] == 12
    assert report["scale"]["compound_episodes"] >= 100_000
    assert not report["principal_positive_authorized"]
    assert report["outcome_b_campaign_authorized"]


def test_canaries_count_expected_nulls_as_scientific_passes() -> None:
    report = experiment.cheap_canaries()
    assert report["all_pass"]
    assert report["passed"] == report["total"] == 21
    assert "identity_after_process_replacement" in report["architecture_nulls_preserved"]
    nulls = [row for row in report["canaries"] if row["classification"].startswith("expected")]
    assert nulls and all(row["effect"] == 0.0 for row in nulls)


def test_reproduced_closure_null_is_exact() -> None:
    report = campaign.reproduce_null(publish=False)
    assert report["all_pass"]
    reproduction = report["reproduction"]
    assert reproduction["instrument_1"]["candidate_effect"] == pytest.approx(-0.01328125)
    assert reproduction["instrument_1"]["oracle_headroom"] == pytest.approx(0.04875)
    assert reproduction["instrument_2"]["mean_paired_effect"] == 0.0
    assert reproduction["instrument_2"]["confidence_interval_95"] == [0.0, 0.0]
    assert report["s2_anatomy"]["effectively_monolithic_substrate_on_bed"]


def test_campaign_pilot_is_outcome_b_only_until_grok_is_real() -> None:
    report = campaign.pilot(publish=False)
    assert report["all_pass"]
    assert report["outcome_b_authorized"]
    assert not report["outcome_a_authorized"]
    challenge = report["documents"]["SUBSTRATE_FINAL_REVISION_CHALLENGE_AUTHORITY.json"]
    assert not challenge["outcome_a_isolation_complete"]
    grok = report["documents"]["SUBSTRATE_FINAL_REVISION_GROK_CHALLENGE_LEDGER.json"]
    assert grok["credited_challenges"] == 0
    assert grok["fabricated_challenges"] == 0


def test_decisive_plan_preserves_zero_variance_power_limit() -> None:
    plan = experiment.decisive_plan(experiment.moderate_pilot())
    assert plan["principal_histories"] == 96
    assert plan["replication_histories"] == 48
    assert plan["hidden_composition_histories"] == 48
    assert plan["planned_microepisodes"] == 1_769_472
    assert not plan["positive_power_estimable"]
    assert not plan["outcome_a_positive_authorized"]


def test_independent_recomputation_matches_raw_receipts() -> None:
    bed = experiment.run_discrimination_bed(
        split="test_independent_recomputation",
        seeds=range(8),
        episodes_per_family=16,
        hidden_composition=True,
    )
    report = verification.recomputation_matches(bed)
    assert report["exact_match"]
    assert report["recomputed"]["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"] == 0.0


def test_mutation_and_counterfeit_verifiers_have_zero_survivors() -> None:
    mutation = verification.mutation_report()
    counterfeit = verification.counterfeit_report()
    assert mutation["baseline_accepted"]
    assert mutation["rejected"] == mutation["total"] == 21
    assert mutation["zero_survivors"]
    assert counterfeit["all_rejected"]


def test_readiness_contract_refuses_external_execution() -> None:
    smoke = bounded_smoke()
    assert smoke["all_pass"]
    assert not smoke["external_action_executed"]
    proposal = ActionProposal(
        action_id="bad",
        task_id="task",
        tool="tool",
        arguments={},
        expected_effect="external change",
        reversibility="reversible",
        external_execution_authorized=True,
    )
    with pytest.raises(io.Refused):
        proposal.validate()


def test_continuity_crosses_two_restore_boundaries(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint.json"
    receipts = []
    for segment in range(3):
        receipts.append(
            run_segment(
                checkpoint_path=checkpoint,
                receipt_path=tmp_path / f"receipt-{segment}.json",
                segment=segment,
                duration_seconds=0.01,
            )
        )
    assert [row["restored_from_prior_process"] for row in receipts] == [False, True, True]
    assert {row["entity_identity"] for row in receipts} == {"continuity-entity"}
    checkpoint_document = io.load_json(checkpoint)
    restored = EventSourcedKernel.restore(checkpoint_document["checkpoint"])
    assert "old-project" in restored.state["unfinished_tasks"]
    assert "history-dependent-new-task" in restored.state["unfinished_tasks"]
    assert restored.query("beliefs")["old-project-ready"]["defeated"]
    assert "model-c" in restored.query("model_fabric")["models"]
