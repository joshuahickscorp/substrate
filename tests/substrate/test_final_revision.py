from __future__ import annotations

import copy
import json

import pytest

from substrate import final_revision_campaign as campaign
from substrate import final_revision_config as C
from substrate import final_revision_experiment as experiment
from substrate import final_revision_grok as grok
from substrate import final_revision_io as io
from substrate import final_revision_verification as verification
from substrate.final_revision_continuity import run_segment
from substrate.final_revision_kernel import (
    ArchitecturePrototype,
    EventSourcedKernel,
    developmental_fixture,
    learning_evaluation_receipt,
)
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
    assert len(C.MUTATIONS) == 22
    assert "checkpoint_omits_self_model" in C.MUTATIONS
    assert len(C.REQUIRED_DELIVERABLES) == len(set(C.REQUIRED_DELIVERABLES))
    assert C.STARTING_CLOSURE_RESULT == "terminal_closed_null"
    assert C.SESOI == 0.05
    assert _no_true_activation(C.configuration())


def test_grok_build_import_is_fail_closed_and_preserves_transport_deviation(tmp_path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    contract = tmp_path / "contract.md"
    prompt = (
        "bounded read-only review contract\n"
        f"ROLE: {C.REVIEW_CELLS[0]}\n"
        f"ROUND: {C.REVIEW_ROUNDS[0]}\n"
        "PUBLIC EVIDENCE COMMIT: c9dbf03802e22ee9c4e3d9852a8d67cd9da0cd08\n"
    )
    contract.write_text(prompt)
    (task / "task.md").write_text(prompt)
    output = {
        "role": C.REVIEW_CELLS[0],
        "round": C.REVIEW_ROUNDS[0],
        "facets": [],
    }
    metadata = {
        "task_id": "grok-build-test",
        "mode": "audit",
        "model": "grok-4.5",
        "sandbox": "read-only",
        "repo": str(io.ROOT),
        "session_id": "session-test",
        "started_at": "2026-07-28T00:00:00Z",
    }
    envelope = {
        "text": f"progress frame\n{json.dumps(output)}",
        "stopReason": "EndTurn",
        "sessionId": "session-test",
        "requestId": "request-test",
        "num_turns": 2,
        "modelUsage": {"grok-4.5-build": {"modelCalls": 2}},
    }
    (task / "metadata.json").write_text(json.dumps(metadata))
    (task / "grok-output.json").write_text(json.dumps(envelope))
    (task / "status").write_text("done\n")
    (task / "exit_code").write_text("0\n")
    record = grok.grok_build_record(task, contract)
    assert record["output"] == output
    assert record["transport"]["non_json_prefix_present"]
    assert record["transport"]["redacted_artifacts_only"]
    assert not record["activation"]
    envelope["text"] += "\ntrailing payload"
    (task / "grok-output.json").write_text(json.dumps(envelope))
    with pytest.raises(io.Refused):
        grok.grok_build_record(task, contract)
    rejected = grok.grok_build_rejected_record(task, contract)
    assert rejected["credited"] is False
    assert rejected["output_received"]
    assert rejected["output"] is None
    assert "non-whitespace payload" in rejected["rejection_reason"]


def test_preflight_preserves_historical_closure_null() -> None:
    report = campaign.preflight(publish=False)
    assert report["all_pass"], report["preflight"]["failed"]
    assert report["immutability"]["historical_evidence_untouched"]
    assert report["immutability"]["immutable_null"]["effect"] == 0.0
    assert report["immutability"]["immutable_null"]["confidence_interval_95"] == [0.0, 0.0]
    assert report["grok"]["completed_distinct_reviewer_count"] >= 0
    assert report["grok"]["completed_distinct_reviewer_count"] <= len(C.REVIEW_CELLS)
    assert all(row["reason"].startswith("validation failed:") for row in report["grok"]["rejected_invocations"])


def test_preflight_accepts_detached_ci_without_local_main(monkeypatch: pytest.MonkeyPatch) -> None:
    original = campaign.io.ref_or_none

    def detached(ref: str, *, peel: bool = False) -> str | None:
        return None if ref == "main" else original(ref, peel=peel)

    monkeypatch.setattr(campaign.io, "ref_or_none", detached)
    report = campaign.preflight(publish=False)
    assert report["all_pass"], report["preflight"]["failed"]
    assert report["preflight"]["checks"]["local_main_absent_or_matches_orientation"]
    assert report["preflight"]["checks"]["remote_main_absent_or_matches_orientation"]
    assert report["preflight"]["checks"]["main_orientation_anchored"]


def test_preflight_accepts_nested_detached_clone_anchored_by_immutable_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    original = campaign.io.ref_or_none

    def nested_detached(ref: str, *, peel: bool = False) -> str | None:
        return None if ref in {"main", "origin/main"} else original(ref, peel=peel)

    monkeypatch.setattr(campaign.io, "ref_or_none", nested_detached)
    report = campaign.preflight(publish=False)
    assert report["all_pass"], report["preflight"]["failed"]
    assert report["preflight"]["checks"]["local_main_absent_or_matches_orientation"]
    assert report["preflight"]["checks"]["remote_main_absent_or_matches_orientation"]
    assert report["preflight"]["checks"]["main_orientation_anchored"]


def test_event_kernel_covers_contracts_and_restores_exactly() -> None:
    prototype = ArchitecturePrototype("I_simplest_sufficient", "entity-test")
    fixture = developmental_fixture(prototype)
    checkpoint = prototype.kernel.checkpoint()
    restored = EventSourcedKernel.restore(checkpoint)
    assert tuple(fixture["interfaces"]) == C.CONTRACTS
    assert restored.state == prototype.kernel.state
    assert restored.state_integrity_digest() == prototype.kernel.state_integrity_digest()
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
    causal = EventSourcedKernel("counterfactual-integrity")
    causal.append(
        "world",
        {"operation": "causal_edge", "value": {"cause": "push", "effect": "door-opens"}},
        provenance="test://causal-edge",
    )
    with pytest.raises(io.Refused):
        causal.append(
            "world",
            {
                "operation": "counterfactual",
                "value": {
                    "changed": {"push": False, "lighting": "dim"},
                    "held_fixed": {"hinge": "intact"},
                    "causal_rule": {"door_opens_if": ["push", "hinge_intact"]},
                },
            },
            provenance="test://undeclared-counterfactual",
        )


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
            "evaluation": learning_evaluation_receipt(
                "bad",
                held_out_before=[True, True, True, True],
                held_out_after=[True, True, True, False],
                retention_before=[True, True, True, True],
                retention_after=[True, False, False, False],
            ),
        },
        provenance="test://learning/admit",
    )
    assert any(row["update_id"] == "bad" for row in prototype.query("learning")["rejected"])
    assert prototype.query("memory")["developmental"] == prior
    assert "poison" not in prototype.query("memory")["semantic"]
    prototype.append(
        "learning_propose",
        {
            "update_id": "tampered-evaluation",
            "namespace": "semantic",
            "key": "forged",
            "value": "forged",
            "data_split": "construction",
            "source": "quarantined-teacher",
        },
        provenance="test://learning/tampered-propose",
    )
    tampered = learning_evaluation_receipt(
        "tampered-evaluation",
        held_out_before=[True, False],
        held_out_after=[True, True],
        retention_before=[True, True],
        retention_after=[True, True],
    )
    tampered["computed"]["held_out_after"] = 0.0
    with pytest.raises(io.Refused, match="corrupt or summary-injected"):
        prototype.append(
            "learning_admit",
            {"update_id": "tampered-evaluation", "evaluation": tampered},
            provenance="test://learning/tampered-admit",
        )
    assert "forged" not in prototype.query("memory")["semantic"]


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
    assert all(row["mechanism_ablation_delta"] == 1.0 for row in rows.values())
    assert len({row["semantic_state_digest"] for row in rows.values()}) == len(rows)
    assert all(row["representation_digest_distinct"] for row in rows.values())
    assert len({row["mechanism_decision"]["mechanism_field"] for row in rows.values()}) == len(rows)
    admitted = experiment.architecture_tournament(
        {
            "candidate_id": "H_causal_temporal_ledger",
            "name": "Intervention-Indexed Dual-Timeline Causal Ledger",
            "provenance": "test-grok-proposal",
        }
    )
    admitted_rows = {row["candidate_id"]: row for row in admitted["candidates"]}
    assert admitted_rows["H_causal_temporal_ledger"]["eligible_after_stage_3"]
    assert admitted_rows["H_causal_temporal_ledger"]["mechanism_decision"]["mechanism_probe"]["projected_delta"] == {
        "door-angle": 0
    }
    assert admitted["selected_candidate"] == "I_simplest_sufficient"


def test_new_bed_has_headroom_but_preserves_architecture_null() -> None:
    bed = experiment.run_discrimination_bed(
        split="test_final_revision",
        seeds=range(8),
        episodes_per_family=16,
    )
    assert bed["oracle_headroom_preferred_0_10"]
    assert bed["oracle_headroom"] > C.SESOI
    assert bed["oracle_headroom_decomposition"]["intentionally_unanswerable_or_sealed_secret_capacity"] == 0.0
    assert bed["class_conditional_scores"]["selected_candidate"]["7"] == 0.0
    assert bed["class_conditional_scores"]["oracle"]["7"] == 1.0
    assert bed["commitments"]["family_programs_mechanically_distinct"]
    assert bed["commitments"]["mechanically_distinct_family_program_count"] == len(C.CHALLENGE_FAMILIES)
    assert not bed["commitments"]["hidden_composition_reuses_construction_template"]
    assert bed["behavioral_execution"]["correctness_recomputed_for_every_episode"]
    assert not bed["behavioral_execution"]["single_correctness_vector_reused_across_episodes"]
    assert bed["behavioral_execution"]["state_updates_executed"] > bed["microepisodes_executed"]
    assert all(
        str(status).startswith("unavailable_no_real_model")
        for system, status in bed["baseline_execution_status"].items()
        if system in {"disconnected_model_ensemble", "stateless_model_router", "largest_model_always", "all_models_always"}
    )
    assert all(
        set(row["family_episode_counts"]) == set(C.CHALLENGE_FAMILIES)
        and all(count == 16 for count in row["family_episode_counts"].values())
        and len(row["decision_receipt_samples"]) == 2 * len(C.CHALLENGE_FAMILIES)
        and len(row["decision_chain_head"]) == 64
        for row in bed["raw_history_execution_receipts"].values()
    )
    effect = bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]
    assert effect["mean_paired_effect"] == 0.0
    assert effect["confidence_interval_95"] == [0.0, 0.0]
    assert not effect["passes"]
    assert not effect["passes_after_holm"]
    assert bed["multiplicity"]["method"] == "Holm-Bonferroni step-down"
    assert bed["multiplicity"]["family_size"] == 3
    transcript = bed["effects"]["P1_selected_minus_full_transcript_replay"]
    assert transcript["mean_paired_effect"] == 0.0
    assert bed["classification"] == "mechanism_null"


def test_generator_cues_are_not_candidate_answer_inputs() -> None:
    events, cue = experiment._history_fixture(7, C.CHALLENGE_FAMILIES[0])
    kernel = EventSourcedKernel("cue-leakage-test")
    for index, (kind, payload) in enumerate(events):
        kernel.append(kind, payload, provenance=f"test://cue-leakage/{index}")
    answers_before = experiment._kernel_answers(kernel, C.CHALLENGE_FAMILIES[0])
    cue["visible"] = "poisoned-visible-answer"
    cue["instruction"] = "poisoned-instruction-answer"
    cue["prediction"] = "poisoned-counterfactual-answer"
    cue["composition_target"] = "poisoned-composition-answer"
    assert experiment._kernel_answers(kernel, C.CHALLENGE_FAMILIES[0]) == answers_before
    assert answers_before[0] != cue["visible"]
    assert answers_before[1] != cue["instruction"]
    assert answers_before[5] != cue["prediction"]
    assert 7 not in answers_before
    counterfactual = next(payload["value"] for kind, payload in events if kind == "world" and payload["operation"] == "counterfactual")
    assert "prediction" not in counterfactual


def test_summary_replay_derives_answers_from_events_not_generator_truth() -> None:
    events, cue = experiment._history_fixture(17, C.CHALLENGE_FAMILIES[0])
    summary = experiment.DeterministicSummaryReplay()
    summary.summarize(events)
    answers_before = summary.answers()
    cue["visible"] = "poisoned-visible-answer"
    cue["body"] = "poisoned-body-answer"
    assert summary.answers() == answers_before
    assert answers_before[0] != cue["visible"]
    assert answers_before[4]["body"] != cue["body"]


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
    assert report["mechanism_positive_count"] == 0
    assert all(not row["contributes_to_facet_binary"] for row in report["canaries"])
    measured = [row for row in report["canaries"] if row["check_kind"] == "measured_endpoint"]
    assert len(measured) == 1
    assert measured[0]["effect"] == 0.0
    assert measured[0]["confidence_interval_95"] == [0.0, 0.0]


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
    assert mutation["rejected"] == mutation["total"] == 22
    assert mutation["zero_survivors"]
    assert mutation["runtime_exercised"] == 22
    assert mutation["all_runtime_baselines_accepted"]
    assert mutation["all_runtime_mutants_rejected"]
    assert all(row["runtime"]["harness"] for row in mutation["rows"])
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
