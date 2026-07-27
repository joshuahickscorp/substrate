from __future__ import annotations

import copy

import pytest

from substrate import v2config as C
from substrate import v2fabric as F
from substrate import v2state as S


def develop(entity: S.DevelopmentalEntity, domain: str, count: int = 12) -> None:
    for index in range(count):
        entity.experience(F.generate_task(0, domain, index), allow_verification=False)


def test_verified_semantic_and_procedural_consolidation_are_causally_used():
    entity = S.DevelopmentalEntity()
    develop(entity, "A")
    assert entity.semantic
    assert entity.procedures
    assert all(record.source_episodes and record.verification_receipts for record in entity.semantic.values())
    assert all(procedure.source_episode_ids and procedure.supporting_receipts for procedure in entity.procedures.values())

    task = F.generate_task(0, "B", 40)
    episode = entity.experience(task, allow_verification=False)
    assert episode.outcome["correct"]
    assert any(component.startswith("procedure:") for component in episode.components_used)
    procedure = next(iter(entity.procedures.values()))
    assert procedure.status == "transferable"
    assert procedure.transfer_ledger[-1]["target_domain"] == "B"
    assert entity.procedure_use_receipts[-1]["correct"]


def test_nonprocedural_controls_do_not_get_consolidated_state():
    for arm in ("fresh_control", "transcript_replay_control", "episodic_only", "semantic_only", "no_procedure"):
        entity = S.DevelopmentalEntity(arm)
        develop(entity, "A")
        assert not entity.procedures
    semantic = S.DevelopmentalEntity("semantic_only")
    develop(semantic, "A")
    assert semantic.semantic


def test_negative_transfer_is_rejected_by_preoutcome_signature():
    entity = S.DevelopmentalEntity()
    develop(entity, "A")
    task = F.generate_task(0, "D", 50)
    episode = entity.experience(task, allow_verification=False)
    assert not any(component.startswith("procedure:") for component in episode.components_used)
    procedure = next(iter(entity.procedures.values()))
    assert procedure.negative_transfer_ledger[-1]["selected"] is False
    assert procedure.negative_transfer_ledger[-1]["preoutcome"] is True


def test_predictions_precede_outcomes_and_credit_names_only_used_components():
    entity = S.DevelopmentalEntity()
    task = F.generate_task(0, "A", 1)
    entity.experience(task, allow_verification=False)
    prediction = entity.self_model.predictions[-1]
    assert prediction.made_at_step < prediction.outcome_step
    credit = entity.credit_ledger[-1]
    assert set(credit["assigned_credit"]) == set(credit["components_used"])
    assert credit["decision_receipt"] == task.identity


def test_generated_unverified_episode_is_refused():
    entity = S.DevelopmentalEntity()
    generated = S.DevelopmentalEpisode(
        identity="generated:1",
        origin="generated",
        domain="A",
        task_signature="conditional ordered selection",
        observation={},
        proposal="select_position_0",
        outcome=None,
        verification=None,
        components_used=[],
        compute=0,
        predicted_accuracy=0.5,
        step=0,
        phase="generated",
        verified=False,
    )
    with pytest.raises(S.Refused, match="generated unverified"):
        entity.promote_generated(generated)


def test_checkpoint_round_trip_covers_all_state_and_refuses_corruption():
    entity = S.DevelopmentalEntity()
    entity.unfinished_tasks.append("resume domain B")
    entity.unresolved_hypotheses.append("whether boundary route transfers")
    develop(entity, "A")
    checkpoint = entity.checkpoint()
    restored = S.DevelopmentalEntity.restore(checkpoint)
    assert restored.identity_hash() == checkpoint["identity"] == entity.identity_hash()
    assert restored.checkpoint() == checkpoint

    for key in ("procedural_memory", "semantic_memory", "credit_ledger", "allocator_state", "active_goals", "unresolved_hypotheses"):
        corrupt = copy.deepcopy(checkpoint)
        value = corrupt["state"][key]
        if isinstance(value, dict):
            value["corrupt"] = True
        else:
            value.append("corrupt")
        with pytest.raises(S.Refused, match="identity"):
            S.DevelopmentalEntity.restore(corrupt)


def test_body_replacement_preserves_cognitive_ownership_and_is_identity_visible():
    entity = S.DevelopmentalEntity()
    entity.uncertainty.append("unresolved transfer boundary")
    develop(entity, "A")
    report = entity.replace_body("tool_dominant")
    assert report["continuing_entity"]
    assert report["goals_preserved"]
    assert report["uncertainty_preserved"]
    assert report["procedures_preserved"]
    assert report["body_change_visible_in_identity"]
    assert S.DevelopmentalEntity.restore(entity.checkpoint()).body_state["name"] == "tool_dominant"


def test_allocation_bed_has_headroom_and_history_policy_beats_simple():
    train = S.allocation_cases(0, 256)
    evaluate = S.allocation_cases(1, 256)
    simple = S.evaluate_allocator("best_fixed_policy", train, evaluate)
    learned = S.evaluate_allocator("tabular_contextual_policy", train, evaluate)
    oracle = S.evaluate_allocator("oracle", train, evaluate)
    assert simple["accuracy"] < 0.95
    assert oracle["mean_utility"] - simple["mean_utility"] > C.SESOI
    assert learned["mean_utility"] - simple["mean_utility"] > C.SESOI
    assert learned["mean_utility"] <= oracle["mean_utility"]
    assert all(row["outcome_revealed_after_action"] for row in learned["rows"])


def test_unsupported_semantic_generalization_is_refused():
    record = S.SemanticRecord(
        id="bad",
        kind="rule",
        content={},
        provenance="",
        source_episodes=[],
        verification_receipts=[],
        confidence=1.0,
        domain_scope=["A"],
        creation_step=0,
    )
    with pytest.raises(S.Refused, match="requires provenance"):
        record.validate()
