"""Independent recomputation, mutation, and counterfeit checks."""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_io as io
from substrate.final_revision_experiment import paired_effect


def recompute_bed(document: Mapping[str, Any]) -> dict[str, Any]:
    raw_document = document.get("raw_history_scores")
    if not isinstance(raw_document, Mapping) or not raw_document:
        raise io.Refused("raw history scores are absent")
    raw: dict[int, dict[str, float]] = {}
    for key, value in raw_document.items():
        if not isinstance(value, Mapping):
            raise io.Refused("raw history score row is malformed")
        raw[int(key)] = {str(system): float(score) for system, score in value.items()}
    systems = tuple(next(iter(raw.values())))
    if any(set(row) != set(systems) for row in raw.values()):
        raise io.Refused("raw history score systems drift across units")
    means = {system: statistics.fmean(row[system] for row in raw.values()) for system in systems}
    candidate = {seed: row["selected_candidate"] for seed, row in raw.items()}
    s2 = {seed: row["S2_task_independent_monolithic_persistent_core"] for seed, row in raw.items()}
    transcript = {seed: row["full_transcript_replay"] for seed, row in raw.items()}
    stateless = {seed: row["stateless_direct_policy"] for seed, row in raw.items()}
    split = str(document["split"])
    return {
        "mean_scores": means,
        "oracle_headroom": means["oracle"] - means["S2_task_independent_monolithic_persistent_core"],
        "effects": {
            "P3_selected_minus_strongest_persistent_alternative": paired_effect(candidate, s2, identity=f"{split}:selected-minus-s2"),
            "P1_selected_minus_full_transcript_replay": paired_effect(candidate, transcript, identity=f"{split}:selected-minus-transcript"),
            "owned_state_minus_stateless": paired_effect(candidate, stateless, identity=f"{split}:selected-minus-stateless"),
        },
        "independent_histories": len(raw),
        "activation": False,
    }


def recomputation_matches(document: Mapping[str, Any]) -> dict[str, Any]:
    recomputed = recompute_bed(document)
    expected = {
        "mean_scores": document["mean_scores"],
        "oracle_headroom": document["oracle_headroom"],
        "effects": document["effects"],
        "independent_histories": document["independent_histories"],
        "activation": False,
    }
    return {
        "expected_digest": io.digest(expected),
        "recomputed_digest": io.digest(recomputed),
        "exact_match": expected == recomputed,
        "recomputed": recomputed,
        "activation": False,
    }


def _valid_dossier() -> dict[str, Any]:
    return {
        "answer_source": "sealed_after_decision",
        "seed_used_as_feature": False,
        "task_identity_used_as_feature": False,
        "modality_digests": {"image": "i", "video": "v", "audio": "a"},
        "model_content_digests": {"model-a": "a", "model-b": "b"},
        "identity_basis": "owned_semantic_state",
        "state_reset": False,
        "model_support_oracle_output": False,
        "active_perception_free_correct_view": False,
        "body_schema_oracle_affordance": False,
        "learning_split": "construction",
        "resource_parity": True,
        "grok_answers_exposed": False,
        "hidden_template_overlap": False,
        "counterfactual_changed": ["door"],
        "counterfactual_held_fixed": ["weather"],
        "counterfactual_undeclared_changes": [],
        "intervention_kind": "intervention",
        "knowledge_warrant": {"confidence": 0.9, "defeated": False},
        "checkpoint_keys": {"goals", "world_model", "model_fabric"},
        "activation": False,
    }


def verify_dossier(dossier: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    if dossier.get("answer_source") != "sealed_after_decision":
        failures.append("answer_leakage")
    if dossier.get("seed_used_as_feature") is not False:
        failures.append("seed_as_key")
    if dossier.get("task_identity_used_as_feature") is not False:
        failures.append("task_identity_leakage")
    modality_digests = dossier.get("modality_digests")
    if not isinstance(modality_digests, Mapping) or len(set(modality_digests.values())) != len(modality_digests):
        failures.append("modality_aliasing")
    model_digests = dossier.get("model_content_digests")
    if not isinstance(model_digests, Mapping) or len(set(model_digests.values())) != len(model_digests):
        failures.append("same_model_under_multiple_names")
    if dossier.get("identity_basis") != "owned_semantic_state":
        failures.append("transcript_replay_credited_as_identity")
    if dossier.get("state_reset") is not False:
        failures.append("state_reset_hidden_behind_metadata_continuity")
    if dossier.get("model_support_oracle_output") is not False:
        failures.append("model_support_given_oracle_output")
    if dossier.get("active_perception_free_correct_view") is not False:
        failures.append("active_perception_given_free_correct_view")
    if dossier.get("body_schema_oracle_affordance") is not False:
        failures.append("body_schema_given_oracle_affordance")
    if dossier.get("learning_split") != "construction":
        failures.append("learning_uses_held_out_outcomes")
    if dossier.get("resource_parity") is not True:
        failures.append("strong_baseline_receives_less_compute")
    if dossier.get("grok_answers_exposed") is not False:
        failures.append("grok_challenge_pack_leaks_answers")
    if dossier.get("hidden_template_overlap") is not False:
        failures.append("hidden_composition_reuses_training_templates")
    if dossier.get("counterfactual_undeclared_changes"):
        failures.append("counterfactual_changes_undeclared_variables")
    if dossier.get("intervention_kind") != "intervention":
        failures.append("intervention_treated_as_observation")
    warrant = dossier.get("knowledge_warrant")
    if not isinstance(warrant, Mapping) or float(warrant.get("confidence", 0.0)) < 0.8 or warrant.get("defeated") is not False:
        failures.append("unsupported_belief_admitted_as_knowledge")
    checkpoint_keys = dossier.get("checkpoint_keys")
    if not isinstance(checkpoint_keys, set):
        failures.extend(["checkpoint_omits_goals", "checkpoint_omits_scene_state", "checkpoint_omits_model_competence"])
    else:
        if "goals" not in checkpoint_keys:
            failures.append("checkpoint_omits_goals")
        if "world_model" not in checkpoint_keys:
            failures.append("checkpoint_omits_scene_state")
        if "model_fabric" not in checkpoint_keys:
            failures.append("checkpoint_omits_model_competence")
    if dossier.get("activation") is not False:
        failures.append("activation_becomes_true")
    return failures


def _mutate(identity: str) -> dict[str, Any]:
    dossier = _valid_dossier()
    changes: dict[str, tuple[str, Any]] = {
        "answer_leakage": ("answer_source", "visible_before_decision"),
        "seed_as_key": ("seed_used_as_feature", True),
        "task_identity_leakage": ("task_identity_used_as_feature", True),
        "modality_aliasing": ("modality_digests", {"image": "same", "video": "same"}),
        "same_model_under_multiple_names": ("model_content_digests", {"model-a": "same", "model-b": "same"}),
        "transcript_replay_credited_as_identity": ("identity_basis", "transcript"),
        "state_reset_hidden_behind_metadata_continuity": ("state_reset", True),
        "model_support_given_oracle_output": ("model_support_oracle_output", True),
        "active_perception_given_free_correct_view": ("active_perception_free_correct_view", True),
        "body_schema_given_oracle_affordance": ("body_schema_oracle_affordance", True),
        "learning_uses_held_out_outcomes": ("learning_split", "hidden_composition"),
        "strong_baseline_receives_less_compute": ("resource_parity", False),
        "grok_challenge_pack_leaks_answers": ("grok_answers_exposed", True),
        "hidden_composition_reuses_training_templates": ("hidden_template_overlap", True),
        "counterfactual_changes_undeclared_variables": ("counterfactual_undeclared_changes", ["lighting"]),
        "intervention_treated_as_observation": ("intervention_kind", "observation"),
        "unsupported_belief_admitted_as_knowledge": ("knowledge_warrant", {"confidence": 0.2, "defeated": True}),
        "checkpoint_omits_goals": ("checkpoint_keys", {"world_model", "model_fabric"}),
        "checkpoint_omits_scene_state": ("checkpoint_keys", {"goals", "model_fabric"}),
        "checkpoint_omits_model_competence": ("checkpoint_keys", {"goals", "world_model"}),
        "activation_becomes_true": ("activation", True),
    }
    key, value = changes[identity]
    dossier[key] = value
    return dossier


def mutation_report() -> dict[str, Any]:
    baseline_failures = verify_dossier(_valid_dossier())
    rows = []
    for identity in C.MUTATIONS:
        failures = verify_dossier(_mutate(identity))
        rows.append(
            {
                "identity": identity,
                "detected_failures": failures,
                "rejected": identity in failures,
                "survived": identity not in failures,
            }
        )
    survivors = [str(row["identity"]) for row in rows if row["survived"]]
    return {
        "baseline_accepted": not baseline_failures,
        "baseline_failures": baseline_failures,
        "rows": rows,
        "total": len(rows),
        "rejected": len(rows) - len(survivors),
        "survivors": survivors,
        "zero_survivors": not survivors,
        "activation": False,
    }


def counterfeit_report() -> dict[str, Any]:
    counterfeits = {
        "transcript_identity": ("transcript_replay_credited_as_identity",),
        "oracle_augmented_agent": (
            "model_support_given_oracle_output",
            "active_perception_given_free_correct_view",
            "body_schema_given_oracle_affordance",
        ),
        "leaky_hidden_solver": (
            "answer_leakage",
            "seed_as_key",
            "task_identity_leakage",
            "grok_challenge_pack_leaks_answers",
            "hidden_composition_reuses_training_templates",
        ),
        "metadata_continuity": ("state_reset_hidden_behind_metadata_continuity",),
        "incomplete_checkpoint": ("checkpoint_omits_goals", "checkpoint_omits_scene_state", "checkpoint_omits_model_competence"),
    }
    rows = []
    for identity, mutations in counterfeits.items():
        dossier = _valid_dossier()
        for mutation in mutations:
            changed = _mutate(mutation)
            for key, value in changed.items():
                if _valid_dossier().get(key) != value:
                    if key == "checkpoint_keys" and isinstance(value, set):
                        dossier[key] = set(dossier[key]) & value
                    else:
                        dossier[key] = value
        failures = verify_dossier(dossier)
        rows.append(
            {
                "identity": identity,
                "mutations": list(mutations),
                "detected_failures": failures,
                "rejected": set(mutations) <= set(failures),
            }
        )
    return {
        "counterfeits": rows,
        "all_rejected": all(bool(row["rejected"]) for row in rows),
        "activation": False,
    }
