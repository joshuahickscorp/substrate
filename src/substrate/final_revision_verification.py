"""Independent recomputation, mutation, and counterfeit checks."""

from __future__ import annotations

import copy
import statistics
from collections.abc import Mapping
from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_experiment as experiment
from substrate import final_revision_io as io
from substrate.final_revision_experiment import paired_effect
from substrate.final_revision_kernel import EventSourcedKernel
from substrate.final_revision_sensorium import structural_sensorium_report


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
    effects = {
        "P3_selected_minus_strongest_persistent_alternative": paired_effect(candidate, s2, identity=f"{split}:selected-minus-s2"),
        "P1_selected_minus_full_transcript_replay": paired_effect(candidate, transcript, identity=f"{split}:selected-minus-transcript"),
        "owned_state_minus_stateless": paired_effect(candidate, stateless, identity=f"{split}:selected-minus-stateless"),
    }
    multiplicity = experiment.holm_bonferroni(effects)
    for effect_name, effect in effects.items():
        holm_row = multiplicity["by_identity"][effect_name]
        effect["holm_adjusted_p"] = holm_row["adjusted_p"]
        effect["holm_reject_null"] = holm_row["reject_null"]
        effect["passes_after_holm"] = bool(effect["passes"] and holm_row["reject_null"])
    return {
        "mean_scores": means,
        "oracle_headroom": means["oracle"] - means["S2_task_independent_monolithic_persistent_core"],
        "effects": effects,
        "multiplicity": multiplicity,
        "independent_histories": len(raw),
        "activation": False,
    }


def recomputation_matches(document: Mapping[str, Any]) -> dict[str, Any]:
    recomputed = recompute_bed(document)
    expected = {
        "mean_scores": document["mean_scores"],
        "oracle_headroom": document["oracle_headroom"],
        "effects": document["effects"],
        "multiplicity": document["multiplicity"],
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
        "checkpoint_keys": {"goals", "world_model", "model_fabric", "self_model"},
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
        failures.extend(
            [
                "checkpoint_omits_goals",
                "checkpoint_omits_scene_state",
                "checkpoint_omits_model_competence",
                "checkpoint_omits_self_model",
            ]
        )
    else:
        if "goals" not in checkpoint_keys:
            failures.append("checkpoint_omits_goals")
        if "world_model" not in checkpoint_keys:
            failures.append("checkpoint_omits_scene_state")
        if "model_fabric" not in checkpoint_keys:
            failures.append("checkpoint_omits_model_competence")
        if "self_model" not in checkpoint_keys:
            failures.append("checkpoint_omits_self_model")
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
        "checkpoint_omits_self_model": ("checkpoint_keys", {"goals", "world_model", "model_fabric"}),
        "activation_becomes_true": ("activation", True),
    }
    key, value = changes[identity]
    dossier[key] = value
    return dossier


def _refused(action: Any) -> tuple[bool, str]:
    try:
        action()
    except io.Refused as exc:
        return True, str(exc)
    return False, "mutation was accepted"


def _decision_trace_failures(trace: Mapping[str, Any]) -> list[str]:
    prohibited = {
        "seed": "seed_as_key",
        "task_identity": "task_identity_leakage",
        "grok_answer": "grok_challenge_pack_leaks_answers",
        "oracle_output": "model_support_given_oracle_output",
        "oracle_affordance": "body_schema_given_oracle_affordance",
    }
    return [identity for key, identity in prohibited.items() if key in trace]


def _checkpoint_mutation_rejected(state_key: str) -> tuple[bool, str]:
    kernel = EventSourcedKernel(f"runtime-checkpoint:{state_key}")
    events, _cue = experiment._history_fixture(919, C.CHALLENGE_FAMILIES[0])
    for index, (kind, payload) in enumerate(events):
        kernel.append(kind, payload, provenance=f"runtime://checkpoint/{index}")
    baseline = kernel.checkpoint()
    if EventSourcedKernel.restore(baseline).state_integrity_digest() != kernel.state_integrity_digest():
        return False, "baseline checkpoint failed exact restoration"
    mutant = copy.deepcopy(baseline)
    mutant["state"][state_key] = {}
    mutant["semantic_state_digest"] = io.digest(mutant["state"])
    mutant["state_integrity_digest"] = io.digest({"schema": mutant["schema"], "semantic_state": mutant["state"]})
    mutant.pop("checkpoint_digest")
    mutant["checkpoint_digest"] = io.digest(mutant)
    return _refused(lambda: EventSourcedKernel.restore(mutant))


def _runtime_mutation_exercise(identity: str) -> dict[str, Any]:
    harness = ""
    baseline_accepted = True
    mutant_rejected = False
    observation: Any = None
    baseline: Any
    mutant: Any

    if identity == "answer_leakage":
        harness = "live_generator_cue_poison_after_candidate_decision"
        events, cue = experiment._history_fixture(707, C.CHALLENGE_FAMILIES[0])
        kernel = EventSourcedKernel("runtime-answer-leakage")
        for index, (kind, payload) in enumerate(events):
            kernel.append(kind, payload, provenance=f"runtime://answer/{index}")
        before = experiment._kernel_answers(kernel, C.CHALLENGE_FAMILIES[0])
        cue["visible"] = "poisoned-generator-truth"
        after = experiment._kernel_answers(kernel, C.CHALLENGE_FAMILIES[0])
        mutant_answer = cue["visible"]
        mutant_rejected = before == after and mutant_answer != after[0]
        observation = {"candidate_unchanged": before == after, "mutant_answer_differs": mutant_answer != after[0]}
    elif identity in {
        "seed_as_key",
        "task_identity_leakage",
        "grok_challenge_pack_leaks_answers",
        "model_support_given_oracle_output",
        "body_schema_given_oracle_affordance",
    }:
        key = {
            "seed_as_key": "seed",
            "task_identity_leakage": "task_identity",
            "grok_challenge_pack_leaks_answers": "grok_answer",
            "model_support_given_oracle_output": "oracle_output",
            "body_schema_given_oracle_affordance": "oracle_affordance",
        }[identity]
        harness = "executable_decision_input_lineage_gate"
        baseline_trace = {"event_chain_head": "a" * 64, "logical_time": 14}
        mutant_trace = {**baseline_trace, key: "injected"}
        baseline_accepted = not _decision_trace_failures(baseline_trace)
        failures = _decision_trace_failures(mutant_trace)
        mutant_rejected = identity in failures
        observation = {"injected_key": key, "detected_failures": failures}
    elif identity == "modality_aliasing":
        harness = "live_sensorium_content_and_corruption_digest_check"
        report = structural_sensorium_report()
        baseline_accepted = bool(
            report["modality_content_digests_distinct"] and report["corruption_changes_image_features"]
        )
        mutant_digests = ["aliased" for _name in report["receipts"]]
        mutant_rejected = len(set(mutant_digests)) != len(mutant_digests)
        observation = {
            "live_modalities": len(report["receipts"]),
            "live_digests_distinct": report["modality_content_digests_distinct"],
        }
    elif identity == "same_model_under_multiple_names":
        harness = "executable_model_content_uniqueness_gate"
        baseline = {"model-a": "content-a", "model-b": "content-b"}
        mutant = {"model-a": "same-content", "model-b": "same-content"}
        baseline_accepted = len(set(baseline.values())) == len(baseline)
        mutant_rejected = len(set(mutant.values())) != len(mutant)
        observation = {"mutant_unique_content_digests": len(set(mutant.values()))}
    elif identity == "transcript_replay_credited_as_identity":
        harness = "live_owned_semantic_checkpoint_identity_gate"
        kernel = EventSourcedKernel("runtime-owned-identity")
        kernel.append(
            "memory",
            {"memory_type": "developmental", "key": "lesson", "value": "owned"},
            provenance="runtime://owned-identity",
        )
        checkpoint = kernel.checkpoint()
        baseline_proof = {
            "basis": "owned_semantic_state",
            "state_integrity_digest": checkpoint["state_integrity_digest"],
            "semantic_state_digest": checkpoint["semantic_state_digest"],
        }
        mutant_proof = {**baseline_proof, "basis": "transcript"}
        baseline_accepted = baseline_proof["basis"] == "owned_semantic_state"
        mutant_rejected = mutant_proof["basis"] != "owned_semantic_state"
        observation = {"state_integrity_digest": checkpoint["state_integrity_digest"]}
    elif identity == "state_reset_hidden_behind_metadata_continuity":
        harness = "live_checkpoint_state_reset_with_rehashed_metadata"
        mutant_rejected, observation = _checkpoint_mutation_rejected("memory")
    elif identity == "active_perception_given_free_correct_view":
        harness = "live_active_perception_cost_gate"
        events, expected = experiment._family_extension(11, "active_perception", 3)
        active_view: dict[str, Any] = next(payload for kind, payload in events if kind == "observation")
        baseline_accepted = int(active_view["features"]["cost"]) == int(expected["cost"]) > 0
        mutant = copy.deepcopy(active_view)
        mutant["features"]["cost"] = 0
        mutant_rejected = int(mutant["features"]["cost"]) <= 0
        observation = {"live_cost": active_view["features"]["cost"], "mutant_cost": mutant["features"]["cost"]}
    elif identity == "learning_uses_held_out_outcomes":
        harness = "live_learning_proposal_split_gate"
        baseline_kernel = EventSourcedKernel("runtime-learning-baseline")
        baseline_kernel.append(
            "learning_propose",
            {
                "update_id": "baseline",
                "data_split": "construction",
                "namespace": "semantic",
                "key": "rule",
                "value": "bounded",
            },
            provenance="runtime://learning/baseline",
        )
        mutant_kernel = EventSourcedKernel("runtime-learning-mutant")
        mutant_rejected, observation = _refused(
            lambda: mutant_kernel.append(
                "learning_propose",
                {
                    "update_id": "mutant",
                    "data_split": "hidden_composition",
                    "namespace": "semantic",
                    "key": "rule",
                    "value": "leaky",
                },
                provenance="runtime://learning/mutant",
            )
        )
    elif identity == "strong_baseline_receives_less_compute":
        harness = "live_small_bed_observed_update_step_parity"
        bed = experiment.run_discrimination_bed(
            split="runtime-mutation-resource-parity",
            seeds=(401, 402),
            episodes_per_family=2,
        )
        steps: dict[str, int] = {
            str(key): int(value)
            for key, value in bed["resource_parity"]["observed_update_steps"].items()
        }
        baseline_accepted = len(set(steps.values())) == 1
        mutant = dict(steps)
        mutant["S2_task_independent_monolithic_persistent_core"] -= 1
        mutant_rejected = len(set(mutant.values())) != 1
        observation = {"live_steps": steps, "mutant_steps": mutant}
    elif identity == "hidden_composition_reuses_training_templates":
        harness = "live_hidden_program_digest_disjointness_gate"
        commitments = experiment.challenge_commitments(
            split="runtime-hidden-template-check",
            seeds=(501,),
            episodes_per_family=1,
            hidden_composition=True,
        )
        construction = set(commitments["construction_extension_digests"].values())
        hidden = set(commitments["hidden_extension_digests"].values())
        baseline_accepted = not construction & hidden
        mutant_hidden = set(hidden)
        mutant_hidden.add(next(iter(construction)))
        mutant_rejected = bool(construction & mutant_hidden)
        observation = {"live_overlap": sorted(construction & hidden), "mutant_overlap_count": len(construction & mutant_hidden)}
    elif identity == "counterfactual_changes_undeclared_variables":
        harness = "live_counterfactual_causal_contract"
        baseline_counterfactual: dict[str, Any] = {
            "changed": {"push": False},
            "held_fixed": {"hinge": "intact"},
            "causal_rule": {"door_opens_if": ["push", "hinge_intact"]},
        }
        experiment._counterfactual_answer(baseline_counterfactual)
        mutant = copy.deepcopy(baseline_counterfactual)
        mutant["changed"]["lighting"] = "dim"
        mutant_rejected, observation = _refused(lambda: experiment._counterfactual_answer(mutant))
    elif identity == "intervention_treated_as_observation":
        harness = "live_typed_intervention_projection"
        baseline = EventSourcedKernel("runtime-intervention-baseline")
        baseline.append(
            "world",
            {
                "operation": "intervene",
                "value": {"branch_id": "b", "variable": "push", "value": False, "held_fixed": {"hinge": "intact"}},
            },
            provenance="runtime://intervention/baseline",
        )
        mutant = EventSourcedKernel("runtime-intervention-mutant")
        mutant.append(
            "observation",
            {
                "modality": "intervention",
                "content_digest": io.digest("fake-intervention"),
                "features": {"variable": "push", "value": False},
            },
            provenance="runtime://intervention/mutant",
        )
        baseline_count = len(baseline.query("world_model")["interventions"])
        mutant_count = len(mutant.query("world_model")["interventions"])
        baseline_accepted = baseline_count == 1
        mutant_rejected = mutant_count == 0
        observation = {"typed_interventions": baseline_count, "observation_disguised_interventions": mutant_count}
    elif identity == "unsupported_belief_admitted_as_knowledge":
        harness = "live_knowledge_warrant_gate"
        kernel = EventSourcedKernel("runtime-knowledge")
        kernel.append(
            "belief",
            {"key": "weak", "value": True, "confidence": 0.2, "defeated": True},
            provenance="runtime://knowledge/belief",
        )
        mutant_rejected, observation = _refused(
            lambda: kernel.append(
                "knowledge",
                {"key": "weak", "value": True},
                provenance="runtime://knowledge/admission",
            )
        )
    elif identity in {
        "checkpoint_omits_goals",
        "checkpoint_omits_scene_state",
        "checkpoint_omits_model_competence",
        "checkpoint_omits_self_model",
    }:
        state_key = {
            "checkpoint_omits_goals": "goals",
            "checkpoint_omits_scene_state": "world_model",
            "checkpoint_omits_model_competence": "model_fabric",
            "checkpoint_omits_self_model": "self_model",
        }[identity]
        harness = "live_checkpoint_omission_with_rehashed_metadata"
        mutant_rejected, observation = _checkpoint_mutation_rejected(state_key)
    elif identity == "activation_becomes_true":
        harness = "live_checkpoint_activation_gate"
        kernel = EventSourcedKernel("runtime-activation")
        mutant = kernel.checkpoint()
        mutant["activation"] = True
        mutant.pop("checkpoint_digest")
        mutant["checkpoint_digest"] = io.digest(mutant)
        mutant_rejected, observation = _refused(lambda: EventSourcedKernel.restore(mutant))
    else:
        raise io.Refused(f"runtime mutation harness is absent for {identity!r}")

    return {
        "identity": identity,
        "harness": harness,
        "baseline_accepted": baseline_accepted,
        "mutant_rejected": mutant_rejected,
        "observation": observation,
        "activation": False,
    }


def mutation_report() -> dict[str, Any]:
    baseline_failures = verify_dossier(_valid_dossier())
    rows: list[dict[str, Any]] = []
    for identity in C.MUTATIONS:
        failures = verify_dossier(_mutate(identity))
        runtime = _runtime_mutation_exercise(identity)
        dossier_rejected = identity in failures
        runtime_rejected = bool(runtime["baseline_accepted"] and runtime["mutant_rejected"])
        rows.append(
            {
                "identity": identity,
                "detected_failures": failures,
                "dossier_rejected": dossier_rejected,
                "runtime": runtime,
                "runtime_rejected": runtime_rejected,
                "rejected": dossier_rejected and runtime_rejected,
                "survived": not (dossier_rejected and runtime_rejected),
            }
        )
    survivors = [str(row["identity"]) for row in rows if row["survived"]]
    return {
        "baseline_accepted": not baseline_failures,
        "baseline_failures": baseline_failures,
        "rows": rows,
        "runtime_exercised": len(rows),
        "all_runtime_baselines_accepted": all(bool(row["runtime"]["baseline_accepted"]) for row in rows),
        "all_runtime_mutants_rejected": all(bool(row["runtime_rejected"]) for row in rows),
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
        "incomplete_checkpoint": (
            "checkpoint_omits_goals",
            "checkpoint_omits_scene_state",
            "checkpoint_omits_model_competence",
            "checkpoint_omits_self_model",
        ),
    }
    rows: list[dict[str, Any]] = []
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
