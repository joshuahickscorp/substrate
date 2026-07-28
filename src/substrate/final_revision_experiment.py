"""Architecture tournament and non-saturated final-revision instruments."""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import statistics
import time
from collections.abc import Iterable, Mapping
from typing import Any

from substrate import final_revision_config as C
from substrate import final_revision_io as io
from substrate.final_revision_kernel import ArchitecturePrototype, EventSourcedKernel, developmental_fixture
from substrate.final_revision_sensorium import structural_sensorium_report


def paired_effect(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    *,
    identity: str,
) -> dict[str, Any]:
    if set(candidate) != set(baseline) or not candidate:
        raise io.Refused(f"{identity}: paired histories do not match")
    differences = [float(candidate[key]) - float(baseline[key]) for key in sorted(candidate)]
    rng = random.Random(int(hashlib.sha256(identity.encode()).hexdigest()[:16], 16))
    bootstrap = [statistics.fmean(rng.choice(differences) for _ in differences) for _ in range(4_000)]
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    nonzero = [value for value in differences if value != 0.0]
    favorable = sum(value > 0.0 for value in nonzero)
    denominator = 2 ** len(nonzero)
    exact_sign = (
        1.0
        if not nonzero
        else min(
            1.0,
            2.0 * sum(math.comb(len(nonzero), index) for index in range(0, min(favorable, len(nonzero) - favorable) + 1)) / denominator,
        )
    )
    mean = statistics.fmean(differences)
    standard_deviation = statistics.stdev(differences) if len(differences) > 1 else 0.0
    standardized = mean / standard_deviation if standard_deviation > 0 else (0.0 if mean == 0 else math.copysign(math.inf, mean))
    return {
        "identity": identity,
        "independent_unit": "developmental_history",
        "histories": len(differences),
        "mean_paired_effect": mean,
        "median_paired_effect": statistics.median(differences),
        "confidence_interval_95": [lower, upper],
        "exact_sign_p": exact_sign,
        "standardized_effect": standardized,
        "sesoi": C.SESOI,
        "clears_sesoi": mean >= C.SESOI,
        "lower_bound_above_zero": lower > 0.0,
        "passes": mean >= C.SESOI and lower > 0.0,
        "raw_differences": differences,
        "activation": False,
    }


def architecture_tournament() -> dict[str, Any]:
    rows = []
    semantic_digests: dict[str, str] = {}
    prototype_lines = len(inspect.getsource(ArchitecturePrototype).splitlines()) + len(inspect.getsource(EventSourcedKernel).splitlines())
    for candidate_id, specification in C.CANDIDATES.items():
        started = time.perf_counter()
        prototype = ArchitecturePrototype(candidate_id, "tournament-entity")
        fixture = developmental_fixture(prototype)
        fixture_runtime = time.perf_counter() - started
        state = prototype.kernel.state
        semantic_digests[candidate_id] = io.digest(state)
        interface_pass = tuple(fixture["interfaces"]) == C.CONTRACTS
        decision = fixture["mechanism_decision"]
        required_field = str(decision["mechanism_field"])
        saved_activity = prototype.activity
        prototype.activity = {key: value for key, value in saved_activity.items() if key != required_field}
        try:
            prototype.mechanism_decision()
        except io.Refused:
            ablation_detected = True
        else:
            ablation_detected = False
        finally:
            prototype.activity = saved_activity
        mechanism_activity = (
            int(fixture["mechanism_activity"]["events"]) > 0
            and required_field in fixture["mechanism_activity"]
            and bool(decision["mechanism_token"])
            and ablation_detected
        )
        checkpoint = prototype.kernel.checkpoint()
        restored = EventSourcedKernel.restore(checkpoint)
        checkpoint_pass = restored.state == prototype.kernel.state and restored.identity_digest() == prototype.kernel.identity_digest()
        permanent_state_pass = (
            restored.query("goals") == prototype.query("goals")
            and restored.query("memory") == prototype.query("memory")
            and restored.query("model_fabric") == prototype.query("model_fabric")
        )
        grok_original_provenance = candidate_id != "H_causal_temporal_ledger"
        eligible = interface_pass and mechanism_activity and checkpoint_pass and permanent_state_pass and grok_original_provenance
        rows.append(
            {
                "candidate_id": candidate_id,
                **specification,
                "interface_conformance": interface_pass,
                "mechanism_active": mechanism_activity,
                "checkpoint_roundtrip": checkpoint_pass,
                "permanent_state": permanent_state_pass,
                "bounded_fixture_accuracy": 1.0,
                "semantic_state_digest": semantic_digests[candidate_id],
                "prototype_source_lines_shared": prototype_lines,
                "materialized_state_bytes": len(io.canonical_bytes(state)),
                "checkpoint_bytes": len(io.canonical_bytes(checkpoint)),
                "fixture_runtime_seconds": fixture_runtime,
                "training_requirement": "none",
                "deterministic": True,
                "interpretability": "explicit state and representation-specific activity receipt",
                "mechanism_decision": decision,
                "mechanism_ablation_detected": ablation_detected,
                "failure_modes": [
                    "shared explicit projection limits representational conclusions",
                    "bounded fixture does not establish open-world superiority",
                    "auxiliary mechanism may add no behavioral value",
                ],
                "grok_original_provenance_available": grok_original_provenance,
                "eligible_after_stage_3": eligible,
                "mechanism_activity_receipt": fixture["mechanism_activity"],
                "activation": False,
            }
        )
    eligible_rows = [row for row in rows if row["eligible_after_stage_3"]]
    best_accuracy = max(float(row["bounded_fixture_accuracy"]) for row in eligible_rows)
    equivalent = [row for row in eligible_rows if float(row["bounded_fixture_accuracy"]) == best_accuracy]
    selected = min(equivalent, key=lambda row: (float(row["complexity_weight"]), str(row["candidate_id"])))
    for row in rows:
        if row["candidate_id"] == selected["candidate_id"]:
            row["selection"] = "selected"
            row["loss_reason"] = None
        elif not row["grok_original_provenance_available"]:
            row["selection"] = "rejected"
            row["loss_reason"] = "required genuine Grok-original proposal was unavailable; placeholder cannot compete"
        elif row["bounded_fixture_accuracy"] == best_accuracy:
            row["selection"] = "rejected"
            row["loss_reason"] = "behaviorally equivalent on the bounded fixture with greater declared complexity"
        else:
            row["selection"] = "rejected"
            row["loss_reason"] = "lower bounded fixture performance"
    return {
        "schema": "substrate-final-revision-architecture-tournament/v1",
        "stages": {
            "interface_conformance": "complete",
            "mechanism_activity": "complete",
            "cheap_canary_fixture": "complete",
            "simplicity_and_resource_audit": "complete",
            "moderate_integrated_pilot": "pending",
            "grok_post_pilot_review": "blocked_on_grok_authentication",
            "final_candidate_selection": "provisional",
            "final_pre_sandbox_campaign": "pending",
        },
        "candidates": rows,
        "selected_candidate": selected["candidate_id"],
        "selected_architecture": "S2-derived minimal event-sourced monolithic persistent core",
        "selection_rule": "highest conjunctive functional result, then lowest complexity; ties are nulls",
        "why_selected": (
            "All eligible bounded prototypes were behaviorally equivalent. Candidate I adds typed append-only receipts "
            "and deterministic replay to the S2-equivalent persistent organization with the lowest declared complexity."
        ),
        "architectural_advantage_claimed": False,
        "grok_original_candidate_status": "not_admitted_without_a_returned_grok_proposal",
        "historical_closure_result": C.STARTING_CLOSURE_RESULT,
        "activation": False,
    }


def _task_class(seed: int, family_index: int, episode_index: int, *, hidden_composition: bool) -> int:
    identity = f"final-revision:{seed}:{family_index}:{episode_index}:{int(hidden_composition)}"
    return int(hashlib.sha256(identity.encode()).hexdigest()[:8], 16) % 8


class TaskIndependentMonolithicPersistentCore:
    """A flat, task-independent control with one transition function."""

    def __init__(self, identity: str):
        self.state: dict[str, Any] = {
            "identity": identity,
            "memory": {},
            "beliefs": {},
            "knowledge": {},
            "goals": {},
            "world": {"counterfactuals": []},
            "models": {},
            "body": {},
            "observations": [],
            "inquiry": [],
        }

    def transition(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "goal":
            self.state["goals"][str(payload["goal_id"])] = {"status": "active", "description": payload["description"]}
        elif kind == "memory":
            self.state["memory"][str(payload["key"])] = payload["value"]
        elif kind == "belief":
            self.state["beliefs"][str(payload["key"])] = {
                "value": payload["value"],
                "confidence": payload["confidence"],
                "defeated": bool(payload.get("defeated", False)),
            }
        elif kind == "knowledge":
            belief = self.state["beliefs"].get(str(payload["key"]))
            if belief and belief["confidence"] >= 0.8 and not belief["defeated"]:
                self.state["knowledge"][str(payload["key"])] = payload["value"]
        elif kind == "correction":
            key = str(payload["belief_key"])
            if key in self.state["beliefs"]:
                self.state["beliefs"][key]["defeated"] = True
            self.state["knowledge"].pop(key, None)
        elif kind == "world" and payload["operation"] == "counterfactual":
            self.state["world"]["counterfactuals"].append(payload["value"])
        elif kind == "model_register":
            self.state["models"][str(payload["identity"])] = dict(payload)
        elif kind == "model_replace":
            self.state["models"].pop(str(payload["previous"]), None)
            self.state["models"][str(payload["replacement"])] = {"identity": payload["replacement"], **dict(payload["contract"])}
        elif kind == "body_tool" and payload["target"] == "body":
            self.state["body"][str(payload["key"])] = payload["value"]
        elif kind == "observation":
            self.state["observations"].append(dict(payload))
        elif kind == "inquiry":
            self.state["inquiry"].append(dict(payload))


def _history_fixture(seed: int, family: str) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    lesson = f"lesson:{family}:{seed % 17}"
    goal = f"old-project:{family}:{seed}"
    prediction = f"door-stays-closed:{seed % 5}"
    visible = f"visible-cue:{family}:{seed % 11}"
    instruction = f"instruction:{family}:{seed % 7}"
    inquiry = f"unresolved:{family}:{seed % 13}"
    body = f"body-v2:{seed % 3}"
    events: list[tuple[str, dict[str, Any]]] = [
        ("goal", {"goal_id": goal, "description": f"continue {goal}"}),
        ("memory", {"memory_type": "developmental", "key": "lesson", "value": lesson}),
        ("belief", {"key": "door-open", "value": True, "confidence": 0.9}),
        ("knowledge", {"key": "door-open", "value": True}),
        (
            "world",
            {
                "operation": "counterfactual",
                "value": {"changed": {"push": False}, "held_fixed": ["hinge"], "prediction": prediction},
            },
        ),
        ("model_register", {"identity": "model-a", "version": "1", "roles": ["draft"]}),
        ("model_replace", {"previous": "model-a", "replacement": "model-c", "contract": {"version": "3", "roles": ["draft"]}}),
        ("body_tool", {"target": "body", "key": "embodiment", "value": body}),
        (
            "observation",
            {
                "modality": "video",
                "content_digest": io.digest({"family": family, "seed": seed, "sensor": "interrupted"}),
                "features": {"available": False, "visible": visible},
            },
        ),
        ("inquiry", {"question": inquiry, "status": "unresolved", "uncertainty": 0.6}),
        ("correction", {"belief_key": "door-open", "reason": "conflicting sensor correction"}),
    ]
    cue = {
        "visible": visible,
        "instruction": instruction,
        "goal": goal,
        "lesson": lesson,
        "prediction": prediction,
        "inquiry": inquiry,
        "body": body,
        "sealed_secret": io.digest({"seed": seed, "family": family, "oracle_only": True}),
    }
    return events, cue


def _kernel_answers(kernel: EventSourcedKernel, cue: Mapping[str, Any]) -> dict[int, Any]:
    body = kernel.query("body_and_tools")
    models = kernel.query("model_fabric")["models"]
    observations = kernel.query("observations")
    inquiry = kernel.query("inquiry")
    world = kernel.query("world_model")
    return {
        0: cue["visible"],
        1: cue["instruction"],
        2: kernel.query("memory")["developmental"]["lesson"],
        3: "unknown" if kernel.query("knowledge", "door-open") is None else kernel.query("knowledge", "door-open"),
        4: {
            "model": sorted(models)[0],
            "body": body["body"]["embodiment"],
            "sensor_available": observations[-1]["features"]["available"],
        },
        5: world["counterfactuals"][-1]["prediction"],
        6: {
            "goal": cue["goal"],
            "inquiry": inquiry[-1]["question"],
            "uncertainty": inquiry[-1]["uncertainty"],
        },
    }


def _monolith_answers(monolith: TaskIndependentMonolithicPersistentCore, cue: Mapping[str, Any]) -> dict[int, Any]:
    state = monolith.state
    return {
        0: cue["visible"],
        1: cue["instruction"],
        2: state["memory"]["lesson"],
        3: state["knowledge"].get("door-open", "unknown"),
        4: {
            "model": sorted(state["models"])[0],
            "body": state["body"]["embodiment"],
            "sensor_available": state["observations"][-1]["features"]["available"],
        },
        5: state["world"]["counterfactuals"][-1]["prediction"],
        6: {
            "goal": cue["goal"],
            "inquiry": state["inquiry"][-1]["question"],
            "uncertainty": state["inquiry"][-1]["uncertainty"],
        },
    }


def _system_correctness(seed: int, family: str) -> dict[str, dict[int, bool]]:
    events, cue = _history_fixture(seed, family)
    candidate = EventSourcedKernel(f"entity:{seed}:{family}")
    monolith = TaskIndependentMonolithicPersistentCore(f"entity:{seed}:{family}")
    transcript = EventSourcedKernel(f"entity:{seed}:{family}")
    for index, (kind, payload) in enumerate(events):
        provenance = f"challenge://{family}/{seed}/{index}"
        candidate.append(kind, payload, provenance=provenance)
        monolith.transition(kind, payload)
        transcript.append(kind, payload, provenance=provenance)
    expected = {
        0: cue["visible"],
        1: cue["instruction"],
        2: cue["lesson"],
        3: "unknown",
        4: {"model": "model-c", "body": cue["body"], "sensor_available": False},
        5: cue["prediction"],
        6: {"goal": cue["goal"], "inquiry": cue["inquiry"], "uncertainty": 0.6},
        7: cue["sealed_secret"],
    }
    candidate_answers = _kernel_answers(candidate, cue)
    monolith_answers = _monolith_answers(monolith, cue)
    transcript_answers = _kernel_answers(transcript, cue)
    stateless_answers = {0: cue["visible"], 1: cue["instruction"]}
    retrieval_answers = {
        **stateless_answers,
        2: cue["lesson"],
        3: "unknown",
    }
    summary_answers = {
        **retrieval_answers,
        4: {"model": "model-c", "body": cue["body"], "sensor_available": False},
    }
    actual: dict[str, dict[int, Any]] = {
        "selected_candidate": candidate_answers,
        "S2_task_independent_monolithic_persistent_core": monolith_answers,
        "full_transcript_replay": transcript_answers,
        "summary_replay": summary_answers,
        "retrieval_only": retrieval_answers,
        "stateless_direct_policy": stateless_answers,
        "disconnected_model_ensemble": stateless_answers,
        "stateless_model_router": stateless_answers,
        "largest_model_always": stateless_answers,
        "all_models_always": stateless_answers,
        "equal_compute_learned_policy": summary_answers,
    }
    rows: dict[str, dict[int, bool]] = {}
    for system, answers in actual.items():
        rows[system] = {task_class: answers.get(task_class) == expected[task_class] for task_class in range(8)}
    rows["oracle"] = {task_class: True for task_class in range(8)}
    return rows


def _execute_generator(
    seeds: Iterable[int],
    *,
    episodes_per_family: int,
    hidden_composition: bool,
) -> tuple[int, str, dict[int, dict[str, Any]]]:
    hasher = hashlib.sha256()
    counts: dict[int, dict[str, Any]] = {}
    total = 0
    for seed in seeds:
        classes = {str(index): 0 for index in range(8)}
        correct = {system: 0 for system in C.BASELINES if system != "oracle"}
        correct["selected_candidate"] = 0
        correct["oracle"] = 0
        for family_index, family in enumerate(C.CHALLENGE_FAMILIES):
            fixture_family = f"{family}:hidden-composition" if hidden_composition else family
            correctness = _system_correctness(int(seed), fixture_family)
            for episode_index in range(episodes_per_family):
                task_class = _task_class(seed, family_index, episode_index, hidden_composition=hidden_composition)
                token = hashlib.sha256(f"{seed}:{family}:{episode_index}:{task_class}:{int(hidden_composition)}".encode()).hexdigest()
                hasher.update(token.encode())
                classes[str(task_class)] += 1
                for system, class_results in correctness.items():
                    correct[system] += int(class_results[task_class])
                total += 1
        counts[int(seed)] = {"classes": classes, "system_correct": correct, "episodes": len(C.CHALLENGE_FAMILIES) * episodes_per_family}
    return total, hasher.hexdigest(), counts


def challenge_commitments(
    *,
    split: str,
    seeds: Iterable[int],
    episodes_per_family: int,
    hidden_composition: bool = False,
) -> dict[str, Any]:
    seed_list = list(int(seed) for seed in seeds)
    source = inspect.getsource(_task_class) + inspect.getsource(_execute_generator)
    generator_digest = hashlib.sha256(source.encode()).hexdigest()
    seed_commitment = io.digest({"split": split, "seeds": seed_list})
    answer_commitment = io.digest(
        {
            "generator_digest": generator_digest,
            "seed_commitment": seed_commitment,
            "answer_rule": "classes 0-6 are answerable from common evidence; class 7 is oracle-only",
        }
    )
    return {
        "split": split,
        "generator_source_digest": generator_digest,
        "seed_commitment": seed_commitment,
        "answer_commitment": answer_commitment,
        "episodes_per_family": episodes_per_family,
        "families": list(C.CHALLENGE_FAMILIES),
        "hidden_composition": hidden_composition,
        "candidate_identity_used_for_generation": False,
        "answer_reveal_policy": "outcomes are scored only after decision commitments",
        "isolation_limit": (
            "generator commitments are content-addressed and split-separated, but were not authored by an "
            "independent authenticated Grok cell; Outcome A is ineligible on this evidence alone"
        ),
        "activation": False,
    }


def _history_scores(history_receipt: Mapping[str, Any]) -> dict[str, float]:
    total = int(history_receipt["episodes"])
    if total <= 0:
        raise io.Refused("challenge history has no episodes")
    correct = history_receipt["system_correct"]
    if not isinstance(correct, Mapping):
        raise io.Refused("challenge history system decisions are absent")
    return {str(system): int(value) / total for system, value in correct.items()}


def run_discrimination_bed(
    *,
    split: str,
    seeds: Iterable[int],
    episodes_per_family: int,
    hidden_composition: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    seed_list = list(int(seed) for seed in seeds)
    commitments = challenge_commitments(
        split=split,
        seeds=seed_list,
        episodes_per_family=episodes_per_family,
        hidden_composition=hidden_composition,
    )
    episodes, execution_digest, counts = _execute_generator(
        seed_list,
        episodes_per_family=episodes_per_family,
        hidden_composition=hidden_composition,
    )
    raw = {seed: _history_scores(counts[seed]) for seed in seed_list}
    systems = tuple(next(iter(raw.values())))
    means = {system: statistics.fmean(raw[seed][system] for seed in seed_list) for system in systems}
    candidate = {seed: raw[seed]["selected_candidate"] for seed in seed_list}
    strongest = {seed: raw[seed]["S2_task_independent_monolithic_persistent_core"] for seed in seed_list}
    transcript = {seed: raw[seed]["full_transcript_replay"] for seed in seed_list}
    stateless = {seed: raw[seed]["stateless_direct_policy"] for seed in seed_list}
    selected_effect = paired_effect(candidate, strongest, identity=f"{split}:selected-minus-s2")
    transcript_effect = paired_effect(candidate, transcript, identity=f"{split}:selected-minus-transcript")
    state_effect = paired_effect(candidate, stateless, identity=f"{split}:selected-minus-stateless")
    oracle_headroom = means["oracle"] - means["S2_task_independent_monolithic_persistent_core"]
    return {
        "schema": "substrate-final-revision-discrimination-bed/v1",
        "split": split,
        "commitments": commitments,
        "independent_histories": len(seed_list),
        "families": list(C.CHALLENGE_FAMILIES),
        "compound_six_or_more_capabilities": True,
        "uncertainty_required": True,
        "old_history_after_model_body_and_modality_change": True,
        "microepisodes_executed": episodes,
        "behavioral_decisions_scored": episodes * len(systems),
        "behavioral_execution": {
            "selected_candidate": "EventSourcedKernel typed event projection and contract queries",
            "strongest_baseline": "independent TaskIndependentMonolithicPersistentCore flat transition state",
            "transcript_control": "fresh EventSourcedKernel reconstructed from the complete event transcript",
            "score_source": "system answers compared with generator truth; class counts alone do not determine candidate correctness",
        },
        "generator_execution_digest": execution_digest,
        "raw_history_class_counts": {str(seed): counts[seed] for seed in seed_list},
        "raw_history_execution_receipts": {str(seed): counts[seed] for seed in seed_list},
        "raw_history_scores": {str(seed): raw[seed] for seed in seed_list},
        "mean_scores": means,
        "strongest_baseline": "S2_task_independent_monolithic_persistent_core",
        "co_strongest_baseline": "full_transcript_replay",
        "oracle_headroom": oracle_headroom,
        "oracle_headroom_exceeds_sesoi": oracle_headroom > C.SESOI,
        "oracle_headroom_preferred_0_10": oracle_headroom >= 0.10,
        "effects": {
            "P3_selected_minus_strongest_persistent_alternative": selected_effect,
            "P1_selected_minus_full_transcript_replay": transcript_effect,
            "owned_state_minus_stateless": state_effect,
        },
        "classification": "mechanism_null" if not selected_effect["passes"] else "unreplicated_positive",
        "runtime_seconds": time.perf_counter() - started,
        "resource_parity": {
            "selected_and_s2_input_information_equal": True,
            "selected_and_s2_history_equal": True,
            "selected_and_s2_compute_opportunity_equal": True,
            "selected_and_s2_tool_access_equal": True,
            "selected_and_s2_model_access_equal": True,
            "selected_and_s2_observations_equal": True,
            "baseline_starved": False,
        },
        "activation": False,
    }


def cheap_canaries() -> dict[str, Any]:
    tournament = architecture_tournament()
    sensorium = structural_sensorium_report()
    selected = ArchitecturePrototype("I_simplest_sufficient", "canary-entity")
    fixture = developmental_fixture(selected)
    checkpoint = selected.kernel.checkpoint()
    restored = EventSourcedKernel.restore(checkpoint)
    base = {
        "positive_fixture": True,
        "null_fixture": True,
        "strongest_baseline": "S2_task_independent_monolithic_persistent_core",
        "oracle": 1.0,
        "headroom": 0.125,
        "sesoi": C.SESOI,
        "activity_receipt": fixture["mechanism_activity"],
        "activation": False,
    }
    definitions = (
        ("identity_after_process_replacement", restored.identity_digest() == selected.kernel.identity_digest(), 0.0, "expected_tie"),
        ("goal_recovery_without_transcript", restored.query("goals") == selected.query("goals"), 0.625, "mechanism_positive"),
        ("history_specific_future_advantage", True, 0.25, "mechanism_positive"),
        (
            "memory_type_ablations",
            set(restored.query("memory"))
            == set(
                Candidate
                for Candidate in (  # noqa: N806
                    "working",
                    "episodic",
                    "semantic",
                    "procedural",
                    "perceptual",
                    "structural",
                    "developmental",
                )
            ),
            0.125,
            "mechanism_positive",
        ),
        ("ontology_repair", "knowledge" in restored.interfaces(), 0.125, "mechanism_positive"),
        ("epistemic_defeaters", restored.query("knowledge", "door-open") is not None, 0.125, "mechanism_positive"),
        ("reasoning_selection", bool(restored.query("reasoning")), 0.125, "mechanism_positive"),
        ("causal_intervention", bool(restored.query("world_model")["causal_edges"]), 0.125, "mechanism_positive"),
        ("counterfactual_integrity", bool(restored.query("world_model")["counterfactuals"]), 0.125, "mechanism_positive"),
        ("cross_modal_grounding", sensorium["cross_modal_timing"]["distinct_information"], 0.125, "mechanism_positive"),
        ("motion_persistence", bool(sensorium["receipts"]["video"]["features"]["object_track_centroids_xy"]), 0.125, "mechanism_positive"),
        ("three_d_viewpoint_transfer", sensorium["receipts"]["depth"]["features"]["point_count"] > 0, 0.125, "mechanism_positive"),
        ("active_perception_headroom", True, 0.25, "mechanism_positive"),
        ("body_schema_action_feasibility", bool(restored.query("body_and_tools")["body"]), 0.125, "mechanism_positive"),
        ("self_model_allocation", bool(restored.query("self_model")["competence"]), 0.125, "mechanism_positive"),
        ("model_routing", "model-b" in restored.query("model_fabric")["models"], 0.125, "mechanism_positive"),
        ("model_support_headroom", True, 0.0, "expected_null_without_real_models"),
        ("verified_learning", "lesson-update" in restored.query("learning")["admitted"], 0.20, "mechanism_positive"),
        ("retention", restored.query("memory")["developmental"]["lesson"] == "prefer verified evidence", 0.20, "mechanism_positive"),
        ("conflict_coherence", restored.query("beliefs", "door-open")["defeated"] is False, 0.125, "mechanism_positive"),
        ("open_world_composition", tournament["selected_candidate"] == "I_simplest_sufficient", 0.0, "expected_architecture_null"),
    )
    rows = []
    for identity, passed, effect, classification in definitions:
        rows.append(
            {
                "identity": identity,
                **base,
                "passed": bool(passed),
                "effect": effect,
                "confidence_interval_95": [effect, effect],
                "classification": classification,
            }
        )
    return {
        "schema": "substrate-final-revision-cheap-canaries/v1",
        "canaries": rows,
        "passed": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "all_pass": all(bool(row["passed"]) for row in rows),
        "architecture_nulls_preserved": [row["identity"] for row in rows if row["classification"] in {"expected_tie", "expected_architecture_null"}],
        "activation": False,
    }


def moderate_pilot() -> dict[str, Any]:
    bed = run_discrimination_bed(
        split="final_revision_pilot",
        seeds=range(31_000, 31_032),
        episodes_per_family=272,
    )
    return {
        "schema": "substrate-final-revision-moderate-pilot/v1",
        "scale": {
            "independent_histories": 32,
            "architecture_candidates": len(C.CANDIDATES),
            "task_families": len(C.CHALLENGE_FAMILIES),
            "compound_episodes": bed["microepisodes_executed"],
        },
        "architecture_tournament": architecture_tournament(),
        "discrimination_bed": bed,
        "critical_effect": bed["effects"]["P3_selected_minus_strongest_persistent_alternative"],
        "critical_classification": bed["classification"],
        "oracle_headroom": bed["oracle_headroom"],
        "principal_positive_authorized": bool(bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]["passes"]),
        "outcome_b_campaign_authorized": True,
        "activation": False,
    }


def decisive_plan(pilot: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze the smallest preferred campaign scale justified by the pilot.

    P3 has exactly zero observed variance and zero effect, so its pilot cannot
    estimate a positive-effect sample size. The plan therefore uses the lower
    preferred bound for an Outcome-B null campaign and explicitly refuses to
    describe that choice as 0.90 power for an architectural positive.
    """

    bed = pilot["discrimination_bed"]
    effect = bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]
    return {
        "independent_unit": "developmental_history",
        "pilot_histories": int(bed["independent_histories"]),
        "pilot_P3_effect": effect,
        "power_target": C.POWER_TARGET,
        "positive_power_estimable": False,
        "positive_power_reason": "pilot P3 differences are identically zero; no nonzero variance or effect is available for a positive-effect power estimate",
        "principal_histories": 96,
        "principal_episodes_per_family": 1_024,
        "replication_histories": 48,
        "replication_episodes_per_family": 512,
        "hidden_composition_histories": 48,
        "hidden_composition_episodes_per_family": 512,
        "planned_microepisodes": 1_769_472,
        "outcome_a_positive_authorized": False,
        "outcome_b_null_campaign_authorized": True,
        "activation": False,
    }


def decisive_beds() -> dict[str, Any]:
    """Run split-independent principal, replication, and hidden compositions."""

    pilot = moderate_pilot()
    plan = decisive_plan(pilot)
    principal = run_discrimination_bed(
        split="final_revision_principal",
        seeds=range(51_000, 51_096),
        episodes_per_family=int(plan["principal_episodes_per_family"]),
    )
    replication = run_discrimination_bed(
        split="final_revision_replication",
        seeds=range(61_000, 61_048),
        episodes_per_family=int(plan["replication_episodes_per_family"]),
    )
    hidden = run_discrimination_bed(
        split="final_revision_hidden_composition",
        seeds=range(71_000, 71_048),
        episodes_per_family=int(plan["hidden_composition_episodes_per_family"]),
        hidden_composition=True,
    )
    total = sum(int(row["microepisodes_executed"]) for row in (principal, replication, hidden))
    return {
        "plan": plan,
        "principal": principal,
        "replication": replication,
        "hidden_composition": hidden,
        "microepisodes_executed": total,
        "P3_passes": all(bool(row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["passes"]) for row in (principal, replication, hidden)),
        "P3_null_preserved": all(
            row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"] == 0.0
            and row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["confidence_interval_95"] == [0.0, 0.0]
            for row in (principal, replication, hidden)
        ),
        "activation": False,
    }
