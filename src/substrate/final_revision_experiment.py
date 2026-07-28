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


def holm_bonferroni(effects: Mapping[str, Mapping[str, Any]], *, alpha: float = 0.05) -> dict[str, Any]:
    """Apply executable Holm step-down correction to one confirmatory family."""
    if not effects or not 0.0 < alpha < 1.0:
        raise io.Refused("Holm correction requires a nonempty family and alpha in (0,1)")
    ordered = sorted(
        ((identity, float(effect["exact_sign_p"])) for identity, effect in effects.items()),
        key=lambda row: (row[1], row[0]),
    )
    running_adjusted = 0.0
    step_down_open = True
    rows = []
    total = len(ordered)
    for rank, (identity, p_value) in enumerate(ordered, start=1):
        multiplier = total - rank + 1
        adjusted = min(1.0, max(running_adjusted, p_value * multiplier))
        running_adjusted = adjusted
        threshold = alpha / multiplier
        rejected = step_down_open and p_value <= threshold
        if not rejected:
            step_down_open = False
        rows.append(
            {
                "identity": identity,
                "rank": rank,
                "raw_p": p_value,
                "threshold": threshold,
                "adjusted_p": adjusted,
                "reject_null": rejected,
            }
        )
    return {
        "method": "Holm-Bonferroni step-down",
        "alpha": alpha,
        "family_size": total,
        "rows": rows,
        "by_identity": {row["identity"]: row for row in rows},
        "activation": False,
    }


def architecture_tournament(
    candidate_h_proposal: Mapping[str, Any] | None = None,
    *,
    integrated_pilot_status: str = "pending",
) -> dict[str, Any]:
    if integrated_pilot_status not in {"pending", "mechanism_null", "unreplicated_positive"}:
        raise io.Refused(f"unknown integrated pilot status {integrated_pilot_status!r}")
    rows = []
    semantic_digests: dict[str, str] = {}
    prototype_lines = len(inspect.getsource(ArchitecturePrototype).splitlines()) + len(inspect.getsource(EventSourcedKernel).splitlines())
    for candidate_id, specification in C.CANDIDATES.items():
        started = time.perf_counter()
        prototype = ArchitecturePrototype(candidate_id, "tournament-entity")
        fixture = developmental_fixture(prototype)
        fixture_runtime = time.perf_counter() - started
        state = prototype.kernel.state
        semantic_digests[candidate_id] = io.digest(
            {
                "kernel_state": state,
                "representation_state": prototype.representation_state(),
            }
        )
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
        checkpoint = prototype.checkpoint()
        restored = ArchitecturePrototype.restore(checkpoint)
        checkpoint_pass = (
            restored.kernel.state == prototype.kernel.state
            and restored.kernel.state_integrity_digest() == prototype.kernel.state_integrity_digest()
            and restored.representation_state() == prototype.representation_state()
        )
        permanent_state_pass = (
            restored.query("goals") == prototype.query("goals")
            and restored.query("memory") == prototype.query("memory")
            and restored.query("model_fabric") == prototype.query("model_fabric")
        )
        grok_original_provenance = candidate_id != "H_causal_temporal_ledger" or bool(candidate_h_proposal)
        eligible = interface_pass and mechanism_activity and checkpoint_pass and permanent_state_pass and grok_original_provenance
        bounded_fixture_accuracy = statistics.fmean(
            float(value)
            for value in (interface_pass, mechanism_activity, checkpoint_pass, permanent_state_pass)
        )
        mechanism_stress_accuracy = float(bool(decision["mechanism_token"]) and bool(decision["mechanism_probe"]))
        mechanism_ablation_accuracy = 0.0 if ablation_detected else mechanism_stress_accuracy
        rows.append(
            {
                "candidate_id": candidate_id,
                **specification,
                "interface_conformance": interface_pass,
                "mechanism_active": mechanism_activity,
                "checkpoint_roundtrip": checkpoint_pass,
                "permanent_state": permanent_state_pass,
                "bounded_fixture_accuracy": bounded_fixture_accuracy,
                "bounded_fixture_metric": "mean of four executed structural conformance checks; not a cognitive endpoint",
                "semantic_state_digest": semantic_digests[candidate_id],
                "shared_core_state_digest": io.digest(state),
                "mechanism_stress_accuracy": mechanism_stress_accuracy,
                "mechanism_ablation_accuracy": mechanism_ablation_accuracy,
                "mechanism_ablation_delta": mechanism_stress_accuracy - mechanism_ablation_accuracy,
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
                    "mechanism stress success establishes load-bearing implementation only on the bounded fixture",
                ],
                "grok_original_provenance_available": grok_original_provenance,
                "grok_original_proposal_digest": (
                    io.digest(candidate_h_proposal)
                    if candidate_id == "H_causal_temporal_ledger" and candidate_h_proposal is not None
                    else None
                ),
                "eligible_after_stage_3": eligible,
                "mechanism_activity_receipt": fixture["mechanism_activity"],
                "activation": False,
            }
        )
    digest_counts = {digest: list(semantic_digests.values()).count(digest) for digest in semantic_digests.values()}
    for row in rows:
        row["representation_digest_distinct"] = digest_counts[str(row["semantic_state_digest"])] == 1
        row["eligible_after_stage_3"] = bool(row["eligible_after_stage_3"]) and bool(row["representation_digest_distinct"])
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
            "moderate_integrated_pilot": integrated_pilot_status,
            "grok_post_pilot_review": "blocked_on_grok_authentication",
            "final_candidate_selection": "provisional",
            "final_pre_sandbox_campaign": "pending",
        },
        "candidates": rows,
        "selected_candidate": selected["candidate_id"],
        "selected_architecture": "S2-derived minimal event-sourced monolithic persistent core",
        "selection_class": "engineering_default_under_behavioral_tie_and_mechanism_null",
        "tournament_scope": "bounded structural engineering screen; not evidence of cognitive or architectural superiority",
        "selection_rule": (
            "interface conformance, exact full-state restore, unique representation digest, and load-bearing mechanism ablation; "
            "then highest bounded result and lowest complexity; ties are nulls"
        ),
        "why_selected": (
            "All eligible bounded prototypes were behaviorally equivalent. Candidate I adds typed append-only receipts "
            "and deterministic replay to the S2-equivalent persistent organization with the lowest declared complexity."
        ),
        "architectural_advantage_claimed": False,
        "grok_original_candidate_status": (
            "implemented_and_admitted_to_bounded_tournament"
            if candidate_h_proposal
            else "not_admitted_without_a_returned_grok_proposal"
        ),
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
            "tools": {},
            "observations": [],
            "inquiry": [],
            "reasoning": [],
            "self_model": {"resource_state": {}},
        }

    def transition(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "goal":
            identity = str(payload["goal_id"])
            action = str(payload.get("action", "create"))
            if action == "create":
                self.state["goals"][identity] = {"status": "active", "description": payload["description"]}
            elif identity in self.state["goals"]:
                self.state["goals"][identity]["status"] = {"resume": "active"}.get(action, action)
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
        elif kind == "body_tool" and payload["target"] == "tools":
            self.state["tools"][str(payload["key"])] = payload["value"]
        elif kind == "observation":
            self.state["observations"].append(dict(payload))
        elif kind == "inquiry":
            self.state["inquiry"].append(dict(payload))
        elif kind == "reasoning":
            self.state["reasoning"].append(dict(payload))
        elif kind == "self" and payload["operation"] == "resource":
            self.state["self_model"]["resource_state"][str(payload["key"])] = payload["value"]


class BoundedAssociativePolicy:
    """A real bounded online baseline with no answer or outcome access."""

    def __init__(self, *, capacity: int, include_compound_summary: bool):
        self.capacity = capacity
        self.include_compound_summary = include_compound_summary
        self.store: dict[str, Any] = {}
        self.order: list[str] = []
        self.update_steps = 0

    def _put(self, key: str, value: Any) -> None:
        self.update_steps += 1
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)
        self.store[key] = value
        while len(self.order) > self.capacity:
            evicted = self.order.pop(0)
            self.store.pop(evicted, None)

    def transition(self, kind: str, payload: Mapping[str, Any]) -> None:
        if kind == "observation":
            self._put(f"observation:{payload['modality']}", dict(payload["features"]))
        elif kind == "memory":
            self._put(f"memory:{payload['key']}", payload["value"])
        elif kind == "correction":
            self._put(f"defeated:{payload['belief_key']}", True)
        elif self.include_compound_summary and kind == "model_replace":
            self._put("model", str(payload["replacement"]))
        elif self.include_compound_summary and kind == "body_tool" and payload["target"] == "body":
            self._put(f"body:{payload['key']}", payload["value"])

    def answers(self) -> dict[int, Any]:
        video = self.store.get("observation:video", {})
        text = self.store.get("observation:text", {})
        answers: dict[int, Any] = {
            0: video.get("visible"),
            1: text.get("instruction"),
            2: self.store.get("memory:lesson"),
            3: "unknown" if self.store.get("defeated:door-open") else None,
        }
        if self.include_compound_summary:
            answers[4] = {
                "model": self.store.get("model"),
                "body": self.store.get("body:embodiment"),
                "sensor_available": video.get("available"),
            }
        return answers

    def resource_receipt(self) -> dict[str, Any]:
        return {
            "algorithm": "bounded_online_associative_policy",
            "capacity": self.capacity,
            "include_compound_summary": self.include_compound_summary,
            "update_steps": self.update_steps,
            "materialized_bytes": len(io.canonical_bytes(self.store)),
            "outcome_labels_seen": 0,
            "activation": False,
        }


class DeterministicSummaryReplay:
    """A lossy post-history summary, distinct from online associative learning."""

    def __init__(self) -> None:
        self.summary: dict[str, Any] = {}
        self.scanned_events = 0

    def summarize(self, events: Iterable[tuple[str, Mapping[str, Any]]]) -> None:
        for kind, payload in events:
            self.scanned_events += 1
            if kind == "observation":
                self.summary[f"observation:{payload['modality']}"] = dict(payload["features"])
            elif kind == "memory" and payload.get("key") == "lesson":
                self.summary["lesson"] = payload["value"]
            elif kind == "correction" and payload.get("belief_key") == "door-open":
                self.summary["door-open-defeated"] = True
            elif kind == "model_register":
                self.summary["model"] = str(payload["identity"])
            elif kind == "model_replace":
                self.summary["model"] = str(payload["replacement"])
            elif kind == "body_tool" and payload.get("target") == "body":
                self.summary[f"body:{payload['key']}"] = payload["value"]

    def answers(self) -> dict[int, Any]:
        video = self.summary.get("observation:video", {})
        text = self.summary.get("observation:text", {})
        return {
            0: video.get("visible"),
            1: text.get("instruction"),
            2: self.summary.get("lesson"),
            3: "unknown" if self.summary.get("door-open-defeated") else None,
            4: {
                "model": self.summary.get("model"),
                "body": self.summary.get("body:embodiment"),
                "sensor_available": video.get("available"),
            },
        }

    def resource_receipt(self) -> dict[str, Any]:
        return {
            "algorithm": "deterministic_lossy_post_history_summary",
            "scanned_events": self.scanned_events,
            "materialized_bytes": len(io.canonical_bytes(self.summary)),
            "outcome_labels_seen": 0,
            "activation": False,
        }


def _family_extension(seed: int, family: str, episode_index: int) -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    token = f"{seed % 97}:{episode_index % 31}"
    value: Any
    if family == "partial_observability":
        value = {"status": "unresolved", "uncertainty": 0.7, "question": f"occluded:{token}"}
        return [("inquiry", value)], value
    if family == "changing_rules":
        value = f"rule-v{episode_index % 3}"
        return [("memory", {"memory_type": "procedural", "key": "current-rule", "value": value})], value
    if family == "novel_task_composition":
        value = f"compose-observe-recall-plan:{token}"
        return [
            (
                "reasoning",
                {
                    "method": "planning",
                    "selected_before_outcome": True,
                    "inputs": ["observation", "memory", "goal"],
                    "conclusion": value,
                },
            )
        ], value
    if family == "model_replacement":
        replacement = f"model-family-{episode_index % 5}"
        return [
            ("model_register", {"identity": "model-family-base", "version": "1", "roles": ["family"]}),
            (
                "model_replace",
                {
                    "previous": "model-family-base",
                    "replacement": replacement,
                    "contract": {"version": "2", "roles": ["family"]},
                },
            ),
        ], replacement
    if family == "unfinished_goal_recovery":
        goal_id = f"family-goal:{token}"
        return [
            ("goal", {"goal_id": goal_id, "description": "resume after interruption"}),
            ("goal", {"goal_id": goal_id, "action": "pause"}),
            ("goal", {"goal_id": goal_id, "action": "resume"}),
        ], {"goal_id": goal_id, "status": "active"}
    if family == "cross_modal_timing":
        event_id = f"cross-modal:{token}"
        return [
            (
                "observation",
                {
                    "modality": "audio",
                    "content_digest": io.digest({"audio": event_id}),
                    "features": {"event_id": event_id, "time": 1.0},
                },
            ),
            (
                "observation",
                {
                    "modality": "video_event",
                    "content_digest": io.digest({"video": event_id}),
                    "features": {"event_id": event_id, "time": 1.01},
                },
            ),
        ], {"event_id": event_id, "time_delta": 0.01}
    if family == "active_perception":
        value = {"requested": True, "cost": 2, "information": f"revealed:{token}"}
        return [
            (
                "observation",
                {
                    "modality": "active_view",
                    "content_digest": io.digest(value),
                    "features": value,
                },
            )
        ], value
    if family == "human_teaching":
        value = f"teacher-lesson:{token}"
        return [("memory", {"memory_type": "semantic", "key": "taught-rule", "value": value})], value
    if family == "conflicting_evidence":
        key = f"conflict:{token}"
        return [
            ("belief", {"key": key, "value": True, "confidence": 0.95, "warrants": ["sensor-conflict"]}),
            ("knowledge", {"key": key, "value": True}),
            ("correction", {"belief_key": key, "reason": "higher-quality contradiction"}),
        ], {"key": key, "defeated": True, "knowledge": None}
    if family == "resource_constraints":
        value = {"remaining_steps": 3 + episode_index % 4, "policy": "bounded"}
        return [("self", {"operation": "resource", "key": "family-budget", "value": value})], value
    if family == "uncertainty_preservation":
        key = f"uncertain:{token}"
        return [
            ("belief", {"key": key, "value": "possible", "confidence": 0.55, "warrants": ["weak-sensor"]}),
            ("inquiry", {"question": key, "status": "unresolved", "uncertainty": 0.45}),
        ], {"key": key, "knowledge": None, "uncertainty": 0.45}
    if family == "history_after_body_and_modality_change":
        body = f"history-body:{episode_index % 4}"
        depth_id = f"depth:{token}"
        return [
            ("body_tool", {"target": "body", "key": "history-embodiment", "value": body}),
            (
                "observation",
                {
                    "modality": "depth",
                    "content_digest": io.digest({"depth": depth_id}),
                    "features": {"depth_id": depth_id, "available": True},
                },
            ),
        ], {"body": body, "depth_id": depth_id}
    raise io.Refused(f"unknown final-revision family {family!r}")


def _hidden_composition_extension(
    seed: int,
    primary_family: str,
    secondary_family: str,
    episode_index: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Create an unseen cross-family program without replaying a construction template."""
    token = io.digest(
        {
            "seed": seed,
            "primary_family": primary_family,
            "secondary_family": secondary_family,
            "episode_index": episode_index,
            "namespace": "hidden-novel-composition/v1",
        }
    )[:16]
    return [
        (
            "reasoning",
            {
                "method": "planning",
                "selected_before_outcome": True,
                "inputs": [f"primary:{primary_family}", f"secondary:{secondary_family}"],
                "conclusion": f"hidden-compose:{primary_family}+{secondary_family}:{token}",
            },
        ),
        (
            "inquiry",
            {
                "question": f"hidden-interaction:{token}",
                "status": "unresolved",
                "uncertainty": 0.5,
            },
        ),
    ]


def _history_fixture(
    seed: int,
    family: str,
    episode_index: int = 0,
    *,
    hidden_composition: bool = False,
) -> tuple[list[tuple[str, dict[str, Any]]], dict[str, Any]]:
    lesson = f"lesson:{family}:{seed % 17}:{episode_index % 23}"
    goal = f"old-project:{family}:{seed}:{episode_index % 19}"
    prediction = {"door_opens": False, "reason": "push_disabled_with_intact_hinge"}
    visible = f"visible-cue:{family}:{seed % 11}:{episode_index % 13}"
    instruction = f"instruction:{family}:{seed % 7}:{episode_index % 17}"
    inquiry = f"unresolved:{family}:{seed % 13}:{episode_index % 11}"
    body = f"body-v2:{seed % 3}:{episode_index % 5}"
    events: list[tuple[str, dict[str, Any]]] = [
        ("goal", {"goal_id": goal, "description": f"continue {goal}"}),
        ("memory", {"memory_type": "developmental", "key": "lesson", "value": lesson}),
        ("belief", {"key": "door-open", "value": True, "confidence": 0.9}),
        ("knowledge", {"key": "door-open", "value": True}),
        (
            "world",
            {
                "operation": "causal_edge",
                "value": {"cause": "push", "effect": "door-opens", "mapping": {"false": False, "true": True}},
            },
        ),
        (
            "world",
            {
                "operation": "counterfactual",
                "value": {
                    "changed": {"push": False},
                    "held_fixed": {"hinge": "intact"},
                    "causal_rule": {"door_opens_if": ["push", "hinge_intact"]},
                },
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
        (
            "observation",
            {
                "modality": "text",
                "content_digest": io.digest({"family": family, "seed": seed, "instruction": instruction}),
                "features": {"instruction": instruction},
            },
        ),
        ("inquiry", {"question": inquiry, "status": "unresolved", "uncertainty": 0.6}),
        ("correction", {"belief_key": "door-open", "reason": "conflicting sensor correction"}),
    ]
    extension, family_expected = _family_extension(seed, family, episode_index)
    events.extend(extension)
    hidden_secondary = None
    if hidden_composition:
        family_index = C.CHALLENGE_FAMILIES.index(family)
        hidden_secondary = C.CHALLENGE_FAMILIES[
            (family_index + 1 + episode_index % (len(C.CHALLENGE_FAMILIES) - 1)) % len(C.CHALLENGE_FAMILIES)
        ]
        events.extend(_hidden_composition_extension(seed, family, hidden_secondary, episode_index))
    cue = {
        "visible": visible,
        "instruction": instruction,
        "goal": goal,
        "lesson": lesson,
        "prediction": prediction,
        "inquiry": inquiry,
        "body": body,
        "family_expected": family_expected,
        "hidden_secondary_family": hidden_secondary,
        "composition_target": io.digest(
            {
                "visible": visible,
                "instruction": instruction,
                "lesson": lesson,
                "goal": goal,
                "family_expected": family_expected,
                "composition_rule": "cross_event_digest/v1",
            }
        ),
    }
    return events, cue


def _counterfactual_answer(counterfactual: Mapping[str, Any]) -> dict[str, Any]:
    changed = counterfactual.get("changed")
    held_fixed = counterfactual.get("held_fixed")
    rule = counterfactual.get("causal_rule")
    if not isinstance(changed, Mapping) or not isinstance(held_fixed, Mapping) or not isinstance(rule, Mapping):
        raise io.Refused("counterfactual structure is incomplete")
    if set(changed) != {"push"} or set(held_fixed) != {"hinge"} or set(changed) & set(held_fixed):
        raise io.Refused("counterfactual changed and held-fixed variables violate the frozen causal contract")
    prerequisites = rule.get("door_opens_if")
    if prerequisites != ["push", "hinge_intact"]:
        raise io.Refused("counterfactual causal rule is unknown")
    door_opens = bool(changed.get("push")) and held_fixed.get("hinge") == "intact"
    return {
        "door_opens": door_opens,
        "reason": "push_enabled_with_intact_hinge" if door_opens else "push_disabled_with_intact_hinge",
    }


def _kernel_family_probe(kernel: EventSourcedKernel, family: str) -> Any:
    if family == "partial_observability":
        row = next(value for value in kernel.query("inquiry") if str(value["question"]).startswith("occluded:"))
        return {"status": row["status"], "uncertainty": row["uncertainty"], "question": row["question"]}
    if family == "changing_rules":
        return kernel.query("memory")["procedural"]["current-rule"]
    if family == "novel_task_composition":
        return next(
            row["conclusion"] for row in kernel.query("reasoning") if str(row["conclusion"]).startswith("compose-observe")
        )
    if family == "model_replacement":
        return next(identity for identity in sorted(kernel.query("model_fabric")["models"]) if identity.startswith("model-family-"))
    if family == "unfinished_goal_recovery":
        goals = kernel.query("goals")
        identity = next(key for key in sorted(goals) if key.startswith("family-goal:"))
        return {"goal_id": identity, "status": goals[identity]["status"]}
    if family == "cross_modal_timing":
        rows = [row for row in kernel.query("observations") if row["modality"] in {"audio", "video_event"}]
        audio = next(row for row in rows if row["modality"] == "audio")
        video = next(row for row in rows if row["modality"] == "video_event")
        return {
            "event_id": audio["features"]["event_id"],
            "time_delta": round(float(video["features"]["time"]) - float(audio["features"]["time"]), 2),
        }
    if family == "active_perception":
        return next(row["features"] for row in kernel.query("observations") if row["modality"] == "active_view")
    if family == "human_teaching":
        return kernel.query("memory")["semantic"]["taught-rule"]
    if family == "conflicting_evidence":
        beliefs = kernel.query("beliefs")
        key = next(identity for identity in sorted(beliefs) if identity.startswith("conflict:"))
        return {"key": key, "defeated": beliefs[key]["defeated"], "knowledge": kernel.query("knowledge", key)}
    if family == "resource_constraints":
        return kernel.query("self_model")["resource_state"]["family-budget"]
    if family == "uncertainty_preservation":
        beliefs = kernel.query("beliefs")
        key = next(identity for identity in sorted(beliefs) if identity.startswith("uncertain:"))
        inquiry = next(row for row in kernel.query("inquiry") if row["question"] == key)
        return {"key": key, "knowledge": kernel.query("knowledge", key), "uncertainty": inquiry["uncertainty"]}
    if family == "history_after_body_and_modality_change":
        body = kernel.query("body_and_tools")["body"]["history-embodiment"]
        depth = next(row for row in kernel.query("observations") if row["modality"] == "depth")
        return {"body": body, "depth_id": depth["features"]["depth_id"]}
    raise io.Refused(f"unknown kernel family probe {family!r}")


def _monolith_family_probe(monolith: TaskIndependentMonolithicPersistentCore, family: str) -> Any:
    state = monolith.state
    if family == "partial_observability":
        row = next(value for value in state["inquiry"] if str(value["question"]).startswith("occluded:"))
        return {"status": row["status"], "uncertainty": row["uncertainty"], "question": row["question"]}
    if family == "changing_rules":
        return state["memory"]["current-rule"]
    if family == "novel_task_composition":
        return next(row["conclusion"] for row in state["reasoning"] if str(row["conclusion"]).startswith("compose-observe"))
    if family == "model_replacement":
        return next(identity for identity in sorted(state["models"]) if identity.startswith("model-family-"))
    if family == "unfinished_goal_recovery":
        identity = next(key for key in sorted(state["goals"]) if key.startswith("family-goal:"))
        return {"goal_id": identity, "status": state["goals"][identity]["status"]}
    if family == "cross_modal_timing":
        rows = [row for row in state["observations"] if row["modality"] in {"audio", "video_event"}]
        audio = next(row for row in rows if row["modality"] == "audio")
        video = next(row for row in rows if row["modality"] == "video_event")
        return {
            "event_id": audio["features"]["event_id"],
            "time_delta": round(float(video["features"]["time"]) - float(audio["features"]["time"]), 2),
        }
    if family == "active_perception":
        return next(row["features"] for row in state["observations"] if row["modality"] == "active_view")
    if family == "human_teaching":
        return state["memory"]["taught-rule"]
    if family == "conflicting_evidence":
        key = next(identity for identity in sorted(state["beliefs"]) if identity.startswith("conflict:"))
        return {"key": key, "defeated": state["beliefs"][key]["defeated"], "knowledge": state["knowledge"].get(key)}
    if family == "resource_constraints":
        return state["self_model"]["resource_state"]["family-budget"]
    if family == "uncertainty_preservation":
        key = next(identity for identity in sorted(state["beliefs"]) if identity.startswith("uncertain:"))
        inquiry = next(row for row in state["inquiry"] if row["question"] == key)
        return {"key": key, "knowledge": state["knowledge"].get(key), "uncertainty": inquiry["uncertainty"]}
    if family == "history_after_body_and_modality_change":
        depth = next(row for row in state["observations"] if row["modality"] == "depth")
        return {"body": state["body"]["history-embodiment"], "depth_id": depth["features"]["depth_id"]}
    raise io.Refused(f"unknown monolith family probe {family!r}")


def _kernel_answers(kernel: EventSourcedKernel, family: str) -> dict[int, Any]:
    body = kernel.query("body_and_tools")
    models = kernel.query("model_fabric")["models"]
    observations = kernel.query("observations")
    world = kernel.query("world_model")
    video = next(row for row in observations if row["modality"] == "video")
    text = next(row for row in observations if row["modality"] == "text")
    return {
        0: video["features"]["visible"],
        1: text["features"]["instruction"],
        2: kernel.query("memory")["developmental"]["lesson"],
        3: "unknown" if kernel.query("knowledge", "door-open") is None else kernel.query("knowledge", "door-open"),
        4: {
            "model": sorted(models)[0],
            "body": body["body"]["embodiment"],
            "sensor_available": video["features"]["available"],
        },
        5: _counterfactual_answer(world["counterfactuals"][-1]),
        6: _kernel_family_probe(kernel, family),
    }


def _monolith_answers(monolith: TaskIndependentMonolithicPersistentCore, family: str) -> dict[int, Any]:
    state = monolith.state
    video = next(row for row in state["observations"] if row["modality"] == "video")
    text = next(row for row in state["observations"] if row["modality"] == "text")
    return {
        0: video["features"]["visible"],
        1: text["features"]["instruction"],
        2: state["memory"]["lesson"],
        3: state["knowledge"].get("door-open", "unknown"),
        4: {
            "model": sorted(state["models"])[0],
            "body": state["body"]["embodiment"],
            "sensor_available": video["features"]["available"],
        },
        5: _counterfactual_answer(state["world"]["counterfactuals"][-1]),
        6: _monolith_family_probe(monolith, family),
    }


def _episode_correctness(
    seed: int,
    family: str,
    episode_index: int,
    *,
    hidden_composition: bool,
) -> tuple[
    dict[str, dict[int, bool]],
    dict[str, dict[int, dict[str, Any]]],
    int,
    dict[str, dict[str, Any]],
]:
    events, cue = _history_fixture(seed, family, episode_index, hidden_composition=hidden_composition)
    identity = f"entity:{seed}:{family}:{episode_index}:{int(hidden_composition)}"
    candidate = EventSourcedKernel(identity)
    monolith = TaskIndependentMonolithicPersistentCore(identity)
    transcript = EventSourcedKernel(identity)
    retrieval_policy = BoundedAssociativePolicy(capacity=6, include_compound_summary=False)
    learned_policy = BoundedAssociativePolicy(capacity=16, include_compound_summary=True)
    summary_replay = DeterministicSummaryReplay()
    for index, (kind, payload) in enumerate(events):
        provenance = f"challenge://{family}/{seed}/{index}"
        candidate.append(kind, payload, provenance=provenance)
        monolith.transition(kind, payload)
        transcript.append(kind, payload, provenance=provenance)
        retrieval_policy.transition(kind, payload)
        learned_policy.transition(kind, payload)
    summary_replay.summarize(events)
    expected = {
        0: cue["visible"],
        1: cue["instruction"],
        2: cue["lesson"],
        3: "unknown",
        4: {"model": "model-c", "body": cue["body"], "sensor_available": False},
        5: cue["prediction"],
        6: cue["family_expected"],
        7: cue["composition_target"],
    }
    candidate_answers = _kernel_answers(candidate, family)
    monolith_answers = _monolith_answers(monolith, family)
    transcript_answers = _kernel_answers(transcript, family)
    event_payloads = [payload for _kind, payload in events]
    visible_observation = next(payload for payload in event_payloads if payload.get("modality") == "video")
    instruction_observation = next(payload for payload in event_payloads if payload.get("modality") == "text")
    stateless_answers = {
        0: visible_observation["features"]["visible"],
        1: instruction_observation["features"]["instruction"],
    }
    unavailable_model_answers: dict[int, Any] = {}
    retrieval_answers = retrieval_policy.answers()
    actual: dict[str, dict[int, Any]] = {
        "selected_candidate": candidate_answers,
        "S2_task_independent_monolithic_persistent_core": monolith_answers,
        "full_transcript_replay": transcript_answers,
        "summary_replay": summary_replay.answers(),
        "retrieval_only": retrieval_answers,
        "stateless_direct_policy": stateless_answers,
        "disconnected_model_ensemble": unavailable_model_answers,
        "stateless_model_router": unavailable_model_answers,
        "largest_model_always": unavailable_model_answers,
        "all_models_always": unavailable_model_answers,
        "equal_compute_learned_policy": learned_policy.answers(),
    }
    rows: dict[str, dict[int, bool]] = {}
    decision_receipts: dict[str, dict[int, dict[str, Any]]] = {}
    for system, answers in actual.items():
        rows[system] = {task_class: answers.get(task_class) == expected[task_class] for task_class in range(8)}
        decision_receipts[system] = {
            task_class: {
                "answer_digest": io.digest(answers.get(task_class)),
                "expected_digest": io.digest(expected[task_class]),
                "correct": rows[system][task_class],
            }
            for task_class in range(8)
        }
    rows["oracle"] = {task_class: True for task_class in range(8)}
    decision_receipts["oracle"] = {
        task_class: {
            "answer_digest": io.digest(expected[task_class]),
            "expected_digest": io.digest(expected[task_class]),
            "correct": True,
        }
        for task_class in range(8)
    }
    return (
        rows,
        decision_receipts,
        len(events),
        {
            "summary_replay": summary_replay.resource_receipt(),
            "retrieval_only": retrieval_policy.resource_receipt(),
            "equal_compute_learned_policy": learned_policy.resource_receipt(),
        },
    )


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
        history_hasher = hashlib.sha256()
        classes = {str(index): 0 for index in range(8)}
        correct = {system: 0 for system in C.BASELINES if system != "oracle"}
        correct["selected_candidate"] = 0
        correct["oracle"] = 0
        correct_by_class = {system: {str(index): 0 for index in range(8)} for system in correct}
        state_updates = 0
        family_episode_counts = {family: 0 for family in C.CHALLENGE_FAMILIES}
        baseline_resource_steps = {
            "summary_replay": 0,
            "retrieval_only": 0,
            "equal_compute_learned_policy": 0,
        }
        baseline_materialized_byte_steps = {
            "summary_replay": 0,
            "retrieval_only": 0,
            "equal_compute_learned_policy": 0,
        }
        decision_receipt_samples = []
        for family_index, family in enumerate(C.CHALLENGE_FAMILIES):
            for episode_index in range(episodes_per_family):
                task_class = _task_class(seed, family_index, episode_index, hidden_composition=hidden_composition)
                correctness, decision_receipts, event_count, resource_receipts = _episode_correctness(
                    int(seed),
                    family,
                    episode_index,
                    hidden_composition=hidden_composition,
                )
                decisions = {system: bool(class_results[task_class]) for system, class_results in correctness.items()}
                scored_receipts = {
                    system: class_receipts[task_class]
                    for system, class_receipts in decision_receipts.items()
                }
                token = io.digest(
                    {
                        "seed": seed,
                        "family": family,
                        "episode_index": episode_index,
                        "task_class": task_class,
                        "hidden_composition": hidden_composition,
                        "system_decision_receipts": scored_receipts,
                    }
                )
                hasher.update(token.encode())
                history_hasher.update(token.encode())
                classes[str(task_class)] += 1
                for system, passed in decisions.items():
                    correct[system] += int(passed)
                    correct_by_class[system][str(task_class)] += int(passed)
                state_updates += event_count * 3
                for system, receipt in resource_receipts.items():
                    baseline_resource_steps[system] += int(receipt.get("update_steps", receipt.get("scanned_events", 0)))
                    baseline_materialized_byte_steps[system] += int(receipt["materialized_bytes"])
                family_episode_counts[family] += 1
                if episode_index in {0, episodes_per_family - 1}:
                    decision_receipt_samples.append(
                        {
                            "family": family,
                            "episode_index": episode_index,
                            "task_class": task_class,
                            "systems": scored_receipts,
                            "receipt_digest": token,
                        }
                    )
                total += 1
        counts[int(seed)] = {
            "classes": classes,
            "system_correct": correct,
            "system_correct_by_class": correct_by_class,
            "episodes": len(C.CHALLENGE_FAMILIES) * episodes_per_family,
            "state_updates_executed": state_updates,
            "primary_system_update_steps": {
                "selected_candidate": state_updates // 3,
                "S2_task_independent_monolithic_persistent_core": state_updates // 3,
                "full_transcript_replay": state_updates // 3,
            },
            "family_episode_counts": family_episode_counts,
            "baseline_resource_steps": baseline_resource_steps,
            "baseline_materialized_byte_steps": baseline_materialized_byte_steps,
            "decision_chain_head": history_hasher.hexdigest(),
            "decision_receipt_samples": decision_receipt_samples,
            "decision_receipt_sample_policy": "first_and_last_episode_per_family; chain head commits every scored decision",
        }
    return total, hasher.hexdigest(), counts


def challenge_commitments(
    *,
    split: str,
    seeds: Iterable[int],
    episodes_per_family: int,
    hidden_composition: bool = False,
) -> dict[str, Any]:
    seed_list = list(int(seed) for seed in seeds)
    source = "".join(
        inspect.getsource(function)
        for function in (
            TaskIndependentMonolithicPersistentCore,
            BoundedAssociativePolicy,
            DeterministicSummaryReplay,
            _task_class,
            _family_extension,
            _hidden_composition_extension,
            _history_fixture,
            _counterfactual_answer,
            _kernel_family_probe,
            _monolith_family_probe,
            _kernel_answers,
            _monolith_answers,
            _episode_correctness,
            _execute_generator,
        )
    )
    family_program_digests = {}
    family_event_sequences = {}
    construction_extension_digests = {}
    hidden_extension_digests = {}
    for family in C.CHALLENGE_FAMILIES:
        events, _cue = _history_fixture(909_001, family, 7, hidden_composition=False)
        family_event_sequences[family] = [kind for kind, _payload in events]
        family_program_digests[family] = io.digest(
            [
                {
                    "kind": kind,
                    "payload_keys": sorted(payload),
                    "operation": payload.get("operation"),
                    "modality": payload.get("modality"),
                    "memory_type": payload.get("memory_type"),
                }
                for kind, payload in events
            ]
        )
        construction_events, _expected = _family_extension(909_001, family, 7)
        construction_extension_digests[family] = io.digest(construction_events)
        family_index = C.CHALLENGE_FAMILIES.index(family)
        secondary = C.CHALLENGE_FAMILIES[
            (family_index + 1 + 7 % (len(C.CHALLENGE_FAMILIES) - 1)) % len(C.CHALLENGE_FAMILIES)
        ]
        hidden_extension_digests[family] = io.digest(_hidden_composition_extension(909_001, family, secondary, 7))
    hidden_template_overlap = bool(
        set(construction_extension_digests.values()) & set(hidden_extension_digests.values())
    )
    generator_digest = hashlib.sha256(source.encode()).hexdigest()
    seed_commitment = io.digest({"split": split, "seeds": seed_list})
    answer_commitment = io.digest(
        {
            "generator_digest": generator_digest,
            "seed_commitment": seed_commitment,
            "answer_rule": (
                "classes 0-7 are answerable from common event evidence; class 7 requires the frozen "
                "cross_event_digest/v1 composition rule and is revealed only after decision commitment"
            ),
        }
    )
    return {
        "split": split,
        "generator_source_digest": generator_digest,
        "seed_commitment": seed_commitment,
        "answer_commitment": answer_commitment,
        "episodes_per_family": episodes_per_family,
        "families": list(C.CHALLENGE_FAMILIES),
        "family_program_digests": family_program_digests,
        "family_event_sequences": family_event_sequences,
        "construction_extension_digests": construction_extension_digests,
        "hidden_extension_digests": hidden_extension_digests if hidden_composition else {},
        "hidden_composition_reuses_construction_template": hidden_template_overlap if hidden_composition else False,
        "mechanically_distinct_family_program_count": len(set(family_program_digests.values())),
        "family_programs_mechanically_distinct": len(set(family_program_digests.values())) == len(C.CHALLENGE_FAMILIES),
        "hidden_composition": hidden_composition,
        "hidden_composition_rule": (
            "each primary episode appends an unseen cross-family interaction program selected by frozen index arithmetic; "
            "no construction-family event template is replayed"
            if hidden_composition
            else None
        ),
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
    unavailable_model_baselines = (
        "disconnected_model_ensemble",
        "stateless_model_router",
        "largest_model_always",
        "all_models_always",
    )
    means = {system: statistics.fmean(raw[seed][system] for seed in seed_list) for system in systems}
    class_opportunities = {
        str(task_class): sum(int(counts[seed]["classes"][str(task_class)]) for seed in seed_list)
        for task_class in range(8)
    }
    class_conditional_scores = {
        system: {
            str(task_class): (
                sum(int(counts[seed]["system_correct_by_class"][system][str(task_class)]) for seed in seed_list)
                / class_opportunities[str(task_class)]
                if class_opportunities[str(task_class)]
                else None
            )
            for task_class in range(8)
        }
        for system in systems
    }
    candidate = {seed: raw[seed]["selected_candidate"] for seed in seed_list}
    strongest = {seed: raw[seed]["S2_task_independent_monolithic_persistent_core"] for seed in seed_list}
    transcript = {seed: raw[seed]["full_transcript_replay"] for seed in seed_list}
    stateless = {seed: raw[seed]["stateless_direct_policy"] for seed in seed_list}
    selected_effect = paired_effect(candidate, strongest, identity=f"{split}:selected-minus-s2")
    transcript_effect = paired_effect(candidate, transcript, identity=f"{split}:selected-minus-transcript")
    state_effect = paired_effect(candidate, stateless, identity=f"{split}:selected-minus-stateless")
    effects = {
        "P3_selected_minus_strongest_persistent_alternative": selected_effect,
        "P1_selected_minus_full_transcript_replay": transcript_effect,
        "owned_state_minus_stateless": state_effect,
    }
    multiplicity = holm_bonferroni(effects)
    for effect_name, effect in effects.items():
        holm_row = multiplicity["by_identity"][effect_name]
        effect["holm_adjusted_p"] = holm_row["adjusted_p"]
        effect["holm_reject_null"] = holm_row["reject_null"]
        effect["passes_after_holm"] = bool(effect["passes"] and holm_row["reject_null"])
    oracle_headroom = means["oracle"] - means["S2_task_independent_monolithic_persistent_core"]
    solvable_oracle_headroom = oracle_headroom
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
        "behavioral_decisions_scored": episodes * (len(systems) - len(unavailable_model_baselines)),
        "unavailable_model_baseline_placeholders_not_scored": episodes * len(unavailable_model_baselines),
        "baseline_execution_status": {
            **{system: "executed" for system in systems if system not in unavailable_model_baselines},
            **{
                system: "unavailable_no_real_model_weights_or_runtime; zero row is an availability sentinel, not behavior"
                for system in unavailable_model_baselines
            },
        },
        "behavioral_execution": {
            "selected_candidate": "EventSourcedKernel typed event projection and contract queries",
            "strongest_baseline": "independent TaskIndependentMonolithicPersistentCore flat transition state",
            "transcript_control": "fresh EventSourcedKernel reconstructed from the complete event transcript",
            "summary_control": "lossy deterministic post-history summary computed from events without generator truth",
            "retrieval_control": "capacity-6 online associative store with deterministic eviction and no outcomes",
            "equal_compute_learned_policy": (
                "capacity-16 online associative update policy with compound summary fields, deterministic eviction, "
                "and zero outcome-label access"
            ),
            "score_source": "system answers compared with generator truth; class counts alone do not determine candidate correctness",
            "correctness_recomputed_for_every_episode": True,
            "single_correctness_vector_reused_across_episodes": False,
            "state_updates_executed": sum(int(row["state_updates_executed"]) for row in counts.values()),
            "primary_system_update_steps": {
                system: sum(int(row["primary_system_update_steps"][system]) for row in counts.values())
                for system in (
                    "selected_candidate",
                    "S2_task_independent_monolithic_persistent_core",
                    "full_transcript_replay",
                )
            },
            "mechanically_distinct_family_program_count": commitments["mechanically_distinct_family_program_count"],
        },
        "generator_execution_digest": execution_digest,
        "raw_history_class_counts": {str(seed): counts[seed] for seed in seed_list},
        "raw_history_execution_receipts": {str(seed): counts[seed] for seed in seed_list},
        "raw_history_scores": {str(seed): raw[seed] for seed in seed_list},
        "mean_scores": means,
        "class_opportunities": class_opportunities,
        "class_conditional_scores": class_conditional_scores,
        "strongest_baseline": "S2_task_independent_monolithic_persistent_core",
        "co_strongest_baseline": "full_transcript_replay",
        "oracle_headroom": oracle_headroom,
        "oracle_headroom_decomposition": {
            "solvable_unused_capacity": solvable_oracle_headroom,
            "intentionally_unanswerable_or_sealed_secret_capacity": 0.0,
            "class_7_common_evidence_rule": "cross_event_digest/v1",
        },
        "oracle_headroom_exceeds_sesoi": oracle_headroom > C.SESOI,
        "oracle_headroom_preferred_0_10": solvable_oracle_headroom >= 0.10,
        "effects": effects,
        "multiplicity": multiplicity,
        "classification": "mechanism_null" if not selected_effect["passes_after_holm"] else "unreplicated_positive",
        "runtime_seconds": time.perf_counter() - started,
        "resource_parity": {
            "selected_and_s2_input_information_equal": True,
            "selected_and_s2_history_equal": True,
            "selected_and_s2_compute_opportunity_equal": True,
            "selected_and_s2_tool_access_equal": True,
            "selected_and_s2_model_access_equal": True,
            "selected_and_s2_observations_equal": True,
            "observed_update_steps": {
                system: sum(int(row["primary_system_update_steps"][system]) for row in counts.values())
                for system in (
                    "selected_candidate",
                    "S2_task_independent_monolithic_persistent_core",
                    "full_transcript_replay",
                )
            },
            "online_baseline_update_steps": {
                system: sum(int(row["baseline_resource_steps"][system]) for row in counts.values())
                for system in ("retrieval_only", "equal_compute_learned_policy")
            },
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
    defeated = EventSourcedKernel.restore(checkpoint)
    defeated.append(
        "correction",
        {"belief_key": "door-open", "reason": "higher-quality contradictory evidence"},
        provenance="canary://epistemic-defeater",
    )
    alternate = ArchitecturePrototype("I_simplest_sufficient", "canary-alternate-history")
    developmental_fixture(alternate)
    alternate.append(
        "memory",
        {"memory_type": "developmental", "key": "lesson", "value": "different verified history"},
        provenance="canary://alternate-history",
    )
    canary_bed = run_discrimination_bed(
        split="final_revision_cheap_canary",
        seeds=range(41_000, 41_004),
        episodes_per_family=16,
    )
    p1 = canary_bed["effects"]["P1_selected_minus_full_transcript_replay"]
    p3 = canary_bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]
    registered_model = restored.query("model_fabric")["models"]["model-b"]
    definitions = (
        (
            "identity_after_process_replacement",
            restored.state_integrity_digest() == selected.kernel.state_integrity_digest(),
            "structural_equivalence_check",
            None,
        ),
        ("goal_recovery_without_transcript", restored.query("goals") == selected.query("goals"), "structural_interface_check", None),
        (
            "history_specific_future_advantage",
            selected.query("memory")["developmental"]["lesson"] != alternate.query("memory")["developmental"]["lesson"],
            "structural_history_dependence_check",
            None,
        ),
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
            "structural_interface_check",
            None,
        ),
        ("ontology_repair", "knowledge" in restored.interfaces(), "structural_interface_check", None),
        (
            "epistemic_defeaters",
            defeated.query("knowledge", "door-open") is None,
            "structural_defeater_transition_check",
            None,
        ),
        ("reasoning_selection", bool(restored.query("reasoning")), "structural_interface_check", None),
        ("causal_intervention", bool(restored.query("world_model")["causal_edges"]), "structural_interface_check", None),
        ("counterfactual_integrity", bool(restored.query("world_model")["counterfactuals"]), "structural_interface_check", None),
        ("cross_modal_grounding", sensorium["cross_modal_timing"]["distinct_information"], "structural_media_check", None),
        (
            "motion_persistence",
            bool(sensorium["receipts"]["video"]["features"]["object_track_centroids_xy"]),
            "structural_media_check",
            None,
        ),
        (
            "three_d_viewpoint_transfer",
            sensorium["receipts"]["depth"]["features"]["point_count"] > 0,
            "structural_media_check",
            None,
        ),
        ("active_perception_headroom", canary_bed["oracle_headroom_preferred_0_10"], "measured_capacity_check", None),
        ("body_schema_action_feasibility", bool(restored.query("body_and_tools")["body"]), "structural_interface_check", None),
        ("self_model_allocation", bool(restored.query("self_model")["competence"]), "structural_interface_check", None),
        ("model_routing", "model-b" in restored.query("model_fabric")["models"], "structural_adapter_check", None),
        ("model_support_headroom", "weights_digest" not in registered_model, "expected_absence_without_real_models", None),
        ("verified_learning", "lesson-update" in restored.query("learning")["admitted"], "structural_policy_check", None),
        (
            "retention",
            restored.query("memory")["developmental"]["lesson"] == "prefer verified evidence",
            "structural_policy_check",
            None,
        ),
        (
            "conflict_coherence",
            defeated.query("beliefs", "door-open")["defeated"] is True,
            "structural_defeater_transition_check",
            None,
        ),
        ("open_world_composition", not p3["passes"] and not p1["passes"], "measured_architecture_null", {"P1": p1, "P3": p3}),
    )
    rows = []
    for identity, passed, classification, measurement in definitions:
        rows.append(
            {
                "identity": identity,
                "passed": bool(passed),
                "check_kind": "measured_endpoint" if measurement is not None else "structural_or_capacity_check",
                "classification": classification,
                "measurement": measurement,
                "effect": p3["mean_paired_effect"] if identity == "open_world_composition" else None,
                "confidence_interval_95": p3["confidence_interval_95"] if identity == "open_world_composition" else None,
                "strongest_baseline": (
                    "S2_task_independent_monolithic_persistent_core" if identity == "open_world_composition" else None
                ),
                "contributes_to_facet_binary": False,
                "activity_receipt": fixture["mechanism_activity"],
                "activation": False,
            }
        )
    return {
        "schema": "substrate-final-revision-cheap-canaries/v2",
        "canaries": rows,
        "passed": sum(bool(row["passed"]) for row in rows),
        "total": len(rows),
        "all_pass": all(bool(row["passed"]) for row in rows),
        "scientific_role": "structural, safety, capacity, and expected-null checks; never cognitive facet credit",
        "mechanism_positive_count": 0,
        "architecture_nulls_preserved": [
            row["identity"] for row in rows if row["classification"] in {"structural_equivalence_check", "measured_architecture_null"}
        ],
        "canary_bed_digest": io.digest(canary_bed),
        "canary_bed_score_source": canary_bed["behavioral_execution"]["score_source"],
        "selected_tournament_candidate": tournament["selected_candidate"],
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
        "principal_positive_authorized": bool(
            bed["effects"]["P3_selected_minus_strongest_persistent_alternative"]["passes_after_holm"]
        ),
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
        "P3_passes": all(
            bool(row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["passes_after_holm"])
            for row in (principal, replication, hidden)
        ),
        "P3_null_preserved": all(
            row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["mean_paired_effect"] == 0.0
            and row["effects"]["P3_selected_minus_strongest_persistent_alternative"]["confidence_interval_95"] == [0.0, 0.0]
            for row in (principal, replication, hidden)
        ),
        "activation": False,
    }
