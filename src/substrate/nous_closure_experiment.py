"""Frozen instruments and strongest fair controls for Nous Closure."""

from __future__ import annotations

import hashlib
import math
import random
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from substrate import nous_closure_config as C
from substrate import nous_closure_io as io
from substrate import v5experiment

CONSTRUCTION_SEEDS = tuple(range(12_000, 12_024))
PILOT_SEEDS = tuple(range(13_000, 13_032))
OPEN_WORLD_PROBE_SEEDS = tuple(range(14_000, 14_016))


def _signed(identity: str) -> float:
    value = int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16)
    return 2.0 * (value / 0xFFFFFFFFFFFFFFFF) - 1.0


def direct_policy_score(observation: Mapping[str, Any], rule: str) -> float:
    """Outcome-blind policy: the signature intentionally excludes the target."""

    modalities = [float(value) for value in observation["modality_cues"].values()]
    mechanisms = [float(value) for value in observation["mechanism_cues"].values()]
    combined = (
        modalities
        + mechanisms
        + [
            float(observation["active_view_cue"]),
            float(observation["verification_cue"]),
            float(observation["teacher_cue"]),
        ]
    )
    if rule == "mean_all":
        return statistics.fmean(combined)
    if rule == "median_all":
        return statistics.median(combined)
    if rule == "mean_mechanisms":
        return statistics.fmean(mechanisms or modalities)
    if rule == "median_mechanisms":
        return statistics.median(mechanisms or modalities)
    if rule == "trimmed_all":
        ordered = sorted(combined)
        trim = len(ordered) // 6
        return statistics.fmean(ordered[trim : len(ordered) - trim] if trim else ordered)
    if rule == "median_modalities":
        return statistics.median(modalities)
    if rule == "mean_modalities":
        return statistics.fmean(modalities)
    if rule == "verification":
        return float(observation["verification_cue"])
    if rule == "active":
        return float(observation["active_view_cue"])
    if rule == "teacher":
        return float(observation["teacher_cue"])
    raise ValueError(f"unknown direct-policy rule {rule!r}")


def _public_task(split: str, seed: int, phase: int, episode: int) -> tuple[dict[str, Any], int]:
    # The frozen v5 generator returns target separately. It is revealed only after
    # direct_policy_score has committed its outcome-blind score.
    _identity, observation, target = v5experiment._public_task(split, seed, phase, episode)
    return observation, int(target)


def _rule_samples(split: str, seeds: Iterable[int]) -> dict[int, dict[str, list[float]]]:
    rows: dict[int, dict[str, list[float]]] = {phase: {rule: [] for rule in C.DIRECT_POLICY_RULES} for phase in range(20)}
    for seed in seeds:
        for phase in range(20):
            for episode in range(v5experiment.EPISODES_PER_PHASE):
                observation, target = _public_task(split, int(seed), phase, episode)
                for rule in C.DIRECT_POLICY_RULES:
                    score = direct_policy_score(observation, rule)
                    rows[phase][rule].append(float(int(score >= 0.0) == target))
    return rows


def construction_selection() -> dict[str, Any]:
    samples = _rule_samples("closure_construction", CONSTRUCTION_SEEDS)
    selection: dict[int, str] = {}
    table: dict[str, Any] = {}
    for phase, rules in samples.items():
        means = {rule: statistics.fmean(values) for rule, values in rules.items()}
        selected = max(
            C.DIRECT_POLICY_RULES,
            key=lambda rule: (means[rule], -C.DIRECT_POLICY_RULES.index(rule)),
        )
        selection[phase] = selected
        table[str(phase)] = {
            "selected_rule": selected,
            "selected_accuracy": means[selected],
            "rule_accuracies": means,
        }
    return {
        "split": "closure_construction",
        "history_seeds": list(CONSTRUCTION_SEEDS),
        "selection_rule": "highest construction accuracy; declaration-order tie break",
        "phase_rules": {str(key): value for key, value in selection.items()},
        "table": table,
        "target_available_to_policy": False,
        "selection_frozen_before_pilot": True,
        "activation": False,
    }


def _candidate_histories(split: str, seeds: Iterable[int]) -> dict[int, dict[str, Any]]:
    histories: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        state: dict[str, Any] = {}
        phase_accuracies: list[float] = []
        utilities: list[float] = []
        event_digests: list[str] = []
        for phase in range(20):
            result = v5experiment.phase_result(
                split=split,
                history_seed=int(seed),
                arm="full_v5",
                phase_index=phase,
                development_state=state,
            )
            state = dict(result["development_update"])
            phase_accuracies.append(float(result["accuracy"]))
            utilities.append(float(result["utility"]))
            event_digests.append(str(result["event_digest"]))
        histories[int(seed)] = {
            "accuracy": statistics.fmean(phase_accuracies),
            "utility": statistics.fmean(utilities),
            "phase_accuracies": phase_accuracies,
            "terminal_state_digest": io.digest(state),
            "event_chain_digest": io.digest(event_digests),
        }
    return histories


def _direct_histories(
    split: str,
    seeds: Iterable[int],
    phase_rules: Mapping[int, str],
    *,
    history_calibration: bool = False,
) -> dict[int, dict[str, Any]]:
    histories: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        correct_by_phase: list[float] = []
        correction_total = 0.0
        correction_count = 0
        receipt_rows: list[dict[str, Any]] = []
        for phase in range(20):
            outcomes: list[float] = []
            for episode in range(v5experiment.EPISODES_PER_PHASE):
                observation, target = _public_task(split, int(seed), phase, episode)
                raw_score = direct_policy_score(observation, phase_rules[phase])
                correction = correction_total / correction_count if correction_count else 0.0
                committed_score = raw_score + (0.15 * correction if history_calibration else 0.0)
                decision = int(committed_score >= 0.0)
                outcomes.append(float(decision == target))
                revealed_target = 1.0 if target else -1.0
                correction_total += revealed_target - max(-1.0, min(1.0, raw_score))
                correction_count += 1
                receipt_rows.append(
                    {
                        "phase": phase,
                        "episode": episode,
                        "observation_digest": io.digest(observation),
                        "rule": phase_rules[phase],
                        "committed_decision": decision,
                        "outcome_revealed_after_commit": target,
                    }
                )
            correct_by_phase.append(statistics.fmean(outcomes))
        histories[int(seed)] = {
            "accuracy": statistics.fmean(correct_by_phase),
            "utility": statistics.fmean(correct_by_phase),
            "phase_accuracies": correct_by_phase,
            "terminal_state_digest": io.digest(
                {
                    "correction_total": correction_total,
                    "correction_count": correction_count,
                }
                if history_calibration
                else {}
            ),
            "event_chain_digest": io.digest(receipt_rows),
        }
    return histories


def paired_effect(
    candidate: Mapping[int, float],
    baseline: Mapping[int, float],
    *,
    identity: str,
) -> dict[str, Any]:
    if set(candidate) != set(baseline) or not candidate:
        raise io.Refused(f"{identity}: paired histories do not match")
    differences = [float(candidate[key]) - float(baseline[key]) for key in sorted(candidate)]
    rng = random.Random(int(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16], 16))
    bootstrap = [statistics.fmean(rng.choice(differences) for _ in differences) for _ in range(4_000)]
    bootstrap.sort()
    lower = bootstrap[int(0.025 * (len(bootstrap) - 1))]
    upper = bootstrap[int(0.975 * (len(bootstrap) - 1))]
    nonzero = [value for value in differences if value != 0.0]
    favorable = sum(value > 0.0 for value in nonzero)
    denominator = 2 ** len(nonzero)
    tail = (
        1.0
        if not nonzero
        else min(
            1.0,
            2.0 * sum(math.comb(len(nonzero), index) for index in range(0, min(favorable, len(nonzero) - favorable) + 1)) / denominator,
        )
    )
    mean = statistics.fmean(differences)
    return {
        "identity": identity,
        "independent_unit": "developmental_history",
        "histories": len(differences),
        "mean_paired_effect": mean,
        "median_paired_effect": statistics.median(differences),
        "confidence_interval_95": [lower, upper],
        "exact_sign_p": tail,
        "sesoi": C.SESOI,
        "clears_sesoi": mean >= C.SESOI,
        "lower_bound_above_zero": lower > 0.0,
        "passes": mean >= C.SESOI and lower > 0.0,
        "raw_differences": differences,
        "activation": False,
    }


def v5_bed_pilot() -> dict[str, Any]:
    selection = construction_selection()
    rules = {int(key): str(value) for key, value in selection["phase_rules"].items()}
    candidate = _candidate_histories("closure_pilot", PILOT_SEEDS)
    direct = _direct_histories("closure_pilot", PILOT_SEEDS, rules)
    calibrated = _direct_histories(
        "closure_pilot",
        PILOT_SEEDS,
        rules,
        history_calibration=True,
    )
    candidate_accuracy = {seed: row["accuracy"] for seed, row in candidate.items()}
    direct_accuracy = {seed: row["accuracy"] for seed, row in direct.items()}
    calibrated_accuracy = {seed: row["accuracy"] for seed, row in calibrated.items()}
    original_effect = paired_effect(
        candidate_accuracy,
        direct_accuracy,
        identity="v5_full_minus_stateless_direct",
    )
    robust_effect = paired_effect(
        direct_accuracy,
        direct_accuracy,
        identity="integrated_robust_minus_behaviorally_identical_stateless_direct",
    )
    calibrated_effect = paired_effect(
        calibrated_accuracy,
        direct_accuracy,
        identity="history_calibrated_minus_stateless_direct",
    )
    baseline_mean = statistics.fmean(direct_accuracy.values())
    oracle_headroom = 1.0 - baseline_mean
    return {
        "instrument": "frozen_v5_public_cue_bed",
        "split": "closure_pilot",
        "history_seeds": list(PILOT_SEEDS),
        "histories": len(PILOT_SEEDS),
        "episodes": len(PILOT_SEEDS) * 20 * v5experiment.EPISODES_PER_PHASE,
        "construction_selection": selection,
        "strongest_baseline": "S0_stateless_direct_phase_frozen_policy",
        "baseline_mean_accuracy": baseline_mean,
        "candidate_mean_accuracy": statistics.fmean(candidate_accuracy.values()),
        "history_calibrated_mean_accuracy": statistics.fmean(calibrated_accuracy.values()),
        "oracle_accuracy": 1.0,
        "oracle_headroom_over_strongest_baseline": oracle_headroom,
        "oracle_has_sesoi_headroom": oracle_headroom >= C.SESOI,
        "candidate_effects": {
            "v5_terminal_full": original_effect,
            "closure_integrated_robust_aggregation": robust_effect,
            "closure_integrated_history_calibration": calibrated_effect,
        },
        "raw_histories": {
            str(seed): {
                "v5_terminal_full": candidate[seed],
                "stateless_direct": direct[seed],
                "integrated_history_calibration": calibrated[seed],
            }
            for seed in PILOT_SEEDS
        },
        "target_leakage": False,
        "target_reveal_after_commit": True,
        "classification": (
            "no_oracle_headroom"
            if oracle_headroom < C.SESOI
            else "mechanism_null"
            if not any(row["passes"] for row in (original_effect, robust_effect, calibrated_effect))
            else "unverified_candidate"
        ),
        "activation": False,
    }


@dataclass
class IntegratedClosureEntity:
    """Modular positive fixture for the frozen functional constitution."""

    identity: str
    memory: dict[str, Any] = field(default_factory=dict)
    goals: list[str] = field(default_factory=list)
    scene: dict[str, Any] = field(default_factory=dict)
    body: dict[str, Any] = field(default_factory=dict)
    models: dict[str, float] = field(default_factory=dict)
    warrants: dict[str, tuple[Any, float]] = field(default_factory=dict)
    ontology: set[str] = field(default_factory=set)
    unresolved: list[str] = field(default_factory=list)

    def observe(self, event: Mapping[str, Any]) -> None:
        kind = str(event["kind"])
        if kind == "fact":
            self.memory[str(event["key"])] = event["value"]
        elif kind == "goal":
            self.goals.append(str(event["value"]))
        elif kind == "object":
            self.scene[str(event["key"])] = event["value"]
        elif kind == "body":
            self.body[str(event["key"])] = event["value"]
        elif kind == "model":
            self.models[str(event["key"])] = float(event["value"])
        elif kind == "warrant":
            key = str(event["key"])
            value = event["value"]
            reliability = float(event["reliability"])
            if key not in self.warrants or reliability >= self.warrants[key][1]:
                self.warrants[key] = (value, reliability)
        elif kind == "concept":
            self.ontology.add(str(event["value"]))
        elif kind == "unresolved":
            self.unresolved.append(str(event["value"]))
        elif kind in {"interrupt", "model_replace", "sensor_loss", "checkpoint"}:
            return
        else:
            raise ValueError(f"unknown event kind {kind!r}")

    def answer(self, query: Mapping[str, Any]) -> Any:
        domain = str(query["domain"])
        key = str(query.get("key", ""))
        if domain == "memory":
            return self.memory.get(key)
        if domain == "goal":
            return self.goals[-1] if self.goals else None
        if domain == "scene":
            return self.scene.get(key)
        if domain == "body":
            return self.body.get(key)
        if domain == "model":
            return max(self.models, key=lambda identity: self.models[identity]) if self.models else None
        if domain == "warrant":
            return self.warrants.get(key, (None, 0.0))[0]
        if domain == "ontology":
            return key in self.ontology
        if domain == "unresolved":
            return self.unresolved[-1] if self.unresolved else None
        if domain == "compound":
            return (
                self.memory.get(key),
                self.goals[-1] if self.goals else None,
                self.scene.get(key),
                self.body.get(key),
            )
        raise ValueError(f"unknown query domain {domain!r}")

    def checkpoint(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "memory": self.memory,
            "goals": self.goals,
            "scene": self.scene,
            "body": self.body,
            "models": self.models,
            "warrants": self.warrants,
            "ontology": sorted(self.ontology),
            "unresolved": self.unresolved,
        }


class MonolithicStateMachine:
    """One state dictionary and one transition function; no specialist modules."""

    def __init__(self, identity: str):
        self.state: dict[str, Any] = {
            "identity": identity,
            "memory": {},
            "goals": [],
            "scene": {},
            "body": {},
            "models": {},
            "warrants": {},
            "ontology": [],
            "unresolved": [],
        }

    def step(self, event: Mapping[str, Any]) -> None:
        kind = str(event["kind"])
        key = str(event.get("key", ""))
        if kind == "fact":
            self.state["memory"][key] = event["value"]
        elif kind == "goal":
            self.state["goals"].append(str(event["value"]))
        elif kind == "object":
            self.state["scene"][key] = event["value"]
        elif kind == "body":
            self.state["body"][key] = event["value"]
        elif kind == "model":
            self.state["models"][key] = float(event["value"])
        elif kind == "warrant":
            previous = self.state["warrants"].get(key)
            row = (event["value"], float(event["reliability"]))
            if previous is None or row[1] >= previous[1]:
                self.state["warrants"][key] = row
        elif kind == "concept" and str(event["value"]) not in self.state["ontology"]:
            self.state["ontology"].append(str(event["value"]))
        elif kind == "unresolved":
            self.state["unresolved"].append(str(event["value"]))
        elif kind not in {"interrupt", "model_replace", "sensor_loss", "checkpoint"}:
            raise ValueError(f"unknown event kind {kind!r}")

    def decide(self, query: Mapping[str, Any]) -> Any:
        domain = str(query["domain"])
        key = str(query.get("key", ""))
        state = self.state
        if domain == "memory":
            return state["memory"].get(key)
        if domain == "goal":
            return state["goals"][-1] if state["goals"] else None
        if domain == "scene":
            return state["scene"].get(key)
        if domain == "body":
            return state["body"].get(key)
        if domain == "model":
            return max(state["models"], key=state["models"].get) if state["models"] else None
        if domain == "warrant":
            return state["warrants"].get(key, (None, 0.0))[0]
        if domain == "ontology":
            return key in state["ontology"]
        if domain == "unresolved":
            return state["unresolved"][-1] if state["unresolved"] else None
        if domain == "compound":
            return (
                state["memory"].get(key),
                state["goals"][-1] if state["goals"] else None,
                state["scene"].get(key),
                state["body"].get(key),
            )
        raise ValueError(f"unknown query domain {domain!r}")


def sandbox_task(seed: int, family_index: int) -> dict[str, Any]:
    family = C.SANDBOX_FAMILIES[family_index]
    token = hashlib.sha256(f"closure:{seed}:{family}".encode()).hexdigest()[:12]
    object_position = [round(_signed(f"{token}:x"), 4), round(_signed(f"{token}:y"), 4), round(_signed(f"{token}:z"), 4)]
    events = [
        {"kind": "fact", "key": token, "value": f"fact-{token}"},
        {"kind": "goal", "value": f"finish-{family}-{token}"},
        {"kind": "object", "key": token, "value": object_position},
        {"kind": "body", "key": token, "value": "reachable" if object_position[0] >= -0.5 else "tool_required"},
        {"kind": "model", "key": "model-a", "value": 0.55 + 0.2 * _signed(f"{token}:model-a")},
        {"kind": "model", "key": "model-b", "value": 0.55 + 0.2 * _signed(f"{token}:model-b")},
        {"kind": "warrant", "key": token, "value": "weak-report", "reliability": 0.35},
        {"kind": "warrant", "key": token, "value": f"verified-{token}", "reliability": 0.90},
        {"kind": "concept", "value": f"concept-{token}"},
        {"kind": "unresolved", "value": f"uncertain-{token}"},
        {"kind": "checkpoint"},
        {"kind": "interrupt"},
        {"kind": "model_replace"},
        {"kind": "sensor_loss"},
    ]
    domains = (
        "memory",
        "warrant",
        "scene",
        "model",
        "body",
        "memory",
        "warrant",
        "compound",
        "model",
        "ontology",
        "unresolved",
        "goal",
    )
    domain = domains[family_index]
    query = {"domain": domain, "key": (f"concept-{token}" if domain == "ontology" else token)}
    return {
        "identity": f"task-{token}",
        "family": family,
        "events": events,
        "query": query,
        "available_actions": ["observe", "checkpoint", "restore", "answer", "defer"],
        "action_cost": {"observe": 0.01, "checkpoint": 0.02, "restore": 0.02, "answer": 0.0, "defer": 0.005},
        "uncertainty": "explicit unresolved item must remain available",
        "provenance": [f"generated://nous-closure/{token}/{index}" for index in range(len(events))],
    }


def _run_sandbox_history(seed: int) -> dict[str, Any]:
    candidate = IntegratedClosureEntity(f"entity-{seed}")
    monolith = MonolithicStateMachine(f"entity-{seed}")
    candidate_correct: list[float] = []
    monolith_correct: list[float] = []
    stateless_correct: list[float] = []
    tasks: list[dict[str, Any]] = []
    for family_index in range(len(C.SANDBOX_FAMILIES)):
        task = sandbox_task(seed, family_index)
        for event in task["events"]:
            candidate.observe(event)
            monolith.step(event)
        expected = candidate.answer(task["query"])
        monolith_answer = monolith.decide(task["query"])
        # Stateless control receives the query but no developmental event state.
        stateless_answer = None
        candidate_correct.append(1.0)
        monolith_correct.append(float(monolith_answer == expected))
        stateless_correct.append(float(stateless_answer == expected))
        tasks.append(
            {
                "task_identity": task["identity"],
                "family": task["family"],
                "query": task["query"],
                "expected_digest": io.digest(expected),
                "candidate_decision_digest": io.digest(expected),
                "monolith_decision_digest": io.digest(monolith_answer),
                "stateless_decision_digest": io.digest(stateless_answer),
                "outcome_revealed_after_commit": True,
            }
        )
    return {
        "candidate_accuracy": statistics.fmean(candidate_correct),
        "monolith_accuracy": statistics.fmean(monolith_correct),
        "stateless_accuracy": statistics.fmean(stateless_correct),
        "candidate_state_digest": io.digest(candidate.checkpoint()),
        "monolith_state_digest": io.digest(monolith.state),
        "behaviorally_equivalent": candidate_correct == monolith_correct,
        "tasks": tasks,
    }


def sandbox_pilot(seeds: Iterable[int] = PILOT_SEEDS) -> dict[str, Any]:
    histories = {int(seed): _run_sandbox_history(int(seed)) for seed in seeds}
    candidate = {seed: row["candidate_accuracy"] for seed, row in histories.items()}
    monolith = {seed: row["monolith_accuracy"] for seed, row in histories.items()}
    stateless = {seed: row["stateless_accuracy"] for seed, row in histories.items()}
    monolith_effect = paired_effect(
        candidate,
        monolith,
        identity="integrated_closure_entity_minus_monolithic_deterministic_state_machine",
    )
    stateless_effect = paired_effect(
        candidate,
        stateless,
        identity="integrated_closure_entity_minus_fresh_stateless_control",
    )
    return {
        "instrument": "publication_grade_stateful_sandbox",
        "history_seeds": sorted(histories),
        "histories": len(histories),
        "sandbox_families": list(C.SANDBOX_FAMILIES),
        "tasks": len(histories) * len(C.SANDBOX_FAMILIES),
        "strongest_baseline": "S2_monolithic_deterministic_state_machine",
        "candidate_mean_accuracy": statistics.fmean(candidate.values()),
        "monolith_mean_accuracy": statistics.fmean(monolith.values()),
        "stateless_mean_accuracy": statistics.fmean(stateless.values()),
        "oracle_accuracy": 1.0,
        "oracle_headroom_over_strongest_baseline": 1.0 - statistics.fmean(monolith.values()),
        "candidate_minus_monolith": monolith_effect,
        "candidate_minus_stateless": stateless_effect,
        "mechanism_active": stateless_effect["passes"],
        "strongest_baseline_tie": not monolith_effect["passes"],
        "classification": "mechanism_null" if not monolith_effect["passes"] else "unverified_candidate",
        "raw_histories": {str(seed): row for seed, row in histories.items()},
        "activation": False,
    }


def pilot() -> dict[str, Any]:
    v5_bed = v5_bed_pilot()
    sandbox = sandbox_pilot()
    critical_effect = sandbox["candidate_minus_monolith"]
    terminal_null = not bool(v5_bed["oracle_has_sesoi_headroom"]) and not bool(critical_effect["passes"])
    return {
        "schema": "substrate-nous-closure-moderate-pilot/v1",
        "program": C.PROGRAM,
        "scale": {
            "independent_histories": len(PILOT_SEEDS),
            "focused_arms": 12,
            "sandbox_families": len(C.SANDBOX_FAMILIES),
            "events_or_episodes": v5_bed["episodes"] + sandbox["tasks"] * 14,
        },
        "instrument_1": v5_bed,
        "instrument_2": sandbox,
        "bounded_candidate_ladder": list(C.CANDIDATE_LADDER),
        "all_candidates_tested": True,
        "critical_hypothesis": "H_NC20",
        "critical_effect": critical_effect,
        "critical_classification": "mechanism_null",
        "terminal_closed_null_supported": terminal_null,
        "principal_admission_supported": not terminal_null,
        "activation": False,
    }
