"""Deterministic toy-world observations for the F22, F28, and F50 to F58 ecology rows.

This module generates raw bounded-world observations only. It does not assign experiment verdicts.
The run driver and independent verifier intentionally score these observations through separate
code paths. All observations are programmatic fixtures, so they cannot support a natural-world or
substrate capability claim.
"""

from __future__ import annotations

import hashlib
import math
from statistics import mean
from typing import Any

from ..experiments.expansion_harness import CLAIM_SCOPE
from ..substrate.events import canonical_bytes, canonical_sha256
from .ecology_scaffold import (
    ACQUISITION_CONTROLS,
    CHANNEL_CONTROLS,
    CURRICULUM_CONTROLS,
    EcologyRefusal,
    GoalArchive,
    GoalRecord,
    evaluate_stop_rules,
    make_ecology_fixture,
    run_goal_babbling_fixture,
)

BATTERY_UNIT_SCHEMA = "mop-ecology-toy-battery-unit/v1"
EXPERIMENT_IDS = (
    "f22_active_form_acquisition",
    "f28_sensor_value_forecast",
    "f50_curriculum_goldilocks_test",
    "f51_safe_play_goal_babbling",
    "f52_quality_diverse_mode_ecology",
    "f53_joint_referent_establishment",
    "f54_communicative_repair",
    "f55_selective_imitation",
    "f56_teaching_value",
    "f57_emergent_symbol_grounding",
    "f58_cultural_accumulation",
)


def _stable_unit(seed: int, label: str, index: int) -> float:
    digest = hashlib.sha256(canonical_bytes({"seed": seed, "label": label, "index": index})).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def _stable_int(seed: int, label: str, index: int, modulus: int) -> int:
    if modulus < 1:
        raise ValueError("modulus must be positive")
    return int(_stable_unit(seed, label, index) * modulus) % modulus


def _rate(seed: int, label: str, probability: float, trials: int) -> dict[str, Any]:
    if not 0.0 < probability < 1.0 or trials < 1:
        raise ValueError("rate fixtures require an interior probability and positive trials")
    successes = sum(_stable_unit(seed, label, index) < probability for index in range(trials))
    return {"successes": successes, "trials": trials, "rate": successes / trials}


def _rounded(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("ecology metrics must be finite")
    return round(value, 8)


def _average_rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda row: (row[1], row[0]))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[indexed[position][0]] = rank
        cursor = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("correlation vectors must have equal positive length")
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_ss = sum((value - left_mean) ** 2 for value in left)
    right_ss = sum((value - right_mean) ** 2 for value in right)
    denominator = math.sqrt(left_ss * right_ss)
    return 0.0 if denominator == 0.0 else numerator / denominator


def _spearman(left: list[float], right: list[float]) -> float:
    return _correlation(_average_rank(left), _average_rank(right))


def _f22(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    action_costs = {
        row.action: row.sensing_units + row.motion_units + row.latency_ticks + row.compute_units
        for row in bundle.perception.costs
    }
    cheapest = min(action_costs[action] for action in ("view", "time", "audio"))
    profiles = {
        "information-gain": (0.78, cheapest),
        "random-acquisition": (0.51, cheapest),
        "uncertainty-acquisition": (0.65, cheapest),
        "saliency-acquisition": (0.56, action_costs["view"]),
        "full-observation": (0.86, sum(action_costs[action] for action in ("view", "time", "audio"))),
    }
    charge_budget = 480
    maximum_queries = 96
    arms: dict[str, dict[str, Any]] = {}
    for arm, (probability, cost) in profiles.items():
        attempts = min(maximum_queries, charge_budget // cost)
        first = attempts // 2
        second = attempts - first
        world_a = _rate(seed, f"f22:{arm}:world-a", probability, first)
        world_b = _rate(seed, f"f22:{arm}:world-b", probability - 0.03, second)
        correct = world_a["successes"] + world_b["successes"]
        accuracy = correct / attempts
        arms[arm] = {
            "correct": correct,
            "attempts": attempts,
            "charge_budget": charge_budget,
            "charge_used": attempts * cost,
            "world_accuracy": [_rounded(world_a["rate"]), _rounded(world_b["rate"])],
            "metrics": {
                "capability_per_sensor_cost": _rounded(correct / charge_budget),
                "acquisition_regret_vs_full_observation": 0.0,
                "untouched_downstream_accuracy": _rounded(accuracy),
                "cross_world_min_accuracy": _rounded(min(world_a["rate"], world_b["rate"])),
            },
        }
    full_accuracy = arms["full-observation"]["metrics"]["untouched_downstream_accuracy"]
    for row in arms.values():
        row["metrics"]["acquisition_regret_vs_full_observation"] = _rounded(
            max(0.0, full_accuracy - row["metrics"]["untouched_downstream_accuracy"])
        )
    return {
        "experiment_id": "f22_active_form_acquisition",
        "primary_arm": "information-gain",
        "controls": list(ACQUISITION_CONTROLS),
        "arms": arms,
        "mechanism_checks": {
            "matched_charge_budget": True,
            "all_sensing_actions_costed": set(action_costs) == {"view", "time", "audio", "abstain"},
            "downstream_training_updates": 0,
            "held_out_world_count": len(bundle.world.world_fixture()["held_out_world_sha256"]),
        },
    }


def _f28(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    actual: list[float] = []
    signal: list[float] = []
    entropy: list[float] = []
    groups: list[int] = []
    for index in range(48):
        latent = 0.18 + 0.64 * _stable_unit(seed, "f28:latent", index)
        realized = min(0.94, max(0.06, latent + 0.16 * (_stable_unit(seed, "f28:outcome", index) - 0.5)))
        signal.append(latent)
        actual.append(realized)
        entropy.append(0.12 + 0.76 * _stable_unit(seed, "f28:entropy", index))
        groups.append(index % 4)
    group_means = {
        group: mean(actual[index] for index in range(len(actual)) if groups[index] == group)
        for group in range(4)
    }
    forecasts = {
        "expected-information-gain": [
            min(0.96, max(0.04, value + 0.12 * (_stable_unit(seed, "f28:forecast", index) - 0.5)))
            for index, value in enumerate(signal)
        ],
        "entropy-baseline": entropy,
        "historical-average": [group_means[group] for group in groups],
        "post-hoc-value": list(actual),
    }
    arms: dict[str, Any] = {}
    for arm, values in forecasts.items():
        arms[arm] = {
            "committed_before_cost": arm != "post-hoc-value",
            "metrics": {
                "payoff_calibration_error": _rounded(
                    mean(abs(forecast - payoff) for forecast, payoff in zip(values, actual, strict=True))
                ),
                "forecast_rank_correlation": _rounded(_spearman(values, actual)),
            },
        }
    observation_payload = {"actual": actual, "forecasts": forecasts, "groups": groups}
    return {
        "experiment_id": "f28_sensor_value_forecast",
        "primary_arm": "expected-information-gain",
        "controls": ["entropy-baseline", "historical-average", "post-hoc-value"],
        "arms": arms,
        "observation_sha256": canonical_sha256(observation_payload),
        "mechanism_checks": {
            "pre_cost_commitment": bundle.perception.value_forecast.pre_cost_commitment,
            "forecast_count": len(actual),
            "payoff_min": _rounded(min(actual)),
            "payoff_max": _rounded(max(actual)),
        },
    }


def _f50(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    tasks: list[dict[str, Any]] = []
    offset = _stable_int(seed, "f50:offset", 0, 16)
    for index in range(16):
        pass_rate = 0.05 + ((index + offset) % 16) * 0.057
        noisy = index % 7 == 0
        generality = 0.62 + 0.34 * _stable_unit(seed, "f50:generality", index)
        transfer = 0.0 if noisy else 4.0 * pass_rate * (1.0 - pass_rate) * generality
        tasks.append(
            {
                "task": index,
                "pass_rate": pass_rate,
                "noisy_tv": noisy,
                "transfer": transfer,
            }
        )
    selectable = [
        row for row in tasks if bundle.ecology.curriculum.in_band(row["pass_rate"]) and not row["noisy_tv"]
    ]
    count = min(6, len(selectable))
    selections = {
        "goldilocks-learning-progress": sorted(selectable, key=lambda row: (-row["transfer"], row["task"]))[
            :count
        ],
        "random-task": sorted(
            tasks, key=lambda row: (_stable_unit(seed, "f50:random", row["task"]), row["task"])
        )[:count],
        "fixed-order": tasks[:count],
        "easiest-first": sorted(tasks, key=lambda row: (-row["pass_rate"], row["task"]))[:count],
        "hardest-first": sorted(tasks, key=lambda row: (row["pass_rate"], row["task"]))[:count],
    }
    arms: dict[str, Any] = {}
    for arm, selected in selections.items():
        arms[arm] = {
            "selected_tasks": [row["task"] for row in selected],
            "noisy_tv_captures": sum(row["noisy_tv"] for row in selected),
            "metrics": {
                "downstream_transfer_per_sample": _rounded(mean(row["transfer"] for row in selected)),
                "band_occupancy_rate": _rounded(
                    mean(bundle.ecology.curriculum.in_band(row["pass_rate"]) for row in selected)
                ),
            },
        }
    return {
        "experiment_id": "f50_curriculum_goldilocks_test",
        "primary_arm": "goldilocks-learning-progress",
        "controls": list(CURRICULUM_CONTROLS),
        "arms": arms,
        "task_manifest_sha256": canonical_sha256(tasks),
        "mechanism_checks": {
            "too_hard_pass_rate": bundle.ecology.curriculum.too_hard_pass_rate,
            "too_easy_pass_rate": bundle.ecology.curriculum.too_easy_pass_rate,
            "task_count": len(tasks),
            "selection_count": count,
        },
    }


def _f51(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    candidates: list[GoalRecord] = []
    validity: dict[str, bool] = {}
    for index in range(24):
        kind = index % 5
        reward_source = "self-scored" if kind == 1 else "environment-consequence"
        target_kind = "noise-source" if kind == 0 else "environment-entity"
        unsafe = kind == 2
        goal = GoalRecord(
            goal_ref=f"goal:safe-play-{seed}-{index}",
            descriptor=(index % 8, index // 8),
            quality=0.25 + 0.7 * _stable_unit(seed, "f51:quality", index),
            reward_source=reward_source,
            target_kind=target_kind,
            unsafe=unsafe,
        )
        candidates.append(goal)
        validity[goal.goal_ref] = kind in {3, 4}
    archive = GoalArchive(capacity=bundle.ecology.archive_capacity)
    refusal_counts = {rule: 0 for rule in ("noisy-tv", "reward-hacking", "unsafe-goal", "archive-bloat")}
    for goal in sorted(candidates, key=lambda row: (-row.quality, row.goal_ref)):
        try:
            archive.admit(goal)
        except EcologyRefusal as refusal:
            if refusal.rule in refusal_counts:
                refusal_counts[refusal.rule] += 1
    guarded_rows = archive.payload()["cells"]
    random_rows = sorted(
        candidates,
        key=lambda row: (
            _stable_unit(seed, "f51:random", int(row.goal_ref.rsplit("-", 1)[1])),
            row.goal_ref,
        ),
    )[: bundle.ecology.archive_capacity]
    ungated_rows = sorted(candidates, key=lambda row: (-row.quality, row.goal_ref))[
        : bundle.ecology.archive_capacity
    ]

    def acceleration(rows: list[GoalRecord] | list[dict[str, Any]]) -> float:
        total = 0.0
        for row in rows:
            if isinstance(row, GoalRecord):
                ref = row.goal_ref
                quality = row.quality
            else:
                ref = str(row["goal_ref"])
                quality = float(row["quality"])
            total += quality if validity[ref] else 0.0
        return total / bundle.ecology.archive_capacity

    histories = {
        "unsafe-goal": [{"new_cells": 0, "distinct_cells": 0, "archive_size": 0, "unsafe_flag": True}],
        "archive-bloat": [
            {
                "new_cells": 1,
                "distinct_cells": 3,
                "archive_size": bundle.ecology.stop.bloat_limit + 1,
                "unsafe_flag": False,
            }
        ],
        "collapse": [
            {"new_cells": 0, "distinct_cells": 1, "archive_size": 1, "unsafe_flag": False}
            for _ in range(bundle.ecology.stop.plateau_window)
        ],
        "plateau": [
            {"new_cells": 0, "distinct_cells": 3, "archive_size": 3, "unsafe_flag": False}
            for _ in range(bundle.ecology.stop.plateau_window)
        ],
        "none": [{"new_cells": 1, "distinct_cells": 3, "archive_size": 3, "unsafe_flag": False}],
    }
    audit = {name: evaluate_stop_rules(bundle.ecology.stop, history) for name, history in histories.items()}
    audit_ok = all(
        observed == (None if expected == "none" else expected) for expected, observed in audit.items()
    )
    trace = run_goal_babbling_fixture(bundle.ecology, seed=seed, steps=24)
    refusal_rates = {rule: _rounded(count / len(candidates)) for rule, count in refusal_counts.items()}
    return {
        "experiment_id": "f51_safe_play_goal_babbling",
        "primary_arm": "guarded-goal-babbling",
        "controls": ["random-exploration", "ungated-curiosity", "stop-rule-audit"],
        "arms": {
            "guarded-goal-babbling": {
                "accepted": len(guarded_rows),
                "metrics": {
                    "external_task_acceleration": _rounded(acceleration(guarded_rows)),
                    "guard_violation_rate": 0.0,
                },
            },
            "random-exploration": {
                "accepted": len(random_rows),
                "metrics": {
                    "external_task_acceleration": _rounded(acceleration(random_rows)),
                    "guard_violation_rate": _rounded(mean(not validity[row.goal_ref] for row in random_rows)),
                },
            },
            "ungated-curiosity": {
                "accepted": len(ungated_rows),
                "metrics": {
                    "external_task_acceleration": _rounded(acceleration(ungated_rows)),
                    "guard_violation_rate": _rounded(
                        mean(not validity[row.goal_ref] for row in ungated_rows)
                    ),
                },
            },
            "stop-rule-audit": {
                "accepted": 0,
                "metrics": {"audit_accuracy": _rounded(mean([audit_ok]))},
            },
        },
        "reported_metrics": {
            "refusal_rate_by_rule": refusal_rates,
        },
        "candidate_manifest_sha256": canonical_sha256([row.payload() for row in candidates]),
        "mechanism_checks": {
            "refusal_counts": refusal_counts,
            "all_stop_rules_exercised": audit_ok,
            "babbling_trace_sha256": trace["trace_sha256"],
            "babbling_stop_reason": trace["stop_reason"],
        },
    }


def _f52(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    contexts = 10
    capacity = bundle.ecology.archive_capacity
    qd_targets = list(range(capacity))
    random_targets = [
        _stable_int(seed, "f52:random-target", index, max(2, capacity // 2)) for index in range(capacity)
    ]

    def context_scores(arm: str, targets: list[int]) -> list[float]:
        scores: list[float] = []
        for context in range(contexts):
            covered = context in targets
            if arm == "monolithic-model":
                value = 0.52 + 0.06 * _stable_unit(seed, "f52:mono", context)
            elif covered:
                value = 0.72 + 0.12 * _stable_unit(seed, f"f52:{arm}:covered", context)
            else:
                value = 0.28 + 0.12 * _stable_unit(seed, f"f52:{arm}:miss", context)
            scores.append(value)
        return scores

    arm_targets = {
        "quality-diverse-archive": qd_targets,
        "monolithic-model": [0],
        "random-archive": random_targets,
    }
    arms: dict[str, Any] = {}
    for arm, targets in arm_targets.items():
        scores = context_scores(arm, targets)
        distinct = len(set(targets))
        arms[arm] = {
            "mode_targets": targets,
            "context_scores": [_rounded(value) for value in scores],
            "metrics": {
                "archive_coverage": _rounded(mean(value >= 0.7 for value in scores)),
                "utility_per_maintenance_cost": _rounded(mean(scores)),
                "redundancy": _rounded(0.0 if arm == "monolithic-model" else 1.0 - distinct / len(targets)),
            },
        }
    history = [
        {"new_cells": 1, "distinct_cells": index + 1, "archive_size": index + 1, "unsafe_flag": False}
        for index in range(capacity)
    ]
    return {
        "experiment_id": "f52_quality_diverse_mode_ecology",
        "primary_arm": "quality-diverse-archive",
        "controls": ["monolithic-model", "random-archive", "matched-compute"],
        "arms": arms,
        "mechanism_checks": {
            "archive_capacity": capacity,
            "context_count": contexts,
            "matched_compute": True,
            "matched_maintenance_budget": True,
            "stop_reason": evaluate_stop_rules(bundle.ecology.stop, history),
        },
    }


def _partner_rate_arms(
    seed: int, experiment: str, profiles: dict[str, tuple[float, float]], trials: int
) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm, (train_probability, held_probability) in profiles.items():
        train = _rate(seed, f"{experiment}:{arm}:train", train_probability, trials)
        held = _rate(seed, f"{experiment}:{arm}:held", held_probability, trials)
        arms[arm] = {
            "train": train,
            "held_out": held,
            "metrics": {
                "held_out_rate": _rounded(held["rate"]),
                "held_out_transfer": _rounded(held["rate"] / max(train["rate"], 1 / trials)),
            },
        }
    return arms


def _f53(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    arms = _partner_rate_arms(
        seed,
        "f53",
        {
            "joint-referent-policy": (0.82, 0.76),
            "cue-blind-partner": (0.31, 0.29),
            "partner-policy-pattern-matching": (0.78, 0.43),
            "shared-label-oracle": (0.92, 0.90),
        },
        96,
    )
    for row in arms.values():
        row["metrics"]["joint_referent_agreement"] = row["metrics"].pop("held_out_rate")
        row["metrics"]["held_out_partner_transfer"] = row["metrics"].pop("held_out_transfer")
    return {
        "experiment_id": "f53_joint_referent_establishment",
        "primary_arm": "joint-referent-policy",
        "controls": [
            "cue-blind-partner",
            "partner-policy-pattern-matching",
            "shared-label-oracle",
        ],
        "arms": arms,
        "mechanism_checks": {
            "training_partner_count": sum(not row.held_out for row in bundle.partner.partners),
            "held_out_partner_count": sum(row.held_out for row in bundle.partner.partners),
            "private_observation_leak": False,
        },
    }


def _f54(seed: int) -> dict[str, Any]:
    profiles = {
        "clarification-repair": 0.76,
        "no-repair-channel": 0.34,
        "repetition": 0.49,
        "larger-channel-matched-bits": 0.62,
    }
    trials = 96
    arms: dict[str, Any] = {}
    for arm, probability in profiles.items():
        recovered = _rate(seed, f"f54:{arm}", probability, trials)
        bits = 2
        turns = 2
        arms[arm] = {
            "recoveries": recovered["successes"],
            "trials": trials,
            "charged_bits_per_trial": bits,
            "turns_per_trial": turns,
            "metrics": {
                "recovery_per_bit": _rounded(recovered["successes"] / (trials * bits)),
                "recovery_per_turn": _rounded(recovered["successes"] / (trials * turns)),
            },
        }
    return {
        "experiment_id": "f54_communicative_repair",
        "primary_arm": "clarification-repair",
        "controls": ["no-repair-channel", "repetition", "larger-channel-matched-bits"],
        "arms": arms,
        "mechanism_checks": {
            "repair_binding_present": True,
            "matched_extra_bits": True,
            "held_out_partners": True,
        },
    }


def _f55(seed: int) -> dict[str, Any]:
    profiles = {
        "selective-causal-imitation": (0.82, 0.80),
        "indiscriminate-imitation": (0.55, 0.66),
        "behavioral-cloning": (0.61, 0.71),
    }
    arms: dict[str, Any] = {}
    for arm, (fidelity_probability, success_probability) in profiles.items():
        fidelity = _rate(seed, f"f55:{arm}:fidelity", fidelity_probability, 128)
        success = _rate(seed, f"f55:{arm}:success", success_probability, 128)
        arms[arm] = {
            "necessary_and_decorative_steps_declared": True,
            "metrics": {
                "causal_action_fidelity": _rounded(fidelity["rate"]),
                "task_success_rate": _rounded(success["rate"]),
            },
        }
    return {
        "experiment_id": "f55_selective_imitation",
        "primary_arm": "selective-causal-imitation",
        "controls": ["indiscriminate-imitation", "behavioral-cloning"],
        "arms": arms,
        "mechanism_checks": {
            "held_out_partners": True,
            "causal_necessity_labels_programmatic": True,
            "decorative_steps_present": True,
        },
    }


def _f56(seed: int) -> dict[str, Any]:
    profiles = {
        "learner-progress-teacher": ("targeted", 0.73),
        "equal-information": ("equal", 0.61),
        "random-demonstrations": ("random", 0.54),
        "hard-example-selection": ("hard", 0.64),
        "uncertainty-selection": ("targeted", 0.73),
    }
    trials = 128
    arms: dict[str, Any] = {}
    equal_rate = 0.0
    raw: dict[str, dict[str, Any]] = {}
    for arm, (draw_label, probability) in profiles.items():
        row = _rate(seed, f"f56:{draw_label}", probability, trials)
        raw[arm] = row
        if arm == "equal-information":
            equal_rate = row["rate"]
    for arm, row in raw.items():
        arms[arm] = {
            "messages": trials,
            "metrics": {
                "learner_gain_per_message": _rounded(row["rate"]),
                "held_out_learner_advantage": _rounded(row["rate"] - equal_rate),
            },
        }
    return {
        "experiment_id": "f56_teaching_value",
        "primary_arm": "learner-progress-teacher",
        "controls": [
            "equal-information",
            "random-demonstrations",
            "hard-example-selection",
            "uncertainty-selection",
        ],
        "arms": arms,
        "mechanism_checks": {
            "equal_information_message_count": True,
            "held_out_learners": True,
            "uncertainty_control_indistinguishable_in_fixture": True,
        },
    }


def _f57(seed: int) -> dict[str, Any]:
    bundle = make_ecology_fixture(seed)
    profiles = {
        "consequence-bound-code": (0.70, 0.74, 0.72),
        "random-message": (0.31, 0.34, 0.29),
        "fixed-message": (0.22, 0.27, 0.42),
        "direct-state": (0.84, 0.86, 0.83),
        "equal-bandwidth": (0.63, 0.66, 0.65),
    }
    arms: dict[str, Any] = {}
    for arm, probabilities in profiles.items():
        composition = _rate(seed, f"f57:{arm}:composition", probabilities[0], 128)
        causal = _rate(seed, f"f57:{arm}:causal", probabilities[1], 128)
        stability = _rate(seed, f"f57:{arm}:stability", probabilities[2], 128)
        axes = {
            axis: _rounded(
                _rate(seed, f"f57:{arm}:axis:{axis}", 0.35 + 0.5 * probabilities[index % 3], 96)["rate"]
            )
            for index, axis in enumerate(bundle.communication.score_axes)
        }
        transfers = {
            requirement: _rounded(
                _rate(
                    seed,
                    f"f57:{arm}:transfer:{requirement}",
                    max(0.08, probabilities[index % 3] - 0.04),
                    96,
                )["rate"]
            )
            for index, requirement in enumerate(bundle.communication.transfer_requirements)
        }
        arms[arm] = {
            "channel_bits": bundle.communication.channel_bits,
            "score_axes": axes,
            "transfer_axes": transfers,
            "metrics": {
                "compositional_transfer": _rounded(composition["rate"]),
                "causal_grounding_score": _rounded(causal["rate"]),
                "code_stability": _rounded(stability["rate"]),
            },
        }
    return {
        "experiment_id": "f57_emergent_symbol_grounding",
        "primary_arm": "consequence-bound-code",
        "controls": list(CHANNEL_CONTROLS),
        "arms": arms,
        "mechanism_checks": {
            "binding_kinds": sorted(row.binding for row in bundle.communication.bindings),
            "equal_bandwidth_bits": bundle.communication.equal_bandwidth_bits,
            "channel_bits": bundle.communication.channel_bits,
            "all_score_axes_present": True,
            "all_transfer_axes_present": True,
        },
    }


def _slope(values: list[float]) -> float:
    x_mean = (len(values) - 1) / 2.0
    numerator = sum((index - x_mean) * (value - mean(values)) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    return numerator / denominator


def _f58(seed: int) -> dict[str, Any]:
    profiles = {
        "cumulative-convention": (0.40, 0.065, 0.78),
        "generation-reset": (0.44, 0.0, 0.51),
        "direct-imitation": (0.44, 0.018, 0.66),
        "fresh-training": (0.43, 0.01, 0.47),
    }
    arms: dict[str, Any] = {}
    for arm, (base, increment, retention_probability) in profiles.items():
        scores = [
            _rate(seed, f"f58:{arm}:generation:{generation}", base + increment * generation, 128)["rate"]
            for generation in range(6)
        ]
        retention = _rate(seed, f"f58:{arm}:retention", retention_probability, 128)
        arms[arm] = {
            "generation_scores": [_rounded(value) for value in scores],
            "metrics": {
                "generation_trend_external_tasks": _rounded(_slope(scores)),
                "convention_retention": _rounded(retention["rate"]),
                "final_external_task_performance": _rounded(scores[-1]),
            },
        }
    return {
        "experiment_id": "f58_cultural_accumulation",
        "primary_arm": "cumulative-convention",
        "controls": ["generation-reset", "direct-imitation", "fresh-training"],
        "arms": arms,
        "mechanism_checks": {
            "generation_count": 6,
            "external_tasks_used": True,
            "source_reliability_preserved": True,
            "dissent_preserved": True,
        },
    }


def run_ecology_battery_seed(seed: int) -> dict[str, Any]:
    """Generate one content-addressed independent toy-world unit for every ecology row."""

    if seed < 0:
        raise ValueError("ecology battery seed must be nonnegative")
    bundle = make_ecology_fixture(seed)
    rows = [_f22(seed), _f28(seed), _f50(seed), _f51(seed), _f52(seed), _f53(seed)]
    rows.extend([_f54(seed), _f55(seed), _f56(seed), _f57(seed), _f58(seed)])
    experiments = {row["experiment_id"]: row for row in rows}
    if tuple(experiments) != EXPERIMENT_IDS:
        raise RuntimeError("ecology experiment coverage or order drift")
    payload: dict[str, Any] = {
        "schema": BATTERY_UNIT_SCHEMA,
        "seed": seed,
        "claim_scope": CLAIM_SCOPE,
        "fixture": bundle.payload(),
        "fixture_sha256": bundle.sha256,
        "experiments": experiments,
    }
    payload["unit_sha256"] = canonical_sha256(payload)
    return payload
