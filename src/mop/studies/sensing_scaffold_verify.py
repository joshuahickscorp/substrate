"""Independent fresh-seed verifier for the f21, f26, and f27 toy executions.

This module intentionally does not import the run evaluator. It independently regenerates the
declared data families, refits every arm, recomputes the load-bearing metrics, and applies stronger
context, source-scale, and intervention-label attacks before closing a favorable toy pattern.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
from random import Random
from statistics import median
from typing import Any

import numpy as np
import yaml

from ..config import REPO_ROOT
from ..substrate.events import canonical_sha256
from ..substrate.sensing_scaffold import CLAIM_SCOPE, make_sensing_fixture

RUN_SCHEMA = "mop-sensing-scaffold-run/v1"
VERIFY_SCHEMA = "mop-sensing-scaffold-verifier/v1"
CONFIG_SCHEMA = "mop-sensing-scaffold-run-config/v1"
DEFAULT_RUN = REPO_ROOT / "proof" / "SENSING_SCAFFOLD_RUN.json"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "sensing_scaffold_runs.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "SENSING_SCAFFOLD_VERIFICATION.json"
EXPERIMENT_IDS = (
    "f21_asynchronous_temporal_binding",
    "f26_cross_form_contradiction_triangulation",
    "f27_causal_crossmodal_binding",
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(raw, encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _seed(seed: int, label: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:8], "big")


def _read_inputs(run_path: Path, config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run = json.loads(run_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if run.get("schema") != RUN_SCHEMA or run.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("sensing verifier received the wrong run receipt")
    run_without_digest = {key: value for key, value in run.items() if key != "payload_sha256"}
    if run.get("payload_sha256") != canonical_sha256(run_without_digest):
        raise ValueError("sensing run receipt digest mismatch")
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("sensing verifier config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("sensing verifier claim scope drift")
    if not isinstance(config.get("null_hypothesis"), str) or not config["null_hypothesis"].strip():
        raise ValueError("sensing verifier requires a nonblank aggregate null hypothesis")
    if run.get("config", {}).get("sha256") != _sha256_file(config_path):
        raise ValueError("sensing run does not bind the verifier config")
    registry = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in registry["experiments"]}
    per_experiment = {
        experiment_id: str(index[experiment_id]["null_hypothesis"]) for experiment_id in EXPERIMENT_IDS
    }
    expected_null_contract = {
        "aggregate": config["null_hypothesis"],
        "per_experiment": per_experiment,
        "per_experiment_sha256": canonical_sha256(per_experiment),
    }
    if run.get("null_contract") != expected_null_contract:
        raise ValueError("sensing run null contract does not match config and live registry")
    for experiment_id in EXPERIMENT_IDS:
        binding = run.get("registry_bindings", {}).get(experiment_id, {})
        if binding.get("null_hypothesis") != per_experiment[experiment_id] or binding.get(
            "row_sha256"
        ) != canonical_sha256(index[experiment_id]):
            raise ValueError(f"sensing registry binding drift for {experiment_id}")
    return run, config


def _f1(labels: list[int], predictions: list[int]) -> float:
    true_positive = sum(
        label == prediction == 1 for label, prediction in zip(labels, predictions, strict=True)
    )
    false_positive = sum(
        label == 0 and prediction == 1 for label, prediction in zip(labels, predictions, strict=True)
    )
    false_negative = sum(
        label == 1 and prediction == 0 for label, prediction in zip(labels, predictions, strict=True)
    )
    return 2 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)


def _choose_window(rows: list[tuple[int, int, float]]) -> float:
    winner_score = -1.0
    winner_window = math.inf
    for window in sorted({row[2] for row in rows}):
        score = _f1([row[1] for row in rows], [int(row[2] <= window) for row in rows])
        if score > winner_score or (score == winner_score and window < winner_window):
            winner_score = score
            winner_window = window
    return winner_window


def _f21_rows(seed: int, count: int, cfg: dict[str, Any], split: str) -> list[tuple[int, int, float]]:
    rng = Random(_seed(seed, f"f21:{split}"))
    rows = []
    for _ in range(count):
        context = rng.randrange(int(cfg["contexts"]))
        label = int(rng.random() < float(cfg["positive_rate"]))
        mean_gap = 2.4 + 2.3 * context + (0.0 if label else 3.4)
        gap = max(0.0, rng.gauss(mean_gap, float(cfg["gap_noise_sd"])))
        if rng.random() < float(cfg["label_noise_rate"]):
            label ^= 1
        rows.append((context, label, gap))
    return rows


def _verify_f21(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train = _f21_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test = _f21_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    context_count = int(cfg["contexts"])
    windows = {
        context: _choose_window([row for row in train if row[0] == context])
        for context in range(context_count)
    }
    fixed = _choose_window(train)
    labels = [row[1] for row in test]
    learned_f1 = _f1(labels, [int(gap <= windows[context]) for context, _, gap in test])
    fixed_f1 = _f1(labels, [int(gap <= fixed) for _, _, gap in test])
    shuffled_context_f1 = _f1(
        labels,
        [int(gap <= windows[(context + 1) % context_count]) for context, _, gap in test],
    )
    control_rng = Random(_seed(seed, "f21:controls"))
    shuffled = []
    wrong = []
    for _ in range(len(test) // 2):
        context = control_rng.randrange(context_count)
        shuffled.append((context, max(0.0, control_rng.gauss(8.8 + 2.3 * context, 2.0))))
        wrong.append((context, max(0.0, control_rng.gauss(11.2 + 2.3 * context, 2.4))))

    def rejection(rows: list[tuple[int, float]]) -> float:
        return sum(gap > windows[context] for context, gap in rows) / len(rows)

    lo, hi = (float(value) for value in cfg["calibration_band"])
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).temporal.sha256,
        "binding_f1_across_delay_and_jitter": round(learned_f1, 8),
        "fixed_window_binding_f1": round(fixed_f1, 8),
        "delta_vs_fixed_window": round(learned_f1 - fixed_f1, 8),
        "shuffled_time_rejection_rate": round(rejection(shuffled), 8),
        "wrong_time_rejection_rate": round(rejection(wrong), 8),
        "context_shuffled_binding_f1": round(shuffled_context_f1, 8),
        "off_floor_and_ceiling": lo <= learned_f1 <= hi,
        "strongest_shell_attack_passed": shuffled_context_f1 < learned_f1,
    }


def _f26_rows(
    seed: int, count: int, cfg: dict[str, Any], split: str
) -> list[tuple[float, int, tuple[float, float, float]]]:
    rng = Random(_seed(seed, f"f26:{split}"))
    source_noise = [float(value) for value in cfg["source_noise_sd"]]
    if len(source_noise) != 3:
        raise ValueError("f26 verifier requires exactly three source noise scales")
    rows: list[tuple[float, int, tuple[float, float, float]]] = []
    for _ in range(count):
        truth = rng.gauss(0.0, 2.2)
        dissenter = rng.randrange(3)
        reports = [truth + rng.gauss(0.0, value) for value in source_noise]
        sign = -1.0 if rng.random() < 0.5 else 1.0
        reports[dissenter] += sign * (
            float(cfg["corruption_magnitude"]) + abs(rng.gauss(0.0, float(cfg["corruption_sd"])))
        )
        rows.append((truth, dissenter, (reports[0], reports[1], reports[2])))
    return rows


def _rank(scores: tuple[float, float, float], target: int) -> float:
    comparisons = []
    for index in range(3):
        if index != target:
            comparisons.append(
                float(scores[target] > scores[index]) + 0.5 * float(scores[target] == scores[index])
            )
    return sum(comparisons) / len(comparisons)


def _verify_f26(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train = _f26_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test = _f26_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    residual_sets: list[list[float]] = [[], [], []]
    for _, target, reports in train:
        midpoint = median(reports)
        for source in range(3):
            if source != target:
                residual_sets[source].append(abs(reports[source] - midpoint))
    scales = [max(0.05, sum(values) / len(values)) for values in residual_sets]
    learned = []
    raw = []
    means = []
    majorities = []
    shuffled = []
    fused_errors = []
    for truth, target, reports in test:
        midpoint = median(reports)
        residual_values = tuple(abs(value - midpoint) for value in reports)
        residual = (residual_values[0], residual_values[1], residual_values[2])
        learned_values = tuple(residual[index] / scales[index] for index in range(3))
        learned_scores = (learned_values[0], learned_values[1], learned_values[2])
        mean_value = sum(reports) / 3
        mean_values = tuple(abs(value - mean_value) for value in reports)
        mean_scores = (mean_values[0], mean_values[1], mean_values[2])
        rounded = [round(value) for value in reports]
        majority_scores = (
            1.0 - rounded.count(rounded[0]) / 3.0,
            1.0 - rounded.count(rounded[1]) / 3.0,
            1.0 - rounded.count(rounded[2]) / 3.0,
        )
        shuffled_values = tuple(residual[index] / scales[(index + 1) % 3] for index in range(3))
        shuffled_scores = (shuffled_values[0], shuffled_values[1], shuffled_values[2])
        learned.append(_rank(learned_scores, target))
        raw.append(_rank(residual, target))
        means.append(_rank(mean_scores, target))
        majorities.append(_rank(majority_scores, target))
        shuffled.append(_rank(shuffled_scores, target))
        predicted = max(range(3), key=lambda index: (learned_scores[index], -index))
        fused_errors.append(abs(sum(reports[index] for index in range(3) if index != predicted) / 2 - truth))

    def average(values: list[float]) -> float:
        return sum(values) / len(values)

    learned_auc = average(learned)
    baselines = {
        "raw-residual": average(raw),
        "mean-fusion": average(means),
        "majority-vote": average(majorities),
    }
    lo, hi = (float(value) for value in cfg["calibration_band"])
    shuffled_auc = average(shuffled)
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).contradiction.sha256,
        "source_localization_auroc": round(learned_auc, 8),
        "baseline_auroc": {key: round(value, 8) for key, value in baselines.items()},
        "delta_vs_best_baseline": round(learned_auc - max(baselines.values()), 8),
        "fused_estimate_error": round(average(fused_errors), 8),
        "source_scale_shuffled_auroc": round(shuffled_auc, 8),
        "off_floor_and_ceiling": lo <= learned_auc <= hi,
        "strongest_shell_attack_passed": shuffled_auc < learned_auc,
    }


def _f27_rows(
    seed: int, count: int, cfg: dict[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = Random(_seed(seed, f"f27:{split}"))
    columns: list[list[float]] = [[], [], [], []]
    for _ in range(count):
        x_value = rng.gauss(0.0, 1.0)
        intervention = (
            rng.uniform(-1.0, 1.0)
            if split == "train"
            else rng.uniform(1.15, 2.0) * (-1.0 if rng.random() < 0.5 else 1.0)
        )
        outcome = (
            0.65 * x_value
            + 0.90 * intervention
            + 0.40 * x_value * intervention
            + rng.gauss(0.0, float(cfg["outcome_noise_sd"]))
        )
        proxy = (
            outcome + rng.gauss(0.0, float(cfg["proxy_noise_sd"]))
            if split == "train"
            else 0.65 * x_value
            - 0.90 * intervention
            - 0.40 * x_value * intervention
            + rng.gauss(0.0, float(cfg["proxy_noise_sd"]))
        )
        for column, value in zip(columns, (x_value, intervention, outcome, proxy), strict=True):
            column.append(value)
    arrays = tuple(np.asarray(values, dtype=np.float64) for values in columns)
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _design(x_values: np.ndarray, interventions: np.ndarray) -> np.ndarray:
    return np.stack((np.ones(len(x_values)), x_values, interventions, x_values * interventions), axis=1)


def _verify_f27(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train_x, train_a, train_y, train_proxy = _f27_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test_x, test_a, test_y, test_proxy = _f27_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    causal_beta = np.linalg.pinv(_design(train_x, train_a)) @ train_y
    temporal_matrix = np.stack((np.ones(len(train_proxy)), train_proxy), axis=1)
    temporal_beta = np.linalg.pinv(temporal_matrix) @ train_y
    causal_prediction = _design(test_x, test_a) @ causal_beta
    temporal_prediction = np.stack((np.ones(len(test_proxy)), test_proxy), axis=1) @ temporal_beta
    causal_error = float(np.mean(np.abs(test_y - causal_prediction)))
    temporal_error = float(np.mean(np.abs(test_y - temporal_prediction)))
    train_error = np.abs(train_y - _design(train_x, train_a) @ causal_beta)
    threshold = float(np.quantile(train_error, 0.90))
    wrong = np.roll(test_y, 1)
    unrelated = np.roll(test_y, len(test_y) // 2)
    wrong_rejection = float(np.mean(np.abs(wrong - causal_prediction) > threshold))
    unrelated_rejection = float(np.mean(np.abs(unrelated - causal_prediction) > threshold))
    shuffled_a = np.roll(train_a, 17)
    shuffled_beta = np.linalg.pinv(_design(train_x, shuffled_a)) @ train_y
    shuffled_error = float(np.mean(np.abs(test_y - _design(test_x, test_a) @ shuffled_beta)))
    normalized = causal_error / float(np.std(test_y))
    lo, hi = (float(value) for value in cfg["calibration_band"])
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).causal.sha256,
        "unseen_intervention_prediction_error": round(causal_error, 8),
        "temporal_correlation_prediction_error": round(temporal_error, 8),
        "relative_error_reduction": round((temporal_error - causal_error) / temporal_error, 8),
        "wrong_event_rejection_rate": round(wrong_rejection, 8),
        "synchronous_unrelated_rejection_rate": round(unrelated_rejection, 8),
        "intervention_label_shuffled_error": round(shuffled_error, 8),
        "off_floor_and_ceiling": lo <= normalized <= hi,
        "strongest_shell_attack_passed": shuffled_error > causal_error,
    }


def _independent_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    experiments = config["experiments"]
    return {
        EXPERIMENT_IDS[0]: _verify_f21(seed, experiments[EXPERIMENT_IDS[0]]),
        EXPERIMENT_IDS[1]: _verify_f26(seed, experiments[EXPERIMENT_IDS[1]]),
        EXPERIMENT_IDS[2]: _verify_f27(seed, experiments[EXPERIMENT_IDS[2]]),
    }


PRIMARY_COMPARE_KEYS: dict[str, tuple[str, ...]] = {
    EXPERIMENT_IDS[0]: (
        "contract_fixture_sha256",
        "binding_f1_across_delay_and_jitter",
        "fixed_window_binding_f1",
        "delta_vs_fixed_window",
        "shuffled_time_rejection_rate",
        "wrong_time_rejection_rate",
        "context_shuffled_binding_f1",
    ),
    EXPERIMENT_IDS[1]: (
        "contract_fixture_sha256",
        "source_localization_auroc",
        "baseline_auroc",
        "delta_vs_best_baseline",
        "fused_estimate_error",
        "source_scale_shuffled_auroc",
    ),
    EXPERIMENT_IDS[2]: (
        "contract_fixture_sha256",
        "unseen_intervention_prediction_error",
        "temporal_correlation_prediction_error",
        "relative_error_reduction",
        "wrong_event_rejection_rate",
        "synchronous_unrelated_rejection_rate",
        "intervention_label_shuffled_error",
    ),
}


def _primary_problems(run: dict[str, Any], config: dict[str, Any]) -> list[str]:
    problems = []
    expected_seeds = [int(value) for value in config["independent_units"]["seeds"]]
    actual_seeds = [int(row["seed"]) for row in run.get("independent_units", [])]
    if actual_seeds != expected_seeds:
        problems.append("primary seed set or order drift")
        return problems
    for unit in run["independent_units"]:
        seed = int(unit["seed"])
        recomputed = _independent_seed(seed, config)
        for experiment_id in EXPERIMENT_IDS:
            recorded = unit["experiments"][experiment_id]
            for key in PRIMARY_COMPARE_KEYS[experiment_id]:
                if recorded.get(key) != recomputed[experiment_id].get(key):
                    problems.append(f"seed {seed} {experiment_id} field {key} does not recompute")
            if (
                recorded.get("difficulty_calibration", {}).get("off_floor_and_ceiling")
                is not recomputed[experiment_id]["off_floor_and_ceiling"]
            ):
                problems.append(f"seed {seed} {experiment_id} difficulty calibration drift")
    return problems


def _fresh_verdict(experiment_id: str, rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    if experiment_id == EXPERIMENT_IDS[0]:
        favorable = all(
            float(row["delta_vs_fixed_window"]) > float(cfg["minimum_f1_delta"])
            and min(float(row["shuffled_time_rejection_rate"]), float(row["wrong_time_rejection_rate"]))
            >= float(cfg["minimum_control_rejection"])
            and row["off_floor_and_ceiling"] is True
            for row in rows
        )
    elif experiment_id == EXPERIMENT_IDS[1]:
        favorable = all(
            float(row["delta_vs_best_baseline"]) > float(cfg["minimum_auroc_delta"])
            and row["off_floor_and_ceiling"] is True
            for row in rows
        )
    else:
        favorable = all(
            float(row["relative_error_reduction"]) > float(cfg["minimum_relative_error_reduction"])
            and min(
                float(row["wrong_event_rejection_rate"]), float(row["synchronous_unrelated_rejection_rate"])
            )
            >= float(cfg["minimum_control_rejection"])
            and row["off_floor_and_ceiling"] is True
            for row in rows
        )
    strongest_shell = all(row["strongest_shell_attack_passed"] is True for row in rows)
    return {
        "fresh_seed_favorable": favorable,
        "strongest_shell_attack_passed": strongest_shell,
        "fresh_seed_count": len(rows),
    }


def _mutation_checks(run: dict[str, Any], config: dict[str, Any]) -> dict[str, bool]:
    mutations: dict[str, Any] = {}
    changed_metric = copy.deepcopy(run)
    changed_metric["independent_units"][0]["experiments"][EXPERIMENT_IDS[0]]["delta_vs_fixed_window"] += 0.01
    mutations["changed_metric"] = changed_metric
    changed_seed = copy.deepcopy(run)
    changed_seed["independent_units"][0]["seed"] += 1
    mutations["changed_seed"] = changed_seed
    changed_control = copy.deepcopy(run)
    changed_control["independent_units"][0]["experiments"][EXPERIMENT_IDS[2]][
        "wrong_event_rejection_rate"
    ] = 1.0
    mutations["changed_control"] = changed_control
    return {name: bool(_primary_problems(payload, config)) for name, payload in mutations.items()}


def build_verification(run_path: Path = DEFAULT_RUN, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    run, config = _read_inputs(run_path, config_path)
    problems = _primary_problems(run, config)
    fresh_seeds = [int(value) for value in config["independent_units"]["fresh_verifier_seeds"]]
    fresh_units: list[dict[str, Any]] = [
        {"seed": seed, "experiments": _independent_seed(seed, config)} for seed in fresh_seeds
    ]
    per_experiment: dict[str, dict[str, Any]] = {}
    for experiment_id in EXPERIMENT_IDS:
        fresh_rows: list[dict[str, Any]] = [unit["experiments"][experiment_id] for unit in fresh_units]
        fresh = _fresh_verdict(experiment_id, fresh_rows, config["experiments"][experiment_id])
        primary_favorable = run["aggregate"][experiment_id]["programmatic_favorable"] is True
        verified = bool(
            primary_favorable and fresh["fresh_seed_favorable"] and fresh["strongest_shell_attack_passed"]
        )
        per_experiment[experiment_id] = {
            "primary_programmatic_favorable": primary_favorable,
            **fresh,
            "programmatic_pattern_verified": verified,
            "scientific_promotion_allowed": False,
            "verdict": (
                "verified-favorable-toy-pattern"
                if verified
                else "verified-null"
                if not primary_favorable
                else "favorable-pattern-not-reproduced"
            ),
        }
        if primary_favorable and not verified:
            problems.append(f"fresh verifier did not close favorable primary {experiment_id}")
    mutations = _mutation_checks(run, config)
    if not all(mutations.values()):
        problems.append("one or more verifier mutations escaped rejection")
    verification = {
        "schema": VERIFY_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "R1 independently re-executed programmatic toy mechanics",
        "run_receipt": {
            "path": _display_path(run_path),
            "sha256": _sha256_file(run_path),
            "payload_sha256": run["payload_sha256"],
        },
        "config": {
            "path": _display_path(config_path),
            "sha256": _sha256_file(config_path),
        },
        "independence": {
            "imports_run_evaluator": False,
            "primary_seed_reexecution": True,
            "fresh_seed_reexecution": True,
            "separate_fit_implementation": True,
        },
        "primary_recompute_exact": not problems,
        "fresh_units": fresh_units,
        "per_experiment": per_experiment,
        "mutation_checks": mutations,
        "problems": problems,
        "all_ok": not problems,
        "scientific_capability_claim": False,
    }
    verification["payload_sha256"] = canonical_sha256(verification)
    return verification


def write_verification(
    output: Path = DEFAULT_OUTPUT,
    run_path: Path = DEFAULT_RUN,
    config_path: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    verification = build_verification(run_path, config_path)
    _atomic_json(output, verification)
    return verification
