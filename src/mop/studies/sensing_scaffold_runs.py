"""Executed toy beds for the f21, f26, and f27 sensing scaffold contracts.

The beds are deterministic, local, and programmatic. They exercise the declared controls and
difficulty gates without natural data or a capability claim. Favorable toy-bed patterns remain
R1 mechanics until an independent verifier closes fresh seeds and the external form gates are met.
"""

from __future__ import annotations

import copy
import hashlib
import json
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

CONFIG_SCHEMA = "mop-sensing-scaffold-run-config/v1"
RECEIPT_SCHEMA = "mop-sensing-scaffold-run/v1"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiment" / "sensing_scaffold_runs.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "proof" / "SENSING_SCAFFOLD_RUN.json"
EXPERIMENT_IDS = (
    "f21_asynchronous_temporal_binding",
    "f26_cross_form_contradiction_triangulation",
    "f27_causal_crossmodal_binding",
)
SOURCE_PATHS = (
    "configs/experiment/sensing_scaffold_runs.yaml",
    "registry/experiments.yaml",
    "src/mop/substrate/sensing_scaffold.py",
    "src/mop/studies/sensing_scaffold_runs.py",
    "src/mop/studies/sensing_scaffold_verify.py",
    "scripts/run_sensing_scaffold.py",
    "scripts/verify_sensing_scaffold.py",
    "tests/unit/test_sensing_scaffold_runs.py",
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


def _file_receipt(relative: str) -> dict[str, Any]:
    path = REPO_ROOT / relative
    return {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _seed(seed: int, label: str) -> int:
    raw = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return int.from_bytes(raw[:8], "big")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema") != CONFIG_SCHEMA:
        raise ValueError("sensing scaffold run config schema drift")
    if config.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("sensing scaffold run claim scope drift")
    if not isinstance(config.get("null_hypothesis"), str) or not config["null_hypothesis"].strip():
        raise ValueError("sensing scaffold run null hypothesis is required")
    units = config.get("independent_units", {})
    seeds = tuple(int(value) for value in units.get("seeds", ()))
    fresh = tuple(int(value) for value in units.get("fresh_verifier_seeds", ()))
    minimum = int(config.get("stop_contract", {}).get("minimum_independent_units", 0))
    if len(seeds) < minimum or len(fresh) < minimum:
        raise ValueError("sensing run needs the preregistered minimum primary and fresh seeds")
    if len(set(seeds + fresh)) != len(seeds) + len(fresh) or min(seeds + fresh) < 0:
        raise ValueError("sensing primary and fresh seeds must be unique and nonnegative")
    experiments = config.get("experiments", {})
    if tuple(experiments) != EXPERIMENT_IDS:
        raise ValueError("sensing experiment order or coverage drift")
    expected_controls = {
        EXPERIMENT_IDS[0]: ("fixed-window-baseline", "shuffled-timing", "wrong-time", "exact-replay"),
        EXPERIMENT_IDS[1]: ("majority-vote", "mean-fusion", "raw-residual", "exact-replay"),
        EXPERIMENT_IDS[2]: (
            "temporal-correlation",
            "wrong-event",
            "synchronous-unrelated",
            "exact-replay",
        ),
    }
    for experiment_id, controls in expected_controls.items():
        if tuple(experiments[experiment_id].get("controls", ())) != controls:
            raise ValueError(f"control drift for {experiment_id}")
    envelope = config.get("resource_envelope", {})
    if (
        envelope.get("device") != "cpu"
        or int(envelope.get("cpu_threads", 0)) != 1
        or envelope.get("model_weights_loaded") is not False
        or envelope.get("external_data_allowed") is not False
    ):
        raise ValueError("sensing run must remain one-thread CPU with no weights or external data")
    return config


def _registry_bindings() -> dict[str, Any]:
    payload = yaml.safe_load((REPO_ROOT / "registry" / "experiments.yaml").read_text(encoding="utf-8"))
    index = {str(row["id"]): row for row in payload["experiments"]}
    out: dict[str, Any] = {}
    for experiment_id in EXPERIMENT_IDS:
        row = index.get(experiment_id)
        if row is None or row.get("status") != "registry-only" or row.get("resource_tier") != "cpu-now":
            raise ValueError(f"live registry row {experiment_id} is absent or outside the declared lane")
        out[experiment_id] = {
            "row_sha256": canonical_sha256(row),
            "null_hypothesis": row["null_hypothesis"],
            "metrics": list(row["metrics"]),
            "controls": list(row["controls"]),
            "status": row["status"],
        }
    return out


def _f1(labels: list[int], predictions: list[int]) -> float:
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions, strict=True))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions, strict=True))
    denominator = 2 * tp + fp + fn
    return 0.0 if denominator == 0 else 2 * tp / denominator


def _fit_threshold(rows: list[tuple[int, int, float]]) -> float:
    candidates = sorted({gap for _, _, gap in rows})
    best = (-1.0, 0.0)
    for threshold in candidates:
        score = _f1([label for _, label, _ in rows], [int(gap <= threshold) for _, _, gap in rows])
        if score > best[0] or (score == best[0] and threshold < best[1]):
            best = (score, threshold)
    return best[1]


def _temporal_rows(seed: int, count: int, cfg: dict[str, Any], split: str) -> list[tuple[int, int, float]]:
    rng = Random(_seed(seed, f"f21:{split}"))
    contexts = int(cfg["contexts"])
    rows = []
    for _ in range(count):
        context = rng.randrange(contexts)
        label = int(rng.random() < float(cfg["positive_rate"]))
        center = 2.4 + 2.3 * context + (0.0 if label else 3.4)
        gap = max(0.0, rng.gauss(center, float(cfg["gap_noise_sd"])))
        if rng.random() < float(cfg["label_noise_rate"]):
            label = 1 - label
        rows.append((context, label, gap))
    return rows


def _temporal_controls(seed: int, count: int, contexts: int) -> tuple[list[tuple[int, float]], ...]:
    rng = Random(_seed(seed, "f21:controls"))
    shuffled = []
    wrong_time = []
    for _ in range(count):
        context = rng.randrange(contexts)
        shuffled.append((context, max(0.0, rng.gauss(8.8 + 2.3 * context, 2.0))))
        wrong_time.append((context, max(0.0, rng.gauss(11.2 + 2.3 * context, 2.4))))
    return shuffled, wrong_time


def _evaluate_f21_once(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train = _temporal_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test = _temporal_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    contexts = int(cfg["contexts"])
    windows = {
        context: _fit_threshold([row for row in train if row[0] == context]) for context in range(contexts)
    }
    fixed_window = _fit_threshold(train)
    labels = [row[1] for row in test]
    learned_predictions = [int(gap <= windows[context]) for context, _, gap in test]
    fixed_predictions = [int(gap <= fixed_window) for _, _, gap in test]
    shuffled_context_predictions = [int(gap <= windows[(context + 1) % contexts]) for context, _, gap in test]
    shuffled, wrong_time = _temporal_controls(seed, len(test) // 2, contexts)

    def rejection(rows: list[tuple[int, float]]) -> float:
        return sum(gap > windows[context] for context, gap in rows) / len(rows)

    learned_f1 = _f1(labels, learned_predictions)
    fixed_f1 = _f1(labels, fixed_predictions)
    shuffled_context_f1 = _f1(labels, shuffled_context_predictions)
    lo, hi = (float(value) for value in cfg["calibration_band"])
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).temporal.sha256,
        "learned_windows": {str(key): round(value, 8) for key, value in windows.items()},
        "fixed_window": round(fixed_window, 8),
        "binding_f1_across_delay_and_jitter": round(learned_f1, 8),
        "fixed_window_binding_f1": round(fixed_f1, 8),
        "delta_vs_fixed_window": round(learned_f1 - fixed_f1, 8),
        "shuffled_time_rejection_rate": round(rejection(shuffled), 8),
        "wrong_time_rejection_rate": round(rejection(wrong_time), 8),
        "context_shuffled_binding_f1": round(shuffled_context_f1, 8),
        "difficulty_calibration": {
            "positive_fraction": round(sum(labels) / len(labels), 8),
            "band": [lo, hi],
            "off_floor_and_ceiling": lo <= learned_f1 <= hi,
        },
        "heldout_examples": len(test),
    }


def _contradiction_rows(
    seed: int, count: int, cfg: dict[str, Any], split: str
) -> list[tuple[float, int, tuple[float, float, float]]]:
    rng = Random(_seed(seed, f"f26:{split}"))
    noise_values = [float(value) for value in cfg["source_noise_sd"]]
    if len(noise_values) != 3:
        raise ValueError("f26 requires exactly three source noise scales")
    noise = (noise_values[0], noise_values[1], noise_values[2])
    rows: list[tuple[float, int, tuple[float, float, float]]] = []
    for _ in range(count):
        truth = rng.gauss(0.0, 2.2)
        dissenter = rng.randrange(3)
        reports = [truth + rng.gauss(0.0, source_sd) for source_sd in noise]
        sign = -1.0 if rng.random() < 0.5 else 1.0
        reports[dissenter] += sign * (
            float(cfg["corruption_magnitude"]) + abs(rng.gauss(0.0, float(cfg["corruption_sd"])))
        )
        rows.append((truth, dissenter, (reports[0], reports[1], reports[2])))
    return rows


def _ranking_auroc(scores: tuple[float, float, float], positive: int) -> float:
    value = 0.0
    for index, score in enumerate(scores):
        if index == positive:
            continue
        value += float(scores[positive] > score) + 0.5 * float(scores[positive] == score)
    return value / 2.0


def _majority_scores(reports: tuple[float, float, float]) -> tuple[float, float, float]:
    bins = [round(value) for value in reports]
    return (
        1.0 - bins.count(bins[0]) / 3.0,
        1.0 - bins.count(bins[1]) / 3.0,
        1.0 - bins.count(bins[2]) / 3.0,
    )


def _evaluate_f26_once(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train = _contradiction_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test = _contradiction_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    honest_residuals: list[list[float]] = [[], [], []]
    for _, dissenter, reports in train:
        center = median(reports)
        for source, value in enumerate(reports):
            if source != dissenter:
                honest_residuals[source].append(abs(value - center))
    scale_values = [max(0.05, sum(values) / len(values)) for values in honest_residuals]
    scales = (scale_values[0], scale_values[1], scale_values[2])
    learned_auc = []
    raw_auc = []
    mean_auc = []
    majority_auc = []
    fused_errors = []
    shuffled_scale_auc = []
    for truth, dissenter, reports in test:
        center = median(reports)
        mean_center = sum(reports) / 3.0
        raw = tuple(abs(value - center) for value in reports)
        raw_scores = (raw[0], raw[1], raw[2])
        learned_scores = tuple(value / scales[index] for index, value in enumerate(raw_scores))
        learned_scores = (learned_scores[0], learned_scores[1], learned_scores[2])
        mean_values = tuple(abs(value - mean_center) for value in reports)
        mean_scores = (mean_values[0], mean_values[1], mean_values[2])
        majority = _majority_scores(reports)
        shuffled_values = tuple(value / scales[(index + 1) % 3] for index, value in enumerate(raw_scores))
        shuffled_scale = (shuffled_values[0], shuffled_values[1], shuffled_values[2])
        learned_auc.append(_ranking_auroc(learned_scores, dissenter))
        raw_auc.append(_ranking_auroc(raw_scores, dissenter))
        mean_auc.append(_ranking_auroc(mean_scores, dissenter))
        majority_auc.append(_ranking_auroc(majority, dissenter))
        shuffled_scale_auc.append(_ranking_auroc(shuffled_scale, dissenter))
        predicted = max(range(3), key=lambda index: (learned_scores[index], -index))
        fused_errors.append(
            abs(sum(value for index, value in enumerate(reports) if index != predicted) / 2 - truth)
        )

    def avg(values: list[float]) -> float:
        return sum(values) / len(values)

    learned_auc_mean = avg(learned_auc)
    baselines = {
        "raw-residual": avg(raw_auc),
        "mean-fusion": avg(mean_auc),
        "majority-vote": avg(majority_auc),
    }
    best_baseline = max(baselines.values())
    lo, hi = (float(value) for value in cfg["calibration_band"])
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).contradiction.sha256,
        "learned_source_scales": [round(value, 8) for value in scales],
        "source_localization_auroc": round(learned_auc_mean, 8),
        "baseline_auroc": {key: round(value, 8) for key, value in baselines.items()},
        "delta_vs_best_baseline": round(learned_auc_mean - best_baseline, 8),
        "fused_estimate_error": round(avg(fused_errors), 8),
        "source_scale_shuffled_auroc": round(avg(shuffled_scale_auc), 8),
        "difficulty_calibration": {
            "band": [lo, hi],
            "off_floor_and_ceiling": lo <= learned_auc_mean <= hi,
        },
        "heldout_examples": len(test),
    }


def _causal_rows(
    seed: int, count: int, cfg: dict[str, Any], split: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = Random(_seed(seed, f"f27:{split}"))
    x_values = []
    interventions = []
    outcomes = []
    proxies = []
    for _ in range(count):
        x_value = rng.gauss(0.0, 1.0)
        if split == "train":
            intervention = rng.uniform(-1.0, 1.0)
        else:
            intervention = rng.uniform(1.15, 2.0) * (-1.0 if rng.random() < 0.5 else 1.0)
        outcome = (
            0.65 * x_value
            + 0.90 * intervention
            + 0.40 * x_value * intervention
            + rng.gauss(0.0, float(cfg["outcome_noise_sd"]))
        )
        if split == "train":
            proxy = outcome + rng.gauss(0.0, float(cfg["proxy_noise_sd"]))
        else:
            proxy = (
                0.65 * x_value
                - 0.90 * intervention
                - 0.40 * x_value * intervention
                + rng.gauss(0.0, float(cfg["proxy_noise_sd"]))
            )
        x_values.append(x_value)
        interventions.append(intervention)
        outcomes.append(outcome)
        proxies.append(proxy)
    arrays = tuple(
        np.asarray(values, dtype=np.float64) for values in (x_values, interventions, outcomes, proxies)
    )
    return arrays[0], arrays[1], arrays[2], arrays[3]


def _causal_design(x_values: np.ndarray, interventions: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(x_values)), x_values, interventions, x_values * interventions))


def _mae(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean(np.abs(left - right)))


def _evaluate_f27_once(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    train_x, train_a, train_y, train_proxy = _causal_rows(seed, int(cfg["train_examples"]), cfg, "train")
    test_x, test_a, test_y, test_proxy = _causal_rows(seed, int(cfg["heldout_examples"]), cfg, "heldout")
    causal_beta = np.linalg.lstsq(_causal_design(train_x, train_a), train_y, rcond=None)[0]
    temporal_beta = np.linalg.lstsq(
        np.column_stack((np.ones(len(train_proxy)), train_proxy)), train_y, rcond=None
    )[0]
    causal_prediction = _causal_design(test_x, test_a) @ causal_beta
    temporal_prediction = np.column_stack((np.ones(len(test_proxy)), test_proxy)) @ temporal_beta
    causal_error = _mae(test_y, causal_prediction)
    temporal_error = _mae(test_y, temporal_prediction)
    relative_reduction = (temporal_error - causal_error) / temporal_error

    train_residuals = np.abs(train_y - _causal_design(train_x, train_a) @ causal_beta)
    rejection_threshold = float(np.quantile(train_residuals, 0.90))
    wrong_event = np.roll(test_y, 1)
    synchronous_unrelated = np.roll(test_y, len(test_y) // 2)
    wrong_rejection = float(np.mean(np.abs(wrong_event - causal_prediction) > rejection_threshold))
    synchronous_rejection = float(
        np.mean(np.abs(synchronous_unrelated - causal_prediction) > rejection_threshold)
    )
    true_acceptance = float(np.mean(np.abs(test_y - causal_prediction) <= rejection_threshold))

    shuffled_a = np.roll(train_a, 17)
    shuffled_beta = np.linalg.lstsq(_causal_design(train_x, shuffled_a), train_y, rcond=None)[0]
    shuffled_error = _mae(test_y, _causal_design(test_x, test_a) @ shuffled_beta)
    normalized_error = causal_error / float(np.std(test_y))
    lo, hi = (float(value) for value in cfg["calibration_band"])
    return {
        "contract_fixture_sha256": make_sensing_fixture(seed=seed).causal.sha256,
        "unseen_intervention_prediction_error": round(causal_error, 8),
        "temporal_correlation_prediction_error": round(temporal_error, 8),
        "relative_error_reduction": round(relative_reduction, 8),
        "wrong_event_rejection_rate": round(wrong_rejection, 8),
        "synchronous_unrelated_rejection_rate": round(synchronous_rejection, 8),
        "true_pair_acceptance_rate": round(true_acceptance, 8),
        "intervention_label_shuffled_error": round(shuffled_error, 8),
        "difficulty_calibration": {
            "normalized_causal_mae": round(normalized_error, 8),
            "band": [lo, hi],
            "off_floor_and_ceiling": lo <= normalized_error <= hi,
        },
        "heldout_examples": len(test_y),
    }


def evaluate_seed(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    experiments = config["experiments"]
    evaluators = (
        (_evaluate_f21_once, EXPERIMENT_IDS[0]),
        (_evaluate_f26_once, EXPERIMENT_IDS[1]),
        (_evaluate_f27_once, EXPERIMENT_IDS[2]),
    )
    results: dict[str, Any] = {}
    for evaluator, experiment_id in evaluators:
        first = evaluator(seed, experiments[experiment_id])
        second = evaluator(seed, experiments[experiment_id])
        first["exact_replay"] = canonical_sha256(first) == canonical_sha256(second)
        results[experiment_id] = first
    return {"seed": seed, "experiments": results, "unit_sha256": canonical_sha256(results)}


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _aggregate(units: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    experiments = config["experiments"]
    by_id = {
        experiment_id: [unit["experiments"][experiment_id] for unit in units]
        for experiment_id in EXPERIMENT_IDS
    }
    f21_cfg = experiments[EXPERIMENT_IDS[0]]
    f21_rows = by_id[EXPERIMENT_IDS[0]]
    f21_deltas = [float(row["delta_vs_fixed_window"]) for row in f21_rows]
    f21_control_min = min(
        min(float(row["shuffled_time_rejection_rate"]), float(row["wrong_time_rejection_rate"]))
        for row in f21_rows
    )
    f21_positive = bool(
        all(delta > float(f21_cfg["minimum_f1_delta"]) for delta in f21_deltas)
        and f21_control_min >= float(f21_cfg["minimum_control_rejection"])
        and all(row["difficulty_calibration"]["off_floor_and_ceiling"] for row in f21_rows)
        and all(row["exact_replay"] for row in f21_rows)
    )

    f26_cfg = experiments[EXPERIMENT_IDS[1]]
    f26_rows = by_id[EXPERIMENT_IDS[1]]
    f26_deltas = [float(row["delta_vs_best_baseline"]) for row in f26_rows]
    f26_positive = bool(
        all(delta > float(f26_cfg["minimum_auroc_delta"]) for delta in f26_deltas)
        and all(row["difficulty_calibration"]["off_floor_and_ceiling"] for row in f26_rows)
        and all(row["exact_replay"] for row in f26_rows)
    )

    f27_cfg = experiments[EXPERIMENT_IDS[2]]
    f27_rows = by_id[EXPERIMENT_IDS[2]]
    f27_reductions = [float(row["relative_error_reduction"]) for row in f27_rows]
    f27_control_min = min(
        min(float(row["wrong_event_rejection_rate"]), float(row["synchronous_unrelated_rejection_rate"]))
        for row in f27_rows
    )
    f27_positive = bool(
        all(value > float(f27_cfg["minimum_relative_error_reduction"]) for value in f27_reductions)
        and f27_control_min >= float(f27_cfg["minimum_control_rejection"])
        and all(row["difficulty_calibration"]["off_floor_and_ceiling"] for row in f27_rows)
        and all(row["exact_replay"] for row in f27_rows)
    )
    return {
        EXPERIMENT_IDS[0]: {
            "mean_binding_f1": round(
                _mean([float(row["binding_f1_across_delay_and_jitter"]) for row in f21_rows]), 8
            ),
            "mean_delta_vs_fixed_window": round(_mean(f21_deltas), 8),
            "minimum_control_rejection": round(f21_control_min, 8),
            "programmatic_favorable": f21_positive,
            "verdict": "favorable-toy-pattern-pending-independent-verification" if f21_positive else "null",
        },
        EXPERIMENT_IDS[1]: {
            "mean_source_localization_auroc": round(
                _mean([float(row["source_localization_auroc"]) for row in f26_rows]), 8
            ),
            "mean_delta_vs_best_baseline": round(_mean(f26_deltas), 8),
            "programmatic_favorable": f26_positive,
            "verdict": "favorable-toy-pattern-pending-independent-verification" if f26_positive else "null",
        },
        EXPERIMENT_IDS[2]: {
            "mean_unseen_intervention_error": round(
                _mean([float(row["unseen_intervention_prediction_error"]) for row in f27_rows]), 8
            ),
            "mean_relative_error_reduction": round(_mean(f27_reductions), 8),
            "minimum_control_rejection": round(f27_control_min, 8),
            "programmatic_favorable": f27_positive,
            "verdict": "favorable-toy-pattern-pending-independent-verification" if f27_positive else "null",
        },
    }


def build_receipt(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_config(config_path)
    units = [evaluate_seed(int(seed), config) for seed in config["independent_units"]["seeds"]]
    aggregate = _aggregate(units, config)
    registry_bindings = _registry_bindings()
    per_experiment_nulls = {
        experiment_id: str(registry_bindings[experiment_id]["null_hypothesis"])
        for experiment_id in EXPERIMENT_IDS
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "claim_scope": CLAIM_SCOPE,
        "evidence_class": "R1 deterministic programmatic toy execution",
        "scientific_capability_claim": False,
        "config": _file_receipt(str(config_path.relative_to(REPO_ROOT))),
        "source_receipts": [_file_receipt(path) for path in SOURCE_PATHS],
        "registry_bindings": registry_bindings,
        "null_contract": {
            "aggregate": config["null_hypothesis"],
            "per_experiment": per_experiment_nulls,
            "per_experiment_sha256": canonical_sha256(per_experiment_nulls),
        },
        "independent_units": units,
        "aggregate": aggregate,
        "favorable_experiments_requiring_fresh_verification": sorted(
            experiment_id for experiment_id, row in aggregate.items() if row["programmatic_favorable"] is True
        ),
        "limitations": [
            "synthetic toy beds do not supply native audiovisual evidence or independent natural forms",
            "f21 and f27 remain blocked on rights-clean native forms and interventions",
            "f26 remains blocked on genuinely independent forms over shared referents",
            "a favorable toy pattern is not a capability result",
        ],
        "resource_envelope": copy.deepcopy(config["resource_envelope"]),
    }
    receipt["payload_sha256"] = canonical_sha256(receipt)
    return receipt


def write_receipt(output: Path = DEFAULT_OUTPUT, config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    receipt = build_receipt(config_path)
    _atomic_json(output, receipt)
    return receipt


def assert_receipt(payload: dict[str, Any]) -> None:
    if payload.get("schema") != RECEIPT_SCHEMA or payload.get("claim_scope") != CLAIM_SCOPE:
        raise ValueError("sensing run receipt schema or claim scope drift")
    expected = canonical_sha256({key: value for key, value in payload.items() if key != "payload_sha256"})
    if payload.get("payload_sha256") != expected:
        raise ValueError("sensing run receipt payload digest mismatch")
    if payload.get("scientific_capability_claim") is not False:
        raise ValueError("sensing toy receipt cannot make a scientific capability claim")
    null_contract = payload.get("null_contract", {})
    per_experiment = null_contract.get("per_experiment", {})
    if (
        not isinstance(null_contract.get("aggregate"), str)
        or not null_contract["aggregate"].strip()
        or set(per_experiment) != set(EXPERIMENT_IDS)
        or null_contract.get("per_experiment_sha256") != canonical_sha256(per_experiment)
    ):
        raise ValueError("sensing receipt null contract is incomplete")
    for experiment_id in EXPERIMENT_IDS:
        if experiment_id not in payload.get("aggregate", {}):
            raise ValueError(f"sensing receipt misses {experiment_id}")
    if not all(
        row["experiments"][experiment_id].get("exact_replay") is True
        for row in payload.get("independent_units", [])
        for experiment_id in EXPERIMENT_IDS
    ):
        raise ValueError("sensing exact replay control did not close")
