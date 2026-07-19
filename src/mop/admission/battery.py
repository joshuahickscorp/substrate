"""The Mechanism Admission Battery: eight deterministic clause functions plus ``run_battery``.

Each clause is a pure, deterministic numpy function over provided arrays. There is no randomness: every
train/test split is a grouped split derived from the sorted distinct group ids, and every probe is a
closed-form ridge fit. A mechanism is admitted only if ALL eight clauses pass; a missing required input is
a clause failure, never a crash, because a mechanism that cannot supply the evidence cannot be admitted.

Input contract (a mapping; arrays are numpy-coercible sequences):

- ``what_true``: (N,) true WHAT target values.
- ``what_pred``: mapping method -> (N,) predictions. Methods should include ``candidate``, ``constant``,
  ``empirical_prior``, ``frozen_random``, and ``handcrafted_control``.
- ``when_features``: (N, F) WHEN feature matrix.
- ``baseline_heuristics``: (N, B) simple energy/rate/change heuristic features.
- ``recompute_value``: (N,) nonnegative marginal value of recomputing at each frame.
- ``labels``: (N,) target-presence labels (used only to contrast with marginal value in clause 3 evidence).
- ``group_ids``: (N,) room/session/referent group identifiers.
- ``budget``: scalar compute budget. A value in (0, 1] is read as a fraction of N; a value > 1 as a count.
- ``design``: mapping with ``sesoi``, ``power``, ``multiplicity_correction``, ``stop_rule``.
- ``noisy_tv_firing_rate`` / ``noisy_tv_base_rate`` / ``shuffled_target_score`` / ``wrong_time_score`` /
  ``chance_level`` / ``primary_control``: control-behavior summaries (arrays may override the rate scalars).
- ``architecture_favorable``: sequence of per-architecture booleans (did the favorable effect hold there).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from mop.admission.prereg import MECHANISM_ADMISSION_PREREG

_RIDGE_LAMBDA = 1e-3


def _as_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=float)


def _clause_result(clause_id: str, passed: bool, evidence: dict[str, Any]) -> dict[str, Any]:
    return {"clause": clause_id, "passed": bool(passed), "evidence": evidence}


def _grouped_split(group_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic grouped split: the first half of the sorted distinct groups train, the rest test."""

    unique = np.unique(group_ids)
    if unique.size < 2:
        mask_all = np.ones(group_ids.shape[0], dtype=bool)
        return mask_all, mask_all
    cut = max(1, unique.size // 2)
    train_groups = set(unique[:cut].tolist())
    train_mask = np.array([gid in train_groups for gid in group_ids], dtype=bool)
    test_mask = ~train_mask
    if not test_mask.any() or not train_mask.any():
        return np.ones_like(train_mask), np.ones_like(train_mask)
    return train_mask, test_mask


def _ridge_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    """Closed-form ridge regression with a bias column. Deterministic, rank-safe."""

    x_train = np.atleast_2d(x_train)
    x_test = np.atleast_2d(x_test)
    if x_train.shape[0] == 0:
        return np.zeros(x_test.shape[0])
    bias_train = np.ones((x_train.shape[0], 1))
    bias_test = np.ones((x_test.shape[0], 1))
    a_train = np.hstack([x_train, bias_train])
    a_test = np.hstack([x_test, bias_test])
    gram = a_train.T @ a_train + _RIDGE_LAMBDA * np.eye(a_train.shape[1])
    weights = np.linalg.solve(gram, a_train.T @ y_train)
    return a_test @ weights


def _balanced_accuracy(y_true_bin: np.ndarray, y_pred_bin: np.ndarray) -> float:
    """Balanced accuracy over a binary target. Falls back to plain accuracy on a degenerate class."""

    pos = y_true_bin
    neg = ~y_true_bin
    if pos.any() and neg.any():
        tpr = float((y_pred_bin & pos).sum()) / float(pos.sum())
        tnr = float((~y_pred_bin & neg).sum()) / float(neg.sum())
        return 0.5 * (tpr + tnr)
    return float((y_pred_bin == y_true_bin).mean())


def _decodability_accuracy(features: np.ndarray, target: np.ndarray, group_ids: np.ndarray) -> float:
    """Grouped-split balanced accuracy of a ridge probe predicting whether the target is above its median."""

    features = np.atleast_2d(features)
    if features.shape[0] != target.shape[0]:
        return 0.5
    train_mask, test_mask = _grouped_split(group_ids)
    y_train = target[train_mask]
    threshold = float(np.median(y_train))
    predictions = _ridge_fit_predict(features[train_mask], y_train, features[test_mask])
    y_true_bin = target[test_mask] > threshold
    pred_threshold = float(np.median(_ridge_fit_predict(features[train_mask], y_train, features[train_mask])))
    y_pred_bin = predictions > pred_threshold
    return _balanced_accuracy(y_true_bin, y_pred_bin)


def _scalar_or_mean(inputs: Mapping[str, Any], scalar_key: str, array_key: str) -> float | None:
    if array_key in inputs and inputs[array_key] is not None:
        arr = _as_float_array(inputs[array_key])
        if arr.size:
            return float(arr.mean())
    if scalar_key in inputs and inputs[scalar_key] is not None:
        return float(inputs[scalar_key])
    return None


def clause_what_absolute_sufficiency(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """1) The WHAT estimator must beat constant, empirical-prior, frozen-random, and a handcrafted control."""

    sesoi = float(prereg["constants"]["sesoi_default"])
    required = ["constant", "empirical_prior", "frozen_random", "handcrafted_control"]
    what_true = inputs.get("what_true")
    what_pred = inputs.get("what_pred") or {}
    if what_true is None or "candidate" not in what_pred:
        return _clause_result(
            "what_absolute_sufficiency", False, {"reason": "missing what_true or candidate prediction"}
        )
    y = _as_float_array(what_true)
    cand_mae = float(np.abs(_as_float_array(what_pred["candidate"]) - y).mean())
    margins: dict[str, float] = {}
    maes: dict[str, float] = {"candidate": cand_mae}
    beaten = True
    for name in required:
        if name not in what_pred:
            return _clause_result(
                "what_absolute_sufficiency", False, {"reason": f"missing baseline {name!r}"}
            )
        base_mae = float(np.abs(_as_float_array(what_pred[name]) - y).mean())
        maes[name] = base_mae
        margins[name] = base_mae - cand_mae
        if base_mae - cand_mae < sesoi:
            beaten = False
    return _clause_result(
        "what_absolute_sufficiency",
        beaten,
        {"sesoi": sesoi, "mae": maes, "margins_over_baselines": margins},
    )


def clause_oracle_budget_headroom(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """2) At the EXACT budget, an oracle change-aligned policy must beat rate-matched-random by the lift."""

    min_lift = float(prereg["constants"]["oracle_headroom_min_lift"])
    primary_required = prereg["constants"]["primary_control"]
    value = inputs.get("recompute_value")
    budget = inputs.get("budget")
    if value is None or budget is None:
        return _clause_result(
            "oracle_budget_headroom", False, {"reason": "missing recompute_value or budget"}
        )
    v = _as_float_array(value)
    n = v.shape[0]
    budget = float(budget)
    k = int(round(budget * n)) if budget <= 1.0 else int(round(budget))
    k = int(np.clip(k, 1, n))
    oracle_captured = float(np.sort(v)[::-1][:k].sum())
    random_expected = float(k * v.mean())
    lift = (oracle_captured - random_expected) / random_expected if random_expected > 0 else 0.0
    primary_control = inputs.get("primary_control", primary_required)
    passed = lift >= min_lift and primary_control == primary_required
    return _clause_result(
        "oracle_budget_headroom",
        passed,
        {
            "budget_frames": k,
            "oracle_captured": oracle_captured,
            "rate_matched_random_expected": random_expected,
            "relative_lift": lift,
            "min_relative_lift": min_lift,
            "primary_control": primary_control,
        },
    )


def clause_when_decodability(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """3) A probe must decode the MARGINAL VALUE of recomputation from WHEN features above chance."""

    floor = float(prereg["constants"]["decodability_floor_over_chance"])
    features = inputs.get("when_features")
    value = inputs.get("recompute_value")
    group_ids = inputs.get("group_ids")
    if features is None or value is None or group_ids is None:
        return _clause_result(
            "when_decodability", False, {"reason": "missing when_features, recompute_value, or group_ids"}
        )
    groups = np.asarray(group_ids)
    marginal_acc = _decodability_accuracy(_as_float_array(features), _as_float_array(value), groups)
    margin = marginal_acc - 0.5
    evidence: dict[str, Any] = {
        "marginal_value_balanced_accuracy": marginal_acc,
        "chance": 0.5,
        "margin_over_chance": margin,
        "floor": floor,
    }
    labels = inputs.get("labels")
    if labels is not None:
        evidence["target_presence_balanced_accuracy"] = _decodability_accuracy(
            _as_float_array(features), _as_float_array(labels), groups
        )
        evidence["note"] = (
            "presence decodability shown only to contrast; the pass criterion is marginal value"
        )
    return _clause_result("when_decodability", margin >= floor, evidence)


def clause_incremental_value(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """4) The WHEN features must add value beyond energy/rate/change heuristics."""

    floor = float(prereg["constants"]["incremental_value_floor"])
    when_features = inputs.get("when_features")
    baselines = inputs.get("baseline_heuristics")
    value = inputs.get("recompute_value")
    group_ids = inputs.get("group_ids")
    if when_features is None or baselines is None or value is None or group_ids is None:
        return _clause_result(
            "incremental_value",
            False,
            {"reason": "missing when_features, baseline_heuristics, recompute_value, or group_ids"},
        )
    groups = np.asarray(group_ids)
    target = _as_float_array(value)
    base = np.atleast_2d(_as_float_array(baselines))
    when = np.atleast_2d(_as_float_array(when_features))
    if base.shape[0] == 1 and base.shape[1] == target.shape[0]:
        base = base.T
    if when.shape[0] == 1 and when.shape[1] == target.shape[0]:
        when = when.T
    acc_baseline = _decodability_accuracy(base, target, groups)
    acc_combined = _decodability_accuracy(np.hstack([base, when]), target, groups)
    increment = acc_combined - acc_baseline
    return _clause_result(
        "incremental_value",
        increment >= floor,
        {
            "baseline_only_accuracy": acc_baseline,
            "baseline_plus_when_accuracy": acc_combined,
            "increment": increment,
            "floor": floor,
        },
    )


def clause_group_disjoint_validity(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """5) Validity must hold under the true grouped unit, not pooled frames."""

    sesoi = float(prereg["constants"]["sesoi_default"])
    min_units = int(prereg["constants"]["min_independent_units"])
    what_true = inputs.get("what_true")
    what_pred = inputs.get("what_pred") or {}
    group_ids = inputs.get("group_ids")
    if what_true is None or "candidate" not in what_pred or group_ids is None:
        return _clause_result(
            "group_disjoint_validity",
            False,
            {"reason": "missing what_true, candidate prediction, or group_ids"},
        )
    base_key = next(
        (
            k
            for k in ("rate_matched_random", "handcrafted_control", "empirical_prior", "constant")
            if k in what_pred
        ),
        None,
    )
    if base_key is None:
        return _clause_result(
            "group_disjoint_validity", False, {"reason": "no baseline present to form an advantage"}
        )
    y = _as_float_array(what_true)
    cand_err = np.abs(_as_float_array(what_pred["candidate"]) - y)
    base_err = np.abs(_as_float_array(what_pred[base_key]) - y)
    advantage = base_err - cand_err
    groups = np.asarray(group_ids)
    pooled = float(advantage.mean())
    unique = np.unique(groups)
    group_means = np.array([advantage[groups == gid].mean() for gid in unique])
    grouped = float(group_means.mean())
    n_units = int(unique.size)
    passed = grouped >= sesoi and n_units >= min_units
    return _clause_result(
        "group_disjoint_validity",
        passed,
        {
            "baseline_used": base_key,
            "pooled_advantage": pooled,
            "grouped_advantage": grouped,
            "n_units": n_units,
            "min_units": min_units,
            "sesoi": sesoi,
        },
    )


def clause_design_adequacy(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """6) SESOI, power, multiplicity, and a stop rule must all be declared and adequate."""

    constants = prereg["constants"]
    power_min = float(constants["power_min"])
    min_units = int(constants["min_independent_units"])
    design = inputs.get("design") or {}
    group_ids = inputs.get("group_ids")
    n_units = int(np.unique(np.asarray(group_ids)).size) if group_ids is not None else 0
    sesoi = design.get("sesoi")
    power = design.get("power")
    multiplicity = design.get("multiplicity_correction")
    stop_rule = design.get("stop_rule")
    checks = {
        "sesoi_declared_positive": isinstance(sesoi, (int, float)) and sesoi > 0,
        "power_adequate": isinstance(power, (int, float)) and power >= power_min,
        "multiplicity_declared": bool(multiplicity),
        "stop_rule_declared": bool(stop_rule),
        "units_adequate": n_units >= min_units,
    }
    return _clause_result(
        "design_adequacy",
        all(checks.values()),
        {"checks": checks, "n_units": n_units, "min_units": min_units, "power_min": power_min},
    )


def clause_control_behavior(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """7) Noisy-TV rejected, shuffled-target and wrong-time at chance, rate-matched-random primary."""

    constants = prereg["constants"]
    tol = float(constants["noisy_tv_tolerance"])
    band = float(constants["chance_band"])
    primary_required = constants["primary_control"]
    firing = _scalar_or_mean(inputs, "noisy_tv_firing_rate", "noisy_tv_scores")
    base_rate = inputs.get("noisy_tv_base_rate")
    shuffled = inputs.get("shuffled_target_score")
    wrong_time = inputs.get("wrong_time_score")
    chance = inputs.get("chance_level")
    primary = inputs.get("primary_control")
    if (
        firing is None
        or base_rate is None
        or shuffled is None
        or wrong_time is None
        or chance is None
        or primary is None
    ):
        return _clause_result(
            "control_behavior", False, {"reason": "missing one or more control-behavior summaries"}
        )
    checks = {
        "noisy_tv_at_chance": firing <= float(base_rate) + tol,
        "shuffled_target_at_chance": abs(float(shuffled) - float(chance)) <= band,
        "wrong_time_at_chance": abs(float(wrong_time) - float(chance)) <= band,
        "primary_control_is_rate_matched_random": primary == primary_required,
    }
    return _clause_result(
        "control_behavior",
        all(checks.values()),
        {
            "checks": checks,
            "noisy_tv_firing_rate": firing,
            "noisy_tv_base_rate": float(base_rate),
            "tolerance": tol,
            "chance_band": band,
        },
    )


def clause_architecture_independence(inputs: Mapping[str, Any], prereg: Mapping[str, Any]) -> dict[str, Any]:
    """8) A favorable result must reproduce across at least two gate architectures."""

    min_arch = int(prereg["constants"]["min_gate_architectures"])
    favorable = inputs.get("architecture_favorable")
    if favorable is None:
        return _clause_result(
            "architecture_independence", False, {"reason": "missing architecture_favorable"}
        )
    flags = [bool(x) for x in favorable]
    n_arch = len(flags)
    n_favorable = int(sum(flags))
    passed = n_arch >= min_arch and n_favorable >= min_arch
    return _clause_result(
        "architecture_independence",
        passed,
        {"n_architectures": n_arch, "n_favorable": n_favorable, "min_architectures": min_arch},
    )


CLAUSE_FUNCTIONS = (
    clause_what_absolute_sufficiency,
    clause_oracle_budget_headroom,
    clause_when_decodability,
    clause_incremental_value,
    clause_group_disjoint_validity,
    clause_design_adequacy,
    clause_control_behavior,
    clause_architecture_independence,
)


def run_battery(inputs: Mapping[str, Any], prereg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run all eight clauses. ``admitted`` is true only when every clause passes."""

    prereg = prereg or MECHANISM_ADMISSION_PREREG
    clauses: dict[str, Any] = {}
    for function in CLAUSE_FUNCTIONS:
        result = function(inputs, prereg)
        clauses[result["clause"]] = {"passed": result["passed"], "evidence": result["evidence"]}
    n_passed = sum(1 for entry in clauses.values() if entry["passed"])
    seal = prereg.get("seal") or {}
    return {
        "schema": "mop-mechanism-admission-battery/v1",
        "admitted": n_passed == len(CLAUSE_FUNCTIONS),
        "n_passed": n_passed,
        "n_clauses": len(CLAUSE_FUNCTIONS),
        "prereg_sha256": seal.get("sha256"),
        "clauses": clauses,
        "activation_allowed": False,
        "scientific_promotion": False,
        "independent_scientific_confirmation": False,
    }


__all__ = [
    "clause_what_absolute_sufficiency",
    "clause_oracle_budget_headroom",
    "clause_when_decodability",
    "clause_incremental_value",
    "clause_group_disjoint_validity",
    "clause_design_adequacy",
    "clause_control_behavior",
    "clause_architecture_independence",
    "CLAUSE_FUNCTIONS",
    "run_battery",
]
