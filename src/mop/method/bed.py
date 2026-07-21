"""Data bed validity.

A dataset being real does not make the task valid. A task being temporal does not make temporal state
necessary. A task being sequentialized does not make it continual. Three sentences that each cost this
program a campaign.

The classifier is deliberately blunt: a bed that fails construct validity or has no headroom is invalid, and
a mechanism null measured on an invalid bed is not a mechanism null.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np

CLASSIFICATIONS = (
    "valid_principal_bed",
    "valid_secondary_bed",
    "invalid_no_construct",
    "invalid_no_headroom",
    "invalid_no_independent_units",
    "invalid_no_intervention",
    "invalid_no_temporal_requirement",
    "invalid_unconverged_baseline",
    "invalid_instrumentation",
)


def unit_audit(train_units, tune_units, test_units, *, test_touched: bool = False) -> dict:
    """Group disjointness is arithmetic over the unit arrays, so it is measured, never assumed."""
    tr, tu, te = set(np.asarray(train_units).tolist()), set(np.asarray(tune_units).tolist()), set(
        np.asarray(test_units).tolist()
    )
    overlaps = {
        "train_tune": sorted(tr & tu),
        "train_test": sorted(tr & te),
        "tune_test": sorted(tu & te),
    }
    return {
        "n_units": len(tr | tu | te),
        "n_train_units": len(tr),
        "n_tune_units": len(tu),
        "n_test_units": len(te),
        "overlaps": overlaps,
        "group_disjoint": not any(overlaps.values()),
        "test_touched": bool(test_touched),
    }


def leakage_audit(feature_fit_units, label_fit_units, normalization_scope: str) -> dict:
    """Any statistic fitted across the split boundary is leakage, whatever it is called."""
    cross = set(np.asarray(feature_fit_units).tolist()) & set(np.asarray(label_fit_units).tolist())
    return {
        "normalization_scope": normalization_scope,
        "normalization_is_train_only": normalization_scope in ("train", "train_only", "per_unit_train"),
        "no_shared_fit_units": not cross,
        "clean": normalization_scope in ("train", "train_only", "per_unit_train") and not cross,
    }


def classify(m: dict) -> dict:
    """m carries measured quantities. Every one of them must be present; absence is invalid, not valid."""
    need = (
        "construct_valid",
        "units",
        "leakage",
        "oracle_headroom",
        "residual_headroom_lcb",
        "baseline_converged",
        "order_necessity",
        "intervention_possible",
        "seed_stability",
    )
    missing = [k for k in need if k not in m]
    if missing:
        return {"classification": "invalid_instrumentation", "reason": f"unmeasured: {missing}", "checks": {}}

    checks = {
        "construct_valid": bool(m["construct_valid"]),
        "group_disjoint": bool(m["units"].get("group_disjoint")),
        "test_untouched": not m["units"].get("test_touched"),
        "enough_units": int(m["units"].get("n_units", 0)) >= 2,
        "no_leakage": bool(m["leakage"].get("clean")),
        "oracle_headroom_positive": float(m["oracle_headroom"]) > 0,
        "residual_headroom_positive": float(m["residual_headroom_lcb"]) > 0,
        "baseline_converged": bool(m["baseline_converged"]),
        "intervention_possible": bool(m["intervention_possible"]),
        "seed_stable": float(m["seed_stability"]) <= float(m.get("seed_stability_bound", 0.05)),
        "order_required": float(m["order_necessity"]) > float(m.get("order_necessity_bound", 0.05)),
    }

    if not checks["construct_valid"]:
        cls = "invalid_no_construct"
    elif not (checks["group_disjoint"] and checks["enough_units"] and checks["test_untouched"]):
        cls = "invalid_no_independent_units"
    elif not checks["no_leakage"]:
        cls = "invalid_instrumentation"
    elif not checks["baseline_converged"]:
        cls = "invalid_unconverged_baseline"
    elif not checks["oracle_headroom_positive"] or not checks["residual_headroom_positive"]:
        cls = "invalid_no_headroom"
    elif not checks["intervention_possible"]:
        cls = "invalid_no_intervention"
    elif not checks["order_required"]:
        cls = "invalid_no_temporal_requirement"
    elif not checks["seed_stable"]:
        cls = "valid_secondary_bed"
    else:
        cls = "valid_principal_bed"
    return {
        "classification": cls,
        "reason": "" if cls.startswith("valid") else f"failed {[k for k, v in checks.items() if not v]}",
        "checks": checks,
        "measurements": {k: m[k] for k in need if not isinstance(m[k], dict)},
    }


def context_boundary(no_adapt_new: float, no_adapt_old: float, adapted_new: float, adapted_old: float,
                     min_gap: float = 0.02) -> dict:
    """Did the run actually cross a context boundary, or is the second context more of the first.

    Two signatures are required. The pretrained model must be measurably worse on the new context than on
    the old one, and adapting to the new context must cost something on the old one. When adaptation
    improves both, no boundary was crossed and there is no stability plasticity tradeoff to study, whatever
    the split was called. This is the reusable form of the defect that made a within domain continual
    battery not continual.
    """
    shift = float(no_adapt_old) - float(no_adapt_new)
    tradeoff = float(adapted_old) - float(no_adapt_old)
    checks = {
        "new_context_is_measurably_harder": shift >= min_gap,
        "adaptation_costs_the_old_context": tradeoff < 0,
    }
    checks["boundary_crossed"] = all(checks.values())
    return {
        "checks": checks,
        "distribution_shift": round(shift, 5),
        "retention_change_under_adaptation": round(tradeoff, 5),
        "classification": "context_boundary_crossed" if checks["boundary_crossed"] else "invalid_no_context_boundary",
    }


def order_necessity(temporal_score: float, order_free_score: float) -> float:
    """How much of the achievable performance requires order. Zero means an order free reader suffices."""
    return round(float(temporal_score) - float(order_free_score), 5)


def residual_headroom(oracle: float, strongest_control: float) -> float:
    return round(float(oracle) - float(strongest_control), 5)
