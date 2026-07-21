"""Baseline convergence authority and baseline identity binding.

Two historical defects live here. One: an effect was reported against LSTM plus GDumb and computed against a
different baseline, so the sentence and the number disagreed. Two: a comparison against an undertrained or
under budgeted baseline flatters the treatment for reasons that have nothing to do with the mechanism.

A comparison whose baseline lacks a convergence receipt is provisional by construction and may not produce a
terminal verdict.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np


def plateau(curve, patience: int = 3, rel_tol: float = 0.005) -> dict:
    """A validation curve has plateaued when the best value stops improving by rel_tol for patience checks."""
    c = [float(x) for x in curve]
    if len(c) < patience + 1:
        return {"converged": False, "reason": f"only {len(c)} validation points, needs {patience + 1}"}
    best, best_i = c[0], 0
    for i, v in enumerate(c):
        if v > best * (1.0 + rel_tol):
            best, best_i = v, i
    tail = len(c) - 1 - best_i
    return {
        "converged": tail >= patience,
        "reason": "" if tail >= patience else f"still improving, best at check {best_i} of {len(c) - 1}",
        "best_value": best,
        "best_index": best_i,
        "checks_after_best": tail,
    }


def receipt(
    name: str,
    *,
    identity: str,
    model: str,
    parameters: int,
    updates: int,
    data_exposure: int,
    memory: int,
    compute_seconds: float,
    validation_curve,
    selected_checkpoint: str,
    seed_scores,
    group_scores=None,
    treatment_budget: dict | None = None,
) -> dict:
    p = plateau(validation_curve)
    s = np.asarray([float(x) for x in seed_scores], float)
    r = {
        "name": name,
        "identity": identity,
        "model": model,
        "parameters": int(parameters),
        "training_updates": int(updates),
        "data_exposure": int(data_exposure),
        "memory": int(memory),
        "compute_seconds": round(float(compute_seconds), 2),
        "validation_curve": [round(float(x), 4) for x in validation_curve],
        "plateau_criterion": "no improvement above 0.5 percent for 3 validation checks",
        "selected_checkpoint": selected_checkpoint,
        "seed_variance": round(float(s.var(ddof=1)), 6) if len(s) > 1 else 0.0,
        "seed_scores": [round(float(x), 4) for x in s],
        "group_variance": (
            round(float(np.var(np.asarray(group_scores, float), ddof=1)), 6)
            if group_scores is not None and len(group_scores) > 1
            else None
        ),
        "converged": p["converged"],
        "reason": p["reason"],
    }
    if treatment_budget:
        r["resource_matched"] = (
            int(updates) >= int(treatment_budget.get("updates", 0))
            and int(memory) >= int(treatment_budget.get("memory", 0))
            and int(data_exposure) >= int(treatment_budget.get("data_exposure", 0))
        )
        r["treatment_budget"] = treatment_budget
    return r


def comparison(effect_name: str, treatment: str, baseline_receipt: dict, declared_baseline: str) -> dict:
    """Bind an effect to the baseline it was actually computed against, and refuse a mismatch."""
    issues = []
    if baseline_receipt.get("identity") != declared_baseline:
        issues.append(
            f"declared baseline {declared_baseline!r} but the effect was computed against "
            f"{baseline_receipt.get('identity')!r}"
        )
    if not baseline_receipt.get("converged"):
        issues.append(f"baseline not converged: {baseline_receipt.get('reason')}")
    if baseline_receipt.get("resource_matched") is False:
        issues.append("baseline resource budget is smaller than the treatment budget")
    return {
        "effect": effect_name,
        "treatment": treatment,
        "declared_baseline": declared_baseline,
        "actual_baseline": baseline_receipt.get("identity"),
        "issues": issues,
        "status": "terminal" if not issues else "provisional",
        "valid": not issues,
    }
