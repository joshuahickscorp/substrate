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


def plateau(curve, patience: int = 3, rel_tol: float = 0.005, abs_tol: float = 0.01) -> dict:
    """Has the validation curve stopped rising.

    Two criteria, both reported, because they disagree in a way that matters. The patience criterion is the
    strict one: the best value must be followed by patience checks with no improvement above rel_tol. On a
    noisy single seed curve its argmax lands late by chance, so a curve that is visibly flat reads as still
    improving. That produced a false unconverged verdict in this program and is recorded as defect D17.

    The plateau criterion asks the question the strict one is a proxy for: is there remaining training
    headroom. It requires the tail of the curve to sit within abs_tol of the best value and the slope across
    the second half to be non positive beyond rel_tol. A genuinely rising curve fails both.

    converged is true when either holds, and criterion_used names which, so nothing is swapped silently.
    """
    c = [float(x) for x in curve]
    if len(c) < patience + 1:
        return {"converged": False, "converged_strict": False, "converged_plateau": False,
                "criterion_used": "none", "reason": f"only {len(c)} validation points, needs {patience + 1}"}
    best, best_i = c[0], 0
    for i, v in enumerate(c):
        if v > best * (1.0 + rel_tol):
            best, best_i = v, i
    tail = len(c) - 1 - best_i
    strict = tail >= patience

    third = max(1, len(c) // 3)
    tail_mean = sum(c[-third:]) / third
    half = c[len(c) // 2 :]
    slope = (half[-1] - half[0]) / max(1, len(half) - 1)
    plateaued = (max(c) - tail_mean) <= abs_tol and slope <= rel_tol

    converged = strict or plateaued
    return {
        "converged": converged,
        "converged_strict": strict,
        "converged_plateau": plateaued,
        "criterion_used": "patience" if strict else ("plateau" if plateaued else "none"),
        "reason": "" if converged else (
            f"still rising: best at check {best_i} of {len(c) - 1}, tail mean {round(tail_mean, 4)} is "
            f"{round(max(c) - tail_mean, 4)} below the best and the second half slope is {round(slope, 4)}"
        ),
        "best_value": best,
        "best_index": best_i,
        "checks_after_best": tail,
        "tail_mean": round(tail_mean, 5),
        "second_half_slope": round(slope, 5),
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
        "plateau_criterion": (
            "either no improvement above 0.5 percent for 3 validation checks, or a tail within 1 point of "
            "the best value with a non positive second half slope. Both are reported"
        ),
        "converged_strict": p["converged_strict"],
        "converged_plateau": p["converged_plateau"],
        "criterion_used": p["criterion_used"],
        "tail_mean": p.get("tail_mean"),
        "second_half_slope": p.get("second_half_slope"),
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
