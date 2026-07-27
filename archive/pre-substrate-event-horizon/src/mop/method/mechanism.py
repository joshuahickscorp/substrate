"""Mechanism activity proof.

A mechanism that is instantiated, named and configured but never acts is not a scientific null. It is
inactive instrumentation. The distinction matters because the two look identical in a results table and
opposite in what they license: a null closes a hypothesis, inactive instrumentation closes nothing.

The proof is a comparison of five conditions on the same bed and seed.

House style: no dashes.
"""

from __future__ import annotations

CONDITIONS = ("enabled", "disabled", "forced_active", "randomized", "shuffled")

REQUIRED_MEASUREMENTS = (
    "intervention_count",
    "intervention_timing",
    "affected_samples",
    "affected_parameter_groups",
    "affected_state",
    "affected_memory",
    "counterfactual_difference",
    "downstream_path",
    "cost",
)


def activity(measurements: dict, min_counterfactual: float = 1e-6) -> dict:
    """measurements maps condition name to the required measurement dict.

    Returns a receipt and a classification. active means the mechanism measurably changed the world in a way
    that reached the outcome; inactive_instrumentation means it did not.
    """
    checks: dict = {}
    missing = [
        f"{c}.{m}"
        for c in ("enabled", "disabled")
        for m in REQUIRED_MEASUREMENTS
        if m not in measurements.get(c, {})
    ]
    if missing:
        return {
            "classification": "inactive_instrumentation",
            "reason": f"missing measurements: {missing[:6]}",
            "checks": {"measurements_complete": False},
            "active": False,
        }
    on, off = measurements["enabled"], measurements["disabled"]
    checks["intervened_at_all"] = int(on["intervention_count"]) > 0
    checks["disabled_does_not_intervene"] = int(off["intervention_count"]) == 0
    checks["touched_samples"] = int(on["affected_samples"]) > 0
    checks["touched_parameters_or_state"] = bool(
        on["affected_parameter_groups"] or on["affected_state"] or on["affected_memory"]
    )
    checks["counterfactual_difference"] = float(on["counterfactual_difference"]) > min_counterfactual
    checks["reaches_outcome"] = bool(on["downstream_path"])
    checks["costs_something"] = float(on["cost"]) > 0
    if "forced_active" in measurements:
        checks["forcing_increases_intervention"] = int(measurements["forced_active"]["intervention_count"]) >= int(
            on["intervention_count"]
        )
    if "randomized" in measurements and "shuffled" in measurements:
        # the real signal must carry information that a rate matched permutation of it does not
        checks["signal_beats_permuted_signal"] = float(on["counterfactual_difference"]) != float(
            measurements["shuffled"]["counterfactual_difference"]
        )
    checks["all_pass"] = all(v for k, v in checks.items() if isinstance(v, bool) and k != "all_pass")
    active = checks["all_pass"]
    return {
        "classification": "active" if active else "inactive_instrumentation",
        "active": active,
        "checks": checks,
        "failed": [k for k, v in checks.items() if v is False and k != "all_pass"],
    }
