from __future__ import annotations
import math
NULL_REFERENCES = (
    "zero",
    "chance",
    "majority_class",
    "permutation_expectation",
    "matched_noise",
    "empirical_null",
    "analytic_value",
)
HISTORY_KINDS = (
    "current_observation_only",
    "last_k_observations",
    "pooled_history",
    "state_carried_from_previous_observations",
    "segment_identity",
    "position",
    "timestamp",
    "future_information",
)
FORBIDDEN_HISTORY = ("future_information",)
def boundary_crossing(*, context_a: dict, context_b: dict, transition_index: int, adapt_window: tuple,
                      baseline_on_a: float, baseline_on_b: float, construct: str,
                      min_gap: float = 0.02) -> dict:
    checks = {
        "context_a_exists": bool(context_a) and int(context_a.get("n", 0)) > 0,
        "context_b_exists": bool(context_b) and int(context_b.get("n", 0)) > 0,
        "contexts_differ_under_the_construct": bool(context_a.get(construct) != context_b.get(construct)),
        "baseline_behaviour_changes_across_the_transition": abs(baseline_on_a - baseline_on_b) >= min_gap,
        "adaptation_window_contains_the_transition": (
            adapt_window[0] <= transition_index <= adapt_window[1]
        ),
    }
    checks["all_pass"] = all(checks.values())
    return {
        "construct": construct,
        "construct_value_a": context_a.get(construct),
        "construct_value_b": context_b.get(construct),
        "baseline_gap": round(float(baseline_on_a - baseline_on_b), 5),
        "transition_index": transition_index,
        "adaptation_window": list(adapt_window),
        "checks": checks,
        "classification": "boundary_crossed" if checks["all_pass"] else "invalid_no_boundary_crossing",
    }
def reset_alignment(reset_indices, segment_boundaries, sequence_length: int, tolerance: int = 1) -> dict:
    r = sorted(int(x) for x in reset_indices)
    b = sorted(int(x) for x in segment_boundaries)
    if not r:
        return {"n_resets": 0, "exact_fraction": 0.0, "within_tolerance_fraction": 0.0,
                "expected_random_alignment": 0.0, "checks": {"is_misaligned": True, "all_pass": True},
                "classification": "no_reset"}
    dists = [min((abs(x - y) for y in b), default=sequence_length) for x in r]
    exact = sum(1 for d in dists if d == 0) / len(r)
    near = sum(1 for d in dists if d <= tolerance) / len(r)
    expected = min(1.0, len(b) * (2 * tolerance + 1) / max(1, sequence_length))
    checks = {
        "is_misaligned": exact <= expected + 1e-9,
        "not_wholly_aligned": exact < 1.0,
        "boundaries_declared": bool(b),
    }
    checks["all_pass"] = all(checks.values())
    return {
        "n_resets": len(r),
        "reset_indices": r[:32],
        "segment_boundaries": b[:32],
        "distance_to_nearest_boundary": dists[:32],
        "exact_fraction": round(exact, 5),
        "within_tolerance_fraction": round(near, 5),
        "expected_random_alignment": round(expected, 5),
        "tolerance": tolerance,
        "checks": checks,
        "classification": "misaligned" if checks["all_pass"] else "oracle_segmented",
    }
def coprime_periods(sequence_length: int, segment_length: int, count: int = 2, lo: int = 5,
                    hi: int = 40) -> list[int]:
    out = []
    for p in range(lo, hi + 1):
        if math.gcd(p, max(1, segment_length)) == 1 and p < sequence_length:
            out.append(p)
        if len(out) >= count * 4:
            break
    picked, seen = [], set()
    for p in out:
        if all(math.gcd(p, q) == 1 for q in picked) and p not in seen:
            picked.append(p)
            seen.add(p)
        if len(picked) >= count:
            break
    return picked
def reset_indices_for(kind: str, sequence_length: int, segment_length: int, period: int = 0,
                      rate: float = 0.0, rng=None) -> list[int]:
    n = sequence_length
    if kind == "none":
        return []
    if kind == "every_observation":
        return list(range(1, n))
    if kind == "fixed_period":
        return list(range(period, n, period))
    if kind == "true_boundary":
        return list(range(segment_length, n, segment_length))
    if kind == "wrong_boundary":
        off = max(1, segment_length // 2)
        return [min(n - 1, x + off) for x in range(segment_length, n, segment_length)]
    if kind == "random_rate_matched":
        k = max(1, int(round(rate * n)))
        idx = rng.choice(range(1, n), size=min(k, n - 1), replace=False)
        return sorted(int(x) for x in idx)
    if kind == "block_shuffled":
        blocks = list(range(0, n, max(1, period or segment_length)))
        perm = rng.permutation(len(blocks))
        return sorted(int(blocks[i]) for i in perm[: max(1, len(blocks) // 2)] if blocks[i] > 0)
    raise ValueError(f"unknown reset kind {kind!r}")
def plateau_validity(curves: dict, *, threshold: float = 0.01, slope_threshold: float = 0.002) -> dict:
    if len(curves) < 4:
        return {"converged": False, "reason": f"{len(curves)} budgets, needs at least 4",
                "checks": {"enough_budgets": False}, "classification": "insufficient_budget_grid"}
    budgets = sorted(int(b) for b in curves)
    vals = [float(curves[b] if b in curves else curves[str(b)]) for b in budgets]
    half = vals[len(vals) // 2 :]
    movement = max(half) - min(half)
    slope = (half[-1] - half[0]) / max(1, len(half) - 1)
    best_i = max(range(len(vals)), key=lambda i: vals[i])
    checks = {
        "enough_budgets": True,
        "largest_budget_is_not_the_best": best_i != len(vals) - 1 or (vals[-1] - max(vals[:-1])) <= threshold,
        "second_half_movement_within_threshold": movement <= threshold,
        "second_half_slope_within_threshold": slope <= slope_threshold,
        "not_overfitting": (max(vals) - vals[-1]) <= threshold * 3,
    }
    checks["all_pass"] = all(checks.values())
    return {
        "budgets": budgets,
        "scores": [round(v, 5) for v in vals],
        "best_budget": budgets[best_i],
        "selected_checkpoint": budgets[best_i],
        "second_half_movement": round(movement, 5),
        "residual_slope": round(slope, 6),
        "threshold": threshold,
        "slope_threshold": slope_threshold,
        "checks": checks,
        "converged": checks["all_pass"],
        "reason": "" if checks["all_pass"] else (
            f"second half moves {round(movement, 4)} with slope {round(slope, 5)}; "
            f"extend the budget grid past {budgets[-1]}"
        ),
        "classification": "converged" if checks["all_pass"] else "unconverged",
    }
def null_reference(kind: str, *, observed: float, reference: float, band: float = 0.05,
                   detail: dict | None = None) -> dict:
    if kind not in NULL_REFERENCES:
        return {"kind": kind, "valid": False, "reason": f"unknown null reference {kind!r}",
                "classification": "invalid_null_reference"}
    delta = float(observed) - float(reference)
    inside = abs(delta) <= band
    return {
        "kind": kind,
        "observed": round(float(observed), 5),
        "reference": round(float(reference), 5),
        "delta": round(delta, 5),
        "band": band,
        "valid": True,
        "behaves_as_a_null": inside,
        "detail": detail or {},
        "classification": "null_as_expected" if inside else "null_reference_violated",
    }
def history_witness(arms: dict) -> dict:
    rows, violations = {}, []
    for name, h in arms.items():
        kinds = set(h.get("kinds") or [])
        unknown = kinds - set(HISTORY_KINDS)
        if unknown:
            violations.append(f"{name}: unknown history kinds {sorted(unknown)}")
        if kinds & set(FORBIDDEN_HISTORY):
            violations.append(f"{name}: sees future information")
        if "last_k_observations" in kinds and not h.get("k"):
            violations.append(f"{name}: declares last k observations without a k")
        rows[name] = {
            "kinds": sorted(kinds),
            "k": h.get("k"),
            "effective_horizon": h.get("effective_horizon"),
            "sees_future": bool(kinds & set(FORBIDDEN_HISTORY)),
        }
    return {
        "arms": rows,
        "violations": violations,
        "all_declared": not violations,
        "distinct_history_profiles": len({tuple(sorted(v["kinds"])) + (v["k"],) for v in rows.values()}),
    }
def matched_information(a: dict, b: dict) -> dict:
    ha, hb = a.get("effective_horizon"), b.get("effective_horizon")
    return {
        "a_horizon": ha,
        "b_horizon": hb,
        "matched": ha is not None and hb is not None and (
            ha == hb or (isinstance(ha, str) and isinstance(hb, str) and ha == hb)
        ),
        "note": "an unmatched comparison measures information, not architecture",
    }
