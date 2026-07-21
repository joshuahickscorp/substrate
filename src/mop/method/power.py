"""Power, sequential design and the decision rule.

The rule that costs the most to follow and saves the most is this one: a tie is a null, and a wrong
direction effect is a failure. Adding seeds after a near miss is only legitimate when the continuation rule
was sealed before the first seed ran.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np

STAGES = ("calibration", "scout", "canary", "principal", "replication")
T95 = {2: 6.314, 3: 2.920, 4: 2.353, 5: 2.132, 6: 2.015, 7: 1.943, 8: 1.895, 9: 1.860, 10: 1.833}


def t95(n: int) -> float:
    return T95.get(n, 1.729 if n > 10 else 6.314)


def lcb(effects) -> float:
    """Random effects lower 95 percent confidence bound over independent units or seeds."""
    e = np.asarray(effects, float)
    n = len(e)
    if n < 2:
        return float(e.mean()) if n else 0.0
    return float(e.mean() - t95(n) * e.std(ddof=1) / np.sqrt(n))


def mde(sd: float, n: int) -> float:
    """Minimum detectable effect at the same one sided 95 percent bar the verdict uses."""
    if n < 2:
        return float("inf")
    return float(t95(n) * sd / np.sqrt(n))


def preregistration(
    *,
    name: str,
    independent_unit: str,
    expected_sd: float,
    sesoi: float,
    seeds: int,
    units: int,
    max_seeds: int,
    futility: float,
    harm: float,
    multiplicity: int = 1,
    continuation_rule: str = "none: the seed count is fixed before the first run",
) -> dict:
    m = mde(expected_sd, seeds)
    return {
        "name": name,
        "independent_unit": independent_unit,
        "expected_sd": round(float(expected_sd), 5),
        "sesoi": float(sesoi),
        "minimum_detectable_effect": round(m, 5),
        "adequately_powered": bool(m <= sesoi),
        "seeds": int(seeds),
        "units": int(units),
        "max_seeds": int(max_seeds),
        "futility_boundary": float(futility),
        "harm_boundary": float(harm),
        "continuation_boundary": continuation_rule,
        "multiplicity": int(multiplicity),
        "bonferroni_sesoi": round(float(sesoi), 5) if multiplicity <= 1 else round(float(sesoi), 5),
        "decision_rule": (
            "positive requires lower_95_cb >= sesoi over independent units; a tie is a null; "
            "an effect in the wrong direction is a failure; seeds may not be added after the fact"
        ),
        "stages": list(STAGES),
    }


def decide(effects, prereg: dict, sesoi: float | None = None) -> dict:
    """Apply the sealed rule to measured per unit effects. No rounding in the caller's favour."""
    e = np.asarray([float(x) for x in effects], float)
    s = float(prereg["sesoi"] if sesoi is None else sesoi)
    lo = lcb(e)
    mean = float(e.mean()) if len(e) else 0.0
    if len(e) < 2:
        return {"verdict": "insufficient_power", "n": len(e), "mean": mean, "lower_95_cb": lo, "sesoi": s}
    if mean <= -abs(float(prereg["harm_boundary"])):
        v = "harm"
    elif lo >= s:
        v = "positive"
    elif mean < 0:
        v = "wrong_direction_failure"
    elif mean <= float(prereg["futility_boundary"]):
        v = "null_futile"
    else:
        v = "null"
    return {
        "verdict": v,
        "n": int(len(e)),
        "mean": round(mean, 5),
        "lower_95_cb": round(lo, 5),
        "sesoi": s,
        "observed_sd": round(float(e.std(ddof=1)), 5),
        "achieved_mde": round(mde(float(e.std(ddof=1)), len(e)), 5),
        "adequately_powered": bool(mde(float(e.std(ddof=1)), len(e)) <= s),
    }


def stage_open(stage: str, previous: dict | None) -> dict:
    """A stage opens only when the previous one passed. Calibration has no predecessor."""
    if stage not in STAGES:
        return {"open": False, "reason": f"unknown stage {stage!r}"}
    i = STAGES.index(stage)
    if i == 0:
        return {"open": True, "reason": "calibration has no predecessor"}
    if previous is None:
        return {"open": False, "reason": f"{STAGES[i - 1]} has not run"}
    ok = bool(previous.get("passed"))
    return {"open": ok, "reason": "" if ok else f"{STAGES[i - 1]} did not pass: {previous.get('reason', '')}"}
