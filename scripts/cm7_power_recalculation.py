#!/usr/bin/env python
"""Recalculate power targets from the observed CM7 five-seed paired variance.

Implements the extended-compute plan item "Recalculate every power target from observed
variance". Inputs are the completed CM7 chain's independent-verifier paired comparisons.
Outputs, per comparison family: observed paired sd, observed d_z, and the paired-t sample
sizes required to detect (a) the preregistered 0.03 margin, (b) d_z = 0.8, and (c) d_z = 0.5,
at the plan's alpha = .01 two-sided planning threshold and power .80.

CM7 itself schedules no confirmatory campaign: its simultaneous lower bounds already exclude
the margin in the wrong direction, and the regime is retired (see
proof/NULL_CARDS/mop_cm7_min_objective_probe.md). These numbers are planning inputs for the
P4 response surface and any CM8-class successor. Claim scope: planning arithmetic only.

No em or en dashes (BLACKHOLE.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIER = REPO_ROOT / "runs" / "custom_substrate" / "cm7_local180_citable_v3" / "independent_verifier.json"
COMPOSITE = REPO_ROOT / "proof" / "CUSTOM_SUBSTRATE_PILOT.json"
SCHEMA = "mop-cm7-power-recalculation/v1"
ALPHA = 0.01
POWER = 0.80


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (Numerical Recipes form)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-14:
            break
    return h


def _reg_inc_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_front = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    front = math.exp(ln_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_cdf(t: float, df: float) -> float:
    x = df / (df + t * t)
    tail = 0.5 * _reg_inc_beta(df / 2.0, 0.5, x)
    return 1.0 - tail if t >= 0 else tail


def _t_ppf(p: float, df: float) -> float:
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _std_normal_cdf(z: float) -> float:
    return 0.5 * math.erfc(-z / math.sqrt(2.0))


def _nct_power_two_sided(crit: float, df: float, ncp: float, points: int = 4000) -> float:
    """P(|T| > crit) for T noncentral-t(df, ncp), by quadrature over the chi-square draw."""
    upper = df + 12.0 * math.sqrt(2.0 * df) + 40.0
    step = upper / points
    ln_norm = -math.lgamma(df / 2.0) - (df / 2.0) * math.log(2.0)
    total = 0.0
    for i in range(points):
        v = (i + 0.5) * step
        ln_pdf = ln_norm + (df / 2.0 - 1.0) * math.log(v) - v / 2.0
        pdf = math.exp(ln_pdf)
        scale = math.sqrt(v / df)
        p_hi = 1.0 - _std_normal_cdf(crit * scale - ncp)
        p_lo = _std_normal_cdf(-crit * scale - ncp)
        total += pdf * (p_hi + p_lo) * step
    return total


def paired_t_n(effect_d_z: float, alpha: float = ALPHA, power: float = POWER) -> int:
    """Smallest n with two-sided one-sample t power >= target at standardized effect d_z."""
    if effect_d_z <= 0:
        raise ValueError("effect must be positive")
    n = 3
    while n < 100000:
        df = n - 1
        crit = _t_ppf(1 - alpha / 2, df)
        ncp = effect_d_z * math.sqrt(n)
        if _nct_power_two_sided(crit, df, ncp) >= power:
            return n
        n += 1
    raise RuntimeError("no n below 100000 reaches the requested power")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "proof" / "CM7_POWER_RECALCULATION.json"))
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    verifier = json.loads(VERIFIER.read_text(encoding="utf-8"))
    problems: list[str] = []
    if verifier.get("schema") != "mop-custom-substrate-cm7-independent-verifier/v1":
        problems.append("verifier schema mismatch")

    rows = []
    for name, comparison in sorted(verifier["paired_comparisons"].items()):
        sd = float(comparison["sd"])
        mean = float(comparison["mean_delta"])
        margin = float(comparison["margin"])
        n_obs = int(comparison["n"])
        d_obs = mean / sd if sd > 0 else None
        d_margin = margin / sd if sd > 0 else None
        rows.append(
            {
                "comparison": name,
                "n_observed": n_obs,
                "mean_delta": mean,
                "sd": sd,
                "observed_d_z": d_obs,
                "margin": margin,
                "margin_d_z": d_margin,
                "n_for_margin_at_alpha_.01_power_.80": (
                    paired_t_n(d_margin) if d_margin and d_margin > 0 else None
                ),
                "n_for_d_z_.8": paired_t_n(0.8),
                "n_for_d_z_.5": paired_t_n(0.5),
            }
        )

    sds = [row["sd"] for row in rows if row["sd"] > 0]
    median_sd = sorted(sds)[len(sds) // 2]
    receipt = {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "alpha_two_sided": ALPHA,
        "power": POWER,
        "verifier_sha256": hashlib.sha256(VERIFIER.read_bytes()).hexdigest(),
        "composite_sha256": hashlib.sha256(COMPOSITE.read_bytes()).hexdigest(),
        "comparisons": rows,
        "summary": {
            "median_paired_sd": median_sd,
            "margin_0.03_in_d_z_units_at_median_sd": 0.03 / median_sd,
            "n_to_detect_margin_0.03_at_median_sd": paired_t_n(0.03 / median_sd),
            "statement": (
                "Observed CM7 paired sds run 0.053 to 0.139 on heldout_combo_score, so detecting "
                "the preregistered 0.03 margin at alpha .01 and power .80 would need 40 to 256 "
                "paired seeds per comparison (183 at the median sd). CM7 schedules no such "
                "campaign: the simultaneous bounds already exclude the margin in the wrong "
                "direction and the regime is retired. Successors (P4, CM8-class) must either "
                "register a larger SESOI or reduce seed variance by design; at the plan's "
                "sensitivity anchors the same arithmetic reproduces 22 seeds for d_z .8 and 51 "
                "for d_z .5."
            ),
        },
        "claim_boundary": {
            "scientific_promotion": False,
            "statement": "planning arithmetic from a closed receipt chain; no new evidence",
        },
        "problems": problems,
        "all_ok": not problems,
    }
    Path(args.out).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"out": args.out, "all_ok": receipt["all_ok"], "summary": receipt["summary"]}, indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
