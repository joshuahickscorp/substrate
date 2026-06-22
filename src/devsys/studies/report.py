"""Frontier 11: reporting scaffolds. Pure functions that turn plain run-output dicts/lists
(seed-sample lists, frontier points, null-registry entries) into the statistics and markdown a
results writeup needs, with NO run and NO scipy. Everything here is deterministic given its
inputs, so the same numbers can be regenerated from a saved summary.json or from dummy data.

Two statistical primitives carry the headline: Cohen's d (effect size, the standardized gap
between two seed samples) and a t-interval mean_ci (where the magnitude actually lands, with
uncertainty that widens as variance grows or n shrinks). seed_summary wraps a single sample into
{mean,std,sem,ci}. frontier_table / null_registry_md / render_report are the markdown renderers;
frontier_table delegates the Pareto math to devsys.metrics.frontier so the report and the gate
agree on who wins.
"""

from __future__ import annotations

import math

from ..metrics.frontier import FrontierPoint, frontier_auc, pareto_front

# Two-sided t critical values at 95% confidence, indexed by degrees of freedom (n-1). Stops at
# df=30 where t is already ~2.04; beyond that we use the normal 1.96, which is the t limit. This
# is the scipy-free substitute the spec asks for (a small t-table + a normal fallback).
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}
_Z95 = 1.96  # normal two-sided 95%; also the t limit as df -> infinity


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _sample_std(xs: list[float]) -> float:
    """Sample standard deviation (Bessel's n-1). Zero for a single value: no spread is
    measurable, which is exactly why one seed can support neither an effect nor an interval."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _t_crit(conf: float, df: int) -> float:
    """Two-sided t critical value. We only table 95%; any other confidence falls back to the
    normal z for that level (good enough for a scaffold, and honestly labelled in mean_ci)."""
    if df < 1:
        return _Z95
    if abs(conf - 0.95) < 1e-9:
        return _T95.get(df, _Z95)
    return _z_crit(conf)


def _z_crit(conf: float) -> float:
    """Normal two-sided critical value for a few common confidences (no scipy)."""
    table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
    best = min(table, key=lambda c: abs(c - conf))
    return table[best]


def cohens_d(a: list[float], b: list[float]) -> float:
    """Effect size between two seed samples a and b: (mean_a - mean_b) / pooled_std.

    Sign follows a - b (positive => a runs higher). Magnitude is in pooled-std units, so the
    usual rough reading is |d| ~ 0.2 small, ~0.5 medium, ~0.8 large. Pooled std uses the n-2
    denominator; if both samples have zero spread the effect is +-inf when the means differ
    (a perfectly separated, infinitely reliable gap) and 0.0 when they coincide.
    """
    if not a or not b:
        raise ValueError("cohens_d needs two non-empty samples")
    na, nb = len(a), len(b)
    ma, mb = _mean(a), _mean(b)
    diff = ma - mb
    if na + nb - 2 <= 0:  # one value each: no within-sample spread to standardize by
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    pooled = math.sqrt((va + vb) / (na + nb - 2))
    if pooled == 0.0:
        return 0.0 if diff == 0 else math.copysign(math.inf, diff)
    return diff / pooled


def mean_ci(samples: list[float], conf: float = 0.95) -> dict:
    """t-interval for the mean of one seed sample.

    Returns {mean, lo, hi, sem, n}. The half-width is t(df=n-1) * sem, so the interval WIDENS
    with sample variance and NARROWS as n grows. A single value has sem=0 and a degenerate
    (zero-width) interval at its own value: one seed cannot bound a mean, and we say so by not
    pretending to.
    """
    if not samples:
        raise ValueError("mean_ci needs a non-empty sample")
    n = len(samples)
    m = _mean(samples)
    sd = _sample_std(samples)
    sem = sd / math.sqrt(n) if n >= 1 else 0.0
    half = _t_crit(conf, n - 1) * sem
    return {"mean": m, "lo": m - half, "hi": m + half, "sem": sem, "n": n}


def seed_summary(values: list[float]) -> dict:
    """Summarize one across-seed sample: {mean, std, sem, n, ci} where ci is the 95% mean_ci.
    The single place a results table should go for "value +- spread" over seeds."""
    if not values:
        raise ValueError("seed_summary needs a non-empty sample")
    sd = _sample_std(values)
    n = len(values)
    return {
        "mean": _mean(values),
        "std": sd,
        "sem": sd / math.sqrt(n),
        "n": n,
        "ci": mean_ci(values, 0.95),
    }


def frontier_table(points: list[dict]) -> str:
    """Render an adaptation-retention frontier as markdown, marking the Pareto winners.

    Each point is {name, adaptation, retention}. Pareto membership and the frontier AUC are
    computed by devsys.metrics.frontier (the program's central metric) so this table and the
    experiment gate never disagree about who is on the front. A leading * flags a Pareto point.
    """
    pts = [FrontierPoint(p["name"], float(p["adaptation"]), float(p["retention"])) for p in points]
    rows = [
        "| Pareto | Method | Adaptation | Retention |",
        "|---|---|---|---|",
    ]
    if not pts:
        rows.append("| | (no points) | | |")
        return "\n".join(rows) + "\n\nFrontier AUC: 0.0000\n"
    front_names = {id(p) for p in pareto_front(pts)}
    # rank by retention then adaptation so the strongest survivors read top-down
    for p in sorted(pts, key=lambda q: (q.retention, q.adaptation), reverse=True):
        star = "*" if id(p) in front_names else ""
        rows.append(f"| {star} | {p.name} | {p.adaptation:.4f} | {p.retention:.4f} |")
    auc = frontier_auc(pts)
    return "\n".join(rows) + f"\n\nFrontier AUC: {auc:.4f}\n"


def null_registry_md(entries: list[dict]) -> str:
    """Render predicted-null verdicts as a markdown table + a verdict tally.

    Each entry: {experiment, verdict, null_hypothesis, taxonomy_category?, taxonomy_label?,
    observed?}. Verdicts are counted (confirmed/refuted/mixed/error) into a one-line summary so a
    reader sees at a glance how many declared nulls actually held.
    """
    tally: dict[str, int] = {"confirmed": 0, "refuted": 0, "mixed": 0, "error": 0}
    rows = [
        "| Experiment | Verdict | Taxonomy | Null hypothesis |",
        "|---|---|---|---|",
    ]
    for e in entries:
        verdict = str(e.get("verdict", "error"))
        tally[verdict] = tally.get(verdict, 0) + 1
        cat = e.get("taxonomy_category")
        label = e.get("taxonomy_label") or ""
        taxonomy = "" if cat is None else f"{cat} {label}".strip()
        null = str(e.get("null_hypothesis", "")).replace("\n", " ")
        rows.append(f"| {e.get('experiment', '?')} | {verdict} | {taxonomy} | {null} |")
    summary = " ".join(f"{k}={v}" for k, v in tally.items())
    head = f"Verdicts: {summary}\n\n"
    return head + "\n".join(rows) + "\n"


def render_report(sections: dict) -> str:
    """Assemble a markdown report from an ordered {heading: body} mapping.

    The first key is treated as the title (rendered as an H1); every remaining key becomes an H2
    with its body underneath. Bodies may be any pre-rendered markdown (tables, prose). Dict order
    is preserved, so callers control section order by insertion order.
    """
    if not sections:
        return "# Report\n\n(empty)\n"
    items = list(sections.items())
    title, title_body = items[0]
    out = [f"# {title}", ""]
    if str(title_body).strip():
        out += [str(title_body).strip(), ""]
    for heading, body in items[1:]:
        out += [f"## {heading}", "", str(body).strip(), ""]
    return "\n".join(out).rstrip() + "\n"
