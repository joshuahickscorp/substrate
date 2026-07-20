from __future__ import annotations

import math

from ..metrics import FrontierPoint, frontier_auc, pareto_front

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
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _t_crit(conf: float, df: int) -> float:
    if df < 1:
        return _Z95
    if abs(conf - 0.95) < 1e-9:
        return _T95.get(df, _Z95)
    return _z_crit(conf)


def _z_crit(conf: float) -> float:
    table = {0.80: 1.282, 0.90: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
    best = min(table, key=lambda c: abs(c - conf))
    return table[best]


def cohens_d(a: list[float], b: list[float]) -> float:
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
    if not samples:
        raise ValueError("mean_ci needs a non-empty sample")
    n = len(samples)
    m = _mean(samples)
    sd = _sample_std(samples)
    sem = sd / math.sqrt(n) if n >= 1 else 0.0
    half = _t_crit(conf, n - 1) * sem
    return {"mean": m, "lo": m - half, "hi": m + half, "sem": sem, "n": n}


def seed_summary(values: list[float]) -> dict:
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
    pts = [FrontierPoint(p["name"], float(p["adaptation"]), float(p["retention"])) for p in points]
    rows = [
        "| Pareto | Method | Adaptation | Retention |",
        "|---|---|---|---|",
    ]
    if not pts:
        rows.append("| | (no points) | | |")
        return "\n".join(rows) + "\n\nFrontier AUC: 0.0000\n"
    front_names = {id(p) for p in pareto_front(pts)}
    for p in sorted(pts, key=lambda q: (q.retention, q.adaptation), reverse=True):
        star = "*" if id(p) in front_names else ""
        rows.append(f"| {star} | {p.name} | {p.adaptation:.4f} | {p.retention:.4f} |")
    auc = frontier_auc(pts)
    return "\n".join(rows) + f"\n\nFrontier AUC: {auc:.4f}\n"


def null_registry_md(entries: list[dict]) -> str:
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
