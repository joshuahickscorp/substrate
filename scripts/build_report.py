#!/usr/bin/env python
"""Frontier 11 build script: turn a cpu-campaign summary (or a built-in dummy) into a single
analysis report markdown. Loads runs/cpu_campaign/summary.json if present, else uses a small
self-contained dummy with the same shape, then assembles effect sizes (Cohen's d) + across-seed
summaries + an adaptation-retention frontier table + a negative-result registry section via the
pure renderers in devsys.studies.report. Writes runs/analysis_report.md and prints its path.

Usage: python scripts/build_report.py [--summary runs/cpu_campaign/summary.json] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from devsys.config import REPO_ROOT
from devsys.logging_utils import get_logger
from devsys.studies.report import (
    cohens_d,
    frontier_table,
    null_registry_md,
    render_report,
    seed_summary,
)

log = get_logger("build_report")

# Self-contained stand-in with the real summary.json shape, so the report builds with NO run and
# NO files on disk. Mirrors the keys this script reads: t3 arm bwt dicts, the seed-variance
# per-seed gaps, an e3 frontier, and a couple of negative-registry entries.
DUMMY_SUMMARY: dict = {
    "t1": {
        "seed_variance_11B": {
            "per_seed": [
                {"seed": 0, "naive_bwt": -0.075, "protected_bwt": 0.112, "gap": 0.187},
                {"seed": 1, "naive_bwt": 0.050, "protected_bwt": 0.050, "gap": 0.000},
                {"seed": 2, "naive_bwt": 0.012, "protected_bwt": 0.137, "gap": 0.125},
                {"seed": 3, "naive_bwt": 0.100, "protected_bwt": 0.125, "gap": 0.025},
                {"seed": 4, "naive_bwt": 0.150, "protected_bwt": 0.137, "gap": -0.012},
            ]
        }
    },
    "t3": {
        "results": {
            "dummy_consol": {
                "metrics": {
                    "bwt": {"naive": -0.178, "protected": -0.043},
                    "frontier": {
                        "constant": {"adaptation": 0.958, "retention": 0.821},
                        "decay": {"adaptation": 0.589, "retention": 1.114},
                        "staged": {"adaptation": 0.696, "retention": 0.993},
                        "triggered": {"adaptation": 0.762, "retention": 0.979},
                    },
                }
            }
        }
    },
    "t0": {
        "negative_registry_11D": {
            "entries": [
                {
                    "experiment": "e1_baseline",
                    "verdict": "confirmed",
                    "taxonomy_category": 3,
                    "taxonomy_label": "target not decodable from the frozen latent (no learning)",
                    "null_hypothesis": "naive and protected show no retention gap",
                },
                {
                    "experiment": "e2_replay",
                    "verdict": "refuted",
                    "taxonomy_category": 5,
                    "taxonomy_label": "stream too short for buffer content to matter",
                    "null_hypothesis": "replay ties no-replay",
                },
            ]
        }
    },
}


def load_summary(path: Path) -> tuple[dict, str]:
    """Return (summary, source-label). Falls back to the built-in dummy if the file is absent or
    unparseable, so the report always builds."""
    if path.is_file():
        try:
            return json.loads(path.read_text()), str(path)
        except (json.JSONDecodeError, OSError) as ex:
            log.info("could not read %s (%s); using built-in dummy", path, ex)
    else:
        log.info("no summary at %s; using built-in dummy", path)
    return DUMMY_SUMMARY, "built-in dummy"


def _seed_samples(summary: dict) -> dict[str, list[float]]:
    """Pull across-seed samples out of the seed-variance leg: the per-seed naive/protected BWT
    and their gap. Empty dict if the leg is absent."""
    sv = summary.get("t1", {}).get("seed_variance_11B", {})
    per_seed = sv.get("per_seed", [])
    if not per_seed:
        return {}
    return {
        "naive_bwt": [float(r["naive_bwt"]) for r in per_seed],
        "protected_bwt": [float(r["protected_bwt"]) for r in per_seed],
        "gap": [float(r["gap"]) for r in per_seed],
    }


def _frontier_points(summary: dict) -> list[dict]:
    """Find the first t3 result that carries a named adaptation/retention frontier and flatten it
    into [{name, adaptation, retention}]. Empty list if none present."""
    results = summary.get("t3", {}).get("results", {})
    for res in results.values():
        frontier = res.get("metrics", {}).get("frontier")
        if frontier:
            return [
                {"name": name, "adaptation": float(v["adaptation"]), "retention": float(v["retention"])}
                for name, v in frontier.items()
            ]
    return []


def _registry_entries(summary: dict) -> list[dict]:
    return summary.get("t0", {}).get("negative_registry_11D", {}).get("entries", [])


def _effect_size_md(samples: dict[str, list[float]]) -> str:
    """Across-seed summaries + the Cohen's d for protected-vs-naive BWT (the headline gap)."""
    if not samples:
        return "(no seed samples in summary)"
    lines = ["| Quantity | mean | std | sem | 95% CI | n |", "|---|---|---|---|---|---|"]
    for name in ("naive_bwt", "protected_bwt", "gap"):
        if name not in samples:
            continue
        s = seed_summary(samples[name])
        ci = s["ci"]
        lines.append(
            f"| {name} | {s['mean']:+.4f} | {s['std']:.4f} | {s['sem']:.4f} | "
            f"[{ci['lo']:+.4f}, {ci['hi']:+.4f}] | {s['n']} |"
        )
    out = ["\n".join(lines)]
    if "protected_bwt" in samples and "naive_bwt" in samples:
        d = cohens_d(samples["protected_bwt"], samples["naive_bwt"])
        out.append("")
        out.append(
            f"Cohen's d (protected vs naive BWT): {d:+.4f} "
            f"(positive => the protected arm retains better; "
            f"|d| ~ 0.2 small / 0.5 medium / 0.8 large)."
        )
    return "\n".join(out)


def build_report(summary: dict, source: str) -> str:
    samples = _seed_samples(summary)
    points = _frontier_points(summary)
    entries = _registry_entries(summary)
    sections = {
        "Analysis report (Frontier 11 scaffold)": (
            f"Source: {source}. Provenance: provisional (effect sizes and intervals are only as "
            f"sound as the underlying runs; this scaffold renders whatever it is given)."
        ),
        "Effect sizes and across-seed summaries": _effect_size_md(samples),
        "Adaptation-retention frontier": frontier_table(points),
        "Negative-result registry": null_registry_md(entries) if entries else "(no registry entries)",
    }
    return render_report(sections)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Frontier 11 analysis-report builder")
    ap.add_argument(
        "--summary",
        type=str,
        default=str(REPO_ROOT / "runs" / "cpu_campaign" / "summary.json"),
        help="campaign summary json to read (falls back to a built-in dummy if absent)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(REPO_ROOT / "runs" / "analysis_report.md"),
        help="markdown output path",
    )
    args = ap.parse_args(argv)

    summary, source = load_summary(Path(args.summary))
    md = build_report(summary, source)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    log.info("wrote %s (%d chars) from %s", out, len(md), source)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
