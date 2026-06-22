"""Frontier 11 tests. The statistics carry results claims, so they are checked against
closed-form values, not just shape: Cohen's d sign+magnitude on a known-separation pair, mean_ci
bracketing the mean and widening with variance / narrowing with n. The renderers are checked for
the load-bearing content (the Pareto winner is flagged, every section heading is present), and
build_report is exercised end to end on the built-in dummy to prove it produces real markdown
with no run and no files on disk.
"""

import importlib.util
import math
from pathlib import Path

from devsys.studies.report import (
    cohens_d,
    frontier_table,
    mean_ci,
    null_registry_md,
    render_report,
    seed_summary,
)

# load scripts/build_report.py by path (scripts/ is not an importable package)
_BR = Path(__file__).resolve().parents[2] / "scripts" / "build_report.py"
_spec = importlib.util.spec_from_file_location("build_report", _BR)
build_report_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_report_mod)


def test_cohens_d_sign_and_magnitude():
    # a clearly above b => positive d; b above a => negated d (antisymmetry).
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    # this pair has zero within-sample spread but differing means -> infinite, signed effect
    assert cohens_d(a, b) == math.inf
    assert cohens_d(b, a) == -math.inf
    # equal samples -> exactly zero effect
    assert cohens_d([0.2, 0.4], [0.2, 0.4]) == 0.0

    # finite, hand-computable case: means differ by 2.0, pooled std is 1.0 -> d == 2.0.
    x = [1.0, 3.0]  # mean 2, sum-sq-dev 2
    y = [-2.0, 0.0]  # mean -1, sum-sq-dev 2
    pooled = math.sqrt((2.0 + 2.0) / (2 + 2 - 2))  # = sqrt(2)
    assert math.isclose(cohens_d(x, y), 3.0 / pooled, rel_tol=1e-9)
    # antisymmetric and magnitude clearly "large" (>0.8)
    assert math.isclose(cohens_d(x, y), -cohens_d(y, x), rel_tol=1e-9)
    assert abs(cohens_d(x, y)) > 0.8


def test_mean_ci_brackets_mean_and_widens_with_variance():
    tight = [0.49, 0.50, 0.51]
    wide = [0.20, 0.50, 0.80]  # same mean (0.5), much larger spread
    ct, cw = mean_ci(tight), mean_ci(wide)
    # mean is recovered and the interval brackets it
    assert math.isclose(ct["mean"], 0.5, abs_tol=1e-9)
    assert ct["lo"] <= ct["mean"] <= ct["hi"]
    assert cw["lo"] <= cw["mean"] <= cw["hi"]
    # wider sample -> wider interval and larger sem
    assert (cw["hi"] - cw["lo"]) > (ct["hi"] - ct["lo"])
    assert cw["sem"] > ct["sem"]


def test_mean_ci_narrows_with_n_and_single_value_is_degenerate():
    # same spread pattern, more samples -> tighter interval (more seeds, more certainty).
    few = [0.4, 0.6]
    many = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
    assert (mean_ci(many)["hi"] - mean_ci(many)["lo"]) < (mean_ci(few)["hi"] - mean_ci(few)["lo"])
    # one value: sem 0, zero-width interval at the value itself (one seed cannot bound a mean).
    one = mean_ci([0.7])
    assert one["sem"] == 0.0 and one["lo"] == one["hi"] == one["mean"] == 0.7 and one["n"] == 1


def test_seed_summary_matches_closed_form():
    xs = [0.1, 0.2, 0.3, 0.4]
    s = seed_summary(xs)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    assert math.isclose(s["mean"], m, rel_tol=1e-9)
    assert math.isclose(s["std"], sd, rel_tol=1e-9)
    assert math.isclose(s["sem"], sd / math.sqrt(len(xs)), rel_tol=1e-9)
    assert s["n"] == 4
    # the embedded ci is the 95% mean_ci of the same sample
    assert math.isclose(s["ci"]["mean"], m, rel_tol=1e-9)


def test_frontier_table_flags_pareto_winner():
    # 'win' dominates 'lose' on both axes; 'tradeoff' is non-dominated (best retention).
    points = [
        {"name": "win", "adaptation": 0.9, "retention": 0.9},
        {"name": "lose", "adaptation": 0.3, "retention": 0.3},
        {"name": "tradeoff", "adaptation": 0.2, "retention": 1.0},
    ]
    md = frontier_table(points)
    lines = {
        ln.split("|")[2].strip(): ln
        for ln in md.splitlines()
        if ln.startswith("|") and "Method" not in ln and "---" not in ln
    }
    # the dominated point is NOT starred; the two Pareto points ARE
    assert lines["win"].split("|")[1].strip() == "*"
    assert lines["tradeoff"].split("|")[1].strip() == "*"
    assert lines["lose"].split("|")[1].strip() == ""
    assert "Frontier AUC:" in md
    # empty input still renders a valid table (no crash, AUC 0)
    assert "Frontier AUC: 0.0000" in frontier_table([])


def test_null_registry_md_tallies_and_lists():
    entries = [
        {
            "experiment": "e1",
            "verdict": "confirmed",
            "taxonomy_category": 3,
            "taxonomy_label": "x",
            "null_hypothesis": "h1",
        },
        {"experiment": "e2", "verdict": "refuted", "null_hypothesis": "h2\nwrapped"},
        {"experiment": "e3", "verdict": "mixed", "null_hypothesis": "h3"},
    ]
    md = null_registry_md(entries)
    assert "confirmed=1" in md and "refuted=1" in md and "mixed=1" in md
    assert "e1" in md and "e2" in md and "e3" in md
    # newline inside a null hypothesis is flattened so the table stays one row per entry
    assert "h2 wrapped" in md and "h2\nwrapped" not in md


def test_render_report_includes_title_and_all_sections():
    sections = {
        "My Title": "intro body",
        "Alpha": "alpha body",
        "Beta": "beta body",
    }
    md = render_report(sections)
    assert md.startswith("# My Title")
    assert "intro body" in md
    assert "## Alpha" in md and "alpha body" in md
    assert "## Beta" in md and "beta body" in md
    # title is H1, not H2
    assert "## My Title" not in md
    # empty mapping degrades gracefully
    assert "# Report" in render_report({})


def test_build_report_on_dummy_produces_nonempty_markdown():
    # Point at a path that does not exist so load_summary falls back to the built-in dummy:
    # build must still produce a real report with every section, no run and no files needed.
    summary, source = build_report_mod.load_summary(Path("/no/such/summary.json"))
    assert source == "built-in dummy"
    md = build_report_mod.build_report(summary, source)
    assert len(md.strip()) > 0
    for heading in (
        "# Analysis report",
        "## Effect sizes and across-seed summaries",
        "## Adaptation-retention frontier",
        "## Negative-result registry",
    ):
        assert heading in md
    # the dummy frontier has a clear winner ('decay' has the top retention) and a Pareto AUC,
    # and the effect-size section reports a Cohen's d line.
    assert "Frontier AUC:" in md
    assert "Cohen's d" in md
    # negative-registry verdict tally rendered from the dummy entries
    assert "confirmed=1" in md and "refuted=1" in md


def test_build_report_reads_real_summary_if_present():
    # If the campaign summary exists, build_report must consume IT (not the dummy) and still
    # render every section. Skipped cleanly when the file is absent.
    real = Path(__file__).resolve().parents[2] / "runs" / "cpu_campaign" / "summary.json"
    summary, source = build_report_mod.load_summary(real)
    if source == "built-in dummy":
        return  # no campaign summary on disk; covered by the dummy test above
    md = build_report_mod.build_report(summary, source)
    assert str(real) == source
    assert "## Effect sizes and across-seed summaries" in md
    assert "Cohen's d" in md  # real summary carries seed-variance per-seed BWT
