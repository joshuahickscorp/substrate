
import importlib.util
import math
from pathlib import Path

from mop.studies.report import (
    cohens_d,
    frontier_table,
    mean_ci,
    null_registry_md,
    render_report,
    seed_summary,
)

_BR = Path(__file__).resolve().parents[2] / "scripts" / "build_report.py"
_spec = importlib.util.spec_from_file_location("build_report", _BR)
build_report_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_report_mod)


def test_cohens_d_sign_and_magnitude():
    a = [1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0]
    assert cohens_d(a, b) == math.inf
    assert cohens_d(b, a) == -math.inf
    assert cohens_d([0.2, 0.4], [0.2, 0.4]) == 0.0

    x = [1.0, 3.0]  # mean 2, sum-sq-dev 2
    y = [-2.0, 0.0]  # mean -1, sum-sq-dev 2
    pooled = math.sqrt((2.0 + 2.0) / (2 + 2 - 2))  # = sqrt(2)
    assert math.isclose(cohens_d(x, y), 3.0 / pooled, rel_tol=1e-9)
    assert math.isclose(cohens_d(x, y), -cohens_d(y, x), rel_tol=1e-9)
    assert abs(cohens_d(x, y)) > 0.8


def test_mean_ci_brackets_mean_and_widens_with_variance():
    tight = [0.49, 0.50, 0.51]
    wide = [0.20, 0.50, 0.80]  # same mean (0.5), much larger spread
    ct, cw = mean_ci(tight), mean_ci(wide)
    assert math.isclose(ct["mean"], 0.5, abs_tol=1e-9)
    assert ct["lo"] <= ct["mean"] <= ct["hi"]
    assert cw["lo"] <= cw["mean"] <= cw["hi"]
    assert (cw["hi"] - cw["lo"]) > (ct["hi"] - ct["lo"])
    assert cw["sem"] > ct["sem"]


def test_mean_ci_narrows_with_n_and_single_value_is_degenerate():
    few = [0.4, 0.6]
    many = [0.4, 0.6, 0.4, 0.6, 0.4, 0.6]
    assert (mean_ci(many)["hi"] - mean_ci(many)["lo"]) < (mean_ci(few)["hi"] - mean_ci(few)["lo"])
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
    assert math.isclose(s["ci"]["mean"], m, rel_tol=1e-9)


def test_frontier_table_flags_pareto_winner():
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
    assert lines["win"].split("|")[1].strip() == "*"
    assert lines["tradeoff"].split("|")[1].strip() == "*"
    assert lines["lose"].split("|")[1].strip() == ""
    assert "Frontier AUC:" in md
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
    assert "## My Title" not in md
    assert "# Report" in render_report({})


def test_build_report_on_dummy_produces_nonempty_markdown():
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
    assert "Frontier AUC:" in md
    assert "Cohen's d" in md
    assert "confirmed=1" in md and "refuted=1" in md


def test_build_report_reads_real_summary_if_present():
    real = Path(__file__).resolve().parents[2] / "runs" / "cpu_campaign" / "summary.json"
    summary, source = build_report_mod.load_summary(real)
    if source == "built-in dummy":
        return  # no campaign summary on disk; covered by the dummy test above
    md = build_report_mod.build_report(summary, source)
    assert str(real) == source
    assert "## Effect sizes and across-seed summaries" in md
    assert "Cohen's d" in md  # real summary carries seed-variance per-seed BWT
