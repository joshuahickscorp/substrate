"""The ONE pytest touchpoint for the microbenchmarks. Asserts each bench_* returns the expected
record shape with finite positive timing, and that run_benches covers every bench. Sizes are the
benches' own tiny defaults, so this is fast; the benches do not otherwise run during the suite.
"""

import math

from devsys import bench

_KEYS = {"name", "seconds", "n", "throughput"}


def _check_record(rec):
    assert set(rec) >= _KEYS, rec
    assert isinstance(rec["name"], str) and rec["name"]
    assert rec["n"] > 0
    assert math.isfinite(rec["seconds"]) and rec["seconds"] > 0
    assert math.isfinite(rec["throughput"]) and rec["throughput"] > 0
    # throughput is items/sec by construction
    assert math.isclose(rec["throughput"], rec["n"] / rec["seconds"], rel_tol=1e-6)


def test_each_bench_returns_valid_record():
    for fn in bench.BENCHES:
        _check_record(fn())


def test_run_benches_covers_all():
    recs = bench.run_benches(reps=2)
    assert len(recs) == len(bench.BENCHES)
    expected = {fn().get("name") for fn in bench.BENCHES}
    assert {r["name"] for r in recs} == expected
    for r in recs:
        _check_record(r)
        assert r["reps"] == 2


def test_faiss_bench_is_honest_when_unavailable():
    """The faiss bench either measures faiss (probe passed) or runs brute and labels itself, so a
    fallback is never mistaken for a real faiss number."""
    rec = bench.bench_retrieve_faiss()
    if bench._faiss_search_safe():
        assert rec["name"] == "retrieve_faiss"
    else:
        assert rec["name"] == "retrieve_faiss_unavailable"
    _check_record(rec)


def test_render_md_has_a_row_per_bench():
    recs = bench.run_benches(reps=1)
    md = bench.render_md(recs)
    assert md.startswith("# Microbenchmarks")
    for r in recs:
        assert r["name"] in md
