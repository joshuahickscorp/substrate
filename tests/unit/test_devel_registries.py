from mop.devel import registries as R


def test_shipped_registries_validate_clean():
    assert R.validate_paradigms() == []
    assert R.validate_capacities() == []
    assert R.validate_paperwatch() == []
    assert R.validate_all() == []


def test_every_paradigm_has_the_full_contract():
    for p in R.load_paradigms():
        for field in ("baseline", "ablation", "metric", "null_hypothesis", "local_test", "status"):
            assert p.get(field), f"{p['slug']} missing {field}"


def test_paradigm_schema_rejects_bad_vocab():
    bad = {
        "slug": "x",
        "name": "X",
        "thesis": "t",
        "family": "memory",
        "changes": "telepathy",  # bad
        "vjepa_compat": "cached-pooled",
        "mvp_local_test": "t",
        "studio_test": "t",
        "compute_risk": "low",
        "expected_signature": "s",
        "null_interpretation": "n",
        "tags": ["memory"],
        "status": "prototype-local",
    }
    bad.update(metric="s", null_hypothesis="n", local_test="t", baseline="b", ablation="a")
    problems = R.validate_paradigm(bad)
    assert any("changes" in p for p in problems)


def test_paradigm_candidate_cannot_claim_canonical_result():
    bad = dict(R.load_paradigms()[0])
    bad["result_tag"] = "real-encoder"  # forbidden on a candidate
    problems = R.validate_paradigm(bad)
    assert any("canonical result_tag" in p for p in problems)


def test_implement_now_paradigm_needs_concrete_local_test():
    bad = dict(R.load_paradigms()[0])
    bad["status"] = "implement-now"
    bad["mvp_local_test"] = "deferred; needs the Studio"
    problems = R.validate_paradigm(bad)
    assert any("implement-now" in p and "deferred" in p for p in problems)


def test_capacity_ladder_has_14_rungs_ordered_and_complete():
    rungs = R.load_capacities()
    assert [c["rung"] for c in rungs] == list(range(1, len(rungs) + 1))
    assert len(rungs) >= 14
    for c in rungs:
        for field in ("metric", "null_hypothesis", "local_test", "studio_test", "failure_interpretation"):
            assert c.get(field), f"rung {c['rung']} missing {field}"


def test_capacity_schema_rejects_missing_field():
    bad = {"rung": 1, "name": "x", "tag": "perception"}  # missing the rest
    problems = R.validate_capability(bad)
    assert any("missing required field" in p for p in problems)


def test_paperwatch_offline_and_complete():
    topics = R.load_paperwatch()
    assert topics  # loads from the cached registry, no network
    for t in topics:
        for field in R.PAPERWATCH_REQUIRED:
            assert field in t, f"{t.get('slug')} missing {field}"


def test_paradigms_cover_the_frontier_families():
    families = {p["family"] for p in R.load_paradigms()}
    tags = {tag for p in R.load_paradigms() for tag in p["tags"]}
    assert {"plasticity", "memory", "abstraction"} <= tags
    assert "local-learning" in tags and "active-curriculum" in tags
    assert "objective" in families or "architecture" in families
