"""The cheap certification: audit, SX2, runtime activity, session and body canaries.

House style: no dashes.
"""

from __future__ import annotations

import json

import pytest

from substrate import audit as A
from substrate import certify as C
from substrate import evidence as io
from substrate import runtime as R
from substrate import workspace as W

# ---------------------------------------------------------------- audit


def test_the_structural_audit_passes_every_check():
    doc = A.run()
    assert doc["all_pass"] is True, doc["failed"]
    assert set(doc["results"]) == set(A.CHECKS)
    assert doc["results"]["no_activation_path"]["source_hits"] == []
    assert doc["results"]["no_activation_path"]["sealed_artifacts_with_activation_true"] == []


def test_a_dynamically_named_artifact_is_refused_by_the_producer_scan():
    """An artifact whose name is built at runtime is invisible to the exclusivity check.

    That is where a second producer would hide, so an artifact on disk that the scan cannot attribute
    fails the audit rather than being quietly ignored.
    """
    report = A.exclusive_producers()
    assert report["duplicated"] == {}
    assert report["on_disk_without_a_producer"] == [], report["on_disk_without_a_producer"]


def test_the_seal_survives_a_json_round_trip():
    """A dict keyed by integers is written with string keys, so a seal taken before the write fails."""
    from substrate import verification as V

    path = io.seal("SUBSTRATE_SEAL_ROUNDTRIP_PROBE.json", {"k_rows": {1: "a", 2: "b"}, "probe": True})
    try:
        doc = json.loads(path.read_text())
        assert V._seal_intact(doc) is True
        assert set(doc["k_rows"]) == {"1", "2"}
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------- SX2


@pytest.fixture(scope="module")
def sx2():
    return json.loads((io.PROOF / "SUBSTRATE_SX2_DIVERSITY.json").read_text())


def test_sx2_closes_because_the_oracle_ceiling_is_low(sx2):
    """Oracle selection is an upper bound. If it does not clear the SESOI, no selector can."""
    assert sx2["n_cells"] >= 70
    margins = [r["margin_over_matched_single"] for r in sx2["k_rows"].values()]
    assert max(margins) < sx2["sesoi"], margins
    assert sx2["k_clearing_sesoi"] == []
    assert sx2["verdict"] == "closed_no_headroom"
    assert "no selector built on this set can" in sx2["reading"]


def test_every_sx2_comparison_is_compute_matched(sx2):
    for k, row in sx2["k_rows"].items():
        matched = row["strongest_compute_matched_single"]
        assert matched["n_affordable"] >= 1, k
        assert row["set_cost"] > 0
        # the bar is the best single cell affordable at the set's own cost, not the best cell overall
        assert matched["value"] is not None
    assert "k times one cell" in sx2["compute_matching"]


def test_the_cells_are_a_real_perspective_set_not_a_designed_one(sx2):
    stats = sx2["diversity_statistics"]
    assert stats["n_cells"] == sx2["n_cells"]
    assert 0.0 <= stats["complementary_fraction"] <= 1.0
    assert stats["mean_pairwise_correlation"] is not None
    assert "none was built to be a perspective set" in sx2["why_admissible"]


# ---------------------------------------------------------------- runtime activity


@pytest.fixture(scope="module")
def activity():
    return C.runtime_activity()


def test_every_ablatable_runtime_stage_is_active(activity):
    assert activity["inactive"] == [], activity["inactive"]
    assert len(activity["active"]) == len(C.ABLATABLE)
    for stage, row in activity["results"].items():
        assert row["classification"] == "active", (stage, row)


def test_a_stage_with_no_possible_null_control_says_so(activity):
    inapplicable = activity["null_control_inapplicable"]
    assert set(inapplicable) == {"arbitrate", "decide", "remember", "checkpoint"}
    for stage, reason in inapplicable.items():
        assert "every cycle by design" in reason
        assert activity["results"][stage]["null_control"]["applicable"] is False
    # and every stage that does admit a control passes it
    assert activity["control_clean"] is True, activity["null_fixture_sensitive"]


def test_an_ablated_stage_is_recorded_as_skipped_not_as_run():
    entity = R.Substrate(ablate=frozenset({"consolidate"}))
    trace = entity.step({"label": "a", "label_confidence": 0.8}, outcome="a")
    assert "consolidate" in trace["stages_skipped"]
    assert trace["stages"]["consolidate"]["reason"] == "ablated"
    assert trace["complete"] is False
    with pytest.raises(R.Refused):
        R.Substrate(ablate=frozenset({"telepathy"}))


# ---------------------------------------------------------------- canaries


@pytest.fixture(scope="module")
def certification():
    return json.loads((io.PROOF / "SUBSTRATE_LONG_RUN_CERTIFICATION.json").read_text())


def test_every_session_canary_passes(certification):
    canaries = certification["session_canaries"]
    assert canaries["failed"] == [], canaries["failed"]
    assert canaries["events_used"] >= 20
    assert canaries["results"]["restoration"]["passes"] is True
    assert canaries["results"]["reliability_update"]["changed"] is True


def test_the_three_bodies_are_pairwise_distinct(certification):
    body = certification["body_canaries"]
    assert body["every_pair_differs_somewhere"] is True
    assert body["every_dimension_separates"] is True
    assert body["non_distinct_dimensions"] == []
    # the substrate changes what the body would have said, or the composition is decoration
    assert body["any_body_facing_change"] is True
    for name, row in body["microtasks"].items():
        assert row["decisions_substrate_changed"] > 0, name


def test_a_failed_component_is_gated_rather_than_blocking(certification):
    assert certification["green"] == (certification["gated_components"] == [])
    assert "gated out of the run rather than blocking" in certification["rule"]


# ---------------------------------------------------------------- regressions from the closure pass


def test_every_attention_candidate_is_a_region_a_perspective_can_read():
    """Correction C_ATTENTION_CANDIDATE_NOT_A_REGION.

    The attended set filters the perspective pool by declared inputs, so a candidate whose id is not a
    region name silently removes every perspective that reads only that region, at every budget. The
    original list called the perceptual region "observation" and dropped the direct perspective forever.
    """
    regions = {r.name for r in W.REGIONS}
    readable = {i for p in R.PS.CATALOG for i in p.spec.inputs}
    for candidate in R.Substrate()._attention_candidates({"label": "a"}):
        assert candidate["id"] in regions, candidate["id"]
        assert candidate["id"] in readable, candidate["id"]


def test_ablation_observes_what_a_stage_writes_not_only_where_it_writes():
    """Correction C_ACTIVITY_PROBE_IGNORED_REGION_CONTENTS.

    The identity hash lists region names, not region contents. Judging activity by it alone called
    arbitration wiring, because its decision can coincide with the first perspective while its preserved
    minority and named missing evidence do not.
    """
    base = C._run(C.positive_fixture())
    ablated = C._run(C.positive_fixture(), frozenset({"arbitrate"}))
    assert base["state"] == ablated["state"], "the defect: names alone cannot see this stage"
    assert base["contents"] != ablated["contents"]
