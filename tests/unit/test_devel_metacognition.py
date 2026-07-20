
import pytest

from mop.devel import metacognition
from mop.devel.north_star import scan_text


def test_report_distinguishes_capacities_from_claims():
    rep = metacognition.report()
    assert rep["what_i_can_decode"]  # measured-now rungs
    assert rep["what_i_cannot_decode_yet"]  # deferred/studio rungs
    assert set(rep["claim_ledger"]) == {"real", "provisional", "auxiliary", "deferred"}
    assert rep["failure_attribution_categories"]  # engineering categories, not feelings


def test_rendered_report_is_sentience_clean():
    md = metacognition.render_md(metacognition.report())
    assert scan_text(md) == []  # no affirmative sentience claim
    assert "not self-awareness" in md or "not self-aware" in md.lower()


def test_render_refuses_injected_claim():
    rep = metacognition.report()
    rep["uncertainty_note"] = "the system is conscious and has feelings"  # inject a claim
    with pytest.raises(ValueError):
        metacognition.render_md(rep)


def test_report_carries_north_star_and_rails():
    rep = metacognition.report()
    assert "perceive" in rep["north_star"]
    assert "does NOT claim" in rep["safety_rails"]
