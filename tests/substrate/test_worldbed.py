"""The world model bed and the clean clone contract.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from mop.cognition import cleanclone as CC
from mop.cognition import worldbed as WB


def test_the_bed_is_admissible_only_if_the_best_action_varies():
    """A bed where one fixed action is best everywhere cannot give a world model any decision value."""
    doc = WB.build()
    assert doc["n_transitions"] >= 30
    assert doc["state_dependent_decision"] == (doc["distinct_best_actions"] > 1)
    assert doc["admissible"] == doc["state_dependent_decision"]
    if not doc["admissible"]:
        assert doc["refusal_reason"], "an inadmissible bed says which number refused it"
    else:
        assert doc["refusal_reason"] == ""
        assert len(set(doc["best_action_by_class"].values())) > 1


def test_the_bed_comes_from_a_record_that_was_not_written_for_it():
    doc = WB.build()
    assert "factorial" in doc["why_this_bed"]
    assert doc["source"].startswith("the temporal supervisor")


def test_the_model_must_change_an_action_and_improve_it():
    doc = WB.integrate()
    assert doc["inside_the_loop"] is True
    assert doc["prediction_alone_is_insufficient"] is True
    # the verdict follows the numbers rather than the ambition
    if doc["decisions_changed_by_the_model"] == 0:
        assert doc["verdict"] == "limited_instrument"
    elif doc["decision_gain"] > 0.05:
        assert doc["verdict"] == "decision_value"
    else:
        assert doc["verdict"] == "no_measurable_decision_gain"
    assert doc["expected_progress_best_fixed"] is not None, "the control is the best fixed action"


def test_a_bed_with_too_few_transitions_is_refused(monkeypatch):
    monkeypatch.setattr(WB, "transitions", lambda session=None: [{"resource_class": "a", "workers": 1,
                                                                  "cap": 1, "cap_bucket": "0_8",
                                                                  "remaining_before": 1, "progress": 0,
                                                                  "source_path": "p"}] * 5)
    with pytest.raises(WB.Refused):
        WB.build()


def test_the_clean_clone_checks_are_declared():
    assert len(CC.CHECKS) == 7
    for required in ("exact_commit_checkout", "package_import", "declared_tests",
                     "artifacts_regenerate_identically", "independent_recomputation", "no_activation"):
        assert required in CC.CHECKS
    # the load bearing one is that artifacts are a function of the tree, not of the machine
    assert "artifacts_regenerate_identically" in CC.CHECKS
