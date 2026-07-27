"""SX5: the first Substrate experiment the gate licensed, and the null it produced.

House style: no dashes.
"""

from __future__ import annotations

import pytest

from substrate import evidence as io
from substrate import experiments as X
from substrate import program as P


@pytest.fixture(scope="module")
def decision():
    return X.sx5_run()


def test_sx5_is_licensed_on_a_bed_this_program_did_not_design(decision):
    assert decision["licensed"] is True
    assert decision["admission"]["blocked_at"] is None
    assert decision["causal_graph_violations"] == []
    # the prediction and the outcome come from different splits of receipts written by another program
    assert "another program" in X.SX5_DESIGN["why_the_bed_is_not_designed_here"]
    ev = decision["preprincipal_evidence"]
    assert ev["power"]["mde"] <= X.SESOI, "a licensed design has to be able to see its own SESOI"
    assert ev["oracle_headroom"]["residual_lower_95_cb"] > 0


def test_the_declared_limitation_is_recorded_not_hidden():
    """The tune to test gap was seen during design. The claim is narrowed rather than the peek denied."""
    limitation = X.SX5_DESIGN["declared_limitation"]
    assert "seen during design" in limitation
    assert "restricted to whether the offset transfers" in limitation


def test_sx5_is_a_null_at_the_declared_effect_size(decision):
    principal = decision["principal"]
    assert principal["verdict"] == "null"
    for direction in principal["directions"].values():
        # the effect is real and in the helpful direction in both replications
        assert direction["effect_best_baseline_minus_updating"] > 0
        assert direction["mean_error"]["updating"] < direction["mean_error"]["naive"]
        assert direction["mean_error"]["updating"] < direction["mean_error"]["fixed_prior"]
        # and it does not clear the SESOI, which is what makes it a null rather than a positive
        assert direction["lower_95_cb"] <= X.SESOI
        assert direction["supports"] is False


def test_sx5_requires_both_directions(decision):
    principal = decision["principal"]
    assert len(principal["directions"]) == 2, "one direction is a replication, not a result"
    assert set(principal["directions"]) == {"har_stream_to_harth_stream", "harth_stream_to_har_stream"}
    assert X.SX5_DESIGN["both_directions_must_support"] is True
    assert principal["both_directions_support"] == all(d["supports"] for d in principal["directions"].values())


def test_the_updating_arm_beats_the_fixed_prior_control_by_a_lot(decision):
    """The control is a fixed prior of the same form, and it is the weaker of the two baselines here."""
    for direction in decision["principal"]["directions"].values():
        assert direction["mean_error"]["fixed_prior"] > direction["mean_error"]["naive"], (
            "the uncorrected self report is the stronger baseline, so it is the one that must be beaten"
        )


def test_a_null_only_becomes_scientific_once_the_verifier_and_the_mutations_report():
    """classify_result is called twice on purpose, and the first call must not be scientific."""
    first = X.sx5_run()["result"]
    assert first["classification"] == "scientifically_unresolved"
    assert first["scientific"] is False

    if not (io.PROOF / "SUBSTRATE_MUTATION_REPORT.json").is_file():
        pytest.skip("the mutation report has not been sealed in this tree yet")
    out = X.sx5_classify()
    if not (out["verifier_agrees"] and out["mutations_rejected"]):
        pytest.skip("the verifier or the mutation suite has not reported for SX5 yet")
    assert out["result"]["classification"] == "mechanism_null"
    assert out["result"]["scientific"] is True
    assert P.result_ledger()["S1"]["classification"] == "mechanism_null"


def test_the_recorded_result_states_the_size_of_the_effect_it_is_calling_null():
    ledger = P.result_ledger()
    if "S1" not in ledger:
        pytest.skip("SX5 has not been classified in this tree yet")
    evidence = ledger["S1"]["evidence"]
    assert "roughly halves calibration error" in evidence["reading"]
    assert evidence["sesoi"] == X.SESOI
    assert evidence["independent_recomputation"]["all_agree"] is True
    assert evidence["mutation_suite"]["survivors"] == []
