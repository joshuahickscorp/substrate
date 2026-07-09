import pytest
import torch

from mop.diagnostics.operational_awareness import (
    OA_SCHEMA,
    compute_value,
    confidence_calibration,
    crisis_detection,
    memory_availability,
    missing_form_detection,
    mode_selection,
    oa_suite,
    render_oa_md,
    report_grounding,
    rewrite_caution,
)


def test_missing_form_detection_perfect_vs_shape_mismatch():
    scores = torch.tensor([0.9, 0.8, 0.1, 0.2])
    absent = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = missing_form_detection(scores, absent)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["absent_rate"] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="absence labels"):
        missing_form_detection(scores, absent[:2])


def test_confidence_calibration_perfect_confidence():
    conf = torch.tensor([0.95, 0.9, 0.1, 0.05])
    correct = torch.tensor([1.0, 1.0, 0.0, 0.0])
    out = confidence_calibration(conf, correct)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["ece"] <= 0.1
    assert out["accuracy"] == pytest.approx(0.5)


def test_memory_availability_ranks_payoff():
    claimed = torch.tensor([0.9, 0.7, 0.2, 0.1])
    payoff = torch.tensor([0.5, 0.3, 0.0, -0.1])
    assert memory_availability(claimed, payoff)["auroc"] == pytest.approx(1.0)


def test_mode_selection_regret_and_guards():
    oracle = torch.tensor([1.0, 0.8, 0.9])
    out = mode_selection(oracle.clone(), oracle, torch.tensor([0.5, 0.5, 0.5]))
    assert out["regret_vs_oracle"] == pytest.approx(0.0)
    assert out["delta_vs_random"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="upper-bound"):
        mode_selection(torch.tensor([1.0]), torch.tensor([0.5]), torch.tensor([0.5]))


def test_compute_value_flags_worthwhile_steps():
    continue_scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    gains = torch.tensor([0.3, 0.2, 0.0, -0.2])
    out = compute_value(continue_scores, gains, step_cost=0.05)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["worth_it_rate"] == pytest.approx(0.5)


def test_crisis_detection_reports_raw_error_margin():
    crisis = torch.tensor([0.9, 0.8, 0.1, 0.2])
    failed = torch.tensor([1.0, 1.0, 0.0, 0.0])
    raw = torch.tensor([0.5, 0.5, 0.5, 0.5])
    out = crisis_detection(crisis, failed, raw_error=raw)
    assert out["auroc"] == pytest.approx(1.0)
    assert out["raw_error_auroc"] == pytest.approx(0.5)
    assert out["auroc_over_raw_error"] == pytest.approx(0.5)


def test_rewrite_caution_margin():
    out = rewrite_caution(torch.tensor([0.0, 0.0, 1.0, 0.0]), torch.tensor([1.0, 1.0, 1.0, 0.0]))
    assert out["false_trigger_rate"] == pytest.approx(0.25)
    assert out["true_trigger_rate"] == pytest.approx(0.75)
    assert out["caution_margin"] == pytest.approx(0.5)


def test_report_grounding_counts_mismatches():
    out = report_grounding(
        {"mode": "planner", "form": "vision", "memory_hit": True},
        {"mode": "planner", "form": "audio", "memory_hit": True},
    )
    assert out["grounded_fraction"] == pytest.approx(2.0 / 3.0)
    assert out["mismatched_fields"] == ["form"]
    with pytest.raises(ValueError, match="shared field"):
        report_grounding({"a": 1}, {"b": 2})


def test_oa_suite_names_missing_components_and_renders_rail_clean():
    suite = oa_suite(
        oa2_calibration=confidence_calibration(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0])),
        oa7_rewrite_caution=rewrite_caution(torch.tensor([0.0]), torch.tensor([1.0])),
    )
    assert suite["schema"] == OA_SCHEMA
    assert suite["components_present"] == ["oa2_calibration", "oa7_rewrite_caution"]
    assert "oa6_crisis_detection" in suite["components_missing"]
    with pytest.raises(ValueError, match="unknown OA components"):
        oa_suite(oa9_soul=None)
    md = render_oa_md(suite, level_note="Ladder position is a measurement target, not a property.")
    assert "oa2_calibration" in md
    assert "not measured" in md


def test_render_refuses_sentience_claims():
    suite = oa_suite(oa8_report_grounding={"note is": 1.0})
    suite["components"]["oa8_report_grounding"] = {"claim": "the system is sentient"}
    with pytest.raises(ValueError, match="safety rail"):
        render_oa_md(suite)
