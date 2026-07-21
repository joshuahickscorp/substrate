import math

from mop.temporal import arch as A
from mop.temporal.runs import analyze
from mop.temporal.runs import mutations
from mop.temporal.runs import verify as V


def test_independent_summary_reports_two_sided_group_information():
    result = V.summarize([0.1, 0.2, 0.3, 0.4])
    assert result["lower_95_cb"] < result["mean"] < result["upper_95_cb"]
    assert result["heterogeneity"] > 0
    assert result["n"] == 4


def test_unit_duplicate_mutation_is_rejected_before_averaging():
    result = mutations.duplicated_unit_mutation()
    assert result["pass"]
    assert result["original_audit"]["all_pass"]
    assert not result["forged_audit"]["all_pass"]
    assert result["forged_audit"]["duplicate_units"] == ["unit_a"]


def test_readout_inventory_checks_each_readout_across_all_cells():
    rows = [{"spec": {"readout": readout}, "params": {"readout": 10 + i}}
            for i, readout in enumerate(A.READOUTS)]
    rows.append({"spec": {"readout": A.READOUTS[0]}, "params": {"readout": 10}})
    valid = V._readout_inventory(rows)
    assert valid["complete"] and valid["depends_only_on_readout"]
    rows[-1]["params"]["readout"] = 999
    assert not V._readout_inventory(rows)["depends_only_on_readout"]


def test_difference_in_differences_reconstruction_preserves_seed_and_unit_pairing():
    components = ["a", "b", "c", "d"]
    by_cell = {
        cell: {seed: {"accuracy": value + seed * 0.01}
               for seed in (0, 1, 2)}
        for cell, value in zip(components, (0.8, 0.5, 0.6, 0.4), strict=True)
    }
    units = {
        cell: {"u1": value, "u2": value + 0.1, "u3": value - 0.1}
        for cell, value in zip(components, (0.8, 0.5, 0.6, 0.4), strict=True)
    }
    effect = V._recompute_effect(by_cell, units, components, [1, -1, -1, 1], [0, 1, 2])
    assert effect["per_seed_effects"] == [0.1, 0.1, 0.1]
    assert effect["group_mean"] == 0.1
    assert effect["group_lower_95_cb"] <= effect["group_mean"] <= effect["group_upper_95_cb"]


def test_equivalence_and_terminal_classification_are_independently_bounded():
    equivalent = {"mean": 0.0, "lower_95_cb": -0.01, "upper_95_cb": 0.01,
                  "group_mean": 0.0, "group_lower_95_cb": -0.01, "group_upper_95_cb": 0.01}
    assert V._equivalent(equivalent, 0.02)
    equivalent["upper_95_cb"] = 0.03
    assert not V._equivalent(equivalent, 0.02)
    equivalent["upper_95_cb"] = 0.01
    equivalent.pop("group_upper_95_cb")
    assert not V._equivalent(equivalent, 0.02)
    effect = {"verdict": "positive", "estimator_sufficient": True,
              "convergence": {"all_converged": True}}
    assert V._terminal_classification(
        effect, instrument_valid=True, bed_valid=True, verifier_agrees=True,
        mutations_rejected=True, implementations_agree=True) == "positive"
    effect["convergence"]["all_converged"] = False
    assert V._terminal_classification(
        effect, instrument_valid=True, bed_valid=True, verifier_agrees=True,
        mutations_rejected=True, implementations_agree=True) == "unconverged_baseline"


def test_welch_cross_bed_difference_keeps_unrelated_units_unpaired():
    result = V._welch([0.30, 0.32, 0.28], [0.10, 0.12, 0.08, 0.11])
    assert result is not None
    assert math.isclose(result["mean"], 0.1975)
    assert result["lower_95_cb"] < result["mean"] < result["upper_95_cb"]
    assert result["degrees"] > 2


def test_cross_bed_producer_and_independent_welch_agree_on_the_estimand():
    left = {"per_seed_effects": [0.30, 0.32, 0.28],
            "per_unit_effects": {"a": 0.31, "b": 0.29, "c": 0.30}}
    right = {"per_seed_effects": [0.10, 0.12, 0.08],
             "per_unit_effects": {"x": 0.11, "y": 0.09, "z": 0.10}}
    produced = analyze.independent_bed_difference(left, right, "bed DID")
    seed = V._welch(left["per_seed_effects"], right["per_seed_effects"])
    group = V._welch(list(left["per_unit_effects"].values()),
                     list(right["per_unit_effects"].values()))
    assert produced["estimand"] == "independent_bed_difference_in_differences"
    assert math.isclose(produced["mean"], seed["mean"], abs_tol=1e-5)
    assert math.isclose(produced["group_lower_95_cb"], group["lower_95_cb"], abs_tol=1e-5)
