import math
import copy
import json

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


def test_training_update_mutation_uses_each_cells_sealed_checkpoint(monkeypatch, tmp_path):
    principal = tmp_path / "e2_principal"
    principal.mkdir()
    selected = {"cell_a": 400, "cell_b": 1600}
    receipt = {
        "convergence_authority": {"selected_checkpoints": selected},
        "runs": [
            {"cell": "cell_a", "steps": 400, "updates": 400},
            {"cell": "cell_b", "steps": 1600, "updates": 1600},
        ],
    }
    path = principal / "bed_0.json"
    path.write_text(json.dumps(receipt))
    monkeypatch.setattr(mutations.io, "RUNS", tmp_path)
    result = mutations.training_update_mutation()
    assert result["pass"] and result["n_receipts"] == 1 and result["n_runs"] == 2
    receipt["runs"][1]["updates"] = 1601
    path.write_text(json.dumps(receipt))
    assert not mutations.training_update_mutation()["pass"]


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


def test_independent_convergence_inventory_is_exact_76_and_preserves_appended_identities():
    specs = V._expected_convergence_specs()
    names = [V.Fx.cell_name(**spec) for spec in specs]
    assert len(names) == len(set(names)) == 76
    assert names[:3] == ["gru|small|linear|none|h1", "lstm|small|linear|none|h1",
                         "mgu|small|linear|none|h1"]
    assert names[-3:] == ["histmlp|large|mlp_strong|none|h1",
                          "tcn|large|mlp_strong|none|h1", "gru|large|mlp_strong|none|h1"]


def test_convergence_receipt_reconstructs_curve_parameters_and_hash(monkeypatch):
    spec = dict(V.Fx.REFERENCE)
    counts = {"core": 80_000, "readout": 1_000, "total": 81_000}
    monkeypatch.setattr(V, "_parameter_count", lambda bed, candidate: counts)
    curve = {400: 0.70, 800: 0.71, 1600: 0.71, 3200: 0.71}
    records = {budget: [{"seed": seed, "updates": budget, "score": score,
                         "checkpoint_sha": f"{seed + 1:x}" * 64}
                        for seed, score in enumerate((value - 0.01, value, value + 0.01))]
               for budget, value in curve.items()}
    witness = V._plateau(curve)
    document = {"bed": "har_stream", "spec": spec, "cell": V.Fx.cell_name(**spec),
                "curve": curve, "seed_spread": {budget: 0.01 for budget in curve},
                "seed_scores": {budget: [row["score"] for row in rows]
                                for budget, rows in records.items()},
                "arm_records": records,
                "seeds": [0, 1, 2], "parameter_count": counts,
                "classification": witness["classification"],
                "selected_checkpoint": witness["selected_checkpoint"],
                "second_half_movement": witness["second_half_movement"],
                "residual_slope": witness["residual_slope"], "converged": witness["all_pass"],
                "program": V.io.PROGRAM, "source_commit": "a" * 40,
                "source_tree_oid": "b" * 40, "result_hash_version": "canonical_json_v2"}
    document["result_sha256"] = V.io.sha_obj(document)
    assert all(V._curve_receipt_checks(document, bed="har_stream", spec=spec,
                                       grid=V.BASE_CONVERGENCE_GRID).values())
    mutated = copy.deepcopy(document)
    mutated["curve"][800] = 0.99
    assert not all(V._curve_receipt_checks(mutated, bed="har_stream", spec=spec,
                                           grid=V.BASE_CONVERGENCE_GRID).values())


def test_local_hypothesis_fold_does_not_reinterpret_family_agreement_as_h1_support():
    folded = V._independent_hypothesis_fold(["all_recurrent_families_agree"])
    assert folded["hypotheses"]["H1_recurrence"]["state"] == "unresolved"
    assert folded["hypotheses"]["H7_architecture_family"]["state"] == "closed"
    assert not folded["unknown_result_keys"]


def test_invalid_optimization_receipt_cannot_close_optimization_hypothesis(monkeypatch):
    effect = {"mean": 0.2, "lower_95_cb": 0.1, "upper_95_cb": 0.3,
              "group_mean": 0.2, "group_lower_95_cb": 0.1, "group_upper_95_cb": 0.3,
              "verdict": "positive", "estimator_sufficient": True,
              "convergence": {"all_converged": True}}
    recomputed = {bed: {"recurrent_versus_matched_history": {
        "gru_vs_histmlp_kfull_window": effect}} for bed in ("har_stream", "speech_stream")}
    recomputed.update({f"optimization:{bed}": {"receipt_valid": False, "converged": True,
                       "scientific_verdict": "invalid_receipt"}
                       for bed in ("har_stream", "speech_stream")})
    monkeypatch.setattr(V.io, "exists", lambda *_: False)
    keys = V._independent_result_keys(
        {"principal_beds": ["har_stream", "speech_stream"]}, recomputed)
    assert "recurrent_beats_matched_history" in keys
    assert "converged_everywhere_and_gap_remains" not in keys


def test_actual_equivalence_is_not_changed_by_a_forged_sealed_pass_flag():
    effect = {"mean": 0.0, "lower_95_cb": -0.01, "upper_95_cb": 0.01,
              "group_mean": 0.0, "group_lower_95_cb": -0.01, "group_upper_95_cb": 0.01}
    sealed = {"seed_equivalent": False, "group_equivalent": False, "mean": 0.0,
              "lower_95_cb": -0.01, "group_lower_95_cb": -0.01, "group_upper_95_cb": 0.01}
    actual_pass, sealed_matches = V._equivalence_row_audit(effect, sealed)
    assert actual_pass
    assert not sealed_matches
