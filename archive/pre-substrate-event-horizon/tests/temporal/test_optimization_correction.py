import copy
import json

import numpy as np
import pytest

from mop.temporal import factorial as Fx
from mop.temporal import hypotheses as H
from mop.temporal import io
from mop.temporal import witness as W
from mop.temporal.runs import analyze, corrections, e2, verify


SEEDS = tuple(e2.CONVERGENCE_SEEDS)
REGULAR_GRID = e2.CONVERGENCE_GRID + e2.EXTENDED_CONVERGENCE_GRID
SMALL_COMPUTE_BUDGET = 4800


def _rows(tier, budget, scores, unit_effects):
    params = 600_000 if tier == "large" else 50_000
    baseline = {unit: 0.4 for unit in unit_effects}
    improved = tier == "large" and budget != min(REGULAR_GRID)
    return [
        {
            "seed": seed,
            "updates": budget,
            "trainable_param_count": params,
            "parameter_update_exposure": params * budget,
            "score": scores[seed],
            "per_unit_accuracy": {
                unit: value + (unit_effects[unit] if improved else 0.0)
                for unit, value in baseline.items()
            },
            "checkpoint_sha": f"{seed + 1:x}" * 64,
            "wall_seconds": 0.01,
        }
        for seed in SEEDS
    ]


def _document(tier, *, seed_effects=(0.2, 0.2, 0.2), unit_effects=None, unconverged=False):
    unit_effects = unit_effects or {"unit_a": 0.2, "unit_b": 0.2, "unit_c": 0.2}
    budgets = REGULAR_GRID + ((SMALL_COMPUTE_BUDGET,) if tier == "small" else ())
    records = {}
    for budget in budgets:
        if tier == "large":
            if unconverged:
                offset = REGULAR_GRID.index(budget) * 0.08
                scores = {seed: 0.4 + offset + seed_effects[seed] / 100 for seed in SEEDS}
            elif budget == min(REGULAR_GRID):
                scores = {seed: 0.5 for seed in SEEDS}
            else:
                scores = {seed: 0.5 + seed_effects[seed] for seed in SEEDS}
        else:
            scores = {seed: 0.5 for seed in SEEDS}
        records[budget] = _rows(tier, budget, scores, unit_effects)
    curve = {budget: float(np.mean([row["score"] for row in records[budget]]))
             for budget in budgets}
    spread = {budget: round(float(np.std([row["score"] for row in records[budget]], ddof=1)), 5)
              for budget in budgets}
    regular_curve = {budget: curve[budget] for budget in REGULAR_GRID}
    witness = W.plateau_validity(regular_curve)
    selected = int(witness["selected_checkpoint"])
    spec = dict(Fx.REFERENCE, tier=tier)
    role = ("large_model_at_same_update_count" if tier == "large"
            else "small_model_at_same_compute")
    anchor = min(REGULAR_GRID) if tier == "large" else SMALL_COMPUTE_BUDGET
    params = 600_000 if tier == "large" else 50_000
    return {
        "schema": "mop-e2-optimization-capacity-correction/v1",
        "bed": "har_stream",
        "tier": tier,
        "spec": spec,
        "cell": Fx.cell_name(**spec),
        "seeds": list(SEEDS),
        "curve": curve,
        "seed_spread": spread,
        "seed_scores": {budget: [row["score"] for row in records[budget]] for budget in budgets},
        "per_unit_seed_scores": {
            budget: {str(row["seed"]): row["per_unit_accuracy"] for row in records[budget]}
            for budget in budgets
        },
        "wall_seconds_per_seed": {budget: [0.01] * len(SEEDS) for budget in budgets},
        "parameter_count": {"core": params, "total": params},
        "arm_records": records,
        "regular_grid": list(REGULAR_GRID),
        "same_update_anchor": min(REGULAR_GRID),
        "compute_match": {
            "large_steps": min(REGULAR_GRID),
            "large_parameter_updates": 600_000 * min(REGULAR_GRID),
            "small_steps": SMALL_COMPUTE_BUDGET,
            "small_parameter_updates": 50_000 * SMALL_COMPUTE_BUDGET,
            "relative_parameter_update_error": 0.0,
        },
        "four_contrast_roles": {
            role: {"budget": anchor, "records": records[anchor]},
            f"{tier}_model_at_strict_selected_convergence": {
                "budget": selected,
                "records": records[selected],
                "classification": witness["classification"],
            },
        },
        "design_checks": {"synthetic_fixture": True},
        "all_checks_pass": True,
        **witness,
    }


def _write_pair(monkeypatch, tmp_path, large=None, small=None):
    runs = tmp_path / "runs"
    proof = tmp_path / "proof"
    stage = runs / "e2_optimization_corrections"
    stage.mkdir(parents=True)
    proof.mkdir()
    monkeypatch.setattr(io, "RUNS", runs)
    monkeypatch.setattr(io, "PROOF", proof)
    pair = {"large": large or _document("large"), "small": small or _document("small")}
    for tier, doc in list(pair.items()):
        doc.update({"program": io.PROGRAM, "source_commit": "a" * 40,
                    "source_tree_oid": "b" * 40, "result_hash_version": "canonical_json_v2"})
        doc = json.loads(json.dumps(doc))
        doc["result_sha256"] = io.sha_obj({k: v for k, v in doc.items() if k != "result_sha256"})
        pair[tier] = doc
        (stage / f"optimization_har_stream_{tier}.json").write_text(json.dumps(doc))
    return pair, runs, proof


def _run_independent_verifier(proof, expected):
    principal = {
        "seeds": list(SEEDS),
        "per_bed": {},
        "principal_beds": [],
        "observed_result_keys": [],
        "hypothesis_fold": H.apply([]),
        "terminal_classification": {},
    }
    (proof / "MOP_E2_PRINCIPAL_RESULT.json").write_text(json.dumps(principal))
    interactions = {
        "architecture_by_bed": {},
        "horizon_by_bed": {},
        "optimization_by_capacity": {"har_stream": expected},
    }
    (proof / "MOP_FACTORIAL_INTERACTION_REPORT.json").write_text(json.dumps(interactions))
    return verify.role_c()


def test_valid_receipts_bind_exact_seeds_roles_compute_and_total_exposure(monkeypatch, tmp_path):
    _write_pair(monkeypatch, tmp_path)
    result = analyze.optimization_interaction("har_stream")

    assert result["receipts_valid"]
    assert result["receipt_checks"]["exact_role_inventory"]
    assert all(result["receipt_checks"][f"{arm}_exact_seed_identity"] for arm in result["components"])
    expected_by_arm = {
        arm: sum(row["parameter_update_exposure"] for row in rows.values())
        for arm, rows in {
            "large_convergence": {
                row["seed"]: row for row in _document("large")["arm_records"][800]
            },
            "large_same_update": {
                row["seed"]: row for row in _document("large")["arm_records"][400]
            },
            "small_convergence": {
                row["seed"]: row for row in _document("small")["arm_records"][400]
            },
            "small_same_compute": {
                row["seed"]: row for row in _document("small")["arm_records"][SMALL_COMPUTE_BUDGET]
            },
        }.items()
    }
    assert result["parameter_update_exposure_by_arm"] == expected_by_arm
    assert result["parameter_update_exposure_denominator"] == sum(expected_by_arm.values())
    verification = _run_independent_verifier(tmp_path / "proof", result)
    assert verification["checks"]["interaction:optimization_by_capacity:har_stream"]


def test_seed_position_reordering_is_handled_by_seed_identity(monkeypatch, tmp_path):
    large, small = _document("large"), _document("small")
    baseline_dir = tmp_path / "baseline"
    _write_pair(monkeypatch, baseline_dir, large=large, small=small)
    baseline = analyze.optimization_interaction("har_stream")

    reordered = copy.deepcopy(large)
    for budget, rows in reordered["arm_records"].items():
        reordered["arm_records"][budget] = list(reversed(rows))
    for role in reordered["four_contrast_roles"].values():
        role["records"] = list(reversed(role["records"]))
    _write_pair(monkeypatch, tmp_path / "reordered", large=reordered, small=small)
    actual = analyze.optimization_interaction("har_stream")

    assert actual["receipts_valid"]
    assert actual["per_seed_effects"] == baseline["per_seed_effects"]
    assert actual["mean"] == baseline["mean"]


@pytest.mark.parametrize("mutation", ["missing_seed", "duplicate_seed", "unit_mismatch"])
def test_malformed_arm_inventories_fail_closed(monkeypatch, tmp_path, mutation):
    large, small = _document("large"), _document("small")
    rows = large["arm_records"][min(REGULAR_GRID)]
    if mutation == "missing_seed":
        rows.pop()
    elif mutation == "duplicate_seed":
        rows[-1] = copy.deepcopy(rows[0])
    else:
        rows[0]["per_unit_accuracy"].pop("unit_c")
    large["four_contrast_roles"]["large_model_at_same_update_count"]["records"] = rows
    _, _, proof = _write_pair(monkeypatch, tmp_path, large=large, small=small)

    result = analyze.optimization_interaction("har_stream")

    assert result["classification"] == "invalid_receipt"
    assert result["verdict"] == "invalid_receipt"
    assert result["mean"] is None
    verification = _run_independent_verifier(proof, result)
    assert not verification["checks"]["interaction:optimization_by_capacity:har_stream"]


def test_group_below_sesoi_cannot_be_a_scientific_positive(monkeypatch, tmp_path):
    unit_effects = {"unit_a": 0.2, "unit_b": 0.0, "unit_c": 0.2}
    _write_pair(monkeypatch, tmp_path, large=_document("large", unit_effects=unit_effects),
                small=_document("small", unit_effects=unit_effects))

    result = analyze.optimization_interaction("har_stream")

    assert result["raw_statistical_verdict"] == "positive"
    assert result["group_lower_95_cb"] < io.SESOI
    assert result["verdict"] == "positive_seed_only_group_floor_not_met"
    assert result["component_floor_status"] == "provisional_or_below_floor"


def test_unconverged_curve_remains_provisional(monkeypatch, tmp_path):
    _write_pair(monkeypatch, tmp_path, large=_document("large", unconverged=True))

    result = analyze.optimization_interaction("har_stream")

    assert result["raw_statistical_verdict"] == "positive"
    assert result["classification"] == "provisional_unconverged"
    assert result["verdict"] == "provisional_unconverged"
    assert result["cost_adjusted_effect_per_billion_parameter_updates"] is None


def test_producer_excludes_extra_small_compute_point_from_strict_plateau(monkeypatch):
    captured = {}

    def fake_build(_splits, *, seed, **spec):
        return spec["tier"], None, None

    def fake_count(tier):
        params = 600_000 if tier == "large" else 50_000
        return {"core": params, "total": params}

    def fake_measure(_bed, spec, budgets):
        tier = spec["tier"]
        curve = {budget: (0.99 if budget == SMALL_COMPUTE_BUDGET else
                          0.8 if budget == min(REGULAR_GRID) else 0.81) for budget in budgets}
        rows = {
            budget: _rows(tier, budget, {seed: curve[budget] for seed in SEEDS}, {"unit": 0.0})
            for budget in budgets
        }
        params = 600_000 if tier == "large" else 50_000
        return {
            "curve": curve,
            "seed_spread": {budget: 0.0 for budget in budgets},
            "seed_scores": {budget: [curve[budget]] * len(SEEDS) for budget in budgets},
            "per_unit_seed_scores": {
                budget: {str(seed): rows[budget][seed]["per_unit_accuracy"] for seed in SEEDS}
                for budget in budgets
            },
            "wall_seconds_per_seed": {budget: [0.0] * len(SEEDS) for budget in budgets},
            "parameter_count": {"core": params, "total": params},
            "arm_records": rows,
        }

    monkeypatch.setattr(corrections.B, "splits", lambda *_: {})
    monkeypatch.setattr(corrections.Fx, "build_cell", fake_build)
    monkeypatch.setattr(corrections.A, "count", fake_count)
    monkeypatch.setattr(corrections, "measure_curve", fake_measure)
    monkeypatch.setattr(corrections.io, "run_json", lambda name, doc, stage: captured.update(doc))

    doc = corrections.optimization_shard("har_stream", "small")
    regular = {budget: doc["curve"][budget] for budget in doc["regular_grid"]}

    assert SMALL_COMPUTE_BUDGET in doc["curve"]
    assert SMALL_COMPUTE_BUDGET not in doc["regular_grid"]
    assert doc["selected_checkpoint"] == verify._plateau(regular)["selected_checkpoint"]
    assert doc["classification"] == verify._plateau(regular)["classification"] == "converged"
    assert verify._plateau(doc["curve"])["selected_checkpoint"] == SMALL_COMPUTE_BUDGET
    assert set(doc["four_contrast_roles"]) == {
        "small_model_at_same_compute", "small_model_at_strict_selected_convergence"
    }
    assert doc["seeds"] == list(SEEDS)
    assert captured["cell"] == doc["cell"]
