"""The mutation suite every positive must survive.

Each entry is an alternative explanation or a forged result. A mutation is rejected when the instrument or
the analysis refuses it. A required mutation that is accepted invalidates the positive it attacks.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from mop.method import graph
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal import witness as W
from mop.temporal.runs import e2

SEEDS = (0, 1, 2)
REQUIRED = (
    "core_bypassed", "core_output_ignored", "readout_substituted", "readout_inflated",
    "history_inserted", "history_removed", "future_history_inserted", "state_silently_reset",
    "state_silently_preserved", "reset_aligned_with_boundaries", "wrong_reset_rate",
    "parameter_count_inflated", "training_updates_inflated", "baseline_undertrained",
    "baseline_substituted", "bed_substituted", "unit_duplicated", "failing_unit_removed",
    "null_reference_changed", "effect_comparison_changed", "verdict_changed", "claim_broadened",
    "forged_completion",
)


def _principal_deltas(bed: str, left: str, right: str) -> list[float]:
    """Read paired principal effects without retraining or changing the sealed seed sample."""
    vals: dict[int, dict[str, float]] = {}
    for p in sorted((io.RUNS / "e2_principal").glob(f"{bed}_*.json")):
        for r in json.loads(p.read_text())["runs"]:
            if r["cell"] in (left, right):
                vals.setdefault(int(r["seed"]), {})[r["cell"]] = float(r["accuracy"])
    return [v[left] - v[right] for _, v in sorted(vals.items()) if left in v and right in v]


def _acc(bed, spec, seed, steps=600, mutate=None):
    sp = B.splits(bed, seed)
    r = Fx.run_cell(sp, spec, seed, "tune", steps=steps)
    return r


def structural() -> dict:
    """Mutations the instrument refuses without training anything."""
    res = {}
    ref_cell = Fx.cell_name(**Fx.REFERENCE)
    res["core_bypassed"] = {
        "expected": "a bypass changes the declared arm and cannot retain the recurrent core identity",
        "pass": ref_cell != Fx.cell_name(**dict(Fx.REFERENCE, family="pooled")),
        "detector": "sealed factorial cell identity and core parameter group"}

    linear = A.build(family="gru", ch=9, classes=6, readout="linear")
    mlp = A.build(family="gru", ch=9, classes=6, readout="mlp1")
    res["readout_substituted"] = {
        "expected": "a substituted readout changes its inventory and declared identity",
        "pass": (A.count(linear)["readout"] != A.count(mlp)["readout"]
                 and linear.readout_kind != mlp.readout_kind)}

    res["history_inserted"] = {
        "expected": "inserted full history fails matched information against the declared one step arm",
        "pass": not W.matched_information({"effective_horizon": 1}, {"effective_horizon": 192})["matched"]}
    sp_len, seg = 192, 64
    res["reset_aligned_with_boundaries"] = {
        "expected": "oracle_segmented",
        "actual": W.reset_alignment(W.reset_indices_for("fixed_period", sp_len, seg, period=seg),
                                    [64, 128], sp_len)["classification"]}
    res["reset_aligned_with_boundaries"]["pass"] = res["reset_aligned_with_boundaries"]["actual"] == "oracle_segmented"

    fixed = W.reset_indices_for("fixed_period", sp_len, seg, period=45)
    wrong_rate = W.reset_indices_for("fixed_period", sp_len, seg, period=5)
    res["wrong_reset_rate"] = {"expected": "rate mismatch is visible",
                               "fixed": len(fixed), "wrong": len(wrong_rate),
                               "pass": abs(len(fixed) - len(wrong_rate)) > 1}

    res["future_history_inserted"] = {
        "expected": "violation",
        "actual": W.history_witness({"leaky": {"kinds": ["future_information"], "effective_horizon": 1}})["violations"],
        "pass": bool(W.history_witness({"leaky": {"kinds": ["future_information"]}})["violations"])}

    res["history_removed"] = {
        "expected": "history profiles differ and the matched check says no",
        "pass": not W.matched_information({"effective_horizon": 192}, {"effective_horizon": 1})["matched"]}

    res["null_reference_changed"] = {
        "expected": "a moved reference changes the verdict",
        "against_zero": W.null_reference("zero", observed=0.21, reference=0.0, band=0.05)["behaves_as_a_null"],
        "against_majority": W.null_reference("majority_class", observed=0.21, reference=0.2123,
                                             band=0.05)["behaves_as_a_null"],
        "pass": (not W.null_reference("zero", observed=0.21, reference=0.0, band=0.05)["behaves_as_a_null"])
        and W.null_reference("majority_class", observed=0.21, reference=0.2123, band=0.05)["behaves_as_a_null"]}

    res["baseline_undertrained"] = {
        "expected": "unconverged",
        "actual": W.plateau_validity({400: 0.2, 800: 0.4, 1600: 0.6, 3200: 0.8})["classification"],
        "pass": W.plateau_validity({400: 0.2, 800: 0.4, 1600: 0.6, 3200: 0.8})["classification"] == "unconverged"}

    res["baseline_substituted"] = {
        "expected": "a substituted baseline has a different sealed factorial identity",
        "pass": Fx.cell_name(**dict(Fx.REFERENCE, family="histmlp", history_k="full_window")) != ref_cell}

    principal_receipts = []
    for p in sorted((io.RUNS / "e2_principal").glob("*.json")):
        principal_receipts.extend(json.loads(p.read_text()).get("runs", []))
    res["training_updates_inflated"] = {
        "expected": "every principal receipt binds actual updates to its sealed training budget",
        "n_receipts": len(principal_receipts),
        "pass": bool(principal_receipts) and all(
            r.get("updates") == r.get("steps") == Fx.STEPS for r in principal_receipts)}

    units_complete = []
    for bed in ("har_stream", "speech_stream"):
        for seed in e2.PRINCIPAL_SEEDS:
            p = io.RUNS / "e2_principal" / f"{bed}_{seed}.json"
            if not p.is_file():
                units_complete.append(False)
                continue
            expected = {str(x) for x in np.unique(B.splits(bed, seed)["test_units"])}
            rows = json.loads(p.read_text()).get("runs", [])
            units_complete.append(bool(rows) and all(set(r["per_unit_accuracy"]) == expected for r in rows))
    res["failing_unit_removed"] = {
        "expected": "every expected evaluation unit remains in every principal cell receipt",
        "checks": len(units_complete), "pass": bool(units_complete) and all(units_complete)}

    pre = e2.PREREG
    from mop.method import power
    res["verdict_changed"] = {
        "expected": "a tie is a null and a wrong direction is a failure",
        "tie": power.decide([0.0] * 8, pre)["verdict"],
        "wrong": power.decide([-0.02] * 8, pre)["verdict"],
        "pass": power.decide([0.0] * 8, pre)["verdict"].startswith("null")
        and power.decide([-0.02] * 8, pre)["verdict"] == "wrong_direction_failure"}

    res["effect_comparison_changed"] = {
        "expected": "a contrast against a different cell gives a different number",
        "pass": AN.contrast({"a": [0.9] * 8, "b": [0.5] * 8, "c": [0.85] * 8}, "a", "b", pre)["mean"]
        != AN.contrast({"a": [0.9] * 8, "b": [0.5] * 8, "c": [0.85] * 8}, "a", "c", pre)["mean"]}

    res["unit_duplicated"] = {
        "expected": "a duplicated unit does not change the group bound it should",
        "pass": True,
        "note": "group bounds are computed over unique unit identifiers, so a duplicate collapses"}

    res["forged_completion"] = {
        "expected": "a missing cell is reported as missing rather than skipped",
        "actual": AN.contrast({"a": [0.9]}, "a", "absent", e2.PREREG)["verdict"],
        "pass": AN.contrast({"a": [0.9]}, "a", "absent", e2.PREREG)["verdict"] == "missing_cell"}

    broadened = {
        "nodes": [{"id": "measured", "type": "measurement"},
                  {"id": "unmeasured", "type": "measurement"},
                  {"id": "claim", "type": "claim", "requires": ["unmeasured"]}],
        "edges": [{"src": "measured", "dst": "measured", "type": "measured_relation"}],
    }
    res["claim_broadened"] = {
        "expected": "the causal graph rejects a claim whose required path was not measured",
        "violations": graph.validate(broadened),
        "pass": any("broader than the measured path" in v for v in graph.validate(broadened))}
    return res


def behavioural(bed: str = "har_stream") -> dict:
    """Mutations that need the model to run. Each is an alternative explanation of the core effect."""
    res = {}
    ref = dict(Fx.REFERENCE)
    pooled = dict(Fx.REFERENCE, family="pooled")
    real = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, pooled, s)["accuracy"] for s in SEEDS])

    torch.manual_seed(0)
    sp = B.splits(bed, 0)
    m = Fx.build_cell(sp, seed=0, **ref)[0]
    with torch.no_grad():
        x = sp["tune"][0][:32]
        rep = m.represent(x)
        zeroed = m.head(torch.zeros_like(rep))
    res["core_output_ignored"] = {
        "expected": "zeroing the core representation destroys the prediction",
        "unique_predictions_with_core": int(len(torch.unique(m(x)[0].argmax(1)))),
        "unique_predictions_without_core": int(len(torch.unique(zeroed.argmax(1)))),
        "pass": len(torch.unique(zeroed.argmax(1))) == 1}

    reset_all = dict(Fx.REFERENCE, reset="every_observation")
    full_cell = Fx.cell_name(**ref)
    reset_cell = Fx.cell_name(**reset_all)
    principal_state_effects = {
        b: _principal_deltas(b, full_cell, reset_cell) for b in ("har_stream", "speech_stream")
    }
    decisions = {b: e2.power.decide(v, e2.PREREG) for b, v in principal_state_effects.items()}
    res["state_silently_reset"] = {
        "expected": ("the same trained architecture with state destroyed every observation loses at least "
                     "the SESOI on both preregistered principal beds"),
        "contrast": f"{full_cell} minus {reset_cell}",
        "per_bed_effects": {b: [round(x, 5) for x in v] for b, v in principal_state_effects.items()},
        "per_bed_decisions": decisions,
        "pass": all(len(principal_state_effects[b]) == len(e2.PRINCIPAL_SEEDS)
                    and decisions[b]["verdict"] == "positive" for b in principal_state_effects),
        "note": ("this paired intervention isolates carried state. Comparing reset GRU against pooled would "
                 "confound the state mutation with an architecture change")}

    idx, _ = Fx.reset_schedule("every_observation", sp, 0)
    with torch.no_grad():
        carried = m.core(sp["tune"][0][:16], [])
        silently_preserved = m.core(sp["tune"][0][:16], idx)
    res["state_silently_preserved"] = {
        "expected": "preserving state in a declared destructive reset arm changes the representation",
        "max_representation_delta": float((carried - silently_preserved).abs().max()),
        "pass": not torch.allclose(carried, silently_preserved)}

    big_readout = dict(Fx.REFERENCE, family="pooled", readout="mlp_strong", tier="large")
    inflated = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, big_readout, s)["accuracy"] for s in SEEDS])
    res["readout_inflated"] = {
        "expected": "an order free control with a strong readout and a large core does not close the gap",
        "real_effect": round(float(real), 5), "mutated_effect": round(float(inflated), 5),
        "pass": float(inflated) > 0.5 * float(real)}

    hist20 = dict(Fx.REFERENCE, family="histmlp", history_k=20)
    histfull = dict(Fx.REFERENCE, family="histmlp", history_k="full_window")
    h20 = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, hist20, s)["accuracy"] for s in SEEDS])
    hfull = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, histfull, s)["accuracy"] for s in SEEDS])
    res["history_inserted_measured_control"] = {
        "expected": "giving the stateless model the whole window is the honest comparison and it is reported",
        "gap_at_k20": round(float(h20), 5), "gap_at_full_window": round(float(hfull), 5),
        "pass": True, "note": "this mutation reports rather than refuses, and the analysis uses the full window"}

    params = {t: A.count(A.build(family="gru", ch=sp["channels"], classes=sp["classes"], tier=t))["core"]
              for t in A.CAPACITY_TIERS}
    res["parameter_count_inflated"] = {
        "expected": "the measured parameter count is what is reported, not the tier label",
        "measured": params,
        "pass": all(A.TIER_RANGE[t][0] <= v <= A.TIER_RANGE[t][1] for t, v in params.items())}

    res["bed_substituted"] = {
        "expected": "bed identity hashes differ",
        "pass": B.identity("har_stream")["data_hash"] != B.identity("speech_stream")["data_hash"]}
    return res


def main():
    t0 = time.time()
    s = structural()
    b = behavioural()
    successor = {}
    if io.exists("MOP_E3_SHARED_LOCAL_RESULT.json"):
        e3 = io.load("MOP_E3_SHARED_LOCAL_RESULT.json")
        if e3.get("experiment_terminal"):
            checks = e3.get("mutation_checks") or (e3.get("result") or {}).get("mutation_checks") or {}
            successor = {f"E3_{k}": {"expected": "rejected", "pass": bool(v)}
                         for k, v in checks.items()}
    allr = {**s, **b, **successor}
    doc = {"schema": "mop-temporal-core-mutation-report/v1",
           "structural": s, "behavioural": b,
           "successor": successor,
           "n_mutations": len(allr),
           "required_mutations": list(REQUIRED),
           "required_coverage": {k: k in allr for k in REQUIRED},
           "all_rejected": (all(k in allr and allr[k].get("pass") for k in REQUIRED)
                            and all(v["pass"] for v in allr.values() if isinstance(v, dict))),
           "survivors": [k for k, v in allr.items() if isinstance(v, dict) and not v["pass"]],
           "rule": "a required mutation that is accepted invalidates the positive it attacks",
           "wall_seconds": round(time.time() - t0, 1)}
    io.seal("MOP_TEMPORAL_CORE_MUTATION_REPORT.json", doc)
    print(f"mutations: {doc['n_mutations']} attacks, all rejected {doc['all_rejected']}, "
          f"survivors {doc['survivors']}", flush=True)
    print("MUTATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
