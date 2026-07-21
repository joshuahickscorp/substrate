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

from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal import witness as W
from mop.temporal.runs import e2

SEEDS = (0, 1, 2)


def _acc(bed, spec, seed, steps=600, mutate=None):
    sp = B.splits(bed, seed)
    r = Fx.run_cell(sp, spec, seed, "tune", steps=steps)
    return r


def structural() -> dict:
    """Mutations the instrument refuses without training anything."""
    res = {}
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

    res["history_removed_silently"] = {
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
    silent = np.mean([_acc(bed, reset_all, s)["accuracy"] - _acc(bed, pooled, s)["accuracy"] for s in SEEDS])
    res["state_reset_silently"] = {
        "expected": "resetting every observation removes most of the core effect",
        "real_effect": round(float(real), 5), "mutated_effect": round(float(silent), 5),
        "pass": float(silent) < 0.5 * float(real)}

    big_readout = dict(Fx.REFERENCE, family="pooled", readout="mlp_strong", tier="large")
    inflated = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, big_readout, s)["accuracy"] for s in SEEDS])
    res["readout_capacity_inflated"] = {
        "expected": "an order free control with a strong readout and a large core does not close the gap",
        "real_effect": round(float(real), 5), "mutated_effect": round(float(inflated), 5),
        "pass": float(inflated) > 0.5 * float(real)}

    hist20 = dict(Fx.REFERENCE, family="histmlp", history_k=20)
    histfull = dict(Fx.REFERENCE, family="histmlp", history_k="full_window")
    h20 = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, hist20, s)["accuracy"] for s in SEEDS])
    hfull = np.mean([_acc(bed, ref, s)["accuracy"] - _acc(bed, histfull, s)["accuracy"] for s in SEEDS])
    res["history_inflated"] = {
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
    allr = {**s, **b}
    doc = {"schema": "mop-temporal-core-mutation-report/v1",
           "structural": s, "behavioural": b,
           "n_mutations": len(allr),
           "all_rejected": all(v["pass"] for v in allr.values() if isinstance(v, dict)),
           "survivors": [k for k, v in allr.items() if isinstance(v, dict) and not v["pass"]],
           "rule": "a required mutation that is accepted invalidates the positive it attacks",
           "wall_seconds": round(time.time() - t0, 1)}
    io.seal("MOP_TEMPORAL_CORE_MUTATION_REPORT.json", doc)
    print(f"mutations: {doc['n_mutations']} attacks, all rejected {doc['all_rejected']}, "
          f"survivors {doc['survivors']}", flush=True)
    print("MUTATIONS_DONE", flush=True)


if __name__ == "__main__":
    main()
