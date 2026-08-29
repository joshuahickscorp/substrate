"""Method extension: seal the five witnesses and prove they reject the four defects they exist for.

D16, D17, D18 and the reset alignment defect are each rebuilt here with the numbers that produced them, and
each must be rejected by the witness that now owns that class.

House style: no dashes.
"""

from __future__ import annotations

import time

import numpy as np

from mop.temporal import io, witness as W


def mutations() -> dict:
    res = {}

    # D16: adapting to B improved A, so no boundary was crossed
    d16 = W.boundary_crossing(
        context_a={"n": 900, "speaker_group": "half_one"}, context_b={"n": 900, "speaker_group": "half_one"},
        transition_index=700, adapt_window=(700, 820), baseline_on_a=0.68154, baseline_on_b=0.68699,
        construct="speaker_group")
    d16_ok = W.boundary_crossing(
        context_a={"n": 900, "speaker_group": "half_one"}, context_b={"n": 900, "speaker_group": "half_two"},
        transition_index=700, adapt_window=(700, 820), baseline_on_a=0.6578, baseline_on_b=0.4897,
        construct="speaker_group")
    res["D16_context_split_that_crossed_no_boundary"] = {
        "expected": "invalid_no_boundary_crossing", "actual": d16["classification"],
        "a_real_boundary_still_passes": d16_ok["classification"] == "boundary_crossed",
        "pass": d16["classification"] == "invalid_no_boundary_crossing"
        and d16_ok["classification"] == "boundary_crossed"}

    # the transition must sit inside the adaptation window
    outside = W.boundary_crossing(
        context_a={"n": 9, "g": "a"}, context_b={"n": 9, "g": "b"}, transition_index=100,
        adapt_window=(200, 300), baseline_on_a=0.7, baseline_on_b=0.4, construct="g")
    res["transition_outside_the_adaptation_window"] = {
        "expected": "invalid_no_boundary_crossing", "actual": outside["classification"],
        "pass": outside["classification"] == "invalid_no_boundary_crossing"}

    # D17: a curve still moving 0.016 is unconverged
    d17 = W.plateau_validity({350: 0.872, 700: 0.8732, 1100: 0.8515, 1600: 0.8696, 2200: 0.878})
    res["D17_curve_still_moving_is_unconverged"] = {
        "expected": "unconverged", "actual": d17["classification"],
        "second_half_movement": d17["second_half_movement"], "pass": d17["classification"] == "unconverged"}

    for name, curve, want in (
        ("known_plateau", {1: 0.80, 2: 0.806, 4: 0.807, 8: 0.807, 16: 0.807}, "converged"),
        ("slow_monotonic_improvement", {1: 0.60, 2: 0.65, 4: 0.70, 8: 0.75, 16: 0.80}, "unconverged"),
        ("temporary_plateau_then_improvement", {1: 0.60, 2: 0.70, 4: 0.70, 8: 0.70, 16: 0.79}, "unconverged"),
        ("overfitting_curve", {1: 0.60, 2: 0.75, 4: 0.82, 8: 0.74, 16: 0.66}, "unconverged"),
        ("high_variance_curve", {1: 0.70, 2: 0.62, 4: 0.78, 8: 0.61, 16: 0.77}, "unconverged"),
    ):
        r = W.plateau_validity(curve)
        res[f"plateau_{name}"] = {"expected": want, "actual": r["classification"], "pass": r["classification"] == want}

    # the reset alignment defect: period three on a three segment stream is an oracle segmentation
    idx3 = W.reset_indices_for("fixed_period", 192, 64, period=64)
    a3 = W.reset_alignment(idx3, [64, 128], 192)
    periods = W.coprime_periods(192, 64, 2, lo=32, hi=83)
    am = W.reset_alignment(W.reset_indices_for("fixed_period", 192, 64, period=periods[0]), [64, 128], 192)
    res["reset_period_three_is_oracle_segmented"] = {
        "expected": "oracle_segmented", "actual": a3["classification"],
        "exact_fraction": a3["exact_fraction"],
        "misaligned_replacement": {"period": periods[0], "classification": am["classification"],
                                   "exact_fraction": am["exact_fraction"]},
        "pass": a3["classification"] == "oracle_segmented" and am["classification"] == "misaligned"}

    tb = W.reset_alignment(W.reset_indices_for("true_boundary", 192, 64), [64, 128], 192)
    wb = W.reset_alignment(W.reset_indices_for("wrong_boundary", 192, 64), [64, 128], 192)
    res["true_boundary_differs_from_wrong_boundary"] = {
        "expected": "true is oracle segmented and wrong is not",
        "true": tb["classification"], "wrong": wb["classification"],
        "pass": tb["classification"] == "oracle_segmented" and wb["classification"] == "misaligned"}

    rng = np.random.default_rng(0)
    fixed = W.reset_indices_for("fixed_period", 192, 64, period=periods[0])
    rnd = W.reset_indices_for("random_rate_matched", 192, 64, rate=len(fixed) / 192, rng=rng)
    res["random_reset_rate_matches_fixed_reset"] = {
        "expected": "equal counts", "fixed": len(fixed), "random": len(rnd),
        "pass": abs(len(fixed) - len(rnd)) <= 1}

    coprime = W.coprime_periods(192, 64, 2, lo=32, hi=83)
    res["two_coprime_misaligned_schedules_exist"] = {
        "expected": "two periods coprime with the segment length and with each other",
        "periods": coprime,
        "pass": len(coprime) >= 2 and all(
            W.reset_alignment(W.reset_indices_for("fixed_period", 192, 64, period=p), [64, 128], 192)[
                "classification"] == "misaligned" for p in coprime)}

    # D18: a permutation control scored against zero rather than the majority class expectation
    naive = W.null_reference("zero", observed=0.04746, reference=0.0, band=0.0419)
    correct_fast = W.null_reference("majority_class", observed=0.2228, reference=0.2123, band=0.05)
    correct_pooled = W.null_reference("majority_class", observed=0.1871, reference=0.2123, band=0.05)
    res["D18_permutation_scored_against_zero"] = {
        "expected": "the zero reference is violated and the majority class reference is satisfied",
        "zero_reference": naive["classification"],
        "majority_reference_fast": correct_fast["classification"],
        "majority_reference_pooled": correct_pooled["classification"],
        "pass": naive["classification"] == "null_reference_violated"
        and correct_fast["behaves_as_a_null"] and correct_pooled["behaves_as_a_null"]}

    bad = W.null_reference("vibes", observed=1.0, reference=0.0)
    res["undeclared_null_reference_is_rejected"] = {
        "expected": "invalid_null_reference", "actual": bad["classification"],
        "pass": bad["classification"] == "invalid_null_reference"}

    # causal history: a hidden difference in what the past looks like
    hid = W.history_witness({
        "recurrent": {"kinds": ["state_carried_from_previous_observations"], "effective_horizon": 192},
        "stateless": {"kinds": ["last_k_observations"], "k": 5, "effective_horizon": 5},
    })
    res["unmatched_history_is_visible"] = {
        "expected": "two distinct history profiles and a matched check that says no",
        "profiles": hid["distinct_history_profiles"],
        "matched": W.matched_information(hid["arms"]["recurrent"], hid["arms"]["stateless"])["matched"],
        "pass": hid["distinct_history_profiles"] == 2 and not W.matched_information(
            hid["arms"]["recurrent"], hid["arms"]["stateless"])["matched"]}

    leak = W.history_witness({"leaky": {"kinds": ["future_information"], "effective_horizon": 1}})
    res["future_information_is_rejected"] = {
        "expected": "violation", "actual": leak["violations"], "pass": bool(leak["violations"])}

    undeclared = W.history_witness({"vague": {"kinds": ["last_k_observations"], "effective_horizon": 5}})
    res["last_k_without_a_k_is_rejected"] = {
        "expected": "violation", "actual": undeclared["violations"], "pass": bool(undeclared["violations"])}

    res["all_pass"] = all(v["pass"] for v in res.values() if isinstance(v, dict))
    return res


def main():
    t0 = time.time()
    mut = mutations()
    io.seal("MOP_TEMPORAL_METHOD_EXTENSION_MUTATIONS.json",
            {"schema": "mop-temporal-method-extension-mutations/v1", **mut})
    io.seal("MOP_TEMPORAL_METHOD_EXTENSION.json", {
        "schema": "mop-temporal-method-extension/v1",
        "witnesses": {
            "boundary_crossing": {
                "implementation": "mop.temporal.witness.boundary_crossing",
                "proves": ["context a exists", "context b exists", "a and b differ under the construct",
                           "the evaluated transition crosses a to b",
                           "baseline behaviour changes across the transition",
                           "the adaptation window contains the transition"],
                "owns_defect": "D16",
            },
            "reset_alignment": {
                "implementation": "mop.temporal.witness.reset_alignment",
                "records": ["reset indices", "true segment boundaries", "distance to nearest boundary",
                            "fraction exactly aligned", "fraction within tolerance",
                            "expected random alignment"],
                "owns_defect": "reset alignment",
                "rule": ("a destructive reset control fails admission when it accidentally functions as an "
                         "oracle boundary reset. Multiple coprime intervals are used and period three is "
                         "retained only as a historical defect mutation"),
            },
            "plateau_validity": {
                "implementation": "mop.temporal.witness.plateau_validity",
                "requires": ["short budget curve", "medium budget curve", "long budget curve",
                             "largest feasible validation budget", "residual slope", "plateau threshold",
                             "selected checkpoint"],
                "owns_defect": "D17",
                "tightening": (
                    "this is stricter than the reformation criterion, which accepted a curve still moving "
                    "0.016 on the argument that the movement sat inside the noise. Here that curve is "
                    "unconverged and the remedy is a longer budget rather than a looser rule. The prior "
                    "receipt stays sealed and untouched; this is an append only tightening"
                ),
            },
            "null_reference": {
                "implementation": "mop.temporal.witness.null_reference",
                "references": list(W.NULL_REFERENCES),
                "owns_defect": "D18",
            },
            "causal_history": {
                "implementation": "mop.temporal.witness.history_witness",
                "kinds": list(W.HISTORY_KINDS),
                "forbidden": list(W.FORBIDDEN_HISTORY),
                "rule": "hidden differences in historical information are rejected",
            },
        },
        "mutations_all_pass": mut["all_pass"],
        "n_mutations": len([k for k, v in mut.items() if isinstance(v, dict)]),
        "wall_seconds": round(time.time() - t0, 1),
    })
    print(f"method extension: {len([k for k, v in mut.items() if isinstance(v, dict)])} mutations, "
          f"all pass {mut['all_pass']}", flush=True)
    for k, v in mut.items():
        if isinstance(v, dict) and not v["pass"]:
            print(f"  FAIL {k}: {v}", flush=True)
    print("METHOD_EXT_DONE", flush=True)


if __name__ == "__main__":
    main()
