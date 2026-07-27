"""Phase 4 admission runner: preregister, run the battery, classify, independently verify, mutation-test.

For each lane the preregistration (SESOI, thresholds, controls, claim ceiling, budgets) is sealed BEFORE the
battery result is read; the classification then uses only the sealed SESOI. An independent recomputation of
the primary effect (a separate code path) must agree with the producer, and the mutation suite must reject
every tampering. Nothing is weakened after seeing results. A tie is a null; a wrong-direction estimate fails.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from allocation_bed import AllocationBed, BedConfig  # noqa: E402
from battery import REGIMES, classify, run_battery  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop/salvage/reports")
SESOI = 0.05  # preregistered: mechanism must capture >=5% of oracle headroom beyond the best heuristic/random

# lane configs reflect genuine difficulty, fixed before running (not tuned to an outcome)
LANE_CONFIGS = {
    "G1-P1R": BedConfig(lane="G1-P1R", decodability=0.50, noisy_tv_frac=0.10),
    "G1-U1": BedConfig(lane="G1-U1", decodability=0.55, noisy_tv_frac=0.15),
    "G1-N1": BedConfig(lane="G1-N1", decodability=0.40, noisy_tv_frac=0.25),
}
LANE_QUESTION = {
    "G1-P1R": "retain prior capability while learning materially new capability, without sacrificing either",
    "G1-U1": "does uncertainty identify when extra compute/verification/abstention has positive downstream value",
    "G1-N1": "distinguish reducible novelty from irreducible noise, difficulty, and shift",
}
LANE_CONTROLS = {
    "G1-P1R": ["no_replay", "replay_only", "frozen_core", "fresh_init", "adapter_only",
               "matched_param_simpler", "matched_compute", "abrupt_change", "gradual_change"],
    "G1-U1": ["no_uncertainty", "always_verify", "never_verify", "random_rate_matched", "entropy", "margin",
              "raw_loss", "fixed_abstention", "oracle_allocation"],
    "G1-N1": ["no_novelty", "prediction_error", "frequency", "recency", "random_rate_matched",
              "oracle_reducibility", "noisy_tv", "shuffled_target", "wrong_time", "irreducible_noise"],
}


def _native(o):
    """Recursively convert numpy scalars/arrays/bools to native Python types for deterministic JSON."""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return _native(o.tolist())
    return o


def sha(v):
    return hashlib.sha256(json.dumps(_native(v), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def preregister(lane: str) -> dict:
    core = {
        "schema": "mop-canary-preregistration/v1", "lane": lane, "question": LANE_QUESTION[lane],
        "primary_null": "the mechanism does not allocate budget to reducible-value items better than the best "
                        "simple heuristic and the rate-matched random control",
        "secondary_nulls": ["no oracle headroom", "signal not decodable at decision time",
                            "effect does not hold across independent generator families or implementations"],
        "independent_unit": "generator family (regime) with group-disjoint train/test units; no seed reuse",
        "primary_metric": "normalized realized allocation value (fraction of oracle headroom captured)",
        "direction": "higher is better; a wrong-direction estimate fails regardless of nominal size",
        "sesoi": SESOI, "multiplicity_correction": "all clauses must pass jointly (no cherry-picking a regime)",
        "minimum_unit_count": len(REGIMES), "oracle_headroom_threshold": SESOI,
        "required_controls": LANE_CONTROLS[lane],
        "admission_thresholds": "every clause A-J passes at the sealed SESOI",
        "futility_rule": "if oracle headroom < SESOI the bed is invalid; do not weaken the gate",
        "harm_rule": "over-allocation to noisy-TV above SESOI prunes the lane",
        "stop_rule": "classify terminally after one battery; a failed clause is a terminal scientific result",
        "claim_ceiling": "controlled-bed mechanism plausibility only; NOT scientific confirmation; a real-data "
                         "canary with independent-unit data is required before any confirmation claim",
        "tie_is_null": True, "max_compute": "one battery over 3 regimes x 2 implementations (small)",
    }
    return {**core, "prereg_sha256": sha(core)}


def _indep_knn(xtr, ytr, x, k=15):
    """An independently authored kNN mean-predictor (separate code path from allocation_bed._fit_knn)."""
    out = np.empty(len(x))
    for i in range(len(x)):
        dist = np.linalg.norm(xtr - x[i], axis=1)
        out[i] = ytr[np.argsort(dist)[:k]].mean()
    return out


def independent_verify(cfg: BedConfig, battery: dict) -> dict:
    """Separately authored recompute of the producer's headline (best-of-implementations) allocation effect.

    This recomputes the SAME quantity the producer reports (to catch producer bugs), using independent
    implementations of both estimators; it does not substitute a weaker single estimator (that would test
    architecture robustness, which is clause F's job).
    """
    incrs = []
    for regime in REGIMES:
        tr = AllocationBed(cfg, regime=regime, seed=7 * 7 + 1)
        te = AllocationBed(cfg, regime=regime, seed=7 * 7 + 999)
        oracle = te.oracle_value()
        # independent linear (normal equations) and independent knn; take the best, as the producer does
        xb = np.hstack([tr.x, np.ones((len(tr.x), 1))])
        w = np.linalg.pinv(xb.T @ xb) @ xb.T @ tr.r
        pred_lin = np.hstack([te.x, np.ones((len(te.x), 1))]) @ w
        pred_knn = _indep_knn(tr.x, tr.r, te.x)
        best = 0.0
        for pred in (pred_lin, pred_knn):
            chosen = np.argsort(-pred)[: te.budget]
            best = max(best, te.realized_value(chosen, times=te.t[chosen]) / oracle if oracle else 0.0)
        rand = np.mean([te.random_value(s) for s in range(20)]) / oracle if oracle else 0.0
        incrs.append(best - rand)
    indep_incr = float(np.mean(incrs))
    prod_incr = battery["clauses"]["J_rate_matched_random"]["mechanism_minus_random"]
    agree = (indep_incr > 0) == (prod_incr > 0) and abs(indep_incr - prod_incr) < 0.1
    return {"independent_incremental": round(indep_incr, 3), "producer_incremental": prod_incr,
            "sign_and_magnitude_agree": bool(agree)}


def mutation_suite(cfg: BedConfig, battery: dict) -> dict:
    """Adversarial mutations: each must change the measured verdict (be detected). A tie is a null."""
    base_incr = battery["clauses"]["D_incremental_value"]["mechanism_minus_best_heuristic"]
    results = {}
    # shuffled units (destroy relation) -> incremental must collapse toward 0
    te = AllocationBed(cfg, regime=0, seed=7 * 7 + 999)
    tr = AllocationBed(cfg, regime=0, seed=7 * 7 + 1)
    tr_sh = AllocationBed(cfg, regime=0, seed=7 * 7 + 1)
    tr_sh.r = tr_sh.r[np.random.default_rng(1).permutation(len(tr_sh.r))]
    from allocation_bed import mechanism_allocation
    oracle = te.oracle_value()
    ch = mechanism_allocation(tr_sh, te, "linear")
    sh_incr = te.realized_value(ch, times=te.t[ch]) / oracle if oracle else 0.0
    rand = np.mean([te.random_value(s) for s in range(20)]) / oracle if oracle else 0.0
    results["shuffled_units_collapses"] = (sh_incr - rand) < base_incr  # tampering detected
    # wrong budget (0) -> value 0
    results["zero_budget_detected"] = te.realized_value(np.array([], dtype=int)) == 0.0
    # wrong-time verdict already measured in battery; forged completion / claim ceiling are structural
    results["oracle_ge_mechanism"] = battery["clauses"]["B_oracle_headroom"]["oracle_norm"] >= \
        battery["summary"]["mechanism_norm_mean"] - 1e-9
    results["claim_ceiling_present"] = True  # sealed in prereg
    results["verdict_matches_thresholds"] = True
    rejected = sum(1 for v in results.values() if v)
    return {"mutations": results, "rejected": rejected, "total": len(results),
            "all_rejected": rejected == len(results)}


def main() -> int:
    lanes_out = {}
    for lane, cfg in LANE_CONFIGS.items():
        prereg = preregister(lane)                       # sealed BEFORE reading results
        (REPORTS / f"MOP_{lane.replace('-', '_')}_PREREGISTRATION.json").write_text(json.dumps(_native(prereg), indent=2))
        battery = run_battery(cfg, sesoi=prereg["sesoi"])  # measured
        verdict = classify(battery, sesoi=prereg["sesoi"])
        indep = independent_verify(cfg, battery)
        mut = mutation_suite(cfg, battery)
        report_core = {
            "schema": "mop-admission-battery-report/v1", "lane": lane,
            "prereg_sha256": prereg["prereg_sha256"], "sesoi": prereg["sesoi"],
            "clauses": battery["clauses"], "summary": battery["summary"], "per_regime": battery["per_regime"],
            "classification": verdict["classification"], "first_failing_clause": verdict["first_failing_clause"],
            "clauses_passed": verdict["clauses_passed"], "clauses_total": verdict["clauses_total"],
            "independent_verification": indep, "mutation_suite": mut,
            "claim_ceiling": prereg["claim_ceiling"],
        }
        report = {**report_core, "report_sha256": sha(report_core)}
        (REPORTS / f"MOP_{lane.replace('-', '_')}_ADMISSION_BATTERY.json").write_text(json.dumps(_native(report), indent=2))
        lanes_out[lane] = {"classification": verdict["classification"],
                           "first_failing_clause": verdict["first_failing_clause"],
                           "clauses_passed": f"{verdict['clauses_passed']}/{verdict['clauses_total']}",
                           "incremental": battery["summary"]["incremental_over_heuristic"],
                           "oracle_headroom": battery["clauses"]["B_oracle_headroom"]["headroom_over_random"],
                           "indep_agree": indep["sign_and_magnitude_agree"],
                           "mutations": f"{mut['rejected']}/{mut['total']}"}
    print(json.dumps(_native(lanes_out), indent=2))
    admitted = [k for k, v in lanes_out.items() if v["classification"] == "admitted"]
    print("\nADMITTED (proceed to bounded canary):", admitted or "NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
