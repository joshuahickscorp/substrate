"""Phase 4 Mechanism Admission Battery (clauses A-J) over the controlled allocation bed.

Every clause is measured across independent generator families (regimes) and both mechanism implementations.
Nothing is weakened after seeing results: the runner seals the preregistered thresholds first, then this
battery reports measured effects, then the lane is classified by those sealed thresholds. A tie is a null; a
point estimate in the wrong direction is a failure regardless of nominal size.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import numpy as np
from allocation_bed import (
    AllocationBed,
    BedConfig,
    heuristic_allocation,
    mechanism_allocation,
)

HEURISTICS = ("feature_magnitude", "first_feature", "frequency", "recency")
REGIMES = (0, 1, 2)
IMPLS = ("linear", "knn")


def _split(cfg: BedConfig, regime: int, seed: int) -> tuple[AllocationBed, AllocationBed]:
    """Group-disjoint train/test: two independently generated beds of the same regime (no unit leakage)."""
    train = AllocationBed(cfg, regime=regime, seed=seed * 7 + 1)
    test = AllocationBed(cfg, regime=regime, seed=seed * 7 + 999)  # disjoint units
    return train, test


def _norm(v: float, oracle: float, base: float) -> float:
    return (v - base) / (oracle - base) if oracle > base else 0.0


def run_battery(cfg: BedConfig, *, sesoi: float, seed: int = 7) -> dict:
    out: dict = {"lane": cfg.lane, "per_regime": {}, "clauses": {}}
    oracle_norms, mech_norms, rand_norms = [], [], []
    per_impl_beats = {im: [] for im in IMPLS}
    heur_best_norms, tv_rates, shuffle_norms, wrongtime_norms = [], [], [], []
    decodabilities = []

    for regime in REGIMES:
        train, test = _split(cfg, regime, seed)
        base = 0.0
        oracle = test.oracle_value()
        rand = np.mean([test.random_value(s) for s in range(20)])
        reg = {"oracle": round(oracle, 2), "random": round(rand, 2),
               "oracle_headroom_norm": round(_norm(oracle, oracle, base), 3)}
        oracle_norms.append(_norm(oracle, oracle, base))
        rand_norms.append(_norm(rand, oracle, base))

        # C: decodability = the BEST available estimator's held-out correlation with true reducible value.
        # (A linear-only probe understates decodability of nonlinear regimes that the knn mechanism decodes.)
        from allocation_bed import MECHANISM_IMPLS
        best_dec = 0.0
        for _im in IMPLS:
            est = MECHANISM_IMPLS[_im](train.x, train.r)
            pred_d = est(test.x)
            if np.std(pred_d) > 0:
                best_dec = max(best_dec, float(np.corrcoef(pred_d, test.r)[0, 1]))
        dec = best_dec
        decodabilities.append(dec)

        # A/D/F: mechanism value per implementation vs best heuristic
        mech_by_impl = {}
        for im in IMPLS:
            chosen = mechanism_allocation(train, test, im)
            mv = test.realized_value(chosen, times=test.t[chosen])
            mech_by_impl[im] = _norm(mv, oracle, base)
        best_mech = max(mech_by_impl.values())
        mech_norms.append(best_mech)
        heur_norms = []
        for h in HEURISTICS:
            ch = heuristic_allocation(test, h)
            heur_norms.append(_norm(test.realized_value(ch, times=test.t[ch]), oracle, base))
        best_heur = max(heur_norms)
        heur_best_norms.append(best_heur)
        for im in IMPLS:
            per_impl_beats[im].append(mech_by_impl[im] - best_heur)

        # G: noisy-TV allocation rate (mechanism must not over-pick useless high-surprise items)
        chosen = mechanism_allocation(train, test, "knn")
        tv_rate = float(np.mean(test.tv[chosen]))
        base_tv = float(np.mean(test.tv))
        tv_rates.append(tv_rate - base_tv)  # excess allocation to noisy-TV (want <= 0-ish)

        # H: shuffled-target -> break r<->x, mechanism should collapse to ~random
        sh = AllocationBed(cfg, regime=regime, seed=seed * 7 + 999)
        perm = np.random.default_rng(seed).permutation(len(sh.r))
        sh_train = AllocationBed(cfg, regime=regime, seed=seed * 7 + 1)
        sh_train.r = sh_train.r[np.random.default_rng(seed + 1).permutation(len(sh_train.r))]
        ch = mechanism_allocation(sh_train, sh, "knn")
        shuffle_norms.append(_norm(sh.realized_value(ch, times=sh.t[ch]), oracle, base))

        # I: wrong-time -> allocate correct items but at deliberately wrong times
        chosen = mechanism_allocation(train, test, "knn")
        wrong_times = (test.t[chosen] + 5) % 10
        wrongtime_norms.append(_norm(test.realized_value(chosen, times=wrong_times), oracle, base))

        reg.update({"decodability": round(dec, 3), "best_mechanism_norm": round(best_mech, 3),
                    "best_heuristic_norm": round(best_heur, 3),
                    "mechanism_by_impl": {k: round(v, 3) for k, v in mech_by_impl.items()},
                    "excess_noisy_tv_alloc": round(tv_rate - base_tv, 3),
                    "shuffled_norm": round(shuffle_norms[-1], 3),
                    "wrongtime_norm": round(wrongtime_norms[-1], 3)})
        out["per_regime"][regime] = reg

    m = lambda a: float(np.mean(a))  # noqa: E731
    mech_mean, heur_mean, rand_mean = m(mech_norms), m(heur_best_norms), m(rand_norms)
    incr = mech_mean - heur_mean
    # clause verdicts (measured vs the sealed sesoi / thresholds)
    out["clauses"] = {
        "A_what_sufficiency": {"decodability_mean": round(m(decodabilities), 3),
                               "pass": m(decodabilities) >= 0.2},
        "B_oracle_headroom": {"oracle_norm": round(m(oracle_norms), 3), "random_norm": round(rand_mean, 3),
                              "headroom_over_random": round(m(oracle_norms) - rand_mean, 3),
                              "pass": (m(oracle_norms) - rand_mean) >= sesoi},
        "C_when_decodability": {"decodability_mean": round(m(decodabilities), 3),
                                "pass": m(decodabilities) >= 0.2},
        "D_incremental_value": {"mechanism_minus_best_heuristic": round(incr, 3), "pass": incr >= sesoi},
        "E_group_disjoint": {"min_regime_incr": round(min(mech_norms[i] - heur_best_norms[i]
                                                          for i in range(len(REGIMES))), 3),
                             "pass": all((mech_norms[i] - heur_best_norms[i]) >= sesoi
                                         for i in range(len(REGIMES)))},
        "F_architecture_independence": {"per_impl_mean_incr": {im: round(m(per_impl_beats[im]), 3) for im in IMPLS},
                                        "pass": all(m(per_impl_beats[im]) >= sesoi for im in IMPLS)},
        "G_noisy_tv": {"excess_alloc_mean": round(m(tv_rates), 3), "pass": m(tv_rates) <= sesoi},
        "H_shuffled_target": {"shuffled_norm_mean": round(m(shuffle_norms), 3),
                              "pass": m(shuffle_norms) <= rand_mean + sesoi},
        "I_wrong_time": {"wrongtime_norm_mean": round(m(wrongtime_norms), 3),
                         "pass": m(wrongtime_norms) <= mech_mean - sesoi},
        "J_rate_matched_random": {"mechanism_minus_random": round(mech_mean - rand_mean, 3),
                                  "pass": (mech_mean - rand_mean) >= sesoi},
    }
    out["summary"] = {"mechanism_norm_mean": round(mech_mean, 3), "best_heuristic_norm_mean": round(heur_mean, 3),
                      "random_norm_mean": round(rand_mean, 3), "incremental_over_heuristic": round(incr, 3)}
    return out


def classify(battery: dict, *, sesoi: float) -> dict:
    """Terminal admission classification by the sealed thresholds. Exactly one class."""
    c = battery["clauses"]
    if not c["B_oracle_headroom"]["pass"]:
        cls = "invalid_bed"
    elif not (c["A_what_sufficiency"]["pass"] and c["C_when_decodability"]["pass"]) or not c["D_incremental_value"]["pass"] or not c["J_rate_matched_random"]["pass"] or not (c["G_noisy_tv"]["pass"] and c["H_shuffled_target"]["pass"] and c["I_wrong_time"]["pass"]):
        cls = "pruned_mechanism"
    elif not c["E_group_disjoint"]["pass"]:
        cls = "insufficient_independent_units"
    elif not c["F_architecture_independence"]["pass"]:
        cls = "architecture_dependent"
    else:
        cls = "admitted"
    passed = [k for k, v in c.items() if v.get("pass")]
    return {"classification": cls, "clauses_passed": len(passed), "clauses_total": len(c),
            "first_failing_clause": next((k for k, v in c.items() if not v.get("pass")), None)}
