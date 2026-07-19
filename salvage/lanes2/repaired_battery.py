"""Phase 4B repaired Mechanism Admission Battery.

Fixes the three Phase 4 construct-validity flaws:
 (1) architecture independence is judged only across SUFFICIENTLY CAPABLE, capacity-matched estimators
     (knn, kernel ridge, random-fourier-feature ridge), each tuned on a tuning partition; linear is a simple
     baseline, not the sole architecture, so failing linear is evidence of nonlinearity, not architecture
     dependence.
 (2) many independent generator families (>=10) with a random-effects LOWER confidence bound on the mean
     family effect, replacing the arbitrary all-groups 0.05 cliff, with a reported power/minimum detectable
     effect.
 (3) residual LEARNABLE headroom (oracle minus best simple control) is reported, so a mechanism cannot pass
     merely by exceeding zero when a simple heuristic already captures nearly all oracle value.

A bed exposes independent families, each with group-disjoint train/tune/test splits carrying decision-time
features x, true reducible value r, right-time t, and a noisy-TV mask. The battery evaluates any allocation
selector (mechanism, heuristic, control, oracle) to a per-family normalized value and computes clauses A-J.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from estimators import CAPABLE, select_and_fit  # noqa: E402

SESOI = 0.05


def _realized(split, chosen, times):
    correct = np.abs(times - split["t"][chosen]) <= 1
    return float(np.sum(split["r"][chosen] * correct))


def oracle_value(split, budget):
    order = np.argsort(-split["r"])[:budget]
    return _realized(split, order, split["t"][order])


def random_value(split, budget, seed):
    rng = np.random.default_rng(seed)
    ch = rng.choice(len(split["r"]), size=budget, replace=False)
    return _realized(split, ch, split["t"][ch])


# ---- selectors: each returns chosen indices on the test split of a family ----
def mechanism_selector(name):
    def sel(fam, budget):
        tr, tu, te = fam["train"], fam["tune"], fam["test"]
        pred = select_and_fit(name, tr["x"], tr["r"], tu["x"], tu["r"], te["x"])
        return np.argsort(-pred)[:budget]
    return sel


def heuristic_selector(kind):
    def sel(fam, budget):
        te = fam["test"]
        if kind == "feature_magnitude":
            s = np.linalg.norm(te["x"], axis=1)
        elif kind == "loss_proxy":
            s = np.abs(te["x"][:, 0])
        elif kind == "frequency":
            s = te["x"][:, 1] if te["x"].shape[1] > 1 else te["x"][:, 0]
        elif kind == "recency":
            s = -te["t"].astype(float)
        else:
            s = np.zeros(len(te["r"]))
        return np.argsort(-s)[:budget]
    return sel


def oracle_selector():
    def sel(fam, budget):
        te = fam["test"]
        return np.argsort(-te["r"])[:budget]
    return sel


def random_selector(seed=0):
    def sel(fam, budget):
        te = fam["test"]
        return np.random.default_rng(seed).choice(len(te["r"]), size=budget, replace=False)
    return sel


def shuffled_train_selector(name, seed=0):
    """Destroy r<->x on the training split; a real mechanism must collapse to ~random."""
    def sel(fam, budget):
        tr, tu, te = fam["train"], fam["tune"], fam["test"]
        r_sh = tr["r"][np.random.default_rng(seed).permutation(len(tr["r"]))]
        pred = select_and_fit(name, tr["x"], r_sh, tu["x"], tu["r"], te["x"])
        return np.argsort(-pred)[:budget]
    return sel


def _family_norm_values(bed, selector, seed=7):
    """Per-family normalized realized value (fraction of oracle headroom over random) for a selector."""
    vals = []
    for f in range(bed.n_families):
        fam = bed.family(f, seed)
        budget = fam["budget"]
        orc = oracle_value(fam["test"], budget)
        rnd = np.mean([random_value(fam["test"], budget, s) for s in range(15)])
        if orc <= rnd:
            vals.append(0.0)
            continue
        chosen = selector(fam, budget)
        val = _realized(fam["test"], chosen, fam["test"]["t"][chosen])
        vals.append((val - rnd) / (orc - rnd))
    return np.array(vals)


def random_effects_rule(effects: np.ndarray, sesoi: float) -> dict:
    """Random-effects lower 95% confidence bound on the mean family effect (replaces all-groups cliff)."""
    n = len(effects)
    mean = float(np.mean(effects))
    sd = float(np.std(effects, ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 0 else 0.0
    t95 = 1.833 if n <= 10 else 1.729  # t_{0.95} approx for df~9..19
    lcb = mean - t95 * se
    mdce = t95 * se  # minimum detectable mean effect at this precision
    return {"mean_family_effect": round(mean, 3), "between_family_sd": round(sd, 3),
            "lower_95_cb": round(lcb, 3), "min_detectable_effect": round(mdce, 3),
            "favorable_family_fraction": round(float(np.mean(effects > 0)), 3),
            "pass": lcb >= sesoi}


def _raw_oracle_headroom(bed, seed=7):
    """RAW oracle-vs-random headroom fraction per family (not the normalized value, which is 1 by design)."""
    fracs = []
    for f in range(bed.n_families):
        fam = bed.family(f, seed)
        b = fam["budget"]
        orc = oracle_value(fam["test"], b)
        rnd = np.mean([random_value(fam["test"], b, s) for s in range(15)])
        fracs.append((orc - rnd) / (orc + 1e-9))
    return np.array(fracs)


def run_repaired_battery(bed, *, sesoi: float = SESOI, seed: int = 7) -> dict:
    # oracle headroom and residual learnable headroom
    orc = _family_norm_values(bed, oracle_selector(), seed)          # ~1.0 by construction
    rnd = _family_norm_values(bed, random_selector(0), seed)         # ~0.0
    raw_headroom = _raw_oracle_headroom(bed, seed)                   # real oracle advantage over random
    simple = {h: _family_norm_values(bed, heuristic_selector(h), seed)
              for h in ("feature_magnitude", "loss_proxy", "frequency", "recency")}
    best_simple = np.max(np.vstack([simple[h] for h in simple]), axis=0)
    residual_headroom = float(np.mean(1.0 - best_simple))            # oracle(1) minus best simple control

    # capable mechanisms (architecture set), each tuned; effect = mechanism - random per family
    mech = {name: _family_norm_values(bed, mechanism_selector(name), seed) for name in CAPABLE}
    mech_incr = {name: mech[name] - best_simple for name in CAPABLE}          # incremental over simple
    baseline_lin = _family_norm_values(bed, mechanism_selector("linear"), seed) if "linear" not in CAPABLE else None

    # decodability across capable estimators (held-out correlation)
    dec = []
    for f in range(bed.n_families):
        fam = bed.family(f, seed)
        best = 0.0
        for name in CAPABLE:
            pred = select_and_fit(name, fam["train"]["x"], fam["train"]["r"], fam["tune"]["x"],
                                  fam["tune"]["r"], fam["test"]["x"])
            if np.std(pred) > 0:
                best = max(best, float(np.corrcoef(pred, fam["test"]["r"])[0, 1]))
        dec.append(best)
    dec = np.array(dec)

    # controls
    tv_excess = []
    for f in range(bed.n_families):
        fam = bed.family(f, seed)
        ch = mechanism_selector("kernel_ridge")(fam, fam["budget"])
        tv_excess.append(float(np.mean(fam["test"]["tv"][ch]) - np.mean(fam["test"]["tv"])))
    shuffled = _family_norm_values(bed, shuffled_train_selector("kernel_ridge"), seed)
    # wrong-time: best mechanism items at wrong times
    wrongtime = []
    for f in range(bed.n_families):
        fam = bed.family(f, seed)
        te = fam["test"]
        orcf = oracle_value(te, fam["budget"]); rndf = np.mean([random_value(te, fam["budget"], s) for s in range(15)])
        ch = mechanism_selector("kernel_ridge")(fam, fam["budget"])
        wt = (te["t"][ch] + 5) % 10
        wrongtime.append((_realized(te, ch, wt) - rndf) / (orcf - rndf) if orcf > rndf else 0.0)

    # architecture independence: how many capable estimators pass the group rule
    per_arch_rule = {name: random_effects_rule(mech_incr[name], sesoi) for name in CAPABLE}
    n_arch_pass = sum(1 for v in per_arch_rule.values() if v["pass"])
    best_arch = max(CAPABLE, key=lambda n: per_arch_rule[n]["mean_family_effect"])
    best_incr_rule = per_arch_rule[best_arch]

    m = lambda a: float(np.mean(a))  # noqa: E731
    clauses = {
        "A_what_sufficiency": {"decodability_mean": round(m(dec), 3), "pass": m(dec) >= 0.2},
        "B_oracle_headroom": {"raw_headroom_fraction": round(m(raw_headroom), 3),
                              "pass": m(raw_headroom) >= 0.10},  # oracle must beat random by >=10% raw value
        "residual_learnable_headroom": {"value": round(residual_headroom, 3),
                                        "best_simple_control_norm": round(m(best_simple), 3),
                                        "pass": residual_headroom >= sesoi},
        "C_when_decodability": {"decodability_mean": round(m(dec), 3), "pass": m(dec) >= 0.2},
        "D_incremental_value": {"best_arch": best_arch, "mean_incr_over_simple": best_incr_rule["mean_family_effect"],
                                "lower_95_cb": best_incr_rule["lower_95_cb"], "pass": best_incr_rule["pass"]},
        "E_group_generalization": {"rule": "random_effects_lower_95_cb", **best_incr_rule},
        "F_architecture_independence": {"capable_estimators": list(CAPABLE), "n_passing": n_arch_pass,
                                        "per_arch": {n: per_arch_rule[n]["lower_95_cb"] for n in CAPABLE},
                                        "linear_baseline_incr": round(m(baseline_lin - best_simple), 3)
                                        if baseline_lin is not None else None,
                                        "pass": n_arch_pass >= 2},
        "G_noisy_tv": {"excess_alloc_mean": round(m(tv_excess), 3), "pass": m(tv_excess) <= sesoi},
        "H_shuffled_target": {"shuffled_norm_mean": round(m(shuffled), 3),
                              "mechanism_norm_mean": round(m(mech[best_arch]), 3),
                              # destroying r<->x must remove most of the mechanism's effect over random
                              "pass": (m(shuffled) - m(rnd)) <= 0.5 * (m(mech[best_arch]) - m(rnd))},
        "I_wrong_time": {"wrongtime_norm_mean": round(m(wrongtime), 3),
                         "pass": m(wrongtime) <= best_incr_rule["mean_family_effect"]},
        "J_rate_matched_random": {"mechanism_minus_random": round(m(mech[best_arch]) - m(rnd), 3),
                                  "pass": (m(mech[best_arch]) - m(rnd)) >= sesoi},
    }
    return {"n_families": bed.n_families, "clauses": clauses, "best_arch": best_arch,
            "residual_learnable_headroom": round(residual_headroom, 3)}


def classify(battery: dict, *, sesoi: float = SESOI) -> dict:
    c = battery["clauses"]
    order = ["B_oracle_headroom", "residual_learnable_headroom", "A_what_sufficiency", "C_when_decodability",
             "G_noisy_tv", "H_shuffled_target", "I_wrong_time", "D_incremental_value", "J_rate_matched_random",
             "E_group_generalization", "F_architecture_independence"]
    first_fail = next((k for k in order if not c[k]["pass"]), None)
    if first_fail in ("B_oracle_headroom", "residual_learnable_headroom"):
        cls = "invalid_bed"
    elif first_fail in ("A_what_sufficiency", "C_when_decodability"):
        cls = "instrumentation_failure" if battery.get("_positive") else "pruned_mechanism"
    elif first_fail in ("G_noisy_tv", "H_shuffled_target", "I_wrong_time", "D_incremental_value",
                        "J_rate_matched_random"):
        cls = "pruned_mechanism"
    elif first_fail == "E_group_generalization":
        cls = "insufficient_independent_units"
    elif first_fail == "F_architecture_independence":
        cls = "architecture_dependent"
    else:
        cls = "admitted"
    passed = sum(1 for v in c.values() if v["pass"])
    return {"classification": cls, "first_failing_clause": first_fail,
            "clauses_passed": passed, "clauses_total": len(c)}
