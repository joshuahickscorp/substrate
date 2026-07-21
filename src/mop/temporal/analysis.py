"""Factorial estimation and the classification rules that read it.

Every contrast is a paired difference over seeds, so the seed is the pairing unit and the lower bound is a
random effects bound over seeds. Group inference over independent units is reported alongside, never instead.

The recovery function is what the calibration worlds test: given a set of cell results whose generative truth
is known, does the analysis name that truth.

House style: no dashes.
"""

from __future__ import annotations

import numpy as np

from mop.method import power
from mop.temporal import arch as A
from mop.temporal import io

SESOI = io.SESOI
EQUIV = io.EQUIVALENCE_MARGIN


def contrast(series: dict, a: str, b: str, prereg: dict, unit_series: dict | None = None) -> dict:
    if a not in series or b not in series:
        return {"contrast": f"{a} minus {b}", "verdict": "missing_cell", "mean": None}
    sa, sb = series[a], series[b]
    n = min(len(sa), len(sb))
    eff = [float(x) - float(y) for x, y in zip(sa[:n], sb[:n], strict=True)]
    d = power.decide(eff, prereg)
    d["contrast"] = f"{a} minus {b}"
    d["per_seed_effects"] = [round(x, 5) for x in eff]
    if unit_series and a in unit_series and b in unit_series:
        ua, ub = unit_series[a], unit_series[b]
        shared = sorted(set(ua) & set(ub))
        ue = [ua[u] - ub[u] for u in shared]
        d["group_lower_95_cb"] = round(power.lcb(ue), 5) if len(ue) > 1 else None
        d["group_upper_95_cb"] = round(-power.lcb([-x for x in ue]), 5) if len(ue) > 1 else None
        d["group_mean"] = round(float(np.mean(ue)), 5) if ue else None
        d["n_units"] = len(shared)
        d["group_heterogeneity"] = round(float(np.std(ue, ddof=1)), 5) if len(ue) > 1 else None
    return d


def equivalent(d: dict, margin: float = EQUIV) -> bool:
    """Two configurations are equivalent when the difference is bounded inside the margin on both sides."""
    if d.get("mean") is None or d.get("lower_95_cb") is None:
        return False
    upper = d["mean"] + (d["mean"] - d["lower_95_cb"])
    return abs(d["mean"]) <= margin and d["lower_95_cb"] >= -margin and upper <= margin


def name(family="gru", tier="small", readout="linear", reset="none", history_k=1) -> str:
    return f"{family}|{tier}|{readout}|{reset}|h{history_k}"


# ---------------------------------------------------------------- factor sweeps


def factor_effects(series: dict, prereg: dict, units: dict | None = None) -> dict:
    """Main effects and interactions, each read off the sweep that was designed to carry it."""
    out: dict[str, dict] = {}
    ref = name()

    out["architecture"] = {
        f: contrast(series, name(family=f), ref, prereg, units)
        for f in A.FAMILIES if f != "gru"
    }
    out["recurrence_versus_best_stateless"] = {}
    for r in A.RECURRENT:
        for s in A.STATELESS:
            out["recurrence_versus_best_stateless"][f"{r}_vs_{s}"] = contrast(
                series, name(family=r), name(family=s), prereg, units)

    out["capacity"] = {}
    for f in ("gru", "lstm", "mgu", "pooled", "histmlp", "tcn"):
        for t in A.CAPACITY_TIERS:
            if t == "small":
                continue
            out["capacity"][f"{f}_{t}_vs_small"] = contrast(
                series, name(family=f, tier=t), name(family=f), prereg, units)

    out["readout"] = {}
    for f in ("gru", "pooled", "histmlp"):
        for r in ("mlp1", "mlp_strong"):
            out["readout"][f"{f}_{r}_vs_linear"] = contrast(
                series, name(family=f, readout=r), name(family=f), prereg, units)

    out["horizon"] = {}
    for f in ("gru", "mgu"):
        for h in (1, 2, 5, 10, 20, 45, 90):
            out["horizon"][f"{f}_h{h}_vs_full"] = contrast(
                series, name(family=f, reset=f"horizon_{h}"), name(family=f, reset="horizon_full"),
                prereg, units)

    out["reset"] = {
        k: contrast(series, name(reset=k), ref, prereg, units)
        for k in ("every_observation", "misaligned_a", "misaligned_b", "random_rate_matched",
                  "block_shuffled", "true_boundary", "wrong_boundary")
    }

    out["history"] = {
        f"histmlp_k{k}_vs_k1": contrast(series, name(family="histmlp", history_k=k),
                                        name(family="histmlp"), prereg, units)
        for k in (2, 5, 10, 20, "full_window", "pooled_summary")
    }
    out["recurrent_versus_matched_history"] = {
        f"gru_vs_histmlp_k{k}": contrast(series, ref, name(family="histmlp", history_k=k), prereg, units)
        for k in (1, 2, 5, 10, 20, "full_window")
    }

    out["capacity_by_horizon"] = {}
    for t in A.CAPACITY_TIERS:
        for h in (5, 45, "full"):
            out["capacity_by_horizon"][f"{t}_h{h}_vs_small_h{h}"] = contrast(
                series, name(tier=t, reset=f"horizon_{h}"), name(tier="small", reset=f"horizon_{h}"),
                prereg, units)

    out["capacity_by_readout"] = {
        f"{f}_large_strong_vs_gru_large_strong": contrast(
            series, name(family=f, tier="large", readout="mlp_strong"),
            name(family="gru", tier="large", readout="mlp_strong"), prereg, units)
        for f in ("pooled", "histmlp", "tcn")
    }
    return out


# ---------------------------------------------------------------- recovery


def recover(series: dict, prereg: dict, units: dict | None = None) -> dict:
    """Name which factors carry real effects. This is what the calibration worlds check."""
    eff = factor_effects(series, prereg, units)

    def any_positive(group: dict, keys=None) -> bool:
        return any(v.get("verdict") == "positive" for k, v in group.items() if keys is None or k in keys)

    def best_mean(group: dict) -> float:
        vals = [v["mean"] for v in group.values() if v.get("mean") is not None]
        return max(vals) if vals else 0.0

    rec_vs_stateless = eff["recurrence_versus_best_stateless"]
    matched_hist = eff["recurrent_versus_matched_history"]
    cap = eff["capacity"]
    hor = eff["horizon"]
    read = eff["readout"]
    cxh = eff["capacity_by_horizon"]

    # Nothing is sufficient for an effect that is not there, and a horizon cannot matter when no horizon
    # helps. Every downstream finding is gated on there being a base effect to explain.
    def spread(group):
        vals = [abs(v["mean"]) for v in group.values() if v.get("mean") is not None]
        return max(vals) if vals else 0.0

    base_effect = max(spread(rec_vs_stateless), spread(hor), spread(cap), spread(read), spread(cxh))
    has_base_effect = base_effect > SESOI

    findings = {
        "base_effect_present": has_base_effect,
        "base_effect_size": round(base_effect, 5),
        "recurrence": any(v.get("verdict") == "positive" for v in rec_vs_stateless.values()),
        "recurrence_survives_matched_history": all(
            v.get("verdict") == "positive" for k, v in matched_hist.items()
            if v.get("verdict") not in (None, "missing_cell")),
        "capacity": any_positive(cap),
        "capacity_monotonic": _monotonic([cap.get(f"gru_{t}_vs_small", {}).get("mean")
                                          for t in ("medium", "large")]),
        "readout": any_positive(read),
        "horizon": has_base_effect and any(
            v.get("mean") is not None and v["mean"] <= -SESOI for v in hor.values()),
        "horizon_threshold": _threshold(hor, "gru"),
        "capacity_by_horizon_interaction": has_base_effect and _interaction(cxh),
        "explicit_history_sufficient": has_base_effect and any(
            equivalent(v) for k, v in matched_hist.items() if v.get("mean") is not None),
        "reset_alignment_artifact": _reset_artifact(eff["reset"]),
    }
    truth = []
    if findings["recurrence"] and findings["recurrence_survives_matched_history"] and has_base_effect:
        truth.append("recurrence")
    if findings["explicit_history_sufficient"]:
        truth.append("explicit_history_sufficiency")
    if findings["capacity"]:
        truth.append("capacity")
    if findings["readout"]:
        truth.append("readout")
    if findings["horizon"]:
        truth.append("horizon")
    if findings["capacity_by_horizon_interaction"]:
        truth.append("core_horizon_interaction")
    if findings["reset_alignment_artifact"]:
        truth.append("reset_alignment_artifact")
    if not truth:
        truth.append("no_effect")
    return {"effects": eff, "findings": findings, "recovered": sorted(set(truth)),
            "best_architecture_gain": round(best_mean(eff["architecture"]), 5)}


def _monotonic(vals) -> bool:
    v = [x for x in vals if x is not None]
    return len(v) >= 2 and all(v[i] <= v[i + 1] + 1e-9 for i in range(len(v) - 1)) and v[-1] > SESOI


def _threshold(horizon_group: dict, family: str) -> int | None:
    """The shortest horizon whose loss against full persistence is inside the equivalence margin."""
    got = []
    for h in (1, 2, 5, 10, 20, 45, 90):
        d = horizon_group.get(f"{family}_h{h}_vs_full")
        if d and d.get("mean") is not None:
            got.append((h, d))
    for h, d in got:
        if abs(d["mean"]) <= EQUIV:
            return h
    return None


def _interaction(group: dict) -> bool:
    """Capacity helps at a long horizon and not at a short one."""
    short = [v["mean"] for k, v in group.items() if k.endswith("h5_vs_small_h5") and v.get("mean") is not None]
    long = [v["mean"] for k, v in group.items() if "hfull" in k and v.get("mean") is not None]
    if not short or not long:
        return False
    return max(long) - max(short) > SESOI


def _reset_artifact(reset_group: dict) -> bool:
    """A reset schedule that beats its misaligned peers by more than the margin is doing oracle work."""
    tb = reset_group.get("true_boundary", {}).get("mean")
    ma = [reset_group.get(k, {}).get("mean") for k in ("misaligned_a", "misaligned_b")]
    ma = [x for x in ma if x is not None]
    return tb is not None and bool(ma) and (tb - max(ma)) > SESOI
