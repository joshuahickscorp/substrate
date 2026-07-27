"""Within domain continual battery runner.

Purpose is not to find a winner. It is to check that localizing plasticity does not destroy within domain
performance, to measure how far the shared fast core can be frozen before acquisition suffers, and to give
the cross domain matrix a converged set of comparisons inside each domain.

Every arm runs the plain protocol, which already carries the missing observation evaluation. The substrate
arms and the strongest baselines additionally run the limited update and limited memory stressors.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import sequence as S
from fastforge import within as W
from fastforge.runs import io

SEEDS = [0, 1, 2, 3, 4]
DOMAINS = ["har", "speech"]
STRESS_ARMS = [
    "G0_always_trainable",
    "G1_frozen_after_first",
    "G4_adapters_only",
    "H_cosine_gate",
    "H_never_update",
    "AT_no_slow",
    "lstm_gdumb",
    "separate_per_context",
]
BASELINES = [
    "gru",
    "lstm",
    "lstm_gdumb",
    "separate_per_context",
    "shared_adapters",
    "ewc",
    "reservoir",
    "joint_upper_bound",
    "bag_order_free",
]


def steps_for(dname):
    return max(60, S.budget(dname) // 2)


def shard(dname, seed):
    steps = steps_for(dname)
    out = {}
    for arm in W.WITHIN_ARMS:
        t0 = time.time()
        out[f"{arm}|plain"] = W.run_within(arm, dname, seed, "plain", steps)
        print(
            f"  {dname} seed{seed} {arm:24s} plain avg={out[f'{arm}|plain']['metrics']['avg_final']:.3f} "
            f"[{time.time() - t0:.0f}s]",
            flush=True,
        )
    for arm in STRESS_ARMS:
        for st in ("limited", "small_mem"):
            out[f"{arm}|{st}"] = W.run_within(arm, dname, seed, st, steps)
    return out


def summarize(dname, rows):
    seeds = sorted(rows)
    arms = sorted({k.split("|")[0] for k in rows[seeds[0]] if k.endswith("|plain")})
    keys = [
        "avg_final",
        "retention",
        "new_context",
        "return_to_prior",
        "future_adaptation",
        "avg_final_missing_observations",
        "missing_observation_drop",
        "params",
        "updates",
    ]
    means = {
        a: {k: round(float(np.mean([rows[s][f"{a}|plain"]["metrics"][k] for s in seeds])), 4) for k in keys}
        for a in arms
    }
    base_params = means["lstm_gdumb"]["params"]

    def util(a, s, stress="plain"):
        m = rows[s][f"{a}|{stress}"]["metrics"]
        return float(m["avg_final"] - 0.05 * (m["params"] / base_params))

    um = {a: round(float(np.mean([util(a, s) for s in seeds])), 4) for a in arms}
    pool = [b for b in BASELINES if b in arms and b != "joint_upper_bound"]
    strongest = max(pool, key=lambda a: um[a])
    effects = {a: S.E.effect([util(a, s) for s in seeds], [util(strongest, s) for s in seeds]) for a in arms}
    stress = {}
    for a in STRESS_ARMS:
        if f"{a}|limited" not in rows[seeds[0]]:
            continue
        stress[a] = {
            st: {
                "utility": round(float(np.mean([util(a, s, st) for s in seeds])), 4),
                "vs_plain": S.E.effect([util(a, s, st) for s in seeds], [util(a, s) for s in seeds]),
            }
            for st in ("limited", "small_mem")
        }
    # the joint upper bound and the order free control are references, not competitors
    passing = [
        a
        for a in arms
        if effects[a]["lower_95_cb"] >= io.SESOI and a not in ("joint_upper_bound", "bag_order_free")
    ]
    return {
        "domain": dname,
        "seeds": seeds,
        "steps_per_context": steps_for(dname),
        "learning_rate": S.lr_for(dname),
        "arms": arms,
        "means": means,
        "utility": um,
        "strongest_matched_baseline": strongest,
        "effects_vs_strongest_baseline": effects,
        "joint_training_upper_bound": um.get("joint_upper_bound"),
        "joint_upper_bound_caveat": (
            "the joint arm interleaves every context round robin at a matched total update budget. Several "
            "sequential arms score above it, so it is not behaving as an upper bound on this protocol. The "
            "likely reason is that per context heads make the sequential arms task incremental, which is "
            "easier than the interleaved multi head problem the joint arm solves. It is reported as a "
            "reference point and is not used as a ceiling in any verdict."
        ),
        "protocol_disclosure": (
            "task incremental: every context owns its projection, adapter, normalization and head, and the "
            "context identity is given at evaluation. Forgetting is therefore confined to the shared core, "
            "which is the object of study, but the absolute numbers are not comparable to class incremental "
            "continual learning results."
        ),
        "order_free_control": um.get("bag_order_free"),
        "stressors": stress,
        "arms_beating_the_strongest_baseline": passing,
        "verdict": "within_domain_positive" if passing else "within_domain_null",
        "simplification_check": {
            "G1_frozen_vs_G0_always": S.E.effect(
                [util("G1_frozen_after_first", s) for s in seeds],
                [util("G0_always_trainable", s) for s in seeds],
            ),
            "G4_adapters_only_vs_G0": S.E.effect(
                [util("G4_adapters_only", s) for s in seeds], [util("G0_always_trainable", s) for s in seeds]
            ),
        },
    }


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        d, s = sys.argv[2].split(":")
        res = shard(d, int(s))
        p = io.RUNS / "within"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{d}_{s}.json").write_text(json.dumps(res, default=str))
        print("SHARD_DONE", d, s, flush=True)
        return
    for dname in DOMAINS:
        rows = {}
        for s in SEEDS:
            f = io.RUNS / "within" / f"{dname}_{s}.json"
            if f.is_file():
                rows[s] = json.loads(f.read_text())
        if len(rows) < len(SEEDS):
            print(f"missing within shards for {dname}: {len(rows)} of {len(SEEDS)}", flush=True)
            continue
        summary = summarize(dname, rows)
        io.seal(
            f"MOP_{dname.upper()}_WITHIN_DOMAIN_REPORT.json",
            {"schema": "mop-fast-state-within-domain/v1", **summary},
        )
        print(dname, summary["verdict"], "strongest", summary["strongest_matched_baseline"], flush=True)
    print("WITHIN_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
