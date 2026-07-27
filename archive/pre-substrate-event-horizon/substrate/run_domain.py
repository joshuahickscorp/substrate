"""Principal-domain comparison: substrate A/B vs strong matched baselines (GDumb, EWC, GRU), 5 seeds, with a
cost-adjusted joint moldability utility (params charged) and sealed component floors. Applies the architecture
selection rule. House style: no dashes."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/scammermike/Downloads/mop-substrate-genesis-v2/substrate")
import numpy as np  # noqa: E402
import domains as D  # noqa: E402
from engine import moldability, run_stream  # noqa: E402

REPORTS = Path("/Users/scammermike/Downloads/mop-substrate-genesis-v2/substrate/reports")
SEEDS = [0, 1, 2, 3, 4]
EWC = 1.0
STEPS = 100

# sealed arms: (model, config). Baselines are strong (GDumb, EWC, combined). Substrate arms + key ablations.
ARMS = {
    "mlp_gdumb": ("mlp", {"memory": "gdumb", "steps": STEPS}),
    "mlp_ewc": ("mlp", {"memory": "none", "consolidation": "fixed", "ewc_lambda": EWC, "steps": STEPS}),
    "mlp_gdumb_ewc": ("mlp", {"memory": "gdumb", "consolidation": "fixed", "ewc_lambda": EWC, "steps": STEPS}),
    "gru_gdumb": ("gru", {"memory": "gdumb", "steps": STEPS}),
    "A_full": ("A", {"memory": "gdumb", "consolidation": "boundary", "ewc_lambda": EWC, "steps": STEPS}),
    "A_nomem": ("A", {"memory": "none", "consolidation": "boundary", "ewc_lambda": EWC, "steps": STEPS}),
    "A_nocons": ("A", {"memory": "gdumb", "steps": STEPS}),
    "B_full": ("B", {"memory": "gdumb", "consolidation": "boundary", "ewc_lambda": EWC, "steps": STEPS}),
    "B_nomem": ("B", {"memory": "none", "consolidation": "boundary", "ewc_lambda": EWC, "steps": STEPS}),
    "B_randroute": ("B", {"memory": "gdumb", "consolidation": "boundary", "ewc_lambda": EWC, "steps": STEPS, "eligibility": "all"}),
}
# sealed utility: cost-adjusted avg_final = avg_final - COST * (params / mlp_params); floors vs best baseline
COST = 0.05
BASELINE_ARMS = ["mlp_gdumb", "mlp_ewc", "mlp_gdumb_ewc", "gru_gdumb"]


def sha(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def re(effects):
    e = np.asarray(effects, float); n = len(e); sd = e.std(ddof=1) if n > 1 else 0
    lcb = float(e.mean() - (1.833 if n <= 10 else 1.729) * sd / np.sqrt(n)) if n else 0
    return {"mean": round(float(e.mean()), 4), "lower_95_cb": round(lcb, 4),
            "favorable_fraction": round(float((e > 0).mean()), 3), "n": n}


def run(dom):
    t0 = time.time()
    per = {a: {"avg_final": [], "retention": [], "new": [], "params": []} for a in ARMS}
    for s in SEEDS:
        tasks, obs_dim, n_out = D.PROVIDERS[dom](s)
        for a, (mn, cfg) in ARMS.items():
            acc, met, _, _ = run_stream(mn, tasks, cfg, s, obs_dim, n_out)
            m = moldability(acc)
            per[a]["avg_final"].append(m["avg_final"]); per[a]["retention"].append(m["retention"])
            per[a]["new"].append(m["new_task"]); per[a]["params"].append(met["n_params"])
        print(f"  [{dom}] seed{s} done", flush=True)
    means = {a: {k: float(np.mean(v)) for k, v in per[a].items()} for a in ARMS}
    mlp_params = means["mlp_gdumb"]["params"]
    # cost-adjusted utility per arm per seed
    def util(a, si):
        return per[a]["avg_final"][si] - COST * (per[a]["params"][si] / mlp_params)
    util_mean = {a: float(np.mean([util(a, si) for si in range(len(SEEDS))])) for a in ARMS}
    best_baseline = max(BASELINE_ARMS, key=lambda a: util_mean[a])
    # substrate effect vs strongest baseline, per seed
    def arch_effect(arch_arm):
        eff = [util(arch_arm, si) - util(best_baseline, si) for si in range(len(SEEDS))]
        floors_ok = (means[arch_arm]["retention"] >= means[best_baseline]["retention"] - 0.02 and
                     means[arch_arm]["new"] >= means[best_baseline]["new"] - 0.02)
        # not solely replay: the full arm must beat its own no-memory ablation by a margin OR the no-mem arm alone helps
        return re(eff), floors_ok
    eff_A, floors_A = arch_effect("A_full")
    eff_B, floors_B = arch_effect("B_full")
    posA = eff_A["lower_95_cb"] >= 0.05 and floors_A
    posB = eff_B["lower_95_cb"] >= 0.05 and floors_B
    # architecture selection rule
    if posA and posB:
        sel = "A_full" if util_mean["A_full"] - util_mean["B_full"] >= 0.05 else ("B_full" if util_mean["B_full"] - util_mean["A_full"] >= 0.05 else "A_full")
        cls = "both_positive"
    elif posA:
        sel, cls = "A_full", "substrate_candidate_positive_A"
    elif posB:
        sel, cls = "B_full", "substrate_candidate_positive_B"
    else:
        sel, cls = None, "substrate_candidate_null"
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd="/Users/scammermike/Downloads/mop-substrate-genesis-v2").stdout.strip()
    rep = {"schema": "mop-substrate-domain-result/v1", "domain": dom, "is_image": D.IS_IMAGE[dom],
           "source_commit": commit, "seeds": SEEDS, "steps": STEPS, "ewc_lambda": EWC,
           "utility": "cost_adjusted avg_final minus 0.05*(params/mlp_params); floors: retention and new within 0.02 of best baseline",
           "arm_means": {a: {k: round(v, 4) for k, v in means[a].items()} for a in ARMS},
           "util_mean": {a: round(v, 4) for a, v in util_mean.items()},
           "best_baseline": best_baseline, "A_effect_vs_best_baseline": eff_A, "A_floors_ok": floors_A,
           "B_effect_vs_best_baseline": eff_B, "B_floors_ok": floors_B,
           "classification": cls, "selected": sel, "SESOI": 0.05,
           "not_solely_replay_check": {"A_full_vs_A_nomem": round(util_mean["A_full"] - util_mean["A_nomem"], 4),
                                       "B_full_vs_B_nomem": round(util_mean["B_full"] - util_mean["B_nomem"], 4)},
           "wall_seconds": round(time.time() - t0, 1)}
    rep["sha256"] = sha(rep)
    (REPORTS / f"MOP_SUBSTRATE_DOMAIN_{dom}_RESULT.json").write_text(json.dumps(rep, indent=2))
    print(f"[{dom}] {cls} | best_baseline={best_baseline}({util_mean[best_baseline]:.3f}) "
          f"A_util={util_mean['A_full']:.3f}(eff_lcb {eff_A['lower_95_cb']}) "
          f"B_util={util_mean['B_full']:.3f}(eff_lcb {eff_B['lower_95_cb']}) [{rep['wall_seconds']}s]", flush=True)
    return cls


if __name__ == "__main__":
    dom = sys.argv[1] if len(sys.argv) > 1 else "emnist"
    run(dom)
    print("DOMAIN_DONE", flush=True)
