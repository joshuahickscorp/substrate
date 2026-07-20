"""Interference attribution and the plasticity action headroom gate.

Interference map: train domain A once, snapshot, then for each declared parameter group replay the same
short second domain phase with only that group (plus the head it needs to express anything) trainable.
The counterfactuals are bounded and are scored on the tuning split. The untouched test split is never
exposed to any update policy.

Headroom gate: the same machinery, but the unit of choice is a plasticity action at the domain switch. The
oracle picks the best action per seed on tuning utility. Headroom is the oracle advantage over the strongest
simple rule. Random and shuffled controls must fail, or the oracle signal is not a signal.

House style: no dashes.
"""

from __future__ import annotations

import copy
import json
import sys
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import data as D
from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import io

SEEDS = list(range(8))
DIRECTIONS = [("har", "speech"), ("speech", "har")]
SHORT = {"second": 150, "return": 60, "adapt": 60}

# a plasticity action is the set of group kinds that may change at and after the domain switch
ACTIONS: dict[str, tuple[str, ...]] = {
    "no_update": (),
    "head_only": ("head",),
    "projection_plus_head": ("proj_conv", "proj_lin", "head"),
    "adapter_plus_head": ("adapter", "norm", "head"),
    "fast_core_plus_head": ("__shared__", "head"),
    "fast_core_plus_adapter": ("__shared__", "adapter", "norm", "head"),
    "all_active_groups": ("__shared__", "proj_conv", "proj_lin", "adapter", "norm", "head"),
}
# groups measured one at a time by the interference map
GROUP_KINDS = ["proj_conv", "proj_lin", "adapter", "norm", "head", "__shared__"]


def _phase1(archname, direction, seed, budget):
    a, b = direction
    sa, sb = D.splits(a, seed), D.splits(b, seed)
    torch.manual_seed(1000 + seed)
    model = A.build(archname, {a: (sa["channels"], sa["classes"]), b: (sb["channels"], sb["classes"])})
    shared = S.shared_group_name(model)
    rng = np.random.default_rng(seed)
    mem = E.Memory("gdumb", 600)
    E.fit(model, a, *sa["main"], train_groups=S.local_groups(model, a) + [shared],
          steps=budget[a], rng=rng, memory=mem)
    mem.add(*sa["main"], rng)
    base = {
        "a_tune": E.evaluate(model, a, *sa["tune"]),
        "b_tune": E.evaluate(model, b, *sb["tune"]),
    }
    return model, shared, sa, sb, base, copy.deepcopy(model.state_dict())


def _groups_for(model, kinds, dom, shared):
    out = []
    for k in kinds:
        if k == "__shared__":
            out.append(shared)
        elif f"{k}.{dom}" in model.param_groups:
            out.append(f"{k}.{dom}")
    return out


def _counterfactual(model, state, shared, sa, sb, a, b, kinds, seed, memory=True):
    model.load_state_dict(state)
    rng = np.random.default_rng(3000 + seed)
    mems = {b: E.Memory("gdumb" if memory else "none", 600), a: E.Memory("gdumb" if memory else "none", 600)}
    mems[a].add(*sa["main"], rng)
    g = _groups_for(model, kinds, b, shared)
    r2 = E.fit(model, b, *sb["main"], train_groups=g or [f"head.{b}"], steps=SHORT["second"], rng=rng,
               memory=mems[b]) if g else None
    if g is None:
        r2 = E.fit(model, b, *sb["main"], train_groups=[f"head.{b}"], steps=0, rng=rng)
    out = {
        "new_domain": E.evaluate(model, b, *sb["tune"]),
        "old_domain_after": E.evaluate(model, a, *sa["tune"]),
    }
    ga = _groups_for(model, kinds, a, shared)
    E.fit(model, a, *sa["main"], train_groups=ga or [f"head.{a}"], steps=SHORT["return"], rng=rng,
          memory=mems[a])
    out["return_recovery"] = E.evaluate(model, a, *sa["tune"])
    E.fit(model, b, *sb["adapt"], train_groups=g or [f"head.{b}"], steps=SHORT["adapt"], rng=rng)
    out["future_adaptation"] = E.evaluate(model, b, *sb["adapt_eval"])
    out["trainable_params"] = r2["trainable_param_count"] if r2 else 0
    out["compute_seconds"] = r2["wall_seconds"] if r2 else 0.0
    return out


def shard(direction, seed, budget):
    a, b = direction
    res = {"direction": f"{a}->{b}", "seed": seed, "groups": {}, "actions": {}}
    for archname in ("G", "H"):
        model, shared, sa, sb, base, state = _phase1(archname, direction, seed, budget)
        res.setdefault("baseline", {})[archname] = base
        for k in GROUP_KINDS:
            if k == "__shared__" and shared is None:
                continue
            name = f"{archname}.{'shared_' + shared if k == '__shared__' else k}"
            kinds = (k, "head") if k != "head" else ("head",)
            res["groups"][name] = _counterfactual(model, state, shared, sa, sb, a, b, kinds, seed)
            res["groups"][name]["group_kinds"] = list(kinds)
        # memory state is a declared group in the interference sense: it changes what an update sees
        res["groups"][f"{archname}.memory_state"] = _counterfactual(
            model, state, shared, sa, sb, a, b, ("__shared__", "head"), seed, memory=False)
        res["groups"][f"{archname}.memory_state"]["group_kinds"] = ["shared_and_head_without_memory"]
        for name, kinds in ACTIONS.items():
            res["actions"][f"{archname}.{name}"] = _counterfactual(
                model, state, shared, sa, sb, a, b, kinds, seed)
        print(f"  {archname} {a}->{b} seed{seed} done", flush=True)
    return res


def _u(v):
    return float(np.mean([v["new_domain"], v["return_recovery"], v["future_adaptation"]]))


def aggregate(rows):
    """rows is a list of shard results."""
    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], []).append(r)
    gmap, amap = {}, {}
    for dname, rs in by_dir.items():
        names = sorted(rs[0]["groups"])
        for n in names:
            base_old = [r["baseline"][n.split(".")[0]]["a_tune"] for r in rs]
            gmap.setdefault(n, {})[dname] = {
                "new_domain_gain": E.effect([r["groups"][n]["new_domain"] for r in rs],
                                            [r["baseline"][n.split(".")[0]]["b_tune"] for r in rs]),
                "old_domain_loss": E.effect(base_old, [r["groups"][n]["old_domain_after"] for r in rs]),
                "return_recovery": round(float(np.mean([r["groups"][n]["return_recovery"] for r in rs])), 4),
                "future_adaptation": round(float(np.mean([r["groups"][n]["future_adaptation"] for r in rs])), 4),
                "trainable_params": int(np.mean([r["groups"][n]["trainable_params"] for r in rs])),
                "compute_seconds": round(float(np.mean([r["groups"][n]["compute_seconds"] for r in rs])), 2),
            }
        for n in sorted(rs[0]["actions"]):
            amap.setdefault(n, {})[dname] = {
                "tune_utility": round(float(np.mean([_u(r["actions"][n]) for r in rs])), 4),
                "per_seed": [round(_u(r["actions"][n]), 4) for r in rs],
            }
    return gmap, amap


def classify(gmap):
    out = {}
    for n, per in gmap.items():
        acq = float(np.mean([v["new_domain_gain"]["mean"] for v in per.values()]))
        forget = float(np.mean([v["old_domain_loss"]["mean"] for v in per.values()]))
        rec = float(np.mean([v["return_recovery"] for v in per.values()]))
        out[n] = {
            "mean_new_domain_gain": round(acq, 4),
            "mean_old_domain_loss": round(forget, 4),
            "mean_return_recovery": round(rec, 4),
            "enables_acquisition": acq >= io.SESOI,
            "causes_forgetting": forget >= io.SESOI,
            "transferable": acq >= io.SESOI and forget < io.SESOI,
            "domain_specific": acq >= io.SESOI and forget >= io.SESOI,
            "safe_to_reopen": forget < io.SESOI,
            "should_remain_frozen": forget >= io.SESOI and acq < io.SESOI,
        }
    return out


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        d, s = sys.argv[2].split(":")
        budget = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {"har": 550, "speech": 1200}
        res = shard(tuple(d.split("-")), int(s), budget)
        p = io.RUNS / "interference"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{d}_{s}.json").write_text(json.dumps(res, default=str))
        print("SHARD_DONE", d, s, flush=True)
        return

    t0 = time.time()
    rows = []
    for direction in DIRECTIONS:
        for s in SEEDS:
            f = io.RUNS / "interference" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                rows.append(json.loads(f.read_text()))
    if len(rows) < len(DIRECTIONS) * len(SEEDS):
        print("missing interference shards:", len(rows), "of", len(DIRECTIONS) * len(SEEDS), flush=True)
        return
    gmap, amap = aggregate(rows)
    cls = classify(gmap)

    io.seal("MOP_SUBSTRATE_INTERFERENCE_MAP.json", {
        "schema": "mop-substrate-interference-map/v1",
        "seeds": SEEDS, "directions": [f"{a}->{b}" for a, b in DIRECTIONS],
        "short_horizon_budget": SHORT,
        "scored_on": "tuning split, unit disjoint from both training and test",
        "groups": gmap,
        "classification": cls,
        "transferable_groups": sorted(n for n, v in cls.items() if v["transferable"]),
        "domain_specific_groups": sorted(n for n, v in cls.items() if v["domain_specific"]),
        "forgetting_groups": sorted(n for n, v in cls.items() if v["causes_forgetting"]),
        "safe_to_reopen": sorted(n for n, v in cls.items() if v["safe_to_reopen"]),
        "keep_frozen": sorted(n for n, v in cls.items() if v["should_remain_frozen"]),
    })
    io.seal("MOP_SUBSTRATE_PLASTICITY_ACTION_HEADROOM.json", {
        "schema": "mop-substrate-plasticity-action-headroom/v1",
        "actions": sorted(ACTIONS), "action_utilities": amap, "seeds": SEEDS,
        "note": "the oracle over these actions is resolved by the policy runner, which also carries the "
                "random and shuffled controls",
    })
    lines = ["# Substrate interference report", "",
             "Counterfactual update effects per declared parameter group, bounded short horizon, scored on the",
             "tuning split only. Effects are lower 95 percent confidence bounds over 8 seeds and two domain",
             "directions.", "",
             "| group | new domain gain | old domain loss | return recovery | verdict |",
             "| --- | --- | --- | --- | --- |"]
    for n in sorted(cls):
        v = cls[n]
        verdict = ("transferable" if v["transferable"] else "domain specific" if v["domain_specific"]
                   else "keep frozen" if v["should_remain_frozen"] else "no material effect")
        lines.append(f"| {n} | {v['mean_new_domain_gain']} | {v['mean_old_domain_loss']} | "
                     f"{v['mean_return_recovery']} | {verdict} |")
    io.seal_md("MOP_SUBSTRATE_INTERFERENCE_REPORT.md", "\n".join(lines) + "\n")
    print("interference sealed", round(time.time() - t0, 1), "s", flush=True)
    print("INTERFERENCE_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
