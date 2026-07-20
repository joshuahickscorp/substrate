"""Within domain continual battery.

The premise under test is the same one the cross domain matrix tests, so the protocol is the same shape: one
shared fast core, context local plasticity. Within a domain the contexts are disjoint class subsets, which
the architectures see as separate registered contexts. That reuses the architectures unchanged and keeps
forgetting confined to the shared core, which is the object of study.

Stressors, all run against the same arms and the same budgets:

    plain       online acquisition, context switch, return to a prior context, and the missing observation
                evaluation, which is an evaluation time stressor and therefore needs no separate run
    limited     one third of the update budget
    small_mem   one third of the memory budget

House style: no dashes.
"""

from __future__ import annotations

import numpy as np
import torch

from . import arch as A
from . import data as D
from . import engine as E

STEPS = 400
MEM_CAP = 600
MASK_FRAC = 0.3

# arch, core trainable after the first context, gate, which context local groups may train, memory policy
WITHIN_ARMS: dict[str, dict] = {
    "G0_always_trainable": dict(arch="G", core_after_first=True),
    "G1_frozen_after_first": dict(arch="G", core_after_first=False),
    "G4_adapters_only": dict(arch="G", core_after_first=False, local_kinds=("adapter", "norm", "head")),
    "G5_projection_and_head": dict(arch="G", core_after_first=False,
                                   local_kinds=("proj_conv", "proj_lin", "head")),
    "H_always_update": dict(arch="H", core_after_first=True, gate="always"),
    "H_never_update": dict(arch="H", core_after_first=False, gate="never"),
    "H_cosine_gate": dict(arch="H", core_after_first=True, gate="cosine", thresh=0.0),
    "H_probe_gate": dict(arch="H", core_after_first=True, gate="probe", thresh=0.05),
    "AT_no_slow": dict(arch="G", core_after_first=True, local_kinds=("proj_conv", "proj_lin", "norm", "head")),
    "gru": dict(arch="gru", core_after_first=True),
    "lstm": dict(arch="lstm", core_after_first=True),
    "lstm_gdumb": dict(arch="lstm", core_after_first=True, memory="gdumb"),
    "separate_per_context": dict(arch="separate", core_after_first=True),
    "shared_adapters": dict(arch="shared_adapters", core_after_first=True),
    "ewc": dict(arch="lstm", core_after_first=True, ewc=50.0),
    "reservoir": dict(arch="lstm", core_after_first=True, memory="reservoir"),
    "no_memory": dict(arch="lstm", core_after_first=True, memory="none"),
    "bag_order_free": dict(arch="bag", core_after_first=True),
    "joint_upper_bound": dict(arch="lstm", core_after_first=True, joint=True),
}


def _contexts(dname: str, seed: int, per: int = 2):
    """Disjoint class subsets over the unit disjoint main split, with a tuning and a future slice per domain."""
    sp = D.splits(dname, seed)
    x, y = sp["main"]
    xt, yt = sp["test"]
    rng = np.random.default_rng(seed)
    order = rng.permutation(sp["classes"])
    ctx = []
    for t in range(sp["classes"] // per):
        cs = order[t * per : (t + 1) * per]
        itr = np.concatenate([np.where(y.numpy() == c)[0] for c in cs])
        ite = np.concatenate([np.where(yt.numpy() == c)[0] for c in cs])
        ctx.append({"name": f"c{t}", "x": x[itr], "y": y[itr], "xte": xt[ite], "yte": yt[ite],
                    "classes": [int(c) for c in cs]})
    return ctx, sp


def run_within(arm: str, dname: str, seed: int, stressor: str = "plain", steps: int = STEPS):
    p = WITHIN_ARMS[arm]
    ctx, sp = _contexts(dname, seed)
    torch.manual_seed(1000 + seed)
    model = A.build(p["arch"], {c["name"]: (sp["channels"], sp["classes"]) for c in ctx})
    shared = None
    for g in ("fast_core", "fast_delta"):
        if g in model.param_groups:
            shared = g
    rng = np.random.default_rng(seed)
    kinds = p.get("local_kinds") or ("proj_conv", "proj_lin", "adapter", "norm", "local_slow", "head")
    if stressor == "limited":
        steps = max(20, steps // 3)
    cap = MEM_CAP // 3 if stressor == "small_mem" else MEM_CAP
    mems = {c["name"]: E.Memory(p.get("memory", "gdumb"), cap) for c in ctx}
    n = len(ctx)
    acc = np.zeros((n, n))
    acc_missing = np.zeros(n)  # missing observation recovery is an evaluation stressor, not a second run
    receipts, ewc = [], None

    if p.get("joint"):  # upper bound: contexts interleaved round robin, total updates matched exactly
        rounds = 4
        for _ in range(rounds):
            for c2 in ctx:
                local = [f"{k}.{c2['name']}" for k in kinds if f"{k}.{c2['name']}" in model.param_groups]
                if p["arch"] == "separate":
                    local += [f"fast_core.{c2['name']}"]
                receipts.append(E.fit(model, c2["name"], c2["x"], c2["y"],
                                      train_groups=local + ([shared] if shared else []),
                                      steps=max(1, steps // rounds), rng=rng, memory=mems[c2["name"]]))
        for t in range(n):
            for j in range(n):
                acc[t, j] = _eval(model, ctx[j], stressor, rng)
    else:
        for t, c in enumerate(ctx):
            local = [f"{k}.{c['name']}" for k in kinds if f"{k}.{c['name']}" in model.param_groups]
            if p["arch"] == "separate":
                local += [f"fast_core.{c['name']}"]
            core_on = shared and (t == 0 or p.get("core_after_first", True))
            gate = None
            ref_batch = ref_domain = None
            if p.get("gate") and shared and t > 0:
                gate = E.Gate(p["gate"], p.get("thresh", 0.0), np.random.default_rng(500 + seed))
                px, py = ctx[t - 1]["x"], ctx[t - 1]["y"]
                bi = rng.choice(len(px), min(128, len(px)), replace=False)
                ref_batch, ref_domain = (px[bi], py[bi]), ctx[t - 1]["name"]
                gate.set_reference(
                    E.reference_gradient(model, model.param_groups[shared], px[bi], py[bi], ref_domain),
                    E.probe_loss(model, px[bi], py[bi], ref_domain))
            if p.get("ewc") and shared and t > 0:
                ewc = E.fisher_diag(model, ctx[t - 1]["name"], ctx[t - 1]["x"], ctx[t - 1]["y"],
                                    model.param_groups[shared], rng)
            receipts.append(E.fit(model, c["name"], c["x"], c["y"],
                                  train_groups=local + ([shared] if core_on else []), steps=steps, rng=rng,
                                  memory=mems[c["name"]], gate=gate, shared_group=shared,
                                  ref_batch=ref_batch, ref_domain=ref_domain, ewc=ewc,
                                  ewc_lambda=p.get("ewc", 0.0), update_recent=True))
            mems[c["name"]].add(c["x"], c["y"], rng)
            for j in range(t + 1):
                acc[t, j] = _eval(model, ctx[j], stressor, rng)

    for j in range(n):
        acc_missing[j] = _eval(model, ctx[j], "missing", np.random.default_rng(31 + seed))
    fin = acc[n - 1]
    # future unit adaptation on the last context, evaluated on rows of units never trained on.
    # the pool is restricted to that context's classes, otherwise a context head is scored on classes it
    # was never given.
    cls = set(ctx[-1]["classes"])
    ax, ay = sp["adapt"]
    ex, ey = sp["adapt_eval"]
    ka = np.array([int(v) in cls for v in ay])
    ke = np.array([int(v) in cls for v in ey])
    ax, ay, ex, ey = ax[ka], ay[ka], ex[ke], ey[ke]
    last = ctx[-1]["name"]
    local = [f"{k}.{last}" for k in kinds if f"{k}.{last}" in model.param_groups]
    if p["arch"] == "separate":
        local += [f"fast_core.{last}"]
    receipts.append(E.fit(model, last, ax, ay, train_groups=local + ([shared] if shared else []),
                          steps=max(20, steps // 3), rng=rng, memory=mems[last]))
    future = E.evaluate(model, last, ex, ey)

    return {
        "arm": arm, "domain": dname, "seed": seed, "stressor": stressor, "arch": p["arch"],
        "metrics": {
            "avg_final": float(fin.mean()),
            "avg_final_missing_observations": float(acc_missing.mean()),
            "missing_observation_drop": float(fin.mean() - acc_missing.mean()),
            "retention": float(fin[: n - 1].mean()) if n > 1 else float(fin.mean()),
            "new_context": float(np.mean([acc[t, t] for t in range(n)])),
            "return_to_prior": float(acc[n - 1, 0]),
            "future_adaptation": float(future),
            "params": A.count_params(model),
            "updates": sum(r["updates"] for r in receipts),
            "memory_cap": cap,
            "train_seconds": round(sum(r["wall_seconds"] for r in receipts), 2),
        },
        "accuracy_matrix": acc.round(4).tolist(),
        "undeclared_changes": sorted({x for r in receipts for x in r["undeclared_changes"]}),
        "checkpoint_final": E.checkpoint_sha(model),
    }


def _eval(model, c, stressor, rng):
    x = D.mask_timesteps(c["xte"], MASK_FRAC, rng) if stressor == "missing" else c["xte"]
    return E.evaluate(model, c["name"], x, c["yte"])
