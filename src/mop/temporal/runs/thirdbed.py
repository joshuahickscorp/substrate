"""Eight seed HARTH admission probe for boundary, adaptation, return and residual headroom."""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch

from fastforge import engine as E
from mop.method import bed, power
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2

SEEDS = e2.PRINCIPAL_SEEDS
ADAPT_STEPS = Fx.STEPS // 4
RETURN_STEPS = Fx.STEPS // 8


def contexts(seed: int) -> dict:
    d = B.load("harth_stream")
    units = np.asarray(d["u"])
    unique = np.random.default_rng(70_000 + seed).permutation(np.unique(units))
    half = len(unique) // 2

    def partition(group):
        n_eval = max(1, int(round(0.3 * len(group))))
        return group[n_eval:], group[:n_eval]

    a_train, a_eval = partition(unique[:half])
    b_train, b_eval = partition(unique[half:])

    def take(group):
        idx = np.where(np.isin(units, list(group)))[0]
        return d["x"][idx], d["y"][idx], units[idx]

    rng = np.random.default_rng(80_000 + seed)
    gain = torch.tensor(rng.uniform(0.45, 1.55, size=d["channels"]), dtype=torch.float32)
    offset = torch.tensor(rng.normal(0, 0.8, size=d["channels"]), dtype=torch.float32)

    def shifted(row):
        x, y, u = row
        return x * gain + offset, y, u

    return {
        "A_train": take(a_train), "A_eval": take(a_eval),
        "B_train": shifted(take(b_train)), "B_eval": shifted(take(b_eval)),
        "channels": d["channels"], "classes": d["classes"],
        "units": {"A_train": a_train.tolist(), "A_eval": a_eval.tolist(),
                  "B_train": b_train.tolist(), "B_eval": b_eval.tolist()},
        "shift": {"kind": "declared per channel affine covariate shift",
                  "gain": [round(float(x), 5) for x in gain],
                  "offset": [round(float(x), 5) for x in offset]},
    }


def evaluate(model, row) -> dict:
    x, y, units = row
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + 256], None)[0].argmax(1) for i in range(0, len(x), 256)])
    correct = (pred == y).numpy()
    return {"accuracy": round(float(correct.mean()), 5),
            "per_unit_accuracy": {str(u): round(float(correct[units == u].mean()), 5)
                                  for u in np.unique(units) if (units == u).sum() >= 5}}


def shard(seed: int) -> dict:
    t0 = time.time()
    ctx = contexts(seed)
    torch.manual_seed(seed)
    model = A.build(family="gru", ch=ctx["channels"], classes=ctx["classes"], tier="small")
    xa, ya, _ = ctx["A_train"]
    pre = E.fit(model, None, xa, ya, train_groups=["core", "readout"], steps=Fx.STEPS,
                lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)
    frozen = copy.deepcopy(model)
    adapted = copy.deepcopy(model)
    before = {"A": evaluate(frozen, ctx["A_eval"]), "B": evaluate(frozen, ctx["B_eval"])}
    xb, yb, _ = ctx["B_train"]
    adapt = E.fit(adapted, None, xb, yb, train_groups=["core", "readout"], steps=ADAPT_STEPS,
                  lr=Fx.LR, rng=np.random.default_rng(90_000 + seed), batch=Fx.BATCH)
    after = {"A": evaluate(adapted, ctx["A_eval"]), "B": evaluate(adapted, ctx["B_eval"])}
    return_before = after["A"]
    returned = copy.deepcopy(adapted)
    recover = E.fit(returned, None, xa, ya, train_groups=["readout"], steps=RETURN_STEPS,
                    lr=Fx.LR, rng=np.random.default_rng(100_000 + seed), batch=Fx.BATCH)
    return_after = evaluate(returned, ctx["A_eval"])
    unit_sets = {k: set(v) for k, v in ctx["units"].items()}
    checks = {"unit_disjoint": not any(unit_sets[a] & unit_sets[b]
                                        for i, a in enumerate(unit_sets) for b in list(unit_sets)[i + 1:]),
              "pretrain_budget": pre["updates"] == Fx.STEPS,
              "adapt_budget": adapt["updates"] == ADAPT_STEPS,
              "return_budget": recover["updates"] == RETURN_STEPS,
              "no_undeclared_changes": not (pre["undeclared_changes"] or adapt["undeclared_changes"]
                                             or recover["undeclared_changes"])}
    doc = {
        "schema": "mop-harth-admission-probe-shard/v1", "bed": "harth_stream", "seed": seed,
        "units": ctx["units"], "shift": ctx["shift"], "before_adaptation": before,
        "after_B_adaptation": after, "return_before_recovery": return_before,
        "return_after_recovery": return_after,
        "receipts": {"pretrain": pre, "adapt_B": adapt, "recover_A_readout": recover},
        "checkpoints": {"pretrained": E.checkpoint_sha(frozen), "adapted_B": E.checkpoint_sha(adapted),
                        "returned_A": E.checkpoint_sha(returned)},
        "checks": checks, "all_checks_pass": all(checks.values()),
        "test_split_untouched": True, "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"harth_preflight_{seed}.json", doc, "third_bed_preflight")
    print(f"HARTH preflight seed {seed}: B {before['B']['accuracy']}->{after['B']['accuracy']} "
          f"return {return_before['accuracy']}->{return_after['accuracy']}", flush=True)
    return doc


def aggregate() -> dict:
    rows = [json.loads((io.RUNS / "third_bed_preflight" / f"harth_preflight_{s}.json").read_text())
            for s in SEEDS]
    gains = [r["after_B_adaptation"]["B"]["accuracy"] - r["before_adaptation"]["B"]["accuracy"]
             for r in rows]
    returns = [r["return_after_recovery"]["accuracy"] - r["return_before_recovery"]["accuracy"]
               for r in rows]
    unit_effects: dict[str, list[float]] = {}
    for row in rows:
        before = row["before_adaptation"]["B"]["per_unit_accuracy"]
        after = row["after_B_adaptation"]["B"]["per_unit_accuracy"]
        for unit in set(before) & set(after):
            unit_effects.setdefault(unit, []).append(after[unit] - before[unit])
    unit_means = [float(np.mean(v)) for v in unit_effects.values()]
    boundary = bed.context_boundary_over_seeds([
        {"no_adapt_new": r["before_adaptation"]["B"]["accuracy"],
         "no_adapt_old": r["before_adaptation"]["A"]["accuracy"],
         "adapted_new": r["after_B_adaptation"]["B"]["accuracy"],
         "adapted_old": r["after_B_adaptation"]["A"]["accuracy"]} for r in rows])
    gain_decision, return_decision = power.decide(gains, e2.PREREG), power.decide(returns, e2.PREREG)
    group_lcb = power.lcb(unit_means) if len(unit_means) > 1 else None
    checks = {
        "all_eight_shards_valid": len(rows) == 8 and all(r["all_checks_pass"] for r in rows),
        "context_boundary_crossed": boundary["checks"]["boundary_crossed"],
        "future_adaptation_headroom": gain_decision["verdict"] == "positive"
        and (group_lcb or float("-inf")) >= io.SESOI,
        "returning_context_recovery": return_decision["verdict"] == "positive",
        "natural_independent_units": len(unit_means) >= 2,
    }
    doc = {
        "schema": "mop-harth-admission-probe/v1", "bed": "harth_stream", "seeds": list(SEEDS),
        "context_boundary": boundary,
        "future_adaptation": {**gain_decision, "group_lower_95_cb": group_lcb,
                              "per_seed_effects": gains, "n_units": len(unit_means)},
        "returning_context": {**return_decision, "per_seed_effects": returns},
        "checks": checks, "all_pass": all(checks.values()),
        "classification": "preflight_pass" if all(checks.values()) else "preflight_failed",
        "shards": [{"path": f"runs/substrate/{io.PROGRAM}/third_bed_preflight/harth_preflight_{s}.json",
                    "sha256": io.sha_file(io.RUNS / "third_bed_preflight" / f"harth_preflight_{s}.json")}
                   for s in SEEDS],
        "claim_ceiling": "secondary natural bed with a declared synthetic covariate-shift admission task",
    }
    io.seal("MOP_THIRD_TEMPORAL_BED_ADMISSION_PROBE.json", doc)
    print(f"HARTH admission probe: {doc['classification']}", flush=True)
    return doc


def run_all() -> dict:
    from mop.temporal.runs import supervisor

    names = [f"harth_preflight_{s}" for s in SEEDS]
    while True:
        pending = supervisor.missing("third_bed_preflight", names)
        if not pending:
            return aggregate()
        for name in pending:
            seed = name.rsplit("_", 1)[1]
            supervisor.launch(["shard", seed], "third_bed_preflight.log", f"tb:{name}",
                              module="mop.temporal.runs.thirdbed")
        time.sleep(5)


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] == "all":
        run_all()
    elif argv[0] == "aggregate":
        aggregate()
    elif argv[0] == "shard":
        shard(int(argv[1]))
    else:
        raise ValueError(argv)
    lock = os.environ.get("TEMPORAL_SHARD_LOCK")
    if lock:
        from pathlib import Path
        Path(lock).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
