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
from mop.temporal.runs import e2, e3

SEEDS = e2.PRINCIPAL_SEEDS
ADAPT_STEPS = Fx.STEPS // 4
RETURN_STEPS = Fx.STEPS // 8


def contexts(seed: int) -> dict:
    d = B.load("harth_stream")
    units = np.asarray(d["u"])
    unique = np.random.default_rng(70_000 + seed).permutation(np.unique(units))
    half = len(unique) // 2

    def partition(group, n_eval):
        return group[n_eval:], group[:n_eval]

    total_eval = max(2, int(round(0.3 * len(unique))))
    a_train, a_eval = partition(unique[:half], (total_eval + 1) // 2)
    b_train, b_eval = partition(unique[half:], total_eval // 2)

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


def permute_time(row, seed: int):
    """Destroy within window order while preserving every example, label and timestep multiset."""
    x, y, units = row
    order = torch.as_tensor(np.random.default_rng(110_000 + seed).permutation(x.shape[1])).long()
    return x[:, order, :], y, units


def _training_exposure(receipt: dict) -> int:
    return int(receipt.get("trainable_param_count", 0)) * int(receipt.get("updates", 0))


def shard(seed: int) -> dict:
    t0 = time.time()
    ctx = contexts(seed)
    torch.manual_seed(seed)
    model = A.build(family="gru", ch=ctx["channels"], classes=ctx["classes"], tier="small")
    xa, ya, _ = ctx["A_train"]
    pre = E.fit(model, None, xa, ya, train_groups=["core", "readout"], steps=Fx.STEPS,
                lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)
    frozen = copy.deepcopy(model)
    torch.manual_seed(seed)
    pooled = A.build(family="pooled", ch=ctx["channels"], classes=ctx["classes"], tier="small")
    pooled_pre = E.fit(pooled, None, xa, ya, train_groups=["core", "readout"], steps=Fx.STEPS,
                       lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)
    ordered_order = evaluate(frozen, ctx["A_eval"])
    permuted_order = evaluate(frozen, permute_time(ctx["A_eval"], seed))
    pooled_ordered = evaluate(pooled, ctx["A_eval"])
    pooled_permuted = evaluate(pooled, permute_time(ctx["A_eval"], seed))
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
              "pooled_control_budget": pooled_pre["updates"] == Fx.STEPS,
              "adapt_budget": adapt["updates"] == ADAPT_STEPS,
              "return_budget": recover["updates"] == RETURN_STEPS,
              "fifteen_train_seven_untouched_evaluation_units": (
                  len(unit_sets["A_train"] | unit_sets["B_train"]) == 15
                  and len(unit_sets["A_eval"] | unit_sets["B_eval"]) == 7),
              "no_undeclared_changes": not (pre["undeclared_changes"] or adapt["undeclared_changes"]
                                             or recover["undeclared_changes"])}
    doc = {
        "schema": "mop-harth-admission-probe-shard/v1", "bed": "harth_stream", "seed": seed,
        "units": ctx["units"], "shift": ctx["shift"], "before_adaptation": before,
        "after_B_adaptation": after, "return_before_recovery": return_before,
        "return_after_recovery": return_after,
        "temporal_order_permutation": {
            "intervention": "within_window_timestep_permutation", "labels_unchanged": True,
            "timestep_multiset_preserved_per_example": True,
            "ordered_accuracy": ordered_order["accuracy"], "permuted_accuracy": permuted_order["accuracy"],
            "pooled_ordered_accuracy": pooled_ordered["accuracy"],
            "pooled_permuted_accuracy": pooled_permuted["accuracy"],
            "ordered_per_unit_accuracy": ordered_order["per_unit_accuracy"],
            "permuted_per_unit_accuracy": permuted_order["per_unit_accuracy"],
            "pooled_ordered_per_unit_accuracy": pooled_ordered["per_unit_accuracy"],
            "pooled_permuted_per_unit_accuracy": pooled_permuted["per_unit_accuracy"],
            "resource_match": {"same_examples": True, "same_model_checkpoint": True,
                               "same_evaluation_code": True}},
        "receipts": {"pretrain": pre, "pooled_control_pretrain": pooled_pre,
                     "adapt_B": adapt, "recover_A_readout": recover},
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
    return_unit_effects: dict[str, list[float]] = {}
    for row in rows:
        before = row["before_adaptation"]["B"]["per_unit_accuracy"]
        after = row["after_B_adaptation"]["B"]["per_unit_accuracy"]
        for unit in set(before) & set(after):
            unit_effects.setdefault(unit, []).append(after[unit] - before[unit])
        return_before = row["return_before_recovery"]["per_unit_accuracy"]
        return_after = row["return_after_recovery"]["per_unit_accuracy"]
        for unit in set(return_before) & set(return_after):
            return_unit_effects.setdefault(unit, []).append(return_after[unit] - return_before[unit])
    unit_means = {unit: float(np.mean(v)) for unit, v in unit_effects.items()}
    return_unit_means = {unit: float(np.mean(v)) for unit, v in return_unit_effects.items()}
    order_units, pooled_units = {}, {}
    for row in rows:
        witness = row["temporal_order_permutation"]
        ordered, permuted = witness["ordered_per_unit_accuracy"], witness["permuted_per_unit_accuracy"]
        pooled_ordered = witness["pooled_ordered_per_unit_accuracy"]
        pooled_permuted = witness["pooled_permuted_per_unit_accuracy"]
        for unit in set(ordered) & set(permuted):
            order_units.setdefault(unit, []).append(ordered[unit] - permuted[unit])
        for unit in set(pooled_ordered) & set(pooled_permuted):
            pooled_units.setdefault(unit, []).append(pooled_ordered[unit] - pooled_permuted[unit])
    order_effects = {u: float(np.mean(v)) for u, v in order_units.items()}
    pooled_effects = {u: float(np.mean(v)) for u, v in pooled_units.items()}
    order_lcb = power.lcb(order_effects.values()) if len(order_effects) > 1 else None
    pooled_lower = power.lcb(pooled_effects.values()) if len(pooled_effects) > 1 else None
    pooled_upper = -power.lcb([-x for x in pooled_effects.values()]) if len(pooled_effects) > 1 else None
    order_seed_effects = [r["temporal_order_permutation"]["ordered_accuracy"] -
                          r["temporal_order_permutation"]["permuted_accuracy"] for r in rows]
    order_seed_decision = power.decide(order_seed_effects, e2.PREREG)
    boundary = bed.context_boundary_over_seeds([
        {"no_adapt_new": r["before_adaptation"]["B"]["accuracy"],
         "no_adapt_old": r["before_adaptation"]["A"]["accuracy"],
         "adapted_new": r["after_B_adaptation"]["B"]["accuracy"],
         "adapted_old": r["after_B_adaptation"]["A"]["accuracy"]} for r in rows])
    future_floors = [r["after_B_adaptation"]["A"]["accuracy"]
                     >= r["before_adaptation"]["A"]["accuracy"] - io.SESOI for r in rows]
    return_floors = [r["return_after_recovery"]["accuracy"]
                     >= r["before_adaptation"]["A"]["accuracy"] - io.SESOI for r in rows]
    future_cost = int(round(float(np.mean([_training_exposure(r["receipts"]["adapt_B"])
                                           for r in rows]))))
    return_cost = int(round(float(np.mean([_training_exposure(r["receipts"]["recover_A_readout"])
                                           for r in rows]))))
    order_cost = int(round(float(np.mean([_training_exposure(r["receipts"]["pretrain"])
                                          for r in rows]))))
    pooled_cost = int(round(float(np.mean([_training_exposure(r["receipts"]["pooled_control_pretrain"])
                                           for r in rows]))))
    def bed_effect(values):
        return {"harth_stream": {"mean": round(float(np.mean(values)), 5),
                                  "per_seed_effects": [round(float(x), 5) for x in values]}}
    gain_decision = e3._effect_summary(gains, unit_means, e2.PREREG, future_cost, {
        "name": "A_context_retention_during_B_adaptation", "per_seed": future_floors,
        "all_pass": all(future_floors), "margin": io.SESOI}, bed_effect(gains))
    return_decision = e3._effect_summary(returns, return_unit_means, e2.PREREG, return_cost, {
        "name": "return_to_A_accuracy", "per_seed": return_floors,
        "all_pass": all(return_floors), "margin": io.SESOI}, bed_effect(returns))
    order_decision = e3._effect_summary(order_seed_effects, order_effects, e2.PREREG, order_cost, {
        "name": "pooled_reader_order_invariance_control", "per_seed": [],
        "all_pass": pooled_lower is not None and pooled_lower >= -io.EQUIVALENCE_MARGIN
        and pooled_upper is not None and pooled_upper <= io.EQUIVALENCE_MARGIN,
        "margin": io.EQUIVALENCE_MARGIN}, bed_effect(order_seed_effects))
    pooled_seed_effects = []
    for r in rows:
        witness = r["temporal_order_permutation"]
        ordered = witness.get("pooled_ordered_accuracy")
        permuted = witness.get("pooled_permuted_accuracy")
        if ordered is None or permuted is None:
            ordered = float(np.mean(list(witness["pooled_ordered_per_unit_accuracy"].values())))
            permuted = float(np.mean(list(witness["pooled_permuted_per_unit_accuracy"].values())))
        pooled_seed_effects.append(float(ordered) - float(permuted))
    pooled_decision = e3._effect_summary(pooled_seed_effects, pooled_effects, e2.PREREG, pooled_cost, {
        "name": "order_invariance_equivalence", "per_seed": [],
        "all_pass": pooled_lower is not None and pooled_lower >= -io.EQUIVALENCE_MARGIN
        and pooled_upper is not None and pooled_upper <= io.EQUIVALENCE_MARGIN,
        "margin": io.EQUIVALENCE_MARGIN}, bed_effect(pooled_seed_effects))
    group_lcb = gain_decision["group_lower_95_cb"]
    checks = {
        "all_eight_shards_valid": len(rows) == 8 and all(r["all_checks_pass"] for r in rows),
        "context_boundary_crossed": boundary["checks"]["boundary_crossed"],
        "future_adaptation_headroom": gain_decision["verdict"] == "positive"
        and (group_lcb or float("-inf")) >= io.SESOI,
        "returning_context_recovery": return_decision["verdict"] == "positive",
        "natural_independent_units": len(unit_means) >= 2,
        "temporal_order_required": order_seed_decision["verdict"] == "positive"
        and order_lcb is not None and order_lcb >= io.SESOI,
        "pooled_reader_order_invariant": pooled_lower is not None and pooled_lower >= -io.EQUIVALENCE_MARGIN
        and pooled_upper is not None and pooled_upper <= io.EQUIVALENCE_MARGIN,
    }
    doc = {
        "schema": "mop-harth-admission-probe/v1", "bed": "harth_stream", "seeds": list(SEEDS),
        "context_boundary": boundary,
        "future_adaptation": gain_decision,
        "returning_context": return_decision,
        "temporal_order_permutation": {
            **order_decision,
            "intervention": "within_window_timestep_permutation", "labels_unchanged": True,
            "ordered_per_unit_accuracy": {u: round(float(np.mean([
                r["temporal_order_permutation"]["ordered_per_unit_accuracy"][u] for r in rows
                if u in r["temporal_order_permutation"]["ordered_per_unit_accuracy"]])), 5)
                for u in order_effects},
            "permuted_per_unit_accuracy": {u: round(float(np.mean([
                r["temporal_order_permutation"]["permuted_per_unit_accuracy"][u] for r in rows
                if u in r["temporal_order_permutation"]["permuted_per_unit_accuracy"]])), 5)
                for u in order_effects},
            "seed_decision": order_seed_decision,
            "pooled_per_unit_effects": {u: round(v, 5) for u, v in pooled_effects.items()},
            "pooled_group_lower_95_cb": round(pooled_lower, 5) if pooled_lower is not None else None,
            "pooled_group_upper_95_cb": round(pooled_upper, 5) if pooled_upper is not None else None,
            "pooled_control_effect": pooled_decision,
            "resource_match": {"same_examples": True, "same_model_checkpoint": True,
                               "same_evaluation_code": True}},
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
        pending = supervisor.recoverable_pending("third_bed_preflight", names)
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
