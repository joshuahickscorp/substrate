"""E3 shared versus domain local temporal representation, sharded by direction and seed."""

from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import torch

from fastforge import engine as E
from mop.method import power
from mop.temporal import analysis as AN
from mop.temporal import arch as A
from mop.temporal import beds as B
from mop.temporal import factorial as Fx
from mop.temporal import io
from mop.temporal.runs import e2

DIRECTIONS = (("har_stream", "speech_stream"), ("speech_stream", "har_stream"))
SEEDS = e2.PRINCIPAL_SEEDS
WIDTH = 76
STEPS = Fx.STEPS
SHARED_STEPS = Fx.STEPS
ARMS = (
    "fresh_component",
    "frozen_transferred_component",
    "fine_tuned_transferred_component",
    "random_transferred_component",
    "wrong_bed_transferred_component",
    "shared_component",
    "domain_local_component",
    "oracle_assignment",
)


def _model(sp: dict, seed: int):
    torch.manual_seed(seed)
    model = A.build(family="gru", ch=sp["channels"], classes=sp["classes"], width=WIDTH,
                    readout="linear")
    names = list(dict(model.named_parameters()))
    model.param_groups.update({
        "local": [n for n in names if n.startswith("core.proj") or n.startswith("head")],
        "shared": [n for n in names if n.startswith("core.rnn") or n.startswith("core.out")],
        "input_to_hidden": [n for n in names if "core.rnn.weight_ih" in n or "core.rnn.bias_ih" in n],
        "hidden_to_hidden": [n for n in names if "core.rnn.weight_hh" in n],
        "hidden_bias": [n for n in names if "core.rnn.bias_hh" in n],
        "output_projection": [n for n in names if n.startswith("core.out")],
    })
    return model


def _copy_group(dst, src, group: str) -> list[str]:
    names = dst.param_groups[group]
    ds, ss = dst.state_dict(), src.state_dict()
    copied = []
    with torch.no_grad():
        for n in names:
            if n in ss and n in ds and ss[n].shape == ds[n].shape:
                ds[n].copy_(ss[n])
                copied.append(n)
    dst.load_state_dict(ds)
    return copied


def _fit(model, sp: dict, seed: int, groups: list[str], steps: int = STEPS) -> dict:
    receipt = E.fit(model, None, sp["main"][0], sp["main"][1], train_groups=groups, steps=steps,
                    lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)
    receipt.update({"optimizer": "Adam", "batch_seed": seed})
    return receipt


def _target_training_match(local: dict, shared: dict) -> dict:
    """Prove that transfer is the only difference in the target training comparison."""
    fields = ("updates", "batch", "lr", "optimizer", "batch_seed", "trainable_params",
              "trainable_param_count")
    checks = {f"same_{field}": local.get(field) == shared.get(field) for field in fields}
    checks.update({
        "same_minibatch_stream": checks["same_batch_seed"] and checks["same_batch"],
    })
    return {
        "optimizer": local.get("optimizer"),
        "batch_seed": local.get("batch_seed"),
        "parameter_exposure_per_arm": local.get("trainable_param_count", 0) * local.get("updates", 0),
        "checks": checks,
        "all_matched": all(checks.values()),
    }


def _evaluate(model, sp: dict) -> dict:
    x, y, units = sp["test"][0], sp["test"][1], np.asarray(sp["test_units"])
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(x[i:i + 256], None)[0].argmax(1) for i in range(0, len(x), 256)])
    correct = (pred == y).numpy()
    return {
        "accuracy": round(float(correct.mean()), 5),
        "per_unit_accuracy": {str(u): round(float(correct[units == u].mean()), 5)
                              for u in np.unique(units) if (units == u).sum() >= 5},
        "prediction_concentration": round(float(np.bincount(
            pred.numpy(), minlength=sp["classes"]).max() / len(pred)), 5),
    }


def _effect_summary(seed_effects: list[float], per_unit: dict[str, float],
                    prereg: dict, parameter_update_cost: int,
                    component_floor_status: dict, bed_specific: dict) -> dict:
    """Emit the complete effect contract used by every successor aggregate."""
    decision = power.decide(seed_effects, prereg)
    unit_values = list(per_unit.values())
    group_lower = power.lcb(unit_values) if len(unit_values) > 1 else None
    group_upper = -power.lcb([-x for x in unit_values]) if len(unit_values) > 1 else None
    mean = float(decision["mean"])
    return {
        **decision,
        "upper_95_cb": round(-power.lcb([-x for x in seed_effects]), 5)
        if len(seed_effects) > 1 else None,
        "per_seed_effects": [round(float(x), 5) for x in seed_effects],
        "per_unit_effects": {str(k): round(float(v), 5) for k, v in sorted(per_unit.items())},
        "group_mean": round(float(np.mean(unit_values)), 5) if unit_values else None,
        "group_lower_95_cb": round(group_lower, 5) if group_lower is not None else None,
        "group_upper_95_cb": round(group_upper, 5) if group_upper is not None else None,
        "group_heterogeneity": round(float(np.std(unit_values, ddof=1)), 5)
        if len(unit_values) > 1 else None,
        "n_units": len(unit_values),
        "bed_specific_effects": bed_specific,
        "cost_adjusted_effect_per_million_parameter_updates": (
            round(mean * 1_000_000 / parameter_update_cost, 8)
            if parameter_update_cost > 0 else None),
        "cost_denominator": {
            "parameter_update_exposure": int(parameter_update_cost),
            "unit": "trainable parameters multiplied by optimizer updates",
            "all_source_and_target_training_charged": True,
        },
        "component_floor_status": component_floor_status,
    }


def _arm(model, sp: dict, receipt: dict | None = None, source_retention: dict | None = None) -> dict:
    out = _evaluate(model, sp)
    out.update({"checkpoint_sha": E.checkpoint_sha(model), "params": A.count(model)})
    if receipt:
        out["training"] = {k: receipt[k] for k in (
            "steps", "updates", "trainable_groups", "trainable_params", "frozen_params",
            "undeclared_changes", "checkpoint_sha_after") if k in receipt}
    if source_retention:
        out["source_retention"] = source_retention
    return out


def shard(source: str, target: str, seed: int) -> dict:
    if (source, target) not in DIRECTIONS:
        raise ValueError(f"undeclared E3 direction {(source, target)}")
    t0 = time.time()
    ssp, tsp = B.splits(source, seed), B.splits(target, seed)

    source_model = _model(ssp, seed)
    source_receipt = _fit(source_model, ssp, seed, ["core", "readout"])
    source_base = _evaluate(source_model, ssp)

    target_initial = _model(tsp, seed)
    target_batch_seed = 50_000 + seed
    local = copy.deepcopy(target_initial)
    local_receipt = _fit(local, tsp, target_batch_seed, ["core", "readout"], SHARED_STEPS)
    domain_local = _arm(local, tsp, local_receipt)

    fresh = copy.deepcopy(local)
    fresh_donor = _model(tsp, 10_000 + seed)
    _copy_group(fresh, fresh_donor, "shared")

    random_transfer = copy.deepcopy(local)
    random_donor = _model(ssp, 20_000 + seed)
    _copy_group(random_transfer, random_donor, "shared")

    wrong_sp = B.splits("harth_stream", seed)
    wrong_donor = _model(wrong_sp, 60_000 + seed)
    wrong_receipt = _fit(wrong_donor, wrong_sp, 60_000 + seed, ["core", "readout"])
    wrong_bed = copy.deepcopy(local)
    wrong_copied = _copy_group(wrong_bed, wrong_donor, "shared")

    frozen = _model(tsp, 30_000 + seed)
    frozen_copied = _copy_group(frozen, source_model, "shared")
    frozen_receipt = _fit(frozen, tsp, 30_000 + seed, ["local"])

    fine = _model(tsp, 40_000 + seed)
    fine_copied = _copy_group(fine, source_model, "shared")
    fine_receipt = _fit(fine, tsp, 40_000 + seed, ["core", "readout"])

    shared = copy.deepcopy(target_initial)
    shared_copied = _copy_group(shared, source_model, "shared")
    shared_receipt = _fit(shared, tsp, target_batch_seed, ["core", "readout"], SHARED_STEPS)
    target_training_match = _target_training_match(local_receipt, shared_receipt)
    retained_source = copy.deepcopy(source_model)
    _copy_group(retained_source, shared, "shared")
    source_after = _evaluate(retained_source, ssp)
    retention = {
        "before": source_base["accuracy"],
        "after": source_after["accuracy"],
        "drop": round(source_base["accuracy"] - source_after["accuracy"], 5),
        "floor_met": source_after["accuracy"] >= source_base["accuracy"] - io.SESOI,
    }

    component_interventions = {}
    for component in ("input_to_hidden", "hidden_to_hidden", "hidden_bias", "output_projection"):
        replaced = copy.deepcopy(local)
        copied = _copy_group(replaced, source_model, component)
        score = _evaluate(replaced, tsp)
        component_interventions[component] = {
            "copied_parameters": copied,
            "accuracy_after_causal_replacement": score["accuracy"],
            "delta_from_domain_local": round(score["accuracy"] - domain_local["accuracy"], 5),
        }
    component_interventions.update({
        "input_projection": {
            "classification": "domain_local_required",
            "source_shape": list(source_model.core.proj.weight.shape),
            "target_shape": list(local.core.proj.weight.shape),
            "intervention": "kept target local in every transfer arm because channel identities differ",
        },
        "normalization": {"classification": "absent_from_selected_core"},
        "readout": {
            "classification": "domain_local_required",
            "source_shape": list(source_model.head.weight.shape),
            "target_shape": list(local.head.weight.shape),
            "intervention": "kept target local because label spaces differ",
        },
        "initial_state": {
            "classification": "fixed_zero_not_an_owned_parameter",
            "intervention": "no transferable parameter exists in Owned Temporal Core v1",
        },
    })

    arms = {
        "fresh_component": _arm(fresh, tsp),
        "frozen_transferred_component": _arm(frozen, tsp, frozen_receipt),
        "fine_tuned_transferred_component": _arm(fine, tsp, fine_receipt),
        "random_transferred_component": _arm(random_transfer, tsp),
        "wrong_bed_transferred_component": _arm(wrong_bed, tsp),
        "shared_component": _arm(shared, tsp, shared_receipt, retention),
        "domain_local_component": domain_local,
    }
    oracle_name = max(("shared_component", "domain_local_component"),
                      key=lambda k: arms[k]["accuracy"])
    arms["oracle_assignment"] = dict(arms[oracle_name], selected_arm=oracle_name,
                                     oracle_uses_test_outcome=True)
    doc = {
        "schema": "mop-e3-shard/v1",
        "source_bed": source,
        "target_bed": target,
        "seed": seed,
        "width": WIDTH,
        "capacity_band": list(A.TIER_RANGE["small"]),
        "arms": arms,
        "component_interventions": component_interventions,
        "copied_parameter_sets": {
            "frozen": frozen_copied, "fine_tuned": fine_copied, "shared": shared_copied,
            "wrong_bed": wrong_copied,
        },
        "wrong_bed_control": {"donor_bed": "harth_stream", "intended_source_bed": source,
                              "distinct_bed": source != "harth_stream",
                              "donor_training": wrong_receipt},
        "source_training": source_receipt,
        "source_baseline": source_base,
        "target_training_match": target_training_match,
        "arm_distinctness": {
            "n_arms": len(arms),
            "n_unique_checkpoints_excluding_oracle_alias": len({
                v["checkpoint_sha"] for k, v in arms.items() if k != "oracle_assignment"}),
            "all_nonoracle_arms_distinct": len({
                v["checkpoint_sha"] for k, v in arms.items() if k != "oracle_assignment"}) == len(arms) - 1,
        },
        "authority_commit": io.commit(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"{source}_to_{target}_{seed}.json", doc, "e3")
    print(f"E3 {source} -> {target} seed {seed}: local {domain_local['accuracy']}, "
          f"shared {arms['shared_component']['accuracy']} in {doc['wall_seconds']}s", flush=True)
    print("E3_SHARD_DONE", flush=True)
    return doc


def aggregate() -> dict:
    shards, missing = [], []
    for source, target in DIRECTIONS:
        for seed in SEEDS:
            p = io.RUNS / "e3" / f"{source}_to_{target}_{seed}.json"
            if p.is_file():
                shards.append(json.loads(p.read_text()))
            else:
                missing.append(p.relative_to(io.ROOT).as_posix())
    if missing:
        raise RuntimeError(f"E3 cannot aggregate with missing shards: {missing}")
    per_direction, direction_verdicts = {}, []
    for source, target in DIRECTIONS:
        rows = [d for d in shards if d["source_bed"] == source and d["target_bed"] == target]
        scores = {a: [d["arms"][a]["accuracy"] for d in rows] for a in ARMS}
        effects = [d["arms"]["shared_component"]["accuracy"]
                   - d["arms"]["domain_local_component"]["accuracy"] for d in rows]
        unit_effects = {}
        for d in rows:
            shared_units = d["arms"]["shared_component"]["per_unit_accuracy"]
            local_units = d["arms"]["domain_local_component"]["per_unit_accuracy"]
            for unit in set(shared_units) & set(local_units):
                unit_effects.setdefault(unit, []).append(shared_units[unit] - local_units[unit])
        per_unit = {unit: float(np.mean(v)) for unit, v in unit_effects.items()}
        unit_means = list(per_unit.values())
        group_lcb = power.lcb(unit_means) if len(unit_means) > 1 else None
        inverse_group_lcb = power.lcb([-x for x in unit_means]) if len(unit_means) > 1 else None
        retention = [d["arms"]["shared_component"]["source_retention"] for d in rows]
        parameter_update_cost = int(round(float(np.mean([
            int(d["source_training"].get("trainable_param_count", 0))
            * int(d["source_training"].get("updates", 0))
            + 2 * int(d["target_training_match"].get("parameter_exposure_per_arm", 0))
            for d in rows]))))
        decision = _effect_summary(
            effects, per_unit, e2.PREREG, parameter_update_cost,
            {"name": "source_retention", "per_seed": [r["floor_met"] for r in retention],
             "all_pass": all(r["floor_met"] for r in retention), "margin": io.SESOI},
            {target: {"mean": round(float(np.mean(effects)), 5),
                      "per_seed_effects": [round(float(x), 5) for x in effects]}})
        if decision["verdict"] == "positive" and group_lcb is not None and group_lcb >= io.SESOI \
                and all(r["floor_met"] for r in retention):
            classification = "shared_component_supported"
        else:
            inverse = power.decide([-x for x in effects], e2.PREREG)
            classification = ("domain_local_component_supported" if inverse["verdict"] == "positive"
                              and inverse_group_lcb is not None and inverse_group_lcb >= io.SESOI
                              else "shared_and_domain_local_inconclusive")
        direction_verdicts.append(classification)
        per_direction[f"{source}_to_{target}"] = {
            "arm_scores": scores,
            "shared_minus_domain_local": {
                **decision,
                "inverse_group_lower_95_cb": round(inverse_group_lcb, 5)
                if inverse_group_lcb is not None else None,
            },
            "retention": retention,
            "classification": classification,
        }
    if all(v == "shared_component_supported" for v in direction_verdicts):
        overall = "shared_temporal_representation_supported"
    elif all(v == "domain_local_component_supported" for v in direction_verdicts):
        overall = "domain_local_temporal_representation_supported"
    else:
        overall = "direction_dependent_or_inconclusive"
    result = {
        "schema": "mop-e3-shared-local-result/v1",
        "experiment_terminal": True,
        "result": {
            "classification": overall,
            "per_direction": per_direction,
            "arms": list(ARMS),
            "n_shards": len(shards),
            "n_seeds_per_direction": len(SEEDS),
            "common_width": WIDTH,
            "component_interventions": {f"{d['source_bed']}_to_{d['target_bed']}_seed_{d['seed']}":
                                        d["component_interventions"] for d in shards},
            "compute": {"supervised_steps_per_full_arm": STEPS,
                        "matched_target_steps_per_comparison_arm": SHARED_STEPS,
                        "pretraining_compute_charged": True,
                        "target_comparison_resource_matched": all(
                            d["target_training_match"]["all_matched"] for d in shards)},
            "claim_ceiling": "causal component sharing on the two principal controlled beds",
        },
        "all_shards_terminal": True,
        "all_nonoracle_arms_distinct": all(
            d["arm_distinctness"]["all_nonoracle_arms_distinct"] for d in shards),
        "mutation_checks": {
            "all_required_arms_present": all(set(d["arms"]) == set(ARMS) for d in shards),
            "wrong_bed_donor_is_distinct": all(
                d["wrong_bed_control"]["distinct_bed"] for d in shards),
            "all_component_replacements_recorded": all(set(d["component_interventions"]) == {
                "input_projection", "normalization", "input_to_hidden", "hidden_to_hidden",
                "hidden_bias", "output_projection", "readout", "initial_state"} for d in shards),
            "all_transfer_arms_have_distinct_checkpoints": all(
                d["arm_distinctness"]["all_nonoracle_arms_distinct"] for d in shards),
            "shared_and_local_target_training_matched": all(
                d["target_training_match"]["all_matched"] for d in shards),
        },
    }
    io.seal("MOP_E3_SHARED_LOCAL_RESULT.json", result)
    print(f"E3 aggregate: {overall}, {len(shards)} shards", flush=True)
    print("E3_AGGREGATE_DONE", flush=True)
    return result


def main(argv=None):
    argv = argv or sys.argv[1:]
    if not argv or argv[0] == "aggregate":
        aggregate()
    elif argv[0] == "shard":
        shard(argv[1], argv[2], int(argv[3]))
    else:
        raise ValueError(argv)
    lock = os.environ.get("TEMPORAL_SHARD_LOCK")
    if lock:
        from pathlib import Path
        Path(lock).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
