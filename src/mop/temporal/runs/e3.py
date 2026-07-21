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
SHARED_STEPS = Fx.STEPS // 2
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
    return E.fit(model, None, sp["main"][0], sp["main"][1], train_groups=groups, steps=steps,
                 lr=Fx.LR, rng=np.random.default_rng(seed), batch=Fx.BATCH)


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

    local = _model(tsp, seed)
    local_receipt = _fit(local, tsp, seed, ["core", "readout"])
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

    shared = copy.deepcopy(local)
    shared_copied = _copy_group(shared, source_model, "shared")
    shared_receipt = _fit(shared, tsp, 50_000 + seed, ["shared"], SHARED_STEPS)
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
        decision = power.decide(effects, e2.PREREG)
        unit_effects = {}
        for d in rows:
            shared_units = d["arms"]["shared_component"]["per_unit_accuracy"]
            local_units = d["arms"]["domain_local_component"]["per_unit_accuracy"]
            for unit in set(shared_units) & set(local_units):
                unit_effects.setdefault(unit, []).append(shared_units[unit] - local_units[unit])
        unit_means = [float(np.mean(v)) for v in unit_effects.values()]
        group_lcb = power.lcb(unit_means) if len(unit_means) > 1 else None
        retention = [d["arms"]["shared_component"]["source_retention"] for d in rows]
        if decision["verdict"] == "positive" and (group_lcb or float("-inf")) >= io.SESOI \
                and all(r["floor_met"] for r in retention):
            classification = "shared_component_supported"
        else:
            inverse = power.decide([-x for x in effects], e2.PREREG)
            classification = ("domain_local_component_supported" if inverse["verdict"] == "positive"
                              else "shared_and_domain_local_inconclusive")
        direction_verdicts.append(classification)
        per_direction[f"{source}_to_{target}"] = {
            "arm_scores": scores,
            "shared_minus_domain_local": {**decision, "per_seed_effects": effects,
                                           "group_lower_95_cb": group_lcb,
                                           "n_units": len(unit_means)},
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
                        "shared_only_steps": SHARED_STEPS,
                        "pretraining_compute_charged": True},
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
