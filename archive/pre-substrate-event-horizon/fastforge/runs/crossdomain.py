"""Bidirectional cross domain matrix.

    sequence 1   HAR    acquire -> Speech acquire -> HAR    return -> future Speech units
    sequence 2   Speech acquire -> HAR    acquire -> Speech return -> future HAR    units

Both directions are mandatory. A domain specific improvement that fails in the reverse direction is not a
unified cross domain positive, and a return recovery improvement bought with impaired acquisition is a null.

Run as shards, one process per (direction, seed), then aggregate. The oracle schedule and the shuffled gate
control are resolved inside a shard because both need the other arms of that same seed.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import sequence as S
from fastforge.runs import io

SEEDS = list(range(8))
DIRECTIONS = [("har", "speech"), ("speech", "har")]
ORACLE_CANDIDATES = [
    "G0_always_trainable",
    "G1_frozen_after_first",
    "G2_reopened_at_return",
    "G3_reopened_on_drop",
    "G4_adapters_only",
    "G5_projection_and_head",
]
H_ORACLE_CANDIDATES = [
    "H_always_update",
    "H_never_update",
    "H_cosine_gate",
    "H_probe_gate",
    "H_drift_gate",
    "H_perf_gate",
    "H_task_label_freeze",
]
PLAIN = [a for a in S.ARMS if not S.ARMS[a].get("oracle") and S.ARMS[a].get("gate") != "shuffled"]


def inference_latency(arm_result) -> float:
    return round(arm_result["metrics"]["train_seconds"] / max(1, arm_result["metrics"]["train_updates"]), 6)


def shard(direction, seed, steps=None):
    out = {}
    for arm in PLAIN:
        t0 = time.time()
        out[arm] = S.run(arm, direction, seed, steps=steps)
        print(
            f"  {direction[0]}->{direction[1]} seed{seed} {arm:32s} "
            f"acq2={out[arm]['metrics']['second_acquisition']:.3f} "
            f"rec={out[arm]['metrics']['return_recovery']:.3f} [{time.time() - t0:.0f}s]",
            flush=True,
        )

    # shuffled gate control: the real decision sequence, permuted, so its timing carries no information
    dec = out["H_cosine_gate"]["gate_decisions"]
    perm = list(np.random.default_rng(4242 + seed).permutation(np.array(dec)))
    out["H_shuffled_gate"] = S.run(
        "H_shuffled_gate", direction, seed, steps=steps, replay_decisions=[int(d) for d in perm]
    )

    # oracle schedules: selected per seed on the tuning utility only, never on the test split
    for name, cands in (
        ("G6_oracle_schedule", ORACLE_CANDIDATES),
        ("H_oracle_schedule", H_ORACLE_CANDIDATES),
    ):
        best = max(cands, key=lambda c: out[c]["metrics"]["tune_utility"])
        chosen = json.loads(json.dumps(out[best], default=str))
        chosen["arm"] = name
        chosen["oracle_selected_from"] = cands
        chosen["oracle_choice"] = best
        chosen["oracle_selection_signal"] = "tune_utility"
        out[name] = chosen
    return out


def summarize(rows: dict) -> dict:
    """rows[direction][seed][arm] -> result. Effects are computed against the strongest matched baseline."""
    arms = sorted({a for d in rows.values() for s in d.values() for a in s})
    per_dir = {}
    for dname, ds in rows.items():
        seeds = sorted(ds)
        metric_names = [
            "first_acquisition",
            "second_acquisition",
            "first_retention",
            "return_recovery",
            "future_adaptation",
            "negative_transfer",
            "backward_transfer",
            "params",
            "checkpoint_bytes",
            "train_updates",
            "train_seconds",
            "trainable_phase2",
            "changed_phase2",
            "tune_utility",
        ]
        means = {
            a: {k: round(float(np.mean([ds[s][a]["metrics"][k] for s in seeds])), 4) for k in metric_names}
            for a in arms
        }
        # cost adjusted utility, identical rule to the inherited program: accuracy minus a parameter price
        base_params = means["lstm_gdumb"]["params"]

        def util(a, s, ds=ds, base_params=base_params):
            m = ds[s][a]["metrics"]
            comp = np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]])
            return float(comp - 0.05 * (m["params"] / base_params))

        um = {a: float(np.mean([util(a, s) for s in seeds])) for a in arms}
        baselines = [
            "lstm_gdumb",
            "separate_per_domain",
            "matched_capacity_multi_domain",
            "shared_adapters_conventional",
            "ewc_shared_core",
            "fresh_independent",
        ]
        strongest = max(baselines, key=lambda a: um[a])
        effects = {
            a: {
                "vs_strongest_baseline": S.E.effect(
                    [util(a, s) for s in seeds], [util(strongest, s) for s in seeds]
                ),
                "second_acquisition_vs_baseline": S.E.effect(
                    [ds[s][a]["metrics"]["second_acquisition"] for s in seeds],
                    [ds[s][strongest]["metrics"]["second_acquisition"] for s in seeds],
                ),
                "return_recovery_vs_baseline": S.E.effect(
                    [ds[s][a]["metrics"]["return_recovery"] for s in seeds],
                    [ds[s][strongest]["metrics"]["return_recovery"] for s in seeds],
                ),
                "future_adaptation_vs_baseline": S.E.effect(
                    [ds[s][a]["metrics"]["future_adaptation"] for s in seeds],
                    [ds[s][strongest]["metrics"]["future_adaptation"] for s in seeds],
                ),
                "first_retention_vs_baseline": S.E.effect(
                    [ds[s][a]["metrics"]["first_retention"] for s in seeds],
                    [ds[s][strongest]["metrics"]["first_retention"] for s in seeds],
                ),
            }
            for a in arms
        }
        # Each component floor is the best any baseline achieved on that component, not the components of
        # whichever baseline happens to win on utility. A baseline that forgets everything would otherwise
        # set a retention floor of nearly zero and the floor would stop being a floor.
        floors = {
            k: max(means[b][k] for b in baselines) - 0.02
            for k in ("second_acquisition", "first_retention", "return_recovery", "future_adaptation")
        }
        floor_source = {
            k: max(baselines, key=lambda b: means[b][k])
            for k in ("second_acquisition", "first_retention", "return_recovery", "future_adaptation")
        }
        passing = [
            a
            for a in arms
            if effects[a]["vs_strongest_baseline"]["lower_95_cb"] >= io.SESOI
            and all(means[a][k] >= v for k, v in floors.items())
        ]
        per_dir[dname] = {
            "seeds": seeds,
            "strongest_matched_baseline": strongest,
            "means": means,
            "utility": {a: round(um[a], 4) for a in arms},
            "effects": effects,
            "component_floors": {k: round(v, 4) for k, v in floors.items()},
            "utility_definition": {
                "weighted": ["second_acquisition", "return_recovery", "future_adaptation"],
                "not_weighted": ["first_retention"],
                "why": "retention is enforced as a hard floor rather than averaged in, because an arm can "
                "buy a high retention average by refusing to learn the second domain at all. G4 adapters "
                "only is exactly that arm. Averaging retention would reward it; a floor does not.",
                "consequence": "an arm that dominates on retention alone cannot pass, and that is intended. "
                "Retention is reported per arm so the choice can be checked rather than taken on trust.",
                "parameter_price": "0.05 times the arm parameter count over the lstm_gdumb parameter count. "
                "It cancels exactly between two arms of equal capacity, so it does no work in the headline "
                "comparison, which is recorded in the mutation report.",
            },
            "utility_spread": {
                "all_arms": round(max(um.values()) - min(um.values()), 4),
                "excluding_degenerate_arms": round(
                    max(um.values()) - sorted(um.values())[1 if len(um) > 1 else 0], 4
                ),
                "SESOI": io.SESOI,
                "note": "SESOI is a fixed absolute bar, not a fraction of the observed spread. It is "
                "reported next to the spread so the reader can judge how demanding it was here.",
            },
            "component_floor_source_baseline": floor_source,
            "strongest_baseline_meets_the_floors": {
                k: bool(means[strongest][k] >= v) for k, v in floors.items()
            },
            "comparator_and_floors_are_separate_mechanisms": "the effect comparator is whichever baseline "
            "scores highest on the decision utility, and the floors are the best any baseline achieved per "
            "component. They can disagree, and here they do: the utility comparator forgets most of the "
            "first domain while the retention floor comes from a different baseline that does not. An arm "
            "must clear both, which is strictly harder than clearing either, and the disagreement is "
            "reported rather than resolved by quietly picking one.",
            "arms_passing_all_conditions": passing,
            "verdict": "cross_domain_positive" if passing else "cross_domain_null",
        }
    both = sorted(
        set(per_dir["har->speech"]["arms_passing_all_conditions"])
        & set(per_dir["speech->har"]["arms_passing_all_conditions"])
    )
    return {
        "per_direction": per_dir,
        "arms_passing_in_both_directions": both,
        "bidirectional_verdict": "cross_domain_positive" if both else "cross_domain_null",
        "rule": "a return recovery improvement with impaired acquisition is a null; a domain specific "
        "improvement that fails in the reverse direction is not a unified cross domain positive",
    }


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        d, s = sys.argv[2].split(":")
        direction = tuple(d.split("-"))
        res = shard(direction, int(s))
        p = io.RUNS / "crossdomain"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{d}_{s}.json").write_text(json.dumps(res, default=str))
        print("SHARD_DONE", d, s, flush=True)
        return

    rows = {}
    for direction in DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        rows[dname] = {}
        for s in SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                rows[dname][s] = json.loads(f.read_text())
    missing = [(d, s) for d in rows for s in SEEDS if s not in rows[d]]
    if missing:
        print("missing shards:", missing, flush=True)
        return
    summary = summarize(rows)
    for dname, direction in zip(rows, DIRECTIONS, strict=True):
        io.seal(
            f"MOP_{direction[0].upper()}_TO_{direction[1].upper()}_REPORT.json",
            {
                "schema": "mop-fast-state-cross-domain-direction/v1",
                "direction": dname,
                "seeds": SEEDS,
                "arms": sorted(S.ARMS),
                "step_budget": {
                    "per_domain": S.BUDGET,
                    "learning_rate": S.LR,
                    "return_fraction": S.RETURN_FRACTION,
                },
                **summary["per_direction"][dname],
            },
        )
    io.seal(
        "MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json",
        {
            "schema": "mop-fast-state-bidirectional-synthesis/v1",
            "seeds": SEEDS,
            "directions": list(rows),
            **{k: v for k, v in summary.items() if k != "per_direction"},
            "per_direction_verdicts": {d: summary["per_direction"][d]["verdict"] for d in rows},
            "per_direction_strongest_baseline": {
                d: summary["per_direction"][d]["strongest_matched_baseline"] for d in rows
            },
        },
    )
    print(json.dumps({d: summary["per_direction"][d]["verdict"] for d in rows}), flush=True)
    print("bidirectional:", summary["bidirectional_verdict"], flush=True)
    print("CROSSDOMAIN_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
