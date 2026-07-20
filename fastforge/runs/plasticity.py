"""Plasticity action headroom gate.

The question is narrow: at a domain switch, is there stable residual value in choosing what updates, beyond
what the strongest simple fixed rule already achieves.

Two comparisons are run, and each stays inside one protocol, because a utility measured under a short
horizon counterfactual is not comparable to one measured under the full budget sequence. Mixing them would
manufacture headroom out of a budget difference.

    policy headroom   over the full budget cross domain policies: every fixed schedule and every simple gate
    action headroom   over the short horizon interference actions: which parameter groups may change

The oracle picks per seed on tuning utility only. Random and shuffled assignment must fail, and the oracle
must beat random, or the selection signal is not a signal. Headroom requires both comparisons to clear SESOI
at the lower confidence bound in both domain directions. Otherwise the verdict is
simple_partition_policy_sufficient and no learned controller opens.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

import numpy as np

from fastforge import engine as E
from fastforge.runs import crossdomain as CD
from fastforge.runs import interference as IF
from fastforge.runs import io

SEEDS = IF.SEEDS
# every full budget policy that decides what may change. Oracle and shuffled arms are excluded: they are
# derived from these, so including them would let the oracle select itself.
POLICIES = [
    "G0_always_trainable",
    "G1_frozen_after_first",
    "G2_reopened_at_return",
    "G3_reopened_on_drop",
    "G4_adapters_only",
    "G5_projection_and_head",
    "H_always_update",
    "H_never_update",
    "H_cosine_gate",
    "H_probe_gate",
    "H_drift_gate",
    "H_perf_gate",
    "H_task_label_freeze",
]
CONTROLS = {"random": "H_random_gate", "shuffled": "H_shuffled_gate", "wrong_domain": "H_wrong_domain_gate"}


def _oracle_block(util: dict[str, list[float]], candidates, seeds, rng_a=11, rng_b=12):
    """Per seed oracle over candidates, with its random and shuffled counterparts, all on the same values."""
    n = len(seeds)
    oracle = [max(util[c][i] for c in candidates) for i in range(n)]
    choice = [max(candidates, key=lambda c: util[c][i]) for i in range(n)]
    rng = np.random.default_rng(rng_a)
    rand = [util[candidates[int(rng.integers(len(candidates)))]][i] for i in range(n)]
    perm = np.random.default_rng(rng_b).permutation(n)
    shuf = [util[choice[perm[i]]][i] for i in range(n)]
    best = max(candidates, key=lambda c: float(np.mean(util[c])))
    return {
        "candidates": list(candidates),
        "best_fixed_policy": best,
        "best_fixed_utility": round(float(np.mean(util[best])), 4),
        "oracle_utility": round(float(np.mean(oracle)), 4),
        "oracle_choice_per_seed": choice,
        "oracle_advantage_over_best_fixed": E.effect(oracle, util[best]),
        "random_control": E.effect(rand, util[best]),
        "shuffled_control": E.effect(shuf, util[best]),
        "positive_control_oracle_beats_random": E.effect(oracle, rand),
    }


def main():
    t0 = time.time()
    rows, actions = {}, {}
    for direction in CD.DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        per = {}
        for s in SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                per[s] = json.loads(f.read_text())
        if len(per) == len(SEEDS):
            rows[dname] = per
        aper = {}
        for s in SEEDS:
            f = io.RUNS / "interference" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                aper[s] = json.loads(f.read_text())
        if len(aper) == len(SEEDS):
            actions[dname] = aper
    if not rows or not actions:
        print(
            f"plasticity gate waiting: cross {len(rows)} directions, interference {len(actions)}", flush=True
        )
        return

    per_dir = {}
    for dname in rows:
        seeds = sorted(rows[dname])
        avail = [p for p in POLICIES if p in rows[dname][seeds[0]]]
        putil = {p: [rows[dname][s][p]["metrics"]["tune_utility"] for s in seeds] for p in avail}
        for name, arm in CONTROLS.items():
            if arm in rows[dname][seeds[0]]:
                putil[f"control_{name}"] = [rows[dname][s][arm]["metrics"]["tune_utility"] for s in seeds]
        policy = _oracle_block(putil, avail, seeds)
        policy["explicit_control_policies"] = {
            name: E.effect(putil[f"control_{name}"], putil[policy["best_fixed_policy"]])
            for name in CONTROLS
            if f"control_{name}" in putil
        }

        aseeds = sorted(actions[dname])
        anames = sorted(actions[dname][aseeds[0]]["actions"])
        autil = {a: [IF._u(actions[dname][s]["actions"][a]) for s in aseeds] for a in anames}
        action = _oracle_block(autil, anames, aseeds, rng_a=21, rng_b=22)
        action["action_utility_mean"] = {a: round(float(np.mean(autil[a])), 4) for a in anames}
        action["cost_accounting"] = {
            a: {
                "trainable_params": int(
                    np.mean([actions[dname][s]["actions"][a]["trainable_params"] for s in aseeds])
                ),
                "compute_seconds": round(
                    float(np.mean([actions[dname][s]["actions"][a]["compute_seconds"] for s in aseeds])), 2
                ),
            }
            for a in anames
        }

        def clean(block):
            return (
                block["oracle_advantage_over_best_fixed"]["lower_95_cb"] >= io.SESOI
                and block["random_control"]["lower_95_cb"] < io.SESOI
                and block["shuffled_control"]["lower_95_cb"] < io.SESOI
                and block["positive_control_oracle_beats_random"]["lower_95_cb"] > 0
            )

        per_dir[dname] = {
            "policy_headroom": policy,
            "policy_utility_mean": {p: round(float(np.mean(putil[p])), 4) for p in putil},
            "action_headroom": action,
            "policy_headroom_present": clean(policy),
            "action_headroom_present": clean(action),
        }

    headroom = all(v["policy_headroom_present"] and v["action_headroom_present"] for v in per_dir.values())
    verdict = "plasticity_action_headroom_present" if headroom else "simple_partition_policy_sufficient"
    io.seal(
        "MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json",
        {
            "schema": "mop-substrate-plasticity-policy/v2",
            "seeds": SEEDS,
            "preregistered_requirements": {
                "seeds": 8,
                "units": "natural (HAR subject, Speech speaker)",
                "decision": "group lower 95 percent confidence bound above SESOI, in both domain directions",
                "oracle_must_beat": "the strongest simple fixed policy inside the same protocol",
                "controls_that_must_fail": ["random", "shuffled"],
                "control_that_must_pass": "oracle beats random",
                "cost_accounting": "trainable parameters and compute seconds recorded per action",
                "protocol_separation": "policy headroom and action headroom are never pooled, because their "
                "utilities come from different budgets",
            },
            "per_direction": per_dir,
            "verdict": verdict,
            "learned_gate_opened": headroom,
            "note": "no learned controller opens without stable residual headroom in both comparisons and "
            "both "
            "domain directions",
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    io.seal(
        "MOP_SUBSTRATE_PLASTICITY_ACTION_HEADROOM.json",
        {
            "schema": "mop-substrate-plasticity-action-headroom/v2",
            "actions": sorted(IF.ACTIONS),
            "action_definition": {k: list(v) for k, v in IF.ACTIONS.items()},
            "per_direction": {d: v["action_headroom"] for d, v in per_dir.items()},
            "verdict": verdict,
        },
    )
    print("plasticity verdict:", verdict, flush=True)
    for d, v in per_dir.items():
        print(
            f"  {d}: policy oracle {v['policy_headroom']['oracle_utility']} vs "
            f"{v['policy_headroom']['best_fixed_policy']} {v['policy_headroom']['best_fixed_utility']} "
            f"lcb {v['policy_headroom']['oracle_advantage_over_best_fixed']['lower_95_cb']}",
            flush=True,
        )
    print("PLASTICITY_DONE", flush=True)


if __name__ == "__main__":
    main()
