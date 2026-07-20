"""Plasticity action headroom gate.

The question is narrow: at a domain switch, is there stable residual value in choosing what updates, beyond
what the strongest simple fixed rule already achieves? The oracle chooses per seed on tuning utility. Random
and shuffled assignment must fail. If the oracle advantage over the strongest simple rule does not clear
SESOI at the lower confidence bound, the verdict is simple_partition_policy_sufficient and no learned
controller opens. That is a decision rule fixed before the numbers exist, not a reading of them.

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
RULE_ARMS = {
    "always_all": "G0_always_trainable",
    "domain_boundary_schedule": "G1_frozen_after_first",
    "reopen_at_return": "G2_reopened_at_return",
    "performance_threshold": "G3_reopened_on_drop",
    "adapter_only": "G4_adapters_only",
    "projection_and_head_only": "G5_projection_and_head",
    "gradient_conflict_threshold": "H_cosine_gate",
    "probe_loss_threshold": "H_probe_gate",
    "drift_threshold": "H_drift_gate",
    "never_update_shared": "H_never_update",
    "random_schedule": "H_random_gate",
    "shuffled_schedule": "H_shuffled_gate",
    "wrong_domain_schedule": "H_wrong_domain_gate",
}


def load_actions():
    rows = []
    for direction in IF.DIRECTIONS:
        for s in SEEDS:
            f = io.RUNS / "interference" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                rows.append(json.loads(f.read_text()))
    return rows


def load_rules():
    out = {}
    for direction in CD.DIRECTIONS:
        d = f"{direction[0]}->{direction[1]}"
        out[d] = {}
        for s in SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                out[d][s] = json.loads(f.read_text())
    return out


def main():
    t0 = time.time()
    rows, rules = load_actions(), load_rules()
    if not rows or not all(len(rules[d]) == len(SEEDS) for d in rules):
        print("plasticity gate waiting on shards", flush=True)
        return

    # per direction, per seed action utility on the tuning split
    per_dir = {}
    for direction in IF.DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        rs = sorted([r for r in rows if r["direction"] == dname], key=lambda r: r["seed"])
        actions = sorted(rs[0]["actions"])
        util = {a: [IF._u(r["actions"][a]) for r in rs] for a in actions}
        rule_util = {name: [rules[dname][s][arm]["metrics"]["tune_utility"] for s in SEEDS]
                     for name, arm in RULE_ARMS.items() if arm in rules[dname][SEEDS[0]]}

        # oracle: best action per seed, chosen on tuning utility only
        oracle = [max(util[a][i] for a in actions) for i in range(len(rs))]
        oracle_choice = [max(actions, key=lambda a: util[a][i]) for i in range(len(rs))]
        # random control: one action per seed, drawn without looking at anything
        rng = np.random.default_rng(11)
        rand = [util[actions[int(rng.integers(len(actions)))]][i] for i in range(len(rs))]
        # shuffled control: each seed is given the action that was best for a different seed
        perm = np.random.default_rng(12).permutation(len(rs))
        shuf = [util[oracle_choice[perm[i]]][i] for i in range(len(rs))]

        simple_pool = {**{a: util[a] for a in actions}, **rule_util}
        best_simple = max(simple_pool, key=lambda k: float(np.mean(simple_pool[k])))
        head = E.effect(oracle, simple_pool[best_simple])
        per_dir[dname] = {
            "action_utility_mean": {a: round(float(np.mean(util[a])), 4) for a in actions},
            "rule_utility_mean": {k: round(float(np.mean(v)), 4) for k, v in rule_util.items()},
            "strongest_simple_policy": best_simple,
            "strongest_simple_utility": round(float(np.mean(simple_pool[best_simple])), 4),
            "oracle_utility": round(float(np.mean(oracle)), 4),
            "oracle_choice_per_seed": oracle_choice,
            "oracle_advantage_over_strongest_simple": head,
            "random_control": E.effect(rand, simple_pool[best_simple]),
            "shuffled_control": E.effect(shuf, simple_pool[best_simple]),
            "positive_control_oracle_beats_random": E.effect(oracle, rand),
            "headroom_present": head["lower_95_cb"] >= io.SESOI,
        }

    controls_ok = all(
        v["random_control"]["lower_95_cb"] < io.SESOI and v["shuffled_control"]["lower_95_cb"] < io.SESOI
        and v["positive_control_oracle_beats_random"]["lower_95_cb"] > 0
        for v in per_dir.values()
    )
    headroom = all(v["headroom_present"] for v in per_dir.values()) and controls_ok
    verdict = "plasticity_action_headroom_present" if headroom else "simple_partition_policy_sufficient"

    io.seal("MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json", {
        "schema": "mop-substrate-plasticity-policy/v1",
        "seeds": SEEDS,
        "preregistered_requirements": {
            "seeds": 8, "units": "natural (HAR subject, Speech speaker)",
            "decision": "group lower 95 percent confidence bound above SESOI",
            "oracle_must_beat": "strongest simple rule",
            "controls_that_must_fail": ["random", "shuffled"],
            "control_that_must_pass": "oracle beats random",
            "cost_accounting": "trainable parameters and compute seconds recorded per action",
        },
        "per_direction": per_dir,
        "controls_behaved": controls_ok,
        "verdict": verdict,
        "learned_gate_opened": headroom,
        "note": "no learned controller opens without stable residual headroom in both domain directions",
        "wall_seconds": round(time.time() - t0, 1),
    })
    print("plasticity verdict:", verdict, flush=True)
    for d, v in per_dir.items():
        print(f"  {d}: oracle {v['oracle_utility']} vs {v['strongest_simple_policy']} "
              f"{v['strongest_simple_utility']} lcb {v['oracle_advantage_over_strongest_simple']['lower_95_cb']}",
              flush=True)
    print("PLASTICITY_DONE", flush=True)


if __name__ == "__main__":
    main()
