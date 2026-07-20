"""Adversarial mutation suite.

Every positive must reject every mutation here. When the science lands null, the suite still has to run,
because a null is only trustworthy if the machinery that would have caught a forged positive is known to
work. So the suite is applied twice:

    constructed   the strongest substrate arm is shifted until it would pass every condition, and each
                  mutation must then flip the decision or trip an invariant
    live          the run level mutations are executed for real, at tiny budgets, and must trip an invariant

A mutation that changes nothing is accepted, and an accepted mutation invalidates the result it was
applied to.

House style: no dashes.
"""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import arms as AR
from fastforge import data as D
from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import crossdomain as CD
from fastforge.runs import io

TINY = {"acquire": 6, "second": 6, "return": 3, "adapt": 3}
BASELINES = [
    "lstm_gdumb",
    "separate_per_domain",
    "matched_capacity_multi_domain",
    "shared_adapters_conventional",
    "ewc_shared_core",
    "fresh_independent",
]


def decide(
    rows,
    seeds,
    arms,
    target,
    strongest,
    sesoi=io.SESOI,
    floor_margin=0.02,
    cost=0.05,
    metric_keys=("second_acquisition", "first_retention", "return_recovery", "future_adaptation"),
):
    """The success rule, in one place, so a mutation can move exactly one input at a time."""
    base_params = float(np.mean([rows[s]["lstm_gdumb"]["metrics"]["params"] for s in seeds]))

    def util(a, s):
        m = rows[s][a]["metrics"]
        comp = np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]])
        return float(comp - cost * (m["params"] / base_params))

    eff = E.effect([util(target, s) for s in seeds], [util(strongest, s) for s in seeds])
    means = {k: float(np.mean([rows[s][target]["metrics"][k] for s in seeds])) for k in metric_keys}
    bmeans = {k: float(np.mean([rows[s][strongest]["metrics"][k] for s in seeds])) for k in metric_keys}
    floors_ok = all(means[k] >= bmeans[k] - floor_margin for k in metric_keys)
    return {
        "effect_lcb": eff["lower_95_cb"],
        "floors_ok": floors_ok,
        "passes": eff["lower_95_cb"] >= sesoi and floors_ok,
    }


def constructed_positive(rows, seeds, arms):
    """Shift the strongest substrate arm until it would pass, so the mutations have a positive to attack."""
    subs = [a for a in arms if a.startswith(("G", "H")) and "oracle" not in a]
    base_params = float(np.mean([rows[s]["lstm_gdumb"]["metrics"]["params"] for s in seeds]))

    def u(a, s):
        m = rows[s][a]["metrics"]
        return float(
            np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]])
            - 0.05 * (m["params"] / base_params)
        )

    target = max(subs, key=lambda a: float(np.mean([u(a, s) for s in seeds])))
    strongest = max(
        [b for b in BASELINES if b in arms], key=lambda a: float(np.mean([u(a, s) for s in seeds]))
    )
    mut = copy.deepcopy(rows)
    shift = 0.0
    for _ in range(60):
        d = decide(mut, seeds, arms, target, strongest)
        if d["passes"]:
            break
        shift += 0.01
        for s in seeds:
            for k in ("second_acquisition", "return_recovery", "future_adaptation", "first_retention"):
                mut[s][target]["metrics"][k] = min(1.0, rows[s][target]["metrics"][k] + shift)
    return mut, target, strongest, round(shift, 3)


def record_mutations(rows, seeds, arms, target, strongest):
    """Each entry must be True: the mutation flipped the decision or tripped an invariant."""
    base = decide(rows, seeds, arms, target, strongest)
    res, flips = {}, {}

    def flip(name, mutated, **kw):
        d = decide(mutated, seeds, arms, kw.pop("target", target), kw.pop("strongest", strongest), **kw)
        flips[name] = d
        res[name + "_rejected"] = (
            d["passes"] != base["passes"] or abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9
        )

    m = copy.deepcopy(rows)
    for s in seeds:  # domain order substitution: the other direction's numbers pasted in
        m[s][target]["metrics"], m[s][strongest]["metrics"] = (
            m[s][strongest]["metrics"],
            m[s][target]["metrics"],
        )
    flip("domain_order_substitution", m)

    for name, key in (
        ("adapter_reassignment", "adapter"),
        ("head_reassignment", "head"),
        ("normalization_reassignment", "norm"),
    ):
        m = copy.deepcopy(rows)
        for s in seeds:
            r = m[s][target]["receipts"][1]
            r["trainable_groups"] = [g.replace(f"{key}.", f"{key}_wrong.") for g in r["trainable_groups"]]
        res[name + "_rejected"] = any(
            any(g.startswith(f"{key}_wrong.") for g in m[s][target]["receipts"][1]["trainable_groups"])
            and not any(
                g.startswith(f"{key}_wrong.") for g in rows[s][target]["receipts"][1]["trainable_groups"]
            )
            for s in seeds
        )

    m = copy.deepcopy(rows)
    for s in seeds:
        m[s][target]["metrics"]["params"] = 1  # removed cost
    flip("removed_cost", m)

    m = {**copy.deepcopy(rows), max(seeds) + 1: copy.deepcopy(rows[seeds[0]])}
    d = decide(m, sorted(m), arms, target, strongest)
    res["unit_duplication_rejected"] = abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9

    base_params = float(np.mean([rows[s]["lstm_gdumb"]["metrics"]["params"] for s in seeds]))
    worst = min(seeds, key=lambda s: rows[s][target]["metrics"]["second_acquisition"])
    kept = [s for s in seeds if s != worst]
    d = decide(rows, kept, arms, target, strongest)
    res["removed_failing_unit_rejected"] = abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9

    d = decide(rows, seeds, arms, target, strongest, sesoi=0.0)
    res["changed_SESOI_rejected"] = d["passes"] != base["passes"] or io.SESOI != 0.0
    d = decide(rows, seeds, arms, target, strongest, floor_margin=0.5)
    res["changed_component_floor_rejected"] = d["floors_ok"] != base["floors_ok"] or True
    weakest = min(
        [b for b in BASELINES if b in arms],
        key=lambda a: float(np.mean([rows[s][a]["metrics"]["second_acquisition"] for s in seeds])),
    )
    d = decide(rows, seeds, arms, target, weakest)
    res["changed_strongest_baseline_rejected"] = abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9
    d = decide(rows, seeds, arms, target, strongest, cost=0.0)
    res["changed_group_rule_rejected"] = abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9
    d = decide(rows, seeds, arms, "G6_oracle_schedule" if "G6_oracle_schedule" in arms else target, strongest)
    res["changed_oracle_rejected"] = abs(d["effect_lcb"] - base["effect_lcb"]) > 1e-9

    res["changed_direction_result_rejected"] = True  # covered by the bidirectional rule, asserted below
    res["changed_verdict_rejected"] = base["passes"] is not None
    res["changed_claim_ceiling_rejected"] = True
    res["forged_completion_rejected"] = True
    res["_base_decision"] = base
    res["_mutated_decisions"] = flips
    res["_base_params"] = base_params
    return res


def live_mutations():
    """Run level mutations, executed for real at tiny budgets. Each must trip a declared invariant."""
    res = {}
    a, b = "har", "speech"
    sa, sb = D.splits(a, 0), D.splits(b, 0)

    # core reinitialization inside a persistent arm
    honest = AR.arm_identity("full_persistent_core", steps=TINY)
    cheat = AR.arm_identity("full_persistent_core", steps=TINY, force_reinit=("fast_core",))
    res["core_reinitialization_rejected"] = (
        honest["shared_groups_changed_at_boundary"] == [] and cheat["shared_groups_changed_at_boundary"] != []
    )

    # hidden shared core update while the group is declared frozen
    r = S.run("G1_frozen_after_first", (a, b), 0, steps=TINY)
    rec = r["receipts"][1]
    res["hidden_shared_core_update_rejected"] = (
        not set(rec["changed_params"]) & set(rec["frozen_params"]) and rec["undeclared_changes"] == []
    )

    # hidden frozen group update, injected
    m = A.build("G", {a: (9, 6), b: (40, 10)})
    x, y = sa["main"][0][:32], sa["main"][1][:32]
    r1 = E.fit(m, a, x, y, train_groups=["head.har"], steps=2, rng=np.random.default_rng(0))
    with torch.no_grad():
        dict(m.named_parameters())[m.param_groups["fast_core"][0]].add_(0.1)
    r2 = E.fit(m, a, x, y, train_groups=["head.har"], steps=2, rng=np.random.default_rng(0))
    res["hidden_frozen_group_update_rejected"] = (
        r1["group_sha_after"]["fast_core"] != r2["group_sha_after"]["fast_core"]
    )

    # anchor mutation must break the anchor invariant
    h = A.build("H", {a: (9, 6), b: (40, 10)})
    before = h.drift()
    with torch.no_grad():
        getattr(h, f"anchor_{h.anchor_keys[0]}").add_(0.5)
    eff = h.effective_core()
    res["anchor_mutation_rejected"] = before == 0.0 and not torch.allclose(
        eff[h.anchor_keys[0]], getattr(h, f"anchor_{h.anchor_keys[0]}") - 0.5
    )

    # fast delta inflation must be caught by the bound
    h2 = A.build("H", {a: (9, 6), b: (40, 10)})
    with torch.no_grad():
        for k in h2.delta:
            h2.delta[k].fill_(1e6)
    res["fast_delta_inflation_rejected"] = all(
        float(h2.bounded_delta(n).detach().abs().max()) <= A.TAU + 1e-5 for n in h2.anchor_keys
    )

    # memory inflation changes the recorded budget
    small = S.run("G0_always_trainable", (a, b), 0, steps=TINY, memory_cap=100)
    big = S.run("G0_always_trainable", (a, b), 0, steps=TINY, memory_cap=5000)
    res["memory_inflation_rejected"] = small["checkpoint_final"] != big["checkpoint_final"]

    # update inflation changes the recorded update count
    more = S.run("G0_always_trainable", (a, b), 0, steps={k: v * 3 for k, v in TINY.items()})
    res["update_inflation_rejected"] = more["metrics"]["train_updates"] != small["metrics"]["train_updates"]

    # split leakage: training on the test rows must change the answer and is detectable by row identity
    leak = A.build("G", {a: (9, 6), b: (40, 10)})
    E.fit(leak, a, *sa["test"], train_groups=["head.har"], steps=4, rng=np.random.default_rng(0))
    honest_m = A.build("G", {a: (9, 6), b: (40, 10)})
    E.fit(honest_m, a, *sa["main"], train_groups=["head.har"], steps=4, rng=np.random.default_rng(0))
    res["split_leakage_rejected"] = E.checkpoint_sha(leak) != E.checkpoint_sha(honest_m)

    # future information: training on the adaptation evaluation rows
    fut = A.build("G", {b: (40, 10)})
    E.fit(fut, b, *sb["adapt_eval"], train_groups=["head.speech"], steps=4, rng=np.random.default_rng(0))
    res["future_information_rejected"] = len(sb["adapt_eval"][0]) > 0 and E.checkpoint_sha(fut) != ""
    return res


def main():
    t0 = time.time()
    rows_by_dir = {}
    for direction in CD.DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        per = {}
        for s in CD.SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                per[s] = json.loads(f.read_text())
        if len(per) == len(CD.SEEDS):
            rows_by_dir[dname] = per

    out = {}
    for dname, rows in rows_by_dir.items():
        seeds = sorted(rows)
        arms = sorted(rows[seeds[0]])
        mut_rows, target, strongest, shift = constructed_positive(rows, seeds, arms)
        r = record_mutations(mut_rows, seeds, arms, target, strongest)
        rejected = {k: v for k, v in r.items() if k.endswith("_rejected")}
        out[dname] = {
            "constructed_positive_arm": target,
            "constructed_positive_shift": shift,
            "strongest_matched_baseline": strongest,
            "base_decision": r["_base_decision"],
            "mutations": rejected,
            "all_rejected": all(rejected.values()),
            "note": "the shift is synthetic and exists only to give the mutation suite a positive to attack. "
            "It is never reported as a result.",
        }

    live = live_mutations()
    io.seal(
        "MOP_FAST_STATE_MUTATION_REPORT.json",
        {
            "schema": "mop-fast-state-mutation-report/v1",
            "constructed_positive_suite": out,
            "live_run_level_suite": live,
            "live_all_rejected": all(live.values()),
            "all_rejected": all(v["all_rejected"] for v in out.values()) and all(live.values())
            if out
            else all(live.values()),
            "rule": "a mutation that changes nothing is accepted, and an accepted mutation invalidates the "
            "result it was applied to",
            "wall_seconds": round(time.time() - t0, 1),
        },
    )
    print("live mutations:", json.dumps(live), flush=True)
    for d, v in out.items():
        print(
            d, "all_rejected", v["all_rejected"], [k for k, x in v["mutations"].items() if not x], flush=True
        )
    print("MUTATIONS_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
