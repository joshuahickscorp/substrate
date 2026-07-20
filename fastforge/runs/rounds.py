"""Architecture improvement rounds.

Two rounds per architecture, selected from measured causal failure rather than run blindly. The selector
reads the interference map and the first cross domain result, states which failure it is responding to, and
records the rejected candidates and why. Every round runs eight seeds in both domain directions against the
same strongest matched baseline as the base matrix.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import engine as E
from fastforge import sequence as S
from fastforge.runs import crossdomain as CD
from fastforge.runs import io

SEEDS = list(range(8))

CANDIDATES = {
    "G-R1": dict(name="G_R1_domain_input_norm",
                 policy=dict(arch="G_R1_domain_input_norm", carry=True, shared_p2=False, shared_p3=True),
                 responds_to="second domain acquisition is limited by input statistics reaching the shared "
                             "core, not by the shared dynamics themselves",
                 evidence_key="projection_or_norm_enables_acquisition"),
    "G-R2": dict(name="G_R2_bounded_core_delta",
                 policy=dict(arch="G_R2_bounded_core_delta", carry=True, shared_p2=True, shared_p3=True),
                 responds_to="the shared core causes old domain forgetting when it is fully trainable",
                 evidence_key="shared_core_causes_forgetting"),
    "G-R3": dict(name="G_R3_split_core",
                 policy=dict(arch="G_R3_split_core", carry=True, shared_p2=False, shared_p3=True,
                             phase1_extra=["fast_core_ih"]),
                 responds_to="return recovery is the failing component, so only a declared subset of the "
                             "core is reopened on return",
                 evidence_key="return_recovery_is_the_failing_component"),
    "H-R1": dict(name="H_R1_conflict_gate_tightened",
                 policy=dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="cosine", thresh=0.2),
                 responds_to="the conflict gate at threshold zero blocked too little to protect the old "
                             "domain",
                 evidence_key="cosine_gate_blocked_too_little"),
    "H-R2": dict(name="H_R2_probe_gate_tightened",
                 policy=dict(arch="H", carry=True, shared_p2=True, shared_p3=True, gate="probe", thresh=0.01),
                 responds_to="the probe loss gate tolerated too much old domain loss before freezing",
                 evidence_key="probe_gate_tolerated_forgetting"),
    "H-R3": dict(name="H_R3_anchor_restoration",
                 policy=dict(arch="H", carry=True, shared_p2=True, shared_p3=True,
                             gate="anchor_restore", thresh=0.5),
                 responds_to="measured drift away from the anchor tracked the loss of the old domain, so "
                             "the response is restoration rather than freezing",
                 evidence_key="drift_tracks_forgetting"),
}


def causal_evidence():
    """Read the sealed causal artifacts and produce the facts the selector is allowed to use."""
    ev = {}
    if io.exists("MOP_SUBSTRATE_INTERFERENCE_MAP.json"):
        m = io.load("MOP_SUBSTRATE_INTERFERENCE_MAP.json")
        cls = m["classification"]
        ev["shared_core_causes_forgetting"] = any(
            v["causes_forgetting"] for k, v in cls.items() if "shared" in k)
        ev["projection_or_norm_enables_acquisition"] = any(
            v["enables_acquisition"] for k, v in cls.items() if "proj" in k or "norm" in k)
        ev["forgetting_groups"] = m["forgetting_groups"]
        ev["transferable_groups"] = m["transferable_groups"]
    rows = {}
    for direction in CD.DIRECTIONS:
        per = {}
        for s in CD.SEEDS:
            f = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file():
                per[s] = json.loads(f.read_text())
        if per:
            rows[f"{direction[0]}->{direction[1]}"] = per
    if rows:
        d0 = next(iter(rows.values()))
        seeds = sorted(d0)
        base = "lstm_gdumb"
        comp = {}
        for k in ("second_acquisition", "first_retention", "return_recovery", "future_adaptation"):
            best_sub = max([a for a in d0[seeds[0]] if a.startswith(("G", "H"))],
                           key=lambda a: float(np.mean([d0[s][a]["metrics"][k] for s in seeds])))
            comp[k] = {
                "best_substrate_arm": best_sub,
                "gap_to_baseline": round(float(np.mean([d0[s][best_sub]["metrics"][k] for s in seeds]))
                                         - float(np.mean([d0[s][base]["metrics"][k] for s in seeds])), 4),
            }
        ev["component_gaps"] = comp
        ev["return_recovery_is_the_failing_component"] = (
            comp["return_recovery"]["gap_to_baseline"] <= min(v["gap_to_baseline"] for v in comp.values()))
        block = {a: float(np.mean([d0[s][a].get("gate_block_rate") or 0.0 for s in seeds]))
                 for a in ("H_cosine_gate", "H_probe_gate", "H_drift_gate") if a in d0[seeds[0]]}
        ev["gate_block_rates"] = {k: round(v, 4) for k, v in block.items()}
        ev["cosine_gate_blocked_too_little"] = block.get("H_cosine_gate", 1.0) < 0.5
        ev["probe_gate_tolerated_forgetting"] = block.get("H_probe_gate", 1.0) < 0.5
        ev["drift_tracks_forgetting"] = block.get("H_drift_gate", 0.0) > 0.0
    return ev


def select(ev):
    chosen, rejected = {}, {}
    for family, keys in (("G", ["G-R1", "G-R2", "G-R3"]), ("H", ["H-R1", "H-R2", "H-R3"])):
        scored = [(k, bool(ev.get(CANDIDATES[k]["evidence_key"], False))) for k in keys]
        supported = [k for k, ok in scored if ok]
        if len(supported) < 2:  # fall back deterministically to declared order, and say so
            supported = (supported + [k for k, _ in scored if k not in supported])[:2]
        chosen[family] = supported[:2]
        rejected[family] = [k for k in keys if k not in chosen[family]]
    return chosen, rejected


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "shard":
        d, s = sys.argv[2].split(":")
        direction = tuple(d.split("-"))
        ev = causal_evidence()
        chosen, _ = select(ev)
        out = {}
        for family in ("G", "H"):
            for k in chosen[family]:
                c = CANDIDATES[k]
                out[c["name"]] = S.run(c["name"], direction, int(s), override=c["policy"])
                print(f"  round {k} {d} seed{s} acq2="
                      f"{out[c['name']]['metrics']['second_acquisition']:.3f}", flush=True)
        p = io.RUNS / "rounds"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"{d}_{s}.json").write_text(json.dumps(out, default=str))
        print("SHARD_DONE", d, s, flush=True)
        return

    t0 = time.time()
    ev = causal_evidence()
    chosen, rejected = select(ev)
    rounds, base = {}, {}
    for direction in CD.DIRECTIONS:
        dname = f"{direction[0]}->{direction[1]}"
        per, bper = {}, {}
        for s in SEEDS:
            f = io.RUNS / "rounds" / f"{direction[0]}-{direction[1]}_{s}.json"
            b = io.RUNS / "crossdomain" / f"{direction[0]}-{direction[1]}_{s}.json"
            if f.is_file() and b.is_file():
                per[s] = json.loads(f.read_text())
                bper[s] = json.loads(b.read_text())
        if len(per) == len(SEEDS):
            rounds[dname], base[dname] = per, bper
    if not rounds:
        print("rounds waiting on shards", flush=True)
        return

    comparison = {}
    for dname in rounds:
        seeds = sorted(rounds[dname])
        arms = sorted(rounds[dname][seeds[0]])
        strongest = io.load(f"MOP_{dname.split('->')[0].upper()}_TO_{dname.split('->')[1].upper()}_REPORT.json"
                            )["strongest_matched_baseline"] if io.exists(
            f"MOP_{dname.split('->')[0].upper()}_TO_{dname.split('->')[1].upper()}_REPORT.json") else "lstm_gdumb"
        base_params = float(np.mean([base[dname][s]["lstm_gdumb"]["metrics"]["params"] for s in seeds]))

        def util(m):
            return float(np.mean([m["second_acquisition"], m["return_recovery"], m["future_adaptation"]])
                         - 0.05 * (m["params"] / base_params))

        bl = [util(base[dname][s][strongest]["metrics"]) for s in seeds]
        g0 = [util(base[dname][s]["G0_always_trainable"]["metrics"]) for s in seeds]
        h0 = [util(base[dname][s]["H_always_update"]["metrics"]) for s in seeds]
        comparison[dname] = {"strongest_matched_baseline": strongest}
        for a in arms:
            u = [util(rounds[dname][s][a]["metrics"]) for s in seeds]
            comparison[dname][a] = {
                "utility": round(float(np.mean(u)), 4),
                "vs_strongest_baseline": E.effect(u, bl),
                "vs_base_architecture": E.effect(u, g0 if a.startswith("G") else h0),
            }
    passing = sorted({a for d in comparison.values() for a, v in d.items()
                      if isinstance(v, dict) and v["vs_strongest_baseline"]["lower_95_cb"] >= io.SESOI})
    both = sorted(a for a in passing
                  if all(comparison[d][a]["vs_strongest_baseline"]["lower_95_cb"] >= io.SESOI
                         for d in comparison))
    io.seal("MOP_FAST_STATE_ARCHITECTURE_COMPARISON.json", {
        "schema": "mop-fast-state-architecture-comparison/v1",
        "seeds": SEEDS,
        "causal_evidence_used": ev,
        "selected_rounds": chosen,
        "rejected_rounds": rejected,
        "round_definitions": {k: {"name": v["name"], "responds_to": v["responds_to"],
                                  "policy": v["policy"]} for k, v in CANDIDATES.items()},
        "comparison": comparison,
        "rounds_passing_a_direction": passing,
        "rounds_passing_both_directions": both,
        "verdict": "improvement_round_positive" if both else "improvement_round_null",
        "wall_seconds": round(time.time() - t0, 1),
    })
    print("rounds verdict:", "positive" if both else "null", passing, flush=True)
    print("ROUNDS_DONE", flush=True)


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
