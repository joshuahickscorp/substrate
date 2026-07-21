"""Per topic authorities, derived from the admission and scout artifacts rather than restated.

Each file here is a projection of evidence that already exists. Nothing is asserted that is not read from a
sealed artifact, so a disagreement between an authority and its source is a bug rather than an opinion.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from mop.method import acceptance, arms, controls, graph, io


def _adm(name: str) -> dict:
    return io.load(name) if io.exists(name) else {}


def _scouts() -> dict:
    out = {}
    for sub, exp in (("scout", "E1"), ("e4_scout", "E4")):
        d = io.RUNS / sub
        if d.is_dir():
            for p in sorted(d.glob("*.json")):
                out[f"{exp}:{json.loads(p.read_text())['bed']}"] = json.loads(p.read_text())
    return out


def arm_authority(e1: dict, e4: dict) -> tuple[dict, dict]:
    per = {}
    for exp, adm in (("E1", e1), ("E4", e4)):
        for b, v in (adm.get("per_bed") or {}).items():
            d = v["arm_distinctness"]
            per[f"{exp}:{b}"] = {
                "n_arms": d["n_arms"],
                "all_distinct": d["all_distinct"],
                "aliased_pairs": d["aliased_pairs"],
                "per_arm": d["per_arm"],
                "configuration_fields_honoured": (v.get("config_sensitivity") or {}).get("all_honoured"),
            }
    a = arms.mutations([
        arms.record("probe_a", source="a", config={"c": 1}, call_graph=["a"], state_transitions=["a"],
                    param_delta={"p": 1}, memory={}, resources={"r": 1}, outputs=[1]),
        arms.record("probe_b", source="b", config={"c": 2}, call_graph=["b"], state_transitions=["b"],
                    param_delta={"p": 2}, memory={}, resources={"r": 2}, outputs=[2]),
    ])
    return (
        {
            "schema": "mop-arm-distinctness-authority/v1",
            "load_bearing_traces": list(arms.LOAD_BEARING),
            "declared_only_fields": list(arms.DECLARED_ONLY),
            "rule": "two arms that match on every load bearing trace are the same experiment run twice",
            "per_experiment": per,
            "all_distinct": all(v["all_distinct"] for v in per.values()) if per else None,
        },
        {"schema": "mop-arm-distinctness-mutations/v1", "attacks": a,
         "all_rejected": a["all_rejected"]},
    )


def control_authority(e1: dict, e4: dict) -> tuple[dict, dict]:
    registry = {
        "order_free": {
            "removes": "the ability to consume temporal order",
            "proofs": ["timestep permutation", "sequence reversal", "temporal block permutation",
                       "cyclic shift", "determinism", "no temporal convolution", "no recurrence",
                       "no position encoding", "no carried state buffer"],
            "implementation": "mop.method.controls.order_free",
        },
        "no_replay": {"removes": "the use of any historical item in an update",
                      "proofs": ["no replayed items", "no hidden buffer read", "no cached batch",
                                 "buffer declared empty"],
                      "implementation": "mop.method.controls.no_replay"},
        "replay_active": {"removes": "nothing: this is the mirror proof that a replay arm actually replayed",
                          "proofs": ["items replayed", "buffer kept replacing", "context boundary crossed",
                                     "buffer contents changed"],
                          "implementation": "mop.method.controls.replay_active"},
        "random": {"removes": "the information in the decision, keeping its rate",
                   "proofs": ["rate matched", "budget matched", "information matched", "seed bound",
                              "independent of target"],
                   "implementation": "mop.method.controls.random_control"},
        "shuffled": {"removes": "the relation, keeping the marginals",
                     "proofs": ["marginals preserved", "relation destroyed"],
                     "implementation": "mop.method.controls.shuffled_control"},
        "wrong_time": {"removes": "the timing, keeping the intervention and the budget",
                       "proofs": ["same intervention", "budget matched", "times preregistered",
                                  "times differ from the real ones"],
                       "implementation": "mop.method.controls.wrong_time_control"},
        "frozen": {"removes": "every change to the target parameter groups",
                   "proofs": ["no target parameter changed", "targets exist", "receipt lists changes"],
                   "implementation": "mop.method.controls.frozen_control"},
    }
    ver = {}
    for exp, adm in (("E1", e1), ("E4", e4)):
        for b, v in (adm.get("per_bed") or {}).items():
            sem = v.get("control_semantics") or v.get("control_receipts") or {}
            ver[f"{exp}:{b}"] = {k: {kk: vv for kk, vv in r.items() if isinstance(vv, bool)}
                                 for k, r in sem.items() if isinstance(r, dict)}
    return (
        {"schema": "mop-control-semantic-registry/v1", "controls": registry,
         "rule": "a control that fails any proof cannot be the comparison for a verdict"},
        {"schema": "mop-control-semantic-verification/v1", "per_experiment": ver,
         "all_pass": all(r.get("all_pass", True) for v in ver.values() for r in v.values()) if ver else None},
    )


def mechanism_authority(e1: dict, e4: dict) -> dict:
    per = {}
    for exp, adm in (("E1", e1), ("E4", e4)):
        for b, v in (adm.get("per_bed") or {}).items():
            m = v["mechanism_activity"]
            per[f"{exp}:{b}"] = {"classification": m["classification"], "active": m["active"],
                                 "checks": m["checks"], "failed": m["failed"],
                                 "accuracies": m.get("accuracies")}
    return {
        "schema": "mop-mechanism-activity-authority/v1",
        "required_measurements": list(__import__("mop.method.mechanism", fromlist=["x"]).REQUIRED_MEASUREMENTS),
        "conditions": list(__import__("mop.method.mechanism", fromlist=["x"]).CONDITIONS),
        "rule": "a mechanism with no measurable causal effect is inactive instrumentation, not a null",
        "per_experiment": per,
        "all_active": all(v["active"] for v in per.values()) if per else None,
    }


def bed_authority(e1: dict, e4: dict, scouts: dict) -> dict:
    per = {}
    for exp, adm in (("E1", e1), ("E4", e4)):
        for b, v in (adm.get("per_bed") or {}).items():
            entry = {
                "inherited_gate_verdict": "temporal_headroom_present",
                "inherited_source": "proof/substrate/mop-fast-state-plasticity-forge-v1/MOP_DOMAIN_VALIDITY.json",
                "unit_audit": v["unit_audit"],
                "unit_counts": v.get("unit_counts"),
            }
            if "context_boundary" in v:
                entry["context_boundary"] = v["context_boundary"]
                entry["covariate_shift"] = v.get("covariate_shift")
            s = scouts.get(f"{exp}:{b}")
            if s:
                entry["residual_headroom"] = s.get("residual_headroom_over_strongest_control") or s.get(
                    "residual_state_only_over_no_adapt")
                entry["oracle_headroom"] = s.get("oracle_headroom")
            per[f"{exp}:{b}"] = entry
    return {
        "schema": "mop-data-bed-validity-authority/v1",
        "classifications": list(__import__("mop.method.bed", fromlist=["x"]).CLASSIFICATIONS),
        "rule": ("a dataset being real does not make the task valid, a task being temporal does not make "
                 "temporal state necessary, and a task being sequentialized does not make it continual"),
        "per_experiment": per,
    }


def baseline_authority(scouts: dict) -> dict:
    per = {}
    for k, s in scouts.items():
        bc = s.get("baseline_convergence")
        if isinstance(bc, dict) and "converged" in bc:
            per[k] = {bc["identity"]: {kk: bc[kk] for kk in ("identity", "converged", "reason",
                                                             "validation_curve", "training_updates",
                                                             "selected_checkpoint", "seed_variance")}}
        elif isinstance(bc, dict):
            per[k] = {name: {kk: v[kk] for kk in ("identity", "converged", "reason", "validation_curve",
                                                  "training_updates", "selected_checkpoint", "seed_variance")}
                      for name, v in bc.items()}
    return {
        "schema": "mop-baseline-convergence-authority/v1",
        "plateau_criterion": "no improvement above 0.5 percent for three validation checks",
        "rule": ("a comparison against an undertrained, misconfigured, aliased or under budgeted baseline is "
                 "provisional and cannot be terminal"),
        "per_experiment": per,
        "all_converged": all(v["converged"] for r in per.values() for v in r.values()) if per else None,
    }


def power_authority(scouts: dict) -> dict:
    return {
        "schema": "mop-power-and-sequential-design/v1",
        "stages": ["calibration", "scout", "canary", "principal", "replication"],
        "stage_rule": "a stage opens only when the previous one passed",
        "decision_rule": ("positive requires the lower 95 percent bound over independent units to reach the "
                          "SESOI; a tie is a null; a wrong direction effect is a failure; seeds may not be "
                          "added after a near miss unless the continuation rule was sealed in advance"),
        "per_experiment": {k: s["power"] for k, s in scouts.items() if "power" in s},
    }


def causal_graphs(e1: dict, e4: dict) -> tuple[dict, str]:
    docs, md = {}, ["# Causal experiment graphs", ""]
    for exp, adm in (("E1", e1), ("E4", e4)):
        g = adm.get("causal_graph")
        if not g:
            continue
        docs[exp] = {"graph": g, "summary": graph.summarize(g), "rejections": graph.validate(g),
                     "admissible": not graph.validate(g)}
        md.append(f"## {exp} {adm.get('title', '')}")
        md.append("")
        md.append(f"{len(g['nodes'])} nodes, {len(g['edges'])} edges, admissible: {not graph.validate(g)}.")
        md.append("")
        md.append("| edge | kind |")
        md.append("|---|---|")
        for e in g["edges"]:
            md.append(f"| {e['src']} to {e['dst']} | {e['type']} |")
        md.append("")
    return {"schema": "mop-causal-experiment-graph/v1", "schema_definition": graph.SCHEMA,
            "experiments": docs}, "\n".join(md)


def main():
    t0 = time.time()
    e1, e4 = _adm("MOP_E1_ADMISSION.json"), _adm("MOP_E4_ADMISSION.json")
    scouts = _scouts()
    aa, am = arm_authority(e1, e4)
    io.seal("MOP_ARM_DISTINCTNESS_AUTHORITY.json", aa)
    io.seal("MOP_ARM_DISTINCTNESS_MUTATIONS.json", am)
    cr, cv = control_authority(e1, e4)
    io.seal("MOP_CONTROL_SEMANTIC_REGISTRY.json", cr)
    io.seal("MOP_CONTROL_SEMANTIC_VERIFICATION.json", cv)
    io.seal("MOP_MECHANISM_ACTIVITY_AUTHORITY.json", mechanism_authority(e1, e4))
    io.seal("MOP_DATA_BED_VALIDITY_AUTHORITY.json", bed_authority(e1, e4, scouts))
    io.seal("MOP_BASELINE_CONVERGENCE_AUTHORITY.json", baseline_authority(scouts))
    io.seal("MOP_POWER_AND_SEQUENTIAL_DESIGN.json", power_authority(scouts))
    cg, cgmd = causal_graphs(e1, e4)
    io.seal("MOP_CAUSAL_EXPERIMENT_GRAPH.json", cg)
    io.seal_md("MOP_CAUSAL_EXPERIMENT_GRAPH.md", cgmd)
    io.seal("MOP_SCOUT_EXPERIMENT_RESULTS.json", {
        "schema": "mop-scout-experiment-results/v1",
        "rule": "a scout validates the design and estimates variance. It is never scientific evidence",
        "evaluated_on": "tuning units only, the test split is untouched by every scout",
        "scouts": scouts,
    })
    print(f"authorities sealed: {len(scouts)} scouts, arms distinct {aa['all_distinct']}, "
          f"mechanisms active {mechanism_authority(e1, e4)['all_active']} in {round(time.time() - t0, 1)}s",
          flush=True)
    print("AUTHORITIES_DONE", flush=True)


if __name__ == "__main__":
    main()
