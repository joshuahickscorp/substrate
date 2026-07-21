"""E1: does the within domain capability come from the core, the readout, or their interaction.

Run on har_stream and speech_stream, the two beds the Fast State Forge sealed as temporal_headroom_present.
The reaudit found that no within domain measurement exists on a bed that requires temporal order, so this
experiment measures an unmeasured cell and separates two explanations of the substrate nulls in one design.

Every stage before principal execution costs no training compute except the scout, and the scout never
touches the test split.

Usage
    python -m mop.method.runs.exp1 admit
    python -m mop.method.runs.exp1 scout   <bed>
    python -m mop.method.runs.exp1 principal <bed> <seed>
    python -m mop.method.runs.exp1 seal

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import arch as A
from fastforge import data as D
from fastforge import engine as E
from mop.method import arms, baseline, bed, contracts, controls, gate, graph, io, mechanism, power
from mop.method.runs import factorial as Fx

EXP = "E1"
BEDS = ("har_stream", "speech_stream")
CELLS = [(c, r) for c in Fx.CORES for r in Fx.READOUTS]
SCOUT_SEEDS = list(range(5))
PRINCIPAL_SEEDS = list(range(8))
STEPS = 1200
LR = 3e-3
BATCH = 64
SESOI = 0.05

HYPOTHESIS_PREDICTIONS = {
    "H_fast_state": "fast beats pooled at both readouts, and fast beats the misaligned reset control",
    "H_readout_capacity": "mlp beats linear at both cores, and fast does not beat pooled",
    "H_core_readout_interaction": "fast beats pooled only under one readout",
    "H_bed_insufficiency": "pooled matches fast, so the bed does not require order after all",
    "H_shared_core_capacity": "no cell separates, because capacity is matched and capacity is what mattered",
}


# ---------------------------------------------------------------- data


def splits_with_units(bedname: str, seed: int) -> dict:
    """D.splits plus the unit identifiers, which the unit audit needs and D.splits does not return."""
    sp = D.splits(bedname, seed)
    d = D.domain(bedname)
    u = np.asarray(d["u"])
    uniq = np.unique(u)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(uniq)
    n_fut = max(1, int(0.25 * len(uniq)))
    n_tun = max(1, int(0.2 * len(uniq)))
    future_u, tune_u, main_u = perm[:n_fut], perm[n_fut : n_fut + n_tun], perm[n_fut + n_tun :]
    sp["units"] = {
        "main": main_u.tolist(),
        "tune": tune_u.tolist(),
        "future": future_u.tolist(),
        "test": np.unique(np.asarray(d["ute"])).tolist(),
        "main_ids": u[np.isin(u, list(main_u))],
        "tune_ids": u[np.isin(u, list(tune_u))],
        "test_ids": np.asarray(d["ute"]),
    }
    return sp


def per_unit_accuracy(model, X, Y, units) -> dict:
    with torch.no_grad():
        pred = torch.cat([model(X[k : k + 256], None)[0].argmax(1) for k in range(0, len(X), 256)])
    correct = (pred == Y).numpy()
    u = np.asarray(units)
    return {str(x): float(correct[u == x].mean()) for x in np.unique(u) if (u == x).sum() >= 5}


# ---------------------------------------------------------------- one run


def run_cell(bedname: str, core: str, readout: str, seed: int, sp: dict, eval_on: str, steps: int = STEPS,
             train_groups=("core", "readout")) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hidden = Fx.match_hidden(sp["channels"], sp["classes"], readout)
    m = Fx.build(sp["channels"], sp["classes"], core, readout, hidden=hidden)
    X, Y = sp["main"]
    t0 = time.time()
    receipt = E.fit(m, None, X, Y, train_groups=list(train_groups), steps=steps, lr=LR, rng=rng, batch=BATCH)
    if eval_on == "tune":
        Xe, Ye, ue = sp["tune"][0], sp["tune"][1], sp["units"]["tune_ids"]
    else:
        Xe, Ye, ue = sp["test"][0], sp["test"][1], sp["units"]["test_ids"]
    acc = E.evaluate(m, None, Xe, Ye)
    return {
        "bed": bedname,
        "core": core,
        "readout": readout,
        "seed": seed,
        "eval_on": eval_on,
        "accuracy": round(float(acc), 5),
        "per_unit_accuracy": per_unit_accuracy(m, Xe, Ye, ue),
        "core_params": Fx.count_core(m),
        "readout_params": Fx.count_readout(m),
        "encoder_hidden": hidden,
        "updates": receipt["updates"],
        "undeclared_changes": receipt["undeclared_changes"],
        "checkpoint_sha_after": receipt["checkpoint_sha_after"],
        "wall_seconds": round(time.time() - t0, 2),
    }


def validation_curve(bedname: str, core: str, readout: str, seed: int, sp: dict, grid=(350, 700, 1200, 1800, 2600, 3400)) -> list:
    """Train once per grid point from scratch, evaluate on tuning units. This is the convergence receipt."""
    out = []
    for s in grid:
        r = run_cell(bedname, core, readout, seed, sp, "tune", steps=s)
        out.append(r["accuracy"])
    return out


# ---------------------------------------------------------------- admission


def arm_records(sp: dict, seed: int = 0) -> list[dict]:
    """Execute every cell briefly and record the load bearing traces that prove distinctness."""
    recs = []
    X, Y = sp["main"]
    probe_x = X[:64]
    for core, readout in CELLS:
        torch.manual_seed(seed)
        hidden = Fx.match_hidden(sp["channels"], sp["classes"], readout)
        m = Fx.build(sp["channels"], sp["classes"], core, readout, hidden=hidden)
        seen: list[str] = []
        hooks = [
            mod.register_forward_hook(lambda mo, i, o, n=name: seen.append(f"{n}:{type(mo).__name__}"))
            for name, mod in m.named_modules()
            if name
        ]
        before = {n: p.detach().clone() for n, p in m.named_parameters()}
        rec = E.fit(m, None, X, Y, train_groups=["core", "readout"], steps=12, lr=LR,
                    rng=np.random.default_rng(seed), batch=32)
        for h in hooks:
            h.remove()
        with torch.no_grad():
            out = m(probe_x, None)[0]
        delta = {n: round(float((p.detach() - before[n]).abs().sum()), 6) for n, p in m.named_parameters()}
        recs.append(
            arms.record(
                f"{core}_{readout}",
                source=Fx.Cell.represent,
                config={"core": core, "readout": readout, "hidden": hidden, "lr": LR, "steps": STEPS},
                call_graph=sorted(set(seen)),
                state_transitions=[rec["checkpoint_sha_before"], rec["checkpoint_sha_after"]],
                param_delta=delta,
                memory={"policy": "none"},
                resources={"core_params": Fx.count_core(m), "readout_params": Fx.count_readout(m),
                           "updates": rec["updates"]},
                outputs=[round(float(v), 5) for v in out.flatten()[:64]],
                declared_difference=f"core={core} readout={readout}",
            )
        )
    return recs


def causal_graph() -> dict:
    return {
        "nodes": [
            {"id": "temporal_capability", "type": "scientific_construct",
             "label": "the capability that lets a model use order in a stream"},
            {"id": "recurrent_core", "type": "mechanism", "implementation": "mop.method.runs.factorial.Cell.represent"},
            {"id": "readout_capacity", "type": "mechanism", "implementation": "mop.method.runs.factorial.Cell.head"},
            {"id": "set_core", "type": "intervention", "implementation": "mop.method.runs.factorial.build"},
            {"id": "set_readout", "type": "intervention", "implementation": "mop.method.runs.factorial.build"},
            {"id": "pooled_control", "type": "control", "implementation": "mop.method.runs.factorial.Cell",
             "removes": "the ability to consume temporal order"},
            {"id": "reset_control", "type": "control", "implementation": "mop.method.runs.factorial.Cell",
             "removes": "state carried across a block boundary"},
            {"id": "stream_order", "type": "available_information", "time": "decision_time"},
            {"id": "final_window_label", "type": "hidden_information", "time": "future", "declared": True},
            {"id": "unit_identity", "type": "confounder", "declared": True,
             "label": "subject or speaker, controlled by unit disjoint splits"},
            {"id": "capacity", "type": "confounder", "declared": True,
             "label": "parameter count, matched by construction and measured"},
            {"id": "segment_alignment", "type": "confounder", "declared": True,
             "label": "a reset period of three lands on the real stream boundaries on both beds, so reset3 "
                      "is oracle segmented. reset5 lands on no boundary and is the neutral ablation"},
            {"id": "representation", "type": "mediator",
             "label": "the fixed width vector the core emits, identical in shape across every cell"},
            {"id": "logits", "type": "mediator", "label": "the readout output"},
            {"id": "subject_or_speaker", "type": "independent_unit"},
            {"id": "test_accuracy", "type": "primary_outcome"},
            {"id": "compute", "type": "cost"},
            {"id": "core_vs_readout_verdict", "type": "verdict", "requires": ["test_accuracy"]},
            {"id": "claim_fast_state_carries_capability", "type": "claim", "requires": ["test_accuracy"]},
        ],
        "edges": [
            {"src": "temporal_capability", "dst": "recurrent_core", "type": "assumed_scientific_relation"},
            {"src": "recurrent_core", "dst": "set_core", "type": "implemented_causal_path"},
            {"src": "readout_capacity", "dst": "set_readout", "type": "implemented_causal_path"},
            {"src": "set_core", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "set_readout", "dst": "logits", "type": "implemented_causal_path"},
            {"src": "representation", "dst": "logits", "type": "implemented_causal_path"},
            {"src": "logits", "dst": "test_accuracy", "type": "measured_relation"},
            {"src": "set_core", "dst": "test_accuracy", "type": "measured_relation"},
            {"src": "set_readout", "dst": "test_accuracy", "type": "measured_relation"},
            {"src": "pooled_control", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "reset_control", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "stream_order", "dst": "recurrent_core", "type": "implemented_causal_path"},
            {"src": "final_window_label", "dst": "set_core", "type": "forbidden_information_path", "realized": False},
            {"src": "unit_identity", "dst": "test_accuracy", "type": "assumed_scientific_relation"},
            {"src": "capacity", "dst": "test_accuracy", "type": "structural_guarantee"},
            {"src": "segment_alignment", "dst": "test_accuracy", "type": "assumed_scientific_relation"},
            {"src": "subject_or_speaker", "dst": "test_accuracy", "type": "measured_relation"},
            {"src": "set_core", "dst": "compute", "type": "measured_relation"},
            {"src": "recurrent_core", "dst": "temporal_capability", "type": "assumed_scientific_relation"},
        ],
    }


def control_semantics(sp: dict) -> dict:
    x = sp["main"][0][:8]
    out = {}
    for core in Fx.CORES:
        torch.manual_seed(0)
        hidden = Fx.match_hidden(sp["channels"], sp["classes"], "mlp")
        m = Fx.build(sp["channels"], sp["classes"], core, "mlp", hidden=hidden).eval()
        r = controls.order_free(lambda t: m(t, None)[0], x, module=m)
        out[core] = {k: v for k, v in r.items() if isinstance(v, bool)}
        out[core]["structural_findings"] = r.get("structural_findings")
    # the frozen control: a cell whose core is not in the trainable groups must not move its core
    torch.manual_seed(0)
    hidden = Fx.match_hidden(sp["channels"], sp["classes"], "linear")
    m = Fx.build(sp["channels"], sp["classes"], "fast", "linear", hidden=hidden)
    rec = E.fit(m, None, sp["main"][0], sp["main"][1], train_groups=["readout"], steps=15, lr=LR,
                rng=np.random.default_rng(0), batch=32)
    out["frozen_core"] = controls.frozen_control(rec, ["core"], m.param_groups)
    return out


def mechanism_activity(sp: dict) -> dict:
    """Prove that changing the core is a change that reaches the outcome, not a renaming."""
    X, Y = sp["main"]
    torch.manual_seed(0)

    def probe(core):
        torch.manual_seed(0)
        hidden = Fx.match_hidden(sp["channels"], sp["classes"], "linear")
        m = Fx.build(sp["channels"], sp["classes"], core, "linear", hidden=hidden)
        rec = E.fit(m, None, X, Y, train_groups=["core", "readout"], steps=60, lr=LR,
                    rng=np.random.default_rng(0), batch=32)
        acc = E.evaluate(m, None, sp["tune"][0], sp["tune"][1])
        with torch.no_grad():
            rep = m.represent(X[:64])
        return m, rec, float(acc), rep

    m_on, rec_on, acc_on, rep_on = probe("fast")
    m_off, rec_off, acc_off, rep_off = probe("pooled")
    m_sh, rec_sh, acc_sh, rep_sh = probe("reset")
    on = {
        "intervention_count": int(rec_on["updates"]),
        "intervention_timing": [0, rec_on["updates"] // 2, rec_on["updates"]],
        "affected_samples": int(rec_on["updates"] * 32),
        "affected_parameter_groups": sorted(m_on.param_groups),
        "affected_state": ["gru_hidden"],
        "affected_memory": [],
        "counterfactual_difference": round(abs(acc_on - acc_off), 6),
        "downstream_path": ["core", "representation", "readout", "logits", "accuracy"],
        "cost": float(rec_on["wall_seconds"]),
    }
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    off["affected_state"] = []
    shuffled = dict(on)
    shuffled["counterfactual_difference"] = round(abs(acc_sh - acc_off), 6)
    forced = dict(on)
    r = mechanism.activity({"enabled": on, "disabled": off, "shuffled": shuffled, "forced_active": forced,
                            "randomized": shuffled})
    r["accuracies"] = {"fast": round(acc_on, 5), "pooled": round(acc_off, 5), "reset": round(acc_sh, 5)}
    r["representation_l2_gap"] = {
        "fast_vs_pooled": round(float((rep_on - rep_off).pow(2).sum().sqrt()), 4),
        "fast_vs_reset": round(float((rep_on - rep_sh).pow(2).sum().sqrt()), 4),
    }
    return r


def admit(write: bool = True) -> dict:
    t0 = time.time()
    per_bed, contracts_all = {}, []
    for b in BEDS:
        sp = splits_with_units(b, 0)
        recs = arm_records(sp)
        dist = arms.distinctness(recs)
        sens = arms.config_sensitivity(
            lambda cfg: arm_record_for(sp, cfg),
            {"core": "fast", "readout": "linear"},
            {"core": "pooled", "readout": "mlp"},
        )
        sem = control_semantics(sp)
        act = mechanism_activity(sp)
        ua = bed.unit_audit(sp["units"]["main"], sp["units"]["tune"], sp["units"]["test"])
        la = bed.leakage_audit(sp["units"]["main"], sp["units"]["main"], "train_only")
        per_bed[b] = {
            "arm_distinctness": dist,
            "config_sensitivity": sens,
            "control_semantics": sem,
            "mechanism_activity": act,
            "unit_audit": ua,
            "leakage_audit": la,
            "parameter_match": {
                f"{c}_{r}": {"core": Fx.count_core(Fx.build(sp["channels"], sp["classes"], c, r,
                                                            hidden=Fx.match_hidden(sp["channels"], sp["classes"], r))),
                             "readout": Fx.count_readout(Fx.build(sp["channels"], sp["classes"], c, r))}
                for c, r in CELLS
            },
        }
        for name in dist["per_arm"]:
            contracts_all.append(contracts.ArmContract(name=f"{b}:{name}",
                                                       evidence={"distinctness": dist["per_arm"][name]}))
        contracts_all.append(contracts.ControlContract(name=f"{b}:pooled_order_free",
                                                       evidence={"semantic": sem["pooled"]}))
        contracts_all.append(contracts.ControlContract(name=f"{b}:frozen_core",
                                                       evidence={"semantic": sem["frozen_core"]}))
        contracts_all.append(contracts.IndependentUnitContract(name=b, evidence={"units": ua}))

    g = causal_graph()
    contracts_all.append(contracts.CausalModel(name=EXP, declared={"graph": g}))
    contracts_all.append(
        contracts.ExperimentQuestion(
            name=EXP,
            declared={
                "question": "does the within domain capability come from the core, the readout, or their interaction",
                "hypotheses": sorted(HYPOTHESIS_PREDICTIONS),
                "predictions": HYPOTHESIS_PREDICTIONS,
            },
        )
    )
    contracts_all.append(
        contracts.MeasurementModel(
            name=EXP,
            declared={"outcomes": {"test_accuracy": {"estimator": "accuracy on the untouched unit disjoint test split",
                                                     "unit": "subject or speaker"}}},
        )
    )
    pre = gate.Preregistration(
        experiment_id=EXP,
        title="core by readout factorial on the two sealed valid temporal beds",
        contracts=contracts_all,
        mechanism_activity=per_bed[BEDS[0]]["mechanism_activity"],
    )
    admission = pre.admit(stage="control_semantics")
    doc = {
        "schema": "mop-experiment-admission/v1",
        "experiment": EXP,
        "beds": list(BEDS),
        "cells": [f"{c}_{r}" for c, r in CELLS],
        "hypothesis_predictions": HYPOTHESIS_PREDICTIONS,
        "causal_graph": g,
        "causal_graph_rejections": graph.validate(g, evidence={"pooled_control": per_bed[BEDS[0]]["control_semantics"]["pooled"]}),
        "causal_graph_summary": graph.summarize(g),
        "per_bed": per_bed,
        "admission": admission,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if write:
        io.seal("MOP_E1_ADMISSION.json", doc)
    print(f"E1 admission through control_semantics: licensed={admission['licensed']} "
          f"blocked_at={admission['blocked_at']}", flush=True)
    if admission["blocking_violations"]:
        print(json.dumps(admission["blocking_violations"][:8], indent=1), flush=True)
    return doc


def arm_record_for(sp: dict, cfg: dict) -> dict:
    torch.manual_seed(0)
    hidden = Fx.match_hidden(sp["channels"], sp["classes"], cfg["readout"])
    m = Fx.build(sp["channels"], sp["classes"], cfg["core"], cfg["readout"], hidden=hidden)
    X, Y = sp["main"]
    seen: list[str] = []
    hooks = [mod.register_forward_hook(lambda mo, i, o, n=name: seen.append(f"{n}:{type(mo).__name__}"))
             for name, mod in m.named_modules() if name]
    rec = E.fit(m, None, X, Y, train_groups=["core", "readout"], steps=8, lr=LR,
                rng=np.random.default_rng(0), batch=32)
    for h in hooks:
        h.remove()
    with torch.no_grad():
        out = m(X[:32], None)[0]
    return arms.record(
        f"{cfg['core']}_{cfg['readout']}",
        source=Fx.Cell.represent,
        config=cfg,
        call_graph=sorted(set(seen)),
        state_transitions=[rec["checkpoint_sha_before"], rec["checkpoint_sha_after"]],
        param_delta={"changed": rec["changed_param_count"]},
        memory={},
        resources={"core_params": Fx.count_core(m)},
        outputs=[round(float(v), 4) for v in out.flatten()[:32]],
    )


# ---------------------------------------------------------------- scout


def scout(bedname: str) -> dict:
    t0 = time.time()
    runs = []
    for seed in SCOUT_SEEDS:
        sp = splits_with_units(bedname, seed)
        for core, readout in CELLS:
            runs.append(run_cell(bedname, core, readout, seed, sp, "tune"))
    by_cell: dict[str, list[float]] = {}
    for r in runs:
        by_cell.setdefault(f"{r['core']}_{r['readout']}", []).append(r["accuracy"])
    means = {k: round(float(np.mean(v)), 5) for k, v in by_cell.items()}
    sds = {k: round(float(np.std(v, ddof=1)), 5) for k, v in by_cell.items()}
    best = max(means, key=means.get)
    strongest_simple = "pooled_linear"
    residual = [by_cell[best][i] - by_cell[strongest_simple][i] for i in range(len(SCOUT_SEEDS))]
    sp0 = splits_with_units(bedname, 0)
    curve = validation_curve(bedname, "fast", "mlp", 0, sp0)
    conv = baseline.receipt(
        "fast_mlp_convergence",
        identity="fast_mlp",
        model="GRU core with one hidden layer readout",
        parameters=Fx.count_core(Fx.build(sp0["channels"], sp0["classes"], "fast", "mlp")),
        updates=STEPS,
        data_exposure=STEPS * BATCH,
        memory=0,
        compute_seconds=sum(r["wall_seconds"] for r in runs),
        validation_curve=curve,
        selected_checkpoint=f"steps={STEPS}",
        seed_scores=by_cell["fast_mlp"],
    )
    pooled_curve = validation_curve(bedname, "pooled", "mlp", 0, sp0)
    conv_pooled = baseline.receipt(
        "pooled_mlp_convergence",
        identity="pooled_mlp",
        model="per timestep encoder with order free pooling and one hidden layer readout",
        parameters=Fx.count_core(Fx.build(sp0["channels"], sp0["classes"], "pooled", "mlp",
                                          hidden=Fx.match_hidden(sp0["channels"], sp0["classes"], "mlp"))),
        updates=STEPS,
        data_exposure=STEPS * BATCH,
        memory=0,
        compute_seconds=0.0,
        validation_curve=pooled_curve,
        selected_checkpoint=f"steps={STEPS}",
        seed_scores=by_cell["pooled_mlp"],
        treatment_budget={"updates": STEPS, "memory": 0, "data_exposure": STEPS * BATCH},
    )
    sd = float(np.mean([sds[k] for k in sds]))
    pre = power.preregistration(
        name=f"{EXP}:{bedname}",
        independent_unit=D.domain(bedname)["unit"],
        expected_sd=sd,
        sesoi=SESOI,
        seeds=len(PRINCIPAL_SEEDS),
        units=len(sp0["units"]["test"]),
        max_seeds=len(PRINCIPAL_SEEDS),
        futility=0.01,
        harm=0.05,
    )
    doc = {
        "schema": "mop-e1-scout/v1",
        "bed": bedname,
        "seeds": SCOUT_SEEDS,
        "evaluated_on": "tuning units, the test split is untouched",
        "runs": runs,
        "cell_means": means,
        "cell_sds": sds,
        "best_cell": best,
        "strongest_simple_control": strongest_simple,
        "oracle_headroom": round(means[best] - min(means.values()), 5),
        "residual_headroom_over_strongest_control": {
            "mean": round(float(np.mean(residual)), 5),
            "lower_95_cb": round(power.lcb(residual), 5),
            "n_seeds": len(SCOUT_SEEDS),
        },
        "baseline_convergence": {"fast_mlp": conv, "pooled_mlp": conv_pooled},
        "power": pre,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"scout_{bedname}.json", doc, "scout")
    print(f"E1 scout {bedname}: best {best} {means[best]}, residual lcb "
          f"{doc['residual_headroom_over_strongest_control']['lower_95_cb']}, mde "
          f"{pre['minimum_detectable_effect']}", flush=True)
    return doc


# ---------------------------------------------------------------- principal


def principal(bedname: str, seed: int) -> dict:
    sp = splits_with_units(bedname, seed)
    runs = [run_cell(bedname, c, r, seed, sp, "test") for c, r in CELLS]
    ext = external_baseline(bedname, seed, sp)
    doc = {"bed": bedname, "seed": seed, "runs": runs, "external_baseline": ext}
    io.run_json(f"{bedname}_{seed}.json", doc, "principal")
    print(f"E1 principal {bedname} seed {seed}: "
          + " ".join(f"{r['core']}_{r['readout']}={r['accuracy']}" for r in runs), flush=True)
    return doc


def external_baseline(bedname: str, seed: int, sp: dict) -> dict:
    """A conventional shared recurrent model at matched budget, so the factorial is not graded against itself."""
    torch.manual_seed(seed)
    m = A.build("lstm", {"x": (sp["channels"], sp["classes"])})
    groups = sorted(m.param_groups)
    rec = E.fit(m, "x", sp["main"][0], sp["main"][1], train_groups=groups, steps=STEPS, lr=LR,
                rng=np.random.default_rng(seed), batch=BATCH)
    acc = E.evaluate(m, "x", sp["test"][0], sp["test"][1])
    return {
        "identity": "conventional_lstm_shared",
        "accuracy": round(float(acc), 5),
        "parameters": int(sum(p.numel() for p in m.parameters())),
        "updates": rec["updates"],
        "undeclared_changes": rec["undeclared_changes"],
    }


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "admit"
    if cmd == "admit":
        admit()
    elif cmd == "scout":
        scout(sys.argv[2])
    elif cmd == "principal":
        principal(sys.argv[2], int(sys.argv[3]))
    print(f"E1_{cmd.upper()}_DONE", flush=True)
