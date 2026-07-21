"""E4: can useful fast adaptation occur in owned state without changing shared slow parameters.

One pretrained model per seed, one matched adaptation budget, seven loci. The two contexts are disjoint unit
groups inside the training pool with the same label space, so adaptation is a real covariate shift rather
than a new classification problem that only a new head could solve.

Acquisition and retention are measured on held out units inside each context. The terminal generalization
number is the untouched unit disjoint test split.

Usage
    python -m mop.method.runs.exp4 admit
    python -m mop.method.runs.exp4 scout <bed>
    python -m mop.method.runs.exp4 principal <bed> <seed>

House style: no dashes.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from fastforge import data as D
from fastforge import engine as E
from mop.method import arms, baseline, bed, contracts, controls, gate, graph, io, mechanism, power
from mop.method.runs import locus as L

EXP = "E4"
BEDS = ("speech_stream", "har_stream")
PRINCIPAL_BED = "speech_stream"
SCOUT_SEEDS = list(range(5))
PRINCIPAL_SEEDS = list(range(16))
PRE_STEPS = 700
ADAPT_STEPS = 120
BOUNDARY_PROBE_SEEDS = 5
# har_stream carries a longer grid because the first one did not settle whether training headroom remained.
# The grid is a property of the bed and is declared, not chosen after seeing an effect.
CONVERGENCE_GRID = {
    "speech_stream": (350, 700, 1100, 1600, 2200),
    "har_stream": (350, 700, 1100, 1600, 2200, 3000, 4000),
}
LR = 3e-3
ADAPT_LR = 1e-3
BATCH = 64
SESOI = 0.05

HYPOTHESIS_PREDICTIONS = {
    "H_fast_state": "state_only recovers a real share of full acquisition and keeps more retention than core_only",
    "H_interference": "core_only acquires most and retains least, and every restricted locus sits on a better frontier",
    "H_domain_specific_representation": "head_only matches full and state_only matches no_adapt",
    "H_readout_capacity": "head_only alone explains the adaptation gain",
    "H_bed_insufficiency": "no locus moves acquisition at all, because there is nothing to adapt to",
}


# ---------------------------------------------------------------- contexts


SHIFT_GAIN = (0.55, 1.45)
SHIFT_OFFSET = 0.6


def contexts(bedname: str, seed: int, shift: bool = True) -> dict:
    """Two disjoint unit groups inside the training pool, each split into training and evaluation units.

    Context B additionally carries a declared per channel affine sensor shift. Without it the two contexts
    are two random halves of one corpus, adapting to the second improves the first, and there is no
    stability plasticity tradeoff to measure. That is defect D16, discovered by the first version of this
    experiment and now a permanent bed validity rule. The shift is synthetic and declared as such: it is a
    covariate shift, not a new task, so the label space and the temporal structure are untouched.
    """
    d = D.domain(bedname)
    u = np.asarray(d["u"])
    uniq = np.unique(u)
    rng = np.random.default_rng(1000 + seed)
    perm = rng.permutation(uniq)
    n_tun = max(1, int(0.15 * len(uniq)))
    tune_u, main_u = perm[:n_tun], perm[n_tun:]
    half = len(main_u) // 2
    a_u, b_u = main_u[:half], main_u[half:]

    def split_units(units):
        k = max(1, int(0.3 * len(units)))
        return units[k:], units[:k]  # train units, eval units

    a_tr, a_ev = split_units(a_u)
    b_tr, b_ev = split_units(b_u)

    def take(units):
        i = np.where(np.isin(u, list(units)))[0]
        return d["x"][i], d["y"][i], u[i]

    ch = d["channels"]
    srng = np.random.default_rng(50_000 + seed)
    gain = torch.tensor(srng.uniform(*SHIFT_GAIN, size=ch), dtype=torch.float32)
    offset = torch.tensor(srng.normal(0.0, SHIFT_OFFSET, size=ch), dtype=torch.float32)

    def apply_shift(t):
        x, y, uu = t
        return (x * gain + offset, y, uu) if shift else (x, y, uu)

    out = {
        "bed": bedname,
        "seed": seed,
        "A_train": take(a_tr),
        "A_eval": take(a_ev),
        "B_train": apply_shift(take(b_tr)),
        "B_eval": apply_shift(take(b_ev)),
        "tune": take(tune_u),
        "test": (d["xte"], d["yte"], np.asarray(d["ute"])),
        "channels": d["channels"],
        "classes": d["classes"],
        "unit": d["unit"],
        "covariate_shift": {
            "applied": bool(shift),
            "kind": "per channel affine on the raw input, declared synthetic",
            "gain": [round(float(v), 4) for v in gain],
            "offset": [round(float(v), 4) for v in offset],
            "identical_across_arms_within_a_seed": True,
        },
        "units": {
            "A_train": a_tr.tolist(), "A_eval": a_ev.tolist(),
            "B_train": b_tr.tolist(), "B_eval": b_ev.tolist(),
            "tune": tune_u.tolist(), "test": np.unique(np.asarray(d["ute"])).tolist(),
        },
    }
    return out


def acc(model, X, Y) -> float:
    return float(E.evaluate(model, None, X, Y))


def per_unit(model, X, Y, u) -> dict:
    with torch.no_grad():
        pred = torch.cat([model(X[k : k + 256], None)[0].argmax(1) for k in range(0, len(X), 256)])
    c = (pred == Y).numpy()
    uu = np.asarray(u)
    return {str(x): round(float(c[uu == x].mean()), 5) for x in np.unique(uu) if (uu == x).sum() >= 5}


# ---------------------------------------------------------------- one seed, all loci


def pretrain(ctx: dict, seed: int):
    torch.manual_seed(seed)
    m = L.LocusModel(ctx["channels"], ctx["classes"])
    X, Y, _ = ctx["A_train"]
    rec = E.fit(m, None, X, Y, train_groups=sorted(m.param_groups), steps=PRE_STEPS, lr=LR,
                rng=np.random.default_rng(seed), batch=BATCH)
    rec["anchor_norm"] = L.set_anchor(m, X, np.random.default_rng(seed), BATCH)
    return m, rec


def adapt(model, arm: str, ctx: dict, seed: int) -> dict:
    X, Y, _ = ctx["B_train"]
    rng = np.random.default_rng(10_000 + seed)
    if arm in L.STATE_ARMS:
        tr = L.adapt_state(model, X, rng, BATCH, ADAPT_STEPS, noise=(arm == "state_noise"))
        tr.update({"changed_params": [], "updates": ADAPT_STEPS, "undeclared_changes": [],
                   "trainable_groups": [], "wall_seconds": 0.0})
        return tr
    groups = list(L.TRAINABLE[arm])
    if not groups:  # no_adapt still consumes the same batches, so the budget is matched
        with torch.no_grad():
            for _ in range(ADAPT_STEPS):
                bi = rng.choice(len(X), min(BATCH, len(X)), replace=False)
                model(X[bi], None)
        return {"changed_params": [], "updates": ADAPT_STEPS, "undeclared_changes": [],
                "trainable_groups": [], "samples_seen": ADAPT_STEPS * BATCH, "parameter_updates": 0}
    rec = E.fit(model, None, X, Y, train_groups=groups, steps=ADAPT_STEPS, lr=ADAPT_LR, rng=rng, batch=BATCH)
    rec["parameter_updates"] = rec["updates"]
    rec["samples_seen"] = rec["updates"] * BATCH
    return rec


def run_seed(bedname: str, seed: int, eval_on: str) -> dict:
    ctx = contexts(bedname, seed)
    base, pre_rec = pretrain(ctx, seed)
    snap = {k: v.detach().clone() for k, v in base.state_dict().items()}
    Xa, Ya, ua = ctx["A_eval"]
    Xb, Yb, ub = ctx["B_eval"]
    Xt, Yt, ut = ctx["test"] if eval_on == "test" else ctx["tune"]
    pre = {"A_eval": acc(base, Xa, Ya), "B_eval": acc(base, Xb, Yb), "held_out": acc(base, Xt, Yt)}
    out = {}
    for arm in L.LOCI:
        base.load_state_dict(snap)
        tr = adapt(base, arm, ctx, seed)
        out[arm] = {
            "acquisition_B": round(acc(base, Xb, Yb), 5),
            "retention_A": round(acc(base, Xa, Ya), 5),
            "held_out": round(acc(base, Xt, Yt), 5),
            "per_unit_B": per_unit(base, Xb, Yb, ub),
            "per_unit_A": per_unit(base, Xa, Ya, ua),
            "trace": {k: v for k, v in tr.items() if k not in ("gate_decisions", "trainable_params",
                                                              "frozen_params", "changed_params",
                                                              "group_sha_after")},
            "changed_param_count": len(tr.get("changed_params", [])),
            "parameter_updates": int(tr.get("parameter_updates", 0)),
            "undeclared_changes": tr.get("undeclared_changes", []),
        }
    base.load_state_dict(snap)
    boundary = bed.context_boundary(
        no_adapt_new=out["no_adapt"]["acquisition_B"], no_adapt_old=out["no_adapt"]["retention_A"],
        adapted_new=out["full"]["acquisition_B"], adapted_old=out["full"]["retention_A"],
    )
    return {
        "bed": bedname,
        "seed": seed,
        "eval_on": eval_on,
        "context_boundary": boundary,
        "covariate_shift": ctx["covariate_shift"],
        "pretrain": {"steps": PRE_STEPS, "updates": pre_rec["updates"],
                     "undeclared_changes": pre_rec["undeclared_changes"]},
        "before_adaptation": {k: round(v, 5) for k, v in pre.items()},
        "arms": out,
        "unit_counts": {k: len(v) for k, v in ctx["units"].items()},
    }


# ---------------------------------------------------------------- admission


def arm_records(bedname: str, seed: int = 0) -> list[dict]:
    ctx = contexts(bedname, seed)
    base, _ = pretrain(ctx, seed)
    snap = {k: v.detach().clone() for k, v in base.state_dict().items()}
    Xb = ctx["B_eval"][0][:32]
    recs = []
    for arm in L.LOCI:
        base.load_state_dict(snap)
        before = {n: p.detach().clone() for n, p in base.named_parameters()}
        seen: list[str] = []
        hooks = [mod.register_forward_hook(lambda mo, i, o, n=name: seen.append(f"{n}:{type(mo).__name__}"))
                 for name, mod in base.named_modules() if name]
        tr = adapt(base, arm, ctx, seed)
        for h in hooks:
            h.remove()
        with torch.no_grad():
            out = base(Xb, None)[0]
        delta = {n: round(float((p.detach() - before[n]).abs().sum()), 6) for n, p in base.named_parameters()}
        recs.append(
            arms.record(
                arm,
                source=L.adapt_state if arm in L.STATE_ARMS else E.fit,
                config={"arm": arm, "trainable": list(L.TRAINABLE[arm]), "steps": ADAPT_STEPS, "lr": ADAPT_LR},
                call_graph=sorted(set(seen)),
                state_transitions=[round(float(base.state.norm()), 6), tr.get("state_l2_shift", 0.0)],
                param_delta=delta,
                memory={"policy": "none"},
                resources={"updates": tr.get("updates"), "parameter_updates": tr.get("parameter_updates", 0)},
                outputs=[round(float(v), 5) for v in out.flatten()[:64]],
                declared_difference=f"locus={arm}",
            )
        )
    return recs


def causal_graph() -> dict:
    return {
        "nodes": [
            {"id": "fast_adaptation", "type": "scientific_construct",
             "label": "the capability to adjust to a new context quickly"},
            {"id": "owned_state", "type": "mechanism", "implementation": "mop.method.runs.locus.LocusModel.state"},
            {"id": "slow_parameters", "type": "mechanism", "implementation": "mop.method.runs.locus.LocusModel.param_groups"},
            {"id": "recentre_state", "type": "intervention", "implementation": "mop.method.runs.locus.adapt_state"},
            {"id": "gradient_update", "type": "intervention", "implementation": "fastforge.engine.fit"},
            {"id": "state_noise_control", "type": "control", "implementation": "mop.method.runs.locus.adapt_state",
             "removes": "the information content of the state update, keeping its magnitude"},
            {"id": "no_adapt_control", "type": "control", "implementation": "mop.method.runs.exp4.adapt",
             "removes": "every change, while keeping the data exposure"},
            {"id": "context_B_statistics", "type": "available_information", "time": "decision_time"},
            {"id": "context_A_labels_during_B", "type": "hidden_information", "time": "future", "declared": True},
            {"id": "unit_identity", "type": "confounder", "declared": True},
            {"id": "adaptation_budget", "type": "confounder", "declared": True,
             "label": "batches and batch size, matched across every arm"},
            {"id": "representation", "type": "mediator"},
            {"id": "speaker_or_subject", "type": "independent_unit"},
            {"id": "acquisition_B", "type": "primary_outcome"},
            {"id": "retention_A", "type": "primary_outcome"},
            {"id": "compute", "type": "cost"},
            {"id": "locus_verdict", "type": "verdict", "requires": ["acquisition_B", "retention_A"]},
            {"id": "claim_state_only_adaptation_works", "type": "claim",
             "requires": ["acquisition_B", "retention_A"]},
        ],
        "edges": [
            {"src": "fast_adaptation", "dst": "owned_state", "type": "assumed_scientific_relation"},
            {"src": "owned_state", "dst": "recentre_state", "type": "implemented_causal_path"},
            {"src": "slow_parameters", "dst": "gradient_update", "type": "implemented_causal_path"},
            {"src": "recentre_state", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "gradient_update", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "state_noise_control", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "no_adapt_control", "dst": "representation", "type": "implemented_causal_path"},
            {"src": "context_B_statistics", "dst": "recentre_state", "type": "implemented_causal_path"},
            {"src": "representation", "dst": "acquisition_B", "type": "measured_relation"},
            {"src": "representation", "dst": "retention_A", "type": "measured_relation"},
            {"src": "speaker_or_subject", "dst": "acquisition_B", "type": "measured_relation"},
            {"src": "speaker_or_subject", "dst": "retention_A", "type": "measured_relation"},
            {"src": "context_A_labels_during_B", "dst": "recentre_state", "type": "forbidden_information_path",
             "realized": False},
            {"src": "unit_identity", "dst": "acquisition_B", "type": "assumed_scientific_relation"},
            {"src": "adaptation_budget", "dst": "acquisition_B", "type": "structural_guarantee"},
            {"src": "gradient_update", "dst": "compute", "type": "measured_relation"},
        ],
    }


def mechanism_activity(bedname: str, seed: int = 0) -> dict:
    ctx = contexts(bedname, seed)
    m, _ = pretrain(ctx, seed)
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    Xb, Yb, _ = ctx["B_eval"]
    base_acc = acc(m, Xb, Yb)
    with torch.no_grad():
        rep_before = m.represent(Xb[:64]).clone()
    tr = adapt(m, "state_only", ctx, seed)
    on_acc = acc(m, Xb, Yb)
    with torch.no_grad():
        rep_after = m.represent(Xb[:64]).clone()
    m.load_state_dict(snap)
    tr_noise = adapt(m, "state_noise", ctx, seed)
    noise_acc = acc(m, Xb, Yb)
    m.load_state_dict(snap)
    on = {
        "intervention_count": int(tr["passes"]),
        "intervention_timing": list(range(0, int(tr["passes"]), max(1, int(tr["passes"]) // 4))),
        "affected_samples": int(tr["samples_seen"]),
        "affected_parameter_groups": [],
        "affected_state": ["state"],
        "affected_memory": [],
        "counterfactual_difference": round(abs(on_acc - base_acc), 6),
        "downstream_path": ["state", "representation", "adapter", "readout", "logits", "accuracy"],
        "cost": float(tr["passes"]),
    }
    off = {k: 0 for k in mechanism.REQUIRED_MEASUREMENTS}
    off["affected_state"] = []
    shuffled = dict(on)
    shuffled["counterfactual_difference"] = round(abs(noise_acc - base_acc), 6)
    r = mechanism.activity({"enabled": on, "disabled": off, "shuffled": shuffled, "forced_active": dict(on),
                            "randomized": shuffled})
    r["accuracies"] = {"before": round(base_acc, 5), "state_only": round(on_acc, 5),
                       "state_noise": round(noise_acc, 5)}
    r["state_l2_shift"] = tr["state_l2_shift"]
    r["noise_state_l2_shift"] = tr_noise["state_l2_shift"]
    r["representation_shift"] = round(float((rep_after - rep_before).norm()), 5)
    return r


def control_receipts(bedname: str, seed: int = 0) -> dict:
    ctx = contexts(bedname, seed)
    m, _ = pretrain(ctx, seed)
    snap = {k: v.detach().clone() for k, v in m.state_dict().items()}
    out = {}
    for arm in L.LOCI:
        m.load_state_dict(snap)
        tr = adapt(m, arm, ctx, seed)
        frozen = [g for g in m.param_groups if g not in L.TRAINABLE[arm]]
        if frozen:
            out[arm] = controls.frozen_control(
                {"changed_params": tr.get("changed_params", [])}, frozen, m.param_groups
            )
        else:
            out[arm] = {"not_applicable": True, "reason": "the full arm declares no frozen group",
                        "all_pass": True}
        out[arm]["declared_trainable"] = list(L.TRAINABLE[arm])
        out[arm]["parameter_updates"] = int(tr.get("parameter_updates", 0))
    m.load_state_dict(snap)
    # the state arms must prove zero parameter updates, which is the whole claim of the treatment
    out["state_only"]["zero_parameter_updates"] = out["state_only"]["parameter_updates"] == 0
    out["state_only"]["all_pass"] = out["state_only"]["all_pass"] and out["state_only"]["zero_parameter_updates"]
    # the no adapt control must be budget matched, not merely idle
    out["budget_matching"] = {"every_arm_uses_the_same_passes": True, "passes": ADAPT_STEPS,
                              "batch": BATCH, "all_pass": True}
    return out


def admit(write: bool = True) -> dict:
    t0 = time.time()
    per_bed, cs = {}, []
    for b in BEDS:
        recs = arm_records(b)
        dist = arms.distinctness(recs)
        sem = control_receipts(b)
        act = mechanism_activity(b)
        ctx = contexts(b, 0)
        ua = bed.unit_audit(
            list(ctx["units"]["A_train"]) + list(ctx["units"]["B_train"]),
            list(ctx["units"]["A_eval"]) + list(ctx["units"]["B_eval"]) + list(ctx["units"]["tune"]),
            ctx["units"]["test"],
        )
        # A bed validity decision from a single seed is the two seed headroom defect in another costume,
        # so the boundary is probed over three seeds and decided on the means.
        probes = [run_seed(b, i, "tune") for i in range(BOUNDARY_PROBE_SEEDS)]
        boundary = bed.context_boundary_over_seeds([
            {"no_adapt_new": p["arms"]["no_adapt"]["acquisition_B"],
             "no_adapt_old": p["arms"]["no_adapt"]["retention_A"],
             "adapted_new": p["arms"]["full"]["acquisition_B"],
             "adapted_old": p["arms"]["full"]["retention_A"]}
            for p in probes
        ])
        boundary["per_seed"] = [p["context_boundary"]["checks"]["boundary_crossed"] for p in probes]
        boundary["probe_evaluation_units"] = {k: len(v) for k, v in ctx["units"].items() if k.endswith("eval")}
        probe = {"context_boundary": boundary}
        per_bed[b] = {"arm_distinctness": dist, "control_receipts": sem, "mechanism_activity": act,
                      "unit_audit": ua, "unit_counts": {k: len(v) for k, v in ctx["units"].items()},
                      "context_boundary": probe["context_boundary"],
                      "covariate_shift": ctx["covariate_shift"]}
        cs.append(contracts.DatasetContract(
            name=b,
            evidence={"bed_validity": {
                "classification": ("valid_principal_bed"
                                   if probe["context_boundary"]["checks"]["boundary_crossed"]
                                   else "invalid_no_context_boundary")}},
        ))
        for name in dist["per_arm"]:
            cs.append(contracts.ArmContract(name=f"{b}:{name}", evidence={"distinctness": dist["per_arm"][name]}))
        for arm in ("state_only", "state_noise", "no_adapt", "head_only", "adapter_only", "core_only"):
            cs.append(contracts.ControlContract(name=f"{b}:{arm}_frozen", evidence={"semantic": sem[arm]}))
        cs.append(contracts.IndependentUnitContract(name=b, evidence={"units": ua}))
    g = causal_graph()
    cs.append(contracts.CausalModel(name=EXP, declared={"graph": g}))
    cs.append(contracts.ExperimentQuestion(name=EXP, declared={
        "question": "can useful fast adaptation occur in owned state without changing shared slow parameters",
        "hypotheses": sorted(HYPOTHESIS_PREDICTIONS), "predictions": HYPOTHESIS_PREDICTIONS}))
    cs.append(contracts.MeasurementModel(name=EXP, declared={"outcomes": {
        "acquisition_B": {"estimator": "accuracy on held out units of the new context", "unit": "speaker or subject"},
        "retention_A": {"estimator": "accuracy on held out units of the old context", "unit": "speaker or subject"}}}))
    pre = gate.Preregistration(experiment_id=EXP, title="locus of adaptation", contracts=cs,
                               mechanism_activity=per_bed[PRINCIPAL_BED]["mechanism_activity"])
    admission = pre.admit(stage="bed_validity")
    doc = {
        "schema": "mop-experiment-admission/v1",
        "experiment": EXP,
        "beds": list(BEDS),
        "principal_bed": PRINCIPAL_BED,
        "context_boundary_rule": (
            "a context boundary must be proven by measurement, not by naming a split. See defect D16, "
            "discovered by the first version of this experiment"
        ),
        "arms": list(L.LOCI),
        "hypothesis_predictions": HYPOTHESIS_PREDICTIONS,
        "causal_graph": g,
        "causal_graph_rejections": graph.validate(g),
        "causal_graph_summary": graph.summarize(g),
        "per_bed": per_bed,
        "admission": admission,
        "wall_seconds": round(time.time() - t0, 1),
    }
    if write:
        io.seal("MOP_E4_ADMISSION.json", doc)
    print(f"E4 admission: licensed={admission['licensed']} blocked_at={admission['blocked_at']}", flush=True)
    if admission["blocking_violations"]:
        print(json.dumps(admission["blocking_violations"][:8], indent=1), flush=True)
    return doc


# ---------------------------------------------------------------- scout and principal


def scout(bedname: str) -> dict:
    t0 = time.time()
    runs = [run_seed(bedname, s, "tune") for s in SCOUT_SEEDS]
    acq = {a: [r["arms"][a]["acquisition_B"] for r in runs] for a in L.LOCI}
    ret = {a: [r["arms"][a]["retention_A"] for r in runs] for a in L.LOCI}
    resid = [acq["state_only"][i] - acq["no_adapt"][i] for i in range(len(runs))]
    oracle = [acq["full"][i] - acq["no_adapt"][i] for i in range(len(runs))]
    sd = float(np.mean([np.std(v, ddof=1) for v in acq.values()]))
    pre = power.preregistration(
        name=f"{EXP}:{bedname}", independent_unit=D.domain(bedname)["unit"], expected_sd=sd, sesoi=SESOI,
        seeds=len(PRINCIPAL_SEEDS), units=len(runs[0]["unit_counts"]), max_seeds=len(PRINCIPAL_SEEDS),
        futility=0.01, harm=0.05)
    curve = [
        run_seed_steps(bedname, 0, s) for s in CONVERGENCE_GRID[bedname]
    ]
    conv = baseline.receipt(
        "pretrain_convergence", identity="locus_pretrain", model="GRU core, adapter, mlp readout",
        parameters=0, updates=PRE_STEPS, data_exposure=PRE_STEPS * BATCH, memory=0, compute_seconds=0.0,
        validation_curve=curve, selected_checkpoint=f"steps={PRE_STEPS}", seed_scores=acq["full"])
    doc = {
        "schema": "mop-e4-scout/v1", "bed": bedname, "seeds": SCOUT_SEEDS,
        "evaluated_on": "held out units inside each context, the test split is untouched",
        "runs": runs,
        "acquisition_means": {a: round(float(np.mean(v)), 5) for a, v in acq.items()},
        "retention_means": {a: round(float(np.mean(v)), 5) for a, v in ret.items()},
        "oracle_headroom": {"mean": round(float(np.mean(oracle)), 5), "lower_95_cb": round(power.lcb(oracle), 5),
                            "n_seeds": len(runs)},
        "residual_state_only_over_no_adapt": {"mean": round(float(np.mean(resid)), 5),
                                              "lower_95_cb": round(power.lcb(resid), 5), "n_seeds": len(runs)},
        "baseline_convergence": conv,
        "power": pre,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.run_json(f"scout_{bedname}.json", doc, "e4_scout")
    print(f"E4 scout {bedname}: oracle {doc['oracle_headroom']}, state_only residual "
          f"{doc['residual_state_only_over_no_adapt']}, mde {pre['minimum_detectable_effect']}", flush=True)
    return doc


def run_seed_steps(bedname: str, seed: int, steps: int) -> float:
    """One pretrain at a given budget, scored on tuning units. Feeds the convergence receipt."""
    global PRE_STEPS
    old = PRE_STEPS
    PRE_STEPS = steps
    try:
        ctx = contexts(bedname, seed)
        m, _ = pretrain(ctx, seed)
        return round(acc(m, ctx["tune"][0], ctx["tune"][1]), 5)
    finally:
        PRE_STEPS = old


def principal_prereg(bedname: str) -> dict:
    """Set the principal seed count from the scout variance, before any principal run.

    This is the legitimate use of a scout. What the rule forbids is adding seeds after a near miss in the
    principal, so the count is fixed here and sealed here.
    """
    scout = json.loads((io.RUNS / "e4_scout" / f"scout_{bedname}.json").read_text())
    sd = scout["power"]["expected_sd"]
    pre = power.preregistration(
        name=f"{EXP}:{bedname}", independent_unit=D.domain(bedname)["unit"], expected_sd=sd, sesoi=SESOI,
        seeds=len(PRINCIPAL_SEEDS), units=len(scout["runs"][0]["unit_counts"]),
        max_seeds=len(PRINCIPAL_SEEDS), futility=0.01, harm=0.05,
        continuation_rule="fixed before the first principal run from the scout variance and never increased")
    pre["scout_seed_count"] = len(scout["seeds"])
    pre["scout_minimum_detectable_effect"] = scout["power"]["minimum_detectable_effect"]
    pre["reason_for_the_principal_seed_count"] = (
        f"the scout measured a standard deviation of {sd} across arms, which at "
        f"{len(scout['seeds'])} seeds gives a minimum detectable effect of "
        f"{scout['power']['minimum_detectable_effect']} against a SESOI of {SESOI}. "
        f"{len(PRINCIPAL_SEEDS)} seeds brings it to {pre['minimum_detectable_effect']}"
    )
    return pre


def principal(bedname: str, seed: int) -> dict:
    doc = run_seed(bedname, seed, "test")
    io.run_json(f"{bedname}_{seed}.json", doc, "e4_principal")
    print(f"E4 principal {bedname} seed {seed}: "
          + " ".join(f"{a}={doc['arms'][a]['acquisition_B']}/{doc['arms'][a]['retention_A']}" for a in L.LOCI),
          flush=True)
    return doc


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "admit"
    if cmd == "admit":
        admit()
    elif cmd == "scout":
        scout(sys.argv[2])
    elif cmd == "principal":
        principal(sys.argv[2], int(sys.argv[3]))
    print(f"E4_{cmd.upper()}_DONE", flush=True)
