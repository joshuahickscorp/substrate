"""Substrate experiments, and the first one the gate refuses.

SX1 is the obvious first experiment: does a typed workspace beat an untyped one of matched capacity. It is
also the first place a Substrate result could be manufactured, so it is worth being precise about why it
cannot run as first designed.

On a synthetic bed the comparison has a closed form. Typing refuses writes from an undeclared component, so
the typed arm reports the declared writer's value and the untyped arm reports whatever wrote last. Which
arm wins follows from the two reliabilities the bed generator was given. Nothing is measured; the answer
was set when the bed was written. The historical defect ledger already carries this failure under its own
name, analytic values reported as measured, and the causal graph validator refuses a claim that is broader
than any measured relation.

So SX1 is preregistered honestly, with the treatment to outcome edge labelled as the structural guarantee
it is, and the gate refuses it. That refusal is a methodological result and not a scientific null, and this
module records the difference.

SX1b is the version that could run: the same question on a bed where the undeclared writer's reliability is
measured from held out data rather than chosen by the generator, which turns a closed form into a real
question about whether a write restriction fitted on training units survives on units it has never seen.

House style: no dashes.
"""

from __future__ import annotations

import json
import sys

from mop.cognition import admission as A
from mop.cognition import io
from mop.method import contracts as C
from mop.method import gate
from mop.method import graph as G

# ---------------------------------------------------------------- SX1, as honestly declared

SX1_GRAPH = {
    "nodes": [
        {"id": "typing", "type": "mechanism", "label": "region write typing",
         "implementation": "mop.cognition.workspace.Workspace._permitted"},
        {"id": "restrict", "type": "intervention", "label": "refuse an undeclared write",
         "implementation": "mop.cognition.workspace.Workspace.write"},
        {"id": "typed", "type": "treatment", "label": "typed workspace",
         "implementation": "mop.cognition.workspace.Workspace"},
        {"id": "untyped", "type": "control", "label": "untyped workspace of matched capacity",
         "removes": "reader and writer sets",
         "implementation": "mop.cognition.workspace.UntypedWorkspace"},
        {"id": "reliabilities", "type": "hidden_information", "label": "the generator's two accuracies",
         "time": "decision_time", "declared": True},
        {"id": "unit", "type": "independent_unit", "label": "a held out source group"},
        {"id": "answer", "type": "primary_outcome", "label": "task accuracy"},
        {"id": "claim", "type": "claim", "label": "typing improves cross perspective accuracy",
         "requires": ["answer"]},
    ],
    "edges": [
        {"src": "typing", "dst": "restrict", "type": "implemented_causal_path"},
        {"src": "restrict", "dst": "typed", "type": "implemented_causal_path"},
        # the honest label. On a synthetic bed this edge is fixed by the generator, not measured.
        {"src": "typed", "dst": "answer", "type": "structural_guarantee"},
        {"src": "untyped", "dst": "answer", "type": "implemented_causal_path"},
        {"src": "reliabilities", "dst": "answer", "type": "analytic_relation"},
        {"src": "unit", "dst": "answer", "type": "analytic_relation"},
    ],
}


def sx1() -> gate.Preregistration:
    """The full preregistration. Every mandatory contract is present, so the refusal is about the science."""
    return gate.Preregistration(
        experiment_id="SX1",
        title="does a typed workspace beat an untyped one of matched capacity on a synthetic bed",
        mechanism_activity={"active": True, "classification": "active",
                            "failed": [],
                            "evidence": "an undeclared write is refused and the region is unchanged"},
        contracts=[
            C.ExperimentQuestion(name="q", declared={
                "question": "does refusing undeclared writes raise cross perspective accuracy",
                "hypotheses": ["H_typed_workspace", "H_capacity_only"],
                "predictions": {"H_typed_workspace": {"typed_beats_untyped": True},
                                "H_capacity_only": {"typed_beats_untyped": False}}}),
            C.CausalModel(name="g", declared={"graph": SX1_GRAPH}),
            C.MeasurementModel(name="m", declared={"outcomes": {
                "accuracy": {"estimator": "mean over held out source groups",
                             "unit": "held out source group"}}}),
            C.InstrumentContract(name="i", evidence={"calibration": {
                "recovers_a_planted_corruption": True, "reports_none_when_none_planted": True,
                "all_pass": True}}),
            C.ArmContract(name="typed", evidence={"distinctness": {"untyped": "distinct"}}),
            C.ControlContract(name="untyped", evidence={"semantic": {
                "typing_absent": True, "capacity_matched": True, "cost_accounting_retained": True,
                "all_pass": True}}),
            C.DatasetContract(name="bed", evidence={"bed_validity": {
                "classification": "valid_principal_bed"}}),
            C.IndependentUnitContract(name="units", evidence={"units": {
                "group_disjoint": True, "n_units": 8, "test_touched": False}}),
            C.BaselineContract(name="base", declared={"identity": "untyped workspace"},
                               evidence={"convergence": {"converged": True, "resource_matched": True,
                                                         "identity": "untyped workspace"}}),
            C.OracleContract(name="oracle", evidence={"headroom": {
                "n_seeds": 8, "residual_lower_95_cb": 0.22}}),
            C.PowerContract(name="power",
                            declared={"sesoi": 0.05, "seeds": 8, "futility": 0.01, "harm": -0.02},
                            evidence={"power": {"mde": 0.03}}),
        ])


def sx1_decision() -> dict:
    """Admit SX1 and record why it is refused, in the vocabulary the method kernel already uses."""
    prereg = sx1()
    report = A.admit(prereg, "principal")
    graph_violations = G.validate(SX1_GRAPH)
    return {
        "schema": "substrate-experiment-decision/v1",
        "experiment_id": "SX1",
        "title": prereg.title,
        # named explicitly so the hypothesis graph can attach the refusal without guessing from prose
        "hypotheses": sorted({h for c in prereg.contracts if c.kind == "ExperimentQuestion"
                              for h in c.declared.get("hypotheses", [])}),
        "admission": report,
        "causal_graph_violations": graph_violations,
        "licensed": report["licensed"],
        "classification": "methodological_refusal" if not report["licensed"] else "licensed",
        "why": ("on a synthetic bed the comparison has a closed form. Typing refuses writes from an "
                "undeclared component, so which arm wins follows from the two reliabilities the generator "
                "was given. The treatment to outcome edge is a structural guarantee, no measured relation "
                "reaches the outcome, and the claim is therefore broader than the measured path"),
        "not_a_scientific_null": ("a methodological failure is not a scientific null. H_typed_workspace is "
                                  "untested, not refuted, and stays open in the hypothesis graph"),
        "successor": "SX1b",
        "activation": False,
    }


# ---------------------------------------------------------------- SX1b, the version that could run

SX1B_DESIGN = {
    "schema": "substrate-experiment-design/v1",
    "experiment_id": "SX1b",
    "question": ("does a write restriction fitted on training units still help on source groups it has "
                 "never seen"),
    "why_it_is_not_a_closed_form": ("the reliability of each writer is measured on training units rather "
                                    "than set by a generator, so the restriction can be fitted to a "
                                    "reliability ordering that does not hold on held out groups. The "
                                    "answer depends on how stable that ordering is across units, which "
                                    "no closed form supplies"),
    "arms": {
        "typed_fitted": "writers restricted to the perspective that was more reliable on training units",
        "typed_oracle": "writers restricted using the held out ordering, an upper bound",
        "untyped": "the matched capacity control, no restriction",
        "typed_wrong": "restricted to the less reliable training writer, the harm direction",
    },
    "bed_requirements": ["a real corpus already under data custody",
                         "source groups that make train and test disjoint",
                         "at least two perspectives whose reliabilities differ by group",
                         "a measured, not declared, reliability ordering"],
    "candidate_beds": ["harth_stream", "pamap2_stream"],
    "primary_outcome": "accuracy per held out source group",
    "sesoi": 0.05,
    "predictions": {
        "H_typed_workspace": {"typed_fitted_beats_untyped": True},
        "H_reliability_instability": {"typed_fitted_beats_untyped": False,
                                      "typed_oracle_beats_untyped": True},
        "H_capacity_only": {"typed_fitted_beats_untyped": False, "typed_oracle_beats_untyped": False},
    },
    "blocked_on": ("principal compute. The temporal core factorial holds every worker slot, and this "
                   "design is queued behind it rather than run alongside it"),
    "activation": False,
}


# ---------------------------------------------------------------- SX1b, implemented

BED_CACHE = io.ROOT.parent / "mop-data" / "harth" / "harth_stream.npz"


def _features(x):
    """Two views of the same window, each holding information the other does not.

    static is the mean over time and knows posture but not motion. dynamic is the mean absolute step to
    step change and knows motion but not posture. Neither is a subset of the other.
    """
    import numpy as np

    return {"static": x.mean(axis=1), "dynamic": np.abs(np.diff(x, axis=1)).mean(axis=1)}


def _centroid_fit(f, y):
    import numpy as np

    mu, sd = f.mean(axis=0), f.std(axis=0) + 1e-8
    z = (f - mu) / sd
    classes = np.unique(y)
    return {"mu": mu, "sd": sd, "classes": classes,
            "centroids": np.stack([z[y == c].mean(axis=0) for c in classes])}


def _centroid_predict(model, f):
    import numpy as np

    z = (f - model["mu"]) / model["sd"]
    d = ((z[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2)
    return model["classes"][d.argmin(axis=1)]


def load_bed() -> dict:
    """harth_stream, already under custody, split by source group."""
    import numpy as np

    if not BED_CACHE.is_file():
        raise FileNotFoundError(f"the harth stream cache is not under custody at {BED_CACHE}")
    d = np.load(BED_CACHE)
    return {"Xtr": d["Xtr"], "Ytr": d["Ytr"], "Utr": d["Utr"],
            "Xte": d["Xte"], "Yte": d["Yte"], "Ute": d["Ute"]}


def per_unit_accuracy(bed: dict, split: str) -> dict:
    """Fit each perspective on train, score it on every unit of the requested split."""
    import numpy as np

    ftr = _features(bed["Xtr"])
    models = {name: _centroid_fit(f, bed["Ytr"]) for name, f in ftr.items()}
    X, Y, U = bed[f"X{split}"], bed[f"Y{split}"], bed[f"U{split}"]
    feats = _features(X)
    out: dict[str, dict] = {}
    for unit in sorted(set(U.tolist())):
        m = U == unit
        out[unit] = {name: float((_centroid_predict(models[name], feats[name][m]) == Y[m]).mean())
                     for name in models}
    return out


def sx1b_evidence() -> dict:
    """Everything the gate needs, measured on train units only. The test split stays untouched here."""
    import numpy as np

    bed = load_bed()
    train_units = sorted(set(bed["Utr"].tolist()))
    test_units = sorted(set(bed["Ute"].tolist()))
    disjoint = not (set(train_units) & set(test_units))

    # reliability is measured on training units, never declared
    tr = per_unit_accuracy(bed, "tr")
    means = {p: float(np.mean([row[p] for row in tr.values()])) for p in ("static", "dynamic")}
    fitted = max(means, key=means.get)
    wrong = min(means, key=means.get)

    # the arms, evaluated on training units so nothing here touches the test split
    def arm_scores(rows: dict) -> dict:
        rng = np.random.default_rng(0)
        untyped = {u: float(rng.choice([r["static"], r["dynamic"]])) for u, r in rows.items()}
        return {"typed_fitted": {u: r[fitted] for u, r in rows.items()},
                "typed_wrong": {u: r[wrong] for u, r in rows.items()},
                "typed_oracle": {u: max(r.values()) for u, r in rows.items()},
                "untyped": untyped}

    arms = arm_scores(tr)
    per_arm = {a: float(np.mean(list(v.values()))) for a, v in arms.items()}
    gaps = [max(r.values()) - min(r.values()) for r in tr.values()]
    residual = per_arm["typed_oracle"] - per_arm["untyped"]
    spread = float(np.std([arms["typed_oracle"][u] - arms["untyped"][u] for u in tr], ddof=1))
    n = len(train_units)
    lower = residual - 1.96 * spread / max(n, 1) ** 0.5
    mde = 1.96 * float(np.std(list(arms["typed_fitted"].values()), ddof=1)) / max(len(test_units), 1) ** 0.5

    return {
        "bed": "harth_stream",
        "cache": str(BED_CACHE),
        "train_units": train_units, "test_units": test_units, "units_disjoint": disjoint,
        "measured_reliability_on_train": means,
        "fitted_writer": fitted, "wrong_writer": wrong,
        "arm_means_on_train": per_arm,
        "per_unit_reliability_gap": {"mean": round(float(np.mean(gaps)), 6),
                                     "max": round(float(np.max(gaps)), 6)},
        "oracle_headroom": {"n_seeds": n, "residual": round(residual, 6),
                            "residual_lower_95_cb": round(lower, 6)},
        "power": {"mde": round(mde, 6), "n_test_units": len(test_units)},
        "arms_distinct": len({tuple(round(v, 9) for v in arms[a].values()) for a in arms}) == len(arms),
        "test_split_touched": False,
    }


SX1B_GRAPH = {
    "nodes": [
        {"id": "restriction", "type": "mechanism", "label": "a fitted write restriction",
         "implementation": "mop.cognition.experiments.sx1b_evidence"},
        {"id": "fit", "type": "intervention", "label": "restrict writes to the train reliable writer",
         "implementation": "mop.cognition.workspace.Workspace.write"},
        {"id": "typed_fitted", "type": "treatment", "label": "fitted restriction",
         "implementation": "mop.cognition.experiments.sx1b_run"},
        {"id": "untyped", "type": "control", "label": "no restriction, an arbitrary writer wins",
         "removes": "the write restriction",
         "implementation": "mop.cognition.workspace.UntypedWorkspace"},
        {"id": "ordering", "type": "confounder", "label": "reliability ordering stability across units",
         "declared": True},
        {"id": "unit", "type": "independent_unit", "label": "a held out source group"},
        {"id": "accuracy", "type": "primary_outcome", "label": "accuracy per held out unit"},
        {"id": "claim", "type": "claim", "label": "a fitted write restriction transfers to unseen units",
         "requires": ["accuracy"]},
    ],
    "edges": [
        {"src": "restriction", "dst": "fit", "type": "implemented_causal_path"},
        {"src": "fit", "dst": "typed_fitted", "type": "implemented_causal_path"},
        {"src": "typed_fitted", "dst": "accuracy", "type": "implemented_causal_path"},
        {"src": "untyped", "dst": "accuracy", "type": "implemented_causal_path"},
        {"src": "ordering", "dst": "accuracy", "type": "assumed_scientific_relation"},
        {"src": "unit", "dst": "accuracy", "type": "measured_relation"},
    ],
}


def sx1b(evidence: dict) -> gate.Preregistration:
    return gate.Preregistration(
        experiment_id="SX1b",
        title=SX1B_DESIGN["question"],
        mechanism_activity={
            "active": evidence["per_unit_reliability_gap"]["mean"] > 0.0,
            "classification": "active" if evidence["per_unit_reliability_gap"]["mean"] > 0.0
            else "inactive_mechanism",
            "failed": [],
            "evidence": "the two writers disagree in reliability on the training units"},
        contracts=[
            C.ExperimentQuestion(name="q", declared={
                "question": SX1B_DESIGN["question"],
                "hypotheses": list(SX1B_DESIGN["predictions"]),
                "predictions": SX1B_DESIGN["predictions"]}),
            C.CausalModel(name="g", declared={"graph": SX1B_GRAPH}),
            C.MeasurementModel(name="m", declared={"outcomes": {
                "accuracy": {"estimator": "mean over held out source groups",
                             "unit": "held out source group"}}}),
            C.InstrumentContract(name="i", evidence={"calibration": {
                "two_views_are_not_subsets": True,
                "reliability_measured_not_declared": True,
                "all_pass": True}}),
            C.ArmContract(name="typed_fitted", evidence={"distinctness": {
                "untyped": "distinct" if evidence["arms_distinct"] else "aliased",
                "typed_wrong": "distinct" if evidence["arms_distinct"] else "aliased"}}),
            C.ControlContract(name="untyped", evidence={"semantic": {
                "restriction_absent": True, "same_two_writers": True, "same_features": True,
                "all_pass": True}}),
            C.DatasetContract(name="bed", evidence={"bed_validity": {
                "classification": "valid_principal_bed"}}),
            C.IndependentUnitContract(name="units", evidence={"units": {
                "group_disjoint": evidence["units_disjoint"],
                "n_units": len(evidence["test_units"]),
                "test_touched": evidence["test_split_touched"]}}),
            C.BaselineContract(name="base", declared={"identity": "no restriction"},
                               evidence={"convergence": {
                                   "converged": True, "resource_matched": True,
                                   "identity": "no restriction"}}),
            C.OracleContract(name="oracle", evidence={"headroom": evidence["oracle_headroom"]}),
            C.PowerContract(name="power",
                            declared={"sesoi": 0.05, "seeds": len(evidence["test_units"]),
                                      "futility": 0.01, "harm": -0.02},
                            evidence={"power": evidence["power"]}),
        ])


SESOI = 0.05


def _sx1b_diagnosis(evidence: dict) -> dict:
    """Which number actually decides the case, and whether more units would change it.

    The power contract blocks on the minimum detectable effect, which is a function of how many held out
    units the bed has. That invites the obvious response of finding more units. It would not help here,
    and saying so is the point of this function: the oracle residual is the ceiling on any restriction
    whatsoever on this bed, it does not depend on n, and it is already below the SESOI.
    """
    residual = evidence["oracle_headroom"]["residual"]
    mde = evidence["power"]["mde"]
    ceiling_below_sesoi = residual <= SESOI
    return {
        "blocking_contract": "power",
        "reported_mde": mde,
        "sesoi": SESOI,
        "oracle_residual": residual,
        "oracle_residual_lower_95_cb": evidence["oracle_headroom"]["residual_lower_95_cb"],
        "more_units_would_help": not ceiling_below_sesoi,
        "decisive_number": "oracle_residual" if ceiling_below_sesoi else "mde",
        "finding": (
            "the oracle restriction, which is the best any write restriction could do on this bed, beats "
            f"the unrestricted control by {residual:.4f}. That ceiling does not depend on how many units "
            f"are held out, and it is below the SESOI of {SESOI}. More units would shrink the minimum "
            "detectable effect and still leave nothing above the SESOI to detect"
            if ceiling_below_sesoi else
            f"the ceiling of {residual:.4f} clears the SESOI, so the block is the unit count alone and a "
            "bed with more held out groups would answer the question"),
        "classification": "bed_cannot_answer_the_question_at_this_effect_size",
        "not_a_null": ("H_typed_workspace is untested on this bed, not refuted. A bed that cannot see the "
                       "declared effect says nothing about whether the effect exists"),
        "successor_requirement": ("a bed where the two writers' reliability ordering varies more between "
                                  "source groups, so that the oracle ceiling itself clears the SESOI"),
    }


def sx1b_run() -> dict:
    """Admit, and only if licensed, measure once on the held out units."""
    import numpy as np

    evidence = sx1b_evidence()
    prereg = sx1b(evidence)
    report = A.admit(prereg, "principal")
    out = {"schema": "substrate-experiment-decision/v1", "experiment_id": "SX1b",
           "title": prereg.title,
           "hypotheses": list(SX1B_DESIGN["predictions"]),
           "preprincipal_evidence": evidence, "admission": report,
           "causal_graph_violations": G.validate(SX1B_GRAPH),
           "licensed": report["licensed"],
           "classification": "licensed" if report["licensed"] else "methodological_refusal",
           "activation": False}
    if not report["licensed"]:
        out["why"] = "; ".join(report["blocking_violations"])
        out["diagnosis"] = _sx1b_diagnosis(evidence)
        return out

    bed = load_bed()
    te = per_unit_accuracy(bed, "te")
    rng = np.random.default_rng(0)
    arms = {
        "typed_fitted": {u: r[evidence["fitted_writer"]] for u, r in te.items()},
        "typed_wrong": {u: r[evidence["wrong_writer"]] for u, r in te.items()},
        "typed_oracle": {u: max(r.values()) for u, r in te.items()},
        "untyped": {u: float(rng.choice([r["static"], r["dynamic"]])) for u, r in te.items()},
    }
    means = {a: float(np.mean(list(v.values()))) for a, v in arms.items()}
    paired = [arms["typed_fitted"][u] - arms["untyped"][u] for u in te]
    n = len(paired)
    effect = float(np.mean(paired))
    half = 1.96 * float(np.std(paired, ddof=1)) / max(n, 1) ** 0.5
    verdict = ("positive" if effect - half > 0.05 else
               "harm" if effect + half < -0.02 else "null")
    out["principal"] = {
        "per_unit": te, "arm_scores": arms, "arm_means": means,
        "effect_typed_fitted_minus_untyped": round(effect, 6),
        "lower_95_cb": round(effect - half, 6), "upper_95_cb": round(effect + half, 6),
        "n_units": n, "sesoi": 0.05, "verdict": verdict,
    }
    out["result"] = gate.classify_result(
        effect={"verdict": verdict, "estimate": round(effect, 6),
                "lower_95_cb": round(effect - half, 6)},
        instrument_valid=True, bed_valid=True,
        mechanism_active=prereg.mechanism_activity["active"],
        baseline_valid=True,
        estimator_sufficient=evidence["power"]["mde"] <= 0.05,
        verifier_agrees=False, mutations_rejected=False, implementations_agreeing=1)
    return out


# ---------------------------------------------------------------- the bed screen

# Declared before the screen was run. Searching beds until one clears the SESOI would be the same defect
# as searching arms until one clears it, so the rule, the candidate list and the outcome if nothing clears
# are all fixed here rather than decided afterwards.
BED_SCREEN_RULE = {
    "schema": "substrate-bed-screen-rule/v1",
    "question": "is there any bed under custody where a write restriction could clear the SESOI at all",
    "screen": ("on training source groups only, measure the oracle ceiling, meaning the mean per unit "
               "best of the two writers minus the unrestricted control. A bed is a candidate only if the "
               "lower 95 percent bound of that ceiling exceeds the SESOI"),
    "candidates": ["harth_stream", "pamap2_stream"],
    "excluded": {"har_stream": "classified an invalid principal bed by the fast state reaudit",
                 "speech_stream": "classified an invalid principal bed by the fast state reaudit"},
    "sesoi": 0.05,
    "outcome_if_none_clears": ("the finding is that no bed under custody can test H_typed_workspace at "
                               "the declared effect size. That is a mapped boundary and not a null, and "
                               "the SESOI is not lowered to manufacture a candidate"),
    "prior_measurement": ("the harth_stream ceiling of 0.036124 was measured by SX1b before this rule was "
                          "written and is carried forward unchanged rather than remeasured"),
    "activation": False,
}

BED_CACHES = {"harth_stream": "harth/harth_stream.npz", "pamap2_stream": "pamap2/pamap2_stream.npz"}


def screen_bed(name: str) -> dict:
    import numpy as np

    cache = io.ROOT.parent / "mop-data" / BED_CACHES[name]
    if not cache.is_file():
        return {"bed": name, "available": False, "reason": f"no cache under custody at {cache}"}
    d = np.load(cache)
    bed = {k: d[k] for k in ("Xtr", "Ytr", "Utr", "Xte", "Yte", "Ute")}
    rows = per_unit_accuracy(bed, "tr")
    rng = np.random.default_rng(0)
    ceiling = [max(r.values()) - float(rng.choice([r["static"], r["dynamic"]])) for r in rows.values()]
    n = len(ceiling)
    mean = float(np.mean(ceiling))
    half = 1.96 * float(np.std(ceiling, ddof=1)) / max(n, 1) ** 0.5
    return {"bed": name, "available": True, "n_train_units": n,
            "mean_reliability": {p: round(float(np.mean([r[p] for r in rows.values()])), 6)
                                 for p in ("static", "dynamic")},
            "oracle_ceiling": round(mean, 6),
            "oracle_ceiling_lower_95_cb": round(mean - half, 6),
            "clears_sesoi": mean - half > BED_SCREEN_RULE["sesoi"]}


def bed_screen() -> dict:
    rows = [screen_bed(name) for name in BED_SCREEN_RULE["candidates"]]
    clearing = [r["bed"] for r in rows if r.get("clears_sesoi")]
    return {
        "schema": "substrate-bed-screen/v1",
        "rule": BED_SCREEN_RULE,
        "screened": rows,
        "beds_clearing_the_sesoi": clearing,
        "any_candidate": bool(clearing),
        "finding": ("SX1c may proceed on " + ", ".join(clearing) if clearing else
                    "no bed under custody can test H_typed_workspace at the declared SESOI of "
                    f"{BED_SCREEN_RULE['sesoi']}. The hypothesis stays untested and the SESOI stays where "
                    "it was preregistered"),
        "classification": "candidate_bed_found" if clearing else "no_bed_can_answer_at_this_effect_size",
        "not_a_null": ("a hypothesis nothing can currently measure is untested, not refuted. Nothing "
                       "downstream of H_typed_workspace closes"),
        "activation": False,
    }


# ---------------------------------------------------------------- value of information
#
# The lesson of SX1, SX1b and the bed screen, stated once so the queue can act on it. A Substrate
# hypothesis whose mechanism is a design choice needs a bed the program did not design. Where the bed is
# synthetic, the oracle ceiling is whatever the generator was told to make it, and the experiment measures
# the generator. Every candidate below is scored on that axis under instrumentation_risk.

def _c(cid, title, question, separated, justification, **scores):
    return {"id": cid, "title": title, "question": question, "hypotheses_separated": separated,
            "justification": justification, "scores": scores}


VOI_CANDIDATES = [
    _c("SX1c", "typed workspace on a bed with a larger reliability spread",
       "does a fitted write restriction transfer to unseen source groups",
       ["H_typed_workspace", "H_capacity_only"],
       {"oracle_headroom": "measured at 0.036 and 0.038 on the only two valid beds, both below the SESOI",
        "closed_premise": "not closed, but no instrument currently reaches it"},
       expected_information_gain=0.6, probability_of_changing_the_substrate_decision=0.5,
       compute_cost=0.1, duration_cost=0.1, instrumentation_risk=0.8, baseline_uncertainty=0.2,
       oracle_headroom=0.0, independent_unit_quality=0.7, reusability_of_implementation=0.9,
       reusability_of_data=0.9, discriminates_competing_hypotheses=0.8,
       risk_of_repeating_a_closed_premise=0.1),
    _c("SX5", "self model calibration against the temporal program's own history",
       "does updating a self model beat a fixed prior of the same form on real paired outcomes",
       ["H_selfmodel_calibration", "H_owned_continuity"],
       {"bed": "the temporal program's sealed shard receipts carry predicted and actual wall seconds, "
               "failures and quarantines. The bed exists already and this program did not design it",
        "oracle_headroom": "a fixed prior is measurably miscalibrated on any stream with drift, and the "
                           "temporal run has drifted across three resource classes",
        "cost": "reading sealed receipts, no principal compute"},
       expected_information_gain=0.7, probability_of_changing_the_substrate_decision=0.6,
       compute_cost=0.05, duration_cost=0.05, instrumentation_risk=0.2, baseline_uncertainty=0.2,
       oracle_headroom=0.6, independent_unit_quality=0.7, reusability_of_implementation=0.8,
       reusability_of_data=0.6, discriminates_competing_hypotheses=0.6,
       risk_of_repeating_a_closed_premise=0.1),
    _c("SX2", "perspective diversity against the best single perspective",
       "does a heterogeneous perspective set beat the strongest single one at matched compute",
       ["H_perspective_diversity", "H_learned_selector"],
       {"oracle_headroom": "unmeasured. The perspectives are implemented but no bed has scored them",
        "instrumentation_risk": "the six implemented perspectives read different regions, so a bed has "
                                "to supply information for all of them or the comparison is unfair"},
       expected_information_gain=0.6, probability_of_changing_the_substrate_decision=0.5,
       compute_cost=0.2, duration_cost=0.2, instrumentation_risk=0.6, baseline_uncertainty=0.4,
       oracle_headroom=0.3, independent_unit_quality=0.5, reusability_of_implementation=0.8,
       reusability_of_data=0.4, discriminates_competing_hypotheses=0.7,
       risk_of_repeating_a_closed_premise=0.1),
    _c("SX4", "owned continuity against matched budget transcript replay",
       "does owned state restore more than a transcript replay of the same size",
       ["H_owned_continuity"],
       {"instrumentation_risk": "high. On a synthetic session the transcript length is a design choice, "
                                "so the margin is set by how much filler the generator adds. This is the "
                                "SX1 objection in a different costume and needs a session nobody wrote "
                                "for the purpose",
        "oracle_headroom": "unbounded replay recovers three of five probes on the demo, so a real ceiling "
                           "exists, but not one this program has measured on an undesigned session"},
       expected_information_gain=0.6, probability_of_changing_the_substrate_decision=0.5,
       compute_cost=0.1, duration_cost=0.1, instrumentation_risk=0.85, baseline_uncertainty=0.5,
       oracle_headroom=0.3, independent_unit_quality=0.3, reusability_of_implementation=0.7,
       reusability_of_data=0.3, discriminates_competing_hypotheses=0.5,
       risk_of_repeating_a_closed_premise=0.2),
    _c("SX7", "learned plasticity policy against the strongest simple rule",
       "does a learned plasticity policy beat a simple triggered rule",
       ["H_learned_plasticity_policy"],
       {"closed_premise": "the fast state program measured no stable headroom and sealed the null. "
                          "Rerunning it without new headroom repeats a closed premise"},
       expected_information_gain=0.3, probability_of_changing_the_substrate_decision=0.2,
       compute_cost=0.7, duration_cost=0.7, instrumentation_risk=0.4, baseline_uncertainty=0.3,
       oracle_headroom=0.1, independent_unit_quality=0.5, reusability_of_implementation=0.5,
       reusability_of_data=0.5, discriminates_competing_hypotheses=0.3,
       risk_of_repeating_a_closed_premise=0.9),
]


def voi_queue() -> dict:
    from mop.method import voi

    q = voi.queue(VOI_CANDIDATES, select=2)
    return {**q, "schema": "substrate-experiment-value-queue/v1",
            "engine": "mop.method.voi, the queue the method reformation already sealed",
            "program_lesson": ("a Substrate hypothesis whose mechanism is a design choice needs a bed the "
                               "program did not design. On a synthetic bed the oracle ceiling is whatever "
                               "the generator was told to make it, and the experiment measures the "
                               "generator. That axis is scored as instrumentation_risk"),
            "activation": False}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    command = argv[0] if argv else "seal"
    if command == "seal":
        decision = sx1_decision()
        io.run_json("SX1_decision.json", decision, "experiments")
        io.run_json("SX1b_design.json", SX1B_DESIGN, "experiments")
        print(json.dumps({"SX1_licensed": decision["licensed"],
                          "classification": decision["classification"],
                          "graph_violations": decision["causal_graph_violations"],
                          "successor": decision["successor"]}, indent=2))
    elif command == "screen":
        out = bed_screen()
        io.run_json("bed_screen.json", out, "experiments")
        print(json.dumps({"screened": [{k: r.get(k) for k in
                                        ("bed", "available", "oracle_ceiling",
                                         "oracle_ceiling_lower_95_cb", "clears_sesoi")}
                                       for r in out["screened"]],
                          "classification": out["classification"],
                          "finding": out["finding"]}, indent=2))
    elif command == "sx1b":
        out = sx1b_run()
        io.run_json("SX1b_decision.json", out, "experiments")
        summary = {"licensed": out["licensed"],
                   "blocked_at": out["admission"]["blocked_at"],
                   "graph_violations": out["causal_graph_violations"]}
        if out["licensed"]:
            summary |= {"arm_means": out["principal"]["arm_means"],
                        "effect": out["principal"]["effect_typed_fitted_minus_untyped"],
                        "lower_95_cb": out["principal"]["lower_95_cb"],
                        "classification": out["result"]["classification"],
                        "reason": out["result"]["reason"]}
        print(json.dumps(summary, indent=2))
    else:
        raise ValueError(argv)


if __name__ == "__main__":
    main()
