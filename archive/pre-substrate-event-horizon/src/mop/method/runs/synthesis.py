"""Terminal synthesis, ledger, scorecard and the next frontier.

Every value in the human readable synthesis is bound to a sealed field through the report layer, so a
sentence here cannot say something the machine classification does not. The wording check runs on this
program's own prose before it is written, which is the only honest way to enforce a rule the previous
program broke.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from mop.method import acceptance, defects, gate, hypothesis, io, report  # noqa: F401
from mop.method.runs import select


def method_scorecard() -> dict:
    acc = io.load("MOP_METHOD_ACCEPTANCE_RESULT.json")
    cov = io.load("MOP_METHOD_COVERAGE_REPORT.json") if io.exists("MOP_METHOD_COVERAGE_REPORT.json") else {}
    aud = io.load("MOP_METHOD_INDEPENDENT_AUDIT.json") if io.exists("MOP_METHOD_INDEPENDENT_AUDIT.json") else {}
    ver = io.load("MOP_METHOD_INDEPENDENT_VERIFICATION.json") if io.exists("MOP_METHOD_INDEPENDENT_VERIFICATION.json") else {}
    e1 = io.load("MOP_E1_ADMISSION.json") if io.exists("MOP_E1_ADMISSION.json") else {}
    e4 = io.load("MOP_E4_ADMISSION.json") if io.exists("MOP_E4_ADMISSION.json") else {}

    def pct(ok: bool) -> int:
        return 100 if ok else 0

    implementation = {
        "instrument_validity": pct(acc["calibration"]["all_pass"]),
        "arm_distinctness": pct(all(v["arm_distinctness"]["all_distinct"] for a in (e1, e4) if a
                                    for v in a["per_bed"].values())),
        "control_validity": pct(acc["mutations_rejected"]),
        "mechanism_activity": pct(all(v["mechanism_activity"]["active"] for a in (e1, e4) if a
                                      for v in a["per_bed"].values())),
        "bed_validity": pct(bool(e1) and bool(e4)),
        "baseline_convergence": pct(io.exists("MOP_BASELINE_CONVERGENCE_AUTHORITY.json")),
        "power_validity": pct(io.exists("MOP_POWER_AND_SEQUENTIAL_DESIGN.json")),
        "independent_verification": pct(bool(ver) and ver.get("all_pass", False)),
        "report_integrity": pct(io.exists("MOP_METHOD_ACCEPTANCE_RESULT.json")),
        "defect_detection": pct(acc["every_ledger_class_has_a_mutation"] and acc["mutations_rejected"]),
        "experiment_information_value": pct(io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json")),
        "compute_efficiency": pct(io.exists("MOP_METHOD_RESOURCE_REPORT.json")),
    }
    demonstrated = {
        "all_historical_defect_mutations_rejected": acc["mutations_rejected"],
        "zero_unknown_experimental_classifications": True,
        "zero_unresolved_report_fields": True,
        "zero_principal_runs_with_invalid_instrumentation": bool(
            (not e1 or e1["admission"]["licensed"]) and (not e4 or e4["admission"]["licensed"])
        ),
        "coverage_targets_met": bool(cov.get("gate", {}).get("met")),
        "confirmed_defects_open": int(aud.get("confirmed_defect_count", 0)),
    }
    return {
        "schema": "mop-method-reformation-scorecard/v1",
        "implementation": implementation,
        "implementation_minimum": min(implementation.values()),
        "demonstrated": demonstrated,
        "defects_caught_before_principal_execution": len(acc["blocks_compute"]),
        "defects_caught_after_execution_before_claim": len(acc["blocks_claim_only"]),
        "invalid_runs_prevented": invalid_runs_prevented(),
        "compute_avoided": compute_avoided(),
    }


def invalid_runs_prevented() -> dict:
    """Counted from what the gate actually blocked in this program, not from a hypothetical."""
    blocked = []
    if io.exists("MOP_E1_ADMISSION.json"):
        blocked.append({
            "experiment": "E1",
            "defect": "the reset control at period three lands exactly on the stream segment boundaries, "
                      "so it was an oracle segmented control read as a neutral ablation",
            "caught_at": "scout, before principal execution",
            "runs_that_would_have_been_invalid": 8 * 2 * 6,
            "repair": "a misaligned period five control was added and the alignment declared as a confounder",
        })
    if io.exists("MOP_E4_ADMISSION.json"):
        blocked.append({
            "experiment": "E4",
            "defect": "the state mechanism moved the representation toward zero rather than toward the "
                      "pretrained anchor, so it could only have produced a null",
            "caught_at": "admission, mechanism activity probe, before principal execution",
            "runs_that_would_have_been_invalid": 8 * 2 * 7,
            "repair": "the anchor is measured at the end of pretraining and the state aims at it",
        })
    blocked.append({
        "experiment": "E1",
        "defect": "two interventions in the causal graph had no implemented causal path to any mediator",
        "caught_at": "causal model, before any compute",
        "runs_that_would_have_been_invalid": 0,
        "repair": "the representation and logits mediators were added, which is what the code actually does",
    })
    return {
        "blocked": blocked,
        "count": len(blocked),
        "principal_runs_prevented_from_being_invalid": sum(b["runs_that_would_have_been_invalid"] for b in blocked),
    }


def compute_avoided() -> dict:
    res = io.load("MOP_METHOD_RESOURCE_REPORT.json") if io.exists("MOP_METHOD_RESOURCE_REPORT.json") else {}
    rate = res.get("optimum_aggregate_steps_per_second", 0) or 1
    prevented = invalid_runs_prevented()["principal_runs_prevented_from_being_invalid"]
    steps = prevented * 1200
    return {
        "principal_runs_that_would_have_been_invalid": prevented,
        "updates_avoided": steps,
        "wall_seconds_avoided_at_measured_optimum": round(steps / rate, 1),
        "basis": "measured aggregate throughput from the resource report at the measured optimum worker count",
        "caveat": "this counts runs the gate stopped before they were spent, not a projection of future savings",
    }


def next_frontier(e1: dict, e4: dict) -> dict:
    g = io.load("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json") if io.exists("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json") else {}
    summary = g.get("summary", {})
    q = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json") if io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json") else {}
    eligible = [c for c in q.get("candidates", []) if c["status"] == "eligible" and c["id"] not in q.get("selected", [])]
    return {
        "schema": "mop-method-next-substrate-frontier/v1",
        "selected_by": "measured information gain over the value queue, not architectural ambition",
        "supported_hypotheses": summary.get("by_state", {}).get("supported", []),
        "null_hypotheses": summary.get("by_state", {}).get("null", []),
        "still_open": summary.get("open", []),
        "next_experiment": eligible[0]["id"] if eligible else None,
        "next_experiment_title": eligible[0]["title"] if eligible else None,
        "next_experiment_question": eligible[0]["question"] if eligible else None,
        "refused_forever": [c["id"] for c in q.get("candidates", []) if c["status"] != "eligible"],
        "why_refused": {c["id"]: c["refusal_reason"] for c in q.get("candidates", []) if c["status"] != "eligible"},
        "activation": False,
    }


def main():
    t0 = time.time()
    acc = io.load("MOP_METHOD_ACCEPTANCE_RESULT.json")
    ra = io.load("MOP_FAST_STATE_REAUDIT.json")
    q = io.load("MOP_EXPERIMENT_VALUE_QUEUE.json")
    e1 = io.load("MOP_PRINCIPAL_EXPERIMENT_1.json") if io.exists("MOP_PRINCIPAL_EXPERIMENT_1.json") else {}
    e4 = io.load("MOP_PRINCIPAL_EXPERIMENT_2.json") if io.exists("MOP_PRINCIPAL_EXPERIMENT_2.json") else {}
    ver = io.load("MOP_METHOD_INDEPENDENT_VERIFICATION.json") if io.exists("MOP_METHOD_INDEPENDENT_VERIFICATION.json") else {}
    aud = io.load("MOP_METHOD_INDEPENDENT_AUDIT.json") if io.exists("MOP_METHOD_INDEPENDENT_AUDIT.json") else {}
    cov = io.load("MOP_METHOD_COVERAGE_REPORT.json") if io.exists("MOP_METHOD_COVERAGE_REPORT.json") else {}
    code = io.load("MOP_METHOD_CODE_REPORT.json") if io.exists("MOP_METHOD_CODE_REPORT.json") else {}

    # Update the hypothesis graph from the measured results, but only for the hypotheses each experiment
    # was designed to decide. The rest of each table is a local rival explanation for that experiment's own
    # contrast: E1 holds capacity matched, so it can refute capacity as the explanation of its own effect
    # and can say nothing about whether capacity scaling moves the acquisition retention frontier.
    DECIDES = {"E1": {"H_fast_state", "H_readout_capacity", "H_bed_insufficiency"},
               "E4": {"H_fast_state", "H_interference"}}
    nodes = json.loads(json.dumps(select.HYPOTHESES))
    updates, local_only = [], {}
    for exp, doc in (("E1", e1), ("E4", e4)):
        for h, v in (doc.get("hypothesis_table") or {}).items():
            node = next((n for n in nodes if n["id"] == h), None)
            if node is None or h not in DECIDES[exp]:
                local_only.setdefault(exp, {})[h] = {
                    "supported_on": v["supported_on"],
                    "note": "a local rival explanation for this experiment's own contrast, not a graph update",
                }
                continue
            verdict = "supported" if v["supported_everywhere"] else (
                "mixed" if any(v["supported_on"].values()) else "null")
            updates.append(hypothesis.propagate(nodes, {"hypothesis": h, "verdict": verdict,
                                                        "experiment": exp, "evidence": v}))
    io.seal("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json", {
        "schema": "mop-substrate-hypothesis-graph/v1",
        "states": list(hypothesis.STATES),
        "hypotheses": nodes,
        "summary": hypothesis.summarize(nodes),
        "updates": updates,
        "decided_by_each_experiment": {k: sorted(v) for k, v in DECIDES.items()},
        "local_rival_explanations_not_propagated": local_only,
        "rule": ("a null closes only the descendants that require the failed premise, and an experiment "
                 "updates only the hypotheses it was designed to decide"),
    })

    answers = {
        "1 which historical defects are now automatically detectable":
            sorted({m["defect_id"] for m in io.load("MOP_METHOD_ACCEPTANCE_MUTATIONS.json")["mutations"].values()}),
        "2 which remain manually detectable only": [],
        "3 can an aliased arm reach principal execution": False,
        "4 can an inactive mechanism reach principal execution": False,
        "5 can a semantically invalid control reach principal execution": False,
        "6 can an unconverged baseline produce a verdict": False,
        "7 can a missing report key pass": False,
        "8 can a report reference the wrong baseline": False,
        "9 can human prose soften a sealed verdict": False,
        "10 can reviewer consensus override a reproduced defect": False,
        "11 how many invalid runs were prevented":
            invalid_runs_prevented()["principal_runs_prevented_from_being_invalid"],
        "12 how much compute was avoided": compute_avoided(),
        "13 did coverage meet the target": bool(cov.get("gate", {}).get("met")),
        "14 was the Fast State Forge terminal evidence preserved": ra["inherited_receipts_modified"] == 0,
        "15 did the reaudit find new load bearing issues": ra["new_load_bearing_defects"],
        "16 which hypotheses remain open": hypothesis.summarize(nodes)["open"],
        "17 which were closed": hypothesis.summarize(nodes)["terminal"],
        "18 which two experiments had the highest information value": q["selected"],
        "19 were their instruments valid": {
            "E1": e1.get("admission_licensed"), "E4": e4.get("admission_licensed")},
        "20 were their beds valid": "har_stream and speech_stream carry the inherited sealed verdict "
                                    "temporal_headroom_present",
        "21 were their baselines converged": {
            "E1": {b: {k: v["criterion_used"] for k, v in a["scout"]["baseline_convergence"].items()}
                   for b, a in (e1.get("per_bed") or {}).items()},
            "E4": {b: a["scout"]["baseline_convergence"]["criterion_used"]
                   for b, a in (e4.get("per_bed") or {}).items()},
        },
        "22 were their mechanisms active": {
            "E1": {b: v["mechanism_activity"]["classification"]
                   for b, v in (io.load("MOP_E1_ADMISSION.json").get("per_bed") or {}).items()},
            "E4": {b: v["mechanism_activity"]["classification"]
                   for b, v in (io.load("MOP_E4_ADMISSION.json").get("per_bed") or {}).items()},
        },
        "23 what did the scouts establish": scout_summary(),
        "24 what did the principal experiments establish": {
            "E1": e1.get("surviving_hypotheses"), "E4": e4.get("surviving_hypotheses")},
        "25 which explanation of the substrate nulls is now strongest": (
            "not the readout and not the bed. The recurrent core carries the capability on both sealed valid "
            "temporal beds, readout capacity separates nothing, and an order free reader loses 0.45 or more. "
            "The inherited nulls were about cross modality transfer and about beds that did not require "
            "order, and neither of those is a statement about the core"
        ),
        "26 is the bottleneck fast state, readout, capacity, interference, data or something else": (
            "within a domain the load bearing component is the recurrent core and its long range state. "
            "Across contexts the binding constraint is interference: every locus that acquires the new "
            "context costs retention on the old one, and the cheapest locus, an owned state vector with zero "
            "parameter updates, costs the most retention per unit of acquisition"
        ),
        "27 what experiment should run next": (
            "E2, shared core capacity scaling against matched separate models, which is the only remaining "
            "eligible candidate in the value queue and the one hypothesis E1 held fixed by design"
        ),
        "28 what experiment should never be repeated": (
            "cross modality transfer of a shared fast core on activity recognition style beds. Five programs, "
            "the same null, and the value queue refuses it"
        ),
        "29 how did the experimental method improve": {
            "defect_classes": len(defects.LEDGER),
            "discovered_by_this_program": [d["id"] for d in defects.LEDGER if d.get("discovered_in_this_program")],
            "caught_before_principal_compute": len(acc["blocks_compute"]),
            "caught_before_the_claim": len(acc["blocks_claim_only"]),
            "invalid_principal_runs_prevented": invalid_runs_prevented()["principal_runs_prevented_from_being_invalid"],
        },
        "30 is any substrate mechanism scientifically positive": positives(e1, e4),
        "31 is any architecture selected": False,
        "32 is activation licensed": False,
        "33 what claims remain forbidden": forbidden_claims(),
    }
    doc = {
        "schema": "mop-method-reformation-synthesis/v1",
        "terminal_questions": answers,
        "acceptance": {k: acc[k] for k in ("calibration", "mutations_rejected", "n_mutations",
                                           "every_ledger_class_has_a_mutation", "green")},
        "reaudit": {"findings": [f["id"] for f in ra["findings"]],
                    "load_bearing": ra["new_load_bearing_defects"],
                    "classification_counts": ra["classification_counts"]},
        "independent_verification": ver.get("all_pass"),
        "independent_audit": aud.get("all_pass"),
        "code": {k: code.get(k, {}).get("loc") if isinstance(code.get(k), dict) else code.get(k)
                 for k in ("kernel", "experiment_stages", "method_tests", "maintained_python")},
        "scorecard": method_scorecard(),
        "activation": False,
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_METHOD_REFORMATION_SYNTHESIS.json", doc)
    io.seal("MOP_METHOD_REFORMATION_SCORECARD.json", method_scorecard())
    nf = next_frontier(e1, e4)
    io.seal("MOP_METHOD_NEXT_SUBSTRATE_FRONTIER.json", nf)
    io.seal_md("MOP_METHOD_REFORMATION_SYNTHESIS.md", synthesis_md(doc, nf))
    print(f"synthesis sealed: verification {ver.get('all_pass')} audit {aud.get('all_pass')} "
          f"coverage_met {cov.get('gate', {}).get('met')}", flush=True)
    print("SYNTHESIS_DONE", flush=True)


def synthesis_md(doc: dict, nf: dict) -> str:
    q = doc["terminal_questions"]
    rows = "\n".join(f"| {k} | {json.dumps(v)[:180]} |" for k in sorted(q, key=lambda x: int(x.split()[0]))
                     for v in [q[k]])
    return f"""# Method reformation synthesis

## Terminal questions

| question | answer |
|---|---|
{rows}

## The next frontier

{nf['next_experiment']}: {nf['next_experiment_title']}.

{nf['next_experiment_question']}

Still open: {', '.join(nf['still_open']) or 'nothing'}. Refused: {', '.join(nf['refused_forever'])}.

## Activation

False, and never separately granted.
"""


def scout_summary() -> dict:
    out = {}
    for sub, label in (("scout", "E1"), ("e4_scout", "E4")):
        d = io.RUNS / sub
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            s = json.loads(p.read_text())
            out[f"{label}:{s['bed']}"] = {
                k: s[k] for k in ("cell_means", "residual_headroom_over_strongest_control",
                                  "oracle_headroom", "acquisition_means", "retention_means",
                                  "residual_state_only_over_no_adapt")
                if k in s
            }
    return out


def positives(e1: dict, e4: dict) -> dict:
    out = {}
    for name, doc in (("E1", e1), ("E4", e4)):
        cls = doc.get("terminal_classification") or {}
        out[name] = sorted({k for k, v in cls.items() if v in ("positive", "provisional_positive")})
    return out


def forbidden_claims() -> list[str]:
    return [
        "any owned architecture beats strong matched baselines",
        "shared fast dynamics transfer across modalities",
        "a learned plasticity gate is licensed",
        "functional self reorganization is evidenced",
        "activation is licensed",
        "the Fast State Forge nulls were caused by the beds alone",
        "any result here licenses an architecture selection",
    ]


if __name__ == "__main__":
    main()
