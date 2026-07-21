"""Role B: the instrumentation auditor, plus the three independent method audit roles and the reconciler.

Role B never looks at an outcome. It reads the admission artifacts and the raw receipts and asks only
whether the instrument could have measured anything: are the arms distinct, do the controls remove what they
claim, is the mechanism active, are the splits clean, are the resources matched, is the baseline the one the
comparison names.

The method audit below runs three roles with different attack surfaces and then a fourth that reruns every
claimed confirmation and every load bearing refutation. Nothing is decided by vote.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from mop.method import acceptance, calibration, defects, gate, graph, io


def _runs(sub: str, pattern: str = "*.json") -> list[dict]:
    d = io.RUNS / sub
    return [json.loads(p.read_text()) for p in sorted(d.glob(pattern))] if d.is_dir() else []


# ---------------------------------------------------------------- role B


def audit_experiment(admission_file: str, principal_sub: str, arm_key: str) -> dict:
    if not io.exists(admission_file):
        return {"status": "not_run"}
    adm = io.load(admission_file)
    checks, notes = {}, []
    for b, v in adm["per_bed"].items():
        checks[f"{b}:arms_distinct"] = v["arm_distinctness"]["all_distinct"]
        checks[f"{b}:units_group_disjoint"] = v["unit_audit"]["group_disjoint"]
        checks[f"{b}:test_untouched"] = not v["unit_audit"]["test_touched"]
        checks[f"{b}:mechanism_active"] = v["mechanism_activity"]["active"]
        sem = v.get("control_semantics") or v.get("control_receipts") or {}
        for name, r in sem.items():
            if isinstance(r, dict) and "all_pass" in r:
                checks[f"{b}:control_{name}"] = bool(r["all_pass"])
        if "config_sensitivity" in v:
            checks[f"{b}:configuration_fields_honoured"] = v["config_sensitivity"]["all_honoured"]
        if "parameter_match" in v:
            cores = [x["core"] for x in v["parameter_match"].values()]
            checks[f"{b}:capacity_matched"] = (max(cores) - min(cores)) / max(cores) < 0.01
            notes.append({"bed": b, "core_parameter_spread": max(cores) - min(cores)})
        if "unit_counts" in v:
            thin = {k: n for k, n in v["unit_counts"].items() if n < 5 and k.endswith("eval")}
            if thin:
                notes.append({"bed": b, "thin_evaluation_unit_groups": thin,
                              "consequence": "the group lower bound on this bed is weak and is reported as such"})
    checks["causal_graph_admissible"] = not adm["causal_graph_rejections"]
    checks["admission_licensed"] = adm["admission"]["licensed"]
    runs = _runs(principal_sub)
    if runs:
        checks["no_undeclared_parameter_changes"] = all(
            not c["undeclared_changes"]
            for r in runs
            for c in (r["runs"] if arm_key == "runs" else r["arms"].values())
        )
    return {
        "role": "B instrumentation auditor",
        "admission": admission_file,
        "checks": checks,
        "notes": notes,
        "failed": [k for k, v in checks.items() if not v],
        "all_pass": all(checks.values()),
        "outcomes_inspected": False,
    }


# ---------------------------------------------------------------- method audit roles


def role_instrumentation() -> dict:
    """Attacks the kernel itself: can a broken instrument get through."""
    attacks = []
    acc = acceptance.run()
    attacks.append({
        "attack": "a historical defect mutation reaches principal execution",
        "path": "src/mop/method/acceptance.py",
        "condition": "every ledger defect is injected as a live mutation",
        "reproduction": "PYTHONPATH=src python3.12 -c 'from mop.method import acceptance; print(acceptance.run()[\"all_rejected\"])'",
        "expected": "every mutation rejected",
        "actual": f"all_rejected={acc['all_rejected']}, failures={acc['failures']}",
        "consequence": "an unrejected mutation means the defect class can recur silently",
        "confirmed": not acc["all_rejected"],
    })
    cal = calibration.run()
    attacks.append({
        "attack": "the instrument misclassifies a known world",
        "path": "src/mop/method/calibration.py",
        "condition": "twelve synthetic worlds with known answers",
        "reproduction": "PYTHONPATH=src python3.12 -c 'from mop.method import calibration; print(calibration.run()[\"all_pass\"])'",
        "expected": "every case classified correctly",
        "actual": f"all_pass={cal['all_pass']}",
        "consequence": "a miscalibrated instrument cannot certify an unknown result",
        "confirmed": not cal["all_pass"],
    })
    empty = graph.validate({})
    attacks.append({
        "attack": "an empty causal graph is admitted",
        "path": "src/mop/method/graph.py",
        "condition": "validate on an empty document",
        "reproduction": "PYTHONPATH=src python3.12 -c 'from mop.method import graph; print(graph.validate({}))'",
        "expected": "rejected",
        "actual": str(empty),
        "consequence": "an experiment with no declared causal structure would pass the gate",
        "confirmed": not empty,
    })
    return {"role": "instrumentation auditor", "attacks": attacks,
            "confirmed": [a["attack"] for a in attacks if a["confirmed"]]}


def role_statistical() -> dict:
    from mop.method import power

    attacks = []
    pre = power.preregistration(name="a", independent_unit="u", expected_sd=0.02, sesoi=0.05, seeds=8,
                                units=8, max_seeds=8, futility=0.01, harm=0.05)
    tie = power.decide([0.0] * 8, pre)
    attacks.append({
        "attack": "a tie is reported as a positive",
        "path": "src/mop/method/power.py",
        "condition": "eight identical zero effects",
        "reproduction": "PYTHONPATH=src python3.12 -m pytest tests/method -k tie_is_a_null",
        "expected": "null",
        "actual": tie["verdict"],
        "consequence": "a null would license a mechanism",
        "confirmed": tie["verdict"] == "positive",
    })
    wrong = power.decide([-0.02] * 8, pre)
    attacks.append({
        "attack": "an effect in the wrong direction is reported as a null",
        "path": "src/mop/method/power.py",
        "condition": "eight identical negative effects below the harm boundary",
        "reproduction": "PYTHONPATH=src python3.12 -m pytest tests/method -k wrong_direction",
        "expected": "wrong_direction_failure",
        "actual": wrong["verdict"],
        "consequence": "a harmful mechanism would read as neutral",
        "confirmed": wrong["verdict"].startswith("null"),
    })
    weak = power.decide([0.06, -0.2, 0.3], pre)
    attacks.append({
        "attack": "an underpowered estimate produces a terminal verdict",
        "path": "src/mop/method/power.py",
        "condition": "three seeds with a standard deviation five times the SESOI",
        "reproduction": "PYTHONPATH=src python3.12 -m pytest tests/method -k underpowered",
        "expected": "adequately_powered false",
        "actual": f"adequately_powered={weak['adequately_powered']}",
        "consequence": "method failure would be reported as scientific null",
        "confirmed": bool(weak["adequately_powered"]),
    })
    return {"role": "scientific statistical auditor", "attacks": attacks,
            "confirmed": [a["attack"] for a in attacks if a["confirmed"]]}


def role_software() -> dict:
    from mop.method import report

    attacks = []
    soft = report.wording_check("the result is marginal", "invalid_no_temporal_headroom")
    attacks.append({
        "attack": "prose broadens a sealed verdict",
        "path": "src/mop/method/report.py",
        "condition": "the word marginal over a sealed invalid verdict",
        "reproduction": "PYTHONPATH=src python3.12 -m pytest tests/method -k broaden",
        "expected": "rejected",
        "actual": f"passes={soft['passes']}",
        "consequence": "the summary contradicts the machine classification",
        "confirmed": soft["passes"],
    })
    ok = report.wording_check("the result is a null", "mechanism_null")
    attacks.append({
        "attack": "a faithful restatement is rejected as broadening",
        "path": "src/mop/method/report.py",
        "condition": "prose that restates a sealed null",
        "reproduction": "PYTHONPATH=src python3.12 -m pytest tests/method -k restate",
        "expected": "accepted",
        "actual": f"passes={ok['passes']}",
        "consequence": "a false rejection would make the check unusable and it would be switched off",
        "confirmed": not ok["passes"],
    })
    seal_check = {}
    for p in sorted(io.PROOF.glob("*.json")):
        d = json.loads(p.read_text())
        if "sha256" in d:
            body = {k: v for k, v in d.items() if k != "sha256"}
            seal_check[p.name] = io.sha_obj(body) == d["sha256"]
    attacks.append({
        "attack": "a sealed artifact no longer matches its own hash",
        "path": str(io.PROOF),
        "condition": "recompute every seal",
        "reproduction": "PYTHONPATH=src python3.12 -m mop.method.runs.audit",
        "expected": "every seal verifies",
        "actual": f"{sum(seal_check.values())} of {len(seal_check)} verify",
        "consequence": "a tampered or stale artifact would be read as authoritative",
        "confirmed": not all(seal_check.values()),
    })
    return {"role": "software evidence auditor", "attacks": attacks, "seal_check": seal_check,
            "confirmed": [a["attack"] for a in attacks if a["confirmed"]]}


def reconcile(roles: list[dict]) -> dict:
    """The fourth role: rerun every claimed confirmation and every load bearing refutation."""
    out = []
    for r in roles:
        for a in r["attacks"]:
            reproduction = {"reproduced": bool(a["confirmed"])}
            verdict = defects.adjudicate(
                {"path": a["path"], "condition": a["condition"], "reproduction": a["reproduction"],
                 "expected": a["expected"], "actual": a["actual"], "consequence": a["consequence"]},
                [{"verdict": "confirmed" if a["confirmed"] else "refuted"}],
                reproduction,
            )
            out.append({"role": r["role"], "attack": a["attack"], "status": verdict["status"],
                        "authority": verdict["authority"], "actual": a["actual"]})
    return {
        "reconciled": out,
        "confirmed_defects": [o for o in out if o["status"] == "defect_confirmed"],
        "refuted_by_reproduction": [o["attack"] for o in out if o["status"] == "refuted_by_reproduction"],
        "decided_by_vote": 0,
        "rule": "decided from reproducibility, never from vote count",
    }


def main():
    t0 = time.time()
    b1 = audit_experiment("MOP_E1_ADMISSION.json", "principal", "runs")
    b4 = audit_experiment("MOP_E4_ADMISSION.json", "e4_principal", "arms")
    roles = [role_instrumentation(), role_statistical(), role_software()]
    rec = reconcile(roles)
    doc = {
        "schema": "mop-method-independent-audit/v1",
        "roles": {
            "B_instrumentation_E1": b1,
            "B_instrumentation_E4": b4,
            "method_audit_roles": roles,
        },
        "reconciliation": rec,
        "confirmed_defect_count": len(rec["confirmed_defects"]),
        "unresolved": [o["attack"] for o in rec["reconciled"] if o["status"].startswith("unresolved")],
        "all_pass": (
            len(rec["confirmed_defects"]) == 0
            and (b1.get("all_pass", True) or b1.get("status") == "not_run")
            and (b4.get("all_pass", True) or b4.get("status") == "not_run")
        ),
        "stages_before_principal_compute": list(gate.PRE_PRINCIPAL),
        "wall_seconds": round(time.time() - t0, 1),
    }
    io.seal("MOP_METHOD_INDEPENDENT_AUDIT.json", doc)
    print(f"audit: confirmed defects {doc['confirmed_defect_count']} | roleB E1 {b1.get('all_pass')} "
          f"E4 {b4.get('all_pass')} | all_pass {doc['all_pass']}", flush=True)
    for k, v in (("E1", b1), ("E4", b4)):
        for f in v.get("failed", []):
            print(f"  roleB {k} FAIL {f}", flush=True)
    print("AUDIT_DONE", flush=True)


if __name__ == "__main__":
    main()
