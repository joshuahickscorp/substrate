"""Durable ledger for the salvage-and-next-evidence-campaign program.

Encodes the 7 phases, 17 final deliverables, the decision tree, the three canary lanes, the three
confirmation clusters, the admission battery, and the stop rules as tracked items, pulling the verified
Phase 1 salvage facts in. Emits MOP_EVIDENCE_CAMPAIGN_STATE.json + MOP_EVIDENCE_CAMPAIGN_LEDGER.md at the
salvage root. Resume from this; do not restart planning.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/Users/scammermike/Downloads/mop")
SAL = ROOT / "salvage"
REP = SAL / "reports"


def load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def item(id_, phase, title, status="pending", evidence=None, next_action=""):
    return {"id": id_, "phase": phase, "title": title, "status": status,
            "evidence": evidence or [], "next_action": next_action}


def main() -> int:
    forensic = load(REP / "MOP_STOPPED_RUN_FORENSIC.json")
    inv = load(REP / "MOP_SALVAGE_INVENTORY.json")
    vf = forensic.get("verified_scientific_facts", {})

    deliverables = [
        ("D1", "stopped-run forensic report", "complete", ["salvage/reports/MOP_STOPPED_RUN_FORENSIC.json"]),
        ("D2", "salvage inventory", "complete", ["salvage/reports/MOP_SALVAGE_INVENTORY.json"]),
        ("D3", "categorized-wave incremental verification", "pending", []),
        ("D4", "runtime failure root-cause report", "partial",
         ["salvage/reports/MOP_STOPPED_RUN_FORENSIC.json (root_cause_of_stop)"]),
        ("D5", "receipt-cache design and proof", "pending", []),
        ("D6", "parallel scheduler benchmarks", "pending", []),
        ("D7", "retry-escalation tests", "pending", []),
        ("D8", "notification-stall tests", "pending", []),
        ("D9", "P1R/U1/N1 admission results", "pending", []),
        ("D10", "canary results", "pending", []),
        ("D11", "three confirmation-cluster preregistrations", "pending", []),
        ("D12", "results + independent verification per executed cluster", "pending", []),
        ("D13", "construction-search decision", "pending", []),
        ("D14", "I1 dependency decision", "pending", []),
        ("D15", "exact resource and wall-time accounting", "pending", []),
        ("D16", "exact scientific claim boundary", "partial",
         ["salvage/reports/MOP_STOPPED_RUN_FORENSIC.json (claim_boundary)"]),
        ("D17", "exact next campaign, if any", "pending", []),
    ]
    checklist = [item(f"DELIV-{d}", "final", t, s, ev) for d, t, s, ev in deliverables]

    phases = [
        ("P1", "forensic closure and salvage", "complete",
         "run closed operator-partial; horizons independently seal-verified; salvage inventory sealed"),
        ("P2", "salvage the categorized wave (incremental resumable verifier)", "active",
         "build receipt-caching incremental verifier; complete wave verify/report from existing 57 receipts "
         "without full-chain replay or new heavy compute"),
        ("P3", "repair the runtime architecture", "pending",
         "work-conserving DAG scheduler + task-class profiles + receipt-cached verification + resumable "
         "verification + retry escalation (deferral->failure_hold->alert) + notifications"),
        ("P4", "P1R/U1/N1 admission and small canaries", "pending",
         "G1-P1R redesigned stability-plasticity, G1-U1 calibrated uncertainty, G1-N1 reducible novelty; "
         "each declares null/units/controls/SESOI/multiplicity/stop/claim; apply Mechanism Admission Battery"),
        ("P5", "three independent-confirmation clusters", "pending",
         "A: V1+K1+M1 (verify/repair/messaging); B: R1+P1R (memory/plasticity); C: A1+S1 (action/simulation); "
         "strongest simple controls; independent units"),
        ("P6", "increase the correct forms of breadth", "pending",
         "data/implementation/context/control breadth; not more seeds on same referents+implementation"),
        ("P7", "construction (G1-G1) then I1 integration, only after confirmation", "pending",
         "G1-G1 search only over confirmed survivors, cost charged; G1-I1 only if dependency closure survives"),
    ]
    for pid, title, status, na in phases:
        checklist.append(item(f"PHASE-{pid}", pid, title, status, [], na))

    # canary lanes
    for cid, title in [("P1R", "redesigned stability-plasticity"), ("U1", "calibrated uncertainty"),
                       ("N1", "reducible novelty")]:
        checklist.append(item(f"CANARY-{cid}", "P4", f"G1-{cid}: {title}", "pending", [],
                              "declare null/units/controls/SESOI/multiplicity/stop/claim; run admission battery"))

    # confirmation clusters
    clusters = [
        ("A", "V1+K1+M1", "does selective verification, contradiction repair, and bounded messaging improve "
                          "downstream decisions enough to justify charged computation"),
        ("B", "R1+P1R", "can the system improve retention and future adaptation jointly, not trade one for other"),
        ("C", "A1+S1", "does simulation improve action selection on independently sourced trajectories vs "
                       "reactive and simpler predictive controls"),
    ]
    for cid, lanes, q in clusters:
        checklist.append(item(f"CLUSTER-{cid}", "P5", f"Cluster {cid} ({lanes}): {q}", "pending", [],
                              "preregister null/units/controls/SESOI/multiplicity/stop/claim; run + verify"))

    # admission battery clauses
    for clause in ["WHAT estimator sufficiency", "oracle budget headroom", "WHEN-value decodability",
                   "value beyond simple heuristics", "group-disjoint validity", "architecture independence",
                   "noisy-TV control", "shuffled-target control", "wrong-time control",
                   "rate-matched-random control"]:
        checklist.append(item(f"BATTERY-{clause[:18].replace(' ','_')}", "P4",
                              f"admission battery: {clause}", "pending", [], "gate before heavy compute"))

    # stop rules (as standing invariants)
    for sr in ["admission battery fails", "point estimate wrong direction", "SESOI unreachable",
               "strongest control wins", "independent units contradict seed-level result",
               "architecture dependence detected", "reproduction fails", "futility boundary crossed",
               "required dependency fails"]:
        checklist.append(item(f"STOP-{sr[:20].replace(' ','_')}", "stop_rules", f"stop when: {sr}", "active",
                              [], "enforce on every branch; do not continue merely because compute remains"))

    from collections import Counter
    by_status = dict(Counter(i["status"] for i in checklist))

    state = {
        "schema": "mop-evidence-campaign-state/v1",
        "mandate": "SALVAGE THE GENERAL RUN AND BUILD THE NEXT EVIDENCE CAMPAIGN",
        "branch": "agent/mop-evidence-salvage",
        "central_question": ("does any surviving mechanism produce independently reproducible value on "
                             "genuinely different data, tasks, contexts, and implementations, after charging "
                             "its full computation?"),
        "hard_constraints": [
            "do not restart the stopped General Run unchanged",
            "do not run the retired frozen D1 design again",
            "do not add another same-code robustness horizon",
            "do not increase mechanism count unnecessarily; increase evidence breadth",
            "do not rewrite historical files to appear complete",
            "do not recompute categorized-wave capsules whose exact receipts already exist",
            "do not promise a wall time before benchmarking; 6-10h is a hypothesis to test",
            "run small canaries first; do not auto-authorize 14 epochs / full waves / integration",
        ],
        "verified_salvage_facts": vf,
        "salvage_categories": {
            "terminal_verified_evidence": "Horizon 1 + Horizon 2 (10 epochs, 20704 rungs, all seals verified)",
            "partial_but_valid_receipts": (inv.get("partial_but_valid_receipts") or {}).get(
                "categorized_wave_complete_capsules"),
            "incomplete_capsules": ["categorized_wave_verify (stuck)", "categorized_wave_report (pending)",
                                    "full_generations_wave (never started)"],
        },
        "decision_tree": ["salvage categorized artifacts", "repair scheduler/verification/retry/notification",
                          "P1R/U1/N1 admission and canaries", "three independent-confirmation clusters",
                          "construction search over confirmed survivors only",
                          "I1 integration only if dependency closure survives"],
        "checklist_summary": {"total": len(checklist), "by_status": by_status},
        "checklist": checklist,
    }
    (SAL / "MOP_EVIDENCE_CAMPAIGN_STATE.json").write_text(json.dumps(state, indent=2))

    lines = ["# MOP Evidence Campaign Ledger", "",
             "Durable ledger for SALVAGE THE GENERAL RUN AND BUILD THE NEXT EVIDENCE CAMPAIGN. Machine "
             "authority: `MOP_EVIDENCE_CAMPAIGN_STATE.json`. Resume from here.", "",
             "## Central question", "", "> " + state["central_question"], "",
             "## Verified salvage facts (Phase 1, independently seal-checked)", ""]
    for k in ("fresh_seed_robustness_epochs", "executed_mechanics_rungs_total", "surviving_mechanics_lanes",
              "pruned_lanes", "claim_boundary"):
        lines.append(f"- {k}: {vf.get(k)}")
    lines += ["", "## Phases", "", "| phase | status | title |", "|---|---|---|"]
    for pid, title, status, _ in phases:
        lines.append(f"| {pid} | {status} | {title} |")
    lines += ["", "## Final deliverables", "", "| id | status | deliverable |", "|---|---|---|"]
    for d, t, s, _ in deliverables:
        lines.append(f"| {d} | {s} | {t} |")
    lines += ["", f"Checklist: {len(checklist)} items, by status: `{json.dumps(by_status)}`.", ""]
    (SAL / "MOP_EVIDENCE_CAMPAIGN_LEDGER.md").write_text("\n".join(lines) + "\n")

    print(f"checklist items: {len(checklist)} | by_status: {json.dumps(by_status)}")
    print("wrote MOP_EVIDENCE_CAMPAIGN_STATE.json + MOP_EVIDENCE_CAMPAIGN_LEDGER.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
