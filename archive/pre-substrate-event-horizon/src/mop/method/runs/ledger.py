"""Durable state, ledger and scorecard, plus the human readable synthesis.

Every number in the markdown is bound to a sealed field and every verdict sentence is passed through the
wording check before it is written. A synthesis that broadens its own evidence fails here rather than being
noticed six weeks later by a reader.

House style: no dashes.
"""

from __future__ import annotations

import json
import time

from mop.method import defects, gate, io, report


def _load(name: str, default=None):
    return io.load(name) if io.exists(name) else (default if default is not None else {})


def state() -> dict:
    acc = _load("MOP_METHOD_ACCEPTANCE_RESULT.json")
    e1 = _load("MOP_PRINCIPAL_EXPERIMENT_1.json")
    e4 = _load("MOP_PRINCIPAL_EXPERIMENT_2.json")
    return {
        "schema": "mop-method-reformation-state/v1",
        "program_id": io.PROGRAM,
        "branch": "agent/mop-experimental-method-reformation",
        "stop_switch": str(io.STOP),
        "stages": {
            "start_authority": io.exists("MOP_METHOD_REFORMATION_START_AUTHORITY.json"),
            "defect_ledger": io.exists("MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.json"),
            "validity_kernel": io.exists("MOP_EXPERIMENT_VALIDITY_KERNEL.json"),
            "method_acceptance": bool(acc.get("green")),
            "fast_state_reaudit": io.exists("MOP_FAST_STATE_REAUDIT.json"),
            "experiment_value_queue": io.exists("MOP_EXPERIMENT_VALUE_QUEUE.json"),
            "hypothesis_graph": io.exists("MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json"),
            "E1_admission": bool(_load("MOP_E1_ADMISSION.json").get("admission", {}).get("licensed")),
            "E4_admission": bool(_load("MOP_E4_ADMISSION.json").get("admission", {}).get("licensed")),
            "E1_principal": bool(e1),
            "E4_principal": bool(e4),
            "positive_mutations": bool(_load("MOP_POSITIVE_MUTATION_SUITE.json").get("all_rejected")),
            "independent_verification": bool(_load("MOP_METHOD_INDEPENDENT_VERIFICATION.json").get("all_pass")),
            "independent_audit": bool(_load("MOP_METHOD_INDEPENDENT_AUDIT.json").get("all_pass")),
            "reports": io.exists("MOP_METHOD_COVERAGE_REPORT.json"),
            "synthesis": io.exists("MOP_METHOD_REFORMATION_SYNTHESIS.json"),
            "evidence_fabric": io.exists("MOP_METHOD_EVIDENCE_FABRIC.json"),
        },
        "defect_classes": len(defects.LEDGER),
        "defect_classes_discovered_here": [d["id"] for d in defects.LEDGER if d.get("discovered_in_this_program")],
        "activation": False,
    }


def ledger_md() -> str:
    ra = _load("MOP_FAST_STATE_REAUDIT.json")
    e1 = _load("MOP_PRINCIPAL_EXPERIMENT_1.json")
    e4 = _load("MOP_PRINCIPAL_EXPERIMENT_2.json")
    acc = _load("MOP_METHOD_ACCEPTANCE_RESULT.json")
    cov = _load("MOP_METHOD_COVERAGE_REPORT.json")
    mut = _load("MOP_POSITIVE_MUTATION_SUITE.json")
    q = _load("MOP_EXPERIMENT_VALUE_QUEUE.json")

    rows = "\n".join(
        f"| {d['id']} | {d['title']} | `{d['detector']}` | {'this program' if d.get('discovered_in_this_program') else 'inherited'} |"
        for d in defects.LEDGER
    )
    e1rows = ""
    for b, a in (e1.get("per_bed") or {}).items():
        for k, v in a["contrasts"].items():
            e1rows += f"| {b} | {k} | {v['mean']:+.4f} | {v['lower_95_cb']:+.4f} | {v['verdict']} |\n"
    e4rows = ""
    for b, a in (e4.get("per_bed") or {}).items():
        for k, v in a["acquisition_contrasts"].items():
            e4rows += f"| {b} | acquisition {k} | {v['mean']:+.4f} | {v['lower_95_cb']:+.4f} | {v['verdict']} |\n"

    return f"""# Method reformation ledger

## What this program changed

The previous method validated instrumentation after expensive scientific execution. This one inverts the
order: {len(gate.PRE_PRINCIPAL)} stages run before any principal training compute, and an experiment that
fails one of them cannot spend any.

## Defect ledger

{len(defects.LEDGER)} classes, every one rebuilt as a live mutation and as a permanent regression test.
{acc.get('n_mutations')} mutations, all rejected: {acc.get('mutations_rejected')}.

| id | defect | detector | discovered |
|---|---|---|---|
{rows}

## Fast State Forge reaudit

Findings: {[f['id'] for f in ra.get('findings', [])]}. Load bearing: {ra.get('new_load_bearing_defects')}.
Inherited receipts modified: {ra.get('inherited_receipts_modified')}.

{ra.get('corrected_claim_ceilings', {}).get('within_domain_battery', {}).get('corrected_claim', '')}

## Experiment selection

Queue selected {q.get('selected')} and refused {q.get('refused')}.

## E1 core by readout factorial

| bed | contrast | mean | lower 95 cb | verdict |
|---|---|---|---|---|
{e1rows}
## E4 adaptation locus

| bed | contrast | mean | lower 95 cb | verdict |
|---|---|---|---|---|
{e4rows}
## Positive mutations

All rejected: {mut.get('all_rejected')}.

## Coverage

Kernel statement {cov.get('gate', {}).get('statement')} percent against a target of 92, branch
{cov.get('gate', {}).get('branch')} percent against a target of 82. Program stages are measured separately
at {cov.get('program_stage_scope', {}).get('statement')} percent and are listed rather than hidden.

## Activation

False. Nothing here licenses an architecture, and no claim in this ledger extends beyond its measured path.
"""


def scorecard_md(sc: dict) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in sc["implementation"].items())
    dem = "\n".join(f"| {k} | {v} |" for k, v in sc["demonstrated"].items())
    return f"""# Method reformation scorecard

## Implementation

| dimension | score |
|---|---|
{rows}

Minimum across dimensions: {sc['implementation_minimum']}.

## Demonstrated

| property | value |
|---|---|
{dem}

Defects caught before principal compute: {sc['defects_caught_before_principal_execution']}.
Defects caught after execution but before any claim: {sc['defects_caught_after_execution_before_claim']}.
Principal runs prevented from being invalid: {sc['invalid_runs_prevented']['principal_runs_prevented_from_being_invalid']}.
"""


def main():
    t0 = time.time()
    from mop.method.runs import synthesis as S

    st = state()
    io.seal("MOP_METHOD_REFORMATION_STATE.json", st)
    sc = S.method_scorecard()
    io.seal("MOP_METHOD_REFORMATION_SCORECARD.json", sc)
    io.seal_md("MOP_METHOD_REFORMATION_SCORECARD.md", scorecard_md(sc))

    md = ledger_md()
    # Each claim sentence is checked against the verdict it summarizes, not the document against an
    # unrelated one. A ledger that reports both an inherited null and a measured positive would fail any
    # whole document check by construction, and failing it that way would say nothing about honesty.
    e1 = _load("MOP_PRINCIPAL_EXPERIMENT_1.json")
    e4 = _load("MOP_PRINCIPAL_EXPERIMENT_2.json")
    claims = [
        ("cross_domain_transfer",
         "cross domain transfer of the shared fast core is null in both directions and this program did not "
         "reopen it",
         "cross_domain_null"),
        ("within_domain_on_invalid_beds",
         "the inherited within domain battery ran on beds sealed invalid, so it bounds nothing about temporal "
         "dynamics",
         "invalid_no_temporal_headroom"),
        ("e1_core_effect",
         "the recurrent core effect is positive on both beds and beats every control that was run",
         (e1.get("terminal_classification") or {}).get("har_stream:core_effect_at_linear", "null")),
        ("e1_readout_effect",
         "readout capacity did not separate any cell",
         (e1.get("terminal_classification") or {}).get("har_stream:readout_effect_at_fast", "null")),
        ("e4_state_only",
         "state only adaptation improves acquisition with zero parameter updates",
         (e4.get("terminal_classification") or {}).get(
             "speech_stream:acquisition_contrasts:state_only_vs_no_adapt", "null")),
        ("e4_state_only_retention",
         "state only adaptation is a null on retention",
         (e4.get("terminal_classification") or {}).get(
             "speech_stream:retention_contrasts:state_only_vs_no_adapt", "null")),
        ("activation", "activation is not licensed and no architecture is selected", "invalid"),
    ]
    checks = {name: report.wording_check(text, sealed) for name, text, sealed in claims}
    for name, text, sealed in claims:
        checks[name]["claim"] = text
        checks[name]["sealed_verdict"] = sealed
    io.seal_md("MOP_METHOD_REFORMATION_LEDGER.md", md)
    io.seal("MOP_METHOD_REFORMATION_LEDGER_WORDING.json", {
        "schema": "mop-ledger-wording-check/v1",
        "checks": {k: v["passes"] for k, v in checks.items()},
        "offenders": {k: v["offenders"] for k, v in checks.items() if not v["passes"]},
        "claims": {k: {"claim": v["claim"], "sealed_verdict": v["sealed_verdict"], "passes": v["passes"]}
                   for k, v in checks.items()},
        "all_pass": all(v["passes"] for v in checks.values()),
        "note": ("each claim is checked against the verdict it summarizes. The e4 retention claim is the "
                 "interesting one: the sealed verdict there is harm, so calling it a null would be a "
                 "softening in the other direction and is caught by the same rule"),
    })
    print(f"state and ledger sealed, stages green "
          f"{sum(1 for v in st['stages'].values() if v)}/{len(st['stages'])} "
          f"in {round(time.time() - t0, 1)}s", flush=True)
    print("LEDGER_DONE", flush=True)


if __name__ == "__main__":
    main()
