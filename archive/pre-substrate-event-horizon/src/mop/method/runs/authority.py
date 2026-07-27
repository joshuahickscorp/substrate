"""Freeze the inherited scientific and methodological boundary.

Nothing here reinterprets the Fast State Forge. It binds what that program concluded, binds the defects it
discovered about its own method, and hashes the artifacts those bindings came from so that a later claim
about what was inherited can be checked rather than trusted.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json

from mop.method import defects, graph, io

FORGE = io.ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1"
FORGE_BRANCH = "agent/mop-fast-state-plasticity-forge"
FORGE_HEAD = "cb2fb61"
FORGE_TERMINAL_SCIENCE = "471fa0e"
FORGE_TAG = "mop-fast-state-terminal"
FORGE_PR = 33

BOUND_CONCLUSIONS = {
    "all_principal_fast_state_forge_results": "null",
    "cross_domain_transfer": "null in both directions",
    "functional_reorganization": "null",
    "task_free_context": "null",
    "learned_plasticity": "no stable headroom",
    "activation": False,
}


def artifact_digest() -> dict:
    out = {}
    for p in sorted(FORGE.rglob("*")):
        if p.is_file():
            out[p.relative_to(FORGE).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    return out


def bind_from_synthesis() -> dict:
    """Read the sealed synthesis rather than restating it from memory."""
    s = json.loads((FORGE / "MOP_FAST_STATE_FORGE_SYNTHESIS.json").read_text())
    q = s["terminal_questions"]
    return {
        "cross_domain_verdict": q["20 did either direction of domain transfer pass"],
        "within_domain_verdict": q["within_domain"],
        "functional_reorganization": q["25 did functional reorganization add value"],
        "task_free_context": q["24 did task free context inference work"],
        "learned_gate_opened": q["15 did a learned plasticity gate open"],
        "architecture_selected": q["30 which architecture was selected"],
        "activation": s["activation"],
        "forbidden_claims": s["forbidden_claims"],
        "domain_validity": q["domain_validity"],
        "unresolved_frontier": q["43 what exact next frontier remains"],
        "self_reported_softening": q["44 were the principal beds actually temporal"],
    }


def main():
    inherited = bind_from_synthesis()
    digest = artifact_digest()

    authority = {
        "schema": "mop-method-reformation-start-authority/v1",
        "program_id": io.PROGRAM,
        "branch": "agent/mop-experimental-method-reformation",
        "worktree": str(io.ROOT),
        "run_root": f"runs/method/{io.PROGRAM}",
        "proof_root": f"proof/method/{io.PROGRAM}",
        "stop_switch": str(io.STOP),
        "inherited_from": {
            "branch": FORGE_BRANCH,
            "head": FORGE_HEAD,
            "terminal_science_commit": FORGE_TERMINAL_SCIENCE,
            "tag": FORGE_TAG,
            "draft_pr": FORGE_PR,
            "pr_status": "immutable, not merged, not reopened by this program",
        },
        "bound_scientific_conclusions": BOUND_CONCLUSIONS,
        "bound_from_sealed_synthesis": inherited,
        "immutability_rule": (
            "all prior scientific results remain immutable. Corrections are append only and preserve the "
            "original evidence. This program reaudits and never rewrites."
        ),
        "inherited_artifact_count": len(digest),
        "inherited_artifact_digest": digest,
        "governing_principle": "experiment on the experiment before experimenting on the substrate",
        "terminal_conditions": [
            "the experimental validity system is implemented",
            "every major historical defect class is reproduced and caught automatically",
            "all experiment arms are machine proven distinct",
            "all controls are proven to have their declared semantics",
            "baselines are proven converged before comparison",
            "a new information value experiment selection system is operational",
            "at least two new high information substrate experiments are designed through the new method",
            "every admitted experiment is executed to its terminal scientific boundary",
            "every result independently verifies",
            "all positives survive adversarial mutation",
            "all nulls become explicit design constraints",
            "the method updates itself from every discovered defect",
            "the next substrate frontier is selected from measured information gain",
            "no dependency ready work remains",
        ],
        "activation": False,
    }
    io.seal("MOP_METHOD_REFORMATION_START_AUTHORITY.json", authority)

    ledger = {
        "schema": "mop-historical-experiment-defect-ledger/v1",
        "substantiation_fields": list(defects.SUBSTANTIATION_FIELDS),
        "defects": defects.LEDGER,
        "n_defects": len(defects.LEDGER),
        "veto_rule": (
            "a reproduced defect is confirmed regardless of reviewer votes. Consensus is evidence, not proof. "
            "Votes alone can never refute an attack; only a failed reproduction can."
        ),
        "required_followups": defects.required_followups("*"),
        "coverage_rule": (
            "coverage misses remain visible. Test scopes may not be narrowed merely to claim compliance."
        ),
    }
    io.seal("MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.json", ledger)
    io.seal("MOP_EXPERIMENT_CAUSAL_GRAPH_SCHEMA.json", dict(graph.SCHEMA))

    rows = "\n".join(
        f"| {d['id']} | {d['title']} | {d['declared']} | {d['actual']} | `{d['detector']}` | {d['stage_caught']} |"
        for d in defects.LEDGER
    )
    io.seal_md(
        "MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.md",
        f"""# Historical experiment defect ledger

{len(defects.LEDGER)} defect classes discovered in the Fast State Plasticity Forge and its predecessors.
Every one is rebuilt as a live mutation in `mop.method.acceptance` and as a permanent regression test.

| id | defect | declared | actual | detector | caught at |
|---|---|---|---|---|---|
{rows}

## The veto rule

{ledger["veto_rule"]}

A confirmed defect requires all six followups: freeze the original result, add a permanent regression test,
open a bounded repair authority, produce the repaired result, write the consequence analysis, revalidate
every dependent artifact.
""",
    )

    io.seal_md(
        "MOP_METHOD_REFORMATION_START_AUTHORITY.md",
        f"""# Method reformation start authority

Program `{io.PROGRAM}` inherits from `{FORGE_BRANCH}` at `{FORGE_HEAD}`, terminal science commit
`{FORGE_TERMINAL_SCIENCE}`, tag `{FORGE_TAG}`, draft PR #{FORGE_PR}. That PR stays immutable.

## Bound scientific conclusions

| result | verdict |
|---|---|
| all principal fast state Forge results | null |
| cross domain transfer | null in both directions |
| functional reorganization | null |
| task free context | null |
| learned plasticity | no stable headroom |
| activation | false |

Read back from the sealed synthesis rather than restated: cross domain
`{inherited["cross_domain_verdict"]}`, within domain `{json.dumps(inherited["within_domain_verdict"])}`,
learned gate opened `{inherited["learned_gate_opened"]}`, architecture selected
`{inherited["architecture_selected"]}`.

## Bound methodological failures

{len(defects.LEDGER)} defect classes, listed in `MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.md`. The one this
program was named after: the order free control contained a Conv1d with kernel 5, so the temporal headroom
interpretation built on it was invalid.

The Forge itself recorded the softening defect in its own answer 44: it called beds carrying the sealed
verdict `invalid_no_temporal_headroom` marginal. That sentence is now a machine checked failure.

## Immutability

{authority["immutability_rule"]}

{len(digest)} inherited artifacts are hashed into the authority. Any later claim about what was inherited is
checkable against those digests.

## Governing principle

{authority["governing_principle"]}
""",
    )
    print(f"authority sealed: {len(digest)} inherited artifacts, {len(defects.LEDGER)} defect classes", flush=True)
    print("AUTHORITY_DONE", flush=True)


if __name__ == "__main__":
    main()
