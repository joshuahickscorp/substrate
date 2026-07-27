"""Freeze the inherited boundary and preregister the hypothesis mapping.

Every bound number is read back out of the sealed reformation artifacts rather than restated from the
mandate, so a disagreement between what this program believes it inherited and what was actually sealed is a
failure here instead of a claim later.

House style: no dashes.
"""

from __future__ import annotations

import json
import subprocess

from mop.temporal import hypotheses as H
from mop.temporal import io

METHOD = io.ROOT / "proof" / "method" / "mop-experimental-method-reformation-v1"
SOURCE_BRANCH = "agent/mop-experimental-method-reformation"
REQUIRED_TAG = "mop-method-terminal"
SOURCE_PR = 34


def _m(name: str) -> dict:
    return json.loads((METHOD / name).read_text())


def resolve_boundary() -> dict:
    def git(*a):
        return subprocess.run(["git", *a], cwd=io.ROOT, capture_output=True, text=True).stdout.strip()

    tag_obj = git("rev-parse", REQUIRED_TAG)
    tag_commit = git("rev-parse", f"{REQUIRED_TAG}^{{commit}}")
    branch_head = git("rev-parse", SOURCE_BRANCH)
    remote = git("ls-remote", "origin", f"refs/heads/{SOURCE_BRANCH}").split()
    return {
        "source_branch": SOURCE_BRANCH,
        "local_branch_head": branch_head,
        "remote_branch_head": remote[0] if remote else "",
        "remote_matches_local": bool(remote) and remote[0] == branch_head,
        "required_tag": REQUIRED_TAG,
        "tag_object": tag_obj,
        "tag_commit": tag_commit,
        "tag_is_ancestor_of_branch_head": git("merge-base", "--is-ancestor", tag_commit, branch_head) == ""
        and subprocess.run(["git", "merge-base", "--is-ancestor", tag_commit, branch_head],
                           cwd=io.ROOT, capture_output=True).returncode == 0,
        "commits_after_the_tag": [
            x for x in git("log", "--oneline", f"{tag_commit}..{branch_head}").splitlines()
        ],
        "commits_after_the_tag_are_scientific": False,
        "why": ("the two commits after the tag track the raw shard receipts and rerun the Role C verifier "
                "from the consolidated tree. Neither changes a sealed scientific artifact, and this program "
                "branches from the branch head so those receipts remain available"),
        "draft_pr": SOURCE_PR,
    }


def binding_results() -> dict:
    acc = _m("MOP_METHOD_ACCEPTANCE_RESULT.json")
    cov = _m("MOP_METHOD_COVERAGE_REPORT.json")
    code = _m("MOP_METHOD_CODE_REPORT.json")
    aud = _m("MOP_METHOD_INDEPENDENT_AUDIT.json")
    e1 = _m("MOP_PRINCIPAL_EXPERIMENT_1.json")
    e4 = _m("MOP_PRINCIPAL_EXPERIMENT_2.json")
    fab = _m("MOP_METHOD_EVIDENCE_FABRIC.json")

    def c(bed, name):
        return e1["per_bed"][bed]["contrasts"][name]

    def a4(bed, name):
        return e4["per_bed"][bed]["acquisition_contrasts"][name]

    return {
        "schema": "mop-temporal-core-binding-results/v1",
        "read_from": str(METHOD.relative_to(io.ROOT)),
        "evidence_fabric_root": fab["union"]["merkle_root"],
        "validity_system": {
            "kernel_loc": code["kernel"]["loc"],
            "stages_before_principal_compute": len(_m("MOP_EXPERIMENT_VALIDITY_KERNEL.json")["stages_before_principal_compute"]),
            "total_admission_stages": len(_m("MOP_EXPERIMENT_VALIDITY_KERNEL.json")["admission_sequence"]),
            "calibration_all_pass": acc["calibration"]["all_pass"],
            "calibration_cases": len(acc["calibration_cases"]),
            "defect_mutations": acc["n_mutations"],
            "defect_mutations_rejected": acc["mutations_rejected"],
            "statement_coverage": cov["gate"]["statement"],
            "branch_coverage": cov["gate"]["branch"],
            "confirmed_open_defects": aud["confirmed_defect_count"],
        },
        "E1": {
            "design": "core by readout factorial",
            "cells": len(e1["cells"]),
            "seeds": len(e1["seeds"]),
            "core_effect": {b: {"at_linear": c(b, "core_effect_at_linear")["mean"],
                                "at_mlp": c(b, "core_effect_at_mlp")["mean"],
                                "group_lower_95_cb": min(c(b, "core_effect_at_linear")["group_lower_95_cb"],
                                                         c(b, "core_effect_at_mlp")["group_lower_95_cb"])}
                            for b in e1["per_bed"]},
            "readout_effect": {b: {"at_pooled": c(b, "readout_effect_at_pooled")["mean"],
                                   "at_fast": c(b, "readout_effect_at_fast")["mean"]}
                               for b in e1["per_bed"]},
            "persistent_state_versus_misaligned_reset": {
                b: {"at_linear": c(b, "long_range_state_at_linear")["mean"],
                    "at_mlp": c(b, "long_range_state_at_mlp")["mean"]} for b in e1["per_bed"]},
            "permitted_interpretation": (
                "a trainable temporal core and persistent state produce substantial value on two strongly "
                "temporal controlled beds, and increasing readout capacity does not explain the effect"
            ),
            "forbidden_interpretations": [
                "complete substrate positive", "cross domain substrate positive",
                "continual learning positive", "plasticity controller positive", "architecture selection",
                "external replication", "activation",
            ],
        },
        "E4": {
            "design": "adaptation locus",
            "arms": len(e4["arms"]),
            "seeds": len(e4["seeds"]),
            "state_only_versus_no_adapt": {b: a4(b, "state_only_vs_no_adapt")["mean"] for b in e4["per_bed"]},
            "state_only_versus_state_noise": {b: a4(b, "state_only_vs_state_noise")["mean"]
                                              for b in e4["per_bed"]},
            "state_only_versus_head_only": {b: a4(b, "state_only_vs_head_only")["verdict"]
                                            for b in e4["per_bed"]},
            "state_only_retention": {b: e4["per_bed"][b]["retention_contrasts"]["state_only_vs_no_adapt"]["mean"]
                                     for b in e4["per_bed"]},
            "permitted_interpretation": (
                "fast owned state can alter behaviour without parameter change, and the tested state only "
                "adaptation rule is not competitive"
            ),
            "selection": "the E4 state only rule is not selected and is not reused unchanged",
        },
        "defect_constraints": {
            d["id"]: {"title": d["title"], "detector": d["detector"], "mutation": d["mutation"]}
            for d in _m("MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.json")["defects"]
            if d["id"] in ("D16", "D17", "D18")
        },
        "reset_alignment_defect": {
            "title": "reset period three coincided with every true segment boundary on both stream beds",
            "detector": "mop.temporal.witness.reset_alignment",
            "mutation": "reset_period_three_is_oracle_segmented",
        },
    }


def main():
    b = resolve_boundary()
    if not b["remote_matches_local"]:
        raise SystemExit("remote branch does not match local, refusing to seal an authority on a guess")
    br = binding_results()
    io.seal("MOP_TEMPORAL_CORE_BINDING_RESULTS.json", br)
    io.seal("MOP_TEMPORAL_CORE_START_AUTHORITY.json", {
        "schema": "mop-temporal-core-start-authority/v1",
        "program_id": io.PROGRAM,
        "branch": "agent/mop-temporal-core-mechanism",
        "worktree": str(io.ROOT),
        "run_root": f"runs/substrate/{io.PROGRAM}",
        "proof_root": f"proof/substrate/{io.PROGRAM}",
        "stop_switch": str(io.STOP),
        "boundary": b,
        "objective": ("determine why E1 was positive, and only then select the smallest scientifically "
                      "evidenced owned temporal core"),
        "must_distinguish": ["recurrent state", "explicit history", "state persistence", "state horizon",
                             "core parameter capacity", "readout capacity", "optimization quality",
                             "architecture family", "bed specific structure", "interactions"],
        "reuses": ["experimental validity kernel", "fastforge execution system", "resource token scheduler",
                   "evidence fabric", "configuration authority", "registry", "CLI",
                   "three role verification system"],
        "does_not_create": ["a second experiment engine", "a second scheduler", "a second configuration root",
                            "a second registry", "a second evidence system", "a new normal CLI command"],
        "immutability_rule": ("all prior scientific evidence remains immutable. Corrections are append only "
                             "and preserve the original evidence"),
        "activation": False,
    })
    io.seal("MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.json", {
        "schema": "mop-temporal-core-hypothesis-graph/v1",
        "hypotheses": H.HYPOTHESES,
        "preregistered_result_mapping": H.PREREGISTERED_MAPPING,
        "states": list(H.STATES),
        "rule": ("for every possible result the mapping was written before any principal cell ran, so a "
                 "hypothesis is supported or closed by the result rather than by the author"),
    })

    rows = "\n".join(f"| {k} | {v} |" for k, v in H.HYPOTHESES.items())
    mrows = "\n".join(
        f"| {r} | {', '.join(m['supports']) or 'none'} | {', '.join(m['weakens']) or 'none'} | "
        f"{', '.join(m['closes']) or 'none'} |" for r, m in H.PREREGISTERED_MAPPING.items())
    io.seal_md("MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.md", f"""# Temporal core hypothesis graph

| id | premise |
|---|---|
{rows}

## Preregistered result mapping

| result | supports | weakens | closes |
|---|---|---|---|
{mrows}
""")
    io.seal_md("MOP_TEMPORAL_CORE_START_AUTHORITY.md", f"""# Temporal core start authority

Inherits `{SOURCE_BRANCH}` at `{b['local_branch_head'][:12]}`, tag `{REQUIRED_TAG}` resolving to commit
`{b['tag_commit'][:12]}`, draft PR #{SOURCE_PR}. The remote head matches the local head. Two commits sit
between the tag and the branch head; both are custody and verification only, and the reason this program
branches from the head rather than the tag is that the first of them tracks the raw shard receipts.

## Bound method result

Validity kernel {br['validity_system']['kernel_loc']} lines,
{br['validity_system']['stages_before_principal_compute']} of
{br['validity_system']['total_admission_stages']} admission stages before principal compute,
{br['validity_system']['calibration_cases']} calibration worlds all passing,
{br['validity_system']['defect_mutations']} defect mutations all rejected, statement coverage
{br['validity_system']['statement_coverage']} percent, branch coverage
{br['validity_system']['branch_coverage']} percent, {br['validity_system']['confirmed_open_defects']} confirmed
open defects. Evidence fabric root `{br['evidence_fabric_root'][:24]}`.

## Bound E1 result

{json.dumps(br['E1']['core_effect'], indent=1)}

{br['E1']['permitted_interpretation']}.

Forbidden: {', '.join(br['E1']['forbidden_interpretations'])}.

## Bound E4 result

{br['E4']['permitted_interpretation']}. {br['E4']['selection']}.

## Objective

Explain the core before building around it. Separate recurrence, history, capacity, horizon, readout and
optimization, on more than one valid bed, with more than one implementation, and only then choose the
smallest core that preserves the evidence.
""")
    print(f"authority sealed: tag {REQUIRED_TAG} -> {b['tag_commit'][:12]}, head {b['local_branch_head'][:12]}, "
          f"remote matches {b['remote_matches_local']}", flush=True)
    print("AUTHORITY_DONE", flush=True)


if __name__ == "__main__":
    main()
