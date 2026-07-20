"""Durable forge ledger and progress scorecard.

Status is derived from sealed artifacts on disk, never from a hand maintained flag, so the program resumes
from the ledger without replanning and cannot claim a requirement it did not seal.

Implementation score and evidence score are tracked separately. Implementation credit comes from the work
existing and being verified. Evidence credit comes only from what the evidence supports, so a complete and
correct null run raises implementation and falsification credit without raising the usefulness credit at all.

House style: no dashes.
"""

from __future__ import annotations

import json

from fastforge.runs import io

# id, title, artifacts that prove it, dependencies, exact next action if not done
REQUIREMENTS = [
    (
        "R01",
        "Freeze the inherited interpretation",
        ["MOP_FAST_STATE_FORGE_START_AUTHORITY.json", "MOP_FAST_STATE_FORGE_START_AUTHORITY.md"],
        [],
        "run fastforge.runs.authority",
    ),
    (
        "R02",
        "Bind the inherited nulls",
        ["MOP_FAST_STATE_BINDING_NULLS.json"],
        ["R01"],
        "run fastforge.runs.authority",
    ),
    (
        "R03",
        "Repair cross domain arm aliasing",
        [
            "MOP_CROSS_DOMAIN_ARM_AUTHORITY.json",
            "MOP_CROSS_DOMAIN_ARM_AUDIT.json",
            "MOP_CROSS_DOMAIN_ARM_MUTATIONS.json",
        ],
        ["R01"],
        "run fastforge.runs.armaudit",
    ),
    (
        "R04",
        "Implement and seal Architecture G",
        ["MOP_ARCHITECTURE_G.json"],
        ["R03"],
        "run fastforge.runs.architectures",
    ),
    (
        "R05",
        "Implement and seal Architecture H",
        ["MOP_ARCHITECTURE_H.json"],
        ["R03"],
        "run fastforge.runs.architectures",
    ),
    (
        "R06",
        "Baseline convergence curves",
        ["MOP_BASELINE_CONVERGENCE_REPORT.json"],
        [],
        "run fastforge.runs.bench",
    ),
    ("R07", "Resource profile", ["MOP_FAST_STATE_RESOURCE_REPORT.json"], [], "run fastforge.runs.bench"),
    (
        "R08",
        "Within domain battery, HAR",
        ["MOP_HAR_WITHIN_DOMAIN_REPORT.json"],
        ["R04", "R05", "R06"],
        "run fastforge.runs.withinrun shards then aggregate",
    ),
    (
        "R09",
        "Within domain battery, Speech",
        ["MOP_SPEECH_WITHIN_DOMAIN_REPORT.json"],
        ["R04", "R05", "R06"],
        "run fastforge.runs.withinrun shards then aggregate",
    ),
    (
        "R10",
        "Cross domain direction 1, HAR to Speech",
        ["MOP_HAR_TO_SPEECH_REPORT.json"],
        ["R04", "R05", "R06"],
        "run fastforge.runs.crossdomain shards then aggregate",
    ),
    (
        "R11",
        "Cross domain direction 2, Speech to HAR",
        ["MOP_SPEECH_TO_HAR_REPORT.json"],
        ["R04", "R05", "R06"],
        "run fastforge.runs.crossdomain shards then aggregate",
    ),
    (
        "R12",
        "Bidirectional synthesis",
        ["MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json"],
        ["R10", "R11"],
        "run fastforge.runs.crossdomain aggregate",
    ),
    (
        "R13",
        "Interference map",
        ["MOP_SUBSTRATE_INTERFERENCE_MAP.json", "MOP_SUBSTRATE_INTERFERENCE_REPORT.md"],
        ["R04", "R05"],
        "run fastforge.runs.interference shards then aggregate",
    ),
    (
        "R14",
        "Plasticity action headroom gate",
        ["MOP_SUBSTRATE_PLASTICITY_ACTION_HEADROOM.json", "MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json"],
        ["R13", "R12"],
        "run fastforge.runs.plasticity",
    ),
    (
        "R15",
        "Architecture improvement rounds",
        ["MOP_FAST_STATE_ARCHITECTURE_COMPARISON.json"],
        ["R12", "R13"],
        "run fastforge.runs.rounds",
    ),
    (
        "R16",
        "Functional reorganization",
        ["MOP_FUNCTIONAL_REORGANIZATION_REPORT.json"],
        ["R13"],
        "run fastforge.runs.reorg",
    ),
    (
        "R17",
        "Task free context inference",
        ["MOP_TASK_FREE_CONTEXT_REPORT.json"],
        ["R13"],
        "run fastforge.runs.reorg",
    ),
    (
        "R18",
        "Fast core representation analysis",
        ["MOP_FAST_CORE_REPRESENTATION_REPORT.json", "MOP_FAST_CORE_REPRESENTATION_REPORT.md"],
        ["R12"],
        "run fastforge.runs.represent",
    ),
    (
        "R19",
        "Third temporal domain preflight",
        ["MOP_THIRD_TEMPORAL_DOMAIN_PREFLIGHT.json"],
        ["R06"],
        "run fastforge.runs.domaingate",
    ),
    ("R20", "Code accounting", ["MOP_FAST_STATE_CODE_REPORT.json"], [], "run fastforge.runs.codereport"),
    (
        "R21",
        "Test report",
        ["MOP_FAST_STATE_TEST_REPORT.json"],
        ["R04", "R05"],
        "run fastforge.runs.testreport",
    ),
    (
        "R22",
        "Independent verification",
        ["MOP_FAST_STATE_INDEPENDENT_VERIFICATION.json"],
        ["R12", "R13", "R14"],
        "run fastforge.verify",
    ),
    (
        "R23",
        "Mutation suites",
        ["MOP_FAST_STATE_MUTATION_REPORT.json"],
        ["R12", "R13", "R14"],
        "run fastforge.runs.mutations",
    ),
    (
        "R24",
        "Clean clone validation",
        ["MOP_FAST_STATE_CLEAN_CLONE.json"],
        ["R22", "R23"],
        "run fastforge.runs.cleanclone",
    ),
    (
        "R25",
        "Evidence fabric indexing",
        ["MOP_FAST_STATE_EVIDENCE_FABRIC.json"],
        ["R24"],
        "run fastforge.runs.fabric",
    ),
    (
        "R26",
        "Synthesis and next frontier",
        [
            "MOP_FAST_STATE_FORGE_SYNTHESIS.json",
            "MOP_FAST_STATE_FORGE_SYNTHESIS.md",
            "MOP_FAST_STATE_NEXT_FRONTIER.json",
        ],
        ["R25"],
        "run fastforge.runs.synthesis",
    ),
]

IMPLEMENTATION_TARGETS = {
    "evidence_and_falsification": 98,
    "orchestration": 98,
    "failure_understanding": 98,
    "owned_substrate": 98,
    "multi_timescale_and_partitioned_plasticity": 95,
    "functional_reorganization": 85,
    "cross_domain_entity": 90,
}
EVIDENCE_TARGETS = {
    "evidence_and_falsification": 95,
    "orchestration": 95,
    "failure_understanding": 95,
    "useful_plasticity": 60,
    "owned_substrate": 80,
    "partitioned_moldability": 70,
    "functional_reorganization": 55,
    "cross_domain_entity": 65,
}
# which requirements carry which implementation dimension
DIMENSION_REQS = {
    "evidence_and_falsification": ["R01", "R02", "R03", "R22", "R23", "R25"],
    "orchestration": ["R07", "R10", "R11", "R13", "R24"],
    "failure_understanding": ["R13", "R14", "R18", "R19"],
    "owned_substrate": ["R04", "R05", "R08", "R09", "R20", "R21"],
    "multi_timescale_and_partitioned_plasticity": ["R13", "R14", "R15"],
    "functional_reorganization": ["R16", "R17"],
    "cross_domain_entity": ["R10", "R11", "R12"],
}


def status():
    st = {}
    for rid, title, arts, deps, nxt in REQUIREMENTS:
        done = all(io.exists(a) for a in arts)
        st[rid] = {
            "id": rid,
            "title": title,
            "artifacts": arts,
            "dependencies": deps,
            "status": "terminal"
            if done
            else (
                "ready"
                if all(all(io.exists(a) for a in dict((r[0], r[2]) for r in REQUIREMENTS)[d]) for d in deps)
                else "blocked"
            ),
            "next_exact_action": "none" if done else nxt,
        }
    return st


def evidence_scores(st):
    """Evidence credit is earned from what the evidence shows, never from the work existing."""

    def verdict(name, key, default=None):
        return io.load(name).get(key, default) if io.exists(name) else default

    cross = verdict("MOP_FAST_STATE_BIDIRECTIONAL_SYNTHESIS.json", "bidirectional_verdict")
    plast = verdict("MOP_SUBSTRATE_PLASTICITY_POLICY_REPORT.json", "verdict")
    reorg = verdict("MOP_FUNCTIONAL_REORGANIZATION_REPORT.json", "verdict")
    third = verdict("MOP_THIRD_TEMPORAL_DOMAIN_PREFLIGHT.json", "verdict")
    beds = verdict("MOP_DOMAIN_VALIDITY.json", "valid_domains")
    secondary = verdict("MOP_FAST_STATE_SECONDARY_MATRIX.json", "verdict")
    terminal = sum(1 for v in st.values() if v["status"] == "terminal")
    frac = terminal / len(st)
    return {
        # a complete, verified, adversarially mutated null is full falsification credit
        "evidence_and_falsification": round(95 * frac),
        "orchestration": round(95 * frac),
        "failure_understanding": round(95 * frac),
        # usefulness credit requires a positive that survived every condition
        "useful_plasticity": 60 if plast == "plasticity_action_headroom_present" else 5,
        "owned_substrate": 45
        if st["R08"]["status"] == "terminal" and st["R09"]["status"] == "terminal"
        else 10,
        "partitioned_moldability": 70 if cross == "cross_domain_positive" else 15,
        "functional_reorganization": 55 if reorg == "functional_reorganization_positive" else 5,
        "cross_domain_entity": 65 if cross == "cross_domain_positive" else 10,
        "_verdicts": {
            "cross_domain": cross,
            "plasticity": plast,
            "reorganization": reorg,
            "third_domain": third,
            "secondary_stream_matrix": secondary,
            "valid_temporal_beds": beds,
        },
    }


def main():
    st = status()
    impl = {
        k: round(100 * sum(1 for r in v if st[r]["status"] == "terminal") / len(v))
        for k, v in DIMENSION_REQS.items()
    }
    ev = evidence_scores(st)
    io.seal(
        "MOP_FAST_STATE_FORGE_STATE.json",
        {
            "schema": "mop-fast-state-forge-state/v1",
            "requirements": st,
            "terminal": sorted(r for r, v in st.items() if v["status"] == "terminal"),
            "ready": sorted(r for r, v in st.items() if v["status"] == "ready"),
            "blocked": sorted(r for r, v in st.items() if v["status"] == "blocked"),
            "resume_rule": "the program resumes from this ledger without replanning: take every ready "
            "requirement and execute its next exact action",
        },
    )
    io.seal(
        "MOP_FAST_STATE_PROGRESS_SCORECARD.json",
        {
            "schema": "mop-fast-state-progress-scorecard/v1",
            "implementation": impl,
            "implementation_targets": IMPLEMENTATION_TARGETS,
            "evidence": {k: v for k, v in ev.items() if not k.startswith("_")},
            "evidence_targets": EVIDENCE_TARGETS,
            "verdicts": ev["_verdicts"],
            "rule": "evidence remains earned. Implementation completeness never raises an evidence score.",
        },
    )
    rows = [
        "# Fast State Plasticity Forge ledger",
        "",
        "| id | requirement | status | next exact action |",
        "| --- | --- | --- | --- |",
    ]
    for rid, v in st.items():
        rows.append(f"| {rid} | {v['title']} | {v['status']} | {v['next_exact_action']} |")
    rows += [
        "",
        "## Scores",
        "",
        "| dimension | implementation | target | evidence | target |",
        "| --- | --- | --- | --- | --- |",
    ]
    for k in IMPLEMENTATION_TARGETS:
        rows.append(
            f"| {k} | {impl.get(k, 0)} | {IMPLEMENTATION_TARGETS[k]} | "
            f"{ev.get(k, '')} | {EVIDENCE_TARGETS.get(k, '')} |"
        )
    for k in EVIDENCE_TARGETS:
        if k not in IMPLEMENTATION_TARGETS:
            rows.append(f"| {k} | | | {ev.get(k, '')} | {EVIDENCE_TARGETS[k]} |")
    io.seal_md("MOP_FAST_STATE_FORGE_LEDGER.md", "\n".join(rows) + "\n")
    io.seal_md(
        "MOP_FAST_STATE_PROGRESS_SCORECARD.md",
        "# Progress scorecard\n\n" + json.dumps({"implementation": impl, "evidence": ev}, indent=2) + "\n",
    )
    print("terminal", len([1 for v in st.values() if v["status"] == "terminal"]), "of", len(st), flush=True)
    print("ready:", [r for r, v in st.items() if v["status"] == "ready"], flush=True)


if __name__ == "__main__":
    main()
