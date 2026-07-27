"""Writes the Substrate master deliverables from the item table and from artifacts already on disk.

Nothing in here composes prose that outruns the tree. The null map binds to the historical authorities by
path and hash and marks an unresolvable source as unresolved rather than quietly dropping it. The hypothesis
graph inherits the method reformation graph instead of restating it, because that graph is sealed evidence
and this program is not licensed to rewrite it.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from mop.cognition import io, program as P

PLAN_PATH = Path(P.PLAN)

# historical authorities Substrate inherits. Each entry is bound by path and hash at write time.
INHERITED_AUTHORITIES = (
    ("method_reformation", "method:MOP_EXPERIMENT_VALIDITY_KERNEL.json",
     "the admission gate every new Substrate experiment must pass"),
    ("method_reformation", "method:MOP_HISTORICAL_EXPERIMENT_DEFECT_LEDGER.json",
     "eighteen reproduced defect classes, each a permanent regression"),
    ("method_reformation", "method:MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json",
     "the inherited hypothesis graph this program extends and never rewrites"),
    ("method_reformation", "method:MOP_METHOD_NEXT_SUBSTRATE_FRONTIER.json",
     "the value of information selection that licensed the temporal core program"),
    ("fast_state_forge", "fastforge:MOP_FAST_STATE_BINDING_NULLS.json",
     "immutable inherited nulls, supersedable only by an appended authority"),
    ("fast_state_forge", "fastforge:MOP_FAST_STATE_FORGE_SYNTHESIS.json",
     "the terminal synthesis of the fast state program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_CORE_START_AUTHORITY.json",
     "the start authority of the live temporal core mechanism program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_METHOD_EXTENSION.json",
     "five method witnesses added by the temporal program"),
    ("temporal_core", "temporal:MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.json",
     "the temporal factorial hypothesis graph"),
    ("temporal_core", "temporal:MOP_DATA_CUSTODY_AUTHORITY.json",
     "corpus custody and the deletion guard"),
)

ROOTS = dict(P.PROOF_ROOTS)
ROOTS["fastforge"] = io.ROOT / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1"


def _resolve(ref: str) -> Path:
    root, _, name = ref.rpartition(":")
    return ROOTS.get(root, io.PROOF) / name


def _bind(ref: str) -> dict:
    path = _resolve(ref)
    if not path.is_file():
        return {"reference": ref, "resolved": False,
                "path": path.relative_to(io.ROOT).as_posix() if io.ROOT in path.parents else str(path),
                "sha256": None}
    return {"reference": ref, "resolved": True,
            "path": path.relative_to(io.ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


# ---------------------------------------------------------------- naming and claim boundary

NAMING = {
    "current_program": "Substrate",
    "map": {
        "MOP": "the original research program, valid inside every historical authority",
        "Mixture of Perspectives": "the perspective generation and arbitration subsystem, items C3 to C5",
        "Mixture of Thinking": "the composition of heterogeneous cognitive processes, items C5, K1 and T1",
    },
    "rule": ("historical files, branches, commits, tags, proofs and program identities are not mass "
             "renamed. This authority maps old terminology onto the current architecture and leaves prior "
             "evidence exactly as sealed"),
    "renamed_nothing": True,
}

CLAIM_BOUNDARY = {
    "levels": ["demonstrated engineering property", "behavioural indication",
               "architectural prerequisite", "philosophical interpretation", "unsupported claim"],
    "permitted_terms": ["sentience adjacent architecture", "entity like continuity",
                        "developmental cognition", "reflective cognitive organization"],
    "forbidden_claims": ["consciousness", "sentience", "feelings", "wants", "suffering",
                         "subjective experience", "life"],
    "rule": "no single architectural property is proof of sentience",
}

PROTECTED_SURFACES = (
    "evidence validation", "audit systems", "claim boundaries", "stop switches",
    "resource limits", "rollback", "adaptation constraints",
)

GOAL_AUTHORITY = {
    "origin": "SUBSTRATE_MASTER_PLAN.md, supplied by the operator",
    "scope": "the items declared in mop.cognition.program.ITEMS",
    "authority": "operator issued, bounded by this file",
    "resources": "local compute already licensed to the temporal core supervisor plus this session",
    "constraints": "activation stays false, no autonomous removal of protected surfaces",
    "termination": "the declared authority is terminal and no dependency ready work remains",
    "audit": "SUBSTRATE_STATE.json, regenerated from the tree on every run",
    "may_not": "silently create unrestricted long term goals",
}

# ---------------------------------------------------------------- Substrate native hypotheses

HYPOTHESES = (
    dict(id="H_typed_workspace", items=["C2", "E2"],
         premise="typed regions with declared readers and writers beat one unrestricted shared state of "
                 "matched capacity on tasks that need cross perspective information",
         predecessor=None, state="unopened",
         required_bed="a bed where at least two perspectives hold information neither one has alone",
         required_headroom="residual headroom over the untyped shared state control",
         strongest_baseline="one opaque state of matched capacity and matched update budget",
         cheapest_falsifier="the same tasks with region typing removed and nothing else changed",
         dependent_hypotheses=["H_arbitration_minority"], blocking_null=None),
    dict(id="H_perspective_diversity", items=["C3", "C4"],
         premise="a set of heterogeneous perspectives beats the single strongest perspective given the "
                 "same total compute",
         predecessor=None, state="unopened",
         required_bed="a bed where the best single perspective leaves measured residual headroom",
         required_headroom="oracle selection over the perspective set must beat the best fixed single one",
         strongest_baseline="best single perspective at the full combined budget",
         cheapest_falsifier="oracle selection with no headroom over the best single perspective",
         dependent_hypotheses=["H_learned_selector"], blocking_null=None),
    dict(id="H_learned_selector", items=["C4"],
         premise="a learned perspective selector beats the strongest simple selection rule",
         predecessor="H_perspective_diversity", state="unopened",
         required_bed="inherits the bed of its predecessor",
         required_headroom="stable residual headroom beyond reliability weighted selection",
         strongest_baseline="reliability weighted selection",
         cheapest_falsifier="oracle minus strong simple selection at or below the SESOI",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_arbitration_minority", items=["C5"],
         premise="preserving a minority hypothesis through arbitration improves the terminal decision "
                 "over forcing consensus at the same budget",
         predecessor="H_typed_workspace", state="unopened",
         required_bed="a bed containing items where the majority perspective is wrong",
         required_headroom="the minority must be correct often enough to be recoverable",
         strongest_baseline="confidence weighted majority with no minority retention",
         cheapest_falsifier="no bed item where the preserved minority changes the terminal answer",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_owned_continuity", items=["E1"],
         premise="owned state restores goals, beliefs and unresolved questions after context removal "
                 "better than replaying the transcript at a matched token budget",
         predecessor=None, state="unopened",
         required_bed="tasks that span an enforced interruption and a checkpoint restore",
         required_headroom="transcript replay must be measurably imperfect at the matched budget",
         strongest_baseline="full transcript replay truncated to the same budget as the owned state",
         cheapest_falsifier="transcript replay matching owned state on every continuity probe",
         dependent_hypotheses=["H_selfmodel_calibration"], blocking_null=None),
    dict(id="H_selfmodel_calibration", items=["S1", "E3"],
         premise="a measured self model improves decisions over a fixed prior of the same form",
         predecessor="H_owned_continuity", state="unopened",
         required_bed="tasks where predicted and actual outcome can both be measured per unit",
         required_headroom="the fixed prior must be measurably miscalibrated",
         strongest_baseline="a fixed prior fitted once and never updated",
         cheapest_falsifier="the fixed prior already calibrated within the SESOI",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_consolidation", items=["M5"],
         premise="verification triggered consolidation beats both no consolidation and a fixed schedule",
         predecessor=None, state="unopened",
         required_bed="a stream with repeated structure and a verifiable outcome per episode",
         required_headroom="oracle consolidation must beat no consolidation",
         strongest_baseline="fixed schedule consolidation at the matched write budget",
         cheapest_falsifier="oracle consolidation at or below no consolidation",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_bounded_reorg", items=["R1"],
         premise="bounded functional reorganization improves downstream utility beyond fixed and simple "
                 "routing after its cost is charged",
         predecessor=None, state="unopened",
         required_bed="a multi domain bed where the useful routing differs by domain",
         required_headroom="oracle routing must beat fixed routing after cost",
         strongest_baseline="simple context rule routing",
         cheapest_falsifier="oracle routing at or below fixed routing once cost is charged",
         dependent_hypotheses=[], blocking_null=None),
    dict(id="H_learned_plasticity_policy", items=["P4"],
         premise="a learned plasticity policy beats the strongest simple plasticity rule",
         predecessor=None, state="closed",
         required_bed="inherits the fast state forge beds",
         required_headroom="stable residual headroom beyond simple rules",
         strongest_baseline="the best simple triggered rule",
         cheapest_falsifier="already run and null",
         dependent_hypotheses=[],
         blocking_null="fastforge:MOP_FAST_STATE_BINDING_NULLS.json#inherited_nulls.learned_plasticity"),
)


def hypothesis_graph(st: dict) -> dict:
    items = st["items"]
    rows = []
    for h in HYPOTHESES:
        row = dict(h)
        row["carrying_items"] = [{"id": i, "level": items.get(i, {}).get("level")} for i in h["items"]]
        if h["blocking_null"]:
            row["state"] = "closed"
            row["closure"] = _bind(h["blocking_null"].split("#", 1)[0])
            row["closes_descendants"] = list(h["dependent_hypotheses"])
        rows.append(row)
    return {
        "schema": "substrate-hypothesis-graph/v1",
        "states": ["unopened", "headroom_pending", "instrument_pending", "admitted", "supported",
                   "mixed", "null", "harm", "invalid", "superseded", "closed"],
        "inherited_graph": _bind("method:MOP_SUBSTRATE_HYPOTHESIS_GRAPH.json"),
        "inherited_temporal_graph": _bind("temporal:MOP_TEMPORAL_CORE_HYPOTHESIS_GRAPH.json"),
        "inheritance_rule": ("the inherited graphs are sealed evidence. This program appends Substrate "
                             "native hypotheses and never edits an inherited node"),
        "hypotheses": rows,
        "open_count": sum(1 for r in rows if r["state"] not in ("closed", "null", "superseded")),
    }


def null_map(st: dict) -> dict:
    inherited = [{"program": prog, **_bind(ref), "role": why}
                 for prog, ref, why in INHERITED_AUTHORITIES]
    native = P.null_ledger()
    unresolved = [row["reference"] for row in inherited if not row["resolved"]]
    return {
        "schema": "substrate-null-map/v1",
        "binding_rule": ("a null is immutable. It may be superseded only by an appended authority that "
                         "states the new evidence, never by rewriting or relabelling the original"),
        "closure_rule": "a failed branch closes only its own descendants, never an independent branch",
        "inherited_authorities": inherited,
        "unresolved_inherited_authorities": unresolved,
        "all_inherited_resolved": not unresolved,
        "substrate_native_nulls": native,
        "native_null_count": len(native),
        "closed_hypotheses": [h["id"] for h in HYPOTHESES if h["blocking_null"]],
    }


def master_authority(st: dict) -> dict:
    plan_sha = hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest() if PLAN_PATH.is_file() else None
    deliverables = sorted({e for item in P.ITEMS for e in item.evidence})
    return {
        "schema": "substrate-master-authority/v1",
        "program": io.PROGRAM,
        "plan": {"path": str(PLAN_PATH), "sha256": plan_sha, "resolved": plan_sha is not None},
        "naming_authority": NAMING,
        "claim_boundary": CLAIM_BOUNDARY,
        "goal_authority": GOAL_AUTHORITY,
        "protected_surfaces": list(PROTECTED_SURFACES),
        "single_infrastructure": {
            "scheduler": "mop.temporal.runs.supervisor",
            "experiment_engine": "fastforge.engine",
            "admission_gate": "mop.method.gate",
            "evidence_fabric": "integrated/evidence_store",
            "registry": "registry/experiments.yaml",
            "configuration_root": "configs/",
            "cli": "python -m mop.<program>.<stage>",
            "rule": "no parallel framework is created; Substrate stages reuse these",
        },
        "inherited_authorities": [{"program": prog, **_bind(ref), "role": why}
                                  for prog, ref, why in INHERITED_AUTHORITIES],
        "item_inventory": [{"id": i.id, "section": i.section, "title": i.title, "batch": i.batch,
                            "kind": i.kind, "category": i.category, "dependencies": list(i.deps)}
                           for i in P.ITEMS],
        "declared_deliverables": deliverables,
        # present means a file exists at the bound path. counts means it also passes its own terminal
        # keys and was sealed at a commit reachable from HEAD. The two are reported apart on purpose.
        "deliverables_present": {d: _bind(d)["resolved"] for d in deliverables},
        "deliverables_counting_as_evidence": {d: P.evidence_state(d)["counts"] for d in deliverables},
        "level_counts": st["level_counts"],
        "activation": False,
    }


def ledger_markdown(st: dict, frontier: dict) -> str:
    lines = [
        "# Substrate ledger",
        "",
        f"Generated from the tree at commit `{io.commit()}`. Status is derived, never asserted: an item is",
        "implemented because its files exist, tested because a recorded test ledger says so, measured",
        "because its evidence is sealed, and terminal because a scientific classification exists for it.",
        "",
        f"Items: {st['total_items']}. Levels: "
        + ", ".join(f"{k} {v}" for k, v in sorted(st["level_counts"].items())) + ".",
        "",
        "| id | section | title | level | dependencies | next action |",
        "|---|---|---|---|---|---|",
    ]
    for row in st["items"].values():
        lines.append(
            f"| {row['id']} | {row['section']} | {row['title']} | {row['level']} | "
            f"{', '.join(row['dependencies']) or 'none'} | {row['next_action']} |"
        )
    primary = frontier.get("primary") or {}
    secondary = frontier.get("secondary") or {}
    lines += [
        "",
        "## Selected next batch",
        "",
        f"Primary: {primary.get('id', 'none')} {primary.get('title', '')}. {primary.get('next_action', '')}",
        f"Secondary: {secondary.get('id', 'none')} {secondary.get('title', '')}. "
        f"{secondary.get('next_action', '')}",
        "",
        "Activation remains false.",
    ]
    return "\n".join(lines) + "\n"


def write_all() -> dict:
    st = P.state()
    frontier = P.next_batches(st)
    written = {
        "SUBSTRATE_MASTER_AUTHORITY.json": io.seal("SUBSTRATE_MASTER_AUTHORITY.json", master_authority(st)),
        "SUBSTRATE_STATE.json": io.seal("SUBSTRATE_STATE.json", st),
        "SUBSTRATE_HYPOTHESIS_GRAPH.json": io.seal("SUBSTRATE_HYPOTHESIS_GRAPH.json", hypothesis_graph(st)),
        "SUBSTRATE_NULL_MAP.json": io.seal("SUBSTRATE_NULL_MAP.json", null_map(st)),
        "SUBSTRATE_PROGRESS_SCORECARD.json": io.seal("SUBSTRATE_PROGRESS_SCORECARD.json", P.scorecard(st)),
        "SUBSTRATE_NEXT_FRONTIER.json": io.seal("SUBSTRATE_NEXT_FRONTIER.json", frontier),
        "SUBSTRATE_LEDGER.md": io.seal_md("SUBSTRATE_LEDGER.md", ledger_markdown(st, frontier)),
    }
    return {name: path.relative_to(io.ROOT).as_posix() for name, path in written.items()}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "write":
        raise ValueError(argv)
    written = write_all()
    print(json.dumps(written, indent=2))
    st = P.state()
    print(f"substrate deliverables: {len(written)} written, "
          f"{st['total_items']} items, levels {st['level_counts']}", flush=True)


if __name__ == "__main__":
    main()
