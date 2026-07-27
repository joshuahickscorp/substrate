"""The final master authority: one binding over every program that fed into this one.

Section 4 asks that the program resume without rereading conversation history. That is the whole test of
this module. Everything a successor needs is derived here from the tree and sealed: which historical
authorities are inherited and at what hash, what every requirement's status is, what its rollback is, and
what to do next. A successor that has to be told what happened has not been given an authority.

Rollback is the field most easily left blank and is filled here for every item, because an item whose
undo is unknown is not bounded work.

House style: no dashes.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from substrate import deliverables as D
from substrate import evidence as io
from substrate import graph as G
from substrate import program as P

FINAL_PLAN = io.ROOT / "docs" / "LONG_RUN_PLAN.md"

# every program whose evidence this authority inherits, and what it contributes
ANCESTRY = (
    ("mop", "the original research program", "historical, preserved in place, never mass renamed"),
    (
        "mop-experimental-method-reformation-v1",
        "the experiment validity kernel",
        "the admission gate every Substrate experiment passes",
    ),
    (
        "mop-fast-state-plasticity-forge-v1",
        "fast state and plasticity",
        "immutable inherited nulls including the learned plasticity closure",
    ),
    (
        "mop-temporal-core-mechanism-v1",
        "the temporal core factorial",
        "terminal, and no core was scientifically licensed",
    ),
    (
        "mop-substrate-master-v1",
        "the predecessor Substrate entity program",
        "historical evidence identity inherited by substrate-v1",
    ),
)


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=io.ROOT, capture_output=True, text=True).stdout.strip()


def _rollback(item) -> str:
    """What undoing this item means. Never blank."""
    if item.kind in ("authority", "boundary"):
        return f"revert the commit that sealed {', '.join(item.evidence) or 'its declaration'}"
    if item.evidence:
        return (
            f"delete {', '.join(e for e in item.evidence if ':' not in e)} and rerun "
            "substrate.deliverables write; the sealed inputs are immutable and are not touched"
        )
    return f"revert the commit that added {', '.join(item.impl)}"


def _experiment_for(item_id: str, results: dict) -> str:
    row = results.get(item_id) or {}
    return row.get("experiment_id") or ""


def requirement_rows(st: dict) -> list[dict]:
    results = P.result_ledger()
    rows = []
    for item in P.ITEMS:
        s = st["items"][item.id]
        rows.append(
            {
                "id": item.id,
                "category": item.category or item.kind,
                "status": s["level"],
                "authority": s["authority"],
                "dependencies": list(item.deps),
                "implementation": list(item.impl),
                "tests": list(item.tests),
                "experiment": _experiment_for(item.id, results),
                "evidence": list(item.evidence),
                "classification": (s["result"] or {}).get("classification", ""),
                "commit": s["commit"],
                "rollback": _rollback(item),
                "next_action": s["next_action"],
            }
        )
    return rows


def master_authority(st: dict) -> dict:
    graph_doc = G.declaration()
    return {
        "schema": "substrate-final-master-authority/v1",
        "supersedes": "the archived predecessor plans without invalidating anything they sealed",
        "plans": {
            "final": {"path": str(FINAL_PLAN), "sha256": _sha(FINAL_PLAN), "resolved": FINAL_PLAN.is_file()},
        },
        "source": {
            "commit": io.commit(),
            "tree": _git("rev-parse", "HEAD^{tree}"),
            "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
            "remote_branch": "agent/substrate-event-horizon",
            "pull_request": 35,
        },
        "proof_roots": {k: str(v.relative_to(io.ROOT)) for k, v in D.ROOTS.items() if k},
        "inherited_authorities": [
            {"program": prog, **D._bind(ref), "role": why} for prog, ref, why in D.INHERITED_AUTHORITIES
        ],
        "naming_authority": D.NAMING,
        "claim_boundary": D.CLAIM_BOUNDARY,
        "goal_authority": D.GOAL_AUTHORITY,
        "single_infrastructure": {
            "scheduler": "substrate.execution over substrate.graph",
            "engine": "substrate.execution",
            "admission_gate": "substrate.method.gate through substrate.admission",
            "registry": "substrate.program.ITEMS",
            "configuration_authority": "configs/substrate/config.json through substrate.config",
            "cli": "substrate",
            "evidence_fabric": "evidence/substrate/v1 through substrate.evidence",
        },
        "program_graph": {
            "nodes": graph_doc["node_count"],
            "valid": graph_doc["valid"],
            "externally_blocked": graph_doc["externally_blocked"],
            "buildable_prerequisites": [b["node"] for b in graph_doc["buildable_prerequisites"]],
        },
        "resume_without_history": (
            "every field a successor needs is in SUBSTRATE_FINAL_STATE.json and "
            "this file. No conversation is required to continue"
        ),
        "activation": False,
    }


def ancestry() -> dict:
    return {
        "schema": "substrate-final-ancestry/v1",
        "programs": [
            {
                "program": name,
                "contribution": what,
                "role": role,
                "proof_root": str((io.ROOT / "proof" / "substrate" / name).relative_to(io.ROOT))
                if (io.ROOT / "proof" / "substrate" / name).is_dir()
                else None,
            }
            for name, what, role in ANCESTRY
        ],
        "rule": (
            "historical evidence is preserved in place. This authority binds it and never rewrites "
            "it, and a superseded claim keeps its original sealed bytes"
        ),
        "temporal_core_verdict": {
            "terminal": True,
            "licensed": False,
            "why": "role B found the load bearing baselines unconverged on all three beds, so the "
            "selection was withdrawn and no successor opened",
            "consequence": "the runtime integrates a declared control implementation and records the "
            "scientific limitation rather than pretending a core was selected",
        },
        "activation": False,
    }


def final_state(st: dict) -> dict:
    return {
        "schema": "substrate-final-state/v1",
        "requirements": requirement_rows(st),
        "total": len(st["items"]),
        "level_counts": st["level_counts"],
        "corrections": st["corrections"],
        "source_tree": st["source_tree"],
        "every_requirement_has_a_rollback": all(r["rollback"] for r in requirement_rows(st)),
        "activation": False,
    }


def final_scorecard(st: dict) -> dict:
    card = P.scorecard(st)
    graph_doc = G.declaration()
    return {
        "schema": "substrate-final-scorecard/v1",
        "categories": card["categories"],
        "categories_with_a_positive": card["categories_with_a_positive"],
        "reading_the_evidence_column": card["reading_the_evidence_column"],
        "implementation_target_band": [80, 95],
        "evidence_target_band": [50, 75],
        "evidence_target_note": (
            "section 43 sets fifty to seventy five percent before a subsystem is "
            "treated as established. No subsystem is there"
        ),
        "sentience_has_no_score": True,
        "graph_terminal_nodes": len(graph_doc["terminal_nodes"]),
        "graph_nodes": graph_doc["node_count"],
        "activation": False,
    }


def value_queue(st: dict) -> dict:
    from substrate import experiments as X

    graph_doc = G.declaration()
    ready = [
        r
        for r in graph_doc["nodes"]
        if not r["exit_passed"]
        and not r["missing_implementation"]
        and not r["external_blocker"]
        and all(G.BY_ID[d].identity in graph_doc["terminal_nodes"] for d in r["dependencies"])
    ]
    return {
        "schema": "substrate-final-value-queue/v1",
        "experiment_queue": X.voi_queue(),
        "graph_ready_nodes": [r["identity"] for r in ready],
        "buildable_prerequisites": graph_doc["buildable_prerequisites"],
        "selection_rule": (
            "expected decision information, scientific headroom, relevance to entity "
            "construction, reuse value, engineering cost, compute cost, validity risk"
        ),
        "do_not_run_everything": True,
        "activation": False,
    }


def ledger_markdown(st: dict) -> str:
    rows = requirement_rows(st)
    lines = [
        "# Substrate final ledger",
        "",
        f"Derived from the tree at `{io.commit()}`. Status is computed, never asserted.",
        "",
        f"{len(rows)} requirements. "
        + ", ".join(f"{k} {v}" for k, v in sorted(st["level_counts"].items()))
        + ".",
        "",
        "| id | category | status | classification | next action | rollback |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['category']} | {r['status']} | {r['classification'] or '-'} | "
            f"{r['next_action']} | {r['rollback'][:70]} |"
        )
    graph_doc = G.declaration()
    lines += [
        "",
        "## Program graph",
        "",
        f"{graph_doc['node_count']} nodes, {len(graph_doc['terminal_nodes'])} terminal, "
        f"{len(graph_doc['buildable_prerequisites'])} buildable prerequisites, "
        f"{len(graph_doc['externally_blocked'])} externally blocked.",
        "",
        "No future wave exists as prose. Every one is a node with an entry and an exit gate.",
        "",
        "Activation remains false.",
    ]
    return "\n".join(lines) + "\n"


def write_all() -> dict:
    st = P.state()
    # SUBSTRATE_FINAL_PROGRAM_GRAPH.json is owned by substrate.graph. One producer per artifact,
    # because two producers means the last one to run decides what the evidence says.
    written = {
        "SUBSTRATE_FINAL_MASTER_AUTHORITY.json": io.seal(
            "SUBSTRATE_FINAL_MASTER_AUTHORITY.json", master_authority(st)
        ),
        "SUBSTRATE_FINAL_ANCESTRY.json": io.seal("SUBSTRATE_FINAL_ANCESTRY.json", ancestry()),
        "SUBSTRATE_FINAL_STATE.json": io.seal("SUBSTRATE_FINAL_STATE.json", final_state(st)),
        "SUBSTRATE_FINAL_SCORECARD.json": io.seal("SUBSTRATE_FINAL_SCORECARD.json", final_scorecard(st)),
        "SUBSTRATE_FINAL_VALUE_QUEUE.json": io.seal("SUBSTRATE_FINAL_VALUE_QUEUE.json", value_queue(st)),
        "SUBSTRATE_FINAL_LEDGER.md": io.seal_md("SUBSTRATE_FINAL_LEDGER.md", ledger_markdown(st)),
    }
    return {k: v.relative_to(io.ROOT).as_posix() for k, v in written.items()}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    written = write_all()
    print(json.dumps({"sealed": sorted(written), "count": len(written)}, indent=2))


if __name__ == "__main__":
    main()
