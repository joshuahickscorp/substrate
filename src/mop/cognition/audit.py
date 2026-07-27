"""The structural audit that runs before anything is certified or launched.

Seven checks, each of which has already caught a real defect at least once in this program. The producer
check found three artifacts written by two modules apiece, where the last writer silently won. The stale
check is what a rebase trips. The activation check is the one that must never go green by accident.

Nothing here is a subsystem. It reads the tree and answers yes or no, and it is cheap enough to run before
every commit.

House style: no dashes.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from mop.cognition import graph as G
from mop.cognition import io
from mop.cognition import program as P
from mop.cognition import runtime as R

SRC = io.ROOT / "src" / "mop" / "cognition"

CHECKS = ("exclusive_producers", "no_stale_outputs", "no_duplicate_stages", "causal_paths_present",
          "every_node_actionable", "runtime_stages_reachable", "no_activation_path")

# an assignment that would turn activation on. The boundary document may describe it; no module may do it.
ACTIVATION_PATTERNS = (
    re.compile(r'"activation"\s*:\s*True'),
    re.compile(r"\bactivation\s*=\s*True\b"),
    re.compile(r"ACTIVATION\s*=\s*True"),
)


def producers() -> dict:
    out = defaultdict(set)
    for f in sorted(SRC.glob("*.py")):
        text = f.read_text()
        # a literal first argument, and a literal inside a dict passed as the first argument. An
        # artifact name built at runtime is invisible to this scan, so building one is refused by the
        # orphan half of the check below rather than tolerated.
        for m in re.finditer(r"io\.seal(?:_md)?\(\s*[\"']([A-Z_0-9]+\.(?:json|md))[\"']", text):
            out[m.group(1)].add(f.stem)
        for m in re.finditer(r"io\.seal(?:_md)?\(\{([^}]*)\}", text, re.S):
            for name in re.findall(r"[\"']([A-Z_0-9]+\.(?:json|md))[\"']", m.group(1)):
                out[name].add(f.stem)
    return {k: sorted(v) for k, v in out.items()}


def exclusive_producers() -> dict:
    p = producers()
    duplicated = {k: v for k, v in p.items() if len(v) > 1}
    orphans = sorted(a.name for a in io.PROOF.glob("SUBSTRATE_*") if a.name not in p)
    return {"artifacts": len(p), "duplicated": duplicated,
            "on_disk_without_a_producer": orphans,
            "ok": not duplicated and not orphans}


def no_stale_outputs() -> dict:
    stale = []
    for path in sorted(io.PROOF.glob("SUBSTRATE_*.json")):
        row = P.evidence_state(path.name)
        if not row["counts"]:
            stale.append({"artifact": path.name, "reason": row["reason"]})
    return {"checked": len(list(io.PROOF.glob("SUBSTRATE_*.json"))), "stale": stale,
            "ok": not stale}


def no_duplicate_stages() -> dict:
    ids = [n.identity for n in G.NODES]
    modules = defaultdict(list)
    for n in G.NODES:
        modules[(n.module, n.args)].append(n.identity)
    same_work = {f"{m} {a}": v for (m, a), v in modules.items() if len(v) > 1 and m}
    return {"nodes": len(ids), "duplicate_identities": [i for i in ids if ids.count(i) > 1],
            "nodes_running_the_same_command": same_work,
            "ok": len(set(ids)) == len(ids)}


def causal_paths_present() -> dict:
    """Every node except the roots must be reachable from a root, and every dependency must exist."""
    missing = [(n.identity, d) for n in G.NODES for d in n.dependencies if d not in G.BY_ID]
    roots = [n.identity for n in G.NODES if not n.dependencies]
    reachable, frontier = set(roots), list(roots)
    while frontier:
        current = frontier.pop()
        for s in G.successors(current):
            if s not in reachable:
                reachable.add(s)
                frontier.append(s)
    unreachable = sorted({n.identity for n in G.NODES} - reachable)
    return {"roots": roots, "missing_dependencies": missing, "unreachable": unreachable,
            "ok": not missing and not unreachable}


def every_node_actionable() -> dict:
    """A node must be runnable, terminally gated by an external blocker, or waiting on a named result."""
    rows = []
    for n in G.NODES:
        runnable = bool(n.module)
        gated = bool(n.external_blocker)
        depends = bool(n.dependencies)
        rows.append({"node": n.identity, "runnable": runnable, "externally_gated": gated,
                     "depends_on_named_results": list(n.dependencies),
                     "actionable": runnable or gated or depends})
    return {"nodes": rows, "not_actionable": [r["node"] for r in rows if not r["actionable"]],
            "ok": all(r["actionable"] for r in rows)}


def runtime_stages_reachable() -> dict:
    """Every declared stage must be written by the loop and leave a receipt when it runs."""
    source = (SRC / "runtime.py").read_text()
    recorded = set(re.findall(r'trace\.(?:record|skip)\(\s*"(\w+)"', source))
    entity = R.Substrate()
    trace = entity.step({"label": "a", "label_confidence": 0.8}, outcome="a", goal=["audit"])
    ran = set(trace["stages_ran"]) | set(trace["stages_skipped"])
    return {"declared": list(R.STAGES), "written_in_the_loop": sorted(recorded),
            "observed_in_one_cycle": sorted(ran),
            "declared_but_never_written": sorted(set(R.STAGES) - recorded),
            "declared_but_not_observed": sorted(set(R.STAGES) - ran),
            "ok": set(R.STAGES) <= recorded and set(R.STAGES) <= ran}


def no_activation_path() -> dict:
    hits = []
    for f in sorted(SRC.glob("*.py")):
        text = f.read_text()
        for pattern in ACTIVATION_PATTERNS:
            for m in pattern.finditer(text):
                line = text[: m.start()].count("\n") + 1
                hits.append({"file": f.name, "line": line, "match": m.group(0)})
    sealed = []
    for path in sorted(io.PROOF.glob("SUBSTRATE_*.json")):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if doc.get("activation") is True:
            sealed.append(path.name)
    return {"source_hits": hits, "sealed_artifacts_with_activation_true": sealed,
            "ok": not hits and not sealed}


def run() -> dict:
    results = {
        "exclusive_producers": exclusive_producers(),
        "no_stale_outputs": no_stale_outputs(),
        "no_duplicate_stages": no_duplicate_stages(),
        "causal_paths_present": causal_paths_present(),
        "every_node_actionable": every_node_actionable(),
        "runtime_stages_reachable": runtime_stages_reachable(),
        "no_activation_path": no_activation_path(),
    }
    failed = sorted(k for k, v in results.items() if not v["ok"])
    return {"schema": "substrate-structural-audit/v1", "checks": list(CHECKS), "results": results,
            "failed": failed, "all_pass": not failed,
            "commit": io.commit(), "activation": False}


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] not in ("run", "seal"):
        raise ValueError(argv)
    doc = run()
    io.seal("SUBSTRATE_STRUCTURAL_AUDIT.json", doc)
    print(json.dumps({"all_pass": doc["all_pass"], "failed": doc["failed"],
                      "summary": {k: v["ok"] for k, v in doc["results"].items()}}, indent=2))
    if not doc["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
