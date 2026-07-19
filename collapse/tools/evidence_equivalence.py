"""Evidence-primitive equivalence analysis (spec section 9: one evidence authority).

Deterministic and read-only. For each integrity/evidence primitive, extract every owned definition's body,
normalize it (AST structure, docstring and comment insensitive), hash the normalized structure, and cluster.
This turns the 168 raw matches into a precise deletion map:

  - identical-body clusters are true duplicates safe to collapse onto one evidence core;
  - distinct bodies are genuinely different implementations that must be inspected or preserved.

Emits collapse/MOP_EVIDENCE_EQUIVALENCE.json. No mass runtime imports (which would pull heavy deps and
compete with the live run); pure source analysis.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse"

TARGETS = {
    "canonical_bytes", "canonical_sha256", "atomic_write_json", "sha256_file",
    "file_sha256", "_validate_seal", "validate_seal", "read_json", "_read_json",
    "canonical_json", "_atomic_write", "atomic_write",
}


def tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout
    return [x for x in out.splitlines() if x]


class _Norm(ast.NodeTransformer):
    """Strip docstrings so structurally identical bodies hash the same regardless of prose."""

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_AsyncFunctionDef = visit_FunctionDef


def norm_hash(fn: ast.AST) -> str:
    try:
        cleaned = _Norm().visit(ast.parse(ast.unparse(fn)))
        dumped = ast.dump(cleaned, annotate_fields=False)
    except Exception:
        dumped = ast.dump(fn, annotate_fields=False)
    return hashlib.sha256(dumped.encode()).hexdigest()[:16]


def main() -> int:
    # name -> list of (path, line, body_hash, loc)
    defs: dict[str, list[dict]] = defaultdict(list)
    for rel in tracked_py():
        p = ROOT / rel
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in TARGETS:
                loc = (node.end_lineno or node.lineno) - node.lineno + 1
                defs[node.name].append({
                    "path": rel, "line": node.lineno, "body_hash": norm_hash(node), "loc": loc,
                })

    analysis = {}
    total_defs = 0
    total_redundant = 0  # definitions beyond one-per-distinct-body (safe collapse candidates)
    for name, occurrences in sorted(defs.items()):
        by_hash: dict[str, list[dict]] = defaultdict(list)
        for o in occurrences:
            by_hash[o["body_hash"]].append(o)
        clusters = sorted(by_hash.items(), key=lambda kv: -len(kv[1]))
        distinct = len(clusters)
        n = len(occurrences)
        redundant = n - distinct  # if all defs collapse to distinct canonical forms
        total_defs += n
        total_redundant += redundant
        analysis[name] = {
            "definitions": n,
            "distinct_bodies": distinct,
            "redundant_defs_collapsible": redundant,
            "dominant_cluster_size": len(clusters[0][1]) if clusters else 0,
            "clusters": [
                {"body_hash": h, "count": len(v), "members": sorted(m["path"] for m in v)}
                for h, v in clusters
            ],
        }

    out = {
        "schema": "mop-collapse-evidence-equivalence/v1",
        "purpose": "Precise deletion map for the one-evidence-authority collapse (section 9).",
        "totals": {
            "primitive_definitions": total_defs,
            "redundant_definitions_collapsible": total_redundant,
            "note": ("redundant_definitions_collapsible = definitions minus distinct normalized bodies; "
                     "these are exact structural duplicates that can be replaced by the evidence core once "
                     "parity + mutation tests pass. distinct bodies must be inspected before deletion."),
        },
        "by_primitive": analysis,
    }
    (OUT / "MOP_EVIDENCE_EQUIVALENCE.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps({
        "primitive_definitions": total_defs,
        "redundant_definitions_collapsible": total_redundant,
        "per_primitive": {k: {"defs": v["definitions"], "distinct": v["distinct_bodies"],
                              "collapsible": v["redundant_defs_collapsible"],
                              "dominant": v["dominant_cluster_size"]}
                          for k, v in analysis.items()},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
