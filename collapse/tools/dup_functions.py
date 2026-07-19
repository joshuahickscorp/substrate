"""Repo-wide duplicated-function census: the honest condensation ceiling (spec sections 9-12).

For every function in every tracked .py, normalize the body (docstring-stripped, own-name-insensitive) and
cluster by structural hash. Byte-identical bodies are the same code copy-pasted and can be condensed onto one
shared implementation. Reports:

  - total redundant LOC (sum over clusters of (count-1) * lines-of-one): the raw dedup ceiling;
  - self_contained redundant LOC: functions whose free names are only builtins / their own args / a small
    safe import set, so they can be moved to a shared module without dragging module state. This is the
    portion safely condensable now with focused parity.

Read-only. Emits collapse/MOP_DUP_FUNCTIONS.json.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse"
BUILTINS = set(dir(builtins))
SAFE_IMPORTS = {"json", "hashlib", "os", "sys", "re", "math", "Path", "pathlib", "time",
                "datetime", "collections", "itertools", "functools", "dataclasses", "typing"}


def tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout
    return [x for x in out.splitlines() if x and not x.startswith("collapse/") and not x.startswith(".collapse/")]


class _Strip(ast.NodeTransformer):
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
        cleaned = _Strip().visit(ast.parse(ast.unparse(fn)))
    except Exception:
        return ""
    for n in ast.walk(cleaned):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.name = "_"
    return hashlib.sha256(ast.dump(cleaned, annotate_fields=False).encode()).hexdigest()[:16]


def free_names(fn: ast.AST) -> set[str]:
    """Names used but not bound locally (args, assignments, comprehensions). Excludes attribute tails."""
    bound: set[str] = set()
    a = fn.args
    for arg in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
        bound.add(arg.arg)
    if a.vararg:
        bound.add(a.vararg.arg)
    if a.kwarg:
        bound.add(a.kwarg.arg)
    used: set[str] = set()
    assigned: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            if isinstance(n.ctx, ast.Store):
                assigned.add(n.id)
            else:
                used.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n is not fn:
            assigned.add(n.name)
    return used - bound - assigned - BUILTINS


def main() -> int:
    clusters: dict[str, list[dict]] = defaultdict(list)
    for rel in tracked_py():
        p = ROOT / rel
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(src)
        except Exception:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                loc = (node.end_lineno or node.lineno) - node.lineno + 1
                if loc < 4:
                    continue  # ignore trivial 1-3 line functions
                h = norm_hash(node)
                if not h:
                    continue
                fn_free = free_names(node)
                self_contained = fn_free.issubset(SAFE_IMPORTS)
                clusters[h].append({"path": rel, "name": node.name, "loc": loc,
                                    "self_contained": self_contained, "free": sorted(fn_free)[:8]})

    total_redundant = 0
    sc_redundant = 0
    big = []
    for h, members in clusters.items():
        if len(members) < 2:
            continue
        loc1 = members[0]["loc"]
        redundant = (len(members) - 1) * loc1
        total_redundant += redundant
        all_sc = all(m["self_contained"] for m in members)
        if all_sc:
            sc_redundant += redundant
        big.append({"hash": h, "count": len(members), "loc_each": loc1,
                    "redundant_LOC": redundant, "all_self_contained": all_sc,
                    "name_sample": members[0]["name"],
                    "paths": sorted({m["path"] for m in members})[:12]})
    big.sort(key=lambda c: -c["redundant_LOC"])

    out = {
        "schema": "mop-collapse-dup-functions/v1",
        "total_functions_clustered": sum(len(v) for v in clusters.values()),
        "duplicate_clusters": sum(1 for v in clusters.values() if len(v) > 1),
        "total_redundant_LOC_ceiling": total_redundant,
        "self_contained_redundant_LOC": sc_redundant,
        "top_clusters": big[:60],
    }
    (OUT / "MOP_DUP_FUNCTIONS.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({
        "duplicate_clusters": out["duplicate_clusters"],
        "total_redundant_LOC_ceiling": total_redundant,
        "self_contained_redundant_LOC": sc_redundant,
        "top10": [{"n": c["count"], "loc_each": c["loc_each"], "redundant": c["redundant_LOC"],
                   "sc": c["all_self_contained"], "name": c["name_sample"]} for c in big[:10]],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
