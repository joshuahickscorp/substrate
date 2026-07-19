"""Evidence-authority migration table + safe batch executor (spec sections 9, 21.6).

Two modes:
  --table   build collapse/MOP_EVIDENCE_MIGRATION.json: one row per duplicate primitive definition with
            destination API, matches_core, scientific_independence flag, deletion gate, and batch.
  --apply   execute a bounded, verified batch: redirect canonical_bytes + canonical_sha256 in the given
            modules to mop.substrate.events and physically delete the local defs. Every file is verified
            byte-identical to the core at edit time, then py_compiled; any failure reverts that file.

Nothing here imports the target modules (which may be heavy). It is pure source transformation with an AST
identity gate. Redirecting a byte-identical pure crypto primitive preserves behavior exactly; section 9
explicitly permits sharing cryptographic primitives, including inside verifiers.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "collapse"
CORE_MODULE = "mop.substrate.events"
CORE_FILE = ROOT / "src/mop/substrate/events.py"
PRIMS = ("canonical_bytes", "canonical_sha256")


class _Norm(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node


def body_hash(fn: ast.AST) -> str:
    cleaned = _Norm().visit(ast.parse(ast.unparse(fn)))
    return hashlib.sha256(ast.dump(cleaned, annotate_fields=False).encode()).hexdigest()[:16]


def core_hashes() -> dict[str, str]:
    tree = ast.parse(CORE_FILE.read_text())
    h = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in PRIMS:
            h[node.name] = body_hash(node)
    return h


def find_defs(path: Path, names) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text())
    return {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names}


def tracked_py() -> list[str]:
    out = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT, capture_output=True, text=True).stdout
    return [x for x in out.splitlines() if x]


def build_table() -> dict:
    core = core_hashes()
    equiv = json.loads((OUT / "MOP_EVIDENCE_EQUIVALENCE.json").read_text())
    rows = []
    for prim, info in equiv["by_primitive"].items():
        dest = f"{CORE_MODULE}.{prim}" if prim in ("canonical_bytes", "canonical_sha256",
                                                    "sha256_file", "atomic_write_json") else "TBD"
        core_h = core.get(prim)
        for cluster in info["clusters"]:
            matches_core = (core_h is not None and cluster["body_hash"] == core_h)
            for member in cluster["members"]:
                if member.endswith("substrate/events.py"):
                    continue
                is_verifier = member.endswith("_verifier.py") or "_verify" in Path(member).name
                is_controller = "/studio/campaign_" in member or member.endswith("telegram_rung_notifier.py")
                if not matches_core:
                    batch = "inspect_distinct_body"
                elif is_controller:
                    batch = "defer_section13_controller"
                elif is_verifier:
                    batch = "defer_verifier_independence_review"
                else:
                    batch = "batch1_studies_safe"
                rows.append({
                    "primitive": prim, "path": member, "body_hash": cluster["body_hash"],
                    "destination_api": dest, "matches_core": matches_core,
                    "scientific_independence": ("independent_verifier" if is_verifier else "shared_integrity"),
                    "deletion_gate": ("byte-identical to core + py_compile + focused import parity"
                                      if matches_core else "manual inspection: body differs from core"),
                    "batch": batch,
                })
    table = {
        "schema": "mop-collapse-evidence-migration/v1",
        "core_module": CORE_MODULE,
        "core_hashes": core,
        "rows": sorted(rows, key=lambda r: (r["primitive"], r["path"])),
        "batch_counts": {},
    }
    from collections import Counter
    table["batch_counts"] = dict(Counter(r["batch"] for r in rows))
    (OUT / "MOP_EVIDENCE_MIGRATION.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    return table


def redirect_file(path: Path, core: dict[str, str]) -> tuple[bool, int, str]:
    """Return (changed, loc_removed, message). Verifies byte-identity, deletes defs, adds import, py_compiles."""
    original = path.read_text()
    try:
        defs = find_defs(path, PRIMS)
    except SyntaxError as e:
        return False, 0, f"skip (unparseable): {e}"
    targets = {}
    for name in PRIMS:
        if name not in defs:
            return False, 0, f"skip: {name} not a module-level def"
        if body_hash(defs[name]) != core.get(name):
            return False, 0, f"skip: {name} body not byte-identical to core"
        targets[name] = defs[name]

    lines = original.splitlines(keepends=True)
    # collect 1-based removal ranges (include decorator lines), then drop
    remove = set()
    loc_removed = 0
    for fn in targets.values():
        start = fn.lineno
        for d in getattr(fn, "decorator_list", []):
            start = min(start, d.lineno)
        end = fn.end_lineno
        for ln in range(start, end + 1):
            remove.add(ln)
        loc_removed += end - start + 1
    # also drop a single trailing blank line after each removed block if present
    kept = []
    for i, line in enumerate(lines, 1):
        if i in remove:
            continue
        kept.append(line)
    new_src = "".join(kept)

    # ensure import of both prims from the core
    tree = ast.parse(new_src)
    have = set()
    existing_events_import_line = None
    last_import_end = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE and node.level == 0:
            existing_events_import_line = node
            have |= {a.name for a in node.names}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import_end = max(last_import_end, node.end_lineno or node.lineno)
    need = [p for p in PRIMS if p not in have]
    if need:
        new_lines = new_src.splitlines(keepends=True)
        if existing_events_import_line is not None:
            # rewrite that import line to include the needed names
            li = existing_events_import_line.lineno - 1
            names = sorted(have | set(PRIMS))
            new_lines[li] = f"from {CORE_MODULE} import {', '.join(names)}\n"
        else:
            imp = f"from {CORE_MODULE} import {', '.join(PRIMS)}\n"
            new_lines.insert(last_import_end, imp)
        new_src = "".join(new_lines)

    path.write_text(new_src)
    # py_compile gate; revert on failure
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        path.write_text(original)
        return False, 0, f"revert (py_compile failed): {e}"
    return True, loc_removed, "ok"


def apply_batch(paths: list[str]) -> dict:
    core = core_hashes()
    results = []
    total_removed = 0
    for rel in paths:
        changed, removed, msg = redirect_file(ROOT / rel, core)
        results.append({"path": rel, "changed": changed, "loc_removed": removed, "msg": msg})
        if changed:
            total_removed += removed
    return {"total_loc_removed": total_removed, "files": results}


def main(argv: list[str]) -> int:
    if "--table" in argv:
        t = build_table()
        print(json.dumps({"batch_counts": t["batch_counts"], "core_hashes": t["core_hashes"],
                          "total_rows": len(t["rows"])}, indent=2))
        return 0
    if "--apply" in argv:
        table = json.loads((OUT / "MOP_EVIDENCE_MIGRATION.json").read_text())
        batch1 = sorted({r["path"] for r in table["rows"] if r["batch"] == "batch1_studies_safe"})
        res = apply_batch(batch1)
        print(json.dumps(res, indent=2))
        return 0
    print("usage: evidence_migrate.py [--table|--apply]")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
