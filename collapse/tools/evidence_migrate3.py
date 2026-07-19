"""Batch 3: consolidate the duplicated _atomic_write helper into one public core primitive (section 9).

Adds atomic_write_json to the evidence core (renamed from the byte-identical private _atomic_write body),
then for each eligible script deletes the local _atomic_write def, renames its single call site to
atomic_write_json, and imports it from the core. No compatibility alias is left behind. Each file is
py_compiled and statically verified (no residual local def, symbol imported, call rewired); any failure
reverts that file. Core behavior is checked by a round-trip write/read/atomicity test in the caller.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_MODULE = "mop.substrate.events"
CORE_FILE = ROOT / "src/mop/substrate/events.py"
OLD = "_atomic_write"
NEW = "atomic_write_json"
MEMBERS = [
    "scripts/run_ecology_scaffold_batteries.py",
    "scripts/run_integrity_scaffold_drills.py",
    "scripts/run_material_twin_batteries.py",
    "scripts/verify_ecology_scaffold_batteries.py",
    "scripts/verify_integrity_scaffold_drills.py",
    "scripts/verify_material_twin_batteries.py",
]


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
    # name-insensitive: compare body/args structure only (the core public name differs from the private one)
    for node in ast.walk(cleaned):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.name = "_"
    return hashlib.sha256(ast.dump(cleaned, annotate_fields=False).encode()).hexdigest()[:16]


def get_def(path: Path, name: str) -> ast.FunctionDef | None:
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def ensure_core() -> int:
    """Add atomic_write_json (public) to the core if missing; return LOC added."""
    if get_def(CORE_FILE, NEW):
        return 0
    text = CORE_FILE.read_text()
    src_fn = get_def(ROOT / MEMBERS[0], OLD)
    body = ast.get_source_segment((ROOT / MEMBERS[0]).read_text(), src_fn)
    body = body.replace(f"def {OLD}(", f"def {NEW}(", 1)
    added = 0
    for imp in ("import os",):
        if not any(line.strip() == imp for line in text.splitlines()):
            tree = ast.parse(text)
            last = max((n.end_lineno or n.lineno) for n in tree.body
                       if isinstance(n, (ast.Import, ast.ImportFrom)))
            lines = text.splitlines(keepends=True)
            lines.insert(last, imp + "\n")
            text = "".join(lines)
            added += 1
    text = text + "\n\n" + body.rstrip() + "\n"
    CORE_FILE.write_text(text)
    added += body.rstrip().count("\n") + 1
    py_compile.compile(str(CORE_FILE), doraise=True)
    return added


def migrate(path: Path, core_h: str) -> tuple[bool, int, str]:
    original = path.read_text()
    fn = get_def(path, OLD)
    if fn is None:
        return False, 0, f"skip: {OLD} not module-level"
    if body_hash(fn) != core_h:
        return False, 0, f"skip: {OLD} body not byte-identical to core"
    start = fn.lineno
    for d in getattr(fn, "decorator_list", []):
        start = min(start, d.lineno)
    end = fn.end_lineno
    removed = end - start + 1
    lines = original.splitlines(keepends=True)
    kept = [ln for i, ln in enumerate(lines, 1) if not (start <= i <= end)]
    new_src = "".join(kept)
    # rename the (now sole) call site
    if f"{OLD}(" not in new_src:
        path.write_text(original)
        return False, 0, "skip: no call site found after def removal"
    new_src = new_src.replace(f"{OLD}(", f"{NEW}(")
    # add import from core
    tree = ast.parse(new_src)
    last_import = 0
    existing = None
    have = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE and node.level == 0:
            existing = node
            have |= {a.name for a in node.names}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = max(last_import, node.end_lineno or node.lineno)
    nl = new_src.splitlines(keepends=True)
    if NEW not in have:
        if existing is not None:
            nl[existing.lineno - 1] = f"from {CORE_MODULE} import {', '.join(sorted(have | {NEW}))}\n"
        else:
            nl.insert(last_import, f"from {CORE_MODULE} import {NEW}\n")
    new_src = "".join(nl)
    path.write_text(new_src)
    # static safety: no residual local def, symbol imported, no bare OLD token remains
    try:
        t2 = ast.parse(new_src)
        py_compile.compile(str(path), doraise=True)
    except Exception as e:
        path.write_text(original)
        return False, 0, f"revert (compile failed): {e}"
    if get_def(path, OLD) is not None or f"{OLD}(" in new_src:
        path.write_text(original)
        return False, 0, "revert (residual _atomic_write reference)"
    imported = any(isinstance(n, ast.ImportFrom) and n.module == CORE_MODULE
                   and any(a.name == NEW for a in n.names) for n in t2.body)
    if not imported:
        path.write_text(original)
        return False, 0, "revert (import not present)"
    return True, removed, "ok"


def main() -> int:
    added = ensure_core()
    core_h = body_hash(get_def(CORE_FILE, NEW))
    results = []
    total = 0
    for rel in MEMBERS:
        changed, removed, msg = migrate(ROOT / rel, core_h)
        results.append({"path": rel, "changed": changed, "loc_removed": removed, "msg": msg})
        if changed:
            total += removed
    print(json.dumps({"core_body_added_LOC": added, "total_loc_removed": total,
                      "net": total - added, "files": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
