"""Primitive-general evidence-core migrator (spec sections 9, 21.6).

Redirects a single integrity primitive's byte-identical duplicates onto the one evidence core
(mop.substrate.events), adding the canonical body to the core once if it is missing. Every eligible file is
verified byte-identical to the canonical body at edit time, then py_compiled with automatic revert on failure.

Usage: evidence_migrate2.py <primitive> <member1.py> <member2.py> ...
Only pass members that are already known byte-identical and eligible (not verifier or controller adjacent).

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import py_compile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_MODULE = "mop.substrate.events"
CORE_FILE = ROOT / "src/mop/substrate/events.py"
# extra imports the core file needs for a given primitive body (kept minimal and explicit)
CORE_IMPORT_NEEDS = {"sha256_file": ["from pathlib import Path"]}


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


def get_def(path: Path, name: str) -> ast.FunctionDef | None:
    tree = ast.parse(path.read_text())
    for n in tree.body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def core_has(name: str) -> bool:
    return get_def(CORE_FILE, name) is not None


def ensure_core_defines(name: str, body_source: str) -> int:
    """Append the canonical primitive body (and any needed imports) to the core once. Returns LOC added."""
    if core_has(name):
        return 0
    text = CORE_FILE.read_text()
    added = 0
    for imp in CORE_IMPORT_NEEDS.get(name, []):
        if imp not in text:
            # insert after the last top-level import
            tree = ast.parse(text)
            last = 0
            for n in tree.body:
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    last = max(last, n.end_lineno or n.lineno)
            lines = text.splitlines(keepends=True)
            lines.insert(last, imp + "\n")
            text = "".join(lines)
            added += 1
    block = "\n\n" + body_source.rstrip() + "\n"
    text = text + block
    CORE_FILE.write_text(text)
    added += body_source.rstrip().count("\n") + 1
    py_compile.compile(str(CORE_FILE), doraise=True)
    return added


def redirect(path: Path, name: str, core_h: str) -> tuple[bool, int, str]:
    original = path.read_text()
    fn = get_def(path, name)
    if fn is None:
        return False, 0, f"skip: {name} not module-level"
    if body_hash(fn) != core_h:
        return False, 0, f"skip: {name} body not byte-identical to core"
    start = fn.lineno
    for d in getattr(fn, "decorator_list", []):
        start = min(start, d.lineno)
    end = fn.end_lineno
    loc_removed = end - start + 1
    lines = original.splitlines(keepends=True)
    kept = [ln for i, ln in enumerate(lines, 1) if not (start <= i <= end)]
    new_src = "".join(kept)
    # ensure import of the primitive from the core
    tree = ast.parse(new_src)
    have = set()
    existing = None
    last_import = 0
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == CORE_MODULE and node.level == 0:
            existing = node
            have |= {a.name for a in node.names}
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            last_import = max(last_import, node.end_lineno or node.lineno)
    if name not in have:
        nl = new_src.splitlines(keepends=True)
        if existing is not None:
            names = sorted(have | {name})
            nl[existing.lineno - 1] = f"from {CORE_MODULE} import {', '.join(names)}\n"
        else:
            nl.insert(last_import, f"from {CORE_MODULE} import {name}\n")
        new_src = "".join(nl)
    path.write_text(new_src)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        path.write_text(original)
        return False, 0, f"revert (py_compile failed): {e}"
    return True, loc_removed, "ok"


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: evidence_migrate2.py <primitive> <member.py> ...")
        return 1
    name, members = argv[0], argv[1:]
    # canonical body: take the byte-identical body from the first member (all verified identical downstream)
    src_fn = get_def(ROOT / members[0], name)
    if src_fn is None:
        print(f"error: {name} not found in {members[0]}")
        return 1
    body_source = ast.get_source_segment((ROOT / members[0]).read_text(), src_fn)
    added = ensure_core_defines(name, body_source)
    core_h = body_hash(get_def(CORE_FILE, name))
    results = []
    total_removed = 0
    for rel in members:
        changed, removed, msg = redirect(ROOT / rel, name, core_h)
        results.append({"path": rel, "changed": changed, "loc_removed": removed, "msg": msg})
        if changed:
            total_removed += removed
    import json
    print(json.dumps({"primitive": name, "core_body_added_LOC": added,
                      "total_loc_removed": total_removed,
                      "net": total_removed - added, "files": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
