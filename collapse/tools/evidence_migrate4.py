"""Generalized private-primitive redirector (spec sections 9, 21.6).

Many modules carry a privately named copy of a core primitive (for example _sha256 == core sha256_file,
_canonical_bytes == core canonical_bytes). This deletes each byte-identical private def, renames every use to
the core public name (word-boundary), imports it from mop.substrate.events, and verifies. Controller-adjacent
modules (studio/campaign_*, generation1_supervisor, telegram) are excluded and deferred to section 13.

Usage: evidence_migrate4.py <private_name> <core_public_name>

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import ast
import hashlib
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORE_MODULE = "mop.substrate.events"
CORE_FILE = ROOT / "src/mop/substrate/events.py"


def _defer(rel: str) -> bool:
    n = Path(rel).name
    return ("/studio/campaign_" in rel or n == "generation1_supervisor.py"
            or n == "telegram_rung_notifier.py")


class _Strip(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if (node.body and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)):
            node.body = node.body[1:] or [ast.Pass()]
        return node


def body_hash(fn: ast.AST) -> str:
    cleaned = _Strip().visit(ast.parse(ast.unparse(fn)))
    for n in ast.walk(cleaned):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n.name = "_"
    return hashlib.sha256(ast.dump(cleaned, annotate_fields=False).encode()).hexdigest()[:16]


def get_def(path: Path, name: str) -> ast.FunctionDef | None:
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    return None


def tracked_defining(name: str) -> list[str]:
    out = subprocess.run(["git", "grep", "-l", f"def {name}(", "--", "*.py"],
                         cwd=ROOT, capture_output=True, text=True).stdout
    return [x for x in out.splitlines() if x and not x.startswith("collapse/")]


def migrate(path: Path, old: str, new: str, core_h: str) -> tuple[bool, int, str]:
    original = path.read_text()
    fn = get_def(path, old)
    if fn is None:
        return False, 0, "skip: not module-level"
    if body_hash(fn) != core_h:
        return False, 0, "skip: body not byte-identical to core"
    start = fn.lineno
    for d in getattr(fn, "decorator_list", []):
        start = min(start, d.lineno)
    end = fn.end_lineno
    removed = end - start + 1
    lines = original.splitlines(keepends=True)
    kept = [ln for i, ln in enumerate(lines, 1) if not (start <= i <= end)]
    new_src = "".join(kept)
    # word-boundary rename of every remaining use of the private name
    new_src = re.sub(rf"\b{re.escape(old)}\b", new, new_src)
    # ensure import
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
    if new not in have:
        nl = new_src.splitlines(keepends=True)
        if existing is not None:
            nl[existing.lineno - 1] = f"from {CORE_MODULE} import {', '.join(sorted(have | {new}))}\n"
        else:
            nl.insert(last_import, f"from {CORE_MODULE} import {new}\n")
        new_src = "".join(nl)
    path.write_text(new_src)
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError as e:
        path.write_text(original)
        return False, 0, f"revert (compile): {str(e)[:80]}"
    if re.search(rf"\b{re.escape(old)}\b", new_src) or get_def(path, old) is not None:
        path.write_text(original)
        return False, 0, "revert (residual private name)"
    return True, removed, "ok"


def main(argv: list[str]) -> int:
    old, new = argv[0], argv[1]
    core_fn = get_def(CORE_FILE, new)
    if core_fn is None:
        print(f"error: core lacks {new}")
        return 1
    core_h = body_hash(core_fn)
    files = [f for f in tracked_defining(old) if not _defer(f)]
    deferred = [f for f in tracked_defining(old) if _defer(f)]
    results = []
    total = 0
    for rel in files:
        changed, removed, msg = migrate(ROOT / rel, old, new, core_h)
        results.append({"path": rel, "changed": changed, "loc_removed": removed, "msg": msg})
        if changed:
            total += removed
    ok = [r for r in results if r["changed"]]
    skipped = [r for r in results if not r["changed"]]
    print(json.dumps({"primitive": f"{old} -> {new}", "eligible_files": len(files),
                      "deferred_controller": deferred, "migrated": len(ok),
                      "total_loc_removed": total,
                      "skips": [f"{r['path']}: {r['msg']}" for r in skipped]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
