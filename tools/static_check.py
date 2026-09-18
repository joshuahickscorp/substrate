"""Source-only inspection and honest physical line accounting.

This tool does not import Substrate, collect tests, run a model, start a service,
launch a sandbox or qualify an experiment. Syntax success is not runtime proof.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import symtable
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOTS = {"src", "tests", "tools", "configs", ".github"}
RUNTIME_FILES = {"pyproject.toml", "Makefile", "Dockerfile", ".dockerignore", ".gitignore"}
NON_AUTHORED_PREFIXES = (
    ".git/",
    ".venv/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "archive/",
    "artifacts/",
    "docs/archive/",
    "evidence/",
    "evidence-compact/",
    "inputs/",
    "logs/",
    "proof/",
    "run/",
    "runs/",
    "refactor/TRANSPLANT_",
)
MAGIC_GLOBALS = {"__name__", "__file__", "__package__", "__doc__", "__spec__", "__loader__", "__builtins__", "__annotations__"}


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def exports(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(alias.asname or alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(part.id for target in targets for part in ast.walk(target) if isinstance(part, ast.Name))
    return names


def review_calls(trees: dict) -> dict:
    """Resolve direct functions/constructors and self/class methods without importing modules."""
    modules = {path.stem: tree for path, tree in trees.items()}
    definitions = {name: {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
                   for name, tree in modules.items()}
    checked, total, errors = 0, 0, []
    for path, tree in trees.items():
        names = {name: (path.stem, name) for name in definitions[path.stem]}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[-1] in definitions:
                names.update({a.asname or a.name: (node.module.split('.')[-1], a.name) for a in node.names})
        def resolve(name):
            module, symbol = names.get(name, ('', ''))
            return definitions.get(module, {}).get(symbol)
        def visit(node, owner=None):
            nonlocal checked, total
            if isinstance(node, ast.ClassDef):
                owner = node
            if isinstance(node, ast.Call):
                total += 1
                target, bound = None, False
                if isinstance(node.func, ast.Name):
                    target = resolve(node.func.id)
                elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    base = owner if node.func.value.id in {'self', 'cls'} else resolve(node.func.value.id)
                    if isinstance(base, ast.ClassDef):
                        target = next((n for n in base.body if isinstance(n, ast.FunctionDef) and n.name == node.func.attr), None)
                        bound = bool(target) and not any(isinstance(d, ast.Name) and d.id == 'staticmethod' for d in target.decorator_list)
                if isinstance(target, ast.ClassDef):
                    cls = target
                    target = next((n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
                    bound = True
                    if target is None and any((isinstance(d, ast.Name) and d.id == 'dataclass') or
                            (isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass') for d in cls.decorator_list):
                        attrs = [n for n in cls.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)]
                        args = [ast.arg(arg=n.target.id) for n in attrs]
                        target = ast.FunctionDef(name=cls.name, args=ast.arguments(posonlyargs=[], args=args, kwonlyargs=[],
                            kw_defaults=[], defaults=[n.value for n in attrs if n.value is not None], vararg=None, kwarg=None),
                            body=[], decorator_list=[], returns=None, type_comment=None, type_params=[])
                        bound = False
                if isinstance(target, ast.FunctionDef):
                    checked += 1
                    spec = target.args
                    positional = [p.arg for p in (*spec.posonlyargs, *spec.args)][int(bound):]
                    required_count = max(0, len(positional) - len(spec.defaults))
                    keys = {k.arg for k in node.keywords if k.arg is not None}
                    star = any(isinstance(a, ast.Starred) for a in node.args)
                    spread = any(k.arg is None for k in node.keywords)
                    supplied = set(positional[:len(node.args)]) | keys
                    required = set(positional[:required_count]) | {a.arg for a, d in zip(spec.kwonlyargs, spec.kw_defaults) if d is None}
                    allowed = set(positional) | {a.arg for a in spec.kwonlyargs}
                    issues = []
                    if not spec.kwarg and keys - allowed:
                        issues.append('unknown keywords ' + repr(sorted(keys - allowed)))
                    if not star and not spec.vararg and len(node.args) > len(positional):
                        issues.append('too many positional arguments')
                    if not star and not spread and required - supplied:
                        issues.append('missing arguments ' + repr(sorted(required - supplied)))
                    if issues:
                        errors.append(f"{relative(path)}:{node.lineno}: {target.name}: {'; '.join(issues)}")
            for child in ast.iter_child_nodes(node):
                visit(child, owner)
        visit(tree)
    return {'calls_present': total, 'resolved_calls_inspected': checked, 'errors': errors,
            'scope': 'AST-only direct functions/constructors and self/class methods; no runtime/type or external API proof'}


def inspect() -> dict:
    paths = sorted(path for path in ROOT.rglob("*") if path.is_file() and "__pycache__" not in path.parts
                   and (path.relative_to(ROOT).parts[0] in RUNTIME_ROOTS
                        or (path.parent == ROOT and path.name in RUNTIME_FILES)))
    errors, inventory, trees, modules = [], [], {}, {}
    for path in paths:
        raw = path.read_bytes()
        inventory.append({"path": relative(path), "physical_lines": len(raw.splitlines()), "bytes": len(raw),
                          "sha256": hashlib.sha256(raw).hexdigest()})
        if path.suffix == ".py":
            try:
                source = raw.decode("utf-8")
                tree = ast.parse(source, filename=relative(path))
                compile(tree, relative(path), "exec")  # Compile a code object; never execute it.
                trees[path] = tree
                if path.is_relative_to(ROOT / "src"):
                    module = path.relative_to(ROOT / "src").with_suffix("").as_posix().replace("/", ".")
                    modules[module.removesuffix(".__init__")] = exports(tree)
                table = symtable.symtable(source, relative(path), "exec")
                bound = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
                bound |= set(dir(builtins)) | MAGIC_GLOBALS

                def symbols(scope):
                    for symbol in scope.get_symbols():
                        if symbol.is_referenced() and symbol.is_global() and symbol.get_name() not in bound:
                            errors.append(f"{relative(path)}:{scope.get_lineno()}: unresolved global {symbol.get_name()}")
                    for child in scope.get_children():
                        symbols(child)
                symbols(table)
            except (SyntaxError, ValueError, UnicodeError) as exc:
                errors.append(f"{relative(path)}: {exc}")
        elif path.suffix == ".json":
            try:
                json.loads(raw)
            except ValueError as exc:
                errors.append(f"{relative(path)}: {exc}")
        elif path.suffix == ".toml":
            try:
                tomllib.loads(raw.decode())
            except (ValueError, UnicodeError) as exc:
                errors.append(f"{relative(path)}: {exc}")
    for path, tree in trees.items():
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom):
                continue
            module = node.module or ""
            if node.level and path.is_relative_to(ROOT / "src"):
                parent = path.relative_to(ROOT / "src").parent.parts
                prefix = parent[:len(parent) - node.level + 1]
                module = ".".join((*prefix, module)).rstrip(".")
            if module == "substrate" or module.startswith("substrate."):
                if module not in modules:
                    errors.append(f"{relative(path)}:{node.lineno}: missing internal module {module}")
                    continue
                for alias in node.names:
                    if alias.name != "*" and alias.name not in modules[module] and f"{module}.{alias.name}" not in modules:
                        errors.append(f"{relative(path)}:{node.lineno}: missing export {module}.{alias.name}")
    frontier_path = ROOT / "configs" / "research_frontier.json"
    if frontier_path.exists():
        frontier = json.loads(frontier_path.read_text())
        source_trees = {p.stem: tree for p, tree in trees.items() if p.parent == ROOT / "src" / "substrate"}
        for entry in frontier["frontiers"]:
            for symbol in entry["implementation_symbols"]:
                parts = symbol.split(".")
                tree = source_trees.get(parts[0])
                if tree is None or parts[1] not in exports(tree):
                    errors.append(f"frontier {entry['id']}: missing symbol {symbol}")
                elif len(parts) == 3:
                    owner = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == parts[1]), None)
                    if owner is None or not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == parts[2] for n in owner.body):
                        errors.append(f"frontier {entry['id']}: missing method {symbol}")
    calls = review_calls(trees)
    errors.extend(calls["errors"])
    evidence_manifest = json.loads((ROOT / "refactor" / "EVIDENCE_MANIFEST.json").read_text())
    historical_paths = {row["path"] for row in evidence_manifest["files"]}
    def is_non_authored(path: Path) -> bool:
        rel = relative(path)
        return rel in historical_paths or any(rel.startswith(prefix) for prefix in NON_AUTHORED_PREFIXES)

    authored_paths = [p for p in ROOT.rglob("*") if p.is_file() and p.name != ".DS_Store" and not is_non_authored(p)
                      and "__pycache__" not in p.relative_to(ROOT).parts]
    authored_lines = sum(len(p.read_bytes().splitlines()) for p in authored_paths)
    if authored_lines > 10000:
        errors.append(f"Entire authored tree has {authored_lines} lines; ceiling includes docs and inventories")
    categories = {}
    for row in inventory:
        category = row["path"].split("/")[0] if "/" in row["path"] else "operational"
        categories[category] = categories.get(category, 0) + row["physical_lines"]
    total = sum(row["physical_lines"] for row in inventory)
    if total > 10000:
        errors.append(f"Physical runtime tree has {total} lines; limit is 10000 including tests and configuration")
    test_definitions = sum(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
                           for path, tree in trees.items() if path.is_relative_to(ROOT / "tests") for node in ast.walk(tree))
    return {"schema": "substrate-static-inspection-v3", "inspector_python": sys.version.split()[0],
            "status": "source-inspection-clean" if not errors else "source-inspection-errors",
            "errors": errors, "call_review": calls, "authored_lines_excluding_historical_evidence": authored_lines,
            "authored_file_count": len(authored_paths), "python_files_parsed_and_compiled_not_executed": len(trees),
            "test_definitions_present_not_collected": test_definitions, "physical_runtime_lines": total,
            "physical_lines_by_category": categories, "runtime_bytes": sum(row["bytes"] for row in inventory),
            "runtime_file_count": len(inventory), "file_inventory": inventory,
            "tests_run": 0, "tests_collected": 0, "odyssey_runs": 0, "model_calls": 0,
            "limits": ["No behavioral equivalence, type-system soundness or test pass is inferred",
                       "No dynamic import, dependency installation, SQL execution or service exercise",
                       "Internal imports and simple global bindings inspected; not a complete linter or type checker"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = inspect()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(json.dumps({key: value for key, value in report.items() if key != "file_inventory"}, indent=2))
    raise SystemExit(bool(report["errors"]))


if __name__ == "__main__":
    main()
