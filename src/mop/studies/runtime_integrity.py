
from __future__ import annotations

import ast
import builtins
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

FORBIDDEN_MODEL_MODULES = (
    "transformers",
    "timm",
    "huggingface_hub",
    "safetensors",
    "vjepa",
    "mop.substrate.vjepa21_official",
    "torch.hub",
)


class ForbiddenRuntimeImport(RuntimeError):
    pass


def _forbidden(name: str, *, importer: str = "") -> bool:
    if (name == "torch.hub" or name.startswith("torch.hub.")) and importer.startswith("torch."):
        return False
    return any(name == prefix or name.startswith(f"{prefix}.") for prefix in FORBIDDEN_MODEL_MODULES)


@contextmanager
def deny_forbidden_runtime_imports() -> Iterator[list[str]]:

    attempts: list[str] = []
    real_import = builtins.__import__
    real_import_module = importlib.import_module

    def guarded_import(
        name: str,
        globals: dict[str, Any] | None = None,
        locals: dict[str, Any] | None = None,
        fromlist: tuple[str, ...] | list[str] = (),
        level: int = 0,
    ) -> Any:
        importer = str((globals or {}).get("__name__", ""))
        if level == 0 and _forbidden(name, importer=importer):
            attempts.append(name)
            raise ForbiddenRuntimeImport(f"forbidden model import attempted during preflight: {name}")
        return real_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name: str, package: str | None = None) -> Any:
        candidate = name if not name.startswith(".") else str(package or "")
        if _forbidden(candidate):
            attempts.append(name)
            raise ForbiddenRuntimeImport(f"forbidden model import attempted during preflight: {name}")
        return real_import_module(name, package)

    builtins.__import__ = guarded_import  # type: ignore[assignment]
    importlib.import_module = guarded_import_module
    try:
        yield attempts
    finally:
        importlib.import_module = real_import_module
        builtins.__import__ = real_import  # type: ignore[assignment]


def forbidden_source_imports(path: Path) -> list[str]:

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    problems: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
        elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            called = node.func
            if (
                isinstance(called, ast.Name)
                and called.id == "__import__"
                or (
                    isinstance(called, ast.Attribute)
                    and isinstance(called.value, ast.Name)
                    and called.value.id == "importlib"
                    and called.attr == "import_module"
                )
            ):
                names.append(str(node.args[0].value))
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "sys"
            and node.value.attr == "modules"
            and isinstance(node.slice, ast.Constant)
        ):
            names.append(str(node.slice.value))
        for name in names:
            if _forbidden(name):
                problems.append(f"line {getattr(node, 'lineno', 0)}: {name}")
    return problems
