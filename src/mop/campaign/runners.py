"""Unified campaign engine: the executable runner registry and node context.

Every node carries an ``entrypoint`` of the form ``"module:function"``. The scheduler imports it in a
worker process and calls it as ``runner(params, ctx) -> RunResult``. This is what makes a node genuinely
executable rather than a registry row: a node with no importable, callable runner cannot be marked
implemented.

Runners must be deterministic and byte-reproducible: they seal a canonical artifact and return its seal,
verdict, and whether it is a null. They must not read a wall clock into the sealed artifact.

House style: no em dashes and no en dashes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mop.substrate.events import canonical_bytes, canonical_sha256


@dataclass
class NodeContext:
    """What a runner is given: its node id, a private work directory, a deterministic seed, and params."""

    node_id: str
    workdir: Path
    seed: int
    proof_root: Path

    def seal_json(self, name: str, content: dict[str, Any]) -> tuple[Path, str]:
        """Seal a canonical JSON artifact under the proof root and return (path, sha256)."""

        seal = canonical_sha256(content)
        body = {**content, "seal": {"sha256": seal}}
        path = self.proof_root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes(body))
        return path, seal


@dataclass
class RunResult:
    """A runner's result: the sealed artifact path, its seal, a verdict, and whether it is a null."""

    artifact_path: str
    seal: str
    verdict: str
    is_null: bool
    detail: dict[str, Any] = field(default_factory=dict)


class RunnerError(RuntimeError):
    pass


_REGISTRY: dict[str, Callable[[dict[str, Any], NodeContext], RunResult]] = {}


def register_runner(key: str) -> Callable[[Callable[..., RunResult]], Callable[..., RunResult]]:
    """Decorator registering a runner under ``key`` (informational; entrypoints are resolved by import)."""

    def deco(fn: Callable[..., RunResult]) -> Callable[..., RunResult]:
        _REGISTRY[key] = fn
        return fn

    return deco


def resolve_entrypoint(entrypoint: str) -> Callable[[dict[str, Any], NodeContext], RunResult]:
    """Import and return the callable named by ``module:function``."""

    if ":" not in entrypoint:
        raise RunnerError(f"entrypoint must be 'module:function', got {entrypoint!r}")
    module_name, func_name = entrypoint.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 (surface any import failure as a runner error)
        raise RunnerError(f"cannot import {module_name!r}: {exc}") from exc
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise RunnerError(f"{entrypoint!r} does not name a callable")
    return fn  # type: ignore[return-value]


def entrypoint_is_runnable(entrypoint: str) -> bool:
    """True if the entrypoint imports to a callable. Used by the compliance verifier to prove a node has a
    real execution path, not just a spec row."""

    try:
        resolve_entrypoint(entrypoint)
        return True
    except RunnerError:
        return False
