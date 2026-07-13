"""Deterministically build exploratory-only Generation 1 injection requests.

This module prepares a request file; it never submits the request or mutates supervisor state.
The live queue identity may be read from a sealed status artifact, or supplied explicitly for
offline construction. Capsules can be loaded from pre-sealed JSON or built through the CLI's
bounded single-capsule argument form.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import REPO_ROOT
from . import generation1_supervisor as supervisor

ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ACTION = "append-capsules"


class InjectionBuildError(ValueError):
    """An injection request cannot be constructed without weakening its contract."""


@dataclass(frozen=True, slots=True)
class InjectionContext:
    program_id: str
    sequence: int
    expected_queue_head_sha256: str
    created_at: str
    existing_capsule_ids: frozenset[str] = frozenset()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InjectionBuildError(message)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        stat = path.lstat()
    except OSError as exc:
        raise InjectionBuildError(f"{label} is unreadable: {path}") from exc
    _require(path.is_file() and not path.is_symlink(), f"{label} must be a regular non-symlink file")
    _require(0 < stat.st_size <= supervisor.MAX_JSON_BYTES, f"{label} byte envelope is invalid")
    try:
        raw = path.read_bytes()
        _require(len(raw) == stat.st_size, f"{label} changed while being read")
        value = json.loads(raw)
    except OSError as exc:
        raise InjectionBuildError(f"{label} is unreadable: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InjectionBuildError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise InjectionBuildError(f"{label} must be a JSON object")
    return value


def _normalized_timestamp(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InjectionBuildError("created_at must be a timestamp string")
    rendered = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(rendered)
    except ValueError as exc:
        raise InjectionBuildError("created_at must be an ISO-8601 timestamp") from exc
    _require(parsed.tzinfo is not None, "created_at must include a UTC offset")
    return parsed.astimezone(UTC).isoformat()


def _validate_identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or ID_RE.fullmatch(value) is None:
        raise InjectionBuildError(f"{label} is invalid")
    return value


def _validate_digest(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        raise InjectionBuildError(f"{label} must be a lowercase SHA-256")
    return value


def _sealed(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    core = dict(payload)
    core.pop(field, None)
    return {**core, field: supervisor.canonical_sha256(core)}


def _valid_seal(payload: Mapping[str, Any], field: str) -> bool:
    core = dict(payload)
    declared = core.pop(field, None)
    return isinstance(declared, str) and declared == supervisor.canonical_sha256(core)


def context_from_status_payload(status: Mapping[str, Any]) -> InjectionContext:
    """Extract the exact append boundary from a self-hashed supervisor status payload."""

    _require(status.get("schema") == supervisor.STATUS_SCHEMA, "status schema is not Generation 1")
    _require(_valid_seal(status, "status_sha256"), "status self-seal mismatch")
    program_id = _validate_identifier(status.get("program_id"), "status program_id")
    sequence = status.get("next_injection_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 1:
        raise InjectionBuildError("status next_injection_sequence must be positive")
    head = _validate_digest(status.get("queue_head_sha256"), "status queue head")
    capsules = status.get("capsules", {})
    _require(isinstance(capsules, Mapping), "status capsules must be an object")
    capsule_ids = frozenset(
        _validate_identifier(capsule_id, "status capsule id") for capsule_id in capsules
    )
    return InjectionContext(
        program_id=program_id,
        sequence=sequence,
        expected_queue_head_sha256=head,
        created_at=_normalized_timestamp(status.get("created_at")),
        existing_capsule_ids=capsule_ids,
    )


def context_from_status(path: Path) -> InjectionContext:
    return context_from_status_payload(_read_object(path, "Generation 1 status"))


def context_from_program(
    program_path: Path,
    *,
    repo_root: Path = REPO_ROOT,
) -> InjectionContext:
    """Load the governed program and its current sealed status without changing either."""

    try:
        program = supervisor.load_program(program_path, repo_root=repo_root)
        status = supervisor.read_status(program)
    except (OSError, ValueError, supervisor.Generation1Refused) as exc:
        raise InjectionBuildError(f"cannot derive live injection context: {exc}") from exc
    return context_from_status_payload(status)


def validate_exploratory_capsule(
    raw: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    label: str = "injection capsule",
) -> dict[str, Any]:
    """Validate the supervisor's complete capsule contract and its injection-only kind rule."""

    _require(raw.get("kind") == "exploratory", f"{label} must have kind 'exploratory'")
    capsule = copy.deepcopy(dict(raw))
    try:
        supervisor._parse_capsule(capsule, repo_root.resolve(), label, injectable=True)
    except (OSError, ValueError, supervisor.Generation1Refused) as exc:
        raise InjectionBuildError(f"{label} is invalid: {exc}") from exc
    return capsule


def load_capsule_json(path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    return validate_exploratory_capsule(
        _read_object(path, "capsule JSON"),
        repo_root=repo_root,
        label=f"capsule {path}",
    )


def _authority(path: Path | str, repo_root: Path) -> dict[str, str]:
    root = repo_root.resolve()
    candidate = Path(path)
    source = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    _require(source.is_relative_to(root), f"authority escapes repository: {path}")
    _require(source.is_file() and not source.is_symlink(), f"authority is not a regular file: {path}")
    return {
        "path": str(source.relative_to(root)),
        "sha256": supervisor.sha256_file(source),
    }


def build_exploratory_capsule(
    *,
    repo_root: Path,
    capsule_id: str,
    command: Sequence[str],
    artifact_path: str,
    artifact_schema: str,
    artifact_fields: Mapping[str, Any] | None = None,
    artifact_seal_field: str | None = "payload_sha256",
    authority_paths: Sequence[Path | str] = (),
    priority: int = 100,
    depends_on: Sequence[str] = (),
    cwd: str = ".",
    environment: Mapping[str, str] | None = None,
    lane: str = "cpu",
    accelerator: str = "none",
    cpu_cores: int = 1,
    estimated_unified_memory_gb: float = 2.0,
    estimated_mps_gb: float = 0.0,
    resource_basis: str = "bounded operator-requested exploratory Generation 1 injection",
    forecast_write_gb: float = 0.25,
    atomic_write_gb: float = 0.1,
    wall_minutes: int = 60,
    process_marker: str | None = None,
) -> dict[str, Any]:
    """Build one sealed capsule while fixing its kind to ``exploratory``."""

    argv = list(command)
    _require(bool(argv) and all(isinstance(item, str) and bool(item) for item in argv), "command is invalid")
    marker = process_marker or (argv[1] if len(argv) > 1 else "")
    _require(bool(marker), "process_marker is required when command has no script argument")
    sources: Sequence[Path | str] = authority_paths or (marker,)
    authorities_by_path = {
        authority["path"]: authority
        for authority in (_authority(path, repo_root) for path in sources)
    }
    core: dict[str, Any] = {
        "schema": supervisor.CAPSULE_SCHEMA,
        "id": capsule_id,
        "kind": "exploratory",
        "priority": priority,
        "depends_on": sorted(depends_on),
        "command": argv,
        "cwd": cwd,
        "environment": dict(sorted((environment or {}).items())),
        "resources": {
            "lane": lane,
            "accelerator": accelerator,
            "cpu_cores": cpu_cores,
            "estimated_unified_memory_gb": estimated_unified_memory_gb,
            "estimated_mps_gb": estimated_mps_gb,
            "resource_basis": resource_basis.strip(),
            "forecast_write_gb": forecast_write_gb,
            "atomic_write_gb": atomic_write_gb,
            "wall_minutes": wall_minutes,
            "process_marker": marker,
        },
        "artifacts": [
            {
                "path": artifact_path,
                "schema": artifact_schema,
                "fields": dict(artifact_fields or {}),
                "seal_field": artifact_seal_field,
            }
        ],
        "authorities": [authorities_by_path[path] for path in sorted(authorities_by_path)],
    }
    return validate_exploratory_capsule(
        _sealed(core, "capsule_sha256"),
        repo_root=repo_root,
        label=f"capsule {capsule_id}",
    )


def build_injection_request(
    *,
    context: InjectionContext,
    capsules: Sequence[Mapping[str, Any]],
    reason: str,
    repo_root: Path = REPO_ROOT,
    injection_id: str | None = None,
) -> dict[str, Any]:
    """Build a canonical request bound to one exact supervisor queue boundary."""

    program_id = _validate_identifier(context.program_id, "program_id")
    _require(
        not isinstance(context.sequence, bool)
        and isinstance(context.sequence, int)
        and context.sequence >= 1,
        "sequence must be positive",
    )
    head = _validate_digest(context.expected_queue_head_sha256, "expected queue head")
    normalized_reason = reason.strip()
    _require(bool(normalized_reason), "reason must be nonempty")
    rows = [
        validate_exploratory_capsule(row, repo_root=repo_root, label=f"capsules[{index}]")
        for index, row in enumerate(capsules)
    ]
    _require(bool(rows), "at least one exploratory capsule is required")
    rows.sort(key=lambda row: str(row["id"]))
    ids = [str(row["id"]) for row in rows]
    _require(len(ids) == len(set(ids)), "injection capsule ids must be unique")
    collisions = sorted(set(ids) & context.existing_capsule_ids)
    _require(not collisions, f"injection capsule ids already exist: {collisions}")
    if injection_id is None:
        identity = supervisor.canonical_sha256(
            {
                "program_id": program_id,
                "sequence": context.sequence,
                "reason": normalized_reason,
                "capsules": [row["capsule_sha256"] for row in rows],
            }
        )[:12]
        injection_id = f"inj-{context.sequence:06d}-{identity}"
    request_id = _validate_identifier(injection_id, "injection_id")
    core = {
        "schema": supervisor.INJECTION_SCHEMA,
        "program_id": program_id,
        "injection_id": request_id,
        "sequence": context.sequence,
        "created_at": _normalized_timestamp(context.created_at),
        "action": ACTION,
        "expected_queue_head_sha256": head,
        "capsules": rows,
        "reason": normalized_reason,
    }
    return _sealed(core, "injection_sha256")


def write_injection_request(path: Path, request: Mapping[str, Any]) -> None:
    """Write once, or accept an exact deterministic rerun at the same path."""

    if path.exists():
        existing = _read_object(path, "existing injection request")
        _require(existing == dict(request), f"existing injection request differs: {path}")
        return
    supervisor.atomic_write_json(path, request)


def _json_argument(value: str, label: str, expected: type) -> Any:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InjectionBuildError(f"{label} is not valid JSON") from exc
    _require(isinstance(parsed, expected), f"{label} must decode to {expected.__name__}")
    return parsed


def _environment(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        key, separator, item = value.partition("=")
        _require(bool(separator) and bool(key) and "=" not in key, f"invalid environment entry: {value}")
        result[key] = item
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    context = parser.add_mutually_exclusive_group()
    context.add_argument("--program", type=Path)
    context.add_argument("--status", type=Path)
    parser.add_argument("--program-id")
    parser.add_argument("--sequence", type=int)
    parser.add_argument("--expected-head")
    parser.add_argument("--created-at")
    parser.add_argument("--injection-id")
    parser.add_argument("--reason", required=True)

    parser.add_argument("--capsule-json", type=Path, action="append", default=[])
    parser.add_argument("--capsule-id")
    parser.add_argument("--command-json")
    parser.add_argument("--artifact-path")
    parser.add_argument("--artifact-schema")
    parser.add_argument("--artifact-fields-json", default="{}")
    parser.add_argument("--artifact-seal-field", default="payload_sha256")
    parser.add_argument("--unsealed-artifact", action="store_true")
    parser.add_argument("--authority", action="append", default=[])
    parser.add_argument("--priority", type=int, default=100)
    parser.add_argument("--depends-on", action="append", default=[])
    parser.add_argument("--cwd", default=".")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("--lane", choices=sorted(supervisor.LANES), default="cpu")
    parser.add_argument("--accelerator", choices=sorted(supervisor.ACCELERATORS), default="none")
    parser.add_argument("--cpu-cores", type=int, default=1)
    parser.add_argument("--memory-gb", type=float, default=2.0)
    parser.add_argument("--mps-gb", type=float, default=0.0)
    parser.add_argument(
        "--resource-basis",
        default="bounded operator-requested exploratory Generation 1 injection",
    )
    parser.add_argument("--forecast-write-gb", type=float, default=0.25)
    parser.add_argument("--atomic-write-gb", type=float, default=0.1)
    parser.add_argument("--wall-minutes", type=int, default=60)
    parser.add_argument("--process-marker")
    return parser


def _explicit_context(arguments: argparse.Namespace) -> InjectionContext:
    missing = [
        name
        for name in ("program_id", "sequence", "expected_head", "created_at")
        if getattr(arguments, name) is None
    ]
    _require(not missing, f"explicit context is missing: {missing}")
    return InjectionContext(
        program_id=arguments.program_id,
        sequence=arguments.sequence,
        expected_queue_head_sha256=arguments.expected_head,
        created_at=arguments.created_at,
    )


def _resolve_context(arguments: argparse.Namespace, repo_root: Path) -> InjectionContext:
    if arguments.program is not None or arguments.status is not None:
        explicit = (arguments.program_id, arguments.sequence, arguments.expected_head)
        _require(
            not any(value is not None for value in explicit),
            "live context cannot be mixed with chain fields",
        )
        context = (
            context_from_program(arguments.program.resolve(), repo_root=repo_root)
            if arguments.program is not None
            else context_from_status(arguments.status)
        )
        if arguments.created_at:
            return replace(context, created_at=_normalized_timestamp(arguments.created_at))
        return context
    return _explicit_context(arguments)


def _resolve_capsules(arguments: argparse.Namespace, repo_root: Path) -> list[dict[str, Any]]:
    if arguments.capsule_json:
        direct_values = (
            arguments.capsule_id,
            arguments.command_json,
            arguments.artifact_path,
            arguments.artifact_schema,
            arguments.process_marker,
            *arguments.authority,
            *arguments.depends_on,
            *arguments.env,
        )
        _require(
            not any(value is not None for value in direct_values),
            "--capsule-json cannot be mixed with direct capsule arguments",
        )
        return [load_capsule_json(path, repo_root=repo_root) for path in arguments.capsule_json]
    required = {
        "capsule_id": arguments.capsule_id,
        "command_json": arguments.command_json,
        "artifact_path": arguments.artifact_path,
        "artifact_schema": arguments.artifact_schema,
    }
    missing = sorted(name for name, value in required.items() if value is None)
    _require(not missing, f"direct capsule arguments are missing: {missing}")
    command = _json_argument(arguments.command_json, "--command-json", list)
    fields = _json_argument(arguments.artifact_fields_json, "--artifact-fields-json", dict)
    authorities: Sequence[Path | str] = arguments.authority
    return [
        build_exploratory_capsule(
            repo_root=repo_root,
            capsule_id=arguments.capsule_id,
            command=command,
            artifact_path=arguments.artifact_path,
            artifact_schema=arguments.artifact_schema,
            artifact_fields=fields,
            artifact_seal_field=None if arguments.unsealed_artifact else arguments.artifact_seal_field,
            authority_paths=authorities,
            priority=arguments.priority,
            depends_on=arguments.depends_on,
            cwd=arguments.cwd,
            environment=_environment(arguments.env),
            lane=arguments.lane,
            accelerator=arguments.accelerator,
            cpu_cores=arguments.cpu_cores,
            estimated_unified_memory_gb=arguments.memory_gb,
            estimated_mps_gb=arguments.mps_gb,
            resource_basis=arguments.resource_basis,
            forecast_write_gb=arguments.forecast_write_gb,
            atomic_write_gb=arguments.atomic_write_gb,
            wall_minutes=arguments.wall_minutes,
            process_marker=arguments.process_marker,
        )
    ]


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        repo_root = arguments.repo_root.resolve()
        _require(repo_root.is_dir(), f"repository root is not a directory: {repo_root}")
        context = _resolve_context(arguments, repo_root)
        capsules = _resolve_capsules(arguments, repo_root)
        request = build_injection_request(
            context=context,
            capsules=capsules,
            reason=arguments.reason,
            repo_root=repo_root,
            injection_id=arguments.injection_id,
        )
        output = arguments.out.resolve()
        write_injection_request(output, request)
        print(
            json.dumps(
                {
                    "built": True,
                    "submitted": False,
                    "path": str(output),
                    "program_id": request["program_id"],
                    "injection_id": request["injection_id"],
                    "sequence": request["sequence"],
                    "capsule_ids": [row["id"] for row in request["capsules"]],
                    "injection_sha256": request["injection_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    except (InjectionBuildError, OSError, ValueError, supervisor.Generation1Refused) as exc:
        print(json.dumps({"built": False, "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2


__all__ = [
    "ACTION",
    "InjectionBuildError",
    "InjectionContext",
    "build_exploratory_capsule",
    "build_injection_request",
    "build_parser",
    "context_from_program",
    "context_from_status",
    "context_from_status_payload",
    "load_capsule_json",
    "main",
    "validate_exploratory_capsule",
    "write_injection_request",
]
