"""Honest repository-shape accounting and offline pack hydration for MOP.

This module is deliberately independent of the training/runtime dependency graph.
It measures the Git-tracked checkout, verifies frozen campaign bindings, and
hydrates checksum-locked directory packs from an offline cache. It never downloads
code and it never treats relocation as elimination.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

CONTRACT_SCHEMA = "mop-extreme-condensation-contract/v1"
BASELINE_SCHEMA = "mop-extreme-condensation-baseline/v1"
PACK_LOCK_SCHEMA = "mop-condensation-pack-lock/v2"
PACK_MANIFEST_SCHEMA = "mop-condensation-pack-manifest/v1"
LIVE_BINDINGS_SCHEMA = "mop-condensation-live-bindings/v1"
ELIMINATION_LEDGER_SCHEMA = "mop-condensation-elimination-ledger/v2"
RUN_RECEIPT_SCHEMA = "mop-extreme-condensation-run-receipt/v1"
MAX_CONTROL_BYTES = 32 * 1024 * 1024
MAX_CHAIN_DEPTH = 10_000
PACK_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}")
PACK_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")


class CondensationError(RuntimeError):
    """The condensation contract, repository, pack, or receipt is invalid."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise CondensationError(f"control document may not be a symlink: {path}")
    try:
        source = path.resolve(strict=True)
        if not source.is_file() or source.stat().st_size > MAX_CONTROL_BYTES:
            raise CondensationError(f"unsafe or oversized control document: {source}")
        value = json.loads(source.read_bytes())
    except OSError as exc:
        raise CondensationError(f"missing or unreadable control document: {path}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CondensationError(f"invalid JSON {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise CondensationError(f"JSON root is not an object: {source}")
    return value


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sealed_document(path: Path, *, schema: str, seal_field: str) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != schema:
        raise CondensationError(f"unexpected schema in {path}: {value.get('schema')!r}")
    declared = value.get(seal_field)
    core = {key: item for key, item in value.items() if key != seal_field}
    if not isinstance(declared, str) or declared != canonical_sha256(core):
        raise CondensationError(f"self-seal mismatch in {path}")
    return value


def _safe_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise CondensationError(f"{field} must be a non-empty relative path")
    raw_parts = value.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise CondensationError(f"{field} contains an unsafe path component: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise CondensationError(f"{field} escapes its declared root: {value!r}")
    return Path(*path.parts)


def _pack_id(value: Any, field: str = "pack_id") -> str:
    if not isinstance(value, str) or PACK_ID_PATTERN.fullmatch(value) is None:
        raise CondensationError(f"{field} has an invalid identifier")
    return value


def _pack_version(value: Any, field: str = "version") -> str:
    if not isinstance(value, str) or PACK_VERSION_PATTERN.fullmatch(value) is None:
        raise CondensationError(f"{field} has an invalid version")
    return value


def load_contract(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema") != CONTRACT_SCHEMA:
        raise CondensationError(f"unexpected condensation contract schema: {value.get('schema')!r}")

    target = value.get("recommended_target_active_repo_loc")
    ladder = value.get("fallback_ladder")
    if (
        isinstance(target, bool)
        or not isinstance(target, int)
        or target <= 0
        or not isinstance(ladder, list)
        or not ladder
        or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in ladder)
        or ladder != sorted(set(ladder), reverse=True)
        or ladder[-1] != target
    ):
        raise CondensationError("fallback_ladder must be unique, descending, and end at the target")
    stretch = value.get("stretch_target_runtime_core_loc")
    if isinstance(stretch, bool) or not isinstance(stretch, int) or stretch <= 0 or stretch > target:
        raise CondensationError("runtime-core stretch target must be positive and no larger than target")

    measurement = value.get("measurement")
    if not isinstance(measurement, dict):
        raise CondensationError("measurement must be an object")
    surfaces = measurement.get("surfaces")
    if not isinstance(surfaces, list) or not surfaces:
        raise CondensationError("measurement.surfaces must be a non-empty list")
    names: set[str] = set()
    defaults = 0
    for row in surfaces:
        if not isinstance(row, dict):
            raise CondensationError("each measurement surface must be an object")
        name = row.get("name")
        include = row.get("include")
        if not isinstance(name, str) or not name or name in names:
            raise CondensationError("measurement surface names must be unique non-empty strings")
        if not isinstance(row.get("active"), bool):
            raise CondensationError(f"surface {name!r} requires an active boolean")
        if row.get("default") is True:
            defaults += 1
        elif (
            not isinstance(include, list)
            or not include
            or not all(isinstance(pattern, str) and pattern for pattern in include)
        ):
            raise CondensationError(f"surface {name!r} requires include patterns")
        exclude = row.get("exclude", [])
        if not isinstance(exclude, list) or not all(isinstance(pattern, str) for pattern in exclude):
            raise CondensationError(f"surface {name!r} has invalid exclude patterns")
        names.add(name)
    if defaults != 1 or surfaces[-1].get("default") is not True:
        raise CondensationError("exactly one final default measurement surface is required")

    max_line = measurement.get("max_active_line_bytes")
    if isinstance(max_line, bool) or not isinstance(max_line, int) or max_line < 200:
        raise CondensationError("measurement.max_active_line_bytes must be an integer >= 200")

    planned = value.get("planned_packs", [])
    if not isinstance(planned, list):
        raise CondensationError("planned_packs must be a list")
    pack_ids: set[str] = set()
    for row in planned:
        if not isinstance(row, dict):
            raise CondensationError("each planned pack requires pack_id")
        pack_id = _pack_id(row.get("pack_id"), "planned_packs.pack_id")
        if pack_id in pack_ids:
            raise CondensationError(f"duplicate planned pack: {pack_id}")
        pack_ids.add(pack_id)
    profile = value.get("elimination_gate_profile")
    gates = value.get("gates")
    commands = gates.get(profile) if isinstance(gates, dict) and isinstance(profile, str) else None
    if not isinstance(commands, list) or not commands:
        raise CondensationError("elimination_gate_profile must name a non-empty valid gate profile")
    gate_ids: list[str] = []
    for row in commands:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not row["id"]
            or not isinstance(row.get("argv"), list)
            or not row["argv"]
            or not all(isinstance(item, str) and item for item in row["argv"])
        ):
            raise CondensationError("elimination_gate_profile must name a non-empty valid gate profile")
        gate_ids.append(row["id"])
    if len(gate_ids) != len(set(gate_ids)):
        raise CondensationError("elimination gate IDs must be unique")
    return value


def _git(root: Path, argv: Sequence[str]) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *argv],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CondensationError(f"git {' '.join(argv)} failed: {detail[:400]}")
    return result.stdout


def tracked_files(root: Path) -> list[str]:
    raw = _git(root, ["ls-files", "-z"])
    paths = [item.decode("utf-8") for item in raw.split(b"\x00") if item]
    if len(paths) != len(set(paths)):
        raise CondensationError("git returned duplicate tracked paths")
    return sorted(paths)


def commit_files(root: Path, commit: str) -> list[str]:
    raw = _git(root, ["ls-tree", "-r", "--name-only", "-z", commit])
    return sorted(item.decode("utf-8") for item in raw.split(b"\x00") if item)


def untracked_files(root: Path) -> list[str]:
    raw = _git(root, ["ls-files", "--others", "--exclude-standard", "-z"])
    return sorted(item.decode("utf-8") for item in raw.split(b"\x00") if item)


def git_head(root: Path) -> str:
    return _git(root, ["rev-parse", "HEAD"]).decode("ascii").strip()


def git_index_tree(root: Path) -> str:
    return _git(root, ["write-tree"]).decode("ascii").strip()


def git_status(root: Path) -> list[str]:
    raw = _git(root, ["status", "--porcelain=v1", "-z"])
    return [item.decode("utf-8", errors="replace") for item in raw.split(b"\x00") if item]


def git_is_ancestor(root: Path, ancestor: str, descendant: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", ancestor, descendant],
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CondensationError(f"git ancestry check failed: {detail[:400]}")
    return result.returncode == 0


def _pytest_context(collection_root: Path, project_root: Path) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    search_paths: list[Path] = []
    for path in (
        collection_root / "src",
        collection_root,
        project_root / "src",
        project_root,
    ):
        resolved = path.resolve()
        if resolved not in search_paths:
            search_paths.append(resolved)
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in search_paths)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.pop("PYTEST_ADDOPTS", None)
    env.pop("PYTEST_PLUGINS", None)
    config = project_root / "pyproject.toml"
    runner = (
        [
            "uv",
            "run",
            "--project",
            str(project_root),
            "--offline",
            "--frozen",
            "--all-extras",
            "python",
        ]
        if config.is_file() and (project_root / "uv.lock").is_file()
        else [sys.executable]
    )
    return runner, env


def pytest_inventory(
    collection_root: Path,
    *,
    project_root: Path | None = None,
    targets: Sequence[Path] = (),
) -> dict[str, Any]:
    collection_root = collection_root.resolve(strict=True)
    project_root = (project_root or collection_root).resolve(strict=True)
    pytest_args = [
        "--collect-only",
        "-q",
        "-p",
        "no:cacheprovider",
        f"--rootdir={collection_root}",
        f"--confcutdir={collection_root}",
    ]
    config = project_root / "pyproject.toml"
    if config.is_file():
        pytest_args.extend(["-c", str(config)])
    pytest_args.extend(path.as_posix() for path in targets)
    plugin = (
        "import json,pytest\n"
        "class Inventory:\n"
        " def pytest_collection_finish(self,session):\n"
        "  print('MOP_NODEIDS='+json.dumps(sorted(item.nodeid for item in session.items)))\n"
        f"raise SystemExit(pytest.main({pytest_args!r},plugins=[Inventory()]))"
    )
    runner, env = _pytest_context(collection_root, project_root)
    result = subprocess.run(
        [*runner, "-c", plugin],
        cwd=collection_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=180,
    )
    marker = "MOP_NODEIDS="
    line = next((row for row in result.stdout.splitlines() if row.startswith(marker)), None)
    if result.returncode not in {0, 5} or line is None:
        detail = (result.stdout + "\n" + result.stderr).strip()
        raise CondensationError(f"pytest inventory collection failed: {detail[-1000:]}")
    try:
        nodeids = json.loads(line[len(marker) :])
    except json.JSONDecodeError as exc:
        raise CondensationError("pytest inventory collection returned malformed identities") from exc
    if (
        not isinstance(nodeids, list)
        or len(nodeids) != len(set(nodeids))
        or not all(isinstance(item, str) and "::" in item for item in nodeids)
    ):
        raise CondensationError("pytest inventory contains duplicate or invalid node identities")
    nodeids = sorted(nodeids)
    return {
        "count": len(nodeids),
        "sha256": canonical_sha256(nodeids),
        "nodeids": nodeids,
    }


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def classify_path(path: str, contract: Mapping[str, Any]) -> Mapping[str, Any]:
    surfaces = contract["measurement"]["surfaces"]
    for row in surfaces:
        if row.get("default") is True:
            return row
        if _matches(path, row["include"]) and not _matches(path, row.get("exclude", [])):
            return row
    raise AssertionError("validated contract always has a default surface")


def physical_loc(data: bytes) -> tuple[int, int] | None:
    if b"\x00" in data:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return None
    lines = text.splitlines()
    longest = max((len(line.encode("utf-8")) for line in lines), default=0)
    return len(lines), longest


def _file_row(
    relative: str,
    data: bytes,
    *,
    is_symlink: bool,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    max_active_line = int(contract["measurement"]["max_active_line_bytes"])
    violations: list[str] = []
    surface = classify_path(relative, contract)
    measured = physical_loc(data)
    row = {
        "path": relative,
        "surface": str(surface["name"]),
        "active": bool(surface["active"]),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
        "symlink": is_symlink,
    }
    if measured is None:
        row.update({"binary": True, "loc": 0, "max_line_bytes": None})
        if surface["active"]:
            violations.append(f"active binary file is not LOC-accountable: {relative}")
    else:
        loc, longest = measured
        row.update({"binary": False, "loc": loc, "max_line_bytes": longest})
        if surface["active"] and longest > max_active_line:
            violations.append(
                f"active file exceeds {max_active_line} bytes on one line: {relative} ({longest})"
            )
    if is_symlink and surface["active"]:
        violations.append(f"active source may not be hidden behind a symlink: {relative}")
    return row, violations


def _shape_summary(
    files: list[dict[str, Any]],
    *,
    git_identity: str,
    untracked: Sequence[str],
    violations: list[str],
) -> dict[str, Any]:
    surface_totals: dict[str, dict[str, Any]] = {}
    directories: set[str] = set()
    for row in files:
        totals = surface_totals.setdefault(
            str(row["surface"]),
            {
                "active": bool(row["active"]),
                "loc": 0,
                "text_files": 0,
                "binary_files": 0,
                "bytes": 0,
            },
        )
        totals["bytes"] += int(row["bytes"])
        if row["binary"]:
            totals["binary_files"] += 1
        else:
            totals["loc"] += int(row["loc"])
            totals["text_files"] += 1
        parent = PurePosixPath(str(row["path"])).parent
        if str(parent) != ".":
            directories.add(str(parent))
    active_files = [row for row in files if row["active"]]
    tree_rows = [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "loc": row["loc"],
            "surface": row["surface"],
            "active": row["active"],
            "binary": row["binary"],
            "symlink": row["symlink"],
        }
        for row in files
    ]
    return {
        "schema": "mop-repository-shape/v1",
        "measured_at": _now(),
        "git_head": git_identity,
        "active_repo_LOC": sum(int(row["loc"]) for row in active_files),
        "tracked_text_LOC": sum(int(row["loc"]) for row in files),
        "tracked_files": len(files),
        "tracked_directories": len(directories),
        "active_files": len(active_files),
        "binary_files": sum(bool(row["binary"]) for row in files),
        "surface_LOC": surface_totals,
        "tree_sha256": canonical_sha256(tree_rows),
        "active_tree_sha256": canonical_sha256([row for row in tree_rows if row["active"]]),
        "no_gaming_violations": violations,
        "untracked_files": list(untracked),
        "files": files,
    }


def measure_repository(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve(strict=True)
    files: list[dict[str, Any]] = []
    violations: list[str] = []

    for relative in tracked_files(root):
        path = root / relative
        if path.is_symlink():
            data = os.readlink(path).encode("utf-8")
            is_symlink = True
        else:
            data = path.read_bytes()
            is_symlink = False
        row, row_violations = _file_row(
            relative,
            data,
            is_symlink=is_symlink,
            contract=contract,
        )
        files.append(row)
        violations.extend(row_violations)
    return _shape_summary(
        files,
        git_identity=git_head(root),
        untracked=untracked_files(root),
        violations=violations,
    )


def measure_commit(root: Path, commit: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    violations: list[str] = []
    raw = _git(root, ["ls-tree", "-r", "-z", commit])
    for record in raw.split(b"\x00"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, object_type, object_id = metadata.decode("ascii").split(" ")
        if object_type != "blob":
            raise CondensationError(f"baseline contains unsupported Git object type: {object_type}")
        relative = raw_path.decode("utf-8")
        data = _git(root, ["cat-file", "blob", object_id])
        row, row_violations = _file_row(
            relative,
            data,
            is_symlink=mode == "120000",
            contract=contract,
        )
        files.append(row)
        violations.extend(row_violations)
    files.sort(key=lambda row: str(row["path"]))
    return _shape_summary(
        files,
        git_identity=commit,
        untracked=[],
        violations=violations,
    )


def _shape_receipt(shape: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in shape.items() if key not in {"measured_at", "files", "untracked_files"}
    }


def _active_inventory(shape: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": row["path"],
            "sha256": row["sha256"],
            "loc": row["loc"],
            "surface": row["surface"],
        }
        for row in shape["files"]
        if row["active"]
    ]


def build_baseline(
    root: Path,
    *,
    contract_path: Path,
    lock_path: Path,
    elimination_ledger_path: Path,
    live_bindings_path: Path,
) -> dict[str, Any]:
    status = git_status(root)
    if status:
        raise CondensationError("baseline generation requires a clean committed checkout")
    contract = load_contract(contract_path)
    lock = _load_pack_lock(lock_path)
    eliminations = _sealed_document(
        elimination_ledger_path,
        schema=ELIMINATION_LEDGER_SCHEMA,
        seal_field="ledger_sha256",
    )
    bindings = _sealed_document(
        live_bindings_path,
        schema=LIVE_BINDINGS_SCHEMA,
        seal_field="snapshot_sha256",
    )
    source_commit = git_head(root)
    recorded_at = _git(root, ["show", "-s", "--format=%cI", source_commit]).decode("ascii").strip()
    shape = measure_commit(root, source_commit, contract)
    if shape["no_gaming_violations"]:
        raise CondensationError("baseline source commit violates the no-gaming contract")
    active_inventory = _active_inventory(shape)
    validation_inventory = [row for row in active_inventory if row["surface"] == "validation"]
    logical_validation = pytest_inventory(root)
    core = {
        "schema": BASELINE_SCHEMA,
        "recorded_at": recorded_at,
        "source_commit": source_commit,
        "source_tree": _git(root, ["rev-parse", f"{source_commit}^{{tree}}"]).decode("ascii").strip(),
        "contract_file_sha256": sha256_file(contract_path),
        "initial_pack_lock_sha256": lock["lock_sha256"],
        "initial_elimination_ledger_sha256": eliminations["ledger_sha256"],
        "live_bindings_snapshot_sha256": bindings["snapshot_sha256"],
        "active_inventory_sha256": canonical_sha256(active_inventory),
        "validation_inventory": {
            "file_count": len(validation_inventory),
            "LOC": sum(int(row["loc"]) for row in validation_inventory),
            "sha256": canonical_sha256(validation_inventory),
        },
        "logical_validation_inventory": {
            "count": logical_validation["count"],
            "sha256": logical_validation["sha256"],
        },
        "shape": _shape_receipt(shape),
    }
    return {**core, "baseline_sha256": canonical_sha256(core)}


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _safe_mount_path(root: Path, relative: Path) -> Path:
    current = root
    for part in relative.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise CondensationError(f"pack mount parent may not be a symlink: {current}")
        current.mkdir(exist_ok=True)
        if not current.is_dir():
            raise CondensationError(f"pack mount parent is not a directory: {current}")
    destination = current / relative.parts[-1]
    if destination.is_symlink():
        raise CondensationError(f"pack mount may not be a symlink: {destination}")
    return destination


def _safe_existing_directory(root: Path, relative: Path, field: str) -> Path:
    if root.is_symlink() or not root.is_dir():
        raise CondensationError(f"{field} root is missing or unsafe: {root}")
    resolved_root = root.resolve(strict=True)
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink() or not current.is_dir():
            raise CondensationError(f"{field} path is missing or unsafe: {current}")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root):
        raise CondensationError(f"{field} path escapes its root: {relative}")
    return current


def _load_sealed_chain(
    path: Path,
    *,
    schema: str,
    seal_field: str,
    previous_field: str,
    history_relpath: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    head = _sealed_document(path, schema=schema, seal_field=seal_field)
    current = head
    lineage: list[str] = []
    history_paths: list[str] = []
    seen: set[str] = set()
    while True:
        seal = current.get(seal_field)
        sequence = current.get("sequence")
        previous = current.get(previous_field)
        if (
            not isinstance(seal, str)
            or re.fullmatch(r"[0-9a-f]{64}", seal) is None
            or isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence < 0
            or sequence > MAX_CHAIN_DEPTH
            or seal in seen
        ):
            raise CondensationError(f"invalid or cyclic {schema} history")
        lineage.append(seal)
        seen.add(seal)
        if sequence == 0:
            if previous is not None:
                raise CondensationError(f"{schema} genesis must not name a predecessor")
            break
        if not isinstance(previous, str) or re.fullmatch(r"[0-9a-f]{64}", previous) is None:
            raise CondensationError(f"{schema} sequence {sequence} requires a predecessor")
        relative = history_relpath / f"{previous}.json"
        ancestor_path = path.parent / relative
        ancestor = _sealed_document(ancestor_path, schema=schema, seal_field=seal_field)
        if ancestor.get(seal_field) != previous or ancestor.get("sequence") != sequence - 1:
            raise CondensationError(f"{schema} predecessor identity or sequence is invalid")
        history_paths.append(relative.as_posix())
        current = ancestor
    if len(lineage) != int(head["sequence"]) + 1:
        raise CondensationError(f"{schema} history depth does not match its sequence")
    history_root = path.parent / history_relpath
    if history_root.exists():
        if history_root.is_symlink() or not history_root.is_dir():
            raise CondensationError(f"{schema} history root is unsafe")
        actual: set[str] = set()
        for candidate in history_root.iterdir():
            if candidate.is_symlink() or not candidate.is_file():
                raise CondensationError(f"{schema} history contains an unsafe entry: {candidate}")
            actual.add(candidate.name)
        expected = {f"{seal}.json" for seal in lineage[1:]}
        if actual != expected:
            raise CondensationError(f"{schema} history is not the exact verified linear ancestry")
    return head, lineage, history_paths


def _load_pack_lock(path: Path) -> dict[str, Any]:
    value, lineage, history_paths = _load_sealed_chain(
        path,
        schema=PACK_LOCK_SCHEMA,
        seal_field="lock_sha256",
        previous_field="previous_lock_sha256",
        history_relpath=Path("condensation/history/pack-lock"),
    )
    packs = value.get("packs")
    if not isinstance(packs, list):
        raise CondensationError("pack lock packs must be a list")
    ids: set[str] = set()
    cache_paths: list[Path] = []
    mount_paths: list[Path] = []
    for entry in packs:
        if not isinstance(entry, dict):
            raise CondensationError("pack lock entries must be objects")
        pack_id = _pack_id(entry.get("pack_id"), "pack lock pack_id")
        if pack_id in ids:
            raise CondensationError("pack lock pack_ids must be unique")
        _pack_version(entry.get("version"), f"{pack_id}.version")
        for field in ("manifest_sha256", "payload_sha256", "cache_relpath", "mount_relpath"):
            if not isinstance(entry.get(field), str) or not entry[field]:
                raise CondensationError(f"pack {pack_id!r} requires {field}")
        for field in ("manifest_sha256", "payload_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", entry[field]) is None:
                raise CondensationError(f"pack {pack_id!r} has an invalid {field}")
        cache_path = _safe_relative(entry["cache_relpath"], f"{pack_id}.cache_relpath")
        mount_path = _safe_relative(entry["mount_relpath"], f"{pack_id}.mount_relpath")
        if any(_paths_overlap(cache_path, existing) for existing in cache_paths):
            raise CondensationError(f"pack {pack_id!r} has an overlapping cache path")
        if any(_paths_overlap(mount_path, existing) for existing in mount_paths):
            raise CondensationError(f"pack {pack_id!r} has an overlapping mount path")
        cache_paths.append(cache_path)
        mount_paths.append(mount_path)
        ids.add(pack_id)
    return {**value, "_verified_lineage": lineage, "_history_paths": history_paths}


def _payload_paths(payload_root: Path, pack_id: str) -> set[str]:
    if payload_root.is_symlink() or not payload_root.is_dir():
        raise CondensationError(f"pack {pack_id} payload root is missing or unsafe")
    paths: set[str] = set()
    for path in sorted(payload_root.rglob("*")):
        if path.is_symlink():
            raise CondensationError(f"pack {pack_id} payload contains a symlink: {path}")
        if path.is_file():
            paths.add(path.relative_to(payload_root).as_posix())
        elif not path.is_dir():
            raise CondensationError(f"pack {pack_id} payload contains a non-regular entry: {path}")
    return paths


def _verify_payload_rows(
    payload_root: Path,
    *,
    pack_id: str,
    files: Sequence[Any],
) -> tuple[int, list[dict[str, Any]]]:
    actual_paths = _payload_paths(payload_root, pack_id)
    seen: set[str] = set()
    owned_loc = 0
    normalized: list[dict[str, Any]] = []
    for row in files:
        if not isinstance(row, dict):
            raise CondensationError(f"pack {pack_id} contains a malformed file row")
        relative = _safe_relative(row.get("path"), f"{pack_id}.files.path")
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise CondensationError(f"pack {pack_id} repeats {relative_text}")
        payload_path = payload_root / relative
        if payload_path.is_symlink() or not payload_path.is_file():
            raise CondensationError(f"pack payload file is missing or unsafe: {payload_path}")
        data = payload_path.read_bytes()
        measured = physical_loc(data)
        actual_loc = measured[0] if measured is not None else 0
        longest = measured[1] if measured is not None else None
        surface = row.get("surface")
        if not isinstance(surface, str) or not surface:
            raise CondensationError(f"pack {pack_id} {relative_text} requires a surface")
        expected = {
            "sha256": sha256_bytes(data),
            "bytes": len(data),
            "loc": actual_loc,
        }
        for field, actual in expected.items():
            if row.get(field) != actual:
                raise CondensationError(
                    f"pack {pack_id} {relative_text} has invalid {field}: {row.get(field)!r} != {actual!r}"
                )
        owned_loc += actual_loc
        seen.add(relative_text)
        normalized.append(
            {
                **row,
                "_binary": measured is None,
                "_max_line_bytes": longest,
            }
        )
    if actual_paths != seen:
        extras = sorted(actual_paths - seen)
        missing = sorted(seen - actual_paths)
        raise CondensationError(
            f"pack {pack_id} manifest/payload inventory mismatch; extras={extras[:5]}, missing={missing[:5]}"
        )
    return owned_loc, normalized


def _verify_pack_manifest(
    pack_root: Path,
    entry: Mapping[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, Any]:
    manifest_path = pack_root / "manifest.json"
    manifest = _sealed_document(
        manifest_path,
        schema=PACK_MANIFEST_SCHEMA,
        seal_field="manifest_sha256",
    )
    if manifest["manifest_sha256"] != entry["manifest_sha256"]:
        raise CondensationError(f"locked manifest mismatch for pack {entry['pack_id']}")
    if manifest.get("pack_id") != entry["pack_id"] or manifest.get("version") != entry["version"]:
        raise CondensationError(f"pack identity mismatch for {entry['pack_id']}")
    files = manifest.get("files")
    if not isinstance(files, list):
        raise CondensationError(f"pack {entry['pack_id']} manifest files must be a list")
    if canonical_sha256(files) != manifest.get("payload_sha256"):
        raise CondensationError(f"payload list seal mismatch for pack {entry['pack_id']}")
    if manifest["payload_sha256"] != entry["payload_sha256"]:
        raise CondensationError(f"locked payload mismatch for pack {entry['pack_id']}")

    owned_loc, normalized = _verify_payload_rows(
        pack_root / "payload",
        pack_id=str(entry["pack_id"]),
        files=files,
    )
    validation_nodeids = _verify_pack_validation(
        pack_root / "payload",
        pack_id=str(entry["pack_id"]),
        payload_sha256=str(manifest["payload_sha256"]),
        files=normalized,
        validation=manifest.get("validation_inventory"),
        project_root=project_root,
    )
    return {
        "pack_id": entry["pack_id"],
        "version": entry["version"],
        "payload_sha256": manifest["payload_sha256"],
        "manifest_sha256": manifest["manifest_sha256"],
        "owned_text_LOC": owned_loc,
        "files": len(files),
        "pack_root": str(pack_root),
        "_file_rows": normalized,
        "_validation_nodeids": validation_nodeids,
        "_validation_inventory": manifest.get("validation_inventory"),
    }


def _verify_pack_validation(
    payload_root: Path,
    *,
    pack_id: str,
    payload_sha256: str,
    files: Sequence[Mapping[str, Any]],
    validation: Any,
    project_root: Path | None,
) -> list[str]:
    validation_files = {str(row["path"]) for row in files if row.get("surface") == "validation"}
    if validation is None:
        if validation_files:
            raise CondensationError(f"pack {pack_id} contains validation files without a collected inventory")
        return []
    if not isinstance(validation, dict):
        raise CondensationError(f"pack {pack_id} has invalid validation inventory")
    declared = validation.get("nodeids")
    raw_targets = validation.get("collection_paths")
    if not isinstance(declared, list) or not isinstance(raw_targets, list) or not raw_targets:
        raise CondensationError(f"pack {pack_id} has invalid validation inventory")
    targets = [_safe_relative(item, f"{pack_id}.collection_paths") for item in raw_targets]
    target_paths = [payload_root / target for target in targets]
    if any(path.is_symlink() or not path.exists() for path in target_paths):
        raise CondensationError(f"pack {pack_id} validation collection path is missing or unsafe")
    validation_nodeids = sorted(declared)
    collected = pytest_inventory(
        payload_root,
        project_root=project_root,
        targets=targets,
    )
    expected = {
        "payload_sha256": payload_sha256,
        "collection_paths": [path.as_posix() for path in targets],
        "count": collected["count"],
        "sha256": collected["sha256"],
        "nodeids": collected["nodeids"],
    }
    nodeid_files = {
        item.split("::", 1)[0] for item in validation_nodeids if isinstance(item, str) and "::" in item
    }
    covered_files = {
        file
        for file in validation_files
        if any(file == target.as_posix() or file.startswith(f"{target.as_posix()}/") for target in targets)
    }
    if (
        not validation_nodeids
        or len(validation_nodeids) != len(set(validation_nodeids))
        or not all(isinstance(item, str) and "::" in item for item in validation_nodeids)
        or not nodeid_files.issubset(covered_files)
        or validation != expected
    ):
        raise CondensationError(f"pack {pack_id} validation inventory is not sealed")
    return validation_nodeids


def _relocated_loc(
    verified: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    baseline_commit: str,
    contract: Mapping[str, Any],
) -> tuple[int, dict[str, int], list[str]]:
    if re.fullmatch(r"[0-9a-f]{40}", baseline_commit) is None:
        raise CondensationError("baseline source commit must be a full lowercase Git object ID")
    current_paths = set(tracked_files(repo_root))
    credited: set[str] = set()
    totals: dict[str, int] = {}
    for pack in verified:
        pack_id = str(pack["pack_id"])
        pack_total = 0
        for row in pack["_file_rows"]:
            origin = row.get("origin")
            if origin is None:
                continue
            if not isinstance(origin, dict):
                raise CondensationError(f"pack {pack_id} has a malformed origin record")
            relative = _safe_relative(origin.get("path"), f"{pack_id}.origin.path")
            relative_text = relative.as_posix()
            if ":" in relative_text:
                raise CondensationError(f"pack {pack_id} origin path may not contain ':'")
            if relative_text in credited:
                raise CondensationError(f"baseline origin is credited more than once: {relative_text}")
            expected_sha = origin.get("sha256")
            expected_loc = origin.get("loc")
            if (
                not isinstance(expected_sha, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None
                or isinstance(expected_loc, bool)
                or not isinstance(expected_loc, int)
                or expected_loc < 0
            ):
                raise CondensationError(f"pack {pack_id} has an invalid origin identity")
            baseline_data = _git(repo_root, ["show", f"{baseline_commit}:{relative_text}"])
            measured = physical_loc(baseline_data)
            baseline_loc = measured[0] if measured is not None else 0
            if sha256_bytes(baseline_data) != expected_sha or baseline_loc != expected_loc:
                raise CondensationError(f"pack {pack_id} origin does not match baseline: {relative_text}")
            origin_surface = classify_path(relative_text, contract)
            if row.get("surface") != origin_surface["name"]:
                raise CondensationError(
                    f"pack {pack_id} origin surface mismatch for {relative_text}: "
                    f"{row.get('surface')!r} != {origin_surface['name']!r}"
                )
            if row.get("sha256") != expected_sha or row.get("loc") != expected_loc:
                continue
            if not bool(origin_surface["active"]):
                continue
            if relative_text in current_paths:
                continue
            pack_total += expected_loc
            credited.add(relative_text)
        totals[pack_id] = pack_total
    return sum(totals.values()), totals, sorted(credited)


def verify_packs(
    lock_path: Path,
    cache_root: Path,
    *,
    repo_root: Path | None = None,
    baseline_commit: str | None = None,
    contract: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    if (repo_root is None) != (baseline_commit is None):
        raise CondensationError("pack relocation accounting requires repo, baseline commit, and contract")
    if repo_root is not None and contract is None:
        raise CondensationError("pack relocation accounting requires the condensation contract")
    lock = _load_pack_lock(lock_path)
    internal: list[dict[str, Any]] = []
    project_root = project_root or repo_root
    for entry in lock["packs"]:
        cache_relative = _safe_relative(entry["cache_relpath"], "cache_relpath")
        pack_root = _safe_existing_directory(cache_root, cache_relative, "pack cache")
        internal.append(
            _verify_pack_manifest(
                pack_root,
                entry,
                project_root=project_root,
            )
        )
    surface_contract = (
        {str(row["name"]): row for row in contract["measurement"]["surfaces"]} if contract is not None else {}
    )
    pack_surface_loc: dict[str, int] = {}
    validation_nodeids: set[str] = set()
    for pack in internal:
        per_pack: dict[str, int] = {}
        for row in pack["_file_rows"]:
            surface = str(row["surface"])
            if contract is not None and surface not in surface_contract:
                raise CondensationError(f"pack {pack['pack_id']} uses unknown surface {surface!r}")
            if contract is not None:
                origin = row.get("origin")
                classification_path = (
                    str(origin["path"]) if isinstance(origin, dict) and "path" in origin else str(row["path"])
                )
                expected_surface = str(classify_path(classification_path, contract)["name"])
                if surface != expected_surface:
                    raise CondensationError(
                        f"pack {pack['pack_id']} surface mismatch for {row['path']}: "
                        f"{surface!r} != {expected_surface!r}"
                    )
            if contract is not None and bool(surface_contract[surface]["active"]):
                if row["_binary"]:
                    raise CondensationError(
                        f"pack {pack['pack_id']} contains active binary payload: {row['path']}"
                    )
                if int(row["_max_line_bytes"]) > int(contract["measurement"]["max_active_line_bytes"]):
                    raise CondensationError(
                        f"pack {pack['pack_id']} contains packed active line: {row['path']}"
                    )
            loc = int(row["loc"])
            per_pack[surface] = per_pack.get(surface, 0) + loc
            pack_surface_loc[surface] = pack_surface_loc.get(surface, 0) + loc
        pack["_surface_LOC"] = per_pack
        pack["_active_owned_LOC"] = sum(
            loc
            for surface, loc in per_pack.items()
            if contract is not None and bool(surface_contract[surface]["active"])
        )
        overlap = validation_nodeids.intersection(pack["_validation_nodeids"])
        if overlap:
            raise CondensationError(
                f"validation node identities are repeated across packs: {sorted(overlap)[:5]}"
            )
        validation_nodeids.update(pack["_validation_nodeids"])
    relocated = 0
    relocated_by_pack: dict[str, int] = {}
    relocated_paths: list[str] = []
    if repo_root is not None and baseline_commit is not None and contract is not None:
        relocated, relocated_by_pack, relocated_paths = _relocated_loc(
            internal,
            repo_root=repo_root,
            baseline_commit=baseline_commit,
            contract=contract,
        )
    verified = []
    for row in internal:
        public = {
            key: value
            for key, value in row.items()
            if key
            not in {
                "_file_rows",
                "_surface_LOC",
                "_active_owned_LOC",
                "_validation_nodeids",
                "_validation_inventory",
            }
        }
        public["surface_LOC"] = row["_surface_LOC"]
        public["active_owned_LOC"] = row["_active_owned_LOC"]
        public["relocated_LOC"] = relocated_by_pack.get(str(row["pack_id"]), 0)
        public["non_relocated_owned_LOC"] = int(public["active_owned_LOC"]) - int(public["relocated_LOC"])
        public["validation_count"] = len(row["_validation_nodeids"])
        inventory = row["_validation_inventory"]
        public["validation_collection_paths"] = (
            inventory["collection_paths"] if isinstance(inventory, dict) else []
        )
        verified.append(public)
    owned_text = sum(int(row["owned_text_LOC"]) for row in verified)
    active_owned = sum(int(row["active_owned_LOC"]) for row in verified)
    if relocated < 0 or relocated > active_owned:
        raise CondensationError("relocated LOC exceeds active owned pack LOC")
    return {
        "schema": "mop-condensation-pack-verification/v1",
        "lock_sha256": lock["lock_sha256"],
        "lineage": lock["_verified_lineage"],
        "history_paths": lock["_history_paths"],
        "packs": verified,
        "pack_count": len(verified),
        "owned_text_LOC": owned_text,
        "active_owned_LOC": active_owned,
        "relocated_LOC": relocated,
        "relocated_paths": relocated_paths,
        "non_relocated_owned_LOC": active_owned - relocated,
        "surface_LOC": pack_surface_loc,
        "validation_nodeids": sorted(validation_nodeids),
    }


def hydrate_packs(
    lock_path: Path,
    *,
    cache_root: Path,
    destination_root: Path,
    contract: Mapping[str, Any] | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    verification = verify_packs(
        lock_path,
        cache_root,
        contract=contract,
        project_root=project_root,
    )
    lock = _load_pack_lock(lock_path)
    hydrated: list[dict[str, Any]] = []
    destination_root.mkdir(parents=True, exist_ok=True)
    if destination_root.is_symlink():
        raise CondensationError("pack hydration destination root may not be a symlink")
    destination_root = destination_root.resolve(strict=True)
    for entry, verified in zip(lock["packs"], verification["packs"], strict=True):
        source = Path(verified["pack_root"]) / "payload"
        manifest = _verify_pack_manifest(
            Path(verified["pack_root"]),
            entry,
            project_root=project_root,
        )
        mount = _safe_relative(entry["mount_relpath"], "mount_relpath")
        destination = _safe_mount_path(destination_root, mount)
        already_present = destination.exists() or destination.is_symlink()
        if already_present:
            _verify_payload_rows(
                destination,
                pack_id=str(entry["pack_id"]),
                files=manifest["_file_rows"],
            )
            _verify_pack_validation(
                destination,
                pack_id=str(entry["pack_id"]),
                payload_sha256=str(entry["payload_sha256"]),
                files=manifest["_file_rows"],
                validation=manifest["_validation_inventory"],
                project_root=project_root,
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix=f".{entry['pack_id']}.",
                dir=destination.parent,
            ) as temp:
                staged = Path(temp) / "payload"
                shutil.copytree(source, staged, symlinks=False)
                _verify_payload_rows(
                    staged,
                    pack_id=str(entry["pack_id"]),
                    files=manifest["_file_rows"],
                )
                _verify_pack_validation(
                    staged,
                    pack_id=str(entry["pack_id"]),
                    payload_sha256=str(entry["payload_sha256"]),
                    files=manifest["_file_rows"],
                    validation=manifest["_validation_inventory"],
                    project_root=project_root,
                )
                os.replace(staged, destination)
        hydrated.append(
            {
                "pack_id": entry["pack_id"],
                "version": entry["version"],
                "destination": str(destination),
                "payload_sha256": entry["payload_sha256"],
                "owned_text_LOC": verified["owned_text_LOC"],
                "already_present": already_present,
            }
        )
    return {
        "schema": "mop-condensation-hydration/v1",
        "hydrated_at": _now(),
        "packs": hydrated,
        "pack_count": len(hydrated),
    }


def verify_live_bindings(root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = _sealed_document(
        manifest_path,
        schema=LIVE_BINDINGS_SCHEMA,
        seal_field="snapshot_sha256",
    )
    bindings = manifest.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise CondensationError("live binding manifest must contain at least one binding")
    seen: set[str] = set()
    mismatches: list[dict[str, str]] = []
    for row in bindings:
        if not isinstance(row, dict):
            raise CondensationError("live binding rows must be objects")
        relative = _safe_relative(row.get("path"), "bindings.path")
        expected = row.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise CondensationError(f"invalid binding digest for {relative}")
        text = relative.as_posix()
        if text in seen:
            raise CondensationError(f"duplicate live binding: {text}")
        path = root / relative
        actual = sha256_file(path) if path.is_file() and not path.is_symlink() else "missing-or-unsafe"
        if actual != expected:
            mismatches.append({"path": text, "expected": expected, "actual": actual})
        seen.add(text)
    return {
        "schema": "mop-condensation-live-binding-verification/v1",
        "observed_at": manifest.get("observed_at"),
        "binding_count": len(bindings),
        "mismatches": mismatches,
        "all_match": not mismatches,
        "snapshot_sha256": manifest["snapshot_sha256"],
    }


def load_baseline(path: Path) -> dict[str, Any]:
    value = _sealed_document(path, schema=BASELINE_SCHEMA, seal_field="baseline_sha256")
    if re.fullmatch(r"[0-9a-f]{40}", str(value.get("source_commit"))) is None:
        raise CondensationError("baseline source_commit must be a full lowercase Git object ID")
    if not isinstance(value.get("shape"), dict):
        raise CondensationError("baseline shape must be an object")
    return value


def verify_baseline(
    root: Path,
    baseline: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    source_commit = str(baseline["source_commit"])
    source_tree = _git(root, ["rev-parse", f"{source_commit}^{{tree}}"]).decode("ascii").strip()
    shape = measure_commit(root, source_commit, contract)
    active_inventory = _active_inventory(shape)
    validation_inventory = [row for row in active_inventory if row["surface"] == "validation"]
    expected_validation = {
        "file_count": len(validation_inventory),
        "LOC": sum(int(row["loc"]) for row in validation_inventory),
        "sha256": canonical_sha256(validation_inventory),
    }
    problems: list[str] = []
    if not git_is_ancestor(root, source_commit, git_head(root)):
        problems.append("baseline source commit is not an ancestor of the current checkout")
    if baseline.get("source_tree") != source_tree:
        problems.append("baseline source tree identity is invalid")
    if baseline.get("contract_file_sha256") != sha256_file(contract_path):
        problems.append("condensation contract differs from the sealed Wave 0 baseline")
    if baseline.get("shape") != _shape_receipt(shape):
        problems.append("baseline shape does not match its source commit")
    if baseline.get("active_inventory_sha256") != canonical_sha256(active_inventory):
        problems.append("baseline active inventory does not match its source commit")
    if baseline.get("validation_inventory") != expected_validation:
        problems.append("baseline validation inventory does not match its source commit")
    return {
        "schema": "mop-extreme-condensation-baseline-verification/v1",
        "source_commit": source_commit,
        "source_tree": source_tree,
        "problems": problems,
        "all_match": not problems,
        "active_paths": [str(row["path"]) for row in active_inventory],
    }


def _file_identity(data: bytes) -> tuple[str, int]:
    measured = physical_loc(data)
    return sha256_bytes(data), measured[0] if measured is not None else 0


def verify_eliminations(
    root: Path,
    ledger_path: Path,
    *,
    baseline_commit: str,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    ledger, lineage, history_paths = _load_sealed_chain(
        ledger_path,
        schema=ELIMINATION_LEDGER_SCHEMA,
        seal_field="ledger_sha256",
        previous_field="previous_ledger_sha256",
        history_relpath=Path("condensation/history/eliminations"),
    )
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise CondensationError("elimination ledger entries must be a list")
    seen_ids: set[str] = set()
    before_paths: set[str] = set()
    after_paths: set[str] = set()
    current_tracked = set(tracked_files(root))
    gate_profile = str(contract["elimination_gate_profile"])
    gate_ids = [str(row["id"]) for row in contract["gates"][gate_profile]]
    required_validation = {
        "gate_profile": gate_profile,
        "gate_ids": gate_ids,
    }
    verified: list[dict[str, Any]] = []
    total = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise CondensationError("elimination entries must be objects")
        entry_id = entry.get("id")
        if (
            not isinstance(entry_id, str)
            or PACK_ID_PATTERN.fullmatch(entry_id) is None
            or entry_id in seen_ids
        ):
            raise CondensationError("elimination entry IDs must be unique safe identifiers")
        before = entry.get("before")
        after = entry.get("after")
        validation = entry.get("validation")
        if (
            not isinstance(before, list)
            or not before
            or not isinstance(after, list)
            or validation != required_validation
        ):
            raise CondensationError(
                f"elimination entry {entry_id!r} requires before, after, and the full gate profile"
            )
        before_loc = 0
        normalized_before: list[dict[str, Any]] = []
        for row in before:
            if not isinstance(row, dict):
                raise CondensationError(f"elimination entry {entry_id!r} has invalid before row")
            relative = _safe_relative(row.get("path"), f"{entry_id}.before.path")
            text = relative.as_posix()
            if text in before_paths or not bool(classify_path(text, contract)["active"]):
                raise CondensationError(f"invalid or repeated eliminated baseline path: {text}")
            data = _git(root, ["show", f"{baseline_commit}:{text}"])
            digest, loc = _file_identity(data)
            if row.get("sha256") != digest or row.get("loc") != loc:
                raise CondensationError(f"elimination baseline identity mismatch: {text}")
            before_loc += loc
            before_paths.add(text)
            normalized_before.append({"path": text, "sha256": digest, "loc": loc})
        after_loc = 0
        normalized_after: list[dict[str, Any]] = []
        entry_after_paths: set[str] = set()
        for row in after:
            if not isinstance(row, dict):
                raise CondensationError(f"elimination entry {entry_id!r} has invalid after row")
            relative = _safe_relative(row.get("path"), f"{entry_id}.after.path")
            text = relative.as_posix()
            if (
                text in after_paths
                or text not in current_tracked
                or not bool(classify_path(text, contract)["active"])
            ):
                raise CondensationError(f"invalid or repeated elimination result path: {text}")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise CondensationError(f"elimination result is missing or unsafe: {text}")
            digest, loc = _file_identity(path.read_bytes())
            if row.get("sha256") != digest or row.get("loc") != loc:
                raise CondensationError(f"elimination result identity mismatch: {text}")
            after_loc += loc
            after_paths.add(text)
            entry_after_paths.add(text)
            normalized_after.append({"path": text, "sha256": digest, "loc": loc})
        retained_without_after = sorted(
            row["path"]
            for row in normalized_before
            if row["path"] in current_tracked and row["path"] not in entry_after_paths
        )
        if retained_without_after:
            raise CondensationError(
                f"elimination entry {entry_id!r} leaves baseline inputs unaccounted: "
                + ", ".join(retained_without_after)
            )
        eliminated = before_loc - after_loc
        if eliminated <= 0 or entry.get("eliminated_LOC") != eliminated:
            raise CondensationError(f"elimination entry {entry_id!r} has no verified LOC reduction")
        total += eliminated
        seen_ids.add(entry_id)
        verified.append(
            {
                "id": entry_id,
                "before": normalized_before,
                "after": normalized_after,
                "eliminated_LOC": eliminated,
                "validation": validation,
            }
        )
    return {
        "schema": "mop-condensation-elimination-verification/v1",
        "ledger_sha256": ledger["ledger_sha256"],
        "lineage": lineage,
        "history_paths": history_paths,
        "entry_count": len(verified),
        "eliminated_LOC": total,
        "before_paths": sorted(before_paths),
        "required_gate_profile": gate_profile,
        "entries": verified,
    }


def next_checkpoint(active_repo_loc: int, ladder: Sequence[int]) -> int | None:
    candidates = [target for target in ladder if target < active_repo_loc]
    return max(candidates) if candidates else None


def accounting(
    shape: Mapping[str, Any],
    *,
    baseline: Mapping[str, Any],
    packs: Mapping[str, Any],
    eliminations: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    active = int(shape["active_repo_LOC"])
    active_owned = int(packs["active_owned_LOC"])
    owned_text = int(packs["owned_text_LOC"])
    relocated = int(packs["relocated_LOC"])
    hydrated = active + active_owned
    baseline_active = int(baseline["shape"]["active_repo_LOC"])
    net_reduction = baseline_active - hydrated
    eliminated = int(eliminations["eliminated_LOC"])
    unexplained_reduction = max(0, net_reduction - eliminated)
    ladder = list(contract["fallback_ladder"])
    active_surfaces = {name: int(row["loc"]) for name, row in shape["surface_LOC"].items()}
    runtime_core = active_surfaces.get("runtime", 0)
    runtime_target = int(contract["stretch_target_runtime_core_loc"])
    hydrated_surfaces = dict(active_surfaces)
    for name, loc in packs["surface_LOC"].items():
        hydrated_surfaces[name] = hydrated_surfaces.get(name, 0) + int(loc)
    return {
        "active_repo_LOC": active,
        "hydrated_owned_LOC": hydrated,
        "hydrated_tracked_text_LOC": int(shape["tracked_text_LOC"]) + owned_text,
        "pack_active_owned_LOC": active_owned,
        "pack_owned_text_LOC": owned_text,
        "relocated_LOC": relocated,
        "non_relocated_owned_LOC": int(packs["non_relocated_owned_LOC"]),
        "eliminated_LOC": eliminated,
        "added_LOC": max(0, eliminated - net_reduction),
        "net_owned_reduction_LOC": net_reduction,
        "unexplained_reduction_LOC": unexplained_reduction,
        "baseline_active_repo_LOC": baseline_active,
        "recommended_target_active_repo_LOC": contract["recommended_target_active_repo_loc"],
        "recommended_target_met": active <= contract["recommended_target_active_repo_loc"],
        "runtime_core_LOC": runtime_core,
        "stretch_target_runtime_core_LOC": runtime_target,
        "stretch_target_runtime_core_met": runtime_core <= runtime_target,
        "next_checkpoint": next_checkpoint(active, ladder),
        "active_surface_LOC": active_surfaces,
        "surface_LOC": hydrated_surfaces,
        "tracked_text_LOC": shape["tracked_text_LOC"],
    }


def verify_repository(
    root: Path,
    *,
    contract_path: Path,
    baseline_path: Path,
    lock_path: Path,
    elimination_ledger_path: Path,
    live_bindings_path: Path,
    cache_root: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    shape = measure_repository(root, contract)
    bindings = verify_live_bindings(root, live_bindings_path)
    baseline = load_baseline(baseline_path)
    baseline_verification = verify_baseline(
        root,
        baseline,
        contract=contract,
        contract_path=contract_path,
    )
    eliminations = verify_eliminations(
        root,
        elimination_ledger_path,
        baseline_commit=str(baseline["source_commit"]),
        contract=contract,
    )
    packs = verify_packs(
        lock_path,
        cache_root,
        repo_root=root,
        baseline_commit=str(baseline["source_commit"]),
        contract=contract,
    )
    problems = list(shape["no_gaming_violations"])
    problems.extend(baseline_verification["problems"])
    current_tracked = set(tracked_files(root))
    if shape["untracked_files"]:
        problems.append(
            "non-ignored untracked files are outside repository accounting: "
            + ", ".join(shape["untracked_files"][:10])
        )
    initial_lock = baseline.get("initial_pack_lock_sha256")
    if initial_lock != packs["lock_sha256"] and initial_lock not in packs["lineage"]:
        problems.append("pack lock does not descend from the sealed Wave 0 lock")
    initial_eliminations = baseline.get("initial_elimination_ledger_sha256")
    if (
        initial_eliminations != eliminations["ledger_sha256"]
        and initial_eliminations not in eliminations["lineage"]
    ):
        problems.append("elimination ledger does not descend from the sealed Wave 0 ledger")
    for control_path, history_paths in (
        (lock_path, packs["history_paths"]),
        (elimination_ledger_path, eliminations["history_paths"]),
    ):
        for history_path in history_paths:
            absolute = (control_path.parent / history_path).resolve(strict=True)
            try:
                relative = absolute.relative_to(root.resolve(strict=True)).as_posix()
            except ValueError:
                problems.append(f"condensation history escapes the repository: {absolute}")
                continue
            if relative not in current_tracked:
                problems.append(f"condensation history is not Git-tracked: {relative}")
    if baseline.get("live_bindings_snapshot_sha256") != bindings["snapshot_sha256"]:
        problems.append("live-binding snapshot identity differs from the sealed Wave 0 baseline")
    if bindings["mismatches"]:
        problems.extend(f"live binding changed: {row['path']}" for row in bindings["mismatches"])
    duplicate_reduction_paths = sorted(
        set(packs["relocated_paths"]).intersection(eliminations["before_paths"])
    )
    if duplicate_reduction_paths:
        problems.append(
            "baseline paths are claimed as both relocated and eliminated: "
            + ", ".join(duplicate_reduction_paths[:10])
        )
    active_validation = pytest_inventory(root)
    packed_validation = set(packs["validation_nodeids"])
    overlap = packed_validation.intersection(active_validation["nodeids"])
    if overlap:
        problems.append(
            "validation nodes are duplicated between active and packed suites: "
            + ", ".join(sorted(overlap)[:5])
        )
    logical_nodeids = sorted(set(active_validation["nodeids"]) | packed_validation)
    logical_validation = {
        "count": len(logical_nodeids),
        "sha256": canonical_sha256(logical_nodeids),
    }
    if logical_validation != baseline.get("logical_validation_inventory"):
        problems.append(
            "logical validation inventory differs from the sealed baseline: "
            f"{logical_validation['count']} collected"
        )
    baseline_active_paths = set(baseline_verification["active_paths"])
    missing_active_paths = baseline_active_paths - current_tracked
    explained_missing_paths = set(packs["relocated_paths"]) | set(eliminations["before_paths"])
    unexplained_missing_paths = sorted(missing_active_paths - explained_missing_paths)
    if unexplained_missing_paths:
        problems.append(
            "baseline active paths disappeared without relocation or elimination evidence: "
            + ", ".join(unexplained_missing_paths[:10])
        )
    result_accounting = accounting(
        shape,
        baseline=baseline,
        packs=packs,
        eliminations=eliminations,
        contract=contract,
    )
    if result_accounting["unexplained_reduction_LOC"]:
        problems.append(
            "active LOC reduction lacks verified elimination evidence: "
            f"{result_accounting['unexplained_reduction_LOC']} LOC"
        )
    return {
        "schema": "mop-extreme-condensation-verification/v1",
        "verified_at": _now(),
        "ok": not problems,
        "problems": problems,
        "accounting": result_accounting,
        "baseline": baseline_verification,
        "shape": {key: value for key, value in shape.items() if key != "files"},
        "packs": packs,
        "eliminations": eliminations,
        "logical_validation_inventory": logical_validation,
        "live_bindings": bindings,
    }


def _gate_commands(contract: Mapping[str, Any], profile: str) -> list[Mapping[str, Any]]:
    gates = contract.get("gates")
    commands = gates.get(profile) if isinstance(gates, dict) else None
    if not isinstance(commands, list):
        raise CondensationError(f"unknown gate profile: {profile}")
    for row in commands:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("id"), str)
            or not isinstance(row.get("argv"), list)
            or not row["argv"]
            or not all(isinstance(item, str) and item for item in row["argv"])
        ):
            raise CondensationError(f"gate profile {profile!r} contains an invalid command")
    return commands


def run_gates(
    root: Path,
    contract: Mapping[str, Any],
    profile: str,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for gate in _gate_commands(contract, profile):
        started = time.monotonic()
        env = os.environ.copy()
        additions = gate.get("env", {})
        if not isinstance(additions, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in additions.items()
        ):
            raise CondensationError(f"gate {gate['id']!r} has invalid env")
        if extra_env is not None:
            env.update(extra_env)
        env.update(additions)
        result = subprocess.run(
            gate["argv"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "id": gate["id"],
                "argv": gate["argv"],
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 3),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        if result.returncode != 0:
            break
    return {
        "profile": profile,
        "ok": len(results) == len(_gate_commands(contract, profile))
        and all(row["returncode"] == 0 for row in results),
        "results": results,
    }


def run_hydrated_validation(
    root: Path,
    packs: Mapping[str, Any],
    hydration: Mapping[str, Any],
) -> dict[str, Any]:
    destinations = {str(row["pack_id"]): Path(str(row["destination"])) for row in hydration["packs"]}
    results: list[dict[str, Any]] = []
    for pack in packs["packs"]:
        if not int(pack["validation_count"]):
            continue
        pack_id = str(pack["pack_id"])
        destination = destinations[pack_id].resolve(strict=True)
        targets = [
            _safe_relative(item, f"{pack_id}.validation_collection_paths")
            for item in pack["validation_collection_paths"]
        ]
        runner, env = _pytest_context(destination, root)
        argv = [
            *runner,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--rootdir={destination}",
            f"--confcutdir={destination}",
        ]
        config = root / "pyproject.toml"
        if config.is_file():
            argv.extend(["-c", str(config)])
        argv.extend(path.as_posix() for path in targets)
        started = time.monotonic()
        result = subprocess.run(
            argv,
            cwd=destination,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        results.append(
            {
                "pack_id": pack_id,
                "returncode": result.returncode,
                "seconds": round(time.monotonic() - started, 3),
                "stdout_tail": result.stdout[-4000:],
                "stderr_tail": result.stderr[-4000:],
            }
        )
        if result.returncode != 0:
            break
    return {
        "schema": "mop-condensation-hydrated-validation/v1",
        "ok": all(row["returncode"] == 0 for row in results),
        "pack_count": len(results),
        "results": results,
    }


def build_run_receipt(
    root: Path,
    *,
    contract_path: Path,
    baseline_path: Path,
    lock_path: Path,
    elimination_ledger_path: Path,
    live_bindings_path: Path,
    cache_root: Path,
    gate_profile: str,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    verification = verify_repository(
        root,
        contract_path=contract_path,
        baseline_path=baseline_path,
        lock_path=lock_path,
        elimination_ledger_path=elimination_ledger_path,
        live_bindings_path=live_bindings_path,
        cache_root=cache_root,
    )
    hydration_root = root / ".mop/packs"
    if verification["ok"]:
        hydration = hydrate_packs(
            lock_path,
            cache_root=cache_root,
            destination_root=hydration_root,
            contract=contract,
            project_root=root,
        )
        gates = run_gates(
            root,
            contract,
            gate_profile,
            extra_env={"MOP_PACK_ROOT": str(hydration_root)},
        )
        packed_validation = (
            run_hydrated_validation(root, verification["packs"], hydration)
            if gate_profile == "full" and gates["ok"]
            else {
                "schema": "mop-condensation-hydrated-validation/v1",
                "ok": True,
                "pack_count": 0,
                "results": [],
                "skipped": "full profile required or an earlier gate failed",
            }
        )
    else:
        hydration = {
            "schema": "mop-condensation-hydration/v1",
            "packs": [],
            "pack_count": 0,
            "skipped": "repository verification failed",
        }
        gates = {
            "profile": gate_profile,
            "ok": False,
            "results": [],
            "skipped": "repository verification failed",
        }
        packed_validation = {
            "schema": "mop-condensation-hydrated-validation/v1",
            "ok": False,
            "pack_count": 0,
            "results": [],
            "skipped": "repository verification failed",
        }
    status = git_status(root)
    clean_release_tree = not status
    elimination_gate_satisfied = (
        not int(verification["eliminations"]["eliminated_LOC"])
        or gate_profile == verification["eliminations"]["required_gate_profile"]
    )
    core = {
        "schema": RUN_RECEIPT_SCHEMA,
        "created_at": _now(),
        "git_head": git_head(root),
        "git_index_tree": git_index_tree(root),
        "git_status": status,
        "working_active_tree_sha256": verification["shape"]["active_tree_sha256"],
        "gate_profile": gate_profile,
        "ok": verification["ok"]
        and gates["ok"]
        and packed_validation["ok"]
        and elimination_gate_satisfied
        and (gate_profile != "full" or clean_release_tree),
        "verification": verification,
        "hydration": hydration,
        "gates": gates,
        "packed_validation": packed_validation,
        "elimination_gate_satisfied": elimination_gate_satisfied,
        "activation": {
            "allowed": False,
            "reason": (
                "live Generation 1 campaign chain must be terminal and release-audited first; "
                "a full release receipt also requires a clean committed tree"
            ),
        },
    }
    return {**core, "receipt_sha256": canonical_sha256(core)}


def _print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _paths(root: Path, args: argparse.Namespace) -> dict[str, Path]:
    return {
        "contract_path": root / args.contract,
        "baseline_path": root / args.baseline,
        "lock_path": root / args.lock,
        "elimination_ledger_path": root / args.eliminations,
        "live_bindings_path": root / args.live_bindings,
        "cache_root": Path(args.cache).expanduser().resolve(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--contract", default="condensation.json")
    parser.add_argument("--baseline", default="condensation.baseline.json")
    parser.add_argument("--lock", default="condensation.packs.lock.json")
    parser.add_argument("--eliminations", default="condensation.eliminations.json")
    parser.add_argument("--live-bindings", default="condensation.live-bindings.json")
    parser.add_argument("--cache", default=os.environ.get("MOP_PACK_CACHE", "~/.cache/mop/packs"))
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline")
    subparsers.add_parser("measure")
    subparsers.add_parser("verify")
    subparsers.add_parser("plan")
    hydrate = subparsers.add_parser("hydrate")
    hydrate.add_argument("--destination", default=".mop/packs")
    run = subparsers.add_parser("run")
    run.add_argument("--profile", default="quick")
    args = parser.parse_args(argv)

    try:
        root = Path(args.root).resolve(strict=True)
        paths = _paths(root, args)
        contract = load_contract(paths["contract_path"])
        if args.command == "baseline":
            baseline = build_baseline(
                root,
                contract_path=paths["contract_path"],
                lock_path=paths["lock_path"],
                elimination_ledger_path=paths["elimination_ledger_path"],
                live_bindings_path=paths["live_bindings_path"],
            )
            _atomic_write_json(paths["baseline_path"], baseline)
            _print_json(
                {
                    "written": str(paths["baseline_path"]),
                    "baseline_sha256": baseline["baseline_sha256"],
                    "source_commit": baseline["source_commit"],
                    "shape": baseline["shape"],
                }
            )
        elif args.command == "measure":
            _print_json(measure_repository(root, contract))
        elif args.command == "verify":
            result = verify_repository(root, **paths)
            _print_json(result)
            return 0 if result["ok"] else 1
        elif args.command == "plan":
            _print_json(
                {
                    "recommended_target_active_repo_LOC": contract["recommended_target_active_repo_loc"],
                    "stretch_target_runtime_core_LOC": contract["stretch_target_runtime_core_loc"],
                    "fallback_ladder": contract["fallback_ladder"],
                    "planned_packs": contract["planned_packs"],
                    "waves": contract["waves"],
                    "activation": contract["activation"],
                }
            )
        elif args.command == "hydrate":
            destination = _safe_relative(args.destination, "hydrate.destination")
            _print_json(
                hydrate_packs(
                    paths["lock_path"],
                    cache_root=paths["cache_root"],
                    destination_root=root / destination,
                    contract=contract,
                    project_root=root,
                )
            )
        else:
            receipt = build_run_receipt(root, gate_profile=args.profile, **paths)
            _print_json(receipt)
            return 0 if receipt["ok"] else 1
    except (CondensationError, OSError, subprocess.SubprocessError) as exc:
        _print_json({"ok": False, "error": str(exc)})
        return 75
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
