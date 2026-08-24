"""Deterministic encoding and atomic file writes for product state."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from substrate.product.contracts import ProductRefused


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON keys before data reaches a signed contract."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant {value!r}")


def _read_regular_text(path: Path) -> str:
    """Read one stable, regular, non-linked text file without a path race."""

    descriptor: int | None = None
    try:
        expected = path.lstat()
        regular_file = stat.S_ISREG(expected.st_mode) and not path.is_symlink() and expected.st_nlink == 1
        if not regular_file:
            raise ProductRefused(f"cannot read {path.name}: expected a regular non-linked file")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        observed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(observed.st_mode)
            or observed.st_nlink != 1
            or (observed.st_dev, observed.st_ino, observed.st_size) != (expected.st_dev, expected.st_ino, expected.st_size)
        ):
            raise ProductRefused(f"cannot read {path.name}: file changed before it could be opened safely")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = None
            return handle.read()
    except OSError as exc:
        raise ProductRefused(f"cannot read {path.name}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def fsync_directory(directory: Path) -> None:
    """Durably publish a directory-entry change on supported POSIX filesystems."""

    descriptor: int | None = None
    try:
        descriptor = os.open(directory, os.O_RDONLY)
        os.fsync(descriptor)
    except OSError as exc:
        raise ProductRefused(f"cannot durably sync directory {directory}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ProductRefused(f"value is not canonical JSON: {exc}") from exc


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            _read_regular_text(path),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonstandard_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ProductRefused(f"cannot read {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProductRefused(f"{path.name} must contain a JSON object")
    return value


def read_json_lines(path: Path) -> list[dict[str, Any]]:
    try:
        lines = _read_regular_text(path).splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ProductRefused(f"cannot read {path.name}: {exc}") from exc
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            raise ProductRefused(f"{path.name} has a blank line at {index}")
        try:
            value = json.loads(line, object_pairs_hook=_strict_object, parse_constant=_reject_nonstandard_constant)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ProductRefused(f"{path.name} has invalid JSON at {index}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProductRefused(f"{path.name} row {index} must be an object")
        rows.append(value)
    return rows


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if temporary is None:
            raise ProductRefused(f"cannot create temporary file for {path.name}")
        os.replace(temporary, path)
        temporary = None
        fsync_directory(path.parent)
    except OSError as exc:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise ProductRefused(f"cannot write {path.name}: {exc}") from exc


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    try:
        text = json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ProductRefused(f"value is not JSON serializable: {exc}") from exc
    atomic_write_text(path, f"{text}\n")


def atomic_write_json_lines(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_text(path, "".join(f"{canonical_json(row)}\n" for row in rows))


def durable_unlink(path: Path) -> None:
    """Remove a transaction marker and durably publish that removal."""

    try:
        path.unlink()
    except OSError as exc:
        raise ProductRefused(f"cannot remove {path.name}: {exc}") from exc
    fsync_directory(path.parent)
