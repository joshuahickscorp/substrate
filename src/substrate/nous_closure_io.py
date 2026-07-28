"""Content-addressed I/O for the Substrate Nous Closure namespace."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from substrate import nous_closure_config as C

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG = ROOT / "configs" / "substrate" / "nous_closure"
EVIDENCE = ROOT / "evidence" / "substrate" / "nous_closure"
RUNS = ROOT / "runs" / "substrate" / "nous_closure"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "nous_closure"
MEDIA = ARTIFACTS / "sandbox_media"
EXTERNAL_REVIEW = ARTIFACTS / "external_review"
STOP = RUNS / "STOP"


class Refused(RuntimeError):
    """A closure authority, receipt, or path violated its frozen contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode:
        raise Refused(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def ref_or_none(ref: str, *, peel: bool = False) -> str | None:
    expression = f"{ref}^{{}}" if peel else ref
    result = subprocess.run(
        ["git", "rev-parse", "--verify", expression],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_digest() -> str:
    rows: list[tuple[str, str]] = []
    for root_name in ("src/substrate", "configs/substrate/nous_closure"):
        root = ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(path for path in root.rglob("*") if path.is_file()):
            if "__pycache__" in path.parts:
                continue
            rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    return digest(rows)


def authority(
    schema: str,
    payload: dict[str, Any],
    *,
    status: str = "implemented",
) -> dict[str, Any]:
    body = {
        "schema": schema,
        "program": C.PROGRAM,
        "scientific_status": status,
        "configuration_digest": C.configuration_digest(),
        "source_digest": source_digest(),
        **payload,
        "activation": False,
    }
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_bytes(value) + b"\n"
    existing = path.read_bytes() if path.exists() else None
    if existing == payload:
        return path
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def write_text(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value.encode("utf-8")
    if path.exists() and path.read_bytes() == payload:
        return path
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} is not a JSON object")
    claimed = value.get("sha256")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} authority digest mismatch")
    if _contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    return value


def _contains_true_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key == "activation" and child is not False for key, child in value.items()) or any(
            _contains_true_activation(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_true_activation(child) for child in value)
    return False


def stop() -> Path:
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.write_text("operator stop\n", encoding="utf-8")
    return STOP


def resume() -> None:
    STOP.unlink(missing_ok=True)
