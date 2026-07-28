"""Content-addressed I/O for the Substrate final revision."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from substrate import final_revision_config as C

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG = ROOT / "configs" / "substrate" / "final_revision"
EVIDENCE = ROOT / "evidence" / "substrate" / "final_revision"
RUNS = ROOT / "runs" / "substrate" / "final_revision"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "final_revision"
READINESS = ARTIFACTS / "real_world_sandbox_readiness"
STOP = RUNS / "STOP"


class Refused(RuntimeError):
    """A final-revision authority, receipt, or state violated its contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def source_digest() -> str:
    rows = []
    for path in sorted((ROOT / "src" / "substrate").glob("final_revision*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    config = CONFIG / "frozen_configuration.json"
    if config.is_file():
        rows.append((str(config.relative_to(ROOT)), file_digest(config)))
    return digest(rows)


def git(*arguments: str, check: bool = True) -> str:
    result = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)
    if check and result.returncode:
        raise Refused(result.stderr.strip() or f"git {' '.join(arguments)} failed")
    return result.stdout.strip()


def ref_or_none(ref: str, *, peel: bool = False) -> str | None:
    expression = f"{ref}^{{}}" if peel else ref
    result = subprocess.run(["git", "rev-parse", "--verify", expression], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def authority(schema: str, payload: dict[str, Any], *, status: str = "implemented") -> dict[str, Any]:
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
    if path.is_file() and path.read_bytes() == payload:
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
    payload = value.encode()
    if path.is_file() and path.read_bytes() == payload:
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


def contains_true_activation(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key == "activation" and child is not False for key, child in value.items()) or any(
            contains_true_activation(child) for child in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_true_activation(child) for child in value)
    return False


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise Refused(f"cannot load {path}: {error}") from error
    if not isinstance(value, dict):
        raise Refused(f"{path} is not a JSON object")
    if contains_true_activation(value):
        raise Refused(f"{path} enables activation")
    claimed = value.get("sha256")
    if isinstance(claimed, str):
        unsigned = dict(value)
        unsigned.pop("sha256")
        if digest(unsigned) != claimed:
            raise Refused(f"{path} authority digest mismatch")
    return value


def stop() -> Path:
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.write_text("operator stop\n")
    return STOP


def resume() -> None:
    STOP.unlink(missing_ok=True)
