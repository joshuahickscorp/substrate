"""Evidence, configuration, and raw receipt ownership for Substrate v4."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

from substrate import evidence as v1

ROOT = v1.ROOT
EVIDENCE = ROOT / "evidence" / "substrate" / "v4"
RUNS = ROOT / "runs" / "substrate" / "v4"
ARTIFACTS = ROOT / "artifacts" / "substrate" / "v4"
CONFIGS = ROOT / "configs" / "substrate" / "v4"
STATE = v1.STATE / "v4"
STOP = STATE / "stop"
PROGRAM = "substrate-v4"
ACTIVATION = False


class Refused(RuntimeError):
    """A v4 state or publication operation failed closed."""


def sha_obj(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_inventory() -> dict[str, str]:
    roots = (ROOT / "src" / "substrate", ROOT / "tests" / "substrate")
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for source_root in roots for path in sorted(source_root.rglob("*.py"))
    }


def source_digest() -> str:
    return sha_obj(source_inventory())


def atomic_write(path: Path, payload: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def atomic_write_bytes(path: Path, payload: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def sealed_document(document: dict) -> dict:
    body = json.loads(json.dumps({key: value for key, value in document.items() if key != "sha256"}, default=str))
    body.setdefault("program", PROGRAM)
    body.setdefault("source_commit", commit())
    body.setdefault("source_digest", source_digest())
    body.setdefault("activation", ACTIVATION)
    if body["activation"] is not False:
        raise Refused("v4 activation must remain false")
    body["sha256"] = sha_obj({key: value for key, value in body.items() if key != "sha256"})
    return body


def seal(name: str, document: dict, *, artifact: bool = False) -> Path:
    root = ARTIFACTS if artifact else EVIDENCE
    return atomic_write(root / name, json.dumps(sealed_document(document), indent=2))


def seal_markdown(name: str, text: str, *, artifact: bool = True) -> Path:
    if "activation=true" in text.replace(" ", "").lower():
        raise Refused("v4 markdown cannot declare activation true")
    root = ARTIFACTS if artifact else EVIDENCE
    return atomic_write(root / name, text)


def run_json(relative: str, document: dict) -> Path:
    body = json.loads(json.dumps(document, default=str))
    body.setdefault("program", PROGRAM)
    body.setdefault("activation", ACTIVATION)
    if body["activation"] is not False:
        raise Refused("v4 raw receipts require activation false")
    return atomic_write(RUNS / relative, json.dumps(body, indent=2))


def config_json(relative: str, document: dict) -> Path:
    return atomic_write(CONFIGS / relative, json.dumps(sealed_document(document), indent=2))


def load(name: str, *, artifact: bool = False) -> dict:
    root = ARTIFACTS if artifact else EVIDENCE
    document = json.loads((root / name).read_text())
    expected = sha_obj({key: value for key, value in document.items() if key != "sha256"})
    if document.get("sha256") != expected or document.get("activation") is not False:
        raise Refused(f"invalid v4 seal for {name}")
    return document


def stop() -> Path:
    return atomic_write(STOP, "operator stop\n")


def resume() -> None:
    STOP.unlink(missing_ok=True)
