"""Evidence, configuration, and unit receipt ownership for Substrate v3."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from substrate import evidence as v1

ROOT = v1.ROOT
EVIDENCE = ROOT / "evidence" / "substrate" / "v3"
RUNS = ROOT / "runs" / "substrate" / "v3"
ARTIFACTS = ROOT / "evidence" / "artifacts" / "substrate" / "v3"
CONFIGS = ROOT / "ops" / "configs" / "substrate" / "v3"
STATE = v1.STATE / "v3"
STOP = STATE / "stop"
PROGRAM = "substrate-v3"
ACTIVATION = False

# The publication mechanics are shared; this module still owns the v3 roots,
# seal defaults, and compatibility surface.
atomic_write = v1.atomic_write
sha_obj = v1.sha_obj


class Refused(RuntimeError):
    """A v3 publication or state operation that fails closed."""


def commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def source_inventory() -> dict[str, str]:
    roots = (ROOT / "src" / "substrate", ROOT / "tests" / "substrate")
    return {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for source_root in roots
        for path in sorted(source_root.rglob("*.py"))
    }


def source_digest() -> str:
    return sha_obj(source_inventory())


def sealed_document(document: dict) -> dict:
    body = json.loads(json.dumps({key: value for key, value in document.items() if key != "sha256"}, default=str))
    body.setdefault("program", PROGRAM)
    body.setdefault("source_commit", commit())
    body.setdefault("source_digest", source_digest())
    body.setdefault("activation", ACTIVATION)
    if body["activation"] is not False:
        raise Refused("v3 activation must remain false")
    body["sha256"] = sha_obj({key: value for key, value in body.items() if key != "sha256"})
    return body


def seal(name: str, document: dict, *, artifact: bool = False) -> Path:
    root = ARTIFACTS if artifact else EVIDENCE
    return atomic_write(root / name, json.dumps(sealed_document(document), indent=2))


def seal_markdown(name: str, text: str, *, artifact: bool = True) -> Path:
    if "activation=true" in text.replace(" ", "").lower():
        raise Refused("v3 markdown cannot declare activation true")
    root = ARTIFACTS if artifact else EVIDENCE
    return atomic_write(root / name, text)


def run_json(relative: str, document: dict) -> Path:
    body = json.loads(json.dumps(document, default=str))
    body.setdefault("program", PROGRAM)
    body.setdefault("activation", ACTIVATION)
    if body["activation"] is not False:
        raise Refused("v3 run receipts require activation false")
    return atomic_write(RUNS / relative, json.dumps(body, indent=2))


def config_json(relative: str, document: dict) -> Path:
    body = sealed_document(document)
    return atomic_write(CONFIGS / relative, json.dumps(body, indent=2))


def load(name: str, *, artifact: bool = False) -> dict:
    root = ARTIFACTS if artifact else EVIDENCE
    document = json.loads((root / name).read_text())
    expected = sha_obj({key: value for key, value in document.items() if key != "sha256"})
    if document.get("sha256") != expected or document.get("activation") is not False:
        raise Refused(f"invalid v3 seal for {name}")
    return document


def validate_document(document: dict) -> bool:
    expected = sha_obj({key: value for key, value in document.items() if key != "sha256"})
    return document.get("sha256") == expected and document.get("activation") is False


def stop() -> Path:
    return atomic_write(STOP, "operator stop\n")


def resume() -> None:
    STOP.unlink(missing_ok=True)
