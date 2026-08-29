"""Content-addressed I/O for Substrate Cognitive Material Genesis.

The hashing, atomic write, activation-guard and git helpers are shared with the
Final Revision namespace; only the paths, the program identity and the source
digest differ.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from substrate import genesis_config as C
from substrate.final_revision_io import (
    Refused,
    canonical_bytes,
    contains_true_activation,
    digest,
    file_digest,
    git,
    ref_or_none,
    write_text,
)
from substrate.final_revision_io import load_json as _load_json
from substrate.final_revision_io import write_json as _write_json

__all__ = [
    "ARTIFACTS",
    "CONFIG",
    "EVIDENCE",
    "ROOT",
    "RUNS",
    "STOP",
    "Refused",
    "authority",
    "canonical_bytes",
    "contains_true_activation",
    "digest",
    "file_digest",
    "git",
    "load_json",
    "read_optional",
    "ref_or_none",
    "resume",
    "source_digest",
    "stop",
    "stopped",
    "write_json",
    "write_text",
]

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG = ROOT / "ops" / "configs" / "substrate" / "genesis"
EVIDENCE = ROOT / "evidence" / "substrate" / "genesis"
RUNS = ROOT / "runs" / "substrate" / "genesis"
ARTIFACTS = ROOT / "evidence" / "artifacts" / "substrate" / "genesis"
STOP = RUNS / "STOP"


def source_digest() -> str:
    """Digest of every genesis source file plus the frozen configuration."""
    rows = []
    for path in sorted((ROOT / "src" / "substrate").glob("genesis*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    frozen = CONFIG / "frozen_configuration.json"
    if frozen.is_file():
        rows.append((str(frozen.relative_to(ROOT)), file_digest(frozen)))
    return digest(rows)


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
    """Refuse to write an artefact that enables activation.

    The inherited writer only guards the read path, so a document that set an
    activation key true would land on disk and be caught later, if at all.
    Guarding the write means it never lands.
    """
    if contains_true_activation(value):
        raise Refused(f"refusing to write {path}: the document enables activation")
    return _write_json(path, value)


def load_json(path: Path) -> dict[str, Any]:
    return _load_json(path)


def read_optional(name: str) -> dict[str, Any] | None:
    path = EVIDENCE / name
    return load_json(path) if path.is_file() else None


def stopped() -> bool:
    return STOP.is_file()


def stop() -> Path:
    STOP.parent.mkdir(parents=True, exist_ok=True)
    STOP.write_text("operator stop\n")
    return STOP


def resume() -> None:
    STOP.unlink(missing_ok=True)
