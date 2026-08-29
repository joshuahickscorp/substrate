"""Content-addressed local authority for Cognitive Material Genesis II."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from substrate import genesis2_config as C2
from substrate.final_revision_io import (
    Refused,
    canonical_bytes,
    contains_true_activation,
    digest,
    file_digest,
    git,
    load_json,
    ref_or_none,
    write_text,
)
from substrate.final_revision_io import write_json as _write_json

ROOT = Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).resolve()
CONFIG = ROOT / "ops" / "configs" / "substrate" / "genesis2"
EVIDENCE = ROOT / "evidence" / "substrate" / "genesis2"
RUNS = ROOT / "runs" / "substrate" / "genesis2"
ARTIFACTS = ROOT / "evidence" / "artifacts" / "substrate" / "genesis2"
STOP = RUNS / "STOP"
MASTER_PLAN_RELATIVE = Path("docs/archive/experiments/genesis2/MASTER_PLAN.md")


def source_digest() -> str:
    """Digest exactly the Genesis II implementation and frozen master plan."""
    rows: list[tuple[str, str]] = []
    for path in sorted((ROOT / "src" / "substrate").glob("genesis2*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    for path in sorted((ROOT / "tests" / "substrate").glob("test_genesis2*.py")):
        rows.append((str(path.relative_to(ROOT)), file_digest(path)))
    plan = ROOT / MASTER_PLAN_RELATIVE
    if plan.is_file():
        rows.append((str(plan.relative_to(ROOT)), file_digest(plan)))
    return digest(rows)


def authority(schema: str, payload: dict[str, Any], *, status: str = "implemented") -> dict[str, Any]:
    body = {
        "schema": schema,
        "program": C2.PROGRAM,
        "scientific_status": status,
        "configuration_digest": C2.configuration_digest(),
        "source_digest": source_digest(),
        **payload,
        "activation": False,
    }
    body.pop("sha256", None)
    body["sha256"] = digest(body)
    return body


def write_json(path: Path, value: dict[str, Any]) -> Path:
    if contains_true_activation(value):
        raise Refused(f"refusing to write {path}: the document enables activation")
    return _write_json(path, value)


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
