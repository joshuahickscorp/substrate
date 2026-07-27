"""Read-only access to sealed MOP paths and schema identities.

This module is the only active compatibility boundary. It never executes a historical runtime and never
provides an ``import mop`` alias.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HISTORICAL_PROGRAM = "mop-substrate-master-v1"
ARCHIVE = "pre-substrate-event-horizon"


def roots(repository: Path) -> dict[str, Path]:
    return {
        "temporal": repository / "proof" / "substrate" / "mop-temporal-core-mechanism-v1",
        "method": repository / "proof" / "method" / "mop-experimental-method-reformation-v1",
        "fastforge": repository / "proof" / "substrate" / "mop-fast-state-plasticity-forge-v1",
        "predecessor": repository / "proof" / "substrate" / HISTORICAL_PROGRAM,
    }


def resolve(repository: Path, reference: str) -> Path:
    namespace, separator, name = reference.rpartition(":")
    if not separator:
        return roots(repository)["predecessor"] / reference
    try:
        return roots(repository)[namespace] / name
    except KeyError as exc:
        raise ValueError(f"unknown historical MOP namespace {namespace!r}") from exc


def active_predecessor(name: str, subdir: str = "") -> Path:
    repository = Path(__file__).resolve().parents[3]
    return roots(repository)["predecessor"] / subdir / name


def run_predecessor(name: str) -> Path:
    repository = Path(__file__).resolve().parents[3]
    return repository / "runs" / "substrate" / HISTORICAL_PROGRAM / name


def archived(repository: Path, relative: str) -> Path:
    """Resolve an immutable predecessor path moved out of the active tree."""
    return repository / "archive" / ARCHIVE / relative


def data_root(repository: Path) -> Path:
    old_environment = os.environ.get("MOP_DATA_ROOT")
    if old_environment:
        from substrate import evidence

        evidence.run_json(
            "mop_data_root.json",
            {
                "schema": "substrate-migration-deprecation/v1",
                "legacy_variable": "MOP_DATA_ROOT",
                "replacement": "SUBSTRATE_DATA_ROOT",
                "value_sha256": hashlib.sha256(old_environment.encode()).hexdigest(),
                "deprecated": True,
                "activation": False,
            },
            "migration",
        )
        return Path(old_environment)
    custody = roots(repository)["temporal"] / "MOP_DATA_CUSTODY_AUTHORITY.json"
    try:
        declared = json.loads(custody.read_text()).get("canonical_root")
        if declared and Path(declared).is_dir():
            return Path(declared)
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    historical = repository.parent / "mop-data"
    return historical if historical.is_dir() else repository.parent / "substrate-data"
