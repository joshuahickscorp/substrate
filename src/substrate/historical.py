"""Hash-bound access to immutable predecessor evidence.

Historical product names and filenames exist only in the migration authority. Active modules use neutral
aliases and this reader refuses any object whose bytes no longer match the recorded SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path


class Refused(RuntimeError):
    """A predecessor object that is missing, unknown, or no longer immutable."""


def repository() -> Path:
    return Path(os.environ.get("SUBSTRATE_REPOSITORY_ROOT", Path(__file__).resolve().parents[2])).expanduser().resolve()


def authority_path() -> Path:
    return repository() / "evidence" / "substrate" / "v1" / "SUBSTRATE_HISTORICAL_EVIDENCE_AUTHORITY.json"


@lru_cache(maxsize=1)
def authority() -> dict:
    document = json.loads(authority_path().read_text())
    if document.get("migration_manifest") is not True:
        raise Refused("historical evidence authority is not a migration manifest")
    return document


def root(alias: str) -> Path:
    try:
        relative = authority()["roots"][alias]
    except KeyError as exc:
        raise Refused(f"unknown historical root alias {alias!r}") from exc
    return repository() / relative


def artifact(alias: str, *, verify: bool = True) -> Path:
    try:
        record = authority()["objects"][alias]
    except KeyError as exc:
        raise Refused(f"unknown historical object alias {alias!r}") from exc
    path = root(record["root"]) / record["name"]
    if not path.is_file():
        raise Refused(f"historical object {alias!r} is missing")
    if verify:
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != record["sha256"]:
            raise Refused(f"historical object {alias!r} failed its immutable SHA-256 binding")
    return path


def predecessor_evidence(name: str, subdir: str = "") -> Path:
    return root("predecessor_evidence") / subdir / name


def predecessor_receipt(name: str) -> Path:
    return root("predecessor_receipts") / name


def archived(relative: str) -> Path:
    return root("historical_archive") / relative


def verify_all() -> dict:
    rows = {}
    for alias in sorted(authority()["objects"]):
        try:
            path = artifact(alias)
            rows[alias] = {"ok": True, "path": path.relative_to(repository()).as_posix()}
        except (OSError, ValueError, json.JSONDecodeError, Refused) as exc:
            rows[alias] = {"ok": False, "error": str(exc)}
    failed = sorted(alias for alias, row in rows.items() if not row["ok"])
    return {"objects": rows, "failed": failed, "all_pass": not failed, "activation": False}


def seal_authority() -> Path:
    from substrate import evidence as io

    document = {key: value for key, value in authority().items() if key not in {"program", "source_commit", "sha256"}}
    path = io.seal("SUBSTRATE_HISTORICAL_EVIDENCE_AUTHORITY.json", document)
    authority.cache_clear()
    return path


def main(argv=None) -> None:
    argv = argv or sys.argv[1:]
    if argv and argv[0] != "seal":
        raise ValueError(argv)
    path = seal_authority()
    sealed = path.relative_to(repository()).as_posix() if repository() in path.parents else str(path)
    print(json.dumps({"sealed": sealed, **verify_all()}, indent=2))


if __name__ == "__main__":
    main()
