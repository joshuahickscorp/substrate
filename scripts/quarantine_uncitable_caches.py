#!/usr/bin/env python
"""Move legacy cache directories out of the active data plane without deleting them.

An array can be internally readable while still being scientifically uncitable. Active caches must
pass strict manifest, referent, form, and encoder-receipt validation. This command inventories every
file before a reversible same-filesystem rename into ``data/cache_quarantine`` and writes a durable
receipt. It never removes or rewrites an array.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from mop.config import REPO_ROOT
from mop.substrate.cache_tools import validate_cache
from mop.substrate.events import sha256_file

SCHEMA = "mop-cache-quarantine/v1"


def _inventory(store: Path) -> list[dict]:
    return [
        {
            "path": str(path.relative_to(store)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(store.rglob("*"))
        if path.is_file()
    ]


def build_quarantine_receipt(
    active_root: Path,
    quarantine_root: Path,
    *,
    execute: bool,
) -> dict:
    active_root = active_root.resolve()
    quarantine_root = quarantine_root.resolve()
    stores = sorted(path for path in active_root.glob("*") if path.is_dir() and (path / "meta.json").exists())
    records: list[dict] = []
    for store in stores:
        problems = validate_cache(store, citable=True)
        destination = quarantine_root / store.name
        record = {
            "name": store.name,
            "source": str(store),
            "destination": str(destination),
            "citable": not problems,
            "validation_problems": problems,
            "files": _inventory(store),
            "action": "retained-active" if not problems else "would-quarantine",
        }
        if problems and execute:
            if destination.exists():
                raise FileExistsError(
                    f"quarantine destination already exists for {store.name}: {destination}"
                )
            quarantine_root.mkdir(parents=True, exist_ok=True)
            store.rename(destination)
            record["action"] = "quarantined"
            record["source_exists_after"] = store.exists()
            record["destination_exists_after"] = destination.exists()
        records.append(record)
    moved = [record for record in records if record["action"] == "quarantined"]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "active_root": str(active_root),
        "quarantine_root": str(quarantine_root),
        "execute": execute,
        "policy": (
            "Only strict-citable caches remain active. Quarantine is a reversible rename; no file "
            "or array is deleted or rewritten."
        ),
        "records": records,
        "summary": {
            "examined": len(records),
            "citable_retained": sum(1 for record in records if record["citable"]),
            "uncitable": sum(1 for record in records if not record["citable"]),
            "quarantined": len(moved),
        },
        "all_ok": all(
            record["citable"]
            or not execute
            or (record.get("source_exists_after") is False and record.get("destination_exists_after") is True)
            for record in records
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-root", type=Path, default=REPO_ROOT / "data/cache")
    parser.add_argument("--quarantine-root", type=Path, default=REPO_ROOT / "data/cache_quarantine")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "proof/CACHE_QUARANTINE_AUDIT.json")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_quarantine_receipt(
        args.active_root,
        args.quarantine_root,
        execute=bool(args.execute),
    )
    out = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["summary"], indent=2))
    return 0 if receipt["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
