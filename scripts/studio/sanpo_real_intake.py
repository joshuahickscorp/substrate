#!/usr/bin/env python
"""Plan, execute, resume, or verify the bounded SANPO-Real natural-video intake.

Dry-run is the default. ``--execute`` is the only mode that downloads official source objects.
The intake remains acquisition-only: it never imports Torch, loads an encoder, or promotes an
F8/F16 result.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from mop.studio.sanpo_real_intake import (
    MIN_FREE_DISK_BYTES,
    RECEIPT_SCHEMA,
    GCSJSONClient,
    SanpoIntakeError,
    build_intake_plan,
    dry_run_receipt,
    execute_intake_plan,
    verify_existing_intake,
    write_receipt,
)

DEFAULT_DESTINATION = Path("data/raw/sanpo_real_smoke_v0")
DEFAULT_PROOF = Path("proof/SANPO_REAL_SMOKE_INTAKE.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="perform generation-pinned downloads")
    mode.add_argument("--verify", action="store_true", help="re-hash an existing completed intake")
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--proof", type=Path, default=DEFAULT_PROOF)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=MIN_FREE_DISK_BYTES / 1e9,
        help="decimal GB floor; values below the current-host 40 GB floor are rejected",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--download-workers",
        type=int,
        default=8,
        help="parallel GCS connections, capped at 8; hashing and safety remain per object",
    )
    return parser


def _failure_receipt(args: argparse.Namespace, exc: Exception) -> dict:
    mode = "verify" if args.verify else ("execute" if args.execute else "dry-run")
    disk_root = args.destination
    while not disk_root.exists() and disk_root != disk_root.parent:
        disk_root = disk_root.parent
    try:
        free = shutil.disk_usage(disk_root).free
    except OSError:
        free = None
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": mode,
        "all_ok": False,
        "download_active": False,
        "destination": str(args.destination.resolve()),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "free_disk_bytes": free,
        "claim_boundary": {
            "status": "blocked-fail-closed",
            "scientific_promotion": False,
            "f8_f16_trusted_provenance_satisfied": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    floor = int(args.min_free_gb * 1e9)
    if floor < MIN_FREE_DISK_BYTES:
        print(
            f"--min-free-gb cannot lower the {MIN_FREE_DISK_BYTES / 1e9:g} GB current-host floor",
            file=sys.stderr,
        )
        return 2
    try:
        if args.verify:
            receipt = verify_existing_intake(args.destination)
            receipt["mode"] = "verified-existing"
            write_receipt(args.proof, receipt, floor_bytes=floor)
        else:
            client = GCSJSONClient(timeout=args.timeout)
            disk_root = args.destination.parent
            while not disk_root.exists() and disk_root != disk_root.parent:
                disk_root = disk_root.parent
            plan = build_intake_plan(
                client=client,
                disk_root=disk_root,
                min_free_disk_bytes=floor,
            )
            if args.execute:
                receipt = execute_intake_plan(
                    plan,
                    destination=args.destination,
                    proof_path=args.proof,
                    timeout=args.timeout,
                    download_workers=args.download_workers,
                )
            else:
                receipt = dry_run_receipt(plan, destination=args.destination)
                write_receipt(args.proof, receipt, floor_bytes=floor)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0
    except (SanpoIntakeError, OSError, ValueError, json.JSONDecodeError) as exc:
        failure = _failure_receipt(args, exc)
        try:
            write_receipt(args.proof, failure, floor_bytes=floor)
        except Exception as write_exc:  # noqa: BLE001
            print(f"could not write failure proof: {write_exc}", file=sys.stderr)
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
