#!/usr/bin/env python
"""Live-authority dry-run, or explicit post-CM7 train/validation AV acquisition.

Dry-run is the default and never opens a media URL. Test media acquisition is deliberately absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from mop.studio.wikimedia_av_intake import (
    DEFAULT_DESTINATION,
    DEFAULT_DRY_RUN_PROOF,
    DEFAULT_MANIFEST,
    RECEIPT_SCHEMA,
    WikimediaAVIntakeError,
    WikimediaCommonsAPI,
    build_dry_run_plan,
    execute_train_validation,
    load_manifest,
    validate_dry_run_plan,
    write_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--proof", type=Path, default=DEFAULT_DRY_RUN_PROOF)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument(
        "--execute-train-validation",
        action="store_true",
        help="download and ffprobe train/validation originals; test remains locked",
    )
    parser.add_argument(
        "--confirm-cm7-complete",
        action="store_true",
        help="required execution acknowledgement; process detection still fails closed",
    )
    return parser


def _manifest_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _failure(args: argparse.Namespace, exc: Exception) -> dict:
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "mode": "execute-train-validation" if args.execute_train_validation else "metadata-only-dry-run",
        "all_ok": False,
        "download_active": False,
        "test_media_accessed": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "claim_boundary": {
            "status": "blocked-fail-closed",
            "scientific_promotion": False,
            "al3_ready": False,
            "dr15_ready": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.confirm_cm7_complete and not args.execute_train_validation:
        print("--confirm-cm7-complete is meaningful only with execution", file=sys.stderr)
        return 2
    if args.execute_train_validation and not args.confirm_cm7_complete:
        print("execution requires --confirm-cm7-complete", file=sys.stderr)
        return 2
    try:
        manifest = load_manifest(args.manifest)
        plan = build_dry_run_plan(
            manifest,
            client=WikimediaCommonsAPI(timeout=args.timeout),
            disk_root=args.destination,
        )
        plan["manifest_file"] = str(args.manifest.resolve())
        plan["manifest_file_sha256"] = _manifest_file_sha256(args.manifest)
        validate_dry_run_plan(plan)
        if args.execute_train_validation:
            receipt = execute_train_validation(plan, destination=args.destination, timeout=args.timeout)
            receipt["preflight_plan_sha256"] = plan["plan_sha256"]
            receipt["manifest_file"] = plan["manifest_file"]
            receipt["manifest_file_sha256"] = plan["manifest_file_sha256"]
        else:
            receipt = plan
        write_receipt(args.proof, receipt)
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt.get("all_ok") else 1
    except (OSError, ValueError, json.JSONDecodeError, WikimediaAVIntakeError) as exc:
        receipt = _failure(args, exc)
        try:
            write_receipt(args.proof, receipt)
        except OSError as write_exc:
            print(f"could not write failure proof: {write_exc}", file=sys.stderr)
        print(json.dumps(receipt, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
