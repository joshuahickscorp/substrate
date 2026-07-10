#!/usr/bin/env python
"""Execute local frontier preflights and write the complete non-F localization audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.frontier_localization import (
    DEFAULT_AUDIT_PROOF,
    DEFAULT_PREFLIGHT_PROOF,
    DEFAULT_RUN_RECEIPT,
    PREFLIGHTS,
    ScientificLaunchBlocked,
    assert_scientific_launch_ready,
    write_frontier_receipts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-receipt", type=Path, default=DEFAULT_RUN_RECEIPT)
    parser.add_argument("--preflight-proof", type=Path, default=DEFAULT_PREFLIGHT_PROOF)
    parser.add_argument("--audit-proof", type=Path, default=DEFAULT_AUDIT_PROOF)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260709, 20260710, 20260711])
    parser.add_argument(
        "--reuse-existing-preflight",
        action="store_true",
        help="rebuild the audit from the existing timing-bearing preflight without rerunning it",
    )
    parser.add_argument(
        "--assert-scientific-ready",
        choices=sorted(PREFLIGHTS),
        help="fail closed unless receipt-backed scientific prerequisites are supplied",
    )
    parser.add_argument(
        "--receipt-map",
        type=Path,
        help="JSON mapping from requirement key to repository-local receipt path",
    )
    args = parser.parse_args()
    if args.assert_scientific_ready:
        supplied = {}
        if args.receipt_map:
            supplied = json.loads(args.receipt_map.read_text())
            if not isinstance(supplied, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in supplied.items()
            ):
                parser.error("--receipt-map must contain a string-to-string JSON mapping")
        try:
            assert_scientific_launch_ready(args.assert_scientific_ready, supplied)
        except ScientificLaunchBlocked as exc:
            parser.exit(2, f"{exc}\n")
        print(json.dumps({"scientific_launch_ready": args.assert_scientific_ready}, indent=2))
        return 0

    preflight, audit = write_frontier_receipts(
        run_receipt=args.run_receipt,
        preflight_proof=args.preflight_proof,
        audit_proof=args.audit_proof,
        seeds=tuple(args.seeds),
        reuse_existing_preflight=bool(args.reuse_existing_preflight),
    )
    print(
        json.dumps(
            {
                "preflight_receipt": str(args.preflight_proof),
                "audit_receipt": str(args.audit_proof),
                "target_count": preflight["target_count"],
                "all_mechanics_verified": preflight["all_mechanics_verified"],
                "all_scientific_paths_fail_closed": preflight["all_scientific_paths_fail_closed"],
                "tagged_non_f_count": audit["coverage"]["tagged_non_f_count"],
                "current_stale_planning_tag_count": audit["coverage"]["current_stale_planning_tag_count"],
                "localization_counts": audit["coverage"]["localization_counts"],
                "measured_hardware_blocked_count": audit["coverage"]["measured_hardware_blocked_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
