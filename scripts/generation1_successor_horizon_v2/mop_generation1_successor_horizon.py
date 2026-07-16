#!/usr/bin/env python3
"""Run one bounded operation of the Generation 1 successor horizon v2."""

# ruff: noqa: E402 - direct execution must bootstrap the repository before MOP imports

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
for _source_root in (REPO_BOOTSTRAP / "src", REPO_BOOTSTRAP):
    if str(_source_root) not in sys.path:
        sys.path.insert(0, str(_source_root))

from mop.studies import generation1_successor_horizon_v2 as horizon


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    admit = commands.add_parser("admit", help="seal admission from the complete v1 horizon")
    admit.add_argument("--output", type=Path, default=horizon.DEFAULT_ADMISSION)
    admit.add_argument("--parent-result", type=Path, default=horizon.DEFAULT_PARENT_RESULT)
    admit.add_argument(
        "--parent-verification",
        type=Path,
        default=horizon.DEFAULT_PARENT_VERIFICATION,
    )
    admit.add_argument(
        "--parent-report-receipt",
        type=Path,
        default=horizon.DEFAULT_PARENT_REPORT_RECEIPT,
    )

    shard = commands.add_parser("run-shard", help="run one result-gated v2 epoch shard")
    shard.add_argument("--root", type=Path, default=horizon.DEFAULT_ROOT)
    shard.add_argument("--admission", type=Path, default=horizon.DEFAULT_ADMISSION)
    shard.add_argument("--epoch-index", type=int, choices=range(len(horizon.EPOCH_IDS)), required=True)
    shard.add_argument("--lane", choices=("d1", "mechanics"), required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--idle-workers", type=int, default=horizon.IDLE_WORKERS)
    shard.add_argument("--hawking-workers", type=int, default=horizon.HAWKING_WORKERS)
    shard.add_argument("--retry-limit", type=int, default=horizon.RETRY_LIMIT)

    classify = commands.add_parser("classify", help="seal one v2 epoch classification barrier")
    classify.add_argument("--root", type=Path, default=horizon.DEFAULT_ROOT)
    classify.add_argument("--admission", type=Path, default=horizon.DEFAULT_ADMISSION)
    classify.add_argument(
        "--epoch-index",
        type=int,
        choices=range(len(horizon.EPOCH_IDS)),
        required=True,
    )

    aggregate = commands.add_parser("aggregate", help="aggregate all five v2 epochs")
    aggregate.add_argument("--root", type=Path, default=horizon.DEFAULT_ROOT)
    aggregate.add_argument("--admission", type=Path, default=horizon.DEFAULT_ADMISSION)
    aggregate.add_argument("--output", type=Path, default=horizon.DEFAULT_RESULT)

    verify = commands.add_parser("verify", help="run the separately authored v2 artifact verifier")
    verify.add_argument("--result", type=Path, default=horizon.DEFAULT_RESULT)
    verify.add_argument("--output", type=Path, default=horizon.DEFAULT_VERIFICATION)

    report = commands.add_parser("report", help="render the v2 evidence report and receipt")
    report.add_argument("--result", type=Path, default=horizon.DEFAULT_RESULT)
    report.add_argument("--verification", type=Path, default=horizon.DEFAULT_VERIFICATION)
    report.add_argument("--report", type=Path, default=horizon.DEFAULT_REPORT)
    report.add_argument("--receipt", type=Path, default=horizon.DEFAULT_REPORT_RECEIPT)
    return parser


def dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "admit":
        return horizon.admit(
            output=arguments.output,
            parent_result_path=arguments.parent_result,
            parent_verification_path=arguments.parent_verification,
            parent_report_receipt_path=arguments.parent_report_receipt,
        )
    if arguments.command == "run-shard":
        count = horizon.D1_SHARD_COUNT if arguments.lane == "d1" else horizon.MECHANICS_SHARD_COUNT
        if not 0 <= arguments.shard_index < count:
            raise ValueError(f"{arguments.lane} shard index must be in [0, {count - 1}]")
        return horizon.run_shard(
            root=arguments.root,
            admission_path=arguments.admission,
            epoch_index=arguments.epoch_index,
            lane=arguments.lane,
            shard_index=arguments.shard_index,
            idle_workers=arguments.idle_workers,
            hawking_workers=arguments.hawking_workers,
            retry_limit=arguments.retry_limit,
        )
    if arguments.command == "classify":
        return horizon.classify_epoch(
            root=arguments.root,
            admission_path=arguments.admission,
            epoch_index=arguments.epoch_index,
        )
    if arguments.command == "aggregate":
        return horizon.aggregate(
            root=arguments.root,
            admission_path=arguments.admission,
            output=arguments.output,
        )
    if arguments.command == "verify":
        verifier = importlib.import_module("mop.studies.generation1_successor_horizon_v2_verify")
        result: dict[str, Any] = verifier.verify(
            result_path=arguments.result,
            output=arguments.output,
        )
        return result
    if arguments.command == "report":
        return horizon.render_report(
            result_path=arguments.result,
            verification_path=arguments.verification,
            report_path=arguments.report,
            receipt_path=arguments.receipt,
        )
    raise ValueError(f"unsupported successor horizon v2 command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    result = dispatch(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
