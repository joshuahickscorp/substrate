#!/usr/bin/env python3
"""Materialize one bounded artifact of the categorized successor scaffold."""

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

from mop.studies import generation1_successor_categorized_batch_wave as wave


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    gate = commands.add_parser("materialize-gate", help="materialize one serial D1 transition gate")
    gate.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)
    gate.add_argument("--gate-index", type=int, choices=range(len(wave.GATE_IDS)), required=True)
    gate.add_argument("--parent-result", type=Path, default=wave.DEFAULT_PARENT_RESULT)
    gate.add_argument(
        "--parent-verification",
        type=Path,
        default=wave.DEFAULT_PARENT_VERIFICATION,
    )
    gate.add_argument(
        "--parent-report-receipt",
        type=Path,
        default=wave.DEFAULT_PARENT_REPORT_RECEIPT,
    )
    gate.add_argument("--d1-result", type=Path, default=wave.DEFAULT_D1_RESULT)
    gate.add_argument("--d1-rung-root", type=Path, default=wave.DEFAULT_D1_RUNG_ROOT)

    category = commands.add_parser(
        "run-category",
        help=(
            "run retained lanes in one fresh-cycle mechanics category through an eight-worker dynamic pool"
        ),
    )
    category.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)
    category.add_argument(
        "--epoch-index",
        type=int,
        choices=range(len(wave.EPOCH_IDS)),
        required=True,
    )
    category.add_argument("--category", choices=wave.CATEGORY_IDS, required=True)

    classify = commands.add_parser("classify-wave", help="seal one serial wave routing barrier")
    classify.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)
    classify.add_argument(
        "--epoch-index",
        type=int,
        choices=range(len(wave.EPOCH_IDS)),
        required=True,
    )

    integration = commands.add_parser(
        "run-integration",
        help="evaluate the conditionally eligible post-W07 I1 mechanics integration",
    )
    integration.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)

    integration_classification = commands.add_parser(
        "classify-integration",
        help="classify the single I1 integration for operator review",
    )
    integration_classification.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)

    aggregate = commands.add_parser("aggregate", help="aggregate the complete scaffold")
    aggregate.add_argument("--root", type=Path, default=wave.DEFAULT_ROOT)
    aggregate.add_argument("--output", type=Path, default=wave.DEFAULT_RESULT)

    verify = commands.add_parser("verify", help="independently verify the complete scaffold")
    verify.add_argument("--result", type=Path, default=wave.DEFAULT_RESULT)
    verify.add_argument("--output", type=Path, default=wave.DEFAULT_VERIFICATION)

    report = commands.add_parser("report", help="render the scaffold report and receipt")
    report.add_argument("--result", type=Path, default=wave.DEFAULT_RESULT)
    report.add_argument("--verification", type=Path, default=wave.DEFAULT_VERIFICATION)
    report.add_argument("--report", type=Path, default=wave.DEFAULT_REPORT)
    report.add_argument("--receipt", type=Path, default=wave.DEFAULT_REPORT_RECEIPT)
    return parser


def dispatch(arguments: argparse.Namespace) -> dict[str, Any]:
    if arguments.command == "materialize-gate":
        return wave.materialize_gate(
            root=arguments.root,
            gate_index=arguments.gate_index,
            parent_result_path=arguments.parent_result,
            parent_verification_path=arguments.parent_verification,
            parent_report_receipt_path=arguments.parent_report_receipt,
            d1_result_path=arguments.d1_result,
            d1_rung_root=arguments.d1_rung_root,
        )
    if arguments.command == "run-category":
        return wave.run_category(
            root=arguments.root,
            epoch_index=arguments.epoch_index,
            category_id=arguments.category,
        )
    if arguments.command == "classify-wave":
        return wave.classify_epoch(
            root=arguments.root,
            epoch_index=arguments.epoch_index,
        )
    if arguments.command == "run-integration":
        return wave.run_integration(root=arguments.root)
    if arguments.command == "classify-integration":
        return wave.classify_integration(root=arguments.root)
    if arguments.command == "aggregate":
        return wave.aggregate(root=arguments.root, output=arguments.output)
    if arguments.command == "verify":
        verifier = importlib.import_module("mop.studies.generation1_successor_categorized_batch_wave_verify")
        result: dict[str, Any] = verifier.verify(
            result_path=arguments.result,
            output=arguments.output,
        )
        return result
    if arguments.command == "report":
        return wave.render_report(
            result_path=arguments.result,
            verification_path=arguments.verification,
            report_path=arguments.report,
            receipt_path=arguments.receipt,
        )
    raise ValueError(f"unsupported categorized-wave command: {arguments.command}")


def main(argv: Sequence[str] | None = None) -> int:
    result = dispatch(build_parser().parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
