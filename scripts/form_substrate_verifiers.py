#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.falsification.form_evidence import (
    build_campaign_scorecard,
    build_form_verdict_gates,
    write_scorecard_inputs,
)
from mop.falsification.form_verifier import (
    CANDIDATE_POSITIVES,
    FRESH_SEEDS,
    run_candidate_verifiers,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="fresh-seed independent adversarial verification for F candidate positives"
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        choices=CANDIDATE_POSITIVES,
        help="verify only this experiment id; repeatable",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(str(value) for value in FRESH_SEEDS),
        help="comma-separated effective verifier seeds; at least five unique values",
    )
    parser.add_argument("--no-write", action="store_true", help="run without durable verifier receipts")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="refresh verdict gates and campaign scorecard after all verifier runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        seeds = [int(value.strip()) for value in str(args.seeds).split(",") if value.strip()]
    except ValueError as exc:
        raise SystemExit(f"--seeds must contain only integers: {exc}") from exc
    result = run_candidate_verifiers(
        experiment_ids=list(args.only) or None,
        repo_root=REPO_ROOT,
        fresh_seeds=seeds,
        write=not bool(args.no_write),
    )
    if args.refresh and not args.no_write:
        result["verdict_gates"] = build_form_verdict_gates(repo_root=REPO_ROOT)
        scorecard = build_campaign_scorecard(repo_root=REPO_ROOT)
        write_scorecard_inputs(scorecard, repo_root=REPO_ROOT)
        result["scorecard"] = scorecard
    print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("all_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
