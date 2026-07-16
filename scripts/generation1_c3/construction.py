#!/usr/bin/env python3
"""Run or validate the sealed G1-C3 construction-search mechanics pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mop.studies.generation1_c3_construction import (
    DEFAULT_SEED_RANGES,
    FRESH_SOURCE,
    atomic_write_result,
    build_pilot_config,
    read_and_validate_result,
    run_pilot,
)


def _seed_range(value: str) -> dict[str, object]:
    try:
        name, role, start, count = value.split(":", 3)
        return {
            "name": name,
            "role": role,
            "start": int(start),
            "count": int(count),
            "source": FRESH_SOURCE,
        }
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "seed range must be NAME:ROLE:START:COUNT"
        ) from exc


def _summary(result: dict) -> dict:
    return {
        "study_id": result["study_id"],
        "seed_count": result["overall"]["seed_count"],
        "discrimination_fraction": result["overall"]["discrimination_fraction"],
        "verdict": result["decision"]["verdict"],
        "result_sha256": result["result_sha256"],
        "activation_allowed": result["activation_allowed"],
        "scientific_promotion": result["scientific_promotion"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/generation1_c3/construction_pilot.json"),
    )
    parser.add_argument(
        "--seed-range",
        action="append",
        type=_seed_range,
        help="repeatable NAME:ROLE:START:COUNT; defaults to the two sealed batch ranges",
    )
    parser.add_argument("--synergy-bonus", type=float, default=1.0)
    parser.add_argument("--per-eval-cost", type=float, default=0.0002)
    parser.add_argument("--size-penalty", type=float, default=0.05)
    parser.add_argument("--num-members", type=int, default=12)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument("--search-restarts", type=int, default=40)
    parser.add_argument("--random-samples", type=int, default=2500)
    parser.add_argument("--minimum-discrimination-fraction", type=float, default=0.75)
    parser.add_argument("--validate", type=Path, help="validate an existing result instead of running")
    args = parser.parse_args()

    if args.validate is not None:
        result = read_and_validate_result(args.validate)
    else:
        config = build_pilot_config(
            seed_ranges=args.seed_range or DEFAULT_SEED_RANGES,
            synergy_bonus=args.synergy_bonus,
            per_eval_cost=args.per_eval_cost,
            size_penalty=args.size_penalty,
            num_members=args.num_members,
            num_tasks=args.num_tasks,
            search_restarts=args.search_restarts,
            random_samples=args.random_samples,
            minimum_discrimination_fraction=args.minimum_discrimination_fraction,
        )
        result = run_pilot(config)
        atomic_write_result(args.output, result)
    print(json.dumps(_summary(result), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
