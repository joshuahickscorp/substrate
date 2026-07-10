#!/usr/bin/env python
"""Run cache-first token-aware E6, or its tiny non-promotable mechanics fixture."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from mop.experiments.e6_dense_relational import (
    E6DenseError,
    TokenReadoutSpec,
    build_mechanics_fixture,
    file_sha256,
    run_dense_relational,
)


def _seeds(value: str) -> tuple[int, ...]:
    if "-" in value:
        start, end = (int(item) for item in value.split("-", maxsplit=1))
        if end < start:
            raise argparse.ArgumentTypeError("seed range end must be >= start")
        return tuple(range(start, end + 1))
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learned-cache", type=Path)
    parser.add_argument("--random-cache", type=Path)
    parser.add_argument("--fixture", action="store_true", help="run an ephemeral 36-row mechanics pair")
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--feature-rank", type=int, default=16)
    parser.add_argument("--summary-dim", type=int, default=32)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--seeds", type=_seeds, default=(0, 1, 2, 3, 4))
    parser.add_argument("--min-margin", type=float, default=0.02)
    parser.add_argument("--ceiling", type=float, default=0.98)
    parser.add_argument("--proof", type=Path, default=Path("proof/E6_DENSE_RELATIONAL_MECHANICS.json"))
    return parser


def _run(args: argparse.Namespace, learned: Path, random: Path, fixture: dict | None) -> dict:
    spec = TokenReadoutSpec(
        bins=args.bins,
        feature_rank=args.feature_rank,
        summary_dim=args.summary_dim,
        ridge=args.ridge,
        seeds=args.seeds,
        min_margin=args.min_margin,
        ceiling=args.ceiling,
    )
    receipt = run_dense_relational(learned, random, spec=spec)
    receipt["implementation"]["runner"] = str(Path(__file__).resolve())
    receipt["implementation"]["runner_sha256"] = file_sha256(Path(__file__).resolve())
    if fixture is not None:
        receipt["fixture"] = fixture
        receipt["scientific_promotion"] = False
        receipt["claim_boundary"]["natural_video_claim"] = False
        receipt["claim_boundary"]["e6_null_rejected"] = False
    return receipt


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.fixture and (args.learned_cache or args.random_cache):
        raise SystemExit("--fixture cannot be combined with explicit cache paths")
    if not args.fixture and bool(args.learned_cache) != bool(args.random_cache):
        raise SystemExit("provide both --learned-cache and --random-cache")
    try:
        if args.fixture or not args.learned_cache:
            with tempfile.TemporaryDirectory(prefix="mop-e6-dense-fixture-") as temporary:
                fixture = build_mechanics_fixture(Path(temporary))
                receipt = _run(
                    args,
                    Path(fixture["stores"]["learned"]),
                    Path(fixture["stores"]["random"]),
                    fixture,
                )
        else:
            receipt = _run(args, args.learned_cache, args.random_cache, None)
        args.proof.parent.mkdir(parents=True, exist_ok=True)
        args.proof.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2, sort_keys=True))
        return 0 if receipt["all_ok"] else 1
    except (E6DenseError, OSError, ValueError) as exc:
        failure = {
            "schema": "mop-e6-dense-relational-failure/v1",
            "all_ok": False,
            "scientific_promotion": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
