#!/usr/bin/env python
"""Write an independent adversarial verification receipt for a completed DR1 cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from mop.config import REPO_ROOT  # noqa: E402
from mop.studio.dr1_verifier import (  # noqa: E402
    DR1VerifierConfig,
    build_dr1_verification,
    write_dr1_verification,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DR1 adversarial artifact verifier")
    parser.add_argument("--cache", default="data/cache/vjepa2_vitl_comp_video")
    parser.add_argument("--out", default=None, help="default: <cache>/dr1_verification.json")
    parser.add_argument("--no-require-a6", action="store_true", help="draft only, do not require A6 receipt")
    parser.add_argument(
        "--no-require-perspective",
        action="store_true",
        help="draft only, do not require PerspectiveMatrix receipt",
    )
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    if not cache.is_absolute():
        cache = REPO_ROOT / cache
    out = Path(args.out) if args.out else cache / "dr1_verification.json"
    if not out.is_absolute():
        out = REPO_ROOT / out
    report = build_dr1_verification(
        DR1VerifierConfig(
            cache_dir=cache,
            require_a6=not args.no_require_a6,
            require_perspective=not args.no_require_perspective,
        )
    )
    report["path"] = str(out)
    write_dr1_verification(report, out)
    print(
        json.dumps(
            {
                "out": str(out),
                "integrity_ok": report["integrity_ok"],
                "passed": report["passed"],
                "summary": report["summary"],
                "problems": report["problems"],
            },
            indent=2,
        )
    )
    return 0 if report["integrity_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
