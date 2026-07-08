#!/usr/bin/env python
"""Write a durable artifact index, optionally copying small receipt files."""

from __future__ import annotations

import argparse
import json
import sys

from mop.config import REPO_ROOT
from mop.studio.artifact_bundle import build_artifact_index, preset_paths, write_artifact_index


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="hash and optionally bundle MOP Studio receipt artifacts")
    p.add_argument("--preset", choices=["pre-studio", "wave0"], default="pre-studio")
    p.add_argument("--path", action="append", default=[], help="extra artifact path, repeatable")
    p.add_argument("--only-paths", action="store_true", help="ignore the preset and use only --path values")
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "proof" / "ARTIFACT_INDEX" / "pre_studio.json"),
        help="artifact index JSON path",
    )
    p.add_argument("--copy-dir", default=None, help="copy untracked small receipts into this bundle dir")
    p.add_argument("--max-copy-mb", type=float, default=5.0)
    p.add_argument(
        "--require-durable",
        action="store_true",
        help="fail unless every existing artifact is tracked or copied",
    )
    p.add_argument("--allow-missing", action="store_true", help="record missing artifacts without failing")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    paths = list(args.path) if args.only_paths else [*preset_paths(args.preset), *args.path]
    index = build_artifact_index(
        paths,
        copy_dir=args.copy_dir,
        max_copy_bytes=int(args.max_copy_mb * 1_000_000),
        require_durable=args.require_durable,
        allow_missing=args.allow_missing,
    )
    write_artifact_index(index, args.out)
    print(json.dumps({"out": args.out, "all_ok": index["all_ok"], "summary": index["summary"]}, indent=2))
    return 0 if index["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
