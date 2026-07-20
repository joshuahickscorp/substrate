#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mop.substrate.cache_manifest import validate_cache_manifest, write_cache_manifest
from mop.substrate.cache_tools import (
    DEFAULT_ROOT,
    cache_info,
    list_caches,
    validate_cache,
)


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="latent cache integrity tools")
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="list every cache under root")
    pl.add_argument("--root", default=str(DEFAULT_ROOT), help="cache root (default data/cache)")

    pi = sub.add_parser("info", help="meta + provenance + sanity facts for one cache")
    pi.add_argument("store_dir", help="path to one cache dir")

    pv = sub.add_parser("validate", help="integrity check one cache (exit !=0 if dirty)")
    pv.add_argument("store_dir", help="path to one cache dir")

    pm = sub.add_parser("manifest", help="write cache_manifest.json for one cache")
    pm.add_argument("store_dir", help="path to one cache dir")
    pm.add_argument(
        "--full-hash-arrays",
        action="store_true",
        help="hash full array files instead of sampled first/last bytes",
    )
    pm.add_argument("--encoder-config", default=None, help="optional encoder config JSON file")

    pvm = sub.add_parser("validate-manifest", help="validate cache_manifest.json only")
    pvm.add_argument("store_dir", help="path to one cache dir")

    args = p.parse_args(argv if argv is not None else sys.argv[1:])

    if args.cmd == "list":
        _emit(list_caches(Path(args.root)))
        return 0
    if args.cmd == "info":
        _emit(cache_info(Path(args.store_dir)))
        return 0
    if args.cmd == "validate":
        problems = validate_cache(Path(args.store_dir))
        _emit({"store_dir": str(args.store_dir), "clean": not problems, "problems": problems})
        return 0 if not problems else 1
    if args.cmd == "manifest":
        encoder_config = None
        if args.encoder_config is not None:
            encoder_config = json.loads(Path(args.encoder_config).read_text())
        manifest = write_cache_manifest(
            Path(args.store_dir),
            encoder_config=encoder_config,
            full_hash_arrays=bool(args.full_hash_arrays),
        )
        _emit({"store_dir": str(args.store_dir), "manifest": manifest})
        return 0
    if args.cmd == "validate-manifest":
        problems = validate_cache_manifest(Path(args.store_dir))
        _emit({"store_dir": str(args.store_dir), "clean": not problems, "problems": problems})
        return 0 if not problems else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
