#!/usr/bin/env python
"""Cache integrity CLI. Read-only: list, describe, and validate the latent caches under
data/cache without touching the encoder. JSON to stdout, logs to stderr.

Usage:
  python scripts/cache_tool.py list [--root data/cache]
  python scripts/cache_tool.py info  data/cache/<name>
  python scripts/cache_tool.py validate data/cache/<name>

`validate` prints {"store_dir","clean","problems"} and exits non-zero if the cache is dirty,
so it doubles as a CI gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from devsys.substrate.cache_tools import (
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
