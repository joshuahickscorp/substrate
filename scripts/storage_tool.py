#!/usr/bin/env python
"""Disk accounting for the latent cache: estimate before you compute, list what is on disk,
prune SAFELY. Pruning defaults to a dry run and refuses to delete without an explicit --apply.

Usage:
  python scripts/storage_tool.py estimate --encoder vjepa2_vitl_fpc64_256 --clips 10000 [--dense]
  python scripts/storage_tool.py estimate --count 1000 --feat 1024            # raw shape form
  python scripts/storage_tool.py list   [--root data/cache]
  python scripts/storage_tool.py prune  --keep vjepa2_vitl_fpc64_256_real [--root data/cache]
  python scripts/storage_tool.py prune  --keep <name> --apply                # actually deletes
JSON to stdout, human log lines to stderr. `prune` without --apply is a dry run (deletes nothing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from mop.config import REPO_ROOT
from mop.logging_utils import get_logger
from mop.substrate.storage import (
    estimate_cache_bytes,
    estimate_for_encoder,
    human_bytes,
    list_caches_with_size,
    prune_caches,
)

log = get_logger("storage_tool")


def _resolve_root(root: str) -> Path:
    p = Path(root)
    return p if p.is_absolute() else REPO_ROOT / p


def _cmd_estimate(a: argparse.Namespace) -> dict:
    if a.encoder:
        cfg_path = REPO_ROOT / "configs" / "encoder" / f"{a.encoder}.yaml"
        cfg = OmegaConf.load(cfg_path)
        out = estimate_for_encoder(
            OmegaConf.to_container(cfg, resolve=True),
            n_clips=a.clips,
            dense=a.dense,
            dense_tokens=a.dense_tokens,
        )
        log.info(
            "estimate %s clips=%d dense=%s -> %s",
            a.encoder,
            a.clips,
            out["dense"],
            out["human"],
        )
        return out
    if a.count is None or a.feat is None:
        raise SystemExit("estimate needs --encoder, or both --count and --feat")
    b = estimate_cache_bytes(a.count, list(a.feat), dtype=a.dtype, has_labels=not a.no_labels)
    out = {
        "bytes": b,
        "human": human_bytes(b),
        "count": a.count,
        "feat_shape": list(a.feat),
        "dtype": a.dtype,
    }
    log.info("estimate count=%d feat=%s -> %s", a.count, list(a.feat), out["human"])
    return out


def _cmd_list(a: argparse.Namespace) -> dict:
    root = _resolve_root(a.root)
    caches = list_caches_with_size(root)
    total = sum(c["bytes"] for c in caches)
    log.info("listed %d caches under %s, total %s", len(caches), root, human_bytes(total))
    return {"root": str(root), "caches": caches, "total_bytes": total, "total_human": human_bytes(total)}


def _cmd_prune(a: argparse.Namespace) -> dict:
    root = _resolve_root(a.root)
    dry_run = not a.apply  # SAFE DEFAULT: no --apply means dry run, nothing is deleted
    plan = prune_caches(root, keep=a.keep, dry_run=dry_run)
    freed = sum(p["bytes"] for p in plan if p["would_delete"])
    if dry_run:
        log.info(
            "DRY RUN: would delete %d caches (%s); pass --apply to actually delete. Kept: %s",
            sum(1 for p in plan if p["would_delete"]),
            human_bytes(freed),
            a.keep or [],
        )
    else:
        log.warning(
            "APPLIED: deleted %d caches (%s freed). Kept: %s",
            sum(1 for p in plan if p["deleted"]),
            human_bytes(freed),
            a.keep or [],
        )
    return {
        "root": str(root),
        "dry_run": dry_run,
        "freed_bytes": freed,
        "freed_human": human_bytes(freed),
        "plan": plan,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="latent-cache storage estimation + cleanup")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("estimate", help="estimate cache bytes (by encoder, or raw count+feat)")
    pe.add_argument("--encoder", type=str, default=None, help="encoder config stem under configs/encoder")
    pe.add_argument("--clips", type=int, default=1000, help="number of clips (encoder form)")
    pe.add_argument("--dense", action="store_true", help="dense (per-token) latents")
    pe.add_argument("--dense-tokens", type=int, default=None, help="override tokens/clip for dense")
    pe.add_argument("--count", type=int, default=None, help="item count (raw form)")
    pe.add_argument("--feat", type=int, nargs="+", default=None, help="per-item feat shape (raw form)")
    pe.add_argument("--dtype", type=str, default="float32")
    pe.add_argument("--no-labels", action="store_true")

    pl = sub.add_parser("list", help="list caches under a root with on-disk size")
    pl.add_argument("--root", type=str, default="data/cache")

    pp = sub.add_parser("prune", help="prune caches NOT in --keep (dry run unless --apply)")
    pp.add_argument("--root", type=str, default="data/cache")
    pp.add_argument("--keep", type=str, nargs="*", default=None, help="cache names to preserve")
    pp.add_argument("--apply", action="store_true", help="ACTUALLY delete (omit for a safe dry run)")

    a = ap.parse_args(sys.argv[1:] if argv is None else argv)
    out = {"estimate": _cmd_estimate, "list": _cmd_list, "prune": _cmd_prune}[a.cmd](a)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
